#!/usr/bin/env python3
"""
TPMA Multi-Asset Toxicity Tracker - Advanced Systemic Risk Detection
Implements Thick Pen Measure of Association based on Fryzlewicz and Oh (2011) mathematical framework.

Mathematical Foundation:
- Thick-Pen Transform (TPT): U^τ_t = max(X_t, ..., X_{t+τ}) + τ/2, L^τ_t = min(X_t, ..., X_{t+τ}) - τ/2
- TPMA Formula: ρ^τ_t = [min(U^τ_t) - max(L^τ_t)] / [max(U^τ_t) - min(L^τ_t)]
- Multi-Thickness TPMA (MTTPMA) for cross-scale dependence analysis
- Real-time adaptation for high-frequency market surveillance and flash crash detection

Two-step process: individual toxicity measurement + cross-asset comovement quantification.
"""

import asyncio
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass
from collections import defaultdict, deque
import statistics
import requests
import networkx as nx
from scipy.stats import pearsonr, spearmanr
import warnings

logger = logging.getLogger(__name__)

@dataclass
class TPMAConfig:
    """Configuration for TPMA multi-asset toxicity tracking based on mathematical framework"""
    # Core TPMA Parameters (from Fryzlewicz and Oh 2011)
    thickness_range: List[int] = None  # Thickness parameters τ (default: [1, 2, 5, 10, 20, 50])
    pen_shape: str = "square"  # Pen shape (square most common for computational efficiency)
    scaling_factor: float = 1.0  # Scaling factor for TPT calculation (default: 1)
    
    # Multi-Thickness TPMA (MTTPMA) Parameters
    enable_mttpma: bool = True  # Enable multi-thickness analysis
    cross_scale_analysis: bool = True  # Enable cross-scale dependence analysis
    
    # Order Flow Toxicity Parameters (Two-Step Process)
    correlation_window: int = 100  # Number of observations for correlation analysis
    toxicity_threshold: float = 0.5  # Individual toxicity level considered significant
    comovement_threshold: float = 0.3  # TPMA threshold for significant comovement
    
    # System Risk Parameters
    min_assets: int = 3  # Minimum assets required for meaningful analysis
    systemic_risk_threshold: float = 0.7  # Threshold for systemic risk alert
    network_centrality_threshold: float = 0.6  # Network centrality alert threshold
    
    # Real-Time Parameters
    update_frequency_seconds: int = 60  # Update frequency for continuous monitoring
    buffer_size: int = 500  # Data buffer size for efficient computation
    standardization_window: int = 200  # Window for time series standardization
    
    # Flash Crash Detection Parameters
    rapid_change_threshold: float = 0.8  # Threshold for rapid TPMA changes
    contagion_speed_threshold: float = 0.8  # Speed threshold for contagion detection
    volume_bucket_analysis: bool = True  # Enable volume-bucket-wise analysis
    
    # Performance Parameters
    recursive_computation: bool = True  # Enable recursive TPT computation for efficiency
    parallel_processing: bool = True  # Enable parallel processing for multiple assets
    
    # External Integration
    feature_engine_url: str = "http://localhost:8500"
    
    def __post_init__(self):
        """Set default thickness range if not provided"""
        if self.thickness_range is None:
            # Default thickness range: small (1-5) to large (20-100+) per research
            self.thickness_range = [1, 2, 5, 10, 20, 50, 100]

@dataclass
class AssetToxicityMetrics:
    """Individual asset toxicity metrics"""
    timestamp: float
    symbol: str
    vpin_toxicity: float
    kyles_lambda: float
    pin_score: float
    spread_toxicity: float
    quote_fade_score: float
    composite_toxicity: float  # Weighted combination of all metrics
    toxicity_trend: str  # INCREASING, DECREASING, STABLE
    risk_level: str  # LOW, MODERATE, HIGH, EXTREME

@dataclass
class TPMAComovementMetrics:
    """TPMA comovement analysis results"""
    asset_pair: Tuple[str, str]
    correlation_coefficient: float
    correlation_p_value: float
    comovement_strength: str  # WEAK, MODERATE, STRONG, VERY_STRONG
    lead_lag_relationship: Dict[str, float]  # Which asset leads in toxicity
    contagion_probability: float
    time_to_propagation_ms: Optional[float]

@dataclass
class SystemicRiskMetrics:
    """Overall systemic risk assessment"""
    timestamp: float
    total_assets_tracked: int
    assets_above_threshold: int
    network_density: float  # How interconnected toxicity levels are
    systemic_risk_score: float  # Overall system risk (0-1)
    dominant_cluster_size: int  # Size of largest connected component
    contagion_pathways: List[List[str]]  # Potential contagion paths
    risk_level: str  # LOW, MODERATE, HIGH, SYSTEMIC
    early_warning_signals: List[str]  # List of warning indicators
    central_risk_nodes: List[str]  # Most connected/risky assets

@dataclass
class TPMASnapshot:
    """Complete TPMA system snapshot"""
    timestamp: float
    individual_metrics: Dict[str, AssetToxicityMetrics]
    comovement_matrix: Dict[Tuple[str, str], TPMAComovementMetrics]
    systemic_metrics: SystemicRiskMetrics
    network_graph: Dict[str, Any]  # NetworkX graph serialized
    alert_level: str  # NORMAL, ELEVATED, HIGH, CRITICAL
    recommendations: List[str]  # Action recommendations

class TPMAMultiAssetTracker:
    """TPMA implementation for multi-asset toxicity tracking and systemic risk detection"""
    
    def __init__(self, config: TPMAConfig = None):
        self.config = config or TPMAConfig()
        self.asset_data = defaultdict(deque)  # Store time series data per asset
        self.toxicity_history = defaultdict(deque)  # Toxicity time series
        self.comovement_history = {}
        self.systemic_snapshots = deque(maxlen=100)  # Keep last 100 snapshots
        self.active_assets = set()
        self.network_graph = nx.Graph()
        
    async def update_asset_data(self, symbol: str, market_data: Dict[str, Any]) -> None:
        """Update market data for an asset"""
        
        # Add timestamp if not present
        if 'timestamp' not in market_data:
            market_data['timestamp'] = datetime.now().timestamp()
        
        # Store in rolling window
        asset_queue = self.asset_data[symbol]
        asset_queue.append(market_data)
        
        # Keep only recent data
        if len(asset_queue) > self.config.correlation_window * 2:
            asset_queue.popleft()
            
        self.active_assets.add(symbol)
        
    async def calculate_individual_toxicity(self, symbol: str) -> AssetToxicityMetrics:
        """
        Step 1 of TPMA: Calculate individual asset toxicity
        Combines multiple toxicity measures into composite score
        """
        
        if symbol not in self.asset_data or len(self.asset_data[symbol]) < 10:
            return self._create_default_individual_metrics(symbol)
        
        asset_history = list(self.asset_data[symbol])
        
        # Calculate individual toxicity components
        vpin_toxicity = await self._calculate_vpin_proxy(asset_history)
        kyles_lambda = await self._calculate_lambda_proxy(asset_history)
        pin_score = await self._calculate_pin_proxy(asset_history)
        spread_toxicity = await self._calculate_spread_proxy(asset_history)
        quote_fade_score = await self._calculate_fade_proxy(asset_history)
        
        # Composite toxicity score (weighted combination)
        composite_toxicity = self._calculate_composite_toxicity(
            vpin_toxicity, kyles_lambda, pin_score, spread_toxicity, quote_fade_score
        )
        
        # Trend analysis
        toxicity_trend = await self._analyze_toxicity_trend(symbol, composite_toxicity)
        
        # Risk level classification
        risk_level = self._classify_risk_level(composite_toxicity)
        
        metrics = AssetToxicityMetrics(
            timestamp=datetime.now().timestamp(),
            symbol=symbol,
            vpin_toxicity=vpin_toxicity,
            kyles_lambda=kyles_lambda,
            pin_score=pin_score,
            spread_toxicity=spread_toxicity,
            quote_fade_score=quote_fade_score,
            composite_toxicity=composite_toxicity,
            toxicity_trend=toxicity_trend,
            risk_level=risk_level
        )
        
        # Store in history
        self.toxicity_history[symbol].append(metrics)
        if len(self.toxicity_history[symbol]) > 200:
            self.toxicity_history[symbol].popleft()
        
        return metrics
    
    async def calculate_thick_pen_transform(self, time_series: List[float], thickness: int) -> Tuple[List[float], List[float]]:
        """
        Calculate Thick-Pen Transform (TPT) using square pen implementation.
        
        Mathematical Formula (Fryzlewicz and Oh 2011):
        U^τ_t = max(X_t, X_{t+1}, ..., X_{t+τ}) + τ/2
        L^τ_t = min(X_t, X_{t+1}, ..., X_{t+τ}) - τ/2
        """
        if len(time_series) < thickness + 1:
            return [], []
        
        n = len(time_series) - thickness
        U_series = []  # Upper boundaries
        L_series = []  # Lower boundaries
        
        if self.config.recursive_computation and thickness > 1:
            # Recursive computation for efficiency: U^τ_t = max(U^{τ-1}_t, U^{τ-1}_{t+1}) + 1/2
            prev_U, prev_L = await self.calculate_thick_pen_transform(time_series, thickness - 1)
            
            for t in range(len(prev_U) - 1):
                if t < len(prev_U) - 1:
                    U_t = max(prev_U[t], prev_U[t + 1]) + self.config.scaling_factor / 2
                    L_t = min(prev_L[t], prev_L[t + 1]) - self.config.scaling_factor / 2
                    U_series.append(U_t)
                    L_series.append(L_t)
        else:
            # Direct computation
            for t in range(n + 1):
                window = time_series[t:t + thickness + 1]
                U_t = max(window) + (thickness * self.config.scaling_factor) / 2
                L_t = min(window) - (thickness * self.config.scaling_factor) / 2
                U_series.append(U_t)
                L_series.append(L_t)
        
        return U_series, L_series
    
    async def calculate_tpma_coefficient(self, series1: List[float], series2: List[float], thickness: int) -> float:
        """
        Calculate TPMA coefficient between two standardized time series.
        
        Mathematical Formula:
        ρ^τ_t = [min(U^τ_t(X^(1)), U^τ_t(X^(2))) - max(L^τ_t(X^(1)), L^τ_t(X^(2)))] /
                [max(U^τ_t(X^(1)), U^τ_t(X^(2))) - min(L^τ_t(X^(1)), L^τ_t(X^(2)))]
        
        Returns: TPMA coefficient ∈ (-1, 1] where values close to 1 indicate strong positive association
        """
        
        # Standardize time series (zero mean, unit variance)
        series1_std = self._standardize_series(series1)
        series2_std = self._standardize_series(series2)
        
        # Calculate TPT for both series
        U1, L1 = await self.calculate_thick_pen_transform(series1_std, thickness)
        U2, L2 = await self.calculate_thick_pen_transform(series2_std, thickness)
        
        if not U1 or not U2 or len(U1) != len(U2):
            return 0.0
        
        # Calculate TPMA for each time point and take average
        tpma_values = []
        
        for i in range(len(U1)):
            numerator = min(U1[i], U2[i]) - max(L1[i], L2[i])
            denominator = max(U1[i], U2[i]) - min(L1[i], L2[i])
            
            if abs(denominator) > 1e-10:  # Avoid division by zero
                tpma_t = numerator / denominator
                tpma_values.append(tpma_t)
        
        # Return average TPMA coefficient
        return statistics.mean(tpma_values) if tpma_values else 0.0
    
    async def calculate_multi_thickness_tpma(self, series1: List[float], series2: List[float]) -> Dict[int, float]:
        """
        Calculate Multi-Thickness TPMA (MTTPMA) for cross-scale dependence analysis.
        Returns TPMA coefficients for all thickness parameters.
        """
        
        mttpma_results = {}
        
        for thickness in self.config.thickness_range:
            if len(series1) >= thickness + 10 and len(series2) >= thickness + 10:  # Ensure sufficient data
                tpma_coef = await self.calculate_tpma_coefficient(series1, series2, thickness)
                mttpma_results[thickness] = tpma_coef
        
        return mttpma_results
    
    async def calculate_volume_bucket_tpma(self, asset1: str, asset2: str) -> Dict[str, float]:
        """
        Calculate volume-bucket-wise TPMA for high-frequency market surveillance.
        Implements proportional order imbalances with volume-bucket standard deviation.
        """
        
        if asset1 not in self.asset_data or asset2 not in self.asset_data:
            return {}
        
        # Get recent volume and price data
        data1 = list(self.asset_data[asset1])[-self.config.correlation_window:]
        data2 = list(self.asset_data[asset2])[-self.config.correlation_window:]
        
        if len(data1) < 50 or len(data2) < 50:
            return {}
        
        # Calculate proportional order imbalances for volume buckets
        volume_buckets = ['small', 'medium', 'large']  # Volume quartiles
        bucket_tpma = {}
        
        for bucket in volume_buckets:
            # Filter data by volume bucket
            bucket_data1 = self._filter_by_volume_bucket(data1, bucket)
            bucket_data2 = self._filter_by_volume_bucket(data2, bucket)
            
            if len(bucket_data1) >= 20 and len(bucket_data2) >= 20:
                # Calculate order imbalances
                imbalances1 = [self._calculate_order_imbalance(d) for d in bucket_data1]
                imbalances2 = [self._calculate_order_imbalance(d) for d in bucket_data2]
                
                # Calculate TPMA on imbalances using optimal thickness
                optimal_thickness = min(10, len(imbalances1) // 4)  # Adaptive thickness
                if optimal_thickness > 0:
                    bucket_tpma[bucket] = await self.calculate_tpma_coefficient(
                        imbalances1, imbalances2, optimal_thickness
                    )
        
        return bucket_tpma

    async def calculate_comovement_analysis(self, assets: List[str]) -> Dict[Tuple[str, str], TPMAComovementMetrics]:
        """
        Step 2 of TPMA: Analyze toxicity comovement between asset pairs using mathematical framework
        """
        
        comovement_results = {}
        
        # Ensure we have enough data for all assets
        valid_assets = [asset for asset in assets 
                       if asset in self.toxicity_history and len(self.toxicity_history[asset]) >= 20]
        
        if len(valid_assets) < 2:
            return comovement_results
        
        # Analyze all asset pairs with TPMA methodology
        for i, asset1 in enumerate(valid_assets):
            for asset2 in valid_assets[i+1:]:
                pair = (asset1, asset2)
                comovement_metrics = await self._analyze_pair_comovement_tpma(asset1, asset2)
                comovement_results[pair] = comovement_metrics
                
                # Store in history
                self.comovement_history[pair] = comovement_metrics
        
        return comovement_results
    
    async def _analyze_pair_comovement_tpma(self, asset1: str, asset2: str) -> TPMAComovementMetrics:
        """Analyze comovement between two assets using TPMA mathematical framework"""
        
        # Get toxicity time series
        history1 = list(self.toxicity_history[asset1])
        history2 = list(self.toxicity_history[asset2])
        
        # Align time series (take minimum length)
        min_length = min(len(history1), len(history2))
        if min_length < 10:
            return self._create_default_comovement_metrics(asset1, asset2)
        
        # Extract toxicity values for TPMA analysis
        toxicity1 = [h.composite_toxicity for h in history1[-min_length:]]
        toxicity2 = [h.composite_toxicity for h in history2[-min_length:]]
        
        # Calculate Multi-Thickness TPMA if enabled
        if self.config.enable_mttpma:
            mttpma_results = await self.calculate_multi_thickness_tpma(toxicity1, toxicity2)
            
            # Use average TPMA across thickness parameters as primary correlation measure
            tpma_coefficients = list(mttpma_results.values())
            correlation_coef = statistics.mean(tpma_coefficients) if tpma_coefficients else 0.0
            
            # Calculate statistical significance (approximate)
            p_value = 1.0 - abs(correlation_coef)  # Simplified p-value approximation
        else:
            # Single thickness TPMA (use medium thickness)
            optimal_thickness = min(10, len(toxicity1) // 4)
            correlation_coef = await self.calculate_tpma_coefficient(toxicity1, toxicity2, optimal_thickness)
            p_value = 1.0 - abs(correlation_coef)
        
        # Volume-bucket analysis if enabled
        volume_bucket_tpma = {}
        if self.config.volume_bucket_analysis:
            volume_bucket_tpma = await self.calculate_volume_bucket_tpma(asset1, asset2)
        
        # Classify comovement strength based on TPMA coefficient
        comovement_strength = self._classify_tpma_strength(abs(correlation_coef))
        
        # Enhanced lead-lag analysis using cross-scale TPMA
        lead_lag_relationship = await self._analyze_tpma_lead_lag(toxicity1, toxicity2, asset1, asset2)
        
        # Contagion probability estimation using TPMA dynamics
        contagion_probability = self._estimate_tpma_contagion_probability(
            correlation_coef, toxicity1, toxicity2, volume_bucket_tpma
        )
        
        # Time to propagation using TPMA-based analysis
        time_to_propagation = await self._estimate_tpma_propagation_time(
            history1[-min_length:], history2[-min_length:], correlation_coef
        ) if abs(correlation_coef) > self.config.comovement_threshold else None
        
        return TPMAComovementMetrics(
            asset_pair=(asset1, asset2),
            correlation_coefficient=correlation_coef,
            correlation_p_value=p_value,
            comovement_strength=comovement_strength,
            lead_lag_relationship=lead_lag_relationship,
            contagion_probability=contagion_probability,
            time_to_propagation_ms=time_to_propagation
        )

    async def _analyze_pair_comovement(self, asset1: str, asset2: str) -> TPMAComovementMetrics:
        """Analyze comovement between two assets"""
        
        # Get toxicity time series
        history1 = list(self.toxicity_history[asset1])
        history2 = list(self.toxicity_history[asset2])
        
        # Align time series (take minimum length)
        min_length = min(len(history1), len(history2))
        if min_length < 10:
            return self._create_default_comovement_metrics(asset1, asset2)
        
        # Extract toxicity values
        toxicity1 = [h.composite_toxicity for h in history1[-min_length:]]
        toxicity2 = [h.composite_toxicity for h in history2[-min_length:]]
        
        # Calculate correlation
        correlation_coef, p_value = pearsonr(toxicity1, toxicity2)
        
        # Classify comovement strength
        comovement_strength = self._classify_comovement_strength(abs(correlation_coef))
        
        # Lead-lag analysis
        lead_lag_relationship = self._analyze_lead_lag(toxicity1, toxicity2, asset1, asset2)
        
        # Contagion probability estimation
        contagion_probability = self._estimate_contagion_probability(
            correlation_coef, toxicity1, toxicity2
        )
        
        # Time to propagation (if significant correlation exists)
        time_to_propagation = self._estimate_propagation_time(
            history1[-min_length:], history2[-min_length:]
        ) if abs(correlation_coef) > self.config.comovement_threshold else None
        
        return TPMAComovementMetrics(
            asset_pair=(asset1, asset2),
            correlation_coefficient=correlation_coef,
            correlation_p_value=p_value,
            comovement_strength=comovement_strength,
            lead_lag_relationship=lead_lag_relationship,
            contagion_probability=contagion_probability,
            time_to_propagation_ms=time_to_propagation
        )
    
    async def calculate_systemic_risk(self, individual_metrics: Dict[str, AssetToxicityMetrics],
                                    comovement_metrics: Dict[Tuple[str, str], TPMAComovementMetrics]) -> SystemicRiskMetrics:
        """Calculate overall systemic risk from TPMA analysis"""
        
        if len(individual_metrics) < self.config.min_assets:
            return self._create_default_systemic_metrics()
        
        # Count assets above toxicity threshold
        assets_above_threshold = sum(
            1 for metrics in individual_metrics.values()
            if metrics.composite_toxicity > self.config.toxicity_threshold
        )
        
        # Build network graph for connectivity analysis
        self.network_graph.clear()
        self.network_graph.add_nodes_from(individual_metrics.keys())
        
        # Add edges based on significant comovements
        for pair, comovement in comovement_metrics.items():
            if abs(comovement.correlation_coefficient) > self.config.comovement_threshold:
                weight = abs(comovement.correlation_coefficient)
                self.network_graph.add_edge(pair[0], pair[1], weight=weight,
                                          contagion_prob=comovement.contagion_probability)
        
        # Network analysis
        network_density = nx.density(self.network_graph) if len(self.network_graph.nodes) > 1 else 0.0
        
        # Find largest connected component
        if len(self.network_graph.nodes) > 0:
            largest_cc = max(nx.connected_components(self.network_graph), key=len, default=set())
            dominant_cluster_size = len(largest_cc)
        else:
            dominant_cluster_size = 0
        
        # Calculate systemic risk score
        systemic_risk_score = self._calculate_systemic_risk_score(
            individual_metrics, network_density, assets_above_threshold
        )
        
        # Identify contagion pathways
        contagion_pathways = self._identify_contagion_pathways()
        
        # Risk level classification
        risk_level = self._classify_systemic_risk_level(systemic_risk_score, assets_above_threshold)
        
        # Early warning signals
        early_warning_signals = self._detect_early_warning_signals(
            individual_metrics, comovement_metrics
        )
        
        # Central risk nodes (most connected assets)
        central_risk_nodes = self._identify_central_risk_nodes()
        
        return SystemicRiskMetrics(
            timestamp=datetime.now().timestamp(),
            total_assets_tracked=len(individual_metrics),
            assets_above_threshold=assets_above_threshold,
            network_density=network_density,
            systemic_risk_score=systemic_risk_score,
            dominant_cluster_size=dominant_cluster_size,
            contagion_pathways=contagion_pathways,
            risk_level=risk_level,
            early_warning_signals=early_warning_signals,
            central_risk_nodes=central_risk_nodes
        )
    
    async def generate_tpma_snapshot(self, assets: List[str]) -> TPMASnapshot:
        """Generate complete TPMA system snapshot"""
        
        # Step 1: Calculate individual toxicity for all assets
        individual_metrics = {}
        for asset in assets:
            if asset in self.active_assets:
                individual_metrics[asset] = await self.calculate_individual_toxicity(asset)
        
        # Step 2: Calculate comovement analysis
        comovement_matrix = await self.calculate_comovement_analysis(list(individual_metrics.keys()))
        
        # Step 3: Calculate systemic risk
        systemic_metrics = await self.calculate_systemic_risk(individual_metrics, comovement_matrix)
        
        # Alert level determination
        alert_level = self._determine_alert_level(systemic_metrics)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(systemic_metrics, individual_metrics)
        
        # Serialize network graph
        network_graph_data = self._serialize_network_graph()
        
        snapshot = TPMASnapshot(
            timestamp=datetime.now().timestamp(),
            individual_metrics=individual_metrics,
            comovement_matrix=comovement_matrix,
            systemic_metrics=systemic_metrics,
            network_graph=network_graph_data,
            alert_level=alert_level,
            recommendations=recommendations
        )
        
        # Store snapshot
        self.systemic_snapshots.append(snapshot)
        
        return snapshot
    
    # Helper methods for toxicity calculation
    async def _calculate_vpin_proxy(self, asset_history: List[Dict[str, Any]]) -> float:
        """Calculate VPIN-like toxicity proxy from price/volume data"""
        if len(asset_history) < 10:
            return 0.0
        
        # Simple VPIN proxy: ratio of volume-weighted price moves
        price_moves = []
        volumes = []
        
        for i in range(1, len(asset_history)):
            prev_price = float(asset_history[i-1].get('price', asset_history[i-1].get('close', 0)))
            curr_price = float(asset_history[i].get('price', asset_history[i].get('close', 0)))
            volume = float(asset_history[i].get('volume', 0))
            
            if prev_price > 0 and volume > 0:
                price_change = abs(curr_price - prev_price) / prev_price
                price_moves.append(price_change)
                volumes.append(volume)
        
        if not price_moves:
            return 0.0
        
        # Volume-weighted average of price moves
        total_volume = sum(volumes)
        if total_volume == 0:
            return statistics.mean(price_moves)
        
        weighted_toxicity = sum(move * vol for move, vol in zip(price_moves, volumes)) / total_volume
        return min(weighted_toxicity * 10, 1.0)  # Scale to 0-1 range
    
    async def _calculate_lambda_proxy(self, asset_history: List[Dict[str, Any]]) -> float:
        """Calculate Kyle's Lambda proxy"""
        if len(asset_history) < 20:
            return 0.0
        
        # Simplified lambda: price impact per unit volume
        price_impacts = []
        
        for i in range(2, len(asset_history)):
            prev_price = float(asset_history[i-2].get('price', asset_history[i-2].get('close', 0)))
            curr_price = float(asset_history[i].get('price', asset_history[i].get('close', 0)))
            volume = float(asset_history[i-1].get('volume', 0))
            
            if prev_price > 0 and volume > 0:
                price_change = abs(curr_price - prev_price) / prev_price
                impact_per_volume = price_change / np.sqrt(volume)
                price_impacts.append(impact_per_volume)
        
        return min(statistics.mean(price_impacts) * 1000, 1.0) if price_impacts else 0.0
    
    async def _calculate_pin_proxy(self, asset_history: List[Dict[str, Any]]) -> float:
        """Calculate PIN proxy from buy/sell order imbalance"""
        if len(asset_history) < 10:
            return 0.0
        
        # Proxy PIN using price direction and volume
        buy_volume = 0
        sell_volume = 0
        
        for i in range(1, len(asset_history)):
            prev_price = float(asset_history[i-1].get('price', asset_history[i-1].get('close', 0)))
            curr_price = float(asset_history[i].get('price', asset_history[i].get('close', 0)))
            volume = float(asset_history[i].get('volume', 0))
            
            if curr_price > prev_price:
                buy_volume += volume
            elif curr_price < prev_price:
                sell_volume += volume
        
        total_volume = buy_volume + sell_volume
        if total_volume == 0:
            return 0.0
        
        imbalance = abs(buy_volume - sell_volume) / total_volume
        return min(imbalance * 2, 1.0)  # Scale to 0-1 range
    
    async def _calculate_spread_proxy(self, asset_history: List[Dict[str, Any]]) -> float:
        """Calculate bid-ask spread toxicity proxy"""
        # For historical data without spread, use high-low range as proxy
        spreads = []
        
        for data in asset_history:
            high = float(data.get('high', data.get('price', 0)))
            low = float(data.get('low', data.get('price', 0)))
            mid = (high + low) / 2 if high > 0 and low > 0 else float(data.get('price', data.get('close', 0)))
            
            if mid > 0:
                spread_proxy = (high - low) / mid
                spreads.append(spread_proxy)
        
        return min(statistics.mean(spreads) * 5, 1.0) if spreads else 0.0
    
    async def _calculate_fade_proxy(self, asset_history: List[Dict[str, Any]]) -> float:
        """Calculate quote fade proxy from volume patterns"""
        if len(asset_history) < 5:
            return 0.0
        
        # Proxy fade using volume decay after price moves
        fade_scores = []
        
        for i in range(2, len(asset_history) - 2):
            prev_vol = float(asset_history[i-1].get('volume', 0))
            curr_vol = float(asset_history[i].get('volume', 0))
            next_vol = float(asset_history[i+1].get('volume', 0))
            
            if curr_vol > 0 and prev_vol > 0:
                vol_fade = 1 - (next_vol / curr_vol) if next_vol < curr_vol else 0
                fade_scores.append(vol_fade)
        
        return min(statistics.mean(fade_scores), 1.0) if fade_scores else 0.0
    
    def _calculate_composite_toxicity(self, vpin: float, lambda_score: float, pin: float, 
                                    spread: float, fade: float) -> float:
        """Calculate weighted composite toxicity score"""
        # Weights based on research importance
        weights = {
            'vpin': 0.30,
            'lambda': 0.25,
            'pin': 0.20,
            'spread': 0.15,
            'fade': 0.10
        }
        
        composite = (
            weights['vpin'] * vpin +
            weights['lambda'] * lambda_score +
            weights['pin'] * pin +
            weights['spread'] * spread +
            weights['fade'] * fade
        )
        
        return min(composite, 1.0)
    
    async def _analyze_toxicity_trend(self, symbol: str, current_toxicity: float) -> str:
        """Analyze toxicity trend for an asset"""
        if len(self.toxicity_history[symbol]) < 3:
            return "INSUFFICIENT_DATA"
        
        recent_values = [m.composite_toxicity for m in list(self.toxicity_history[symbol])[-5:]]
        trend_slope = np.polyfit(range(len(recent_values)), recent_values, 1)[0]
        
        if trend_slope > 0.05:
            return "INCREASING"
        elif trend_slope < -0.05:
            return "DECREASING"
        else:
            return "STABLE"
    
    def _classify_risk_level(self, toxicity: float) -> str:
        """Classify individual asset risk level"""
        if toxicity < 0.25:
            return "LOW"
        elif toxicity < 0.50:
            return "MODERATE"
        elif toxicity < 0.75:
            return "HIGH"
        else:
            return "EXTREME"
    
    def _classify_comovement_strength(self, correlation: float) -> str:
        """Classify comovement strength"""
        if correlation < 0.3:
            return "WEAK"
        elif correlation < 0.5:
            return "MODERATE"
        elif correlation < 0.7:
            return "STRONG"
        else:
            return "VERY_STRONG"
    
    def _analyze_lead_lag(self, series1: List[float], series2: List[float], 
                         asset1: str, asset2: str) -> Dict[str, float]:
        """Analyze lead-lag relationship between two assets"""
        
        # Cross-correlation analysis with lags
        max_lag = min(5, len(series1) // 3)
        correlations = {}
        
        for lag in range(-max_lag, max_lag + 1):
            if lag == 0:
                corr, _ = pearsonr(series1, series2)
            elif lag > 0:
                # asset1 leads asset2
                if len(series1) > lag and len(series2) > lag:
                    corr, _ = pearsonr(series1[:-lag], series2[lag:])
                else:
                    continue
            else:  # lag < 0
                # asset2 leads asset1
                abs_lag = abs(lag)
                if len(series1) > abs_lag and len(series2) > abs_lag:
                    corr, _ = pearsonr(series1[abs_lag:], series2[:-abs_lag])
                else:
                    continue
            
            correlations[lag] = abs(corr)
        
        if not correlations:
            return {asset1: 0.0, asset2: 0.0}
        
        # Find lag with highest correlation
        best_lag = max(correlations.keys(), key=lambda k: correlations[k])
        
        if best_lag > 0:
            return {asset1: 1.0, asset2: 0.0}  # asset1 leads
        elif best_lag < 0:
            return {asset1: 0.0, asset2: 1.0}  # asset2 leads
        else:
            return {asset1: 0.5, asset2: 0.5}  # simultaneous
    
    def _estimate_contagion_probability(self, correlation: float, series1: List[float], 
                                      series2: List[float]) -> float:
        """Estimate probability of contagion between assets"""
        
        # High correlation suggests higher contagion probability
        correlation_factor = abs(correlation)
        
        # Volatility factor (higher volatility = higher contagion risk)
        vol1 = np.std(series1) if len(series1) > 1 else 0
        vol2 = np.std(series2) if len(series2) > 1 else 0
        volatility_factor = min((vol1 + vol2) / 2, 0.5)
        
        # Toxicity level factor
        toxicity_factor = min((statistics.mean(series1) + statistics.mean(series2)) / 2, 0.5)
        
        # Combined probability
        contagion_prob = (
            0.5 * correlation_factor +
            0.3 * volatility_factor +
            0.2 * toxicity_factor
        )
        
        return min(contagion_prob, 1.0)
    
    def _estimate_propagation_time(self, history1: List[AssetToxicityMetrics], 
                                  history2: List[AssetToxicityMetrics]) -> Optional[float]:
        """Estimate time for toxicity to propagate between assets"""
        
        if len(history1) < 5 or len(history2) < 5:
            return None
        
        # Look for toxicity spikes and measure time difference
        toxicity_spikes = []
        
        for i, (h1, h2) in enumerate(zip(history1, history2)):
            if h1.composite_toxicity > 0.6:  # Toxicity spike threshold
                toxicity_spikes.append(('asset1', i, h1.timestamp))
            if h2.composite_toxicity > 0.6:
                toxicity_spikes.append(('asset2', i, h2.timestamp))
        
        if len(toxicity_spikes) < 2:
            return None
        
        # Find consecutive spikes between different assets
        for i in range(len(toxicity_spikes) - 1):
            spike1 = toxicity_spikes[i]
            spike2 = toxicity_spikes[i + 1]
            
            if spike1[0] != spike2[0]:  # Different assets
                time_diff = abs(spike2[2] - spike1[2]) * 1000  # Convert to milliseconds
                if time_diff < 300000:  # Less than 5 minutes
                    return time_diff
        
        return None
    
    def _calculate_systemic_risk_score(self, individual_metrics: Dict[str, AssetToxicityMetrics],
                                     network_density: float, assets_above_threshold: int) -> float:
        """Calculate overall systemic risk score"""
        
        # Component 1: Average toxicity level
        avg_toxicity = statistics.mean([m.composite_toxicity for m in individual_metrics.values()])
        
        # Component 2: Proportion of assets above threshold
        proportion_above_threshold = assets_above_threshold / len(individual_metrics)
        
        # Component 3: Network connectivity
        connectivity_factor = network_density
        
        # Component 4: Extreme values factor
        extreme_assets = sum(1 for m in individual_metrics.values() if m.composite_toxicity > 0.8)
        extreme_factor = extreme_assets / len(individual_metrics)
        
        # Weighted combination
        systemic_score = (
            0.3 * avg_toxicity +
            0.3 * proportion_above_threshold +
            0.2 * connectivity_factor +
            0.2 * extreme_factor
        )
        
        return min(systemic_score, 1.0)
    
    def _identify_contagion_pathways(self) -> List[List[str]]:
        """Identify potential contagion pathways through network analysis"""
        
        if len(self.network_graph.nodes) < 3:
            return []
        
        pathways = []
        
        # Find paths between high-toxicity nodes
        high_risk_nodes = []
        for node in self.network_graph.nodes():
            # Check if node has high centrality or high connections
            degree = self.network_graph.degree(node)
            if degree > len(self.network_graph.nodes) * 0.3:  # Highly connected
                high_risk_nodes.append(node)
        
        # Find shortest paths between high-risk nodes
        for i, node1 in enumerate(high_risk_nodes):
            for node2 in high_risk_nodes[i+1:]:
                try:
                    path = nx.shortest_path(self.network_graph, node1, node2)
                    if len(path) <= 4:  # Reasonable path length
                        pathways.append(path)
                except nx.NetworkXNoPath:
                    continue
        
        return pathways[:10]  # Return top 10 pathways
    
    def _classify_systemic_risk_level(self, systemic_score: float, assets_above_threshold: int) -> str:
        """Classify overall systemic risk level"""
        
        if systemic_score > self.config.systemic_risk_threshold and assets_above_threshold > 2:
            return "SYSTEMIC"
        elif systemic_score > 0.5:
            return "HIGH"
        elif systemic_score > 0.3:
            return "MODERATE"
        else:
            return "LOW"
    
    def _detect_early_warning_signals(self, individual_metrics: Dict[str, AssetToxicityMetrics],
                                    comovement_metrics: Dict[Tuple[str, str], TPMAComovementMetrics]) -> List[str]:
        """Detect early warning signals for systemic risk"""
        
        warnings = []
        
        # Check for increasing toxicity trends
        increasing_assets = [asset for asset, metrics in individual_metrics.items()
                           if metrics.toxicity_trend == "INCREASING"]
        if len(increasing_assets) > len(individual_metrics) * 0.5:
            warnings.append(f"Rising toxicity trend in {len(increasing_assets)} assets")
        
        # Check for high correlations
        high_correlations = sum(1 for comovement in comovement_metrics.values()
                              if comovement.comovement_strength in ["STRONG", "VERY_STRONG"])
        if high_correlations > len(comovement_metrics) * 0.3:
            warnings.append(f"High correlation detected in {high_correlations} asset pairs")
        
        # Check for contagion risks
        high_contagion_risk = sum(1 for comovement in comovement_metrics.values()
                                if comovement.contagion_probability > 0.7)
        if high_contagion_risk > 0:
            warnings.append(f"High contagion risk in {high_contagion_risk} asset pairs")
        
        # Check network centrality
        if len(self.network_graph.nodes) > 3:
            centrality = nx.degree_centrality(self.network_graph)
            high_centrality_nodes = [node for node, cent in centrality.items()
                                   if cent > self.config.network_centrality_threshold]
            if high_centrality_nodes:
                warnings.append(f"High centrality nodes detected: {high_centrality_nodes}")
        
        return warnings
    
    def _identify_central_risk_nodes(self) -> List[str]:
        """Identify most central/risky nodes in the network"""
        
        if len(self.network_graph.nodes) < 2:
            return []
        
        # Calculate various centrality measures
        degree_centrality = nx.degree_centrality(self.network_graph)
        
        try:
            betweenness_centrality = nx.betweenness_centrality(self.network_graph)
            eigenvector_centrality = nx.eigenvector_centrality(self.network_graph, max_iter=1000)
        except:
            betweenness_centrality = {}
            eigenvector_centrality = {}
        
        # Combine centrality measures
        combined_centrality = {}
        for node in self.network_graph.nodes():
            score = (
                degree_centrality.get(node, 0) * 0.4 +
                betweenness_centrality.get(node, 0) * 0.3 +
                eigenvector_centrality.get(node, 0) * 0.3
            )
            combined_centrality[node] = score
        
        # Return top nodes by centrality
        sorted_nodes = sorted(combined_centrality.items(), key=lambda x: x[1], reverse=True)
        return [node for node, score in sorted_nodes[:5] if score > 0.1]
    
    def _determine_alert_level(self, systemic_metrics: SystemicRiskMetrics) -> str:
        """Determine overall system alert level"""
        
        if systemic_metrics.risk_level == "SYSTEMIC":
            return "CRITICAL"
        elif systemic_metrics.systemic_risk_score > 0.7:
            return "HIGH"
        elif systemic_metrics.systemic_risk_score > 0.4:
            return "ELEVATED"
        else:
            return "NORMAL"
    
    def _generate_recommendations(self, systemic_metrics: SystemicRiskMetrics,
                                individual_metrics: Dict[str, AssetToxicityMetrics]) -> List[str]:
        """Generate actionable recommendations based on TPMA analysis"""
        
        recommendations = []
        
        # System-level recommendations
        if systemic_metrics.risk_level == "SYSTEMIC":
            recommendations.append("CRITICAL: Consider reducing overall portfolio risk exposure")
            recommendations.append("Implement enhanced monitoring for all tracked assets")
        
        # High-risk asset recommendations
        high_risk_assets = [asset for asset, metrics in individual_metrics.items()
                          if metrics.risk_level in ["HIGH", "EXTREME"]]
        if high_risk_assets:
            recommendations.append(f"Monitor high-risk assets closely: {', '.join(high_risk_assets)}")
        
        # Network-based recommendations
        if systemic_metrics.network_density > 0.6:
            recommendations.append("High network connectivity detected - consider diversification")
        
        # Central node recommendations
        if systemic_metrics.central_risk_nodes:
            recommendations.append(f"Focus risk management on central nodes: {', '.join(systemic_metrics.central_risk_nodes)}")
        
        # Early warning recommendations
        if systemic_metrics.early_warning_signals:
            recommendations.append("Early warning signals active - increase monitoring frequency")
        
        return recommendations
    
    def _serialize_network_graph(self) -> Dict[str, Any]:
        """Serialize NetworkX graph for storage/transmission"""
        
        if len(self.network_graph.nodes) == 0:
            return {"nodes": [], "edges": []}
        
        nodes = list(self.network_graph.nodes())
        edges = [(u, v, data) for u, v, data in self.network_graph.edges(data=True)]
        
        return {
            "nodes": nodes,
            "edges": edges,
            "node_count": len(nodes),
            "edge_count": len(edges),
            "density": nx.density(self.network_graph)
        }
    
    # Default metrics creation methods
    def _create_default_individual_metrics(self, symbol: str) -> AssetToxicityMetrics:
        """Create default individual metrics when insufficient data"""
        return AssetToxicityMetrics(
            timestamp=datetime.now().timestamp(),
            symbol=symbol,
            vpin_toxicity=0.0,
            kyles_lambda=0.0,
            pin_score=0.0,
            spread_toxicity=0.0,
            quote_fade_score=0.0,
            composite_toxicity=0.0,
            toxicity_trend="INSUFFICIENT_DATA",
            risk_level="UNKNOWN"
        )
    
    def _create_default_comovement_metrics(self, asset1: str, asset2: str) -> TPMAComovementMetrics:
        """Create default comovement metrics when insufficient data"""
        return TPMAComovementMetrics(
            asset_pair=(asset1, asset2),
            correlation_coefficient=0.0,
            correlation_p_value=1.0,
            comovement_strength="INSUFFICIENT_DATA",
            lead_lag_relationship={asset1: 0.0, asset2: 0.0},
            contagion_probability=0.0,
            time_to_propagation_ms=None
        )
    
    def _create_default_systemic_metrics(self) -> SystemicRiskMetrics:
        """Create default systemic metrics when insufficient data"""
        return SystemicRiskMetrics(
            timestamp=datetime.now().timestamp(),
            total_assets_tracked=0,
            assets_above_threshold=0,
            network_density=0.0,
            systemic_risk_score=0.0,
            dominant_cluster_size=0,
            contagion_pathways=[],
            risk_level="INSUFFICIENT_DATA",
            early_warning_signals=[],
            central_risk_nodes=[]
        )
    
    # TPMA Mathematical Framework Helper Methods
    
    def _standardize_series(self, series: List[float]) -> List[float]:
        """Standardize time series to zero mean and unit variance (required for TPMA)"""
        if len(series) < 2:
            return series
        
        mean_val = statistics.mean(series)
        std_val = statistics.stdev(series) if len(series) > 1 else 1.0
        
        if std_val == 0:
            return [0.0] * len(series)
        
        return [(x - mean_val) / std_val for x in series]
    
    def _filter_by_volume_bucket(self, data: List[Dict[str, Any]], bucket: str) -> List[Dict[str, Any]]:
        """Filter data by volume bucket for volume-bucket-wise TPMA analysis"""
        if not data:
            return []
        
        # Calculate volume quartiles
        volumes = [float(d.get('volume', 0)) for d in data if d.get('volume', 0) > 0]
        if not volumes:
            return []
        
        q25 = np.percentile(volumes, 25)
        q75 = np.percentile(volumes, 75)
        
        filtered_data = []
        for d in data:
            volume = float(d.get('volume', 0))
            if bucket == 'small' and volume <= q25:
                filtered_data.append(d)
            elif bucket == 'medium' and q25 < volume <= q75:
                filtered_data.append(d)
            elif bucket == 'large' and volume > q75:
                filtered_data.append(d)
        
        return filtered_data
    
    def _calculate_order_imbalance(self, data: Dict[str, Any]) -> float:
        """Calculate proportional order imbalance for a single data point"""
        # Proxy for order imbalance using price change and volume
        price = float(data.get('price', data.get('close', 0)))
        volume = float(data.get('volume', 0))
        
        # Simple order imbalance proxy: signed volume based on price direction
        # In practice, would use actual bid/ask order flow data
        if 'price_change' in data:
            price_change = float(data['price_change'])
        else:
            price_change = 0.0  # Would need previous price to calculate
        
        # Imbalance = sign(price_change) * sqrt(volume)
        if price_change > 0:
            return np.sqrt(volume) if volume > 0 else 0.0
        elif price_change < 0:
            return -np.sqrt(volume) if volume > 0 else 0.0
        else:
            return 0.0
    
    def _classify_tpma_strength(self, tpma_coef: float) -> str:
        """Classify TPMA comovement strength (TPMA-specific thresholds)"""
        # TPMA coefficients are bounded in (-1, 1]
        # Research suggests different thresholds than traditional correlation
        if tpma_coef < 0.2:
            return "WEAK"
        elif tpma_coef < 0.4:
            return "MODERATE"
        elif tpma_coef < 0.6:
            return "STRONG"
        else:
            return "VERY_STRONG"
    
    async def _analyze_tpma_lead_lag(self, series1: List[float], series2: List[float], 
                                   asset1: str, asset2: str) -> Dict[str, float]:
        """Enhanced lead-lag analysis using cross-scale TPMA"""
        
        # Multi-thickness lead-lag analysis
        max_lag = min(5, len(series1) // 5)
        thickness_range = [2, 5, 10]  # Multiple thickness parameters
        
        lead_lag_scores = {asset1: 0.0, asset2: 0.0}
        total_weight = 0.0
        
        for thickness in thickness_range:
            if len(series1) > thickness + max_lag and len(series2) > thickness + max_lag:
                for lag in range(-max_lag, max_lag + 1):
                    if lag == 0:
                        continue
                    
                    # Create lagged series
                    if lag > 0:
                        # asset1 leads asset2
                        series1_lagged = series1[:-lag]
                        series2_lagged = series2[lag:]
                    else:
                        # asset2 leads asset1
                        abs_lag = abs(lag)
                        series1_lagged = series1[abs_lag:]
                        series2_lagged = series2[:-abs_lag]
                    
                    if len(series1_lagged) >= thickness + 5:
                        # Calculate TPMA coefficient for lagged series
                        tpma_coef = await self.calculate_tpma_coefficient(
                            series1_lagged, series2_lagged, thickness
                        )
                        
                        weight = abs(tpma_coef)
                        total_weight += weight
                        
                        if lag > 0 and weight > 0:
                            lead_lag_scores[asset1] += weight  # asset1 leads
                        elif lag < 0 and weight > 0:
                            lead_lag_scores[asset2] += weight  # asset2 leads
        
        # Normalize scores
        if total_weight > 0:
            lead_lag_scores[asset1] /= total_weight
            lead_lag_scores[asset2] /= total_weight
            
            # Ensure scores sum to 1
            total_score = lead_lag_scores[asset1] + lead_lag_scores[asset2]
            if total_score > 0:
                lead_lag_scores[asset1] /= total_score
                lead_lag_scores[asset2] /= total_score
            else:
                lead_lag_scores = {asset1: 0.5, asset2: 0.5}
        else:
            lead_lag_scores = {asset1: 0.5, asset2: 0.5}
        
        return lead_lag_scores
    
    def _estimate_tpma_contagion_probability(self, tpma_coef: float, series1: List[float], 
                                           series2: List[float], volume_bucket_tpma: Dict[str, float]) -> float:
        """Estimate contagion probability using TPMA dynamics and volume analysis"""
        
        # Base probability from TPMA coefficient
        tpma_factor = abs(tpma_coef) * 0.4
        
        # Volatility factor (higher volatility = higher contagion risk)
        vol1 = np.std(series1) if len(series1) > 1 else 0
        vol2 = np.std(series2) if len(series2) > 1 else 0
        volatility_factor = min((vol1 + vol2) / 4, 0.3)  # Cap at 0.3
        
        # Toxicity level factor
        toxicity_factor = min((statistics.mean(series1) + statistics.mean(series2)) / 4, 0.2)
        
        # Volume bucket consistency factor
        volume_consistency_factor = 0.0
        if volume_bucket_tpma:
            # If TPMA is consistent across volume buckets, higher contagion risk
            bucket_values = list(volume_bucket_tpma.values())
            if len(bucket_values) > 1:
                consistency = 1.0 - (np.std(bucket_values) / (np.mean([abs(v) for v in bucket_values]) + 1e-8))
                volume_consistency_factor = min(consistency * 0.1, 0.1)
        
        # Combined probability
        contagion_prob = tpma_factor + volatility_factor + toxicity_factor + volume_consistency_factor
        
        return min(contagion_prob, 1.0)
    
    async def _estimate_tpma_propagation_time(self, history1: List[AssetToxicityMetrics], 
                                            history2: List[AssetToxicityMetrics],
                                            tpma_coef: float) -> Optional[float]:
        """Estimate propagation time using TPMA-based change detection"""
        
        if len(history1) < 10 or len(history2) < 10 or abs(tpma_coef) < 0.3:
            return None
        
        # Extract timestamps and toxicity levels
        times1 = [h.timestamp for h in history1]
        toxicity1 = [h.composite_toxicity for h in history1]
        times2 = [h.timestamp for h in history2]
        toxicity2 = [h.composite_toxicity for h in history2]
        
        # Detect rapid changes in toxicity (potential flash crash precursors)
        change_threshold = 0.3  # 30% toxicity increase
        
        # Find significant toxicity spikes
        spikes1 = []
        spikes2 = []
        
        for i in range(1, len(toxicity1)):
            if toxicity1[i] - toxicity1[i-1] > change_threshold:
                spikes1.append((times1[i], toxicity1[i]))
        
        for i in range(1, len(toxicity2)):
            if toxicity2[i] - toxicity2[i-1] > change_threshold:
                spikes2.append((times2[i], toxicity2[i]))
        
        # Find closest spike pairs and calculate propagation time
        min_time_diff = float('inf')
        
        for time1, _ in spikes1:
            for time2, _ in spikes2:
                time_diff = abs(time2 - time1)
                if time_diff < min_time_diff and time_diff > 0:
                    min_time_diff = time_diff
        
        if min_time_diff != float('inf') and min_time_diff < 600:  # Less than 10 minutes
            return min_time_diff * 1000  # Convert to milliseconds
        
        return None

# Global tracker instance
tpma_tracker = TPMAMultiAssetTracker()