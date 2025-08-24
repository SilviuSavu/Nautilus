"""
Intelligent Edge Placement Optimizer for Regional Trading Operations

This module optimizes edge node placement based on trading activity patterns,
latency requirements, and market dynamics for maximum performance efficiency.
"""

import asyncio
import logging
import math
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple, Any, Set
from datetime import datetime, timedelta
import numpy as np


class PlacementStrategy(Enum):
    """Edge node placement strategies"""
    LATENCY_OPTIMIZED = "latency_optimized"      # Minimize latency to markets
    COST_OPTIMIZED = "cost_optimized"            # Minimize deployment costs
    BALANCED = "balanced"                        # Balance latency and cost
    TRADING_VOLUME = "trading_volume"            # Based on trading volume patterns
    GEOGRAPHIC_SPREAD = "geographic_spread"      # Maximum geographic coverage
    MARKET_HOURS = "market_hours"                # Optimized for market trading hours


class TradingActivityType(Enum):
    """Types of trading activities for placement optimization"""
    HIGH_FREQUENCY = "high_frequency"           # Ultra-low latency requirements
    ALGORITHMIC = "algorithmic"                 # Moderate latency requirements
    DISCRETIONARY = "discretionary"             # Standard latency requirements
    MARKET_MAKING = "market_making"             # Ultra-low latency + high volume
    ARBITRAGE = "arbitrage"                     # Cross-market latency sensitive
    EXECUTION = "execution"                     # Order execution focused


@dataclass
class TradingActivityPattern:
    """Trading activity pattern for placement optimization"""
    activity_id: str
    activity_type: TradingActivityType
    
    # Geographic requirements
    primary_markets: List[str]  # e.g., ["NYSE", "NASDAQ", "LSE"]
    secondary_markets: List[str] = field(default_factory=list)
    
    # Latency requirements
    max_latency_us: float = 1000.0
    target_latency_us: float = 500.0
    
    # Volume characteristics
    peak_volume_per_second: float = 10000.0
    average_volume_per_second: float = 1000.0
    
    # Temporal patterns
    active_hours_utc: List[Tuple[int, int]] = field(default_factory=list)  # [(9, 16), (21, 5)]
    peak_hours_utc: List[Tuple[int, int]] = field(default_factory=list)
    
    # Resource requirements
    cpu_cores_required: int = 4
    memory_gb_required: int = 16
    network_bandwidth_mbps: float = 1000.0
    
    # Criticality and priority
    business_priority: int = 5  # 1-10 scale
    sla_requirement: float = 0.9999  # 99.99% availability


@dataclass 
class MarketConnectivityRequirement:
    """Market connectivity requirements for trading"""
    market_code: str  # e.g., "NYSE", "NASDAQ", "LSE"
    market_name: str
    
    # Geographic location
    region: str
    country: str
    timezone: str
    
    # Connectivity requirements
    required_latency_us: float
    required_bandwidth_mbps: float
    required_availability: float
    
    # Market characteristics
    trading_hours_local: Tuple[int, int]  # (9, 16) for 9 AM - 4 PM
    market_data_volume_mbps: float
    order_volume_per_second: float
    
    # Access requirements
    direct_market_access: bool = False
    co_location_available: bool = False
    special_connectivity_required: bool = False


@dataclass
class EdgePlacementCandidate:
    """Candidate location for edge node placement"""
    candidate_id: str
    region_name: str
    
    # Geographic information
    latitude: float
    longitude: float
    country: str
    city: str
    
    # Infrastructure characteristics  
    cloud_provider: str  # "aws", "gcp", "azure", "bare_metal"
    availability_zone: str
    instance_types_available: List[str]
    network_tier: str  # "premium", "standard"
    
    # Performance characteristics
    estimated_latency_to_markets: Dict[str, float]  # market_code -> latency_us
    available_bandwidth_gbps: float
    estimated_monthly_cost_usd: float
    
    # Suitability scores
    latency_score: float = 0.0      # 0.0 - 1.0
    cost_score: float = 0.0         # 0.0 - 1.0  
    availability_score: float = 0.0 # 0.0 - 1.0
    overall_score: float = 0.0      # 0.0 - 1.0


@dataclass
class PlacementOptimizationResult:
    """Result of edge placement optimization"""
    optimization_id: str
    strategy_used: PlacementStrategy
    
    # Optimization parameters
    total_candidates_evaluated: int
    total_activities_considered: int
    optimization_time_seconds: float
    
    # Recommended placements
    recommended_placements: List[EdgePlacementCandidate]
    placement_rationale: Dict[str, str]  # candidate_id -> rationale
    
    # Performance projections
    projected_average_latency_us: float
    projected_total_cost_usd_monthly: float
    projected_availability: float
    projected_capacity_utilization: float
    
    # Alternative options
    alternative_strategies: List[Dict[str, Any]]
    cost_benefit_analysis: Dict[str, float]


class EdgePlacementOptimizer:
    """
    Intelligent Edge Placement Optimizer for Trading Operations
    
    Optimizes edge node placement based on:
    - Trading activity patterns and volume
    - Latency requirements to major markets  
    - Cost optimization across cloud providers
    - Geographic distribution for resilience
    - Market hours and temporal patterns
    - Regulatory and compliance requirements
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Trading activity patterns
        self.trading_patterns: Dict[str, TradingActivityPattern] = {}
        
        # Market connectivity requirements
        self.market_requirements: Dict[str, MarketConnectivityRequirement] = {}
        
        # Placement candidates
        self.placement_candidates: Dict[str, EdgePlacementCandidate] = {}
        
        # Optimization history
        self.optimization_history: List[PlacementOptimizationResult] = []
        
        # Initialize default configurations
        self._initialize_market_requirements()
        self._initialize_placement_candidates()
        
        self.logger.info("Edge Placement Optimizer initialized")
    
    def _initialize_market_requirements(self):
        """Initialize major market connectivity requirements"""
        
        markets = [
            {
                "code": "NYSE", "name": "New York Stock Exchange",
                "region": "us_east", "country": "USA", "timezone": "America/New_York",
                "latency_us": 50.0, "bandwidth_mbps": 10000.0, "availability": 0.99999,
                "hours": (9, 16), "data_volume": 5000.0, "order_volume": 50000.0,
                "direct_access": True, "co_location": True
            },
            {
                "code": "NASDAQ", "name": "NASDAQ",
                "region": "us_east", "country": "USA", "timezone": "America/New_York", 
                "latency_us": 50.0, "bandwidth_mbps": 8000.0, "availability": 0.99999,
                "hours": (9, 16), "data_volume": 4000.0, "order_volume": 45000.0,
                "direct_access": True, "co_location": True
            },
            {
                "code": "LSE", "name": "London Stock Exchange",
                "region": "eu_west", "country": "UK", "timezone": "Europe/London",
                "latency_us": 100.0, "bandwidth_mbps": 5000.0, "availability": 0.9999,
                "hours": (8, 16), "data_volume": 2000.0, "order_volume": 20000.0,
                "direct_access": True, "co_location": True  
            },
            {
                "code": "TSE", "name": "Tokyo Stock Exchange",
                "region": "ap_northeast", "country": "Japan", "timezone": "Asia/Tokyo",
                "latency_us": 100.0, "bandwidth_mbps": 3000.0, "availability": 0.9999,
                "hours": (9, 15), "data_volume": 1500.0, "order_volume": 15000.0,
                "direct_access": False, "co_location": False
            },
            {
                "code": "HKEX", "name": "Hong Kong Exchange",
                "region": "ap_southeast", "country": "Hong Kong", "timezone": "Asia/Hong_Kong",
                "latency_us": 200.0, "bandwidth_mbps": 2000.0, "availability": 0.999,
                "hours": (9, 16), "data_volume": 1000.0, "order_volume": 10000.0,
                "direct_access": False, "co_location": False
            },
            {
                "code": "CME", "name": "Chicago Mercantile Exchange",
                "region": "us_central", "country": "USA", "timezone": "America/Chicago",
                "latency_us": 75.0, "bandwidth_mbps": 6000.0, "availability": 0.99999,
                "hours": (8, 17), "data_volume": 3000.0, "order_volume": 30000.0,
                "direct_access": True, "co_location": True
            }
        ]
        
        for market in markets:
            requirement = MarketConnectivityRequirement(
                market_code=market["code"],
                market_name=market["name"],
                region=market["region"],
                country=market["country"],
                timezone=market["timezone"],
                required_latency_us=market["latency_us"],
                required_bandwidth_mbps=market["bandwidth_mbps"], 
                required_availability=market["availability"],
                trading_hours_local=market["hours"],
                market_data_volume_mbps=market["data_volume"],
                order_volume_per_second=market["order_volume"],
                direct_market_access=market["direct_access"],
                co_location_available=market["co_location"]
            )
            
            self.market_requirements[market["code"]] = requirement
        
        self.logger.info(f"Initialized {len(self.market_requirements)} market connectivity requirements")
    
    def _initialize_placement_candidates(self):
        """Initialize edge placement candidate locations"""
        
        candidates = [
            # Ultra-low latency candidates (market proximity)
            {
                "id": "nyse_mahwah", "region": "NYSE Mahwah", 
                "lat": 41.0875, "lng": -74.1444, "country": "USA", "city": "Mahwah, NJ",
                "provider": "bare_metal", "az": "co_location", 
                "instances": ["dedicated_servers"], "tier": "premium",
                "market_latencies": {"NYSE": 10.0, "NASDAQ": 15.0, "CME": 800.0},
                "bandwidth": 100.0, "cost": 15000.0
            },
            {
                "id": "nasdaq_carteret", "region": "NASDAQ Carteret",
                "lat": 40.6076, "lng": -74.2272, "country": "USA", "city": "Carteret, NJ", 
                "provider": "bare_metal", "az": "co_location",
                "instances": ["dedicated_servers"], "tier": "premium",
                "market_latencies": {"NASDAQ": 10.0, "NYSE": 20.0, "CME": 850.0},
                "bandwidth": 100.0, "cost": 15000.0
            },
            {
                "id": "cme_chicago", "region": "CME Chicago",
                "lat": 41.8781, "lng": -87.6298, "country": "USA", "city": "Chicago, IL",
                "provider": "bare_metal", "az": "co_location", 
                "instances": ["dedicated_servers"], "tier": "premium",
                "market_latencies": {"CME": 10.0, "NYSE": 800.0, "NASDAQ": 850.0},
                "bandwidth": 100.0, "cost": 12000.0
            },
            {
                "id": "lse_basildon", "region": "LSE Basildon",
                "lat": 51.5707, "lng": 0.4889, "country": "UK", "city": "Basildon",
                "provider": "bare_metal", "az": "co_location",
                "instances": ["dedicated_servers"], "tier": "premium", 
                "market_latencies": {"LSE": 15.0, "NYSE": 6000.0, "NASDAQ": 6100.0},
                "bandwidth": 50.0, "cost": 8000.0
            },
            
            # Cloud region candidates
            {
                "id": "aws_us_east_1", "region": "AWS US East 1",
                "lat": 38.9517, "lng": -77.4481, "country": "USA", "city": "Virginia",
                "provider": "aws", "az": "us-east-1a",
                "instances": ["c6i.4xlarge", "c6i.8xlarge", "c6i.16xlarge"], "tier": "premium",
                "market_latencies": {"NYSE": 500.0, "NASDAQ": 450.0, "CME": 1200.0},
                "bandwidth": 25.0, "cost": 2000.0
            },
            {
                "id": "aws_us_west_1", "region": "AWS US West 1", 
                "lat": 37.3861, "lng": -122.0839, "country": "USA", "city": "California",
                "provider": "aws", "az": "us-west-1a",
                "instances": ["c6i.4xlarge", "c6i.8xlarge"], "tier": "premium",
                "market_latencies": {"NYSE": 2500.0, "NASDAQ": 2400.0, "CME": 2000.0},
                "bandwidth": 25.0, "cost": 2200.0
            },
            {
                "id": "gcp_eu_west_1", "region": "GCP EU West 1",
                "lat": 53.4084, "lng": -6.2719, "country": "Ireland", "city": "Dublin", 
                "provider": "gcp", "az": "eu-west-1-a",
                "instances": ["c2-standard-8", "c2-standard-16"], "tier": "premium",
                "market_latencies": {"LSE": 200.0, "NYSE": 6500.0, "NASDAQ": 6600.0},
                "bandwidth": 20.0, "cost": 1800.0
            },
            {
                "id": "azure_ap_northeast_1", "region": "Azure AP Northeast 1",
                "lat": 35.6762, "lng": 139.6503, "country": "Japan", "city": "Tokyo",
                "provider": "azure", "az": "japaneast-1",
                "instances": ["Standard_F8s_v2", "Standard_F16s_v2"], "tier": "standard",
                "market_latencies": {"TSE": 150.0, "HKEX": 2000.0, "NYSE": 15000.0},
                "bandwidth": 15.0, "cost": 2500.0
            },
            {
                "id": "gcp_ap_southeast_1", "region": "GCP AP Southeast 1", 
                "lat": 1.3521, "lng": 103.8198, "country": "Singapore", "city": "Singapore",
                "provider": "gcp", "az": "asia-southeast1-a",
                "instances": ["c2-standard-4", "c2-standard-8"], "tier": "premium",
                "market_latencies": {"HKEX": 800.0, "TSE": 4000.0, "NYSE": 18000.0},
                "bandwidth": 18.0, "cost": 2800.0
            }
        ]
        
        for candidate in candidates:
            placement_candidate = EdgePlacementCandidate(
                candidate_id=candidate["id"],
                region_name=candidate["region"],
                latitude=candidate["lat"],
                longitude=candidate["lng"], 
                country=candidate["country"],
                city=candidate["city"],
                cloud_provider=candidate["provider"],
                availability_zone=candidate["az"],
                instance_types_available=candidate["instances"],
                network_tier=candidate["tier"],
                estimated_latency_to_markets=candidate["market_latencies"],
                available_bandwidth_gbps=candidate["bandwidth"],
                estimated_monthly_cost_usd=candidate["cost"]
            )
            
            self.placement_candidates[candidate["id"]] = placement_candidate
        
        self.logger.info(f"Initialized {len(self.placement_candidates)} placement candidates")
    
    def add_trading_activity_pattern(self, pattern: TradingActivityPattern):
        """Add trading activity pattern for placement optimization"""
        
        self.trading_patterns[pattern.activity_id] = pattern
        self.logger.info(f"Added trading activity pattern: {pattern.activity_id} ({pattern.activity_type.value})")
    
    async def optimize_edge_placement(
        self,
        strategy: PlacementStrategy = PlacementStrategy.BALANCED,
        max_nodes: int = 10,
        budget_constraint_usd: Optional[float] = None,
        latency_constraint_us: Optional[float] = None,
        geographic_constraints: Optional[List[str]] = None
    ) -> PlacementOptimizationResult:
        """
        Optimize edge node placement based on trading patterns and constraints
        """
        
        optimization_start = time.time()
        optimization_id = f"opt_{int(optimization_start)}"
        
        self.logger.info(f"Starting edge placement optimization: {optimization_id}")
        self.logger.info(f"Strategy: {strategy.value}, Max nodes: {max_nodes}")
        
        try:
            # Calculate placement scores for all candidates
            await self._calculate_placement_scores(strategy)
            
            # Apply constraints
            viable_candidates = await self._apply_constraints(
                budget_constraint_usd, latency_constraint_us, geographic_constraints
            )
            
            # Select optimal placements
            optimal_placements = await self._select_optimal_placements(
                viable_candidates, max_nodes, strategy
            )
            
            # Calculate performance projections
            performance_projections = await self._calculate_performance_projections(optimal_placements)
            
            # Generate alternative strategies
            alternatives = await self._generate_alternative_strategies(
                viable_candidates, max_nodes, strategy
            )
            
            optimization_time = time.time() - optimization_start
            
            result = PlacementOptimizationResult(
                optimization_id=optimization_id,
                strategy_used=strategy,
                total_candidates_evaluated=len(self.placement_candidates),
                total_activities_considered=len(self.trading_patterns),
                optimization_time_seconds=optimization_time,
                recommended_placements=optimal_placements,
                placement_rationale=self._generate_placement_rationale(optimal_placements),
                projected_average_latency_us=performance_projections["avg_latency_us"],
                projected_total_cost_usd_monthly=performance_projections["total_cost_usd"],
                projected_availability=performance_projections["availability"],
                projected_capacity_utilization=performance_projections["capacity_utilization"],
                alternative_strategies=alternatives,
                cost_benefit_analysis=self._perform_cost_benefit_analysis(optimal_placements)
            )
            
            self.optimization_history.append(result)
            
            self.logger.info(f"Optimization completed in {optimization_time:.2f}s")
            self.logger.info(f"Recommended {len(optimal_placements)} placements")
            self.logger.info(f"Projected cost: ${performance_projections['total_cost_usd']:,.2f}/month")
            self.logger.info(f"Projected latency: {performance_projections['avg_latency_us']:.1f}μs")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Optimization failed: {e}")
            raise
    
    async def _calculate_placement_scores(self, strategy: PlacementStrategy):
        """Calculate placement scores for all candidates based on strategy"""
        
        for candidate_id, candidate in self.placement_candidates.items():
            
            # Calculate latency score (lower latency = higher score)
            latency_score = await self._calculate_latency_score(candidate)
            
            # Calculate cost score (lower cost = higher score)  
            cost_score = await self._calculate_cost_score(candidate)
            
            # Calculate availability score
            availability_score = await self._calculate_availability_score(candidate)
            
            # Weight scores based on strategy
            if strategy == PlacementStrategy.LATENCY_OPTIMIZED:
                weights = {"latency": 0.7, "cost": 0.1, "availability": 0.2}
            elif strategy == PlacementStrategy.COST_OPTIMIZED:
                weights = {"latency": 0.1, "cost": 0.7, "availability": 0.2}
            elif strategy == PlacementStrategy.BALANCED:
                weights = {"latency": 0.4, "cost": 0.3, "availability": 0.3}
            elif strategy == PlacementStrategy.TRADING_VOLUME:
                weights = {"latency": 0.5, "cost": 0.2, "availability": 0.3}
            else:
                weights = {"latency": 0.33, "cost": 0.33, "availability": 0.34}
            
            # Calculate overall score
            overall_score = (
                latency_score * weights["latency"] +
                cost_score * weights["cost"] + 
                availability_score * weights["availability"]
            )
            
            # Update candidate scores
            candidate.latency_score = latency_score
            candidate.cost_score = cost_score
            candidate.availability_score = availability_score 
            candidate.overall_score = overall_score
        
        self.logger.info("Calculated placement scores for all candidates")
    
    async def _calculate_latency_score(self, candidate: EdgePlacementCandidate) -> float:
        """Calculate latency score for candidate based on trading patterns"""
        
        if not self.trading_patterns:
            # If no trading patterns, use market latencies
            market_latencies = list(candidate.estimated_latency_to_markets.values())
            if market_latencies:
                avg_latency = sum(market_latencies) / len(market_latencies)
                # Score inversely related to latency (lower latency = higher score)
                return max(0.0, min(1.0, (5000.0 - avg_latency) / 5000.0))
            return 0.5
        
        total_score = 0.0
        total_weight = 0.0
        
        for pattern in self.trading_patterns.values():
            pattern_score = 0.0
            pattern_weight = pattern.business_priority / 10.0  # Normalize to 0-1
            
            # Calculate latency score for this pattern
            for market in pattern.primary_markets:
                if market in candidate.estimated_latency_to_markets:
                    market_latency = candidate.estimated_latency_to_markets[market]
                    
                    # Score based on meeting latency requirements
                    if market_latency <= pattern.target_latency_us:
                        market_score = 1.0
                    elif market_latency <= pattern.max_latency_us:
                        # Linear decay between target and max
                        market_score = 1.0 - ((market_latency - pattern.target_latency_us) / 
                                            (pattern.max_latency_us - pattern.target_latency_us))
                    else:
                        market_score = 0.0
                    
                    pattern_score += market_score
            
            # Average score across markets for this pattern
            if pattern.primary_markets:
                pattern_score /= len(pattern.primary_markets)
            
            total_score += pattern_score * pattern_weight
            total_weight += pattern_weight
        
        return total_score / total_weight if total_weight > 0 else 0.5
    
    async def _calculate_cost_score(self, candidate: EdgePlacementCandidate) -> float:
        """Calculate cost score for candidate"""
        
        # Normalize cost score (lower cost = higher score)
        max_cost = max(c.estimated_monthly_cost_usd for c in self.placement_candidates.values())
        min_cost = min(c.estimated_monthly_cost_usd for c in self.placement_candidates.values())
        
        if max_cost == min_cost:
            return 0.5
        
        # Linear normalization (inverted - lower cost gets higher score)
        normalized_cost = (candidate.estimated_monthly_cost_usd - min_cost) / (max_cost - min_cost)
        return 1.0 - normalized_cost
    
    async def _calculate_availability_score(self, candidate: EdgePlacementCandidate) -> float:
        """Calculate availability score for candidate"""
        
        # Base availability score based on provider and location type
        if candidate.cloud_provider == "bare_metal":
            base_availability = 0.9999  # 99.99% for co-location
        elif candidate.cloud_provider in ["aws", "gcp", "azure"]:
            if candidate.network_tier == "premium":
                base_availability = 0.999   # 99.9% for premium cloud
            else:
                base_availability = 0.99    # 99% for standard cloud  
        else:
            base_availability = 0.95        # 95% for other providers
        
        # Adjust based on geographic factors
        if candidate.country in ["USA", "UK", "Germany", "Japan", "Singapore"]:
            geographic_factor = 1.0  # Major markets have good infrastructure
        else:
            geographic_factor = 0.95 # Other regions slightly lower
        
        return min(1.0, base_availability * geographic_factor)
    
    async def _apply_constraints(
        self,
        budget_constraint: Optional[float],
        latency_constraint: Optional[float], 
        geographic_constraints: Optional[List[str]]
    ) -> List[EdgePlacementCandidate]:
        """Apply constraints to filter viable candidates"""
        
        viable_candidates = []
        
        for candidate in self.placement_candidates.values():
            # Check budget constraint
            if budget_constraint and candidate.estimated_monthly_cost_usd > budget_constraint:
                continue
            
            # Check latency constraint
            if latency_constraint:
                max_market_latency = max(candidate.estimated_latency_to_markets.values())
                if max_market_latency > latency_constraint:
                    continue
            
            # Check geographic constraints
            if geographic_constraints:
                if candidate.country not in geographic_constraints:
                    continue
            
            viable_candidates.append(candidate)
        
        self.logger.info(f"Applied constraints: {len(viable_candidates)} viable candidates remaining")
        return viable_candidates
    
    async def _select_optimal_placements(
        self,
        candidates: List[EdgePlacementCandidate],
        max_nodes: int,
        strategy: PlacementStrategy
    ) -> List[EdgePlacementCandidate]:
        """Select optimal placements from viable candidates"""
        
        if not candidates:
            return []
        
        # Sort candidates by overall score (descending)
        sorted_candidates = sorted(candidates, key=lambda c: c.overall_score, reverse=True)
        
        selected_placements = []
        
        if strategy == PlacementStrategy.GEOGRAPHIC_SPREAD:
            # Ensure geographic diversity
            selected_countries = set()
            selected_providers = set()
            
            for candidate in sorted_candidates:
                if len(selected_placements) >= max_nodes:
                    break
                
                # Prioritize geographic and provider diversity
                diversity_bonus = 0.0
                if candidate.country not in selected_countries:
                    diversity_bonus += 0.1
                if candidate.cloud_provider not in selected_providers:
                    diversity_bonus += 0.05
                
                # Add diversity bonus to score
                adjusted_score = candidate.overall_score + diversity_bonus
                
                # Select candidate
                selected_placements.append(candidate)
                selected_countries.add(candidate.country)
                selected_providers.add(candidate.cloud_provider)
        
        else:
            # Simple greedy selection based on score
            selected_placements = sorted_candidates[:max_nodes]
        
        self.logger.info(f"Selected {len(selected_placements)} optimal placements")
        return selected_placements
    
    async def _calculate_performance_projections(
        self, placements: List[EdgePlacementCandidate]
    ) -> Dict[str, float]:
        """Calculate projected performance metrics"""
        
        if not placements:
            return {
                "avg_latency_us": 0.0,
                "total_cost_usd": 0.0,
                "availability": 0.0,
                "capacity_utilization": 0.0
            }
        
        # Calculate average latency across all placements and markets
        total_latency = 0.0
        total_measurements = 0
        
        for placement in placements:
            for latency in placement.estimated_latency_to_markets.values():
                total_latency += latency
                total_measurements += 1
        
        avg_latency = total_latency / total_measurements if total_measurements > 0 else 0.0
        
        # Calculate total cost
        total_cost = sum(p.estimated_monthly_cost_usd for p in placements)
        
        # Calculate combined availability (assuming independent failures)
        combined_availability = 1.0
        for placement in placements:
            combined_availability *= placement.availability_score
        
        # Estimate capacity utilization based on trading patterns
        total_capacity = len(placements) * 100000  # Assume 100K ops/sec per node
        total_demand = sum(p.peak_volume_per_second for p in self.trading_patterns.values())
        capacity_utilization = min(1.0, total_demand / total_capacity) if total_capacity > 0 else 0.0
        
        return {
            "avg_latency_us": avg_latency,
            "total_cost_usd": total_cost,
            "availability": combined_availability,
            "capacity_utilization": capacity_utilization
        }
    
    async def _generate_alternative_strategies(
        self,
        candidates: List[EdgePlacementCandidate],
        max_nodes: int,
        current_strategy: PlacementStrategy
    ) -> List[Dict[str, Any]]:
        """Generate alternative placement strategies for comparison"""
        
        alternatives = []
        
        # Generate alternatives for other strategies
        other_strategies = [s for s in PlacementStrategy if s != current_strategy]
        
        for alt_strategy in other_strategies[:3]:  # Limit to 3 alternatives
            # Recalculate scores with alternative strategy
            temp_scores = {}
            for candidate in candidates:
                if alt_strategy == PlacementStrategy.LATENCY_OPTIMIZED:
                    score = candidate.latency_score * 0.8 + candidate.availability_score * 0.2
                elif alt_strategy == PlacementStrategy.COST_OPTIMIZED:
                    score = candidate.cost_score * 0.8 + candidate.availability_score * 0.2
                else:
                    score = candidate.overall_score
                
                temp_scores[candidate.candidate_id] = score
            
            # Select top candidates for alternative strategy
            alt_candidates = sorted(candidates, key=lambda c: temp_scores[c.candidate_id], reverse=True)[:max_nodes]
            alt_projections = await self._calculate_performance_projections(alt_candidates)
            
            alternatives.append({
                "strategy": alt_strategy.value,
                "placements": [c.candidate_id for c in alt_candidates],
                "projected_cost": alt_projections["total_cost_usd"],
                "projected_latency": alt_projections["avg_latency_us"],
                "projected_availability": alt_projections["availability"]
            })
        
        return alternatives
    
    def _generate_placement_rationale(
        self, placements: List[EdgePlacementCandidate]
    ) -> Dict[str, str]:
        """Generate human-readable rationale for each placement"""
        
        rationale = {}
        
        for placement in placements:
            reasons = []
            
            # Latency rationale
            if placement.latency_score > 0.8:
                reasons.append("Excellent latency to target markets")
            elif placement.latency_score > 0.6:
                reasons.append("Good latency performance")
            
            # Cost rationale
            if placement.cost_score > 0.8:
                reasons.append("Cost-effective deployment option")
            elif placement.cost_score < 0.4:
                reasons.append("Premium pricing for specialized requirements")
            
            # Provider rationale
            if placement.cloud_provider == "bare_metal":
                reasons.append("Co-location for ultra-low latency")
            else:
                reasons.append(f"Cloud deployment on {placement.cloud_provider.upper()}")
            
            # Geographic rationale
            reasons.append(f"Strategic location in {placement.city}, {placement.country}")
            
            rationale[placement.candidate_id] = "; ".join(reasons)
        
        return rationale
    
    def _perform_cost_benefit_analysis(
        self, placements: List[EdgePlacementCandidate]
    ) -> Dict[str, float]:
        """Perform cost-benefit analysis of selected placements"""
        
        if not placements:
            return {}
        
        total_cost = sum(p.estimated_monthly_cost_usd for p in placements)
        
        # Estimate benefits based on latency improvement and trading volume
        latency_benefit = 0.0
        volume_benefit = 0.0
        
        for pattern in self.trading_patterns.values():
            # Estimate revenue impact of latency improvement
            # (Simplified model - 1μs improvement = $1000/month value for HFT)
            if pattern.activity_type in [TradingActivityType.HIGH_FREQUENCY, TradingActivityType.MARKET_MAKING]:
                latency_factor = 1000.0  # High value for HFT
            else:
                latency_factor = 100.0   # Lower value for other strategies
            
            # Find best latency for this pattern's markets
            best_latency = float('inf')
            for placement in placements:
                for market in pattern.primary_markets:
                    if market in placement.estimated_latency_to_markets:
                        best_latency = min(best_latency, placement.estimated_latency_to_markets[market])
            
            if best_latency < float('inf'):
                # Benefit from latency improvement (assuming baseline of 5000μs)
                baseline_latency = 5000.0
                latency_improvement = max(0, baseline_latency - best_latency)
                latency_benefit += latency_improvement * latency_factor * (pattern.business_priority / 10.0)
        
        # Calculate ROI and payback period
        total_benefit = latency_benefit + volume_benefit
        roi = (total_benefit - total_cost) / total_cost if total_cost > 0 else 0.0
        payback_months = total_cost / (total_benefit / 12) if total_benefit > 0 else float('inf')
        
        return {
            "total_monthly_cost_usd": total_cost,
            "estimated_monthly_benefit_usd": total_benefit,
            "roi_ratio": roi,
            "payback_period_months": min(payback_months, 999.0),  # Cap at 999 months
            "net_monthly_value_usd": total_benefit - total_cost
        }
    
    def get_placement_recommendations_summary(self) -> Dict[str, Any]:
        """Get summary of all placement optimization results"""
        
        if not self.optimization_history:
            return {"message": "No optimization history available"}
        
        latest_result = self.optimization_history[-1]
        
        return {
            "total_optimizations": len(self.optimization_history),
            "latest_optimization": {
                "optimization_id": latest_result.optimization_id,
                "strategy": latest_result.strategy_used.value,
                "placements_count": len(latest_result.recommended_placements),
                "projected_cost_usd": latest_result.projected_total_cost_usd_monthly,
                "projected_latency_us": latest_result.projected_average_latency_us,
                "projected_availability": latest_result.projected_availability
            },
            "placement_distribution": self._get_placement_distribution(latest_result.recommended_placements),
            "optimization_trends": self._get_optimization_trends(),
            "trading_patterns_configured": len(self.trading_patterns),
            "market_requirements_configured": len(self.market_requirements)
        }
    
    def _get_placement_distribution(self, placements: List[EdgePlacementCandidate]) -> Dict[str, int]:
        """Get distribution of placements by various dimensions"""
        
        distribution = {
            "by_provider": {},
            "by_country": {},
            "by_region": {},
            "by_network_tier": {}
        }
        
        for placement in placements:
            # By provider
            provider = placement.cloud_provider
            distribution["by_provider"][provider] = distribution["by_provider"].get(provider, 0) + 1
            
            # By country
            country = placement.country
            distribution["by_country"][country] = distribution["by_country"].get(country, 0) + 1
            
            # By region name
            region = placement.region_name
            distribution["by_region"][region] = distribution["by_region"].get(region, 0) + 1
            
            # By network tier
            tier = placement.network_tier
            distribution["by_network_tier"][tier] = distribution["by_network_tier"].get(tier, 0) + 1
        
        return distribution
    
    def _get_optimization_trends(self) -> Dict[str, Any]:
        """Get trends across optimization history"""
        
        if len(self.optimization_history) < 2:
            return {"message": "Insufficient history for trend analysis"}
        
        recent_results = self.optimization_history[-5:]  # Last 5 optimizations
        
        # Calculate trends
        costs = [r.projected_total_cost_usd_monthly for r in recent_results]
        latencies = [r.projected_average_latency_us for r in recent_results]
        availabilities = [r.projected_availability for r in recent_results]
        
        return {
            "cost_trend": "increasing" if costs[-1] > costs[0] else "decreasing",
            "latency_trend": "increasing" if latencies[-1] > latencies[0] else "decreasing", 
            "availability_trend": "increasing" if availabilities[-1] > availabilities[0] else "decreasing",
            "average_optimization_time": sum(r.optimization_time_seconds for r in recent_results) / len(recent_results)
        }