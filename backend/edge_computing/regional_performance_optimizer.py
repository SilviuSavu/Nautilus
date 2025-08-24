"""
Regional Performance Optimizer for Edge Trading Operations

This module optimizes trading performance across regions by analyzing
market patterns, latency characteristics, and resource allocation.
"""

import asyncio
import logging
import math
import statistics
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple, Any, Set
from datetime import datetime, timedelta
import numpy as np


class OptimizationObjective(Enum):
    """Performance optimization objectives"""
    MINIMIZE_LATENCY = "minimize_latency"           # Prioritize latency reduction
    MAXIMIZE_THROUGHPUT = "maximize_throughput"     # Prioritize throughput optimization
    MINIMIZE_COST = "minimize_cost"                 # Cost-effective optimization
    BALANCE_PERFORMANCE = "balance_performance"     # Balance latency, throughput, cost
    MAXIMIZE_AVAILABILITY = "maximize_availability" # Prioritize uptime and reliability


class PerformanceTier(Enum):
    """Regional performance tiers"""
    ULTRA_PERFORMANCE = "ultra_performance"     # < 100μs, 100k+ ops/sec
    HIGH_PERFORMANCE = "high_performance"       # < 500μs, 50k+ ops/sec  
    STANDARD = "standard"                       # < 2ms, 10k+ ops/sec
    ECONOMY = "economy"                         # < 10ms, 1k+ ops/sec


class MarketSession(Enum):
    """Trading market sessions"""
    PRE_MARKET = "pre_market"           # Before market open
    MARKET_OPEN = "market_open"         # Market opening period
    REGULAR_HOURS = "regular_hours"     # Regular trading hours
    MARKET_CLOSE = "market_close"       # Market closing period
    AFTER_HOURS = "after_hours"         # After market close
    OVERNIGHT = "overnight"             # Overnight session


@dataclass
class RegionalPerformanceProfile:
    """Performance profile for a regional trading setup"""
    region_id: str
    region_name: str
    
    # Geographic and network characteristics
    latitude: float
    longitude: float
    timezone: str
    primary_markets: List[str]  # ["NYSE", "NASDAQ", etc.]
    
    # Performance characteristics
    performance_tier: PerformanceTier
    target_latency_us: float
    current_latency_us: float
    target_throughput_ops: float
    current_throughput_ops: float
    
    # Resource allocation
    allocated_cpu_cores: int
    allocated_memory_gb: int
    allocated_bandwidth_gbps: float
    current_cpu_utilization: float
    current_memory_utilization: float
    current_bandwidth_utilization: float
    
    # Market timing optimization
    peak_trading_hours: List[Tuple[int, int]]  # [(9, 16), (21, 5)]
    market_session_patterns: Dict[MarketSession, Dict[str, float]]
    
    # Performance metrics
    p50_latency_us: float = 0.0
    p95_latency_us: float = 0.0
    p99_latency_us: float = 0.0
    p999_latency_us: float = 0.0
    error_rate: float = 0.0
    availability: float = 0.999
    
    # Optimization potential
    latency_improvement_potential: float = 0.0
    throughput_improvement_potential: float = 0.0
    cost_reduction_potential: float = 0.0


@dataclass
class OptimizationRecommendation:
    """Performance optimization recommendation"""
    recommendation_id: str
    region_id: str
    category: str  # "latency", "throughput", "resource", "network", "strategy"
    
    # Recommendation details
    title: str
    description: str
    rationale: str
    implementation_complexity: str  # "low", "medium", "high"
    
    # Expected impact
    expected_latency_improvement_us: float
    expected_throughput_improvement_ops: float
    expected_cost_impact_usd: float
    confidence_score: float  # 0.0 - 1.0
    
    # Implementation details
    required_resources: Dict[str, Any]
    estimated_implementation_hours: float
    prerequisites: List[str]
    risks: List[str]
    
    # Priority and timing
    priority: int  # 1-10, 10 = highest
    recommended_timing: str  # "immediate", "next_window", "planned_maintenance"
    impact_on_trading: str   # "none", "minimal", "moderate", "significant"


@dataclass
class PerformanceBaseline:
    """Performance baseline for comparison"""
    baseline_id: str
    region_id: str
    measurement_date: float
    
    # Latency baseline (microseconds)
    baseline_latencies: Dict[str, float]  # market -> latency
    baseline_p95_latency: float
    baseline_p99_latency: float
    
    # Throughput baseline
    baseline_throughput_ops: float
    baseline_peak_throughput_ops: float
    
    # Resource utilization baseline
    baseline_cpu_utilization: float
    baseline_memory_utilization: float
    baseline_network_utilization: float
    
    # Market timing baseline
    baseline_session_performance: Dict[MarketSession, Dict[str, float]]


@dataclass
class OptimizationResult:
    """Result of regional performance optimization"""
    optimization_id: str
    region_id: str
    optimization_date: float
    optimization_objective: OptimizationObjective
    
    # Optimization summary
    recommendations_generated: int
    high_priority_recommendations: int
    estimated_total_improvement: Dict[str, float]
    estimated_total_cost: float
    
    # Before/after comparison
    performance_before: Dict[str, float]
    performance_projected: Dict[str, float]
    improvement_percentage: Dict[str, float]
    
    # Implementation plan
    immediate_actions: List[OptimizationRecommendation]
    scheduled_actions: List[OptimizationRecommendation]
    long_term_actions: List[OptimizationRecommendation]
    
    # Risk assessment
    implementation_risks: List[str]
    rollback_plan: Dict[str, str]
    success_criteria: Dict[str, float]


class RegionalPerformanceOptimizer:
    """
    Regional Performance Optimizer for Edge Trading Operations
    
    Analyzes and optimizes trading performance across regions by:
    - Monitoring latency and throughput patterns
    - Analyzing market session performance characteristics  
    - Optimizing resource allocation and utilization
    - Generating actionable performance recommendations
    - Tracking optimization results and improvements
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Regional performance profiles
        self.regional_profiles: Dict[str, RegionalPerformanceProfile] = {}
        
        # Performance baselines for comparison
        self.performance_baselines: Dict[str, PerformanceBaseline] = {}
        
        # Optimization history
        self.optimization_history: List[OptimizationResult] = []
        
        # Real-time performance data
        self.performance_metrics: Dict[str, Dict[str, List[float]]] = {}  # region -> metric -> values
        
        # Market characteristics
        self.market_characteristics = self._initialize_market_characteristics()
        
        # Monitoring state
        self.monitoring_active = False
        self.optimization_active = False
        
        self.logger.info("Regional Performance Optimizer initialized")
    
    def _initialize_market_characteristics(self) -> Dict[str, Dict[str, Any]]:
        """Initialize market characteristics for optimization"""
        
        return {
            "NYSE": {
                "region": "us_east",
                "trading_hours": (9.5, 16),  # 9:30 AM - 4:00 PM EST
                "peak_volume_hours": [(9.5, 10.5), (15, 16)],  # Open and close
                "typical_latency_tolerance_us": 50.0,
                "volume_characteristics": {
                    "pre_market": 0.1,    # 10% of regular volume
                    "market_open": 3.0,   # 300% spike at open
                    "regular_hours": 1.0, # Baseline
                    "market_close": 2.5,  # 250% spike at close
                    "after_hours": 0.05   # 5% of regular volume
                }
            },
            "NASDAQ": {
                "region": "us_east", 
                "trading_hours": (9.5, 16),
                "peak_volume_hours": [(9.5, 10.5), (15, 16)],
                "typical_latency_tolerance_us": 50.0,
                "volume_characteristics": {
                    "pre_market": 0.15,
                    "market_open": 2.8,
                    "regular_hours": 1.0,
                    "market_close": 2.2,
                    "after_hours": 0.08
                }
            },
            "LSE": {
                "region": "eu_west",
                "trading_hours": (8, 16.5),  # 8:00 AM - 4:30 PM GMT
                "peak_volume_hours": [(8, 9), (15.5, 16.5)],
                "typical_latency_tolerance_us": 100.0,
                "volume_characteristics": {
                    "pre_market": 0.05,
                    "market_open": 2.5,
                    "regular_hours": 1.0,
                    "market_close": 2.0,
                    "after_hours": 0.02
                }
            },
            "TSE": {
                "region": "ap_northeast",
                "trading_hours": (9, 15),    # 9:00 AM - 3:00 PM JST
                "peak_volume_hours": [(9, 10), (14, 15)],
                "typical_latency_tolerance_us": 100.0,
                "volume_characteristics": {
                    "pre_market": 0.03,
                    "market_open": 2.0,
                    "regular_hours": 1.0,
                    "market_close": 1.8,
                    "after_hours": 0.01
                }
            }
        }
    
    async def add_regional_profile(self, profile: RegionalPerformanceProfile):
        """Add regional performance profile for optimization"""
        
        self.regional_profiles[profile.region_id] = profile
        
        # Initialize performance metrics tracking
        self.performance_metrics[profile.region_id] = {
            "latency_us": [],
            "throughput_ops": [],
            "cpu_utilization": [],
            "memory_utilization": [],
            "error_rate": []
        }
        
        self.logger.info(f"Added regional performance profile: {profile.region_id} ({profile.region_name})")
    
    async def create_performance_baseline(self, region_id: str) -> PerformanceBaseline:
        """Create performance baseline for region"""
        
        if region_id not in self.regional_profiles:
            raise ValueError(f"Region {region_id} not found")
        
        profile = self.regional_profiles[region_id]
        
        # Measure current performance to establish baseline
        baseline_metrics = await self._measure_baseline_performance(region_id)
        
        baseline = PerformanceBaseline(
            baseline_id=f"baseline_{region_id}_{int(time.time())}",
            region_id=region_id,
            measurement_date=time.time(),
            baseline_latencies=baseline_metrics["market_latencies"],
            baseline_p95_latency=baseline_metrics["p95_latency"],
            baseline_p99_latency=baseline_metrics["p99_latency"],
            baseline_throughput_ops=baseline_metrics["throughput_ops"],
            baseline_peak_throughput_ops=baseline_metrics["peak_throughput_ops"],
            baseline_cpu_utilization=baseline_metrics["cpu_utilization"],
            baseline_memory_utilization=baseline_metrics["memory_utilization"],
            baseline_network_utilization=baseline_metrics["network_utilization"],
            baseline_session_performance=baseline_metrics["session_performance"]
        )
        
        self.performance_baselines[region_id] = baseline
        
        self.logger.info(f"Created performance baseline for region {region_id}")
        return baseline
    
    async def _measure_baseline_performance(self, region_id: str) -> Dict[str, Any]:
        """Measure current performance to establish baseline"""
        
        profile = self.regional_profiles[region_id]
        
        # Simulate performance measurements
        market_latencies = {}
        for market in profile.primary_markets:
            market_char = self.market_characteristics.get(market, {})
            base_latency = market_char.get("typical_latency_tolerance_us", 200.0)
            # Add some realistic variance
            measured_latency = base_latency * (0.8 + 0.4 * np.random.random())
            market_latencies[market] = measured_latency
        
        # Generate realistic performance metrics
        avg_latency = statistics.mean(market_latencies.values()) if market_latencies else profile.current_latency_us
        
        return {
            "market_latencies": market_latencies,
            "p95_latency": avg_latency * 1.8,
            "p99_latency": avg_latency * 2.5,
            "throughput_ops": profile.current_throughput_ops,
            "peak_throughput_ops": profile.current_throughput_ops * 3.0,
            "cpu_utilization": profile.current_cpu_utilization,
            "memory_utilization": profile.current_memory_utilization, 
            "network_utilization": profile.current_bandwidth_utilization,
            "session_performance": {
                MarketSession.PRE_MARKET: {"latency": avg_latency * 0.7, "throughput": profile.current_throughput_ops * 0.1},
                MarketSession.MARKET_OPEN: {"latency": avg_latency * 1.5, "throughput": profile.current_throughput_ops * 3.0},
                MarketSession.REGULAR_HOURS: {"latency": avg_latency, "throughput": profile.current_throughput_ops},
                MarketSession.MARKET_CLOSE: {"latency": avg_latency * 1.3, "throughput": profile.current_throughput_ops * 2.5},
                MarketSession.AFTER_HOURS: {"latency": avg_latency * 0.8, "throughput": profile.current_throughput_ops * 0.05}
            }
        }
    
    async def optimize_regional_performance(
        self,
        region_id: str,
        objective: OptimizationObjective = OptimizationObjective.BALANCE_PERFORMANCE,
        target_improvement_percent: float = 20.0
    ) -> OptimizationResult:
        """Optimize performance for specific region"""
        
        if region_id not in self.regional_profiles:
            raise ValueError(f"Region {region_id} not found")
        
        optimization_start = time.time()
        optimization_id = f"opt_{region_id}_{int(optimization_start)}"
        
        self.logger.info(f"Starting regional performance optimization: {optimization_id}")
        self.logger.info(f"Region: {region_id}, Objective: {objective.value}")
        
        try:
            profile = self.regional_profiles[region_id]
            
            # Analyze current performance
            performance_analysis = await self._analyze_current_performance(region_id)
            
            # Generate optimization recommendations
            recommendations = await self._generate_optimization_recommendations(
                region_id, objective, target_improvement_percent, performance_analysis
            )
            
            # Categorize recommendations by priority and timing
            immediate_actions = [r for r in recommendations if r.recommended_timing == "immediate"]
            scheduled_actions = [r for r in recommendations if r.recommended_timing == "next_window"]
            long_term_actions = [r for r in recommendations if r.recommended_timing == "planned_maintenance"]
            
            # Calculate projected improvements
            projected_performance = await self._calculate_projected_performance(region_id, recommendations)
            
            # Create optimization result
            result = OptimizationResult(
                optimization_id=optimization_id,
                region_id=region_id,
                optimization_date=optimization_start,
                optimization_objective=objective,
                recommendations_generated=len(recommendations),
                high_priority_recommendations=len([r for r in recommendations if r.priority >= 8]),
                estimated_total_improvement={
                    "latency_improvement_us": sum(r.expected_latency_improvement_us for r in recommendations),
                    "throughput_improvement_ops": sum(r.expected_throughput_improvement_ops for r in recommendations),
                    "cost_impact_usd": sum(r.expected_cost_impact_usd for r in recommendations)
                },
                estimated_total_cost=sum(abs(r.expected_cost_impact_usd) for r in recommendations if r.expected_cost_impact_usd > 0),
                performance_before=performance_analysis["current_metrics"],
                performance_projected=projected_performance,
                improvement_percentage=self._calculate_improvement_percentage(
                    performance_analysis["current_metrics"], projected_performance
                ),
                immediate_actions=immediate_actions,
                scheduled_actions=scheduled_actions,
                long_term_actions=long_term_actions,
                implementation_risks=self._assess_implementation_risks(recommendations),
                rollback_plan=self._create_rollback_plan(recommendations),
                success_criteria=self._define_success_criteria(objective, target_improvement_percent)
            )
            
            self.optimization_history.append(result)
            
            optimization_time = time.time() - optimization_start
            self.logger.info(f"Optimization completed in {optimization_time:.2f}s")
            self.logger.info(f"Generated {len(recommendations)} recommendations")
            self.logger.info(f"High priority actions: {len(immediate_actions)}")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Optimization failed for region {region_id}: {e}")
            raise
    
    async def _analyze_current_performance(self, region_id: str) -> Dict[str, Any]:
        """Analyze current performance characteristics"""
        
        profile = self.regional_profiles[region_id]
        
        # Get recent performance metrics
        metrics = self.performance_metrics.get(region_id, {})
        
        # Calculate current performance statistics
        current_metrics = {
            "avg_latency_us": profile.current_latency_us,
            "p95_latency_us": profile.p95_latency_us,
            "p99_latency_us": profile.p99_latency_us,
            "throughput_ops": profile.current_throughput_ops,
            "cpu_utilization": profile.current_cpu_utilization,
            "memory_utilization": profile.current_memory_utilization,
            "error_rate": profile.error_rate,
            "availability": profile.availability
        }
        
        # Identify performance bottlenecks
        bottlenecks = []
        
        if profile.current_latency_us > profile.target_latency_us * 1.2:
            bottlenecks.append("latency_exceeds_target")
        
        if profile.current_throughput_ops < profile.target_throughput_ops * 0.8:
            bottlenecks.append("throughput_below_target")
        
        if profile.current_cpu_utilization > 80:
            bottlenecks.append("cpu_overutilized")
        
        if profile.current_memory_utilization > 85:
            bottlenecks.append("memory_overutilized")
        
        if profile.error_rate > 0.001:  # 0.1% error rate
            bottlenecks.append("high_error_rate")
        
        # Analyze market session performance
        session_analysis = {}
        for session in MarketSession:
            session_perf = profile.market_session_patterns.get(session, {})
            session_analysis[session.value] = {
                "performance_score": session_perf.get("performance_score", 0.5),
                "optimization_potential": 1.0 - session_perf.get("performance_score", 0.5)
            }
        
        return {
            "current_metrics": current_metrics,
            "bottlenecks": bottlenecks,
            "session_analysis": session_analysis,
            "improvement_potential": {
                "latency": profile.latency_improvement_potential,
                "throughput": profile.throughput_improvement_potential,
                "cost": profile.cost_reduction_potential
            }
        }
    
    async def _generate_optimization_recommendations(
        self,
        region_id: str,
        objective: OptimizationObjective,
        target_improvement: float,
        analysis: Dict[str, Any]
    ) -> List[OptimizationRecommendation]:
        """Generate optimization recommendations based on analysis"""
        
        recommendations = []
        profile = self.regional_profiles[region_id]
        bottlenecks = analysis["bottlenecks"]
        
        # Latency optimization recommendations
        if "latency_exceeds_target" in bottlenecks or objective == OptimizationObjective.MINIMIZE_LATENCY:
            
            if profile.performance_tier != PerformanceTier.ULTRA_PERFORMANCE:
                recommendations.append(OptimizationRecommendation(
                    recommendation_id=f"lat_001_{region_id}",
                    region_id=region_id,
                    category="latency",
                    title="Upgrade to Ultra Performance Tier",
                    description="Upgrade regional infrastructure to ultra-performance tier with kernel bypass and dedicated cores",
                    rationale="Current latency exceeds target. Ultra-performance tier can reduce latency by 70-80%",
                    implementation_complexity="high",
                    expected_latency_improvement_us=profile.current_latency_us * 0.7,
                    expected_throughput_improvement_ops=profile.current_throughput_ops * 0.5,
                    expected_cost_impact_usd=5000.0,  # Monthly increase
                    confidence_score=0.85,
                    required_resources={
                        "cpu_cores": 8,
                        "memory_gb": 64,
                        "network_bandwidth_gbps": 25,
                        "storage_ssd": True
                    },
                    estimated_implementation_hours=40,
                    prerequisites=["maintenance_window", "testing_environment"],
                    risks=["service_disruption", "configuration_complexity"],
                    priority=9,
                    recommended_timing="planned_maintenance",
                    impact_on_trading="moderate"
                ))
            
            # CPU pinning and isolation
            recommendations.append(OptimizationRecommendation(
                recommendation_id=f"lat_002_{region_id}",
                region_id=region_id,
                category="latency",
                title="Enable CPU Core Isolation",
                description="Isolate specific CPU cores for trading applications to reduce context switching latency",
                rationale="CPU core isolation can reduce latency jitter by 40-60% for real-time applications",
                implementation_complexity="medium",
                expected_latency_improvement_us=profile.current_latency_us * 0.3,
                expected_throughput_improvement_ops=0.0,
                expected_cost_impact_usd=0.0,
                confidence_score=0.75,
                required_resources={"cpu_isolation_config": True},
                estimated_implementation_hours=8,
                prerequisites=["kernel_parameters_access"],
                risks=["reduced_cpu_flexibility"],
                priority=7,
                recommended_timing="next_window",
                impact_on_trading="minimal"
            ))
        
        # Throughput optimization recommendations
        if "throughput_below_target" in bottlenecks or objective == OptimizationObjective.MAXIMIZE_THROUGHPUT:
            
            recommendations.append(OptimizationRecommendation(
                recommendation_id=f"thr_001_{region_id}",
                region_id=region_id,
                category="throughput",
                title="Implement Connection Pooling",
                description="Implement advanced connection pooling to reduce connection overhead and increase throughput",
                rationale="Connection pooling can improve throughput by 30-50% by reusing established connections",
                implementation_complexity="low",
                expected_latency_improvement_us=profile.current_latency_us * 0.1,
                expected_throughput_improvement_ops=profile.current_throughput_ops * 0.4,
                expected_cost_impact_usd=0.0,
                confidence_score=0.9,
                required_resources={"memory_gb": 8},
                estimated_implementation_hours=16,
                prerequisites=["application_modification"],
                risks=["connection_state_management"],
                priority=8,
                recommended_timing="immediate",
                impact_on_trading="none"
            ))
            
            if profile.current_cpu_utilization > 70:
                recommendations.append(OptimizationRecommendation(
                    recommendation_id=f"thr_002_{region_id}",
                    region_id=region_id,
                    category="throughput",
                    title="Scale CPU Resources",
                    description="Increase CPU cores allocation to handle higher throughput demands",
                    rationale="CPU utilization over 70% indicates bottleneck. Additional cores can increase capacity by 50%+",
                    implementation_complexity="medium",
                    expected_latency_improvement_us=0.0,
                    expected_throughput_improvement_ops=profile.current_throughput_ops * 0.6,
                    expected_cost_impact_usd=1500.0,  # Monthly increase
                    confidence_score=0.8,
                    required_resources={"cpu_cores": 4},
                    estimated_implementation_hours=4,
                    prerequisites=["resource_availability"],
                    risks=["cost_increase"],
                    priority=6,
                    recommended_timing="next_window", 
                    impact_on_trading="minimal"
                ))
        
        # Resource optimization recommendations
        if "cpu_overutilized" in bottlenecks or "memory_overutilized" in bottlenecks:
            
            recommendations.append(OptimizationRecommendation(
                recommendation_id=f"res_001_{region_id}",
                region_id=region_id,
                category="resource",
                title="Optimize Memory Usage",
                description="Implement memory-efficient data structures and garbage collection tuning",
                rationale="Memory optimization can reduce allocation overhead and improve overall performance",
                implementation_complexity="medium",
                expected_latency_improvement_us=profile.current_latency_us * 0.15,
                expected_throughput_improvement_ops=profile.current_throughput_ops * 0.2,
                expected_cost_impact_usd=-500.0,  # Cost savings
                confidence_score=0.7,
                required_resources={"development_time": 40},
                estimated_implementation_hours=32,
                prerequisites=["performance_profiling"],
                risks=["code_complexity_increase"],
                priority=5,
                recommended_timing="planned_maintenance",
                impact_on_trading="minimal"
            ))
        
        # Network optimization recommendations
        recommendations.append(OptimizationRecommendation(
            recommendation_id=f"net_001_{region_id}",
            region_id=region_id,
            category="network",
            title="Optimize TCP Parameters",
            description="Fine-tune TCP buffer sizes, congestion control, and kernel network parameters",
            rationale="Network parameter tuning can reduce latency by 10-25% and improve throughput",
            implementation_complexity="low",
            expected_latency_improvement_us=profile.current_latency_us * 0.2,
            expected_throughput_improvement_ops=profile.current_throughput_ops * 0.15,
            expected_cost_impact_usd=0.0,
            confidence_score=0.8,
            required_resources={"system_access": True},
            estimated_implementation_hours=4,
            prerequisites=["kernel_parameter_access"],
            risks=["network_instability_if_misconfigured"],
            priority=7,
            recommended_timing="immediate",
            impact_on_trading="none"
        ))
        
        # Strategy-specific optimizations based on market session analysis
        session_analysis = analysis["session_analysis"]
        
        for session_name, session_data in session_analysis.items():
            if session_data["optimization_potential"] > 0.3:  # 30% improvement potential
                
                recommendations.append(OptimizationRecommendation(
                    recommendation_id=f"str_{session_name}_{region_id}",
                    region_id=region_id,
                    category="strategy",
                    title=f"Optimize {session_name.replace('_', ' ').title()} Performance",
                    description=f"Implement session-specific optimizations for {session_name} trading patterns",
                    rationale=f"Session shows {session_data['optimization_potential']:.1%} improvement potential",
                    implementation_complexity="medium",
                    expected_latency_improvement_us=profile.current_latency_us * session_data["optimization_potential"] * 0.5,
                    expected_throughput_improvement_ops=profile.current_throughput_ops * session_data["optimization_potential"] * 0.3,
                    expected_cost_impact_usd=200.0,
                    confidence_score=0.65,
                    required_resources={"configuration_tuning": True},
                    estimated_implementation_hours=12,
                    prerequisites=["session_pattern_analysis"],
                    risks=["session_specific_issues"],
                    priority=4,
                    recommended_timing="next_window",
                    impact_on_trading="minimal"
                ))
        
        # Sort recommendations by priority (descending)
        recommendations.sort(key=lambda r: r.priority, reverse=True)
        
        return recommendations
    
    async def _calculate_projected_performance(
        self, 
        region_id: str, 
        recommendations: List[OptimizationRecommendation]
    ) -> Dict[str, float]:
        """Calculate projected performance after implementing recommendations"""
        
        profile = self.regional_profiles[region_id]
        
        # Start with current performance
        projected = {
            "avg_latency_us": profile.current_latency_us,
            "throughput_ops": profile.current_throughput_ops,
            "cpu_utilization": profile.current_cpu_utilization,
            "memory_utilization": profile.current_memory_utilization,
            "error_rate": profile.error_rate,
            "availability": profile.availability
        }
        
        # Apply improvements from recommendations
        total_latency_improvement = 0.0
        total_throughput_improvement = 0.0
        
        for rec in recommendations:
            # Apply latency improvements (multiplicative for realism)
            if rec.expected_latency_improvement_us > 0:
                improvement_factor = rec.expected_latency_improvement_us / profile.current_latency_us
                projected["avg_latency_us"] *= (1.0 - improvement_factor)
            
            # Apply throughput improvements (additive)
            if rec.expected_throughput_improvement_ops > 0:
                projected["throughput_ops"] += rec.expected_throughput_improvement_ops
        
        # Apply resource scaling effects
        cpu_scaling_recommendations = [r for r in recommendations if "cpu_cores" in r.required_resources]
        if cpu_scaling_recommendations:
            additional_cores = sum(r.required_resources["cpu_cores"] for r in cpu_scaling_recommendations)
            current_cores = profile.allocated_cpu_cores
            scaling_factor = current_cores / (current_cores + additional_cores)
            projected["cpu_utilization"] *= scaling_factor
        
        # Ensure realistic bounds
        projected["avg_latency_us"] = max(projected["avg_latency_us"], profile.target_latency_us * 0.8)
        projected["cpu_utilization"] = max(10.0, min(95.0, projected["cpu_utilization"]))
        projected["memory_utilization"] = max(10.0, min(90.0, projected["memory_utilization"]))
        projected["error_rate"] = max(0.0001, min(0.01, projected["error_rate"] * 0.8))
        projected["availability"] = min(0.9999, projected["availability"] * 1.001)
        
        return projected
    
    def _calculate_improvement_percentage(
        self, 
        current: Dict[str, float], 
        projected: Dict[str, float]
    ) -> Dict[str, float]:
        """Calculate percentage improvement between current and projected performance"""
        
        improvements = {}
        
        for metric in current.keys():
            current_val = current[metric]
            projected_val = projected[metric]
            
            if current_val == 0:
                improvements[metric] = 0.0
            else:
                if metric in ["avg_latency_us", "error_rate", "cpu_utilization", "memory_utilization"]:
                    # For these metrics, lower is better
                    improvements[metric] = ((current_val - projected_val) / current_val) * 100
                else:
                    # For these metrics, higher is better
                    improvements[metric] = ((projected_val - current_val) / current_val) * 100
        
        return improvements
    
    def _assess_implementation_risks(self, recommendations: List[OptimizationRecommendation]) -> List[str]:
        """Assess implementation risks across all recommendations"""
        
        risk_categories = set()
        
        for rec in recommendations:
            risk_categories.update(rec.risks)
        
        # Add aggregate risks
        high_complexity_count = len([r for r in recommendations if r.implementation_complexity == "high"])
        if high_complexity_count > 2:
            risk_categories.add("multiple_high_complexity_changes")
        
        total_cost_impact = sum(r.expected_cost_impact_usd for r in recommendations if r.expected_cost_impact_usd > 0)
        if total_cost_impact > 10000:
            risk_categories.add("significant_cost_increase")
        
        return list(risk_categories)
    
    def _create_rollback_plan(self, recommendations: List[OptimizationRecommendation]) -> Dict[str, str]:
        """Create rollback plan for recommendations"""
        
        rollback_plan = {}
        
        for rec in recommendations:
            if rec.implementation_complexity == "high":
                rollback_plan[rec.recommendation_id] = "full_system_restore_required"
            elif rec.implementation_complexity == "medium":
                rollback_plan[rec.recommendation_id] = "configuration_rollback_available"
            else:
                rollback_plan[rec.recommendation_id] = "immediate_rollback_available"
        
        return rollback_plan
    
    def _define_success_criteria(
        self, 
        objective: OptimizationObjective, 
        target_improvement: float
    ) -> Dict[str, float]:
        """Define success criteria for optimization"""
        
        criteria = {}
        
        if objective == OptimizationObjective.MINIMIZE_LATENCY:
            criteria["latency_improvement_percent"] = target_improvement
            criteria["p95_latency_improvement_percent"] = target_improvement * 0.8
            criteria["availability_maintained_percent"] = 99.9
            
        elif objective == OptimizationObjective.MAXIMIZE_THROUGHPUT:
            criteria["throughput_improvement_percent"] = target_improvement
            criteria["latency_degradation_max_percent"] = 10.0  # Max 10% latency degradation
            criteria["error_rate_max"] = 0.001  # Max 0.1% error rate
            
        elif objective == OptimizationObjective.MINIMIZE_COST:
            criteria["cost_reduction_percent"] = target_improvement
            criteria["performance_degradation_max_percent"] = 5.0  # Max 5% performance loss
            
        else:  # BALANCE_PERFORMANCE
            criteria["latency_improvement_percent"] = target_improvement * 0.7
            criteria["throughput_improvement_percent"] = target_improvement * 0.5
            criteria["cost_increase_max_percent"] = target_improvement * 0.3
            criteria["availability_maintained_percent"] = 99.9
        
        return criteria
    
    async def start_performance_monitoring(self):
        """Start continuous performance monitoring"""
        
        if self.monitoring_active:
            return
        
        self.monitoring_active = True
        
        # Start monitoring task
        asyncio.create_task(self._performance_monitoring_loop())
        
        self.logger.info("Regional performance monitoring started")
    
    async def _performance_monitoring_loop(self):
        """Main performance monitoring loop"""
        
        while self.monitoring_active:
            try:
                for region_id in self.regional_profiles.keys():
                    await self._collect_performance_metrics(region_id)
                
                await asyncio.sleep(10)  # Collect metrics every 10 seconds
                
            except Exception as e:
                self.logger.error(f"Performance monitoring error: {e}")
                await asyncio.sleep(30)
    
    async def _collect_performance_metrics(self, region_id: str):
        """Collect performance metrics for specific region"""
        
        try:
            profile = self.regional_profiles[region_id]
            metrics = self.performance_metrics[region_id]
            
            # Simulate realistic metric collection with some variance
            import random
            
            # Latency with realistic variance
            base_latency = profile.current_latency_us
            current_latency = base_latency * (0.8 + 0.4 * random.random())
            metrics["latency_us"].append(current_latency)
            
            # Throughput with time-based patterns
            current_hour = datetime.now().hour
            volume_multiplier = 1.0
            
            # Simulate market hours effect
            for market in profile.primary_markets:
                market_char = self.market_characteristics.get(market, {})
                trading_hours = market_char.get("trading_hours", (9, 17))
                
                if trading_hours[0] <= current_hour <= trading_hours[1]:
                    volume_multiplier *= 2.0  # Higher throughput during market hours
            
            current_throughput = profile.current_throughput_ops * volume_multiplier * (0.7 + 0.6 * random.random())
            metrics["throughput_ops"].append(current_throughput)
            
            # Resource utilization
            cpu_util = min(95, profile.current_cpu_utilization + random.uniform(-10, 15))
            memory_util = min(90, profile.current_memory_utilization + random.uniform(-5, 10))
            error_rate = max(0, profile.error_rate + random.uniform(-0.0005, 0.001))
            
            metrics["cpu_utilization"].append(cpu_util)
            metrics["memory_utilization"].append(memory_util)
            metrics["error_rate"].append(error_rate)
            
            # Keep only recent metrics (last 1000 measurements)
            for metric_name, values in metrics.items():
                if len(values) > 1000:
                    metrics[metric_name] = values[-1000:]
            
            # Update profile with recent averages
            if len(metrics["latency_us"]) >= 10:
                profile.current_latency_us = statistics.mean(metrics["latency_us"][-10:])
                profile.p95_latency_us = np.percentile(metrics["latency_us"][-100:], 95) if len(metrics["latency_us"]) >= 100 else profile.current_latency_us * 1.8
                profile.p99_latency_us = np.percentile(metrics["latency_us"][-100:], 99) if len(metrics["latency_us"]) >= 100 else profile.current_latency_us * 2.5
            
            if len(metrics["throughput_ops"]) >= 10:
                profile.current_throughput_ops = statistics.mean(metrics["throughput_ops"][-10:])
            
            if len(metrics["cpu_utilization"]) >= 10:
                profile.current_cpu_utilization = statistics.mean(metrics["cpu_utilization"][-10:])
            
            if len(metrics["memory_utilization"]) >= 10:
                profile.current_memory_utilization = statistics.mean(metrics["memory_utilization"][-10:])
            
            if len(metrics["error_rate"]) >= 10:
                profile.error_rate = statistics.mean(metrics["error_rate"][-10:])
                
        except Exception as e:
            self.logger.error(f"Error collecting metrics for region {region_id}: {e}")
    
    async def get_optimization_summary(self) -> Dict[str, Any]:
        """Get comprehensive optimization summary across all regions"""
        
        if not self.optimization_history:
            return {"message": "No optimization history available"}
        
        # Latest optimization results by region
        latest_optimizations = {}
        for result in self.optimization_history:
            region_id = result.region_id
            if region_id not in latest_optimizations or result.optimization_date > latest_optimizations[region_id].optimization_date:
                latest_optimizations[region_id] = result
        
        # Calculate aggregate statistics
        total_recommendations = sum(r.recommendations_generated for r in latest_optimizations.values())
        total_high_priority = sum(r.high_priority_recommendations for r in latest_optimizations.values())
        
        # Performance improvements
        avg_latency_improvement = statistics.mean([
            r.improvement_percentage.get("avg_latency_us", 0) for r in latest_optimizations.values()
        ])
        
        avg_throughput_improvement = statistics.mean([
            r.improvement_percentage.get("throughput_ops", 0) for r in latest_optimizations.values()
        ])
        
        return {
            "timestamp": time.time(),
            "global_optimization_summary": {
                "total_regions_optimized": len(latest_optimizations),
                "total_recommendations": total_recommendations,
                "high_priority_recommendations": total_high_priority,
                "avg_latency_improvement_percent": avg_latency_improvement,
                "avg_throughput_improvement_percent": avg_throughput_improvement
            },
            "regional_summaries": {
                region_id: {
                    "optimization_id": result.optimization_id,
                    "optimization_date": result.optimization_date,
                    "objective": result.optimization_objective.value,
                    "recommendations_count": result.recommendations_generated,
                    "immediate_actions": len(result.immediate_actions),
                    "projected_improvements": result.improvement_percentage
                }
                for region_id, result in latest_optimizations.items()
            },
            "performance_trends": self._calculate_performance_trends(),
            "monitoring_active": self.monitoring_active
        }
    
    def _calculate_performance_trends(self) -> Dict[str, Any]:
        """Calculate performance trends across regions"""
        
        trends = {}
        
        for region_id, metrics in self.performance_metrics.items():
            if len(metrics["latency_us"]) < 2:
                continue
                
            # Calculate recent trend (last 50% of data vs first 50%)
            latency_values = metrics["latency_us"]
            half_point = len(latency_values) // 2
            
            if half_point > 0:
                early_avg = statistics.mean(latency_values[:half_point])
                recent_avg = statistics.mean(latency_values[half_point:])
                latency_trend = "improving" if recent_avg < early_avg else "degrading"
            else:
                latency_trend = "stable"
            
            # Similar for throughput
            throughput_values = metrics["throughput_ops"]
            if len(throughput_values) >= half_point * 2:
                early_thr = statistics.mean(throughput_values[:half_point])
                recent_thr = statistics.mean(throughput_values[half_point:])
                throughput_trend = "improving" if recent_thr > early_thr else "degrading"
            else:
                throughput_trend = "stable"
            
            trends[region_id] = {
                "latency_trend": latency_trend,
                "throughput_trend": throughput_trend,
                "data_points": len(latency_values)
            }
        
        return trends
    
    def stop_monitoring(self):
        """Stop performance monitoring"""
        
        self.monitoring_active = False
        self.logger.info("Regional performance monitoring stopped")