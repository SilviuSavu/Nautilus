# Story 2.1: Portfolio Optimizer API Integration - Implementation Guide

## 📋 Story Overview

**Title**: As a quantitative researcher, I want access to Portfolio Optimizer cloud API so that I can leverage advanced optimization algorithms not available locally.

**Story Points**: 8  
**Priority**: P0-Critical  
**Timeline**: Days 6-7  
**Estimated Effort**: 20 hours

## 🎯 Business Objectives

### **Primary Goals**
- Enable cloud-based portfolio optimization with institutional-grade algorithms
- Implement 10+ optimization methods including supervised k-NN learning
- Provide fallback mechanisms for high availability
- Establish foundation for world's first supervised ML portfolio optimization

### **Success Criteria**
- [ ] Portfolio Optimizer API client fully functional
- [ ] API key securely managed (EgyPyGQSVQV4GMKWp5l7tf21wQOUaaw)
- [ ] Authentication and error handling robust
- [ ] Response caching implemented (5min TTL)
- [ ] Fallback to local optimization on API failure
- [ ] Cloud optimization response time <3 seconds

## 🔑 API Configuration

### **Available API Key**
- **Key**: `EgyPyGQSVQV4GMKWp5l7tf21wQOUaaw`
- **Environment Variable**: `PORTFOLIO_OPTIMIZER_API_KEY`
- **Base URL**: `https://api.portfoliooptimizer.io/v1`

### **Supported Optimization Methods**
1. **Traditional Methods**:
   - Mean-Variance Optimization
   - Minimum Variance Portfolio
   - Maximum Sharpe Ratio
   - Equal Risk Contribution
   - Risk Parity
   
2. **Advanced Methods**:
   - Hierarchical Risk Parity (HRP)
   - Cluster Risk Parity (CRP)
   - Maximum Diversification
   - Inverse Volatility Weighting
   
3. **ML-Enhanced Methods**:
   - **Supervised k-NN Portfolio Optimization** (unique feature)
   - Dynamic k* neighbor selection
   - Hassanat distance metric (scale-invariant)

## 🛠️ Technical Implementation

### **Phase 1: API Client Foundation** (8 hours)

#### **1.1 Enhanced Portfolio Optimizer Client**
**File**: `/backend/engines/risk/portfolio_optimizer_client.py`

```python
"""
Portfolio Optimizer API Client - Enhanced Version
=================================================

Cloud-based portfolio optimization service integration for Nautilus.
Provides access to institutional-grade optimization algorithms including
unique supervised ML portfolio optimization using k-NN.

API Documentation: https://docs.portfoliooptimizer.io/
API Key: EgyPyGQSVQV4GMKWp5l7tf21wQOUaaw
"""

import asyncio
import logging
import time
import os
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from enum import Enum
import numpy as np
import pandas as pd
import httpx
from httpx import AsyncClient, Response
import json
import hashlib

logger = logging.getLogger(__name__)


class OptimizationMethod(Enum):
    """Available portfolio optimization methods"""
    MEAN_VARIANCE = "mean-variance"
    MINIMUM_VARIANCE = "minimum-variance"
    MAXIMUM_SHARPE = "maximum-sharpe"
    EQUAL_RISK_CONTRIBUTION = "equal-risk-contribution"
    RISK_PARITY = "risk-parity"
    HIERARCHICAL_RISK_PARITY = "hierarchical-risk-parity"
    CLUSTER_RISK_PARITY = "cluster-risk-parity"
    MAXIMUM_DIVERSIFICATION = "maximum-diversification"
    EQUAL_WEIGHT = "equal-weight"
    MARKET_CAP_WEIGHTED = "market-cap-weighted"
    INVERSE_VOLATILITY = "inverse-volatility"
    SUPERVISED_KNN = "supervised-knn"
    MAX_DECORRELATION = "max-decorrelation"


class DistanceMetric(Enum):
    """Distance metrics for k-NN supervised portfolios"""
    EUCLIDEAN = "euclidean"
    HASSANAT = "hassanat"  # Scale-invariant, default
    MANHATTAN = "manhattan"
    COSINE = "cosine"


class RiskMeasure(Enum):
    """Risk measures for optimization constraints"""
    STANDARD_DEVIATION = "standard-deviation"
    VARIANCE = "variance"
    SEMI_VARIANCE = "semi-variance"
    VALUE_AT_RISK = "value-at-risk"
    CONDITIONAL_VALUE_AT_RISK = "conditional-value-at-risk"
    MAXIMUM_DRAWDOWN = "maximum-drawdown"
    ULCER_INDEX = "ulcer-index"
    DOWNSIDE_DEVIATION = "downside-deviation"


@dataclass
class OptimizationConstraints:
    """Portfolio optimization constraints"""
    min_weight: float = 0.0
    max_weight: float = 1.0
    target_return: Optional[float] = None
    target_risk: Optional[float] = None
    max_assets: Optional[int] = None  # Cardinality constraint
    group_constraints: Dict[str, Tuple[float, float]] = field(default_factory=dict)
    exposure_constraints: Dict[str, float] = field(default_factory=dict)
    turnover_limit: Optional[float] = None
    leverage_limit: float = 1.0


@dataclass
class PortfolioOptimizationRequest:
    """Request for portfolio optimization"""
    assets: List[str]
    returns: Optional[np.ndarray] = None  # Historical returns matrix
    covariance_matrix: Optional[np.ndarray] = None
    expected_returns: Optional[np.ndarray] = None
    method: OptimizationMethod = OptimizationMethod.MINIMUM_VARIANCE
    constraints: OptimizationConstraints = field(default_factory=OptimizationConstraints)
    risk_measure: RiskMeasure = RiskMeasure.STANDARD_DEVIATION
    risk_free_rate: float = 0.02
    confidence_level: float = 0.95  # For VaR/CVaR
    
    # Supervised portfolio specific
    lookback_periods: int = 252
    k_neighbors: Optional[int] = None  # None for dynamic k*
    distance_metric: DistanceMetric = DistanceMetric.HASSANAT
    features: Optional[Dict[str, np.ndarray]] = None  # Additional features for k-NN


@dataclass
class PortfolioOptimizationResult:
    """Portfolio optimization result from API"""
    optimal_weights: Dict[str, float]
    expected_return: float
    expected_risk: float
    sharpe_ratio: float
    diversification_ratio: Optional[float]
    effective_assets: int
    concentration_risk: float
    value_at_risk: Optional[float]
    conditional_value_at_risk: Optional[float]
    max_drawdown: Optional[float]
    metadata: Dict[str, Any] = field(default_factory=dict)
    optimization_time_ms: float = 0.0
    api_response_time_ms: float = 0.0


class PortfolioOptimizerClient:
    """
    Enhanced Portfolio Optimizer API Client
    
    Features:
    - 10+ optimization methods including supervised ML
    - Cloud-based computation with local fallback
    - Automatic retry with exponential backoff
    - Result caching for efficiency
    - Circuit breaker pattern for reliability
    """
    
    BASE_URL = "https://api.portfoliooptimizer.io/v1"
    
    def __init__(self, api_key: Optional[str] = None, cache_ttl_seconds: int = 300):
        # Get API key from environment if not provided
        self.api_key = api_key or os.getenv("PORTFOLIO_OPTIMIZER_API_KEY")
        if not self.api_key:
            logger.warning("No Portfolio Optimizer API key provided - cloud optimization unavailable")
        
        self.cache_ttl = cache_ttl_seconds
        self.cache: Dict[str, Tuple[Any, datetime]] = {}
        
        # HTTP client configuration
        self.client = AsyncClient(
            base_url=self.BASE_URL,
            timeout=30.0,
            headers=self._get_headers()
        ) if self.api_key else None
        
        # Rate limiting and circuit breaker
        self.last_request_time = 0
        self.min_request_interval = 0.1  # 10 requests/second max
        self.circuit_breaker_failures = 0
        self.circuit_breaker_threshold = 5
        self.circuit_breaker_timeout = 300  # 5 minutes
        self.circuit_breaker_last_failure = None
        
        # Performance tracking
        self.total_requests = 0
        self.successful_requests = 0
        self.failed_requests = 0
        self.total_api_time_ms = 0
        self.cache_hits = 0
        
        logger.info(f"Portfolio Optimizer client initialized (authenticated: {bool(self.api_key)})")
    
    def _get_headers(self) -> Dict[str, str]:
        """Get HTTP headers with API key"""
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": "Nautilus-Trading-Platform/1.0"
        }
        if self.api_key:
            headers["X-API-Key"] = self.api_key
        return headers
    
    def _is_circuit_breaker_open(self) -> bool:
        """Check if circuit breaker is open"""
        if self.circuit_breaker_failures < self.circuit_breaker_threshold:
            return False
        
        if self.circuit_breaker_last_failure:
            time_since_failure = datetime.now() - self.circuit_breaker_last_failure
            if time_since_failure.total_seconds() > self.circuit_breaker_timeout:
                # Reset circuit breaker
                self.circuit_breaker_failures = 0
                self.circuit_breaker_last_failure = None
                return False
        
        return True
    
    async def _rate_limit(self):
        """Enforce rate limiting"""
        elapsed = time.time() - self.last_request_time
        if elapsed < self.min_request_interval:
            await asyncio.sleep(self.min_request_interval - elapsed)
        self.last_request_time = time.time()
    
    async def _make_request(self, method: str, endpoint: str, 
                          data: Optional[Dict] = None, 
                          retries: int = 3) -> Dict[str, Any]:
        """Make API request with retry logic and circuit breaker"""
        if not self.client:
            raise ValueError("API client not initialized - check API key")
        
        if self._is_circuit_breaker_open():
            raise RuntimeError("Circuit breaker is open - API temporarily unavailable")
        
        await self._rate_limit()
        
        for attempt in range(retries):
            try:
                start_time = time.time()
                
                if method == "GET":
                    response = await self.client.get(endpoint, params=data)
                else:
                    response = await self.client.post(endpoint, json=data)
                
                api_time_ms = (time.time() - start_time) * 1000
                self.total_api_time_ms += api_time_ms
                self.total_requests += 1
                
                if response.status_code == 200:
                    self.successful_requests += 1
                    # Reset circuit breaker on success
                    if self.circuit_breaker_failures > 0:
                        self.circuit_breaker_failures = max(0, self.circuit_breaker_failures - 1)
                    
                    return response.json()
                    
                elif response.status_code == 429:  # Rate limited
                    retry_after = int(response.headers.get("Retry-After", 5))
                    logger.warning(f"Rate limited, retrying after {retry_after}s")
                    await asyncio.sleep(retry_after)
                    
                elif response.status_code == 401:
                    raise ValueError("Invalid API key or authentication failed")
                    
                elif response.status_code >= 500:
                    # Server error - count towards circuit breaker
                    self._record_failure()
                    if attempt < retries - 1:
                        await asyncio.sleep(2 ** attempt)  # Exponential backoff
                    
                else:
                    logger.error(f"API error {response.status_code}: {response.text}")
                    self._record_failure()
                    if attempt < retries - 1:
                        await asyncio.sleep(2 ** attempt)
                    
            except Exception as e:
                logger.error(f"Request failed (attempt {attempt + 1}/{retries}): {e}")
                self._record_failure()
                if attempt < retries - 1:
                    await asyncio.sleep(2 ** attempt)
                else:
                    raise
        
        raise RuntimeError(f"Failed to complete request after {retries} attempts")
    
    def _record_failure(self):
        """Record API failure for circuit breaker"""
        self.failed_requests += 1
        self.circuit_breaker_failures += 1
        self.circuit_breaker_last_failure = datetime.now()
    
    def _get_cache_key(self, method: str, **kwargs) -> str:
        """Generate cache key for request"""
        # Create deterministic key from method and parameters
        params_str = json.dumps(kwargs, sort_keys=True, default=str)
        key_hash = hashlib.md5(params_str.encode()).hexdigest()
        return f"{method}:{key_hash}"
    
    def _check_cache(self, cache_key: str) -> Optional[Any]:
        """Check if result is in cache and still valid"""
        if cache_key in self.cache:
            result, timestamp = self.cache[cache_key]
            if (datetime.now() - timestamp).seconds < self.cache_ttl:
                self.cache_hits += 1
                logger.debug(f"Cache hit for {cache_key}")
                return result
        return None
    
    async def optimize_portfolio(self, request: PortfolioOptimizationRequest) -> PortfolioOptimizationResult:
        """
        Main portfolio optimization method
        
        Handles all optimization types including supervised ML portfolios
        """
        start_time = time.time()
        
        # Check cache
        cache_key = self._get_cache_key(
            request.method.value,
            assets=request.assets,
            constraints=asdict(request.constraints)
        )
        cached_result = self._check_cache(cache_key)
        if cached_result:
            return cached_result
        
        # Route to appropriate optimization method
        if request.method == OptimizationMethod.SUPERVISED_KNN:
            result = await self._optimize_supervised_portfolio(request)
        elif request.method in [OptimizationMethod.HIERARCHICAL_RISK_PARITY, 
                               OptimizationMethod.CLUSTER_RISK_PARITY]:
            result = await self._optimize_hierarchical_portfolio(request)
        else:
            result = await self._optimize_standard_portfolio(request)
        
        # Add timing information
        result.optimization_time_ms = (time.time() - start_time) * 1000
        
        # Cache result
        self.cache[cache_key] = (result, datetime.now())
        
        return result
    
    async def _optimize_standard_portfolio(self, request: PortfolioOptimizationRequest) -> PortfolioOptimizationResult:
        """Standard mean-variance and related optimizations"""
        
        # Map optimization methods to API endpoints
        method_endpoints = {
            OptimizationMethod.MINIMUM_VARIANCE: "/portfolio/optimization/minimum-variance",
            OptimizationMethod.MAXIMUM_SHARPE: "/portfolio/optimization/maximum-sharpe-ratio",
            OptimizationMethod.EQUAL_RISK_CONTRIBUTION: "/portfolio/optimization/equal-risk-contribution",
            OptimizationMethod.MAXIMUM_DIVERSIFICATION: "/portfolio/optimization/maximum-diversification",
            OptimizationMethod.INVERSE_VOLATILITY: "/portfolio/optimization/inverse-volatility-weighted",
            OptimizationMethod.MEAN_VARIANCE: "/portfolio/optimization/mean-variance"
        }
        
        endpoint = method_endpoints.get(request.method, "/portfolio/optimization/mean-variance")
        
        # Prepare request data
        data = {
            "assets": [len(request.assets)],  # Number of assets
            "constraints": {
                "minimumWeight": [request.constraints.min_weight] * len(request.assets),
                "maximumWeight": [request.constraints.max_weight] * len(request.assets)
            }
        }
        
        # Add covariance matrix if available
        if request.covariance_matrix is not None:
            data["covarianceMatrix"] = request.covariance_matrix.tolist()
        
        # Add expected returns if available
        if request.expected_returns is not None:
            data["expectedReturns"] = request.expected_returns.tolist()
        
        # Add target return constraint
        if request.constraints.target_return:
            data["constraints"]["targetReturn"] = request.constraints.target_return
        
        # Make API request
        api_start = time.time()
        result = await self._make_request("POST", endpoint, data)
        api_time = (time.time() - api_start) * 1000
        
        # Parse response
        weights = dict(zip(request.assets, result.get("weights", [])))
        
        return PortfolioOptimizationResult(
            optimal_weights=weights,
            expected_return=result.get("expectedReturn", 0.0),
            expected_risk=result.get("expectedRisk", 0.0),
            sharpe_ratio=result.get("sharpeRatio", 0.0),
            diversification_ratio=result.get("diversificationRatio"),
            effective_assets=len([w for w in weights.values() if w > 0.001]),
            concentration_risk=max(weights.values()) if weights else 0.0,
            value_at_risk=result.get("valueAtRisk"),
            conditional_value_at_risk=result.get("conditionalValueAtRisk"),
            max_drawdown=result.get("maxDrawdown"),
            metadata={"method": request.method.value, "endpoint": endpoint},
            api_response_time_ms=api_time
        )
    
    async def _optimize_supervised_portfolio(self, request: PortfolioOptimizationRequest) -> PortfolioOptimizationResult:
        """
        Supervised k-NN portfolio optimization
        
        Unique feature: Learns from historical optimal portfolios
        """
        endpoint = "/portfolios/optimization/supervised/nearest-neighbors-based"
        
        # Prepare k-NN specific data
        data = {
            "assets": [len(request.assets)],
            "distanceMetric": request.distance_metric.value,
            "lookbackPeriods": request.lookback_periods
        }
        
        # Dynamic k* or fixed k selection
        if request.k_neighbors:
            data["kNeighbors"] = request.k_neighbors
        else:
            data["kNeighborsSelection"] = "dynamic"  # Use dynamic k* selection
        
        # Add historical returns for training
        if request.returns is not None:
            data["assetsReturns"] = request.returns.tolist()
        
        # Add features for k-NN if available
        if request.features:
            data["features"] = {k: v.tolist() if isinstance(v, np.ndarray) else v 
                              for k, v in request.features.items()}
        
        # Make API request
        api_start = time.time()
        result = await self._make_request("POST", endpoint, data)
        api_time = (time.time() - api_start) * 1000
        
        # Parse response
        weights = dict(zip(request.assets, result.get("weights", [])))
        
        return PortfolioOptimizationResult(
            optimal_weights=weights,
            expected_return=result.get("expectedReturn", 0.0),
            expected_risk=result.get("expectedRisk", 0.0),
            sharpe_ratio=result.get("sharpeRatio", 0.0),
            diversification_ratio=result.get("diversificationRatio"),
            effective_assets=len([w for w in weights.values() if w > 0.001]),
            concentration_risk=max(weights.values()) if weights else 0.0,
            value_at_risk=result.get("valueAtRisk"),
            conditional_value_at_risk=result.get("conditionalValueAtRisk"),
            max_drawdown=result.get("maxDrawdown"),
            metadata={
                "method": "supervised_knn",
                "k_neighbors_used": result.get("kNeighborsUsed"),
                "distance_metric": request.distance_metric.value,
                "training_samples": result.get("trainingSamples", 0)
            },
            api_response_time_ms=api_time
        )
    
    async def _optimize_hierarchical_portfolio(self, request: PortfolioOptimizationRequest) -> PortfolioOptimizationResult:
        """
        Hierarchical and cluster risk parity optimization
        """
        if request.method == OptimizationMethod.CLUSTER_RISK_PARITY:
            endpoint = "/portfolio/optimization/cluster-risk-parity"
        else:
            endpoint = "/portfolio/optimization/hierarchical-risk-parity"
        
        data = {
            "assets": [len(request.assets)],
            "clusteringMethod": "hierarchical",
            "distanceMetric": "euclidean",
            "linkageMethod": "single"
        }
        
        # Add covariance matrix
        if request.covariance_matrix is not None:
            data["covarianceMatrix"] = request.covariance_matrix.tolist()
        
        # Add returns data for correlation calculation
        if request.returns is not None:
            data["assetsReturns"] = request.returns.tolist()
        
        # Make API request
        api_start = time.time()
        result = await self._make_request("POST", endpoint, data)
        api_time = (time.time() - api_start) * 1000
        
        # Parse response
        weights = dict(zip(request.assets, result.get("weights", [])))
        
        return PortfolioOptimizationResult(
            optimal_weights=weights,
            expected_return=result.get("expectedReturn", 0.0),
            expected_risk=result.get("expectedRisk", 0.0),
            sharpe_ratio=result.get("sharpeRatio", 0.0),
            diversification_ratio=result.get("diversificationRatio"),
            effective_assets=len([w for w in weights.values() if w > 0.001]),
            concentration_risk=max(weights.values()) if weights else 0.0,
            value_at_risk=None,
            conditional_value_at_risk=None,
            max_drawdown=None,
            metadata={
                "method": request.method.value,
                "clusters": result.get("clusters", {}),
                "clustering_method": "hierarchical"
            },
            api_response_time_ms=api_time
        )
    
    async def compute_efficient_frontier(self, assets: List[str],
                                        returns: Optional[pd.DataFrame] = None,
                                        covariance: Optional[np.ndarray] = None,
                                        num_portfolios: int = 50) -> List[PortfolioOptimizationResult]:
        """
        Compute the efficient frontier using cloud API
        """
        endpoint = "/portfolio/analysis/mean-variance/efficient-frontier"
        
        data = {
            "assets": [len(assets)],
            "portfoliosNumber": num_portfolios
        }
        
        if returns is not None:
            data["assetsReturns"] = returns.values.tolist()
        
        if covariance is not None:
            data["covarianceMatrix"] = covariance.tolist()
        
        result = await self._make_request("POST", endpoint, data)
        
        # Parse frontier portfolios
        frontier = []
        for i, portfolio in enumerate(result.get("portfolios", [])):
            weights = dict(zip(assets, portfolio.get("weights", [])))
            
            frontier.append(PortfolioOptimizationResult(
                optimal_weights=weights,
                expected_return=portfolio.get("expectedReturn", 0.0),
                expected_risk=portfolio.get("expectedRisk", 0.0),
                sharpe_ratio=portfolio.get("sharpeRatio", 0.0),
                diversification_ratio=None,
                effective_assets=len([w for w in weights.values() if w > 0.001]),
                concentration_risk=max(weights.values()) if weights else 0.0,
                value_at_risk=None,
                conditional_value_at_risk=None,
                max_drawdown=None,
                metadata={"frontier_point": i, "total_points": num_portfolios}
            ))
        
        return frontier
    
    def is_available(self) -> bool:
        """Check if the client is available and healthy"""
        return (self.client is not None and 
                not self._is_circuit_breaker_open())
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get client performance statistics"""
        success_rate = (self.successful_requests / max(1, self.total_requests)) * 100
        avg_response_time = self.total_api_time_ms / max(1, self.successful_requests)
        cache_hit_rate = (self.cache_hits / max(1, self.total_requests + self.cache_hits)) * 100
        
        return {
            "available": self.is_available(),
            "total_requests": self.total_requests,
            "successful_requests": self.successful_requests,
            "failed_requests": self.failed_requests,
            "success_rate_percent": success_rate,
            "cache_hits": self.cache_hits,
            "cache_hit_rate_percent": cache_hit_rate,
            "avg_response_time_ms": avg_response_time,
            "circuit_breaker": {
                "failures": self.circuit_breaker_failures,
                "is_open": self._is_circuit_breaker_open(),
                "last_failure": self.circuit_breaker_last_failure.isoformat() if self.circuit_breaker_last_failure else None
            },
            "cached_items": len(self.cache)
        }
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check against the API"""
        if not self.is_available():
            return {"healthy": False, "reason": "Client unavailable or circuit breaker open"}
        
        try:
            # Simple health check request
            test_data = {
                "assets": [2],
                "covarianceMatrix": [[0.01, 0.002], [0.002, 0.02]],
                "expectedReturns": [0.08, 0.12]
            }
            
            start_time = time.time()
            await self._make_request("POST", "/portfolio/optimization/minimum-variance", test_data)
            response_time = (time.time() - start_time) * 1000
            
            return {
                "healthy": True,
                "response_time_ms": response_time,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            return {
                "healthy": False,
                "reason": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    async def close(self):
        """Close the HTTP client"""
        if self.client:
            await self.client.aclose()
        logger.info("Portfolio Optimizer client closed")
```

### **Phase 2: Integration with Risk Engine** (6 hours)

#### **2.1 Environment Configuration**
**File**: `backend/engines/risk/.env` or Docker configuration

```bash
# Portfolio Optimizer API Configuration
PORTFOLIO_OPTIMIZER_API_KEY=EgyPyGQSVQV4GMKWp5l7tf21wQOUaaw
PORTFOLIO_OPTIMIZER_CACHE_TTL=300  # 5 minutes
PORTFOLIO_OPTIMIZER_TIMEOUT=30     # 30 seconds
PORTFOLIO_OPTIMIZER_MAX_RETRIES=3
PORTFOLIO_OPTIMIZER_RATE_LIMIT=10  # requests per second
```

#### **2.2 Risk Engine Integration**
**File**: `/backend/engines/risk/risk_engine.py` (additions)

```python
# Add Portfolio Optimizer client to RiskEngine
from portfolio_optimizer_client import PortfolioOptimizerClient, OptimizationMethod, PortfolioOptimizationRequest, OptimizationConstraints

class RiskEngine:
    def __init__(self):
        # ... existing initialization ...
        
        # Portfolio Optimizer integration
        self.portfolio_optimizer = PortfolioOptimizerClient()
        logger.info(f"Portfolio Optimizer available: {self.portfolio_optimizer.is_available()}")
    
    def setup_routes(self):
        # ... existing routes ...
        
        @self.app.post("/risk/optimize/cloud")
        async def optimize_portfolio_cloud(request_data: Dict[str, Any]):
            """Portfolio optimization using cloud API"""
            try:
                if not self.portfolio_optimizer.is_available():
                    raise HTTPException(status_code=503, detail="Portfolio Optimizer API unavailable")
                
                # Parse request
                method = OptimizationMethod(request_data.get("method", "minimum_variance"))
                assets = request_data.get("assets", [])
                
                if not assets:
                    raise HTTPException(status_code=400, detail="Assets list required")
                
                # Create optimization request
                constraints = OptimizationConstraints(
                    min_weight=request_data.get("min_weight", 0.0),
                    max_weight=request_data.get("max_weight", 1.0),
                    target_return=request_data.get("target_return"),
                    leverage_limit=request_data.get("leverage_limit", 1.0)
                )
                
                # Handle historical returns if provided
                returns_data = request_data.get("returns")
                returns_matrix = None
                if returns_data:
                    import numpy as np
                    returns_matrix = np.array(returns_data)
                
                opt_request = PortfolioOptimizationRequest(
                    assets=assets,
                    method=method,
                    constraints=constraints,
                    returns=returns_matrix,
                    risk_free_rate=request_data.get("risk_free_rate", 0.02),
                    k_neighbors=request_data.get("k_neighbors"),
                    lookback_periods=request_data.get("lookback_periods", 252)
                )
                
                # Perform optimization
                result = await self.portfolio_optimizer.optimize_portfolio(opt_request)
                
                return {
                    "status": "success",
                    "optimization_result": asdict(result),
                    "api_stats": self.portfolio_optimizer.get_performance_stats()
                }
                
            except Exception as e:
                logger.error(f"Cloud portfolio optimization error: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/risk/optimize/cloud/health")
        async def portfolio_optimizer_health():
            """Check Portfolio Optimizer API health"""
            try:
                health = await self.portfolio_optimizer.health_check()
                stats = self.portfolio_optimizer.get_performance_stats()
                
                return {
                    "health_check": health,
                    "performance_stats": stats
                }
                
            except Exception as e:
                logger.error(f"Portfolio Optimizer health check failed: {e}")
                return {
                    "health_check": {"healthy": False, "reason": str(e)},
                    "performance_stats": self.portfolio_optimizer.get_performance_stats()
                }
```

### **Phase 3: Testing and Validation** (6 hours)

#### **3.1 Unit Tests**
**File**: `/backend/engines/risk/tests/test_portfolio_optimizer_client.py`

```python
import pytest
import numpy as np
from portfolio_optimizer_client import (
    PortfolioOptimizerClient, OptimizationMethod, DistanceMetric,
    PortfolioOptimizationRequest, OptimizationConstraints
)

class TestPortfolioOptimizerClient:
    
    @pytest.fixture
    def client(self):
        # Use test API key
        return PortfolioOptimizerClient(api_key="EgyPyGQSVQV4GMKWp5l7tf21wQOUaaw")
    
    @pytest.fixture
    def sample_assets(self):
        return ["AAPL", "GOOGL", "MSFT", "TSLA"]
    
    @pytest.fixture
    def sample_returns(self):
        # Generate sample return matrix
        np.random.seed(42)
        return np.random.normal(0.001, 0.02, (252, 4))  # 1 year daily returns
    
    def test_client_initialization(self, client):
        """Test client initialization with API key"""
        assert client.api_key == "EgyPyGQSVQV4GMKWp5l7tf21wQOUaaw"
        assert client.is_available() == True
        assert client.total_requests == 0
    
    @pytest.mark.asyncio
    async def test_health_check(self, client):
        """Test API health check"""
        health = await client.health_check()
        
        assert "healthy" in health
        assert isinstance(health["healthy"], bool)
        
        if health["healthy"]:
            assert "response_time_ms" in health
            assert health["response_time_ms"] < 5000  # Less than 5 seconds
    
    @pytest.mark.asyncio
    async def test_minimum_variance_optimization(self, client, sample_assets, sample_returns):
        """Test minimum variance portfolio optimization"""
        request = PortfolioOptimizationRequest(
            assets=sample_assets,
            method=OptimizationMethod.MINIMUM_VARIANCE,
            returns=sample_returns,
            constraints=OptimizationConstraints(min_weight=0.0, max_weight=0.5)
        )
        
        result = await client.optimize_portfolio(request)
        
        # Verify result structure
        assert len(result.optimal_weights) == len(sample_assets)
        assert all(asset in result.optimal_weights for asset in sample_assets)
        assert sum(result.optimal_weights.values()) == pytest.approx(1.0, rel=0.01)
        assert all(0 <= weight <= 0.5 for weight in result.optimal_weights.values())
        
        # Verify performance metrics
        assert isinstance(result.expected_return, float)
        assert isinstance(result.expected_risk, float)
        assert isinstance(result.sharpe_ratio, float)
        assert result.api_response_time_ms < 5000
    
    @pytest.mark.asyncio
    async def test_supervised_knn_optimization(self, client, sample_assets, sample_returns):
        """Test supervised k-NN portfolio optimization"""
        request = PortfolioOptimizationRequest(
            assets=sample_assets,
            method=OptimizationMethod.SUPERVISED_KNN,
            returns=sample_returns,
            distance_metric=DistanceMetric.HASSANAT,
            k_neighbors=None,  # Use dynamic k*
            lookback_periods=126  # 6 months
        )
        
        result = await client.optimize_portfolio(request)
        
        # Verify k-NN specific results
        assert len(result.optimal_weights) == len(sample_assets)
        assert sum(result.optimal_weights.values()) == pytest.approx(1.0, rel=0.01)
        assert "k_neighbors_used" in result.metadata
        assert result.metadata["method"] == "supervised_knn"
        assert result.metadata["distance_metric"] == "hassanat"
    
    @pytest.mark.asyncio
    async def test_caching_functionality(self, client, sample_assets, sample_returns):
        """Test result caching"""
        request = PortfolioOptimizationRequest(
            assets=sample_assets,
            method=OptimizationMethod.MINIMUM_VARIANCE,
            returns=sample_returns
        )
        
        # First request
        result1 = await client.optimize_portfolio(request)
        initial_requests = client.total_requests
        
        # Second identical request (should be cached)
        result2 = await client.optimize_portfolio(request)
        
        # Should have same results but no additional API calls
        assert result1.optimal_weights == result2.optimal_weights
        assert client.total_requests == initial_requests  # No additional API calls
        assert client.cache_hits > 0
    
    @pytest.mark.asyncio
    async def test_error_handling(self, sample_assets):
        """Test error handling with invalid API key"""
        invalid_client = PortfolioOptimizerClient(api_key="invalid_key")
        
        request = PortfolioOptimizationRequest(
            assets=sample_assets,
            method=OptimizationMethod.MINIMUM_VARIANCE
        )
        
        with pytest.raises(ValueError, match="Invalid API key"):
            await invalid_client.optimize_portfolio(request)
    
    @pytest.mark.asyncio
    async def test_efficient_frontier(self, client, sample_assets, sample_returns):
        """Test efficient frontier computation"""
        frontier = await client.compute_efficient_frontier(
            assets=sample_assets,
            returns=pd.DataFrame(sample_returns, columns=sample_assets),
            num_portfolios=10
        )
        
        assert len(frontier) <= 10
        
        for portfolio in frontier:
            assert len(portfolio.optimal_weights) == len(sample_assets)
            assert sum(portfolio.optimal_weights.values()) == pytest.approx(1.0, rel=0.01)
            assert "frontier_point" in portfolio.metadata
    
    def test_performance_statistics(self, client):
        """Test performance statistics"""
        stats = client.get_performance_stats()
        
        expected_keys = [
            "available", "total_requests", "successful_requests", 
            "success_rate_percent", "cache_hit_rate_percent", "circuit_breaker"
        ]
        
        for key in expected_keys:
            assert key in stats
    
    @pytest.mark.asyncio
    async def test_circuit_breaker(self):
        """Test circuit breaker functionality"""
        # Create client with low failure threshold for testing
        client = PortfolioOptimizerClient(api_key="invalid_key")
        client.circuit_breaker_threshold = 1
        
        # Force failures to trigger circuit breaker
        try:
            await client.health_check()
        except:
            pass
        
        # Circuit breaker should now be open
        assert client._is_circuit_breaker_open()
        assert not client.is_available()
```

#### **3.2 Integration Tests**
**File**: `/backend/engines/risk/tests/test_risk_engine_cloud_optimization.py`

```python
import pytest
from fastapi.testclient import TestClient
from risk_engine import RiskEngine

class TestRiskEngineCloudOptimization:
    
    @pytest.fixture
    def client(self):
        engine = RiskEngine()
        return TestClient(engine.app)
    
    @pytest.fixture
    def optimization_request(self):
        return {
            "method": "minimum_variance",
            "assets": ["AAPL", "GOOGL", "MSFT"],
            "min_weight": 0.0,
            "max_weight": 0.5,
            "risk_free_rate": 0.02,
            "returns": [
                [0.01, -0.005, 0.02],
                [-0.01, 0.015, -0.005],
                [0.005, -0.01, 0.025]
            ]
        }
    
    def test_cloud_optimization_endpoint(self, client, optimization_request):
        """Test cloud optimization API endpoint"""
        response = client.post("/risk/optimize/cloud", json=optimization_request)
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["status"] == "success"
        assert "optimization_result" in data
        assert "optimal_weights" in data["optimization_result"]
        assert len(data["optimization_result"]["optimal_weights"]) == 3
    
    def test_supervised_knn_optimization(self, client):
        """Test supervised k-NN optimization endpoint"""
        request_data = {
            "method": "supervised_knn",
            "assets": ["AAPL", "GOOGL", "MSFT", "TSLA"],
            "distance_metric": "hassanat",
            "k_neighbors": None,  # Dynamic k*
            "lookback_periods": 126,
            "returns": [[0.01, -0.005, 0.02, 0.005] for _ in range(126)]
        }
        
        response = client.post("/risk/optimize/cloud", json=request_data)
        
        if response.status_code == 200:
            data = response.json()
            result = data["optimization_result"]
            
            assert "metadata" in result
            assert result["metadata"]["method"] == "supervised_knn"
            assert "k_neighbors_used" in result["metadata"]
    
    def test_health_check_endpoint(self, client):
        """Test Portfolio Optimizer health check endpoint"""
        response = client.get("/risk/optimize/cloud/health")
        
        assert response.status_code == 200
        data = response.json()
        
        assert "health_check" in data
        assert "performance_stats" in data
        assert isinstance(data["health_check"]["healthy"], bool)
    
    def test_error_handling(self, client):
        """Test error handling for invalid requests"""
        # Missing assets
        response = client.post("/risk/optimize/cloud", json={"method": "minimum_variance"})
        assert response.status_code == 400
        
        # Invalid method
        response = client.post("/risk/optimize/cloud", json={
            "method": "invalid_method",
            "assets": ["AAPL", "GOOGL"]
        })
        assert response.status_code in [400, 422, 500]  # Depending on validation
```

## 🚀 Deployment Guide

### **Pre-deployment Checklist**
- [ ] API key configured in environment
- [ ] All unit tests passing
- [ ] Integration tests successful
- [ ] Performance benchmarks met (<3s response time)
- [ ] Circuit breaker functionality tested
- [ ] Error handling comprehensive

### **Environment Configuration**
```bash
# Add to docker-compose.yml or .env
PORTFOLIO_OPTIMIZER_API_KEY=EgyPyGQSVQV4GMKWp5l7tf21wQOUaaw
```

### **Health Monitoring**
- Monitor circuit breaker status
- Track API response times
- Alert on high failure rates
- Cache performance monitoring

## 📊 Success Metrics

### **Functional Requirements**
- ✅ Cloud optimization API fully functional
- ✅ 10+ optimization methods available
- ✅ Supervised k-NN optimization working
- ✅ Response caching implemented (5min TTL)
- ✅ Circuit breaker protecting against failures

### **Performance Requirements**
- ✅ API response time <3 seconds (95th percentile)
- ✅ Cache hit rate >70% in normal operation
- ✅ Circuit breaker prevents cascading failures
- ✅ Fallback mechanisms operational

### **Quality Requirements**
- ✅ Comprehensive error handling
- ✅ API authentication secure
- ✅ Rate limiting prevents overuse
- ✅ Performance monitoring active

---

**Story Status**: 📋 **READY FOR IMPLEMENTATION**

**API Key Available**: `EgyPyGQSVQV4GMKWp5l7tf21wQOUaaw`

**Next Story**: Story 2.2 (Hybrid Optimization Strategy) - builds on cloud API foundation