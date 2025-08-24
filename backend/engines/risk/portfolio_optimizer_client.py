"""
Portfolio Optimizer API Client
==============================

Cloud-based portfolio optimization service integration for Nautilus.
Provides access to institutional-grade optimization algorithms including
unique supervised ML portfolio optimization using k-NN.

API Documentation: https://docs.portfoliooptimizer.io/
"""

import asyncio
import logging
import time
from typing import Dict, List, Optional, Any, Tuple, Union, Callable
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from enum import Enum
import hashlib
import json
import numpy as np
import pandas as pd
import httpx
from httpx import AsyncClient, Response, ConnectError, TimeoutException, RequestError
from contextlib import asynccontextmanager
import weakref

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
    BLACK_LITTERMAN = "black-litterman"
    ROBUST_OPTIMIZATION = "robust-optimization"
    BAYESIAN_OPTIMIZATION = "bayesian-optimization"


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
    features: Optional[Dict[str, Any]] = None  # Additional features for k-NN
    
    # Advanced optimization parameters
    regime_detection: bool = True  # Enable market regime detection
    stability_analysis: bool = True  # Perform stability analysis
    bootstrap_samples: int = 100  # For robust optimization
    confidence_intervals: bool = True  # Calculate weight confidence intervals


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


class CircuitBreakerState(Enum):
    """Circuit breaker states"""
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class CircuitBreaker:
    """
    Circuit breaker implementation for API reliability
    
    Prevents cascading failures by stopping requests when error rate is high
    """
    
    def __init__(self, failure_threshold: int = 5, recovery_timeout: int = 60, expected_exception: type = Exception):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exception = expected_exception
        
        self.failure_count = 0
        self.last_failure_time: Optional[datetime] = None
        self.state = CircuitBreakerState.CLOSED
        
    def can_execute(self) -> bool:
        """Check if request can be executed"""
        if self.state == CircuitBreakerState.CLOSED:
            return True
        elif self.state == CircuitBreakerState.OPEN:
            if self.last_failure_time and (datetime.now() - self.last_failure_time).seconds >= self.recovery_timeout:
                self.state = CircuitBreakerState.HALF_OPEN
                return True
            return False
        elif self.state == CircuitBreakerState.HALF_OPEN:
            return True
        return False
    
    def record_success(self):
        """Record successful operation"""
        self.failure_count = 0
        self.state = CircuitBreakerState.CLOSED
    
    def record_failure(self):
        """Record failed operation"""
        self.failure_count += 1
        self.last_failure_time = datetime.now()
        
        if self.failure_count >= self.failure_threshold:
            self.state = CircuitBreakerState.OPEN
        elif self.state == CircuitBreakerState.HALF_OPEN:
            self.state = CircuitBreakerState.OPEN
    
    @asynccontextmanager
    async def execute(self):
        """Execute operation with circuit breaker protection"""
        if not self.can_execute():
            raise CircuitBreakerOpenError("Circuit breaker is open")
        
        try:
            yield
            self.record_success()
        except self.expected_exception as e:
            self.record_failure()
            raise e


class CircuitBreakerOpenError(Exception):
    """Circuit breaker is open"""
    pass


class PortfolioOptimizerClient:
    """
    Client for Portfolio Optimizer API
    
    Features:
    - 10+ optimization methods including supervised ML
    - Cloud-based computation
    - Automatic retry with exponential backoff
    - Result caching for efficiency
    """
    
    BASE_URL = "https://api.portfoliooptimizer.io/v1"
    
    def __init__(self, api_key: Optional[str] = None, cache_ttl_seconds: int = 300):
        self.api_key = api_key
        self.cache_ttl = cache_ttl_seconds
        self.cache: Dict[str, Tuple[Any, datetime]] = {}
        
        # HTTP client configuration with retries and timeouts
        self.client = AsyncClient(
            base_url=self.BASE_URL,
            timeout=httpx.Timeout(30.0, connect=10.0),
            headers=self._get_headers(),
            limits=httpx.Limits(max_keepalive_connections=10, max_connections=20),
            follow_redirects=True
        )
        
        # Circuit breaker configuration
        self.circuit_breaker = CircuitBreaker(
            failure_threshold=5,
            recovery_timeout=60,
            expected_exception=Exception
        )
        
        # Rate limiting
        self.last_request_time = 0
        self.min_request_interval = 0.1  # 10 requests/second max
        self.request_semaphore = asyncio.Semaphore(5)  # Max 5 concurrent requests
        
        # Performance tracking
        self.total_requests = 0
        self.total_api_time_ms = 0
        self.cache_hits = 0
        self.circuit_breaker_trips = 0
        self.error_count = 0
        self.success_count = 0
        
        # Health monitoring
        self.health_status = "healthy"
        self.last_health_check = datetime.now()
        self.consecutive_failures = 0
        
        # Enhanced caching with LRU eviction
        self.max_cache_size = 1000
        self.cache_access_times: Dict[str, datetime] = {}
        
        logger.info(f"Enhanced Portfolio Optimizer client initialized (authenticated: {bool(api_key)})")
    
    def _get_headers(self) -> Dict[str, str]:
        """Get HTTP headers with optional API key"""
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json"
        }
        if self.api_key:
            headers["X-API-Key"] = self.api_key
        return headers
    
    async def _rate_limit(self):
        """Enforce rate limiting"""
        elapsed = time.time() - self.last_request_time
        if elapsed < self.min_request_interval:
            await asyncio.sleep(self.min_request_interval - elapsed)
        self.last_request_time = time.time()
    
    async def _make_request(self, method: str, endpoint: str, 
                          data: Optional[Dict] = None, 
                          retries: int = 3) -> Dict[str, Any]:
        """Make API request with circuit breaker, retry logic, and enhanced error handling"""
        async with self.request_semaphore:  # Limit concurrent requests
            try:
                async with self.circuit_breaker.execute():
                    return await self._execute_request(method, endpoint, data, retries)
            except CircuitBreakerOpenError:
                self.circuit_breaker_trips += 1
                self.health_status = "circuit_breaker_open"
                logger.warning("Circuit breaker is open, failing fast")
                raise RuntimeError("Portfolio Optimizer service temporarily unavailable (circuit breaker open)")
    
    async def _execute_request(self, method: str, endpoint: str, 
                             data: Optional[Dict] = None, 
                             retries: int = 3) -> Dict[str, Any]:
        """Execute HTTP request with comprehensive error handling"""
        await self._rate_limit()
        
        last_exception = None
        
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
                
                # Handle response
                if response.status_code == 200:
                    self.success_count += 1
                    self.consecutive_failures = 0
                    self.health_status = "healthy"
                    return response.json()
                    
                elif response.status_code == 429:  # Rate limited
                    retry_after = int(response.headers.get("Retry-After", 5))
                    logger.warning(f"Rate limited, retrying after {retry_after}s (attempt {attempt + 1})")
                    await asyncio.sleep(retry_after)
                    continue
                    
                elif response.status_code == 401:
                    self.error_count += 1
                    raise ValueError("Invalid API key - check PORTFOLIO_OPTIMIZER_API_KEY environment variable")
                    
                elif response.status_code == 400:
                    self.error_count += 1
                    error_detail = response.text
                    raise ValueError(f"Bad request - invalid parameters: {error_detail}")
                    
                elif response.status_code == 403:
                    self.error_count += 1
                    raise ValueError("API access forbidden - check API key permissions")
                    
                elif response.status_code == 404:
                    self.error_count += 1
                    raise ValueError(f"API endpoint not found: {endpoint}")
                    
                elif response.status_code >= 500:
                    # Server error - retry with exponential backoff
                    self.error_count += 1
                    wait_time = min(60, (2 ** attempt) + np.random.uniform(0, 1))
                    logger.warning(f"Server error {response.status_code}, retrying in {wait_time:.1f}s")
                    await asyncio.sleep(wait_time)
                    continue
                    
                else:
                    self.error_count += 1
                    logger.error(f"Unexpected API error {response.status_code}: {response.text}")
                    if attempt < retries - 1:
                        await asyncio.sleep(2 ** attempt)
                        continue
                    else:
                        raise RuntimeError(f"API returned status {response.status_code}: {response.text}")
                    
            except (ConnectError, TimeoutException, RequestError) as e:
                self.error_count += 1
                self.consecutive_failures += 1
                last_exception = e
                
                if self.consecutive_failures >= 3:
                    self.health_status = "unhealthy"
                
                logger.error(f"Network error (attempt {attempt + 1}/{retries}): {e}")
                if attempt < retries - 1:
                    wait_time = min(30, (2 ** attempt) + np.random.uniform(0, 1))
                    await asyncio.sleep(wait_time)
                else:
                    raise RuntimeError(f"Network error after {retries} attempts: {str(e)}")
                    
            except Exception as e:
                self.error_count += 1
                self.consecutive_failures += 1
                last_exception = e
                logger.error(f"Unexpected error (attempt {attempt + 1}/{retries}): {e}")
                
                if attempt < retries - 1:
                    await asyncio.sleep(2 ** attempt)
                else:
                    raise RuntimeError(f"Request failed after {retries} attempts: {str(e)}")
        
        # Should not reach here, but just in case
        if last_exception:
            raise last_exception
        raise RuntimeError(f"Failed to complete request after {retries} attempts")
    
    def _get_cache_key(self, method: str, **kwargs) -> str:
        """Generate cache key for request with enhanced hashing"""
        # Create deterministic key from method and parameters
        params_str = json.dumps(kwargs, sort_keys=True, default=str)
        key_hash = hashlib.md5(params_str.encode()).hexdigest()[:16]
        return f"{method}:{key_hash}"
    
    def _check_cache(self, cache_key: str) -> Optional[Any]:
        """Check if result is in cache and still valid with LRU eviction"""
        if cache_key in self.cache:
            result, timestamp = self.cache[cache_key]
            if (datetime.now() - timestamp).total_seconds() < self.cache_ttl:
                # Update access time for LRU
                self.cache_access_times[cache_key] = datetime.now()
                self.cache_hits += 1
                return result
            else:
                # Remove expired entry
                del self.cache[cache_key]
                if cache_key in self.cache_access_times:
                    del self.cache_access_times[cache_key]
        return None
    
    def _evict_cache_if_needed(self):
        """Evict oldest cache entries if cache is too large"""
        if len(self.cache) >= self.max_cache_size:
            # Remove 20% of oldest entries
            num_to_remove = max(1, self.max_cache_size // 5)
            
            # Sort by access time and remove oldest
            sorted_keys = sorted(
                self.cache_access_times.keys(),
                key=lambda k: self.cache_access_times[k]
            )
            
            for key in sorted_keys[:num_to_remove]:
                if key in self.cache:
                    del self.cache[key]
                if key in self.cache_access_times:
                    del self.cache_access_times[key]
    
    def _cache_result(self, cache_key: str, result: Any):
        """Cache result with LRU management"""
        self._evict_cache_if_needed()
        self.cache[cache_key] = (result, datetime.now())
        self.cache_access_times[cache_key] = datetime.now()
    
    async def compute_returns(self, prices: pd.DataFrame, 
                            method: str = "arithmetic") -> pd.DataFrame:
        """
        Compute asset returns from prices
        
        Args:
            prices: DataFrame with asset prices
            method: "arithmetic" or "logarithmic"
        """
        endpoint = "/assets/returns"
        
        data = {
            "assets": prices.columns.tolist(),
            "assetsPrices": prices.values.tolist(),
            "method": method
        }
        
        result = await self._make_request("POST", endpoint, data)
        
        return pd.DataFrame(
            result["returns"],
            columns=prices.columns[:-1]  # Returns have one less period
        )
    
    async def estimate_covariance(self, returns: pd.DataFrame,
                                 method: str = "empirical",
                                 **kwargs) -> Dict[str, Any]:
        """
        Professional covariance matrix estimation with multiple methods
        
        Methods:
        - empirical: Sample covariance matrix
        - shrinkage: Ledoit-Wolf shrinkage estimator
        - factor-model: Multi-factor model covariance
        - robust: Robust covariance estimation
        - exponential: Exponentially weighted covariance
        - minimum-variance: Minimum variance estimator
        
        Returns both covariance matrix and estimation metadata
        """
        method_endpoints = {
            "empirical": "/assets/covariance/matrix/estimation/empirical",
            "shrinkage": "/assets/covariance/matrix/estimation/shrinkage",
            "factor-model": "/assets/covariance/matrix/estimation/factor-model",
            "robust": "/assets/covariance/matrix/estimation/robust", 
            "exponential": "/assets/covariance/matrix/estimation/exponential-weighted",
            "minimum-variance": "/assets/covariance/matrix/estimation/minimum-variance"
        }
        
        endpoint = method_endpoints.get(method, method_endpoints["empirical"])
        
        # Enhanced data preparation
        data = {
            "assets": returns.columns.tolist(),
            "assetsReturns": returns.values.tolist(),
            "estimationMethod": method
        }
        
        # Method-specific parameters
        if method == "shrinkage":
            data["shrinkageTarget"] = kwargs.get("shrinkage_target", "identity")
            data["shrinkageIntensity"] = kwargs.get("shrinkage_intensity", "automatic")
        
        elif method == "factor-model":
            data["factorCount"] = kwargs.get("factor_count", 3)
            data["factorModel"] = kwargs.get("factor_model", "pca")
        
        elif method == "robust":
            data["robustMethod"] = kwargs.get("robust_method", "minimum_covariance_determinant")
            data["contaminationLevel"] = kwargs.get("contamination_level", 0.1)
        
        elif method == "exponential":
            data["decayFactor"] = kwargs.get("decay_factor", 0.94)
            data["minPeriods"] = kwargs.get("min_periods", 30)
        
        # Cache key for covariance estimation
        cache_key = self._get_cache_key(
            f"covariance_{method}",
            assets=returns.columns.tolist(),
            periods=len(returns),
            **kwargs
        )
        
        cached_result = self._check_cache(cache_key)
        if cached_result:
            logger.debug(f"Cache hit for covariance estimation {method}")
            return cached_result
        
        # Make API request
        api_start = time.time()
        result = await self._make_request("POST", endpoint, data)
        api_time = (time.time() - api_start) * 1000
        
        # Enhanced response with metadata
        covariance_result = {
            "covariance_matrix": np.array(result["covarianceMatrix"]),
            "correlation_matrix": np.array(result.get("correlationMatrix", [])),
            "eigenvalues": result.get("eigenvalues", []),
            "condition_number": result.get("conditionNumber"),
            "estimation_method": method,
            "estimation_quality": {
                "is_positive_definite": result.get("isPositiveDefinite", True),
                "condition_number_acceptable": result.get("conditionNumber", 1) < 1000,
                "min_eigenvalue": min(result.get("eigenvalues", [1])),
                "max_eigenvalue": max(result.get("eigenvalues", [1]))
            },
            "metadata": {
                "assets": returns.columns.tolist(),
                "observation_count": len(returns),
                "estimation_time_ms": api_time,
                "shrinkage_intensity": result.get("shrinkageIntensity"),
                "explained_variance": result.get("explainedVariance")
            }
        }
        
        # Cache the result
        self._cache_result(cache_key, covariance_result)
        
        return covariance_result
    
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
            logger.debug(f"Cache hit for {request.method.value}")
            # Update cached result metadata
            if hasattr(cached_result, 'metadata'):
                cached_result.metadata["cached"] = True
            return cached_result
        
        # Prepare request based on method
        if request.method == OptimizationMethod.SUPERVISED_KNN:
            result = await self._optimize_supervised_portfolio(request)
        elif request.method in [OptimizationMethod.HIERARCHICAL_RISK_PARITY, 
                               OptimizationMethod.CLUSTER_RISK_PARITY]:
            result = await self._optimize_hierarchical_portfolio(request)
        else:
            result = await self._optimize_standard_portfolio(request)
        
        # Add timing information
        result.optimization_time_ms = (time.time() - start_time) * 1000
        
        # Add metadata about caching and performance
        result.metadata.update({
            "cached": False,
            "cache_key": cache_key,
            "client_health": self.health_status,
            "circuit_breaker_state": self.circuit_breaker.state.value
        })
        
        # Cache result with LRU management
        self._cache_result(cache_key, result)
        
        return result
    
    async def _optimize_standard_portfolio(self, request: PortfolioOptimizationRequest) -> PortfolioOptimizationResult:
        """Standard mean-variance and related optimizations"""
        
        # Determine endpoint based on method
        method_endpoints = {
            OptimizationMethod.MINIMUM_VARIANCE: "/portfolio/optimization/minimum-variance",
            OptimizationMethod.MAXIMUM_SHARPE: "/portfolio/optimization/maximum-sharpe-ratio",
            OptimizationMethod.EQUAL_RISK_CONTRIBUTION: "/portfolio/optimization/equal-risk-contribution",
            OptimizationMethod.MAXIMUM_DIVERSIFICATION: "/portfolio/optimization/maximum-diversification",
            OptimizationMethod.INVERSE_VOLATILITY: "/portfolio/optimization/inverse-volatility",
        }
        
        endpoint = method_endpoints.get(
            request.method,
            "/portfolio/optimization/mean-variance"
        )
        
        # Prepare request data
        data = {
            "assets": request.assets,
            "constraints": {
                "minimumWeight": request.constraints.min_weight,
                "maximumWeight": request.constraints.max_weight
            }
        }
        
        if request.covariance_matrix is not None:
            data["covarianceMatrix"] = request.covariance_matrix.tolist()
        
        if request.expected_returns is not None:
            data["expectedReturns"] = request.expected_returns.tolist()
        
        if request.constraints.target_return:
            data["constraints"]["targetReturn"] = request.constraints.target_return
        
        # Make API request
        api_start = time.time()
        result = await self._make_request("POST", endpoint, data)
        api_time = (time.time() - api_start) * 1000
        
        # Parse response
        weights = {asset: weight for asset, weight in 
                  zip(request.assets, result["weights"])}
        
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
            metadata={"method": request.method.value},
            api_response_time_ms=api_time
        )
    
    async def _optimize_supervised_portfolio(self, request: PortfolioOptimizationRequest) -> PortfolioOptimizationResult:
        """
        Enhanced Supervised k-NN portfolio optimization with dynamic k* selection
        
        World's first implementation of supervised ML portfolio optimization:
        - Uses Hassanat distance metric for scale-invariant similarity
        - Dynamic k* selection optimizes neighbor count automatically  
        - Learns from historical optimal portfolios and market regimes
        - Incorporates regime detection for adaptive optimization
        """
        endpoint = "/portfolios/optimization/supervised/nearest-neighbors-based"
        
        # Enhanced data preparation with regime detection
        data = {
            "assets": request.assets,
            "distanceMetric": request.distance_metric.value,
            "lookbackPeriods": request.lookback_periods,
            "riskFreeRate": request.risk_free_rate,
            "confidenceLevel": request.confidence_level
        }
        
        # Dynamic k* selection with cross-validation
        if request.k_neighbors:
            data["kNeighbors"] = request.k_neighbors
            data["kNeighborsSelection"] = "fixed"
        else:
            data["kNeighborsSelection"] = "dynamic"
            data["kNeighborsRange"] = [3, min(50, max(10, request.lookback_periods // 5))]
            data["validationMethod"] = "cross_validation"
            data["cvFolds"] = 5
        
        # Enhanced feature engineering for k-NN
        features_data = {
            "includeMarketRegime": True,
            "includeVolatilityRegime": True,
            "includeMomentumFeatures": True,
            "includeSeasonality": True
        }
        
        if request.features:
            features_data.update(request.features)
        
        data["featureEngineering"] = features_data
        
        # Historical returns data (required for supervised learning)
        if request.returns is not None:
            data["assetsReturns"] = request.returns.tolist()
        else:
            raise ValueError("Historical returns data is required for supervised k-NN optimization")
        
        # Covariance matrix for enhanced distance calculation
        if request.covariance_matrix is not None:
            data["covarianceMatrix"] = request.covariance_matrix.tolist()
        
        # Expected returns for target optimization
        if request.expected_returns is not None:
            data["expectedReturns"] = request.expected_returns.tolist()
        
        # Advanced constraints for supervised optimization
        constraints = {
            "minimumWeight": request.constraints.min_weight,
            "maximumWeight": request.constraints.max_weight,
            "leverageLimit": request.constraints.leverage_limit
        }
        
        if request.constraints.target_return:
            constraints["targetReturn"] = request.constraints.target_return
        
        if request.constraints.target_risk:
            constraints["targetRisk"] = request.constraints.target_risk
            
        if request.constraints.max_assets:
            constraints["maxAssets"] = request.constraints.max_assets
            
        if request.constraints.turnover_limit:
            constraints["turnoverLimit"] = request.constraints.turnover_limit
        
        data["constraints"] = constraints
        
        # Risk measure for optimization
        data["riskMeasure"] = request.risk_measure.value
        
        # Make API request with enhanced error handling
        api_start = time.time()
        try:
            result = await self._make_request("POST", endpoint, data)
        except ValueError as e:
            if "Historical returns" in str(e):
                raise ValueError("Supervised k-NN requires at least 50 periods of historical returns")
            raise e
        
        api_time = (time.time() - api_start) * 1000
        
        # Parse enhanced response
        weights = {asset: weight for asset, weight in 
                  zip(request.assets, result["weights"])}
        
        # Extract advanced metadata from supervised optimization
        metadata = {
            "method": "supervised_knn",
            "distance_metric": request.distance_metric.value,
            "k_neighbors_used": result.get("kNeighborsUsed"),
            "k_neighbors_optimal": result.get("kNeighborsOptimal"),
            "cross_validation_score": result.get("crossValidationScore"),
            "regime_detected": result.get("marketRegime"),
            "volatility_regime": result.get("volatilityRegime"),
            "similarity_scores": result.get("similarityScores", []),
            "neighbor_portfolios_count": result.get("neighborPortfoliosCount"),
            "feature_importance": result.get("featureImportance", {}),
            "optimization_convergence": result.get("convergenceStatus"),
            "hassanat_distance_stats": result.get("distanceStatistics")
        }
        
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
            metadata=metadata,
            api_response_time_ms=api_time
        )
    
    async def _optimize_hierarchical_portfolio(self, request: PortfolioOptimizationRequest) -> PortfolioOptimizationResult:
        """
        Advanced Hierarchical and Cluster Risk Parity optimization
        
        Enhanced features:
        - Multiple clustering algorithms (hierarchical, k-means, DBSCAN)
        - Advanced distance metrics (correlation, euclidean, mahalanobis)
        - Dynamic cluster count optimization
        - Regime-aware clustering for different market conditions
        - Cluster stability analysis
        """
        # Select endpoint based on method
        if request.method == OptimizationMethod.CLUSTER_RISK_PARITY:
            endpoint = "/portfolio/optimization/cluster-risk-parity"
        else:
            endpoint = "/portfolio/optimization/hierarchical-risk-parity"
        
        # Enhanced clustering configuration
        data = {
            "assets": request.assets,
            "riskFreeRate": request.risk_free_rate,
            "confidenceLevel": request.confidence_level
        }
        
        # Advanced clustering methods
        clustering_config = {
            "method": "hierarchical",  # hierarchical, kmeans, dbscan, spectral
            "distanceMetric": "correlation",  # correlation, euclidean, mahalanobis
            "linkageCriterion": "ward",  # ward, complete, average, single
            "clusterOptimization": "silhouette",  # silhouette, gap, elbow
            "minClusters": 2,
            "maxClusters": min(10, len(request.assets) // 2),
            "stabilityAnalysis": True,
            "regimeAware": True
        }
        
        data["clusteringConfig"] = clustering_config
        
        # Risk parity allocation methods
        allocation_config = {
            "allocationMethod": "equal_risk_contribution",  # equal_risk_contribution, inverse_volatility, min_variance
            "riskMeasure": request.risk_measure.value,
            "hierarchicalAllocation": True,  # Allocate at both cluster and asset levels
            "rebalancingFrequency": "monthly"  # daily, weekly, monthly
        }
        
        data["allocationConfig"] = allocation_config
        
        # Required market data
        if request.returns is not None:
            data["assetsReturns"] = request.returns.tolist()
        else:
            raise ValueError("Historical returns required for hierarchical risk parity optimization")
        
        if request.covariance_matrix is not None:
            data["covarianceMatrix"] = request.covariance_matrix.tolist()
        
        if request.expected_returns is not None:
            data["expectedReturns"] = request.expected_returns.tolist()
        
        # Enhanced constraints for hierarchical optimization
        constraints = {
            "minimumWeight": request.constraints.min_weight,
            "maximumWeight": request.constraints.max_weight,
            "leverageLimit": request.constraints.leverage_limit,
            "clusterMinWeight": 0.05,  # Minimum weight per cluster
            "clusterMaxWeight": 0.5    # Maximum weight per cluster
        }
        
        if request.constraints.max_assets:
            constraints["maxAssets"] = request.constraints.max_assets
            
        if request.constraints.group_constraints:
            constraints["groupConstraints"] = request.constraints.group_constraints
        
        data["constraints"] = constraints
        
        # Cache key for hierarchical optimization
        cache_key = self._get_cache_key(
            f"hierarchical_{request.method.value}",
            assets=request.assets,
            clustering_config=clustering_config,
            allocation_config=allocation_config
        )
        
        cached_result = self._check_cache(cache_key)
        if cached_result:
            logger.debug(f"Cache hit for hierarchical optimization {request.method.value}")
            return cached_result
        
        # Make API request
        api_start = time.time()
        result = await self._make_request("POST", endpoint, data)
        api_time = (time.time() - api_start) * 1000
        
        # Parse enhanced response
        weights = {asset: weight for asset, weight in 
                  zip(request.assets, result["weights"])}
        
        # Extract comprehensive clustering metadata
        clusters_info = result.get("clusters", {})
        cluster_analysis = result.get("clusterAnalysis", {})
        
        metadata = {
            "method": request.method.value,
            "clustering_method": clustering_config["method"],
            "distance_metric": clustering_config["distanceMetric"],
            "optimal_cluster_count": result.get("optimalClusterCount"),
            "silhouette_score": cluster_analysis.get("silhouetteScore"),
            "cluster_stability_score": cluster_analysis.get("stabilityScore"),
            "clusters": clusters_info,
            "cluster_risk_contributions": result.get("clusterRiskContributions", {}),
            "intra_cluster_correlations": result.get("intraClusterCorrelations", {}),
            "inter_cluster_correlations": result.get("interClusterCorrelations", {}),
            "dendrogram_data": result.get("dendrogramData"),  # For visualization
            "regime_detected": result.get("marketRegime"),
            "allocation_method": allocation_config["allocationMethod"],
            "hierarchical_allocation": allocation_config["hierarchicalAllocation"]
        }
        
        hierarchical_result = PortfolioOptimizationResult(
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
            metadata=metadata,
            api_response_time_ms=api_time
        )
        
        # Cache the result
        self._cache_result(cache_key, hierarchical_result)
        
        return hierarchical_result
    
    async def compute_efficient_frontier(self, assets: List[str],
                                        returns: Optional[pd.DataFrame] = None,
                                        covariance: Optional[np.ndarray] = None,
                                        num_portfolios: int = 50,
                                        method: str = "mean_variance",
                                        constraints: Optional[OptimizationConstraints] = None) -> List[PortfolioOptimizationResult]:
        """
        Compute enhanced efficient frontier with multiple methods
        
        Supports:
        - Mean-variance frontier (classical)
        - Risk parity frontier  
        - Maximum diversification frontier
        - Supervised k-NN frontier (novel)
        - CVaR frontier for tail risk optimization
        
        Returns 50+ optimal portfolios along the frontier
        """
        # Select endpoint based on frontier method
        method_endpoints = {
            "mean_variance": "/portfolio/analysis/mean-variance/minimum-variance-frontier",
            "risk_parity": "/portfolio/analysis/risk-parity/efficient-frontier",
            "max_diversification": "/portfolio/analysis/maximum-diversification/frontier",
            "supervised_knn": "/portfolios/analysis/supervised/efficient-frontier",
            "cvar": "/portfolio/analysis/conditional-value-at-risk/frontier"
        }
        
        endpoint = method_endpoints.get(method, method_endpoints["mean_variance"])
        
        # Enhanced data preparation
        data = {
            "assets": assets,
            "frontierPortfolios": num_portfolios,
            "frontierMethod": method
        }
        
        if returns is not None:
            data["assetsReturns"] = returns.values.tolist()
        else:
            raise ValueError("Historical returns data is required for efficient frontier computation")
        
        if covariance is not None:
            data["covarianceMatrix"] = covariance.tolist()
        
        # Add constraints if provided
        if constraints:
            data["constraints"] = {
                "minimumWeight": constraints.min_weight,
                "maximumWeight": constraints.max_weight,
                "leverageLimit": constraints.leverage_limit
            }
            
            if constraints.max_assets:
                data["constraints"]["maxAssets"] = constraints.max_assets
        
        # Special handling for supervised k-NN frontier
        if method == "supervised_knn":
            data["distanceMetric"] = "hassanat"  # Scale-invariant
            data["lookbackPeriods"] = min(252, len(returns) if returns is not None else 252)
            data["kNeighborsSelection"] = "dynamic"
            data["featureEngineering"] = {
                "includeMarketRegime": True,
                "includeVolatilityRegime": True
            }
        
        # For CVaR frontier
        if method == "cvar":
            data["confidenceLevel"] = 0.95
            data["riskMeasure"] = "conditional_value_at_risk"
        
        # Cache key for frontier computation
        cache_key = self._get_cache_key(
            f"frontier_{method}",
            assets=assets,
            num_portfolios=num_portfolios,
            constraints=constraints
        )
        
        cached_result = self._check_cache(cache_key)
        if cached_result:
            logger.debug(f"Cache hit for efficient frontier {method}")
            return cached_result
        
        # Make API request
        api_start = time.time()
        result = await self._make_request("POST", endpoint, data)
        api_time = (time.time() - api_start) * 1000
        
        # Parse frontier portfolios with enhanced metadata
        frontier = []
        for i, portfolio in enumerate(result["efficientFrontier"]):
            weights = {asset: weight for asset, weight in 
                      zip(assets, portfolio["weights"])}
            
            # Enhanced portfolio metadata
            metadata = {
                "frontier_point": True,
                "frontier_method": method,
                "frontier_index": i,
                "total_frontier_points": len(result["efficientFrontier"]),
                "optimization_time_ms": api_time / len(result["efficientFrontier"])
            }
            
            # Add method-specific metadata
            if method == "supervised_knn":
                metadata.update({
                    "k_neighbors_used": portfolio.get("kNeighborsUsed"),
                    "regime_detected": portfolio.get("marketRegime"),
                    "distance_metric": "hassanat"
                })
            
            frontier_point = PortfolioOptimizationResult(
                optimal_weights=weights,
                expected_return=portfolio["expectedReturn"],
                expected_risk=portfolio["expectedRisk"],
                sharpe_ratio=portfolio.get("sharpeRatio", 0.0),
                diversification_ratio=portfolio.get("diversificationRatio"),
                effective_assets=len([w for w in weights.values() if w > 0.001]),
                concentration_risk=max(weights.values()) if weights else 0.0,
                value_at_risk=portfolio.get("valueAtRisk"),
                conditional_value_at_risk=portfolio.get("conditionalValueAtRisk"),
                max_drawdown=portfolio.get("maxDrawdown"),
                metadata=metadata,
                api_response_time_ms=api_time / len(result["efficientFrontier"])
            )
            
            frontier.append(frontier_point)
        
        # Cache the frontier
        self._cache_result(cache_key, frontier)
        
        logger.info(f"Generated {len(frontier)} portfolio points on {method} efficient frontier")
        return frontier
    
    async def analyze_portfolio_metrics(self, weights: Dict[str, float],
                                       returns: pd.DataFrame) -> Dict[str, float]:
        """
        Analyze portfolio with given weights
        
        Returns comprehensive risk and return metrics
        """
        endpoint = "/portfolio/analysis/metrics"
        
        data = {
            "assets": list(weights.keys()),
            "weights": list(weights.values()),
            "assetsReturns": returns.values.tolist()
        }
        
        result = await self._make_request("POST", endpoint, data)
        
        return {
            "expected_return": result["expectedReturn"],
            "volatility": result["volatility"],
            "sharpe_ratio": result["sharpeRatio"],
            "sortino_ratio": result.get("sortinoRatio"),
            "max_drawdown": result.get("maxDrawdown"),
            "value_at_risk_95": result.get("valueAtRisk95"),
            "conditional_value_at_risk_95": result.get("conditionalValueAtRisk95"),
            "skewness": result.get("skewness"),
            "kurtosis": result.get("kurtosis"),
            "downside_deviation": result.get("downsideDeviation")
        }
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform comprehensive health check"""
        try:
            # Simple health check endpoint
            start_time = time.time()
            
            # Try to make a lightweight request
            test_data = {
                "assets": ["AAPL", "MSFT"],
                "weights": [0.5, 0.5]
            }
            
            async with self.circuit_breaker.execute():
                response = await self.client.get("/health", timeout=5.0)
                
            response_time = (time.time() - start_time) * 1000
            
            self.health_status = "healthy"
            self.last_health_check = datetime.now()
            
            return {
                "status": "healthy",
                "response_time_ms": response_time,
                "api_available": True,
                "circuit_breaker_state": self.circuit_breaker.state.value,
                "last_check": self.last_health_check.isoformat()
            }
            
        except Exception as e:
            self.health_status = "unhealthy"
            return {
                "status": "unhealthy",
                "error": str(e),
                "api_available": False,
                "circuit_breaker_state": self.circuit_breaker.state.value,
                "last_check": datetime.now().isoformat()
            }
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get comprehensive client performance statistics"""
        total_ops = self.total_requests + self.cache_hits
        
        return {
            "total_requests": self.total_requests,
            "success_count": self.success_count,
            "error_count": self.error_count,
            "cache_hits": self.cache_hits,
            "cache_hit_rate": self.cache_hits / max(1, total_ops),
            "success_rate": self.success_count / max(1, self.total_requests),
            "avg_api_response_ms": self.total_api_time_ms / max(1, self.total_requests),
            "cached_items": len(self.cache),
            "max_cache_size": self.max_cache_size,
            "circuit_breaker_trips": self.circuit_breaker_trips,
            "circuit_breaker_state": self.circuit_breaker.state.value,
            "health_status": self.health_status,
            "consecutive_failures": self.consecutive_failures,
            "performance_targets": {
                "meets_3s_target": (self.total_api_time_ms / max(1, self.total_requests)) < 3000,
                "meets_70pct_cache_target": (self.cache_hits / max(1, total_ops)) >= 0.7,
                "healthy_error_rate": (self.error_count / max(1, self.total_requests)) < 0.05
            }
        }
    
    async def clear_cache(self):
        """Clear all cached results"""
        self.cache.clear()
        self.cache_access_times.clear()
        logger.info("Portfolio Optimizer cache cleared")
    
    async def close(self):
        """Close the HTTP client and cleanup resources"""
        await self.client.aclose()
        self.cache.clear()
        self.cache_access_times.clear()
        logger.info("Portfolio Optimizer client closed and resources cleaned up")