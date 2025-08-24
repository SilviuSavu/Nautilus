"""
Pydantic Models for Volatility Forecasting API

This module defines all request and response models for the volatility API.
"""

from datetime import datetime
from typing import Dict, List, Optional, Any, Union
from pydantic import BaseModel, Field, validator


class ForecastRequest(BaseModel):
    """Request model for volatility forecast"""
    recent_data: Optional[List[Dict[str, Any]]] = Field(
        None, 
        description="Recent market data for forecasting (OHLCV format)"
    )
    horizon: int = Field(
        default=5, 
        ge=1, 
        le=30, 
        description="Forecast horizon in days"
    )
    confidence_level: float = Field(
        default=0.95, 
        ge=0.5, 
        le=0.99, 
        description="Confidence level for prediction intervals"
    )
    include_individual_forecasts: bool = Field(
        default=False,
        description="Include individual model forecasts in response"
    )
    
    @validator('recent_data')
    def validate_recent_data(cls, v):
        if v is not None and len(v) == 0:
            raise ValueError("Recent data cannot be empty if provided")
        return v


class TrainingRequest(BaseModel):
    """Request model for model training"""
    training_data: Union[List[Dict[str, Any]], str] = Field(
        ..., 
        description="Training data in JSON or CSV format"
    )
    data_format: str = Field(
        default="json",
        description="Data format: 'json' or 'csv'"
    )
    validation_split: float = Field(
        default=0.2,
        ge=0.1,
        le=0.4,
        description="Fraction of data to use for validation"
    )
    include_ohlc: bool = Field(
        default=True,
        description="Whether data includes OHLC columns"
    )
    async_training: bool = Field(
        default=False,
        description="Whether to train models asynchronously"
    )
    
    @validator('data_format')
    def validate_data_format(cls, v):
        if v not in ['json', 'csv']:
            raise ValueError("Data format must be 'json' or 'csv'")
        return v


class RealtimeDataRequest(BaseModel):
    """Request model for real-time data updates"""
    market_data: Dict[str, Any] = Field(
        ...,
        description="Real-time market data (OHLCV)"
    )
    timestamp: Optional[datetime] = Field(
        default=None,
        description="Data timestamp (uses current time if not provided)"
    )
    
    @validator('market_data')
    def validate_market_data(cls, v):
        required_fields = ['close']
        missing_fields = [field for field in required_fields if field not in v]
        if missing_fields:
            raise ValueError(f"Missing required fields in market_data: {missing_fields}")
        return v


class ForecastResponse(BaseModel):
    """Response model for volatility forecast"""
    status: str
    symbol: str
    forecast: Dict[str, Any]
    generation_time_ms: Optional[float] = None
    source: Optional[str] = None
    individual_forecasts: Optional[Dict[str, Any]] = None


class TrainingResponse(BaseModel):
    """Response model for training results"""
    status: str
    symbol: str
    training_results: Optional[Dict[str, Any]] = None
    message: str
    async_training: bool = False
    training_time_ms: Optional[float] = None


class ModelStatusResponse(BaseModel):
    """Response model for model status"""
    symbol: str
    total_models: int
    models: Dict[str, Any]
    last_forecast: Optional[datetime] = None
    forecast_count: int
    hardware_acceleration: bool


class EngineStatusResponse(BaseModel):
    """Response model for engine status"""
    status: str
    uptime_seconds: float
    active_symbols: List[str]
    models_per_symbol: Dict[str, int]
    performance_stats: Dict[str, Any]
    hardware_acceleration: Dict[str, Any]
    external_services: Dict[str, bool]
    configuration: Dict[str, Any]


class WebSocketMessage(BaseModel):
    """WebSocket message model"""
    type: str = Field(..., description="Message type")
    symbol: Optional[str] = Field(None, description="Trading symbol")
    data: Dict[str, Any] = Field(..., description="Message data")
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class SubscriptionRequest(BaseModel):
    """WebSocket subscription request"""
    action: str = Field(..., description="'subscribe' or 'unsubscribe'")
    symbol: str = Field(..., description="Trading symbol to subscribe to")
    stream_type: str = Field(
        default="forecasts",
        description="Stream type: 'forecasts', 'realtime', or 'all'"
    )
    
    @validator('action')
    def validate_action(cls, v):
        if v not in ['subscribe', 'unsubscribe']:
            raise ValueError("Action must be 'subscribe' or 'unsubscribe'")
        return v
    
    @validator('stream_type')
    def validate_stream_type(cls, v):
        if v not in ['forecasts', 'realtime', 'all']:
            raise ValueError("Stream type must be 'forecasts', 'realtime', or 'all'")
        return v


class ErrorResponse(BaseModel):
    """Error response model"""
    error: str
    detail: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class HealthResponse(BaseModel):
    """Health check response model"""
    status: str = "healthy"
    service: str = "volatility-forecasting-engine"
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    version: str = "1.0.0"


class BenchmarkRequest(BaseModel):
    """Benchmark request model"""
    symbols: List[str] = Field(
        default=["AAPL"],
        description="Symbols to benchmark"
    )
    iterations: int = Field(
        default=10,
        ge=1,
        le=100,
        description="Number of forecast iterations per symbol"
    )
    include_individual_timings: bool = Field(
        default=False,
        description="Include individual iteration timings"
    )


class BenchmarkResponse(BaseModel):
    """Benchmark response model"""
    symbols_tested: List[str]
    iterations: int
    results: List[Dict[str, Any]]
    summary: Dict[str, Any]
    benchmark_timestamp: datetime = Field(default_factory=datetime.utcnow)


class ModelConfiguration(BaseModel):
    """Model configuration model"""
    model_type: str
    parameters: Dict[str, Any] = Field(default_factory=dict)
    enabled: bool = True
    use_gpu: bool = False
    lookback_days: int = 252
    forecast_horizon: int = 5


class EnsembleConfiguration(BaseModel):
    """Ensemble configuration model"""
    method: str = "dynamic_weight"
    min_models: int = 3
    max_models: int = 8
    rebalance_frequency_minutes: int = 60
    use_confidence_weighting: bool = True


class HardwareConfiguration(BaseModel):
    """Hardware configuration model"""
    use_metal_gpu: bool = True
    use_neural_engine: bool = True
    use_cpu_optimization: bool = True
    auto_hardware_routing: bool = True
    max_gpu_memory_gb: float = 16.0


class ConfigurationUpdateRequest(BaseModel):
    """Configuration update request"""
    ensemble: Optional[EnsembleConfiguration] = None
    hardware: Optional[HardwareConfiguration] = None
    models: Optional[Dict[str, ModelConfiguration]] = None


class VolatilitySurface(BaseModel):
    """Volatility surface model"""
    symbol: str
    strikes: List[float]
    expiries: List[float]
    volatilities: List[List[float]]
    surface_timestamp: datetime = Field(default_factory=datetime.utcnow)
    model_used: str
    confidence_surfaces: Optional[Dict[str, List[List[float]]]] = None


class VolatilitySurfaceRequest(BaseModel):
    """Volatility surface generation request"""
    symbol: str
    min_strike: float = Field(ge=0.01, description="Minimum strike price")
    max_strike: float = Field(ge=0.01, description="Maximum strike price") 
    strike_steps: int = Field(default=20, ge=5, le=100)
    min_expiry_days: int = Field(default=7, ge=1)
    max_expiry_days: int = Field(default=365, le=1000)
    expiry_steps: int = Field(default=10, ge=3, le=50)
    model_type: str = Field(default="heston", description="Model for surface generation")
    
    @validator('max_strike')
    def validate_strikes(cls, v, values):
        if 'min_strike' in values and v <= values['min_strike']:
            raise ValueError("max_strike must be greater than min_strike")
        return v