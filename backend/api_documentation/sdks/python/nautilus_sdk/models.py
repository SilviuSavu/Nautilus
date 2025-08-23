"""
Pydantic models for Nautilus Trading Platform API
Comprehensive data models with validation and serialization
"""

from pydantic import BaseModel, Field, validator
from typing import Dict, List, Optional, Any, Union
from datetime import datetime
from enum import Enum


class DataSource(str, Enum):
    """Supported data sources"""
    IBKR = "IBKR"
    ALPHA_VANTAGE = "ALPHA_VANTAGE"
    FRED = "FRED"
    EDGAR = "EDGAR"
    DATAGOV = "DATAGOV"
    TRADING_ECONOMICS = "TRADING_ECONOMICS"
    DBNOMICS = "DBNOMICS"
    YAHOO = "YAHOO"


class HealthStatus(str, Enum):
    """System health status levels"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"


class ComponentStatus(str, Enum):
    """Individual component status"""
    UP = "up"
    DOWN = "down"
    DEGRADED = "degraded"


class RiskLimitType(str, Enum):
    """Types of risk limits"""
    POSITION_LIMIT = "position_limit"
    LOSS_LIMIT = "loss_limit"
    EXPOSURE_LIMIT = "exposure_limit"
    CONCENTRATION_LIMIT = "concentration_limit"
    VAR_LIMIT = "var_limit"
    DRAWDOWN_LIMIT = "drawdown_limit"
    LEVERAGE_LIMIT = "leverage_limit"
    CORRELATION_LIMIT = "correlation_limit"
    SECTOR_LIMIT = "sector_limit"
    CURRENCY_LIMIT = "currency_limit"
    VOLATILITY_LIMIT = "volatility_limit"
    LIQUIDITY_LIMIT = "liquidity_limit"


class RiskLimitStatus(str, Enum):
    """Risk limit status"""
    ACTIVE = "active"
    BREACHED = "breached"
    WARNING = "warning"
    DISABLED = "disabled"


class StrategyStatus(str, Enum):
    """Strategy deployment status"""
    DEVELOPMENT = "development"
    TESTING = "testing"
    DEPLOYED = "deployed"
    PAUSED = "paused"
    RETIRED = "retired"


class WebSocketMessageType(str, Enum):
    """WebSocket message types"""
    MARKET_DATA = "market_data"
    TRADE_UPDATE = "trade_update"
    ORDER_UPDATE = "order_update"
    PORTFOLIO_UPDATE = "portfolio_update"
    RISK_ALERT = "risk_alert"
    SYSTEM_STATUS = "system_status"
    STRATEGY_UPDATE = "strategy_update"
    ANALYTICS_UPDATE = "analytics_update"
    CONNECTION_STATUS = "connection_status"
    SUBSCRIPTION_UPDATE = "subscription_update"
    HEARTBEAT = "heartbeat"
    ERROR = "error"
    USER_NOTIFICATION = "user_notification"
    MARKET_EVENT = "market_event"
    ECONOMIC_DATA = "economic_data"


# Base Models
class BaseNautilusModel(BaseModel):
    """Base model with common configuration"""
    
    class Config:
        use_enum_values = True
        allow_population_by_field_name = True
        validate_assignment = True
        extra = "forbid"


class ErrorResponse(BaseNautilusModel):
    """Standard error response"""
    error: str = Field(..., description="Error code")
    message: str = Field(..., description="Human-readable error message")
    details: Optional[Dict[str, Any]] = Field(None, description="Additional error details")
    timestamp: datetime = Field(..., description="Error timestamp")
    request_id: Optional[str] = Field(None, description="Request identifier for debugging")


class PaginationRequest(BaseNautilusModel):
    """Pagination parameters for requests"""
    page: int = Field(1, ge=1, description="Page number (1-based)")
    limit: int = Field(50, ge=1, le=1000, description="Number of items per page")
    sort: Optional[str] = Field(None, description="Sort field and direction (e.g., 'timestamp:desc')")


class PaginationResponse(BaseNautilusModel):
    """Pagination information in responses"""
    page: int = Field(..., description="Current page number")
    limit: int = Field(..., description="Items per page")
    total: int = Field(..., description="Total number of items")
    pages: int = Field(..., description="Total number of pages")
    has_next: bool = Field(..., description="Whether there are more pages")
    has_prev: bool = Field(..., description="Whether there are previous pages")


# System Models
class ComponentHealthInfo(BaseNautilusModel):
    """Individual component health information"""
    status: ComponentStatus = Field(..., description="Component status")
    response_time: Optional[float] = Field(None, description="Response time in milliseconds")
    last_check: Optional[datetime] = Field(None, description="Last health check timestamp")
    error: Optional[str] = Field(None, description="Error message if component is down")


class HealthCheck(BaseNautilusModel):
    """System health check response"""
    status: HealthStatus = Field(..., description="Overall system health status")
    timestamp: datetime = Field(..., description="Health check timestamp")
    version: Optional[str] = Field(None, description="API version")
    components: Optional[Dict[str, ComponentHealthInfo]] = Field(
        None, description="Individual component health status"
    )


# Market Data Models
class MarketData(BaseNautilusModel):
    """Market data for a financial instrument"""
    symbol: str = Field(..., description="Financial instrument symbol")
    timestamp: datetime = Field(..., description="Data timestamp")
    source: DataSource = Field(..., description="Data source provider")
    price: Optional[float] = Field(None, description="Current or closing price")
    volume: Optional[int] = Field(None, description="Trading volume")
    bid: Optional[float] = Field(None, description="Bid price")
    ask: Optional[float] = Field(None, description="Ask price")
    open: Optional[float] = Field(None, description="Opening price")
    high: Optional[float] = Field(None, description="Highest price")
    low: Optional[float] = Field(None, description="Lowest price")
    change: Optional[float] = Field(None, description="Price change")
    change_percent: Optional[float] = Field(None, description="Percentage change")
    
    @validator('symbol')
    def validate_symbol(cls, v):
        if not v or len(v.strip()) == 0:
            raise ValueError("Symbol cannot be empty")
        return v.upper().strip()


class HistoricalDataResponse(BaseNautilusModel):
    """Historical market data response"""
    symbol: str = Field(..., description="Financial instrument symbol")
    interval: str = Field(..., description="Data interval")
    data: List[MarketData] = Field(..., description="Historical data points")
    pagination: Optional[PaginationResponse] = Field(None, description="Pagination information")


# Risk Management Models
class RiskLimit(BaseNautilusModel):
    """Risk limit configuration"""
    id: Optional[str] = Field(None, description="Unique risk limit identifier")
    type: RiskLimitType = Field(..., description="Type of risk limit")
    value: float = Field(..., description="Limit value")
    current_value: Optional[float] = Field(None, description="Current exposure value")
    utilization: Optional[float] = Field(
        None, ge=0, le=1, description="Limit utilization percentage (0-1)"
    )
    status: RiskLimitStatus = Field(..., description="Current limit status")
    symbol: Optional[str] = Field(None, description="Symbol for the limit")
    warning_threshold: float = Field(0.8, ge=0, le=1, description="Warning threshold (0-1)")
    auto_adjust: bool = Field(False, description="Whether limit auto-adjusts")
    created_at: Optional[datetime] = Field(None, description="Creation timestamp")
    updated_at: Optional[datetime] = Field(None, description="Last update timestamp")


class RiskBreach(BaseNautilusModel):
    """Risk limit breach information"""
    id: str = Field(..., description="Unique breach identifier")
    limit_id: str = Field(..., description="Associated risk limit ID")
    timestamp: datetime = Field(..., description="Breach timestamp")
    severity: str = Field(..., description="Breach severity level")
    current_value: float = Field(..., description="Value at time of breach")
    limit_value: float = Field(..., description="Limit that was breached")
    actions_taken: Optional[List[str]] = Field(None, description="Automated actions taken")
    resolved: bool = Field(False, description="Whether breach has been resolved")


# Strategy Models
class PerformanceMetrics(BaseNautilusModel):
    """Strategy performance metrics"""
    total_return: Optional[float] = Field(None, description="Total return")
    sharpe_ratio: Optional[float] = Field(None, description="Sharpe ratio")
    max_drawdown: Optional[float] = Field(None, description="Maximum drawdown")
    win_rate: Optional[float] = Field(None, description="Win rate")
    profit_factor: Optional[float] = Field(None, description="Profit factor")
    alpha: Optional[float] = Field(None, description="Alpha")
    beta: Optional[float] = Field(None, description="Beta")
    sortino_ratio: Optional[float] = Field(None, description="Sortino ratio")


class DeploymentInfo(BaseNautilusModel):
    """Strategy deployment information"""
    deployed_at: Optional[datetime] = Field(None, description="Deployment timestamp")
    deployment_id: Optional[str] = Field(None, description="Deployment identifier")
    environment: Optional[str] = Field(None, description="Deployment environment")
    auto_rollback: bool = Field(False, description="Whether auto-rollback is enabled")
    rollback_threshold: Optional[float] = Field(None, description="Rollback threshold")


class Strategy(BaseNautilusModel):
    """Trading strategy definition"""
    id: Optional[str] = Field(None, description="Unique strategy identifier")
    name: str = Field(..., description="Strategy name")
    version: str = Field(..., description="Strategy version")
    status: StrategyStatus = Field(..., description="Current strategy status")
    description: Optional[str] = Field(None, description="Strategy description")
    parameters: Optional[Dict[str, Any]] = Field(None, description="Strategy parameters")
    performance_metrics: Optional[PerformanceMetrics] = Field(None, description="Performance metrics")
    deployment_info: Optional[DeploymentInfo] = Field(None, description="Deployment information")
    risk_limits: Optional[Dict[str, Any]] = Field(None, description="Associated risk limits")
    created_at: Optional[datetime] = Field(None, description="Creation timestamp")
    updated_at: Optional[datetime] = Field(None, description="Last update timestamp")


# WebSocket Models
class WebSocketMessage(BaseNautilusModel):
    """WebSocket message structure"""
    type: WebSocketMessageType = Field(..., description="Message type identifier")
    timestamp: datetime = Field(..., description="Message timestamp")
    data: Optional[Dict[str, Any]] = Field(None, description="Message payload")
    sequence: Optional[int] = Field(None, description="Message sequence number")
    channel: Optional[str] = Field(None, description="WebSocket channel/topic")


class WebSocketSubscription(BaseNautilusModel):
    """WebSocket subscription configuration"""
    type: str = Field(..., description="Subscription type")
    symbols: Optional[List[str]] = Field(None, description="Symbols to subscribe to")
    data_types: Optional[List[str]] = Field(None, description="Types of data to receive")
    filters: Optional[Dict[str, Any]] = Field(None, description="Additional filters")


# Authentication Models
class LoginRequest(BaseNautilusModel):
    """User login request"""
    username: str = Field(..., description="User email or username")
    password: str = Field(..., description="User password")
    remember_me: bool = Field(False, description="Whether to use long-lived token")


class LoginResponse(BaseNautilusModel):
    """User login response"""
    access_token: str = Field(..., description="JWT access token")
    token_type: str = Field("Bearer", description="Token type")
    expires_in: int = Field(..., description="Token expiration in seconds")
    refresh_token: Optional[str] = Field(None, description="Refresh token")
    user_id: Optional[str] = Field(None, description="User identifier")
    permissions: Optional[List[str]] = Field(None, description="User permissions")


class TokenRefreshRequest(BaseNautilusModel):
    """Token refresh request"""
    refresh_token: str = Field(..., description="Refresh token")


class TokenRefreshResponse(BaseNautilusModel):
    """Token refresh response"""
    access_token: str = Field(..., description="New JWT access token")
    token_type: str = Field("Bearer", description="Token type")
    expires_in: int = Field(..., description="Token expiration in seconds")


# Analytics Models
class PerformanceAnalytics(BaseNautilusModel):
    """Performance analytics data"""
    portfolio_id: Optional[str] = Field(None, description="Portfolio identifier")
    period_start: datetime = Field(..., description="Analysis period start")
    period_end: datetime = Field(..., description="Analysis period end")
    total_return: float = Field(..., description="Total return for period")
    annualized_return: float = Field(..., description="Annualized return")
    volatility: float = Field(..., description="Volatility")
    sharpe_ratio: float = Field(..., description="Sharpe ratio")
    max_drawdown: float = Field(..., description="Maximum drawdown")
    var_95: Optional[float] = Field(None, description="95% Value at Risk")
    trades_count: int = Field(..., description="Number of trades")
    win_rate: float = Field(..., description="Win rate")
    profit_factor: float = Field(..., description="Profit factor")


class RiskAnalytics(BaseNautilusModel):
    """Risk analytics data"""
    portfolio_id: str = Field(..., description="Portfolio identifier")
    timestamp: datetime = Field(..., description="Analysis timestamp")
    var_95: float = Field(..., description="95% Value at Risk")
    var_99: float = Field(..., description="99% Value at Risk")
    expected_shortfall: float = Field(..., description="Expected shortfall")
    beta: float = Field(..., description="Portfolio beta")
    correlation_matrix: Optional[Dict[str, Dict[str, float]]] = Field(
        None, description="Asset correlation matrix"
    )
    sector_exposure: Optional[Dict[str, float]] = Field(None, description="Sector exposure")
    currency_exposure: Optional[Dict[str, float]] = Field(None, description="Currency exposure")


# Request/Response Models for API Operations
class CreateRiskLimitRequest(BaseNautilusModel):
    """Request to create a new risk limit"""
    type: RiskLimitType = Field(..., description="Type of risk limit")
    value: float = Field(..., description="Limit value")
    symbol: Optional[str] = Field(None, description="Symbol for the limit")
    warning_threshold: float = Field(0.8, ge=0, le=1, description="Warning threshold")
    auto_adjust: bool = Field(False, description="Whether limit auto-adjusts")


class DeployStrategyRequest(BaseNautilusModel):
    """Request to deploy a trading strategy"""
    name: str = Field(..., description="Strategy name")
    version: str = Field(..., description="Strategy version")
    description: str = Field(..., description="Strategy description")
    parameters: Dict[str, Any] = Field(..., description="Strategy parameters")
    risk_limits: Optional[Dict[str, Any]] = Field(None, description="Risk limits")
    deployment_config: Optional[Dict[str, Any]] = Field(None, description="Deployment config")


class DeployStrategyResponse(BaseNautilusModel):
    """Response from strategy deployment"""
    deployment_id: str = Field(..., description="Unique deployment identifier")
    strategy_id: str = Field(..., description="Strategy identifier")
    status: str = Field(..., description="Initial deployment status")
    message: str = Field(..., description="Deployment message")
    pipeline_url: Optional[str] = Field(None, description="Pipeline monitoring URL")


# Export all models
__all__ = [
    # Enums
    "DataSource",
    "HealthStatus", 
    "ComponentStatus",
    "RiskLimitType",
    "RiskLimitStatus", 
    "StrategyStatus",
    "WebSocketMessageType",
    
    # Base Models
    "BaseNautilusModel",
    "ErrorResponse",
    "PaginationRequest", 
    "PaginationResponse",
    
    # System Models
    "ComponentHealthInfo",
    "HealthCheck",
    
    # Market Data Models
    "MarketData",
    "HistoricalDataResponse",
    
    # Risk Models
    "RiskLimit",
    "RiskBreach",
    
    # Strategy Models  
    "PerformanceMetrics",
    "DeploymentInfo",
    "Strategy",
    
    # WebSocket Models
    "WebSocketMessage",
    "WebSocketSubscription",
    
    # Auth Models
    "LoginRequest",
    "LoginResponse", 
    "TokenRefreshRequest",
    "TokenRefreshResponse",
    
    # Analytics Models
    "PerformanceAnalytics",
    "RiskAnalytics",
    
    # Request/Response Models
    "CreateRiskLimitRequest",
    "DeployStrategyRequest",
    "DeployStrategyResponse"
]