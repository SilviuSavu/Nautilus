"""
WebSocket Message Protocols - Sprint 3 Priority 1

Message format definitions, validation schemas, and protocol versioning for:
- Structured message formats
- Data validation and serialization
- Protocol versioning
- Message type definitions
"""

import json
import logging
from datetime import datetime
from typing import Dict, Any, Optional, List, Union, Literal
from pydantic import BaseModel, Field, validator
from enum import Enum

logger = logging.getLogger(__name__)


class MessageType(Enum):
    """WebSocket message types"""
    # Connection management
    CONNECTION_ESTABLISHED = "connection_established"
    CONNECTION_CLOSED = "connection_closed"
    HEARTBEAT = "heartbeat"
    HEARTBEAT_RESPONSE = "heartbeat_response"
    
    # Subscription management
    SUBSCRIBE = "subscribe"
    UNSUBSCRIBE = "unsubscribe"
    SUBSCRIPTION_CONFIRMED = "subscription_confirmed"
    SUBSCRIPTION_ERROR = "subscription_error"
    
    # Data streaming
    ENGINE_STATUS = "engine_status"
    MARKET_DATA = "market_data"
    TRADE_UPDATES = "trade_updates"
    SYSTEM_HEALTH = "system_health"
    
    # Sprint 3: Advanced Analytics & Risk
    PERFORMANCE_UPDATE = "performance_update"
    RISK_ALERT = "risk_alert"
    ORDER_UPDATE = "order_update"
    POSITION_UPDATE = "position_update"
    STRATEGY_PERFORMANCE = "strategy_performance"
    EXECUTION_ANALYTICS = "execution_analytics"
    RISK_METRICS = "risk_metrics"
    
    # Events & Notifications
    EVENT = "event"
    ALERT = "alert"
    NOTIFICATION = "notification"
    BREACH_ALERT = "breach_alert"
    SYSTEM_ALERT = "system_alert"
    
    # Responses
    ACK = "ack"
    ERROR = "error"
    SUCCESS = "success"
    
    # Commands
    COMMAND = "command"
    COMMAND_RESPONSE = "command_response"


class ProtocolVersion(Enum):
    """Protocol version for backward compatibility"""
    V1_0 = "1.0"
    V1_1 = "1.1"
    V2_0 = "2.0"


class MessagePriority(Enum):
    """Message priority levels"""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4


# Base message models

class BaseMessage(BaseModel):
    """Base WebSocket message structure"""
    type: str = Field(..., description="Message type identifier")
    version: str = Field(default="2.0", description="Protocol version")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Message timestamp")
    message_id: Optional[str] = Field(None, description="Unique message identifier")
    correlation_id: Optional[str] = Field(None, description="Correlation ID for request/response")
    priority: int = Field(default=2, ge=1, le=4, description="Message priority (1-4)")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }
        
    @validator('timestamp', pre=True)
    def parse_timestamp(cls, v):
        if isinstance(v, str):
            return datetime.fromisoformat(v.replace('Z', '+00:00'))
        return v


class ConnectionMessage(BaseMessage):
    """Connection management messages"""
    connection_id: str = Field(..., description="WebSocket connection identifier")
    user_id: Optional[str] = Field(None, description="Authenticated user identifier")
    session_info: Optional[Dict[str, Any]] = Field(None, description="Session metadata")


class SubscriptionMessage(BaseMessage):
    """Subscription management messages"""
    subscription_id: str = Field(..., description="Subscription identifier")
    topic: str = Field(..., description="Subscription topic")
    parameters: Optional[Dict[str, Any]] = Field(None, description="Subscription parameters")
    filters: Optional[Dict[str, Any]] = Field(None, description="Message filters")


class DataMessage(BaseMessage):
    """Data streaming messages"""
    data: Dict[str, Any] = Field(..., description="Message payload data")
    source: Optional[str] = Field(None, description="Data source identifier")
    sequence: Optional[int] = Field(None, description="Message sequence number")


class ErrorMessage(BaseMessage):
    """Error messages"""
    error_code: str = Field(..., description="Error code identifier")
    error_message: str = Field(..., description="Human-readable error message")
    details: Optional[Dict[str, Any]] = Field(None, description="Additional error details")


class CommandMessage(BaseMessage):
    """Command messages"""
    command: str = Field(..., description="Command identifier")
    parameters: Optional[Dict[str, Any]] = Field(None, description="Command parameters")
    target: Optional[str] = Field(None, description="Command target")


# Specific message types

class EngineStatusMessage(DataMessage):
    """Engine status update message"""
    type: Literal["engine_status"] = Field(default="engine_status")
    engine_id: Optional[str] = Field(None, description="Engine identifier")
    
    @validator('data')
    def validate_engine_data(cls, v):
        required_fields = ['state', 'uptime']
        for field in required_fields:
            if field not in v:
                raise ValueError(f"Missing required engine status field: {field}")
        return v


class MarketDataMessage(DataMessage):
    """Market data update message"""
    type: Literal["market_data"] = Field(default="market_data")
    symbol: str = Field(..., description="Trading symbol")
    venue: Optional[str] = Field(None, description="Market venue")
    data_type: Optional[str] = Field(None, description="Type of market data (tick, quote, bar)")
    
    @validator('data')
    def validate_market_data(cls, v):
        # Validate that at least one price field exists
        price_fields = ['price', 'bid_price', 'ask_price', 'open', 'high', 'low', 'close']
        if not any(field in v for field in price_fields):
            raise ValueError("Market data must contain at least one price field")
        return v


class TradeUpdateMessage(DataMessage):
    """Trade execution update message"""
    type: Literal["trade_updates"] = Field(default="trade_updates")
    user_id: str = Field(..., description="User identifier")
    trade_type: Optional[str] = Field(None, description="Type of trade update")
    
    @validator('data')
    def validate_trade_data(cls, v):
        # Ensure trade data contains required fields
        if 'timestamp' not in v:
            v['timestamp'] = datetime.utcnow().isoformat()
        return v


class SystemHealthMessage(DataMessage):
    """System health monitoring message"""
    type: Literal["system_health"] = Field(default="system_health")
    component: Optional[str] = Field(None, description="System component")
    status: Optional[str] = Field(None, description="Health status")
    
    @validator('data')
    def validate_health_data(cls, v):
        if 'status' not in v:
            raise ValueError("System health data must contain status field")
        return v


class EventMessage(DataMessage):
    """Event notification message"""
    type: Literal["event"] = Field(default="event")
    event_type: str = Field(..., description="Event type identifier")
    event_id: str = Field(..., description="Unique event identifier")
    source: str = Field(..., description="Event source")


class AlertMessage(DataMessage):
    """Alert notification message"""
    type: Literal["alert"] = Field(default="alert")
    alert_level: str = Field(..., description="Alert severity level")
    alert_category: Optional[str] = Field(None, description="Alert category")
    
    @validator('alert_level')
    def validate_alert_level(cls, v):
        valid_levels = ['info', 'warning', 'error', 'critical']
        if v.lower() not in valid_levels:
            raise ValueError(f"Invalid alert level: {v}")
        return v.lower()


class HeartbeatMessage(BaseMessage):
    """Heartbeat message"""
    type: Literal["heartbeat"] = Field(default="heartbeat")
    client_timestamp: Optional[datetime] = Field(None, description="Client timestamp")
    server_timestamp: datetime = Field(default_factory=datetime.utcnow, description="Server timestamp")


class AckMessage(BaseMessage):
    """Acknowledgment message"""
    type: Literal["ack"] = Field(default="ack")
    ack_message_id: str = Field(..., description="ID of message being acknowledged")
    status: str = Field(default="success", description="Acknowledgment status")


# Sprint 3: Advanced Analytics & Risk Message Types

class PerformanceUpdateMessage(DataMessage):
    """Performance analytics update message"""
    type: Literal["performance_update"] = Field(default="performance_update")
    portfolio_id: str = Field(..., description="Portfolio identifier")
    strategy_id: Optional[str] = Field(None, description="Strategy identifier")
    
    @validator('data')
    def validate_performance_data(cls, v):
        required_fields = ['pnl', 'returns', 'timestamp']
        for field in required_fields:
            if field not in v:
                raise ValueError(f"Missing required performance field: {field}")
        return v


class RiskAlertMessage(DataMessage):
    """Risk management alert message"""
    type: Literal["risk_alert"] = Field(default="risk_alert")
    portfolio_id: str = Field(..., description="Portfolio identifier")
    risk_type: str = Field(..., description="Type of risk alert")
    severity: str = Field(..., description="Alert severity level")
    
    @validator('severity')
    def validate_severity(cls, v):
        valid_severities = ['low', 'medium', 'high', 'critical']
        if v.lower() not in valid_severities:
            raise ValueError(f"Invalid severity level: {v}")
        return v.lower()
    
    @validator('data')
    def validate_risk_data(cls, v):
        required_fields = ['metric', 'current_value', 'threshold', 'timestamp']
        for field in required_fields:
            if field not in v:
                raise ValueError(f"Missing required risk alert field: {field}")
        return v


class OrderUpdateMessage(DataMessage):
    """Order status update message"""
    type: Literal["order_update"] = Field(default="order_update")
    order_id: str = Field(..., description="Order identifier")
    user_id: str = Field(..., description="User identifier")
    symbol: str = Field(..., description="Trading symbol")
    order_status: str = Field(..., description="Current order status")
    
    @validator('order_status')
    def validate_order_status(cls, v):
        valid_statuses = ['pending', 'open', 'filled', 'partial', 'cancelled', 'rejected']
        if v.lower() not in valid_statuses:
            raise ValueError(f"Invalid order status: {v}")
        return v.lower()


class PositionUpdateMessage(DataMessage):
    """Position update message"""
    type: Literal["position_update"] = Field(default="position_update")
    portfolio_id: str = Field(..., description="Portfolio identifier")
    symbol: str = Field(..., description="Trading symbol")
    position_side: str = Field(..., description="Position side (long/short)")
    
    @validator('position_side')
    def validate_position_side(cls, v):
        valid_sides = ['long', 'short', 'flat']
        if v.lower() not in valid_sides:
            raise ValueError(f"Invalid position side: {v}")
        return v.lower()
    
    @validator('data')
    def validate_position_data(cls, v):
        required_fields = ['quantity', 'avg_price', 'unrealized_pnl']
        for field in required_fields:
            if field not in v:
                raise ValueError(f"Missing required position field: {field}")
        return v


class StrategyPerformanceMessage(DataMessage):
    """Strategy performance analytics message"""
    type: Literal["strategy_performance"] = Field(default="strategy_performance")
    strategy_id: str = Field(..., description="Strategy identifier")
    strategy_name: str = Field(..., description="Strategy name")
    
    @validator('data')
    def validate_strategy_data(cls, v):
        required_fields = ['total_return', 'sharpe_ratio', 'max_drawdown', 'trade_count']
        for field in required_fields:
            if field not in v:
                raise ValueError(f"Missing required strategy performance field: {field}")
        return v


class ExecutionAnalyticsMessage(DataMessage):
    """Execution quality analytics message"""
    type: Literal["execution_analytics"] = Field(default="execution_analytics")
    execution_id: str = Field(..., description="Execution identifier")
    symbol: str = Field(..., description="Trading symbol")
    
    @validator('data')
    def validate_execution_data(cls, v):
        required_fields = ['fill_price', 'benchmark_price', 'slippage', 'timestamp']
        for field in required_fields:
            if field not in v:
                raise ValueError(f"Missing required execution analytics field: {field}")
        return v


class RiskMetricsMessage(DataMessage):
    """Risk metrics update message"""
    type: Literal["risk_metrics"] = Field(default="risk_metrics")
    portfolio_id: str = Field(..., description="Portfolio identifier")
    metric_type: str = Field(..., description="Type of risk metric")
    
    @validator('data')
    def validate_risk_metrics_data(cls, v):
        required_fields = ['value', 'timestamp']
        for field in required_fields:
            if field not in v:
                raise ValueError(f"Missing required risk metrics field: {field}")
        return v


class BreachAlertMessage(DataMessage):
    """Risk limit breach alert message"""
    type: Literal["breach_alert"] = Field(default="breach_alert")
    portfolio_id: str = Field(..., description="Portfolio identifier")
    breach_type: str = Field(..., description="Type of breach")
    severity: str = Field(default="high", description="Breach severity")
    
    @validator('severity')
    def validate_breach_severity(cls, v):
        valid_severities = ['medium', 'high', 'critical']
        if v.lower() not in valid_severities:
            raise ValueError(f"Invalid breach severity: {v}")
        return v.lower()
    
    @validator('data')
    def validate_breach_data(cls, v):
        required_fields = ['limit_type', 'current_value', 'limit_value', 'timestamp']
        for field in required_fields:
            if field not in v:
                raise ValueError(f"Missing required breach alert field: {field}")
        return v


class SystemAlertMessage(DataMessage):
    """System-level alert message"""
    type: Literal["system_alert"] = Field(default="system_alert")
    component: str = Field(..., description="System component")
    alert_type: str = Field(..., description="Type of system alert")
    severity: str = Field(..., description="Alert severity")
    
    @validator('severity')
    def validate_system_severity(cls, v):
        valid_severities = ['info', 'warning', 'error', 'critical']
        if v.lower() not in valid_severities:
            raise ValueError(f"Invalid system alert severity: {v}")
        return v.lower()


# Protocol validation and utilities

class MessageProtocol:
    """
    WebSocket message protocol handler
    
    Provides:
    - Message validation
    - Serialization/deserialization
    - Protocol versioning
    - Message routing
    """
    
    def __init__(self, version: ProtocolVersion = ProtocolVersion.V2_0):
        self.version = version
        self.message_classes = self._register_message_classes()
        
    def _register_message_classes(self) -> Dict[str, BaseMessage]:
        """Register message classes by type"""
        return {
            # Connection management
            "connection_established": ConnectionMessage,
            "connection_closed": ConnectionMessage,
            "heartbeat": HeartbeatMessage,
            "heartbeat_response": HeartbeatMessage,
            
            # Subscription management
            "subscribe": SubscriptionMessage,
            "unsubscribe": SubscriptionMessage,
            "subscription_confirmed": SubscriptionMessage,
            "subscription_error": ErrorMessage,
            
            # Core data streaming
            "engine_status": EngineStatusMessage,
            "market_data": MarketDataMessage,
            "trade_updates": TradeUpdateMessage,
            "system_health": SystemHealthMessage,
            
            # Sprint 3: Advanced Analytics & Risk
            "performance_update": PerformanceUpdateMessage,
            "risk_alert": RiskAlertMessage,
            "order_update": OrderUpdateMessage,
            "position_update": PositionUpdateMessage,
            "strategy_performance": StrategyPerformanceMessage,
            "execution_analytics": ExecutionAnalyticsMessage,
            "risk_metrics": RiskMetricsMessage,
            
            # Alerts & Events
            "event": EventMessage,
            "alert": AlertMessage,
            "breach_alert": BreachAlertMessage,
            "system_alert": SystemAlertMessage,
            "notification": DataMessage,
            
            # System responses
            "error": ErrorMessage,
            "ack": AckMessage,
            "success": DataMessage,
            
            # Commands
            "command": CommandMessage,
            "command_response": DataMessage
        }
        
    def validate_message(self, message_data: Dict[str, Any]) -> bool:
        """
        Validate a message against the protocol
        
        Args:
            message_data: Raw message data
            
        Returns:
            bool: True if message is valid
        """
        try:
            message_type = message_data.get("type")
            if not message_type:
                logger.warning("Message missing type field")
                return False
                
            # Check if message type is supported
            if message_type not in self.message_classes:
                logger.warning(f"Unsupported message type: {message_type}")
                return False
                
            # Validate against message class
            message_class = self.message_classes[message_type]
            message_class(**message_data)
            
            return True
            
        except Exception as e:
            logger.error(f"Message validation failed: {e}")
            return False
            
    def serialize_message(self, message: BaseMessage) -> str:
        """
        Serialize message to JSON string
        
        Args:
            message: Message instance
            
        Returns:
            str: JSON serialized message
        """
        try:
            return message.json()
        except Exception as e:
            logger.error(f"Message serialization failed: {e}")
            raise
            
    def deserialize_message(self, message_data: str) -> Optional[BaseMessage]:
        """
        Deserialize JSON string to message instance
        
        Args:
            message_data: JSON message string
            
        Returns:
            BaseMessage: Deserialized message or None if failed
        """
        try:
            # Parse JSON
            data = json.loads(message_data)
            
            # Get message type
            message_type = data.get("type")
            if not message_type or message_type not in self.message_classes:
                logger.warning(f"Unknown message type: {message_type}")
                return None
                
            # Create message instance
            message_class = self.message_classes[message_type]
            return message_class(**data)
            
        except Exception as e:
            logger.error(f"Message deserialization failed: {e}")
            return None
            
    def create_error_message(
        self,
        error_code: str,
        error_message: str,
        correlation_id: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None
    ) -> ErrorMessage:
        """Create a standardized error message"""
        return ErrorMessage(
            type="error",
            error_code=error_code,
            error_message=error_message,
            correlation_id=correlation_id,
            details=details or {},
            version=self.version.value
        )
        
    def create_ack_message(
        self,
        ack_message_id: str,
        status: str = "success",
        correlation_id: Optional[str] = None
    ) -> AckMessage:
        """Create a standardized acknowledgment message"""
        return AckMessage(
            type="ack",
            ack_message_id=ack_message_id,
            status=status,
            correlation_id=correlation_id,
            version=self.version.value
        )
        
    def create_heartbeat_response(
        self,
        client_timestamp: Optional[datetime] = None,
        correlation_id: Optional[str] = None
    ) -> HeartbeatMessage:
        """Create a heartbeat response message"""
        return HeartbeatMessage(
            type="heartbeat_response",
            client_timestamp=client_timestamp,
            server_timestamp=datetime.utcnow(),
            correlation_id=correlation_id,
            version=self.version.value
        )
        
    def get_supported_message_types(self) -> List[str]:
        """Get list of supported message types"""
        return list(self.message_classes.keys())
        
    def get_protocol_info(self) -> Dict[str, Any]:
        """Get protocol information"""
        return {
            "version": self.version.value,
            "supported_message_types": self.get_supported_message_types(),
            "message_count": len(self.message_classes),
            "protocol_features": [
                "message_validation",
                "serialization",
                "error_handling",
                "heartbeat_support",
                "subscription_management",
                "event_routing"
            ]
        }


# Protocol-specific utilities

def create_subscription_request(
    subscription_id: str,
    topic: str,
    parameters: Optional[Dict[str, Any]] = None,
    filters: Optional[Dict[str, Any]] = None
) -> SubscriptionMessage:
    """Create a subscription request message"""
    return SubscriptionMessage(
        type="subscribe",
        subscription_id=subscription_id,
        topic=topic,
        parameters=parameters or {},
        filters=filters or {}
    )


def create_market_data_message(
    symbol: str,
    data: Dict[str, Any],
    venue: Optional[str] = None,
    data_type: Optional[str] = None,
    sequence: Optional[int] = None
) -> MarketDataMessage:
    """Create a market data message"""
    return MarketDataMessage(
        type="market_data",
        symbol=symbol,
        data=data,
        venue=venue,
        data_type=data_type,
        sequence=sequence
    )


def create_engine_status_message(
    data: Dict[str, Any],
    engine_id: Optional[str] = None,
    sequence: Optional[int] = None
) -> EngineStatusMessage:
    """Create an engine status message"""
    return EngineStatusMessage(
        type="engine_status",
        data=data,
        engine_id=engine_id,
        sequence=sequence
    )


def create_alert_message(
    alert_level: str,
    data: Dict[str, Any],
    alert_category: Optional[str] = None
) -> AlertMessage:
    """Create an alert message"""
    return AlertMessage(
        type="alert",
        alert_level=alert_level,
        data=data,
        alert_category=alert_category,
        priority=MessagePriority.HIGH.value if alert_level in ['error', 'critical'] else MessagePriority.NORMAL.value
    )


# Sprint 3: Message creation utilities

def create_performance_update_message(
    portfolio_id: str,
    data: Dict[str, Any],
    strategy_id: Optional[str] = None
) -> PerformanceUpdateMessage:
    """Create a performance update message"""
    return PerformanceUpdateMessage(
        type="performance_update",
        portfolio_id=portfolio_id,
        strategy_id=strategy_id,
        data=data,
        priority=MessagePriority.NORMAL.value
    )


def create_risk_alert_message(
    portfolio_id: str,
    risk_type: str,
    severity: str,
    data: Dict[str, Any]
) -> RiskAlertMessage:
    """Create a risk alert message"""
    priority = MessagePriority.HIGH.value if severity in ['high', 'critical'] else MessagePriority.NORMAL.value
    return RiskAlertMessage(
        type="risk_alert",
        portfolio_id=portfolio_id,
        risk_type=risk_type,
        severity=severity,
        data=data,
        priority=priority
    )


def create_order_update_message(
    order_id: str,
    user_id: str,
    symbol: str,
    order_status: str,
    data: Dict[str, Any]
) -> OrderUpdateMessage:
    """Create an order update message"""
    return OrderUpdateMessage(
        type="order_update",
        order_id=order_id,
        user_id=user_id,
        symbol=symbol,
        order_status=order_status,
        data=data,
        priority=MessagePriority.HIGH.value
    )


def create_position_update_message(
    portfolio_id: str,
    symbol: str,
    position_side: str,
    data: Dict[str, Any]
) -> PositionUpdateMessage:
    """Create a position update message"""
    return PositionUpdateMessage(
        type="position_update",
        portfolio_id=portfolio_id,
        symbol=symbol,
        position_side=position_side,
        data=data,
        priority=MessagePriority.NORMAL.value
    )


def create_strategy_performance_message(
    strategy_id: str,
    strategy_name: str,
    data: Dict[str, Any]
) -> StrategyPerformanceMessage:
    """Create a strategy performance message"""
    return StrategyPerformanceMessage(
        type="strategy_performance",
        strategy_id=strategy_id,
        strategy_name=strategy_name,
        data=data,
        priority=MessagePriority.NORMAL.value
    )


def create_execution_analytics_message(
    execution_id: str,
    symbol: str,
    data: Dict[str, Any]
) -> ExecutionAnalyticsMessage:
    """Create an execution analytics message"""
    return ExecutionAnalyticsMessage(
        type="execution_analytics",
        execution_id=execution_id,
        symbol=symbol,
        data=data,
        priority=MessagePriority.NORMAL.value
    )


def create_risk_metrics_message(
    portfolio_id: str,
    metric_type: str,
    data: Dict[str, Any]
) -> RiskMetricsMessage:
    """Create a risk metrics message"""
    return RiskMetricsMessage(
        type="risk_metrics",
        portfolio_id=portfolio_id,
        metric_type=metric_type,
        data=data,
        priority=MessagePriority.NORMAL.value
    )


def create_breach_alert_message(
    portfolio_id: str,
    breach_type: str,
    data: Dict[str, Any],
    severity: str = "high"
) -> BreachAlertMessage:
    """Create a breach alert message"""
    return BreachAlertMessage(
        type="breach_alert",
        portfolio_id=portfolio_id,
        breach_type=breach_type,
        severity=severity,
        data=data,
        priority=MessagePriority.CRITICAL.value
    )


def create_system_alert_message(
    component: str,
    alert_type: str,
    severity: str,
    data: Dict[str, Any]
) -> SystemAlertMessage:
    """Create a system alert message"""
    priority = MessagePriority.CRITICAL.value if severity == 'critical' else MessagePriority.HIGH.value
    return SystemAlertMessage(
        type="system_alert",
        component=component,
        alert_type=alert_type,
        severity=severity,
        data=data,
        priority=priority
    )


# Message validation utilities

def validate_symbol_format(symbol: str) -> bool:
    """Validate trading symbol format"""
    if not symbol or len(symbol) < 1 or len(symbol) > 20:
        return False
    # Allow alphanumeric characters, dots, and dashes
    return symbol.replace('.', '').replace('-', '').isalnum()


def validate_price_data(data: Dict[str, Any]) -> bool:
    """Validate price data structure"""
    price_fields = ['price', 'bid_price', 'ask_price', 'open', 'high', 'low', 'close']
    
    # Check if at least one price field exists and is numeric
    for field in price_fields:
        if field in data:
            try:
                float(data[field])
                return True
            except (ValueError, TypeError):
                continue
                
    return False


def sanitize_message_data(data: Dict[str, Any]) -> Dict[str, Any]:
    """Sanitize message data for security"""
    sanitized = {}
    
    for key, value in data.items():
        # Remove potentially dangerous keys
        if key.startswith('_') or key in ['__class__', '__module__']:
            continue
            
        # Sanitize string values
        if isinstance(value, str):
            # Remove control characters and limit length
            sanitized[key] = ''.join(char for char in value if ord(char) >= 32)[:1000]
        elif isinstance(value, (int, float, bool)):
            sanitized[key] = value
        elif isinstance(value, dict):
            sanitized[key] = sanitize_message_data(value)
        elif isinstance(value, list):
            sanitized[key] = [sanitize_message_data(item) if isinstance(item, dict) else item for item in value[:100]]
        else:
            # Convert other types to string representation
            sanitized[key] = str(value)[:1000]
            
    return sanitized


# Global protocol instance
default_protocol = MessageProtocol()