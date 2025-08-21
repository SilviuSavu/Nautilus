"""
Strategy Management Service for NautilusTrader Integration
Handles strategy templates, configuration, deployment, and lifecycle management
"""

import json
import uuid
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
from decimal import Decimal
from dataclasses import dataclass, asdict
from enum import Enum
import logging

logger = logging.getLogger(__name__)

class ParameterType(Enum):
    STRING = "string"
    INTEGER = "integer"
    DECIMAL = "decimal"
    BOOLEAN = "boolean"
    INSTRUMENT_ID = "instrument_id"
    TIMEFRAME = "timeframe"
    CURRENCY = "currency"
    PERCENTAGE = "percentage"

class StrategyState(Enum):
    INITIALIZING = "initializing"
    RUNNING = "running"
    PAUSED = "paused"
    STOPPING = "stopping"
    STOPPED = "stopped"
    ERROR = "error"
    COMPLETED = "completed"

@dataclass
class ValidationRule:
    type: str  # 'range', 'regex', 'custom'
    params: Dict[str, Any]
    error_message: str

@dataclass
class ParameterDefinition:
    name: str
    display_name: str
    type: ParameterType
    required: bool
    default_value: Optional[Any] = None
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    allowed_values: Optional[List[Any]] = None
    validation_rules: List[ValidationRule] = None
    help_text: str = ""
    group: str = "General"

    def __post_init__(self):
        if self.validation_rules is None:
            self.validation_rules = []

@dataclass
class RiskParameterDefinition(ParameterDefinition):
    impact_level: str = "medium"  # 'low', 'medium', 'high', 'critical'

@dataclass
class ExampleConfig:
    name: str
    description: str
    parameters: Dict[str, Any]

@dataclass
class StrategyTemplate:
    id: str
    name: str
    category: str  # 'trend_following', 'mean_reversion', 'arbitrage', 'market_making'
    description: str
    python_class: str
    parameters: List[ParameterDefinition]
    risk_parameters: List[RiskParameterDefinition]
    example_configs: List[ExampleConfig]
    documentation_url: Optional[str] = None
    created_at: datetime = None
    updated_at: datetime = None

    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.utcnow()
        if self.updated_at is None:
            self.updated_at = datetime.utcnow()

@dataclass
class RiskSettings:
    max_position_size: Decimal
    stop_loss_atr: Optional[float] = None
    take_profit_atr: Optional[float] = None
    max_daily_loss: Optional[Decimal] = None
    position_sizing_method: str = "fixed"  # 'fixed', 'percentage', 'volatility_adjusted'

@dataclass
class DeploymentSettings:
    mode: str  # 'live', 'paper', 'backtest'
    venue: str
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    initial_balance: Optional[Decimal] = None

@dataclass
class StrategyConfig:
    id: str
    name: str
    template_id: str
    user_id: str
    parameters: Dict[str, Any]
    risk_settings: RiskSettings
    deployment_settings: DeploymentSettings
    version: int = 1
    status: str = "draft"  # 'draft', 'validated', 'deployed', 'archived'
    tags: List[str] = None
    created_at: datetime = None
    updated_at: datetime = None

    def __post_init__(self):
        if self.tags is None:
            self.tags = []
        if self.created_at is None:
            self.created_at = datetime.utcnow()
        if self.updated_at is None:
            self.updated_at = datetime.utcnow()

@dataclass
class PerformanceMetrics:
    total_pnl: Decimal
    unrealized_pnl: Decimal
    total_trades: int
    winning_trades: int
    win_rate: float
    max_drawdown: Decimal
    sharpe_ratio: Optional[float] = None
    last_updated: datetime = None

    def __post_init__(self):
        if self.last_updated is None:
            self.last_updated = datetime.utcnow()

@dataclass
class RuntimeInfo:
    orders_placed: int
    positions_opened: int
    last_signal_time: Optional[datetime] = None
    cpu_usage: Optional[float] = None
    memory_usage: Optional[float] = None
    uptime_seconds: int = 0

@dataclass
class ErrorEntry:
    timestamp: datetime
    level: str  # 'warning', 'error', 'critical'
    message: str
    nautilus_error: Optional[str] = None
    stack_trace: Optional[str] = None

@dataclass
class StrategyInstance:
    id: str
    config_id: str
    nautilus_strategy_id: str
    deployment_id: str
    state: StrategyState
    performance_metrics: PerformanceMetrics
    runtime_info: RuntimeInfo
    error_log: List[ErrorEntry]
    started_at: datetime
    stopped_at: Optional[datetime] = None

class StrategyService:
    """Service for managing strategy templates, configurations, and deployments"""
    
    def __init__(self):
        self._templates: Dict[str, StrategyTemplate] = {}
        self._configurations: Dict[str, StrategyConfig] = {}
        self._instances: Dict[str, StrategyInstance] = {}
        self._initialize_default_templates()

    def _initialize_default_templates(self):
        """Initialize default strategy templates"""
        
        # Moving Average Cross Strategy
        ma_cross_template = StrategyTemplate(
            id="ma_cross_001",
            name="Moving Average Cross",
            category="trend_following",
            description="A trend-following strategy that generates signals when fast MA crosses above/below slow MA",
            python_class="MovingAverageCrossStrategy",
            parameters=[
                ParameterDefinition(
                    name="instrument_id",
                    display_name="Instrument",
                    type=ParameterType.INSTRUMENT_ID,
                    required=True,
                    help_text="The trading instrument to apply the strategy to",
                    group="Basic Settings"
                ),
                ParameterDefinition(
                    name="fast_period",
                    display_name="Fast MA Period",
                    type=ParameterType.INTEGER,
                    required=True,
                    default_value=10,
                    min_value=1,
                    max_value=100,
                    help_text="Period for the fast moving average",
                    group="Technical Parameters"
                ),
                ParameterDefinition(
                    name="slow_period",
                    display_name="Slow MA Period",
                    type=ParameterType.INTEGER,
                    required=True,
                    default_value=20,
                    min_value=2,
                    max_value=200,
                    help_text="Period for the slow moving average",
                    group="Technical Parameters"
                ),
                ParameterDefinition(
                    name="trade_size",
                    display_name="Trade Size",
                    type=ParameterType.DECIMAL,
                    required=True,
                    default_value=Decimal("100000"),
                    min_value=1000,
                    help_text="Size of each trade in base currency units",
                    group="Position Sizing"
                )
            ],
            risk_parameters=[
                RiskParameterDefinition(
                    name="max_position_size",
                    display_name="Max Position Size",
                    type=ParameterType.DECIMAL,
                    required=True,
                    default_value=Decimal("1000000"),
                    help_text="Maximum position size allowed",
                    group="Risk Management",
                    impact_level="high"
                ),
                RiskParameterDefinition(
                    name="stop_loss_atr",
                    display_name="Stop Loss (ATR Multiplier)",
                    type=ParameterType.DECIMAL,
                    required=False,
                    default_value=2.0,
                    min_value=0.5,
                    max_value=5.0,
                    help_text="Stop loss as multiple of Average True Range",
                    group="Risk Management",
                    impact_level="critical"
                )
            ],
            example_configs=[
                ExampleConfig(
                    name="Conservative EUR/USD",
                    description="Conservative settings for EUR/USD trading",
                    parameters={
                        "instrument_id": "EUR/USD.SIM",
                        "fast_period": 15,
                        "slow_period": 30,
                        "trade_size": "50000",
                        "max_position_size": "500000",
                        "stop_loss_atr": 2.5
                    }
                ),
                ExampleConfig(
                    name="Aggressive GBP/USD",
                    description="More aggressive settings for volatile pairs",
                    parameters={
                        "instrument_id": "GBP/USD.SIM",
                        "fast_period": 8,
                        "slow_period": 21,
                        "trade_size": "100000",
                        "max_position_size": "1000000",
                        "stop_loss_atr": 1.5
                    }
                )
            ],
            documentation_url="https://docs.nautilustrader.io/strategies/moving-average-cross"
        )
        
        # Mean Reversion Strategy
        mean_reversion_template = StrategyTemplate(
            id="mean_revert_001",
            name="Mean Reversion RSI",
            category="mean_reversion",
            description="Mean reversion strategy using RSI to identify overbought/oversold conditions",
            python_class="MeanReversionRSIStrategy",
            parameters=[
                ParameterDefinition(
                    name="instrument_id",
                    display_name="Instrument",
                    type=ParameterType.INSTRUMENT_ID,
                    required=True,
                    help_text="The trading instrument",
                    group="Basic Settings"
                ),
                ParameterDefinition(
                    name="rsi_period",
                    display_name="RSI Period",
                    type=ParameterType.INTEGER,
                    required=True,
                    default_value=14,
                    min_value=2,
                    max_value=50,
                    help_text="Period for RSI calculation",
                    group="Technical Parameters"
                ),
                ParameterDefinition(
                    name="oversold_threshold",
                    display_name="Oversold Threshold",
                    type=ParameterType.DECIMAL,
                    required=True,
                    default_value=30.0,
                    min_value=10.0,
                    max_value=40.0,
                    help_text="RSI level considered oversold (buy signal)",
                    group="Technical Parameters"
                ),
                ParameterDefinition(
                    name="overbought_threshold",
                    display_name="Overbought Threshold",
                    type=ParameterType.DECIMAL,
                    required=True,
                    default_value=70.0,
                    min_value=60.0,
                    max_value=90.0,
                    help_text="RSI level considered overbought (sell signal)",
                    group="Technical Parameters"
                )
            ],
            risk_parameters=[
                RiskParameterDefinition(
                    name="max_position_size",
                    display_name="Max Position Size",
                    type=ParameterType.DECIMAL,
                    required=True,
                    default_value=Decimal("500000"),
                    help_text="Maximum position size",
                    group="Risk Management",
                    impact_level="high"
                )
            ],
            example_configs=[
                ExampleConfig(
                    name="Standard RSI Setup",
                    description="Standard RSI mean reversion parameters",
                    parameters={
                        "instrument_id": "USD/JPY.SIM",
                        "rsi_period": 14,
                        "oversold_threshold": 30.0,
                        "overbought_threshold": 70.0,
                        "max_position_size": "500000"
                    }
                )
            ]
        )
        
        self._templates[ma_cross_template.id] = ma_cross_template
        self._templates[mean_reversion_template.id] = mean_reversion_template

    # Template Management
    def get_templates(self, category: Optional[str] = None, search: Optional[str] = None) -> Dict[str, Any]:
        """Get available strategy templates with optional filtering"""
        templates = list(self._templates.values())
        
        if category:
            templates = [t for t in templates if t.category == category]
        
        if search:
            search_lower = search.lower()
            templates = [
                t for t in templates 
                if (search_lower in t.name.lower() or 
                    search_lower in t.description.lower() or
                    search_lower in t.python_class.lower())
            ]
        
        categories = list(set(t.category for t in self._templates.values()))
        
        return {
            "templates": [self._template_to_dict(t) for t in templates],
            "categories": sorted(categories)
        }

    def get_template(self, template_id: str) -> Optional[Dict[str, Any]]:
        """Get specific template by ID"""
        template = self._templates.get(template_id)
        return self._template_to_dict(template) if template else None

    # Configuration Management
    def create_configuration(self, template_id: str, name: str, parameters: Dict[str, Any], 
                           user_id: str = "default", risk_settings: Optional[Dict] = None) -> Dict[str, Any]:
        """Create a new strategy configuration"""
        template = self._templates.get(template_id)
        if not template:
            raise ValueError(f"Template {template_id} not found")
        
        # Validate parameters
        validation_result = self.validate_parameters(template_id, parameters)
        if not validation_result["valid"]:
            raise ValueError(f"Parameter validation failed: {validation_result['errors']}")
        
        config_id = str(uuid.uuid4())
        
        # Set default risk settings if not provided
        if not risk_settings:
            risk_settings = {
                "max_position_size": "1000000",
                "position_sizing_method": "fixed"
            }
        
        # Convert risk settings
        risk_settings_obj = RiskSettings(
            max_position_size=Decimal(str(risk_settings.get("max_position_size", "1000000"))),
            stop_loss_atr=risk_settings.get("stop_loss_atr"),
            take_profit_atr=risk_settings.get("take_profit_atr"),
            max_daily_loss=Decimal(str(risk_settings["max_daily_loss"])) if risk_settings.get("max_daily_loss") else None,
            position_sizing_method=risk_settings.get("position_sizing_method", "fixed")
        )
        
        deployment_settings = DeploymentSettings(
            mode="paper",
            venue="SIM"
        )
        
        config = StrategyConfig(
            id=config_id,
            name=name,
            template_id=template_id,
            user_id=user_id,
            parameters=parameters,
            risk_settings=risk_settings_obj,
            deployment_settings=deployment_settings
        )
        
        self._configurations[config_id] = config
        
        return {
            "strategy_id": config_id,
            "config": self._config_to_dict(config),
            "validation_result": validation_result
        }

    def validate_parameters(self, template_id: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Validate strategy parameters against template definition"""
        template = self._templates.get(template_id)
        if not template:
            return {"valid": False, "errors": [f"Template {template_id} not found"]}
        
        errors = []
        warnings = []
        
        all_params = template.parameters + template.risk_parameters
        
        for param in all_params:
            value = parameters.get(param.name)
            
            # Check required parameters
            if param.required and (value is None or value == ""):
                errors.append(f"{param.display_name} is required")
                continue
            
            if value is not None:
                # Type validation
                if param.type == ParameterType.INTEGER:
                    try:
                        int(value)
                    except (ValueError, TypeError):
                        errors.append(f"{param.display_name} must be an integer")
                
                elif param.type == ParameterType.DECIMAL:
                    try:
                        float(value)
                    except (ValueError, TypeError):
                        errors.append(f"{param.display_name} must be a number")
                
                elif param.type == ParameterType.BOOLEAN:
                    if not isinstance(value, bool):
                        errors.append(f"{param.display_name} must be true or false")
                
                # Range validation
                if param.type in [ParameterType.INTEGER, ParameterType.DECIMAL]:
                    try:
                        num_value = float(value)
                        if param.min_value is not None and num_value < param.min_value:
                            errors.append(f"{param.display_name} must be at least {param.min_value}")
                        if param.max_value is not None and num_value > param.max_value:
                            errors.append(f"{param.display_name} must be at most {param.max_value}")
                    except (ValueError, TypeError):
                        pass  # Type error already caught above
                
                # Allowed values validation
                if param.allowed_values and value not in param.allowed_values:
                    errors.append(f"{param.display_name} must be one of: {', '.join(map(str, param.allowed_values))}")
        
        return {
            "valid": len(errors) == 0,
            "errors": errors,
            "warnings": warnings
        }

    def get_configuration(self, config_id: str) -> Optional[Dict[str, Any]]:
        """Get strategy configuration by ID"""
        config = self._configurations.get(config_id)
        return self._config_to_dict(config) if config else None

    def list_configurations(self, user_id: str = None) -> List[Dict[str, Any]]:
        """List all strategy configurations, optionally filtered by user"""
        configs = list(self._configurations.values())
        if user_id:
            configs = [c for c in configs if c.user_id == user_id]
        
        return [self._config_to_dict(c) for c in configs]

    def delete_configuration(self, config_id: str) -> bool:
        """Delete a strategy configuration"""
        if config_id in self._configurations:
            del self._configurations[config_id]
            return True
        return False

    # Deployment and Control (Placeholder implementations)
    def deploy_strategy(self, config_id: str, deployment_mode: str) -> Dict[str, Any]:
        """Deploy a strategy configuration"""
        config = self._configurations.get(config_id)
        if not config:
            raise ValueError(f"Configuration {config_id} not found")
        
        deployment_id = str(uuid.uuid4())
        nautilus_strategy_id = f"strategy_{deployment_id[:8]}"
        
        # This would integrate with actual NautilusTrader deployment
        logger.info(f"Deploying strategy {config.name} in {deployment_mode} mode")
        
        return {
            "deployment_id": deployment_id,
            "status": "running",
            "nautilus_strategy_id": nautilus_strategy_id
        }

    def control_strategy(self, strategy_id: str, action: str, force: bool = False) -> Dict[str, Any]:
        """Control strategy execution (start, stop, pause, resume)"""
        logger.info(f"Strategy control: {strategy_id} -> {action} (force={force})")
        
        # This would integrate with actual NautilusTrader control
        return {
            "status": "success",
            "new_state": "running" if action == "start" else "stopped",
            "message": f"Strategy {action} completed"
        }

    def get_strategy_status(self, strategy_id: str) -> Dict[str, Any]:
        """Get current strategy status and performance"""
        # This would fetch from actual NautilusTrader instance
        return {
            "strategy_id": strategy_id,
            "state": "running",
            "performance_metrics": {
                "total_pnl": "1250.50",
                "unrealized_pnl": "125.30",
                "total_trades": 15,
                "winning_trades": 9,
                "win_rate": 0.6,
                "max_drawdown": "250.00",
                "last_updated": datetime.utcnow().isoformat()
            },
            "runtime_info": {
                "orders_placed": 30,
                "positions_opened": 15,
                "uptime_seconds": 3600
            }
        }

    # Utility methods
    def get_available_instruments(self) -> List[str]:
        """Get list of available instruments"""
        return [
            "EUR/USD.SIM", "GBP/USD.SIM", "USD/JPY.SIM", "AUD/USD.SIM",
            "USD/CAD.SIM", "USD/CHF.SIM", "NZD/USD.SIM", "EUR/GBP.SIM"
        ]

    def get_available_timeframes(self) -> List[str]:
        """Get list of available timeframes"""
        return ["1m", "5m", "15m", "30m", "1h", "4h", "1D", "1W"]

    def get_available_venues(self) -> List[str]:
        """Get list of available venues"""
        return ["SIM", "IB", "BINANCE", "DYDX"]

    def health_check(self) -> Dict[str, Any]:
        """Strategy service health check"""
        return {
            "status": "healthy",
            "timestamp": datetime.utcnow().isoformat(),
            "templates_loaded": len(self._templates),
            "configurations_active": len(self._configurations)
        }

    # Private helper methods
    def _template_to_dict(self, template: StrategyTemplate) -> Dict[str, Any]:
        """Convert StrategyTemplate to dictionary for API response"""
        def param_to_dict(param):
            result = {
                "name": param.name,
                "display_name": param.display_name,
                "type": param.type.value,
                "required": param.required,
                "help_text": param.help_text,
                "group": param.group
            }
            
            if param.default_value is not None:
                # Handle Decimal serialization
                if isinstance(param.default_value, Decimal):
                    result["default_value"] = str(param.default_value)
                else:
                    result["default_value"] = param.default_value
            if param.min_value is not None:
                result["min_value"] = param.min_value
            if param.max_value is not None:
                result["max_value"] = param.max_value
            if param.allowed_values is not None:
                result["allowed_values"] = param.allowed_values
            
            if param.validation_rules:
                result["validation_rules"] = [asdict(rule) for rule in param.validation_rules]
            
            if isinstance(param, RiskParameterDefinition):
                result["impact_level"] = param.impact_level
                
            return result

        return {
            "id": template.id,
            "name": template.name,
            "category": template.category,
            "description": template.description,
            "python_class": template.python_class,
            "parameters": [param_to_dict(param) for param in template.parameters],
            "risk_parameters": [param_to_dict(param) for param in template.risk_parameters],
            "example_configs": [asdict(example) for example in template.example_configs],
            "documentation_url": template.documentation_url,
            "created_at": template.created_at.isoformat(),
            "updated_at": template.updated_at.isoformat()
        }

    def _config_to_dict(self, config: StrategyConfig) -> Dict[str, Any]:
        """Convert StrategyConfig to dictionary for API response"""
        return {
            "id": config.id,
            "name": config.name,
            "template_id": config.template_id,
            "user_id": config.user_id,
            "parameters": config.parameters,
            "risk_settings": {
                "max_position_size": str(config.risk_settings.max_position_size),
                "stop_loss_atr": config.risk_settings.stop_loss_atr,
                "take_profit_atr": config.risk_settings.take_profit_atr,
                "max_daily_loss": str(config.risk_settings.max_daily_loss) if config.risk_settings.max_daily_loss else None,
                "position_sizing_method": config.risk_settings.position_sizing_method
            },
            "deployment_settings": asdict(config.deployment_settings),
            "version": config.version,
            "status": config.status,
            "tags": config.tags,
            "created_at": config.created_at.isoformat(),
            "updated_at": config.updated_at.isoformat()
        }

# Global service instance
strategy_service = StrategyService()