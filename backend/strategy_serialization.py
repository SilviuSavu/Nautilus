"""
Strategy Configuration Serialization Service
Converts frontend strategy configurations to NautilusTrader-compatible formats and validates parameters.
"""

import json
import logging
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass, asdict
from decimal import Decimal
from datetime import datetime, timedelta
from enum import Enum

from nautilus_trader.model.identifiers import InstrumentId, StrategyId
from nautilus_trader.model.enums import OrderSide, OrderType, TimeInForce
from nautilus_trader.model.objects import Price, Quantity
from nautilus_trader.core.uuid import UUID4


class ParameterType(Enum):
    """Parameter types for validation"""
    STRING = "string"
    INTEGER = "integer"
    DECIMAL = "decimal"
    BOOLEAN = "boolean"
    INSTRUMENT_ID = "instrument_id"
    TIMEFRAME = "timeframe"
    CURRENCY = "currency"
    PERCENTAGE = "percentage"
    ENUM = "enum"


@dataclass
class ParameterDefinition:
    """Strategy parameter definition with validation rules"""
    name: str
    display_name: str
    type: ParameterType
    required: bool
    default_value: Any = None
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    allowed_values: Optional[List[Any]] = None
    description: str = ""
    group: str = "general"


@dataclass
class ValidationResult:
    """Parameter validation result"""
    is_valid: bool
    errors: List[str]
    warnings: List[str]
    normalized_values: Dict[str, Any]


class StrategyConfigSerializer:
    """
    Strategy Configuration Serialization Service
    
    Handles conversion between frontend strategy configurations and NautilusTrader formats,
    including parameter validation, type conversion, and configuration serialization.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Define strategy templates with their parameters
        self.strategy_templates = self._initialize_strategy_templates()
        
        # Type conversion mappings
        self.type_converters = {
            ParameterType.STRING: str,
            ParameterType.INTEGER: int,
            ParameterType.DECIMAL: Decimal,
            ParameterType.BOOLEAN: bool,
            ParameterType.PERCENTAGE: lambda x: float(x) / 100.0 if isinstance(x, (int, float)) else float(x),
            ParameterType.INSTRUMENT_ID: self._convert_instrument_id,
            ParameterType.TIMEFRAME: self._convert_timeframe,
            ParameterType.CURRENCY: str,
            ParameterType.ENUM: str
        }
    
    def _initialize_strategy_templates(self) -> Dict[str, Dict[str, Any]]:
        """Initialize strategy templates with parameter definitions"""
        return {
            "MovingAverageCross": {
                "name": "Moving Average Cross",
                "category": "trend_following",
                "description": "Simple moving average crossover strategy",
                "python_class": "MovingAverageCross",
                "parameters": [
                    ParameterDefinition(
                        name="instrument_id",
                        display_name="Instrument",
                        type=ParameterType.INSTRUMENT_ID,
                        required=True,
                        description="Trading instrument (e.g., EUR/USD.SIM, AAPL.NASDAQ)",
                        group="instrument"
                    ),
                    ParameterDefinition(
                        name="fast_period",
                        display_name="Fast MA Period",
                        type=ParameterType.INTEGER,
                        required=True,
                        default_value=10,
                        min_value=1,
                        max_value=100,
                        description="Fast moving average period",
                        group="indicators"
                    ),
                    ParameterDefinition(
                        name="slow_period",
                        display_name="Slow MA Period",
                        type=ParameterType.INTEGER,
                        required=True,
                        default_value=20,
                        min_value=2,
                        max_value=200,
                        description="Slow moving average period",
                        group="indicators"
                    ),
                    ParameterDefinition(
                        name="trade_size",
                        display_name="Trade Size",
                        type=ParameterType.DECIMAL,
                        required=True,
                        default_value=Decimal("100000"),
                        min_value=1000,
                        description="Position size for each trade",
                        group="risk"
                    )
                ],
                "risk_parameters": [
                    ParameterDefinition(
                        name="max_position_size",
                        display_name="Max Position Size",
                        type=ParameterType.DECIMAL,
                        required=True,
                        default_value=Decimal("1000000"),
                        min_value=1000,
                        description="Maximum position size",
                        group="risk"
                    ),
                    ParameterDefinition(
                        name="stop_loss_atr",
                        display_name="Stop Loss (ATR)",
                        type=ParameterType.DECIMAL,
                        required=False,
                        default_value=Decimal("2.0"),
                        min_value=0.1,
                        max_value=10.0,
                        description="Stop loss in ATR multiples",
                        group="risk"
                    )
                ]
            },
            
            "MeanReversion": {
                "name": "Mean Reversion",
                "category": "mean_reversion",
                "description": "Mean reversion strategy using Z-score",
                "python_class": "MeanReversion",
                "parameters": [
                    ParameterDefinition(
                        name="instrument_id",
                        display_name="Instrument",
                        type=ParameterType.INSTRUMENT_ID,
                        required=True,
                        description="Trading instrument",
                        group="instrument"
                    ),
                    ParameterDefinition(
                        name="lookback_period",
                        display_name="Lookback Period",
                        type=ParameterType.INTEGER,
                        required=True,
                        default_value=20,
                        min_value=5,
                        max_value=100,
                        description="Lookback period for mean calculation",
                        group="indicators"
                    ),
                    ParameterDefinition(
                        name="z_score_threshold",
                        display_name="Z-Score Threshold",
                        type=ParameterType.DECIMAL,
                        required=True,
                        default_value=Decimal("2.0"),
                        min_value=0.5,
                        max_value=5.0,
                        description="Z-score threshold for signal generation",
                        group="indicators"
                    ),
                    ParameterDefinition(
                        name="position_size_pct",
                        display_name="Position Size (%)",
                        type=ParameterType.PERCENTAGE,
                        required=True,
                        default_value=5.0,
                        min_value=0.1,
                        max_value=20.0,
                        description="Position size as percentage of equity",
                        group="risk"
                    )
                ],
                "risk_parameters": [
                    ParameterDefinition(
                        name="max_daily_loss",
                        display_name="Max Daily Loss",
                        type=ParameterType.DECIMAL,
                        required=False,
                        default_value=Decimal("1000"),
                        min_value=100,
                        description="Maximum daily loss limit",
                        group="risk"
                    )
                ]
            },
            
            "TrendFollowing": {
                "name": "Trend Following",
                "category": "trend_following",
                "description": "Trend following strategy with momentum confirmation",
                "python_class": "TrendFollowing",
                "parameters": [
                    ParameterDefinition(
                        name="instrument_id",
                        display_name="Instrument",
                        type=ParameterType.INSTRUMENT_ID,
                        required=True,
                        description="Trading instrument",
                        group="instrument"
                    ),
                    ParameterDefinition(
                        name="trend_period",
                        display_name="Trend Period",
                        type=ParameterType.INTEGER,
                        required=True,
                        default_value=50,
                        min_value=10,
                        max_value=200,
                        description="Period for trend identification",
                        group="indicators"
                    ),
                    ParameterDefinition(
                        name="momentum_period",
                        display_name="Momentum Period",
                        type=ParameterType.INTEGER,
                        required=True,
                        default_value=14,
                        min_value=5,
                        max_value=50,
                        description="Period for momentum calculation",
                        group="indicators"
                    ),
                    ParameterDefinition(
                        name="trend_strength_threshold",
                        display_name="Trend Strength Threshold",
                        type=ParameterType.DECIMAL,
                        required=True,
                        default_value=Decimal("0.6"),
                        min_value=0.1,
                        max_value=1.0,
                        description="Minimum trend strength for signal",
                        group="indicators"
                    )
                ],
                "risk_parameters": [
                    ParameterDefinition(
                        name="trailing_stop_atr",
                        display_name="Trailing Stop (ATR)",
                        type=ParameterType.DECIMAL,
                        required=False,
                        default_value=Decimal("3.0"),
                        min_value=1.0,
                        max_value=10.0,
                        description="Trailing stop in ATR multiples",
                        group="risk"
                    )
                ]
            }
        }
    
    def get_strategy_templates(self) -> Dict[str, Dict[str, Any]]:
        """Get all available strategy templates"""
        result = {}
        for template_id, template in self.strategy_templates.items():
            result[template_id] = {
                "id": template_id,
                "name": template["name"],
                "category": template["category"],
                "description": template["description"],
                "python_class": template["python_class"],
                "parameters": [asdict(p) for p in template["parameters"]],
                "risk_parameters": [asdict(p) for p in template["risk_parameters"]]
            }
        return result
    
    def get_template_by_id(self, template_id: str) -> Optional[Dict[str, Any]]:
        """Get a specific strategy template"""
        if template_id not in self.strategy_templates:
            return None
        
        template = self.strategy_templates[template_id]
        return {
            "id": template_id,
            "name": template["name"],
            "category": template["category"],
            "description": template["description"],
            "python_class": template["python_class"],
            "parameters": [asdict(p) for p in template["parameters"]],
            "risk_parameters": [asdict(p) for p in template["risk_parameters"]]
        }
    
    def validate_strategy_config(self, template_id: str, parameters: Dict[str, Any]) -> ValidationResult:
        """Validate strategy configuration parameters"""
        if template_id not in self.strategy_templates:
            return ValidationResult(
                is_valid=False,
                errors=[f"Unknown strategy template: {template_id}"],
                warnings=[],
                normalized_values={}
            )
        
        template = self.strategy_templates[template_id]
        all_params = template["parameters"] + template["risk_parameters"]
        
        errors = []
        warnings = []
        normalized_values = {}
        
        # Check required parameters
        for param_def in all_params:
            param_name = param_def.name
            param_value = parameters.get(param_name)
            
            # Handle missing required parameters
            if param_def.required and param_value is None:
                if param_def.default_value is not None:
                    param_value = param_def.default_value
                    warnings.append(f"Using default value for {param_name}: {param_value}")
                else:
                    errors.append(f"Required parameter missing: {param_name}")
                    continue
            
            # Skip validation for None optional parameters
            if param_value is None:
                continue
            
            # Validate and convert parameter
            try:
                validated_value = self._validate_parameter(param_def, param_value)
                normalized_values[param_name] = validated_value
            except ValueError as e:
                errors.append(f"Invalid value for {param_name}: {str(e)}")
        
        # Check for unknown parameters
        for param_name in parameters:
            if not any(p.name == param_name for p in all_params):
                warnings.append(f"Unknown parameter: {param_name}")
        
        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            normalized_values=normalized_values
        )
    
    def _validate_parameter(self, param_def: ParameterDefinition, value: Any) -> Any:
        """Validate and convert a single parameter"""
        # Type conversion
        try:
            converter = self.type_converters[param_def.type]
            converted_value = converter(value)
        except (ValueError, TypeError) as e:
            raise ValueError(f"Cannot convert to {param_def.type.value}: {value}")
        
        # Range validation for numeric types
        if param_def.type in [ParameterType.INTEGER, ParameterType.DECIMAL, ParameterType.PERCENTAGE]:
            numeric_value = float(converted_value)
            
            if param_def.min_value is not None and numeric_value < param_def.min_value:
                raise ValueError(f"Value {numeric_value} below minimum {param_def.min_value}")
            
            if param_def.max_value is not None and numeric_value > param_def.max_value:
                raise ValueError(f"Value {numeric_value} above maximum {param_def.max_value}")
        
        # Allowed values validation
        if param_def.allowed_values is not None:
            if converted_value not in param_def.allowed_values:
                raise ValueError(f"Value must be one of: {param_def.allowed_values}")
        
        return converted_value
    
    def _convert_instrument_id(self, value: Any) -> str:
        """Convert instrument identifier"""
        if isinstance(value, str):
            # Validate instrument ID format
            if not value or '.' not in value:
                # Try to add default venue
                if value and '/' in value:  # FX pair
                    return f"{value}.SIM"
                elif value:  # Stock symbol
                    return f"{value}.NASDAQ"
            return value
        raise ValueError(f"Invalid instrument ID: {value}")
    
    def _convert_timeframe(self, value: Any) -> str:
        """Convert timeframe specification"""
        if isinstance(value, str):
            valid_timeframes = ["1s", "5s", "10s", "15s", "30s", "1m", "5m", "15m", "30m", "1h", "4h", "1d", "1w", "1M"]
            if value in valid_timeframes:
                return value
        raise ValueError(f"Invalid timeframe: {value}. Must be one of {valid_timeframes}")
    
    def serialize_to_nautilus_config(self, template_id: str, parameters: Dict[str, Any], strategy_name: str) -> Dict[str, Any]:
        """Serialize frontend config to NautilusTrader strategy configuration"""
        # Validate configuration first
        validation_result = self.validate_strategy_config(template_id, parameters)
        
        if not validation_result.is_valid:
            raise ValueError(f"Invalid configuration: {validation_result.errors}")
        
        template = self.strategy_templates[template_id]
        normalized_params = validation_result.normalized_values
        
        # Create NautilusTrader strategy configuration
        nautilus_config = {
            "strategy_id": StrategyId(f"{template['python_class']}-{UUID4()}"),
            "strategy_name": strategy_name,
            "strategy_class": template["python_class"],
            "config": {}
        }
        
        # Convert parameters to NautilusTrader format
        for param_name, param_value in normalized_params.items():
            nautilus_config["config"][param_name] = self._convert_to_nautilus_type(param_name, param_value)
        
        # Add default NautilusTrader configuration
        nautilus_config["config"].update({
            "order_id_tag": f"WEB-{strategy_name[:8]}",
            "oms_type": "NETTING",  # Default to netting OMS
            "manage_gtd_expiry": True,
            "manage_contingent_orders": True
        })
        
        return nautilus_config
    
    def _convert_to_nautilus_type(self, param_name: str, param_value: Any) -> Any:
        """Convert parameter to NautilusTrader-specific type"""
        if param_name == "instrument_id":
            return str(param_value)  # Keep as string for now
        elif isinstance(param_value, Decimal):
            return str(param_value)  # Convert Decimal to string for JSON serialization
        else:
            return param_value
    
    def serialize_to_json(self, config: Dict[str, Any]) -> str:
        """Serialize configuration to JSON string"""
        def json_serializer(obj):
            if isinstance(obj, Decimal):
                return str(obj)
            elif isinstance(obj, datetime):
                return obj.isoformat()
            elif hasattr(obj, '__dict__'):
                return obj.__dict__
            raise TypeError(f"Object of type {type(obj)} is not JSON serializable")
        
        return json.dumps(config, default=json_serializer, indent=2)
    
    def deserialize_from_json(self, json_str: str) -> Dict[str, Any]:
        """Deserialize configuration from JSON string"""
        try:
            return json.loads(json_str)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON configuration: {e}")
    
    def create_deployment_config(self, strategy_config: Dict[str, Any]) -> Dict[str, Any]:
        """Create deployment configuration for strategy execution engine"""
        template_id = strategy_config.get("template_id")
        if not template_id or template_id not in self.strategy_templates:
            raise ValueError(f"Invalid template ID: {template_id}")
        
        template = self.strategy_templates[template_id]
        
        # Validate and normalize parameters
        validation_result = self.validate_strategy_config(template_id, strategy_config.get("parameters", {}))
        if not validation_result.is_valid:
            raise ValueError(f"Invalid configuration: {validation_result.errors}")
        
        # Extract risk settings
        risk_settings = {}
        for param_def in template["risk_parameters"]:
            param_name = param_def.name
            if param_name in validation_result.normalized_values:
                risk_settings[param_name] = validation_result.normalized_values[param_name]
        
        # Remove risk settings from regular parameters
        regular_parameters = {}
        for param_def in template["parameters"]:
            param_name = param_def.name
            if param_name in validation_result.normalized_values:
                regular_parameters[param_name] = validation_result.normalized_values[param_name]
        
        return {
            "strategy_id": strategy_config.get("id", str(UUID4())),
            "name": strategy_config.get("name", f"{template['name']} Strategy"),
            "strategy_class": template["python_class"],
            "parameters": regular_parameters,
            "risk_settings": risk_settings,
            "deployment_mode": strategy_config.get("deployment_mode", "paper"),
            "auto_start": strategy_config.get("auto_start", True),
            "risk_check": strategy_config.get("risk_check", True)
        }


# Global serializer instance
_serializer: Optional[StrategyConfigSerializer] = None

def get_strategy_serializer() -> StrategyConfigSerializer:
    """Get or create the strategy serializer singleton"""
    global _serializer
    
    if _serializer is None:
        _serializer = StrategyConfigSerializer()
    
    return _serializer