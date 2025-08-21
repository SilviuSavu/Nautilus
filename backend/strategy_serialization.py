"""
Strategy Serialization - Production Compatible Version
Provides strategy configuration serialization without direct NautilusTrader imports
"""

import json
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime
from dataclasses import asdict
from fastapi import HTTPException

logger = logging.getLogger(__name__)

class StrategyConfigSerializer:
    """Serializes and deserializes strategy configurations"""
    
    def __init__(self):
        self.supported_formats = ["json", "yaml"]
    
    def serialize_config(self, config: Dict[str, Any], format: str = "json") -> str:
        """Serialize strategy configuration to string"""
        try:
            if format.lower() == "json":
                return json.dumps(config, indent=2, default=self._json_serializer)
            elif format.lower() == "yaml":
                # Basic YAML serialization (could be enhanced with PyYAML)
                return self._dict_to_yaml(config)
            else:
                raise ValueError(f"Unsupported format: {format}")
        except Exception as e:
            logger.error(f"❌ Failed to serialize config: {e}")
            raise HTTPException(status_code=500, detail=f"Serialization failed: {e}")
    
    def deserialize_config(self, config_str: str, format: str = "json") -> Dict[str, Any]:
        """Deserialize strategy configuration from string"""
        try:
            if format.lower() == "json":
                return json.loads(config_str)
            elif format.lower() == "yaml":
                return self._yaml_to_dict(config_str)
            else:
                raise ValueError(f"Unsupported format: {format}")
        except Exception as e:
            logger.error(f"❌ Failed to deserialize config: {e}")
            raise HTTPException(status_code=500, detail=f"Deserialization failed: {e}")
    
    def validate_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Validate strategy configuration"""
        validation_result = {
            "valid": True,
            "errors": [],
            "warnings": []
        }
        
        # Required fields
        required_fields = ["strategy_id", "strategy_type", "instruments", "venues"]
        for field in required_fields:
            if field not in config:
                validation_result["valid"] = False
                validation_result["errors"].append(f"Missing required field: {field}")
        
        # Validate data types
        if "risk_limits" in config and not isinstance(config["risk_limits"], dict):
            validation_result["valid"] = False
            validation_result["errors"].append("risk_limits must be a dictionary")
        
        if "instruments" in config and not isinstance(config["instruments"], list):
            validation_result["valid"] = False
            validation_result["errors"].append("instruments must be a list")
        
        # Add warnings for optional but recommended fields
        recommended_fields = ["risk_limits", "max_position_size"]
        for field in recommended_fields:
            if field not in config:
                validation_result["warnings"].append(f"Recommended field missing: {field}")
        
        return validation_result
    
    def get_strategy_templates(self) -> Dict[str, Dict[str, Any]]:
        """Get available strategy templates"""
        template_list = [
            {
                "id": "mean_reversion",
                "name": "Mean Reversion Strategy",
                "description": "Buy low, sell high based on moving averages",
                "category": "statistical_arbitrage",
                "parameters": {
                    "lookback_period": {"type": "int", "default": 20, "min": 5, "max": 100},
                    "entry_threshold": {"type": "float", "default": 2.0, "min": 1.0, "max": 5.0},
                    "position_size": {"type": "float", "default": 0.1, "min": 0.01, "max": 1.0}
                },
                "risk_limits": {
                    "max_position_size": 10000,
                    "stop_loss_percent": 0.02,
                    "max_drawdown": 0.05
                }
            },
            {
                "id": "momentum",
                "name": "Momentum Strategy", 
                "description": "Follow trending markets with momentum indicators",
                "category": "trend_following",
                "parameters": {
                    "fast_ma": {"type": "int", "default": 10, "min": 5, "max": 50},
                    "slow_ma": {"type": "int", "default": 30, "min": 10, "max": 200},
                    "rsi_period": {"type": "int", "default": 14, "min": 7, "max": 30}
                },
                "risk_limits": {
                    "max_position_size": 15000,
                    "stop_loss_percent": 0.03,
                    "max_drawdown": 0.08
                }
            },
            {
                "id": "pairs_trading",
                "name": "Pairs Trading Strategy",
                "description": "Statistical arbitrage between correlated assets",
                "category": "statistical_arbitrage", 
                "parameters": {
                    "correlation_lookback": {"type": "int", "default": 60, "min": 30, "max": 252},
                    "entry_zscore": {"type": "float", "default": 2.0, "min": 1.5, "max": 3.0},
                    "exit_zscore": {"type": "float", "default": 0.5, "min": 0.1, "max": 1.0}
                },
                "risk_limits": {
                    "max_position_size": 20000,
                    "stop_loss_percent": 0.04,
                    "max_drawdown": 0.06
                }
            }
        ]
        # Convert to dictionary with id as key
        templates = {template["id"]: template for template in template_list}
        return templates
    
    def _json_serializer(self, obj):
        """Custom JSON serializer for special types"""
        if isinstance(obj, datetime):
            return obj.isoformat()
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")
    
    def _dict_to_yaml(self, data: Dict[str, Any], indent: int = 0) -> str:
        """Simple YAML serialization"""
        yaml_str = ""
        for key, value in data.items():
            yaml_str += "  " * indent + f"{key}:"
            if isinstance(value, dict):
                yaml_str += "\n" + self._dict_to_yaml(value, indent + 1)
            elif isinstance(value, list):
                yaml_str += "\n"
                for item in value:
                    yaml_str += "  " * (indent + 1) + f"- {item}\n"
            else:
                yaml_str += f" {value}\n"
        return yaml_str
    
    def _yaml_to_dict(self, yaml_str: str) -> Dict[str, Any]:
        """Simple YAML deserialization (basic implementation)"""
        # This is a simplified YAML parser for basic configs
        # In production, you'd use PyYAML
        result = {}
        lines = yaml_str.strip().split('\n')
        
        for line in lines:
            if ':' in line and not line.strip().startswith('-'):
                key, value = line.split(':', 1)
                key = key.strip()
                value = value.strip()
                
                # Simple type conversion
                if value.lower() in ['true', 'false']:
                    result[key] = value.lower() == 'true'
                elif value.isdigit():
                    result[key] = int(value)
                elif '.' in value and value.replace('.', '').isdigit():
                    result[key] = float(value)
                else:
                    result[key] = value
        
        return result

# Global instance
_strategy_serializer: Optional[StrategyConfigSerializer] = None

def get_strategy_serializer() -> StrategyConfigSerializer:
    """Get or create the global strategy serializer"""
    global _strategy_serializer
    if _strategy_serializer is None:
        _strategy_serializer = StrategyConfigSerializer()
        logger.info("✅ Strategy Serializer initialized")
    return _strategy_serializer