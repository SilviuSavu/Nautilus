"""
Strategy Template Manager
Manages strategy templates for NautilusTrader engine deployment

Sprint 2: Strategy Deployment System
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)

class StrategyTemplateManager:
    """Manages strategy templates and deployment configurations"""
    
    def __init__(self, templates_path: Optional[Path] = None):
        if templates_path is None:
            templates_path = Path(__file__).parent / "engine_templates"
        
        self.templates_path = templates_path
        self.strategies_path = templates_path / "strategies"
        
    def list_available_strategies(self) -> Dict[str, Any]:
        """List all available strategy templates"""
        try:
            strategies = []
            
            if self.strategies_path.exists():
                for template_file in self.strategies_path.glob("*.json"):
                    try:
                        with open(template_file, 'r') as f:
                            template_data = json.load(f)
                            
                        strategies.append({
                            "name": template_file.stem,
                            "file": template_file.name,
                            "strategy_class": template_data.get("strategy_class", "unknown"),
                            "description": template_data.get("description", "No description available"),
                            "risk_parameters": template_data.get("risk_parameters", {})
                        })
                        
                    except Exception as e:
                        logger.warning(f"Error loading strategy template {template_file}: {e}")
                        
            return {
                "success": True,
                "strategies": strategies,
                "total_count": len(strategies)
            }
            
        except Exception as e:
            logger.error(f"Error listing strategy templates: {e}")
            return {
                "success": False,
                "error": str(e)
            }
            
    def get_strategy_template(self, strategy_name: str) -> Dict[str, Any]:
        """Get a specific strategy template"""
        try:
            template_file = self.strategies_path / f"{strategy_name}.json"
            
            if not template_file.exists():
                return {
                    "success": False,
                    "error": f"Strategy template not found: {strategy_name}"
                }
                
            with open(template_file, 'r') as f:
                template_data = json.load(f)
                
            return {
                "success": True,
                "template": template_data,
                "name": strategy_name
            }
            
        except Exception as e:
            logger.error(f"Error loading strategy template {strategy_name}: {e}")
            return {
                "success": False,
                "error": str(e)
            }
            
    def create_strategy_config(self, strategy_name: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Create strategy configuration from template with parameters"""
        try:
            template_result = self.get_strategy_template(strategy_name)
            
            if not template_result["success"]:
                return template_result
                
            template = template_result["template"]
            
            # Clone the template
            strategy_config = json.loads(json.dumps(template))
            
            # Apply parameter substitutions
            strategy_config = self._apply_template_substitutions(strategy_config, parameters)
            
            return {
                "success": True,
                "config": strategy_config,
                "strategy_name": strategy_name
            }
            
        except Exception as e:
            logger.error(f"Error creating strategy config for {strategy_name}: {e}")
            return {
                "success": False,
                "error": str(e)
            }
            
    def _apply_template_substitutions(self, config: Any, parameters: Dict[str, Any]) -> Any:
        """Recursively apply template substitutions"""
        if isinstance(config, dict):
            result = {}
            for key, value in config.items():
                result[key] = self._apply_template_substitutions(value, parameters)
            return result
        elif isinstance(config, list):
            return [self._apply_template_substitutions(item, parameters) for item in config]
        elif isinstance(config, str):
            # Apply template substitutions
            for param_key, param_value in parameters.items():
                placeholder = f"{{{param_key}}}"
                if placeholder in config:
                    config = config.replace(placeholder, str(param_value))
            return config
        else:
            return config
            
    def validate_strategy_parameters(self, strategy_name: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Validate strategy parameters against template requirements"""
        try:
            template_result = self.get_strategy_template(strategy_name)
            
            if not template_result["success"]:
                return template_result
                
            template = template_result["template"]
            
            # Extract required parameters from template
            required_params = self._extract_required_parameters(template)
            
            # Check for missing parameters
            missing_params = []
            for param in required_params:
                if param not in parameters:
                    missing_params.append(param)
                    
            if missing_params:
                return {
                    "success": False,
                    "error": f"Missing required parameters: {missing_params}",
                    "required_parameters": required_params,
                    "missing_parameters": missing_params
                }
                
            return {
                "success": True,
                "message": "All required parameters provided",
                "required_parameters": required_params
            }
            
        except Exception as e:
            logger.error(f"Error validating strategy parameters: {e}")
            return {
                "success": False,
                "error": str(e)
            }
            
    def _extract_required_parameters(self, template: Any) -> List[str]:
        """Extract required parameters from template"""
        required_params = set()
        
        def extract_params(obj):
            if isinstance(obj, dict):
                for value in obj.values():
                    extract_params(value)
            elif isinstance(obj, list):
                for item in obj:
                    extract_params(item)
            elif isinstance(obj, str):
                # Find {parameter_name} patterns
                import re
                params = re.findall(r'\{(\w+)\}', obj)
                required_params.update(params)
                
        extract_params(template)
        return sorted(list(required_params))

# Global instance
_strategy_template_manager: Optional[StrategyTemplateManager] = None

def get_strategy_template_manager() -> StrategyTemplateManager:
    """Get global strategy template manager instance"""
    global _strategy_template_manager
    if _strategy_template_manager is None:
        _strategy_template_manager = StrategyTemplateManager()
    return _strategy_template_manager