"""
Strategy Error Handler - Production Compatible Version
Provides error handling and logging for strategy operations
"""

import logging
from typing import Dict, Any, Optional, List
from datetime import datetime
from enum import Enum
from dataclasses import dataclass
from fastapi import HTTPException

logger = logging.getLogger(__name__)

class ErrorSeverity(Enum):
    """Error severity levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class ErrorCategory(Enum):
    """Error categories for classification"""
    CONFIGURATION = "configuration"
    EXECUTION = "execution"
    CONNECTION = "connection"
    VALIDATION = "validation"
    PERFORMANCE = "performance"
    SECURITY = "security"

@dataclass
class StrategyError:
    """Strategy error information"""
    error_id: str
    strategy_id: str
    category: ErrorCategory
    severity: ErrorSeverity
    message: str
    details: Optional[Dict[str, Any]] = None
    timestamp: Optional[datetime] = None
    resolved: bool = False

class StrategyErrorHandler:
    """Handles and tracks strategy errors"""
    
    def __init__(self):
        self.errors: Dict[str, StrategyError] = {}
        self.error_counter = 0
    
    def log_error(self, strategy_id: str, category: ErrorCategory, 
                  severity: ErrorSeverity, message: str, 
                  details: Optional[Dict[str, Any]] = None) -> str:
        """Log a strategy error"""
        self.error_counter += 1
        error_id = f"ERR_{self.error_counter:06d}"
        
        error = StrategyError(
            error_id=error_id,
            strategy_id=strategy_id,
            category=category,
            severity=severity,
            message=message,
            details=details or {},
            timestamp=datetime.now(),
            resolved=False
        )
        
        self.errors[error_id] = error
        
        # Log to system logger based on severity
        log_message = f"Strategy {strategy_id} [{category.value}:{severity.value}] {message}"
        
        if severity == ErrorSeverity.CRITICAL:
            logger.critical(log_message)
        elif severity == ErrorSeverity.HIGH:
            logger.error(log_message)
        elif severity == ErrorSeverity.MEDIUM:
            logger.warning(log_message)
        else:
            logger.info(log_message)
        
        return error_id
    
    def resolve_error(self, error_id: str) -> bool:
        """Mark an error as resolved"""
        if error_id in self.errors:
            self.errors[error_id].resolved = True
            logger.info(f"✅ Error {error_id} marked as resolved")
            return True
        return False
    
    def get_errors(self, strategy_id: Optional[str] = None, 
                   resolved: Optional[bool] = None) -> List[StrategyError]:
        """Get errors with optional filters"""
        errors = list(self.errors.values())
        
        if strategy_id:
            errors = [e for e in errors if e.strategy_id == strategy_id]
        
        if resolved is not None:
            errors = [e for e in errors if e.resolved == resolved]
        
        return sorted(errors, key=lambda x: x.timestamp, reverse=True)
    
    def get_error_summary(self, strategy_id: Optional[str] = None) -> Dict[str, Any]:
        """Get error summary statistics"""
        errors = self.get_errors(strategy_id)
        unresolved = [e for e in errors if not e.resolved]
        
        severity_counts = {}
        category_counts = {}
        
        for error in unresolved:
            severity_counts[error.severity.value] = severity_counts.get(error.severity.value, 0) + 1
            category_counts[error.category.value] = category_counts.get(error.category.value, 0) + 1
        
        return {
            "total_errors": len(errors),
            "unresolved_errors": len(unresolved),
            "resolved_errors": len(errors) - len(unresolved),
            "severity_breakdown": severity_counts,
            "category_breakdown": category_counts,
            "last_error": errors[0].timestamp.isoformat() if errors else None
        }

# Global instance
_error_handler: Optional[StrategyErrorHandler] = None

def get_strategy_error_handler() -> StrategyErrorHandler:
    """Get or create the global strategy error handler"""
    global _error_handler
    if _error_handler is None:
        _error_handler = StrategyErrorHandler()
        logger.info("✅ Strategy Error Handler initialized")
    return _error_handler

def log_configuration_error(strategy_id: str, message: str, details: Optional[Dict[str, Any]] = None) -> str:
    """Convenience function to log configuration errors"""
    handler = get_strategy_error_handler()
    return handler.log_error(strategy_id, ErrorCategory.CONFIGURATION, ErrorSeverity.HIGH, message, details)

def log_execution_error(strategy_id: str, message: str, details: Optional[Dict[str, Any]] = None) -> str:
    """Convenience function to log execution errors"""
    handler = get_strategy_error_handler()
    return handler.log_error(strategy_id, ErrorCategory.EXECUTION, ErrorSeverity.MEDIUM, message, details)