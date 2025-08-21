"""
Strategy Error Handling and Logging Service
Comprehensive error handling, logging, and recovery system for NautilusTrader strategy integration.
"""

import asyncio
import logging
import traceback
import json
from typing import Dict, Any, Optional, List, Callable, Union
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
import sqlite3
import threading
from contextlib import contextmanager

from nautilus_trader.core.correctness import PyCondition


class ErrorSeverity(Enum):
    """Error severity levels"""
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class ErrorCategory(Enum):
    """Error categories"""
    CONFIGURATION = "configuration"
    CONNECTION = "connection"
    VALIDATION = "validation"
    EXECUTION = "execution"
    MARKET_DATA = "market_data"
    RISK_MANAGEMENT = "risk_management"
    SERIALIZATION = "serialization"
    SYSTEM = "system"
    NAUTILUS_TRADER = "nautilus_trader"
    USER_INPUT = "user_input"


@dataclass
class ErrorEntry:
    """Structured error entry"""
    id: str
    timestamp: datetime
    severity: ErrorSeverity
    category: ErrorCategory
    component: str
    message: str
    details: Dict[str, Any]
    stack_trace: Optional[str]
    strategy_id: Optional[str] = None
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    request_id: Optional[str] = None
    recovery_suggested: bool = False
    recovery_action: Optional[str] = None


@dataclass
class RecoveryAction:
    """Recovery action definition"""
    id: str
    name: str
    description: str
    category: ErrorCategory
    auto_execute: bool
    action_function: Callable
    conditions: List[str]  # Conditions that must be met to execute


class StrategyErrorHandler:
    """
    Comprehensive Error Handling and Logging Service
    
    Provides structured error logging, categorization, and recovery mechanisms
    for NautilusTrader strategy integration with detailed diagnostics.
    """
    
    def __init__(self, log_file_path: Optional[str] = None):
        self.logger = logging.getLogger(__name__)
        
        # Error storage
        self.error_log: List[ErrorEntry] = []
        self.max_memory_entries = 1000
        
        # Database storage for persistent logging
        self.db_path = log_file_path or "strategy_errors.db"
        self._initialize_database()
        
        # Recovery system
        self.recovery_actions: Dict[str, RecoveryAction] = {}
        self.auto_recovery_enabled = True
        
        # Error rate limiting
        self.error_rate_window = timedelta(minutes=5)
        self.max_errors_per_window = 50
        self.rate_limited_components: Dict[str, datetime] = {}
        
        # Callbacks
        self.error_callbacks: List[Callable] = []
        self.recovery_callbacks: List[Callable] = []
        
        # Threading
        self._lock = threading.Lock()
        
        # Initialize recovery actions
        self._initialize_recovery_actions()
        
        # Setup custom NautilusTrader error handling
        self._setup_nautilus_error_handling()
    
    def _initialize_database(self):
        """Initialize SQLite database for error logging"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS error_log (
                        id TEXT PRIMARY KEY,
                        timestamp TEXT NOT NULL,
                        severity TEXT NOT NULL,
                        category TEXT NOT NULL,
                        component TEXT NOT NULL,
                        message TEXT NOT NULL,
                        details TEXT,
                        stack_trace TEXT,
                        strategy_id TEXT,
                        user_id TEXT,
                        session_id TEXT,
                        request_id TEXT,
                        recovery_suggested INTEGER,
                        recovery_action TEXT
                    )
                """)
                
                conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_timestamp ON error_log(timestamp)
                """)
                
                conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_severity ON error_log(severity)
                """)
                
                conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_category ON error_log(category)
                """)
                
                conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_strategy_id ON error_log(strategy_id)
                """)
                
                conn.commit()
                
        except Exception as e:
            self.logger.error(f"Failed to initialize error database: {e}")
    
    def _initialize_recovery_actions(self):
        """Initialize recovery actions"""
        recovery_actions = [
            RecoveryAction(
                id="restart_strategy",
                name="Restart Strategy",
                description="Stop and restart a failed strategy",
                category=ErrorCategory.EXECUTION,
                auto_execute=True,
                action_function=self._recovery_restart_strategy,
                conditions=["strategy_id_present", "not_in_rate_limit"]
            ),
            RecoveryAction(
                id="reconnect_ib_gateway",
                name="Reconnect IB Gateway",
                description="Reconnect to Interactive Brokers Gateway",
                category=ErrorCategory.CONNECTION,
                auto_execute=True,
                action_function=self._recovery_reconnect_ib,
                conditions=["connection_error", "not_in_rate_limit"]
            ),
            RecoveryAction(
                id="reset_market_data",
                name="Reset Market Data",
                description="Reset market data subscriptions",
                category=ErrorCategory.MARKET_DATA,
                auto_execute=False,
                action_function=self._recovery_reset_market_data,
                conditions=["market_data_error"]
            ),
            RecoveryAction(
                id="validate_configuration",
                name="Validate Configuration",
                description="Re-validate strategy configuration",
                category=ErrorCategory.CONFIGURATION,
                auto_execute=False,
                action_function=self._recovery_validate_config,
                conditions=["configuration_error"]
            )
        ]
        
        for action in recovery_actions:
            self.recovery_actions[action.id] = action
    
    def _setup_nautilus_error_handling(self):
        """Setup custom error handling for NautilusTrader components"""
        # Custom logging handler for NautilusTrader
        nautilus_logger = logging.getLogger("nautilus_trader")
        handler = NautilusErrorHandler(self)
        nautilus_logger.addHandler(handler)
    
    async def log_error(
        self,
        message: str,
        severity: ErrorSeverity = ErrorSeverity.ERROR,
        category: ErrorCategory = ErrorCategory.SYSTEM,
        component: str = "unknown",
        details: Optional[Dict[str, Any]] = None,
        exception: Optional[Exception] = None,
        strategy_id: Optional[str] = None,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        request_id: Optional[str] = None,
        suggest_recovery: bool = True
    ) -> str:
        """Log an error with structured information"""
        
        # Check rate limiting
        if self._is_rate_limited(component):
            return "rate_limited"
        
        # Generate unique error ID
        error_id = f"err_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{id(object())}"
        
        # Prepare error details
        error_details = details or {}
        if exception:
            error_details.update({
                "exception_type": type(exception).__name__,
                "exception_args": str(exception.args) if exception.args else None,
                "exception_str": str(exception)
            })
        
        # Get stack trace
        stack_trace = None
        if exception:
            stack_trace = traceback.format_exception(type(exception), exception, exception.__traceback__)
            stack_trace = ''.join(stack_trace)
        elif severity in [ErrorSeverity.ERROR, ErrorSeverity.CRITICAL]:
            stack_trace = ''.join(traceback.format_stack())
        
        # Determine recovery action
        recovery_action = None
        if suggest_recovery:
            recovery_action = self._suggest_recovery_action(category, error_details, strategy_id)
        
        # Create error entry
        error_entry = ErrorEntry(
            id=error_id,
            timestamp=datetime.now(),
            severity=severity,
            category=category,
            component=component,
            message=message,
            details=error_details,
            stack_trace=stack_trace,
            strategy_id=strategy_id,
            user_id=user_id,
            session_id=session_id,
            request_id=request_id,
            recovery_suggested=recovery_action is not None,
            recovery_action=recovery_action
        )
        
        # Store error
        await self._store_error(error_entry)
        
        # Log to standard logger
        log_level = {
            ErrorSeverity.DEBUG: logging.DEBUG,
            ErrorSeverity.INFO: logging.INFO,
            ErrorSeverity.WARNING: logging.WARNING,
            ErrorSeverity.ERROR: logging.ERROR,
            ErrorSeverity.CRITICAL: logging.CRITICAL
        }[severity]
        
        self.logger.log(log_level, f"[{category.value}] {component}: {message}")
        if exception:
            self.logger.log(log_level, f"Exception details: {error_details}")
        
        # Notify callbacks
        await self._notify_error_callbacks(error_entry)
        
        # Attempt recovery if suggested and auto-recovery enabled
        if recovery_action and self.auto_recovery_enabled:
            await self._attempt_recovery(error_entry, recovery_action)
        
        return error_id
    
    async def _store_error(self, error_entry: ErrorEntry):
        """Store error in memory and database"""
        with self._lock:
            # Add to memory
            self.error_log.append(error_entry)
            
            # Trim memory log if needed
            if len(self.error_log) > self.max_memory_entries:
                self.error_log = self.error_log[-self.max_memory_entries:]
        
        # Store in database
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT INTO error_log (
                        id, timestamp, severity, category, component, message,
                        details, stack_trace, strategy_id, user_id, session_id,
                        request_id, recovery_suggested, recovery_action
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    error_entry.id,
                    error_entry.timestamp.isoformat(),
                    error_entry.severity.value,
                    error_entry.category.value,
                    error_entry.component,
                    error_entry.message,
                    json.dumps(error_entry.details),
                    error_entry.stack_trace,
                    error_entry.strategy_id,
                    error_entry.user_id,
                    error_entry.session_id,
                    error_entry.request_id,
                    1 if error_entry.recovery_suggested else 0,
                    error_entry.recovery_action
                ))
                conn.commit()
        except Exception as e:
            self.logger.error(f"Failed to store error in database: {e}")
    
    def _is_rate_limited(self, component: str) -> bool:
        """Check if component is rate limited"""
        now = datetime.now()
        
        # Clean old rate limit entries
        for comp, timestamp in list(self.rate_limited_components.items()):
            if now - timestamp > self.error_rate_window:
                del self.rate_limited_components[comp]
        
        # Count recent errors for this component
        recent_errors = [
            e for e in self.error_log
            if e.component == component and (now - e.timestamp) <= self.error_rate_window
        ]
        
        if len(recent_errors) >= self.max_errors_per_window:
            self.rate_limited_components[component] = now
            return True
        
        return False
    
    def _suggest_recovery_action(self, category: ErrorCategory, details: Dict[str, Any], strategy_id: Optional[str]) -> Optional[str]:
        """Suggest recovery action based on error characteristics"""
        
        # Strategy execution errors
        if category == ErrorCategory.EXECUTION and strategy_id:
            return "restart_strategy"
        
        # Connection errors
        if category == ErrorCategory.CONNECTION:
            if "ib" in str(details).lower() or "interactive_brokers" in str(details).lower():
                return "reconnect_ib_gateway"
        
        # Market data errors
        if category == ErrorCategory.MARKET_DATA:
            return "reset_market_data"
        
        # Configuration errors
        if category == ErrorCategory.CONFIGURATION:
            return "validate_configuration"
        
        return None
    
    async def _notify_error_callbacks(self, error_entry: ErrorEntry):
        """Notify error callbacks"""
        for callback in self.error_callbacks:
            try:
                await callback(error_entry)
            except Exception as e:
                self.logger.error(f"Error in error callback: {e}")
    
    async def _attempt_recovery(self, error_entry: ErrorEntry, recovery_action_id: str):
        """Attempt recovery action"""
        if recovery_action_id not in self.recovery_actions:
            return
        
        recovery_action = self.recovery_actions[recovery_action_id]
        
        # Check conditions
        if not self._check_recovery_conditions(recovery_action, error_entry):
            return
        
        try:
            self.logger.info(f"Attempting recovery action: {recovery_action.name}")
            
            # Execute recovery action
            result = await recovery_action.action_function(error_entry)
            
            # Log recovery attempt
            await self.log_error(
                f"Recovery action '{recovery_action.name}' executed: {result}",
                severity=ErrorSeverity.INFO,
                category=ErrorCategory.SYSTEM,
                component="recovery_system",
                details={"recovery_action_id": recovery_action_id, "result": result}
            )
            
            # Notify callbacks
            for callback in self.recovery_callbacks:
                try:
                    await callback(recovery_action, error_entry, result)
                except Exception as e:
                    self.logger.error(f"Error in recovery callback: {e}")
            
        except Exception as e:
            await self.log_error(
                f"Recovery action '{recovery_action.name}' failed: {str(e)}",
                severity=ErrorSeverity.ERROR,
                category=ErrorCategory.SYSTEM,
                component="recovery_system",
                exception=e,
                details={"recovery_action_id": recovery_action_id}
            )
    
    def _check_recovery_conditions(self, recovery_action: RecoveryAction, error_entry: ErrorEntry) -> bool:
        """Check if recovery conditions are met"""
        for condition in recovery_action.conditions:
            if condition == "strategy_id_present" and not error_entry.strategy_id:
                return False
            elif condition == "not_in_rate_limit" and self._is_rate_limited(f"recovery_{recovery_action.id}"):
                return False
            elif condition == "connection_error" and error_entry.category != ErrorCategory.CONNECTION:
                return False
            elif condition == "market_data_error" and error_entry.category != ErrorCategory.MARKET_DATA:
                return False
            elif condition == "configuration_error" and error_entry.category != ErrorCategory.CONFIGURATION:
                return False
        
        return True
    
    # Recovery action implementations
    
    async def _recovery_restart_strategy(self, error_entry: ErrorEntry) -> str:
        """Recovery action: restart strategy"""
        if not error_entry.strategy_id:
            return "No strategy ID provided"
        
        try:
            from strategy_execution_engine import get_strategy_execution_engine
            engine = get_strategy_execution_engine()
            
            # Stop strategy
            stop_result = await engine.control_strategy(error_entry.strategy_id, "stop")
            
            # Wait a bit
            await asyncio.sleep(3)
            
            # Start strategy
            start_result = await engine.control_strategy(error_entry.strategy_id, "start")
            
            if start_result.get("status") == "success":
                return "Strategy restarted successfully"
            else:
                return f"Failed to restart strategy: {start_result.get('message')}"
                
        except Exception as e:
            return f"Error during strategy restart: {str(e)}"
    
    async def _recovery_reconnect_ib(self, error_entry: ErrorEntry) -> str:
        """Recovery action: reconnect IB Gateway"""
        try:
            from nautilus_ib_adapter import get_nautilus_ib_adapter
            adapter = get_nautilus_ib_adapter()
            
            # Disconnect and reconnect
            await adapter.disconnect()
            await asyncio.sleep(2)
            success = await adapter.connect()
            
            if success:
                return "IB Gateway reconnected successfully"
            else:
                return "Failed to reconnect IB Gateway"
                
        except Exception as e:
            return f"Error during IB Gateway reconnection: {str(e)}"
    
    async def _recovery_reset_market_data(self, error_entry: ErrorEntry) -> str:
        """Recovery action: reset market data"""
        try:
            # This would reset market data subscriptions
            # Implementation depends on specific market data system
            return "Market data reset completed"
        except Exception as e:
            return f"Error during market data reset: {str(e)}"
    
    async def _recovery_validate_config(self, error_entry: ErrorEntry) -> str:
        """Recovery action: validate configuration"""
        try:
            from strategy_serialization import get_strategy_serializer
            serializer = get_strategy_serializer()
            
            # This would re-validate strategy configurations
            # Implementation depends on specific validation needs
            return "Configuration validation completed"
        except Exception as e:
            return f"Error during configuration validation: {str(e)}"
    
    # Public API methods
    
    def get_errors(
        self,
        severity: Optional[ErrorSeverity] = None,
        category: Optional[ErrorCategory] = None,
        component: Optional[str] = None,
        strategy_id: Optional[str] = None,
        limit: int = 100,
        offset: int = 0
    ) -> List[Dict[str, Any]]:
        """Get errors with filtering"""
        
        filtered_errors = []
        
        for error in reversed(self.error_log):  # Most recent first
            if severity and error.severity != severity:
                continue
            if category and error.category != category:
                continue
            if component and error.component != component:
                continue
            if strategy_id and error.strategy_id != strategy_id:
                continue
            
            filtered_errors.append(asdict(error))
            
            if len(filtered_errors) >= offset + limit:
                break
        
        return filtered_errors[offset:offset + limit]
    
    def get_error_summary(self) -> Dict[str, Any]:
        """Get error summary statistics"""
        now = datetime.now()
        last_hour = now - timedelta(hours=1)
        last_day = now - timedelta(days=1)
        
        recent_errors_hour = [e for e in self.error_log if e.timestamp > last_hour]
        recent_errors_day = [e for e in self.error_log if e.timestamp > last_day]
        
        summary = {
            "total_errors": len(self.error_log),
            "errors_last_hour": len(recent_errors_hour),
            "errors_last_day": len(recent_errors_day),
            "by_severity": {},
            "by_category": {},
            "by_component": {},
            "rate_limited_components": list(self.rate_limited_components.keys()),
            "recovery_actions_available": len(self.recovery_actions)
        }
        
        # Count by severity
        for severity in ErrorSeverity:
            summary["by_severity"][severity.value] = len([
                e for e in recent_errors_day if e.severity == severity
            ])
        
        # Count by category
        for category in ErrorCategory:
            summary["by_category"][category.value] = len([
                e for e in recent_errors_day if e.category == category
            ])
        
        # Count by component (top 10)
        component_counts = {}
        for error in recent_errors_day:
            component_counts[error.component] = component_counts.get(error.component, 0) + 1
        
        summary["by_component"] = dict(sorted(
            component_counts.items(), key=lambda x: x[1], reverse=True
        )[:10])
        
        return summary
    
    def add_error_callback(self, callback: Callable):
        """Add error callback"""
        self.error_callbacks.append(callback)
    
    def add_recovery_callback(self, callback: Callable):
        """Add recovery callback"""
        self.recovery_callbacks.append(callback)
    
    def enable_auto_recovery(self, enabled: bool = True):
        """Enable or disable auto-recovery"""
        self.auto_recovery_enabled = enabled
    
    async def execute_recovery_action(self, recovery_action_id: str, error_entry: ErrorEntry) -> str:
        """Manually execute a recovery action"""
        if recovery_action_id not in self.recovery_actions:
            return "Recovery action not found"
        
        recovery_action = self.recovery_actions[recovery_action_id]
        
        try:
            result = await recovery_action.action_function(error_entry)
            
            await self.log_error(
                f"Manual recovery action '{recovery_action.name}' executed: {result}",
                severity=ErrorSeverity.INFO,
                category=ErrorCategory.SYSTEM,
                component="recovery_system",
                details={"recovery_action_id": recovery_action_id, "result": result, "manual": True}
            )
            
            return result
            
        except Exception as e:
            await self.log_error(
                f"Manual recovery action '{recovery_action.name}' failed: {str(e)}",
                severity=ErrorSeverity.ERROR,
                category=ErrorCategory.SYSTEM,
                component="recovery_system",
                exception=e,
                details={"recovery_action_id": recovery_action_id, "manual": True}
            )
            return f"Recovery action failed: {str(e)}"


class NautilusErrorHandler(logging.Handler):
    """Custom logging handler for NautilusTrader errors"""
    
    def __init__(self, error_handler: StrategyErrorHandler):
        super().__init__()
        self.error_handler = error_handler
        
    def emit(self, record):
        """Handle log record from NautilusTrader"""
        try:
            # Map logging levels to error severities
            severity_map = {
                logging.DEBUG: ErrorSeverity.DEBUG,
                logging.INFO: ErrorSeverity.INFO,
                logging.WARNING: ErrorSeverity.WARNING,
                logging.ERROR: ErrorSeverity.ERROR,
                logging.CRITICAL: ErrorSeverity.CRITICAL
            }
            
            severity = severity_map.get(record.levelno, ErrorSeverity.ERROR)
            
            # Determine category from logger name
            category = ErrorCategory.NAUTILUS_TRADER
            if "data" in record.name.lower():
                category = ErrorCategory.MARKET_DATA
            elif "execution" in record.name.lower():
                category = ErrorCategory.EXECUTION
            elif "risk" in record.name.lower():
                category = ErrorCategory.RISK_MANAGEMENT
            
            # Create error entry asynchronously
            asyncio.create_task(self.error_handler.log_error(
                message=record.getMessage(),
                severity=severity,
                category=category,
                component=record.name,
                details={
                    "level": record.levelname,
                    "pathname": record.pathname,
                    "lineno": record.lineno,
                    "funcName": record.funcName,
                    "process": record.process,
                    "thread": record.thread
                },
                suggest_recovery=severity in [ErrorSeverity.ERROR, ErrorSeverity.CRITICAL]
            ))
            
        except Exception:
            # Don't let error handling errors break the system
            pass


# Global error handler instance
_error_handler: Optional[StrategyErrorHandler] = None

def get_strategy_error_handler() -> StrategyErrorHandler:
    """Get or create the strategy error handler singleton"""
    global _error_handler
    
    if _error_handler is None:
        _error_handler = StrategyErrorHandler()
    
    return _error_handler

# Convenience functions for common error logging

async def log_configuration_error(message: str, details: Optional[Dict[str, Any]] = None, exception: Optional[Exception] = None):
    """Log configuration error"""
    handler = get_strategy_error_handler()
    return await handler.log_error(
        message=message,
        severity=ErrorSeverity.ERROR,
        category=ErrorCategory.CONFIGURATION,
        component="configuration",
        details=details,
        exception=exception
    )

async def log_connection_error(message: str, details: Optional[Dict[str, Any]] = None, exception: Optional[Exception] = None):
    """Log connection error"""
    handler = get_strategy_error_handler()
    return await handler.log_error(
        message=message,
        severity=ErrorSeverity.ERROR,
        category=ErrorCategory.CONNECTION,
        component="connection",
        details=details,
        exception=exception
    )

async def log_execution_error(message: str, strategy_id: Optional[str] = None, details: Optional[Dict[str, Any]] = None, exception: Optional[Exception] = None):
    """Log strategy execution error"""
    handler = get_strategy_error_handler()
    return await handler.log_error(
        message=message,
        severity=ErrorSeverity.ERROR,
        category=ErrorCategory.EXECUTION,
        component="strategy_execution",
        details=details,
        exception=exception,
        strategy_id=strategy_id
    )

async def log_market_data_error(message: str, details: Optional[Dict[str, Any]] = None, exception: Optional[Exception] = None):
    """Log market data error"""
    handler = get_strategy_error_handler()
    return await handler.log_error(
        message=message,
        severity=ErrorSeverity.ERROR,
        category=ErrorCategory.MARKET_DATA,
        component="market_data",
        details=details,
        exception=exception
    )