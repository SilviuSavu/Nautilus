"""
Strategy Runtime Management Service
Manages strategy lifecycle, monitoring, performance tracking, and error handling for NautilusTrader strategies.
"""

import asyncio
import logging
import threading
import time
import traceback
from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from enum import Enum
from decimal import Decimal
import psutil
import json

from strategy_execution_engine import StrategyExecutionEngine, StrategyState, StrategyInstance
from strategy_serialization import get_strategy_serializer


class RuntimeStatus(Enum):
    """Runtime status for strategy manager"""
    STOPPED = "stopped"
    STARTING = "starting"
    RUNNING = "running"
    STOPPING = "stopping"
    ERROR = "error"


@dataclass
class SystemMetrics:
    """System resource metrics"""
    cpu_usage: float
    memory_usage: float
    disk_usage: float
    network_io: Dict[str, int]
    timestamp: datetime


@dataclass
class StrategyMetrics:
    """Strategy-specific performance metrics"""
    deployment_id: str
    pnl: Decimal
    unrealized_pnl: Decimal
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    max_drawdown: Decimal
    current_drawdown: Decimal
    sharpe_ratio: Optional[float]
    orders_placed: int
    positions_opened: int
    avg_trade_duration: Optional[timedelta]
    last_trade_time: Optional[datetime]
    last_updated: datetime


@dataclass
class AlertRule:
    """Performance alert rule"""
    id: str
    name: str
    strategy_id: Optional[str]  # None for global rules
    metric: str
    condition: str  # 'gt', 'lt', 'eq', 'change_pct'
    threshold: float
    enabled: bool
    last_triggered: Optional[datetime] = None


class StrategyRuntimeManager:
    """
    Strategy Runtime Management Service
    
    Provides comprehensive runtime management for NautilusTrader strategies including:
    - Performance monitoring and metrics collection
    - Resource usage tracking
    - Error detection and recovery
    - Alert system for performance thresholds
    - Runtime optimization and health checks
    """
    
    def __init__(self, execution_engine: StrategyExecutionEngine):
        self.logger = logging.getLogger(__name__)
        self.execution_engine = execution_engine
        
        # Runtime state
        self.status = RuntimeStatus.STOPPED
        self.start_time: Optional[datetime] = None
        
        # Monitoring
        self.monitoring_interval = 5.0  # seconds
        self.monitoring_task: Optional[asyncio.Task] = None
        self.metrics_history: Dict[str, List[StrategyMetrics]] = {}
        self.system_metrics_history: List[SystemMetrics] = []
        
        # Alert system
        self.alert_rules: Dict[str, AlertRule] = {}
        self.alert_callbacks: List[Callable] = []
        
        # Performance tracking
        self.strategy_snapshots: Dict[str, Dict[str, Any]] = {}
        
        # Error handling
        self.error_callbacks: List[Callable] = []
        self.max_error_rate = 10  # errors per minute
        self.error_window = timedelta(minutes=1)
        self.error_log: List[Dict[str, Any]] = []
        
        # Recovery settings
        self.auto_recovery_enabled = True
        self.max_recovery_attempts = 3
        self.recovery_attempt_interval = timedelta(minutes=5)
        self.strategy_recovery_attempts: Dict[str, int] = {}
        
        # Initialize default alert rules
        self._initialize_default_alerts()
    
    def _initialize_default_alerts(self):
        """Initialize default alert rules"""
        default_rules = [
            AlertRule(
                id="high_drawdown",
                name="High Drawdown Alert",
                strategy_id=None,
                metric="current_drawdown",
                condition="gt",
                threshold=0.05,  # 5% drawdown
                enabled=True
            ),
            AlertRule(
                id="low_win_rate",
                name="Low Win Rate Alert",
                strategy_id=None,
                metric="win_rate",
                condition="lt",
                threshold=0.3,  # 30% win rate
                enabled=True
            ),
            AlertRule(
                id="high_memory_usage",
                name="High Memory Usage",
                strategy_id=None,
                metric="memory_usage",
                condition="gt",
                threshold=80.0,  # 80% memory usage
                enabled=True
            ),
            AlertRule(
                id="strategy_error_rate",
                name="High Error Rate",
                strategy_id=None,
                metric="error_rate",
                condition="gt",
                threshold=5.0,  # 5 errors per minute
                enabled=True
            )
        ]
        
        for rule in default_rules:
            self.alert_rules[rule.id] = rule
    
    async def start(self) -> bool:
        """Start the runtime manager"""
        try:
            if self.status == RuntimeStatus.RUNNING:
                self.logger.warning("Runtime manager already running")
                return True
            
            self.logger.info("Starting strategy runtime manager...")
            self.status = RuntimeStatus.STARTING
            
            # Initialize execution engine if needed
            if not self.execution_engine.trading_node:
                if not await self.execution_engine.initialize_trading_node():
                    raise RuntimeError("Failed to initialize execution engine")
            
            # Start monitoring
            self.monitoring_task = asyncio.create_task(self._monitoring_loop())
            
            self.start_time = datetime.now()
            self.status = RuntimeStatus.RUNNING
            
            self.logger.info("Strategy runtime manager started successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to start runtime manager: {e}")
            self.logger.error(traceback.format_exc())
            self.status = RuntimeStatus.ERROR
            return False
    
    async def stop(self):
        """Stop the runtime manager"""
        try:
            if self.status == RuntimeStatus.STOPPED:
                return
            
            self.logger.info("Stopping strategy runtime manager...")
            self.status = RuntimeStatus.STOPPING
            
            # Cancel monitoring
            if self.monitoring_task:
                self.monitoring_task.cancel()
                try:
                    await self.monitoring_task
                except asyncio.CancelledError:
                    pass
            
            self.status = RuntimeStatus.STOPPED
            self.logger.info("Strategy runtime manager stopped")
            
        except Exception as e:
            self.logger.error(f"Error stopping runtime manager: {e}")
    
    async def _monitoring_loop(self):
        """Main monitoring loop"""
        while self.status == RuntimeStatus.RUNNING:
            try:
                # Collect system metrics
                system_metrics = self._collect_system_metrics()
                self.system_metrics_history.append(system_metrics)
                
                # Keep only last 1000 system metrics (about 1.4 hours at 5s interval)
                if len(self.system_metrics_history) > 1000:
                    self.system_metrics_history = self.system_metrics_history[-1000:]
                
                # Collect strategy metrics
                await self._collect_strategy_metrics()
                
                # Check alerts
                await self._check_alerts()
                
                # Perform health checks
                await self._perform_health_checks()
                
                # Clean up old data
                self._cleanup_old_data()
                
                await asyncio.sleep(self.monitoring_interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {e}")
                await self._handle_monitoring_error(e)
                await asyncio.sleep(self.monitoring_interval)
    
    def _collect_system_metrics(self) -> SystemMetrics:
        """Collect system resource metrics"""
        try:
            cpu_usage = psutil.cpu_percent(interval=None)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            network = psutil.net_io_counters()
            
            return SystemMetrics(
                cpu_usage=cpu_usage,
                memory_usage=memory.percent,
                disk_usage=(disk.used / disk.total) * 100,
                network_io={
                    'bytes_sent': network.bytes_sent,
                    'bytes_recv': network.bytes_recv
                },
                timestamp=datetime.now()
            )
        except Exception as e:
            self.logger.warning(f"Failed to collect system metrics: {e}")
            return SystemMetrics(
                cpu_usage=0.0,
                memory_usage=0.0,
                disk_usage=0.0,
                network_io={'bytes_sent': 0, 'bytes_recv': 0},
                timestamp=datetime.now()
            )
    
    async def _collect_strategy_metrics(self):
        """Collect metrics for all deployed strategies"""
        deployed_strategies = self.execution_engine.get_deployed_strategies()
        
        for deployment_id, instance in deployed_strategies.items():
            try:
                # Get updated status from execution engine
                status = await self.execution_engine.get_strategy_status(deployment_id)
                
                # Calculate additional metrics
                metrics = self._calculate_strategy_metrics(deployment_id, instance, status)
                
                # Store metrics
                if deployment_id not in self.metrics_history:
                    self.metrics_history[deployment_id] = []
                
                self.metrics_history[deployment_id].append(metrics)
                
                # Keep only last 1000 metrics per strategy
                if len(self.metrics_history[deployment_id]) > 1000:
                    self.metrics_history[deployment_id] = self.metrics_history[deployment_id][-1000:]
                
            except Exception as e:
                self.logger.warning(f"Failed to collect metrics for strategy {deployment_id}: {e}")
    
    def _calculate_strategy_metrics(self, deployment_id: str, instance: StrategyInstance, status: Dict[str, Any]) -> StrategyMetrics:
        """Calculate comprehensive strategy metrics"""
        perf_metrics = status.get("performance_metrics", {})
        runtime_info = status.get("runtime_info", {})
        
        # Get previous metrics for calculations
        previous_metrics = None
        if deployment_id in self.metrics_history and self.metrics_history[deployment_id]:
            previous_metrics = self.metrics_history[deployment_id][-1]
        
        # Calculate win/loss counts
        total_trades = perf_metrics.get("total_trades", 0)
        win_rate = perf_metrics.get("win_rate", 0.0)
        winning_trades = int(total_trades * win_rate) if total_trades > 0 else 0
        losing_trades = total_trades - winning_trades
        
        # Calculate current drawdown (simplified)
        current_pnl = Decimal(str(perf_metrics.get("total_pnl", 0)))
        max_pnl = current_pnl  # Simplified - should track historical max
        if previous_metrics:
            max_pnl = max(max_pnl, previous_metrics.pnl)
        
        current_drawdown = max_pnl - current_pnl if max_pnl > 0 else Decimal('0')
        
        # Calculate Sharpe ratio (simplified)
        sharpe_ratio = None
        if len(self.metrics_history.get(deployment_id, [])) > 20:  # Need some history
            # Simplified Sharpe calculation
            returns = []
            for i in range(1, min(len(self.metrics_history[deployment_id]), 100)):
                prev_pnl = self.metrics_history[deployment_id][i-1].pnl
                curr_pnl = self.metrics_history[deployment_id][i].pnl
                if prev_pnl != 0:
                    returns.append(float((curr_pnl - prev_pnl) / prev_pnl))
            
            if returns and len(returns) > 5:
                avg_return = sum(returns) / len(returns)
                std_return = (sum((r - avg_return) ** 2 for r in returns) / len(returns)) ** 0.5
                sharpe_ratio = avg_return / std_return if std_return > 0 else 0
        
        return StrategyMetrics(
            deployment_id=deployment_id,
            pnl=current_pnl,
            unrealized_pnl=Decimal(str(perf_metrics.get("unrealized_pnl", 0))),
            total_trades=total_trades,
            winning_trades=winning_trades,
            losing_trades=losing_trades,
            win_rate=win_rate,
            max_drawdown=Decimal(str(perf_metrics.get("max_drawdown", 0))),
            current_drawdown=current_drawdown,
            sharpe_ratio=sharpe_ratio,
            orders_placed=runtime_info.get("orders_placed", 0),
            positions_opened=runtime_info.get("positions_opened", 0),
            avg_trade_duration=None,  # TODO: Calculate from trade history
            last_trade_time=None,     # TODO: Get from last trade
            last_updated=datetime.now()
        )
    
    async def _check_alerts(self):
        """Check all alert rules and trigger alerts"""
        current_time = datetime.now()
        
        for rule in self.alert_rules.values():
            if not rule.enabled:
                continue
            
            # Skip if recently triggered (within 5 minutes)
            if rule.last_triggered and (current_time - rule.last_triggered) < timedelta(minutes=5):
                continue
            
            try:
                if await self._evaluate_alert_rule(rule):
                    await self._trigger_alert(rule)
                    rule.last_triggered = current_time
            except Exception as e:
                self.logger.warning(f"Error evaluating alert rule {rule.id}: {e}")
    
    async def _evaluate_alert_rule(self, rule: AlertRule) -> bool:
        """Evaluate a single alert rule"""
        if rule.strategy_id:
            # Strategy-specific rule
            if rule.strategy_id not in self.metrics_history or not self.metrics_history[rule.strategy_id]:
                return False
            
            latest_metrics = self.metrics_history[rule.strategy_id][-1]
            metric_value = getattr(latest_metrics, rule.metric, None)
        else:
            # Global/system rule
            if rule.metric == "memory_usage" and self.system_metrics_history:
                metric_value = self.system_metrics_history[-1].memory_usage
            elif rule.metric == "cpu_usage" and self.system_metrics_history:
                metric_value = self.system_metrics_history[-1].cpu_usage
            elif rule.metric == "error_rate":
                metric_value = self._calculate_error_rate()
            else:
                return False
        
        if metric_value is None:
            return False
        
        # Convert to float for comparison
        if isinstance(metric_value, Decimal):
            metric_value = float(metric_value)
        
        # Evaluate condition
        if rule.condition == "gt":
            return metric_value > rule.threshold
        elif rule.condition == "lt":
            return metric_value < rule.threshold
        elif rule.condition == "eq":
            return abs(metric_value - rule.threshold) < 0.01
        elif rule.condition == "change_pct":
            # TODO: Implement percentage change logic
            return False
        
        return False
    
    def _calculate_error_rate(self) -> float:
        """Calculate current error rate (errors per minute)"""
        current_time = datetime.now()
        window_start = current_time - self.error_window
        
        recent_errors = [
            error for error in self.error_log
            if datetime.fromisoformat(error["timestamp"]) > window_start
        ]
        
        return len(recent_errors)
    
    async def _trigger_alert(self, rule: AlertRule):
        """Trigger an alert"""
        alert_data = {
            "rule_id": rule.id,
            "rule_name": rule.name,
            "strategy_id": rule.strategy_id,
            "metric": rule.metric,
            "threshold": rule.threshold,
            "timestamp": datetime.now().isoformat(),
            "severity": self._get_alert_severity(rule)
        }
        
        self.logger.warning(f"Alert triggered: {rule.name} - {rule.metric} {rule.condition} {rule.threshold}")
        
        # Notify callbacks
        for callback in self.alert_callbacks:
            try:
                await callback(alert_data)
            except Exception as e:
                self.logger.error(f"Error in alert callback: {e}")
    
    def _get_alert_severity(self, rule: AlertRule) -> str:
        """Determine alert severity"""
        if rule.metric in ["error_rate", "memory_usage"] and rule.threshold > 90:
            return "critical"
        elif rule.metric in ["current_drawdown"] and rule.threshold > 0.1:
            return "high"
        else:
            return "medium"
    
    async def _perform_health_checks(self):
        """Perform health checks on running strategies"""
        deployed_strategies = self.execution_engine.get_deployed_strategies()
        
        for deployment_id, instance in deployed_strategies.items():
            if instance.state != StrategyState.RUNNING:
                continue
            
            try:
                # Check if strategy is responsive
                status = await self.execution_engine.get_strategy_status(deployment_id)
                
                # Check for error conditions
                if status.get("last_error"):
                    await self._handle_strategy_error(deployment_id, status["last_error"])
                
                # Check for stale data
                last_updated = datetime.fromisoformat(status["performance_metrics"]["last_updated"])
                if (datetime.now() - last_updated) > timedelta(minutes=10):
                    self.logger.warning(f"Strategy {deployment_id} has stale data")
                
            except Exception as e:
                self.logger.warning(f"Health check failed for strategy {deployment_id}: {e}")
                await self._handle_strategy_error(deployment_id, str(e))
    
    async def _handle_strategy_error(self, deployment_id: str, error_message: str):
        """Handle strategy error"""
        error_entry = {
            "deployment_id": deployment_id,
            "error_message": error_message,
            "timestamp": datetime.now().isoformat(),
            "handled": False
        }
        
        self.error_log.append(error_entry)
        
        # Notify error callbacks
        for callback in self.error_callbacks:
            try:
                await callback(error_entry)
            except Exception as e:
                self.logger.error(f"Error in error callback: {e}")
        
        # Auto-recovery if enabled
        if self.auto_recovery_enabled:
            await self._attempt_recovery(deployment_id, error_message)
    
    async def _attempt_recovery(self, deployment_id: str, error_message: str):
        """Attempt to recover a failed strategy"""
        if deployment_id not in self.strategy_recovery_attempts:
            self.strategy_recovery_attempts[deployment_id] = 0
        
        self.strategy_recovery_attempts[deployment_id] += 1
        
        if self.strategy_recovery_attempts[deployment_id] > self.max_recovery_attempts:
            self.logger.error(f"Max recovery attempts reached for strategy {deployment_id}")
            return
        
        try:
            self.logger.info(f"Attempting recovery for strategy {deployment_id} (attempt {self.strategy_recovery_attempts[deployment_id]})")
            
            # Stop and restart strategy
            await self.execution_engine.control_strategy(deployment_id, "stop")
            await asyncio.sleep(5)  # Wait a bit
            result = await self.execution_engine.control_strategy(deployment_id, "start")
            
            if result.get("status") == "success":
                self.logger.info(f"Successfully recovered strategy {deployment_id}")
                self.strategy_recovery_attempts[deployment_id] = 0  # Reset counter
            else:
                self.logger.warning(f"Recovery failed for strategy {deployment_id}: {result.get('message')}")
                
        except Exception as e:
            self.logger.error(f"Error during recovery attempt for strategy {deployment_id}: {e}")
    
    async def _handle_monitoring_error(self, error: Exception):
        """Handle monitoring loop error"""
        self.logger.error(f"Monitoring error: {error}")
        
        # Add to error log
        error_entry = {
            "deployment_id": None,
            "error_message": f"Monitoring error: {str(error)}",
            "timestamp": datetime.now().isoformat(),
            "handled": False
        }
        
        self.error_log.append(error_entry)
        
        # If too many monitoring errors, consider stopping
        recent_monitoring_errors = [
            e for e in self.error_log
            if e.get("deployment_id") is None and 
            datetime.fromisoformat(e["timestamp"]) > (datetime.now() - timedelta(minutes=5))
        ]
        
        if len(recent_monitoring_errors) > 10:
            self.logger.critical("Too many monitoring errors - stopping runtime manager")
            self.status = RuntimeStatus.ERROR
    
    def _cleanup_old_data(self):
        """Clean up old metrics and error data"""
        # Keep only last 100 error log entries
        if len(self.error_log) > 100:
            self.error_log = self.error_log[-100:]
    
    # Public API methods
    
    def get_strategy_metrics(self, deployment_id: str, limit: int = 100) -> List[Dict[str, Any]]:
        """Get strategy metrics history"""
        if deployment_id not in self.metrics_history:
            return []
        
        metrics = self.metrics_history[deployment_id][-limit:]
        return [asdict(m) for m in metrics]
    
    def get_system_metrics(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get system metrics history"""
        metrics = self.system_metrics_history[-limit:]
        return [asdict(m) for m in metrics]
    
    def get_alert_rules(self) -> List[Dict[str, Any]]:
        """Get all alert rules"""
        return [asdict(rule) for rule in self.alert_rules.values()]
    
    def add_alert_rule(self, rule: AlertRule) -> bool:
        """Add a new alert rule"""
        if rule.id in self.alert_rules:
            return False
        
        self.alert_rules[rule.id] = rule
        return True
    
    def update_alert_rule(self, rule_id: str, updates: Dict[str, Any]) -> bool:
        """Update an existing alert rule"""
        if rule_id not in self.alert_rules:
            return False
        
        rule = self.alert_rules[rule_id]
        for key, value in updates.items():
            if hasattr(rule, key):
                setattr(rule, key, value)
        
        return True
    
    def remove_alert_rule(self, rule_id: str) -> bool:
        """Remove an alert rule"""
        if rule_id in self.alert_rules:
            del self.alert_rules[rule_id]
            return True
        return False
    
    def add_alert_callback(self, callback: Callable):
        """Add alert callback"""
        self.alert_callbacks.append(callback)
    
    def add_error_callback(self, callback: Callable):
        """Add error callback"""
        self.error_callbacks.append(callback)
    
    def get_runtime_status(self) -> Dict[str, Any]:
        """Get runtime manager status"""
        uptime = None
        if self.start_time:
            uptime = (datetime.now() - self.start_time).total_seconds()
        
        return {
            "status": self.status.value,
            "uptime_seconds": uptime,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "monitoring_interval": self.monitoring_interval,
            "strategies_monitored": len(self.metrics_history),
            "error_count": len(self.error_log),
            "alert_rules_active": sum(1 for rule in self.alert_rules.values() if rule.enabled)
        }


# Global runtime manager instance
_runtime_manager: Optional[StrategyRuntimeManager] = None

def get_strategy_runtime_manager(execution_engine: Optional[StrategyExecutionEngine] = None) -> StrategyRuntimeManager:
    """Get or create the strategy runtime manager singleton"""
    global _runtime_manager
    
    if _runtime_manager is None:
        if execution_engine is None:
            from strategy_execution_engine import get_strategy_execution_engine
            execution_engine = get_strategy_execution_engine()
        
        _runtime_manager = StrategyRuntimeManager(execution_engine)
    
    return _runtime_manager

async def initialize_runtime_manager() -> bool:
    """Initialize the strategy runtime manager"""
    manager = get_strategy_runtime_manager()
    return await manager.start()

async def shutdown_runtime_manager():
    """Shutdown the strategy runtime manager"""
    global _runtime_manager
    if _runtime_manager:
        await _runtime_manager.stop()
        _runtime_manager = None