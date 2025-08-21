"""
NautilusTrader Strategy Execution Engine Integration
Handles strategy deployment, execution, and lifecycle management using NautilusTrader framework.
"""

import asyncio
import logging
import json
import traceback
from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from enum import Enum
from decimal import Decimal

# NautilusTrader imports
from nautilus_trader.trading.strategy import Strategy
from nautilus_trader.live.node import TradingNode
from nautilus_trader.live.node_builder import TradingNodeBuilder
from nautilus_trader.config import LiveExecEngineConfig, LiveDataEngineConfig, LiveRiskEngineConfig
from nautilus_trader.config import LoggingConfig, TradingNodeConfig
from nautilus_trader.adapters.interactive_brokers.config import InteractiveBrokersDataClientConfig, InteractiveBrokersExecClientConfig
from nautilus_trader.common.component import Clock
from nautilus_trader.model.identifiers import TraderId, StrategyId, AccountId, InstrumentId
from nautilus_trader.model.enums import OrderSide, OrderType, TimeInForce
from nautilus_trader.model.objects import Price, Quantity
from nautilus_trader.core.uuid import UUID4


class StrategyState(Enum):
    """Strategy execution states"""
    INITIALIZING = "initializing"
    RUNNING = "running"
    PAUSED = "paused"
    STOPPING = "stopping"
    STOPPED = "stopped"
    ERROR = "error"
    COMPLETED = "completed"


@dataclass
class StrategyDeploymentConfig:
    """Strategy deployment configuration"""
    strategy_id: str
    name: str
    strategy_class: str
    parameters: Dict[str, Any]
    risk_settings: Dict[str, Any]
    deployment_mode: str  # 'live', 'paper', 'backtest'
    auto_start: bool = True
    risk_check: bool = True


@dataclass
class StrategyInstance:
    """Running strategy instance"""
    id: str
    config_id: str
    nautilus_strategy_id: str
    deployment_id: str
    state: StrategyState
    strategy_class: str
    parameters: Dict[str, Any]
    performance_metrics: Dict[str, Any]
    runtime_info: Dict[str, Any]
    error_log: List[Dict[str, Any]]
    started_at: datetime
    stopped_at: Optional[datetime] = None


@dataclass
class PerformanceMetrics:
    """Strategy performance metrics"""
    total_pnl: Decimal
    unrealized_pnl: Decimal
    total_trades: int
    winning_trades: int
    win_rate: float
    max_drawdown: Decimal
    sharpe_ratio: Optional[float]
    last_updated: datetime


class StrategyExecutionEngine:
    """
    NautilusTrader Strategy Execution Engine
    
    Manages strategy deployment, execution, and lifecycle using the NautilusTrader framework.
    Provides integration between web dashboard and NautilusTrader's trading engine.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # NautilusTrader components
        self.trading_node: Optional[TradingNode] = None
        self.node_config: Optional[TradingNodeConfig] = None
        
        # Strategy management
        self.deployed_strategies: Dict[str, StrategyInstance] = {}
        self.strategy_classes: Dict[str, type] = {}
        
        # Callbacks
        self.state_change_callbacks: List[Callable] = []
        self.performance_callbacks: List[Callable] = []
        
        # Configuration
        self.trader_id = TraderId("WEB-TRADER")
        self.account_id = AccountId("IB-001")
        
        # Initialize built-in strategy classes
        self._register_builtin_strategies()
        
    def _register_builtin_strategies(self):
        """Register built-in strategy classes"""
        # Import and register common strategy templates
        try:
            from nautilus_trader.examples.strategies.ema_cross import EMACross
            from nautilus_trader.examples.strategies.volatility_surface import VolatilitySurface
            
            self.strategy_classes["EMACross"] = EMACross
            self.strategy_classes["VolatilitySurface"] = VolatilitySurface
            
            # Register our custom strategies
            self.strategy_classes["MovingAverageCross"] = self._create_ma_cross_strategy()
            self.strategy_classes["MeanReversion"] = self._create_mean_reversion_strategy()
            self.strategy_classes["TrendFollowing"] = self._create_trend_following_strategy()
            
        except ImportError as e:
            self.logger.warning(f"Could not import built-in strategies: {e}")
    
    def _create_ma_cross_strategy(self) -> type:
        """Create Moving Average Cross strategy class"""
        class MovingAverageCross(Strategy):
            def __init__(self, config):
                super().__init__(config)
                self.fast_period = config.get('fast_period', 10)
                self.slow_period = config.get('slow_period', 20)
                self.instrument_id = InstrumentId.from_str(config.get('instrument_id', 'EUR/USD.SIM'))
                
            def on_start(self):
                self.log.info(f"Starting MA Cross strategy: {self.fast_period}/{self.slow_period}")
                
            def on_stop(self):
                self.log.info("Stopping MA Cross strategy")
                
            def on_data(self, data):
                # Implement MA cross logic here
                pass
                
        return MovingAverageCross
    
    def _create_mean_reversion_strategy(self) -> type:
        """Create Mean Reversion strategy class"""
        class MeanReversion(Strategy):
            def __init__(self, config):
                super().__init__(config)
                self.lookback_period = config.get('lookback_period', 20)
                self.z_score_threshold = config.get('z_score_threshold', 2.0)
                self.instrument_id = InstrumentId.from_str(config.get('instrument_id', 'EUR/USD.SIM'))
                
            def on_start(self):
                self.log.info(f"Starting Mean Reversion strategy")
                
            def on_stop(self):
                self.log.info("Stopping Mean Reversion strategy")
                
            def on_data(self, data):
                # Implement mean reversion logic here
                pass
                
        return MeanReversion
    
    def _create_trend_following_strategy(self) -> type:
        """Create Trend Following strategy class"""
        class TrendFollowing(Strategy):
            def __init__(self, config):
                super().__init__(config)
                self.trend_period = config.get('trend_period', 50)
                self.momentum_period = config.get('momentum_period', 14)
                self.instrument_id = InstrumentId.from_str(config.get('instrument_id', 'EUR/USD.SIM'))
                
            def on_start(self):
                self.log.info(f"Starting Trend Following strategy")
                
            def on_stop(self):
                self.log.info("Stopping Trend Following strategy")
                
            def on_data(self, data):
                # Implement trend following logic here
                pass
                
        return TrendFollowing

    async def initialize_trading_node(self) -> bool:
        """Initialize NautilusTrader trading node"""
        try:
            self.logger.info("Initializing NautilusTrader trading node...")
            
            # Create trading node configuration
            self.node_config = TradingNodeConfig(
                trader_id=self.trader_id,
                logging=LoggingConfig(
                    log_level="INFO",
                    log_component_levels={
                        "Portfolio": "DEBUG",
                        "RiskEngine": "INFO",
                        "ExecEngine": "INFO",
                        "DataEngine": "INFO"
                    }
                ),
                data_engine=LiveDataEngineConfig(),
                risk_engine=LiveRiskEngineConfig(),
                exec_engine=LiveExecEngineConfig(),
                streaming=None  # No streaming config for now
            )
            
            # Build trading node
            builder = TradingNodeBuilder()
            builder.set_config(self.node_config)
            
            # Add IB adapter configuration
            ib_data_config = InteractiveBrokersDataClientConfig(
                ibg_host="127.0.0.1",
                ibg_port=4002,
                ibg_client_id=100,  # Different from web adapter
                account_id="DU12345",
                trading_mode="paper"
            )
            
            ib_exec_config = InteractiveBrokersExecClientConfig(
                ibg_host="127.0.0.1",
                ibg_port=4002,
                ibg_client_id=101,  # Different client ID
                account_id="DU12345",
                trading_mode="paper"
            )
            
            builder.add_data_client("IB", ib_data_config)
            builder.add_exec_client("IB", ib_exec_config)
            
            # Build the node
            self.trading_node = builder.build()
            
            self.logger.info("NautilusTrader trading node initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize trading node: {e}")
            self.logger.error(traceback.format_exc())
            return False
    
    async def start_trading_node(self) -> bool:
        """Start the trading node"""
        try:
            if not self.trading_node:
                if not await self.initialize_trading_node():
                    return False
            
            self.logger.info("Starting NautilusTrader trading node...")
            await self.trading_node.start()
            
            self.logger.info("NautilusTrader trading node started successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to start trading node: {e}")
            return False
    
    async def stop_trading_node(self):
        """Stop the trading node"""
        try:
            if self.trading_node:
                self.logger.info("Stopping NautilusTrader trading node...")
                await self.trading_node.stop()
                self.logger.info("NautilusTrader trading node stopped")
        except Exception as e:
            self.logger.error(f"Error stopping trading node: {e}")
    
    async def deploy_strategy(self, deployment_config: StrategyDeploymentConfig) -> Dict[str, Any]:
        """Deploy a strategy to the execution engine"""
        try:
            self.logger.info(f"Deploying strategy: {deployment_config.name}")
            
            # Ensure trading node is running
            if not self.trading_node:
                if not await self.start_trading_node():
                    raise RuntimeError("Failed to start trading node")
            
            # Validate strategy class
            if deployment_config.strategy_class not in self.strategy_classes:
                raise ValueError(f"Unknown strategy class: {deployment_config.strategy_class}")
            
            # Create strategy instance
            strategy_class = self.strategy_classes[deployment_config.strategy_class]
            
            # Generate unique strategy ID
            nautilus_strategy_id = StrategyId(f"{deployment_config.strategy_class}-{UUID4()}")
            deployment_id = str(UUID4())
            
            # Create strategy configuration for NautilusTrader
            strategy_config = self._create_nautilus_strategy_config(
                deployment_config, nautilus_strategy_id
            )
            
            # Create and add strategy to trading node
            strategy_instance = strategy_class(strategy_config)
            self.trading_node.add_strategy(strategy_instance)
            
            # Create our tracking instance
            instance = StrategyInstance(
                id=deployment_id,
                config_id=deployment_config.strategy_id,
                nautilus_strategy_id=str(nautilus_strategy_id),
                deployment_id=deployment_id,
                state=StrategyState.INITIALIZING,
                strategy_class=deployment_config.strategy_class,
                parameters=deployment_config.parameters.copy(),
                performance_metrics=self._create_initial_metrics(),
                runtime_info=self._create_initial_runtime_info(),
                error_log=[],
                started_at=datetime.now()
            )
            
            # Store deployed strategy
            self.deployed_strategies[deployment_id] = instance
            
            # Auto-start if requested
            if deployment_config.auto_start:
                await self._start_strategy_instance(deployment_id)
            
            self.logger.info(f"Strategy deployed successfully: {deployment_id}")
            
            return {
                "deployment_id": deployment_id,
                "status": "deployed",
                "nautilus_strategy_id": str(nautilus_strategy_id),
                "message": f"Strategy '{deployment_config.name}' deployed successfully"
            }
            
        except Exception as e:
            self.logger.error(f"Failed to deploy strategy: {e}")
            self.logger.error(traceback.format_exc())
            return {
                "deployment_id": None,
                "status": "failed",
                "error_message": str(e)
            }
    
    def _create_nautilus_strategy_config(self, deployment_config: StrategyDeploymentConfig, strategy_id: StrategyId) -> Dict[str, Any]:
        """Create NautilusTrader strategy configuration"""
        config = {
            "strategy_id": strategy_id,
            "instrument_id": deployment_config.parameters.get("instrument_id", "EUR/USD.SIM"),
            "order_id_tag": f"WEB-{deployment_config.strategy_id[:8]}",
            **deployment_config.parameters,
            **deployment_config.risk_settings
        }
        
        return config
    
    def _create_initial_metrics(self) -> Dict[str, Any]:
        """Create initial performance metrics"""
        return {
            "total_pnl": 0.0,
            "unrealized_pnl": 0.0,
            "total_trades": 0,
            "winning_trades": 0,
            "win_rate": 0.0,
            "max_drawdown": 0.0,
            "sharpe_ratio": None,
            "last_updated": datetime.now().isoformat()
        }
    
    def _create_initial_runtime_info(self) -> Dict[str, Any]:
        """Create initial runtime information"""
        return {
            "orders_placed": 0,
            "positions_opened": 0,
            "last_signal_time": None,
            "cpu_usage": 0.0,
            "memory_usage": 0.0,
            "uptime_seconds": 0
        }
    
    async def _start_strategy_instance(self, deployment_id: str):
        """Start a deployed strategy instance"""
        if deployment_id not in self.deployed_strategies:
            raise ValueError(f"Strategy not found: {deployment_id}")
        
        instance = self.deployed_strategies[deployment_id]
        
        try:
            # Start strategy in NautilusTrader
            strategy_id = StrategyId(instance.nautilus_strategy_id)
            strategy = self.trading_node.get_strategy(strategy_id)
            
            if strategy:
                strategy.start()
                instance.state = StrategyState.RUNNING
                self.logger.info(f"Strategy started: {deployment_id}")
            else:
                raise RuntimeError(f"Strategy not found in trading node: {strategy_id}")
                
        except Exception as e:
            instance.state = StrategyState.ERROR
            instance.error_log.append({
                "timestamp": datetime.now().isoformat(),
                "level": "error",
                "message": f"Failed to start strategy: {str(e)}"
            })
            raise
    
    async def control_strategy(self, deployment_id: str, action: str, force: bool = False) -> Dict[str, Any]:
        """Control strategy execution (start, stop, pause, resume)"""
        try:
            if deployment_id not in self.deployed_strategies:
                raise ValueError(f"Strategy not found: {deployment_id}")
            
            instance = self.deployed_strategies[deployment_id]
            strategy_id = StrategyId(instance.nautilus_strategy_id)
            strategy = self.trading_node.get_strategy(strategy_id) if self.trading_node else None
            
            if not strategy:
                raise RuntimeError(f"Strategy not found in trading node: {strategy_id}")
            
            if action == "start":
                if instance.state in [StrategyState.STOPPED, StrategyState.PAUSED, StrategyState.ERROR]:
                    strategy.start()
                    instance.state = StrategyState.RUNNING
                    message = f"Strategy started: {deployment_id}"
                else:
                    raise ValueError(f"Cannot start strategy in state: {instance.state}")
                    
            elif action == "stop":
                if instance.state in [StrategyState.RUNNING, StrategyState.PAUSED]:
                    strategy.stop()
                    instance.state = StrategyState.STOPPED
                    instance.stopped_at = datetime.now()
                    message = f"Strategy stopped: {deployment_id}"
                else:
                    raise ValueError(f"Cannot stop strategy in state: {instance.state}")
                    
            elif action == "pause":
                if instance.state == StrategyState.RUNNING:
                    # NautilusTrader doesn't have native pause, so we stop it
                    strategy.stop()
                    instance.state = StrategyState.PAUSED
                    message = f"Strategy paused: {deployment_id}"
                else:
                    raise ValueError(f"Cannot pause strategy in state: {instance.state}")
                    
            elif action == "resume":
                if instance.state == StrategyState.PAUSED:
                    strategy.start()
                    instance.state = StrategyState.RUNNING
                    message = f"Strategy resumed: {deployment_id}"
                else:
                    raise ValueError(f"Cannot resume strategy in state: {instance.state}")
            else:
                raise ValueError(f"Unknown action: {action}")
            
            self.logger.info(message)
            
            # Notify callbacks
            for callback in self.state_change_callbacks:
                try:
                    await callback(deployment_id, instance.state, action)
                except Exception as e:
                    self.logger.error(f"Error in state change callback: {e}")
            
            return {
                "status": "success",
                "new_state": instance.state.value,
                "message": message
            }
            
        except Exception as e:
            self.logger.error(f"Failed to {action} strategy {deployment_id}: {e}")
            return {
                "status": "error",
                "message": str(e)
            }
    
    async def get_strategy_status(self, deployment_id: str) -> Dict[str, Any]:
        """Get strategy status and performance metrics"""
        if deployment_id not in self.deployed_strategies:
            raise ValueError(f"Strategy not found: {deployment_id}")
        
        instance = self.deployed_strategies[deployment_id]
        
        # Update runtime metrics
        instance.runtime_info["uptime_seconds"] = (
            datetime.now() - instance.started_at
        ).total_seconds()
        
        # Get performance metrics from NautilusTrader if available
        if self.trading_node:
            try:
                strategy_id = StrategyId(instance.nautilus_strategy_id)
                portfolio = self.trading_node.portfolio
                
                # Update performance metrics from portfolio
                if portfolio:
                    instance.performance_metrics.update({
                        "total_pnl": float(portfolio.net_liquidation_value() or 0),
                        "unrealized_pnl": float(portfolio.unrealized_pnl() or 0),
                        "last_updated": datetime.now().isoformat()
                    })
                    
            except Exception as e:
                self.logger.warning(f"Could not update performance metrics: {e}")
        
        return {
            "strategy_id": deployment_id,
            "state": instance.state.value,
            "performance_metrics": instance.performance_metrics,
            "runtime_info": instance.runtime_info,
            "last_error": instance.error_log[-1]["message"] if instance.error_log else None
        }
    
    def get_deployed_strategies(self) -> Dict[str, StrategyInstance]:
        """Get all deployed strategies"""
        return self.deployed_strategies.copy()
    
    def get_strategy_classes(self) -> Dict[str, str]:
        """Get available strategy classes"""
        return {name: cls.__doc__ or f"{name} strategy" for name, cls in self.strategy_classes.items()}
    
    async def remove_strategy(self, deployment_id: str) -> bool:
        """Remove a deployed strategy"""
        try:
            if deployment_id not in self.deployed_strategies:
                return False
            
            instance = self.deployed_strategies[deployment_id]
            
            # Stop strategy if running
            if instance.state in [StrategyState.RUNNING, StrategyState.PAUSED]:
                await self.control_strategy(deployment_id, "stop")
            
            # Remove from trading node
            if self.trading_node:
                strategy_id = StrategyId(instance.nautilus_strategy_id)
                # NautilusTrader doesn't have direct remove method, so we just stop it
                
            # Remove from our tracking
            del self.deployed_strategies[deployment_id]
            
            self.logger.info(f"Strategy removed: {deployment_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to remove strategy {deployment_id}: {e}")
            return False
    
    def add_state_change_callback(self, callback: Callable):
        """Add callback for strategy state changes"""
        self.state_change_callbacks.append(callback)
    
    def add_performance_callback(self, callback: Callable):
        """Add callback for performance updates"""
        self.performance_callbacks.append(callback)
    
    async def shutdown(self):
        """Shutdown the execution engine"""
        try:
            self.logger.info("Shutting down strategy execution engine...")
            
            # Stop all running strategies
            for deployment_id, instance in list(self.deployed_strategies.items()):
                if instance.state in [StrategyState.RUNNING, StrategyState.PAUSED]:
                    await self.control_strategy(deployment_id, "stop")
            
            # Stop trading node
            await self.stop_trading_node()
            
            self.logger.info("Strategy execution engine shutdown complete")
            
        except Exception as e:
            self.logger.error(f"Error during shutdown: {e}")


# Global execution engine instance
_execution_engine: Optional[StrategyExecutionEngine] = None

def get_strategy_execution_engine() -> StrategyExecutionEngine:
    """Get or create the strategy execution engine singleton"""
    global _execution_engine
    
    if _execution_engine is None:
        _execution_engine = StrategyExecutionEngine()
    
    return _execution_engine

async def initialize_execution_engine() -> bool:
    """Initialize the strategy execution engine"""
    engine = get_strategy_execution_engine()
    return await engine.initialize_trading_node()

async def shutdown_execution_engine():
    """Shutdown the strategy execution engine"""
    global _execution_engine
    if _execution_engine:
        await _execution_engine.shutdown()
        _execution_engine = None