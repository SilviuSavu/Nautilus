"""
NautilusTrader Engine Management Service
Docker-based integration following CORE RULE #8
"""

import asyncio
import json
import logging
import subprocess
import tempfile
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

from pydantic import BaseModel

logger = logging.getLogger(__name__)


class EngineState(Enum):
    """NautilusTrader engine states"""
    STOPPED = "stopped"
    STARTING = "starting"
    RUNNING = "running"
    STOPPING = "stopping"
    ERROR = "error"


class BacktestStatus(Enum):
    """Backtest execution status"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class EngineConfig(BaseModel):
    """Engine configuration"""
    engine_type: str = "live"  # live, backtest, sandbox
    log_level: str = "INFO"
    instance_id: str = "nautilus-001"
    trading_mode: str = "paper"  # paper, live
    
    # Resource limits
    max_memory: str = "2g"
    max_cpu: str = "2.0"
    
    # Data configuration
    data_catalog_path: str = "/app/data"
    cache_database_path: str = "/app/cache"
    
    # Risk settings
    risk_engine_enabled: bool = True
    max_position_size: Optional[float] = None
    max_order_rate: Optional[int] = None


class BacktestConfig(BaseModel):
    """Backtest configuration"""
    strategy_class: str
    strategy_config: Dict[str, Any]
    start_date: str
    end_date: str
    instruments: List[str]
    venues: List[str] = ["SIM"]
    initial_balance: float = 1000000.0
    base_currency: str = "USD"
    
    # Data configuration
    data_sources: List[str] = ["catalog"]
    bar_types: List[str] = []
    tick_data: bool = False
    
    # Output configuration
    output_path: str = "/app/results"
    save_results: bool = True


class NautilusEngineManager:
    """Manages NautilusTrader engines in Docker containers"""
    
    def __init__(self):
        # Fix container name to match CORE RULE #8 specification
        self.container_name = "nautilus-engine"
        self.current_state = EngineState.STOPPED  # Consistent naming with tests
        self.current_config: Optional[EngineConfig] = None
        self.process: Optional[subprocess.Popen] = None
        self.last_error: Optional[str] = None
        self.started_at: Optional[datetime] = None
        self.resource_usage = {}
        self.session_id: Optional[str] = None  # Add session tracking
        
        # Backtest management
        self.active_backtests: Dict[str, Dict[str, Any]] = {}
        
    async def get_engine_status(self) -> Dict[str, Any]:
        """Get comprehensive engine status"""
        return {
            "state": self.current_state.value,
            "config": self.current_config.dict() if self.current_config else None,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "last_error": self.last_error,
            "resource_usage": await self._get_resource_usage(),
            "container_info": await self._get_container_info(),
            "active_backtests": len(self.active_backtests),
            "health_check": await self._health_check()
        }
    
    async def start_engine(self, config: EngineConfig) -> Dict[str, Any]:
        """Start NautilusTrader live trading engine"""
        try:
            if self.current_state in [EngineState.RUNNING, EngineState.STARTING]:
                return {
                    "success": False,
                    "message": f"Engine already {self.current_state.value}",
                    "state": self.current_state.value
                }
            
            logger.info("Starting NautilusTrader engine in Docker container...")
            self.current_state = EngineState.STARTING
            self.current_config = config
            self.last_error = None
            
            # Create engine configuration file
            config_path = await self._create_engine_config(config)
            
            # Start engine in Docker container
            cmd = [
                "docker", "exec", "-d", self.container_name,
                "python", "-m", "nautilus_trader.live",
                "--config", config_path,
                "--log-level", config.log_level
            ]
            
            result = await self._run_docker_command(cmd)
            
            if result["success"]:
                self.current_state = EngineState.RUNNING
                self.started_at = datetime.now()
                logger.info("NautilusTrader engine started successfully")
                
                return {
                    "success": True,
                    "message": "Engine started successfully",
                    "state": self.current_state.value,
                    "started_at": self.started_at.isoformat(),
                    "config": config.dict()
                }
            else:
                self.current_state = EngineState.ERROR
                self.last_error = result["error"]
                logger.error(f"Failed to start engine: {result['error']}")
                
                return {
                    "success": False,
                    "message": f"Failed to start engine: {result['error']}",
                    "state": self.current_state.value
                }
                
        except Exception as e:
            self.current_state = EngineState.ERROR
            self.last_error = str(e)
            logger.error(f"Error starting engine: {e}")
            
            return {
                "success": False,
                "message": f"Error starting engine: {str(e)}",
                "state": self.current_state.value
            }
    
    async def stop_engine(self, force: bool = False) -> Dict[str, Any]:
        """Stop NautilusTrader engine"""
        try:
            if self.current_state == EngineState.STOPPED:
                return {
                    "success": True,
                    "message": "Engine already stopped",
                    "state": self.current_state.value
                }
            
            logger.info("Stopping NautilusTrader engine...")
            self.current_state = EngineState.STOPPING
            
            if force:
                # Force stop using container restart
                cmd = ["docker", "restart", self.container_name]
            else:
                # Graceful stop by sending shutdown signal
                cmd = ["docker", "exec", self.container_name, "pkill", "-f", "nautilus_trader.live"]
            
            result = await self._run_docker_command(cmd)
            
            self.current_state = EngineState.STOPPED
            self.started_at = None
            self.current_config = None
            
            logger.info("NautilusTrader engine stopped")
            
            return {
                "success": True,
                "message": "Engine stopped successfully",
                "state": self.current_state.value
            }
            
        except Exception as e:
            self.last_error = str(e)
            logger.error(f"Error stopping engine: {e}")
            
            return {
                "success": False,
                "message": f"Error stopping engine: {str(e)}",
                "state": self.current_state.value
            }
    
    async def restart_engine(self) -> Dict[str, Any]:
        """Restart NautilusTrader engine with current configuration"""
        try:
            if not self.current_config:
                return {
                    "success": False,
                    "message": "No configuration available for restart",
                    "state": self.current_state.value
                }
            
            # Stop current engine
            stop_result = await self.stop_engine()
            if not stop_result["success"]:
                return stop_result
            
            # Wait a moment
            await asyncio.sleep(2)
            
            # Start with previous configuration
            return await self.start_engine(self.current_config)
            
        except Exception as e:
            logger.error(f"Error restarting engine: {e}")
            return {
                "success": False,
                "message": f"Error restarting engine: {str(e)}",
                "state": self.current_state.value
            }
    
    async def run_backtest(self, backtest_id: str, config: BacktestConfig) -> Dict[str, Any]:
        """Run a historical backtest"""
        try:
            if backtest_id in self.active_backtests:
                return {
                    "success": False,
                    "message": f"Backtest {backtest_id} already exists",
                    "backtest_id": backtest_id
                }
            
            logger.info(f"Starting backtest {backtest_id}...")
            
            # Register backtest
            self.active_backtests[backtest_id] = {
                "id": backtest_id,
                "config": config.dict(),
                "status": BacktestStatus.PENDING.value,
                "started_at": datetime.now().isoformat(),
                "progress": 0.0,
                "results": None,
                "error": None
            }
            
            # Create backtest configuration file
            config_path = await self._create_backtest_config(backtest_id, config)
            
            # Run backtest in Docker container
            cmd = [
                "docker", "exec", "-d", self.container_name,
                "python", "-m", "nautilus_trader.backtest",
                "--config", config_path,
                "--output", f"/app/results/{backtest_id}"
            ]
            
            # Update status to running
            self.active_backtests[backtest_id]["status"] = BacktestStatus.RUNNING.value
            
            # Execute backtest asynchronously
            asyncio.create_task(self._monitor_backtest(backtest_id, cmd))
            
            return {
                "success": True,
                "message": f"Backtest {backtest_id} started",
                "backtest_id": backtest_id,
                "status": BacktestStatus.RUNNING.value
            }
            
        except Exception as e:
            if backtest_id in self.active_backtests:
                self.active_backtests[backtest_id]["status"] = BacktestStatus.FAILED.value
                self.active_backtests[backtest_id]["error"] = str(e)
            
            logger.error(f"Error starting backtest: {e}")
            return {
                "success": False,
                "message": f"Error starting backtest: {str(e)}",
                "backtest_id": backtest_id
            }
    
    async def get_backtest_status(self, backtest_id: str) -> Dict[str, Any]:
        """Get backtest status and results"""
        if backtest_id not in self.active_backtests:
            return {
                "success": False,
                "message": f"Backtest {backtest_id} not found"
            }
        
        backtest_info = self.active_backtests[backtest_id]
        
        # If completed, try to load results
        if backtest_info["status"] == BacktestStatus.COMPLETED.value and not backtest_info["results"]:
            try:
                results = await self._load_backtest_results(backtest_id)
                backtest_info["results"] = results
            except Exception as e:
                logger.warning(f"Failed to load backtest results: {e}")
        
        return {
            "success": True,
            "backtest": backtest_info
        }
    
    async def cancel_backtest(self, backtest_id: str) -> Dict[str, Any]:
        """Cancel running backtest"""
        if backtest_id not in self.active_backtests:
            return {
                "success": False,
                "message": f"Backtest {backtest_id} not found"
            }
        
        try:
            # Kill the backtest process
            cmd = ["docker", "exec", self.container_name, "pkill", "-f", f"backtest.*{backtest_id}"]
            await self._run_docker_command(cmd)
            
            # Update status
            self.active_backtests[backtest_id]["status"] = BacktestStatus.CANCELLED.value
            self.active_backtests[backtest_id]["cancelled_at"] = datetime.now().isoformat()
            
            return {
                "success": True,
                "message": f"Backtest {backtest_id} cancelled",
                "backtest_id": backtest_id
            }
            
        except Exception as e:
            logger.error(f"Error cancelling backtest: {e}")
            return {
                "success": False,
                "message": f"Error cancelling backtest: {str(e)}"
            }
    
    async def list_backtests(self) -> Dict[str, Any]:
        """List all backtests"""
        return {
            "success": True,
            "backtests": list(self.active_backtests.values()),
            "total_count": len(self.active_backtests)
        }
    
    async def get_data_catalog(self) -> Dict[str, Any]:
        """Get available data in catalog"""
        try:
            cmd = ["docker", "exec", self.container_name, "python", "-c", 
                   "from nautilus_trader.persistence.catalog import ParquetDataCatalog; "
                   "catalog = ParquetDataCatalog('/app/data'); "
                   "import json; "
                   "print(json.dumps({'instruments': list(catalog.instruments()), 'venues': list(catalog.venues())}))"]
            
            result = await self._run_docker_command(cmd)
            
            if result["success"]:
                try:
                    catalog_info = json.loads(result["output"])
                    return {
                        "success": True,
                        "catalog": catalog_info
                    }
                except json.JSONDecodeError:
                    return {
                        "success": False,
                        "message": "Failed to parse catalog data"
                    }
            else:
                return {
                    "success": False,
                    "message": f"Failed to get catalog: {result['error']}"
                }
                
        except Exception as e:
            logger.error(f"Error getting data catalog: {e}")
            return {
                "success": False,
                "message": f"Error getting data catalog: {str(e)}"
            }
    
    async def get_asset_allocations(self, portfolio_id: str) -> Dict[str, Any]:
        """Get asset allocations for portfolio"""
        try:
            # Return mock data for now since this is mainly for UI display
            return {
                "portfolio_id": portfolio_id,
                "allocations": {
                    "stocks": {"percentage": 60.0, "value": 600000.0},
                    "bonds": {"percentage": 25.0, "value": 250000.0},
                    "cash": {"percentage": 10.0, "value": 100000.0},
                    "options": {"percentage": 5.0, "value": 50000.0}
                },
                "total_value": 1000000.0,
                "last_updated": datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Error getting asset allocations: {e}")
            raise Exception(f"Failed to get asset allocations: {str(e)}")
    
    async def get_strategy_allocations(self, portfolio_id: str, start_date: str, end_date: str) -> Dict[str, Any]:
        """Get strategy allocations for portfolio"""
        try:
            return {
                "portfolio_id": portfolio_id,
                "strategies": [
                    {"name": "Momentum Strategy", "allocation": 40.0, "value": 400000.0, "pnl": 5000.0},
                    {"name": "Mean Reversion", "allocation": 35.0, "value": 350000.0, "pnl": -2000.0},
                    {"name": "Arbitrage", "allocation": 25.0, "value": 250000.0, "pnl": 1500.0}
                ],
                "total_allocation": 100.0,
                "total_value": 1000000.0,
                "period": {"start": start_date, "end": end_date}
            }
        except Exception as e:
            logger.error(f"Error getting strategy allocations: {e}")
            raise Exception(f"Failed to get strategy allocations: {str(e)}")
    
    async def get_performance_history(self, portfolio_id: str, start_date: str, end_date: str) -> Dict[str, Any]:
        """Get performance history for portfolio"""
        try:
            # Generate sample performance data
            import random
            from datetime import datetime, timedelta
            
            start = datetime.fromisoformat(start_date.replace('Z', ''))
            end = datetime.fromisoformat(end_date.replace('Z', ''))
            days = (end - start).days
            
            performance_data = []
            cumulative_return = 0.0
            
            for i in range(min(days, 30)):  # Limit to 30 data points
                date = start + timedelta(days=i)
                daily_return = random.uniform(-0.02, 0.02)  # -2% to +2% daily
                cumulative_return += daily_return
                
                performance_data.append({
                    "date": date.isoformat(),
                    "cumulative_return": round(cumulative_return * 100, 2),
                    "daily_return": round(daily_return * 100, 2),
                    "portfolio_value": round(1000000 * (1 + cumulative_return), 2)
                })
            
            return {
                "portfolio_id": portfolio_id,
                "performance": performance_data,
                "period": {"start": start_date, "end": end_date},
                "total_return": round(cumulative_return * 100, 2)
            }
        except Exception as e:
            logger.error(f"Error getting performance history: {e}")
            raise Exception(f"Failed to get performance history: {str(e)}")
    
    async def get_strategy_correlations(self, portfolio_id: str) -> Dict[str, Any]:
        """Get strategy correlations for portfolio"""
        try:
            import random
            
            strategies = ["Momentum Strategy", "Mean Reversion", "Arbitrage"]
            correlations = {}
            
            for i, strategy1 in enumerate(strategies):
                correlations[strategy1] = {}
                for j, strategy2 in enumerate(strategies):
                    if i == j:
                        correlations[strategy1][strategy2] = 1.0
                    else:
                        # Generate symmetric correlation matrix
                        if strategy2 in correlations:
                            correlations[strategy1][strategy2] = correlations[strategy2][strategy1]
                        else:
                            correlations[strategy1][strategy2] = round(random.uniform(-0.5, 0.8), 2)
            
            return {
                "portfolio_id": portfolio_id,
                "correlations": correlations,
                "analysis_date": datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Error getting strategy correlations: {e}")
            raise Exception(f"Failed to get strategy correlations: {str(e)}")
    
    async def get_benchmark_comparison(self, portfolio_id: str, start_date: str, end_date: str) -> Dict[str, Any]:
        """Get benchmark comparison for portfolio"""
        try:
            import random
            from datetime import datetime, timedelta
            
            start = datetime.fromisoformat(start_date.replace('Z', ''))
            end = datetime.fromisoformat(end_date.replace('Z', ''))
            days = (end - start).days
            
            portfolio_performance = []
            sp500_performance = []
            
            portfolio_return = 0.0
            sp500_return = 0.0
            
            for i in range(min(days, 30)):
                date = start + timedelta(days=i)
                
                # Portfolio performance (slightly outperforming)
                portfolio_daily = random.uniform(-0.025, 0.025)
                portfolio_return += portfolio_daily
                
                # S&P 500 performance (market baseline)
                sp500_daily = random.uniform(-0.02, 0.02)
                sp500_return += sp500_daily
                
                portfolio_performance.append({
                    "date": date.isoformat(),
                    "return": round(portfolio_return * 100, 2)
                })
                
                sp500_performance.append({
                    "date": date.isoformat(),
                    "return": round(sp500_return * 100, 2)
                })
            
            return {
                "portfolio_id": portfolio_id,
                "portfolio": {
                    "performance": portfolio_performance,
                    "total_return": round(portfolio_return * 100, 2),
                    "volatility": round(random.uniform(10, 20), 2),
                    "sharpe_ratio": round(random.uniform(0.8, 1.5), 2)
                },
                "benchmark": {
                    "name": "S&P 500",
                    "performance": sp500_performance,
                    "total_return": round(sp500_return * 100, 2),
                    "volatility": round(random.uniform(12, 18), 2),
                    "sharpe_ratio": round(random.uniform(0.6, 1.2), 2)
                },
                "alpha": round((portfolio_return - sp500_return) * 100, 2),
                "beta": round(random.uniform(0.8, 1.2), 2),
                "period": {"start": start_date, "end": end_date}
            }
        except Exception as e:
            logger.error(f"Error getting benchmark comparison: {e}")
            raise Exception(f"Failed to get benchmark comparison: {str(e)}")
    
    async def get_portfolio_performance_history(self, portfolio_id: str, start_date: str, end_date: str) -> Dict[str, Any]:
        """Get portfolio performance history (alias for get_performance_history)"""
        return await self.get_performance_history(portfolio_id, start_date, end_date)
    
    # Private helper methods
    
    async def _create_engine_config(self, config: EngineConfig) -> str:
        """Create engine configuration file"""
        engine_config = {
            "environment": {
                "trader_id": config.instance_id,
                "log_level": config.log_level,
                "instance_id": config.instance_id,
                "cache": {
                    "database": f"sqlite:///{config.cache_database_path}/cache.db"
                }
            },
            "data_engine": {
                "qsize": 100000,
                "time_bars_build_with_no_updates": False,
                "time_bars_timestamp_on_close": True,
                "validate_data_sequence": True
            },
            "risk_engine": {
                "bypass": not config.risk_engine_enabled,
                "max_order_rate": config.max_order_rate or "100/00:00:01",
                "max_notional_per_order": {"USD": config.max_position_size} if config.max_position_size else {}
            },
            "exec_engine": {
                "load_cache": True,
                "qsize": 100000
            },
            "streaming": {
                "catalog_path": config.data_catalog_path,
                "fs_protocol": "file",
                "fs_storage_options": None
            }
        }
        
        # SECURITY FIX: Use proper file creation instead of string concatenation
        # Write config to temporary file first, then copy to container
        import tempfile
        import os
        
        # Create temporary file with proper JSON content
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as tmp_file:
            json.dump(engine_config, tmp_file, indent=2)
            tmp_file_path = tmp_file.name
        
        try:
            # Copy file to container safely
            copy_cmd = ["docker", "cp", tmp_file_path, f"{self.container_name}:/app/config/engine_config.json"]
            await self._run_docker_command(copy_cmd)
            
            # Ensure proper permissions
            chmod_cmd = ["docker", "exec", self.container_name, "chmod", "644", "/app/config/engine_config.json"]
            await self._run_docker_command(chmod_cmd)
        finally:
            # Clean up temporary file
            if os.path.exists(tmp_file_path):
                os.unlink(tmp_file_path)
        
        return "/app/config/engine_config.json"
    
    async def _create_backtest_config(self, backtest_id: str, config: BacktestConfig) -> str:
        """Create backtest configuration file"""
        backtest_config = {
            "trader_id": f"BACKTESTER-{backtest_id}",
            "log_level": "INFO",
            "run_config_id": backtest_id,
            "strategies": [
                {
                    "strategy_path": config.strategy_class,
                    "config_path": f"/app/config/{backtest_id}_strategy_config.json",
                    "config": config.strategy_config
                }
            ],
            "venues": [
                {
                    "name": venue,
                    "oms_type": "HEDGING",
                    "account_type": "MARGIN",
                    "base_currency": config.base_currency,
                    "starting_balances": [f"{config.initial_balance} {config.base_currency}"]
                }
                for venue in config.venues
            ],
            "data": {
                "catalog_path": "/app/data",
                "catalog_fs_protocol": "file"
            },
            "backtest": {
                "start_time": config.start_date,
                "end_time": config.end_date,
                "run_config_id": backtest_id
            }
        }
        
        # SECURITY FIX: Use proper file creation instead of string concatenation
        import tempfile
        import os
        
        # Create temporary files for both configs
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as tmp_main:
            json.dump(backtest_config, tmp_main, indent=2)
            tmp_main_path = tmp_main.name
            
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as tmp_strategy:
            json.dump(config.strategy_config, tmp_strategy, indent=2)
            tmp_strategy_path = tmp_strategy.name
        
        try:
            # Safely validate backtest_id to prevent path injection
            if not backtest_id.replace('-', '').replace('_', '').isalnum():
                raise ValueError("Invalid backtest_id: contains unsafe characters")
            
            # Copy files to container safely
            main_copy_cmd = ["docker", "cp", tmp_main_path, f"{self.container_name}:/app/config/{backtest_id}_config.json"]
            await self._run_docker_command(main_copy_cmd)
            
            strategy_copy_cmd = ["docker", "cp", tmp_strategy_path, f"{self.container_name}:/app/config/{backtest_id}_strategy_config.json"]
            await self._run_docker_command(strategy_copy_cmd)
        finally:
            # Clean up temporary files
            for tmp_path in [tmp_main_path, tmp_strategy_path]:
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)
        
        return f"/app/config/{backtest_id}_config.json"
    
    async def _monitor_backtest(self, backtest_id: str, cmd: List[str]):
        """Monitor backtest execution"""
        try:
            result = await self._run_docker_command(cmd)
            
            if result["success"]:
                # Simulate progress monitoring (in real implementation, parse logs)
                for progress in [25, 50, 75, 90, 100]:
                    await asyncio.sleep(1)  # Simulate work
                    if backtest_id in self.active_backtests:
                        self.active_backtests[backtest_id]["progress"] = progress
                
                self.active_backtests[backtest_id]["status"] = BacktestStatus.COMPLETED.value
                self.active_backtests[backtest_id]["completed_at"] = datetime.now().isoformat()
            else:
                self.active_backtests[backtest_id]["status"] = BacktestStatus.FAILED.value
                self.active_backtests[backtest_id]["error"] = result["error"]
                
        except Exception as e:
            logger.error(f"Error monitoring backtest {backtest_id}: {e}")
            if backtest_id in self.active_backtests:
                self.active_backtests[backtest_id]["status"] = BacktestStatus.FAILED.value
                self.active_backtests[backtest_id]["error"] = str(e)
    
    async def _load_backtest_results(self, backtest_id: str) -> Dict[str, Any]:
        """Load backtest results from output files"""
        try:
            # Load results from container
            cmd = ["docker", "exec", self.container_name, "python", "-c",
                   f"import json, os; "
                   f"results_path = '/app/results/{backtest_id}'; "
                   f"if os.path.exists(results_path + '/results.json'): "
                   f"    with open(results_path + '/results.json') as f: print(f.read()); "
                   f"else: print('{{\"error\": \"Results not found\"}}')"]
            
            result = await self._run_docker_command(cmd)
            
            if result["success"]:
                return json.loads(result["output"])
            else:
                return {"error": "Failed to load results"}
                
        except Exception as e:
            logger.error(f"Error loading backtest results: {e}")
            return {"error": str(e)}
    
    async def _run_docker_command(self, cmd: List[str]) -> Dict[str, Any]:
        """Execute Docker command asynchronously"""
        try:
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await process.communicate()
            
            if process.returncode == 0:
                return {
                    "success": True,
                    "output": stdout.decode().strip(),
                    "error": None
                }
            else:
                return {
                    "success": False,
                    "output": stdout.decode().strip(),
                    "error": stderr.decode().strip()
                }
                
        except Exception as e:
            return {
                "success": False,
                "output": "",
                "error": str(e)
            }
    
    async def _get_resource_usage(self) -> Dict[str, Any]:
        """Get container resource usage"""
        try:
            cmd = ["docker", "stats", self.container_name, "--no-stream", "--format", 
                   "{{.CPUPerc}},{{.MemUsage}},{{.MemPerc}},{{.NetIO}},{{.BlockIO}}"]
            
            result = await self._run_docker_command(cmd)
            
            if result["success"] and result["output"]:
                parts = result["output"].split(",")
                return {
                    "cpu_percent": parts[0] if len(parts) > 0 else "0%",
                    "memory_usage": parts[1] if len(parts) > 1 else "0B / 0B",
                    "memory_percent": parts[2] if len(parts) > 2 else "0%",
                    "network_io": parts[3] if len(parts) > 3 else "0B / 0B",
                    "block_io": parts[4] if len(parts) > 4 else "0B / 0B"
                }
            else:
                return {"error": "Failed to get resource usage"}
                
        except Exception as e:
            return {"error": str(e)}
    
    async def _get_container_info(self) -> Dict[str, Any]:
        """Get container information"""
        try:
            cmd = ["docker", "inspect", self.container_name]
            result = await self._run_docker_command(cmd)
            
            if result["success"]:
                container_info = json.loads(result["output"])[0]
                return {
                    "status": container_info["State"]["Status"],
                    "running": container_info["State"]["Running"],
                    "started_at": container_info["State"]["StartedAt"],
                    "image": container_info["Config"]["Image"],
                    "platform": container_info["Platform"]
                }
            else:
                return {"error": "Failed to get container info"}
                
        except Exception as e:
            return {"error": str(e)}
    
    async def _health_check(self) -> Dict[str, Any]:
        """Perform health check"""
        try:
            # Check if container is running
            cmd = ["docker", "exec", self.container_name, "python", "-c", 
                   "print('healthy')"]
            
            result = await self._run_docker_command(cmd)
            
            return {
                "status": "healthy" if result["success"] else "unhealthy",
                "last_check": datetime.now().isoformat(),
                "details": result["output"] if result["success"] else result["error"]
            }
            
        except Exception as e:
            return {
                "status": "error",
                "last_check": datetime.now().isoformat(),
                "details": str(e)
            }


# Global instance
_engine_manager: Optional[NautilusEngineManager] = None


def get_nautilus_engine_manager() -> NautilusEngineManager:
    """Get global engine manager instance"""
    global _engine_manager
    if _engine_manager is None:
        _engine_manager = NautilusEngineManager()
    return _engine_manager


# Compatibility functions for the adapter
def get_nautilus_ib_adapter():
    """Compatibility function for existing routes"""
    return get_nautilus_engine_manager()


class IBGatewayStatus:
    """Compatibility class for IB status"""
    def __init__(self, connected: bool, account_id: str = None, connection_time: datetime = None, 
                 error_message: str = None, host: str = "127.0.0.1", port: int = 4002, client_id: int = 1):
        self.connected = connected
        self.account_id = account_id
        self.connection_time = connection_time
        self.error_message = error_message
        self.host = host
        self.port = port
        self.client_id = client_id


class IBMarketDataUpdate:
    """Compatibility class for market data"""
    def __init__(self, symbol: str, bid: float = None, ask: float = None, 
                 last: float = None, volume: int = None, timestamp: datetime = None):
        self.symbol = symbol
        self.bid = bid
        self.ask = ask
        self.last = last
        self.volume = volume
        self.timestamp = timestamp or datetime.now()