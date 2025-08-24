#!/usr/bin/env python3
"""
Simple Strategy Engine - Containerized Strategy Deployment Service
Automated strategy deployment, testing, and lifecycle management
"""

import asyncio
import logging
import os
import time
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from enum import Enum
import json
import uuid
from fastapi import FastAPI, HTTPException
import uvicorn

# Basic MessageBus integration
from enhanced_messagebus_client import BufferedMessageBusClient, MessageBusConfig

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class StrategyStatus(Enum):
    DRAFT = "draft"
    TESTING = "testing"
    APPROVED = "approved"
    DEPLOYED = "deployed"
    PAUSED = "paused"
    STOPPED = "stopped"
    ERROR = "error"

class DeploymentType(Enum):
    DIRECT = "direct"
    BLUE_GREEN = "blue_green"
    CANARY = "canary"
    ROLLING = "rolling"

class TestingStage(Enum):
    SYNTAX_CHECK = "syntax_check"
    UNIT_TESTS = "unit_tests"
    BACKTEST = "backtest"
    PAPER_TRADING = "paper_trading"
    RISK_VALIDATION = "risk_validation"
    PERFORMANCE_VALIDATION = "performance_validation"

@dataclass
class StrategyDefinition:
    strategy_id: str
    strategy_name: str
    version: str
    code: str
    parameters: Dict[str, Any]
    risk_limits: Dict[str, Any]
    status: StrategyStatus
    created_at: datetime
    updated_at: datetime

@dataclass
class DeploymentPipeline:
    pipeline_id: str
    strategy_id: str
    deployment_type: DeploymentType
    stages: List[TestingStage]
    current_stage: Optional[TestingStage]
    status: str
    started_at: datetime
    completed_at: Optional[datetime]
    results: Dict[str, Any]

@dataclass
class StrategyExecution:
    execution_id: str
    strategy_id: str
    status: str
    start_time: datetime
    end_time: Optional[datetime]
    performance_metrics: Dict[str, float]
    trade_count: int
    pnl: float

class SimpleStrategyEngine:
    """
    Simple Strategy Engine demonstrating containerization approach
    Automated strategy deployment with CI/CD pipeline
    """
    
    def __init__(self):
        self.app = FastAPI(title="Nautilus Simple Strategy Engine", version="1.0.0")
        self.is_running = False
        self.strategies_deployed = 0
        self.pipelines_executed = 0
        self.tests_completed = 0
        self.start_time = time.time()
        
        # Strategy management state
        self.strategies: Dict[str, StrategyDefinition] = {}
        self.deployments: Dict[str, DeploymentPipeline] = {}
        self.active_executions: Dict[str, StrategyExecution] = {}
        
        # MessageBus configuration
        self.messagebus_config = MessageBusConfig(
            redis_host=os.getenv("REDIS_HOST", "redis"),
            redis_port=int(os.getenv("REDIS_PORT", "6379")),
            redis_db=0
        )
        
        self.messagebus = None
        self.setup_routes()
        
    def setup_routes(self):
        """Setup FastAPI routes"""
        
        @self.app.get("/health")
        async def health_check():
            return {
                "status": "healthy" if self.is_running else "stopped",
                "strategies_deployed": self.strategies_deployed,
                "pipelines_executed": self.pipelines_executed,
                "tests_completed": self.tests_completed,
                "active_strategies": len(self.strategies),
                "active_executions": len(self.active_executions),
                "uptime_seconds": time.time() - self.start_time,
                "messagebus_connected": self.messagebus is not None and self.messagebus.is_connected
            }
        
        @self.app.get("/metrics")
        async def get_metrics():
            uptime = time.time() - self.start_time
            return {
                "deployments_per_hour": (self.strategies_deployed / max(1, uptime)) * 3600,
                "pipelines_per_hour": (self.pipelines_executed / max(1, uptime)) * 3600,
                "total_strategies": len(self.strategies),
                "total_deployments": self.strategies_deployed,
                "total_pipelines": self.pipelines_executed,
                "active_executions": len(self.active_executions),
                "success_rate": self._calculate_success_rate(),
                "uptime": uptime,
                "engine_type": "strategy_deployment",
                "containerized": True
            }
        
        @self.app.get("/strategies")
        async def get_strategies():
            """Get all strategies"""
            strategies = []
            for strategy in self.strategies.values():
                strategies.append({
                    "strategy_id": strategy.strategy_id,
                    "strategy_name": strategy.strategy_name,
                    "version": strategy.version,
                    "status": strategy.status.value,
                    "created_at": strategy.created_at.isoformat(),
                    "updated_at": strategy.updated_at.isoformat()
                })
            
            return {
                "strategies": strategies,
                "count": len(strategies)
            }
        
        @self.app.post("/strategies")
        async def create_strategy(strategy_data: Dict[str, Any]):
            """Create new strategy"""
            try:
                strategy = StrategyDefinition(
                    strategy_id=str(uuid.uuid4()),
                    strategy_name=strategy_data.get("strategy_name", "Unnamed Strategy"),
                    version=strategy_data.get("version", "1.0.0"),
                    code=strategy_data.get("code", ""),
                    parameters=strategy_data.get("parameters", {}),
                    risk_limits=strategy_data.get("risk_limits", {}),
                    status=StrategyStatus.DRAFT,
                    created_at=datetime.now(),
                    updated_at=datetime.now()
                )
                
                self.strategies[strategy.strategy_id] = strategy
                
                return {
                    "status": "created",
                    "strategy_id": strategy.strategy_id,
                    "strategy_name": strategy.strategy_name,
                    "version": strategy.version
                }
                
            except Exception as e:
                logger.error(f"Strategy creation error: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.post("/strategies/{strategy_id}/deploy")
        async def deploy_strategy(strategy_id: str, deployment_config: Dict[str, Any]):
            """Deploy strategy with automated pipeline"""
            try:
                if strategy_id not in self.strategies:
                    raise HTTPException(status_code=404, detail="Strategy not found")
                
                strategy = self.strategies[strategy_id]
                deployment_type = DeploymentType(deployment_config.get("deployment_type", "direct"))
                
                # Create deployment pipeline
                pipeline = DeploymentPipeline(
                    pipeline_id=str(uuid.uuid4()),
                    strategy_id=strategy_id,
                    deployment_type=deployment_type,
                    stages=[
                        TestingStage.SYNTAX_CHECK,
                        TestingStage.UNIT_TESTS,
                        TestingStage.BACKTEST,
                        TestingStage.PAPER_TRADING,
                        TestingStage.RISK_VALIDATION,
                        TestingStage.PERFORMANCE_VALIDATION
                    ],
                    current_stage=None,
                    status="queued",
                    started_at=datetime.now(),
                    completed_at=None,
                    results={}
                )
                
                self.deployments[pipeline.pipeline_id] = pipeline
                
                # Start pipeline execution
                asyncio.create_task(self._execute_deployment_pipeline(pipeline))
                
                return {
                    "status": "deployment_started",
                    "pipeline_id": pipeline.pipeline_id,
                    "strategy_id": strategy_id,
                    "deployment_type": deployment_type.value,
                    "stages": [stage.value for stage in pipeline.stages]
                }
                
            except HTTPException:
                raise
            except Exception as e:
                logger.error(f"Strategy deployment error: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/deployments/{pipeline_id}/status")
        async def get_deployment_status(pipeline_id: str):
            """Get deployment pipeline status"""
            if pipeline_id not in self.deployments:
                raise HTTPException(status_code=404, detail="Pipeline not found")
            
            pipeline = self.deployments[pipeline_id]
            
            return {
                "pipeline_id": pipeline_id,
                "strategy_id": pipeline.strategy_id,
                "deployment_type": pipeline.deployment_type.value,
                "current_stage": pipeline.current_stage.value if pipeline.current_stage else None,
                "status": pipeline.status,
                "started_at": pipeline.started_at.isoformat(),
                "completed_at": pipeline.completed_at.isoformat() if pipeline.completed_at else None,
                "stages_completed": len([stage for stage in pipeline.results.keys()]),
                "total_stages": len(pipeline.stages),
                "results": pipeline.results
            }
        
        @self.app.post("/strategies/{strategy_id}/test")
        async def test_strategy(strategy_id: str, test_config: Dict[str, Any]):
            """Test strategy with specific configuration"""
            try:
                if strategy_id not in self.strategies:
                    raise HTTPException(status_code=404, detail="Strategy not found")
                
                strategy = self.strategies[strategy_id]
                test_type = test_config.get("test_type", "backtest")
                
                # Execute test
                test_result = await self._execute_strategy_test(strategy, test_type, test_config)
                self.tests_completed += 1
                
                return {
                    "status": "test_completed",
                    "strategy_id": strategy_id,
                    "test_type": test_type,
                    "result": test_result
                }
                
            except HTTPException:
                raise
            except Exception as e:
                logger.error(f"Strategy test error: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.post("/strategies/{strategy_id}/start")
        async def start_strategy_execution(strategy_id: str, execution_config: Dict[str, Any]):
            """Start live strategy execution"""
            try:
                if strategy_id not in self.strategies:
                    raise HTTPException(status_code=404, detail="Strategy not found")
                
                strategy = self.strategies[strategy_id]
                
                if strategy.status != StrategyStatus.DEPLOYED:
                    raise HTTPException(status_code=400, detail="Strategy must be deployed before execution")
                
                # Create execution record
                execution = StrategyExecution(
                    execution_id=str(uuid.uuid4()),
                    strategy_id=strategy_id,
                    status="running",
                    start_time=datetime.now(),
                    end_time=None,
                    performance_metrics={},
                    trade_count=0,
                    pnl=0.0
                )
                
                self.active_executions[execution.execution_id] = execution
                
                # Start execution monitoring
                asyncio.create_task(self._monitor_strategy_execution(execution))
                
                return {
                    "status": "execution_started",
                    "execution_id": execution.execution_id,
                    "strategy_id": strategy_id,
                    "start_time": execution.start_time.isoformat()
                }
                
            except HTTPException:
                raise
            except Exception as e:
                logger.error(f"Strategy execution error: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.post("/strategies/{strategy_id}/stop")
        async def stop_strategy_execution(strategy_id: str):
            """Stop strategy execution"""
            try:
                # Find active execution for strategy
                execution = None
                for exec_record in self.active_executions.values():
                    if exec_record.strategy_id == strategy_id and exec_record.status == "running":
                        execution = exec_record
                        break
                
                if not execution:
                    raise HTTPException(status_code=404, detail="No active execution found")
                
                # Stop execution
                execution.status = "stopped"
                execution.end_time = datetime.now()
                
                # Update strategy status
                if strategy_id in self.strategies:
                    self.strategies[strategy_id].status = StrategyStatus.STOPPED
                
                return {
                    "status": "execution_stopped",
                    "execution_id": execution.execution_id,
                    "strategy_id": strategy_id,
                    "duration_minutes": (execution.end_time - execution.start_time).total_seconds() / 60
                }
                
            except HTTPException:
                raise
            except Exception as e:
                logger.error(f"Strategy stop error: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/executions")
        async def get_active_executions():
            """Get all active strategy executions"""
            executions = []
            for execution in self.active_executions.values():
                executions.append({
                    "execution_id": execution.execution_id,
                    "strategy_id": execution.strategy_id,
                    "status": execution.status,
                    "start_time": execution.start_time.isoformat(),
                    "end_time": execution.end_time.isoformat() if execution.end_time else None,
                    "trade_count": execution.trade_count,
                    "pnl": execution.pnl,
                    "performance_metrics": execution.performance_metrics
                })
            
            return {
                "executions": executions,
                "count": len(executions)
            }

    async def start_engine(self):
        """Start the strategy engine"""
        try:
            logger.info("Starting Simple Strategy Engine...")
            
            # Try to initialize MessageBus
            try:
                self.messagebus = BufferedMessageBusClient(self.messagebus_config)
                await self.messagebus.start()
                logger.info("MessageBus connected successfully")
            except Exception as e:
                logger.warning(f"MessageBus connection failed: {e}. Running without MessageBus.")
                self.messagebus = None
            
            # Initialize sample strategies for demonstration
            await self._initialize_sample_strategies()
            
            self.is_running = True
            logger.info("Simple Strategy Engine started successfully")
            
        except Exception as e:
            logger.error(f"Failed to start Strategy Engine: {e}")
            raise
    
    async def stop_engine(self):
        """Stop the strategy engine"""
        logger.info("Stopping Simple Strategy Engine...")
        self.is_running = False
        
        # Stop all active executions
        for execution in self.active_executions.values():
            if execution.status == "running":
                execution.status = "stopped"
                execution.end_time = datetime.now()
        
        if self.messagebus:
            await self.messagebus.stop()
        
        logger.info("Simple Strategy Engine stopped")
    
    async def _initialize_sample_strategies(self):
        """Initialize sample strategies for demonstration"""
        sample_strategies = [
            {
                "strategy_name": "Mean Reversion RSI",
                "version": "1.0.0",
                "code": "# RSI-based mean reversion strategy\nclass RSIMeanReversionStrategy:\n    def __init__(self):\n        self.rsi_period = 14\n        self.oversold = 30\n        self.overbought = 70",
                "parameters": {"rsi_period": 14, "oversold_threshold": 30, "overbought_threshold": 70},
                "risk_limits": {"max_position_size": 10000, "daily_loss_limit": -1000}
            },
            {
                "strategy_name": "Moving Average Crossover",
                "version": "2.1.0",
                "code": "# Simple moving average crossover strategy\nclass MACrossoverStrategy:\n    def __init__(self):\n        self.fast_period = 12\n        self.slow_period = 26",
                "parameters": {"fast_ma": 12, "slow_ma": 26, "signal_threshold": 0.02},
                "risk_limits": {"max_position_size": 15000, "daily_loss_limit": -1500}
            }
        ]
        
        for strategy_data in sample_strategies:
            strategy = StrategyDefinition(
                strategy_id=str(uuid.uuid4()),
                strategy_name=strategy_data["strategy_name"],
                version=strategy_data["version"],
                code=strategy_data["code"],
                parameters=strategy_data["parameters"],
                risk_limits=strategy_data["risk_limits"],
                status=StrategyStatus.DRAFT,
                created_at=datetime.now(),
                updated_at=datetime.now()
            )
            self.strategies[strategy.strategy_id] = strategy
        
        logger.info(f"Initialized {len(sample_strategies)} sample strategies")
    
    async def _execute_deployment_pipeline(self, pipeline: DeploymentPipeline):
        """Execute deployment pipeline stages"""
        try:
            pipeline.status = "running"
            
            for stage in pipeline.stages:
                pipeline.current_stage = stage
                logger.info(f"Executing pipeline stage: {stage.value}")
                
                # Execute stage
                stage_result = await self._execute_pipeline_stage(pipeline, stage)
                pipeline.results[stage.value] = stage_result
                
                # Check if stage passed
                if not stage_result.get("passed", False):
                    pipeline.status = "failed"
                    pipeline.completed_at = datetime.now()
                    logger.error(f"Pipeline stage {stage.value} failed")
                    return
                
                await asyncio.sleep(0.1)  # Brief pause between stages
            
            # All stages passed
            pipeline.status = "completed"
            pipeline.completed_at = datetime.now()
            
            # Update strategy status to deployed
            if pipeline.strategy_id in self.strategies:
                self.strategies[pipeline.strategy_id].status = StrategyStatus.DEPLOYED
            
            self.strategies_deployed += 1
            self.pipelines_executed += 1
            
            logger.info(f"Deployment pipeline {pipeline.pipeline_id} completed successfully")
            
        except Exception as e:
            logger.error(f"Pipeline execution error: {e}")
            pipeline.status = "error"
            pipeline.completed_at = datetime.now()
    
    async def _execute_pipeline_stage(self, pipeline: DeploymentPipeline, stage: TestingStage) -> Dict[str, Any]:
        """Execute individual pipeline stage"""
        # Simulate stage execution time
        await asyncio.sleep(0.5)  # 500ms per stage
        
        if stage == TestingStage.SYNTAX_CHECK:
            return {"passed": True, "message": "Syntax validation passed", "duration_ms": 150}
        elif stage == TestingStage.UNIT_TESTS:
            return {"passed": True, "tests_run": 15, "tests_passed": 15, "duration_ms": 800}
        elif stage == TestingStage.BACKTEST:
            return {"passed": True, "sharpe_ratio": 1.8, "max_drawdown": -0.12, "duration_ms": 2500}
        elif stage == TestingStage.PAPER_TRADING:
            return {"passed": True, "trades": 25, "win_rate": 0.68, "duration_ms": 5000}
        elif stage == TestingStage.RISK_VALIDATION:
            return {"passed": True, "risk_score": 0.3, "var_95": -850, "duration_ms": 400}
        elif stage == TestingStage.PERFORMANCE_VALIDATION:
            return {"passed": True, "alpha": 0.05, "beta": 0.8, "duration_ms": 600}
        else:
            return {"passed": True, "message": "Stage completed", "duration_ms": 100}
    
    async def _execute_strategy_test(self, strategy: StrategyDefinition, test_type: str, config: Dict[str, Any]) -> Dict[str, Any]:
        """Execute strategy test"""
        # Simulate test execution
        await asyncio.sleep(1.0)  # 1 second test time
        
        if test_type == "backtest":
            return {
                "total_return": 0.15,
                "sharpe_ratio": 1.6,
                "max_drawdown": -0.08,
                "win_rate": 0.62,
                "trade_count": 45,
                "duration_days": config.get("duration_days", 365)
            }
        elif test_type == "paper_trading":
            return {
                "duration_hours": 24,
                "trades_executed": 8,
                "pnl": 350.75,
                "win_rate": 0.75,
                "avg_trade_duration_minutes": 45
            }
        else:
            return {
                "test_type": test_type,
                "status": "completed",
                "score": 0.85
            }
    
    async def _monitor_strategy_execution(self, execution: StrategyExecution):
        """Monitor live strategy execution"""
        try:
            while execution.status == "running":
                # Simulate execution monitoring
                await asyncio.sleep(10)  # Check every 10 seconds
                
                # Update mock performance metrics
                execution.trade_count += 1 if time.time() % 30 < 1 else 0
                execution.pnl += (time.time() % 10) - 5  # Random P&L change
                
                # Update performance metrics
                execution.performance_metrics = {
                    "sharpe_ratio": round(1.2 + (time.time() % 0.8), 2),
                    "max_drawdown": round(-0.05 - (time.time() % 0.03), 3),
                    "win_rate": round(0.6 + (time.time() % 0.2), 2)
                }
                
        except Exception as e:
            logger.error(f"Strategy execution monitoring error: {e}")
            execution.status = "error"
    
    def _calculate_success_rate(self) -> float:
        """Calculate deployment success rate"""
        if self.pipelines_executed == 0:
            return 1.0
        
        successful_pipelines = sum(1 for pipeline in self.deployments.values() if pipeline.status == "completed")
        return successful_pipelines / len(self.deployments)

# Create and start the engine
simple_strategy_engine = SimpleStrategyEngine()

if __name__ == "__main__":
    # Configuration from environment
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8700"))
    
    logger.info(f"Starting Simple Strategy Engine on {host}:{port}")
    
    # Start the engine on startup
    async def lifespan():
        await simple_strategy_engine.start_engine()
    
    # Run startup
    asyncio.run(lifespan())
    
    # Start FastAPI server
    uvicorn.run(
        simple_strategy_engine.app,
        host=host,
        port=port,
        log_level="info",
        access_log=True
    )