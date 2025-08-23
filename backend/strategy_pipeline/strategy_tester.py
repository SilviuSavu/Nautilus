"""
Automated Strategy Testing Framework
Handles syntax validation, code analysis, automated backtesting, paper trading validation, and performance benchmarking
"""

import ast
import asyncio
import json
import logging
import subprocess
import tempfile
import uuid
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel

logger = logging.getLogger(__name__)


class TestStatus(Enum):
    """Test execution status"""
    PENDING = "pending"
    RUNNING = "running"
    PASSED = "passed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class TestType(Enum):
    """Types of tests to run"""
    SYNTAX_VALIDATION = "syntax_validation"
    CODE_ANALYSIS = "code_analysis"
    BACKTEST = "backtest"
    PAPER_TRADING = "paper_trading"
    PERFORMANCE_BENCHMARK = "performance_benchmark"
    INTEGRATION_TEST = "integration_test"


class TestResult(BaseModel):
    """Individual test result"""
    test_id: str
    test_type: TestType
    status: TestStatus
    score: Optional[float] = None
    message: str = ""
    details: Dict[str, Any] = {}
    duration_seconds: float = 0.0
    started_at: datetime
    completed_at: Optional[datetime] = None


class ValidationResult(BaseModel):
    """Syntax validation result"""
    valid: bool
    errors: List[str] = []
    warnings: List[str] = []
    imports: List[str] = []
    classes: List[str] = []
    methods: List[str] = []


class CodeAnalysisResult(BaseModel):
    """Code quality analysis result"""
    complexity_score: float
    maintainability_index: float
    code_quality_score: float
    issues: List[Dict[str, Any]] = []
    metrics: Dict[str, Any] = {}


class BacktestResult(BaseModel):
    """Backtest execution result"""
    total_pnl: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    total_trades: int
    winning_trades: int
    avg_trade_duration: float
    metrics: Dict[str, Any] = {}


class BenchmarkResult(BaseModel):
    """Performance benchmark result"""
    cpu_usage: float
    memory_usage: float
    execution_time: float
    throughput: float
    latency_p95: float
    resource_efficiency: float


class TestSuite(BaseModel):
    """Complete test suite result"""
    suite_id: str
    strategy_id: str
    strategy_version: str
    status: TestStatus
    tests: List[TestResult]
    overall_score: float
    started_at: datetime
    completed_at: Optional[datetime] = None
    environment: str = "testing"


class StrategyTester:
    """Automated strategy testing framework"""
    
    def __init__(self, nautilus_engine_service=None):
        self.nautilus_engine_service = nautilus_engine_service
        self._active_tests: Dict[str, TestSuite] = {}
        self._test_results: Dict[str, TestSuite] = {}
        
    async def run_full_test_suite(self, 
                                  strategy_code: str, 
                                  strategy_config: Dict[str, Any],
                                  test_config: Dict[str, Any] = None) -> TestSuite:
        """Run complete test suite for a strategy"""
        suite_id = str(uuid.uuid4())
        strategy_id = strategy_config.get("id", "unknown")
        strategy_version = strategy_config.get("version", "1.0.0")
        
        test_suite = TestSuite(
            suite_id=suite_id,
            strategy_id=strategy_id,
            strategy_version=strategy_version,
            status=TestStatus.RUNNING,
            tests=[],
            overall_score=0.0,
            started_at=datetime.utcnow()
        )
        
        self._active_tests[suite_id] = test_suite
        
        try:
            # Run all tests in sequence
            tests_to_run = [
                TestType.SYNTAX_VALIDATION,
                TestType.CODE_ANALYSIS,
                TestType.BACKTEST,
                TestType.PAPER_TRADING,
                TestType.PERFORMANCE_BENCHMARK
            ]
            
            for test_type in tests_to_run:
                test_result = await self._run_single_test(
                    test_type, strategy_code, strategy_config, test_config
                )
                test_suite.tests.append(test_result)
                
                # Stop if critical tests fail
                if test_type in [TestType.SYNTAX_VALIDATION, TestType.CODE_ANALYSIS] and test_result.status == TestStatus.FAILED:
                    logger.warning(f"Critical test {test_type.value} failed, stopping test suite")
                    break
            
            # Calculate overall score
            test_suite.overall_score = self._calculate_overall_score(test_suite.tests)
            test_suite.status = TestStatus.PASSED if test_suite.overall_score >= 70 else TestStatus.FAILED
            test_suite.completed_at = datetime.utcnow()
            
        except Exception as e:
            logger.error(f"Test suite {suite_id} failed: {str(e)}")
            test_suite.status = TestStatus.FAILED
            test_suite.completed_at = datetime.utcnow()
        
        finally:
            self._test_results[suite_id] = test_suite
            if suite_id in self._active_tests:
                del self._active_tests[suite_id]
        
        return test_suite
    
    async def _run_single_test(self, 
                               test_type: TestType, 
                               strategy_code: str, 
                               strategy_config: Dict[str, Any],
                               test_config: Dict[str, Any] = None) -> TestResult:
        """Run a single test"""
        test_id = str(uuid.uuid4())
        start_time = datetime.utcnow()
        
        test_result = TestResult(
            test_id=test_id,
            test_type=test_type,
            status=TestStatus.RUNNING,
            started_at=start_time
        )
        
        try:
            if test_type == TestType.SYNTAX_VALIDATION:
                result = await self._run_syntax_validation(strategy_code)
                test_result.status = TestStatus.PASSED if result.valid else TestStatus.FAILED
                test_result.message = f"Found {len(result.errors)} errors, {len(result.warnings)} warnings"
                test_result.details = result.dict()
                test_result.score = 100.0 if result.valid else 0.0
                
            elif test_type == TestType.CODE_ANALYSIS:
                result = await self._run_code_analysis(strategy_code)
                test_result.status = TestStatus.PASSED if result.code_quality_score >= 70 else TestStatus.FAILED
                test_result.message = f"Code quality score: {result.code_quality_score:.1f}"
                test_result.details = result.dict()
                test_result.score = result.code_quality_score
                
            elif test_type == TestType.BACKTEST:
                result = await self._run_backtest(strategy_code, strategy_config, test_config)
                test_result.status = TestStatus.PASSED if result.sharpe_ratio > 1.0 else TestStatus.FAILED
                test_result.message = f"Sharpe: {result.sharpe_ratio:.2f}, Max DD: {result.max_drawdown:.2f}%"
                test_result.details = result.dict()
                test_result.score = min(100.0, max(0.0, result.sharpe_ratio * 50))
                
            elif test_type == TestType.PAPER_TRADING:
                result = await self._run_paper_trading_test(strategy_code, strategy_config, test_config)
                test_result.status = TestStatus.PASSED if result.get("success", False) else TestStatus.FAILED
                test_result.message = result.get("message", "Paper trading test completed")
                test_result.details = result
                test_result.score = result.get("score", 0.0)
                
            elif test_type == TestType.PERFORMANCE_BENCHMARK:
                result = await self._run_performance_benchmark(strategy_code, strategy_config)
                test_result.status = TestStatus.PASSED if result.resource_efficiency >= 70 else TestStatus.FAILED
                test_result.message = f"Resource efficiency: {result.resource_efficiency:.1f}%"
                test_result.details = result.dict()
                test_result.score = result.resource_efficiency
            
        except Exception as e:
            logger.error(f"Test {test_type.value} failed: {str(e)}")
            test_result.status = TestStatus.FAILED
            test_result.message = f"Test failed: {str(e)}"
            test_result.score = 0.0
        
        finally:
            end_time = datetime.utcnow()
            test_result.completed_at = end_time
            test_result.duration_seconds = (end_time - start_time).total_seconds()
        
        return test_result
    
    async def _run_syntax_validation(self, strategy_code: str) -> ValidationResult:
        """Run syntax validation on strategy code"""
        validation_result = ValidationResult(valid=True)
        
        try:
            # Parse the Python code
            tree = ast.parse(strategy_code)
            
            # Extract imports, classes, and methods
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        validation_result.imports.append(alias.name)
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        for alias in node.names:
                            validation_result.imports.append(f"{node.module}.{alias.name}")
                elif isinstance(node, ast.ClassDef):
                    validation_result.classes.append(node.name)
                elif isinstance(node, ast.FunctionDef):
                    validation_result.methods.append(node.name)
            
            # Check for required NautilusTrader patterns
            code_lower = strategy_code.lower()
            if "class" not in code_lower:
                validation_result.warnings.append("No strategy class found")
            if "nautilustrader" not in code_lower and "nautilus_trader" not in code_lower:
                validation_result.warnings.append("No NautilusTrader imports detected")
            if "on_start" not in code_lower and "on_data" not in code_lower:
                validation_result.warnings.append("No standard strategy lifecycle methods found")
            
        except SyntaxError as e:
            validation_result.valid = False
            validation_result.errors.append(f"Syntax error at line {e.lineno}: {e.msg}")
        except Exception as e:
            validation_result.valid = False
            validation_result.errors.append(f"Code validation failed: {str(e)}")
        
        return validation_result
    
    async def _run_code_analysis(self, strategy_code: str) -> CodeAnalysisResult:
        """Run code quality analysis"""
        try:
            # Basic complexity analysis
            tree = ast.parse(strategy_code)
            
            # Count various code elements
            total_nodes = sum(1 for _ in ast.walk(tree))
            functions = sum(1 for node in ast.walk(tree) if isinstance(node, ast.FunctionDef))
            classes = sum(1 for node in ast.walk(tree) if isinstance(node, ast.ClassDef))
            conditionals = sum(1 for node in ast.walk(tree) if isinstance(node, (ast.If, ast.While, ast.For)))
            
            # Calculate complexity score (simplified)
            complexity_score = max(0, 100 - (conditionals * 5) - (total_nodes * 0.1))
            
            # Calculate maintainability index (simplified)
            lines_of_code = len(strategy_code.split('\n'))
            maintainability_index = max(0, 100 - (lines_of_code * 0.5) - (functions * 2))
            
            # Overall code quality score
            code_quality_score = (complexity_score + maintainability_index) / 2
            
            issues = []
            if lines_of_code > 500:
                issues.append({"type": "warning", "message": "Strategy is quite large, consider breaking it down"})
            if functions < 3:
                issues.append({"type": "info", "message": "Consider adding more helper methods for better organization"})
            if classes == 0:
                issues.append({"type": "error", "message": "No strategy class found"})
            
            return CodeAnalysisResult(
                complexity_score=complexity_score,
                maintainability_index=maintainability_index,
                code_quality_score=code_quality_score,
                issues=issues,
                metrics={
                    "lines_of_code": lines_of_code,
                    "functions": functions,
                    "classes": classes,
                    "total_nodes": total_nodes,
                    "conditionals": conditionals
                }
            )
            
        except Exception as e:
            logger.error(f"Code analysis failed: {str(e)}")
            return CodeAnalysisResult(
                complexity_score=0.0,
                maintainability_index=0.0,
                code_quality_score=0.0,
                issues=[{"type": "error", "message": f"Analysis failed: {str(e)}"}]
            )
    
    async def _run_backtest(self, 
                            strategy_code: str, 
                            strategy_config: Dict[str, Any],
                            test_config: Dict[str, Any] = None) -> BacktestResult:
        """Run automated backtest"""
        try:
            # Default backtest configuration
            backtest_config = {
                "start_date": "2024-01-01",
                "end_date": "2024-12-31",
                "initial_balance": 100000.0,
                "instrument": "EUR/USD.SIM"
            }
            
            if test_config and "backtest" in test_config:
                backtest_config.update(test_config["backtest"])
            
            # Mock backtest results for now
            # In production, this would integrate with actual NautilusTrader backtesting
            total_pnl = 5000.0 + (hash(strategy_code) % 10000) - 5000
            max_drawdown = abs(total_pnl) * 0.3
            total_trades = 50 + (hash(strategy_code) % 100)
            winning_trades = int(total_trades * (0.6 + (hash(strategy_code) % 20) / 100))
            
            win_rate = winning_trades / total_trades if total_trades > 0 else 0
            sharpe_ratio = total_pnl / 10000 if total_pnl > 0 else -1.0
            avg_trade_duration = 4.5 + (hash(strategy_code) % 10)
            
            return BacktestResult(
                total_pnl=total_pnl,
                sharpe_ratio=sharpe_ratio,
                max_drawdown=max_drawdown,
                win_rate=win_rate,
                total_trades=total_trades,
                winning_trades=winning_trades,
                avg_trade_duration=avg_trade_duration,
                metrics={
                    "start_date": backtest_config["start_date"],
                    "end_date": backtest_config["end_date"],
                    "initial_balance": backtest_config["initial_balance"],
                    "final_balance": backtest_config["initial_balance"] + total_pnl
                }
            )
            
        except Exception as e:
            logger.error(f"Backtest failed: {str(e)}")
            return BacktestResult(
                total_pnl=0.0,
                sharpe_ratio=-1.0,
                max_drawdown=100.0,
                win_rate=0.0,
                total_trades=0,
                winning_trades=0,
                avg_trade_duration=0.0
            )
    
    async def _run_paper_trading_test(self, 
                                      strategy_code: str, 
                                      strategy_config: Dict[str, Any],
                                      test_config: Dict[str, Any] = None) -> Dict[str, Any]:
        """Run paper trading validation test"""
        try:
            # Configuration for paper trading test
            paper_config = {
                "duration_minutes": 5,
                "initial_balance": 10000.0,
                "instrument": "EUR/USD.SIM"
            }
            
            if test_config and "paper_trading" in test_config:
                paper_config.update(test_config["paper_trading"])
            
            if self.nautilus_engine_service:
                # Deploy strategy in paper trading mode
                deployment_result = await self._deploy_for_testing(
                    strategy_code, 
                    strategy_config, 
                    "paper"
                )
                
                if deployment_result.get("success"):
                    # Run for specified duration
                    await asyncio.sleep(paper_config["duration_minutes"] * 60)
                    
                    # Collect results
                    status = await self._get_test_deployment_status(deployment_result["deployment_id"])
                    
                    # Stop the test deployment
                    await self._stop_test_deployment(deployment_result["deployment_id"])
                    
                    return {
                        "success": True,
                        "message": f"Paper trading test completed successfully",
                        "score": min(100.0, max(0.0, status.get("score", 50.0))),
                        "details": status
                    }
                else:
                    return {
                        "success": False,
                        "message": f"Failed to deploy for paper trading: {deployment_result.get('error')}",
                        "score": 0.0
                    }
            else:
                # Mock paper trading result
                return {
                    "success": True,
                    "message": "Paper trading test completed (mocked)",
                    "score": 75.0,
                    "details": {
                        "trades_executed": 3,
                        "pnl": 150.0,
                        "duration_minutes": paper_config["duration_minutes"]
                    }
                }
                
        except Exception as e:
            logger.error(f"Paper trading test failed: {str(e)}")
            return {
                "success": False,
                "message": f"Paper trading test failed: {str(e)}",
                "score": 0.0
            }
    
    async def _run_performance_benchmark(self, 
                                         strategy_code: str, 
                                         strategy_config: Dict[str, Any]) -> BenchmarkResult:
        """Run performance benchmarking"""
        try:
            # Mock performance metrics
            # In production, this would run actual performance tests
            
            code_size = len(strategy_code)
            complexity_factor = code_size / 1000.0
            
            cpu_usage = min(100.0, max(5.0, complexity_factor * 10))
            memory_usage = min(100.0, max(10.0, complexity_factor * 15))
            execution_time = max(0.1, complexity_factor * 0.5)
            throughput = max(10.0, 1000.0 / complexity_factor)
            latency_p95 = max(1.0, complexity_factor * 2)
            
            # Calculate resource efficiency score
            resource_efficiency = max(0.0, 100.0 - cpu_usage * 0.3 - memory_usage * 0.2 - latency_p95 * 5)
            
            return BenchmarkResult(
                cpu_usage=cpu_usage,
                memory_usage=memory_usage,
                execution_time=execution_time,
                throughput=throughput,
                latency_p95=latency_p95,
                resource_efficiency=resource_efficiency
            )
            
        except Exception as e:
            logger.error(f"Performance benchmark failed: {str(e)}")
            return BenchmarkResult(
                cpu_usage=100.0,
                memory_usage=100.0,
                execution_time=10.0,
                throughput=1.0,
                latency_p95=100.0,
                resource_efficiency=0.0
            )
    
    async def _deploy_for_testing(self, 
                                  strategy_code: str, 
                                  strategy_config: Dict[str, Any],
                                  mode: str) -> Dict[str, Any]:
        """Deploy strategy for testing purposes"""
        try:
            if not self.nautilus_engine_service:
                return {"success": False, "error": "NautilusTrader engine service not available"}
            
            # Create temporary deployment configuration
            test_deployment_config = {
                "strategy_code": strategy_code,
                "strategy_config": strategy_config,
                "mode": mode,
                "test_deployment": True
            }
            
            # Mock deployment for now
            deployment_id = str(uuid.uuid4())
            return {
                "success": True,
                "deployment_id": deployment_id,
                "message": f"Test deployment created in {mode} mode"
            }
            
        except Exception as e:
            logger.error(f"Test deployment failed: {str(e)}")
            return {"success": False, "error": str(e)}
    
    async def _get_test_deployment_status(self, deployment_id: str) -> Dict[str, Any]:
        """Get status of test deployment"""
        # Mock status
        return {
            "deployment_id": deployment_id,
            "status": "running",
            "score": 75.0,
            "trades_executed": 2,
            "pnl": 100.0
        }
    
    async def _stop_test_deployment(self, deployment_id: str) -> Dict[str, Any]:
        """Stop test deployment"""
        # Mock stop
        return {"success": True, "message": "Test deployment stopped"}
    
    def _calculate_overall_score(self, test_results: List[TestResult]) -> float:
        """Calculate overall test suite score"""
        if not test_results:
            return 0.0
        
        weights = {
            TestType.SYNTAX_VALIDATION: 0.25,
            TestType.CODE_ANALYSIS: 0.20,
            TestType.BACKTEST: 0.30,
            TestType.PAPER_TRADING: 0.15,
            TestType.PERFORMANCE_BENCHMARK: 0.10
        }
        
        weighted_score = 0.0
        total_weight = 0.0
        
        for result in test_results:
            if result.score is not None:
                weight = weights.get(result.test_type, 0.1)
                weighted_score += result.score * weight
                total_weight += weight
        
        return weighted_score / total_weight if total_weight > 0 else 0.0
    
    def get_test_suite(self, suite_id: str) -> Optional[TestSuite]:
        """Get test suite by ID"""
        return self._test_results.get(suite_id) or self._active_tests.get(suite_id)
    
    def list_test_suites(self, 
                         strategy_id: str = None, 
                         status: TestStatus = None) -> List[TestSuite]:
        """List test suites with optional filtering"""
        all_suites = list(self._test_results.values()) + list(self._active_tests.values())
        
        if strategy_id:
            all_suites = [s for s in all_suites if s.strategy_id == strategy_id]
        
        if status:
            all_suites = [s for s in all_suites if s.status == status]
        
        return sorted(all_suites, key=lambda s: s.started_at, reverse=True)
    
    async def cancel_test_suite(self, suite_id: str) -> bool:
        """Cancel running test suite"""
        if suite_id in self._active_tests:
            test_suite = self._active_tests[suite_id]
            test_suite.status = TestStatus.CANCELLED
            test_suite.completed_at = datetime.utcnow()
            
            # Move to results
            self._test_results[suite_id] = test_suite
            del self._active_tests[suite_id]
            
            logger.info(f"Test suite {suite_id} cancelled")
            return True
        
        return False
    
    def get_testing_statistics(self) -> Dict[str, Any]:
        """Get testing statistics"""
        all_suites = list(self._test_results.values()) + list(self._active_tests.values())
        
        total_suites = len(all_suites)
        passed_suites = len([s for s in all_suites if s.status == TestStatus.PASSED])
        failed_suites = len([s for s in all_suites if s.status == TestStatus.FAILED])
        running_suites = len([s for s in all_suites if s.status == TestStatus.RUNNING])
        
        avg_score = 0.0
        if all_suites:
            avg_score = sum(s.overall_score for s in all_suites) / len(all_suites)
        
        return {
            "total_test_suites": total_suites,
            "passed_suites": passed_suites,
            "failed_suites": failed_suites,
            "running_suites": running_suites,
            "success_rate": (passed_suites / total_suites * 100) if total_suites > 0 else 0,
            "average_score": avg_score,
            "timestamp": datetime.utcnow().isoformat()
        }


# Global service instance
strategy_tester = StrategyTester()