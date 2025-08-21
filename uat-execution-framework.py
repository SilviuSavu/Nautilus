#!/usr/bin/env python3
"""
Nautilus Trading Platform - UAT Execution Framework
Automated testing framework for 21 production-ready stories
"""

import asyncio
import aiohttp
import json
import time
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class TestStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    PASSED = "passed"
    FAILED = "failed"
    SKIPPED = "skipped"

@dataclass
class TestResult:
    test_id: str
    test_name: str
    epic: str
    status: TestStatus
    execution_time: float
    error_message: Optional[str] = None
    response_data: Optional[Dict] = None

class UATFramework:
    def __init__(self, base_url: str = "http://localhost:8001"):
        self.base_url = base_url
        self.frontend_url = "http://localhost:3001"
        self.results: List[TestResult] = []
        self.session: Optional[aiohttp.ClientSession] = None
        
    async def __aenter__(self):
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=30)
        )
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()

    async def execute_test(self, test_id: str, test_name: str, epic: str, 
                          test_func, *args, **kwargs) -> TestResult:
        """Execute a single test with timing and error handling"""
        logger.info(f"🧪 Executing: {test_id} - {test_name}")
        
        start_time = time.time()
        try:
            result = await test_func(*args, **kwargs)
            execution_time = time.time() - start_time
            
            if result:
                logger.info(f"✅ PASSED: {test_id} ({execution_time:.2f}s)")
                return TestResult(
                    test_id=test_id,
                    test_name=test_name,
                    epic=epic,
                    status=TestStatus.PASSED,
                    execution_time=execution_time,
                    response_data=result if isinstance(result, dict) else None
                )
            else:
                logger.error(f"❌ FAILED: {test_id} ({execution_time:.2f}s)")
                return TestResult(
                    test_id=test_id,
                    test_name=test_name,
                    epic=epic,
                    status=TestStatus.FAILED,
                    execution_time=execution_time,
                    error_message="Test returned False"
                )
                
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"❌ ERROR: {test_id} - {str(e)} ({execution_time:.2f}s)")
            return TestResult(
                test_id=test_id,
                test_name=test_name,
                epic=epic,
                status=TestStatus.FAILED,
                execution_time=execution_time,
                error_message=str(e)
            )

    async def http_get(self, endpoint: str, expected_status: int = 200) -> bool:
        """Generic HTTP GET test"""
        url = f"{self.base_url}{endpoint}"
        try:
            async with self.session.get(url) as response:
                return response.status == expected_status
        except Exception:
            return False

    async def http_get_with_data(self, endpoint: str) -> Optional[Dict]:
        """HTTP GET that returns response data"""
        url = f"{self.base_url}{endpoint}"
        try:
            async with self.session.get(url) as response:
                if response.status == 200:
                    return await response.json()
                return None
        except Exception:
            return None

    async def test_response_time(self, endpoint: str, max_time: float = 0.1) -> bool:
        """Test API response time"""
        url = f"{self.base_url}{endpoint}"
        start_time = time.time()
        try:
            async with self.session.get(url) as response:
                response_time = time.time() - start_time
                return response.status == 200 and response_time <= max_time
        except Exception:
            return False

    # EPIC 1: Foundation Infrastructure Tests
    async def test_health_check(self) -> bool:
        return await self.http_get("/health")

    async def test_database_connection(self) -> bool:
        data = await self.http_get_with_data("/health")
        return data and data.get("database") == "connected"

    async def test_redis_connection(self) -> bool:
        data = await self.http_get_with_data("/health")
        return data and data.get("redis") == "connected"

    async def test_websocket_endpoint(self) -> bool:
        return await self.http_get("/ws", expected_status=426)  # Upgrade required

    async def test_auth_health(self) -> bool:
        return await self.http_get("/api/auth/health")

    async def test_messagebus_status(self) -> bool:
        return await self.http_get("/api/messagebus/status")

    # EPIC 2: Market Data Tests
    async def test_market_data_stream_status(self) -> bool:
        return await self.http_get("/api/market-data/stream/status")

    async def test_instrument_search(self) -> bool:
        data = await self.http_get_with_data("/api/instruments/search?query=AAPL")
        return data and len(data) > 0

    async def test_order_book_data(self) -> bool:
        return await self.http_get("/api/market-data/orderbook/AAPL")

    async def test_price_data(self) -> bool:
        return await self.http_get("/api/market-data/prices/AAPL")

    # EPIC 3: Trading & Position Management Tests
    async def test_trade_history(self) -> bool:
        return await self.http_get("/api/trades/history")

    async def test_position_monitoring(self) -> bool:
        return await self.http_get("/api/positions/current")

    async def test_portfolio_summary(self) -> bool:
        return await self.http_get("/api/portfolio/summary")

    async def test_pnl_calculation(self) -> bool:
        return await self.http_get("/api/portfolio/pnl")

    # EPIC 4: Strategy & Portfolio Tools Tests
    async def test_strategy_config(self) -> bool:
        return await self.http_get("/api/strategies/config")

    async def test_performance_metrics(self) -> bool:
        return await self.http_get("/api/performance/metrics")

    async def test_risk_assessment(self) -> bool:
        return await self.http_get("/api/risk/assessment")

    async def test_portfolio_visualization(self) -> bool:
        return await self.http_get("/api/portfolio/visualization")

    # EPIC 5: Analytics Suite Tests
    async def test_system_monitoring(self) -> bool:
        return await self.http_get("/api/monitoring/system")

    async def test_data_export(self) -> bool:
        return await self.http_get("/api/export/formats")

    async def test_advanced_charting(self) -> bool:
        return await self.http_get("/api/charts/indicators")

    async def test_performance_analytics(self) -> bool:
        return await self.http_get("/api/analytics/performance")

    # EPIC 6: Nautilus Engine Integration Tests
    async def test_engine_status(self) -> bool:
        return await self.http_get("/api/nautilus/engine/status")

    async def test_backtest_status(self) -> bool:
        return await self.http_get("/api/nautilus/backtest/status")

    async def test_deployment_status(self) -> bool:
        return await self.http_get("/api/deployment/status")

    async def test_data_pipeline_status(self) -> bool:
        return await self.http_get("/api/data/pipeline/status")

    # IB Integration Tests
    async def test_ib_connection(self) -> bool:
        return await self.http_get("/api/ib/connection/status")

    async def test_ib_market_data(self) -> bool:
        return await self.http_get("/api/ib/market-data/status")

    async def test_ib_orders(self) -> bool:
        return await self.http_get("/api/ib/orders/status")

    async def test_ib_positions(self) -> bool:
        return await self.http_get("/api/ib/positions")

    # Performance Tests
    async def test_api_response_time(self) -> bool:
        return await self.test_response_time("/health", max_time=0.1)

    async def test_frontend_accessibility(self) -> bool:
        try:
            async with self.session.get(self.frontend_url) as response:
                return response.status == 200
        except Exception:
            return False

    async def run_epic_1_tests(self):
        """Foundation Infrastructure Tests"""
        tests = [
            ("UAT-1.1", "Health Check", self.test_health_check),
            ("UAT-1.2", "Database Connection", self.test_database_connection),
            ("UAT-1.3", "Redis Connection", self.test_redis_connection),
            ("UAT-1.4", "WebSocket Endpoint", self.test_websocket_endpoint),
            ("UAT-1.5", "Auth Health", self.test_auth_health),
            ("UAT-1.6", "MessageBus Status", self.test_messagebus_status),
        ]
        
        for test_id, test_name, test_func in tests:
            result = await self.execute_test(test_id, test_name, "Epic 1", test_func)
            self.results.append(result)

    async def run_epic_2_tests(self):
        """Market Data Tests"""
        tests = [
            ("UAT-2.1", "Market Data Stream", self.test_market_data_stream_status),
            ("UAT-2.2", "Instrument Search", self.test_instrument_search),
            ("UAT-2.3", "Order Book Data", self.test_order_book_data),
            ("UAT-2.4", "Price Data", self.test_price_data),
        ]
        
        for test_id, test_name, test_func in tests:
            result = await self.execute_test(test_id, test_name, "Epic 2", test_func)
            self.results.append(result)

    async def run_epic_3_tests(self):
        """Trading & Position Management Tests"""
        tests = [
            ("UAT-3.1", "Trade History", self.test_trade_history),
            ("UAT-3.2", "Position Monitoring", self.test_position_monitoring),
            ("UAT-3.3", "Portfolio Summary", self.test_portfolio_summary),
            ("UAT-3.4", "P&L Calculation", self.test_pnl_calculation),
        ]
        
        for test_id, test_name, test_func in tests:
            result = await self.execute_test(test_id, test_name, "Epic 3", test_func)
            self.results.append(result)

    async def run_epic_4_tests(self):
        """Strategy & Portfolio Tools Tests"""
        tests = [
            ("UAT-4.1", "Strategy Config", self.test_strategy_config),
            ("UAT-4.2", "Performance Metrics", self.test_performance_metrics),
            ("UAT-4.3", "Risk Assessment", self.test_risk_assessment),
            ("UAT-4.4", "Portfolio Visualization", self.test_portfolio_visualization),
        ]
        
        for test_id, test_name, test_func in tests:
            result = await self.execute_test(test_id, test_name, "Epic 4", test_func)
            self.results.append(result)

    async def run_epic_5_tests(self):
        """Analytics Suite Tests"""
        tests = [
            ("UAT-5.1", "System Monitoring", self.test_system_monitoring),
            ("UAT-5.2", "Data Export", self.test_data_export),
            ("UAT-5.3", "Advanced Charting", self.test_advanced_charting),
            ("UAT-5.4", "Performance Analytics", self.test_performance_analytics),
        ]
        
        for test_id, test_name, test_func in tests:
            result = await self.execute_test(test_id, test_name, "Epic 5", test_func)
            self.results.append(result)

    async def run_epic_6_tests(self):
        """Nautilus Engine Integration Tests"""
        tests = [
            ("UAT-6.1", "Engine Status", self.test_engine_status),
            ("UAT-6.2", "Backtest Status", self.test_backtest_status),
            ("UAT-6.3", "Deployment Status", self.test_deployment_status),
            ("UAT-6.4", "Data Pipeline Status", self.test_data_pipeline_status),
        ]
        
        for test_id, test_name, test_func in tests:
            result = await self.execute_test(test_id, test_name, "Epic 6", test_func)
            self.results.append(result)

    async def run_integration_tests(self):
        """Integration and Performance Tests"""
        tests = [
            ("UAT-INT-1", "IB Connection", self.test_ib_connection),
            ("UAT-INT-2", "IB Market Data", self.test_ib_market_data),
            ("UAT-INT-3", "IB Orders", self.test_ib_orders),
            ("UAT-INT-4", "IB Positions", self.test_ib_positions),
            ("UAT-PERF-1", "API Response Time", self.test_api_response_time),
            ("UAT-PERF-2", "Frontend Accessibility", self.test_frontend_accessibility),
        ]
        
        for test_id, test_name, test_func in tests:
            result = await self.execute_test(test_id, test_name, "Integration", test_func)
            self.results.append(result)

    async def run_all_tests(self):
        """Execute all UAT test suites"""
        logger.info("🚀 Starting UAT Execution for 21 Production-Ready Stories")
        logger.info("=" * 60)
        
        start_time = time.time()
        
        # Run all test suites
        await self.run_epic_1_tests()
        await self.run_epic_2_tests()
        await self.run_epic_3_tests()
        await self.run_epic_4_tests()
        await self.run_epic_5_tests()
        await self.run_epic_6_tests()
        await self.run_integration_tests()
        
        total_time = time.time() - start_time
        
        # Generate summary report
        self.generate_summary_report(total_time)

    def generate_summary_report(self, total_time: float):
        """Generate comprehensive UAT summary report"""
        passed = len([r for r in self.results if r.status == TestStatus.PASSED])
        failed = len([r for r in self.results if r.status == TestStatus.FAILED])
        total = len(self.results)
        pass_rate = (passed / total * 100) if total > 0 else 0
        
        # Console report
        print("\n" + "=" * 60)
        print("🎯 UAT EXECUTION SUMMARY REPORT")
        print("=" * 60)
        print(f"📊 Total Tests:    {total}")
        print(f"✅ Passed:        {passed}")
        print(f"❌ Failed:        {failed}")
        print(f"📈 Pass Rate:     {pass_rate:.1f}%")
        print(f"⏱️  Total Time:    {total_time:.2f} seconds")
        print()
        
        # Epic breakdown
        epics = {}
        for result in self.results:
            if result.epic not in epics:
                epics[result.epic] = {"passed": 0, "failed": 0, "total": 0}
            epics[result.epic]["total"] += 1
            if result.status == TestStatus.PASSED:
                epics[result.epic]["passed"] += 1
            else:
                epics[result.epic]["failed"] += 1
        
        print("📋 EPIC BREAKDOWN:")
        for epic, stats in epics.items():
            epic_pass_rate = (stats["passed"] / stats["total"] * 100) if stats["total"] > 0 else 0
            print(f"  {epic}: {stats['passed']}/{stats['total']} ({epic_pass_rate:.1f}%)")
        
        # Failed tests
        failed_tests = [r for r in self.results if r.status == TestStatus.FAILED]
        if failed_tests:
            print("\n❌ FAILED TESTS:")
            for test in failed_tests:
                print(f"  {test.test_id}: {test.test_name}")
                if test.error_message:
                    print(f"    Error: {test.error_message}")
        
        # Production readiness assessment
        print("\n🎖️  PRODUCTION READINESS ASSESSMENT:")
        if pass_rate >= 95:
            print("✅ READY FOR PRODUCTION - Excellent test coverage")
        elif pass_rate >= 85:
            print("🟡 CONDITIONAL GO - Minor issues to address")
        else:
            print("❌ NOT READY - Critical issues must be resolved")
        
        # Save detailed JSON report
        self.save_json_report()

    def save_json_report(self):
        """Save detailed test results to JSON file"""
        report = {
            "timestamp": datetime.now().isoformat(),
            "summary": {
                "total_tests": len(self.results),
                "passed": len([r for r in self.results if r.status == TestStatus.PASSED]),
                "failed": len([r for r in self.results if r.status == TestStatus.FAILED]),
                "pass_rate": len([r for r in self.results if r.status == TestStatus.PASSED]) / len(self.results) * 100 if self.results else 0
            },
            "test_results": [
                {
                    "test_id": r.test_id,
                    "test_name": r.test_name,
                    "epic": r.epic,
                    "status": r.status.value,
                    "execution_time": r.execution_time,
                    "error_message": r.error_message,
                    "response_data": r.response_data
                }
                for r in self.results
            ]
        }
        
        filename = f"uat-results-{datetime.now().strftime('%Y%m%d-%H%M%S')}.json"
        with open(filename, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"📄 Detailed report saved: {filename}")

async def main():
    """Main UAT execution entry point"""
    async with UATFramework() as uat:
        await uat.run_all_tests()

if __name__ == "__main__":
    asyncio.run(main())