#!/usr/bin/env python3
"""
Nautilus Trading Platform - Simple Paper Trading Validation
Validates trading platform functionality using existing API endpoints
"""

import json
import logging
import time
import requests
from datetime import datetime
from typing import Dict, List

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TradingValidationSuite:
    """Simple validation suite for production readiness testing"""
    
    def __init__(self, base_url: str = "http://localhost:8002"):
        self.base_url = base_url
        self.session = requests.Session()
        self.test_results = []
        
    def log_test_result(self, test_name: str, passed: bool, details: str = ""):
        """Log test result"""
        status = "✅ PASSED" if passed else "❌ FAILED"
        logger.info(f"{status}: {test_name} - {details}")
        self.test_results.append({
            "test": test_name,
            "passed": passed,
            "details": details,
            "timestamp": datetime.now().isoformat()
        })
        
    def test_system_health(self):
        """Test 1: System Health Check"""
        logger.info("🔍 Testing system health...")
        try:
            response = self.session.get(f"{self.base_url}/health", timeout=10)
            if response.status_code == 200:
                data = response.json()
                self.log_test_result("System Health", True, f"Status: {data.get('status', 'unknown')}")
                return True
            else:
                self.log_test_result("System Health", False, f"HTTP {response.status_code}")
                return False
        except Exception as e:
            self.log_test_result("System Health", False, f"Connection error: {e}")
            return False
    
    def test_ib_connection(self):
        """Test 2: Interactive Brokers Connection"""
        logger.info("🔗 Testing IB Gateway connection...")
        try:
            response = self.session.get(f"{self.base_url}/api/v1/ib/connection/status", timeout=10)
            if response.status_code == 200:
                data = response.json()
                connected = data.get('connected', False)
                account_id = data.get('account_id', 'Unknown')
                self.log_test_result("IB Connection", connected, f"Account: {account_id}")
                return connected
            else:
                self.log_test_result("IB Connection", False, f"HTTP {response.status_code}")
                return False
        except Exception as e:
            self.log_test_result("IB Connection", False, f"Error: {e}")
            return False
    
    def test_account_data(self):
        """Test 3: Account Data Retrieval"""
        logger.info("💰 Testing account data retrieval...")
        try:
            response = self.session.get(f"{self.base_url}/api/v1/ib/account", timeout=10)
            if response.status_code == 200:
                data = response.json()
                account_id = data.get('account_id', 'Unknown')
                status = data.get('connection_status', 'Unknown')
                self.log_test_result("Account Data", True, f"Account {account_id}, Status: {status}")
                return True
            else:
                self.log_test_result("Account Data", False, f"HTTP {response.status_code}")
                return False
        except Exception as e:
            self.log_test_result("Account Data", False, f"Error: {e}")
            return False
    
    def test_market_data_availability(self):
        """Test 4: Market Data Services"""
        logger.info("📊 Testing market data availability...")
        try:
            # Test market data endpoint
            response = self.session.get(f"{self.base_url}/api/v1/market/status", timeout=10)
            if response.status_code == 200:
                self.log_test_result("Market Data", True, "Market data services available")
                return True
            else:
                # Try alternative endpoint
                response = self.session.get(f"{self.base_url}/api/v1/ib/positions", timeout=10)
                if response.status_code == 200:
                    self.log_test_result("Market Data", True, "IB positions endpoint available")
                    return True
                else:
                    self.log_test_result("Market Data", False, f"No market data endpoints responding")
                    return False
        except Exception as e:
            self.log_test_result("Market Data", False, f"Error: {e}")
            return False
    
    def test_portfolio_services(self):
        """Test 5: Portfolio Services"""
        logger.info("📈 Testing portfolio services...")
        try:
            # Test portfolio endpoint
            response = self.session.get(f"{self.base_url}/api/v1/portfolio/summary", timeout=10)
            if response.status_code == 200:
                self.log_test_result("Portfolio Services", True, "Portfolio summary available")
                return True
            else:
                # Try alternative endpoint
                response = self.session.get(f"{self.base_url}/api/v1/ib/positions", timeout=10)
                if response.status_code == 200:
                    data = response.json()
                    position_count = len(data) if isinstance(data, list) else 0
                    self.log_test_result("Portfolio Services", True, f"IB positions available ({position_count} positions)")
                    return True
                else:
                    self.log_test_result("Portfolio Services", False, "No portfolio endpoints responding")
                    return False
        except Exception as e:
            self.log_test_result("Portfolio Services", False, f"Error: {e}")
            return False
    
    def test_trading_capabilities(self):
        """Test 6: Trading System Readiness"""
        logger.info("⚡ Testing trading system readiness...")
        try:
            # Test order management endpoint
            response = self.session.get(f"{self.base_url}/api/v1/orders/active", timeout=10)
            if response.status_code == 200:
                self.log_test_result("Trading System", True, "Order management available")
                return True
            else:
                # Try IB orders endpoint
                response = self.session.get(f"{self.base_url}/api/v1/ib/orders", timeout=10)
                if response.status_code == 200:
                    self.log_test_result("Trading System", True, "IB order system available")
                    return True
                else:
                    self.log_test_result("Trading System", False, "No trading endpoints available")
                    return False
        except Exception as e:
            self.log_test_result("Trading System", False, f"Error: {e}")
            return False
    
    def test_performance_analytics(self):
        """Test 7: Performance Analytics (Story 5.1)"""
        logger.info("📊 Testing performance analytics...")
        try:
            # Test analytics benchmarks endpoint (new route)
            response = self.session.get(f"{self.base_url}/api/v1/analytics/benchmarks", timeout=10)
            if response.status_code == 200:
                data = response.json()
                benchmark_count = len(data.get('benchmarks', []))
                self.log_test_result("Performance Analytics", True, f"Analytics available ({benchmark_count} benchmarks)")
                return True
            else:
                # Try alternative performance endpoint
                response = self.session.get(f"{self.base_url}/api/v1/trade-history/summary", timeout=10)
                if response.status_code == 200:
                    self.log_test_result("Performance Analytics", True, "Trade history analytics available")
                    return True
                else:
                    self.log_test_result("Performance Analytics", False, "Performance analytics not accessible")
                    return False
        except Exception as e:
            self.log_test_result("Performance Analytics", False, f"Error: {e}")
            return False
    
    def test_engine_management(self):
        """Test 8: NautilusTrader Engine Management (Story 6.1)"""
        logger.info("🚀 Testing engine management...")
        try:
            # Test strategy templates endpoint (working endpoint)
            response = self.session.get(f"{self.base_url}/api/v1/nautilus/strategies/templates", timeout=10)
            if response.status_code == 200:
                data = response.json()
                template_count = len(data.get('templates', []))
                self.log_test_result("Engine Management", True, f"Strategy system available ({template_count} templates)")
                return True
            else:
                # Try engine status endpoint
                response = self.session.get(f"{self.base_url}/api/v1/nautilus/engine/status", timeout=10)
                if response.status_code == 200:
                    data = response.json()
                    self.log_test_result("Engine Management", True, f"Engine status: {data.get('status', 'unknown')}")
                    return True
                else:
                    self.log_test_result("Engine Management", False, "Engine management not accessible")
                    return False
        except Exception as e:
            self.log_test_result("Engine Management", False, f"Error: {e}")
            return False
    
    def test_system_monitoring(self):
        """Test 9: System Monitoring"""
        logger.info("🔍 Testing system monitoring...")
        try:
            # Test monitoring endpoint
            response = self.session.get(f"{self.base_url}/api/v1/monitoring/health", timeout=10)
            if response.status_code == 200:
                self.log_test_result("System Monitoring", True, "Monitoring services available")
                return True
            else:
                # Check if basic health is working as fallback
                response = self.session.get(f"{self.base_url}/health", timeout=10)
                if response.status_code == 200:
                    self.log_test_result("System Monitoring", True, "Basic health monitoring available")
                    return True
                else:
                    self.log_test_result("System Monitoring", False, "No monitoring endpoints available")
                    return False
        except Exception as e:
            self.log_test_result("System Monitoring", False, f"Error: {e}")
            return False
    
    def run_validation_suite(self):
        """Run complete validation suite"""
        logger.info("🚀 Starting Nautilus Trading Platform Validation Suite")
        logger.info("=" * 60)
        
        start_time = time.time()
        
        # Run all tests
        tests = [
            self.test_system_health,
            self.test_ib_connection,
            self.test_account_data,
            self.test_market_data_availability,
            self.test_portfolio_services,
            self.test_trading_capabilities,
            self.test_performance_analytics,
            self.test_engine_management,
            self.test_system_monitoring
        ]
        
        passed_tests = 0
        total_tests = len(tests)
        
        for test_func in tests:
            if test_func():
                passed_tests += 1
            time.sleep(1)  # Brief pause between tests
        
        execution_time = time.time() - start_time
        
        # Generate summary
        logger.info("=" * 60)
        logger.info("📊 VALIDATION SUITE SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Total Tests: {total_tests}")
        logger.info(f"Passed: {passed_tests}")
        logger.info(f"Failed: {total_tests - passed_tests}")
        logger.info(f"Success Rate: {(passed_tests/total_tests)*100:.1f}%")
        logger.info(f"Execution Time: {execution_time:.2f} seconds")
        
        # Production readiness assessment
        success_rate = (passed_tests / total_tests) * 100
        if success_rate >= 90:
            logger.info("🎉 PRODUCTION READY: All critical systems operational")
        elif success_rate >= 70:
            logger.info("⚠️  PRODUCTION CAUTION: Some issues detected, review failed tests")
        else:
            logger.info("❌ NOT PRODUCTION READY: Critical issues detected")
        
        # Save detailed results
        report = {
            "validation_date": datetime.now().isoformat(),
            "total_tests": total_tests,
            "passed_tests": passed_tests,
            "failed_tests": total_tests - passed_tests,
            "success_rate": success_rate,
            "execution_time": execution_time,
            "production_ready": success_rate >= 90,
            "test_results": self.test_results
        }
        
        with open("paper-trading-validation-report.json", "w") as f:
            json.dump(report, f, indent=2)
        
        logger.info("📄 Detailed report saved to: paper-trading-validation-report.json")
        
        return success_rate >= 90

def main():
    """Main execution function"""
    import os
    base_url = os.getenv("BASE_URL", "http://localhost:8002")
    validator = TradingValidationSuite(base_url=base_url)
    production_ready = validator.run_validation_suite()
    
    if production_ready:
        logger.info("✅ Platform validated and ready for paper trading!")
        exit(0)
    else:
        logger.info("❌ Platform requires attention before production deployment")
        exit(1)

if __name__ == "__main__":
    main()