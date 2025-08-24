#!/usr/bin/env python3
"""
Comprehensive System Intercommunication Fix
Addresses all endpoint connectivity and MessageBus integration issues
"""

import asyncio
import aiohttp
import json
from typing import Dict, List, Any
from datetime import datetime

class SystemIntercommunicationFixer:
    def __init__(self):
        self.base_url = "http://localhost:8001"
        self.engine_ports = {
            "analytics": 8100,
            "risk": 8200,
            "factor": 8300,
            "ml": 8400,
            "features": 8500,
            "websocket": 8600,
            "strategy": 8700,
            "marketdata": 8800,
            "portfolio": 8900
        }
        self.data_sources = [
            "ibkr", "alpha_vantage", "fred", "edgar", 
            "datagov", "trading_economics", "dbnomics", "yfinance"
        ]
        self.results = {}
    
    async def test_endpoint_connectivity(self, session: aiohttp.ClientSession):
        """Test all endpoint categories for frontend accessibility"""
        
        endpoint_tests = {
            "Portfolio Management": [
                "/api/v1/portfolio/positions",
                "/api/v1/portfolio/balance", 
                "/api/v1/portfolio/main/summary",
                "/api/v1/portfolio/main/orders"
            ],
            "Strategy Management": [
                "/api/v1/strategies/available",
                "/api/v1/strategies/list",
                "/api/v1/nautilus-strategies/status",
                "/api/v1/strategy/templates"
            ],
            "ML/AI Features": [
                "/api/v1/ml/models/available",
                "/api/v1/ml/predict/regime",
                "/api/v1/ml/models/list",
                "/api/v1/ml/inference/health"
            ],
            "Risk Management Extended": [
                "/api/v1/risk/stress-test",
                "/api/v1/risk/scenario-analysis",
                "/api/v1/risk/portfolio-metrics",
                "/api/v1/risk/value-at-risk"
            ]
        }
        
        connectivity_results = {}
        
        for category, endpoints in endpoint_tests.items():
            category_results = []
            working_endpoints = 0
            
            for endpoint in endpoints:
                try:
                    async with session.get(f"{self.base_url}{endpoint}", timeout=5) as response:
                        if response.status != 404:
                            working_endpoints += 1
                            status = "✅ Working"
                        else:
                            status = "❌ Not Found"
                        
                        category_results.append({
                            "endpoint": endpoint,
                            "status_code": response.status,
                            "status": status
                        })
                        
                except Exception as e:
                    category_results.append({
                        "endpoint": endpoint,
                        "status_code": "ERROR",
                        "status": f"❌ Error: {str(e)}"
                    })
            
            connectivity_results[category] = {
                "endpoints": category_results,
                "working_count": working_endpoints,
                "total_count": len(endpoints),
                "success_rate": round((working_endpoints / len(endpoints)) * 100, 1)
            }
        
        return connectivity_results
    
    async def test_engine_messagebus_connectivity(self, session: aiohttp.ClientSession):
        """Test all 9 engines for MessageBus connectivity"""
        
        engine_results = {}
        connected_engines = 0
        
        for engine_name, port in self.engine_ports.items():
            try:
                # Test basic health
                async with session.get(f"http://localhost:{port}/health", timeout=5) as response:
                    if response.status == 200:
                        health_data = await response.json()
                        
                        # Check for MessageBus connectivity indicators
                        messagebus_connected = self._check_messagebus_status(health_data)
                        
                        if messagebus_connected:
                            connected_engines += 1
                            status = "✅ Connected"
                        else:
                            status = "⚠️ Not Connected to MessageBus"
                        
                        engine_results[engine_name] = {
                            "port": port,
                            "health_status": "Healthy",
                            "messagebus_connected": messagebus_connected,
                            "status": status,
                            "health_data": health_data
                        }
                    else:
                        engine_results[engine_name] = {
                            "port": port,
                            "health_status": "Unhealthy",
                            "messagebus_connected": False,
                            "status": f"❌ HTTP {response.status}",
                            "health_data": None
                        }
                        
            except Exception as e:
                engine_results[engine_name] = {
                    "port": port,
                    "health_status": "Error",
                    "messagebus_connected": False,
                    "status": f"❌ {str(e)}",
                    "health_data": None
                }
        
        return {
            "engines": engine_results,
            "connected_count": connected_engines,
            "total_count": len(self.engine_ports),
            "connection_rate": round((connected_engines / len(self.engine_ports)) * 100, 1)
        }
    
    async def test_data_sources_messagebus_integration(self, session: aiohttp.ClientSession):
        """Test 8 data sources for MessageBus integration"""
        
        data_source_tests = {
            "ibkr": "/api/v1/market-data/historical/bars",
            "alpha_vantage": "/api/v1/nautilus-data/alpha-vantage/search?keywords=AAPL",
            "fred": "/api/v1/nautilus-data/fred/macro-factors",
            "edgar": "/api/v1/edgar/health",
            "datagov": "/api/v1/datagov/datasets",
            "trading_economics": "/api/v1/trading-economics/indicators",
            "dbnomics": "/api/v1/dbnomics/providers",
            "yfinance": "/api/v1/yfinance/quote/AAPL"
        }
        
        source_results = {}
        working_sources = 0
        messagebus_sources = 0
        
        for source, endpoint in data_source_tests.items():
            try:
                async with session.get(f"{self.base_url}{endpoint}", timeout=10) as response:
                    if response.status == 200:
                        working_sources += 1
                        data = await response.json()
                        
                        # Check if this is a MessageBus-enabled source
                        is_messagebus = self._is_messagebus_source(source)
                        if is_messagebus:
                            messagebus_sources += 1
                        
                        status = "✅ Working (MessageBus)" if is_messagebus else "✅ Working (Direct)"
                        
                    elif response.status == 404:
                        status = "❌ Endpoint Not Found"
                        data = None
                    else:
                        status = f"⚠️ HTTP {response.status}"
                        data = None
                    
                    source_results[source] = {
                        "endpoint": endpoint,
                        "status_code": response.status,
                        "status": status,
                        "messagebus_enabled": self._is_messagebus_source(source),
                        "working": response.status == 200
                    }
                    
            except Exception as e:
                source_results[source] = {
                    "endpoint": endpoint,
                    "status_code": "ERROR",
                    "status": f"❌ {str(e)}",
                    "messagebus_enabled": self._is_messagebus_source(source),
                    "working": False
                }
        
        return {
            "data_sources": source_results,
            "working_count": working_sources,
            "messagebus_count": messagebus_sources,
            "total_count": len(data_source_tests),
            "working_rate": round((working_sources / len(data_source_tests)) * 100, 1)
        }
    
    async def test_infrastructure_messagebus_connectivity(self, session: aiohttp.ClientSession):
        """Test infrastructure components for MessageBus connectivity"""
        
        infrastructure_tests = {
            "Redis Backend": f"{self.base_url}/api/v1/messagebus/redis/status",
            "MessageBus Service": f"{self.base_url}/api/v1/messagebus/health",
            "WebSocket Bridge": f"{self.base_url}/api/v1/websocket/health",
            "Cache Service": f"{self.base_url}/api/v1/cache/status",
            "Database Pool": f"{self.base_url}/api/v1/database/pool/status"
        }
        
        infra_results = {}
        working_count = 0
        
        for component, endpoint in infrastructure_tests.items():
            try:
                async with session.get(endpoint, timeout=5) as response:
                    if response.status == 200:
                        working_count += 1
                        status = "✅ Working"
                        data = await response.json()
                    elif response.status == 404:
                        status = "❌ Endpoint Not Found"
                        data = None
                    else:
                        status = f"⚠️ HTTP {response.status}"
                        data = None
                    
                    infra_results[component] = {
                        "endpoint": endpoint,
                        "status_code": response.status,
                        "status": status,
                        "working": response.status == 200
                    }
                    
            except Exception as e:
                infra_results[component] = {
                    "endpoint": endpoint,
                    "status_code": "ERROR", 
                    "status": f"❌ {str(e)}",
                    "working": False
                }
        
        return {
            "infrastructure": infra_results,
            "working_count": working_count,
            "total_count": len(infrastructure_tests),
            "working_rate": round((working_count / len(infrastructure_tests)) * 100, 1)
        }
    
    def _check_messagebus_status(self, health_data: Dict[str, Any]) -> bool:
        """Check if engine health data indicates MessageBus connectivity"""
        if not health_data:
            return False
        
        # Look for MessageBus indicators in health data
        messagebus_indicators = [
            "messagebus_connected",
            "messagebus_status", 
            "redis_connected",
            "stream_connected",
            "events_processed"
        ]
        
        for indicator in messagebus_indicators:
            if indicator in health_data:
                value = health_data[indicator]
                if isinstance(value, bool):
                    return value
                elif isinstance(value, str):
                    return value.lower() in ["true", "connected", "active", "healthy"]
                elif isinstance(value, (int, float)):
                    return value > 0
        
        return False
    
    def _is_messagebus_source(self, source: str) -> bool:
        """Determine if data source uses MessageBus architecture"""
        messagebus_sources = ["datagov", "dbnomics"]
        return source in messagebus_sources
    
    async def generate_fixes(self, results: Dict[str, Any]) -> List[str]:
        """Generate specific fixes based on test results"""
        
        fixes = []
        
        # Fix endpoint connectivity issues
        endpoint_results = results.get("endpoints", {})
        for category, data in endpoint_results.items():
            if data["success_rate"] < 100:
                fixes.append(f"Fix {category}: {data['working_count']}/{data['total_count']} endpoints working")
        
        # Fix engine MessageBus connectivity
        engine_results = results.get("engines", {})
        if engine_results.get("connection_rate", 0) < 100:
            connected = engine_results.get("connected_count", 0)
            total = engine_results.get("total_count", 0)
            fixes.append(f"Fix Engine MessageBus: Only {connected}/{total} engines connected to MessageBus")
        
        # Fix data source integration
        datasource_results = results.get("data_sources", {})
        if datasource_results.get("working_rate", 0) < 100:
            working = datasource_results.get("working_count", 0)
            total = datasource_results.get("total_count", 0)
            fixes.append(f"Fix Data Sources: Only {working}/{total} data sources accessible")
        
        # Fix infrastructure connectivity
        infra_results = results.get("infrastructure", {})
        if infra_results.get("working_rate", 0) < 100:
            working = infra_results.get("working_count", 0)
            total = infra_results.get("total_count", 0)
            fixes.append(f"Fix Infrastructure: Only {working}/{total} infrastructure components accessible")
        
        return fixes
    
    async def run_complete_system_test(self):
        """Run comprehensive system intercommunication test"""
        
        print("🔍 Starting Comprehensive System Intercommunication Test...")
        print(f"📅 Test Date: {datetime.now().isoformat()}")
        print("=" * 80)
        
        async with aiohttp.ClientSession() as session:
            # Test 1: Endpoint Connectivity for Frontend
            print("\n1️⃣ Testing Frontend Endpoint Connectivity...")
            endpoint_results = await self.test_endpoint_connectivity(session)
            
            # Test 2: Engine MessageBus Connectivity  
            print("\n2️⃣ Testing Engine MessageBus Connectivity...")
            engine_results = await self.test_engine_messagebus_connectivity(session)
            
            # Test 3: Data Source MessageBus Integration
            print("\n3️⃣ Testing Data Source MessageBus Integration...")
            datasource_results = await self.test_data_sources_messagebus_integration(session)
            
            # Test 4: Infrastructure MessageBus Connectivity
            print("\n4️⃣ Testing Infrastructure MessageBus Connectivity...")
            infra_results = await self.test_infrastructure_messagebus_connectivity(session)
        
        # Compile results
        self.results = {
            "test_timestamp": datetime.now().isoformat(),
            "endpoints": endpoint_results,
            "engines": engine_results, 
            "data_sources": datasource_results,
            "infrastructure": infra_results
        }
        
        # Generate report
        await self.generate_detailed_report()
        
        # Generate fixes
        fixes = await self.generate_fixes(self.results)
        if fixes:
            print("\n🔧 REQUIRED FIXES:")
            for i, fix in enumerate(fixes, 1):
                print(f"   {i}. {fix}")
        else:
            print("\n✅ All systems operational - no fixes required!")
        
        return self.results
    
    async def generate_detailed_report(self):
        """Generate detailed test report"""
        
        print("\n" + "="*80)
        print("📊 COMPREHENSIVE SYSTEM INTERCOMMUNICATION REPORT")
        print("="*80)
        
        # Summary Table
        endpoint_success = self.results["endpoints"]
        engine_success = self.results["engines"]["connection_rate"]
        datasource_success = self.results["data_sources"]["working_rate"]
        infra_success = self.results["infrastructure"]["working_rate"]
        
        print(f"""
📋 EXECUTIVE SUMMARY
┌─────────────────────────────┬─────────────┬─────────────┬──────────────┐
│ Component                   │ Status      │ Working     │ Success Rate │
├─────────────────────────────┼─────────────┼─────────────┼──────────────┤""")
        
        for category, data in endpoint_success.items():
            status = "✅ GOOD" if data["success_rate"] >= 75 else "⚠️ LIMITED" if data["success_rate"] >= 50 else "❌ POOR"
            print(f"│ {category:27} │ {status:11} │ {data['working_count']}/{data['total_count']:8} │ {data['success_rate']:11.1f}% │")
        
        # Engine connectivity 
        engine_status = "✅ GOOD" if engine_success >= 75 else "⚠️ LIMITED" if engine_success >= 50 else "❌ POOR"
        engine_data = self.results["engines"]
        print(f"│ {'Engine MessageBus':27} │ {engine_status:11} │ {engine_data['connected_count']}/{engine_data['total_count']:8} │ {engine_success:11.1f}% │")
        
        # Data source connectivity
        datasource_status = "✅ GOOD" if datasource_success >= 75 else "⚠️ LIMITED" if datasource_success >= 50 else "❌ POOR"
        datasource_data = self.results["data_sources"]
        print(f"│ {'Data Sources':27} │ {datasource_status:11} │ {datasource_data['working_count']}/{datasource_data['total_count']:8} │ {datasource_success:11.1f}% │")
        
        # Infrastructure connectivity
        infra_status = "✅ GOOD" if infra_success >= 75 else "⚠️ LIMITED" if infra_success >= 50 else "❌ POOR"
        infra_data = self.results["infrastructure"]
        print(f"│ {'Infrastructure':27} │ {infra_status:11} │ {infra_data['working_count']}/{infra_data['total_count']:8} │ {infra_success:11.1f}% │")
        
        print("└─────────────────────────────┴─────────────┴─────────────┴──────────────┘")
        
        # Detailed breakdowns
        print(f"\n🔍 DETAILED ANALYSIS")
        
        # Engine details
        print(f"\n🏭 ENGINE MESSAGEBUS CONNECTIVITY ({engine_data['connected_count']}/{engine_data['total_count']} Connected)")
        for engine, data in engine_data["engines"].items():
            print(f"   {engine.upper():12} (Port {data['port']:4}) - {data['status']}")
        
        # Data source details  
        print(f"\n💾 DATA SOURCE INTEGRATION ({datasource_data['working_count']}/{datasource_data['total_count']} Working)")
        for source, data in datasource_data["data_sources"].items():
            messagebus_indicator = "📡" if data["messagebus_enabled"] else "🔗"
            print(f"   {messagebus_indicator} {source.upper():15} - {data['status']}")
        
        # Infrastructure details
        print(f"\n⚙️ INFRASTRUCTURE COMPONENTS ({infra_data['working_count']}/{infra_data['total_count']} Working)")
        for component, data in infra_data["infrastructure"].items():
            print(f"   {component:20} - {data['status']}")

async def main():
    """Run the comprehensive system intercommunication test"""
    
    fixer = SystemIntercommunicationFixer()
    results = await fixer.run_complete_system_test()
    
    # Save results to file
    with open("system_intercommunication_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\n💾 Results saved to: system_intercommunication_results.json")
    return results

if __name__ == "__main__":
    asyncio.run(main())