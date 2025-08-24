#!/usr/bin/env python3
"""
Corrected Comprehensive System Intercommunication Test
Uses actual endpoint paths that exist in the system
"""

import asyncio
import aiohttp
import json
from typing import Dict, List, Any
from datetime import datetime

class CorrectedSystemTest:
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
        
    async def test_corrected_endpoint_connectivity(self, session: aiohttp.ClientSession):
        """Test endpoints using actual paths that exist in the system"""
        
        # These are the ACTUAL endpoints that exist based on route files
        actual_endpoint_tests = {
            "Portfolio Management": [
                "/api/v1/portfolio/positions",           # ✅ Works (tested)
                "/api/v1/portfolio/balance",             # ✅ Works (tested) 
                "/api/v1/portfolio/main/summary",        # ✅ Works (tested)
                "/api/v1/portfolio/main/orders"          # ✅ Works (tested)
            ],
            "Strategy Management": [
                "/api/v1/strategies/templates",          # ✅ Works (tested)
                "/api/v1/strategies/configurations",     # ✅ Works (tested)
                "/api/v1/strategies/health",             # ✅ Works (tested)
                "/api/v1/strategies/active"              # From routes file
            ],
            "ML/AI Features": [
                "/api/v1/ml/status",                     # Has implementation issues
                "/api/v1/ml/inference/models",           # Has implementation issues
                "/api/v1/ml/config",                     # Has implementation issues
                "/api/v1/ml/regime/current"              # From routes file
            ],
            "Risk Management": [
                "/api/v1/risk/health",                   # Basic risk endpoint
                "/api/v1/risk/calculate-var",            # From risk routes
                "/api/v1/risk/breach-detection",         # From risk routes
                "/api/v1/risk/monitoring/metrics"        # From risk routes
            ]
        }
        
        connectivity_results = {}
        
        for category, endpoints in actual_endpoint_tests.items():
            category_results = []
            working_endpoints = 0
            
            for endpoint in endpoints:
                try:
                    async with session.get(f"{self.base_url}{endpoint}", timeout=10) as response:
                        if response.status == 200:
                            working_endpoints += 1
                            status = "✅ Working"
                        elif response.status == 422:
                            working_endpoints += 1  # Endpoint exists, just needs parameters
                            status = "✅ Exists (needs params)"
                        elif response.status == 404:
                            status = "❌ Not Found"
                        elif response.status == 500:
                            status = "⚠️ Server Error (implementation issue)"
                        else:
                            status = f"⚠️ HTTP {response.status}"
                        
                        category_results.append({
                            "endpoint": endpoint,
                            "status_code": response.status,
                            "status": status,
                            "working": response.status in [200, 422]
                        })
                        
                except Exception as e:
                    category_results.append({
                        "endpoint": endpoint,
                        "status_code": "ERROR",
                        "status": f"❌ Error: {str(e)}",
                        "working": False
                    })
            
            connectivity_results[category] = {
                "endpoints": category_results,
                "working_count": working_endpoints,
                "total_count": len(endpoints),
                "success_rate": round((working_endpoints / len(endpoints)) * 100, 1)
            }
        
        return connectivity_results

    async def test_engine_status_detailed(self, session: aiohttp.ClientSession):
        """Get detailed status of all 9 engines"""
        
        engine_status = {}
        healthy_engines = 0
        
        for engine_name, port in self.engine_ports.items():
            try:
                async with session.get(f"http://localhost:{port}/health", timeout=5) as response:
                    if response.status == 200:
                        healthy_engines += 1
                        health_data = await response.json()
                        
                        # Try to get additional metrics if available
                        try:
                            async with session.get(f"http://localhost:{port}/metrics", timeout=3) as metrics_response:
                                if metrics_response.status == 200:
                                    metrics_data = await metrics_response.json()
                                    health_data["metrics"] = metrics_data
                        except:
                            pass  # Metrics endpoint not available
                        
                        engine_status[engine_name] = {
                            "port": port,
                            "status": "✅ Healthy",
                            "accessible": True,
                            "health_data": health_data
                        }
                    else:
                        engine_status[engine_name] = {
                            "port": port,
                            "status": f"⚠️ HTTP {response.status}",
                            "accessible": False,
                            "health_data": None
                        }
                        
            except Exception as e:
                engine_status[engine_name] = {
                    "port": port,
                    "status": f"❌ {str(e)}",
                    "accessible": False,
                    "health_data": None
                }
        
        return {
            "engines": engine_status,
            "healthy_count": healthy_engines,
            "total_count": len(self.engine_ports),
            "availability_rate": round((healthy_engines / len(self.engine_ports)) * 100, 1)
        }

    async def test_data_source_detailed_status(self, session: aiohttp.ClientSession):
        """Test data sources with correct endpoints"""
        
        # Use endpoints that actually exist and work
        working_data_sources = {
            "alpha_vantage": "/api/v1/nautilus-data/alpha-vantage/search?keywords=AAPL",
            "fred": "/api/v1/nautilus-data/fred/macro-factors", 
            "edgar": "/api/v1/edgar/health",
            "trading_economics": "/api/v1/trading-economics/indicators",
            "dbnomics": "/api/v1/dbnomics/providers"
        }
        
        # Test problematic endpoints separately
        problematic_data_sources = {
            "ibkr": "/api/v1/market-data/historical/bars",  # Needs parameters
            "datagov": "/api/v1/datagov/datasets",         # Endpoint not found
            "yfinance": "/api/v1/yfinance/quote/AAPL"      # May have issues
        }
        
        all_sources = {**working_data_sources, **problematic_data_sources}
        source_results = {}
        working_count = 0
        
        for source_name, endpoint in all_sources.items():
            try:
                async with session.get(f"{self.base_url}{endpoint}", timeout=10) as response:
                    if response.status == 200:
                        working_count += 1
                        status = "✅ Working"
                        data = await response.json()
                        has_data = len(str(data)) > 50  # Has substantial response
                    elif response.status == 422:
                        # Endpoint exists but needs parameters
                        working_count += 0.5  # Half credit
                        status = "⚠️ Needs Parameters"
                        has_data = False
                    elif response.status == 404:
                        status = "❌ Not Found"
                        has_data = False
                    else:
                        status = f"⚠️ HTTP {response.status}"
                        has_data = False
                    
                    source_results[source_name] = {
                        "endpoint": endpoint,
                        "status_code": response.status,
                        "status": status,
                        "working": response.status == 200,
                        "has_data": has_data
                    }
                    
            except Exception as e:
                source_results[source_name] = {
                    "endpoint": endpoint,
                    "status_code": "ERROR",
                    "status": f"❌ {str(e)}",
                    "working": False,
                    "has_data": False
                }
        
        return {
            "data_sources": source_results,
            "working_count": int(working_count),
            "total_count": len(all_sources),
            "success_rate": round((working_count / len(all_sources)) * 100, 1)
        }
    
    async def run_corrected_system_test(self):
        """Run the corrected comprehensive system test"""
        
        print("🔧 Running CORRECTED System Intercommunication Test")
        print("=" * 70)
        print(f"📅 Test Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("🎯 Using ACTUAL endpoint paths from route definitions")
        print("=" * 70)
        
        async with aiohttp.ClientSession() as session:
            # Test 1: Corrected Endpoint Connectivity
            print("\n1️⃣ Testing Frontend Endpoint Connectivity (Corrected Paths)")
            endpoint_results = await self.test_corrected_endpoint_connectivity(session)
            
            # Test 2: Detailed Engine Status
            print("\n2️⃣ Testing Engine Health and Accessibility")
            engine_results = await self.test_engine_status_detailed(session)
            
            # Test 3: Detailed Data Source Status
            print("\n3️⃣ Testing Data Source Integration (Corrected Endpoints)")
            datasource_results = await self.test_data_source_detailed_status(session)
        
        # Generate corrected report
        await self.generate_corrected_report(endpoint_results, engine_results, datasource_results)
        
        return {
            "test_timestamp": datetime.now().isoformat(),
            "endpoints": endpoint_results,
            "engines": engine_results,
            "data_sources": datasource_results
        }
    
    async def generate_corrected_report(self, endpoint_results, engine_results, datasource_results):
        """Generate corrected detailed report"""
        
        print("\n" + "="*70)
        print("📊 CORRECTED SYSTEM INTERCOMMUNICATION REPORT")
        print("="*70)
        
        # Corrected summary table
        print(f"""
🎯 CORRECTED SYSTEM STATUS (Using Actual Endpoint Paths)
┌─────────────────────────────┬─────────────┬─────────────┬──────────────┐
│ Component Category          │ Status      │ Working     │ Success Rate │
├─────────────────────────────┼─────────────┼─────────────┼──────────────┤""")
        
        for category, data in endpoint_results.items():
            if data["success_rate"] >= 90:
                status = "✅ EXCELLENT"
            elif data["success_rate"] >= 75:
                status = "✅ GOOD"
            elif data["success_rate"] >= 50:
                status = "⚠️ LIMITED"
            else:
                status = "❌ POOR"
            
            print(f"│ {category:27} │ {status:11} │ {data['working_count']}/{data['total_count']:8} │ {data['success_rate']:11.1f}% │")
        
        # Engine status
        engine_success = engine_results["availability_rate"]
        if engine_success >= 90:
            engine_status = "✅ EXCELLENT"
        elif engine_success >= 75:
            engine_status = "✅ GOOD" 
        elif engine_success >= 50:
            engine_status = "⚠️ LIMITED"
        else:
            engine_status = "❌ POOR"
        
        print(f"│ {'Processing Engines':27} │ {engine_status:11} │ {engine_results['healthy_count']}/{engine_results['total_count']:8} │ {engine_success:11.1f}% │")
        
        # Data source status
        datasource_success = datasource_results["success_rate"]
        if datasource_success >= 90:
            datasource_status = "✅ EXCELLENT"
        elif datasource_success >= 75:
            datasource_status = "✅ GOOD"
        elif datasource_success >= 50:
            datasource_status = "⚠️ LIMITED"
        else:
            datasource_status = "❌ POOR"
        
        print(f"│ {'Data Source Integration':27} │ {datasource_status:11} │ {datasource_results['working_count']}/{datasource_results['total_count']:8} │ {datasource_success:11.1f}% │")
        
        print("└─────────────────────────────┴─────────────┴─────────────┴──────────────┘")
        
        # Detailed breakdowns
        print(f"\n🔍 DETAILED COMPONENT STATUS")
        
        # Endpoint details with actual paths
        print(f"\n📡 FRONTEND ENDPOINT CONNECTIVITY")
        for category, data in endpoint_results.items():
            print(f"\n   {category} ({data['working_count']}/{data['total_count']} Working):")
            for endpoint_info in data["endpoints"]:
                print(f"      {endpoint_info['endpoint']:40} - {endpoint_info['status']}")
        
        # Engine details
        print(f"\n🏭 PROCESSING ENGINE STATUS ({engine_results['healthy_count']}/{engine_results['total_count']} Healthy)")
        for engine, data in engine_results["engines"].items():
            print(f"   {engine.upper():12} (Port {data['port']:4}) - {data['status']}")
        
        # Data source details  
        print(f"\n💾 DATA SOURCE INTEGRATION ({datasource_results['working_count']}/{datasource_results['total_count']} Working)")
        for source, data in datasource_results["data_sources"].items():
            has_data_indicator = "📊" if data.get("has_data", False) else "📄"
            print(f"   {has_data_indicator} {source.upper():15} - {data['status']}")
        
        # Summary assessment
        overall_score = (
            sum(d["success_rate"] for d in endpoint_results.values()) / len(endpoint_results) * 0.4 +
            engine_results["availability_rate"] * 0.4 +
            datasource_results["success_rate"] * 0.2
        )
        
        if overall_score >= 90:
            overall_status = "✅ EXCELLENT"
        elif overall_score >= 75:
            overall_status = "✅ GOOD"
        elif overall_score >= 50:
            overall_status = "⚠️ NEEDS IMPROVEMENT"
        else:
            overall_status = "❌ CRITICAL ISSUES"
        
        print(f"\n🎯 OVERALL SYSTEM HEALTH: {overall_status} ({overall_score:.1f}%)")


async def main():
    """Run the corrected comprehensive system test"""
    
    tester = CorrectedSystemTest()
    results = await tester.run_corrected_system_test()
    
    # Save corrected results
    with open("corrected_system_test_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\n💾 Corrected results saved to: corrected_system_test_results.json")
    return results

if __name__ == "__main__":
    asyncio.run(main())