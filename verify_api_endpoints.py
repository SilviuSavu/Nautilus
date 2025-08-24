#!/usr/bin/env python3
"""
API Endpoints Verification Script
=================================

Verifies that all documented Enhanced Risk Engine API endpoints 
are properly implemented in the enhanced_risk_api.py file.
"""

import sys
import os

# Add risk engine path
sys.path.insert(0, '/Users/savusilviu/Desktop/SilviuCorneliuSavu/Nautilus/backend/engines/risk')

def verify_api_endpoints():
    """Verify all Enhanced Risk API endpoints are implemented"""
    
    print("🔍 ENHANCED RISK API ENDPOINTS VERIFICATION")
    print("=" * 50)
    
    try:
        from enhanced_risk_api import router
        
        # Get all routes from the router
        routes = []
        for route in router.routes:
            if hasattr(route, 'path') and hasattr(route, 'methods'):
                for method in route.methods:
                    if method not in ['HEAD', 'OPTIONS']:  # Skip utility methods
                        routes.append(f"{method} {route.path}")
        
        print(f"📋 Found {len(routes)} API routes:")
        print("-" * 30)
        
        for route in sorted(routes):
            print(f"✅ {route}")
        
        # Expected endpoints from documentation
        expected_endpoints = [
            "GET /api/v1/enhanced-risk/health",
            "GET /api/v1/enhanced-risk/system/metrics", 
            "POST /api/v1/enhanced-risk/backtest/run",
            "GET /api/v1/enhanced-risk/backtest/results/{backtest_id}",
            "POST /api/v1/enhanced-risk/data/store",
            "GET /api/v1/enhanced-risk/data/retrieve/{symbol}",
            "POST /api/v1/enhanced-risk/xva/calculate",
            "GET /api/v1/enhanced-risk/xva/results/{calculation_id}",
            "POST /api/v1/enhanced-risk/alpha/generate",
            "GET /api/v1/enhanced-risk/alpha/signals/{generation_id}",
            "POST /api/v1/enhanced-risk/hybrid/submit",
            "GET /api/v1/enhanced-risk/hybrid/status/{workload_id}",
            "POST /api/v1/enhanced-risk/dashboard/generate",
            "GET /api/v1/enhanced-risk/dashboard/views"
        ]
        
        print(f"\n📊 ENDPOINT VERIFICATION SUMMARY")
        print("=" * 40)
        
        implemented_count = 0
        for expected in expected_endpoints:
            # Check if endpoint exists (allowing for parameter variations)
            base_endpoint = expected.split("{")[0].rstrip("/")
            method = expected.split(" ")[0]
            
            found = False
            for implemented in routes:
                if method in implemented and base_endpoint in implemented:
                    found = True
                    break
            
            status = "✅ IMPLEMENTED" if found else "❌ MISSING"
            print(f"{status} {expected}")
            
            if found:
                implemented_count += 1
        
        completion_rate = (implemented_count / len(expected_endpoints)) * 100
        
        print(f"\n🎯 API IMPLEMENTATION STATUS")
        print(f"Endpoints Implemented: {implemented_count}/{len(expected_endpoints)}")
        print(f"Completion Rate: {completion_rate:.1f}%")
        
        if completion_rate >= 95:
            print("🏆 STATUS: ✅ FULLY IMPLEMENTED")
            grade = "A+"
        elif completion_rate >= 85:
            print("🏆 STATUS: ✅ WELL IMPLEMENTED") 
            grade = "A"
        elif completion_rate >= 70:
            print("🏆 STATUS: ⚠️ MOSTLY IMPLEMENTED")
            grade = "B+"
        else:
            print("🏆 STATUS: ❌ NEEDS WORK")
            grade = "C"
        
        print(f"📈 GRADE: {grade}")
        
        return completion_rate, grade
        
    except Exception as e:
        print(f"❌ ERROR: Failed to verify API endpoints - {e}")
        import traceback
        traceback.print_exc()
        return 0, "F"

if __name__ == "__main__":
    completion_rate, grade = verify_api_endpoints()
    
    if completion_rate >= 85:
        print(f"\n✅ Enhanced Risk API is production ready!")
        sys.exit(0)
    else:
        print(f"\n⚠️  Enhanced Risk API needs attention")
        sys.exit(1)