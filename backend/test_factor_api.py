"""
Quick API test for Factor Engine endpoints
"""
import httpx
import asyncio
import json
from datetime import date, timedelta

async def test_factor_api():
    """Test the factor engine API endpoints"""
    
    base_url = "http://localhost:8000"
    
    try:
        async with httpx.AsyncClient() as client:
            # Test health endpoint
            response = await client.get(f"{base_url}/api/v1/factor-engine/health")
            print(f"Health check: {response.status_code}")
            if response.status_code == 200:
                print(f"Response: {response.json()}")
            
            # Test engine info
            response = await client.get(f"{base_url}/api/v1/factor-engine/info")
            print(f"Engine info: {response.status_code}")
            if response.status_code == 200:
                info = response.json()
                print(f"Engine: {info['name']} v{info['version']}")
                print(f"Capabilities: {len(info['capabilities'])} features")
            
            # Test list factors
            response = await client.get(f"{base_url}/api/v1/factor-engine/factors/list")
            print(f"List factors: {response.status_code}")
            if response.status_code == 200:
                factors = response.json()
                print(f"Available factors: {factors['style_factors']}")
                
    except httpx.ConnectError:
        print("❌ Could not connect to server. Make sure the backend is running on port 8000")
    except Exception as e:
        print(f"❌ Error testing API: {e}")

if __name__ == "__main__":
    print("🧪 Testing Factor Engine API...")
    asyncio.run(test_factor_api())