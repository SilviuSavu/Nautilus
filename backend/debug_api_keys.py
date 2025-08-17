#!/usr/bin/env python3
"""
Debug script to check API keys in the running database
"""

import asyncio
import aiohttp
import json

async def test_api_key_login():
    """Test API key login with current running server"""
    
    # Get admin API key from running server
    async with aiohttp.ClientSession() as session:
        try:
            # First try username/password to get a token
            login_data = {
                "username": "admin",
                "password": "admin123"
            }
            
            async with session.post(
                "http://localhost:8002/api/v1/auth/login",
                json=login_data,
                headers={"Content-Type": "application/json"}
            ) as resp:
                if resp.status == 200:
                    result = await resp.json()
                    print("✅ Username/password login successful")
                    print(f"Token: {result['access_token'][:50]}...")
                else:
                    print(f"❌ Username/password login failed: {resp.status}")
                    text = await resp.text()
                    print(f"Response: {text}")
                    return
            
            # Now try to get user info to see if we can find API key info
            token = result['access_token']
            async with session.get(
                "http://localhost:8002/api/v1/auth/me",
                headers={"Authorization": f"Bearer {token}"}
            ) as resp:
                if resp.status == 200:
                    user_info = await resp.json()
                    print("✅ User info retrieved")
                    print(f"User: {json.dumps(user_info, indent=2)}")
                else:
                    print(f"❌ Failed to get user info: {resp.status}")
            
            # Test the API keys we know about
            api_keys_to_test = [
                "QSHjFuFH7U-gk70KRjRZFwTcK-Fj1hvRla-ow_hM2uQ",  # Latest from script
                "bqQdnz0aVzu_Y_BiNGwuxf3YjREd54n_BdzKGDR1r_M"   # Previous one
            ]
            
            for api_key in api_keys_to_test:
                print(f"\n🔑 Testing API key: {api_key[:20]}...")
                async with session.post(
                    "http://localhost:8002/api/v1/auth/login",
                    json={"api_key": api_key},
                    headers={"Content-Type": "application/json"}
                ) as resp:
                    if resp.status == 200:
                        result = await resp.json()
                        print("✅ API key login successful!")
                        print(f"Token: {result['access_token'][:50]}...")
                        return api_key
                    else:
                        print(f"❌ API key login failed: {resp.status}")
                        text = await resp.text()
                        print(f"Response: {text}")
                        
        except Exception as e:
            print(f"❌ Error: {e}")
            
    return None

if __name__ == "__main__":
    asyncio.run(test_api_key_login())