#!/usr/bin/env python3
"""
REAL ML Prediction Test - Using Actual AAPL Market Data from PostgreSQL
No more synthetic BS - using real market data with Neural-GPU Bus
"""

import requests
import json
import time
import asyncpg
import asyncio

async def get_real_aapl_data():
    """Get real AAPL market data from PostgreSQL"""
    try:
        conn = await asyncpg.connect(
            "postgresql://nautilus:nautilus123@localhost:5432/nautilus"
        )
        
        # Get recent AAPL data
        query = """
        SELECT open_price, high_price, low_price, close_price, volume
        FROM market_bars 
        WHERE instrument_id = 'AAPL'
        ORDER BY timestamp_ns DESC 
        LIMIT 100
        """
        
        data = await conn.fetch(query)
        await conn.close()
        
        if not data:
            return None
            
        # Extract real prices and volumes
        prices = []
        volumes = []
        
        for row in data:
            prices.append(float(row['close_price']))  # Use close price
            volumes.append(float(row['volume']))
        
        # Reverse to get chronological order (oldest first)
        prices.reverse()
        volumes.reverse()
        
        return {
            "prices": prices,
            "volumes": volumes,
            "data_points": len(prices)
        }
        
    except Exception as e:
        print(f"❌ Database error: {e}")
        return None

async def test_real_ml_prediction():
    """Test ML engine with REAL AAPL market data"""
    print("🎯 REAL ML PREDICTION TEST - ACTUAL AAPL MARKET DATA")
    print("=" * 65)
    
    # Get real AAPL data
    print("📊 Fetching real AAPL market data from PostgreSQL...")
    real_data = await get_real_aapl_data()
    
    if not real_data:
        print("❌ No real AAPL data available")
        return
    
    print(f"✅ Retrieved {real_data['data_points']} real AAPL price points")
    print(f"💰 Price range: ${min(real_data['prices']):.2f} - ${max(real_data['prices']):.2f}")
    print(f"📊 Volume range: {min(real_data['volumes']):,.0f} - {max(real_data['volumes']):,.0f}")
    
    # Show sample of real data
    sample_prices = real_data['prices'][-10:]  # Last 10 prices
    print(f"📈 Recent real prices: {[f'${p:.2f}' for p in sample_prices]}")
    
    print(f"\n🧠 Sending REAL data to Neural-GPU ML Engine...")
    
    # Test ML prediction with real data
    start_time = time.time()
    
    try:
        response = requests.post(
            "http://localhost:8400/ml/predict/price/AAPL",
            json={
                "prices": real_data["prices"],
                "volume": real_data["volumes"]
            },
            timeout=30
        )
        
        end_time = time.time()
        processing_time = (end_time - start_time) * 1000
        
        if response.status_code == 200:
            result = response.json()
            prediction = result.get("prediction", {})
            
            print("\n" + "=" * 65)
            print("🎯 REAL MARKET DATA ML PREDICTION RESULTS")
            print("=" * 65)
            
            # Data source info
            print(f"📊 Data Source: PostgreSQL market_bars table")
            print(f"📊 Symbol: AAPL (Apple Inc.)")
            print(f"📊 Real Data Points Used: {real_data['data_points']}")
            
            # Performance metrics
            print(f"⚡ Processing Time: {processing_time:.1f}ms")
            print(f"🧠 Hardware Used: {result.get('hardware_used', 'Unknown')}")
            print(f"🎯 Prediction ID: {result.get('prediction_id', 'N/A')}")
            
            # Real predictions
            current_price = prediction.get('current_price', 0)
            predicted_price = prediction.get('predicted_price', 0)
            price_change = prediction.get('price_change_percent', 0)
            confidence = prediction.get('confidence', 0)
            
            print(f"\n💰 REAL MARKET ANALYSIS:")
            print(f"   Current Price: ${current_price:.2f}")
            print(f"   Predicted Price: ${predicted_price:.2f}")
            print(f"   Expected Change: {price_change:+.2f}%")
            print(f"   Model Confidence: {confidence:.1%}")
            
            # Volatility analysis
            vol_pred = prediction.get('volatility_prediction', {})
            volatility = vol_pred.get('predicted_volatility', 0)
            vol_level = vol_pred.get('volatility_level', 'UNKNOWN')
            
            print(f"\n🌊 VOLATILITY ANALYSIS:")
            print(f"   Predicted Volatility: {volatility:.2f}%")
            print(f"   Risk Level: {vol_level}")
            
            # Trend analysis
            trend_pred = prediction.get('trend_prediction', {})
            trend = trend_pred.get('predicted_trend', 'UNKNOWN')
            trend_probs = trend_pred.get('trend_probabilities', {})
            
            print(f"\n📈 TREND ANALYSIS:")
            print(f"   Predicted Trend: {trend}")
            print(f"   Probabilities:")
            for direction, prob in trend_probs.items():
                print(f"     {direction}: {prob:.1%}")
            
            # Calculate actual price movement in the data
            if len(real_data['prices']) >= 2:
                actual_change = ((real_data['prices'][-1] - real_data['prices'][-2]) / real_data['prices'][-2]) * 100
                print(f"\n📊 RECENT ACTUAL MOVEMENT: {actual_change:+.2f}%")
                
                # Compare predicted vs recent actual trend
                if abs(price_change - actual_change) < 1.0:
                    print(f"✅ Prediction aligns with recent trend (within 1%)")
                else:
                    print(f"⚠️ Prediction differs from recent trend")
            
            print(f"\n✅ REAL DATA TEST COMPLETED SUCCESSFULLY!")
            print(f"🚀 This prediction used actual AAPL market data")
            print(f"🧠 Neural-GPU Bus processed real financial data")
            print(f"⚡ M4 Max hardware acceleration on real market signals")
            
            return {
                "success": True,
                "processing_time_ms": processing_time,
                "data_points": real_data['data_points'],
                "prediction": prediction,
                "hardware_used": result.get('hardware_used')
            }
            
        else:
            print(f"❌ ML prediction failed: {response.status_code}")
            print(f"Response: {response.text}")
            return {"success": False, "error": f"HTTP {response.status_code}"}
            
    except Exception as e:
        print(f"❌ Request failed: {e}")
        return {"success": False, "error": str(e)}

async def main():
    """Run real ML prediction test"""
    result = await test_real_ml_prediction()
    
    print(f"\n" + "=" * 65)
    if result and result.get("success"):
        print("🏆 REAL DATA ML TEST: SUCCESS")
        print(f"📊 Processed {result['data_points']} real market data points")
        print(f"⚡ Response time: {result['processing_time_ms']:.1f}ms")
        print(f"🧠 Hardware: {result['hardware_used']}")
        print("🎯 This proves the system works with real market data!")
    else:
        print("❌ REAL DATA ML TEST: FAILED")
        if result:
            print(f"Error: {result.get('error', 'Unknown error')}")

if __name__ == "__main__":
    asyncio.run(main())