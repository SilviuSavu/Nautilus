"""
Test Toraniko Factor Engine Integration
Quick test to ensure the factor engine is properly integrated
"""
import asyncio
import sys
import os
import pytest
from datetime import date, datetime, timedelta
import polars as pl
import numpy as np

# Add the backend directory to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from factor_engine_service import factor_engine_service

@pytest.mark.asyncio
async def test_factor_engine_initialization():
    """Test that the factor engine service initializes correctly"""
    await factor_engine_service.initialize()
    assert factor_engine_service is not None

@pytest.mark.asyncio
async def test_momentum_factor_calculation():
    """Test momentum factor calculation with sample data"""
    # Create sample returns data
    sample_data = []
    symbols = ['AAPL', 'MSFT', 'GOOGL']
    start_date = date(2023, 1, 1)
    
    # Generate 300 days of sample returns for 3 symbols
    for i in range(300):
        current_date = start_date + timedelta(days=i)
        for symbol in symbols:
            # Generate random returns around 0 with some momentum
            base_return = np.random.normal(0.001, 0.02)  # 0.1% daily return with 2% volatility
            sample_data.append({
                'symbol': symbol,
                'date': current_date,
                'asset_returns': base_return
            })
    
    # Convert to Polars DataFrame
    returns_df = pl.DataFrame(sample_data)
    
    # Calculate momentum factor
    momentum_scores = await factor_engine_service.calculate_momentum_factor(
        returns_df, 
        trailing_days=252,
        winsor_factor=0.01
    )
    
    # Verify results
    assert momentum_scores is not None
    assert len(momentum_scores) > 0
    assert 'symbol' in momentum_scores.columns
    assert 'date' in momentum_scores.columns

@pytest.mark.asyncio
async def test_value_factor_calculation():
    """Test value factor calculation with sample fundamental data"""
    # Create sample fundamental data
    sample_data = []
    symbols = ['AAPL', 'MSFT', 'GOOGL']
    start_date = date(2023, 1, 1)
    
    # Generate sample fundamental data
    for i in range(50):  # Less data for fundamentals
        current_date = start_date + timedelta(days=i*7)  # Weekly data
        for symbol in symbols:
            sample_data.append({
                'symbol': symbol,
                'date': current_date,
                'book_price': np.random.uniform(0.1, 0.5),  # Book-to-price ratio
                'sales_price': np.random.uniform(0.05, 0.15),  # Sales-to-price ratio  
                'cf_price': np.random.uniform(0.001, 0.02)  # Cash flow-to-price ratio
            })
    
    # Convert to Polars DataFrame
    fundamentals_df = pl.DataFrame(sample_data)
    
    # Calculate value factor
    value_scores = await factor_engine_service.calculate_value_factor(fundamentals_df)
    
    # Verify results
    assert value_scores is not None
    assert len(value_scores) > 0

def test_toraniko_import():
    """Test that toraniko modules can be imported correctly"""
    try:
        # Add the toraniko path
        toraniko_path = os.path.join(os.path.dirname(__file__), 'engines', 'toraniko')
        sys.path.append(toraniko_path)
        
        from toraniko.model import estimate_factor_returns
        from toraniko.styles import factor_mom, factor_val, factor_sze
        from toraniko.utils import top_n_by_group
        
        # If we get here without errors, imports work
        assert True
        
    except ImportError as e:
        pytest.fail(f"Failed to import toraniko modules: {e}")

async def main():
    """Manual test runner for development"""
    print("🧪 Testing Toraniko Factor Engine Integration")
    
    # Test imports
    print("1. Testing toraniko imports...")
    test_toraniko_import()
    print("✅ Toraniko modules imported successfully")
    
    # Test initialization
    print("2. Testing factor engine initialization...")
    await test_factor_engine_initialization()
    print("✅ Factor engine initialized successfully")
    
    # Test momentum factor
    print("3. Testing momentum factor calculation...")
    await test_momentum_factor_calculation()
    print("✅ Momentum factor calculation successful")
    
    # Test value factor
    print("4. Testing value factor calculation...")
    await test_value_factor_calculation()
    print("✅ Value factor calculation successful")
    
    print("🎉 All tests passed! Toraniko Factor Engine is ready to use.")

if __name__ == "__main__":
    asyncio.run(main())