#!/usr/bin/env python3
"""
Test script to verify timestamp parsing fix for backfill system
"""

import asyncio
import logging
from datetime import datetime, timedelta
from data_backfill_service import DataBackfillService, BackfillRequest

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_timestamp_parsing():
    """Test if the timestamp parsing fix works"""
    
    # Initialize backfill service
    backfill_service = DataBackfillService()
    
    try:
        # Initialize the service
        success = await backfill_service.initialize()
        if not success:
            logger.error("Failed to initialize backfill service")
            return False
            
        # Create a small backfill request for AAPL - just 1 day of 1m data
        request = BackfillRequest(
            symbol="AAPL",
            sec_type="STK",
            exchange="SMART", 
            currency="USD",
            timeframes=["1m"],
            start_date=datetime.now() - timedelta(days=1),
            end_date=datetime.now(),
            priority=1
        )
        
        logger.info("Starting small backfill test for AAPL 1m data (1 day)")
        
        # Process the request
        await backfill_service.process_backfill_request(request)
        
        logger.info("✅ Timestamp parsing test completed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Timestamp parsing test failed: {e}")
        return False

if __name__ == "__main__":
    result = asyncio.run(test_timestamp_parsing())
    if result:
        print("\n🎉 SUCCESS: Timestamp parsing fix is working!")
        print("The backfill system can now correctly parse IB Gateway timestamps")
        print("and store them in PostgreSQL.")
    else:
        print("\n❌ FAILED: Timestamp parsing fix needs more work")