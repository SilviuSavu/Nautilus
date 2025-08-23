#!/usr/bin/env python3
# -------------------------------------------------------------------------------------------------
#  Copyright (C) 2015-2025 Nautech Systems Pty Ltd. All rights reserved.
#  https://nautechsystems.io
#
#  Licensed under the GNU Lesser General Public License Version 3.0 (the "License");
#  You may not use this file except in compliance with the License.
#  You may obtain a copy of the License at https://www.gnu.org/licenses/lgpl-3.0.en.html
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
# -------------------------------------------------------------------------------------------------

"""
Example script demonstrating basic usage of the DBnomics adapter.

This script shows how to:
1. Configure the DBnomics data client
2. Load economic instruments 
3. Fetch economic time series data
4. Handle the data in a trading context

Run this script to see the adapter in action with real data.
"""

import asyncio
import logging
from decimal import Decimal

from nautilus_trader.adapters.dbnomics import DBnomicsDataClient
from nautilus_trader.adapters.dbnomics import DBnomicsDataClientConfig
from nautilus_trader.adapters.dbnomics import DBnomicsInstrumentProvider
from nautilus_trader.adapters.dbnomics.core import DBNOMICS_VENUE
from nautilus_trader.adapters.dbnomics.types import DBnomicsTimeSeriesData
from nautilus_trader.core.uuid import UUID4
from nautilus_trader.data.messages import DataType
from nautilus_trader.data.messages import RequestData
from nautilus_trader.model.identifiers import ClientId
from nautilus_trader.model.identifiers import InstrumentId
from nautilus_trader.model.identifiers import Symbol


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def demonstrate_instrument_loading():
    """Demonstrate loading economic instruments from DBnomics."""
    logger.info("=== Demonstrating Instrument Loading ===")
    
    # Create instrument provider
    provider = DBnomicsInstrumentProvider(
        max_nb_series=10,  # Limit for demo
        timeout=30,
    )
    
    # Define filters for economic indicators
    filters = {
        'providers': ['IMF'],
        'datasets': {'IMF': ['CPI']},  # Consumer Price Index
        'dimensions': {
            'geo': ['FR', 'DE'],  # France and Germany
            'freq': ['A']  # Annual data only
        }
    }
    
    try:
        # Load instruments
        logger.info("Loading instruments with filters: %s", filters)
        await provider.load_all_async(filters)
        
        # Display loaded instruments
        instruments = provider.get_all()
        logger.info("Loaded %d instruments:", len(instruments))
        
        for instrument in instruments:
            logger.info("  - %s (%s)", 
                       instrument.id, 
                       instrument.info.get('series_name', 'Unknown'))
                       
        return instruments
        
    except Exception as e:
        logger.error("Failed to load instruments: %s", e)
        return []


async def demonstrate_data_fetching():
    """Demonstrate fetching time series data."""
    logger.info("=== Demonstrating Data Fetching ===")
    
    # Mock components (in real usage, these come from NautilusTrader engine)
    class MockMsgBus:
        def publish(self, topic: str, msg):
            logger.info("Published data: %s on topic: %s", 
                       type(msg).__name__, topic)
    
    class MockCache:
        pass
    
    class MockClock:
        pass
    
    # Create data client
    config = DBnomicsDataClientConfig(
        max_nb_series=5,
        timeout=30,
    )
    
    client = DBnomicsDataClient(
        loop=asyncio.get_event_loop(),
        client_id=ClientId("DBNOMICS-DEMO"),
        config=config,
        msgbus=MockMsgBus(),
        cache=MockCache(),
        clock=MockClock(),
    )
    
    try:
        # Connect to API
        logger.info("Connecting to DBnomics API...")
        await client._connect()
        
        # Fetch data for a specific series
        instrument_id = InstrumentId(
            Symbol("IMF-CPI-A.FR.PCPIEC_WT"),  # France annual CPI
            DBNOMICS_VENUE
        )
        
        metadata = {
            'start_date': '2020-01-01',
            'end_date': '2023-12-31',
        }
        
        logger.info("Fetching data for: %s", instrument_id)
        await client._fetch_series_data(instrument_id, metadata)
        
        # Disconnect
        await client._disconnect()
        logger.info("Successfully fetched and published data")
        
    except Exception as e:
        logger.error("Data fetching failed: %s", e)


async def demonstrate_data_request():
    """Demonstrate making a data request."""
    logger.info("=== Demonstrating Data Request ===")
    
    # This would typically be done through the engine's data request mechanism
    instrument_id = InstrumentId(
        Symbol("OECD-KEI-M.USA.LORSGPRT.STSA"),  # US Unemployment rate
        DBNOMICS_VENUE  
    )
    
    data_type = DataType(
        type=DBnomicsTimeSeriesData,
        metadata={
            'instrument_id': instrument_id,
            'start_date': '2022-01-01',
            'filters': [
                {
                    'code': 'interpolate',
                    'parameters': {
                        'frequency': 'monthly',
                        'method': 'linear'
                    }
                }
            ]
        }
    )
    
    request = RequestData(
        client_id=ClientId("STRATEGY-001"),
        venue=DBNOMICS_VENUE,
        data_type=data_type,
        correlation_id=UUID4(),
        request_id=UUID4(),
        ts_init=0,
    )
    
    logger.info("Data request created for: %s", instrument_id)
    logger.info("Request would be processed by data client in live environment")


def demonstrate_data_usage():
    """Demonstrate how to use DBnomics data in trading logic."""
    logger.info("=== Demonstrating Data Usage ===")
    
    # Example: Creating and using economic data points
    instrument_id = InstrumentId(
        Symbol("IMF-CPI-A.US.PCPIEC_WT"),
        DBNOMICS_VENUE
    )
    
    # Sample data point (as would be received from DBnomics)
    import pandas as pd
    
    data_point = DBnomicsTimeSeriesData(
        instrument_id=instrument_id,
        timestamp=pd.Timestamp("2023-12-01"),
        value=Decimal("3.2"),  # 3.2% inflation
        series_code="A.US.PCPIEC_WT",
        provider_code="IMF",
        dataset_code="CPI",
        frequency="A",
        unit="Percent",
    )
    
    # Trading logic example
    inflation_rate = float(data_point.value)
    
    if inflation_rate > 3.0:
        logger.info("High inflation detected (%.1f%%) - Consider defensive positions", 
                   inflation_rate)
    elif inflation_rate < 1.0:
        logger.info("Low inflation detected (%.1f%%) - Consider growth positions", 
                   inflation_rate)
    else:
        logger.info("Moderate inflation (%.1f%%) - Normal market conditions", 
                   inflation_rate)
    
    logger.info("Data point: %s", data_point)


async def main():
    """Run all demonstration examples."""
    logger.info("Starting DBnomics Adapter Demonstration")
    logger.info("=" * 60)
    
    try:
        # Run demonstrations
        instruments = await demonstrate_instrument_loading()
        
        if instruments:  # Only proceed if we loaded instruments
            await demonstrate_data_fetching()
        
        await demonstrate_data_request()
        demonstrate_data_usage()
        
        logger.info("=" * 60)
        logger.info("Demonstration completed successfully!")
        
    except KeyboardInterrupt:
        logger.info("Demonstration interrupted by user")
    except Exception as e:
        logger.error("Demonstration failed: %s", e)
        raise


if __name__ == "__main__":
    # Run the demonstration
    asyncio.run(main())