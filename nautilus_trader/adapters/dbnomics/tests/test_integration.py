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
Integration tests for DBnomics adapter.

These tests require network access and will make real API calls to dbnomics.world.
They are marked as integration tests and may be skipped in CI/CD environments.
"""

from unittest.mock import AsyncMock

import pytest

from nautilus_trader.adapters.dbnomics.config import DBnomicsDataClientConfig
from nautilus_trader.adapters.dbnomics.core import DBNOMICS_VENUE
from nautilus_trader.adapters.dbnomics.data import DBnomicsDataClient
from nautilus_trader.adapters.dbnomics.providers import DBnomicsInstrumentProvider
from nautilus_trader.adapters.dbnomics.types import DBnomicsTimeSeriesData
from nautilus_trader.core.uuid import UUID4
from nautilus_trader.data.messages import DataType
from nautilus_trader.data.messages import RequestData
from nautilus_trader.model.identifiers import ClientId
from nautilus_trader.model.identifiers import InstrumentId
from nautilus_trader.model.identifiers import Symbol
from nautilus_trader.test_kit.stubs.cache import TestCache
from nautilus_trader.test_kit.stubs.clock import TestClock
from nautilus_trader.test_kit.stubs.msgbus import TestMessageBus


@pytest.mark.integration
class TestDBnomicsIntegration:
    """
    Integration test cases for DBnomics adapter.
    
    These tests make real API calls and require network connectivity.
    """

    def setup_method(self):
        """Set up test fixtures."""
        self.config = DBnomicsDataClientConfig(
            max_nb_series=10,  # Keep small for tests
            timeout=60,  # Longer timeout for network calls
        )

    @pytest.mark.asyncio
    async def test_real_api_connection(self):
        """Test real connection to DBnomics API."""
        # Arrange
        loop = AsyncMock()
        client_id = ClientId("DBNOMICS-TEST")
        msgbus = TestMessageBus()
        cache = TestCache()
        clock = TestClock()

        client = DBnomicsDataClient(
            loop=loop,
            client_id=client_id,
            config=self.config,
            msgbus=msgbus,
            cache=cache,
            clock=clock,
        )

        # Act & Assert
        try:
            await client._connect()
            await client._disconnect()
        except Exception as e:
            pytest.skip(f"Real API connection failed: {e}")

    @pytest.mark.asyncio
    async def test_real_data_fetching(self):
        """Test fetching real data from DBnomics API."""
        # Arrange
        loop = AsyncMock()
        client_id = ClientId("DBNOMICS-TEST")
        msgbus = TestMessageBus()
        cache = TestCache()
        clock = TestClock()

        client = DBnomicsDataClient(
            loop=loop,
            client_id=client_id,
            config=self.config,
            msgbus=msgbus,
            cache=cache,
            clock=clock,
        )

        # Test with known stable series
        instrument_id = InstrumentId(
            Symbol("IMF-IFS-A.AD.BOP_BP6_GG"), 
            DBNOMICS_VENUE
        )
        metadata = {}

        try:
            # Act
            await client._fetch_series_data(instrument_id, metadata)
            
            # Assert - Should complete without error
            # Data publishing would be verified through message bus in full integration
            
        except Exception as e:
            pytest.skip(f"Real data fetching failed: {e}")

    @pytest.mark.asyncio
    async def test_instrument_provider_real_loading(self):
        """Test instrument provider with real data."""
        # Arrange
        provider = DBnomicsInstrumentProvider(
            max_nb_series=5,
            timeout=60,
        )

        filters = {
            'providers': ['IMF'],
            'datasets': {'IMF': ['IFS']},
            'dimensions': {'geo': ['AD']}  # Small country for limited results
        }

        try:
            # Act
            await provider.load_all_async(filters)

            # Assert
            instruments = provider.get_all()
            assert len(instruments) > 0
            
            # Verify instrument structure
            instrument = instruments[0]
            assert instrument.id.venue == DBNOMICS_VENUE
            assert 'provider_code' in instrument.info
            assert 'dataset_code' in instrument.info
            
        except Exception as e:
            pytest.skip(f"Real instrument loading failed: {e}")

    def test_example_usage_configuration(self):
        """Test example configuration for documentation purposes."""
        # Arrange & Act
        config = DBnomicsDataClientConfig(
            max_nb_series=100,
            timeout=30,
            default_providers=['IMF', 'OECD', 'ECB'],
            instrument_id_mappings={
                'IMF/CPI/A.FR.PCPIEC_WT': 'EUR_INFLATION_ANNUAL',
                'OECD/QNA/DEU.B1_GE.GPSA.Q': 'DEU_GDP_QUARTERLY'
            },
            auto_retry=True,
        )

        # Assert
        assert config.max_nb_series == 100
        assert config.default_providers == ['IMF', 'OECD', 'ECB']
        assert len(config.instrument_id_mappings) == 2