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

from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import patch

import pandas as pd
import pytest

from nautilus_trader.adapters.dbnomics.config import DBnomicsDataClientConfig
from nautilus_trader.adapters.dbnomics.core import DBNOMICS_VENUE
from nautilus_trader.adapters.dbnomics.data import DBnomicsDataClient
from nautilus_trader.adapters.dbnomics.errors import DBnomicsConnectionError
from nautilus_trader.adapters.dbnomics.errors import DBnomicsDataError
from nautilus_trader.adapters.dbnomics.types import DBnomicsTimeSeriesData
from nautilus_trader.common.enums import LogColor
from nautilus_trader.core.uuid import UUID4
from nautilus_trader.data.messages import DataRequest
from nautilus_trader.data.messages import DataType
from nautilus_trader.data.messages import RequestData
from nautilus_trader.data.messages import SubscribeData
from nautilus_trader.model.identifiers import ClientId
from nautilus_trader.model.identifiers import InstrumentId
from nautilus_trader.model.identifiers import Symbol
from nautilus_trader.test_kit.providers import TestInstrumentProvider
from nautilus_trader.test_kit.stubs.cache import TestCache
from nautilus_trader.test_kit.stubs.clock import TestClock
from nautilus_trader.test_kit.stubs.msgbus import TestMessageBus


class TestDBnomicsDataClient:
    """
    Test cases for DBnomicsDataClient.
    """

    def setup_method(self):
        """Set up test fixtures."""
        self.loop = AsyncMock()
        self.client_id = ClientId("DBNOMICS-001")
        self.config = DBnomicsDataClientConfig()
        self.msgbus = TestMessageBus()
        self.cache = TestCache()
        self.clock = TestClock()
        
        self.client = DBnomicsDataClient(
            loop=self.loop,
            client_id=self.client_id,
            config=self.config,
            msgbus=self.msgbus,
            cache=self.cache,
            clock=self.clock,
        )

    @pytest.mark.asyncio
    @patch("nautilus_trader.adapters.dbnomics.data.dbnomics.fetch_series")
    async def test_connect_success(self, mock_fetch_series):
        """Test successful connection."""
        # Arrange
        mock_df = pd.DataFrame({
            'period': ['2023-01-01'],
            'value': [100.0],
            'series_code': ['M.FR.PCPIEC_WT'],
            'provider_code': ['IMF'],
            'dataset_code': ['CPI']
        })
        mock_fetch_series.return_value = mock_df

        # Act
        await self.client._connect()

        # Assert
        mock_fetch_series.assert_called_once()

    @pytest.mark.asyncio
    @patch("nautilus_trader.adapters.dbnomics.data.dbnomics.fetch_series")
    async def test_connect_fetch_error(self, mock_fetch_series):
        """Test connection failure due to fetch error."""
        # Arrange
        import dbnomics
        mock_fetch_series.side_effect = dbnomics.FetchError(url="test")

        # Act & Assert
        with pytest.raises(DBnomicsConnectionError):
            await self.client._connect()

    @pytest.mark.asyncio
    async def test_disconnect(self):
        """Test disconnection."""
        # Arrange
        self.client._subscriptions = {"test": "data"}

        # Act
        await self.client._disconnect()

        # Assert
        assert len(self.client._subscriptions) == 0

    def test_reset(self):
        """Test client reset."""
        # Arrange
        self.client._subscriptions = {"test": "data"}

        # Act
        self.client.reset()

        # Assert
        assert len(self.client._subscriptions) == 0

    @pytest.mark.asyncio
    async def test_subscribe_valid_data_type(self):
        """Test subscription to valid data type."""
        # Arrange
        instrument_id = InstrumentId(Symbol("IMF-CPI-M.FR.PCPIEC_WT"), DBNOMICS_VENUE)
        data_type = DataType(
            type=DBnomicsTimeSeriesData,
            metadata={'instrument_id': instrument_id}
        )
        command = SubscribeData(
            client_id=self.client_id,
            venue=DBNOMICS_VENUE,
            data_type=data_type,
            command_id=UUID4(),
        )

        with patch.object(self.client, '_fetch_series_data', new_callable=AsyncMock) as mock_fetch:
            # Act
            await self.client._subscribe(command)

            # Assert
            assert instrument_id in self.client._subscriptions
            mock_fetch.assert_called_once()

    @pytest.mark.asyncio
    async def test_subscribe_invalid_data_type(self):
        """Test subscription to invalid data type."""
        # Arrange
        data_type = DataType(type=str)  # Invalid type
        command = SubscribeData(
            client_id=self.client_id,
            venue=DBNOMICS_VENUE,
            data_type=data_type,
            command_id=UUID4(),
        )

        # Act
        await self.client._subscribe(command)

        # Assert
        assert len(self.client._subscriptions) == 0

    @pytest.mark.asyncio
    @patch("nautilus_trader.adapters.dbnomics.data.dbnomics.fetch_series")
    async def test_fetch_series_data_success(self, mock_fetch_series):
        """Test successful data fetching."""
        # Arrange
        mock_df = pd.DataFrame({
            'period': ['2023-01-01', '2023-02-01'],
            'value': [100.0, 101.0],
            'series_code': ['M.FR.PCPIEC_WT', 'M.FR.PCPIEC_WT'],
            'provider_code': ['IMF', 'IMF'],
            'dataset_code': ['CPI', 'CPI'],
            'freq': ['M', 'M'],
            'unit': ['Index', 'Index']
        })
        mock_fetch_series.return_value = mock_df
        
        instrument_id = InstrumentId(Symbol("IMF-CPI-M.FR.PCPIEC_WT"), DBNOMICS_VENUE)
        metadata = {}

        # Act
        await self.client._fetch_series_data(instrument_id, metadata)

        # Assert
        mock_fetch_series.assert_called_once()

    @pytest.mark.asyncio
    @patch("nautilus_trader.adapters.dbnomics.data.dbnomics.fetch_series")
    async def test_fetch_series_data_fetch_error(self, mock_fetch_series):
        """Test data fetching with fetch error."""
        # Arrange
        import dbnomics
        mock_fetch_series.side_effect = dbnomics.FetchError(url="test")
        
        instrument_id = InstrumentId(Symbol("IMF-CPI-M.FR.PCPIEC_WT"), DBNOMICS_VENUE)
        metadata = {}

        # Act & Assert
        with pytest.raises(DBnomicsDataError):
            await self.client._fetch_series_data(instrument_id, metadata)

    @pytest.mark.asyncio
    async def test_process_and_publish_data(self):
        """Test data processing and publishing."""
        # Arrange
        df = pd.DataFrame({
            'period': ['2023-01-01', '2023-02-01'],
            'value': [100.0, 101.0],
            'series_code': ['M.FR.PCPIEC_WT', 'M.FR.PCPIEC_WT'],
            'provider_code': ['IMF', 'IMF'],
            'dataset_code': ['CPI', 'CPI'],
            'freq': ['M', 'M'],
            'unit': ['Index', 'Index']
        })
        
        instrument_id = InstrumentId(Symbol("IMF-CPI-M.FR.PCPIEC_WT"), DBNOMICS_VENUE)

        with patch.object(self.client, '_handle_data') as mock_handle:
            # Act
            await self.client._process_and_publish_data(df, instrument_id)

            # Assert
            assert mock_handle.call_count == 2  # Two data points

    def test_handle_data(self):
        """Test data handling."""
        # Arrange
        instrument_id = InstrumentId(Symbol("IMF-CPI-M.FR.PCPIEC_WT"), DBNOMICS_VENUE)
        timestamp = pd.Timestamp("2023-01-01")
        
        data = DBnomicsTimeSeriesData(
            instrument_id=instrument_id,
            timestamp=timestamp,
            value=pd.Decimal("100.0"),
            series_code="M.FR.PCPIEC_WT",
            provider_code="IMF",
            dataset_code="CPI",
        )

        # Act
        self.client._handle_data(data)

        # Assert - Check that message was published
        # This would depend on the specific msgbus implementation