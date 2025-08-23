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

from unittest.mock import patch

import pandas as pd
import pytest

from nautilus_trader.adapters.dbnomics.core import DBNOMICS_VENUE
from nautilus_trader.adapters.dbnomics.providers import DBnomicsInstrumentProvider
from nautilus_trader.model.identifiers import InstrumentId
from nautilus_trader.model.identifiers import Symbol


class TestDBnomicsInstrumentProvider:
    """
    Test cases for DBnomicsInstrumentProvider.
    """

    def setup_method(self):
        """Set up test fixtures."""
        self.provider = DBnomicsInstrumentProvider(
            max_nb_series=50,
            timeout=30,
        )

    @pytest.mark.asyncio
    @patch("nautilus_trader.adapters.dbnomics.providers.dbnomics.fetch_series")
    async def test_load_all_async_success(self, mock_fetch_series):
        """Test successful loading of all series."""
        # Arrange
        mock_df = pd.DataFrame({
            'period': ['2023-01-01', '2023-02-01'],
            'value': [100.0, 101.0],
            'series_code': ['M.FR.PCPIEC_WT', 'M.FR.PCPIEC_WT'],
            'provider_code': ['IMF', 'IMF'],
            'dataset_code': ['CPI', 'CPI'],
            'series_name': ['Consumer Price Index', 'Consumer Price Index'],
            'frequency': ['M', 'M'],
            'unit': ['Index', 'Index'],
            'last_update': ['2023-12-01', '2023-12-01']
        })
        mock_fetch_series.return_value = mock_df

        filters = {
            'providers': ['IMF'],
            'datasets': {'IMF': ['CPI']},
            'dimensions': {'geo': ['FR']}
        }

        # Act
        await self.provider.load_all_async(filters)

        # Assert
        mock_fetch_series.assert_called_once()
        assert len(self.provider.get_all()) > 0

    @pytest.mark.asyncio
    async def test_load_all_async_no_filters(self):
        """Test loading all series without filters."""
        # Act
        await self.provider.load_all_async(None)

        # Assert - Should log warning and return without loading

    @pytest.mark.asyncio
    @patch("nautilus_trader.adapters.dbnomics.providers.dbnomics.fetch_series")
    async def test_load_ids_async_success(self, mock_fetch_series):
        """Test successful loading of specific series IDs."""
        # Arrange
        mock_df = pd.DataFrame({
            'period': ['2023-01-01'],
            'value': [100.0],
            'series_code': ['M.FR.PCPIEC_WT'],
            'provider_code': ['IMF'],
            'dataset_code': ['CPI'],
            'series_name': ['Consumer Price Index'],
            'frequency': ['M'],
            'unit': ['Index'],
            'last_update': ['2023-12-01']
        })
        mock_fetch_series.return_value = mock_df

        instrument_ids = [
            InstrumentId(Symbol("IMF-CPI-M.FR.PCPIEC_WT"), DBNOMICS_VENUE)
        ]

        # Act
        await self.provider.load_ids_async(instrument_ids)

        # Assert
        mock_fetch_series.assert_called_once_with(
            series_ids=['IMF/CPI/M.FR.PCPIEC_WT'],
            max_nb_series=50,
            api_base_url=None,
            timeout=30,
        )

    @pytest.mark.asyncio
    @patch("nautilus_trader.adapters.dbnomics.providers.dbnomics.fetch_series")
    async def test_load_async_success(self, mock_fetch_series):
        """Test successful loading of single series."""
        # Arrange
        mock_df = pd.DataFrame({
            'period': ['2023-01-01'],
            'value': [100.0],
            'series_code': ['M.FR.PCPIEC_WT'],
            'provider_code': ['IMF'],
            'dataset_code': ['CPI'],
            'series_name': ['Consumer Price Index'],
            'frequency': ['M'],
            'unit': ['Index'],
            'last_update': ['2023-12-01']
        })
        mock_fetch_series.return_value = mock_df

        instrument_id = InstrumentId(Symbol("IMF-CPI-M.FR.PCPIEC_WT"), DBNOMICS_VENUE)

        # Act
        await self.provider.load_async(instrument_id)

        # Assert
        mock_fetch_series.assert_called_once()

    @pytest.mark.asyncio
    async def test_process_series_dataframe_empty(self):
        """Test processing empty dataframe."""
        # Arrange
        empty_df = pd.DataFrame()

        # Act
        await self.provider._process_series_dataframe(empty_df)

        # Assert - Should return without error

    @pytest.mark.asyncio
    async def test_process_series_dataframe_success(self):
        """Test successful processing of series dataframe."""
        # Arrange
        df = pd.DataFrame({
            'series_code': ['M.FR.PCPIEC_WT'],
            'provider_code': ['IMF'],
            'dataset_code': ['CPI'],
            'series_name': ['Consumer Price Index'],
            'frequency': ['M'],
            'unit': ['Index'],
            'last_update': ['2023-12-01']
        })

        # Act
        await self.provider._process_series_dataframe(df)

        # Assert
        instruments = self.provider.get_all()
        assert len(instruments) == 1
        
        instrument = instruments[0]
        assert str(instrument.id.symbol) == "IMF-CPI-M.FR.PCPIEC_WT"
        assert instrument.id.venue == DBNOMICS_VENUE
        assert instrument.info['provider_code'] == 'IMF'
        assert instrument.info['dataset_code'] == 'CPI'
        assert instrument.info['series_name'] == 'Consumer Price Index'