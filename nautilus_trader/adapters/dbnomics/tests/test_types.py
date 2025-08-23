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

from decimal import Decimal

import pandas as pd
import pytest

from nautilus_trader.adapters.dbnomics.core import DBNOMICS_VENUE
from nautilus_trader.adapters.dbnomics.types import DBnomicsTimeSeriesData
from nautilus_trader.model.identifiers import InstrumentId
from nautilus_trader.model.identifiers import Symbol


class TestDBnomicsTimeSeriesData:
    """
    Test cases for DBnomicsTimeSeriesData.
    """

    def test_initialization(self):
        """Test DBnomicsTimeSeriesData initialization."""
        instrument_id = InstrumentId(Symbol("IMF-CPI-M.FR.PCPIEC_WT"), DBNOMICS_VENUE)
        timestamp = pd.Timestamp("2023-01-01")
        value = Decimal("100.5")
        
        data = DBnomicsTimeSeriesData(
            instrument_id=instrument_id,
            timestamp=timestamp,
            value=value,
            series_code="M.FR.PCPIEC_WT",
            provider_code="IMF",
            dataset_code="CPI",
            frequency="M",
            unit="Index",
        )
        
        assert data.instrument_id == instrument_id
        assert data.timestamp == timestamp
        assert data.value == value
        assert data.series_code == "M.FR.PCPIEC_WT"
        assert data.provider_code == "IMF"
        assert data.dataset_code == "CPI"
        assert data.frequency == "M"
        assert data.unit == "Index"

    def test_initialization_optional_params(self):
        """Test initialization with optional parameters."""
        instrument_id = InstrumentId(Symbol("IMF-CPI-M.FR.PCPIEC_WT"), DBNOMICS_VENUE)
        timestamp = pd.Timestamp("2023-01-01")
        value = Decimal("100.5")
        
        data = DBnomicsTimeSeriesData(
            instrument_id=instrument_id,
            timestamp=timestamp,
            value=value,
            series_code="M.FR.PCPIEC_WT",
            provider_code="IMF",
            dataset_code="CPI",
        )
        
        assert data.frequency is None
        assert data.unit is None

    def test_repr(self):
        """Test string representation."""
        instrument_id = InstrumentId(Symbol("IMF-CPI-M.FR.PCPIEC_WT"), DBNOMICS_VENUE)
        timestamp = pd.Timestamp("2023-01-01")
        value = Decimal("100.5")
        
        data = DBnomicsTimeSeriesData(
            instrument_id=instrument_id,
            timestamp=timestamp,
            value=value,
            series_code="M.FR.PCPIEC_WT",
            provider_code="IMF",
            dataset_code="CPI",
        )
        
        expected = (
            f"DBnomicsTimeSeriesData("
            f"instrument_id={instrument_id}, "
            f"timestamp={timestamp}, "
            f"value={value}, "
            f"series_code='M.FR.PCPIEC_WT', "
            f"provider_code='IMF', "
            f"dataset_code='CPI'"
            f")"
        )
        
        assert repr(data) == expected