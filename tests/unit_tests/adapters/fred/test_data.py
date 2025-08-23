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

import pytest

from nautilus_trader.adapters.fred.data import EconomicData
from nautilus_trader.model.identifiers import InstrumentId
from nautilus_trader.model.identifiers import Symbol
from nautilus_trader.model.identifiers import Venue


class TestEconomicData:
    def setup_method(self):
        self.instrument_id = InstrumentId(Symbol("GDP"), Venue("FRED"))
        self.series_id = "GDP"
        self.value = Decimal("25000.5")
        self.units = "Billions of Dollars"
        self.frequency = "Quarterly"
        
    def test_create_economic_data_with_decimal(self):
        # Arrange & Act
        data = EconomicData.create(
            instrument_id=self.instrument_id,
            series_id=self.series_id,
            value=self.value,
            units=self.units,
            frequency=self.frequency,
            ts_event=1000000000,
            ts_init=1000000000,
        )
        
        # Assert
        assert data.instrument_id == self.instrument_id
        assert data.series_id == self.series_id
        assert data.value == self.value
        assert data.units == self.units
        assert data.frequency == self.frequency
        assert data.ts_event == 1000000000
        assert data.ts_init == 1000000000

    def test_create_economic_data_with_float(self):
        # Arrange & Act
        data = EconomicData.create(
            instrument_id=self.instrument_id,
            series_id=self.series_id,
            value=25000.5,
            units=self.units,
            frequency=self.frequency,
        )
        
        # Assert
        assert data.value == Decimal("25000.5")

    def test_create_economic_data_with_string(self):
        # Arrange & Act
        data = EconomicData.create(
            instrument_id=self.instrument_id,
            series_id=self.series_id,
            value="25000.5",
            units=self.units,
            frequency=self.frequency,
        )
        
        # Assert
        assert data.value == Decimal("25000.5")

    def test_create_economic_data_with_missing_value_dot(self):
        # Arrange & Act
        data = EconomicData.create(
            instrument_id=self.instrument_id,
            series_id=self.series_id,
            value=".",  # FRED uses "." for missing values
            units=self.units,
            frequency=self.frequency,
        )
        
        # Assert
        assert data.value == Decimal("0")

    def test_create_economic_data_with_empty_string(self):
        # Arrange & Act
        data = EconomicData.create(
            instrument_id=self.instrument_id,
            series_id=self.series_id,
            value="",
            units=self.units,
            frequency=self.frequency,
        )
        
        # Assert
        assert data.value == Decimal("0")

    def test_is_valid_value_with_valid_data(self):
        # Arrange
        data = EconomicData.create(
            instrument_id=self.instrument_id,
            series_id=self.series_id,
            value="25000.5",
            units=self.units,
            frequency=self.frequency,
        )
        
        # Act & Assert
        assert data.is_valid_value

    def test_is_valid_value_with_zero_but_units(self):
        # Arrange
        data = EconomicData.create(
            instrument_id=self.instrument_id,
            series_id=self.series_id,
            value="0",
            units=self.units,
            frequency=self.frequency,
        )
        
        # Act & Assert
        assert data.is_valid_value  # Valid because it has units

    def test_is_valid_value_with_zero_no_units(self):
        # Arrange
        data = EconomicData.create(
            instrument_id=self.instrument_id,
            series_id=self.series_id,
            value="0",
            units="",
            frequency=self.frequency,
        )
        
        # Act & Assert
        assert not data.is_valid_value

    def test_to_dict(self):
        # Arrange
        data = EconomicData.create(
            instrument_id=self.instrument_id,
            series_id=self.series_id,
            value="25000.5",
            units=self.units,
            frequency=self.frequency,
            seasonal_adjustment="Seasonally Adjusted",
            ts_event=1000000000,
            ts_init=1000000000,
            last_updated=2000000000,
            release_date=1500000000,
        )
        
        # Act
        result = data.to_dict()
        
        # Assert
        expected = {
            "instrument_id": "GDP.FRED",
            "series_id": "GDP",
            "value": 25000.5,
            "units": "Billions of Dollars",
            "frequency": "Quarterly",
            "seasonal_adjustment": "Seasonally Adjusted",
            "ts_event": 1000000000,
            "ts_init": 1000000000,
            "last_updated": 2000000000,
            "release_date": 1500000000,
        }
        assert result == expected

    def test_repr(self):
        # Arrange
        data = EconomicData.create(
            instrument_id=self.instrument_id,
            series_id=self.series_id,
            value="25000.5",
            units=self.units,
            frequency=self.frequency,
            ts_event=1000000000,
            ts_init=1000000000,
        )
        
        # Act
        result = repr(data)
        
        # Assert
        assert "EconomicData" in result
        assert "GDP" in result
        assert "25000.5" in result
        assert "Billions of Dollars" in result
        assert "Quarterly" in result