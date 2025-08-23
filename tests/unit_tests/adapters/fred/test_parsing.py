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

from nautilus_trader.adapters.fred.parsing import create_fred_instrument
from nautilus_trader.adapters.fred.parsing import format_economic_value
from nautilus_trader.adapters.fred.parsing import get_popular_fred_series
from nautilus_trader.adapters.fred.parsing import parse_fred_observations
from nautilus_trader.adapters.fred.parsing import parse_fred_series_info
from nautilus_trader.adapters.fred.parsing import validate_fred_series_id
from nautilus_trader.model.identifiers import InstrumentId
from nautilus_trader.model.identifiers import Symbol
from nautilus_trader.model.identifiers import Venue
from nautilus_trader.model.instruments import Equity


class TestFREDParsing:
    
    def test_parse_fred_series_info(self):
        # Arrange
        raw_data = {
            "id": "GDP",
            "title": "Gross Domestic Product",
            "units": "Billions of Dollars",
            "units_short": "Bil. of $",
            "frequency": "Quarterly",
            "frequency_short": "Q",
            "seasonal_adjustment": "Seasonally Adjusted Annual Rate",
            "seasonal_adjustment_short": "SAAR",
            "last_updated": "2024-01-01",
            "popularity": 100,
            "notes": "GDP measurement",
        }
        
        # Act
        result = parse_fred_series_info(raw_data)
        
        # Assert
        assert result["id"] == "GDP"
        assert result["title"] == "Gross Domestic Product"
        assert result["units"] == "Billions of Dollars"
        assert result["frequency"] == "Quarterly"
        assert result["seasonal_adjustment"] == "Seasonally Adjusted Annual Rate"
        assert result["last_updated"] == "2024-01-01"
        assert result["popularity"] == "100"

    def test_parse_fred_series_info_with_missing_fields(self):
        # Arrange
        raw_data = {"id": "GDP"}  # Minimal data
        
        # Act
        result = parse_fred_series_info(raw_data)
        
        # Assert
        assert result["id"] == "GDP"
        assert result["title"] == ""
        assert result["units"] == ""
        assert result["frequency"] == ""

    def test_parse_fred_observations_valid_data(self):
        # Arrange
        series_id = "GDP"
        observations_data = {
            "observations": [
                {
                    "date": "2023-01-01",
                    "value": "25000.5",
                    "realtime_end": "2023-01-15",
                },
                {
                    "date": "2022-10-01", 
                    "value": "24900.0",
                    "realtime_end": "2022-10-15",
                },
            ]
        }
        series_info = {
            "units": "Billions of Dollars",
            "frequency": "Quarterly",
            "seasonal_adjustment": "Seasonally Adjusted",
        }
        
        # Act
        result = parse_fred_observations(series_id, observations_data, series_info)
        
        # Assert
        assert len(result) == 2
        
        data1 = result[0]
        assert data1.series_id == "GDP"
        assert data1.instrument_id.symbol.value == "GDP"
        assert data1.instrument_id.venue.value == "FRED"
        assert data1.value == Decimal("25000.5")
        assert data1.units == "Billions of Dollars"
        assert data1.frequency == "Quarterly"

    def test_parse_fred_observations_with_missing_values(self):
        # Arrange
        series_id = "GDP"
        observations_data = {
            "observations": [
                {
                    "date": "2023-01-01",
                    "value": "25000.5",
                },
                {
                    "date": "2022-10-01", 
                    "value": ".",  # Missing value
                },
                {
                    "date": "2022-07-01", 
                    "value": "",  # Empty value
                },
            ]
        }
        
        # Act
        result = parse_fred_observations(series_id, observations_data)
        
        # Assert
        assert len(result) == 1  # Only valid observations
        assert result[0].value == Decimal("25000.5")

    def test_parse_fred_observations_with_invalid_dates(self):
        # Arrange
        series_id = "GDP"
        observations_data = {
            "observations": [
                {
                    "date": "invalid-date",
                    "value": "25000.5",
                },
                {
                    "date": "2023-01-01",
                    "value": "24000.0",
                },
            ]
        }
        
        # Act
        result = parse_fred_observations(series_id, observations_data)
        
        # Assert
        assert len(result) == 1  # Only valid date observations
        assert result[0].value == Decimal("24000.0")

    def test_create_fred_instrument(self):
        # Arrange
        series_info = {
            "id": "GDP",
            "title": "Gross Domestic Product",
            "units": "Billions of Dollars",
            "frequency": "Quarterly",
        }
        
        # Act
        result = create_fred_instrument(series_info)
        
        # Assert
        assert isinstance(result, Equity)
        assert result.id.symbol.value == "GDP"
        assert result.id.venue.value == "FRED"
        assert result.info["title"] == "Gross Domestic Product"
        assert result.info["type"] == "economic_indicator"

    def test_get_popular_fred_series(self):
        # Act
        result = get_popular_fred_series()
        
        # Assert
        assert isinstance(result, dict)
        assert len(result) > 0
        assert "GDP" in result
        assert "UNRATE" in result
        assert "FEDFUNDS" in result
        assert isinstance(result["GDP"], str)

    def test_validate_fred_series_id_valid(self):
        # Act & Assert
        assert validate_fred_series_id("GDP") is True
        assert validate_fred_series_id("UNRATE") is True
        assert validate_fred_series_id("DGS10") is True
        assert validate_fred_series_id("CPIAUCSL") is True
        assert validate_fred_series_id("GDP_TEST") is True
        assert validate_fred_series_id("GDP-TEST") is True

    def test_validate_fred_series_id_invalid(self):
        # Act & Assert
        assert validate_fred_series_id("") is False
        assert validate_fred_series_id(None) is False
        assert validate_fred_series_id("x" * 51) is False  # Too long
        assert validate_fred_series_id("GDP@TEST") is False  # Invalid character
        assert validate_fred_series_id("GDP TEST") is False  # Space not allowed
        assert validate_fred_series_id(123) is False  # Not a string

    def test_format_economic_value_percent(self):
        # Act & Assert
        assert format_economic_value(Decimal("5.25"), "Percent") == "5.25%"
        assert format_economic_value(Decimal("3.14"), "Interest Rate") == "3.14%"

    def test_format_economic_value_billions(self):
        # Act & Assert
        assert format_economic_value(Decimal("25000.5"), "Billions of Dollars") == "$25,000.5B"

    def test_format_economic_value_millions(self):
        # Act & Assert
        assert format_economic_value(Decimal("1500.2"), "Millions of Dollars") == "$1,500.2M"

    def test_format_economic_value_thousands(self):
        # Act & Assert
        assert format_economic_value(Decimal("500.1"), "Thousands of Dollars") == "$500.1K"

    def test_format_economic_value_index(self):
        # Act & Assert
        assert format_economic_value(Decimal("105.25"), "Index 1982-1984=100") == "105.25"

    def test_format_economic_value_dollar(self):
        # Act & Assert
        assert format_economic_value(Decimal("1234.56"), "Dollars") == "$1,234.56"
        assert format_economic_value(Decimal("1234.56"), "$") == "$1,234.56"

    def test_format_economic_value_default(self):
        # Act & Assert
        assert format_economic_value(Decimal("1234.5678"), "Units") == "1,234.5678"