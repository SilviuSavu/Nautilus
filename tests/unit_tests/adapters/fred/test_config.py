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

import pytest

from nautilus_trader.adapters.fred.config import FREDDataClientConfig
from nautilus_trader.adapters.fred.config import FREDInstrumentProviderConfig
from nautilus_trader.model.identifiers import InstrumentId
from nautilus_trader.model.identifiers import Symbol
from nautilus_trader.model.identifiers import Venue


class TestFREDDataClientConfig:
    
    def test_fred_data_client_config_defaults(self):
        # Arrange
        api_key = "test_api_key_12345"
        
        # Act
        config = FREDDataClientConfig(api_key=api_key)
        
        # Assert
        assert config.api_key == api_key
        assert config.base_url == "https://api.stlouisfed.org/fred"
        assert config.request_timeout == 30.0
        assert config.rate_limit_delay == 1.0
        assert config.max_retries == 3
        assert config.retry_delay == 2.0
        assert config.default_limit == 1000
        assert config.update_interval == 3600
        assert config.instrument_ids is None
        assert config.series_ids is None
        assert config.auto_subscribe is False

    def test_fred_data_client_config_custom_values(self):
        # Arrange
        api_key = "custom_api_key"
        instrument_ids = [
            InstrumentId(Symbol("GDP"), Venue("FRED")),
            InstrumentId(Symbol("UNRATE"), Venue("FRED")),
        ]
        series_ids = ["GDP", "UNRATE", "FEDFUNDS"]
        
        # Act
        config = FREDDataClientConfig(
            api_key=api_key,
            base_url="https://custom.fred.api.com",
            request_timeout=60.0,
            rate_limit_delay=2.0,
            max_retries=5,
            retry_delay=3.0,
            default_limit=500,
            update_interval=1800,
            instrument_ids=instrument_ids,
            series_ids=series_ids,
            auto_subscribe=True,
        )
        
        # Assert
        assert config.api_key == api_key
        assert config.base_url == "https://custom.fred.api.com"
        assert config.request_timeout == 60.0
        assert config.rate_limit_delay == 2.0
        assert config.max_retries == 5
        assert config.retry_delay == 3.0
        assert config.default_limit == 500
        assert config.update_interval == 1800
        assert config.instrument_ids == instrument_ids
        assert config.series_ids == series_ids
        assert config.auto_subscribe is True

    def test_fred_data_client_config_frozen(self):
        # Arrange
        config = FREDDataClientConfig(api_key="test_key")
        
        # Act & Assert
        with pytest.raises(AttributeError):
            config.api_key = "new_key"  # Should fail - config is frozen


class TestFREDInstrumentProviderConfig:
    
    def test_fred_instrument_provider_config_defaults(self):
        # Arrange
        api_key = "test_api_key_12345"
        
        # Act
        config = FREDInstrumentProviderConfig(api_key=api_key)
        
        # Assert
        assert config.api_key == api_key
        assert config.base_url == "https://api.stlouisfed.org/fred"
        assert config.request_timeout == 30.0
        assert config.rate_limit_delay == 1.0
        assert config.series_ids == []
        assert config.load_all is False
        assert config.category_ids == []
        assert config.search_terms == []
        assert config.max_search_results == 100
        assert config.cache_instruments is True

    def test_fred_instrument_provider_config_custom_values(self):
        # Arrange
        api_key = "custom_api_key"
        series_ids = ["GDP", "UNRATE", "FEDFUNDS"]
        category_ids = [1, 2, 3]
        search_terms = ["GDP", "unemployment"]
        
        # Act
        config = FREDInstrumentProviderConfig(
            api_key=api_key,
            base_url="https://custom.fred.api.com",
            request_timeout=45.0,
            rate_limit_delay=1.5,
            series_ids=series_ids,
            load_all=True,
            category_ids=category_ids,
            search_terms=search_terms,
            max_search_results=200,
            cache_instruments=False,
        )
        
        # Assert
        assert config.api_key == api_key
        assert config.base_url == "https://custom.fred.api.com"
        assert config.request_timeout == 45.0
        assert config.rate_limit_delay == 1.5
        assert config.series_ids == series_ids
        assert config.load_all is True
        assert config.category_ids == category_ids
        assert config.search_terms == search_terms
        assert config.max_search_results == 200
        assert config.cache_instruments is False

    def test_fred_instrument_provider_config_frozen(self):
        # Arrange
        config = FREDInstrumentProviderConfig(api_key="test_key")
        
        # Act & Assert
        with pytest.raises(AttributeError):
            config.api_key = "new_key"  # Should fail - config is frozen