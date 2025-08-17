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

from nautilus_trader.adapters.yfinance.config import YFinanceDataClientConfig
from nautilus_trader.adapters.yfinance.data import YFinanceDataClient
from nautilus_trader.adapters.yfinance.factories import YFinanceLiveDataClientFactory
from nautilus_trader.adapters.yfinance.factories import create_yfinance_data_client
from nautilus_trader.adapters.yfinance.factories import create_yfinance_instrument_provider
from nautilus_trader.adapters.yfinance.factories import get_cached_yfinance_instrument_provider
from nautilus_trader.adapters.yfinance.providers import YFinanceInstrumentProvider


class TestYFinanceFactories:
    """Tests for YFinance factory functions."""

    def test_create_yfinance_instrument_provider(self):
        """Test creating instrument provider via factory."""
        provider = create_yfinance_instrument_provider()
        
        assert isinstance(provider, YFinanceInstrumentProvider)
        assert str(provider._venue) == "YAHOO"

    def test_get_cached_yfinance_instrument_provider(self):
        """Test cached instrument provider factory."""
        provider1 = get_cached_yfinance_instrument_provider()
        provider2 = get_cached_yfinance_instrument_provider()
        
        # Should return the same cached instance
        assert provider1 is provider2
        assert isinstance(provider1, YFinanceInstrumentProvider)

    def test_create_yfinance_data_client(self, event_loop, msgbus, cache, clock):
        """Test creating data client via convenience function."""
        config = YFinanceDataClientConfig(symbols=["AAPL"])
        
        client = create_yfinance_data_client(
            loop=event_loop,
            msgbus=msgbus,
            cache=cache,
            clock=clock,
            config=config,
            name="TestClient",
        )
        
        assert isinstance(client, YFinanceDataClient)
        assert str(client.id) == "TestClient"
        assert client._config.symbols == ["AAPL"]

    def test_create_yfinance_data_client_default_config(self, event_loop, msgbus, cache, clock):
        """Test creating data client with default config."""
        client = create_yfinance_data_client(
            loop=event_loop,
            msgbus=msgbus,
            cache=cache,
            clock=clock,
        )
        
        assert isinstance(client, YFinanceDataClient)
        assert client._config is not None
        assert client._config.cache_expiry_seconds == 3600  # Default value

    def test_yfinance_live_data_client_factory(self, event_loop, msgbus, cache, clock):
        """Test YFinanceLiveDataClientFactory."""
        config = YFinanceDataClientConfig(symbols=["MSFT"])
        
        client = YFinanceLiveDataClientFactory.create(
            loop=event_loop,
            name="FactoryClient",
            config=config,
            msgbus=msgbus,
            cache=cache,
            clock=clock,
        )
        
        assert isinstance(client, YFinanceDataClient)
        assert str(client.id) == "FactoryClient"
        assert client._config.symbols == ["MSFT"]

    def test_factory_creates_different_instances(self, event_loop, msgbus, cache, clock):
        """Test that factory creates different client instances."""
        config1 = YFinanceDataClientConfig(symbols=["AAPL"])
        config2 = YFinanceDataClientConfig(symbols=["MSFT"])
        
        client1 = create_yfinance_data_client(
            loop=event_loop,
            msgbus=msgbus,
            cache=cache,
            clock=clock,
            config=config1,
            name="Client1",
        )
        
        client2 = create_yfinance_data_client(
            loop=event_loop,
            msgbus=msgbus,
            cache=cache,
            clock=clock,
            config=config2,
            name="Client2",
        )
        
        assert client1 is not client2
        assert str(client1.id) == "Client1"
        assert str(client2.id) == "Client2"
        assert client1._config.symbols == ["AAPL"]
        assert client2._config.symbols == ["MSFT"]