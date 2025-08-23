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

from nautilus_trader.adapters.dbnomics.config import DBnomicsDataClientConfig
from nautilus_trader.adapters.dbnomics.data import DBnomicsDataClient
from nautilus_trader.adapters.dbnomics.factories import DBnomicsInstrumentProviderFactory
from nautilus_trader.adapters.dbnomics.factories import DBnomicsLiveDataClientFactory
from nautilus_trader.adapters.dbnomics.providers import DBnomicsInstrumentProvider
from nautilus_trader.model.identifiers import ClientId
from nautilus_trader.test_kit.stubs.cache import TestCache
from nautilus_trader.test_kit.stubs.clock import TestClock
from nautilus_trader.test_kit.stubs.msgbus import TestMessageBus


class TestDBnomicsLiveDataClientFactory:
    """
    Test cases for DBnomicsLiveDataClientFactory.
    """

    def test_create_client(self):
        """Test creating a data client via factory."""
        # Arrange
        loop = AsyncMock()
        name = "DBNOMICS-001"
        config = DBnomicsDataClientConfig()
        msgbus = TestMessageBus()
        cache = TestCache()
        clock = TestClock()

        # Act
        client = DBnomicsLiveDataClientFactory.create(
            loop=loop,
            name=name,
            config=config,
            msgbus=msgbus,
            cache=cache,
            clock=clock,
        )

        # Assert
        assert isinstance(client, DBnomicsDataClient)
        assert client._client_id == ClientId(name)


class TestDBnomicsInstrumentProviderFactory:
    """
    Test cases for DBnomicsInstrumentProviderFactory.
    """

    def test_create_provider(self):
        """Test creating an instrument provider via factory."""
        # Arrange
        config = DBnomicsDataClientConfig(
            max_nb_series=100,
            api_base_url="https://test.api.url",
            timeout=60,
        )

        # Act
        provider = DBnomicsInstrumentProviderFactory.create(config)

        # Assert
        assert isinstance(provider, DBnomicsInstrumentProvider)
        assert provider._max_nb_series == 100
        assert provider._api_base_url == "https://test.api.url"
        assert provider._timeout == 60

    def test_create_provider_default_config(self):
        """Test creating provider with default configuration."""
        # Arrange
        config = DBnomicsDataClientConfig()

        # Act
        provider = DBnomicsInstrumentProviderFactory.create(config)

        # Assert
        assert isinstance(provider, DBnomicsInstrumentProvider)
        assert provider._max_nb_series == 50
        assert provider._api_base_url is None
        assert provider._timeout == 30