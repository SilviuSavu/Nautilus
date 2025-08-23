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
DBnomics adapter factory for creating data clients and providers.
"""

from nautilus_trader.adapters.dbnomics.config import DBnomicsDataClientConfig
from nautilus_trader.adapters.dbnomics.data import DBnomicsDataClient
from nautilus_trader.adapters.dbnomics.providers import DBnomicsInstrumentProvider
from nautilus_trader.live.factories import LiveDataClientFactory


class DBnomicsLiveDataClientFactory(LiveDataClientFactory):
    """
    Provides DBnomics live data clients from a configuration.
    """

    @staticmethod
    def create(
        loop,
        name: str,
        config: DBnomicsDataClientConfig,
        msgbus,
        cache,
        clock,
    ) -> DBnomicsDataClient:
        """
        Create a DBnomics data client.

        Parameters
        ----------
        loop
            The event loop.
        name : str
            The client name.
        config : DBnomicsDataClientConfig
            The client configuration.
        msgbus
            The message bus.
        cache
            The cache.
        clock
            The clock.

        Returns
        -------
        DBnomicsDataClient
        """
        return DBnomicsDataClient(
            loop=loop,
            client_id=name,
            config=config,
            msgbus=msgbus,
            cache=cache,
            clock=clock,
        )


class DBnomicsInstrumentProviderFactory:
    """
    Provides DBnomics instrument providers from a configuration.
    """

    @staticmethod
    def create(
        config: DBnomicsDataClientConfig,
    ) -> DBnomicsInstrumentProvider:
        """
        Create a DBnomics instrument provider.

        Parameters
        ----------
        config : DBnomicsDataClientConfig
            The configuration.

        Returns
        -------
        DBnomicsInstrumentProvider
        """
        return DBnomicsInstrumentProvider(
            max_nb_series=config.max_nb_series,
            api_base_url=config.api_base_url,
            timeout=config.timeout,
        )