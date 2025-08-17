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

import asyncio
from functools import lru_cache

from nautilus_trader.adapters.yfinance.config import YFinanceDataClientConfig
from nautilus_trader.adapters.yfinance.data import YFinanceDataClient
from nautilus_trader.adapters.yfinance.providers import YFinanceInstrumentProvider
from nautilus_trader.cache.cache import Cache
from nautilus_trader.common.component import LiveClock
from nautilus_trader.common.component import MessageBus
from nautilus_trader.config import InstrumentProviderConfig
from nautilus_trader.live.factories import LiveDataClientFactory


@lru_cache(1)
def get_cached_yfinance_instrument_provider(
    config: InstrumentProviderConfig | None = None,
) -> YFinanceInstrumentProvider:
    """
    Cache and return a YFinance instrument provider.

    If a cached provider already exists, then that provider will be returned.

    Parameters
    ----------
    config : InstrumentProviderConfig, optional
        The configuration for the instrument provider.

    Returns
    -------
    YFinanceInstrumentProvider

    """
    return YFinanceInstrumentProvider(config=config)


class YFinanceLiveDataClientFactory(LiveDataClientFactory):
    """
    Provides a YFinance live data client factory.
    """

    @staticmethod
    def create(  # type: ignore
        loop: asyncio.AbstractEventLoop,
        name: str,
        config: YFinanceDataClientConfig,
        msgbus: MessageBus,
        cache: Cache,
        clock: LiveClock,
    ) -> YFinanceDataClient:
        """
        Create a new YFinance data client.

        Parameters
        ----------
        loop : asyncio.AbstractEventLoop
            The event loop for the client.
        name : str
            The custom client name.
        config : YFinanceDataClientConfig
            The client configuration.
        msgbus : MessageBus
            The message bus for the client.
        cache : Cache
            The cache for the client.
        clock : LiveClock
            The clock for the client.

        Returns
        -------
        YFinanceDataClient

        """
        # Get instrument provider
        provider = get_cached_yfinance_instrument_provider(
            config=config.instrument_provider,
        )

        return YFinanceDataClient(
            loop=loop,
            msgbus=msgbus,
            cache=cache,
            clock=clock,
            instrument_provider=provider,
            config=config,
            name=name,
        )


def create_yfinance_data_client(
    loop: asyncio.AbstractEventLoop,
    msgbus: MessageBus,
    cache: Cache,
    clock: LiveClock,
    config: YFinanceDataClientConfig | None = None,
    name: str | None = None,
) -> YFinanceDataClient:
    """
    Create a YFinance data client.

    This is a convenience function for creating a YFinance data client
    without using the factory pattern.

    Parameters
    ----------
    loop : asyncio.AbstractEventLoop
        The event loop for the client.
    msgbus : MessageBus
        The message bus for the client.
    cache : Cache
        The cache for the client.
    clock : LiveClock
        The clock for the client.
    config : YFinanceDataClientConfig, optional
        The client configuration.
    name : str, optional
        The custom client name.

    Returns
    -------
    YFinanceDataClient

    """
    if config is None:
        config = YFinanceDataClientConfig()

    provider = YFinanceInstrumentProvider(config=config.instrument_provider)

    return YFinanceDataClient(
        loop=loop,
        msgbus=msgbus,
        cache=cache,
        clock=clock,
        instrument_provider=provider,
        config=config,
        name=name,
    )


def create_yfinance_instrument_provider(
    config: InstrumentProviderConfig | None = None,
) -> YFinanceInstrumentProvider:
    """
    Create a YFinance instrument provider.

    This is a convenience function for creating a YFinance instrument provider.

    Parameters
    ----------
    config : InstrumentProviderConfig, optional
        The configuration for the instrument provider.

    Returns
    -------
    YFinanceInstrumentProvider

    """
    return YFinanceInstrumentProvider(config=config)