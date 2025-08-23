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
import os
from typing import Any

from nautilus_trader.adapters.fred.client import FREDDataClient
from nautilus_trader.adapters.fred.config import FREDDataClientConfig
from nautilus_trader.adapters.fred.config import FREDInstrumentProviderConfig
from nautilus_trader.adapters.fred.providers import FREDInstrumentProvider
from nautilus_trader.cache.cache import Cache
from nautilus_trader.common.component import LiveClock
from nautilus_trader.common.component import MessageBus
from nautilus_trader.live.factories import LiveDataClientFactory


# Global instrument provider cache
_FRED_INSTRUMENT_PROVIDERS: dict[str, FREDInstrumentProvider] = {}


class FREDLiveDataClientFactory(LiveDataClientFactory):
    """
    Provides a `FREDDataClient` factory.
    """

    @staticmethod
    def create(
        loop: asyncio.AbstractEventLoop,
        name: str,
        config: dict[str, Any],
        msgbus: MessageBus,
        cache: Cache,
        clock: LiveClock,
    ) -> FREDDataClient:
        """
        Create a FRED data client.

        Parameters
        ----------
        loop : asyncio.AbstractEventLoop
            The event loop for the client.
        name : str
            The client name.
        config : dict[str, Any]
            The configuration dictionary.
        msgbus : MessageBus
            The message bus for the client.
        cache : Cache
            The cache for the client.
        clock : LiveClock
            The clock for the client.

        Returns
        -------
        FREDDataClient

        """
        # Create data client config
        client_config = FREDDataClientConfig(**config)
        
        # Create or get cached instrument provider
        provider_config = FREDInstrumentProviderConfig(
            api_key=client_config.api_key,
            base_url=client_config.base_url,
            request_timeout=client_config.request_timeout,
            rate_limit_delay=client_config.rate_limit_delay,
            series_ids=client_config.series_ids or [],
        )
        
        instrument_provider = get_cached_fred_instrument_provider(provider_config)

        return FREDDataClient(
            loop=loop,
            msgbus=msgbus,
            cache=cache,
            clock=clock,
            instrument_provider=instrument_provider,
            config=client_config,
            name=name,
        )


def create_fred_data_client(
    loop: asyncio.AbstractEventLoop,
    msgbus: MessageBus,
    cache: Cache,
    clock: LiveClock,
    instrument_provider: FREDInstrumentProvider,
    config: FREDDataClientConfig | None = None,
    name: str | None = None,
) -> FREDDataClient:
    """
    Create a FRED data client.

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
    instrument_provider : FREDInstrumentProvider
        The instrument provider for the client.
    config : FREDDataClientConfig, optional
        The configuration for the client.
    name : str, optional
        The client name.

    Returns
    -------
    FREDDataClient

    """
    return FREDDataClient(
        loop=loop,
        msgbus=msgbus,
        cache=cache,
        clock=clock,
        instrument_provider=instrument_provider,
        config=config,
        name=name,
    )


def create_fred_instrument_provider(
    config: FREDInstrumentProviderConfig | None = None,
    api_key: str | None = None,
) -> FREDInstrumentProvider:
    """
    Create a FRED instrument provider.

    Parameters
    ----------
    config : FREDInstrumentProviderConfig, optional
        The configuration for the provider.
    api_key : str, optional
        The FRED API key. If not provided, will try to get from config or environment.

    Returns
    -------
    FREDInstrumentProvider

    Raises
    ------
    ValueError
        If no API key is provided and none found in environment.

    """
    # Get API key from various sources
    if config and config.api_key:
        final_api_key = config.api_key
    elif api_key:
        final_api_key = api_key
    else:
        final_api_key = os.getenv("FRED_API_KEY")
        
    if not final_api_key:
        raise ValueError(
            "FRED API key is required. Provide it via config, parameter, or FRED_API_KEY environment variable. "
            "Get a free API key from: https://fred.stlouisfed.org/docs/api/api_key.html"
        )
    
    if config is None:
        config = FREDInstrumentProviderConfig(api_key=final_api_key)
    elif not config.api_key:
        # Update config with API key
        config = FREDInstrumentProviderConfig(
            api_key=final_api_key,
            base_url=config.base_url,
            request_timeout=config.request_timeout,
            rate_limit_delay=config.rate_limit_delay,
            series_ids=config.series_ids,
            load_all=config.load_all,
            category_ids=config.category_ids,
            search_terms=config.search_terms,
            max_search_results=config.max_search_results,
            cache_instruments=config.cache_instruments,
        )

    return FREDInstrumentProvider(config=config)


def get_cached_fred_instrument_provider(
    config: FREDInstrumentProviderConfig,
) -> FREDInstrumentProvider:
    """
    Get a cached FRED instrument provider.

    If a provider with the same configuration exists in cache, returns it.
    Otherwise creates a new provider and caches it.

    Parameters
    ----------
    config : FREDInstrumentProviderConfig
        The configuration for the provider.

    Returns
    -------
    FREDInstrumentProvider

    """
    # Create cache key based on config
    cache_key = f"{config.api_key}_{config.base_url}_{hash(tuple(config.series_ids))}"
    
    if cache_key not in _FRED_INSTRUMENT_PROVIDERS:
        _FRED_INSTRUMENT_PROVIDERS[cache_key] = create_fred_instrument_provider(config)
        
    return _FRED_INSTRUMENT_PROVIDERS[cache_key]


def clear_fred_instrument_provider_cache() -> None:
    """
    Clear the cached FRED instrument providers.
    
    This is useful for testing or when you want to force recreation of providers.
    """
    global _FRED_INSTRUMENT_PROVIDERS
    _FRED_INSTRUMENT_PROVIDERS.clear()