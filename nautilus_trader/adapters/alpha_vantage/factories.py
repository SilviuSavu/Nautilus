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

from nautilus_trader.adapters.alpha_vantage.client import AlphaVantageDataClient
from nautilus_trader.adapters.alpha_vantage.config import AlphaVantageDataClientConfig
from nautilus_trader.adapters.alpha_vantage.config import AlphaVantageInstrumentProviderConfig
from nautilus_trader.adapters.alpha_vantage.providers import AlphaVantageInstrumentProvider
from nautilus_trader.cache.cache import Cache
from nautilus_trader.common.component import LiveClock
from nautilus_trader.common.component import MessageBus
from nautilus_trader.live.factories import LiveDataClientFactory


# Global instrument provider cache
_ALPHA_VANTAGE_INSTRUMENT_PROVIDERS: dict[str, AlphaVantageInstrumentProvider] = {}


class AlphaVantageLiveDataClientFactory(LiveDataClientFactory):
    """
    Provides an `AlphaVantageDataClient` factory.
    """

    @staticmethod
    def create(
        loop: asyncio.AbstractEventLoop,
        name: str,
        config: dict[str, Any],
        msgbus: MessageBus,
        cache: Cache,
        clock: LiveClock,
    ) -> AlphaVantageDataClient:
        """
        Create an Alpha Vantage data client.

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
        AlphaVantageDataClient
        """
        # Create data client config
        client_config = AlphaVantageDataClientConfig(**config)
        
        # Create or get cached instrument provider
        provider_config = AlphaVantageInstrumentProviderConfig(
            api_key=client_config.api_key,
            base_url=client_config.base_url,
            request_timeout=client_config.request_timeout,
            rate_limit_delay=client_config.rate_limit_delay,
            symbols=client_config.symbols or [],
        )
        
        instrument_provider = get_cached_alpha_vantage_instrument_provider(provider_config)

        return AlphaVantageDataClient(
            loop=loop,
            msgbus=msgbus,
            cache=cache,
            clock=clock,
            instrument_provider=instrument_provider,
            config=client_config,
            name=name,
        )


def create_alpha_vantage_data_client(
    loop: asyncio.AbstractEventLoop,
    msgbus: MessageBus,
    cache: Cache,
    clock: LiveClock,
    instrument_provider: AlphaVantageInstrumentProvider,
    config: AlphaVantageDataClientConfig | None = None,
    name: str | None = None,
) -> AlphaVantageDataClient:
    """
    Create an Alpha Vantage data client.

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
    instrument_provider : AlphaVantageInstrumentProvider
        The instrument provider for the client.
    config : AlphaVantageDataClientConfig, optional
        The configuration for the client.
    name : str, optional
        The client name.

    Returns
    -------
    AlphaVantageDataClient
    """
    return AlphaVantageDataClient(
        loop=loop,
        msgbus=msgbus,
        cache=cache,
        clock=clock,
        instrument_provider=instrument_provider,
        config=config,
        name=name,
    )


def create_alpha_vantage_instrument_provider(
    config: AlphaVantageInstrumentProviderConfig | None = None,
    api_key: str | None = None,
) -> AlphaVantageInstrumentProvider:
    """
    Create an Alpha Vantage instrument provider.

    Parameters
    ----------
    config : AlphaVantageInstrumentProviderConfig, optional
        The configuration for the provider.
    api_key : str, optional
        The Alpha Vantage API key. If not provided, will try to get from config or environment.

    Returns
    -------
    AlphaVantageInstrumentProvider

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
        final_api_key = os.getenv("ALPHA_VANTAGE_API_KEY")
        
    if not final_api_key:
        raise ValueError(
            "Alpha Vantage API key is required. Provide it via config, parameter, or ALPHA_VANTAGE_API_KEY "
            "environment variable. Get a free API key from: https://www.alphavantage.co/support/#api-key"
        )
    
    if config is None:
        config = AlphaVantageInstrumentProviderConfig(api_key=final_api_key)
    elif not config.api_key:
        # Update config with API key
        config = AlphaVantageInstrumentProviderConfig(
            api_key=final_api_key,
            base_url=config.base_url,
            request_timeout=config.request_timeout,
            rate_limit_delay=config.rate_limit_delay,
            symbols=config.symbols,
            load_all=config.load_all,
            search_terms=config.search_terms,
            max_search_results=config.max_search_results,
            cache_instruments=config.cache_instruments,
        )

    return AlphaVantageInstrumentProvider(config=config)


def get_cached_alpha_vantage_instrument_provider(
    config: AlphaVantageInstrumentProviderConfig,
) -> AlphaVantageInstrumentProvider:
    """
    Get a cached Alpha Vantage instrument provider.

    If a provider with the same configuration exists in cache, returns it.
    Otherwise creates a new provider and caches it.

    Parameters
    ----------
    config : AlphaVantageInstrumentProviderConfig
        The configuration for the provider.

    Returns
    -------
    AlphaVantageInstrumentProvider
    """
    # Create cache key based on config
    cache_key = f"{config.api_key}_{config.base_url}_{hash(tuple(config.symbols or []))}"
    
    if cache_key not in _ALPHA_VANTAGE_INSTRUMENT_PROVIDERS:
        _ALPHA_VANTAGE_INSTRUMENT_PROVIDERS[cache_key] = create_alpha_vantage_instrument_provider(config)
        
    return _ALPHA_VANTAGE_INSTRUMENT_PROVIDERS[cache_key]


def clear_alpha_vantage_instrument_provider_cache() -> None:
    """
    Clear the cached Alpha Vantage instrument providers.
    
    This is useful for testing or when you want to force recreation of providers.
    """
    global _ALPHA_VANTAGE_INSTRUMENT_PROVIDERS
    _ALPHA_VANTAGE_INSTRUMENT_PROVIDERS.clear()