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

import os
from typing import Any

from nautilus_trader.config import NautilusConfig
from nautilus_trader.live.config import LiveDataClientConfig
from nautilus_trader.model.identifiers import InstrumentId


class AlphaVantageDataClientConfig(LiveDataClientConfig, frozen=True):
    """
    Configuration for `AlphaVantageDataClient` instances.
    
    Parameters
    ----------
    api_key : str
        The Alpha Vantage API key. Get a free key from: https://www.alphavantage.co/support/#api-key
    base_url : str, default "https://www.alphavantage.co"
        The Alpha Vantage API base URL.
    request_timeout : float, default 30.0
        The request timeout for HTTP calls in seconds.
    rate_limit_delay : float, default 12.0
        The delay between requests in seconds (free tier: 5 req/min = 12s delay).
    default_limit : int, default 100
        The default number of data points to request.
    update_interval : float, default 300.0
        The interval for checking data updates in seconds (5 minutes).
    auto_subscribe : bool, default False
        Whether to automatically subscribe to configured instruments.
    instrument_ids : list[InstrumentId] | None, default None
        The initial instrument IDs to load and optionally subscribe to.
    symbols : list[str] | None, default None  
        The initial symbols to load and optionally subscribe to.
    """

    api_key: str = ""
    base_url: str = "https://www.alphavantage.co"
    request_timeout: float = 30.0
    rate_limit_delay: float = 12.0  # 5 requests per minute for free tier
    default_limit: int = 100
    update_interval: float = 300.0  # 5 minutes
    auto_subscribe: bool = False
    instrument_ids: list[InstrumentId] | None = None
    symbols: list[str] | None = None

    def __post_init__(self) -> None:
        if not self.api_key:
            # Try to get from environment
            api_key = os.getenv("ALPHA_VANTAGE_API_KEY")
            if api_key:
                object.__setattr__(self, "api_key", api_key)
            else:
                raise ValueError(
                    "Alpha Vantage API key is required. Provide it via config or set ALPHA_VANTAGE_API_KEY "
                    "environment variable. Get a free API key from: https://www.alphavantage.co/support/#api-key"
                )


class AlphaVantageInstrumentProviderConfig(NautilusConfig, frozen=True):
    """
    Configuration for `AlphaVantageInstrumentProvider` instances.
    
    Parameters
    ----------
    api_key : str
        The Alpha Vantage API key.
    base_url : str, default "https://www.alphavantage.co"
        The Alpha Vantage API base URL.
    request_timeout : float, default 30.0
        The request timeout for HTTP calls in seconds.
    rate_limit_delay : float, default 12.0
        The delay between requests in seconds.
    symbols : list[str] | None, default None
        The initial symbols to load instrument data for.
    load_all : bool, default False
        Whether to load all available instruments (not recommended for Alpha Vantage).
    search_terms : list[str] | None, default None
        Search terms to use for finding instruments.
    max_search_results : int, default 100
        Maximum number of search results to return.
    cache_instruments : bool, default True
        Whether to cache instrument metadata.
    """

    api_key: str = ""
    base_url: str = "https://www.alphavantage.co"
    request_timeout: float = 30.0
    rate_limit_delay: float = 12.0
    symbols: list[str] | None = None
    load_all: bool = False
    search_terms: list[str] | None = None
    max_search_results: int = 100
    cache_instruments: bool = True

    def __post_init__(self) -> None:
        if not self.api_key:
            # Try to get from environment
            api_key = os.getenv("ALPHA_VANTAGE_API_KEY")
            if api_key:
                object.__setattr__(self, "api_key", api_key)
            else:
                raise ValueError(
                    "Alpha Vantage API key is required. Provide it via config or set ALPHA_VANTAGE_API_KEY "
                    "environment variable. Get a free API key from: https://www.alphavantage.co/support/#api-key"
                )