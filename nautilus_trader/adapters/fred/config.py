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

from nautilus_trader.config import InstrumentProviderConfig
from nautilus_trader.config import LiveDataClientConfig
from nautilus_trader.model.identifiers import InstrumentId


class FREDDataClientConfig(LiveDataClientConfig, frozen=True):
    """
    Configuration for ``FREDDataClient`` instances.

    Parameters
    ----------
    api_key : str
        The FRED API key for authentication. Can be obtained free from
        https://fred.stlouisfed.org/docs/api/api_key.html
    base_url : str, default "https://api.stlouisfed.org/fred"
        The base URL for the FRED API.
    request_timeout : float, default 30.0
        The timeout (seconds) for individual FRED API requests.
        If a request takes longer than this, it will be cancelled.
    rate_limit_delay : float, default 1.0
        The minimum delay (seconds) between consecutive API calls to respect FRED limits.
        FRED allows 120 requests per 60 seconds, so 1 second delay provides safety margin.
    max_retries : int, default 3
        The maximum number of retry attempts for failed API requests.
        Helps handle transient network issues and temporary API failures.
    retry_delay : float, default 2.0
        The delay (seconds) between retry attempts for failed requests.
        Exponential backoff is applied: delay * (2 ** attempt_number).
    default_limit : int, default 1000
        The default maximum number of observations to retrieve per request.
        FRED allows up to 100,000 observations per request.
    update_interval : int, default 3600
        The interval (seconds) for checking for new economic data updates.
        Economic data is typically published monthly or quarterly.
    instrument_ids : list[InstrumentId], optional
        The instrument IDs to load and subscribe to on start.
        If None, instruments must be loaded manually or requested dynamically.
    series_ids : list[str], optional
        The FRED series IDs to create instruments for on start.
        These will be converted to InstrumentId objects using the FRED venue.
    auto_subscribe : bool, default False
        Whether to automatically subscribe to data updates for loaded instruments.
        If True, the client will periodically check for new data.

    """

    api_key: str
    base_url: str = "https://api.stlouisfed.org/fred"
    request_timeout: float = 30.0
    rate_limit_delay: float = 1.0
    max_retries: int = 3
    retry_delay: float = 2.0
    default_limit: int = 1000
    update_interval: int = 3600
    instrument_ids: list[InstrumentId] | None = None
    series_ids: list[str] | None = None
    auto_subscribe: bool = False


class FREDInstrumentProviderConfig(InstrumentProviderConfig, frozen=True):
    """
    Configuration for ``FREDInstrumentProvider`` instances.

    Parameters
    ----------
    api_key : str
        The FRED API key for authentication.
    base_url : str, default "https://api.stlouisfed.org/fred"
        The base URL for the FRED API.
    request_timeout : float, default 30.0
        The timeout (seconds) for individual FRED API requests.
    rate_limit_delay : float, default 1.0
        The minimum delay (seconds) between consecutive API calls.
    series_ids : list[str], default []
        The FRED series IDs to load instrument definitions for.
        Popular series include: "GDP", "UNRATE", "CPIAUCSL", "FEDFUNDS", "DGS10".
    load_all : bool, default False
        Whether to attempt loading all available FRED series.
        Warning: This is not recommended as FRED has 800,000+ series.
    category_ids : list[int], default []
        The FRED category IDs to load series from.
        Useful for loading all series within specific economic categories.
    search_terms : list[str], default []
        Search terms to find relevant FRED series.
        The provider will search for series matching these terms.
    max_search_results : int, default 100
        The maximum number of search results to process per search term.
    cache_instruments : bool, default True
        Whether to cache loaded instrument definitions to reduce API calls.

    """

    api_key: str
    base_url: str = "https://api.stlouisfed.org/fred"
    request_timeout: float = 30.0
    rate_limit_delay: float = 1.0
    series_ids: list[str] = []
    load_all: bool = False
    category_ids: list[int] = []
    search_terms: list[str] = []
    max_search_results: int = 100
    cache_instruments: bool = True