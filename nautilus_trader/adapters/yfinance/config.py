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

from nautilus_trader.config import LiveDataClientConfig
from nautilus_trader.model.identifiers import InstrumentId


class YFinanceDataClientConfig(LiveDataClientConfig, frozen=True):
    """
    Configuration for ``YFinanceDataClient`` instances.

    Parameters
    ----------
    cache_expiry_seconds : int, default 3600
        The time (seconds) to cache yfinance Ticker objects before re-creating them.
        Reduces API calls by reusing Ticker instances for multiple requests.
    rate_limit_delay : float, default 0.1
        The minimum delay (seconds) between consecutive API calls to respect Yahoo Finance limits.
        Helps prevent rate limiting and connection errors.
    request_timeout : float, default 30.0
        The timeout (seconds) for individual yfinance API requests.
        If a request takes longer than this, it will be cancelled.
    max_retries : int, default 3
        The maximum number of retry attempts for failed API requests.
        Helps handle transient network issues and temporary API failures.
    retry_delay : float, default 1.0
        The delay (seconds) between retry attempts for failed requests.
        Exponential backoff is applied: delay * (2 ** attempt_number).
    default_period : str, default "1y"
        The default time period for historical data requests when not specified.
        Valid periods: 1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max.
    default_interval : str, default "1d"
        The default bar interval for historical data requests when not specified.
        Valid intervals: 1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo.
    instrument_ids : list[InstrumentId], optional
        The instrument IDs to load instrument definitions for on start.
        If None, instruments must be loaded manually or requested dynamically.
    symbols : list[str], optional
        The Yahoo Finance symbols to create instruments for on start.
        These will be converted to InstrumentId objects using the YAHOO venue.

    """

    cache_expiry_seconds: int = 3600
    rate_limit_delay: float = 0.1
    request_timeout: float = 30.0
    max_retries: int = 3
    retry_delay: float = 1.0
    default_period: str = "1y"
    default_interval: str = "1d"
    instrument_ids: list[InstrumentId] | None = None
    symbols: list[str] | None = None