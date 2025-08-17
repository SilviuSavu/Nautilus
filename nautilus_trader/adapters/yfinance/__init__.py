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
Yahoo Finance market data integration adapter.

This subpackage provides a data client factory, instrument provider,
configurations, and utilities for connecting to and interacting with
the Yahoo Finance API through the yfinance library.

The adapter provides historical market data access with intelligent caching,
rate limiting, and robust error handling. It supports OHLCV data for stocks,
ETFs, currencies, and other instruments available on Yahoo Finance.

Key Features:
- Historical data access for any Yahoo Finance symbol
- Intelligent caching to reduce API calls  
- Rate limiting to respect Yahoo Finance limits
- Automatic data conversion to Nautilus Bar objects
- Proper timezone handling (UTC)
- Robust exception handling for API failures

Limitations:
- No real-time streaming (historical data only)
- Rate limiting constraints from Yahoo Finance
- Limited to OHLCV data (no order book or tick data)

For convenience, the most commonly used symbols are re-exported at the
subpackage's top level, so downstream code can simply import from
``nautilus_trader.adapters.yfinance``.

"""
from nautilus_trader.adapters.yfinance.config import YFinanceDataClientConfig
from nautilus_trader.adapters.yfinance.data import YFinanceDataClient
from nautilus_trader.adapters.yfinance.factories import YFinanceLiveDataClientFactory
from nautilus_trader.adapters.yfinance.factories import create_yfinance_data_client
from nautilus_trader.adapters.yfinance.factories import create_yfinance_instrument_provider
from nautilus_trader.adapters.yfinance.factories import get_cached_yfinance_instrument_provider
from nautilus_trader.adapters.yfinance.providers import YFinanceInstrumentProvider


__all__ = [
    "YFinanceDataClientConfig",
    "YFinanceDataClient", 
    "YFinanceInstrumentProvider",
    "YFinanceLiveDataClientFactory",
    "create_yfinance_data_client",
    "create_yfinance_instrument_provider", 
    "get_cached_yfinance_instrument_provider",
]