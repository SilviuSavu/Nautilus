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
Federal Reserve Economic Data (FRED) API integration adapter.

This subpackage provides a data client factory, instrument provider,
configurations, and utilities for connecting to and interacting with
the Federal Reserve Economic Data (FRED) API.

The adapter provides access to over 800,000 U.S. and international economic
time series from 89 sources. This rich economic data can be integrated
into NautilusTrader strategies for quantitative trading that incorporates
macroeconomic factors into decision-making processes.

Key Features:
- Access to 800,000+ economic time series
- Historical economic data retrieval
- Custom EconomicData type for economic indicators
- Rate limiting to respect FRED API constraints  
- Robust error handling for API failures
- Support for popular indicators (GDP, unemployment, inflation, interest rates)
- Automatic data conversion to Nautilus data objects
- Proper timezone handling (UTC)

Limitations:
- No real-time streaming (economic data is published with delays)
- Rate limiting constraints from FRED API
- Economic data has low frequency (monthly/quarterly updates)
- Requires free FRED API key from Federal Reserve Bank of St. Louis

For convenience, the most commonly used symbols are re-exported at the
subpackage's top level, so downstream code can simply import from
``nautilus_trader.adapters.fred``.

"""
from nautilus_trader.adapters.fred.config import FREDDataClientConfig
from nautilus_trader.adapters.fred.config import FREDInstrumentProviderConfig
from nautilus_trader.adapters.fred.client import FREDDataClient
from nautilus_trader.adapters.fred.data import EconomicData
from nautilus_trader.adapters.fred.factories import FREDLiveDataClientFactory
from nautilus_trader.adapters.fred.factories import create_fred_data_client
from nautilus_trader.adapters.fred.factories import create_fred_instrument_provider
from nautilus_trader.adapters.fred.factories import get_cached_fred_instrument_provider
from nautilus_trader.adapters.fred.providers import FREDInstrumentProvider


__all__ = [
    "FREDDataClientConfig",
    "FREDInstrumentProviderConfig", 
    "FREDDataClient",
    "EconomicData",
    "FREDInstrumentProvider",
    "FREDLiveDataClientFactory",
    "create_fred_data_client",
    "create_fred_instrument_provider",
    "get_cached_fred_instrument_provider",
]