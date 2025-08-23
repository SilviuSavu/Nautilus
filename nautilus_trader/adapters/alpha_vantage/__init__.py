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
Alpha Vantage adapter for live market data and fundamental analysis.

This adapter provides access to Alpha Vantage's market data API including:
- Real-time stock quotes
- Daily and intraday OHLCV data
- Company fundamental data (earnings, financials, ratios)
- Symbol search and discovery

The adapter implements proper rate limiting (5 requests/minute, 500 requests/day for free tier)
and integrates with NautilusTrader's message bus for real-time data distribution.
"""

from nautilus_trader.adapters.alpha_vantage.client import AlphaVantageDataClient
from nautilus_trader.adapters.alpha_vantage.config import AlphaVantageDataClientConfig
from nautilus_trader.adapters.alpha_vantage.config import AlphaVantageInstrumentProviderConfig
from nautilus_trader.adapters.alpha_vantage.data import AlphaVantageBar
from nautilus_trader.adapters.alpha_vantage.data import AlphaVantageQuote
from nautilus_trader.adapters.alpha_vantage.data import AlphaVantageCompanyData
from nautilus_trader.adapters.alpha_vantage.factories import AlphaVantageLiveDataClientFactory
from nautilus_trader.adapters.alpha_vantage.factories import create_alpha_vantage_data_client
from nautilus_trader.adapters.alpha_vantage.factories import create_alpha_vantage_instrument_provider
from nautilus_trader.adapters.alpha_vantage.providers import AlphaVantageInstrumentProvider

__all__ = [
    "AlphaVantageDataClient",
    "AlphaVantageDataClientConfig", 
    "AlphaVantageInstrumentProviderConfig",
    "AlphaVantageBar",
    "AlphaVantageQuote", 
    "AlphaVantageCompanyData",
    "AlphaVantageLiveDataClientFactory",
    "AlphaVantageInstrumentProvider",
    "create_alpha_vantage_data_client",
    "create_alpha_vantage_instrument_provider",
]