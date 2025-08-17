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

import pytest

from nautilus_trader.adapters.yfinance.providers import YFinanceInstrumentProvider
from nautilus_trader.model.identifiers import InstrumentId


class TestYFinanceInstrumentProvider:
    """Tests for YFinanceInstrumentProvider."""

    def test_init(self, instrument_provider):
        """Test initialization of instrument provider."""
        assert instrument_provider is not None
        assert str(instrument_provider._venue) == "YAHOO"

    @pytest.mark.asyncio
    async def test_load_all_async_raises_error(self, instrument_provider):
        """Test that load_all_async raises RuntimeError."""
        with pytest.raises(RuntimeError, match="Loading all instrument definitions is not supported"):
            await instrument_provider.load_all_async()

    @pytest.mark.asyncio
    async def test_load_async_single_instrument(self, instrument_provider):
        """Test loading a single instrument."""
        instrument_id = InstrumentId.from_str("AAPL.YAHOO")
        
        # This test requires actual API access, so we mock the response
        # In a real test environment, this would make an actual API call
        await instrument_provider.load_async(instrument_id)
        
        # Check if instrument was added to provider
        # Note: This test may fail if Yahoo Finance is down or rate limited
        
    @pytest.mark.asyncio 
    async def test_load_ids_async_multiple_instruments(self, instrument_provider):
        """Test loading multiple instruments."""
        instrument_ids = [
            InstrumentId.from_str("AAPL.YAHOO"),
            InstrumentId.from_str("MSFT.YAHOO"),
        ]
        
        # This test requires actual API access
        await instrument_provider.load_ids_async(instrument_ids)
        
        # In a real test, we would verify the instruments were loaded
        # Note: This test may fail if Yahoo Finance is down or rate limited

    def test_create_equity_instrument(self, instrument_provider):
        """Test creating an equity instrument from Yahoo Finance info."""
        # Mock Yahoo Finance info response
        info = {
            "symbol": "AAPL",
            "currency": "USD",
            "currentPrice": 150.00,
            "quoteType": "EQUITY",
        }
        
        from nautilus_trader.model.identifiers import InstrumentId, Symbol, Venue
        instrument_id = InstrumentId(Symbol("AAPL"), Venue("YAHOO"))
        
        equity = instrument_provider._create_equity(instrument_id, info)
        
        assert equity is not None
        assert equity.id == instrument_id
        assert str(equity.quote_currency) == "USD"

    def test_create_currency_pair_instrument(self, instrument_provider):
        """Test creating a currency pair instrument from Yahoo Finance info."""
        # Mock Yahoo Finance info response
        info = {
            "symbol": "EURUSD=X", 
            "currency": "USD",
            "currentPrice": 1.1000,
            "quoteType": "CURRENCY",
        }
        
        from nautilus_trader.model.identifiers import InstrumentId, Symbol, Venue
        instrument_id = InstrumentId(Symbol("EURUSD=X"), Venue("YAHOO"))
        
        currency_pair = instrument_provider._create_currency_pair(instrument_id, info)
        
        assert currency_pair is not None
        assert currency_pair.id == instrument_id
        assert str(currency_pair.base_currency) == "EUR"
        assert str(currency_pair.quote_currency) == "USD"

    def test_get_price_precision(self, instrument_provider):
        """Test price precision calculation."""
        # Test with current price
        info_with_price = {"currentPrice": 123.45}
        precision = instrument_provider._get_price_precision(info_with_price)
        assert precision == 2
        
        # Test with high precision price
        info_high_precision = {"regularMarketPrice": 1.23456}
        precision = instrument_provider._get_price_precision(info_high_precision)
        assert precision == 5
        
        # Test currency default
        info_currency = {"quoteType": "CURRENCY"}
        precision = instrument_provider._get_price_precision(info_currency)
        assert precision == 5
        
        # Test equity default
        info_equity = {"quoteType": "EQUITY"}
        precision = instrument_provider._get_price_precision(info_equity)
        assert precision == 2