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

import pandas as pd
import pytest

from nautilus_trader.adapters.yfinance.data import YFinanceDataClient
from nautilus_trader.model.data import BarType
from nautilus_trader.model.enums import BarAggregation
from nautilus_trader.model.enums import PriceType
from nautilus_trader.model.identifiers import InstrumentId


class TestYFinanceDataClient:
    """Tests for YFinanceDataClient."""

    def test_init(self, data_client):
        """Test initialization of data client."""
        assert data_client is not None
        assert str(data_client.id) == "YFinanceTest"
        assert data_client._config is not None

    @pytest.mark.asyncio
    async def test_connect(self, data_client):
        """Test client connection."""
        await data_client._connect()
        # Should connect without errors

    @pytest.mark.asyncio
    async def test_disconnect(self, data_client):
        """Test client disconnection."""
        await data_client._disconnect()
        # Should disconnect without errors
        assert len(data_client._ticker_cache) == 0

    def test_reset(self, data_client):
        """Test client reset."""
        # Add some test data to cache
        data_client._ticker_cache["AAPL"] = ("test_ticker", 123456)
        data_client._last_request_time = 123456
        
        data_client.reset()
        
        assert len(data_client._ticker_cache) == 0
        assert data_client._last_request_time == 0.0

    def test_dispose(self, data_client):
        """Test client disposal."""
        # Add some test data to cache
        data_client._ticker_cache["AAPL"] = ("test_ticker", 123456)
        
        data_client.dispose()
        
        assert len(data_client._ticker_cache) == 0

    def test_convert_bar_spec_to_interval(self, data_client):
        """Test converting Nautilus bar spec to yfinance interval."""
        # Test minute intervals
        bar_type = BarType.from_str("AAPL.YAHOO-1-MINUTE-LAST-EXTERNAL")
        interval = data_client._convert_bar_spec_to_interval(bar_type)
        assert interval == "1m"
        
        bar_type = BarType.from_str("AAPL.YAHOO-5-MINUTE-LAST-EXTERNAL")
        interval = data_client._convert_bar_spec_to_interval(bar_type)
        assert interval == "5m"
        
        bar_type = BarType.from_str("AAPL.YAHOO-15-MINUTE-LAST-EXTERNAL")
        interval = data_client._convert_bar_spec_to_interval(bar_type)
        assert interval == "15m"
        
        # Test hour intervals
        bar_type = BarType.from_str("AAPL.YAHOO-1-HOUR-LAST-EXTERNAL")
        interval = data_client._convert_bar_spec_to_interval(bar_type)
        assert interval == "1h"
        
        # Test day intervals
        bar_type = BarType.from_str("AAPL.YAHOO-1-DAY-LAST-EXTERNAL")
        interval = data_client._convert_bar_spec_to_interval(bar_type)
        assert interval == "1d"

    def test_determine_period(self, data_client):
        """Test determining period for yfinance request."""
        # Test with no start/end
        period = data_client._determine_period(None, None)
        assert period == data_client._default_period
        
        # Test with date range
        start = pd.Timestamp("2023-01-01", tz="UTC")
        end = pd.Timestamp("2023-01-05", tz="UTC")
        period = data_client._determine_period(start, end)
        assert period == "5d"
        
        # Test longer range
        start = pd.Timestamp("2023-01-01", tz="UTC")
        end = pd.Timestamp("2023-03-01", tz="UTC")
        period = data_client._determine_period(start, end)
        assert period == "3mo"

    def test_convert_to_nautilus_bars(self, data_client):
        """Test converting yfinance DataFrame to Nautilus bars."""
        # Create sample yfinance data
        data = pd.DataFrame({
            "Open": [100.0, 101.0],
            "High": [102.0, 103.0], 
            "Low": [99.0, 100.0],
            "Close": [101.0, 102.0],
            "Volume": [1000000, 1100000],
        }, index=pd.to_datetime(["2023-01-01", "2023-01-02"], utc=True))
        
        bar_type = BarType.from_str("AAPL.YAHOO-1-DAY-LAST-EXTERNAL")
        bars = data_client._convert_to_nautilus_bars(data, bar_type)
        
        assert len(bars) == 2
        assert bars[0].bar_type == bar_type
        assert float(bars[0].open) == 100.0
        assert float(bars[0].high) == 102.0
        assert float(bars[0].low) == 99.0
        assert float(bars[0].close) == 101.0
        assert int(bars[0].volume) == 1000000

    @pytest.mark.asyncio
    async def test_get_ticker_caching(self, data_client):
        """Test ticker caching mechanism."""
        # This test would require mocking yfinance.Ticker
        # In a real implementation, we would mock the yfinance.Ticker call
        pass

    @pytest.mark.asyncio
    async def test_fetch_historical_data_rate_limiting(self, data_client):
        """Test rate limiting in historical data fetching."""
        # This test would require mocking yfinance and testing timing
        # In a real implementation, we would verify rate limiting behavior
        pass

    @pytest.mark.asyncio
    async def test_request_bars_integration(self, data_client):
        """Test full bars request integration."""
        # This would be an integration test requiring actual API access
        # In practice, this would test against real Yahoo Finance data
        # but should be marked as requiring internet access
        pass

    def test_config_validation(self, config):
        """Test configuration validation."""
        assert config.cache_expiry_seconds == 60
        assert config.rate_limit_delay == 0.01
        assert config.request_timeout == 5.0
        assert config.symbols == ["AAPL", "MSFT"]