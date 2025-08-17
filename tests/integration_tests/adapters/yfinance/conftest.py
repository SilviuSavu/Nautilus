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
from unittest.mock import Mock

import pytest

from nautilus_trader.adapters.yfinance.config import YFinanceDataClientConfig
from nautilus_trader.adapters.yfinance.data import YFinanceDataClient
from nautilus_trader.adapters.yfinance.providers import YFinanceInstrumentProvider
from nautilus_trader.cache.cache import Cache
from nautilus_trader.common.component import LiveClock
from nautilus_trader.common.component import MessageBus
from nautilus_trader.config import InstrumentProviderConfig
from nautilus_trader.model.identifiers import TraderId
from nautilus_trader.test_kit.mocks import MockActor


@pytest.fixture
def event_loop():
    """Create event loop for async tests."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def clock():
    """Create a LiveClock for testing."""
    return LiveClock()


@pytest.fixture
def trader_id():
    """Create a TraderId for testing."""
    return TraderId("TESTER-001")


@pytest.fixture
def msgbus(trader_id):
    """Create a MessageBus for testing."""
    return MessageBus(
        trader_id=trader_id,
        clock=LiveClock(),
        name="MessageBus",
    )


@pytest.fixture
def cache():
    """Create a Cache for testing."""
    return Cache()


@pytest.fixture
def instrument_provider():
    """Create a YFinanceInstrumentProvider for testing."""
    return YFinanceInstrumentProvider()


@pytest.fixture
def config():
    """Create a YFinanceDataClientConfig for testing."""
    return YFinanceDataClientConfig(
        cache_expiry_seconds=60,  # Short cache for testing
        rate_limit_delay=0.01,   # Minimal delay for testing
        request_timeout=5.0,     # Short timeout for testing
        symbols=["AAPL", "MSFT"], # Test symbols
    )


@pytest.fixture
def data_client(event_loop, msgbus, cache, clock, instrument_provider, config):
    """Create a YFinanceDataClient for testing."""
    return YFinanceDataClient(
        loop=event_loop,
        msgbus=msgbus,
        cache=cache,
        clock=clock,
        instrument_provider=instrument_provider,
        config=config,
        name="YFinanceTest",
    )