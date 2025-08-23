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

from decimal import Decimal
from typing import Any

from nautilus_trader.core.data import Data
from nautilus_trader.model.identifiers import InstrumentId


class AlphaVantageBar(Data):
    """
    Represents an OHLCV bar from Alpha Vantage.
    
    Parameters
    ----------
    instrument_id : InstrumentId
        The instrument ID for the bar.
    open : Decimal
        The opening price.
    high : Decimal
        The highest price.
    low : Decimal
        The lowest price.
    close : Decimal
        The closing price.
    volume : int
        The trading volume.
    adjusted_close : Decimal | None, default None
        The adjusted closing price (for splits/dividends).
    dividend_amount : Decimal | None, default None
        The dividend amount (if applicable).
    split_coefficient : Decimal | None, default None
        The split coefficient (if applicable).
    ts_event : int
        The UNIX timestamp (nanoseconds) when the bar data was recorded.
    ts_init : int
        The UNIX timestamp (nanoseconds) when the data object was initialized.
    """

    def __init__(
        self,
        instrument_id: InstrumentId,
        open: Decimal,
        high: Decimal, 
        low: Decimal,
        close: Decimal,
        volume: int,
        adjusted_close: Decimal | None = None,
        dividend_amount: Decimal | None = None,
        split_coefficient: Decimal | None = None,
        ts_event: int = 0,
        ts_init: int = 0,
    ) -> None:
        super().__init__(ts_event, ts_init)
        
        self.instrument_id = instrument_id
        self.open = open
        self.high = high
        self.low = low
        self.close = close
        self.volume = volume
        self.adjusted_close = adjusted_close
        self.dividend_amount = dividend_amount
        self.split_coefficient = split_coefficient

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"instrument_id={self.instrument_id}, "
            f"open={self.open}, "
            f"high={self.high}, "
            f"low={self.low}, "
            f"close={self.close}, "
            f"volume={self.volume}, "
            f"ts_event={self.ts_event})"
        )


class AlphaVantageQuote(Data):
    """
    Represents a real-time quote from Alpha Vantage.
    
    Parameters
    ----------
    instrument_id : InstrumentId
        The instrument ID for the quote.
    symbol : str
        The symbol for the quote.
    open : Decimal
        The opening price for the day.
    high : Decimal
        The highest price for the day.
    low : Decimal
        The lowest price for the day. 
    price : Decimal
        The current/latest price.
    volume : int
        The trading volume for the day.
    latest_trading_day : str
        The latest trading day.
    previous_close : Decimal
        The previous day's closing price.
    change : Decimal
        The price change from previous close.
    change_percent : str
        The percentage change from previous close.
    ts_event : int
        The UNIX timestamp (nanoseconds) when the quote was recorded.
    ts_init : int
        The UNIX timestamp (nanoseconds) when the data object was initialized.
    """

    def __init__(
        self,
        instrument_id: InstrumentId,
        symbol: str,
        open: Decimal,
        high: Decimal,
        low: Decimal,
        price: Decimal,
        volume: int,
        latest_trading_day: str,
        previous_close: Decimal,
        change: Decimal,
        change_percent: str,
        ts_event: int = 0,
        ts_init: int = 0,
    ) -> None:
        super().__init__(ts_event, ts_init)
        
        self.instrument_id = instrument_id
        self.symbol = symbol
        self.open = open
        self.high = high
        self.low = low
        self.price = price
        self.volume = volume
        self.latest_trading_day = latest_trading_day
        self.previous_close = previous_close
        self.change = change
        self.change_percent = change_percent

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"instrument_id={self.instrument_id}, "
            f"symbol={self.symbol}, "
            f"price={self.price}, "
            f"change={self.change}, "
            f"change_percent={self.change_percent}, "
            f"ts_event={self.ts_event})"
        )


class AlphaVantageCompanyData(Data):
    """
    Represents company fundamental data from Alpha Vantage.
    
    Parameters
    ----------
    instrument_id : InstrumentId
        The instrument ID for the company.
    symbol : str
        The stock symbol.
    name : str
        The company name.
    description : str | None
        Company description.
    exchange : str | None
        The exchange where the stock is traded.
    currency : str | None
        The trading currency.
    country : str | None
        The country of incorporation.
    sector : str | None
        The business sector.
    industry : str | None
        The business industry.
    market_cap : int | None
        Market capitalization.
    pe_ratio : Decimal | None
        Price-to-earnings ratio.
    peg_ratio : Decimal | None
        Price/earnings-to-growth ratio.
    book_value : Decimal | None
        Book value per share.
    dividend_yield : Decimal | None
        Dividend yield.
    eps : Decimal | None
        Earnings per share.
    revenue_ttm : int | None
        Trailing twelve months revenue.
    gross_profit_ttm : int | None
        Trailing twelve months gross profit.
    ebitda : int | None
        Earnings before interest, taxes, depreciation, and amortization.
    ts_event : int
        The UNIX timestamp (nanoseconds) when the data was recorded.
    ts_init : int
        The UNIX timestamp (nanoseconds) when the data object was initialized.
    """

    def __init__(
        self,
        instrument_id: InstrumentId,
        symbol: str,
        name: str,
        description: str | None = None,
        exchange: str | None = None,
        currency: str | None = None,
        country: str | None = None,
        sector: str | None = None,
        industry: str | None = None,
        market_cap: int | None = None,
        pe_ratio: Decimal | None = None,
        peg_ratio: Decimal | None = None,
        book_value: Decimal | None = None,
        dividend_yield: Decimal | None = None,
        eps: Decimal | None = None,
        revenue_ttm: int | None = None,
        gross_profit_ttm: int | None = None,
        ebitda: int | None = None,
        ts_event: int = 0,
        ts_init: int = 0,
    ) -> None:
        super().__init__(ts_event, ts_init)
        
        self.instrument_id = instrument_id
        self.symbol = symbol
        self.name = name
        self.description = description
        self.exchange = exchange
        self.currency = currency
        self.country = country
        self.sector = sector
        self.industry = industry
        self.market_cap = market_cap
        self.pe_ratio = pe_ratio
        self.peg_ratio = peg_ratio
        self.book_value = book_value
        self.dividend_yield = dividend_yield
        self.eps = eps
        self.revenue_ttm = revenue_ttm
        self.gross_profit_ttm = gross_profit_ttm
        self.ebitda = ebitda

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"instrument_id={self.instrument_id}, "
            f"symbol={self.symbol}, "
            f"name='{self.name}', "
            f"sector={self.sector}, "
            f"market_cap={self.market_cap}, "
            f"ts_event={self.ts_event})"
        )