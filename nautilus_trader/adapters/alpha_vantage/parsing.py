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
from datetime import datetime
from typing import Any

from nautilus_trader.adapters.alpha_vantage.data import AlphaVantageBar
from nautilus_trader.adapters.alpha_vantage.data import AlphaVantageQuote
from nautilus_trader.adapters.alpha_vantage.data import AlphaVantageCompanyData
from nautilus_trader.model.identifiers import InstrumentId
from nautilus_trader.model.identifiers import Symbol
from nautilus_trader.model.identifiers import Venue


def parse_alpha_vantage_quote(
    symbol: str,
    quote_data: dict[str, Any],
    ts_init: int,
) -> AlphaVantageQuote | None:
    """
    Parse Alpha Vantage quote data into AlphaVantageQuote object.
    
    Parameters
    ----------
    symbol : str
        The stock symbol.
    quote_data : dict[str, Any]
        The raw quote data from Alpha Vantage API.
    ts_init : int
        The initialization timestamp.
        
    Returns
    -------
    AlphaVantageQuote | None
        The parsed quote or None if parsing failed.
    """
    try:
        global_quote = quote_data.get("Global Quote", {})
        if not global_quote:
            return None
            
        # Create instrument ID
        instrument_id = InstrumentId(Symbol(symbol), Venue("ALPHA_VANTAGE"))
        
        # Parse values with error handling
        def safe_decimal(value: str, default: Decimal = Decimal("0")) -> Decimal:
            try:
                return Decimal(value) if value and value != "None" else default
            except (ValueError, TypeError):
                return default
        
        def safe_int(value: str, default: int = 0) -> int:
            try:
                return int(value) if value and value != "None" else default
            except (ValueError, TypeError):
                return default
        
        # Parse quote data
        open_price = safe_decimal(global_quote.get("02. open", "0"))
        high = safe_decimal(global_quote.get("03. high", "0"))
        low = safe_decimal(global_quote.get("04. low", "0"))
        price = safe_decimal(global_quote.get("05. price", "0"))
        volume = safe_int(global_quote.get("06. volume", "0"))
        latest_trading_day = global_quote.get("07. latest trading day", "")
        previous_close = safe_decimal(global_quote.get("08. previous close", "0"))
        change = safe_decimal(global_quote.get("09. change", "0"))
        change_percent = global_quote.get("10. change percent", "0%")
        
        # Convert latest trading day to timestamp
        ts_event = ts_init
        if latest_trading_day:
            try:
                dt = datetime.strptime(latest_trading_day, "%Y-%m-%d")
                ts_event = int(dt.timestamp() * 1_000_000_000)
            except ValueError:
                pass  # Use ts_init as fallback
        
        return AlphaVantageQuote(
            instrument_id=instrument_id,
            symbol=symbol,
            open=open_price,
            high=high,
            low=low,
            price=price,
            volume=volume,
            latest_trading_day=latest_trading_day,
            previous_close=previous_close,
            change=change,
            change_percent=change_percent,
            ts_event=ts_event,
            ts_init=ts_init,
        )
        
    except Exception:
        return None


def parse_alpha_vantage_bars(
    symbol: str,
    time_series_data: dict[str, Any],
    adjusted: bool = True,
    ts_init: int = 0,
) -> list[AlphaVantageBar]:
    """
    Parse Alpha Vantage time series data into AlphaVantageBar objects.
    
    Parameters
    ----------
    symbol : str
        The stock symbol.
    time_series_data : dict[str, Any]
        The raw time series data from Alpha Vantage API.
    adjusted : bool, default True
        Whether the data includes adjustment factors.
    ts_init : int, default 0
        The initialization timestamp.
        
    Returns
    -------
    list[AlphaVantageBar]
        The parsed bars.
    """
    bars = []
    
    # Find the time series key
    time_series_key = None
    for key in time_series_data.keys():
        if "Time Series" in key:
            time_series_key = key
            break
    
    if not time_series_key or time_series_key not in time_series_data:
        return bars
    
    time_series = time_series_data[time_series_key]
    instrument_id = InstrumentId(Symbol(symbol), Venue("ALPHA_VANTAGE"))
    
    def safe_decimal(value: str, default: Decimal = Decimal("0")) -> Decimal:
        try:
            return Decimal(value) if value and value != "None" else default
        except (ValueError, TypeError):
            return default
    
    def safe_int(value: str, default: int = 0) -> int:
        try:
            return int(value) if value and value != "None" else default
        except (ValueError, TypeError):
            return default
    
    for timestamp_str, ohlcv_data in time_series.items():
        try:
            # Parse timestamp
            try:
                # Try datetime with time first
                dt = datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M:%S")
            except ValueError:
                try:
                    # Try date only
                    dt = datetime.strptime(timestamp_str, "%Y-%m-%d")
                except ValueError:
                    continue  # Skip invalid timestamps
            
            ts_event = int(dt.timestamp() * 1_000_000_000)
            
            # Parse OHLCV data
            open_price = safe_decimal(ohlcv_data.get("1. open", "0"))
            high = safe_decimal(ohlcv_data.get("2. high", "0"))
            low = safe_decimal(ohlcv_data.get("3. low", "0"))
            close = safe_decimal(ohlcv_data.get("4. close", "0"))
            volume = safe_int(ohlcv_data.get("5. volume", "0"))
            
            # Parse adjusted data if available
            adjusted_close = None
            dividend_amount = None
            split_coefficient = None
            
            if adjusted:
                adjusted_close = safe_decimal(ohlcv_data.get("5. adjusted close"))
                dividend_amount = safe_decimal(ohlcv_data.get("7. dividend amount"))
                split_coefficient = safe_decimal(ohlcv_data.get("8. split coefficient"))
            
            bar = AlphaVantageBar(
                instrument_id=instrument_id,
                open=open_price,
                high=high,
                low=low,
                close=close,
                volume=volume,
                adjusted_close=adjusted_close,
                dividend_amount=dividend_amount,
                split_coefficient=split_coefficient,
                ts_event=ts_event,
                ts_init=ts_init,
            )
            bars.append(bar)
            
        except Exception:
            continue  # Skip bars that can't be parsed
    
    # Sort bars by timestamp (oldest first)
    bars.sort(key=lambda x: x.ts_event)
    return bars


def parse_alpha_vantage_company_data(
    symbol: str,
    overview_data: dict[str, Any],
    ts_init: int,
) -> AlphaVantageCompanyData | None:
    """
    Parse Alpha Vantage company overview data into AlphaVantageCompanyData object.
    
    Parameters
    ----------
    symbol : str
        The stock symbol.
    overview_data : dict[str, Any]
        The raw company overview data from Alpha Vantage API.
    ts_init : int
        The initialization timestamp.
        
    Returns
    -------
    AlphaVantageCompanyData | None
        The parsed company data or None if parsing failed.
    """
    try:
        if not overview_data or "Symbol" not in overview_data:
            return None
            
        instrument_id = InstrumentId(Symbol(symbol), Venue("ALPHA_VANTAGE"))
        
        def safe_decimal(value: str) -> Decimal | None:
            try:
                return Decimal(value) if value and value != "None" else None
            except (ValueError, TypeError):
                return None
        
        def safe_int(value: str) -> int | None:
            try:
                return int(value) if value and value != "None" else None
            except (ValueError, TypeError):
                return None
        
        return AlphaVantageCompanyData(
            instrument_id=instrument_id,
            symbol=symbol,
            name=overview_data.get("Name", ""),
            description=overview_data.get("Description"),
            exchange=overview_data.get("Exchange"),
            currency=overview_data.get("Currency"),
            country=overview_data.get("Country"),
            sector=overview_data.get("Sector"),
            industry=overview_data.get("Industry"),
            market_cap=safe_int(overview_data.get("MarketCapitalization")),
            pe_ratio=safe_decimal(overview_data.get("PERatio")),
            peg_ratio=safe_decimal(overview_data.get("PEGRatio")),
            book_value=safe_decimal(overview_data.get("BookValue")),
            dividend_yield=safe_decimal(overview_data.get("DividendYield")),
            eps=safe_decimal(overview_data.get("EPS")),
            revenue_ttm=safe_int(overview_data.get("RevenueTTM")),
            gross_profit_ttm=safe_int(overview_data.get("GrossProfitTTM")),
            ebitda=safe_int(overview_data.get("EBITDA")),
            ts_event=ts_init,
            ts_init=ts_init,
        )
        
    except Exception:
        return None