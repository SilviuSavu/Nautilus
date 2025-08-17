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
from typing import Any

import yfinance as yf

from nautilus_trader.common.enums import LogColor
from nautilus_trader.common.providers import InstrumentProvider
from nautilus_trader.config import InstrumentProviderConfig
from nautilus_trader.core.correctness import PyCondition
from nautilus_trader.model.currencies import USD
from nautilus_trader.model.enums import AssetClass
from nautilus_trader.model.enums import InstrumentClass
from nautilus_trader.model.identifiers import InstrumentId
from nautilus_trader.model.identifiers import Symbol
from nautilus_trader.model.identifiers import Venue
from nautilus_trader.model.instruments import CurrencyPair
from nautilus_trader.model.instruments import Equity
from nautilus_trader.model.instruments import Instrument
from nautilus_trader.model.objects import Money
from nautilus_trader.model.objects import Price
from nautilus_trader.model.objects import Quantity


class YFinanceInstrumentProvider(InstrumentProvider):
    """
    Provides a means of loading `Instrument` objects from Yahoo Finance.

    Parameters
    ----------
    config : InstrumentProviderConfig, optional
        The configuration for the provider.

    """

    def __init__(
        self,
        config: InstrumentProviderConfig | None = None,
    ) -> None:
        super().__init__(config=config)
        self._venue = Venue("YAHOO")

    async def load_all_async(self, filters: dict | None = None) -> None:
        """
        Load all available instruments for the provider.

        Since Yahoo Finance contains millions of symbols, this method is not supported.
        Use `load_ids_async` or `load_async` with specific symbol lists instead.

        Parameters
        ----------
        filters : dict, optional
            Not used for this implementation.

        Raises
        ------
        RuntimeError
            Always, as loading all instruments is not supported.

        """
        raise RuntimeError(
            "Loading all instrument definitions is not supported for Yahoo Finance, "
            "as this would potentially include millions of symbols. "
            "Use load_ids_async() or load_async() with specific symbols instead.",
        )

    async def load_ids_async(
        self,
        instrument_ids: list[InstrumentId],
        filters: dict | None = None,
    ) -> None:
        """
        Load the instrument definitions for the given instrument IDs into the provider.

        This method fetches instrument information from Yahoo Finance for each symbol
        and creates appropriate Nautilus Instrument objects.

        Parameters
        ----------
        instrument_ids : list[InstrumentId]
            The instrument IDs to load.
        filters : dict, optional
            Optional filters for the instrument request (not used).

        """
        PyCondition.not_empty(instrument_ids, "instrument_ids")

        self._log.info(
            f"Loading {len(instrument_ids)} instrument(s) from Yahoo Finance...",
            LogColor.BLUE,
        )

        # Extract symbols from instrument IDs
        symbols = [instrument_id.symbol.value for instrument_id in instrument_ids]

        # Load instruments in parallel
        tasks = [self._load_single_instrument(symbol) for symbol in symbols]
        instruments = await asyncio.gather(*tasks, return_exceptions=True)

        # Process results
        loaded_count = 0
        for i, result in enumerate(instruments):
            if isinstance(result, Exception):
                self._log.error(f"Failed to load {symbols[i]}: {result}")
            elif result is not None:
                self.add(instrument=result)
                loaded_count += 1
                self._log.debug(f"Added instrument {result.id}")

        self._log.info(
            f"Loaded {loaded_count}/{len(instrument_ids)} instrument(s) from Yahoo Finance",
            LogColor.BLUE,
        )

    async def load_async(
        self,
        instrument_id: InstrumentId,
        filters: dict | None = None,
    ) -> None:
        """
        Load the instrument definition for the given instrument ID into the provider.

        Parameters
        ----------
        instrument_id : InstrumentId
            The instrument ID to load.
        filters : dict, optional
            Optional filters for the instrument request (not used).

        """
        await self.load_ids_async([instrument_id], filters=filters)

    async def _load_single_instrument(self, symbol: str) -> Instrument | None:
        """
        Load a single instrument from Yahoo Finance.

        Parameters
        ----------
        symbol : str
            The Yahoo Finance symbol to load.

        Returns
        -------
        Instrument or None
            The loaded instrument, or None if loading failed.

        """
        try:
            # Run yfinance call in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            ticker = await loop.run_in_executor(None, yf.Ticker, symbol)
            info = await loop.run_in_executor(None, lambda: ticker.info)

            if not info or "symbol" not in info:
                self._log.warning(f"No information available for symbol {symbol}")
                return None

            # Create instrument based on type
            instrument_id = InstrumentId(Symbol(symbol), self._venue)
            
            # Determine instrument type from Yahoo Finance data
            quote_type = info.get("quoteType", "").upper()
            
            if quote_type in ("EQUITY", "ETF"):
                return self._create_equity(instrument_id, info)
            elif quote_type == "CURRENCY":
                return self._create_currency_pair(instrument_id, info)
            else:
                # Default to equity for unknown types
                self._log.warning(f"Unknown quote type '{quote_type}' for {symbol}, treating as equity")
                return self._create_equity(instrument_id, info)

        except Exception as e:
            self._log.error(f"Error loading instrument {symbol}: {e}")
            return None

    def _create_equity(self, instrument_id: InstrumentId, info: dict[str, Any]) -> Equity:
        """Create an Equity instrument from Yahoo Finance info."""
        
        # Extract basic information
        currency_str = info.get("currency", "USD")
        price_precision = self._get_price_precision(info)
        
        # Get current price for min notional calculation
        current_price = info.get("currentPrice") or info.get("regularMarketPrice") or 1.0
        
        return Equity(
            instrument_id=instrument_id,
            raw_symbol=Symbol(instrument_id.symbol.value),
            asset_class=AssetClass.EQUITY,
            instrument_class=InstrumentClass.SPOT,
            quote_currency=currency_str,
            price_precision=price_precision,
            price_increment=Price(10 ** -price_precision, price_precision),
            size_precision=0,  # Whole shares
            size_increment=Quantity.from_int(1),
            multiplier=Quantity.from_int(1),
            lot_size=Quantity.from_int(1),
            margin_init=Money(0, USD),  # No margin requirements for cash equity
            margin_maint=Money(0, USD),
            maker_fee=Money(0, USD),  # Fees handled by broker
            taker_fee=Money(0, USD),
            financing={},
            timestamp_ns=0,  # Will be set by the engine
            info=info,
        )

    def _create_currency_pair(self, instrument_id: InstrumentId, info: dict[str, Any]) -> CurrencyPair:
        """Create a CurrencyPair instrument from Yahoo Finance info."""
        
        # Parse currency pair symbol (e.g., "EURUSD=X" -> EUR/USD)
        symbol_str = instrument_id.symbol.value
        if "=" in symbol_str:
            symbol_str = symbol_str.split("=")[0]
        
        if len(symbol_str) >= 6:
            base_currency = symbol_str[:3]
            quote_currency = symbol_str[3:6]
        else:
            # Fallback
            base_currency = "USD"
            quote_currency = "USD"
            
        price_precision = self._get_price_precision(info)
        
        return CurrencyPair(
            instrument_id=instrument_id,
            raw_symbol=Symbol(instrument_id.symbol.value),
            asset_class=AssetClass.FX,
            instrument_class=InstrumentClass.SPOT,
            base_currency=base_currency,
            quote_currency=quote_currency,
            price_precision=price_precision,
            price_increment=Price(10 ** -price_precision, price_precision),
            size_precision=2,  # Standard FX precision
            size_increment=Quantity.from_str("0.01"),
            lot_size=Quantity.from_int(1000),  # Mini lot
            max_quantity=Quantity.from_int(1_000_000),
            min_quantity=Quantity.from_str("0.01"),
            max_price=None,
            min_price=None,
            margin_init=Money(0, USD),
            margin_maint=Money(0, USD),
            maker_fee=Money(0, USD),
            taker_fee=Money(0, USD),
            financing={},
            timestamp_ns=0,
            info=info,
        )

    def _get_price_precision(self, info: dict[str, Any]) -> int:
        """
        Determine appropriate price precision from Yahoo Finance info.
        
        Parameters
        ----------
        info : dict
            The Yahoo Finance instrument info.
            
        Returns
        -------
        int
            The price precision (number of decimal places).
            
        """
        # Try to get precision from various price fields
        current_price = info.get("currentPrice") or info.get("regularMarketPrice")
        if current_price:
            price_str = str(current_price)
            if "." in price_str:
                decimal_places = len(price_str.split(".")[1])
                return min(decimal_places, 8)  # Cap at 8 decimal places
        
        # Default precision based on quote type
        quote_type = info.get("quoteType", "").upper()
        if quote_type == "CURRENCY":
            return 5  # Standard for FX
        else:
            return 2  # Standard for equities