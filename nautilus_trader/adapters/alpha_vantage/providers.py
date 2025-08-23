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

from nautilus_trader.adapters.alpha_vantage.config import AlphaVantageInstrumentProviderConfig
from nautilus_trader.adapters.alpha_vantage.http import AlphaVantageHttpClient
from nautilus_trader.common.component import Logger
from nautilus_trader.common.providers import InstrumentProvider
from nautilus_trader.model.identifiers import InstrumentId
from nautilus_trader.model.identifiers import Symbol
from nautilus_trader.model.identifiers import Venue
from nautilus_trader.model.instruments import Equity
from nautilus_trader.model.objects import Price
from nautilus_trader.model.objects import Quantity


class AlphaVantageInstrumentProvider(InstrumentProvider):
    """
    Provides instruments from Alpha Vantage.
    
    Parameters
    ----------
    config : AlphaVantageInstrumentProviderConfig
        The configuration for the provider.
    logger : Logger, optional
        The logger for the provider.
    """

    def __init__(
        self,
        config: AlphaVantageInstrumentProviderConfig,
        logger: Logger | None = None,
    ) -> None:
        super().__init__(
            venue=Venue("ALPHA_VANTAGE"),
            logger=logger,
        )
        
        self._config = config
        
        # HTTP client for Alpha Vantage API
        self._http_client = AlphaVantageHttpClient(
            api_key=config.api_key,
            base_url=config.base_url,
            request_timeout=config.request_timeout,
            rate_limit_delay=config.rate_limit_delay,
            logger=self._log,
        )
        
        # Cache for instrument data
        self._instrument_cache: dict[str, Equity] = {}

    async def load_ids_async(
        self,
        instrument_ids: list[InstrumentId],
        filters: dict[str, Any] | None = None,
    ) -> None:
        """
        Load instruments for the given instrument IDs.
        
        Parameters
        ----------
        instrument_ids : list[InstrumentId]
            The instrument IDs to load.
        filters : dict[str, Any] | None, optional
            Not used for Alpha Vantage.
        """
        await self._http_client.connect()
        
        try:
            for instrument_id in instrument_ids:
                if instrument_id.venue != self._venue:
                    continue
                    
                symbol = instrument_id.symbol.value
                
                # Check cache first
                if symbol in self._instrument_cache:
                    self.add(self._instrument_cache[symbol])
                    continue
                
                # Fetch instrument data from Alpha Vantage
                instrument = await self._load_instrument(symbol)
                if instrument:
                    self._instrument_cache[symbol] = instrument
                    self.add(instrument)
                    
        finally:
            await self._http_client.disconnect()

    async def load_all_async(self, filters: dict[str, Any] | None = None) -> None:
        """
        Load all configured instruments.
        
        Parameters
        ----------
        filters : dict[str, Any] | None, optional
            Not used for Alpha Vantage.
        """
        if not self._config.symbols and not self._config.search_terms:
            self._log.warning("No symbols or search terms configured")
            return
        
        await self._http_client.connect()
        
        try:
            # Load specific symbols if configured
            if self._config.symbols:
                for symbol in self._config.symbols:
                    if symbol in self._instrument_cache:
                        self.add(self._instrument_cache[symbol])
                        continue
                        
                    instrument = await self._load_instrument(symbol)
                    if instrument:
                        self._instrument_cache[symbol] = instrument
                        self.add(instrument)
            
            # Search for instruments if search terms configured
            if self._config.search_terms:
                await self._search_and_load_instruments()
                
        finally:
            await self._http_client.disconnect()

    async def _load_instrument(self, symbol: str) -> Equity | None:
        """
        Load instrument data for a specific symbol.
        
        Parameters
        ----------
        symbol : str
            The symbol to load.
            
        Returns
        -------
        Equity | None
            The loaded instrument or None if loading failed.
        """
        try:
            # Get company overview data
            company_data = await self._http_client.get_company_overview(symbol)
            if not company_data:
                # Create basic instrument without company data
                return self._create_basic_instrument(symbol)
            
            # Parse company data
            instrument_id = InstrumentId(Symbol(symbol), self._venue)
            
            # Get basic info with defaults
            name = company_data.get("Name", symbol)
            exchange = company_data.get("Exchange", "NYSE")  # Default to NYSE
            currency = company_data.get("Currency", "USD")   # Default to USD
            
            # Create equity instrument
            instrument = Equity(
                instrument_id=instrument_id,
                raw_symbol=Symbol(symbol),
                currency=currency,
                price_precision=2,  # Default precision for stocks
                price_increment=Price(0.01, precision=2),
                lot_size=Quantity(1, precision=0),
                isin=None,
                margin_init=0.5,  # Default 50% initial margin
                margin_maint=0.25,  # Default 25% maintenance margin
                maker_fee=0.0005,  # Default 0.05% maker fee
                taker_fee=0.0005,  # Default 0.05% taker fee
                ts_event=self._clock.timestamp_ns(),
                ts_init=self._clock.timestamp_ns(),
                info={
                    "name": name,
                    "exchange": exchange,
                    "sector": company_data.get("Sector"),
                    "industry": company_data.get("Industry"),
                    "market_cap": company_data.get("MarketCapitalization"),
                    "description": company_data.get("Description"),
                },
            )
            
            self._log.info(f"Loaded Alpha Vantage instrument: {symbol} ({name})")
            return instrument
            
        except Exception as e:
            self._log.error(f"Failed to load instrument {symbol}: {e}")
            return self._create_basic_instrument(symbol)

    def _create_basic_instrument(self, symbol: str) -> Equity:
        """
        Create a basic instrument with minimal information.
        
        Parameters
        ----------
        symbol : str
            The symbol.
            
        Returns
        -------
        Equity
            The basic instrument.
        """
        instrument_id = InstrumentId(Symbol(symbol), self._venue)
        
        return Equity(
            instrument_id=instrument_id,
            raw_symbol=Symbol(symbol),
            currency="USD",  # Default to USD
            price_precision=2,
            price_increment=Price(0.01, precision=2),
            lot_size=Quantity(1, precision=0),
            isin=None,
            margin_init=0.5,
            margin_maint=0.25,
            maker_fee=0.0005,
            taker_fee=0.0005,
            ts_event=self._clock.timestamp_ns(),
            ts_init=self._clock.timestamp_ns(),
            info={"name": symbol},
        )

    async def _search_and_load_instruments(self) -> None:
        """Search for and load instruments based on search terms."""
        if not self._config.search_terms:
            return
            
        for search_term in self._config.search_terms:
            try:
                search_results = await self._http_client.search_symbols(search_term)
                if not search_results or "bestMatches" not in search_results:
                    continue
                
                matches = search_results["bestMatches"]
                loaded_count = 0
                
                for match in matches:
                    if loaded_count >= self._config.max_search_results:
                        break
                        
                    symbol = match.get("1. symbol")
                    if not symbol or symbol in self._instrument_cache:
                        continue
                    
                    instrument = await self._load_instrument(symbol)
                    if instrument:
                        self._instrument_cache[symbol] = instrument
                        self.add(instrument)
                        loaded_count += 1
                
                self._log.info(f"Loaded {loaded_count} instruments from search: '{search_term}'")
                
            except Exception as e:
                self._log.error(f"Failed to search for instruments with term '{search_term}': {e}")

    async def find_by_symbol(self, symbol: str) -> InstrumentId | None:
        """
        Find instrument ID by symbol.
        
        Parameters
        ----------
        symbol : str
            The symbol to search for.
            
        Returns
        -------
        InstrumentId | None
            The instrument ID or None if not found.
        """
        # Check if already loaded
        instrument_id = InstrumentId(Symbol(symbol), self._venue)
        if self.find(instrument_id):
            return instrument_id
        
        # Try to load the instrument
        await self._http_client.connect()
        try:
            instrument = await self._load_instrument(symbol)
            if instrument:
                self._instrument_cache[symbol] = instrument
                self.add(instrument)
                return instrument_id
        finally:
            await self._http_client.disconnect()
        
        return None