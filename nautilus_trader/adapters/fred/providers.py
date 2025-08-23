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

from nautilus_trader.adapters.fred.config import FREDInstrumentProviderConfig
from nautilus_trader.adapters.fred.http import FREDHttpClient
from nautilus_trader.adapters.fred.parsing import create_fred_instrument
from nautilus_trader.adapters.fred.parsing import get_popular_fred_series
from nautilus_trader.adapters.fred.parsing import parse_fred_series_info
from nautilus_trader.adapters.fred.parsing import validate_fred_series_id
from nautilus_trader.common.enums import LogColor
from nautilus_trader.common.providers import InstrumentProvider
from nautilus_trader.core.correctness import PyCondition
from nautilus_trader.model.identifiers import InstrumentId
from nautilus_trader.model.identifiers import Symbol
from nautilus_trader.model.identifiers import Venue


class FREDInstrumentProvider(InstrumentProvider):
    """
    Provides a means of loading `Instrument` objects from FRED economic data series.

    This provider fetches economic series metadata from the FRED API and creates
    corresponding Nautilus Instrument objects that can be used in trading strategies.

    Parameters
    ----------
    config : FREDInstrumentProviderConfig, optional
        The configuration for the provider.

    """

    def __init__(
        self,
        config: FREDInstrumentProviderConfig | None = None,
    ) -> None:
        if config is None:
            config = FREDInstrumentProviderConfig()
            
        super().__init__(config=config)
        
        self._config = config
        self._venue = Venue("FRED")
        self._http_client = FREDHttpClient(
            api_key=config.api_key,
            base_url=config.base_url,
            request_timeout=config.request_timeout,
            rate_limit_delay=config.rate_limit_delay,
            logger=self._log,
        )
        self._series_cache: dict[str, dict[str, Any]] = {}

    async def load_all_async(self, filters: dict | None = None) -> None:
        """
        Load all available instruments for the provider.

        Since FRED contains over 800,000 series, this method loads a curated
        set of popular economic indicators unless specifically configured otherwise.

        Parameters
        ----------
        filters : dict, optional
            Optional filters for the instrument request (not used).

        """
        if self._config.load_all:
            self._log.warning(
                "Loading all FRED instruments is not recommended (800,000+ series). "
                "Consider using specific series_ids or category_ids instead.",
                LogColor.YELLOW,
            )
            # For safety, we'll load popular series instead of everything
            
        # Load popular series if no specific configuration provided
        if not self._config.series_ids and not self._config.category_ids and not self._config.search_terms:
            popular_series = get_popular_fred_series()
            series_ids = list(popular_series.keys())
            self._log.info(f"Loading {len(series_ids)} popular FRED series")
        else:
            series_ids = self._config.series_ids.copy()
            
        # Load series from categories
        if self._config.category_ids:
            for category_id in self._config.category_ids:
                category_series = await self._load_category_series(category_id)
                series_ids.extend(category_series)
                
        # Load series from search terms
        if self._config.search_terms:
            for search_term in self._config.search_terms:
                search_series = await self._search_series(search_term)
                series_ids.extend(search_series)
                
        # Remove duplicates and load instruments
        unique_series_ids = list(set(series_ids))
        self._log.info(f"Loading {len(unique_series_ids)} FRED series as instruments")
        
        instrument_ids = [
            InstrumentId(Symbol(series_id), self._venue) 
            for series_id in unique_series_ids
        ]
        
        await self.load_ids_async(instrument_ids)

    async def load_ids_async(
        self,
        instrument_ids: list[InstrumentId],
        filters: dict | None = None,
    ) -> None:
        """
        Load the instrument definitions for the given instrument IDs into the provider.

        This method fetches series information from FRED for each series ID
        and creates appropriate Nautilus Instrument objects.

        Parameters
        ----------
        instrument_ids : list[InstrumentId]
            The instrument IDs to load.
        filters : dict, optional
            Optional filters for the instrument request (not used).

        """
        PyCondition.not_empty(instrument_ids, "instrument_ids")

        # Validate venue
        for instrument_id in instrument_ids:
            if instrument_id.venue != self._venue:
                raise ValueError(
                    f"Invalid venue {instrument_id.venue}, expected {self._venue}"
                )

        await self._http_client.connect()
        
        try:
            instruments = []
            
            for instrument_id in instrument_ids:
                series_id = instrument_id.symbol.value
                
                # Validate series ID format
                if not validate_fred_series_id(series_id):
                    self._log.warning(f"Invalid FRED series ID format: {series_id}")
                    continue
                    
                try:
                    # Get series info from FRED API
                    series_data = await self._http_client.get_series_info(series_id)
                    
                    if not series_data:
                        self._log.warning(f"No data found for FRED series: {series_id}")
                        continue
                        
                    # Parse and cache series info
                    series_info = parse_fred_series_info(series_data)
                    if self._config.cache_instruments:
                        self._series_cache[series_id] = series_info
                        
                    # Create instrument
                    instrument = create_fred_instrument(series_info)
                    instruments.append(instrument)
                    
                    self._log.info(
                        f"Loaded FRED series {series_id}: {series_info.get('title', 'Unknown')}"
                    )
                    
                except Exception as e:
                    self._log.error(f"Failed to load FRED series {series_id}: {e}")
                    continue
            
            # Add instruments to provider
            self.add_bulk(instruments)
            
            self._log.info(
                f"Successfully loaded {len(instruments)} FRED instruments",
                LogColor.GREEN,
            )
            
        finally:
            await self._http_client.disconnect()

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
        await self.load_ids_async([instrument_id], filters)

    def get_series_info(self, series_id: str) -> dict[str, Any] | None:
        """
        Get cached series information for a FRED series.
        
        Parameters
        ----------
        series_id : str
            The FRED series ID.
            
        Returns
        -------
        dict[str, Any] | None
            The cached series information or None if not found.
            
        """
        return self._series_cache.get(series_id)

    async def _load_category_series(self, category_id: int) -> list[str]:
        """
        Load series IDs from a FRED category.
        
        Parameters
        ----------
        category_id : int
            The FRED category ID.
            
        Returns
        -------
        list[str]
            The list of series IDs in the category.
            
        """
        try:
            response = await self._http_client.get_category_series(
                category_id=category_id,
                limit=self._config.max_search_results,
            )
            
            series_list = response.get("seriess", [])
            series_ids = [series["id"] for series in series_list]
            
            self._log.info(f"Found {len(series_ids)} series in category {category_id}")
            return series_ids
            
        except Exception as e:
            self._log.error(f"Failed to load category {category_id}: {e}")
            return []

    async def _search_series(self, search_term: str) -> list[str]:
        """
        Search for series IDs matching a search term.
        
        Parameters
        ----------
        search_term : str
            The search term.
            
        Returns
        -------
        list[str]
            The list of matching series IDs.
            
        """
        try:
            response = await self._http_client.search_series(
                search_text=search_term,
                limit=self._config.max_search_results,
            )
            
            series_list = response.get("seriess", [])
            series_ids = [series["id"] for series in series_list]
            
            self._log.info(f"Found {len(series_ids)} series matching '{search_term}'")
            return series_ids
            
        except Exception as e:
            self._log.error(f"Failed to search for '{search_term}': {e}")
            return []