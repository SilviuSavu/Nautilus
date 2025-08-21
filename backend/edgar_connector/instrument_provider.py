"""
EDGAR Instrument Provider
========================

Instrument provider for SEC entities, managing CIK/ticker mappings and entity data.
"""

import asyncio
import logging
from datetime import datetime
from typing import Dict, List, Optional, Set

from nautilus_trader.common.providers import InstrumentProvider
from nautilus_trader.model.identifiers import InstrumentId
from nautilus_trader.model.instruments import Instrument

from edgar_connector.api_client import EDGARAPIClient
from edgar_connector.config import EDGARConfig, EDGARInstrumentConfig
from edgar_connector.data_types import SECEntity
from edgar_connector.utils import DataCache, normalize_cik, normalize_ticker

logger = logging.getLogger(__name__)


class EDGARInstrumentProvider(InstrumentProvider):
    """
    Provides SEC entity data as NautilusTrader instruments.
    
    This class manages the mapping between stock tickers/CIKs and SEC entities,
    allowing NautilusTrader strategies to treat SEC entities as tradeable instruments.
    """
    
    def __init__(
        self,
        api_client: EDGARAPIClient,
        config: EDGARInstrumentConfig,
    ):
        super().__init__()
        self.api_client = api_client
        self.config = config
        self.cache = DataCache(config.entities_cache_path)
        
        # Internal state
        self._entities: Dict[str, SECEntity] = {}  # CIK -> SECEntity
        self._ticker_to_cik: Dict[str, str] = {}  # Ticker -> CIK
        self._cik_to_ticker: Dict[str, str] = {}  # CIK -> Ticker
        self._is_loaded = False
        self._last_update = None
        
        logger.info(f"EDGAR Instrument Provider initialized")
    
    async def load_all_async(self) -> None:
        """Load all SEC entity data."""
        if self.config.update_entities_on_startup:
            await self.update_entities()
        else:
            await self.load_cached_entities()
        
        self._is_loaded = True
        logger.info(f"Loaded {len(self._entities)} SEC entities")
    
    async def load_ids_async(
        self,
        instrument_ids: List[InstrumentId],
        filters: Optional[Dict] = None,
    ) -> None:
        """Load specific instrument IDs."""
        # For EDGAR, we'll load all entities since IDs are CIKs
        await self.load_all_async()
    
    async def load_async(self, instrument_id: InstrumentId) -> Optional[Instrument]:
        """Load a specific instrument (SEC entity)."""
        # Extract CIK from instrument ID
        cik = instrument_id.symbol.value
        
        if not self._is_loaded:
            await self.load_all_async()
        
        if cik in self._entities:
            # Create a synthetic "instrument" for the SEC entity
            # Note: This is a placeholder - in practice you might create
            # custom instrument types for SEC entities
            return None  # SEC entities aren't tradeable instruments
        
        return None
    
    def get_all(self) -> Dict[InstrumentId, Instrument]:
        """Get all loaded instruments."""
        # SEC entities aren't traditional instruments, return empty dict
        return {}
    
    def get(self, instrument_id: InstrumentId) -> Optional[Instrument]:
        """Get instrument by ID."""
        return None  # SEC entities aren't traditional instruments
    
    # EDGAR-specific methods
    
    async def update_entities(self) -> None:
        """Update SEC entity data from API."""
        logger.info("Updating SEC entity data from API")
        
        try:
            # Get company ticker mappings from SEC
            ticker_data = await self.api_client.get_company_tickers()
            
            entities = {}
            ticker_to_cik = {}
            cik_to_ticker = {}
            
            for cik_key, company_info in ticker_data.items():
                if not isinstance(company_info, dict):
                    continue
                
                cik = normalize_cik(company_info.get("cik_str", cik_key))
                ticker = company_info.get("ticker", "").upper()
                company_name = company_info.get("title", "")
                exchange = company_info.get("exchange", "")
                
                entity = SECEntity(
                    cik=cik,
                    name=company_name,
                    ticker=ticker if ticker else None,
                    exchange=exchange if exchange else None
                )
                
                entities[cik] = entity
                
                if ticker:
                    ticker_to_cik[ticker] = cik
                    cik_to_ticker[cik] = ticker
            
            # Update internal state
            self._entities = entities
            self._ticker_to_cik = ticker_to_cik
            self._cik_to_ticker = cik_to_ticker
            self._last_update = datetime.now()
            
            # Cache the data
            cache_data = {
                "entities": {cik: entity.dict() for cik, entity in entities.items()},
                "ticker_to_cik": ticker_to_cik,
                "cik_to_ticker": cik_to_ticker,
                "last_update": self._last_update.isoformat()
            }
            
            self.cache.set("entities_data", cache_data)
            
            logger.info(f"Updated {len(entities)} SEC entities")
            
        except Exception as e:
            logger.error(f"Error updating SEC entities: {e}")
            raise
    
    async def load_cached_entities(self) -> None:
        """Load SEC entity data from cache."""
        logger.info("Loading SEC entity data from cache")
        
        cache_data = self.cache.get(
            "entities_data",
            ttl_seconds=self.config.ticker_cache_ttl
        )
        
        if cache_data:
            try:
                # Restore entities from cache
                self._entities = {
                    cik: SECEntity(**entity_data)
                    for cik, entity_data in cache_data.get("entities", {}).items()
                }
                
                self._ticker_to_cik = cache_data.get("ticker_to_cik", {})
                self._cik_to_ticker = cache_data.get("cik_to_ticker", {})
                
                last_update_str = cache_data.get("last_update")
                if last_update_str:
                    self._last_update = datetime.fromisoformat(last_update_str)
                
                logger.info(f"Loaded {len(self._entities)} entities from cache")
                return
                
            except Exception as e:
                logger.warning(f"Error loading cached entities: {e}")
        
        # Fall back to API update if cache is invalid
        await self.update_entities()
    
    def get_entity_by_cik(self, cik: str) -> Optional[SECEntity]:
        """Get SEC entity by CIK."""
        normalized_cik = normalize_cik(cik)
        return self._entities.get(normalized_cik)
    
    def get_entity_by_ticker(self, ticker: str) -> Optional[SECEntity]:
        """Get SEC entity by ticker symbol."""
        normalized_ticker = normalize_ticker(ticker)
        cik = self._ticker_to_cik.get(normalized_ticker)
        if cik:
            return self._entities.get(cik)
        return None
    
    def resolve_ticker_to_cik(self, ticker: str) -> Optional[str]:
        """Resolve ticker symbol to CIK."""
        normalized_ticker = normalize_ticker(ticker)
        return self._ticker_to_cik.get(normalized_ticker)
    
    def resolve_cik_to_ticker(self, cik: str) -> Optional[str]:
        """Resolve CIK to ticker symbol."""
        normalized_cik = normalize_cik(cik)
        return self._cik_to_ticker.get(normalized_cik)
    
    def search_entities(self, query: str, limit: int = 10) -> List[SECEntity]:
        """
        Search SEC entities by name or ticker.
        
        Args:
            query: Search query (company name or ticker)
            limit: Maximum results to return
            
        Returns:
            List of matching SECEntity objects
        """
        query_lower = query.lower()
        results = []
        
        for entity in self._entities.values():
            # Search in company name
            if query_lower in entity.name.lower():
                results.append(entity)
            # Search in ticker
            elif entity.ticker and query_lower in entity.ticker.lower():
                results.append(entity)
            
            if len(results) >= limit:
                break
        
        return results
    
    def get_entities_by_exchange(self, exchange: str) -> List[SECEntity]:
        """Get entities listed on specific exchange."""
        return [
            entity for entity in self._entities.values()
            if entity.exchange and entity.exchange.upper() == exchange.upper()
        ]
    
    def get_all_tickers(self) -> List[str]:
        """Get list of all available ticker symbols."""
        return list(self._ticker_to_cik.keys())
    
    def get_all_ciks(self) -> List[str]:
        """Get list of all available CIKs."""
        return list(self._entities.keys())
    
    def get_entity_count(self) -> int:
        """Get total number of loaded entities."""
        return len(self._entities)
    
    def is_valid_ticker(self, ticker: str) -> bool:
        """Check if ticker is valid (exists in SEC data)."""
        normalized_ticker = normalize_ticker(ticker)
        return normalized_ticker in self._ticker_to_cik
    
    def is_valid_cik(self, cik: str) -> bool:
        """Check if CIK is valid (exists in SEC data)."""
        normalized_cik = normalize_cik(cik)
        return normalized_cik in self._entities
    
    def get_entities_with_recent_filings(
        self,
        days_back: int = 30
    ) -> List[SECEntity]:
        """
        Get entities that have had recent filings.
        
        Note: This is a placeholder method. In practice, you'd need to 
        track filing dates and filter accordingly.
        
        Args:
            days_back: Number of days to look back
            
        Returns:
            List of entities with recent filings
        """
        # Placeholder implementation
        # In practice, you'd track filing dates and filter
        return list(self._entities.values())[:50]  # Return first 50 as example
    
    async def refresh_if_stale(self, max_age_hours: int = 24) -> None:
        """Refresh entity data if it's stale."""
        if (
            self._last_update is None or
            (datetime.now() - self._last_update).total_seconds() > max_age_hours * 3600
        ):
            logger.info("Entity data is stale, refreshing...")
            await self.update_entities()
    
    def get_statistics(self) -> Dict[str, int]:
        """Get statistics about loaded entities."""
        return {
            "total_entities": len(self._entities),
            "entities_with_tickers": len(self._ticker_to_cik),
            "unique_exchanges": len(set(
                entity.exchange for entity in self._entities.values()
                if entity.exchange
            ))
        }