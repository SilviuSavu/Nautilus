"""
YFinance Service with Rate Limiting and Error Handling
====================================================

Robust YFinance implementation with:
- Rate limiting to avoid 429 errors
- Exponential backoff retry logic
- Proper error handling
- Caching to reduce API calls
- Bulk download support

Based on 2025 best practices for yfinance reliability.
"""

import yfinance as yf
import asyncio
import time
import logging
from typing import Dict, List, Any, Optional, Union
from datetime import datetime, timedelta
from cachetools import TTLCache
import pandas as pd
from pydantic import BaseModel
import random

logger = logging.getLogger(__name__)

class YFinanceConfig:
    """Configuration for YFinance service."""
    
    # Rate limiting
    MIN_REQUEST_INTERVAL = 1.0  # Minimum 1 second between requests
    MAX_RETRIES = 3
    BASE_DELAY = 2.0  # Base delay for exponential backoff
    MAX_DELAY = 30.0  # Maximum delay
    
    # Cache settings
    CACHE_TTL_SECONDS = 300  # 5 minutes cache
    MAX_CACHE_SIZE = 1000
    
    # Bulk request settings  
    BULK_CHUNK_SIZE = 10  # Process 10 symbols at once max
    BULK_DELAY = 2.0  # Delay between bulk requests

class YFinanceQuote(BaseModel):
    """YFinance quote data model."""
    symbol: str
    price: Optional[float] = None
    currency: Optional[str] = None
    market_cap: Optional[float] = None
    pe_ratio: Optional[float] = None
    dividend_yield: Optional[float] = None
    volume: Optional[int] = None
    timestamp: datetime
    source: str = "yfinance"

class YFinanceHistorical(BaseModel):
    """YFinance historical data model."""
    symbol: str
    data: List[Dict[str, Any]]
    period: str
    interval: str
    timestamp: datetime
    source: str = "yfinance"

class YFinanceService:
    """YFinance service with rate limiting and reliability improvements."""
    
    def __init__(self):
        self.config = YFinanceConfig()
        self.cache = TTLCache(
            maxsize=self.config.MAX_CACHE_SIZE,
            ttl=self.config.CACHE_TTL_SECONDS
        )
        self.last_request_time = 0.0
        self.request_count = 0
        
    async def _rate_limit_delay(self):
        """Ensure minimum delay between requests."""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        
        if time_since_last < self.config.MIN_REQUEST_INTERVAL:
            delay = self.config.MIN_REQUEST_INTERVAL - time_since_last
            # Add small random jitter to avoid synchronized requests
            delay += random.uniform(0.1, 0.5)
            await asyncio.sleep(delay)
            
        self.last_request_time = time.time()
        self.request_count += 1
        
    async def _exponential_backoff_retry(self, func, *args, **kwargs):
        """Execute function with exponential backoff retry logic."""
        last_exception = None
        
        for attempt in range(self.config.MAX_RETRIES):
            try:
                await self._rate_limit_delay()
                
                # Execute the function in a thread pool to avoid blocking
                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(None, func, *args, **kwargs)
                
                return result
                
            except Exception as e:
                last_exception = e
                error_str = str(e).lower()
                
                # Check for rate limiting errors
                if "429" in error_str or "too many requests" in error_str:
                    delay = min(
                        self.config.BASE_DELAY * (2 ** attempt) + random.uniform(0, 1),
                        self.config.MAX_DELAY
                    )
                    logger.warning(
                        f"Rate limit hit on attempt {attempt + 1}, "
                        f"waiting {delay:.1f}s before retry"
                    )
                    await asyncio.sleep(delay)
                    continue
                
                # For other errors, shorter delay
                if attempt < self.config.MAX_RETRIES - 1:
                    delay = min(self.config.BASE_DELAY + random.uniform(0, 1), 5.0)
                    logger.warning(f"Request failed on attempt {attempt + 1}: {e}")
                    await asyncio.sleep(delay)
                
        # All retries exhausted
        logger.error(f"All {self.config.MAX_RETRIES} attempts failed")
        raise last_exception
    
    def _get_cache_key(self, symbol: str, data_type: str, **kwargs) -> str:
        """Generate cache key for requests."""
        key_parts = [symbol.upper(), data_type]
        for k, v in sorted(kwargs.items()):
            key_parts.append(f"{k}={v}")
        return ":".join(key_parts)
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check on YFinance service."""
        try:
            # Test with a reliable ticker
            test_symbol = "AAPL"
            cache_key = self._get_cache_key(test_symbol, "health_check")
            
            if cache_key in self.cache:
                return self.cache[cache_key]
            
            def _test_ticker():
                ticker = yf.Ticker(test_symbol)
                info = ticker.info
                return bool(info.get('symbol') or info.get('shortName'))
            
            is_working = await self._exponential_backoff_retry(_test_ticker)
            
            result = {
                "status": "operational" if is_working else "error",
                "service": "yfinance",
                "version": yf.__version__,
                "timestamp": datetime.now().isoformat(),
                "test_symbol": test_symbol,
                "request_count": self.request_count,
                "cache_size": len(self.cache)
            }
            
            # Cache successful health checks
            if is_working:
                self.cache[cache_key] = result
                
            return result
            
        except Exception as e:
            logger.error(f"YFinance health check failed: {e}")
            return {
                "status": "error",
                "service": "yfinance", 
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    async def get_quote(self, symbol: str) -> YFinanceQuote:
        """Get real-time quote for a symbol."""
        cache_key = self._get_cache_key(symbol, "quote")
        
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        def _get_ticker_info():
            ticker = yf.Ticker(symbol.upper())
            info = ticker.info
            hist = ticker.history(period="1d", interval="1m").tail(1)
            
            # Extract price from history if available
            current_price = None
            volume = None
            if not hist.empty:
                current_price = float(hist['Close'].iloc[0])
                volume = int(hist['Volume'].iloc[0]) if not pd.isna(hist['Volume'].iloc[0]) else None
            
            return {
                'symbol': info.get('symbol', symbol.upper()),
                'price': current_price or info.get('currentPrice') or info.get('regularMarketPrice'),
                'currency': info.get('currency'),
                'market_cap': info.get('marketCap'),
                'pe_ratio': info.get('forwardPE') or info.get('trailingPE'),
                'dividend_yield': info.get('dividendYield'),
                'volume': volume or info.get('volume')
            }
        
        try:
            data = await self._exponential_backoff_retry(_get_ticker_info)
            
            quote = YFinanceQuote(
                symbol=symbol.upper(),
                price=data.get('price'),
                currency=data.get('currency', 'USD'),
                market_cap=data.get('market_cap'),
                pe_ratio=data.get('pe_ratio'),
                dividend_yield=data.get('dividend_yield'),
                volume=data.get('volume'),
                timestamp=datetime.now()
            )
            
            self.cache[cache_key] = quote
            return quote
            
        except Exception as e:
            logger.error(f"Failed to get quote for {symbol}: {e}")
            raise
    
    async def get_historical_data(
        self, 
        symbol: str, 
        period: str = "1mo", 
        interval: str = "1d"
    ) -> YFinanceHistorical:
        """Get historical data for a symbol."""
        cache_key = self._get_cache_key(symbol, "historical", period=period, interval=interval)
        
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        def _get_history():
            ticker = yf.Ticker(symbol.upper())
            hist = ticker.history(period=period, interval=interval)
            
            if hist.empty:
                return []
            
            # Convert to list of dictionaries
            data = []
            for idx, row in hist.iterrows():
                data.append({
                    'timestamp': idx.isoformat(),
                    'open': float(row['Open']) if not pd.isna(row['Open']) else None,
                    'high': float(row['High']) if not pd.isna(row['High']) else None, 
                    'low': float(row['Low']) if not pd.isna(row['Low']) else None,
                    'close': float(row['Close']) if not pd.isna(row['Close']) else None,
                    'volume': int(row['Volume']) if not pd.isna(row['Volume']) else None
                })
            
            return data
        
        try:
            data = await self._exponential_backoff_retry(_get_history)
            
            historical = YFinanceHistorical(
                symbol=symbol.upper(),
                data=data,
                period=period,
                interval=interval,
                timestamp=datetime.now()
            )
            
            self.cache[cache_key] = historical
            return historical
            
        except Exception as e:
            logger.error(f"Failed to get historical data for {symbol}: {e}")
            raise
    
    async def bulk_quotes(self, symbols: List[str]) -> Dict[str, YFinanceQuote]:
        """Get quotes for multiple symbols efficiently."""
        results = {}
        
        # Process in chunks to avoid overwhelming the API
        for i in range(0, len(symbols), self.config.BULK_CHUNK_SIZE):
            chunk = symbols[i:i + self.config.BULK_CHUNK_SIZE]
            
            # Add delay between chunks
            if i > 0:
                await asyncio.sleep(self.config.BULK_DELAY)
            
            try:
                # Use yfinance bulk download for efficiency
                def _bulk_download():
                    symbols_str = " ".join(s.upper() for s in chunk)
                    data = yf.download(symbols_str, period="1d", interval="1m", group_by='ticker')
                    return data
                
                bulk_data = await self._exponential_backoff_retry(_bulk_download)
                
                # Process results for each symbol
                for symbol in chunk:
                    try:
                        symbol_upper = symbol.upper()
                        
                        # Handle single vs multiple symbol data structure
                        if len(chunk) == 1:
                            symbol_data = bulk_data
                        else:
                            symbol_data = bulk_data[symbol_upper] if symbol_upper in bulk_data else None
                        
                        if symbol_data is not None and not symbol_data.empty:
                            latest = symbol_data.iloc[-1]
                            results[symbol] = YFinanceQuote(
                                symbol=symbol_upper,
                                price=float(latest['Close']) if not pd.isna(latest['Close']) else None,
                                volume=int(latest['Volume']) if not pd.isna(latest['Volume']) else None,
                                timestamp=datetime.now()
                            )
                        else:
                            # Fallback to individual quote
                            results[symbol] = await self.get_quote(symbol)
                            
                    except Exception as e:
                        logger.warning(f"Failed to process bulk quote for {symbol}: {e}")
                        # Continue with other symbols
                        continue
                        
            except Exception as e:
                logger.warning(f"Bulk download failed for chunk {chunk}: {e}")
                
                # Fallback to individual requests
                for symbol in chunk:
                    try:
                        results[symbol] = await self.get_quote(symbol)
                    except Exception as individual_error:
                        logger.error(f"Individual fallback failed for {symbol}: {individual_error}")
        
        return results
    
    async def search_symbols(self, query: str) -> List[Dict[str, Any]]:
        """Search for symbols by company name or ticker."""
        # Note: YFinance doesn't have a built-in search API
        # This is a simplified implementation
        cache_key = self._get_cache_key(query, "search")
        
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        try:
            # Try to get ticker info as a simple search
            def _search_ticker():
                ticker = yf.Ticker(query.upper())
                info = ticker.info
                
                if info.get('symbol') or info.get('shortName'):
                    return [{
                        'symbol': info.get('symbol', query.upper()),
                        'name': info.get('shortName') or info.get('longName', ''),
                        'type': 'equity',
                        'exchange': info.get('exchange'),
                        'currency': info.get('currency', 'USD')
                    }]
                return []
            
            results = await self._exponential_backoff_retry(_search_ticker)
            self.cache[cache_key] = results
            return results
            
        except Exception as e:
            logger.warning(f"Symbol search failed for {query}: {e}")
            return []

# Global service instance
yfinance_service = YFinanceService()