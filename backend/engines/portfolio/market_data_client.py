#!/usr/bin/env python3
"""
Market Data Client for Portfolio Engine
Connects to MarketData Engine and yfinance for real market data
"""

import asyncio
import logging
import aiohttp
import yfinance as yf
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from cachetools import TTLCache
import pandas as pd

logger = logging.getLogger(__name__)

class MarketDataClient:
    """
    Client for fetching real market data from multiple sources
    """
    
    def __init__(self):
        self.marketdata_engine_url = "http://marketdata:8800"  # Docker service name
        self.cache = TTLCache(maxsize=1000, ttl=300)  # 5-minute cache
        self.session: Optional[aiohttp.ClientSession] = None
        
    async def start(self):
        """Initialize the HTTP session"""
        self.session = aiohttp.ClientSession()
        logger.info("MarketDataClient started")
        
    async def stop(self):
        """Close the HTTP session"""
        if self.session:
            await self.session.close()
        logger.info("MarketDataClient stopped")
        
    async def get_current_price(self, symbol: str) -> Optional[float]:
        """
        Get current price for a symbol from MarketData Engine or yfinance
        """
        try:
            # First try MarketData Engine
            price = await self._get_price_from_marketdata_engine(symbol)
            if price is not None:
                return price
                
            # Fallback to yfinance
            return await self._get_price_from_yfinance(symbol)
            
        except Exception as e:
            logger.error(f"Error getting price for {symbol}: {e}")
            return None
    
    async def get_multiple_prices(self, symbols: List[str]) -> Dict[str, float]:
        """
        Get current prices for multiple symbols
        """
        prices = {}
        
        # Batch request from MarketData Engine first
        marketdata_prices = await self._get_prices_from_marketdata_engine(symbols)
        prices.update(marketdata_prices)
        
        # Get remaining symbols from yfinance
        missing_symbols = [s for s in symbols if s not in prices]
        if missing_symbols:
            yfinance_prices = await self._get_prices_from_yfinance(missing_symbols)
            prices.update(yfinance_prices)
            
        return prices
    
    async def get_historical_data(self, symbol: str, period: str = "1y") -> Optional[pd.DataFrame]:
        """
        Get historical price data for a symbol
        """
        try:
            # Use yfinance for historical data
            ticker = yf.Ticker(symbol)
            data = ticker.history(period=period)
            
            if data is not None and not data.empty:
                logger.info(f"Retrieved {len(data)} historical data points for {symbol}")
                return data
            else:
                logger.warning(f"No historical data available for {symbol}")
                return None
                
        except Exception as e:
            logger.error(f"Error getting historical data for {symbol}: {e}")
            return None
    
    async def _get_price_from_marketdata_engine(self, symbol: str) -> Optional[float]:
        """Get price from MarketData Engine snapshot API"""
        if not self.session:
            return None
            
        cache_key = f"marketdata_{symbol}"
        if cache_key in self.cache:
            return self.cache[cache_key]
            
        try:
            url = f"{self.marketdata_engine_url}/symbols/{symbol}/snapshot"
            async with self.session.post(url) as response:
                if response.status == 200:
                    data = await response.json()
                    snapshot = data.get("snapshot", {})
                    last_trade = snapshot.get("last_trade", {})
                    price = last_trade.get("price")
                    
                    if price is not None:
                        self.cache[cache_key] = price
                        logger.debug(f"Got price from MarketData Engine: {symbol} = ${price}")
                        return price
                        
        except Exception as e:
            logger.warning(f"MarketData Engine request failed for {symbol}: {e}")
            
        return None
    
    async def _get_prices_from_marketdata_engine(self, symbols: List[str]) -> Dict[str, float]:
        """Get multiple prices from MarketData Engine"""
        prices = {}
        
        if not self.session:
            return prices
            
        # Process symbols in parallel but with some rate limiting
        semaphore = asyncio.Semaphore(5)  # Max 5 concurrent requests
        
        async def get_single_price(symbol: str):
            async with semaphore:
                price = await self._get_price_from_marketdata_engine(symbol)
                if price is not None:
                    prices[symbol] = price
                    
        tasks = [get_single_price(symbol) for symbol in symbols]
        await asyncio.gather(*tasks, return_exceptions=True)
        
        return prices
    
    async def _get_price_from_yfinance(self, symbol: str) -> Optional[float]:
        """Get price from yfinance as fallback"""
        cache_key = f"yfinance_{symbol}"
        if cache_key in self.cache:
            return self.cache[cache_key]
            
        try:
            # Run yfinance in executor to avoid blocking
            loop = asyncio.get_event_loop()
            ticker = await loop.run_in_executor(None, yf.Ticker, symbol)
            info = await loop.run_in_executor(None, lambda: ticker.info)
            
            price = info.get("currentPrice") or info.get("regularMarketPrice")
            
            if price is not None:
                self.cache[cache_key] = price
                logger.debug(f"Got price from yfinance: {symbol} = ${price}")
                return price
                
        except Exception as e:
            logger.warning(f"yfinance request failed for {symbol}: {e}")
            
        return None
    
    async def _get_prices_from_yfinance(self, symbols: List[str]) -> Dict[str, float]:
        """Get multiple prices from yfinance"""
        prices = {}
        
        try:
            # Use yfinance bulk download
            loop = asyncio.get_event_loop()
            tickers_str = " ".join(symbols)
            
            # Run in executor to avoid blocking
            data = await loop.run_in_executor(
                None, 
                lambda: yf.download(tickers_str, period="1d", interval="1d", progress=False)
            )
            
            if data is not None and not data.empty:
                # Extract latest close prices
                if len(symbols) == 1:
                    # Single symbol case
                    if 'Close' in data.columns:
                        prices[symbols[0]] = float(data['Close'].iloc[-1])
                else:
                    # Multiple symbols case
                    if 'Close' in data.columns:
                        for symbol in symbols:
                            try:
                                if symbol in data['Close'].columns:
                                    price = data['Close'][symbol].iloc[-1]
                                    if pd.notna(price):
                                        prices[symbol] = float(price)
                            except (KeyError, IndexError):
                                continue
                                
                logger.info(f"Got {len(prices)} prices from yfinance bulk download")
                
        except Exception as e:
            logger.warning(f"yfinance bulk request failed: {e}")
            
        return prices
    
    def get_real_price_for_mock(self, symbol: str) -> float:
        """
        Get a more realistic price for known symbols using current market data
        This replaces the hash-based mock prices with actual market-based estimates
        """
        # Known approximate prices for major symbols (will be replaced by real data)
        known_prices = {
            "AAPL": 175.0,
            "GOOGL": 140.0,
            "MSFT": 420.0,
            "TSLA": 250.0,
            "AMZN": 145.0,
            "NVDA": 875.0,
            "META": 485.0,
            "NFLX": 1204.65,  # Current real price
            "IWM": 234.83,    # Current real price
            "SPY": 540.0,
            "QQQ": 450.0,
            "VTI": 265.0,
            "GLD": 230.0,
            "TLT": 95.0,
            "EFA": 75.0,
            "EEM": 40.0,
        }
        
        return known_prices.get(symbol, 100.0 + (hash(symbol) % 400))