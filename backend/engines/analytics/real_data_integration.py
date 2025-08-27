#!/usr/bin/env python3
"""
Real Data Integration Module for Analytics Engine
Integrates with database and MarketData Hub for real analytics calculations
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple
import pandas as pd
import numpy as np
from dataclasses import dataclass
import json
import asyncpg
# Direct API calls BLOCKED - use MarketData Client only
# import yfinance as yf  # BLOCKED - use MarketData Client
# import aiohttp  # BLOCKED - use MarketData Client
from marketdata_client import create_marketdata_client, DataType, DataSource
from universal_enhanced_messagebus_client import EngineType
from concurrent.futures import ThreadPoolExecutor
import time

# Configure logging
logger = logging.getLogger(__name__)

@dataclass 
class MarketDataPoint:
    """Market data point with timestamp and OHLCV data"""
    symbol: str
    timestamp: datetime
    open_price: float
    high_price: float
    low_price: float
    close_price: float
    volume: float
    venue: str = "SMART"
    
@dataclass
class PerformanceMetrics:
    """Real performance metrics calculated from actual data"""
    symbol: str
    period_start: datetime
    period_end: datetime
    total_return: float
    sharpe_ratio: float
    max_drawdown: float
    volatility: float
    beta: float
    alpha: float
    calmar_ratio: float
    sortino_ratio: float
    win_rate: float
    avg_win: float
    avg_loss: float
    profit_factor: float

@dataclass
class TechnicalIndicators:
    """Technical indicators calculated from real price data"""
    symbol: str
    timestamp: datetime
    sma_20: float
    sma_50: float
    ema_12: float
    ema_26: float
    rsi_14: float
    macd: float
    macd_signal: float
    macd_histogram: float
    bb_upper: float
    bb_middle: float
    bb_lower: float
    atr_14: float
    stoch_k: float
    stoch_d: float

class RealDataIntegration:
    """
    Real data integration for Analytics Engine
    Connects to database and MarketData Hub for centralized data access
    """
    
    def __init__(self, database_url: str, marketdata_engine_url: str = "http://marketdata-engine:8800"):
        self.database_url = database_url
        self.marketdata_engine_url = marketdata_engine_url  # Legacy - not used
        self.db_pool = None
        # MarketData Client - MANDATORY for all external data
        self.marketdata_client = None
        self.executor = ThreadPoolExecutor(max_workers=4)
        
        # Benchmark symbols for beta calculations
        self.benchmark_symbols = ["SPY.SMART", "QQQ.SMART", "IWM.SMART"]
        
        # Cache for market data
        self.price_cache: Dict[str, pd.DataFrame] = {}
        self.cache_expiry: Dict[str, datetime] = {}
        self.cache_duration = timedelta(minutes=15)
        
    async def initialize(self):
        """Initialize database connection pool and HTTP session"""
        try:
            # Initialize database connection pool
            self.db_pool = await asyncpg.create_pool(
                self.database_url,
                min_size=2,
                max_size=10,
                command_timeout=30,
                server_settings={
                    'jit': 'off'  # Disable JIT for faster simple queries
                }
            )
            
            # Initialize MarketData Client for external data
            self.marketdata_client = create_marketdata_client(
                EngineType.ANALYTICS,
                8100  # Analytics Engine port
            )
            
            logger.info("RealDataIntegration initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize RealDataIntegration: {e}")
            raise
    
    async def cleanup(self):
        """Cleanup resources"""
        if self.db_pool:
            await self.db_pool.close()
        # MarketData Client cleanup handled automatically
        if self.executor:
            self.executor.shutdown(wait=True)
    
    async def get_market_data_from_db(
        self, 
        symbol: str, 
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        limit: int = 1000
    ) -> List[MarketDataPoint]:
        """
        Retrieve market data from database market_bars table
        """
        try:
            # Use cache if available and fresh
            cache_key = f"{symbol}_{start_date}_{end_date}_{limit}"
            if (cache_key in self.price_cache and 
                cache_key in self.cache_expiry and 
                datetime.now() < self.cache_expiry[cache_key]):
                
                df = self.price_cache[cache_key]
                return [MarketDataPoint(
                    symbol=row['instrument_id'],
                    timestamp=datetime.fromtimestamp(row['timestamp_ns'] / 1_000_000_000),
                    open_price=float(row['open_price']),
                    high_price=float(row['high_price']),
                    low_price=float(row['low_price']),
                    close_price=float(row['close_price']),
                    volume=float(row['volume']),
                    venue=row['venue']
                ) for _, row in df.iterrows()]
            
            # Build query
            query = """
                SELECT instrument_id, venue, timestamp_ns, open_price, high_price, 
                       low_price, close_price, volume
                FROM market_bars 
                WHERE instrument_id = $1
            """
            params = [symbol]
            param_count = 2
            
            if start_date:
                query += f" AND timestamp_ns >= ${param_count}"
                params.append(int(start_date.timestamp() * 1_000_000_000))
                param_count += 1
                
            if end_date:
                query += f" AND timestamp_ns <= ${param_count}"
                params.append(int(end_date.timestamp() * 1_000_000_000))
                param_count += 1
                
            query += f" ORDER BY timestamp_ns DESC LIMIT ${param_count}"
            params.append(limit)
            
            async with self.db_pool.acquire() as conn:
                rows = await conn.fetch(query, *params)
                
            if not rows:
                logger.warning(f"No market data found for symbol {symbol}")
                return []
            
            # Convert to DataFrame and cache
            df = pd.DataFrame([dict(row) for row in rows])
            self.price_cache[cache_key] = df
            self.cache_expiry[cache_key] = datetime.now() + self.cache_duration
            
            # Convert to MarketDataPoint objects
            result = []
            for _, row in df.iterrows():
                result.append(MarketDataPoint(
                    symbol=row['instrument_id'],
                    timestamp=datetime.fromtimestamp(row['timestamp_ns'] / 1_000_000_000),
                    open_price=float(row['open_price']),
                    high_price=float(row['high_price']),
                    low_price=float(row['low_price']),
                    close_price=float(row['close_price']),
                    volume=float(row['volume']),
                    venue=row['venue']
                ))
            
            logger.info(f"Retrieved {len(result)} market data points for {symbol}")
            return result
            
        except Exception as e:
            logger.error(f"Error retrieving market data for {symbol}: {e}")
            return []
    
    async def get_marketdata_via_hub(
        self, 
        symbol: str, 
        data_types: List[DataType] = None,
        sources: List[DataSource] = None
    ) -> List[MarketDataPoint]:
        """
        Get market data via MarketData Hub (sub-5ms performance)
        REPLACES yfinance direct calls with centralized hub access
        """
        try:
            if not self.marketdata_client:
                logger.error("MarketData Client not initialized")
                return []
            
            # Default to comprehensive data types
            if data_types is None:
                data_types = [DataType.BAR, DataType.QUOTE, DataType.TRADE]
            
            # Default to all available sources
            if sources is None:
                sources = [DataSource.YAHOO, DataSource.ALPHA_VANTAGE, DataSource.IBKR]
            
            # Request data via MarketData Hub (sub-5ms)
            hub_data = await self.marketdata_client.get_data(
                symbols=[symbol],
                data_types=data_types,
                sources=sources,
                cache=True  # Use cache for maximum performance
            )
            
            # Convert hub data to MarketDataPoint objects
            result = []
            if hub_data and symbol in hub_data:
                symbol_data = hub_data[symbol]
                
                # Extract OHLCV bars if available
                if 'bars' in symbol_data:
                    for bar in symbol_data['bars']:
                        result.append(MarketDataPoint(
                            symbol=symbol,
                            timestamp=datetime.fromisoformat(bar['timestamp']),
                            open_price=float(bar['open']),
                            high_price=float(bar['high']),
                            low_price=float(bar['low']),
                            close_price=float(bar['close']),
                            volume=float(bar['volume']),
                            venue="HUB"
                        ))
                
                # Extract quotes if bars not available
                elif 'quotes' in symbol_data:
                    for quote in symbol_data['quotes']:
                        price = (float(quote['bid']) + float(quote['ask'])) / 2
                        result.append(MarketDataPoint(
                            symbol=symbol,
                            timestamp=datetime.fromisoformat(quote['timestamp']),
                            open_price=price,
                            high_price=price,
                            low_price=price,
                            close_price=price,
                            volume=0.0,  # Quote data doesn't have volume
                            venue="HUB"
                        ))
            
            logger.info(f"Retrieved {len(result)} data points for {symbol} via MarketData Hub")
            return result
            
        except Exception as e:
            logger.error(f"Error retrieving hub data for {symbol}: {e}")
            return []
    
    async def get_marketdata_snapshot(self, symbol: str) -> Optional[MarketDataPoint]:
        """
        Get real-time snapshot via MarketData Hub (sub-5ms performance)
        REPLACES direct HTTP calls with centralized hub access
        """
        try:
            if not self.marketdata_client:
                logger.error("MarketData Client not initialized")
                return None
            
            # Get real-time snapshot via hub
            snapshot_data = await self.marketdata_client.get_data(
                symbols=[symbol],
                data_types=[DataType.QUOTE, DataType.TICK],
                sources=[DataSource.IBKR],  # Real-time data from IBKR
                cache=False  # Real-time data shouldn't use cache
            )
            
            if snapshot_data and symbol in snapshot_data:
                symbol_data = snapshot_data[symbol]
                
                # Use latest quote or tick data
                if 'quotes' in symbol_data and symbol_data['quotes']:
                    quote = symbol_data['quotes'][-1]  # Latest quote
                    price = (float(quote['bid']) + float(quote['ask'])) / 2
                    
                    return MarketDataPoint(
                        symbol=symbol,
                        timestamp=datetime.fromisoformat(quote['timestamp']),
                        open_price=price,
                        high_price=price,
                        low_price=price,
                        close_price=price,
                        volume=0.0,
                        venue="HUB_REALTIME"
                    )
                
                elif 'ticks' in symbol_data and symbol_data['ticks']:
                    tick = symbol_data['ticks'][-1]  # Latest tick
                    price = float(tick['price'])
                    
                    return MarketDataPoint(
                        symbol=symbol,
                        timestamp=datetime.fromisoformat(tick['timestamp']),
                        open_price=price,
                        high_price=price,
                        low_price=price,
                        close_price=price,
                        volume=float(tick['volume']),
                        venue="HUB_REALTIME"
                    )
            
            # Fallback to database if real-time not available
            recent_data = await self.get_market_data_from_db(symbol, limit=1)
            return recent_data[0] if recent_data else None
                
        except Exception as e:
            logger.error(f"Error getting market data snapshot for {symbol}: {e}")
            return None
    
    def calculate_technical_indicators(self, data_points: List[MarketDataPoint]) -> Optional[TechnicalIndicators]:
        """
        Calculate technical indicators from real price data
        """
        try:
            if len(data_points) < 50:  # Need enough data for indicators
                return None
                
            # Convert to DataFrame for easier calculation
            df = pd.DataFrame([{
                'timestamp': dp.timestamp,
                'close': dp.close_price,
                'high': dp.high_price,
                'low': dp.low_price,
                'volume': dp.volume
            } for dp in data_points])
            
            df = df.sort_values('timestamp').reset_index(drop=True)
            
            if len(df) < 50:
                return None
            
            # Simple Moving Averages
            df['sma_20'] = df['close'].rolling(window=20).mean()
            df['sma_50'] = df['close'].rolling(window=50).mean()
            
            # Exponential Moving Averages
            df['ema_12'] = df['close'].ewm(span=12).mean()
            df['ema_26'] = df['close'].ewm(span=26).mean()
            
            # MACD
            df['macd'] = df['ema_12'] - df['ema_26']
            df['macd_signal'] = df['macd'].ewm(span=9).mean()
            df['macd_histogram'] = df['macd'] - df['macd_signal']
            
            # RSI
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            df['rsi_14'] = 100 - (100 / (1 + rs))
            
            # Bollinger Bands
            df['bb_middle'] = df['close'].rolling(window=20).mean()
            bb_std = df['close'].rolling(window=20).std()
            df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
            df['bb_lower'] = df['bb_middle'] - (bb_std * 2)
            
            # ATR (Average True Range)
            df['tr1'] = df['high'] - df['low']
            df['tr2'] = abs(df['high'] - df['close'].shift())
            df['tr3'] = abs(df['low'] - df['close'].shift())
            df['tr'] = df[['tr1', 'tr2', 'tr3']].max(axis=1)
            df['atr_14'] = df['tr'].rolling(window=14).mean()
            
            # Stochastic Oscillator
            df['lowest_low'] = df['low'].rolling(window=14).min()
            df['highest_high'] = df['high'].rolling(window=14).max()
            df['stoch_k'] = ((df['close'] - df['lowest_low']) / (df['highest_high'] - df['lowest_low'])) * 100
            df['stoch_d'] = df['stoch_k'].rolling(window=3).mean()
            
            # Get the latest values
            latest = df.iloc[-1]
            
            return TechnicalIndicators(
                symbol=data_points[0].symbol,
                timestamp=latest['timestamp'],
                sma_20=float(latest['sma_20']) if pd.notna(latest['sma_20']) else 0.0,
                sma_50=float(latest['sma_50']) if pd.notna(latest['sma_50']) else 0.0,
                ema_12=float(latest['ema_12']) if pd.notna(latest['ema_12']) else 0.0,
                ema_26=float(latest['ema_26']) if pd.notna(latest['ema_26']) else 0.0,
                rsi_14=float(latest['rsi_14']) if pd.notna(latest['rsi_14']) else 50.0,
                macd=float(latest['macd']) if pd.notna(latest['macd']) else 0.0,
                macd_signal=float(latest['macd_signal']) if pd.notna(latest['macd_signal']) else 0.0,
                macd_histogram=float(latest['macd_histogram']) if pd.notna(latest['macd_histogram']) else 0.0,
                bb_upper=float(latest['bb_upper']) if pd.notna(latest['bb_upper']) else 0.0,
                bb_middle=float(latest['bb_middle']) if pd.notna(latest['bb_middle']) else 0.0,
                bb_lower=float(latest['bb_lower']) if pd.notna(latest['bb_lower']) else 0.0,
                atr_14=float(latest['atr_14']) if pd.notna(latest['atr_14']) else 0.0,
                stoch_k=float(latest['stoch_k']) if pd.notna(latest['stoch_k']) else 50.0,
                stoch_d=float(latest['stoch_d']) if pd.notna(latest['stoch_d']) else 50.0,
            )
            
        except Exception as e:
            logger.error(f"Error calculating technical indicators: {e}")
            return None
    
    async def calculate_performance_metrics(
        self, 
        symbol: str, 
        period_days: int = 252,
        benchmark_symbol: str = "SPY.SMART"
    ) -> Optional[PerformanceMetrics]:
        """
        Calculate real performance metrics from market data
        """
        try:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=period_days)
            
            # Get price data for the symbol
            symbol_data = await self.get_market_data_from_db(
                symbol, start_date=start_date, end_date=end_date, limit=period_days
            )
            
            if len(symbol_data) < 30:  # Need at least 30 days of data
                logger.warning(f"Insufficient data for performance calculation: {len(symbol_data)} days")
                return None
            
            # Get benchmark data for beta calculation
            benchmark_data = await self.get_market_data_from_db(
                benchmark_symbol, start_date=start_date, end_date=end_date, limit=period_days
            )
            
            # Convert to DataFrame for analysis
            symbol_df = pd.DataFrame([{
                'timestamp': dp.timestamp,
                'close': dp.close_price
            } for dp in symbol_data]).sort_values('timestamp')
            
            benchmark_df = pd.DataFrame([{
                'timestamp': dp.timestamp,
                'close': dp.close_price
            } for dp in benchmark_data]).sort_values('timestamp') if benchmark_data else None
            
            # Calculate returns
            symbol_df['returns'] = symbol_df['close'].pct_change().fillna(0)
            
            # Performance calculations
            total_return = (symbol_df['close'].iloc[-1] / symbol_df['close'].iloc[0] - 1) * 100
            
            # Risk metrics
            volatility = symbol_df['returns'].std() * np.sqrt(252) * 100  # Annualized
            
            # Sharpe ratio (assuming 2% risk-free rate)
            risk_free_rate = 0.02
            excess_returns = symbol_df['returns'].mean() * 252 - risk_free_rate
            sharpe_ratio = excess_returns / (volatility / 100) if volatility > 0 else 0
            
            # Maximum drawdown
            cumulative_returns = (1 + symbol_df['returns']).cumprod()
            running_max = cumulative_returns.expanding().max()
            drawdown = (cumulative_returns - running_max) / running_max
            max_drawdown = drawdown.min() * 100
            
            # Beta calculation
            beta = 0.0
            alpha = 0.0
            if benchmark_df is not None and len(benchmark_df) > 0:
                benchmark_df['returns'] = benchmark_df['returns'].pct_change().fillna(0)
                
                # Align dates
                merged = pd.merge(symbol_df[['timestamp', 'returns']], 
                                benchmark_df[['timestamp', 'returns']], 
                                on='timestamp', suffixes=('_symbol', '_benchmark'))
                
                if len(merged) > 10:
                    covariance = np.cov(merged['returns_symbol'], merged['returns_benchmark'])[0, 1]
                    benchmark_variance = np.var(merged['returns_benchmark'])
                    beta = covariance / benchmark_variance if benchmark_variance > 0 else 0.0
                    
                    # Alpha calculation
                    benchmark_return = merged['returns_benchmark'].mean() * 252
                    alpha = excess_returns - beta * (benchmark_return - risk_free_rate)
            
            # Calmar ratio
            calmar_ratio = (total_return / 100) / abs(max_drawdown / 100) if max_drawdown < 0 else 0
            
            # Sortino ratio (downside deviation)
            downside_returns = symbol_df['returns'][symbol_df['returns'] < 0]
            downside_std = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else volatility / 100
            sortino_ratio = excess_returns / downside_std if downside_std > 0 else 0
            
            # Win rate and profit factor
            positive_returns = symbol_df['returns'][symbol_df['returns'] > 0]
            negative_returns = symbol_df['returns'][symbol_df['returns'] < 0]
            
            win_rate = len(positive_returns) / len(symbol_df['returns']) * 100 if len(symbol_df['returns']) > 0 else 0
            avg_win = positive_returns.mean() * 100 if len(positive_returns) > 0 else 0
            avg_loss = abs(negative_returns.mean()) * 100 if len(negative_returns) > 0 else 0
            profit_factor = (avg_win * len(positive_returns)) / (avg_loss * len(negative_returns)) if avg_loss > 0 and len(negative_returns) > 0 else 0
            
            return PerformanceMetrics(
                symbol=symbol,
                period_start=symbol_df['timestamp'].iloc[0],
                period_end=symbol_df['timestamp'].iloc[-1],
                total_return=total_return,
                sharpe_ratio=sharpe_ratio,
                max_drawdown=max_drawdown,
                volatility=volatility,
                beta=beta,
                alpha=alpha,
                calmar_ratio=calmar_ratio,
                sortino_ratio=sortino_ratio,
                win_rate=win_rate,
                avg_win=avg_win,
                avg_loss=avg_loss,
                profit_factor=profit_factor
            )
            
        except Exception as e:
            logger.error(f"Error calculating performance metrics for {symbol}: {e}")
            return None
    
    async def get_comprehensive_analysis(self, symbol: str) -> Dict[str, Any]:
        """
        Get comprehensive analysis combining all data sources
        """
        try:
            start_time = time.time()
            
            # Parallel data retrieval via MarketData Hub
            tasks = [
                self.get_market_data_from_db(symbol, limit=250),  # ~1 year of daily data
                self.get_marketdata_via_hub(symbol),  # Hub data replacing yfinance
                self.get_marketdata_snapshot(symbol),
                self.calculate_performance_metrics(symbol)
            ]
            
            db_data, hub_data, snapshot, performance = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Handle exceptions
            if isinstance(db_data, Exception):
                logger.error(f"Database data error: {db_data}")
                db_data = []
            if isinstance(hub_data, Exception):
                logger.error(f"MarketData Hub error: {hub_data}")
                hub_data = []
            if isinstance(snapshot, Exception):
                logger.error(f"Snapshot data error: {snapshot}")
                snapshot = None
            if isinstance(performance, Exception):
                logger.error(f"Performance calculation error: {performance}")
                performance = None
            
            # Use database data as primary, MarketData Hub as secondary
            primary_data = db_data if db_data else hub_data
            
            # Calculate technical indicators
            technical_indicators = None
            if primary_data:
                technical_indicators = self.calculate_technical_indicators(primary_data)
            
            processing_time = (time.time() - start_time) * 1000  # Convert to milliseconds
            
            return {
                "symbol": symbol,
                "timestamp": datetime.now().isoformat(),
                "data_sources": {
                    "database_points": len(db_data) if db_data else 0,
                    "hub_points": len(hub_data) if hub_data else 0,
                    "real_time_snapshot": snapshot is not None,
                    "primary_source": "database" if db_data else ("hub" if hub_data else "none")
                },
                "current_price": {
                    "price": snapshot.close_price if snapshot else (primary_data[-1].close_price if primary_data else 0),
                    "timestamp": snapshot.timestamp.isoformat() if snapshot else (primary_data[-1].timestamp.isoformat() if primary_data else ""),
                    "venue": snapshot.venue if snapshot else (primary_data[-1].venue if primary_data else "")
                },
                "performance_metrics": {
                    "total_return_pct": performance.total_return if performance else 0,
                    "sharpe_ratio": performance.sharpe_ratio if performance else 0,
                    "max_drawdown_pct": performance.max_drawdown if performance else 0,
                    "volatility_pct": performance.volatility if performance else 0,
                    "beta": performance.beta if performance else 0,
                    "alpha": performance.alpha if performance else 0,
                    "calmar_ratio": performance.calmar_ratio if performance else 0,
                    "sortino_ratio": performance.sortino_ratio if performance else 0,
                    "win_rate_pct": performance.win_rate if performance else 0,
                    "profit_factor": performance.profit_factor if performance else 0
                } if performance else {},
                "technical_indicators": {
                    "sma_20": technical_indicators.sma_20 if technical_indicators else 0,
                    "sma_50": technical_indicators.sma_50 if technical_indicators else 0,
                    "rsi_14": technical_indicators.rsi_14 if technical_indicators else 50,
                    "macd": technical_indicators.macd if technical_indicators else 0,
                    "macd_signal": technical_indicators.macd_signal if technical_indicators else 0,
                    "bb_upper": technical_indicators.bb_upper if technical_indicators else 0,
                    "bb_lower": technical_indicators.bb_lower if technical_indicators else 0,
                    "atr_14": technical_indicators.atr_14 if technical_indicators else 0,
                    "stoch_k": technical_indicators.stoch_k if technical_indicators else 50,
                } if technical_indicators else {},
                "processing_time_ms": processing_time,
                "data_quality": {
                    "sufficient_data": len(primary_data) >= 50 if primary_data else False,
                    "real_time_available": snapshot is not None,
                    "performance_calculated": performance is not None,
                    "technical_indicators_calculated": technical_indicators is not None
                }
            }
            
        except Exception as e:
            logger.error(f"Error in comprehensive analysis for {symbol}: {e}")
            return {
                "symbol": symbol,
                "timestamp": datetime.now().isoformat(),
                "error": str(e),
                "data_sources": {},
                "processing_time_ms": 0
            }