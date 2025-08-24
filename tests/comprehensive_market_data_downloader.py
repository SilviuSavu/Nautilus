#!/usr/bin/env python3
"""
Comprehensive Market Data Downloader for Engine Testing
Downloads extensive market data from all available sources to stress-test containerized engines
"""
import asyncio
import json
import logging
import os
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import pandas as pd
import numpy as np
import yfinance as yf
import requests
import aiohttp
import asyncpg
from sqlalchemy import create_engine

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ComprehensiveMarketDataDownloader:
    def __init__(self):
        self.data_sources = {
            'yfinance': True,
            'alpha_vantage': True,
            'fred': True,
            'data_gov': True,
            'trading_economics': True,
            'dbnomics': True
        }
        
        # API Keys from environment
        self.alpha_vantage_key = "271AHP91HVAPDRGP"
        self.fred_key = "1f1ba9c949e988e12796b7c1f6cce1bf"
        self.datagov_key = "4alUJkyWfUMtRAKsx4gOJXgffG1P0rSPVjRooMvt"
        
        # Test symbols for comprehensive coverage
        self.test_symbols = {
            'stocks': ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'NVDA', 'JPM', 'JNJ', 'V', 'PG',
                      'UNH', 'HD', 'MA', 'DIS', 'PYPL', 'BAC', 'NFLX', 'CRM', 'CMCSA', 'XOM'],
            'etfs': ['SPY', 'QQQ', 'IWM', 'VTI', 'VEA', 'VWO', 'BND', 'TLT', 'GLD', 'XLF'],
            'forex': ['EURUSD=X', 'GBPUSD=X', 'JPYUSD=X', 'AUDUSD=X', 'CADUSD=X'],
            'commodities': ['GC=F', 'SI=F', 'CL=F', 'NG=F', 'ZC=F'],
            'crypto': ['BTC-USD', 'ETH-USD', 'ADA-USD', 'DOT-USD', 'SOL-USD']
        }
        
        self.downloaded_data = {}
        self.performance_metrics = {}
        
    async def download_yfinance_data(self) -> Dict:
        """Download comprehensive data from Yahoo Finance"""
        logger.info("Starting YFinance data download...")
        start_time = time.time()
        
        yf_data = {}
        
        # Get all symbols
        all_symbols = []
        for category, symbols in self.test_symbols.items():
            all_symbols.extend(symbols)
        
        # Download data in batches to avoid rate limiting
        batch_size = 10
        for i in range(0, len(all_symbols), batch_size):
            batch = all_symbols[i:i + batch_size]
            logger.info(f"Downloading batch {i//batch_size + 1}: {batch}")
            
            for symbol in batch:
                try:
                    ticker = yf.Ticker(symbol)
                    
                    # Historical data - 2 years
                    hist_data = ticker.history(period="2y", interval="1d")
                    if not hist_data.empty:
                        yf_data[f"{symbol}_daily"] = hist_data
                    
                    # Hourly data - 1 month
                    hourly_data = ticker.history(period="1mo", interval="1h")
                    if not hourly_data.empty:
                        yf_data[f"{symbol}_hourly"] = hourly_data
                    
                    # Get financials for stocks
                    if symbol in self.test_symbols['stocks']:
                        try:
                            financials = ticker.financials
                            if financials is not None and not financials.empty:
                                yf_data[f"{symbol}_financials"] = financials
                        except Exception as e:
                            logger.warning(f"Could not get financials for {symbol}: {e}")
                    
                    # Brief delay to respect rate limits
                    await asyncio.sleep(0.1)
                    
                except Exception as e:
                    logger.error(f"Error downloading {symbol}: {e}")
                    continue
        
        execution_time = time.time() - start_time
        self.performance_metrics['yfinance'] = {
            'execution_time': execution_time,
            'symbols_downloaded': len(yf_data),
            'records_downloaded': sum(len(df) for df in yf_data.values() if isinstance(df, pd.DataFrame))
        }
        
        logger.info(f"YFinance download complete: {len(yf_data)} datasets in {execution_time:.2f}s")
        return yf_data
    
    async def download_alpha_vantage_data(self) -> Dict:
        """Download data from Alpha Vantage API"""
        logger.info("Starting Alpha Vantage data download...")
        start_time = time.time()
        
        av_data = {}
        base_url = "https://www.alphavantage.co/query"
        
        # Focus on key stocks for detailed analysis
        key_stocks = self.test_symbols['stocks'][:5]  # Limit to avoid API limits
        
        async with aiohttp.ClientSession() as session:
            for symbol in key_stocks:
                try:
                    # Daily adjusted data
                    params = {
                        'function': 'TIME_SERIES_DAILY_ADJUSTED',
                        'symbol': symbol,
                        'apikey': self.alpha_vantage_key,
                        'outputsize': 'full'
                    }
                    
                    async with session.get(base_url, params=params) as response:
                        if response.status == 200:
                            data = await response.json()
                            if 'Time Series (Daily)' in data:
                                av_data[f"{symbol}_av_daily"] = data['Time Series (Daily)']
                    
                    # Technical indicators - RSI
                    params = {
                        'function': 'RSI',
                        'symbol': symbol,
                        'interval': 'daily',
                        'time_period': 14,
                        'series_type': 'close',
                        'apikey': self.alpha_vantage_key
                    }
                    
                    await asyncio.sleep(12)  # Alpha Vantage rate limit: 5 calls per minute
                    
                    async with session.get(base_url, params=params) as response:
                        if response.status == 200:
                            data = await response.json()
                            if 'Technical Analysis: RSI' in data:
                                av_data[f"{symbol}_rsi"] = data['Technical Analysis: RSI']
                    
                    await asyncio.sleep(12)  # Rate limit compliance
                    
                except Exception as e:
                    logger.error(f"Error downloading Alpha Vantage data for {symbol}: {e}")
                    continue
        
        execution_time = time.time() - start_time
        self.performance_metrics['alpha_vantage'] = {
            'execution_time': execution_time,
            'datasets_downloaded': len(av_data),
            'api_calls_made': len(key_stocks) * 2
        }
        
        logger.info(f"Alpha Vantage download complete: {len(av_data)} datasets in {execution_time:.2f}s")
        return av_data
    
    async def download_fred_data(self) -> Dict:
        """Download economic data from FRED"""
        logger.info("Starting FRED data download...")
        start_time = time.time()
        
        fred_data = {}
        base_url = "https://api.stlouisfed.org/fred/series/observations"
        
        # Key economic indicators
        fred_series = [
            'GDP',           # Gross Domestic Product
            'UNRATE',        # Unemployment Rate
            'CPIAUCSL',      # Consumer Price Index
            'FEDFUNDS',      # Federal Funds Rate
            'DGS10',         # 10-Year Treasury Rate
            'DEXUSEU',       # USD/EUR Exchange Rate
            'VIXCLS',        # VIX Volatility Index
            'PAYEMS',        # Non-farm Payrolls
            'HOUST',         # Housing Starts
            'INDPRO'         # Industrial Production Index
        ]
        
        async with aiohttp.ClientSession() as session:
            for series_id in fred_series:
                try:
                    params = {
                        'series_id': series_id,
                        'api_key': self.fred_key,
                        'file_type': 'json',
                        'observation_start': '2020-01-01',
                        'observation_end': datetime.now().strftime('%Y-%m-%d')
                    }
                    
                    async with session.get(base_url, params=params) as response:
                        if response.status == 200:
                            data = await response.json()
                            if 'observations' in data:
                                fred_data[series_id] = data['observations']
                    
                    await asyncio.sleep(0.1)  # Brief delay
                    
                except Exception as e:
                    logger.error(f"Error downloading FRED series {series_id}: {e}")
                    continue
        
        execution_time = time.time() - start_time
        self.performance_metrics['fred'] = {
            'execution_time': execution_time,
            'series_downloaded': len(fred_data),
            'total_observations': sum(len(obs) for obs in fred_data.values())
        }
        
        logger.info(f"FRED download complete: {len(fred_data)} series in {execution_time:.2f}s")
        return fred_data
    
    async def generate_synthetic_data(self) -> Dict:
        """Generate synthetic high-frequency data for stress testing"""
        logger.info("Generating synthetic high-frequency data...")
        start_time = time.time()
        
        synthetic_data = {}
        
        # Generate high-frequency price data
        np.random.seed(42)
        
        for symbol in ['AAPL', 'MSFT', 'GOOGL']:
            # 1-minute bars for 1 month
            n_minutes = 30 * 24 * 60  # 30 days
            base_price = 100.0
            
            # Generate realistic price movements
            returns = np.random.normal(0, 0.001, n_minutes)  # 0.1% volatility per minute
            prices = [base_price]
            
            for i in range(n_minutes):
                new_price = prices[-1] * (1 + returns[i])
                prices.append(max(new_price, 0.01))  # Ensure positive prices
            
            # Create timestamps
            start_date = datetime.now() - timedelta(days=30)
            timestamps = [start_date + timedelta(minutes=i) for i in range(n_minutes + 1)]
            
            # Create OHLCV data
            ohlcv_data = []
            for i in range(len(prices) - 1):
                high = prices[i] * (1 + abs(np.random.normal(0, 0.0005)))
                low = prices[i] * (1 - abs(np.random.normal(0, 0.0005)))
                volume = int(np.random.exponential(1000000))  # Realistic volume distribution
                
                ohlcv_data.append({
                    'timestamp': timestamps[i].isoformat(),
                    'open': round(prices[i], 2),
                    'high': round(high, 2),
                    'low': round(low, 2),
                    'close': round(prices[i + 1], 2),
                    'volume': volume
                })
            
            synthetic_data[f"{symbol}_hf"] = ohlcv_data
        
        # Generate order book data
        for symbol in ['AAPL', 'MSFT']:
            order_book_snapshots = []
            
            for i in range(10000):  # 10k snapshots
                mid_price = 100 + np.random.normal(0, 5)
                
                bids = []
                asks = []
                
                # Generate 10 bid/ask levels
                for level in range(10):
                    bid_price = mid_price - (level + 1) * 0.01 - np.random.exponential(0.01)
                    ask_price = mid_price + (level + 1) * 0.01 + np.random.exponential(0.01)
                    
                    bid_size = int(np.random.exponential(1000))
                    ask_size = int(np.random.exponential(1000))
                    
                    bids.append([round(bid_price, 2), bid_size])
                    asks.append([round(ask_price, 2), ask_size])
                
                order_book_snapshots.append({
                    'timestamp': (datetime.now() - timedelta(seconds=i)).isoformat(),
                    'symbol': symbol,
                    'bids': bids,
                    'asks': asks
                })
            
            synthetic_data[f"{symbol}_orderbook"] = order_book_snapshots
        
        execution_time = time.time() - start_time
        self.performance_metrics['synthetic'] = {
            'execution_time': execution_time,
            'datasets_generated': len(synthetic_data),
            'total_records': sum(len(data) for data in synthetic_data.values())
        }
        
        logger.info(f"Synthetic data generation complete: {len(synthetic_data)} datasets in {execution_time:.2f}s")
        return synthetic_data
    
    async def save_data_to_files(self):
        """Save all downloaded data to files for engine testing"""
        logger.info("Saving data to files...")
        
        # Create test data directory
        os.makedirs('/Users/savusilviu/Desktop/SilviuCorneliuSavu/Nautilus/tests/data', exist_ok=True)
        
        # Save each dataset
        for source, data in self.downloaded_data.items():
            for dataset_name, dataset in data.items():
                try:
                    file_path = f'/Users/savusilviu/Desktop/SilviuCorneliuSavu/Nautilus/tests/data/{source}_{dataset_name}.json'
                    
                    if isinstance(dataset, pd.DataFrame):
                        # Convert DataFrame to JSON
                        dataset.to_json(file_path, orient='index', date_format='iso')
                    else:
                        # Save as JSON
                        with open(file_path, 'w') as f:
                            json.dump(dataset, f, indent=2, default=str)
                    
                    logger.info(f"Saved {dataset_name} to {file_path}")
                    
                except Exception as e:
                    logger.error(f"Error saving {dataset_name}: {e}")
    
    async def run_comprehensive_download(self):
        """Execute comprehensive data download from all sources"""
        logger.info("Starting comprehensive market data download...")
        overall_start = time.time()
        
        # Download from all sources
        tasks = []
        
        if self.data_sources['yfinance']:
            tasks.append(('yfinance', self.download_yfinance_data()))
        
        if self.data_sources['alpha_vantage']:
            tasks.append(('alpha_vantage', self.download_alpha_vantage_data()))
        
        if self.data_sources['fred']:
            tasks.append(('fred', self.download_fred_data()))
        
        # Add synthetic data generation
        tasks.append(('synthetic', self.generate_synthetic_data()))
        
        # Execute downloads
        for source_name, task in tasks:
            try:
                data = await task
                self.downloaded_data[source_name] = data
                logger.info(f"Completed {source_name} download")
            except Exception as e:
                logger.error(f"Failed to download from {source_name}: {e}")
        
        # Save all data
        await self.save_data_to_files()
        
        # Generate summary report
        total_time = time.time() - overall_start
        
        summary = {
            'download_completed': datetime.now().isoformat(),
            'total_execution_time': total_time,
            'data_sources': list(self.downloaded_data.keys()),
            'performance_metrics': self.performance_metrics,
            'total_datasets': sum(len(data) for data in self.downloaded_data.values()),
            'total_records': sum(
                len(dataset) if isinstance(dataset, (list, dict)) else len(dataset) if hasattr(dataset, '__len__') else 0 
                for data in self.downloaded_data.values() 
                for dataset in data.values()
            )
        }
        
        # Save summary
        with open('/Users/savusilviu/Desktop/SilviuCorneliuSavu/Nautilus/tests/data/download_summary.json', 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        logger.info("=" * 80)
        logger.info("COMPREHENSIVE DATA DOWNLOAD COMPLETE")
        logger.info(f"Total execution time: {total_time:.2f} seconds")
        logger.info(f"Data sources: {len(self.downloaded_data)}")
        logger.info(f"Total datasets: {summary['total_datasets']}")
        logger.info(f"Total records: {summary['total_records']:,}")
        logger.info("=" * 80)
        
        return summary

async def main():
    """Main execution function"""
    downloader = ComprehensiveMarketDataDownloader()
    summary = await downloader.run_comprehensive_download()
    return summary

if __name__ == "__main__":
    asyncio.run(main())