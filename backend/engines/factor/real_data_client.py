#!/usr/bin/env python3
"""
Real Data Client for Factor Engine
==================================

HTTP client to connect to existing operational APIs:
- FRED Economic Data API
- EDGAR SEC Filings API  
- Alpha Vantage Market Data API
- Database Economic Indicators

This replaces mock data with real data from the main backend's APIs.
"""

import asyncio
import logging
import aiohttp
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
import pandas as pd
import numpy as np
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class RealDataPoint:
    """Real data point from external APIs"""
    value: float
    date: datetime
    source: str
    indicator: str
    confidence: float = 1.0

class RealDataClient:
    """Client for connecting to real data APIs"""
    
    def __init__(self, base_url: str = "http://172.19.0.18:8001"):
        self.base_url = base_url
        self.session: Optional[aiohttp.ClientSession] = None
        self.cache: Dict[str, List[RealDataPoint]] = {}
        self.cache_ttl = 300  # 5 minutes cache
        self.last_update: Dict[str, datetime] = {}
        
    async def __aenter__(self):
        """Async context manager entry"""
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=30)
        )
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self.session:
            await self.session.close()
    
    async def _make_request(self, endpoint: str, params: Optional[Dict] = None) -> Dict[str, Any]:
        """Make HTTP request to backend API"""
        if not self.session:
            raise RuntimeError("Client not initialized. Use async context manager.")
            
        url = f"{self.base_url}{endpoint}"
        try:
            async with self.session.get(url, params=params or {}) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    logger.error(f"API request failed: {response.status} for {url}")
                    response_text = await response.text()
                    raise aiohttp.ClientError(f"HTTP {response.status}: {response_text}")
        except Exception as e:
            logger.error(f"Request to {url} failed: {e}")
            raise
    
    def _is_cache_valid(self, key: str) -> bool:
        """Check if cache entry is still valid"""
        if key not in self.last_update:
            return False
        return (datetime.now() - self.last_update[key]).seconds < self.cache_ttl
    
    async def get_fred_series(self, series_id: str, limit: Optional[int] = None) -> List[RealDataPoint]:
        """Get FRED economic series data"""
        cache_key = f"fred_{series_id}_{limit or 'all'}"
        
        if self._is_cache_valid(cache_key) and cache_key in self.cache:
            return self.cache[cache_key]
        
        try:
            params = {}
            if limit:
                params['limit'] = limit
                
            data = await self._make_request(f"/api/v1/fred/series/{series_id}", params)
            
            if not data.get('success'):
                logger.error(f"FRED API returned error for {series_id}")
                return []
            
            data_points = []
            for item in data.get('data', []):
                try:
                    data_points.append(RealDataPoint(
                        value=float(item['value']),
                        date=datetime.strptime(item['date'], '%Y-%m-%d'),
                        source='FRED',
                        indicator=series_id,
                        confidence=1.0
                    ))
                except (ValueError, KeyError) as e:
                    logger.warning(f"Skipping invalid FRED data point: {e}")
                    continue
            
            self.cache[cache_key] = data_points
            self.last_update[cache_key] = datetime.now()
            
            logger.info(f"Retrieved {len(data_points)} data points for FRED series {series_id}")
            return data_points
            
        except Exception as e:
            logger.error(f"Failed to get FRED series {series_id}: {e}")
            return []
    
    async def get_fred_macro_factors(self) -> Dict[str, float]:
        """Get calculated macro factors from FRED"""
        try:
            data = await self._make_request("/api/v1/fred/macro-factors")
            return data.get('factors', {})
        except Exception as e:
            logger.error(f"Failed to get FRED macro factors: {e}")
            return {}
    
    async def get_edgar_company_facts(self, ticker: str) -> Dict[str, Any]:
        """Get EDGAR company facts for fundamental analysis"""
        cache_key = f"edgar_facts_{ticker}"
        
        if self._is_cache_valid(cache_key) and cache_key in self.cache:
            # Return cached dict data (not RealDataPoint for complex structures)
            return getattr(self.cache[cache_key], 'raw_data', {})
        
        try:
            data = await self._make_request(f"/api/v1/edgar/ticker/{ticker}/facts")
            
            # Store in cache with special handling for complex data
            cache_entry = type('CacheEntry', (), {'raw_data': data})()
            self.cache[cache_key] = cache_entry
            self.last_update[cache_key] = datetime.now()
            
            logger.info(f"Retrieved EDGAR facts for {ticker}")
            return data
            
        except Exception as e:
            logger.error(f"Failed to get EDGAR facts for {ticker}: {e}")
            return {}
    
    async def get_edgar_company_filings(self, ticker: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent company filings from EDGAR"""
        try:
            data = await self._make_request(
                f"/api/v1/edgar/ticker/{ticker}/filings",
                params={'limit': limit}
            )
            return data.get('filings', [])
        except Exception as e:
            logger.error(f"Failed to get EDGAR filings for {ticker}: {e}")
            return []
    
    async def get_alpha_vantage_quote(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get real-time quote from Alpha Vantage"""
        try:
            data = await self._make_request(f"/api/v1/nautilus-data/alpha-vantage/quote/{symbol}")
            return data.get('data', {})
        except Exception as e:
            logger.warning(f"Alpha Vantage quote failed for {symbol}: {e}")
            return None
    
    async def search_alpha_vantage_symbols(self, keywords: str) -> List[Dict[str, Any]]:
        """Search symbols using Alpha Vantage"""
        try:
            data = await self._make_request(
                "/api/v1/nautilus-data/alpha-vantage/search",
                params={'keywords': keywords}
            )
            return data.get('results', [])
        except Exception as e:
            logger.error(f"Alpha Vantage search failed for {keywords}: {e}")
            return []
    
    async def get_multiple_fred_series(self, series_ids: List[str], limit: Optional[int] = None) -> Dict[str, List[RealDataPoint]]:
        """Get multiple FRED series concurrently"""
        tasks = []
        for series_id in series_ids:
            tasks.append(self.get_fred_series(series_id, limit))
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        series_data = {}
        for i, result in enumerate(results):
            series_id = series_ids[i]
            if isinstance(result, Exception):
                logger.error(f"Failed to get series {series_id}: {result}")
                series_data[series_id] = []
            else:
                series_data[series_id] = result
        
        return series_data
    
    def get_latest_value(self, data_points: List[RealDataPoint]) -> Optional[float]:
        """Get the most recent value from data points"""
        if not data_points:
            return None
        # Sort by date and get the latest
        sorted_points = sorted(data_points, key=lambda x: x.date, reverse=True)
        return sorted_points[0].value
    
    def calculate_growth_rate(self, data_points: List[RealDataPoint], periods: int = 4) -> Optional[float]:
        """Calculate growth rate over specified periods"""
        if len(data_points) < periods + 1:
            return None
        
        sorted_points = sorted(data_points, key=lambda x: x.date, reverse=True)
        current_value = sorted_points[0].value
        past_value = sorted_points[periods].value
        
        if past_value == 0:
            return None
        
        return (current_value - past_value) / abs(past_value)
    
    def to_pandas_series(self, data_points: List[RealDataPoint], value_col: str = 'value') -> pd.Series:
        """Convert data points to pandas Series"""
        if not data_points:
            return pd.Series(dtype=float)
        
        dates = [dp.date for dp in data_points]
        values = [getattr(dp, value_col, dp.value) for dp in data_points]
        
        series = pd.Series(values, index=pd.DatetimeIndex(dates))
        return series.sort_index()

# Create global instance for factor engine use
real_data_client = RealDataClient()

# Economic indicators mapping for quick access
ECONOMIC_INDICATORS = {
    # Interest Rates and Monetary Policy
    'FED_FUNDS_RATE': 'FEDFUNDS',
    'TREASURY_10Y': 'GS10', 
    'TREASURY_2Y': 'GS2',
    'REAL_INTEREST_RATE': 'REAINTRATREARAT10Y',
    
    # GDP and Growth
    'GDP_GROWTH': 'GDPC1',
    'GDP_REAL': 'GDPC1',
    'GDP_NOMINAL': 'GDP',
    'GDP_PER_CAPITA': 'A939RX0Q048SBEA',
    
    # Inflation
    'CPI_ALL_ITEMS': 'CPIAUCSL',
    'CORE_CPI': 'CPILFESL', 
    'PCE_PRICE_INDEX': 'PCEPI',
    'CORE_PCE': 'PCEPILFE',
    
    # Labor Market
    'UNEMPLOYMENT_RATE': 'UNRATE',
    'LABOR_FORCE_PARTICIPATION': 'CIVPART',
    'NONFARM_PAYROLLS': 'PAYEMS',
    'INITIAL_CLAIMS': 'ICSA',
    
    # Consumer and Business
    'CONSUMER_SENTIMENT': 'UMCSENT',
    'RETAIL_SALES': 'RSAFS',
    'INDUSTRIAL_PRODUCTION': 'INDPRO',
    'CAPACITY_UTILIZATION': 'TCU',
    
    # Housing
    'HOUSING_STARTS': 'HOUST',
    'EXISTING_HOME_SALES': 'EXHOSLUSM495S',
    'CASE_SHILLER_INDEX': 'CSUSHPISA',
    
    # Money Supply
    'M1_MONEY_SUPPLY': 'M1SL',
    'M2_MONEY_SUPPLY': 'M2SL',
    
    # Financial Markets
    'VIX': 'VIXCLS',
    'TERM_SPREAD': 'T10Y2Y',
    'CREDIT_SPREAD': 'BAA10Y'
}

# Fundamental factors mapping
FUNDAMENTAL_INDICATORS = {
    # Income Statement
    'NET_INCOME': ['us-gaap:NetIncomeLoss'],
    'REVENUE': ['us-gaap:Revenues', 'us-gaap:RevenueFromContractWithCustomerExcludingAssessedTax'],
    'GROSS_PROFIT': ['us-gaap:GrossProfit'],
    'OPERATING_INCOME': ['us-gaap:OperatingIncomeLoss'],
    'EBITDA': ['us-gaap:IncomeLossFromContinuingOperationsBeforeIncomeTaxesMinorityInterestAndIncomeLossFromEquityMethodInvestments'],
    
    # Balance Sheet  
    'TOTAL_ASSETS': ['us-gaap:Assets'],
    'TOTAL_DEBT': ['us-gaap:LongTermDebtAndCapitalLeaseObligations', 'us-gaap:ShortTermBorrowings'],
    'SHAREHOLDERS_EQUITY': ['us-gaap:StockholdersEquity'],
    'CASH_AND_EQUIVALENTS': ['us-gaap:CashAndCashEquivalentsAtCarryingValue'],
    'CURRENT_ASSETS': ['us-gaap:AssetsCurrent'],
    'CURRENT_LIABILITIES': ['us-gaap:LiabilitiesCurrent'],
    
    # Per Share Data
    'EARNINGS_PER_SHARE': ['us-gaap:EarningsPerShareBasic', 'us-gaap:EarningsPerShareDiluted'],
    'BOOK_VALUE_PER_SHARE': ['us-gaap:BookValuePerShare'],
    'SHARES_OUTSTANDING': ['us-gaap:CommonStockSharesOutstanding']
}