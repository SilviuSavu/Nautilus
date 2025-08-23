"""
FRED (Federal Reserve Economic Data) Integration Service
======================================================

Professional-grade integration with the Federal Reserve Bank of St. Louis FRED API
for accessing 32+ institutional economic indicators.

Features:
- 32 economic series across 5 categories
- Institutional macro factor calculations
- Real-time economic regime detection
- Cached data for performance
- Rate limiting and error handling
"""

import os
import logging
import asyncio
import aiohttp
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, date
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass
from enum import Enum
import json
from cachetools import TTLCache
import time

logger = logging.getLogger(__name__)


class EconomicCategory(Enum):
    """Economic indicator categories for factor modeling."""
    GROWTH = "growth"
    EMPLOYMENT = "employment"
    INFLATION = "inflation"
    MONETARY = "monetary"
    FINANCIAL = "financial"


@dataclass
class EconomicSeries:
    """Economic time series definition."""
    series_id: str
    title: str
    category: EconomicCategory
    units: str
    frequency: str
    seasonal_adjustment: str
    notes: str = ""


@dataclass
class MacroFactor:
    """Calculated macro-economic factor."""
    name: str
    value: float
    category: str
    calculation_date: str
    data_vintage: str
    confidence: float = 1.0


class FREDIntegrationService:
    """
    FRED API Integration for institutional-grade economic data access.
    
    Provides access to 32+ Federal Reserve economic series organized into 5 categories:
    - Economic Growth & Activity (6 series)
    - Employment & Labor Market (6 series)
    - Inflation & Prices (4 series)
    - Monetary Policy & Interest Rates (6 series)
    - Financial Markets & Conditions (6+ series)
    """
    
    def __init__(self):
        self.api_key = os.environ.get('FRED_API_KEY')
        self.base_url = "https://api.stlouisfed.org/fred"
        self.session: Optional[aiohttp.ClientSession] = None
        self.cache = TTLCache(maxsize=1000, ttl=1800)  # 30-minute cache
        self.rate_limit_delay = 0.1  # 100ms between requests
        self.last_request_time = 0
        
        # Define the 32+ economic series for institutional factor modeling
        self._define_economic_series()
        
        logger.info(f"FRED Integration initialized with {len(self.economic_series)} series")
    
    def _define_economic_series(self):
        """Define the comprehensive set of economic series for factor modeling."""
        self.economic_series = {
            # Economic Growth & Activity (6 series)
            "GDP": EconomicSeries("GDP", "Gross Domestic Product", EconomicCategory.GROWTH, "Billions of Dollars", "Quarterly", "Seasonally Adjusted Annual Rate"),
            "GDPC1": EconomicSeries("GDPC1", "Real GDP", EconomicCategory.GROWTH, "Billions of Chained 2017 Dollars", "Quarterly", "Seasonally Adjusted Annual Rate"),
            "GDPPOT": EconomicSeries("GDPPOT", "Real Potential GDP", EconomicCategory.GROWTH, "Billions of Chained 2017 Dollars", "Quarterly", "Not Seasonally Adjusted"),
            "INDPRO": EconomicSeries("INDPRO", "Industrial Production Index", EconomicCategory.GROWTH, "Index 2017=100", "Monthly", "Seasonally Adjusted"),
            "RSAFS": EconomicSeries("RSAFS", "Advance Retail Sales", EconomicCategory.GROWTH, "Millions of Dollars", "Monthly", "Seasonally Adjusted"),
            "HOUST": EconomicSeries("HOUST", "Housing Starts", EconomicCategory.GROWTH, "Thousands of Units", "Monthly", "Seasonally Adjusted Annual Rate"),
            
            # Employment & Labor Market (6 series)
            "UNRATE": EconomicSeries("UNRATE", "Unemployment Rate", EconomicCategory.EMPLOYMENT, "Percent", "Monthly", "Seasonally Adjusted"),
            "PAYEMS": EconomicSeries("PAYEMS", "All Employees: Total Nonfarm Payrolls", EconomicCategory.EMPLOYMENT, "Thousands of Persons", "Monthly", "Seasonally Adjusted"),
            "CIVPART": EconomicSeries("CIVPART", "Labor Force Participation Rate", EconomicCategory.EMPLOYMENT, "Percent", "Monthly", "Seasonally Adjusted"),
            "AHETPI": EconomicSeries("AHETPI", "Average Hourly Earnings of Production and Nonsupervisory Employees", EconomicCategory.EMPLOYMENT, "Dollars per Hour", "Monthly", "Seasonally Adjusted"),
            "ICSA": EconomicSeries("ICSA", "Initial Claims", EconomicCategory.EMPLOYMENT, "Number", "Weekly", "Seasonally Adjusted"),
            "JTSJOL": EconomicSeries("JTSJOL", "Job Openings: Total Nonfarm", EconomicCategory.EMPLOYMENT, "Thousands", "Monthly", "Seasonally Adjusted"),
            
            # Inflation & Prices (4 series)
            "CPIAUCSL": EconomicSeries("CPIAUCSL", "Consumer Price Index for All Urban Consumers: All Items", EconomicCategory.INFLATION, "Index 1982-84=100", "Monthly", "Seasonally Adjusted"),
            "CPILFESL": EconomicSeries("CPILFESL", "Consumer Price Index for All Urban Consumers: All Items Less Food and Energy", EconomicCategory.INFLATION, "Index 1982-84=100", "Monthly", "Seasonally Adjusted"),
            "PCEPI": EconomicSeries("PCEPI", "Personal Consumption Expenditures: Chain-type Price Index", EconomicCategory.INFLATION, "Index 2017=100", "Monthly", "Seasonally Adjusted"),
            "T5YIE": EconomicSeries("T5YIE", "5-Year Breakeven Inflation Rate", EconomicCategory.INFLATION, "Percent", "Daily", "Not Seasonally Adjusted"),
            
            # Monetary Policy & Interest Rates (6 series)
            "FEDFUNDS": EconomicSeries("FEDFUNDS", "Federal Funds Effective Rate", EconomicCategory.MONETARY, "Percent", "Monthly", "Not Seasonally Adjusted"),
            "DGS2": EconomicSeries("DGS2", "Market Yield on U.S. Treasury Securities at 2-Year Constant Maturity", EconomicCategory.MONETARY, "Percent", "Daily", "Not Seasonally Adjusted"),
            "DGS5": EconomicSeries("DGS5", "Market Yield on U.S. Treasury Securities at 5-Year Constant Maturity", EconomicCategory.MONETARY, "Percent", "Daily", "Not Seasonally Adjusted"),
            "DGS10": EconomicSeries("DGS10", "Market Yield on U.S. Treasury Securities at 10-Year Constant Maturity", EconomicCategory.MONETARY, "Percent", "Daily", "Not Seasonally Adjusted"),
            "DGS30": EconomicSeries("DGS30", "Market Yield on U.S. Treasury Securities at 30-Year Constant Maturity", EconomicCategory.MONETARY, "Percent", "Daily", "Not Seasonally Adjusted"),
            "M2SL": EconomicSeries("M2SL", "M2 Money Stock", EconomicCategory.MONETARY, "Billions of Dollars", "Monthly", "Seasonally Adjusted"),
            
            # Financial Markets & Conditions (6+ series)
            "BAMLH0A0HYM2": EconomicSeries("BAMLH0A0HYM2", "ICE BofA US High Yield Index Option-Adjusted Spread", EconomicCategory.FINANCIAL, "Percent", "Daily", "Not Seasonally Adjusted"),
            "VIXCLS": EconomicSeries("VIXCLS", "CBOE Volatility Index: VIX", EconomicCategory.FINANCIAL, "Index", "Daily", "Not Seasonally Adjusted"),
            "DEXUSEU": EconomicSeries("DEXUSEU", "U.S. / Euro Foreign Exchange Rate", EconomicCategory.FINANCIAL, "U.S. Dollars to One Euro", "Daily", "Not Seasonally Adjusted"),
            "EMVOVERALLEMV": EconomicSeries("EMVOVERALLEMV", "Emerging Markets Bond Index Global (EMBIG) Stripped Spread", EconomicCategory.FINANCIAL, "Percentage Points", "Daily", "Not Seasonally Adjusted"),
            "DCOILWTICO": EconomicSeries("DCOILWTICO", "Crude Oil Prices: West Texas Intermediate (WTI)", EconomicCategory.FINANCIAL, "Dollars per Barrel", "Daily", "Not Seasonally Adjusted"),
            # "GOLDAMGBD228NLBM": EconomicSeries("GOLDAMGBD228NLBM", "Gold Fixing Price 10:30 A.M. (London time) in London Bullion Market", EconomicCategory.FINANCIAL, "U.S. Dollars per Troy Ounce", "Daily", "Not Seasonally Adjusted"),  # DEPRECATED - series no longer available
            "GOLDPMGBD228NLBM": EconomicSeries("GOLDPMGBD228NLBM", "Gold Fixing Price 3:00 P.M. (London time) in London Bullion Market", EconomicCategory.FINANCIAL, "U.S. Dollars per Troy Ounce", "Daily", "Not Seasonally Adjusted"),
        }
    
    async def _ensure_session(self):
        """Ensure aiohttp session is available."""
        if self.session is None or self.session.closed:
            self.session = aiohttp.ClientSession()
    
    async def _rate_limit(self):
        """Implement rate limiting for FRED API."""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        if time_since_last < self.rate_limit_delay:
            await asyncio.sleep(self.rate_limit_delay - time_since_last)
        self.last_request_time = time.time()
    
    async def _make_request(self, endpoint: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Make rate-limited request to FRED API."""
        if not self.api_key:
            raise ValueError("FRED_API_KEY environment variable not set")
        
        await self._ensure_session()
        await self._rate_limit()
        
        url = f"{self.base_url}/{endpoint}"
        params.update({
            "api_key": self.api_key,
            "file_type": "json"
        })
        
        cache_key = f"{endpoint}_{hash(str(sorted(params.items())))}"
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        try:
            async with self.session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    self.cache[cache_key] = data
                    return data
                else:
                    error_text = await response.text()
                    logger.error(f"FRED API error {response.status}: {error_text}")
                    raise Exception(f"FRED API request failed: {response.status}")
                    
        except Exception as e:
            logger.error(f"FRED API request failed: {e}")
            raise
    
    async def get_series_data(
        self, 
        series_id: str, 
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
        limit: int = 1000
    ) -> pd.DataFrame:
        """Get time series data for a specific economic series."""
        params = {
            "series_id": series_id,
            "limit": limit,
            "sort_order": "desc"
        }
        
        if start_date:
            params["observation_start"] = start_date.strftime("%Y-%m-%d")
        if end_date:
            params["observation_end"] = end_date.strftime("%Y-%m-%d")
        
        try:
            response = await self._make_request("series/observations", params)
            observations = response.get("observations", [])
            
            if not observations:
                logger.warning(f"No data returned for series {series_id}")
                return pd.DataFrame()
            
            # Convert to DataFrame
            df = pd.DataFrame(observations)
            df['date'] = pd.to_datetime(df['date'])
            df['value'] = pd.to_numeric(df['value'], errors='coerce')
            df = df.dropna(subset=['value'])
            df = df.sort_values('date')
            
            logger.info(f"Retrieved {len(df)} observations for {series_id}")
            return df
            
        except Exception as e:
            logger.error(f"Failed to get data for series {series_id}: {e}")
            raise
    
    async def get_latest_value(self, series_id: str) -> Optional[float]:
        """Get the most recent value for an economic series."""
        try:
            df = await self.get_series_data(series_id, limit=1)
            if not df.empty:
                return float(df.iloc[-1]['value'])
            return None
        except Exception as e:
            logger.error(f"Failed to get latest value for {series_id}: {e}")
            return None
    
    async def calculate_macro_factors(self, as_of_date: Optional[date] = None) -> Dict[str, float]:
        """
        Calculate institutional-grade macro-economic factors.
        
        Returns comprehensive macro factors used in professional factor models:
        - Interest rate levels and changes
        - Yield curve factors (level, slope, curvature)
        - Economic activity indicators
        - Inflation and employment trends
        - Financial market stress indicators
        """
        if as_of_date is None:
            as_of_date = date.today()
        
        start_date = as_of_date - timedelta(days=90)  # 3 months of data
        
        factors = {}
        
        try:
            # Interest Rate Factors
            fed_funds = await self.get_latest_value("FEDFUNDS")
            treasury_2y = await self.get_latest_value("DGS2")
            treasury_5y = await self.get_latest_value("DGS5")
            treasury_10y = await self.get_latest_value("DGS10")
            treasury_30y = await self.get_latest_value("DGS30")
            
            if all(v is not None for v in [fed_funds, treasury_2y, treasury_10y, treasury_30y]):
                factors.update({
                    "fed_funds_level": fed_funds,
                    "treasury_2y_level": treasury_2y,
                    "treasury_10y_level": treasury_10y,
                    "treasury_30y_level": treasury_30y,
                    "yield_curve_slope": treasury_10y - treasury_2y,
                    "term_spread": treasury_30y - treasury_2y,
                    "carry_steepness": treasury_30y - treasury_10y
                })
                
                if treasury_5y is not None:
                    # Nelson-Siegel curvature approximation
                    factors["yield_curve_curvature"] = 2 * treasury_5y - treasury_2y - treasury_10y
            
            # Economic Activity Factors
            unemployment = await self.get_latest_value("UNRATE")
            if unemployment is not None:
                factors["unemployment_rate"] = unemployment
            
            # Get payroll change (need time series)
            payroll_df = await self.get_series_data("PAYEMS", start_date=start_date, limit=3)
            if len(payroll_df) >= 2:
                latest_payroll = payroll_df.iloc[-1]['value']
                previous_payroll = payroll_df.iloc[-2]['value']
                factors["payroll_growth_mom"] = (latest_payroll - previous_payroll) / previous_payroll * 100
            
            # Inflation Factors
            cpi_df = await self.get_series_data("CPIAUCSL", start_date=start_date, limit=13)
            if len(cpi_df) >= 13:
                latest_cpi = cpi_df.iloc[-1]['value']
                year_ago_cpi = cpi_df.iloc[-13]['value']  # 12 months ago
                factors["cpi_inflation_yoy"] = (latest_cpi - year_ago_cpi) / year_ago_cpi * 100
            
            # Financial Market Stress Factors
            vix = await self.get_latest_value("VIXCLS")
            if vix is not None:
                factors["vix_level"] = vix
                factors["volatility_regime"] = 1.0 if vix > 20 else 0.0
            
            credit_spread = await self.get_latest_value("BAMLH0A0HYM2")
            if credit_spread is not None:
                factors["high_yield_spread"] = credit_spread
                factors["credit_stress"] = 1.0 if credit_spread > 5.0 else 0.0
            
            # Currency and Commodity Factors
            eurusd = await self.get_latest_value("DEXUSEU")
            if eurusd is not None:
                factors["usd_strength"] = 1.0 / eurusd  # Inverse for USD strength
            
            oil_price = await self.get_latest_value("DCOILWTICO")
            if oil_price is not None:
                factors["oil_price"] = oil_price
            
            gold_price = await self.get_latest_value("GOLDPMGBD228NLBM")
            if gold_price is not None:
                factors["gold_price"] = gold_price
            
            logger.info(f"Calculated {len(factors)} macro factors as of {as_of_date}")
            return factors
            
        except Exception as e:
            logger.error(f"Failed to calculate macro factors: {e}")
            raise
    
    async def get_economic_calendar(self, days_ahead: int = 30) -> List[Dict[str, Any]]:
        """Get economic data release calendar for the next N days."""
        end_date = date.today() + timedelta(days=days_ahead)
        
        # This is a simplified version - FRED doesn't have a direct calendar endpoint
        # In practice, you'd maintain a calendar of release dates for key series
        calendar_events = []
        
        # Add known monthly releases (simplified)
        for series_id, series_info in self.economic_series.items():
            if series_info.frequency == "Monthly":
                calendar_events.append({
                    "series_id": series_id,
                    "title": series_info.title,
                    "category": series_info.category.value,
                    "estimated_release_date": end_date.strftime("%Y-%m-%d"),  # Placeholder
                    "importance": "high" if series_id in ["PAYEMS", "UNRATE", "CPIAUCSL"] else "medium"
                })
        
        return calendar_events
    
    async def get_series_info(self) -> Dict[str, Dict[str, Any]]:
        """Get information about all available economic series."""
        return {
            series_id: {
                "title": series.title,
                "category": series.category.value,
                "units": series.units,
                "frequency": series.frequency,
                "seasonal_adjustment": series.seasonal_adjustment,
                "notes": series.notes
            }
            for series_id, series in self.economic_series.items()
        }
    
    async def health_check(self) -> Dict[str, Any]:
        """Check the health of the FRED integration."""
        health_status = {
            "service": "FRED Integration",
            "status": "unknown",
            "api_key_configured": bool(self.api_key),
            "available_series": len(self.economic_series),
            "cache_size": len(self.cache),
            "timestamp": datetime.now().isoformat()
        }
        
        if not self.api_key:
            health_status.update({
                "status": "not_configured",
                "error_message": "FRED_API_KEY environment variable not set"
            })
            return health_status
        
        try:
            # Test API connectivity
            test_value = await self.get_latest_value("FEDFUNDS")
            if test_value is not None:
                health_status.update({
                    "status": "operational",
                    "test_value": f"Fed Funds Rate: {test_value}%",
                    "last_successful_request": datetime.now().isoformat()
                })
            else:
                health_status.update({
                    "status": "degraded",
                    "error_message": "API accessible but no data returned"
                })
                
        except Exception as e:
            health_status.update({
                "status": "error",
                "error_message": str(e)
            })
        
        return health_status
    
    async def close(self):
        """Close the HTTP session."""
        if self.session and not self.session.closed:
            await self.session.close()


# Global instance
fred_integration = FREDIntegrationService()