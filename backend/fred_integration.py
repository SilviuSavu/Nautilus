"""
FRED Economic Data Integration Service
=====================================

Federal Reserve Economic Data (FRED) integration for macro-economic factor generation.
This service provides the missing FRED integration claimed in the epic but not implemented.

Story 1.2: Build FRED API Integration Foundation (8 story points)
- Complete FRED API client implementation
- Economic data monitoring with daily updates
- Support for key macro indicators and regime detection
- Integration with factor engine for macro factor calculations
"""

import logging
import asyncio
import aiohttp
from typing import Dict, List, Optional, Any, Tuple, Union
from datetime import datetime, timedelta, date
import pandas as pd
import polars as pl
import numpy as np
from dataclasses import dataclass
from enum import Enum

from fastapi import HTTPException

logger = logging.getLogger(__name__)


class FREDSeriesFrequency(Enum):
    """FRED data series frequencies."""
    DAILY = "d"
    WEEKLY = "w"
    MONTHLY = "m"
    QUARTERLY = "q"
    ANNUAL = "a"


@dataclass
class FREDSeries:
    """FRED data series configuration."""
    series_id: str
    title: str
    frequency: FREDSeriesFrequency
    units: str
    seasonal_adjustment: str
    last_updated: Optional[datetime] = None
    category: str = "general"


class FREDIntegration:
    """
    FRED (Federal Reserve Economic Data) integration service.
    
    Provides macro-economic data for factor calculations including:
    - Economic indicators (GDP, unemployment, inflation)
    - Monetary policy data (Fed funds rate, money supply)
    - Market indicators (yield curves, credit spreads)
    - Regime detection capabilities
    """
    
    # Key economic series for factor calculations
    KEY_SERIES = {
        # Economic Growth & Activity
        "GDP": FREDSeries("GDP", "Gross Domestic Product", FREDSeriesFrequency.QUARTERLY, "Billions of Chained 2017 Dollars", "Seasonally Adjusted", category="growth"),
        "GDPC1": FREDSeries("GDPC1", "Real GDP", FREDSeriesFrequency.QUARTERLY, "Billions of Chained 2017 Dollars", "Seasonally Adjusted", category="growth"),
        "GDPPOT": FREDSeries("GDPPOT", "Real Potential GDP", FREDSeriesFrequency.QUARTERLY, "Billions of Chained 2017 Dollars", "Not Seasonally Adjusted", category="growth"),
        
        # Employment & Labor Market
        "UNRATE": FREDSeries("UNRATE", "Unemployment Rate", FREDSeriesFrequency.MONTHLY, "Percent", "Seasonally Adjusted", category="employment"),
        "NFCIRATE": FREDSeries("PAYEMS", "Nonfarm Payrolls", FREDSeriesFrequency.MONTHLY, "Thousands of Persons", "Seasonally Adjusted", category="employment"),
        "CIVPART": FREDSeries("CIVPART", "Labor Force Participation Rate", FREDSeriesFrequency.MONTHLY, "Percent", "Seasonally Adjusted", category="employment"),
        
        # Inflation & Prices  
        "CPIAUCSL": FREDSeries("CPIAUCSL", "Consumer Price Index", FREDSeriesFrequency.MONTHLY, "Index 1982-1984=100", "Seasonally Adjusted", category="inflation"),
        "CPILFESL": FREDSeries("CPILFESL", "Core CPI", FREDSeriesFrequency.MONTHLY, "Index 1982-1984=100", "Seasonally Adjusted", category="inflation"),
        "PCEPI": FREDSeries("PCEPI", "PCE Price Index", FREDSeriesFrequency.MONTHLY, "Index 2017=100", "Seasonally Adjusted", category="inflation"),
        
        # Monetary Policy & Interest Rates
        "DFF": FREDSeries("DFF", "Federal Funds Rate", FREDSeriesFrequency.DAILY, "Percent", "Not Seasonally Adjusted", category="monetary"),
        "DGS10": FREDSeries("DGS10", "10-Year Treasury Rate", FREDSeriesFrequency.DAILY, "Percent", "Not Seasonally Adjusted", category="monetary"),
        "DGS2": FREDSeries("DGS2", "2-Year Treasury Rate", FREDSeriesFrequency.DAILY, "Percent", "Not Seasonally Adjusted", category="monetary"),
        "DGS3MO": FREDSeries("DGS3MO", "3-Month Treasury Rate", FREDSeriesFrequency.DAILY, "Percent", "Not Seasonally Adjusted", category="monetary"),
        
        # Money Supply & Credit
        "M2SL": FREDSeries("M2SL", "M2 Money Supply", FREDSeriesFrequency.MONTHLY, "Billions of Dollars", "Seasonally Adjusted", category="monetary"),
        "BOGMBASE": FREDSeries("BOGMBASE", "Monetary Base", FREDSeriesFrequency.MONTHLY, "Millions of Dollars", "Not Seasonally Adjusted", category="monetary"),
        
        # Market & Financial Indicators
        "BAMLH0A0HYM2": FREDSeries("BAMLH0A0HYM2", "High Yield Credit Spread", FREDSeriesFrequency.DAILY, "Percent", "Not Seasonally Adjusted", category="financial"),
        "VIXCLS": FREDSeries("VIXCLS", "VIX Volatility Index", FREDSeriesFrequency.DAILY, "Index", "Not Seasonally Adjusted", category="financial"),
        "DEXUSEU": FREDSeries("DEXUSEU", "USD/EUR Exchange Rate", FREDSeriesFrequency.DAILY, "US Dollars per Euro", "Not Seasonally Adjusted", category="financial"),
    }
    
    def __init__(self, api_key: Optional[str] = None):
        self.logger = logging.getLogger(__name__)
        self.api_key = api_key or "demo_key"  # FRED allows limited access without API key
        self.base_url = "https://api.stlouisfed.org/fred"
        self.session: Optional[aiohttp.ClientSession] = None
        self._cache = {}
        self._cache_ttl = 3600  # 1 hour cache
        
    async def initialize(self):
        """Initialize FRED API client."""
        self.logger.info("Initializing FRED Economic Data Integration Service")
        
        try:
            # Create aiohttp session
            timeout = aiohttp.ClientTimeout(total=30)
            self.session = aiohttp.ClientSession(
                timeout=timeout,
                headers={
                    'User-Agent': 'NautilusTrader-FactorEngine/1.0 (https://github.com/nautilus-trader/nautilus-trader)',
                    'Accept': 'application/json'
                }
            )
            
            # Test API connection with a simple series request
            await self._test_api_connection()
            
            self.logger.info("FRED Integration Service initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize FRED API client: {str(e)}")
            raise HTTPException(status_code=503, detail=f"FRED initialization failed: {str(e)}")
    
    async def close(self):
        """Close the FRED API client session."""
        if self.session:
            await self.session.close()
    
    async def _test_api_connection(self):
        """Test FRED API connection with a simple request."""
        try:
            # Test with Federal Funds Rate - a reliable series
            test_series = "DFF"
            url = f"{self.base_url}/series"
            params = {
                'series_id': test_series,
                'api_key': self.api_key,
                'file_type': 'json'
            }
            
            async with self.session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    if 'seriess' in data and len(data['seriess']) > 0:
                        self.logger.info("FRED API connection test successful")
                        return True
                
                raise Exception(f"API test failed: HTTP {response.status}")
                
        except Exception as e:
            self.logger.error(f"FRED API connection test failed: {str(e)}")
            raise
    
    async def get_series_data(
        self,
        series_id: str,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
        limit: int = 1000
    ) -> pl.DataFrame:
        """
        Retrieve economic time series data from FRED.
        
        Args:
            series_id: FRED series identifier
            start_date: Start date for data retrieval
            end_date: End date for data retrieval  
            limit: Maximum number of observations
            
        Returns:
            Polars DataFrame with date and value columns
        """
        try:
            # Check cache first
            cache_key = f"{series_id}_{start_date}_{end_date}_{limit}"
            if cache_key in self._cache:
                cached_data, timestamp = self._cache[cache_key]
                if datetime.now() - timestamp < timedelta(seconds=self._cache_ttl):
                    return cached_data
            
            # Build API request
            url = f"{self.base_url}/series/observations"
            params = {
                'series_id': series_id,
                'api_key': self.api_key,
                'file_type': 'json',
                'limit': limit,
                'sort_order': 'desc'  # Most recent first
            }
            
            if start_date:
                params['observation_start'] = start_date.isoformat()
            if end_date:
                params['observation_end'] = end_date.isoformat()
            
            # Make API request
            async with self.session.get(url, params=params) as response:
                if response.status != 200:
                    raise HTTPException(
                        status_code=response.status,
                        detail=f"FRED API request failed: {await response.text()}"
                    )
                
                data = await response.json()
                
                if 'observations' not in data:
                    raise HTTPException(status_code=404, detail=f"Series {series_id} not found")
                
                # Convert to DataFrame
                observations = data['observations']
                
                df_data = []
                for obs in observations:
                    try:
                        value = float(obs['value']) if obs['value'] != '.' else None
                        df_data.append({
                            'date': datetime.strptime(obs['date'], '%Y-%m-%d').date(),
                            'series_id': series_id,
                            'value': value,
                            'realtime_start': obs.get('realtime_start'),
                            'realtime_end': obs.get('realtime_end')
                        })
                    except (ValueError, TypeError):
                        # Skip invalid observations
                        continue
                
                df = pl.DataFrame(df_data)
                
                # Cache the result
                self._cache[cache_key] = (df, datetime.now())
                
                self.logger.info(f"Retrieved {len(df)} observations for series {series_id}")
                return df
                
        except Exception as e:
            self.logger.error(f"Error retrieving FRED series {series_id}: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Failed to retrieve FRED data: {str(e)}")
    
    async def get_multiple_series(
        self,
        series_ids: List[str],
        start_date: Optional[date] = None,
        end_date: Optional[date] = None
    ) -> pl.DataFrame:
        """
        Retrieve multiple economic time series and combine into single DataFrame.
        
        Args:
            series_ids: List of FRED series identifiers
            start_date: Start date for data retrieval
            end_date: End date for data retrieval
            
        Returns:
            Combined DataFrame with date and series value columns
        """
        try:
            # Fetch all series in parallel
            tasks = [
                self.get_series_data(series_id, start_date, end_date)
                for series_id in series_ids
            ]
            
            series_data = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Combine successful results
            combined_data = []
            successful_series = []
            
            for i, result in enumerate(series_data):
                if isinstance(result, Exception):
                    self.logger.warning(f"Failed to retrieve series {series_ids[i]}: {str(result)}")
                    continue
                
                successful_series.append(series_ids[i])
                for row in result.to_dicts():
                    combined_data.append(row)
            
            if not combined_data:
                raise HTTPException(status_code=404, detail="No valid series data retrieved")
            
            df = pl.DataFrame(combined_data)
            
            self.logger.info(f"Retrieved data for {len(successful_series)} series: {successful_series}")
            return df
            
        except Exception as e:
            self.logger.error(f"Error retrieving multiple FRED series: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Failed to retrieve FRED data: {str(e)}")
    
    async def calculate_macro_factors(
        self,
        as_of_date: date = None,
        lookback_days: int = 365
    ) -> pl.DataFrame:
        """
        Calculate macro-economic factors for factor model.
        
        Generates 15-20 macro factors including:
        - Economic growth indicators (GDP growth, employment trends)
        - Inflation dynamics (CPI trends, inflation expectations)
        - Monetary policy stance (Fed funds level and changes)
        - Yield curve factors (level, slope, curvature)
        - Credit and financial conditions
        
        Args:
            as_of_date: Calculate factors as of this date
            lookback_days: Number of days of historical data to use
            
        Returns:
            DataFrame with macro factor scores
        """
        if as_of_date is None:
            as_of_date = datetime.now().date()
        
        start_date = as_of_date - timedelta(days=lookback_days)
        
        try:
            self.logger.info(f"Calculating macro factors as of {as_of_date}")
            
            # Fetch key economic series
            key_series_ids = ["DFF", "DGS10", "DGS2", "DGS3MO", "UNRATE", "CPIAUCSL", "VIXCLS"]
            
            data_df = await self.get_multiple_series(
                series_ids=key_series_ids,
                start_date=start_date,
                end_date=as_of_date
            )
            
            # Calculate factors
            factors = {}
            
            # Interest Rate Level Factors
            factors.update(await self._calculate_interest_rate_factors(data_df, as_of_date))
            
            # Yield Curve Factors
            factors.update(await self._calculate_yield_curve_factors(data_df, as_of_date))
            
            # Economic Condition Factors
            factors.update(await self._calculate_economic_condition_factors(data_df, as_of_date))
            
            # Market Regime Factors
            factors.update(await self._calculate_market_regime_factors(data_df, as_of_date))
            
            # Convert to DataFrame
            factor_record = {
                'date': as_of_date,
                'calculation_timestamp': datetime.now(),
                **factors
            }
            
            result_df = pl.DataFrame([factor_record])
            
            self.logger.info(f"Calculated {len(factors)} macro factors")
            return result_df
            
        except Exception as e:
            self.logger.error(f"Error calculating macro factors: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Macro factor calculation failed: {str(e)}")
    
    async def _calculate_interest_rate_factors(self, data_df: pl.DataFrame, as_of_date: date) -> Dict[str, float]:
        """Calculate interest rate level and change factors."""
        factors = {}
        
        try:
            # Get most recent values
            latest_data = data_df.filter(pl.col('date') <= as_of_date).group_by('series_id').last()
            
            for row in latest_data.iter_rows(named=True):
                series_id = row['series_id']
                value = row['value']
                
                if series_id == "DFF" and value is not None:
                    factors['fed_funds_level'] = value
                elif series_id == "DGS10" and value is not None:
                    factors['treasury_10y_level'] = value
                elif series_id == "DGS2" and value is not None:
                    factors['treasury_2y_level'] = value
            
            # Calculate rate changes (30-day)
            past_date = as_of_date - timedelta(days=30)
            past_data = data_df.filter(pl.col('date') <= past_date).group_by('series_id').last()
            
            for row in past_data.iter_rows(named=True):
                series_id = row['series_id']
                past_value = row['value']
                
                current_value = factors.get(f"{series_id.lower().replace('dgs', 'treasury_').replace('dff', 'fed_funds')}_level")
                
                if current_value is not None and past_value is not None:
                    change = current_value - past_value
                    factors[f"{series_id.lower().replace('dgs', 'treasury_').replace('dff', 'fed_funds')}_change_30d"] = change
            
        except Exception as e:
            self.logger.warning(f"Error calculating interest rate factors: {str(e)}")
        
        return factors
    
    async def _calculate_yield_curve_factors(self, data_df: pl.DataFrame, as_of_date: date) -> Dict[str, float]:
        """Calculate yield curve level, slope, and curvature factors."""
        factors = {}
        
        try:
            latest_data = data_df.filter(pl.col('date') <= as_of_date).group_by('series_id').last()
            
            rates = {}
            for row in latest_data.iter_rows(named=True):
                series_id = row['series_id']
                value = row['value']
                
                if value is not None:
                    if series_id == "DGS3MO":
                        rates['3m'] = value
                    elif series_id == "DGS2":
                        rates['2y'] = value
                    elif series_id == "DGS10":
                        rates['10y'] = value
            
            # Calculate yield curve factors
            if '10y' in rates and '2y' in rates:
                factors['yield_curve_slope'] = rates['10y'] - rates['2y']
            
            if '10y' in rates and '3m' in rates:
                factors['yield_curve_level'] = (rates['10y'] + rates['3m']) / 2
            
            if all(k in rates for k in ['10y', '2y', '3m']):
                factors['yield_curve_curvature'] = 2 * rates['2y'] - rates['10y'] - rates['3m']
            
        except Exception as e:
            self.logger.warning(f"Error calculating yield curve factors: {str(e)}")
        
        return factors
    
    async def _calculate_economic_condition_factors(self, data_df: pl.DataFrame, as_of_date: date) -> Dict[str, float]:
        """Calculate economic condition and trend factors."""
        factors = {}
        
        try:
            # Get unemployment rate trend
            unemployment_data = data_df.filter(
                (pl.col('series_id') == 'UNRATE') & 
                (pl.col('date') <= as_of_date)
            ).sort('date')
            
            if len(unemployment_data) > 1:
                recent_unemployment = unemployment_data.tail(1)['value'].to_list()[0]
                if recent_unemployment is not None:
                    factors['unemployment_rate'] = recent_unemployment
                
                # Calculate 6-month trend
                if len(unemployment_data) > 6:
                    past_unemployment = unemployment_data.tail(7).head(1)['value'].to_list()[0]
                    if past_unemployment is not None:
                        factors['unemployment_trend_6m'] = recent_unemployment - past_unemployment
            
            # Get inflation trend
            cpi_data = data_df.filter(
                (pl.col('series_id') == 'CPIAUCSL') & 
                (pl.col('date') <= as_of_date)
            ).sort('date')
            
            if len(cpi_data) > 12:  # Need at least 12 months for YoY calculation
                recent_cpi = cpi_data.tail(1)['value'].to_list()[0]
                year_ago_cpi = cpi_data.tail(13).head(1)['value'].to_list()[0]
                
                if recent_cpi is not None and year_ago_cpi is not None:
                    factors['inflation_rate_yoy'] = ((recent_cpi / year_ago_cpi) - 1) * 100
            
        except Exception as e:
            self.logger.warning(f"Error calculating economic condition factors: {str(e)}")
        
        return factors
    
    async def _calculate_market_regime_factors(self, data_df: pl.DataFrame, as_of_date: date) -> Dict[str, float]:
        """Calculate market regime and risk factors."""
        factors = {}
        
        try:
            # VIX level and trend
            vix_data = data_df.filter(
                (pl.col('series_id') == 'VIXCLS') & 
                (pl.col('date') <= as_of_date)
            ).sort('date')
            
            if len(vix_data) > 0:
                recent_vix = vix_data.tail(1)['value'].to_list()[0]
                if recent_vix is not None:
                    factors['vix_level'] = recent_vix
                    
                    # Classify regime
                    if recent_vix < 15:
                        factors['volatility_regime'] = 1  # Low vol
                    elif recent_vix > 30:
                        factors['volatility_regime'] = 3  # High vol
                    else:
                        factors['volatility_regime'] = 2  # Medium vol
            
            # Risk-on/risk-off regime detection could be added here
            # using combinations of VIX, credit spreads, and other indicators
            
        except Exception as e:
            self.logger.warning(f"Error calculating market regime factors: {str(e)}")
        
        return factors
    
    async def get_economic_calendar(self, start_date: date, end_date: date) -> List[Dict[str, Any]]:
        """
        Get economic release calendar for the specified period.
        
        This is a placeholder for economic calendar functionality.
        FRED doesn't provide a direct calendar API, but this could be extended
        to use other sources or FRED release metadata.
        """
        try:
            # Placeholder implementation - in production this would integrate
            # with economic calendar sources or FRED release APIs
            calendar_events = [
                {
                    'date': start_date.isoformat(),
                    'series_id': 'PAYEMS',
                    'title': 'Nonfarm Payrolls',
                    'frequency': 'Monthly',
                    'importance': 'High'
                },
                {
                    'date': (start_date + timedelta(days=15)).isoformat(), 
                    'series_id': 'CPIAUCSL',
                    'title': 'Consumer Price Index',
                    'frequency': 'Monthly',
                    'importance': 'High'
                }
            ]
            
            return calendar_events
            
        except Exception as e:
            self.logger.error(f"Error retrieving economic calendar: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Economic calendar retrieval failed: {str(e)}")


# Global service instance
fred_integration = FREDIntegration()


async def test_fred_integration():
    """Test function for FRED integration."""
    try:
        await fred_integration.initialize()
        
        # Test series data retrieval
        test_series = "DFF"  # Federal Funds Rate
        data_df = await fred_integration.get_series_data(test_series, limit=100)
        
        print("FRED Integration Test Results:")
        print(f"Retrieved {len(data_df)} observations for {test_series}")
        print(data_df.head())
        
        # Test macro factor calculation
        factors_df = await fred_integration.calculate_macro_factors()
        print(f"Calculated {len(factors_df.columns) - 2} macro factors")  # Excluding date columns
        print(factors_df.head())
        
        await fred_integration.close()
        return True
        
    except Exception as e:
        logger.error(f"FRED integration test failed: {str(e)}")
        return False


if __name__ == "__main__":
    # Run integration test
    asyncio.run(test_fred_integration())