"""
Alpha Vantage Service
====================

High-level service interface for Alpha Vantage market data integration.
"""

import logging
from typing import Dict, List, Optional, Any
from datetime import datetime, date
from .config import AlphaVantageConfig, alpha_vantage_config
from .http_client import AlphaVantageHTTPClient
from .models import (
    AlphaVantageQuote, AlphaVantageTimeSeries, AlphaVantageBarData,
    AlphaVantageCompany, AlphaVantageSearchResult, AlphaVantageEarnings,
    AlphaVantageHealthStatus
)

logger = logging.getLogger(__name__)


class AlphaVantageService:
    """
    Alpha Vantage market data service.
    
    Provides comprehensive access to Alpha Vantage API including:
    - Real-time stock quotes
    - Historical time series data (intraday, daily, weekly, monthly)
    - Company fundamental data and overview
    - Symbol search functionality
    - Earnings data
    """
    
    def __init__(self, config: Optional[AlphaVantageConfig] = None):
        self.config = config or alpha_vantage_config
        self.http_client = AlphaVantageHTTPClient(self.config)
        logger.info("Alpha Vantage service initialized")
    
    async def get_quote(self, symbol: str) -> AlphaVantageQuote:
        """Get real-time quote for a stock symbol."""
        try:
            response = await self.http_client.get_quote(symbol)
            
            # Parse Alpha Vantage quote response
            quote_data = response.get('Global Quote', {})
            if not quote_data:
                raise ValueError(f"No quote data returned for symbol {symbol}")
            
            # Extract values from Alpha Vantage response format
            current_price = float(quote_data.get('05. price', 0))
            change = float(quote_data.get('09. change', 0))
            change_percent = float(quote_data.get('10. change percent', '0').replace('%', ''))
            volume = int(quote_data.get('06. volume', 0))
            previous_close = float(quote_data.get('08. previous close', 0))
            open_price = float(quote_data.get('02. open', 0))
            high = float(quote_data.get('03. high', 0))
            low = float(quote_data.get('04. low', 0))
            
            return AlphaVantageQuote(
                symbol=symbol.upper(),
                price=current_price,
                change=change,
                change_percent=change_percent,
                volume=volume,
                previous_close=previous_close,
                open_price=open_price,
                high=high,
                low=low,
                timestamp=datetime.now()
            )
            
        except Exception as e:
            logger.error(f"Failed to get quote for {symbol}: {e}")
            raise
    
    async def get_multiple_quotes(self, symbols: List[str]) -> Dict[str, AlphaVantageQuote]:
        """Get quotes for multiple symbols."""
        quotes = {}
        for symbol in symbols:
            try:
                quote = await self.get_quote(symbol)
                quotes[symbol.upper()] = quote
            except Exception as e:
                logger.warning(f"Failed to get quote for {symbol}: {e}")
                # Continue with other symbols
        return quotes
    
    async def get_intraday_data(
        self, 
        symbol: str, 
        interval: str = '5min',
        extended_hours: bool = True
    ) -> AlphaVantageTimeSeries:
        """Get intraday time series data."""
        try:
            response = await self.http_client.get_intraday(symbol, interval, extended_hours)
            
            # Parse time series data
            time_series_key = f'Time Series ({interval})'
            time_series = response.get(time_series_key, {})
            
            if not time_series:
                raise ValueError(f"No intraday data returned for {symbol}")
            
            bars = []
            for timestamp_str, bar_data in time_series.items():
                timestamp = datetime.strptime(timestamp_str, '%Y-%m-%d %H:%M:%S')
                bar = AlphaVantageBarData(
                    timestamp=timestamp,
                    open=float(bar_data['1. open']),
                    high=float(bar_data['2. high']),
                    low=float(bar_data['3. low']),
                    close=float(bar_data['4. close']),
                    volume=int(bar_data['5. volume'])
                )
                bars.append(bar)
            
            # Sort by timestamp (most recent first)
            bars.sort(key=lambda x: x.timestamp, reverse=True)
            
            # Get metadata
            metadata = response.get('Meta Data', {})
            last_refreshed = datetime.strptime(
                metadata.get('3. Last Refreshed', ''),
                '%Y-%m-%d %H:%M:%S'
            )
            
            return AlphaVantageTimeSeries(
                symbol=symbol.upper(),
                interval=interval,
                data=bars,
                last_refreshed=last_refreshed,
                time_zone=metadata.get('6. Time Zone', 'US/Eastern')
            )
            
        except Exception as e:
            logger.error(f"Failed to get intraday data for {symbol}: {e}")
            raise
    
    async def get_daily_data(self, symbol: str, outputsize: str = 'compact') -> AlphaVantageTimeSeries:
        """Get daily time series data."""
        try:
            response = await self.http_client.get_daily(symbol, outputsize)
            
            # Parse time series data
            time_series = response.get('Time Series (Daily)', {})
            
            if not time_series:
                raise ValueError(f"No daily data returned for {symbol}")
            
            bars = []
            for date_str, bar_data in time_series.items():
                timestamp = datetime.strptime(date_str, '%Y-%m-%d')
                bar = AlphaVantageBarData(
                    timestamp=timestamp,
                    open=float(bar_data['1. open']),
                    high=float(bar_data['2. high']),
                    low=float(bar_data['3. low']),
                    close=float(bar_data['4. close']),
                    volume=int(bar_data['6. volume'])  # Adjusted volume for daily adjusted
                )
                bars.append(bar)
            
            # Sort by timestamp (most recent first)
            bars.sort(key=lambda x: x.timestamp, reverse=True)
            
            # Get metadata
            metadata = response.get('Meta Data', {})
            last_refreshed = datetime.strptime(
                metadata.get('3. Last Refreshed', ''),
                '%Y-%m-%d'
            )
            
            return AlphaVantageTimeSeries(
                symbol=symbol.upper(),
                interval='daily',
                data=bars,
                last_refreshed=last_refreshed,
                time_zone=metadata.get('5. Time Zone', 'US/Eastern')
            )
            
        except Exception as e:
            logger.error(f"Failed to get daily data for {symbol}: {e}")
            raise
    
    async def search_symbols(self, keywords: str, max_results: int = 10) -> List[AlphaVantageSearchResult]:
        """Search for symbols by keywords."""
        try:
            response = await self.http_client.search_symbols(keywords)
            
            best_matches = response.get('bestMatches', [])
            if not best_matches:
                return []
            
            results = []
            for match in best_matches[:max_results]:
                result = AlphaVantageSearchResult(
                    symbol=match.get('1. symbol', ''),
                    name=match.get('2. name', ''),
                    type=match.get('3. type', ''),
                    region=match.get('4. region', ''),
                    market_open=match.get('5. marketOpen', ''),
                    market_close=match.get('6. marketClose', ''),
                    timezone=match.get('7. timezone', ''),
                    currency=match.get('8. currency', ''),
                    match_score=float(match.get('9. matchScore', 0))
                )
                results.append(result)
            
            # Sort by match score (highest first)
            results.sort(key=lambda x: x.match_score, reverse=True)
            return results
            
        except Exception as e:
            logger.error(f"Failed to search symbols for '{keywords}': {e}")
            raise
    
    async def get_company_fundamentals(self, symbol: str) -> Optional[AlphaVantageCompany]:
        """Get company fundamental data and overview."""
        try:
            response = await self.http_client.get_company_overview(symbol)
            
            if not response or 'Symbol' not in response:
                return None
            
            # Helper function to safely convert to float
            def safe_float(value: str) -> Optional[float]:
                if value == 'None' or value == '-' or not value:
                    return None
                try:
                    return float(value)
                except (ValueError, TypeError):
                    return None
            
            # Helper function to safely convert to int
            def safe_int(value: str) -> Optional[int]:
                if value == 'None' or value == '-' or not value:
                    return None
                try:
                    return int(float(value))  # Convert via float to handle scientific notation
                except (ValueError, TypeError):
                    return None
            
            company = AlphaVantageCompany(
                symbol=response.get('Symbol', ''),
                name=response.get('Name', ''),
                description=response.get('Description'),
                exchange=response.get('Exchange'),
                currency=response.get('Currency'),
                country=response.get('Country'),
                sector=response.get('Sector'),
                industry=response.get('Industry'),
                market_capitalization=safe_int(response.get('MarketCapitalization')),
                ebitda=safe_int(response.get('EBITDA')),
                pe_ratio=safe_float(response.get('PERatio')),
                peg_ratio=safe_float(response.get('PEGRatio')),
                book_value=safe_float(response.get('BookValue')),
                dividend_per_share=safe_float(response.get('DividendPerShare')),
                dividend_yield=safe_float(response.get('DividendYield')),
                earnings_per_share=safe_float(response.get('EPS')),
                revenue_per_share_ttm=safe_float(response.get('RevenuePerShareTTM')),
                profit_margin=safe_float(response.get('ProfitMargin')),
                operating_margin_ttm=safe_float(response.get('OperatingMarginTTM')),
                return_on_assets_ttm=safe_float(response.get('ReturnOnAssetsTTM')),
                return_on_equity_ttm=safe_float(response.get('ReturnOnEquityTTM')),
                revenue_ttm=safe_int(response.get('RevenueTTM')),
                gross_profit_ttm=safe_int(response.get('GrossProfitTTM')),
                diluted_eps_ttm=safe_float(response.get('DilutedEPSTTM')),
                quarterly_earnings_growth_yoy=safe_float(response.get('QuarterlyEarningsGrowthYOY')),
                quarterly_revenue_growth_yoy=safe_float(response.get('QuarterlyRevenueGrowthYOY')),
                analyst_target_price=safe_float(response.get('AnalystTargetPrice')),
                trailing_pe=safe_float(response.get('TrailingPE')),
                forward_pe=safe_float(response.get('ForwardPE')),
                price_to_sales_ratio_ttm=safe_float(response.get('PriceToSalesRatioTTM')),
                price_to_book_ratio=safe_float(response.get('PriceToBookRatio')),
                ev_to_revenue=safe_float(response.get('EVToRevenue')),
                ev_to_ebitda=safe_float(response.get('EVToEBITDA')),
                beta=safe_float(response.get('Beta')),
                week_52_high=safe_float(response.get('52WeekHigh')),
                week_52_low=safe_float(response.get('52WeekLow')),
                day_50_moving_average=safe_float(response.get('50DayMovingAverage')),
                day_200_moving_average=safe_float(response.get('200DayMovingAverage')),
                shares_outstanding=safe_int(response.get('SharesOutstanding'))
            )
            
            return company
            
        except Exception as e:
            logger.error(f"Failed to get company fundamentals for {symbol}: {e}")
            return None
    
    async def get_earnings_data(self, symbol: str) -> List[AlphaVantageEarnings]:
        """Get quarterly earnings data."""
        try:
            response = await self.http_client.get_earnings(symbol)
            
            quarterly_earnings = response.get('quarterlyEarnings', [])
            if not quarterly_earnings:
                return []
            
            earnings = []
            for earning in quarterly_earnings:
                earning_data = AlphaVantageEarnings(
                    fiscal_date_ending=datetime.strptime(earning.get('fiscalDateEnding'), '%Y-%m-%d').date(),
                    reported_date=datetime.strptime(earning.get('reportedDate'), '%Y-%m-%d').date(),
                    reported_eps=float(earning.get('reportedEPS')) if earning.get('reportedEPS') != 'None' else None,
                    estimated_eps=float(earning.get('estimatedEPS')) if earning.get('estimatedEPS') != 'None' else None,
                    surprise=float(earning.get('surprise')) if earning.get('surprise') != 'None' else None,
                    surprise_percentage=float(earning.get('surprisePercentage')) if earning.get('surprisePercentage') != 'None' else None
                )
                earnings.append(earning_data)
            
            # Sort by fiscal date (most recent first)
            earnings.sort(key=lambda x: x.fiscal_date_ending, reverse=True)
            return earnings
            
        except Exception as e:
            logger.error(f"Failed to get earnings data for {symbol}: {e}")
            return []
    
    async def get_supported_functions(self) -> List[str]:
        """Get list of supported Alpha Vantage API functions."""
        return await self.http_client.get_supported_functions()
    
    async def health_check(self) -> AlphaVantageHealthStatus:
        """Perform comprehensive health check."""
        status = AlphaVantageHealthStatus(
            status="checking",
            api_key_configured=self.config.is_configured
        )
        
        if not self.config.is_configured:
            status.status = "not_configured"
            status.error_message = "ALPHA_VANTAGE_API_KEY environment variable not set"
            return status
        
        try:
            health_result = await self.http_client.health_check()
            if isinstance(health_result, dict) and 'status' in health_result:
                status.status = health_result['status']
                if health_result['status'] == 'operational':
                    status.last_successful_request = datetime.fromisoformat(health_result['last_successful_request']) if health_result['last_successful_request'] else None
                else:
                    status.error_message = health_result.get('error_message')
            else:
                status.status = "error"
                status.error_message = f"Invalid health check response: {health_result}"
                
        except Exception as e:
            status.status = "error"
            status.error_message = str(e)
        
        return status
    
    async def close(self):
        """Close the service and cleanup resources."""
        await self.http_client.close()


# Global service instance
alpha_vantage_service = AlphaVantageService()