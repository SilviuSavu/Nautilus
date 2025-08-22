"""
Alpha Vantage Data Models
========================

Pydantic models for Alpha Vantage API responses.
"""

from pydantic import BaseModel, Field, validator
from typing import Dict, List, Optional, Any
from datetime import datetime, date
from decimal import Decimal


class AlphaVantageQuote(BaseModel):
    """Real-time stock quote from Alpha Vantage."""
    
    symbol: str = Field(description="Stock ticker symbol")
    price: float = Field(description="Current stock price")
    change: float = Field(description="Price change from previous close")
    change_percent: float = Field(description="Percentage change from previous close")
    volume: int = Field(description="Trading volume")
    previous_close: Optional[float] = Field(None, description="Previous closing price")
    open_price: Optional[float] = Field(None, description="Opening price")
    high: Optional[float] = Field(None, description="Day's high price")
    low: Optional[float] = Field(None, description="Day's low price")
    timestamp: datetime = Field(description="Quote timestamp")
    
    @validator('symbol')
    def symbol_must_be_uppercase(cls, v):
        return v.upper()


class AlphaVantageBarData(BaseModel):
    """Price bar data (OHLCV)."""
    
    timestamp: datetime = Field(description="Bar timestamp")
    open: float = Field(description="Opening price")
    high: float = Field(description="High price")
    low: float = Field(description="Low price")
    close: float = Field(description="Closing price")
    volume: int = Field(description="Trading volume")


class AlphaVantageTimeSeries(BaseModel):
    """Time series data response."""
    
    symbol: str = Field(description="Stock ticker symbol")
    interval: str = Field(description="Time interval (1min, 5min, daily, etc.)")
    data: List[AlphaVantageBarData] = Field(description="Historical price data")
    last_refreshed: datetime = Field(description="Last data refresh timestamp")
    time_zone: str = Field(default="US/Eastern", description="Data timezone")
    
    @validator('symbol')
    def symbol_must_be_uppercase(cls, v):
        return v.upper()


class AlphaVantageEarnings(BaseModel):
    """Quarterly earnings data."""
    
    fiscal_date_ending: date = Field(description="Fiscal quarter end date")
    reported_date: date = Field(description="Earnings report date")
    reported_eps: Optional[float] = Field(None, description="Reported earnings per share")
    estimated_eps: Optional[float] = Field(None, description="Estimated earnings per share")
    surprise: Optional[float] = Field(None, description="Earnings surprise")
    surprise_percentage: Optional[float] = Field(None, description="Earnings surprise percentage")


class AlphaVantageCompany(BaseModel):
    """Company overview and fundamental data."""
    
    symbol: str = Field(description="Stock ticker symbol")
    name: str = Field(description="Company name")
    description: Optional[str] = Field(None, description="Business description")
    exchange: Optional[str] = Field(None, description="Stock exchange")
    currency: Optional[str] = Field(None, description="Trading currency")
    country: Optional[str] = Field(None, description="Country of incorporation")
    sector: Optional[str] = Field(None, description="Business sector")
    industry: Optional[str] = Field(None, description="Industry classification")
    market_capitalization: Optional[int] = Field(None, description="Market cap in USD")
    ebitda: Optional[int] = Field(None, description="EBITDA")
    pe_ratio: Optional[float] = Field(None, description="Price-to-earnings ratio")
    peg_ratio: Optional[float] = Field(None, description="PEG ratio")
    book_value: Optional[float] = Field(None, description="Book value per share")
    dividend_per_share: Optional[float] = Field(None, description="Annual dividend per share")
    dividend_yield: Optional[float] = Field(None, description="Dividend yield percentage")
    earnings_per_share: Optional[float] = Field(None, description="Trailing 12-month EPS")
    revenue_per_share_ttm: Optional[float] = Field(None, description="Revenue per share TTM")
    profit_margin: Optional[float] = Field(None, description="Net profit margin")
    operating_margin_ttm: Optional[float] = Field(None, description="Operating margin TTM")
    return_on_assets_ttm: Optional[float] = Field(None, description="Return on assets TTM")
    return_on_equity_ttm: Optional[float] = Field(None, description="Return on equity TTM")
    revenue_ttm: Optional[int] = Field(None, description="Revenue TTM")
    gross_profit_ttm: Optional[int] = Field(None, description="Gross profit TTM")
    diluted_eps_ttm: Optional[float] = Field(None, description="Diluted EPS TTM")
    quarterly_earnings_growth_yoy: Optional[float] = Field(None, description="Quarterly earnings growth YoY")
    quarterly_revenue_growth_yoy: Optional[float] = Field(None, description="Quarterly revenue growth YoY")
    analyst_target_price: Optional[float] = Field(None, description="Average analyst target price")
    trailing_pe: Optional[float] = Field(None, description="Trailing P/E ratio")
    forward_pe: Optional[float] = Field(None, description="Forward P/E ratio")
    price_to_sales_ratio_ttm: Optional[float] = Field(None, description="Price-to-sales ratio TTM")
    price_to_book_ratio: Optional[float] = Field(None, description="Price-to-book ratio")
    ev_to_revenue: Optional[float] = Field(None, description="EV to revenue ratio")
    ev_to_ebitda: Optional[float] = Field(None, description="EV to EBITDA ratio")
    beta: Optional[float] = Field(None, description="Stock beta")
    week_52_high: Optional[float] = Field(None, description="52-week high price")
    week_52_low: Optional[float] = Field(None, description="52-week low price")
    day_50_moving_average: Optional[float] = Field(None, description="50-day moving average")
    day_200_moving_average: Optional[float] = Field(None, description="200-day moving average")
    shares_outstanding: Optional[int] = Field(None, description="Shares outstanding")
    date_first_added: Optional[date] = Field(None, description="Date first added to exchange")
    
    @validator('symbol')
    def symbol_must_be_uppercase(cls, v):
        return v.upper()


class AlphaVantageSearchResult(BaseModel):
    """Symbol search result."""
    
    symbol: str = Field(description="Stock ticker symbol")
    name: str = Field(description="Company name")
    type: str = Field(description="Security type (Equity, ETF, etc.)")
    region: str = Field(description="Geographic region")
    market_open: str = Field(description="Market open time")
    market_close: str = Field(description="Market close time")
    timezone: str = Field(description="Market timezone")
    currency: str = Field(description="Trading currency")
    match_score: float = Field(description="Search relevance score")
    
    @validator('symbol')
    def symbol_must_be_uppercase(cls, v):
        return v.upper()


class AlphaVantageError(BaseModel):
    """Error response from Alpha Vantage API."""
    
    error_type: str = Field(description="Error type")
    message: str = Field(description="Error message")
    timestamp: datetime = Field(default_factory=datetime.now, description="Error timestamp")


class AlphaVantageHealthStatus(BaseModel):
    """Health check response."""
    
    service: str = Field(default="Alpha Vantage", description="Service name")
    status: str = Field(description="Service status")
    api_key_configured: bool = Field(description="Whether API key is configured")
    timestamp: datetime = Field(default_factory=datetime.now, description="Health check timestamp")
    last_successful_request: Optional[datetime] = Field(None, description="Last successful API request")
    error_message: Optional[str] = Field(None, description="Error message if any")
    rate_limit_remaining: Optional[int] = Field(None, description="API calls remaining")
    rate_limit_reset: Optional[datetime] = Field(None, description="Rate limit reset time")