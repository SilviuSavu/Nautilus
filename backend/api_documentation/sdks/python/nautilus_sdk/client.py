"""
Main Nautilus Trading Platform Python Client
Provides comprehensive API access with async/await support
"""

import asyncio
import aiohttp
from typing import Dict, List, Optional, Any, Union
from datetime import datetime, timedelta
import json

from .auth import AuthenticatedSession
from .exceptions import NautilusException, AuthenticationError, RateLimitError
from .models import (
    MarketData, RiskLimit, Strategy, HealthCheck, 
    PaginationResponse, WebSocketMessage
)
from .websocket import WebSocketClient


class NautilusClient:
    """
    Main client for interacting with the Nautilus Trading Platform API
    
    Supports both synchronous and asynchronous operations with comprehensive
    error handling, rate limiting, and automatic retry logic.
    """
    
    def __init__(
        self,
        base_url: str = "http://localhost:8001",
        api_key: Optional[str] = None,
        username: Optional[str] = None,
        password: Optional[str] = None,
        timeout: int = 30,
        max_retries: int = 3,
        retry_delay: float = 1.0
    ):
        """
        Initialize Nautilus client
        
        Args:
            base_url: Base URL for the API (default: http://localhost:8001)
            api_key: API key for authentication
            username: Username for login authentication
            password: Password for login authentication
            timeout: Request timeout in seconds
            max_retries: Maximum number of retries for failed requests
            retry_delay: Delay between retries in seconds
        """
        self.base_url = base_url.rstrip('/')
        self.timeout = aiohttp.ClientTimeout(total=timeout)
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        
        # Initialize authentication session
        self.auth = AuthenticatedSession(
            base_url=base_url,
            api_key=api_key,
            username=username,
            password=password
        )
        
        # WebSocket client for real-time data
        self.websocket = WebSocketClient(base_url, self.auth)
        
        # Session will be created on first use
        self._session: Optional[aiohttp.ClientSession] = None
        
    async def __aenter__(self):
        """Async context manager entry"""
        await self._ensure_session()
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self._session:
            await self._session.close()
            
    async def _ensure_session(self):
        """Ensure aiohttp session is created"""
        if not self._session:
            self._session = aiohttp.ClientSession(
                timeout=self.timeout,
                headers={"User-Agent": "Nautilus-Python-SDK/3.0.0"}
            )
            
    async def _request(
        self,
        method: str,
        endpoint: str,
        params: Optional[Dict] = None,
        data: Optional[Dict] = None,
        headers: Optional[Dict] = None,
        auth_required: bool = True
    ) -> Dict[str, Any]:
        """
        Make authenticated API request with retry logic
        
        Args:
            method: HTTP method (GET, POST, PUT, DELETE)
            endpoint: API endpoint path
            params: Query parameters
            data: Request body data
            headers: Additional headers
            auth_required: Whether authentication is required
            
        Returns:
            Response data as dictionary
            
        Raises:
            NautilusException: For API errors
            AuthenticationError: For authentication errors
            RateLimitError: For rate limit errors
        """
        await self._ensure_session()
        
        url = f"{self.base_url}{endpoint}"
        request_headers = headers or {}
        
        # Add authentication if required
        if auth_required:
            auth_headers = await self.auth.get_auth_headers()
            request_headers.update(auth_headers)
            
        # Retry logic
        last_exception = None
        for attempt in range(self.max_retries + 1):
            try:
                async with self._session.request(
                    method=method,
                    url=url,
                    params=params,
                    json=data,
                    headers=request_headers
                ) as response:
                    
                    # Handle different response codes
                    if response.status == 401:
                        # Try to refresh token and retry once
                        if attempt == 0 and auth_required:
                            await self.auth.refresh_token()
                            auth_headers = await self.auth.get_auth_headers()
                            request_headers.update(auth_headers)
                            continue
                        raise AuthenticationError("Authentication failed")
                        
                    elif response.status == 429:
                        # Rate limit exceeded
                        retry_after = int(response.headers.get('Retry-After', self.retry_delay))
                        raise RateLimitError(f"Rate limit exceeded. Retry after {retry_after}s")
                        
                    elif response.status >= 400:
                        error_data = await response.json() if response.content_type == 'application/json' else {}
                        raise NautilusException(
                            f"API error {response.status}: {error_data.get('message', 'Unknown error')}"
                        )
                    
                    # Parse successful response
                    if response.content_type == 'application/json':
                        return await response.json()
                    else:
                        return {"data": await response.text()}
                        
            except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                last_exception = e
                if attempt < self.max_retries:
                    await asyncio.sleep(self.retry_delay * (attempt + 1))
                    continue
                    
        # All retries failed
        raise NautilusException(f"Request failed after {self.max_retries} retries: {last_exception}")
    
    # Authentication Methods
    async def login(self, username: str, password: str, remember_me: bool = False) -> Dict[str, str]:
        """
        Authenticate user and get access token
        
        Args:
            username: User email/username
            password: User password
            remember_me: Whether to use long-lived token
            
        Returns:
            Authentication response with tokens
        """
        data = {
            "username": username,
            "password": password,
            "remember_me": remember_me
        }
        
        response = await self._request(
            "POST", 
            "/api/v1/auth/login", 
            data=data,
            auth_required=False
        )
        
        # Store tokens in auth session
        await self.auth.set_tokens(
            access_token=response["access_token"],
            refresh_token=response.get("refresh_token")
        )
        
        return response
        
    async def refresh_token(self) -> Dict[str, str]:
        """Refresh access token using refresh token"""
        return await self.auth.refresh_token()
        
    # System & Health Methods
    async def get_health(self) -> HealthCheck:
        """Get system health status"""
        response = await self._request("GET", "/health", auth_required=False)
        return HealthCheck(**response)
        
    async def get_system_metrics(self) -> Dict[str, Any]:
        """Get comprehensive system metrics"""
        return await self._request("GET", "/api/v1/system/metrics")
    
    # Market Data Methods
    async def get_quote(
        self, 
        symbol: str, 
        source: Optional[str] = None
    ) -> MarketData:
        """
        Get real-time quote for a symbol
        
        Args:
            symbol: Trading symbol (e.g., 'AAPL')
            source: Data source preference ('IBKR', 'ALPHA_VANTAGE', 'YAHOO')
            
        Returns:
            Real-time market data
        """
        params = {}
        if source:
            params["source"] = source
            
        response = await self._request(
            "GET", 
            f"/api/v1/market-data/quote/{symbol}",
            params=params
        )
        return MarketData(**response)
        
    async def get_historical_data(
        self,
        symbol: str,
        interval: str = "1day",
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        limit: int = 100
    ) -> Dict[str, Any]:
        """
        Get historical market data
        
        Args:
            symbol: Trading symbol
            interval: Data interval ('1min', '5min', '1hour', '1day', etc.)
            start_date: Start date for data
            end_date: End date for data
            limit: Maximum number of data points
            
        Returns:
            Historical market data with pagination
        """
        params = {
            "interval": interval,
            "limit": limit
        }
        
        if start_date:
            params["start_date"] = start_date.strftime("%Y-%m-%d")
        if end_date:
            params["end_date"] = end_date.strftime("%Y-%m-%d")
            
        return await self._request(
            "GET",
            f"/api/v1/market-data/historical/{symbol}",
            params=params
        )
    
    # Risk Management Methods
    async def create_risk_limit(
        self,
        limit_type: str,
        value: float,
        symbol: Optional[str] = None,
        warning_threshold: float = 0.8,
        auto_adjust: bool = False
    ) -> RiskLimit:
        """
        Create a new risk limit
        
        Args:
            limit_type: Type of limit ('position_limit', 'loss_limit', etc.)
            value: Limit value
            symbol: Symbol for the limit (optional)
            warning_threshold: Warning threshold (0-1)
            auto_adjust: Whether to auto-adjust limit
            
        Returns:
            Created risk limit
        """
        data = {
            "type": limit_type,
            "value": value,
            "warning_threshold": warning_threshold,
            "auto_adjust": auto_adjust
        }
        
        if symbol:
            data["symbol"] = symbol
            
        response = await self._request("POST", "/api/v1/risk/limits", data=data)
        return RiskLimit(**response)
        
    async def get_risk_limits(self) -> List[RiskLimit]:
        """Get all risk limits"""
        response = await self._request("GET", "/api/v1/risk/limits")
        return [RiskLimit(**limit) for limit in response["limits"]]
        
    async def check_risk_limit(self, limit_id: str) -> Dict[str, Any]:
        """Check specific risk limit status"""
        return await self._request("GET", f"/api/v1/risk/limits/{limit_id}/check")
        
    async def get_risk_breaches(self) -> List[Dict[str, Any]]:
        """Get recent risk breaches"""
        response = await self._request("GET", "/api/v1/risk/breaches")
        return response["breaches"]
    
    # Strategy Management Methods
    async def deploy_strategy(
        self,
        name: str,
        version: str,
        description: str,
        parameters: Dict[str, Any],
        risk_limits: Optional[Dict[str, Any]] = None,
        deployment_config: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Deploy a trading strategy
        
        Args:
            name: Strategy name
            version: Strategy version
            description: Strategy description
            parameters: Strategy parameters
            risk_limits: Risk limits for the strategy
            deployment_config: Deployment configuration
            
        Returns:
            Deployment response with deployment_id
        """
        data = {
            "name": name,
            "version": version,
            "description": description,
            "parameters": parameters
        }
        
        if risk_limits:
            data["risk_limits"] = risk_limits
        if deployment_config:
            data["deployment_config"] = deployment_config
            
        return await self._request("POST", "/api/v1/strategies/deploy", data=data)
        
    async def get_strategies(self) -> List[Strategy]:
        """Get all strategies"""
        response = await self._request("GET", "/api/v1/strategies")
        return [Strategy(**strategy) for strategy in response["strategies"]]
        
    async def get_strategy(self, strategy_id: str) -> Strategy:
        """Get specific strategy"""
        response = await self._request("GET", f"/api/v1/strategies/{strategy_id}")
        return Strategy(**response)
        
    async def get_deployment_status(self, deployment_id: str) -> Dict[str, Any]:
        """Get deployment pipeline status"""
        return await self._request("GET", f"/api/v1/strategies/pipeline/{deployment_id}/status")
    
    # Analytics Methods
    async def get_performance_analytics(
        self, 
        portfolio_id: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """
        Get performance analytics
        
        Args:
            portfolio_id: Portfolio ID (optional)
            start_date: Analysis start date
            end_date: Analysis end date
            
        Returns:
            Performance analytics data
        """
        params = {}
        if portfolio_id:
            params["portfolio_id"] = portfolio_id
        if start_date:
            params["start_date"] = start_date.isoformat()
        if end_date:
            params["end_date"] = end_date.isoformat()
            
        return await self._request("GET", "/api/v1/analytics/performance", params=params)
        
    async def get_risk_analytics(self, portfolio_id: str) -> Dict[str, Any]:
        """Get risk analytics for a portfolio"""
        return await self._request("GET", f"/api/v1/analytics/risk/{portfolio_id}")
    
    # WebSocket Methods
    async def stream_market_data(
        self, 
        symbols: List[str],
        callback: callable,
        data_types: Optional[List[str]] = None
    ):
        """
        Start real-time market data stream
        
        Args:
            symbols: List of symbols to stream
            callback: Function to call with each message
            data_types: Types of data to stream ('quotes', 'trades', etc.)
        """
        await self.websocket.connect()
        
        subscription = {
            "type": "subscribe",
            "symbols": symbols,
            "data_types": data_types or ["quotes", "trades"]
        }
        
        await self.websocket.subscribe("market_data", subscription, callback)
        
    async def stream_trade_updates(self, callback: callable):
        """Stream real-time trade updates"""
        await self.websocket.connect()
        await self.websocket.subscribe("trade_updates", {}, callback)
        
    async def stream_risk_alerts(self, callback: callable):
        """Stream real-time risk alerts"""
        await self.websocket.connect()
        await self.websocket.subscribe("risk_alerts", {}, callback)
    
    # Utility Methods
    def close(self):
        """Close all connections"""
        if self._session and not self._session.closed:
            asyncio.create_task(self._session.close())
        asyncio.create_task(self.websocket.disconnect())


# Convenience functions for quick operations
async def quick_quote(symbol: str, base_url: str = "http://localhost:8001") -> MarketData:
    """Quick function to get a quote without managing client lifecycle"""
    async with NautilusClient(base_url=base_url) as client:
        return await client.get_quote(symbol)


async def quick_health_check(base_url: str = "http://localhost:8001") -> HealthCheck:
    """Quick function to check system health"""
    async with NautilusClient(base_url=base_url) as client:
        return await client.get_health()


# Example usage
if __name__ == "__main__":
    async def main():
        # Create client with authentication
        client = NautilusClient(
            base_url="http://localhost:8001",
            username="trader@nautilus.com",
            password="your_password"
        )
        
        async with client:
            # Login
            auth_response = await client.login("trader@nautilus.com", "password")
            print(f"Authenticated: {auth_response['access_token'][:20]}...")
            
            # Get system health
            health = await client.get_health()
            print(f"System status: {health.status}")
            
            # Get real-time quote
            quote = await client.get_quote("AAPL")
            print(f"AAPL: ${quote.price} ({quote.change_percent:+.2f}%)")
            
            # Create risk limit
            risk_limit = await client.create_risk_limit(
                limit_type="position_limit",
                value=1000000,
                symbol="AAPL",
                warning_threshold=0.8
            )
            print(f"Risk limit created: {risk_limit.id}")
            
            # Stream market data
            def handle_market_data(message):
                print(f"Market update: {message}")
                
            await client.stream_market_data(["AAPL", "GOOGL"], handle_market_data)
    
    # Run example
    asyncio.run(main())