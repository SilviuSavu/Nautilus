# Nautilus API Developer Experience Guide

## Table of Contents

1. [API Design Principles](#api-design-principles)
2. [Error Handling Best Practices](#error-handling-best-practices)
3. [Rate Limiting & Performance](#rate-limiting--performance)
4. [Security Guidelines](#security-guidelines)
5. [Testing Strategies](#testing-strategies)
6. [Monitoring & Observability](#monitoring--observability)
7. [Documentation Standards](#documentation-standards)
8. [Integration Patterns](#integration-patterns)
9. [Troubleshooting Guide](#troubleshooting-guide)
10. [Performance Optimization](#performance-optimization)

## API Design Principles

### RESTful Design Standards

The Nautilus API follows RESTful design principles with consistent patterns:

```
GET    /api/v1/resource          # List resources
GET    /api/v1/resource/{id}     # Get specific resource
POST   /api/v1/resource          # Create new resource
PUT    /api/v1/resource/{id}     # Update entire resource
PATCH  /api/v1/resource/{id}     # Partial update
DELETE /api/v1/resource/{id}     # Delete resource
```

### Resource Naming Conventions

- **Use plural nouns**: `/strategies`, `/risk-limits`, `/market-data`
- **Use kebab-case**: `/market-data`, `/risk-limits`, `/trading-economics`
- **Be descriptive**: `/market-data/historical` instead of `/data/hist`
- **Avoid deep nesting**: Maximum 3 levels deep

### HTTP Status Codes

| Status Code | Description | Use Case |
|-------------|-------------|----------|
| 200 | OK | Successful GET, PUT, PATCH |
| 201 | Created | Successful POST |
| 204 | No Content | Successful DELETE |
| 400 | Bad Request | Invalid request syntax |
| 401 | Unauthorized | Missing or invalid authentication |
| 403 | Forbidden | Valid auth but insufficient permissions |
| 404 | Not Found | Resource doesn't exist |
| 409 | Conflict | Resource already exists |
| 422 | Unprocessable Entity | Invalid data format |
| 429 | Too Many Requests | Rate limit exceeded |
| 500 | Internal Server Error | Server-side error |

### Consistent Response Format

All API responses follow a consistent structure:

```json
{
  "data": {},           // Response data
  "meta": {             // Metadata
    "timestamp": "2025-01-23T10:30:00.000Z",
    "request_id": "req_123abc",
    "version": "3.0.0"
  },
  "pagination": {       // For paginated responses
    "page": 1,
    "limit": 50,
    "total": 1000,
    "pages": 20,
    "has_next": true,
    "has_prev": false
  }
}
```

## Error Handling Best Practices

### Standard Error Response

```json
{
  "error": "VALIDATION_ERROR",
  "message": "Request validation failed",
  "details": {
    "symbol": ["Symbol is required and must be a valid ticker"],
    "amount": ["Amount must be a positive number"]
  },
  "timestamp": "2025-01-23T10:30:00.000Z",
  "request_id": "req_123abc456def",
  "documentation_url": "https://docs.nautilus-trading.com/errors#validation_error"
}
```

### Error Categories

#### 1. Client Errors (4xx)

```python
# Example: Input validation error
{
  "error": "VALIDATION_ERROR",
  "message": "Invalid symbol format",
  "details": {
    "symbol": "Symbol must be 1-5 uppercase letters"
  }
}
```

#### 2. Authentication Errors

```python
# Example: Token expired
{
  "error": "TOKEN_EXPIRED",
  "message": "Access token has expired",
  "details": {
    "expired_at": "2025-01-23T09:30:00.000Z",
    "refresh_endpoint": "/api/v1/auth/refresh"
  }
}
```

#### 3. Rate Limiting Errors

```python
# Example: Rate limit exceeded
{
  "error": "RATE_LIMIT_EXCEEDED",
  "message": "API rate limit exceeded",
  "details": {
    "limit": 1000,
    "window": 3600,
    "retry_after": 60,
    "remaining": 0
  }
}
```

### Error Handling Implementation

```python
# Python SDK Example
try:
    quote = await client.get_quote("AAPL")
except AuthenticationError as e:
    # Handle authentication issues
    await client.refresh_token()
    quote = await client.get_quote("AAPL")
    
except RateLimitError as e:
    # Handle rate limiting
    await asyncio.sleep(e.retry_after)
    quote = await client.get_quote("AAPL")
    
except ValidationError as e:
    # Handle validation errors
    print(f"Invalid request: {e.details}")
    
except NautilusException as e:
    # Handle general API errors
    print(f"API error: {e.message}")
```

```typescript
// TypeScript SDK Example
try {
  const quote = await client.marketData.getQuote('AAPL');
} catch (error) {
  if (error instanceof AuthenticationError) {
    await client.refreshToken();
    return await client.marketData.getQuote('AAPL');
  }
  
  if (error instanceof RateLimitError) {
    await new Promise(resolve => setTimeout(resolve, error.retryAfter * 1000));
    return await client.marketData.getQuote('AAPL');
  }
  
  if (error instanceof ValidationError) {
    console.error('Validation failed:', error.details);
  }
  
  throw error;
}
```

## Rate Limiting & Performance

### Rate Limiting Strategy

Nautilus API implements intelligent rate limiting with multiple tiers:

#### Standard Rate Limits

| Endpoint Category | Requests/Minute | Burst Limit |
|------------------|-----------------|-------------|
| Authentication | 10 | 20 |
| Market Data | 1000 | 2000 |
| Trading Operations | 100 | 200 |
| Risk Management | 500 | 1000 |
| Analytics | 200 | 400 |
| WebSocket Connections | 10/user | 20/user |

#### Rate Limit Headers

```http
HTTP/1.1 200 OK
X-RateLimit-Limit: 1000
X-RateLimit-Remaining: 999
X-RateLimit-Reset: 1642950000
X-RateLimit-Retry-After: 60
```

### Performance Optimization Strategies

#### 1. Caching

```python
# Implement client-side caching
import time
from typing import Dict, Any

class CachedClient:
    def __init__(self):
        self.cache: Dict[str, Any] = {}
        self.cache_ttl = 300  # 5 minutes
    
    async def get_quote_cached(self, symbol: str):
        cache_key = f"quote:{symbol}"
        now = time.time()
        
        # Check cache first
        if cache_key in self.cache:
            cached_data, timestamp = self.cache[cache_key]
            if now - timestamp < self.cache_ttl:
                return cached_data
        
        # Fetch from API
        quote = await self.client.get_quote(symbol)
        self.cache[cache_key] = (quote, now)
        return quote
```

#### 2. Batch Requests

```python
# Batch multiple symbol requests
symbols = ["AAPL", "GOOGL", "MSFT", "TSLA"]
quotes = await client.get_quotes_batch(symbols)

# Instead of individual requests:
# for symbol in symbols:
#     quote = await client.get_quote(symbol)  # Inefficient
```

#### 3. Pagination

```python
# Efficient pagination
async def get_all_strategies():
    strategies = []
    page = 1
    
    while True:
        response = await client.get_strategies(page=page, limit=100)
        strategies.extend(response.data)
        
        if not response.pagination.has_next:
            break
            
        page += 1
    
    return strategies
```

## Security Guidelines

### Authentication Best Practices

#### 1. Token Management

```python
class SecureTokenManager:
    def __init__(self):
        self.access_token = None
        self.refresh_token = None
        self.token_expires_at = None
    
    def store_tokens_securely(self, access_token: str, refresh_token: str, expires_in: int):
        """Store tokens using encrypted storage"""
        import keyring
        
        keyring.set_password("nautilus_api", "access_token", access_token)
        keyring.set_password("nautilus_api", "refresh_token", refresh_token)
        
        self.access_token = access_token
        self.refresh_token = refresh_token
        self.token_expires_at = time.time() + expires_in
    
    def get_access_token(self) -> Optional[str]:
        """Get access token from secure storage"""
        if self.needs_refresh():
            self.refresh_access_token()
        
        return keyring.get_password("nautilus_api", "access_token")
```

#### 2. Environment Configuration

```bash
# .env file (never commit to version control)
NAUTILUS_API_URL=https://api.nautilus-trading.com
NAUTILUS_API_KEY=your_secure_api_key
NAUTILUS_USERNAME=your_username
NAUTILUS_PASSWORD=your_secure_password

# Use environment variables
NAUTILUS_CLIENT_ID=client_123
NAUTILUS_CLIENT_SECRET=secret_abc
```

#### 3. HTTPS and Certificate Validation

```python
import ssl
import certifi

# Always use HTTPS in production
client = NautilusClient(
    base_url="https://api.nautilus-trading.com",  # HTTPS only
    verify_ssl=True,
    ca_bundle=certifi.where()  # Use updated CA bundle
)
```

### API Key Security

```python
# DO NOT hardcode API keys
# ❌ Bad
client = NautilusClient(api_key="sk_live_1234567890")

# ✅ Good
import os
client = NautilusClient(
    api_key=os.getenv("NAUTILUS_API_KEY"),
    base_url=os.getenv("NAUTILUS_API_URL", "https://api.nautilus-trading.com")
)
```

## Testing Strategies

### Unit Testing

```python
import pytest
from unittest.mock import AsyncMock, patch
from nautilus_sdk import NautilusClient

class TestNautilusClient:
    @pytest.fixture
    async def client(self):
        return NautilusClient(base_url="http://test.example.com")
    
    @patch('nautilus_sdk.client.aiohttp.ClientSession.request')
    async def test_get_quote_success(self, mock_request, client):
        # Mock successful response
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json.return_value = {
            "symbol": "AAPL",
            "price": 150.25,
            "timestamp": "2025-01-23T10:30:00.000Z"
        }
        mock_request.return_value.__aenter__.return_value = mock_response
        
        quote = await client.get_quote("AAPL")
        
        assert quote.symbol == "AAPL"
        assert quote.price == 150.25
        mock_request.assert_called_once()
    
    @patch('nautilus_sdk.client.aiohttp.ClientSession.request')
    async def test_rate_limit_handling(self, mock_request, client):
        # Mock rate limit response
        mock_response = AsyncMock()
        mock_response.status = 429
        mock_response.headers = {'Retry-After': '60'}
        mock_request.return_value.__aenter__.return_value = mock_response
        
        with pytest.raises(RateLimitError) as exc_info:
            await client.get_quote("AAPL")
        
        assert exc_info.value.retry_after == 60
```

### Integration Testing

```python
class TestIntegration:
    @pytest.mark.asyncio
    async def test_end_to_end_workflow(self):
        """Test complete workflow from authentication to trading"""
        client = NautilusClient(base_url=TEST_API_URL)
        
        # 1. Authentication
        await client.login(TEST_USERNAME, TEST_PASSWORD)
        assert client.is_authenticated()
        
        # 2. Market data
        quote = await client.get_quote("AAPL")
        assert quote.symbol == "AAPL"
        assert quote.price > 0
        
        # 3. Risk management
        risk_limit = await client.create_risk_limit(
            limit_type="position_limit",
            value=1000000,
            symbol="AAPL"
        )
        assert risk_limit.id is not None
        
        # 4. Strategy deployment
        deployment = await client.deploy_strategy({
            "name": "Test_Strategy",
            "version": "1.0.0",
            "description": "Integration test strategy",
            "parameters": {"symbols": ["AAPL"]}
        })
        assert deployment.deployment_id is not None
        
        # 5. Cleanup
        await client.delete_risk_limit(risk_limit.id)
```

### Performance Testing

```python
import asyncio
import time
from concurrent.futures import ThreadPoolExecutor

async def performance_test():
    """Test API performance under load"""
    client = NautilusClient()
    await client.login(USERNAME, PASSWORD)
    
    # Test concurrent requests
    symbols = ["AAPL", "GOOGL", "MSFT"] * 10  # 30 symbols
    start_time = time.time()
    
    # Run concurrent requests
    tasks = [client.get_quote(symbol) for symbol in symbols]
    quotes = await asyncio.gather(*tasks, return_exceptions=True)
    
    end_time = time.time()
    duration = end_time - start_time
    
    # Analyze results
    successful_requests = sum(1 for q in quotes if not isinstance(q, Exception))
    failed_requests = len(quotes) - successful_requests
    requests_per_second = len(quotes) / duration
    
    print(f"Performance Results:")
    print(f"  Total requests: {len(quotes)}")
    print(f"  Successful: {successful_requests}")
    print(f"  Failed: {failed_requests}")
    print(f"  Duration: {duration:.2f}s")
    print(f"  Requests/second: {requests_per_second:.2f}")
    
    assert requests_per_second > 10  # Minimum performance threshold
    assert failed_requests < len(quotes) * 0.05  # Less than 5% failure rate
```

## Monitoring & Observability

### Logging Best Practices

```python
import logging
import structlog
from nautilus_sdk import NautilusClient

# Configure structured logging
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.dev.ConsoleRenderer()
    ],
    wrapper_class=structlog.stdlib.BoundLogger,
    logger_factory=structlog.stdlib.LoggerFactory(),
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger()

class MonitoredNautilusClient:
    def __init__(self):
        self.client = NautilusClient()
        self.logger = logger.bind(component="nautilus_client")
    
    async def get_quote_with_monitoring(self, symbol: str):
        request_id = f"req_{int(time.time())}"
        
        self.logger.info(
            "market_data_request_started",
            symbol=symbol,
            request_id=request_id
        )
        
        try:
            start_time = time.time()
            quote = await self.client.get_quote(symbol)
            duration = time.time() - start_time
            
            self.logger.info(
                "market_data_request_completed",
                symbol=symbol,
                request_id=request_id,
                duration=duration,
                price=quote.price
            )
            
            return quote
            
        except Exception as e:
            self.logger.error(
                "market_data_request_failed",
                symbol=symbol,
                request_id=request_id,
                error=str(e),
                error_type=type(e).__name__
            )
            raise
```

### Metrics Collection

```python
import time
from collections import defaultdict, deque
from threading import Lock

class MetricsCollector:
    def __init__(self):
        self.request_counts = defaultdict(int)
        self.response_times = defaultdict(lambda: deque(maxlen=1000))
        self.error_counts = defaultdict(int)
        self.lock = Lock()
    
    def record_request(self, endpoint: str, duration: float, success: bool):
        with self.lock:
            self.request_counts[endpoint] += 1
            self.response_times[endpoint].append(duration)
            
            if not success:
                self.error_counts[endpoint] += 1
    
    def get_metrics(self) -> dict:
        with self.lock:
            metrics = {}
            
            for endpoint in self.request_counts:
                response_times = list(self.response_times[endpoint])
                
                metrics[endpoint] = {
                    "request_count": self.request_counts[endpoint],
                    "error_count": self.error_counts[endpoint],
                    "error_rate": self.error_counts[endpoint] / self.request_counts[endpoint],
                    "avg_response_time": sum(response_times) / len(response_times) if response_times else 0,
                    "p95_response_time": sorted(response_times)[int(len(response_times) * 0.95)] if response_times else 0
                }
            
            return metrics

# Usage
metrics = MetricsCollector()

async def monitored_api_call(client, endpoint, *args, **kwargs):
    start_time = time.time()
    success = True
    
    try:
        method = getattr(client, endpoint)
        result = await method(*args, **kwargs)
        return result
    except Exception as e:
        success = False
        raise
    finally:
        duration = time.time() - start_time
        metrics.record_request(endpoint, duration, success)
```

## Documentation Standards

### OpenAPI Documentation

```yaml
# Example OpenAPI specification
openapi: 3.0.3
info:
  title: Nautilus Trading Platform API
  version: 3.0.0
  description: Enterprise trading platform with 8-source data integration
  
paths:
  /api/v1/market-data/quote/{symbol}:
    get:
      summary: Get real-time quote
      description: |
        Retrieve real-time quote data from integrated data sources.
        
        **Data Sources:**
        - IBKR (Primary)
        - Alpha Vantage (Backup)
        - Yahoo Finance (Backup)
        
        **Rate Limits:**
        - 1000 requests/minute
        - Burst limit: 2000 requests
      parameters:
        - name: symbol
          in: path
          required: true
          schema:
            type: string
            pattern: '^[A-Z]{1,5}$'
          example: AAPL
          description: Stock symbol (1-5 uppercase letters)
        - name: source
          in: query
          schema:
            type: string
            enum: [IBKR, ALPHA_VANTAGE, YAHOO]
          description: Preferred data source
      responses:
        '200':
          description: Real-time quote data
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/MarketData'
              examples:
                apple_stock:
                  summary: Apple Inc. stock quote
                  value:
                    symbol: AAPL
                    price: 150.25
                    timestamp: '2025-01-23T15:30:00.000Z'
        '404':
          description: Symbol not found
        '429':
          description: Rate limit exceeded
```

### Code Documentation

```python
class NautilusClient:
    """
    Official Nautilus Trading Platform API client.
    
    Provides comprehensive access to trading, market data, risk management,
    and analytics capabilities with built-in error handling and rate limiting.
    
    Args:
        base_url: API base URL (default: http://localhost:8001)
        timeout: Request timeout in seconds (default: 30)
        max_retries: Maximum retry attempts (default: 3)
        
    Examples:
        Basic usage:
        >>> client = NautilusClient()
        >>> await client.login("user@example.com", "password")
        >>> quote = await client.get_quote("AAPL")
        >>> print(f"AAPL: ${quote.price}")
        
        With custom configuration:
        >>> client = NautilusClient(
        ...     base_url="https://api.nautilus-trading.com",
        ...     timeout=60,
        ...     max_retries=5
        ... )
        
    Note:
        All methods require authentication unless explicitly stated.
        Use context manager for automatic cleanup:
        
        >>> async with NautilusClient() as client:
        ...     await client.login("user", "pass")
        ...     # Client will be closed automatically
    """
    
    async def get_quote(self, symbol: str, source: Optional[str] = None) -> MarketData:
        """
        Get real-time quote for a financial instrument.
        
        Args:
            symbol: Financial instrument symbol (e.g., 'AAPL', 'GOOGL')
            source: Preferred data source ('IBKR', 'ALPHA_VANTAGE', 'YAHOO')
                   If not specified, uses best available source
        
        Returns:
            MarketData object containing quote information
            
        Raises:
            AuthenticationError: If not authenticated or token expired
            ValidationError: If symbol format is invalid
            RateLimitError: If rate limit is exceeded
            NautilusException: For other API errors
            
        Examples:
            >>> quote = await client.get_quote("AAPL")
            >>> print(f"Price: ${quote.price}")
            >>> print(f"Volume: {quote.volume:,}")
            
            With specific source:
            >>> quote = await client.get_quote("AAPL", source="IBKR")
            
        Note:
            Quote data is real-time during market hours and delayed otherwise.
            Check the timestamp field for data freshness.
        """
```

## Integration Patterns

### Circuit Breaker Pattern

```python
import asyncio
from enum import Enum
from typing import Callable, Any

class CircuitState(Enum):
    CLOSED = "closed"      # Normal operation
    OPEN = "open"         # Circuit breaker active
    HALF_OPEN = "half_open"  # Testing if service recovered

class CircuitBreaker:
    def __init__(self, failure_threshold: int = 5, recovery_timeout: int = 60):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failure_count = 0
        self.last_failure_time = None
        self.state = CircuitState.CLOSED
    
    async def call(self, func: Callable, *args, **kwargs) -> Any:
        if self.state == CircuitState.OPEN:
            if self._should_attempt_reset():
                self.state = CircuitState.HALF_OPEN
            else:
                raise Exception("Circuit breaker is OPEN")
        
        try:
            result = await func(*args, **kwargs)
            self._on_success()
            return result
        except Exception as e:
            self._on_failure()
            raise
    
    def _should_attempt_reset(self) -> bool:
        return (
            self.last_failure_time and
            time.time() - self.last_failure_time >= self.recovery_timeout
        )
    
    def _on_success(self):
        self.failure_count = 0
        self.state = CircuitState.CLOSED
    
    def _on_failure(self):
        self.failure_count += 1
        self.last_failure_time = time.time()
        
        if self.failure_count >= self.failure_threshold:
            self.state = CircuitState.OPEN

# Usage
circuit_breaker = CircuitBreaker(failure_threshold=3, recovery_timeout=30)

async def robust_api_call(client, symbol):
    return await circuit_breaker.call(client.get_quote, symbol)
```

### Retry Strategy with Exponential Backoff

```python
import asyncio
import random
from typing import Type, Tuple

async def retry_with_backoff(
    func: Callable,
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    exponential_base: float = 2.0,
    jitter: bool = True,
    retry_exceptions: Tuple[Type[Exception], ...] = (Exception,)
):
    """
    Retry function with exponential backoff and jitter.
    
    Args:
        func: Async function to retry
        max_retries: Maximum number of retry attempts
        base_delay: Base delay in seconds
        max_delay: Maximum delay in seconds
        exponential_base: Base for exponential backoff
        jitter: Add random jitter to prevent thundering herd
        retry_exceptions: Exceptions that trigger retry
    """
    last_exception = None
    
    for attempt in range(max_retries + 1):
        try:
            return await func()
        except retry_exceptions as e:
            last_exception = e
            
            if attempt == max_retries:
                break
            
            # Calculate delay with exponential backoff
            delay = min(base_delay * (exponential_base ** attempt), max_delay)
            
            # Add jitter
            if jitter:
                delay = delay * (0.5 + random.random() * 0.5)
            
            await asyncio.sleep(delay)
    
    raise last_exception

# Usage
async def reliable_market_data_fetch(client, symbol):
    async def fetch():
        return await client.get_quote(symbol)
    
    return await retry_with_backoff(
        fetch,
        max_retries=3,
        base_delay=1.0,
        retry_exceptions=(RateLimitError, NetworkError)
    )
```

## Troubleshooting Guide

### Common Issues and Solutions

#### 1. Authentication Issues

**Problem**: `401 Unauthorized` responses

**Solutions**:
```python
# Check token expiration
if client.auth_manager.is_token_expired():
    await client.refresh_token()

# Verify credentials
try:
    await client.login(username, password)
except AuthenticationError as e:
    print(f"Login failed: {e.message}")
    # Check credentials, account status, etc.

# Debug token
token_info = client.decode_token()  # For debugging only
print(f"Token expires at: {token_info['exp']}")
```

#### 2. Rate Limiting

**Problem**: `429 Too Many Requests`

**Solutions**:
```python
# Implement exponential backoff
try:
    result = await client.api_call()
except RateLimitError as e:
    await asyncio.sleep(e.retry_after)
    result = await client.api_call()

# Use batch requests
symbols = ["AAPL", "GOOGL", "MSFT"]
quotes = await client.get_quotes_batch(symbols)  # More efficient

# Implement client-side rate limiting
from asyncio import Semaphore
semaphore = Semaphore(10)  # Max 10 concurrent requests

async def limited_api_call():
    async with semaphore:
        return await client.api_call()
```

#### 3. Network Connectivity

**Problem**: Connection timeouts or network errors

**Solutions**:
```python
# Configure timeouts
client = NautilusClient(
    timeout=30,  # 30 second timeout
    connect_timeout=5,  # 5 second connection timeout
    read_timeout=25  # 25 second read timeout
)

# Use connection pooling
import aiohttp
connector = aiohttp.TCPConnector(
    limit=100,  # Total connection pool size
    limit_per_host=30,  # Connections per host
    keepalive_timeout=30,
    enable_cleanup_closed=True
)

# Health check before requests
if not await client.health_check():
    print("API is not healthy, retrying later...")
    await asyncio.sleep(30)
```

#### 4. Data Inconsistencies

**Problem**: Unexpected data format or missing fields

**Solutions**:
```python
# Validate responses
def validate_market_data(data):
    required_fields = ['symbol', 'price', 'timestamp']
    missing_fields = [field for field in required_fields if field not in data]
    
    if missing_fields:
        raise ValueError(f"Missing required fields: {missing_fields}")
    
    if data['price'] <= 0:
        raise ValueError(f"Invalid price: {data['price']}")
    
    return data

# Use schema validation
from pydantic import BaseModel, validator

class MarketDataResponse(BaseModel):
    symbol: str
    price: float
    timestamp: datetime
    
    @validator('price')
    def price_must_be_positive(cls, v):
        if v <= 0:
            raise ValueError('Price must be positive')
        return v

# Parse with validation
try:
    market_data = MarketDataResponse(**response_data)
except ValidationError as e:
    logger.error(f"Invalid response format: {e}")
```

### Debug Mode

```python
# Enable debug logging
import logging
logging.basicConfig(level=logging.DEBUG)

# Enable request/response logging
client = NautilusClient(debug=True)

# Inspect raw responses
async with client.session.get(url) as response:
    raw_text = await response.text()
    print(f"Raw response: {raw_text}")
    
    headers = dict(response.headers)
    print(f"Response headers: {headers}")
```

## Performance Optimization

### Client-Side Optimization

#### 1. Connection Reuse

```python
# Use session context manager
async with NautilusClient() as client:
    # Reuses the same connection pool
    quote1 = await client.get_quote("AAPL")
    quote2 = await client.get_quote("GOOGL")
    quote3 = await client.get_quote("MSFT")

# Configure connection pooling
client = NautilusClient(
    connection_pool_size=100,
    connection_pool_maxsize=100,
    keepalive_timeout=30
)
```

#### 2. Async Batch Operations

```python
# Concurrent requests
import asyncio

async def fetch_multiple_quotes(symbols):
    async with NautilusClient() as client:
        await client.login(username, password)
        
        # Create tasks for concurrent execution
        tasks = [client.get_quote(symbol) for symbol in symbols]
        
        # Execute concurrently
        quotes = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle results and exceptions
        successful_quotes = []
        for i, quote in enumerate(quotes):
            if isinstance(quote, Exception):
                print(f"Failed to get quote for {symbols[i]}: {quote}")
            else:
                successful_quotes.append(quote)
        
        return successful_quotes

# Usage
symbols = ["AAPL", "GOOGL", "MSFT", "TSLA", "AMZN"]
quotes = await fetch_multiple_quotes(symbols)
```

#### 3. Smart Caching

```python
from functools import lru_cache
import time

class SmartCache:
    def __init__(self, ttl=300):  # 5 minutes TTL
        self.cache = {}
        self.ttl = ttl
    
    def get(self, key):
        if key in self.cache:
            data, timestamp = self.cache[key]
            if time.time() - timestamp < self.ttl:
                return data
            else:
                del self.cache[key]
        return None
    
    def set(self, key, value):
        self.cache[key] = (value, time.time())

cache = SmartCache(ttl=60)  # 1 minute cache for quotes

async def cached_get_quote(client, symbol):
    # Check cache first
    cached_quote = cache.get(f"quote:{symbol}")
    if cached_quote:
        return cached_quote
    
    # Fetch from API
    quote = await client.get_quote(symbol)
    
    # Cache result
    cache.set(f"quote:{symbol}", quote)
    
    return quote
```

### Memory Optimization

```python
# Use generators for large datasets
async def stream_historical_data(client, symbol, days=365):
    """Stream historical data without loading everything into memory"""
    page = 1
    page_size = 100
    
    while True:
        data = await client.get_historical_data(
            symbol=symbol,
            page=page,
            limit=page_size
        )
        
        for record in data.records:
            yield record
        
        if not data.has_next:
            break
            
        page += 1

# Usage
async for data_point in stream_historical_data(client, "AAPL"):
    process_data_point(data_point)  # Process one at a time
```

### Monitoring Performance

```python
import time
from contextlib import asynccontextmanager

@asynccontextmanager
async def measure_time(operation_name: str):
    """Context manager to measure operation duration"""
    start_time = time.perf_counter()
    try:
        yield
    finally:
        duration = time.perf_counter() - start_time
        print(f"{operation_name} took {duration:.2f} seconds")

# Usage
async def performance_test():
    async with measure_time("Login"):
        await client.login(username, password)
    
    async with measure_time("Fetch 100 quotes"):
        symbols = [f"SYMBOL{i}" for i in range(100)]
        quotes = await asyncio.gather(*[
            client.get_quote(symbol) for symbol in symbols
        ])
```

This comprehensive guide provides developers with everything they need to effectively integrate with the Nautilus Trading Platform API while following best practices for security, performance, and reliability.