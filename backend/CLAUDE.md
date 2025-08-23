# Backend Module Configuration

This file provides backend-specific configuration and context for the Nautilus trading platform Python API.

## Backend Architecture
- **Framework**: FastAPI for high-performance async API
- **Database**: PostgreSQL with SQLAlchemy ORM
- **Trading Integration**: Interactive Brokers API via ib_insync
- **Data Sources**: Multi-source approach (IBKR + Alpha Vantage + FRED + EDGAR)
- **Message Bus**: Internal event system for real-time communication
- **Authentication**: JWT-based auth with secure middleware
- **Data Processing**: Pandas for market data analysis

## Development Commands (Docker Only)
**IMPORTANT: Backend runs ONLY in Docker container. Do NOT run locally.**
**CRITICAL: Frontend has NO hardcoded values - uses environment variables to connect.**

### Container Commands
- Start backend container: `docker-compose up backend`
- View backend logs: `docker-compose logs backend`
- Execute commands in container: `docker exec -it nautilus-backend [command]`
- Restart backend: `docker-compose restart backend`

### Testing in Container
- Run tests: `docker exec nautilus-backend pytest`
- Run specific test: `docker exec nautilus-backend pytest tests/test_filename.py`
- Run with verbose: `docker exec nautilus-backend python -m pytest tests/ -v`

### Health Checks (All Containerized - Port 8001)
- Backend health: `curl http://localhost:8001/health`
- Unified data sources: `curl http://localhost:8001/api/v1/nautilus-data/health`
- FRED integration: `curl http://localhost:8001/api/v1/nautilus-data/fred/macro-factors`
- Alpha Vantage: `curl "http://localhost:8001/api/v1/nautilus-data/alpha-vantage/search?keywords=AAPL"`
- EDGAR health: `curl http://localhost:8001/api/v1/edgar/health`

### Database Operations (Containerized)
- Access PostgreSQL: `docker exec -it nautilus-postgres psql -U nautilus -d nautilus`
- Access Redis: `docker exec -it nautilus-redis redis-cli`

## API Patterns
- Use FastAPI dependency injection for database sessions
- Implement proper async/await patterns
- Use Pydantic models for request/response validation
- Follow REST conventions for endpoint design
- Implement proper error handling with HTTP status codes

## Database Operations
- **Connection**: PostgreSQL with connection pooling
- **ORM**: SQLAlchemy with async support
- **Models**: Located in `auth/models.py` and service-specific files
- **Migrations**: Handle schema changes through SQLAlchemy
- **Transactions**: Use proper transaction boundaries

## Testing Approach
- **Unit Tests**: pytest with async support
- **Integration Tests**: Test database and external API interactions
- **Fixtures**: Use pytest fixtures for test data setup
- **Mocking**: Mock external services (IB API, Redis) in tests
- **Coverage**: Aim for >85% on business logic

## Data Services Architecture

### Multi-Source Data Flow
1. **IBKR Gateway Integration** (`ib_*.py`)
   - Real-time market data feeds for trading
   - Historical data requests
   - Professional-grade trading data source
   - Primary source for all trading operations

2. **Alpha Vantage Integration** (`alpha_vantage/`)
   - Fundamental and supplementary market data
   - Company financials and earnings data
   - Symbol search and discovery
   - Rate-limited supplementary data source

3. **FRED Integration** (`fred_integration.py`, `fred_routes.py`)
   - Federal Reserve Economic Data for macro-economic factors
   - 32 economic series across 5 categories (Growth, Employment, Inflation, Monetary, Financial)
   - Real-time economic indicators and regime detection
   - Institutional-grade macro factor calculations

4. **EDGAR Integration** (`edgar_routes.py`, `edgar_connector/`)
   - SEC filing data and company fundamental information
   - 7,861+ public company entities with CIK/ticker mapping
   - Real-time access to 10-K, 10-Q, 8-K filings and other SEC forms
   - Company search, financial facts, and regulatory compliance data

5. **Database Layer** (PostgreSQL with TimescaleDB)
   - Cached historical market data
   - Nanosecond precision timestamps
   - Optimized for time-series data queries
   - Primary source for API responses

6. **NautilusTrader Engine Service** (`nautilus_engine_service.py`) **- SPRINT 2 REAL INTEGRATION**
   - Container-in-container orchestration for dynamic NautilusTrader engines
   - Real-time engine lifecycle management (start/stop/restart/health)
   - Resource monitoring and container metrics collection
   - Session-based engine instances with unique container naming
   - **CRITICAL**: Flattened API responses for single container scenarios

### API Endpoints (All Containerized - Port 8001)
```
# System Health
/health                                # Backend health check

# Unified Nautilus Data Integration (NEW)
/api/v1/nautilus-data/health           # All data sources health check
/api/v1/nautilus-data/status           # Integration status overview
/api/v1/nautilus-data/fred/macro-factors # FRED macro factors via Nautilus
/api/v1/nautilus-data/alpha-vantage/search # Alpha Vantage search via Nautilus
/api/v1/nautilus-data/alpha-vantage/quote/{symbol} # Quotes via Nautilus

# Interactive Brokers (via Nautilus)
/api/v1/market-data/historical/bars     # Historical data from IBKR
/api/v1/ib/backfill                    # Manual historical data backfill
/api/v1/historical/backfill/status     # Backfill operation status
/api/v1/historical/backfill/stop       # Stop running backfill operations

# EDGAR SEC Filing Data (Direct Integration)
/api/v1/edgar/health                   # EDGAR API health check
/api/v1/edgar/companies/search?q=      # Search companies by name/ticker
/api/v1/edgar/companies/{cik}/facts    # Get company financial facts
/api/v1/edgar/companies/{cik}/filings  # Get recent company filings
/api/v1/edgar/ticker/{ticker}/resolve  # Resolve ticker to CIK and company name
/api/v1/edgar/ticker/{ticker}/facts    # Get financial facts by ticker
/api/v1/edgar/ticker/{ticker}/filings  # Get filings by ticker
/api/v1/edgar/filing-types             # List supported SEC form types
/api/v1/edgar/statistics               # EDGAR service statistics

# NautilusTrader Engine Management (SPRINT 2 - REAL INTEGRATION)
/api/v1/nautilus/engine/start          # Start real NautilusTrader container
/api/v1/nautilus/engine/stop           # Stop engine (graceful or force)
/api/v1/nautilus/engine/restart        # Restart with current config
/api/v1/nautilus/engine/status         # **FIXED**: Flattened structures for single container
/api/v1/nautilus/engine/config         # Update engine configuration
/api/v1/nautilus/engine/logs           # Real-time engine logs from container
/api/v1/nautilus/engine/health         # Engine health check from running container
/api/v1/nautilus/engine/emergency-stop # Emergency force stop
/api/v1/nautilus/engine/backtest       # Start backtest in dedicated container
```

## Key Services
```
backend/
├── main.py                    # FastAPI application entry
├── auth/                      # Authentication & user management
├── ib_*.py                   # Interactive Brokers integration (primary trading data)
├── alpha_vantage/            # Alpha Vantage integration (supplementary data)
│   ├── config.py             # Configuration and settings
│   ├── http_client.py        # HTTP client with rate limiting
│   ├── models.py             # Pydantic data models
│   ├── service.py            # Data service layer
│   └── routes.py             # FastAPI route definitions
├── fred_integration.py       # FRED economic data integration service
├── fred_routes.py           # FRED API routes and endpoints
├── edgar_routes.py          # EDGAR SEC filing data routes and endpoints
├── edgar_connector/         # EDGAR API integration components
│   ├── api_client.py        # SEC EDGAR API client
│   ├── config.py            # EDGAR configuration settings
│   ├── data_types.py        # SEC data models and types
│   ├── instrument_provider.py # Company/ticker resolution
│   └── utils.py             # EDGAR utility functions
├── market_data_*.py          # Market data processing & unified API
├── historical_data_service.py # Database operations for historical data
├── portfolio_*.py            # Portfolio management
├── strategy_*.py             # Trading strategy execution
├── monitoring_*.py           # System monitoring
├── nautilus_engine_service.py # **SPRINT 2**: Real NautilusTrader container orchestration
├── nautilus_engine_routes.py # Engine API endpoints and request validation
├── Dockerfile.engine         # NautilusTrader container build specification
├── engine_bootstrap.py       # Container entry point for NautilusTrader engines
├── nautilus_trading_node.py  # NautilusTrader TradingNode integration (may be mocked)
└── tests/                    # Test suite
```

## Error Handling
- Use HTTPException for API errors
- Implement proper logging with structured formats
- Handle database connection errors gracefully
- Provide meaningful error messages to clients
- Log errors for debugging without exposing internals

## Security Guidelines
- Never commit secrets or API keys
- Use environment variables for configuration
- Implement proper input validation
- Use parameterized queries to prevent SQL injection
- Validate JWT tokens on protected endpoints

## Integration Patterns - Hybrid Architecture Implementation

### MessageBus Integration Pattern (Event-Driven)
**Used by**: Data.gov, DBnomics  
**Implementation**: Redis Streams + Event Handlers

```python
# MessageBus Service Pattern
class DataSourceMessageBusService:
    def __init__(self):
        self.messagebus_client = get_messagebus_client()
        self._handlers = {
            "source.health_check": self._handle_health_check,
            "source.data.request": self._handle_data_request,
            "source.search": self._handle_search_request,
        }
    
    async def start(self):
        for event_type, handler in self._handlers.items():
            self.messagebus_client.add_message_handler(
                self._create_message_filter(event_type, handler)
            )
```

**File Structure for MessageBus Sources**:
```
backend/
├── [source]_routes.py              # HTTP endpoints (compatibility)
├── [source]_messagebus_routes.py   # Event-triggered endpoints
├── [source]_messagebus_service.py  # Event handlers & pub/sub logic
└── messagebus_client.py           # Shared MessageBus client
```

**Benefits**:
- **Horizontal Scaling**: Multiple workers can process events
- **Fault Tolerance**: Dead letter queues and retry logic
- **Async Processing**: Non-blocking operations for high-volume data
- **Event Traceability**: Full audit trail of data operations

### Direct REST Integration Pattern (Request/Response)
**Used by**: IBKR, Alpha Vantage, FRED, EDGAR, Trading Economics  
**Implementation**: FastAPI + Direct HTTP Clients

```python
# Direct REST Service Pattern
class DirectAPIService:
    def __init__(self):
        self.http_client = httpx.AsyncClient()
        self.rate_limiter = AdvancedRateLimiter()
    
    async def fetch_data(self, params: dict):
        async with self.rate_limiter:
            response = await self.http_client.get(endpoint, params=params)
            return self._process_response(response)
```

**File Structure for Direct Sources**:
```
backend/
├── [source]_routes.py         # FastAPI route definitions
├── [source]_service.py        # Business logic layer  
├── [source]_client.py         # HTTP client with rate limiting
└── [source]_models.py         # Pydantic data models
```

**Benefits**:
- **Low Latency**: Direct HTTP connections, minimal overhead
- **Simple Debugging**: Straightforward request/response flow
- **Rate Control**: Direct management of API quotas
- **Real-time Response**: Immediate data delivery

### Architecture Decision Framework

**Choose MessageBus When**:
```python
# High-volume async processing example
if data_volume > 100_000 and processing_time > 1_sec:
    use_messagebus = True
    pattern = "event_driven"
```

**Choose Direct REST When**:
```python
# Low-latency real-time example  
if latency_requirement < 100_ms or trading_operation:
    use_direct_rest = True
    pattern = "request_response"
```

### Performance Characteristics

**MessageBus Throughput**: 10,000+ events/second per worker  
**Direct REST Latency**: < 50ms average response time  
**Scaling**: MessageBus horizontal, REST vertical (connection pooling)

### Comprehensive Latency Architecture Implementation

#### Connection Layer Performance Specifications

**WebSocket Infrastructure** (Real-time streaming):
```python
# WebSocket Manager Configuration
WEBSOCKET_MAX_CONNECTIONS = 1000
WEBSOCKET_HEARTBEAT_INTERVAL = 30  # seconds
SUBSCRIPTION_TIMEOUT = 300  # seconds
MAX_SUBSCRIPTIONS_PER_CLIENT = 50
CLEANUP_TIMEOUT = 300  # seconds

# Performance Targets (Validated)
MESSAGE_THROUGHPUT = 50_000  # messages/second
AVERAGE_LATENCY = 50  # milliseconds
CONNECTION_LIMIT = 1000  # concurrent connections
```

**Database Connection Pools** (TimescaleDB optimized):
```python
# Pool Strategy Configurations
HIGH_THROUGHPUT = {
    "min_connections": 10,
    "max_connections": 50,
    "command_timeout": 30.0,  # seconds
    "max_inactive_connection_lifetime": 300.0,  # 5 minutes
    "tcp_keepalives_idle": "600",
    "tcp_keepalives_interval": "30",
    "tcp_keepalives_count": "3"
}

BALANCED = {
    "min_connections": 5,
    "max_connections": 20,
    "command_timeout": 60.0,  # seconds
    "max_inactive_connection_lifetime": 600.0,  # 10 minutes
    "tcp_keepalives_idle": "900",
    "tcp_keepalives_interval": "60",
    "tcp_keepalives_count": "2"
}

CONSERVATIVE = {
    "min_connections": 2,
    "max_connections": 10,
    "command_timeout": 120.0,  # seconds
    "max_inactive_connection_lifetime": 1800.0,  # 30 minutes
}
```

**MessageBus Client Configuration** (Redis Streams):
```python
# MessageBus Latency Settings
CONNECTION_TIMEOUT = 5.0  # seconds
RECONNECT_BASE_DELAY = 1.0  # seconds
RECONNECT_MAX_DELAY = 60.0  # seconds
HEALTH_CHECK_INTERVAL = 30.0  # seconds
MAX_RECONNECT_ATTEMPTS = 10

# Performance Characteristics
EVENTS_PER_SECOND_PER_WORKER = 10_000
HORIZONTAL_SCALING = True  # Redis pub/sub
```

**External API Rate Limiting**:
```python
# Alpha Vantage Configuration
ALPHA_VANTAGE_RATE_LIMIT = 5  # calls per minute (free tier)
ALPHA_VANTAGE_TIMEOUT = 30  # seconds
ALPHA_VANTAGE_CACHE_TTL = 300  # 5 minutes

# EDGAR SEC API
EDGAR_RATE_LIMIT = 5  # requests per second
EDGAR_TIMEOUT = 30  # seconds
EDGAR_MAX_CONNECTIONS = 100

# YFinance (if enabled)
YFINANCE_MIN_REQUEST_INTERVAL = 2.0  # seconds
YFINANCE_MAX_DELAY = 30.0  # seconds
YFINANCE_BACKOFF_FACTOR = 1.0
```

#### Trading System Latency Monitoring

**Real-Time Performance Metrics** (from monitoring service):
```python
# Order Execution Latency Distribution
ORDER_EXECUTION_LATENCY = {
    "min_ms": 2.1,
    "avg_ms": 12.3,
    "p50_ms": 10.2,
    "p95_ms": 28.5,
    "p99_ms": 41.2,
    "max_ms": 45.7
}

# Market Data Feed Latency
MARKET_DATA_LATENCY = {
    "tick_to_trade_ms": 5.8,
    "feed_latency_ms": 3.2,
    "processing_latency_ms": 2.6,
    "total_latency_ms": 8.8
}

# Connection Health
CONNECTION_LATENCY = {
    "ping_ms": 15.4,
    "jitter_ms": 2.1,
    "packet_loss_percent": 0.02
}
```

#### Performance Testing & Validation

**Load Testing Configuration**:
```python
# WebSocket Scalability Tests
MAX_CONCURRENT_CONNECTIONS = 1000
MAX_MESSAGES_PER_SECOND = 50_000
LATENCY_REQUIREMENT_MS = 50  # Average latency target
PERFORMANCE_TEST_DURATION = 300  # 5 minutes

# Database Performance Tests  
MAX_DATABASE_CONNECTIONS = 100
QUERY_PERFORMANCE_TARGET_MS = 100
BULK_OPERATION_TARGET_MS = 500
```

**Monitoring & Alerting Thresholds**:
```python
# Performance Alert Thresholds
API_RESPONSE_TIME_ALERT = 200  # milliseconds
DATABASE_QUERY_ALERT = 100  # milliseconds  
WEBSOCKET_LATENCY_ALERT = 50  # milliseconds
CONNECTION_FAILURE_ALERT = 5  # percent
MEMORY_USAGE_ALERT = 80  # percent
CPU_USAGE_ALERT = 75  # percent
```

## Integration Patterns - Legacy
- **IB Gateway**: Use ib_insync for Interactive Brokers API (primary trading data)
- **Alpha Vantage**: HTTP-based API with rate limiting for supplementary data
- **Redis**: Cache frequently accessed data
- **WebSocket**: Real-time data streaming to frontend
- **Database**: Proper connection pooling and async operations

## Data Architecture Best Practices
- **Hybrid approach**: MessageBus for high-volume, Direct REST for low-latency
- **Professional trading focus**: IBKR direct connection for millisecond requirements
- **Event-driven scaling**: MessageBus sources can scale horizontally
- **Database caching**: PostgreSQL with TimescaleDB for optimal time-series performance
- **Rate limiting**: Queue-based (MessageBus) vs Direct control (REST)
- **Real-time priority**: Direct connections for trading, async for analytics

## FRED Economic Data Categories

The FRED integration provides 32+ economic series organized into 5 categories:

### Growth Indicators (6 series)
- GDP, Real GDP, Real Potential GDP
- Industrial Production Index
- Retail Sales, Housing Starts

### Employment & Labor Market (6 series)  
- Unemployment Rate, Nonfarm Payrolls
- Labor Force Participation Rate
- Average Hourly Earnings, Initial Claims, Job Openings

### Inflation & Prices (4 series)
- Consumer Price Index (CPI), Core CPI
- PCE Price Index, Fed Inflation Target

### Monetary Policy & Interest Rates (6 series)
- Federal Funds Rate
- Treasury yields (2Y, 5Y, 10Y, 30Y)
- Money Supply (M2), Monetary Base

### Financial Markets (6 series)
- High Yield Credit Spreads, VIX Volatility
- USD/EUR Exchange Rate, Emerging Markets Spreads
- WTI Oil Price, Gold Price

### Calculated Macro Factors
- Interest rate levels and 30-day changes
- Yield curve slope, level, and curvature  
- Economic activity composite indicators
- Unemployment and employment trends
- Volatility regimes and market stress indicators

## EDGAR SEC Filing Data Integration

The EDGAR integration provides comprehensive access to SEC filing data:

### Company Data & Search
- **7,861+ Public Companies**: Complete database with CIK and ticker mapping
- **Company Search**: Real-time search by company name or ticker symbol
- **Ticker Resolution**: Convert stock tickers to SEC Central Index Keys (CIK)
- **Entity Information**: Company names, tickers, and exchange data

### SEC Filing Types
- **10-K**: Annual reports with comprehensive business overview
- **10-Q**: Quarterly reports with financial statements
- **8-K**: Current reports for material events
- **DEF 14A**: Proxy statements for shareholder meetings
- **S-1**: Registration statements for new securities
- **13F-HR**: Institutional investment manager holdings
- **4, 3, 5**: Insider trading and ownership reports
- **SC 13D/13G**: Beneficial ownership disclosure statements

### Financial Data Access
- **Company Facts**: Key financial metrics from XBRL filings
- **Filing History**: Recent regulatory filings with metadata
- **Real-time Updates**: Direct access to newly published SEC data
- **Regulatory Compliance**: Official SEC data for institutional analysis

### Integration Features
- **Rate Limiting**: Respects SEC API limits (5 requests/second)
- **Caching**: 30-minute cache TTL for optimal performance
- **Error Handling**: Robust handling of SEC API changes
- **Data Validation**: CIK and ticker format validation

## Environment Variables (Configured in Docker)
All environment variables are pre-configured in `docker-compose.yml`:

### API Keys (Already Configured)
- **ALPHA_VANTAGE_API_KEY**: 271AHP91HVAPDRGP (configured)
- **FRED_API_KEY**: 1f1ba9c949e988e12796b7c1f6cce1bf (configured)
- **IB_CLIENT_ID**: 1 (default for Gateway connection)

### Database & Services (Containerized)
- **DATABASE_URL**: postgresql://nautilus:nautilus123@postgres:5432/nautilus
- **REDIS_URL**: redis://redis:6379
- **CORS_ORIGINS**: http://localhost:3000,http://localhost:80

### Container Network
All services communicate via Docker network `nautilus-network`:
- Backend: nautilus-backend:8000 (exposed as localhost:8001)
- Frontend: nautilus-frontend:3000 (exposed as localhost:3000)
- Database: nautilus-postgres:5432 (exposed as localhost:5432)
- Redis: nautilus-redis:6379 (exposed as localhost:6379)

## 🚨 CRITICAL: Sprint 2 Engine API Structure Fix

### Problem: Frontend-Backend API Structure Mismatch
**File**: `nautilus_engine_service.py` 
**Method**: `get_engine_status()` lines 446-456
**Issue**: Backend returned nested structures by session ID, frontend expected flat objects

### The Fix (DO NOT BREAK THIS AGAIN)
```python
async def get_engine_status(self) -> Dict[str, Any]:
    """Get comprehensive engine status"""
    resource_usage = await self._get_resource_usage()
    container_info = await self._get_container_info()
    health_check = await self._health_check()
    
    # CRITICAL FIX: For single container, flatten the nested structure for frontend compatibility
    if len(self.dynamic_containers) == 1:
        session_id = list(self.dynamic_containers.keys())[0]
        if isinstance(resource_usage, dict) and session_id in resource_usage:
            resource_usage = resource_usage[session_id]
        if isinstance(container_info, dict) and session_id in container_info:
            container_info = container_info[session_id]
```

### Expected Frontend API Response Structure
```json
{
  "success": true,
  "status": {
    "state": "running",
    "resource_usage": {
      "cpu_percent": "1.5%",
      "memory_usage": "100MB / 2GB",
      "memory_percent": "4.9%"
    },
    "container_info": {
      "status": "running",
      "running": true,
      "name": "nautilus-engine-xyz"
    }
  }
}
```

### DO NOT RETURN (This Breaks Frontend)
```json
{
  "resource_usage": {
    "session-id-123": {"cpu_percent": "1.5%"}
  },
  "container_info": {  
    "session-id-123": {"status": "running"}
  }
}
```

### Network Configuration Issues to Avoid
- **Wrong Network**: `nautilus-network` (correct: `nautilus_nautilus-network`)
- **Missing CORS**: Must include both port 3000 and 3001
- **IB Import Issues**: Mock `get_nautilus_node_manager()` if import fails
- **Port Changes**: DO NOT change ports (Frontend: 3000, Backend: 8001)