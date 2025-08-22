# Claude Code Configuration

This file provides basic configuration and context for Claude Code operations on the Nautilus trading platform.

## Project Overview
- **Purpose**: Multi-source trading platform with professional data integrations
- **Architecture**: Python backend, React frontend, NautilusTrader core
- **Database**: PostgreSQL for market data storage
- **Real-time data**: Interactive Brokers API + Alpha Vantage API + FRED + EDGAR integration
- **Historical data**: Interactive Brokers API + Alpha Vantage API + FRED + EDGAR integration

## Key Technologies
- **Backend**: FastAPI, Python 3.13, SQLAlchemy
- **Frontend**: React, TypeScript, Vite
- **Trading**: NautilusTrader platform (Rust/Python) - **REAL INTEGRATION COMPLETE (Sprint 2)**
- **Engine Management**: Container-in-container pattern with dynamic NautilusTrader instances
- **Database**: PostgreSQL with nanosecond precision
- **Containerization**: Docker and Docker Compose

## Development Guidelines
- Follow standard coding practices for each language
- Write comprehensive tests for new functionality
- Use proper error handling and logging
- Maintain clean, readable code with good documentation

## Repository Information
- **Location**: https://github.com/SilviuSavu/Nautilus.git
- **Branch**: main
- **License**: MIT

## Data Architecture
The platform uses a multi-source data architecture focused on comprehensive market coverage:

### Data Flow Hierarchy
1. **Primary Trading**: IBKR Gateway → Live trading and professional market data
2. **Supplementary Data**: Alpha Vantage API → Fundamental data and additional market coverage
3. **Economic Data**: FRED API → Macro-economic indicators and regime detection
4. **Regulatory Data**: EDGAR API → SEC filing data and company fundamentals
5. **Cache**: PostgreSQL Database → Cached data for fast retrieval  

### Data Sources
- **Interactive Brokers (IBKR)**: Professional-grade trading and market data
  - Live market data feeds for trading operations
  - Historical data with multiple timeframes
  - Multi-asset class support (stocks, options, futures, forex)
  - Primary source for all trading operations

- **Alpha Vantage**: Comprehensive market and fundamental data
  - Real-time and historical stock quotes
  - Daily and intraday price data (1min-60min intervals)
  - Company fundamental data (earnings, financials, ratios)
  - Symbol search and company overview data
  - Rate-limited: 5 requests/minute, 500 requests/day (free tier)

- **FRED (Federal Reserve Economic Data)**: Institutional-grade macro-economic data
  - 32+ economic indicators across 5 categories
  - Real-time economic regime detection
  - Yield curve analysis and monetary policy indicators
  - Employment, inflation, and growth metrics
  - Market volatility and financial stress indicators

- **EDGAR (SEC Filing Data)**: Comprehensive regulatory and fundamental data
  - 7,861+ public company database with CIK/ticker mapping
  - Real-time SEC filing access (10-K, 10-Q, 8-K, proxy statements)
  - Company search and ticker resolution services
  - Financial facts extraction from XBRL filings
  - Insider trading and institutional holdings data

### API Endpoints

#### Interactive Brokers (IBKR)
- `/api/v1/market-data/historical/bars` - Historical data from IBKR
- `/api/v1/ib/backfill` - Manual historical data backfill via IB Gateway
- `/api/v1/historical/backfill/status` - Backfill operation status
- `/api/v1/historical/backfill/stop` - Stop running backfill operations

#### Alpha Vantage
- `/api/v1/alpha-vantage/health` - Integration health check
- `/api/v1/alpha-vantage/quote/{symbol}` - Real-time stock quotes
- `/api/v1/alpha-vantage/daily/{symbol}` - Daily historical data
- `/api/v1/alpha-vantage/intraday/{symbol}` - Intraday data (1min-60min)
- `/api/v1/alpha-vantage/search` - Symbol search by keywords
- `/api/v1/alpha-vantage/company/{symbol}` - Company fundamental data
- `/api/v1/alpha-vantage/earnings/{symbol}` - Quarterly/annual earnings
- `/api/v1/alpha-vantage/supported-functions` - List available functions

#### FRED Economic Data
- `/api/v1/fred/health` - FRED API health check
- `/api/v1/fred/series` - List all 32+ available economic series
- `/api/v1/fred/series/{series_id}` - Get time series data for specific indicator
- `/api/v1/fred/series/{series_id}/latest` - Get latest value for economic series
- `/api/v1/fred/macro-factors` - Calculate institutional macro factors
- `/api/v1/fred/economic-calendar` - Economic release calendar
- `/api/v1/fred/cache/refresh` - Refresh economic data cache

#### EDGAR SEC Filing Data
- `/api/v1/edgar/health` - EDGAR API health check
- `/api/v1/edgar/companies/search` - Search companies by name/ticker
- `/api/v1/edgar/companies/{cik}/facts` - Get company financial facts
- `/api/v1/edgar/companies/{cik}/filings` - Get recent company filings
- `/api/v1/edgar/ticker/{ticker}/resolve` - Resolve ticker to CIK and company name
- `/api/v1/edgar/ticker/{ticker}/facts` - Get financial facts by ticker
- `/api/v1/edgar/ticker/{ticker}/filings` - Get filings by ticker
- `/api/v1/edgar/filing-types` - List supported SEC form types
- `/api/v1/edgar/statistics` - EDGAR service statistics

#### NautilusTrader Engine Management (Sprint 2 - REAL INTEGRATION)
**🚨 CRITICAL: This is now REAL NautilusTrader integration, NOT mocks**
- `/api/v1/nautilus/engine/start` - Start real NautilusTrader container with live configuration
- `/api/v1/nautilus/engine/stop` - Stop NautilusTrader engine (graceful or force)
- `/api/v1/nautilus/engine/restart` - Restart engine with current configuration
- `/api/v1/nautilus/engine/status` - **FIXED**: Returns flattened structures for single container
- `/api/v1/nautilus/engine/config` - Update engine configuration
- `/api/v1/nautilus/engine/logs` - Get real-time engine logs from container
- `/api/v1/nautilus/engine/health` - Engine health check from running container
- `/api/v1/nautilus/engine/emergency-stop` - Emergency force stop
- `/api/v1/nautilus/engine/backtest` - Start backtest in dedicated container

**Container Architecture**: Container-in-container pattern
- **Base Image**: `nautilus-engine:latest` (NautilusTrader 1.219.0)
- **Network**: `nautilus_nautilus-network`
- **Container Naming**: `nautilus-engine-{session-id}-{timestamp}`
- **Resource Limits**: Configurable CPU/memory limits per engine
- **Session Management**: Dynamic container creation/cleanup per engine instance

## Getting Started (Docker Only)
**IMPORTANT: All services run in Docker containers. Do NOT run services locally.**
**CRITICAL: NO hardcoded values in frontend - all use environment variables.**

1. Ensure Docker and Docker Compose are installed
2. API keys are already configured in docker-compose.yml:
   - ALPHA_VANTAGE_API_KEY=271AHP91HVAPDRGP
   - FRED_API_KEY=1f1ba9c949e988e12796b7c1f6cce1bf
3. Run `docker-compose up` to start all services
4. **Backend**: http://localhost:8001 (containerized only)
5. **Frontend**: http://localhost:3000 (containerized only)
6. **Database**: localhost:5432 (containerized)
7. **Redis**: localhost:6379 (containerized)

### Environment Variables (Pre-configured)
- **VITE_API_BASE_URL**: http://localhost:8001 (frontend → backend)
- **VITE_WS_URL**: localhost:8001 (WebSocket connections)
- **All frontend components**: Use environment variables, NO hardcoded URLs

## Data Provider Configuration
- **IBKR**: Requires Interactive Brokers Gateway connection
- **Alpha Vantage**: Set environment variable `ALPHA_VANTAGE_API_KEY`
- **FRED**: Set environment variable `FRED_API_KEY` (32-character lowercase alphanumeric)
- **EDGAR**: No API key required (uses SEC public API with rate limiting)

## Testing Framework
- **E2E Testing**: Playwright for browser automation and integration testing
- **Playwright MCP**: Available for advanced test automation via MCP protocol
- **Unit Tests**: Vitest (frontend), pytest (backend)
- **Component Tests**: React Testing Library with Vitest

## Common Commands (Docker Only)
**All commands assume Docker containers are running with `docker-compose up`**

### Container Management
- Start all services: `docker-compose up`
- Start in background: `docker-compose up -d`
- Stop all services: `docker-compose down`
- Rebuild containers: `docker-compose up --build`
- View logs: `docker-compose logs [service-name]`

### Testing
- Run frontend tests: `cd frontend && npm test`
- Run backend tests: `docker exec nautilus-backend pytest`
- Run Playwright tests: `cd frontend && npx playwright test`
- Run Playwright headed: `cd frontend && npx playwright test --headed`

### Health Checks (All Containerized)
- System: `curl http://localhost:8001/health`
- Unified data sources: `curl http://localhost:8001/api/v1/nautilus-data/health`
- FRED macro factors: `curl http://localhost:8001/api/v1/nautilus-data/fred/macro-factors`
- Alpha Vantage search: `curl "http://localhost:8001/api/v1/nautilus-data/alpha-vantage/search?keywords=AAPL"`
- EDGAR: `curl http://localhost:8001/api/v1/edgar/health`

### Database Access
- PostgreSQL: `docker exec -it nautilus-postgres psql -U nautilus -d nautilus`
- Redis: `docker exec -it nautilus-redis redis-cli`

### Engine Management (Sprint 2 Real Integration)
- Start engine: `curl -X POST http://localhost:8001/api/v1/nautilus/engine/start -H "Content-Type: application/json" -d '{"config": {"engine_type": "live", "log_level": "INFO", "instance_id": "test-001", "trading_mode": "paper", "max_memory": "2g", "max_cpu": "2.0", "data_catalog_path": "/app/data", "cache_database_path": "/app/cache", "risk_engine_enabled": true}, "confirm_live_trading": false}'`
- Check status: `curl http://localhost:8001/api/v1/nautilus/engine/status`  
- Stop engine: `curl -X POST http://localhost:8001/api/v1/nautilus/engine/stop -H "Content-Type: application/json" -d '{"force": false}'`
- View logs: `curl http://localhost:8001/api/v1/nautilus/engine/logs?lines=50`

## 🚨 CRITICAL FIXES & TROUBLESHOOTING (Sprint 2)

### Frontend-Backend API Structure Mismatch (FIXED)
**Problem**: Frontend shows `TypeError: Cannot read properties of undefined (reading 'replace')`
**Root Cause**: Backend returned nested structures by session ID, frontend expected flat objects
**Fix Location**: `/backend/nautilus_engine_service.py` lines 446-456 in `get_engine_status()`

```python
# CRITICAL FIX: Flatten nested structures for single container scenarios
if len(self.dynamic_containers) == 1:
    session_id = list(self.dynamic_containers.keys())[0]
    if isinstance(resource_usage, dict) and session_id in resource_usage:
        resource_usage = resource_usage[session_id]
    if isinstance(container_info, dict) and session_id in container_info:
        container_info = container_info[session_id]
```

**Before (BROKEN)**:
```json
{
  "resource_usage": {
    "session-id-123": {"cpu_percent": "1.5%", "memory_usage": "100MB"}
  },
  "container_info": {
    "session-id-123": {"status": "running", "name": "nautilus-engine-xyz"}
  }
}
```

**After (FIXED)**:
```json
{
  "resource_usage": {"cpu_percent": "1.5%", "memory_usage": "100MB"},
  "container_info": {"status": "running", "name": "nautilus-engine-xyz"}
}
```

### Network Configuration Issues
- **Docker Network**: MUST use `nautilus_nautilus-network` (not `nautilus-network`)
- **CORS Origins**: Include both port 3000 and 3001 in docker-compose.yml
- **IB Gateway**: Use mocked imports if real nautilus_trading_node fails import

### Container Management Issues
- **Container Cleanup**: Use `docker container prune -f` to clean stopped containers
- **Image Updates**: `docker-compose build backend` to rebuild engine image
- **Health Checks**: Engine containers expose health endpoint on port 8001

### Port Configuration (DO NOT CHANGE)
- **Frontend**: localhost:3000 (Docker container)
- **Backend**: localhost:8001 (Docker container) 
- **Database**: localhost:5432 (Docker container)
- **Frontend Environment**: VITE_API_BASE_URL=http://localhost:8001

## Playwright Integration
- Playwright tests located in `frontend/tests/e2e/`
- Use Playwright for end-to-end workflow testing
- Playwright MCP server available for advanced automation scenarios
- Browser automation for real user interaction testing