# Backend Module Configuration

This file provides backend-specific configuration and context for the Nautilus trading platform Python API.

## Backend Architecture
- **Framework**: FastAPI for high-performance async API
- **Database**: PostgreSQL with SQLAlchemy ORM
- **Trading Integration**: Interactive Brokers API via ib_insync
- **Data Architecture**: Multi-source approach following NautilusTrader patterns
- **Message Bus**: Internal event system for real-time communication
- **Authentication**: JWT-based auth with secure middleware
- **Data Processing**: Pandas for market data analysis

## Development Commands
- Start dev server: `python -m uvicorn main:app --reload --host 0.0.0.0 --port 8000`
- With IB client: `IB_CLIENT_ID=1 DATABASE_URL=postgresql://nautilus:nautilus123@localhost:5432/nautilus python -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload`
- Run tests: `pytest` or `python -m pytest tests/ -v`
- Run specific test: `pytest tests/test_filename.py`
- Database migration: Check SQLAlchemy models and run migrations
- Health check: `curl http://localhost:8000/health`

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

### Primary Data Flow (IBKR → Database → API)
1. **IBKR Gateway Integration** (`ib_*.py`)
   - Real-time market data feeds
   - Historical data requests
   - Professional-grade data source
   - Primary and only data provider for all operations

2. **Database Layer** (PostgreSQL with TimescaleDB)
   - Cached historical market data
   - Nanosecond precision timestamps
   - Optimized for time-series data queries
   - Primary source for API responses

### API Endpoints
```
/api/v1/market-data/historical/bars     # Historical data from IBKR
/api/v1/ib/backfill                    # Manual historical data backfill
/api/v1/historical/backfill/status     # Backfill operation status
/api/v1/historical/backfill/stop       # Stop running backfill operations
```

## Key Services
```
backend/
├── main.py                    # FastAPI application entry
├── auth/                      # Authentication & user management
├── ib_*.py                   # Interactive Brokers integration (primary data)
# YFinance service removed - using IBKR only
├── market_data_*.py          # Market data processing & unified API
├── historical_data_service.py # Database operations for historical data
├── portfolio_*.py            # Portfolio management
├── strategy_*.py             # Trading strategy execution
├── monitoring_*.py           # System monitoring
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

## Integration Patterns
- **IB Gateway**: Use ib_insync for Interactive Brokers API (primary data source)
# YFinance Integration: Completely removed - using IBKR only
- **Message Bus**: Implement event-driven architecture
- **Redis**: Cache frequently accessed data
- **WebSocket**: Real-time data streaming to frontend
- **Database**: Proper connection pooling and async operations

## Data Architecture Best Practices
- **Single-source reliability**: IBKR Gateway as primary and only data source
- **Professional data focus**: IBKR for all trading operations and historical data
- **Database caching**: PostgreSQL with TimescaleDB for optimal time-series performance
- **Real-time priority**: Live market data feeds with historical backfill capabilities