# Claude Code Configuration

This file provides basic configuration and context for Claude Code operations on the Nautilus trading platform.

## Project Overview
- **Purpose**: Trading platform with Interactive Brokers Gateway integration
- **Architecture**: Python backend, React frontend, NautilusTrader core
- **Database**: PostgreSQL for market data storage
- **Real-time data**: Interactive Brokers API integration
- **Historical data**: Interactive Brokers API integration

## Key Technologies
- **Backend**: FastAPI, Python 3.13, SQLAlchemy
- **Frontend**: React, TypeScript, Vite
- **Trading**: NautilusTrader platform (Rust/Python)
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
The platform uses a streamlined data architecture focused on professional-grade market data:

### Data Flow Hierarchy
1. **Primary**: IBKR Gateway → Live market data and historical data
2. **Cache**: PostgreSQL Database → Cached historical data for fast retrieval  

### Data Sources
- **Interactive Brokers (IBKR)**: Professional-grade real-time and historical market data
  - Live market data feeds
  - Historical data with multiple timeframes
  - Multi-asset class support (stocks, options, futures, forex)
  - Primary and only source for all trading operations

### API Endpoints
- `/api/v1/market-data/historical/bars` - Historical data from IBKR
- `/api/v1/ib/backfill` - Manual historical data backfill via IB Gateway
- `/api/v1/historical/backfill/status` - Backfill operation status
- `/api/v1/historical/backfill/stop` - Stop running backfill operations

## Getting Started
1. Ensure Docker and Docker Compose are installed
2. Run `docker-compose up` to start all services
3. Backend available at http://localhost:8001 (containerized)
4. Frontend available at http://localhost:3000

## Testing Framework
- **E2E Testing**: Playwright for browser automation and integration testing
- **Playwright MCP**: Available for advanced test automation via MCP protocol
- **Unit Tests**: Vitest (frontend), pytest (backend)
- **Component Tests**: React Testing Library with Vitest

## Common Commands
- Start backend: `cd backend && python -m uvicorn main:app --reload`
- Start frontend: `cd frontend && npm run dev`
- Run unit tests: `npm test` (frontend) or `pytest` (backend)
- Run Playwright tests: `npx playwright test`
- Run Playwright headed: `npx playwright test --headed`
- Health check: `curl http://localhost:8001/health`

## Playwright Integration
- Playwright tests located in `frontend/tests/e2e/`
- Use Playwright for end-to-end workflow testing
- Playwright MCP server available for advanced automation scenarios
- Browser automation for real user interaction testing