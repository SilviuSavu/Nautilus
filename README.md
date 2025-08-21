# Nautilus Trader Dashboard

A containerized web dashboard for the Nautilus Trader algorithmic trading platform, featuring a React frontend and FastAPI backend with Docker Compose orchestration.

## Quick Start

### Prerequisites

- Docker Engine 20.10+
- Docker Compose 2.0+
- Git

### Setup and Run

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd nautilus-trader-dashboard
   ```

2. **Set up environment variables:**
   ```bash
   cp .env.example .env
   # Optionally edit .env file to customize configuration
   ```

3. **Build and start the development environment:**
   ```bash
   docker-compose up --build
   ```

4. **Access the application:**
   - **Frontend Dashboard**: http://localhost:3000 
   - **Backend API**: http://localhost:8000
   - **API Documentation**: http://localhost:8000/docs
   - **Nginx Proxy**: http://localhost:80
   
   **Note**: Current development setup runs on ports 3000 (frontend) and 8000 (backend)

## Authentication & Login

The application includes a complete authentication system:

### Default Credentials
- **Username**: `admin`
- **Password**: `admin123`

### Authentication Methods
1. **Username/Password**: Standard login form
2. **API Key**: For programmatic access (see API documentation)

### Features
- JWT token-based session management
- Automatic token refresh
- Session persistence across browser restarts
- Protected routes with automatic redirects
- Secure logout functionality

## Financial Charting Integration

The dashboard includes professional financial charting capabilities:

### Current Implementation Status
- ✅ **Backend API**: Real-time market data from Interactive Brokers Gateway
- ✅ **Asset Classes**: Stocks, Forex, Futures, Indices, ETFs
- ✅ **Data Integration**: 124+ historical OHLCV bars successfully retrieved
- ⚠️ **Chart Display**: UI rendering issue requiring resolution
- 🔄 **In Development**: Chart visualization troubleshooting in progress

### Features Implemented
- **TradingView Integration**: Lightweight Charts v4.2.3 library
- **Instrument Selection**: 30+ predefined instruments across multiple asset classes
- **Timeframe Options**: 1m, 5m, 15m, 1h, 4h, 1d intervals
- **Real Market Data**: Multi-source architecture (IBKR primary, YFinance historical supplement)
- **Professional UI**: Integrated chart tab in main dashboard

### Usage
1. Navigate to "Financial Chart" tab in the dashboard
2. Select instrument from dropdown (AAPL, EURUSD, ES, etc.)
3. Choose timeframe (1h default)
4. View real-time financial charts with market data

### Known Issues
- Chart displays as black screen (data retrieval working correctly)
- Requires browser console investigation for rendering issues
- All backend infrastructure and data flow functional

### API Endpoints
- `GET /api/v1/market-data/historical/bars` - Unified historical OHLCV data (IBKR → Cache → YFinance)
- `GET /api/v1/yfinance/status` - YFinance service health and configuration
- `POST /api/v1/yfinance/backfill` - Manual historical data import
- Supports query parameters: symbol, timeframe, asset_class, exchange, currency

## Data Architecture

The platform implements a professional multi-source data architecture following NautilusTrader best practices:

### Data Flow Hierarchy
1. **Primary Source**: Interactive Brokers Gateway
   - Real-time market data feeds
   - Professional-grade historical data
   - Multi-asset class support (stocks, forex, futures, options)
   - Primary source for all trading operations

2. **Cache Layer**: PostgreSQL with TimescaleDB
   - Optimized time-series data storage
   - Nanosecond precision timestamps
   - Fast retrieval for API responses
   - Persistent storage of historical data

3. **Fallback Source**: YFinance Service
   - **Role**: Historical data supplement (not live data)
   - **Use Case**: Extended historical data, symbol coverage gaps
   - **Implementation**: Follows NautilusTrader patterns for data import
   - **Rate Limiting**: 2-second delays, graceful 429 error handling
   - **Architecture**: Historical data importer, not real-time feed

### Data Source Status
- **IBKR Status**: Monitored via Dashboard with connection health indicators
- **YFinance Status**: Available as "operational" fallback service
- **Unified API**: Single endpoint automatically selects best available data source

## Architecture

### Services

The application consists of three main services orchestrated by Docker Compose:

#### Frontend (React + Vite)
- **Port**: 3000
- **Technology**: React 18.3+, TypeScript, Ant Design, Vite
- **Features**: Hot reload, proxy to backend API, WebSocket support
- **Container**: `nautilus-frontend`

#### Backend (FastAPI)
- **Port**: 8001 (containerized) / 8000 (direct)  
- **Technology**: FastAPI, Python 3.11, uvicorn, PostgreSQL, TimescaleDB
- **Data Architecture**: Multi-source approach following NautilusTrader patterns
  - **Primary**: Interactive Brokers Gateway (live + historical data)
  - **Cache**: PostgreSQL with TimescaleDB (optimized time-series storage)  
  - **Fallback**: YFinance service (historical data supplement)
- **Features**: REST API, WebSocket endpoints, auto-reload, graceful data source fallback
- **Container**: `nautilus-backend`

#### Nginx (Reverse Proxy)
- **Port**: 80
- **Technology**: Nginx Alpine
- **Features**: Routes frontend/backend requests, WebSocket proxying
- **Container**: `nautilus-nginx`

### Development Workflow

#### Hot Reload
- **Frontend**: Vite development server with HMR (Hot Module Replacement)
- **Backend**: uvicorn with auto-reload on Python file changes
- **Volumes**: Source code mounted for instant updates

#### API Integration
- Frontend proxies API calls through Vite dev server to backend
- WebSocket connections supported for real-time features
- CORS configured for development environment

## Docker Compose Commands

### Development Commands

```bash
# Start all services
docker-compose up

# Start services in background
docker-compose up -d

# Build and start (rebuild containers)
docker-compose up --build

# Stop all services
docker-compose down

# Stop and remove volumes
docker-compose down -v

# View logs
docker-compose logs -f

# View logs for specific service
docker-compose logs -f frontend
docker-compose logs -f backend
docker-compose logs -f nginx
```

### Individual Service Management

```bash
# Start only frontend
docker-compose up frontend

# Start frontend and backend (without nginx)
docker-compose up frontend backend

# Rebuild specific service
docker-compose build frontend
docker-compose up --no-deps frontend
```

### Development Utilities

```bash
# Access container shell
docker-compose exec frontend sh
docker-compose exec backend bash

# Run npm commands in frontend
docker-compose exec frontend npm run test
docker-compose exec frontend npm run lint

# Install new packages
docker-compose exec frontend npm install <package>
docker-compose exec backend pip install <package>
```

## Environment Configuration

### Environment Files

- **`.env.example`**: Template with all available variables
- **`.env.development`**: Development-specific configuration  
- **`.env.production`**: Production-specific configuration
- **`.env`**: Local environment file (copy from .env.example)

### Key Environment Variables

#### Frontend (VITE_ prefixed)
```bash
VITE_API_BASE_URL=http://localhost:8002
VITE_WS_URL=ws://localhost:8002/ws
VITE_ENV=development
VITE_DEBUG=true
```

#### Backend
```bash
ENVIRONMENT=development
DEBUG=true
HOST=0.0.0.0
PORT=8002
CORS_ORIGINS=http://localhost:3001,http://localhost:80
```

#### Development Tools
```bash
NODE_ENV=development
CHOKIDAR_USEPOLLING=true  # For file watching in containers
RELOAD=true               # Enable backend auto-reload
```

## API Endpoints

### Health and Status
- `GET /health` - Health check endpoint
- `GET /api/v1/status` - API status and feature availability
- `GET /` - Root endpoint with API information

### WebSocket
- `WS /ws` - WebSocket endpoint for real-time communication

### API Documentation
- `GET /docs` - Interactive Swagger UI documentation
- `GET /redoc` - Alternative ReDoc documentation

## Testing

### Frontend Testing

```bash
# Run tests
docker-compose exec frontend npm run test

# Run tests with UI
docker-compose exec frontend npm run test:ui

# Run tests with coverage
docker-compose exec frontend npm run test:coverage

# Lint code
docker-compose exec frontend npm run lint
```

### Backend Testing

```bash
# Access backend container
docker-compose exec backend bash

# Install test dependencies (if not in requirements.txt)
pip install pytest pytest-asyncio httpx

# Run tests (when implemented)
pytest
```

## Production Deployment

### Environment Setup

1. **Copy production environment:**
   ```bash
   cp .env.production .env
   ```

2. **Update production variables:**
   ```bash
   # Edit .env with your production domain and settings
   VITE_API_BASE_URL=https://yourdomain.com
   VITE_WS_URL=wss://yourdomain.com/ws
   CORS_ORIGINS=https://yourdomain.com
   ```

3. **Use production compose file:**
   ```bash
   docker-compose -f docker-compose.yml -f docker-compose.prod.yml up -d
   ```

### Production Considerations

- Use environment-specific Docker files
- Configure SSL/TLS certificates for nginx
- Set up proper logging and monitoring
- Configure database persistence
- Implement proper secret management
- Set up CI/CD pipeline

## Troubleshooting

### Common Issues

#### Port Conflicts
```bash
# Check if ports are in use
netstat -tulpn | grep :3000
netstat -tulpn | grep :8000
netstat -tulpn | grep :80

# Stop conflicting services or change ports in docker-compose.yml
```

#### Permission Issues
```bash
# Fix file permissions (Linux/Mac)
sudo chown -R $USER:$USER .

# For Windows, ensure Docker Desktop has access to project directory
```

#### Container Build Failures
```bash
# Clean Docker cache
docker system prune -a

# Rebuild without cache
docker-compose build --no-cache

# Check Docker daemon status
docker version
```

#### Hot Reload Not Working
```bash
# Ensure CHOKIDAR_USEPOLLING=true in environment
# Verify volume mounts in docker-compose.yml
# Check file permissions
```

#### Network Connectivity Issues
```bash
# Check container networking
docker network ls
docker network inspect nautilus_nautilus-network

# Verify service communication
docker-compose exec frontend ping backend
docker-compose exec backend ping frontend
```

### Logs and Debugging

```bash
# View all service logs
docker-compose logs -f

# View specific service logs with timestamps
docker-compose logs -f -t frontend

# Follow logs in real-time
docker-compose logs -f --tail=100

# Debug container issues
docker-compose exec frontend env  # Check environment variables
docker-compose exec backend ps aux  # Check running processes
```

### Health Checks

```bash
# Check backend health
curl http://localhost:8000/health

# Check API status
curl http://localhost:8000/api/v1/status

# Check frontend availability
curl http://localhost:3000

# Check nginx proxy
curl http://localhost:80
```

### IB Gateway and Backfill Issues

#### Backfill Process Hanging After Disconnect

**Problem**: Historical data backfill process hangs indefinitely after IB Gateway disconnection, causing resource waste and preventing reconnection.

**Symptoms**:
- Backfill shows `is_running: true` but no progress
- Repeated "Not connected to IB Gateway" errors in logs
- Client ID conflicts when trying to reconnect
- Process won't stop with `/api/v1/historical/backfill/stop`

**Solution** (Fixed in CORE RULE #14):
```bash
# 1. Check current backfill status
curl http://localhost:8000/api/v1/historical/backfill/status

# 2. If hanging, restart the backend process
# The new code automatically detects disconnects and stops gracefully

# 3. Verify the fix is working:
# - Start backfill while connected
# - Disconnect IB Gateway  
# - Backfill should stop within 5 seconds with "IB Gateway disconnected" message
```

**Prevention**:
- Updated backfill service with disconnect detection
- Connection verification before each data request
- Exponential backoff for temporary errors
- Proper progress tracking with disconnect errors

#### IB Gateway Client ID Conflicts

**Problem**: "IB Code 326: Unable to connect as the client id is already in use"

**Solution**:
```bash
# Use different client IDs for different processes
IB_CLIENT_ID=1 python3 -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload

# For multiple instances, use different IDs:
# Backend: IB_CLIENT_ID=1
# Backfill: IB_CLIENT_ID=2  
# Testing: IB_CLIENT_ID=3
```

#### Instrument Search Issues

**Problem**: Searching for stocks like "PLTR" returns currency pairs instead

**Solution** (Fixed):
- Frontend now defaults to `sec_type=STK` for stock-only results
- Backend filtering logic properly handles security type filters
- API endpoint: `/api/v1/ib/instruments/search/PLTR?sec_type=STK`

## Development Guidelines

### Code Standards
- Follow TypeScript best practices for frontend
- Use Python type hints and follow PEP 8 for backend
- Implement proper error handling
- Write comprehensive tests

### Container Best Practices
- Use multi-stage builds for production
- Minimize image layers
- Use .dockerignore files
- Implement health checks

### Security Considerations
- Never commit .env files
- Use secrets management for production
- Implement proper CORS configuration
- Validate all inputs
- Use HTTPS in production

## License

[Add license information here]

## Contributing

[Add contributing guidelines here]