# Technology Stack

## Core Technologies

### Backend Engines & Services
- **Python 3.13**: Primary backend language for 11 specialized engines
- **Rust**: High-performance NautilusTrader core components
- **Cython**: Python/C integration layer for performance-critical paths
- **FastAPI**: Async API framework for engine orchestration
- **SQLAlchemy**: ORM for database operations across engines
- **Pydantic**: Data validation and engine configuration management

### Frontend Architecture
- **TypeScript**: Primary frontend language with strict typing
- **React 18**: UI framework with concurrent features
- **Vite**: Build tool optimized for development speed
- **Zustand**: Global state management for engine status
- **Ant Design**: UI component library for trading interfaces

### Infrastructure & Container Orchestration
- **Docker**: Multi-container engine architecture
- **Docker Compose**: Engine lifecycle orchestration
- **Nginx**: Reverse proxy for multi-engine routing
- **Container-in-Container**: Dynamic NautilusTrader engine deployment

### Communication Architecture
- **Redis Streams**: MessageBus backbone for engine communication
- **WebSocket (Enterprise-grade)**: 1000+ concurrent connections, 50k+ msg/sec
- **Redis Pub/Sub**: Horizontal scaling for real-time messaging  
- **HTTP/REST**: Direct engine-to-engine communication for low-latency operations
- **MessageBus Client**: Async event-driven communication layer

### Data Storage & Processing
- **PostgreSQL + TimescaleDB**: Time-series optimization for market data
- **Redis Cache**: Multi-tier caching (tick/quote/OHLCV) with TTL policies
- **Parquet**: Data persistence format for analytics engines
- **Apache Arrow**: In-memory columnar data processing

### Machine Learning & AI
- **Scikit-learn**: ML model training and inference engine
- **NumPy/Pandas**: Numerical computing for feature engineering
- **TA-Lib**: Technical analysis indicators (200+ indicators)
- **PyTorch**: Advanced ML models for market regime detection
- **Feature Engineering Pipeline**: Real-time factor calculation (380k+ factors)

### Data Source Connectors
- **Interactive Brokers API**: Primary trading data connector
- **Alpha Vantage SDK**: Market data and fundamentals connector  
- **FRED API Client**: Economic data connector (32+ macro indicators)
- **SEC EDGAR Connector**: Regulatory filing data (7,861+ companies)
- **Data.gov Connector**: Federal datasets (346,000+ datasets)
- **DBnomics Connector**: Statistical data (800M+ time series)
- **Trading Economics API**: Global economic indicators (300k+ indicators)
- **Yahoo Finance**: Backup market data connector

### Real-Time Processing
- **asyncio**: Asynchronous I/O for engine coordination
- **uvloop**: High-performance async event loop
- **Redis Streams**: Event sourcing and message durability
- **WebSocket Manager**: Connection pooling and subscription management
- **Event Dispatcher**: Real-time event routing across engines

### Monitoring & Observability
- **Prometheus**: Metrics collection from all 11 engines
- **Grafana**: Real-time dashboards and visualization
- **Structured Logging**: JSON logging across all engine components
- **Health Check Framework**: Engine-level health monitoring
- **Performance Metrics**: Sub-second latency tracking

### Security & Authentication
- **JWT**: Token-based authentication for engine access
- **CORS**: Cross-origin resource sharing for frontend-backend communication
- **Rate Limiting**: Advanced rate limiting per engine and data source
- **Input Validation**: Comprehensive validation across all engine APIs

### Testing & Quality Assurance
- **pytest**: Python testing framework for backend engines
- **Vitest**: Frontend testing with React Testing Library
- **Playwright**: End-to-end testing for full engine workflows
- **Load Testing**: 1000+ concurrent WebSocket connection validation
- **Integration Testing**: Cross-engine communication testing

### Development & CI/CD
- **Strategy Deployment Engine**: Automated CI/CD pipeline
- **Docker Build Pipeline**: Multi-stage container building
- **Automated Testing**: Pre-deployment validation
- **Version Control Integration**: Git-based strategy versioning
- **Rollback Automation**: Performance-based automatic rollbacks

### Performance Characteristics by Engine Tier
- **Tier 1 (Trading)**: <10ms latency - Python/Rust hybrid
- **Tier 2 (Analytics/Risk)**: <1s processing - Python with NumPy acceleration
- **Tier 3 (AI/ML/Streaming)**: <100ms inference - Python with C extensions
- **Tier 4 (Deployment/Data)**: Batch processing - Python with async I/O

### Database Technologies by Use Case
- **TimescaleDB**: Market data storage with nanosecond precision
- **Redis**: Real-time caching and message queuing
- **PostgreSQL**: Transactional data and engine configuration
- **In-Memory Caching**: Factor computation results and ML model cache