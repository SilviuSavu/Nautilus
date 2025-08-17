# NautilusTrader Dashboard Backend

FastAPI backend service for the NautilusTrader Dashboard providing dual data source integration, real-time monitoring, and comprehensive market data infrastructure.

## Epic 2.0: Dual Data Source Integration ✅ COMPLETED

### Features

- **🔄 Dual Data Sources**: IB Gateway + YFinance integration for market data redundancy
- **📊 Historical Data Backfill**: PostgreSQL-based data storage with real-time progress tracking  
- **🌐 YFinance Service**: Standalone adapter with rate limiting, caching, and auto-initialization
- **⚡ Real-time Monitoring**: Live status updates for both data sources with 5-second polling
- **🔌 IB Gateway Integration**: Direct connection to Interactive Brokers for live market data
- **💾 PostgreSQL Storage**: Unified schema supporting multiple data sources (3,390+ bars stored)
- **🔐 API Security**: Protected endpoints with API key authentication
- **📈 Dashboard Interface**: Professional card-based status display with improved layout
- **🚀 Auto-initialization**: YFinance automatically starts on backend startup
- **🛡️ Error Handling**: Comprehensive error states and graceful failure handling

## Architecture

```
Frontend (WebSocket) ← FastAPI Backend ← Redis Streams ← NautilusTrader MessageBus
```

The backend acts as a bridge between NautilusTrader's Redis-based MessageBus and the web frontend, providing:

1. **MessageBus Client**: Connects to Redis streams published by NautilusTrader
2. **WebSocket Server**: Broadcasts real-time messages to connected web clients  
3. **REST API**: Provides status and health check endpoints
4. **Connection Management**: Handles reconnection, error recovery, and health monitoring

## Configuration

### Environment Variables

Create a `.env` file in the backend directory:

```env
# Application Settings
ENVIRONMENT=development
DEBUG=true
HOST=0.0.0.0
PORT=8000
CORS_ORIGINS=http://localhost:3000,http://localhost:80

# Redis/MessageBus Settings
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_DB=0
NAUTILUS_STREAM_KEY=nautilus-streams
```

### Settings

The application supports configuration via environment variables or `.env` file:

| Variable | Default | Description |
|----------|---------|-------------|
| `ENVIRONMENT` | development | Application environment |
| `DEBUG` | true | Enable debug mode |
| `HOST` | 0.0.0.0 | Server host |
| `PORT` | 8000 | Server port |
| `CORS_ORIGINS` | localhost:3000,localhost:80 | CORS allowed origins |
| `REDIS_HOST` | localhost | Redis server host |
| `REDIS_PORT` | 6379 | Redis server port |
| `REDIS_DB` | 0 | Redis database number |
| `NAUTILUS_STREAM_KEY` | nautilus-streams | Redis stream key for NautilusTrader messages |

## MessageBus Connection

### Connection Process

1. **Initialization**: Client configured with Redis connection parameters
2. **Connection**: Establishes connection to Redis server
3. **Consumer Group Setup**: Creates Redis consumer group for stream processing
4. **Message Consumption**: Continuously reads from Redis streams
5. **Health Monitoring**: Periodic health checks with automatic reconnection

### Message Format

Messages from NautilusTrader are processed in the following format:

```json
{
  "topic": "market.eurusd.quote",
  "payload": {
    "symbol": "EURUSD",
    "bid": 1.0850,
    "ask": 1.0852,
    "timestamp": 1630000000000000000
  },
  "timestamp": 1630000000000000000,
  "message_type": "market_data"
}
```

### Connection States

- **DISCONNECTED**: No active connection
- **CONNECTING**: Attempting to establish connection
- **CONNECTED**: Successfully connected and consuming messages
- **RECONNECTING**: Connection lost, attempting to reconnect
- **ERROR**: Maximum reconnection attempts exceeded

### Reconnection Logic

- **Exponential Backoff**: Base delay of 1s, doubles with each attempt
- **Maximum Delay**: Capped at 60 seconds
- **Maximum Attempts**: 10 attempts before entering error state
- **Health Checks**: Every 30 seconds when connected

## API Endpoints

### Health Check

```http
GET /health
```

Returns basic health status of the application.

**Response:**
```json
{
  "status": "healthy",
  "environment": "development", 
  "debug": true
}
```

### API Status

```http
GET /api/v1/status
```

Returns API status and feature availability.

**Response:**
```json
{
  "api_version": "1.0.0",
  "status": "operational",
  "features": {
    "websocket": true,
    "messagebus": true,
    "market_data": false,
    "trading": false,
    "portfolio": false
  }
}
```

### MessageBus Status

```http
GET /api/v1/messagebus/status  
```

Returns detailed MessageBus connection status.

**Response:**
```json
{
  "connection_state": "connected",
  "connected_at": "2023-08-27T10:30:00Z",
  "last_message_at": "2023-08-27T10:35:00Z", 
  "reconnect_attempts": 0,
  "error_message": null,
  "messages_received": 1234
}
```

## WebSocket API

### Connection

```javascript
const ws = new WebSocket('ws://localhost:8000/ws');
```

### Message Types

#### Connection Status
```json
{
  "type": "connection",
  "status": "connected", 
  "message": "Connected to Nautilus Trader API"
}
```

#### MessageBus Data
```json
{
  "type": "messagebus",
  "topic": "market.eurusd.quote",
  "payload": {
    "symbol": "EURUSD",
    "bid": 1.0850,
    "ask": 1.0852
  },
  "timestamp": 1630000000000000000,
  "message_type": "market_data"
}
```

## Installation & Development

### Prerequisites

- Python 3.11+
- Redis server
- NautilusTrader instance configured with Redis MessageBus

### Setup

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Configure environment:**
   ```bash
   cp .env.example .env
   # Edit .env with your configuration
   ```

3. **Start Redis (if not running):**
   ```bash
   redis-server
   ```

4. **Run the application:**
   ```bash
   python main.py
   ```

   Or with uvicorn:
   ```bash
   uvicorn main:app --host 0.0.0.0 --port 8000 --reload
   ```

### Docker Development

```bash
# Build development image
docker build -f Dockerfile.dev -t nautilus-backend .

# Run with docker-compose
docker-compose up backend
```

## Testing

### Unit Tests

```bash
pytest tests/test_messagebus_client.py -v
pytest tests/test_main.py -v
```

### Integration Tests

```bash
pytest tests/test_integration.py -v
```

### All Tests

```bash
pytest -v
```

### Test Coverage

```bash
pytest --cov=. --cov-report=html
```

## Performance Requirements

The implementation meets the following performance requirements:

- **Connection Establishment**: < 5 seconds
- **Automatic Reconnection**: < 10 seconds after disconnection
- **Message Processing Latency**: < 10ms per message
- **API Response Time**: < 100ms for status endpoints

## Error Handling

### Connection Errors

- **Redis Unavailable**: Automatic reconnection with exponential backoff
- **Network Issues**: Connection state tracking and health monitoring
- **Authentication Errors**: Logged and reported via status API

### Message Processing Errors

- **Malformed Messages**: Logged and skipped, processing continues
- **JSON Parse Errors**: Gracefully handled without crashing
- **Handler Exceptions**: Isolated to prevent affecting other handlers

### WebSocket Errors  

- **Client Disconnection**: Automatic cleanup of connection resources
- **Broadcast Failures**: Logged with connection state management

## Logging

The application uses Python's built-in logging with the following levels:

- **INFO**: Connection status changes, startup/shutdown events
- **WARNING**: Connection health check failures, retry attempts  
- **ERROR**: Connection failures, message processing errors
- **DEBUG**: Detailed message processing, Redis operations

Log configuration can be adjusted via environment variables or logging configuration files.

## Security Considerations

- **CORS Configuration**: Properly configured for allowed origins
- **Input Validation**: All API inputs validated via Pydantic models
- **Error Disclosure**: Sensitive error details not exposed in API responses
- **Connection Security**: Redis connection supports SSL/TLS when configured

## Monitoring

### Health Checks

- Kubernetes/Docker health checks via `/health` endpoint
- Application-level health monitoring via MessageBus status

### Metrics

Key metrics available through status endpoints:

- Connection uptime and state
- Message processing rate and count
- Reconnection attempt frequency
- Error rates and types

### Alerting

Monitor the following for alerting:

- MessageBus connection state != "connected"
- High reconnection attempt frequency
- API endpoint response times > 100ms
- Error message presence in status

## Troubleshooting

### Common Issues

**Connection Refused**
- Verify Redis server is running
- Check `REDIS_HOST` and `REDIS_PORT` configuration
- Ensure firewall allows Redis port access

**No Messages Received**
- Verify NautilusTrader is publishing to Redis streams
- Check `NAUTILUS_STREAM_KEY` matches NautilusTrader configuration
- Confirm Redis consumer group creation

**High Memory Usage**
- Monitor message queue size during high-frequency data
- Adjust Redis stream trimming configuration
- Review WebSocket client connection management

**WebSocket Disconnections**
- Check CORS configuration for client domain
- Monitor network stability and timeout settings
- Review client-side connection handling

### Debug Mode

Enable debug mode for detailed logging:

```bash
DEBUG=true python main.py
```

### Redis Connection Testing

Test Redis connectivity:

```bash
redis-cli -h localhost -p 6379 ping
```

### Stream Monitoring

Monitor Redis streams:

```bash
redis-cli XINFO STREAM nautilus-streams
redis-cli XREAD STREAMS nautilus-streams $
```