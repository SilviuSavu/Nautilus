# Sprint 3 API Documentation

## Overview

The Sprint 3 API routes provide comprehensive endpoints for advanced trading infrastructure including analytics, risk management, strategy deployment, WebSocket management, and system monitoring.

## Base URL
All endpoints are prefixed with `/api/v1/sprint3`

## API Categories

### 1. Analytics APIs

#### Performance Analytics
- `POST /analytics/performance/analyze` - Comprehensive performance analysis with real-time calculations
- `GET /analytics/portfolio/{portfolio_id}/summary` - Get comprehensive portfolio summary with real-time metrics

#### Risk Analytics  
- `POST /analytics/risk/analyze` - Advanced risk analytics with VaR, stress testing, and exposure analysis

#### Strategy Analytics
- `POST /analytics/strategy/analyze` - Strategy performance analysis and comparison

#### Execution Analytics
- `POST /analytics/execution/analyze` - Trade execution quality analysis

#### Data Aggregation
- `POST /aggregation/create-job` - Create data aggregation job for historical analysis
- `POST /aggregation/run/{job_id}` - Execute aggregation job
- `GET /aggregation/data/{data_type}/{interval}` - Retrieve aggregated analytical data

### 2. Risk Management APIs

#### Risk Limits (CRUD Operations)
- `POST /risk/limits` - Create a new risk limit
- `GET /risk/limits` - List all risk limits with optional portfolio filtering
- `PUT /risk/limits/{limit_id}` - Update an existing risk limit
- `DELETE /risk/limits/{limit_id}` - Delete a risk limit

#### Real-time Risk Monitoring
- `GET /risk/realtime/{portfolio_id}` - Get real-time risk metrics for portfolio
- `POST /risk/monitoring/start` - Start real-time risk monitoring for portfolios
- `POST /risk/monitoring/stop` - Stop real-time risk monitoring

#### Limit Monitoring
- `POST /risk/limits/monitoring/start` - Start real-time limit monitoring
- `POST /risk/limits/monitoring/stop` - Stop real-time limit monitoring

#### Breach Detection and Reporting
- `GET /risk/alerts/{portfolio_id}` - Check for active risk alerts for portfolio
- `GET /risk/breaches/{portfolio_id}` - Get risk breaches for a portfolio
- `POST /risk/pre-trade-check` - Check if a proposed trade would breach any limits

### 3. Strategy Management APIs

#### Strategy Deployment and Management
- `POST /strategy/deploy` - Deploy a strategy to specified environment
- `GET /strategy/deployments` - List strategy deployments with optional filtering
- `GET /strategy/deployments/{deployment_id}` - Get detailed deployment information

#### Version Control Operations
- `POST /strategy/versions` - Create a new strategy version
- `GET /strategy/versions/{strategy_id}` - List all versions of a strategy

#### Rollback and Recovery
- `POST /strategy/rollback` - Rollback strategy to previous version

#### Pipeline Monitoring
- `GET /strategy/statistics` - Get deployment statistics and metrics

### 4. WebSocket Management APIs

#### Connection Management
- `POST /websocket/connections` - Register a new WebSocket connection
- `DELETE /websocket/connections/{connection_id}` - Unregister a WebSocket connection
- `GET /websocket/connections` - Get current WebSocket connection statistics

#### Subscription Management
- `POST /websocket/subscribe` - Subscribe to WebSocket events for real-time updates
- `POST /websocket/subscriptions` - Create a WebSocket topic subscription
- `DELETE /websocket/subscriptions/{subscription_id}` - Delete a WebSocket subscription

#### Message Broadcasting
- `POST /websocket/broadcast` - Broadcast message to WebSocket subscribers

#### Connection Statistics
- `GET /websocket/stats` - Get WebSocket connection and subscription statistics

### 5. System Monitoring APIs

#### System Health and Performance
- `GET /system/health` - Get comprehensive system health status for Sprint 3 components
- `GET /system/metrics` - Get system performance metrics for monitoring
- `GET /system/performance` - Get detailed system performance metrics
- `GET /system/components` - Get status of all Sprint 3 components

#### Alert Management
- `POST /system/alerts` - Create a system alert

## Request/Response Models

### Risk Management Models

#### RiskLimitRequest
```json
{
  "name": "string",
  "portfolio_id": "string",
  "user_id": "string",
  "strategy_id": "string", 
  "limit_type": "var|concentration|position_size|leverage|exposure|drawdown|volatility|notional",
  "threshold_value": 0.0,
  "warning_threshold": 0.0,
  "action": "warn|block|reduce|notify|liquidate|freeze",
  "parameters": {}
}
```

### Strategy Management Models

#### StrategyDeploymentRequest
```json
{
  "strategy_id": "string",
  "version": "string", 
  "target_environment": "development|testing|staging|production",
  "deployment_strategy": "direct|blue_green|canary|rolling",
  "auto_rollback": true,
  "rollback_threshold": 0.05,
  "canary_percentage": 10.0,
  "approval_required": false
}
```

#### VersionCreateRequest
```json
{
  "strategy_id": "string",
  "version": "string",
  "strategy_code": "string",
  "strategy_config": {},
  "description": "string",
  "tags": ["string"]
}
```

### WebSocket Management Models

#### SubscriptionRequest
```json
{
  "connection_id": "string",
  "topic": "string",
  "filters": {}
}
```

#### BroadcastRequest
```json
{
  "topic": "string",
  "message": {},
  "target_connections": ["string"]
}
```

## Key Features

### Authentication and Authorization
- Production-ready authentication middleware
- Role-based access control for sensitive operations
- API key validation

### Rate Limiting and Throttling
- Advanced rate limiting for API protection
- Configurable throttling based on user tiers
- Request queuing for burst handling

### Input Validation
- Comprehensive Pydantic model validation
- Type checking and data sanitization
- Error handling with meaningful messages

### Error Handling
- Consistent error response format
- Detailed error logging
- Graceful degradation for component failures

### Real-time Capabilities
- WebSocket integration for live updates
- Real-time risk monitoring
- Live performance metrics streaming

### Production Features

#### Comprehensive Logging
- Structured logging with correlation IDs
- Performance metrics tracking
- Audit trails for sensitive operations

#### Health Checks
- Component health monitoring
- System performance metrics
- Automated failover capabilities

#### Data Validation
- Input sanitization and validation
- Business logic validation
- Data integrity checks

#### Scalability
- Async/await for high performance
- Background task processing
- Resource pooling and connection management

## Integration Points

### Sprint 3 Components
- **Analytics Aggregator**: Real-time data processing and analysis
- **Risk Monitor**: Continuous portfolio risk assessment
- **Limit Engine**: Dynamic risk limit enforcement
- **Deployment Manager**: Strategy lifecycle management
- **WebSocket Manager**: Real-time communication
- **Subscription Manager**: Event-driven notifications

### External Systems
- **NautilusTrader Engine**: Trading execution and management
- **PostgreSQL**: Data persistence and analytics
- **Redis**: Caching and pub/sub messaging
- **Interactive Brokers**: Live trading data
- **Alpha Vantage**: Market data enrichment

## Security Considerations

### API Security
- HTTPS-only endpoints in production
- Request signing and validation
- SQL injection prevention
- XSS protection headers

### Data Protection
- Sensitive data encryption
- PII data handling compliance
- Audit logging for regulatory requirements

### Access Control
- Fine-grained permissions
- Session management
- Rate limiting per user/endpoint

## Performance Optimizations

### Caching Strategy
- Redis caching for frequently accessed data
- Response caching for static data
- Database query optimization

### Async Processing
- Background jobs for heavy computations
- Non-blocking I/O operations
- Connection pooling

### Resource Management
- Memory-efficient data structures
- Garbage collection optimization
- Connection lifecycle management

## Monitoring and Observability

### Metrics Collection
- API response times and throughput
- Error rates and types
- Resource utilization tracking

### Alerting
- System health alerts
- Performance degradation notifications
- Business logic breach alerts

### Dashboard Integration
- Real-time system status
- Performance trending
- Business metrics visualization

## Usage Examples

### Create Risk Limit
```bash
curl -X POST "http://localhost:8001/api/v1/sprint3/risk/limits" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "Portfolio VaR Limit",
    "portfolio_id": "PORTFOLIO_001",
    "limit_type": "var",
    "threshold_value": 50000.0,
    "warning_threshold": 40000.0,
    "action": "warn"
  }'
```

### Deploy Strategy
```bash
curl -X POST "http://localhost:8001/api/v1/sprint3/strategy/deploy" \
  -H "Content-Type: application/json" \
  -d '{
    "strategy_id": "MOMENTUM_V1",
    "version": "1.2.0",
    "target_environment": "staging",
    "deployment_strategy": "blue_green"
  }'
```

### Get System Health
```bash
curl -X GET "http://localhost:8001/api/v1/sprint3/system/health"
```

## Error Handling

All API endpoints return consistent error responses:

```json
{
  "success": false,
  "error": "Error message",
  "detail": "Detailed error information",
  "timestamp": "2024-01-01T00:00:00Z",
  "request_id": "uuid"
}
```

## Status Codes
- `200`: Success
- `201`: Created
- `400`: Bad Request
- `401`: Unauthorized
- `403`: Forbidden
- `404`: Not Found
- `422`: Validation Error
- `500`: Internal Server Error
- `503`: Service Unavailable

This comprehensive API layer provides production-ready endpoints for all Sprint 3 functionality with proper validation, error handling, authentication, and documentation.