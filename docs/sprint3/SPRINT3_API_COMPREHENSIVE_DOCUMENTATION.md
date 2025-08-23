# Sprint 3 API Comprehensive Documentation

## Overview

This document provides detailed documentation for all Sprint 3 API endpoints, including comprehensive examples, detailed request/response schemas, authentication requirements, and integration patterns.

## Table of Contents

1. [Authentication & Authorization](#authentication--authorization)
2. [Analytics APIs](#analytics-apis)
3. [Risk Management APIs](#risk-management-apis)
4. [Strategy Management APIs](#strategy-management-apis)
5. [WebSocket Management APIs](#websocket-management-apis)
6. [System Monitoring APIs](#system-monitoring-apis)
7. [Performance Metrics APIs](#performance-metrics-apis)
8. [Error Handling](#error-handling)
9. [Integration Examples](#integration-examples)
10. [Rate Limiting](#rate-limiting)

## Base Configuration

### Base URLs
- **Development**: `http://localhost:8001`
- **Staging**: `https://staging-api.nautilus.com`
- **Production**: `https://api.nautilus.com`

### Common Headers
```http
Content-Type: application/json
Authorization: Bearer <jwt_token>
X-API-Key: <api_key>
X-Request-ID: <unique_request_id>
```

## Authentication & Authorization

### API Key Authentication
```bash
curl -X GET "${BASE_URL}/api/v1/sprint3/system/health" \
  -H "X-API-Key: your-api-key-here"
```

### JWT Bearer Token
```bash
curl -X POST "${BASE_URL}/api/v1/sprint3/risk/limits" \
  -H "Authorization: Bearer ${JWT_TOKEN}" \
  -H "Content-Type: application/json"
```

### User Roles and Permissions
- **ADMIN**: Full access to all endpoints
- **TRADER**: Trading and portfolio operations
- **RISK_MANAGER**: Risk management and monitoring
- **ANALYST**: Analytics and reporting
- **VIEWER**: Read-only access

## Analytics APIs

### 1. Performance Analytics

#### POST /api/v1/sprint3/analytics/performance/analyze
Comprehensive performance analysis with real-time calculations.

**Request Schema:**
```json
{
  "portfolio_id": "string",
  "start_date": "2024-01-01T00:00:00Z",
  "end_date": "2024-12-31T23:59:59Z",
  "benchmark": "SPY",
  "include_attribution": true,
  "risk_free_rate": 0.05,
  "metrics": [
    "sharpe_ratio",
    "sortino_ratio",
    "max_drawdown",
    "var_95",
    "beta",
    "alpha",
    "tracking_error",
    "information_ratio"
  ]
}
```

**Response Schema:**
```json
{
  "success": true,
  "data": {
    "portfolio_id": "PORTFOLIO_001",
    "period": {
      "start": "2024-01-01T00:00:00Z",
      "end": "2024-12-31T23:59:59Z"
    },
    "performance_metrics": {
      "total_return": 0.1547,
      "annualized_return": 0.1234,
      "volatility": 0.0876,
      "sharpe_ratio": 1.4567,
      "sortino_ratio": 1.7892,
      "max_drawdown": -0.0543,
      "var_95": -0.0234,
      "cvar_95": -0.0345,
      "beta": 0.9876,
      "alpha": 0.0123,
      "tracking_error": 0.0456,
      "information_ratio": 0.2789
    },
    "attribution": {
      "sector_attribution": {...},
      "security_attribution": {...},
      "style_attribution": {...}
    },
    "time_series": {
      "daily_returns": [...],
      "cumulative_returns": [...],
      "rolling_sharpe": [...]
    }
  },
  "metadata": {
    "calculation_time_ms": 145,
    "data_points": 252,
    "benchmark_data": "SPY"
  }
}
```

**Usage Example:**
```python
import requests

response = requests.post(
    f"{BASE_URL}/api/v1/sprint3/analytics/performance/analyze",
    headers=headers,
    json={
        "portfolio_id": "PORTFOLIO_001",
        "start_date": "2024-01-01T00:00:00Z",
        "end_date": "2024-12-31T23:59:59Z",
        "benchmark": "SPY",
        "include_attribution": True
    }
)
performance_data = response.json()
```

#### GET /api/v1/sprint3/analytics/portfolio/{portfolio_id}/summary
Get comprehensive portfolio summary with real-time metrics.

**Query Parameters:**
- `include_positions`: boolean (default: true)
- `include_performance`: boolean (default: true)
- `include_risk`: boolean (default: true)
- `currency`: string (default: USD)

**Response Schema:**
```json
{
  "success": true,
  "data": {
    "portfolio_id": "PORTFOLIO_001",
    "name": "Equity Growth Portfolio",
    "total_value": 1000000.00,
    "cash_balance": 50000.00,
    "invested_value": 950000.00,
    "unrealized_pnl": 25000.00,
    "realized_pnl": 15000.00,
    "positions": [
      {
        "symbol": "AAPL",
        "quantity": 1000,
        "market_value": 175000.00,
        "unrealized_pnl": 5000.00,
        "weight": 0.175
      }
    ],
    "performance": {
      "daily_return": 0.0123,
      "mtd_return": 0.0456,
      "ytd_return": 0.1234
    },
    "risk": {
      "var_95": -15000.00,
      "beta": 0.95,
      "tracking_error": 0.045
    }
  }
}
```

### 2. Risk Analytics

#### POST /api/v1/sprint3/analytics/risk/analyze
Advanced risk analytics with VaR, stress testing, and exposure analysis.

**Request Schema:**
```json
{
  "portfolio_id": "string",
  "risk_models": ["parametric", "historical", "monte_carlo"],
  "confidence_levels": [0.95, 0.99],
  "holding_period": 1,
  "stress_scenarios": [
    {
      "name": "2008_crisis",
      "shocks": {
        "SPY": -0.20,
        "TLT": 0.05
      }
    }
  ],
  "include_correlations": true
}
```

**Response Schema:**
```json
{
  "success": true,
  "data": {
    "portfolio_id": "PORTFOLIO_001",
    "var_calculations": {
      "parametric": {
        "var_95": -25000.00,
        "var_99": -35000.00
      },
      "historical": {
        "var_95": -27000.00,
        "var_99": -38000.00
      },
      "monte_carlo": {
        "var_95": -26500.00,
        "var_99": -36500.00
      }
    },
    "stress_test_results": [
      {
        "scenario": "2008_crisis",
        "portfolio_impact": -185000.00,
        "position_impacts": {...}
      }
    ],
    "exposure_analysis": {
      "sector_exposure": {...},
      "geographic_exposure": {...},
      "currency_exposure": {...}
    },
    "correlation_matrix": {...}
  }
}
```

### 3. Strategy Analytics

#### POST /api/v1/sprint3/analytics/strategy/analyze
Strategy performance analysis and comparison.

**Request Schema:**
```json
{
  "strategy_ids": ["STRATEGY_001", "STRATEGY_002"],
  "benchmark": "SPY",
  "period": {
    "start": "2024-01-01T00:00:00Z",
    "end": "2024-12-31T23:59:59Z"
  },
  "analysis_type": "comparative",
  "include_factor_analysis": true
}
```

## Risk Management APIs

### 1. Risk Limits Management

#### POST /api/v1/sprint3/risk/limits
Create a new risk limit with comprehensive validation.

**Request Schema:**
```json
{
  "name": "Portfolio VaR Limit",
  "description": "Maximum daily VaR exposure for equity portfolio",
  "portfolio_id": "PORTFOLIO_001",
  "user_id": "USER_001",
  "strategy_id": "STRATEGY_001",
  "limit_type": "var",
  "threshold_value": 50000.0,
  "warning_threshold": 40000.0,
  "currency": "USD",
  "action": "warn",
  "escalation_actions": ["notify_risk_manager", "reduce_positions"],
  "parameters": {
    "confidence_level": 0.95,
    "holding_period": 1,
    "lookback_period": 252
  },
  "schedule": {
    "active_hours": {
      "start": "09:30",
      "end": "16:00",
      "timezone": "America/New_York"
    },
    "active_days": ["monday", "tuesday", "wednesday", "thursday", "friday"]
  }
}
```

**Response Schema:**
```json
{
  "success": true,
  "data": {
    "limit_id": "LIMIT_001",
    "name": "Portfolio VaR Limit",
    "status": "active",
    "created_at": "2024-01-01T00:00:00Z",
    "created_by": "USER_001",
    "current_value": 35000.0,
    "utilization": 0.70,
    "last_check": "2024-01-01T12:00:00Z",
    "next_check": "2024-01-01T12:05:00Z"
  }
}
```

#### GET /api/v1/sprint3/risk/limits
List all risk limits with comprehensive filtering.

**Query Parameters:**
- `portfolio_id`: Filter by portfolio
- `strategy_id`: Filter by strategy
- `limit_type`: Filter by limit type
- `status`: Filter by status (active, inactive, breached)
- `user_id`: Filter by user
- `page`: Pagination page number
- `page_size`: Number of items per page
- `sort_by`: Sort field (created_at, name, threshold_value)
- `sort_order`: Sort direction (asc, desc)

**Response Schema:**
```json
{
  "success": true,
  "data": {
    "limits": [...],
    "pagination": {
      "total_count": 150,
      "page": 1,
      "page_size": 50,
      "total_pages": 3
    }
  }
}
```

### 2. Real-time Risk Monitoring

#### GET /api/v1/sprint3/risk/realtime/{portfolio_id}
Get real-time risk metrics for portfolio.

**Query Parameters:**
- `metrics`: Comma-separated list of metrics to include
- `include_positions`: Include position-level risk
- `currency`: Risk reporting currency

**Response Schema:**
```json
{
  "success": true,
  "data": {
    "portfolio_id": "PORTFOLIO_001",
    "timestamp": "2024-01-01T12:00:00Z",
    "risk_metrics": {
      "var_95": 25000.00,
      "var_99": 35000.00,
      "expected_shortfall": 42000.00,
      "beta": 0.95,
      "tracking_error": 0.045,
      "concentration_risk": 0.25
    },
    "limit_status": {
      "total_limits": 5,
      "breached_limits": 0,
      "warning_limits": 1
    },
    "position_risks": [
      {
        "symbol": "AAPL",
        "var_contribution": 5000.00,
        "concentration": 0.175
      }
    ]
  }
}
```

## Strategy Management APIs

### 1. Strategy Deployment

#### POST /api/v1/sprint3/strategy/deploy
Deploy a strategy with comprehensive deployment options.

**Request Schema:**
```json
{
  "strategy_id": "MOMENTUM_V1",
  "version": "1.2.0",
  "target_environment": "staging",
  "deployment_strategy": "blue_green",
  "auto_rollback": true,
  "rollback_threshold": 0.05,
  "canary_percentage": 10.0,
  "approval_required": false,
  "configuration_override": {
    "risk_budget": 100000,
    "max_position_size": 50000
  },
  "notifications": {
    "on_success": ["email:trader@company.com"],
    "on_failure": ["slack:#trading-alerts"],
    "on_rollback": ["email:risk@company.com"]
  }
}
```

**Response Schema:**
```json
{
  "success": true,
  "data": {
    "deployment_id": "DEPLOY_001",
    "status": "in_progress",
    "strategy_id": "MOMENTUM_V1",
    "version": "1.2.0",
    "target_environment": "staging",
    "deployment_strategy": "blue_green",
    "started_at": "2024-01-01T12:00:00Z",
    "estimated_completion": "2024-01-01T12:15:00Z",
    "pipeline_url": "https://pipeline.nautilus.com/deployments/DEPLOY_001"
  }
}
```

## WebSocket Management APIs

### 1. Connection Management

#### POST /api/v1/sprint3/websocket/connections
Register a new WebSocket connection.

**Request Schema:**
```json
{
  "connection_id": "CONN_001",
  "user_id": "USER_001",
  "connection_type": "dashboard",
  "metadata": {
    "ip_address": "192.168.1.100",
    "user_agent": "Mozilla/5.0...",
    "session_id": "SESSION_001"
  }
}
```

### 2. Subscription Management

#### POST /api/v1/sprint3/websocket/subscriptions
Create a WebSocket topic subscription.

**Request Schema:**
```json
{
  "connection_id": "CONN_001",
  "topic": "portfolio.updates",
  "filters": {
    "portfolio_id": "PORTFOLIO_001",
    "event_types": ["position_update", "trade_execution", "risk_alert"]
  },
  "rate_limit": {
    "max_messages_per_second": 10,
    "burst_limit": 50
  }
}
```

## System Monitoring APIs

### 1. System Health

#### GET /api/v1/sprint3/system/health
Comprehensive system health status for all Sprint 3 components.

**Response Schema:**
```json
{
  "success": true,
  "data": {
    "overall_status": "healthy",
    "timestamp": "2024-01-01T12:00:00Z",
    "components": {
      "analytics_service": {
        "status": "healthy",
        "response_time_ms": 45,
        "last_check": "2024-01-01T12:00:00Z",
        "metrics": {
          "requests_per_minute": 150,
          "error_rate": 0.001,
          "memory_usage": 0.65
        }
      },
      "risk_service": {
        "status": "healthy",
        "response_time_ms": 32,
        "active_monitors": 15,
        "processed_checks": 1200
      },
      "websocket_service": {
        "status": "healthy",
        "active_connections": 45,
        "messages_per_minute": 850,
        "connection_success_rate": 0.998
      },
      "database": {
        "status": "healthy",
        "connection_pool_usage": 0.40,
        "query_response_time_ms": 15
      }
    }
  }
}
```

## Performance Metrics APIs

### 1. System Performance

#### GET /api/v1/sprint3/system/performance
Detailed system performance metrics.

**Query Parameters:**
- `component`: Filter by component name
- `metric_type`: Filter by metric type
- `time_range`: Time range for metrics (1h, 24h, 7d, 30d)
- `aggregation`: Aggregation method (avg, max, min, sum)

**Response Schema:**
```json
{
  "success": true,
  "data": {
    "time_range": "24h",
    "metrics": {
      "api_performance": {
        "avg_response_time_ms": 125,
        "requests_per_minute": 450,
        "error_rate": 0.002,
        "p95_response_time_ms": 300,
        "p99_response_time_ms": 800
      },
      "websocket_performance": {
        "active_connections": 45,
        "messages_per_second": 850,
        "connection_latency_ms": 15,
        "reconnection_rate": 0.001
      },
      "system_resources": {
        "cpu_usage": 0.35,
        "memory_usage": 0.65,
        "disk_usage": 0.25,
        "network_io_mbps": 125
      }
    }
  }
}
```

## Error Handling

### Error Response Format
All API endpoints return consistent error responses with detailed information:

```json
{
  "success": false,
  "error": {
    "code": "VALIDATION_ERROR",
    "message": "Invalid request parameters",
    "details": [
      {
        "field": "threshold_value",
        "message": "Must be a positive number",
        "value": -1000
      }
    ]
  },
  "metadata": {
    "timestamp": "2024-01-01T12:00:00Z",
    "request_id": "req_123456789",
    "endpoint": "/api/v1/sprint3/risk/limits",
    "method": "POST"
  }
}
```

### Common Error Codes
- `VALIDATION_ERROR`: Request validation failed
- `AUTHENTICATION_ERROR`: Authentication required or failed
- `AUTHORIZATION_ERROR`: Insufficient permissions
- `RESOURCE_NOT_FOUND`: Requested resource not found
- `RATE_LIMIT_EXCEEDED`: Too many requests
- `INTERNAL_ERROR`: Server error
- `SERVICE_UNAVAILABLE`: Service temporarily unavailable

## Rate Limiting

### Rate Limit Headers
All responses include rate limiting information:

```http
X-RateLimit-Limit: 1000
X-RateLimit-Remaining: 999
X-RateLimit-Reset: 1640995200
X-RateLimit-RetryAfter: 60
```

### Rate Limit Tiers
- **Free Tier**: 100 requests/hour
- **Basic Tier**: 1,000 requests/hour
- **Professional Tier**: 10,000 requests/hour
- **Enterprise Tier**: Unlimited

## Integration Examples

### Python SDK Example
```python
from nautilus_sprint3_sdk import NautilusClient

# Initialize client
client = NautilusClient(
    base_url="http://localhost:8001",
    api_key="your-api-key",
    timeout=30
)

# Performance analysis
performance = client.analytics.analyze_performance(
    portfolio_id="PORTFOLIO_001",
    start_date="2024-01-01",
    end_date="2024-12-31",
    benchmark="SPY"
)

# Risk monitoring
risk_metrics = client.risk.get_realtime_metrics(
    portfolio_id="PORTFOLIO_001"
)

# Strategy deployment
deployment = client.strategy.deploy(
    strategy_id="MOMENTUM_V1",
    version="1.2.0",
    environment="staging"
)
```

### JavaScript/Node.js Example
```javascript
const { NautilusClient } = require('@nautilus/sprint3-sdk');

const client = new NautilusClient({
  baseUrl: 'http://localhost:8001',
  apiKey: 'your-api-key'
});

// WebSocket subscription
const ws = client.websocket.subscribe({
  topics: ['portfolio.updates', 'risk.alerts'],
  filters: { portfolio_id: 'PORTFOLIO_001' }
});

ws.on('message', (data) => {
  console.log('Received update:', data);
});
```

### cURL Examples

#### Create Risk Limit
```bash
curl -X POST "${BASE_URL}/api/v1/sprint3/risk/limits" \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-api-key" \
  -d '{
    "name": "Portfolio VaR Limit",
    "portfolio_id": "PORTFOLIO_001",
    "limit_type": "var",
    "threshold_value": 50000.0,
    "action": "warn"
  }'
```

#### Get System Health
```bash
curl -X GET "${BASE_URL}/api/v1/sprint3/system/health" \
  -H "X-API-Key: your-api-key"
```

#### Deploy Strategy
```bash
curl -X POST "${BASE_URL}/api/v1/sprint3/strategy/deploy" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer ${JWT_TOKEN}" \
  -d '{
    "strategy_id": "MOMENTUM_V1",
    "version": "1.2.0",
    "target_environment": "staging",
    "deployment_strategy": "blue_green"
  }'
```

## Best Practices

### API Usage
1. Always include request IDs for tracing
2. Use appropriate rate limiting strategies
3. Implement proper error handling and retries
4. Cache responses where appropriate
5. Use WebSockets for real-time data

### Security
1. Store API keys securely
2. Use HTTPS in production
3. Implement proper authentication
4. Validate all inputs
5. Log security events

### Performance
1. Use pagination for large datasets
2. Implement connection pooling
3. Cache frequently accessed data
4. Monitor API performance metrics
5. Use async/await patterns

This comprehensive documentation provides detailed information for integrating with all Sprint 3 APIs, including examples, best practices, and troubleshooting guidance.