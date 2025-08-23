"""
Enhanced OpenAPI 3.0 Specification Generator for Nautilus Trading Platform
Generates comprehensive API documentation with schemas, examples, and interactive features
"""

from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field
import json
from datetime import datetime
from enum import Enum


class APIVersionInfo(BaseModel):
    title: str = "Nautilus Trading Platform API"
    description: str = """
    Enterprise-grade trading platform with 8-source data integration, real-time streaming,
    advanced risk management, and automated strategy deployment capabilities.
    
    ## Key Features
    - **8 Integrated Data Sources**: IBKR, Alpha Vantage, FRED, EDGAR, Data.gov, Trading Economics, DBnomics, Yahoo Finance
    - **380,000+ Factors**: Multi-source synthesis and cross-correlation analysis
    - **Real-time WebSocket Streaming**: 1000+ concurrent connections with Redis pub/sub
    - **Advanced Risk Management**: ML-based breach detection with dynamic limits
    - **Strategy Deployment**: Automated CI/CD pipeline with version control
    - **Enterprise Monitoring**: Prometheus/Grafana with comprehensive dashboards
    
    ## Authentication
    All API endpoints require Bearer token authentication unless otherwise specified.
    Use `/api/v1/auth/login` to obtain access tokens.
    
    ## Rate Limiting
    - Standard tier: 1000 requests/minute
    - Premium tier: 5000 requests/minute  
    - WebSocket connections: 100/user, 1000/total
    
    ## Data Format
    All timestamps are in ISO 8601 format (UTC). Market data follows industry standards.
    """
    version: str = "3.0.0"
    contact: Dict[str, str] = {
        "name": "Nautilus API Support",
        "email": "api-support@nautilus-trading.com",
        "url": "https://docs.nautilus-trading.com"
    }
    license: Dict[str, str] = {
        "name": "MIT License",
        "url": "https://opensource.org/licenses/MIT"
    }


class SecurityScheme(BaseModel):
    type: str = "http"
    scheme: str = "bearer"
    bearerFormat: str = "JWT"
    description: str = "JWT Bearer token authentication"


class ServerInfo(BaseModel):
    url: str
    description: str


class TagInfo(BaseModel):
    name: str
    description: str
    externalDocs: Optional[Dict[str, str]] = None


class OpenAPIGenerator:
    """Advanced OpenAPI 3.0 specification generator with comprehensive schemas"""
    
    def __init__(self):
        self.spec = {
            "openapi": "3.0.3",
            "info": APIVersionInfo().dict(),
            "servers": [
                {"url": "http://localhost:8001", "description": "Development Server"},
                {"url": "https://api.nautilus-trading.com", "description": "Production Server"},
                {"url": "https://staging-api.nautilus-trading.com", "description": "Staging Server"}
            ],
            "security": [{"bearerAuth": []}],
            "components": {
                "securitySchemes": {
                    "bearerAuth": SecurityScheme().dict()
                },
                "schemas": {},
                "responses": {},
                "parameters": {},
                "examples": {},
                "requestBodies": {}
            },
            "paths": {},
            "tags": self._generate_tags()
        }
        
    def _generate_tags(self) -> List[Dict[str, Any]]:
        """Generate comprehensive API tags for organization"""
        return [
            {
                "name": "Authentication",
                "description": "User authentication and authorization endpoints",
                "externalDocs": {"url": "https://docs.nautilus-trading.com/auth"}
            },
            {
                "name": "Market Data",
                "description": "Real-time and historical market data from 8 integrated sources",
                "externalDocs": {"url": "https://docs.nautilus-trading.com/market-data"}
            },
            {
                "name": "WebSocket Streaming",
                "description": "Real-time data streaming with Redis pub/sub scaling",
                "externalDocs": {"url": "https://docs.nautilus-trading.com/websocket"}
            },
            {
                "name": "Risk Management",
                "description": "Advanced risk management with ML-based breach detection",
                "externalDocs": {"url": "https://docs.nautilus-trading.com/risk"}
            },
            {
                "name": "Strategy Management",
                "description": "Automated strategy deployment and version control",
                "externalDocs": {"url": "https://docs.nautilus-trading.com/strategies"}
            },
            {
                "name": "Analytics",
                "description": "Real-time analytics and performance monitoring",
                "externalDocs": {"url": "https://docs.nautilus-trading.com/analytics"}
            },
            {
                "name": "Portfolio Management",
                "description": "Portfolio tracking and visualization",
                "externalDocs": {"url": "https://docs.nautilus-trading.com/portfolio"}
            },
            {
                "name": "Trading Engine",
                "description": "NautilusTrader engine management and execution",
                "externalDocs": {"url": "https://docs.nautilus-trading.com/engine"}
            },
            {
                "name": "Data Sources",
                "description": "Integration with 8 external data providers",
                "externalDocs": {"url": "https://docs.nautilus-trading.com/data-sources"}
            },
            {
                "name": "System Monitoring",
                "description": "System health and performance monitoring",
                "externalDocs": {"url": "https://docs.nautilus-trading.com/monitoring"}
            }
        ]
    
    def add_common_schemas(self):
        """Add comprehensive common schemas used across the API"""
        self.spec["components"]["schemas"].update({
            "ErrorResponse": {
                "type": "object",
                "required": ["error", "message", "timestamp"],
                "properties": {
                    "error": {
                        "type": "string",
                        "description": "Error code",
                        "example": "VALIDATION_ERROR"
                    },
                    "message": {
                        "type": "string",
                        "description": "Human-readable error message",
                        "example": "Invalid symbol format"
                    },
                    "details": {
                        "type": "object",
                        "description": "Additional error details",
                        "additionalProperties": True
                    },
                    "timestamp": {
                        "type": "string",
                        "format": "date-time",
                        "description": "Error timestamp in ISO 8601 format",
                        "example": "2025-01-23T10:30:00.000Z"
                    },
                    "request_id": {
                        "type": "string",
                        "description": "Unique request identifier for debugging",
                        "example": "req_123abc456def"
                    }
                }
            },
            "HealthCheck": {
                "type": "object",
                "required": ["status", "timestamp"],
                "properties": {
                    "status": {
                        "type": "string",
                        "enum": ["healthy", "degraded", "unhealthy"],
                        "description": "Overall system health status"
                    },
                    "timestamp": {
                        "type": "string",
                        "format": "date-time",
                        "description": "Health check timestamp"
                    },
                    "version": {
                        "type": "string",
                        "description": "API version",
                        "example": "3.0.0"
                    },
                    "components": {
                        "type": "object",
                        "description": "Individual component health status",
                        "additionalProperties": {
                            "type": "object",
                            "properties": {
                                "status": {"type": "string", "enum": ["up", "down", "degraded"]},
                                "response_time": {"type": "number", "description": "Response time in milliseconds"},
                                "last_check": {"type": "string", "format": "date-time"}
                            }
                        }
                    }
                }
            },
            "PaginationRequest": {
                "type": "object",
                "properties": {
                    "page": {
                        "type": "integer",
                        "minimum": 1,
                        "default": 1,
                        "description": "Page number (1-based)"
                    },
                    "limit": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 1000,
                        "default": 50,
                        "description": "Number of items per page"
                    },
                    "sort": {
                        "type": "string",
                        "description": "Sort field and direction (e.g., 'timestamp:desc')"
                    }
                }
            },
            "PaginationResponse": {
                "type": "object",
                "required": ["page", "limit", "total", "pages"],
                "properties": {
                    "page": {
                        "type": "integer",
                        "description": "Current page number"
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Items per page"
                    },
                    "total": {
                        "type": "integer",
                        "description": "Total number of items"
                    },
                    "pages": {
                        "type": "integer",
                        "description": "Total number of pages"
                    },
                    "has_next": {
                        "type": "boolean",
                        "description": "Whether there are more pages"
                    },
                    "has_prev": {
                        "type": "boolean",
                        "description": "Whether there are previous pages"
                    }
                }
            },
            "MarketData": {
                "type": "object",
                "required": ["symbol", "timestamp", "source"],
                "properties": {
                    "symbol": {
                        "type": "string",
                        "description": "Financial instrument symbol",
                        "example": "AAPL"
                    },
                    "timestamp": {
                        "type": "string",
                        "format": "date-time",
                        "description": "Data timestamp in ISO 8601 format"
                    },
                    "source": {
                        "type": "string",
                        "enum": ["IBKR", "ALPHA_VANTAGE", "FRED", "EDGAR", "DATAGOV", "TRADING_ECONOMICS", "DBNOMICS", "YAHOO"],
                        "description": "Data source provider"
                    },
                    "price": {
                        "type": "number",
                        "format": "double",
                        "description": "Current or closing price"
                    },
                    "volume": {
                        "type": "integer",
                        "description": "Trading volume"
                    },
                    "bid": {
                        "type": "number",
                        "format": "double",
                        "description": "Bid price"
                    },
                    "ask": {
                        "type": "number",
                        "format": "double",
                        "description": "Ask price"
                    },
                    "open": {
                        "type": "number",
                        "format": "double",
                        "description": "Opening price"
                    },
                    "high": {
                        "type": "number",
                        "format": "double",
                        "description": "Highest price"
                    },
                    "low": {
                        "type": "number",
                        "format": "double",
                        "description": "Lowest price"
                    },
                    "change": {
                        "type": "number",
                        "format": "double",
                        "description": "Price change"
                    },
                    "change_percent": {
                        "type": "number",
                        "format": "double",
                        "description": "Percentage change"
                    }
                }
            },
            "WebSocketMessage": {
                "type": "object",
                "required": ["type", "timestamp"],
                "properties": {
                    "type": {
                        "type": "string",
                        "enum": [
                            "market_data", "trade_update", "order_update", "portfolio_update",
                            "risk_alert", "system_status", "strategy_update", "analytics_update",
                            "connection_status", "subscription_update", "heartbeat", "error",
                            "user_notification", "market_event", "economic_data"
                        ],
                        "description": "Message type identifier"
                    },
                    "timestamp": {
                        "type": "string",
                        "format": "date-time",
                        "description": "Message timestamp"
                    },
                    "data": {
                        "type": "object",
                        "description": "Message payload",
                        "additionalProperties": True
                    },
                    "sequence": {
                        "type": "integer",
                        "description": "Message sequence number for ordering"
                    },
                    "channel": {
                        "type": "string",
                        "description": "WebSocket channel/topic"
                    }
                }
            },
            "RiskLimit": {
                "type": "object",
                "required": ["id", "type", "value", "status"],
                "properties": {
                    "id": {
                        "type": "string",
                        "description": "Unique risk limit identifier"
                    },
                    "type": {
                        "type": "string",
                        "enum": [
                            "position_limit", "loss_limit", "exposure_limit", "concentration_limit",
                            "var_limit", "drawdown_limit", "leverage_limit", "correlation_limit",
                            "sector_limit", "currency_limit", "volatility_limit", "liquidity_limit"
                        ],
                        "description": "Type of risk limit"
                    },
                    "value": {
                        "type": "number",
                        "format": "double",
                        "description": "Limit value"
                    },
                    "current_value": {
                        "type": "number",
                        "format": "double",
                        "description": "Current exposure value"
                    },
                    "utilization": {
                        "type": "number",
                        "format": "double",
                        "minimum": 0,
                        "maximum": 1,
                        "description": "Limit utilization percentage (0-1)"
                    },
                    "status": {
                        "type": "string",
                        "enum": ["active", "breached", "warning", "disabled"],
                        "description": "Current limit status"
                    },
                    "warning_threshold": {
                        "type": "number",
                        "format": "double",
                        "description": "Warning threshold (0-1)"
                    },
                    "auto_adjust": {
                        "type": "boolean",
                        "description": "Whether limit auto-adjusts based on market conditions"
                    }
                }
            },
            "Strategy": {
                "type": "object",
                "required": ["id", "name", "version", "status"],
                "properties": {
                    "id": {
                        "type": "string",
                        "description": "Unique strategy identifier"
                    },
                    "name": {
                        "type": "string",
                        "description": "Strategy name"
                    },
                    "version": {
                        "type": "string",
                        "description": "Strategy version (semantic versioning)"
                    },
                    "status": {
                        "type": "string",
                        "enum": ["development", "testing", "deployed", "paused", "retired"],
                        "description": "Current strategy status"
                    },
                    "description": {
                        "type": "string",
                        "description": "Strategy description"
                    },
                    "parameters": {
                        "type": "object",
                        "description": "Strategy parameters",
                        "additionalProperties": True
                    },
                    "performance_metrics": {
                        "type": "object",
                        "properties": {
                            "total_return": {"type": "number", "format": "double"},
                            "sharpe_ratio": {"type": "number", "format": "double"},
                            "max_drawdown": {"type": "number", "format": "double"},
                            "win_rate": {"type": "number", "format": "double"},
                            "profit_factor": {"type": "number", "format": "double"}
                        }
                    },
                    "deployment_info": {
                        "type": "object",
                        "properties": {
                            "deployed_at": {"type": "string", "format": "date-time"},
                            "deployment_id": {"type": "string"},
                            "environment": {"type": "string", "enum": ["paper", "live"]},
                            "auto_rollback": {"type": "boolean"}
                        }
                    }
                }
            }
        })
    
    def add_authentication_endpoints(self):
        """Add authentication and authorization endpoints"""
        auth_paths = {
            "/api/v1/auth/login": {
                "post": {
                    "tags": ["Authentication"],
                    "summary": "User login",
                    "description": "Authenticate user and receive JWT access token",
                    "security": [],  # No auth required for login
                    "requestBody": {
                        "required": True,
                        "content": {
                            "application/json": {
                                "schema": {
                                    "type": "object",
                                    "required": ["username", "password"],
                                    "properties": {
                                        "username": {"type": "string", "example": "trader@nautilus.com"},
                                        "password": {"type": "string", "format": "password"},
                                        "remember_me": {"type": "boolean", "default": False}
                                    }
                                }
                            }
                        }
                    },
                    "responses": {
                        "200": {
                            "description": "Login successful",
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "type": "object",
                                        "properties": {
                                            "access_token": {"type": "string"},
                                            "token_type": {"type": "string", "example": "Bearer"},
                                            "expires_in": {"type": "integer", "example": 3600},
                                            "refresh_token": {"type": "string"},
                                            "user_id": {"type": "string"},
                                            "permissions": {"type": "array", "items": {"type": "string"}}
                                        }
                                    }
                                }
                            }
                        },
                        "401": {"$ref": "#/components/responses/UnauthorizedError"},
                        "422": {"$ref": "#/components/responses/ValidationError"}
                    }
                }
            },
            "/api/v1/auth/refresh": {
                "post": {
                    "tags": ["Authentication"],
                    "summary": "Refresh access token",
                    "description": "Refresh JWT access token using refresh token",
                    "security": [],
                    "requestBody": {
                        "required": True,
                        "content": {
                            "application/json": {
                                "schema": {
                                    "type": "object",
                                    "required": ["refresh_token"],
                                    "properties": {
                                        "refresh_token": {"type": "string"}
                                    }
                                }
                            }
                        }
                    },
                    "responses": {
                        "200": {
                            "description": "Token refreshed successfully",
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "type": "object",
                                        "properties": {
                                            "access_token": {"type": "string"},
                                            "token_type": {"type": "string", "example": "Bearer"},
                                            "expires_in": {"type": "integer", "example": 3600}
                                        }
                                    }
                                }
                            }
                        },
                        "401": {"$ref": "#/components/responses/UnauthorizedError"}
                    }
                }
            }
        }
        self.spec["paths"].update(auth_paths)
    
    def add_market_data_endpoints(self):
        """Add comprehensive market data endpoints"""
        market_data_paths = {
            "/api/v1/market-data/quote/{symbol}": {
                "get": {
                    "tags": ["Market Data"],
                    "summary": "Get real-time quote",
                    "description": "Retrieve real-time quote data from integrated data sources",
                    "parameters": [
                        {
                            "name": "symbol",
                            "in": "path",
                            "required": True,
                            "schema": {"type": "string", "example": "AAPL"},
                            "description": "Financial instrument symbol"
                        },
                        {
                            "name": "source",
                            "in": "query",
                            "schema": {
                                "type": "string",
                                "enum": ["IBKR", "ALPHA_VANTAGE", "YAHOO"],
                                "default": "IBKR"
                            },
                            "description": "Preferred data source"
                        }
                    ],
                    "responses": {
                        "200": {
                            "description": "Real-time quote data",
                            "content": {
                                "application/json": {
                                    "schema": {"$ref": "#/components/schemas/MarketData"},
                                    "examples": {
                                        "aapl_quote": {
                                            "value": {
                                                "symbol": "AAPL",
                                                "timestamp": "2025-01-23T15:30:00.000Z",
                                                "source": "IBKR",
                                                "price": 150.25,
                                                "volume": 45000000,
                                                "bid": 150.20,
                                                "ask": 150.30,
                                                "open": 149.80,
                                                "high": 151.50,
                                                "low": 149.50,
                                                "change": 0.45,
                                                "change_percent": 0.30
                                            }
                                        }
                                    }
                                }
                            }
                        },
                        "404": {"$ref": "#/components/responses/NotFoundError"},
                        "429": {"$ref": "#/components/responses/RateLimitError"}
                    }
                }
            },
            "/api/v1/market-data/historical/{symbol}": {
                "get": {
                    "tags": ["Market Data"],
                    "summary": "Get historical data",
                    "description": "Retrieve historical market data with multiple timeframes",
                    "parameters": [
                        {
                            "name": "symbol",
                            "in": "path",
                            "required": True,
                            "schema": {"type": "string"},
                            "description": "Financial instrument symbol"
                        },
                        {
                            "name": "interval",
                            "in": "query",
                            "schema": {
                                "type": "string",
                                "enum": ["1min", "5min", "15min", "30min", "1hour", "4hour", "1day", "1week", "1month"],
                                "default": "1day"
                            },
                            "description": "Data interval"
                        },
                        {
                            "name": "start_date",
                            "in": "query",
                            "schema": {"type": "string", "format": "date"},
                            "description": "Start date (YYYY-MM-DD)"
                        },
                        {
                            "name": "end_date",
                            "in": "query",
                            "schema": {"type": "string", "format": "date"},
                            "description": "End date (YYYY-MM-DD)"
                        },
                        {
                            "name": "limit",
                            "in": "query",
                            "schema": {"type": "integer", "minimum": 1, "maximum": 5000, "default": 100},
                            "description": "Maximum number of data points"
                        }
                    ],
                    "responses": {
                        "200": {
                            "description": "Historical market data",
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "type": "object",
                                        "properties": {
                                            "symbol": {"type": "string"},
                                            "interval": {"type": "string"},
                                            "data": {
                                                "type": "array",
                                                "items": {"$ref": "#/components/schemas/MarketData"}
                                            },
                                            "pagination": {"$ref": "#/components/schemas/PaginationResponse"}
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        self.spec["paths"].update(market_data_paths)
    
    def add_websocket_endpoints(self):
        """Add WebSocket streaming endpoints documentation"""
        websocket_paths = {
            "/ws/market-data/{symbol}": {
                "get": {
                    "tags": ["WebSocket Streaming"],
                    "summary": "Real-time market data stream",
                    "description": """
                    WebSocket endpoint for real-time market data streaming.
                    
                    **Connection Process:**
                    1. Establish WebSocket connection with authentication header
                    2. Send subscription message with desired symbols
                    3. Receive real-time market data updates
                    4. Handle heartbeat messages for connection health
                    
                    **Message Format:**
                    All messages follow the WebSocketMessage schema with type-specific payloads.
                    """,
                    "parameters": [
                        {
                            "name": "symbol",
                            "in": "path",
                            "required": True,
                            "schema": {"type": "string"},
                            "description": "Symbol to stream (supports wildcards like 'AAPL,MSFT,GOOGL')"
                        },
                        {
                            "name": "Authorization",
                            "in": "header",
                            "required": True,
                            "schema": {"type": "string"},
                            "description": "Bearer token for authentication"
                        }
                    ],
                    "responses": {
                        "101": {
                            "description": "WebSocket connection established",
                            "content": {
                                "application/json": {
                                    "schema": {"$ref": "#/components/schemas/WebSocketMessage"},
                                    "examples": {
                                        "market_data_update": {
                                            "value": {
                                                "type": "market_data",
                                                "timestamp": "2025-01-23T15:30:00.000Z",
                                                "sequence": 12345,
                                                "channel": "market_data.AAPL",
                                                "data": {
                                                    "symbol": "AAPL",
                                                    "price": 150.25,
                                                    "volume": 1000,
                                                    "bid": 150.20,
                                                    "ask": 150.30
                                                }
                                            }
                                        },
                                        "heartbeat": {
                                            "value": {
                                                "type": "heartbeat",
                                                "timestamp": "2025-01-23T15:30:30.000Z",
                                                "data": {"status": "alive"}
                                            }
                                        }
                                    }
                                }
                            }
                        },
                        "401": {"$ref": "#/components/responses/UnauthorizedError"},
                        "429": {"$ref": "#/components/responses/RateLimitError"}
                    }
                }
            }
        }
        self.spec["paths"].update(websocket_paths)
    
    def add_common_responses(self):
        """Add common response schemas"""
        self.spec["components"]["responses"].update({
            "UnauthorizedError": {
                "description": "Authentication required",
                "content": {
                    "application/json": {
                        "schema": {"$ref": "#/components/schemas/ErrorResponse"},
                        "example": {
                            "error": "UNAUTHORIZED",
                            "message": "Authentication required. Please provide valid Bearer token.",
                            "timestamp": "2025-01-23T10:30:00.000Z",
                            "request_id": "req_123abc456def"
                        }
                    }
                }
            },
            "ForbiddenError": {
                "description": "Access forbidden",
                "content": {
                    "application/json": {
                        "schema": {"$ref": "#/components/schemas/ErrorResponse"},
                        "example": {
                            "error": "FORBIDDEN",
                            "message": "Insufficient permissions to access this resource.",
                            "timestamp": "2025-01-23T10:30:00.000Z"
                        }
                    }
                }
            },
            "NotFoundError": {
                "description": "Resource not found",
                "content": {
                    "application/json": {
                        "schema": {"$ref": "#/components/schemas/ErrorResponse"},
                        "example": {
                            "error": "NOT_FOUND",
                            "message": "The requested resource could not be found.",
                            "timestamp": "2025-01-23T10:30:00.000Z"
                        }
                    }
                }
            },
            "ValidationError": {
                "description": "Request validation failed",
                "content": {
                    "application/json": {
                        "schema": {"$ref": "#/components/schemas/ErrorResponse"},
                        "example": {
                            "error": "VALIDATION_ERROR",
                            "message": "Request validation failed",
                            "details": {
                                "symbol": ["Symbol is required and must be a valid ticker"]
                            },
                            "timestamp": "2025-01-23T10:30:00.000Z"
                        }
                    }
                }
            },
            "RateLimitError": {
                "description": "Rate limit exceeded",
                "content": {
                    "application/json": {
                        "schema": {"$ref": "#/components/schemas/ErrorResponse"},
                        "example": {
                            "error": "RATE_LIMIT_EXCEEDED",
                            "message": "Rate limit exceeded. Please try again later.",
                            "details": {
                                "retry_after": 60,
                                "limit": 1000,
                                "window": 3600
                            },
                            "timestamp": "2025-01-23T10:30:00.000Z"
                        }
                    }
                }
            },
            "ServerError": {
                "description": "Internal server error",
                "content": {
                    "application/json": {
                        "schema": {"$ref": "#/components/schemas/ErrorResponse"},
                        "example": {
                            "error": "INTERNAL_ERROR",
                            "message": "An unexpected error occurred. Please try again later.",
                            "timestamp": "2025-01-23T10:30:00.000Z",
                            "request_id": "req_123abc456def"
                        }
                    }
                }
            }
        })
    
    def generate_complete_spec(self) -> Dict[str, Any]:
        """Generate the complete OpenAPI specification"""
        self.add_common_schemas()
        self.add_common_responses()
        self.add_authentication_endpoints()
        self.add_market_data_endpoints()
        self.add_websocket_endpoints()
        
        # Add health check endpoint
        self.spec["paths"]["/health"] = {
            "get": {
                "tags": ["System Monitoring"],
                "summary": "System health check",
                "description": "Get overall system health status and component information",
                "security": [],  # No auth required for health check
                "responses": {
                    "200": {
                        "description": "System health information",
                        "content": {
                            "application/json": {
                                "schema": {"$ref": "#/components/schemas/HealthCheck"},
                                "example": {
                                    "status": "healthy",
                                    "timestamp": "2025-01-23T10:30:00.000Z",
                                    "version": "3.0.0",
                                    "components": {
                                        "database": {"status": "up", "response_time": 5.2},
                                        "redis": {"status": "up", "response_time": 1.1},
                                        "websocket": {"status": "up", "active_connections": 150}
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        
        return self.spec
    
    def save_spec(self, filepath: str):
        """Save the OpenAPI specification to a file"""
        spec = self.generate_complete_spec()
        with open(filepath, 'w') as f:
            json.dump(spec, f, indent=2, default=str)
        return filepath


if __name__ == "__main__":
    generator = OpenAPIGenerator()
    spec_path = generator.save_spec("openapi_spec.json")
    print(f"OpenAPI 3.0 specification saved to: {spec_path}")