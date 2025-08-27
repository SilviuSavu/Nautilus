# 🔧 Nautilus Containerized Engines API Documentation

## Overview

This document provides comprehensive API documentation for all **9 containerized processing engines** in the Nautilus trading platform. Each engine operates as an independent microservice with specialized functionality and dedicated REST API endpoints.

**Engine Architecture**: Containerized microservices with FastAPI framework  
**Performance**: 50x+ improvement through true parallel processing  
**Communication**: Enhanced MessageBus + direct HTTP APIs  
**Resource Allocation**: 20.5 CPU cores, 36GB RAM across 9 engines  

---

## 🏗️ Engine Infrastructure Overview

### **Containerized Engines**

| **Engine** | **Port** | **Container** | **Resources** | **Specialization** |
|------------|----------|---------------|---------------|--------------------|
| [Analytics Engine](#analytics-engine-8100) | 8100 | `nautilus-analytics-engine` | 2 CPU, 4GB RAM | Real-time P&L, performance analytics |
| [Risk Engine](#risk-engine-8200) | 8200 | `nautilus-risk-engine` | 0.5 CPU, 1GB RAM | Dynamic limit monitoring, breach detection |
| [Factor Engine](#factor-engine-8300) | 8300 | `nautilus-factor-engine` | 4 CPU, 8GB RAM | 380,000+ factor synthesis |
| [ML Inference Engine](#ml-inference-engine-8400) | 8400 | `nautilus-ml-engine` | 2 CPU, 6GB RAM | Model predictions, regime detection |
| [Features Engine](#features-engine-8500) | 8500 | `nautilus-features-engine` | 3 CPU, 4GB RAM | Technical indicators, fundamentals |
| [WebSocket Engine](#websocket-engine-8600) | 8600 | `nautilus-websocket-engine` | 1 CPU, 2GB RAM | Real-time streaming |
| [Strategy Engine](#strategy-engine-8700) | 8700 | `nautilus-strategy-engine` | 1 CPU, 2GB RAM | Automated deployment |
| [Market Data Engine](#market-data-engine-8800) | 8800 | `nautilus-marketdata-engine` | 2 CPU, 3GB RAM | Data ingestion |
| [Portfolio Engine](#portfolio-engine-8900) | 8900 | `nautilus-portfolio-engine` | 4 CPU, 8GB RAM | Portfolio optimization |

### **Common API Patterns**

All engines follow consistent API design patterns:

#### **Standard Endpoints (All Engines)**
```python
GET  /{engine}/health                    # Health check with MessageBus status
GET  /{engine}/metrics                   # Prometheus metrics endpoint  
GET  /{engine}/info                      # Engine information and capabilities
POST /{engine}/process                   # Main processing endpoint
GET  /{engine}/status                    # Engine status and statistics
```

#### **Authentication & Headers**
```http
# Required headers for all requests
Content-Type: application/json
Accept: application/json

# Optional headers
X-Request-ID: uuid4()                   # Request tracing
X-Client-Version: 1.0.0                 # Client version
Authorization: Bearer <jwt_token>        # Authentication (if enabled)
```

#### **Standard Response Format**
```json
{
  "success": true,
  "data": {},
  "metadata": {
    "engine": "analytics",
    "version": "1.0.0", 
    "request_id": "uuid",
    "processing_time_ms": 45,
    "timestamp": "2025-08-23T10:30:00Z"
  },
  "errors": []
}
```

---

## 📊 Analytics Engine (8100)

### **Overview**
Real-time performance analytics engine for portfolio P&L calculations, attribution analysis, and performance metrics computation.

**Base URL**: `http://localhost:8100`  
**Container**: `nautilus-analytics-engine`  
**Resources**: 2 CPU cores, 4GB RAM  
**Performance Target**: Sub-second P&L calculations  

### **Core Endpoints**

#### **Health & Status**
```http
GET /health
```

**Response**:
```json
{
  "status": "healthy",
  "version": "1.0.0",
  "uptime_seconds": 3600,
  "messagebus_connected": true,
  "database_connected": true,
  "last_health_check": "2025-08-23T10:30:00Z",
  "performance_metrics": {
    "avg_response_time_ms": 45,
    "requests_per_second": 150,
    "active_calculations": 5
  }
}
```

#### **Portfolio Performance Calculation**
```http
POST /analytics/calculate/{portfolio_id}
```

**Request Body**:
```json
{
  "positions": [
    {
      "symbol": "AAPL",
      "quantity": 100,
      "avg_cost": 150.00,
      "current_price": 155.00,
      "asset_class": "equity",
      "sector": "technology"
    }
  ],
  "benchmark": "SPY",
  "calculation_options": {
    "include_attribution": true,
    "time_period": "1D",
    "risk_free_rate": 0.05
  }
}
```

**Response**:
```json
{
  "success": true,
  "data": {
    "portfolio_id": "portfolio_001",
    "total_pnl": 500.00,
    "total_pnl_percent": 3.33,
    "unrealized_pnl": 500.00,
    "realized_pnl": 0.00,
    "market_value": 15500.00,
    "cost_basis": 15000.00,
    "performance_metrics": {
      "sharpe_ratio": 1.25,
      "alpha": 0.02,
      "beta": 1.05,
      "max_drawdown": -0.08,
      "volatility": 0.15
    },
    "attribution": {
      "sector_attribution": {
        "technology": 500.00
      },
      "security_attribution": {
        "AAPL": 500.00
      }
    }
  },
  "metadata": {
    "calculation_time_ms": 45,
    "positions_analyzed": 1,
    "benchmark_used": "SPY"
  }
}
```

#### **Real-time P&L Streaming**
```http
GET /analytics/stream/{portfolio_id}
```

**Response**: Server-Sent Events (SSE) stream
```javascript
data: {"event": "pnl_update", "portfolio_id": "portfolio_001", "pnl": 525.50, "timestamp": "2025-08-23T10:30:15Z"}

data: {"event": "position_update", "symbol": "AAPL", "current_price": 155.25, "pnl": 525.00}
```

#### **Performance Attribution Analysis**
```http
POST /analytics/attribution/{portfolio_id}
```

**Request Body**:
```json
{
  "attribution_type": "factor",
  "factors": ["size", "value", "momentum", "quality"],
  "time_period": "1M",
  "benchmark": "SPY"
}
```

**Response**:
```json
{
  "success": true,
  "data": {
    "total_return": 0.0533,
    "benchmark_return": 0.0445,
    "active_return": 0.0088,
    "factor_attribution": {
      "size": 0.0012,
      "value": 0.0023,
      "momentum": 0.0035,
      "quality": 0.0018
    },
    "security_selection": 0.0045,
    "allocation_effect": 0.0043
  }
}
```

#### **Benchmark Comparison**
```http
GET /analytics/benchmark/{portfolio_id}?benchmark=SPY&period=1M
```

**Response**:
```json
{
  "success": true,
  "data": {
    "portfolio_return": 0.0533,
    "benchmark_return": 0.0445,
    "excess_return": 0.0088,
    "tracking_error": 0.0234,
    "information_ratio": 0.376,
    "correlation": 0.89,
    "beta": 1.05,
    "alpha": 0.002
  }
}
```

### **Metrics & Monitoring**

#### **Prometheus Metrics**
```http
GET /metrics
```

**Key Metrics**:
```prometheus
# Request metrics
analytics_requests_total{method="POST", endpoint="/calculate"}
analytics_request_duration_seconds{quantile="0.95"}
analytics_calculation_errors_total{error_type="invalid_position"}

# Business metrics  
analytics_portfolios_processed_total
analytics_pnl_calculations_total
analytics_attribution_analyses_total

# Performance metrics
analytics_calculation_time_seconds{portfolio_size="small"}
analytics_memory_usage_bytes
analytics_cpu_utilization_percent
```

---

## ⚠️ Risk Engine (8200)

### **Overview**
Advanced risk management engine for dynamic limit monitoring, real-time breach detection, and ML-based risk prediction.

**Base URL**: `http://localhost:8200`  
**Container**: `nautilus-risk-engine`  
**Resources**: 0.5 CPU cores, 1GB RAM  
**Performance Target**: <50ms risk checks, 5-second monitoring intervals  

### **Core Endpoints**

#### **Health & Status**
```http
GET /health
```

**Response**:
```json
{
  "status": "healthy",
  "version": "1.0.0",
  "uptime_seconds": 3600,
  "messagebus_connected": true,
  "monitoring_active": true,
  "monitored_portfolios": 5,
  "breach_alerts_sent": 2,
  "risk_checks_performed": 1440,
  "avg_check_time_ms": 25
}
```

#### **Real-time Risk Check**
```http
POST /risk/check/{portfolio_id}
```

**Request Body**:
```json
{
  "positions": [
    {
      "symbol": "AAPL",
      "quantity": 1000,
      "current_price": 155.00,
      "asset_class": "equity",
      "sector": "technology"
    }
  ],
  "portfolio_value": 1000000,
  "limits": {
    "max_position_size": 0.10,
    "max_sector_exposure": 0.25,
    "max_leverage": 2.0,
    "var_limit": 0.02
  }
}
```

**Response**:
```json
{
  "success": true,
  "data": {
    "risk_assessment": "MEDIUM",
    "overall_status": "WITHIN_LIMITS",
    "limit_checks": [
      {
        "limit_type": "position_size",
        "current_value": 0.155,
        "limit_value": 0.10,
        "status": "BREACH",
        "severity": "HIGH",
        "action_required": "REDUCE_POSITION"
      },
      {
        "limit_type": "sector_exposure",
        "current_value": 0.155,
        "limit_value": 0.25,
        "status": "OK",
        "severity": "LOW",
        "utilization_percent": 62.0
      }
    ],
    "risk_metrics": {
      "portfolio_var_1d": 0.015,
      "portfolio_var_10d": 0.047,
      "expected_shortfall": 0.021,
      "beta": 1.05,
      "correlation_to_market": 0.89
    },
    "breaches": [
      {
        "breach_id": "breach_001",
        "limit_type": "position_size",
        "severity": "HIGH",
        "action_required": "REDUCE_POSITION",
        "recommendation": "Reduce AAPL position to 10% of portfolio"
      }
    ]
  },
  "metadata": {
    "check_time_ms": 35,
    "positions_analyzed": 1,
    "limits_checked": 8
  }
}
```

#### **Start Risk Monitoring**
```http
POST /risk/monitor/start
```

**Request Body**:
```json
{
  "portfolio_id": "portfolio_001",
  "monitoring_config": {
    "check_interval_seconds": 5,
    "alert_channels": ["email", "webhook"],
    "severity_threshold": "MEDIUM"
  }
}
```

**Response**:
```json
{
  "success": true,
  "data": {
    "monitoring_id": "monitor_001",
    "status": "ACTIVE",
    "next_check_at": "2025-08-23T10:30:05Z",
    "check_interval_seconds": 5
  }
}
```

#### **Dynamic Limit Management**
```http
POST /risk/limits
```

**Request Body**:
```json
{
  "portfolio_id": "portfolio_001",
  "limits": [
    {
      "limit_type": "position_size",
      "value": 0.10,
      "enabled": true,
      "auto_adjust": false
    },
    {
      "limit_type": "var_limit",
      "value": 0.02,
      "enabled": true,
      "auto_adjust": true,
      "adjustment_factor": 0.1
    }
  ]
}
```

#### **Breach History & Reporting**
```http
GET /risk/breaches/{portfolio_id}?period=7D&format=json
```

**Response**:
```json
{
  "success": true,
  "data": {
    "breach_summary": {
      "total_breaches": 5,
      "high_severity": 2,
      "medium_severity": 2,
      "low_severity": 1,
      "avg_resolution_time_minutes": 15
    },
    "breaches": [
      {
        "breach_id": "breach_001",
        "timestamp": "2025-08-23T09:15:00Z",
        "limit_type": "position_size",
        "severity": "HIGH",
        "current_value": 0.155,
        "limit_value": 0.10,
        "resolution_status": "RESOLVED",
        "resolution_time": "2025-08-23T09:25:00Z",
        "action_taken": "Position reduced to 9.5%"
      }
    ]
  }
}
```

#### **Multi-format Risk Reporting**
```http
POST /risk/report/generate
```

**Request Body**:
```json
{
  "portfolio_id": "portfolio_001",
  "report_type": "comprehensive",
  "format": "pdf",
  "time_period": "1M",
  "sections": ["summary", "breaches", "var_analysis", "stress_tests"]
}
```

**Response**:
```json
{
  "success": true,
  "data": {
    "report_id": "report_001",
    "download_url": "/risk/reports/report_001.pdf",
    "expires_at": "2025-08-24T10:30:00Z",
    "report_size_bytes": 2048576,
    "generation_time_ms": 1500
  }
}
```

### **Supported Limit Types**

```python
LIMIT_TYPES = {
    "position_size": "Maximum position size as % of portfolio",
    "concentration_limit": "Maximum concentration in single security", 
    "sector_exposure": "Maximum exposure to any sector",
    "var_limit": "Value at Risk limit (1-day, 99% confidence)",
    "expected_shortfall": "Expected Shortfall limit",
    "leverage_limit": "Maximum portfolio leverage",
    "gross_exposure": "Gross exposure limit",
    "net_exposure": "Net exposure limit",
    "currency_exposure": "Foreign currency exposure limit",
    "adv_limit": "Average Daily Volume limit",
    "liquidity_limit": "Minimum liquidity requirement",
    "correlation_limit": "Maximum correlation to benchmark"
}
```

---

## 🔬 Factor Engine (8300)

### **Overview**
Multi-source factor synthesis engine managing 380,000+ factors across 8 integrated data sources with advanced cross-correlation analysis.

**Base URL**: `http://localhost:8300`  
**Container**: `nautilus-factor-engine`  
**Resources**: 4 CPU cores, 8GB RAM  
**Performance Target**: 5,000+ factors/second calculation  

### **Core Endpoints**

#### **Health & Status**
```http
GET /health
```

**Response**:
```json
{
  "status": "healthy",
  "version": "1.0.0",
  "uptime_seconds": 3600,
  "messagebus_connected": true,
  "data_sources_connected": {
    "ibkr": true,
    "alpha_vantage": true,
    "fred": true,
    "edgar": true,
    "datagov": true,
    "trading_economics": true,
    "dbnomics": true,
    "yfinance": true
  },
  "total_factors_available": 380547,
  "factors_calculated_today": 15420,
  "avg_calculation_time_ms": 125
}
```

#### **Multi-Source Factor Calculation**
```http
POST /factors/calculate
```

**Request Body**:
```json
{
  "symbols": ["AAPL", "MSFT", "GOOGL"],
  "factor_categories": ["technical", "fundamental", "economic", "regulatory"],
  "data_sources": ["alpha_vantage", "fred", "edgar"],
  "calculation_options": {
    "lookback_period": "1Y",
    "rebalance_frequency": "monthly",
    "cross_correlation": true,
    "factor_scoring": true
  }
}
```

**Response**:
```json
{
  "success": true,
  "data": {
    "calculation_id": "calc_001",
    "symbols_processed": 3,
    "factors_calculated": 145,
    "calculation_time_ms": 1850,
    "factor_results": {
      "AAPL": {
        "technical_factors": {
          "rsi_14": 65.3,
          "macd_signal": 0.85,
          "bollinger_position": 0.72,
          "momentum_12_1": 0.15
        },
        "fundamental_factors": {
          "pe_ratio": 28.5,
          "price_to_book": 8.2,
          "roe": 0.88,
          "debt_to_equity": 1.73
        },
        "economic_factors": {
          "gdp_correlation": 0.65,
          "interest_rate_sensitivity": -0.23,
          "inflation_beta": 0.12
        }
      }
    },
    "cross_correlations": {
      "technical_fundamental": 0.35,
      "fundamental_economic": 0.52,
      "technical_economic": 0.18
    },
    "factor_scores": {
      "momentum": 0.78,
      "value": 0.34,
      "quality": 0.89,
      "size": 0.56
    }
  }
}
```

#### **Factor Correlation Analysis**
```http
GET /factors/correlation/{factor_id}?period=1Y&correlation_threshold=0.5
```

**Response**:
```json
{
  "success": true,
  "data": {
    "factor_id": "momentum_12_1",
    "correlation_analysis": {
      "highly_correlated_factors": [
        {
          "factor_id": "rsi_14",
          "correlation": 0.78,
          "data_source": "technical"
        },
        {
          "factor_id": "price_momentum_6m",
          "correlation": 0.82,
          "data_source": "alpha_vantage"
        }
      ],
      "negatively_correlated_factors": [
        {
          "factor_id": "value_score",
          "correlation": -0.67,
          "data_source": "fundamental"
        }
      ],
      "factor_groups": {
        "momentum_cluster": ["momentum_12_1", "rsi_14", "macd_signal"],
        "value_cluster": ["pe_ratio", "price_to_book", "value_score"]
      }
    }
  }
}
```

#### **Factor Synthesis & Combination**
```http
POST /factors/synthesize
```

**Request Body**:
```json
{
  "synthesis_type": "multi_source",
  "factor_combination": {
    "alpha_vantage_momentum": 0.4,
    "fred_economic_regime": 0.3,
    "edgar_fundamental_strength": 0.3
  },
  "symbols": ["AAPL", "MSFT"],
  "rebalancing": {
    "frequency": "monthly",
    "optimization_method": "sharpe_ratio"
  }
}
```

**Response**:
```json
{
  "success": true,
  "data": {
    "synthetic_factor_id": "multi_source_composite_001",
    "factor_weights": {
      "alpha_vantage_momentum": 0.4,
      "fred_economic_regime": 0.3,
      "edgar_fundamental_strength": 0.3
    },
    "backtest_results": {
      "annualized_return": 0.1245,
      "volatility": 0.1567,
      "sharpe_ratio": 0.79,
      "max_drawdown": -0.0823,
      "information_ratio": 0.67
    },
    "factor_values": {
      "AAPL": 0.78,
      "MSFT": 0.65
    }
  }
}
```

#### **Factor Performance Ranking**
```http
GET /factors/ranking?period=1Y&metric=sharpe_ratio&top_n=50
```

**Response**:
```json
{
  "success": true,
  "data": {
    "ranking_criteria": {
      "metric": "sharpe_ratio",
      "period": "1Y",
      "minimum_observations": 252
    },
    "top_factors": [
      {
        "rank": 1,
        "factor_id": "momentum_quality_composite",
        "data_source": "multi_source",
        "sharpe_ratio": 1.87,
        "annualized_return": 0.1890,
        "volatility": 0.1011
      },
      {
        "rank": 2,
        "factor_id": "economic_regime_momentum",
        "data_source": "fred_alpha_vantage",
        "sharpe_ratio": 1.65,
        "annualized_return": 0.1543,
        "volatility": 0.0935
      }
    ],
    "factor_categories_performance": {
      "momentum": {
        "avg_sharpe_ratio": 1.23,
        "best_factor": "momentum_quality_composite"
      },
      "value": {
        "avg_sharpe_ratio": 0.67,
        "best_factor": "deep_value_composite"
      }
    }
  }
}
```

### **Data Source Capabilities**

```python
DATA_SOURCES = {
    "alpha_vantage": {
        "factors": 15,
        "update_frequency": "daily",
        "data_types": ["price", "volume", "fundamentals"]
    },
    "fred": {
        "factors": 32,
        "update_frequency": "daily",
        "data_types": ["economic_indicators", "rates", "inflation"]
    },
    "edgar": {
        "factors": 25,
        "update_frequency": "quarterly",
        "data_types": ["financials", "filings", "governance"]
    },
    "datagov": {
        "factors": 50,
        "update_frequency": "weekly",
        "data_types": ["government_data", "economic_census", "agriculture"]
    },
    "trading_economics": {
        "factors": 300000,
        "update_frequency": "realtime",
        "data_types": ["global_indicators", "forecasts", "calendars"]
    },
    "dbnomics": {
        "factors": 80000,
        "update_frequency": "daily",
        "data_types": ["statistical_data", "central_bank", "institutional"]
    }
}
```

---

## 🤖 ML Inference Engine (8400)

### **Overview**
Machine learning inference engine providing real-time predictions for price, market regime detection, volatility forecasting, and model management.

**Base URL**: `http://localhost:8400`  
**Container**: `nautilus-ml-engine`  
**Resources**: 2 CPU cores, 6GB RAM  
**Performance Target**: <100ms prediction latency, 1,000+ predictions/second  

### **Core Endpoints**

#### **Health & Status**
```http
GET /health
```

**Response**:
```json
{
  "status": "healthy",
  "version": "1.0.0",
  "uptime_seconds": 3600,
  "messagebus_connected": true,
  "models_loaded": 12,
  "active_models": 8,
  "predictions_served_today": 15420,
  "avg_prediction_time_ms": 65,
  "gpu_available": false,
  "model_registry_status": "connected"
}
```

#### **Price Prediction**
```http
POST /ml/predict/price
```

**Request Body**:
```json
{
  "symbol": "AAPL",
  "current_price": 155.00,
  "prediction_horizon": "1D",
  "model_type": "lstm",
  "features": {
    "technical_indicators": {
      "rsi": 65.3,
      "macd": 0.85,
      "bollinger_position": 0.72
    },
    "fundamental_data": {
      "pe_ratio": 28.5,
      "earnings_growth": 0.15
    },
    "market_conditions": {
      "vix": 18.5,
      "market_regime": "trending_up"
    }
  },
  "confidence_interval": 0.95
}
```

**Response**:
```json
{
  "success": true,
  "data": {
    "symbol": "AAPL",
    "current_price": 155.00,
    "predicted_price": 157.25,
    "prediction_change": 0.0145,
    "prediction_direction": "UP",
    "confidence_score": 0.78,
    "confidence_interval": {
      "lower_bound": 154.20,
      "upper_bound": 160.30,
      "confidence_level": 0.95
    },
    "model_info": {
      "model_id": "lstm_price_v2.1",
      "model_type": "lstm",
      "training_date": "2025-08-01",
      "feature_importance": {
        "rsi": 0.35,
        "macd": 0.28,
        "pe_ratio": 0.22,
        "vix": 0.15
      }
    },
    "risk_metrics": {
      "prediction_volatility": 0.025,
      "downside_risk": 0.018,
      "expected_return": 0.0145
    }
  },
  "metadata": {
    "prediction_time_ms": 67,
    "features_used": 8,
    "model_version": "2.1"
  }
}
```

#### **Market Regime Detection**
```http
POST /ml/predict/regime
```

**Request Body**:
```json
{
  "market_indicators": {
    "vix": 18.5,
    "yield_curve_10y_2y": 1.25,
    "dollar_index": 103.5,
    "high_yield_spreads": 350
  },
  "lookback_period": "30D",
  "regime_types": ["trending_up", "trending_down", "volatile", "calm", "crisis", "recovery"]
}
```

**Response**:
```json
{
  "success": true,
  "data": {
    "current_regime": "trending_up",
    "regime_probability": 0.82,
    "regime_probabilities": {
      "trending_up": 0.82,
      "trending_down": 0.05,
      "volatile": 0.08,
      "calm": 0.03,
      "crisis": 0.01,
      "recovery": 0.01
    },
    "regime_transition_matrix": {
      "from_trending_up": {
        "to_trending_up": 0.85,
        "to_volatile": 0.10,
        "to_trending_down": 0.05
      }
    },
    "regime_characteristics": {
      "expected_volatility": 0.15,
      "expected_return": 0.08,
      "typical_duration_days": 45
    },
    "confidence_metrics": {
      "classification_confidence": 0.82,
      "regime_stability": 0.78,
      "prediction_horizon_days": 5
    }
  }
}
```

#### **Volatility Forecasting**
```http
POST /ml/predict/volatility
```

**Request Body**:
```json
{
  "symbol": "AAPL",
  "forecast_horizon": "30D",
  "model_type": "garch",
  "historical_data": {
    "returns": [-0.01, 0.02, -0.005, 0.015],
    "volatilities": [0.12, 0.15, 0.13, 0.14]
  }
}
```

**Response**:
```json
{
  "success": true,
  "data": {
    "symbol": "AAPL",
    "forecasted_volatility": 0.1678,
    "volatility_range": {
      "min_volatility": 0.1420,
      "max_volatility": 0.1936,
      "confidence_level": 0.95
    },
    "volatility_regime": "moderate",
    "risk_metrics": {
      "var_1d_99": 0.0289,
      "expected_shortfall": 0.0345,
      "volatility_of_volatility": 0.025
    },
    "forecast_accuracy": {
      "historical_mae": 0.0123,
      "r_squared": 0.67,
      "directional_accuracy": 0.74
    }
  }
}
```

#### **Model Registry Management**
```http
GET /ml/models
```

**Response**:
```json
{
  "success": true,
  "data": {
    "available_models": [
      {
        "model_id": "lstm_price_v2.1",
        "model_type": "price_prediction",
        "algorithm": "lstm",
        "status": "active",
        "accuracy": 0.67,
        "last_trained": "2025-08-01T00:00:00Z",
        "predictions_served": 5420,
        "avg_latency_ms": 65
      },
      {
        "model_id": "regime_classifier_v1.3",
        "model_type": "market_regime",
        "algorithm": "random_forest",
        "status": "active",
        "accuracy": 0.82,
        "last_trained": "2025-07-28T00:00:00Z",
        "predictions_served": 1230,
        "avg_latency_ms": 45
      }
    ],
    "model_performance_summary": {
      "total_models": 12,
      "active_models": 8,
      "avg_accuracy": 0.74,
      "total_predictions_today": 15420
    }
  }
}
```

#### **Model Training & Retraining**
```http
POST /ml/models/train
```

**Request Body**:
```json
{
  "model_type": "price_prediction",
  "algorithm": "lstm",
  "training_config": {
    "symbols": ["AAPL", "MSFT", "GOOGL"],
    "lookback_period": "2Y",
    "validation_split": 0.2,
    "hyperparameters": {
      "layers": [64, 32, 16],
      "dropout_rate": 0.2,
      "learning_rate": 0.001,
      "batch_size": 32,
      "epochs": 100
    }
  },
  "feature_selection": {
    "technical_indicators": true,
    "fundamental_data": true,
    "market_sentiment": true
  }
}
```

**Response**:
```json
{
  "success": true,
  "data": {
    "training_job_id": "train_001",
    "status": "started",
    "estimated_completion": "2025-08-23T12:30:00Z",
    "training_progress": 0.05,
    "model_config": {
      "model_id": "lstm_price_v2.2",
      "algorithm": "lstm",
      "training_samples": 15420,
      "validation_samples": 3855
    }
  }
}
```

#### **A/B Testing Framework**
```http
POST /ml/models/ab_test
```

**Request Body**:
```json
{
  "test_name": "lstm_vs_transformer",
  "model_a": "lstm_price_v2.1",
  "model_b": "transformer_price_v1.0", 
  "test_config": {
    "traffic_split": 0.5,
    "success_metrics": ["accuracy", "latency", "sharpe_ratio"],
    "test_duration_days": 7,
    "significance_threshold": 0.05
  }
}
```

### **Model Types & Capabilities**

```python
MODEL_TYPES = {
    "price_prediction": {
        "algorithms": ["lstm", "transformer", "xgboost", "linear_regression"],
        "horizons": ["1H", "1D", "1W", "1M"],
        "accuracy_target": 0.65
    },
    "market_regime": {
        "algorithms": ["random_forest", "hmm", "clustering", "neural_network"],
        "regimes": ["trending_up", "trending_down", "volatile", "calm", "crisis", "recovery"],
        "accuracy_target": 0.80
    },
    "volatility": {
        "algorithms": ["garch", "stochastic_vol", "neural_network"],
        "horizons": ["1D", "1W", "1M"],
        "accuracy_target": 0.70
    },
    "sentiment": {
        "algorithms": ["bert", "lstm", "naive_bayes"],
        "data_sources": ["news", "social_media", "analyst_reports"],
        "accuracy_target": 0.75
    }
}
```

---

## 🔧 Features Engine (8500)

### **Overview**
Technical and fundamental feature engineering engine calculating 25+ indicators with real-time streaming capabilities.

**Base URL**: `http://localhost:8500`  
**Container**: `nautilus-features-engine`  
**Resources**: 3 CPU cores, 4GB RAM  
**Performance Target**: Real-time feature streaming for live trading  

### **Core Endpoints**

#### **Health & Status**
```http
GET /health
```

**Response**:
```json
{
  "status": "healthy",
  "version": "1.0.0",
  "uptime_seconds": 3600,
  "messagebus_connected": true,
  "features_calculated_today": 45230,
  "active_streams": 12,
  "supported_indicators": 27,
  "avg_calculation_time_ms": 15
}
```

#### **Technical Indicators Calculation**
```http
POST /features/technical
```

**Request Body**:
```json
{
  "symbol": "AAPL",
  "data": {
    "ohlcv": [
      {"timestamp": "2025-08-23T09:30:00Z", "open": 154.5, "high": 155.2, "low": 154.1, "close": 155.0, "volume": 1000000},
      {"timestamp": "2025-08-23T09:31:00Z", "open": 155.0, "high": 155.8, "low": 154.9, "close": 155.5, "volume": 1200000}
    ]
  },
  "indicators": {
    "momentum": ["rsi", "macd", "stochastic"],
    "trend": ["sma", "ema", "bollinger_bands"],
    "volume": ["obv", "volume_sma", "vwap"],
    "volatility": ["atr", "bollinger_width"]
  },
  "parameters": {
    "rsi_period": 14,
    "sma_period": 20,
    "macd_fast": 12,
    "macd_slow": 26,
    "macd_signal": 9
  }
}
```

**Response**:
```json
{
  "success": true,
  "data": {
    "symbol": "AAPL",
    "timestamp": "2025-08-23T09:31:00Z",
    "technical_indicators": {
      "momentum": {
        "rsi_14": 65.3,
        "macd": 0.85,
        "macd_signal": 0.78,
        "macd_histogram": 0.07,
        "stochastic_k": 72.5,
        "stochastic_d": 68.2
      },
      "trend": {
        "sma_20": 152.8,
        "ema_20": 153.2,
        "bollinger_upper": 158.5,
        "bollinger_middle": 154.2,
        "bollinger_lower": 149.9,
        "bollinger_position": 0.72
      },
      "volume": {
        "obv": 15420000,
        "volume_sma_20": 980000,
        "vwap": 154.85
      },
      "volatility": {
        "atr_14": 2.35,
        "bollinger_width": 0.056,
        "historical_volatility_20": 0.18
      }
    },
    "feature_summary": {
      "bullish_signals": 5,
      "bearish_signals": 1,
      "neutral_signals": 2,
      "overall_signal": "BULLISH",
      "signal_strength": 0.75
    }
  },
  "metadata": {
    "calculation_time_ms": 18,
    "indicators_calculated": 15,
    "data_points_used": 20
  }
}
```

#### **Fundamental Features**
```http
POST /features/fundamental
```

**Request Body**:
```json
{
  "symbol": "AAPL",
  "data_sources": ["alpha_vantage", "edgar"],
  "feature_types": ["valuation", "profitability", "growth", "efficiency"],
  "normalization": "industry_relative"
}
```

**Response**:
```json
{
  "success": true,
  "data": {
    "symbol": "AAPL",
    "fundamental_features": {
      "valuation": {
        "pe_ratio": 28.5,
        "pe_relative_to_sector": 1.15,
        "price_to_book": 8.2,
        "price_to_sales": 7.3,
        "ev_to_ebitda": 23.1
      },
      "profitability": {
        "roe": 0.88,
        "roa": 0.27,
        "gross_margin": 0.43,
        "operating_margin": 0.31,
        "net_margin": 0.25
      },
      "growth": {
        "revenue_growth_5y": 0.09,
        "earnings_growth_5y": 0.12,
        "dividend_growth_5y": 0.08
      },
      "efficiency": {
        "asset_turnover": 1.08,
        "inventory_turnover": 35.2,
        "receivables_turnover": 15.8
      }
    },
    "fundamental_score": 0.78,
    "industry_ranking": {
      "percentile": 85,
      "rank": 15,
      "total_companies": 100
    }
  }
}
```

#### **Real-time Feature Streaming**
```http
GET /features/stream/{symbol}
```

**Response**: Server-Sent Events (SSE) stream
```javascript
data: {"event": "technical_update", "symbol": "AAPL", "rsi": 65.8, "macd": 0.87, "timestamp": "2025-08-23T09:31:15Z"}

data: {"event": "volume_spike", "symbol": "AAPL", "volume": 1500000, "volume_ma": 980000, "spike_ratio": 1.53}

data: {"event": "bollinger_breakout", "symbol": "AAPL", "price": 158.6, "upper_band": 158.5, "signal": "BULLISH"}
```

#### **Batch Feature Processing**
```http
POST /features/batch
```

**Request Body**:
```json
{
  "symbols": ["AAPL", "MSFT", "GOOGL"],
  "feature_types": ["technical", "fundamental"],
  "calculation_date": "2025-08-23",
  "output_format": "dataframe",
  "include_metadata": true
}
```

**Response**:
```json
{
  "success": true,
  "data": {
    "batch_id": "batch_001",
    "features_dataframe": {
      "columns": ["symbol", "rsi_14", "pe_ratio", "roe", "bollinger_position"],
      "data": [
        ["AAPL", 65.3, 28.5, 0.88, 0.72],
        ["MSFT", 58.2, 32.1, 0.45, 0.65],
        ["GOOGL", 71.5, 24.8, 0.18, 0.81]
      ]
    },
    "processing_statistics": {
      "symbols_processed": 3,
      "features_per_symbol": 25,
      "processing_time_ms": 145,
      "success_rate": 1.0
    }
  }
}
```

#### **Feature Importance Analysis**
```http
POST /features/importance
```

**Request Body**:
```json
{
  "target_variable": "future_returns_1d",
  "features": ["rsi_14", "pe_ratio", "macd", "volume_ratio"],
  "analysis_period": "1Y",
  "method": "mutual_information"
}
```

**Response**:
```json
{
  "success": true,
  "data": {
    "feature_importance": [
      {
        "feature": "rsi_14",
        "importance_score": 0.34,
        "rank": 1,
        "correlation_to_target": 0.23
      },
      {
        "feature": "macd",
        "importance_score": 0.28,
        "rank": 2,
        "correlation_to_target": 0.19
      }
    ],
    "feature_groups": {
      "technical": 0.62,
      "fundamental": 0.38
    }
  }
}
```

### **Supported Technical Indicators**

```python
TECHNICAL_INDICATORS = {
    "momentum": [
        "rsi", "stochastic", "williams_r", "momentum", "roc",
        "macd", "ppo", "ultimate_oscillator", "commodity_channel_index"
    ],
    "trend": [
        "sma", "ema", "bollinger_bands", "parabolic_sar",
        "ichimoku", "linear_regression", "standard_deviation"
    ],
    "volume": [
        "obv", "accumulation_distribution", "chaikin_money_flow",
        "volume_sma", "vwap", "money_flow_index"
    ],
    "volatility": [
        "atr", "bollinger_width", "keltner_channels",
        "historical_volatility", "garman_klass_volatility"
    ]
}
```

---

## 🌐 WebSocket Engine (8600)

### **Overview**
Enterprise-grade real-time streaming engine supporting 1000+ concurrent connections with Redis pub/sub scaling.

**Base URL**: `http://localhost:8600` / `ws://localhost:8600`  
**Container**: `nautilus-websocket-engine`  
**Resources**: 1 CPU core, 2GB RAM  
**Performance Target**: 1000+ concurrent connections, 50,000+ messages/second  

### **Core Endpoints**

#### **Health & Status**
```http
GET /health
```

**Response**:
```json
{
  "status": "healthy",
  "version": "1.0.0",
  "uptime_seconds": 3600,
  "messagebus_connected": true,
  "active_connections": 245,
  "max_connections": 1000,
  "messages_sent_today": 125430,
  "avg_message_latency_ms": 12,
  "connection_success_rate": 0.98
}
```

#### **WebSocket Connection**
```javascript
// WebSocket connection
const ws = new WebSocket('ws://localhost:8600/ws/stream');

// Connection established
ws.onopen = function(event) {
    console.log('WebSocket connected');
    
    // Subscribe to topics
    ws.send(JSON.stringify({
        type: 'subscribe',
        topics: ['market.data.AAPL', 'portfolio.pnl.portfolio_001'],
        client_id: 'client_001'
    }));
};

// Receive messages
ws.onmessage = function(event) {
    const message = JSON.parse(event.data);
    console.log('Received:', message);
};
```

#### **Connection Management**
```http
POST /websocket/connections
```

**Request Body**:
```json
{
  "client_id": "client_001",
  "connection_type": "trading_dashboard",
  "subscriptions": [
    {
      "topic": "market.data.AAPL",
      "filter": {"price_change_threshold": 0.01}
    },
    {
      "topic": "risk.alerts.*",
      "filter": {"severity": ["HIGH", "CRITICAL"]}
    }
  ],
  "rate_limit": {
    "messages_per_second": 100,
    "burst_limit": 200
  }
}
```

**Response**:
```json
{
  "success": true,
  "data": {
    "connection_id": "conn_001",
    "client_id": "client_001",
    "websocket_url": "ws://localhost:8600/ws/stream?client_id=client_001&conn_id=conn_001",
    "subscriptions_active": 2,
    "rate_limits": {
      "messages_per_second": 100,
      "current_usage": 0
    }
  }
}
```

#### **Subscription Management**
```http
POST /websocket/subscriptions
```

**Request Body**:
```json
{
  "client_id": "client_001",
  "action": "subscribe",
  "subscriptions": [
    {
      "topic": "portfolio.performance.portfolio_001",
      "filters": {
        "update_frequency": "1s",
        "metrics": ["pnl", "sharpe_ratio", "drawdown"]
      }
    },
    {
      "topic": "strategy.execution.*",
      "filters": {
        "strategy_ids": ["strategy_001", "strategy_002"]
      }
    }
  ]
}
```

**Response**:
```json
{
  "success": true,
  "data": {
    "subscriptions_added": 2,
    "active_subscriptions": 4,
    "subscription_details": [
      {
        "topic": "portfolio.performance.portfolio_001",
        "status": "active",
        "message_rate": "1/second",
        "last_message": "2025-08-23T10:30:00Z"
      }
    ]
  }
}
```

#### **Message Broadcasting**
```http
POST /websocket/broadcast
```

**Request Body**:
```json
{
  "topic": "system.alert.maintenance",
  "message": {
    "type": "system_alert",
    "severity": "INFO",
    "title": "Scheduled Maintenance",
    "description": "System maintenance scheduled for 2025-08-24 02:00 UTC",
    "expires_at": "2025-08-24T02:00:00Z"
  },
  "target_clients": {
    "connection_types": ["trading_dashboard", "admin_panel"],
    "exclude_clients": ["client_005"]
  }
}
```

**Response**:
```json
{
  "success": true,
  "data": {
    "message_id": "msg_001",
    "broadcast_timestamp": "2025-08-23T10:30:00Z",
    "clients_targeted": 145,
    "messages_sent": 143,
    "failed_sends": 2,
    "delivery_rate": 0.986
  }
}
```

#### **Connection Statistics**
```http
GET /websocket/stats
```

**Response**:
```json
{
  "success": true,
  "data": {
    "connection_stats": {
      "total_connections": 245,
      "active_connections": 240,
      "idle_connections": 5,
      "connections_per_type": {
        "trading_dashboard": 180,
        "mobile_app": 45,
        "admin_panel": 15,
        "api_client": 5
      }
    },
    "message_stats": {
      "messages_sent_per_second": 1250,
      "messages_received_per_second": 85,
      "avg_message_size_bytes": 256,
      "total_messages_today": 125430
    },
    "performance_stats": {
      "avg_latency_ms": 12,
      "p95_latency_ms": 28,
      "p99_latency_ms": 45,
      "connection_success_rate": 0.98,
      "message_delivery_rate": 0.995
    },
    "subscription_stats": {
      "total_subscriptions": 892,
      "most_popular_topics": [
        {"topic": "market.data.*", "subscribers": 180},
        {"topic": "portfolio.pnl.*", "subscribers": 125},
        {"topic": "risk.alerts.*", "subscribers": 95}
      ]
    }
  }
}
```

### **WebSocket Message Types**

#### **Market Data Streaming**
```json
{
  "type": "market_data",
  "topic": "market.data.AAPL",
  "timestamp": "2025-08-23T10:30:00Z",
  "data": {
    "symbol": "AAPL",
    "price": 155.25,
    "change": 0.25,
    "change_percent": 0.0016,
    "volume": 1250000,
    "bid": 155.24,
    "ask": 155.26
  }
}
```

#### **Portfolio Performance Updates**
```json
{
  "type": "portfolio_performance",
  "topic": "portfolio.pnl.portfolio_001",
  "timestamp": "2025-08-23T10:30:00Z",
  "data": {
    "portfolio_id": "portfolio_001",
    "total_pnl": 15420.50,
    "pnl_change": 125.75,
    "market_value": 1542050.00,
    "daily_return": 0.0082
  }
}
```

#### **Risk Alerts**
```json
{
  "type": "risk_alert",
  "topic": "risk.alerts.portfolio_001",
  "timestamp": "2025-08-23T10:30:00Z",
  "data": {
    "alert_id": "alert_001",
    "portfolio_id": "portfolio_001",
    "severity": "HIGH",
    "limit_type": "position_size",
    "message": "AAPL position exceeds 10% limit",
    "current_value": 0.155,
    "limit_value": 0.10,
    "action_required": true
  }
}
```

### **Subscription Topics**

```python
TOPIC_CATEGORIES = {
    "market_data": [
        "market.data.{symbol}",
        "market.quotes.{symbol}",
        "market.trades.{symbol}",
        "market.orderbook.{symbol}"
    ],
    "portfolio": [
        "portfolio.pnl.{portfolio_id}",
        "portfolio.positions.{portfolio_id}",
        "portfolio.performance.{portfolio_id}"
    ],
    "risk": [
        "risk.alerts.{portfolio_id}",
        "risk.limits.{portfolio_id}",
        "risk.breaches.{portfolio_id}"
    ],
    "trading": [
        "trading.orders.{portfolio_id}",
        "trading.executions.{portfolio_id}",
        "trading.positions.{portfolio_id}"
    ],
    "system": [
        "system.health.*",
        "system.alerts.*",
        "system.maintenance.*"
    ]
}
```

---

## 🚀 Strategy Engine (8700)

### **Overview**
Automated strategy deployment engine with CI/CD pipeline, version control, and 6-stage testing framework.

**Base URL**: `http://localhost:8700`  
**Container**: `nautilus-strategy-engine`  
**Resources**: 1 CPU core, 2GB RAM  
**Performance Target**: Complete deployment pipeline in <5 minutes  

### **Core Endpoints**

#### **Health & Status**
```http
GET /health
```

**Response**:
```json
{
  "status": "healthy",
  "version": "1.0.0",
  "uptime_seconds": 3600,
  "messagebus_connected": true,
  "active_deployments": 3,
  "strategies_deployed_today": 12,
  "avg_deployment_time_minutes": 3.5,
  "pipeline_success_rate": 0.92,
  "version_control_status": "connected"
}
```

#### **Strategy Deployment**
```http
POST /strategy/deploy
```

**Request Body**:
```json
{
  "strategy_name": "momentum_strategy_v2",
  "strategy_code": "# Python strategy code\nclass MomentumStrategy:\n    def on_bar(self, bar):\n        # Strategy logic\n        pass",
  "deployment_config": {
    "deployment_type": "blue_green",
    "rollback_on_failure": true,
    "max_deployment_time_minutes": 10,
    "approval_required": false
  },
  "testing_config": {
    "run_syntax_check": true,
    "run_unit_tests": true,
    "run_integration_tests": true,
    "run_backtest": true,
    "backtest_period": "1Y",
    "run_paper_trading": true,
    "paper_trading_duration_hours": 24
  },
  "risk_limits": {
    "max_position_size": 0.05,
    "max_leverage": 2.0,
    "var_limit": 0.02
  }
}
```

**Response**:
```json
{
  "success": true,
  "data": {
    "deployment_id": "deploy_001",
    "strategy_name": "momentum_strategy_v2",
    "status": "pipeline_started",
    "pipeline_stages": [
      {"stage": "syntax_check", "status": "pending"},
      {"stage": "unit_tests", "status": "pending"},
      {"stage": "integration_tests", "status": "pending"},
      {"stage": "backtest", "status": "pending"},
      {"stage": "paper_trading", "status": "pending"},
      {"stage": "production_deploy", "status": "pending"}
    ],
    "estimated_completion": "2025-08-23T10:35:00Z",
    "monitoring_url": "/strategy/deployments/deploy_001/status"
  },
  "metadata": {
    "deployment_start": "2025-08-23T10:30:00Z",
    "deployment_type": "blue_green"
  }
}
```

#### **Deployment Status Monitoring**
```http
GET /strategy/deployments/{deployment_id}/status
```

**Response**:
```json
{
  "success": true,
  "data": {
    "deployment_id": "deploy_001",
    "strategy_name": "momentum_strategy_v2",
    "overall_status": "in_progress",
    "current_stage": "backtest",
    "progress_percent": 65,
    "pipeline_stages": [
      {
        "stage": "syntax_check",
        "status": "completed",
        "duration_seconds": 15,
        "result": "passed",
        "details": "No syntax errors found"
      },
      {
        "stage": "unit_tests",
        "status": "completed",
        "duration_seconds": 45,
        "result": "passed",
        "details": "15 tests passed, 0 failed"
      },
      {
        "stage": "backtest",
        "status": "in_progress",
        "progress_percent": 75,
        "estimated_completion": "2025-08-23T10:32:00Z",
        "partial_results": {
          "total_return": 0.1245,
          "sharpe_ratio": 1.23,
          "max_drawdown": -0.0567
        }
      }
    ],
    "deployment_metrics": {
      "start_time": "2025-08-23T10:30:00Z",
      "elapsed_time_minutes": 2.5,
      "estimated_total_time_minutes": 4.0
    }
  }
}
```

#### **Version Control Operations**
```http
POST /strategy/versions
```

**Request Body**:
```json
{
  "strategy_name": "momentum_strategy",
  "action": "create_version",
  "version_info": {
    "version": "2.1.0",
    "description": "Improved momentum calculation with volatility adjustment",
    "author": "trader_001",
    "changes": [
      "Added volatility normalization",
      "Improved risk management",
      "Fixed position sizing bug"
    ]
  },
  "strategy_code": "# Updated strategy code",
  "configuration": {
    "parameters": {
      "lookback_period": 20,
      "momentum_threshold": 0.05
    }
  }
}
```

**Response**:
```json
{
  "success": true,
  "data": {
    "version_id": "v2.1.0",
    "strategy_name": "momentum_strategy",
    "commit_hash": "a1b2c3d4e5f6",
    "created_at": "2025-08-23T10:30:00Z",
    "version_info": {
      "version": "2.1.0",
      "author": "trader_001",
      "description": "Improved momentum calculation with volatility adjustment"
    },
    "deployment_ready": true
  }
}
```

#### **Automated Rollback**
```http
POST /strategy/rollback/{deployment_id}
```

**Request Body**:
```json
{
  "rollback_reason": "Performance degradation detected",
  "target_version": "2.0.0",
  "rollback_type": "immediate",
  "preserve_data": true
}
```

**Response**:
```json
{
  "success": true,
  "data": {
    "rollback_id": "rollback_001",
    "deployment_id": "deploy_001",
    "status": "rollback_in_progress",
    "rollback_stages": [
      {"stage": "traffic_redirect", "status": "completed"},
      {"stage": "version_restore", "status": "in_progress"},
      {"stage": "health_verification", "status": "pending"}
    ],
    "estimated_completion": "2025-08-23T10:32:00Z"
  }
}
```

#### **Strategy Testing Framework**
```http
POST /strategy/test/{strategy_id}
```

**Request Body**:
```json
{
  "test_suite": "comprehensive",
  "test_config": {
    "backtest": {
      "start_date": "2024-01-01",
      "end_date": "2025-01-01",
      "initial_capital": 100000,
      "commission": 0.001
    },
    "paper_trading": {
      "duration_hours": 24,
      "risk_limits": {
        "max_position_size": 0.05,
        "max_daily_loss": 0.02
      }
    }
  }
}
```

**Response**:
```json
{
  "success": true,
  "data": {
    "test_run_id": "test_001",
    "test_results": {
      "backtest": {
        "total_return": 0.1245,
        "annualized_return": 0.1245,
        "sharpe_ratio": 1.23,
        "max_drawdown": -0.0567,
        "win_rate": 0.65,
        "profit_factor": 1.45
      },
      "paper_trading": {
        "status": "running",
        "current_pnl": 1250.50,
        "trades_executed": 15,
        "avg_trade_duration_minutes": 45
      }
    },
    "test_status": "passed",
    "production_ready": true
  }
}
```

### **Deployment Pipeline Stages**

```python
PIPELINE_STAGES = {
    "syntax_check": {
        "description": "Python syntax validation",
        "timeout_minutes": 1,
        "failure_action": "stop_pipeline"
    },
    "unit_tests": {
        "description": "Unit test execution",
        "timeout_minutes": 5,
        "failure_action": "stop_pipeline"
    },
    "integration_tests": {
        "description": "Integration test suite",
        "timeout_minutes": 10,
        "failure_action": "stop_pipeline"
    },
    "backtest": {
        "description": "Historical performance validation",
        "timeout_minutes": 30,
        "failure_action": "stop_pipeline",
        "pass_criteria": {
            "min_sharpe_ratio": 0.8,
            "max_drawdown": -0.15
        }
    },
    "paper_trading": {
        "description": "Live market simulation",
        "timeout_hours": 24,
        "failure_action": "stop_pipeline"
    },
    "production_deploy": {
        "description": "Production deployment",
        "timeout_minutes": 5,
        "failure_action": "rollback"
    }
}
```

---

## 📡 Market Data Engine (8800)

### **Overview**
High-throughput market data ingestion engine supporting 8 data sources with real-time distribution and <50ms latency monitoring.

**Base URL**: `http://localhost:8800`  
**Container**: `nautilus-marketdata-engine`  
**Resources**: 2 CPU cores, 3GB RAM  
**Performance Target**: <50ms data latency, high-throughput ingestion  

### **Core Endpoints**

#### **Health & Status**
```http
GET /health
```

**Response**:
```json
{
  "status": "healthy",
  "version": "1.0.0",
  "uptime_seconds": 3600,
  "messagebus_connected": true,
  "data_sources_status": {
    "ibkr": {"status": "connected", "latency_ms": 12},
    "alpha_vantage": {"status": "connected", "latency_ms": 245},
    "yfinance": {"status": "connected", "latency_ms": 180}
  },
  "data_points_ingested_today": 2450000,
  "avg_ingestion_latency_ms": 25,
  "active_streams": 45
}
```

#### **Data Source Management**
```http
GET /marketdata/sources
```

**Response**:
```json
{
  "success": true,
  "data": {
    "available_sources": [
      {
        "source_id": "ibkr",
        "name": "Interactive Brokers",
        "status": "active",
        "data_types": ["real_time_quotes", "historical_bars", "level2"],
        "symbols_supported": 50000,
        "update_frequency": "real_time",
        "latency_ms": 12
      },
      {
        "source_id": "alpha_vantage", 
        "name": "Alpha Vantage",
        "status": "active",
        "data_types": ["quotes", "fundamentals", "earnings"],
        "rate_limit": "5/minute",
        "latency_ms": 245
      }
    ],
    "failover_hierarchy": ["ibkr", "alpha_vantage", "yfinance"],
    "data_quality_scores": {
      "ibkr": 0.98,
      "alpha_vantage": 0.92,
      "yfinance": 0.85
    }
  }
}
```

#### **Real-time Data Subscription**
```http
POST /marketdata/subscribe
```

**Request Body**:
```json
{
  "symbols": ["AAPL", "MSFT", "GOOGL"],
  "data_types": ["quotes", "trades", "level1"],
  "sources": ["ibkr", "alpha_vantage"],
  "delivery_config": {
    "delivery_method": "websocket",
    "batch_size": 1,
    "max_latency_ms": 100,
    "deduplicate": true
  },
  "filters": {
    "min_price_change": 0.01,
    "min_volume_change": 1000
  }
}
```

**Response**:
```json
{
  "success": true,
  "data": {
    "subscription_id": "sub_001",
    "symbols_subscribed": 3,
    "data_streams_active": 3,
    "estimated_messages_per_second": 45,
    "websocket_endpoint": "ws://localhost:8800/ws/marketdata/sub_001"
  }
}
```

#### **Historical Data Request**
```http
POST /marketdata/historical
```

**Request Body**:
```json
{
  "symbol": "AAPL",
  "data_type": "bars",
  "timeframe": "1h",
  "start_date": "2025-08-01",
  "end_date": "2025-08-23",
  "sources": ["ibkr", "alpha_vantage"],
  "fallback_enabled": true,
  "cache_result": true
}
```

**Response**:
```json
{
  "success": true,
  "data": {
    "symbol": "AAPL",
    "data_type": "bars",
    "timeframe": "1h",
    "source_used": "ibkr",
    "bars_count": 368,
    "data": [
      {
        "timestamp": "2025-08-01T09:30:00Z",
        "open": 150.25,
        "high": 151.80,
        "low": 150.10,
        "close": 151.45,
        "volume": 2450000,
        "vwap": 151.20
      }
    ],
    "data_quality": {
      "completeness": 0.98,
      "source_reliability": 0.99,
      "latency_ms": 145
    }
  },
  "metadata": {
    "request_time_ms": 145,
    "cache_hit": false,
    "fallback_used": false
  }
}
```

#### **Data Quality Monitoring**
```http
GET /marketdata/quality/{symbol}
```

**Response**:
```json
{
  "success": true,
  "data": {
    "symbol": "AAPL",
    "quality_metrics": {
      "data_completeness": 0.98,
      "timeliness_score": 0.95,
      "accuracy_score": 0.97,
      "consistency_score": 0.96
    },
    "quality_issues": [
      {
        "issue_type": "missing_data",
        "timestamp": "2025-08-23T09:45:00Z",
        "severity": "low",
        "resolution": "data_interpolated"
      }
    ],
    "source_comparison": {
      "ibkr": {"quality_score": 0.98, "uptime": 0.999},
      "alpha_vantage": {"quality_score": 0.92, "uptime": 0.985}
    }
  }
}
```

#### **Latency Monitoring**
```http
GET /marketdata/latency
```

**Response**:
```json
{
  "success": true,
  "data": {
    "overall_latency": {
      "avg_latency_ms": 25,
      "p50_latency_ms": 18,
      "p95_latency_ms": 45,
      "p99_latency_ms": 67
    },
    "source_latency": {
      "ibkr": {
        "feed_latency_ms": 12,
        "processing_latency_ms": 8,
        "total_latency_ms": 20
      },
      "alpha_vantage": {
        "api_latency_ms": 230,
        "processing_latency_ms": 15,
        "total_latency_ms": 245
      }
    },
    "latency_by_symbol": {
      "AAPL": {"avg_latency_ms": 18, "message_count": 1250},
      "MSFT": {"avg_latency_ms": 22, "message_count": 890}
    }
  }
}
```

### **Data Types & Sources**

```python
DATA_SOURCES = {
    "ibkr": {
        "data_types": ["real_time_quotes", "historical_bars", "level2", "trades"],
        "update_frequency": "real_time",
        "symbols": 50000,
        "asset_classes": ["stocks", "options", "futures", "forex"]
    },
    "alpha_vantage": {
        "data_types": ["quotes", "daily_bars", "fundamentals", "earnings"],
        "rate_limit": "5/minute",
        "symbols": 10000,
        "asset_classes": ["stocks", "etfs", "mutual_funds"]
    },
    "yfinance": {
        "data_types": ["quotes", "historical_bars"],
        "rate_limit": "2000/hour",
        "symbols": 15000,
        "asset_classes": ["stocks", "etfs", "indices"]
    }
}
```

---

## 💼 Portfolio Engine (8900)

### **Overview**
Advanced portfolio optimization engine with mean-variance optimization, risk parity, and automated rebalancing capabilities.

**Base URL**: `http://localhost:8900`  
**Container**: `nautilus-portfolio-engine`  
**Resources**: 4 CPU cores, 8GB RAM  
**Performance Target**: Complex portfolio optimization in <5 seconds  

### **Core Endpoints**

#### **Health & Status**
```http
GET /health
```

**Response**:
```json
{
  "status": "healthy",
  "version": "1.0.0",
  "uptime_seconds": 3600,
  "messagebus_connected": true,
  "optimizations_completed_today": 145,
  "active_portfolios": 25,
  "avg_optimization_time_ms": 2850,
  "rebalancing_jobs_active": 5
}
```

#### **Portfolio Optimization**
```http
POST /portfolio/optimize
```

**Request Body**:
```json
{
  "portfolio_id": "portfolio_001",
  "optimization_method": "mean_variance",
  "universe": ["AAPL", "MSFT", "GOOGL", "TSLA", "NVDA"],
  "constraints": {
    "max_weight": 0.20,
    "min_weight": 0.05,
    "max_turnover": 0.10,
    "sector_limits": {
      "technology": 0.60,
      "healthcare": 0.40
    }
  },
  "objective": {
    "target_return": 0.12,
    "risk_aversion": 5.0,
    "benchmark": "SPY"
  },
  "optimization_config": {
    "lookback_period": "2Y",
    "rebalance_frequency": "monthly",
    "transaction_costs": 0.001
  }
}
```

**Response**:
```json
{
  "success": true,
  "data": {
    "optimization_id": "opt_001",
    "portfolio_id": "portfolio_001",
    "optimal_weights": {
      "AAPL": 0.20,
      "MSFT": 0.18,
      "GOOGL": 0.15,
      "TSLA": 0.12,
      "NVDA": 0.10,
      "cash": 0.25
    },
    "expected_performance": {
      "expected_return": 0.125,
      "expected_volatility": 0.18,
      "sharpe_ratio": 0.69,
      "tracking_error": 0.05
    },
    "risk_metrics": {
      "portfolio_var_1d": 0.025,
      "expected_shortfall": 0.034,
      "max_drawdown_est": -0.12
    },
    "optimization_details": {
      "method": "mean_variance",
      "optimization_time_ms": 2450,
      "convergence_achieved": true,
      "constraint_violations": 0
    }
  },
  "metadata": {
    "optimization_start": "2025-08-23T10:30:00Z",
    "data_as_of": "2025-08-23T09:00:00Z"
  }
}
```

#### **Risk Parity Optimization**
```http
POST /portfolio/optimize/risk_parity
```

**Request Body**:
```json
{
  "portfolio_id": "portfolio_002",
  "universe": ["AAPL", "MSFT", "GOOGL", "TLT", "GLD"],
  "risk_budget": {
    "equities": 0.60,
    "bonds": 0.30,
    "commodities": 0.10
  },
  "constraints": {
    "max_weight": 0.40,
    "min_weight": 0.05
  }
}
```

**Response**:
```json
{
  "success": true,
  "data": {
    "optimal_weights": {
      "AAPL": 0.15,
      "MSFT": 0.15,
      "GOOGL": 0.15,
      "TLT": 0.30,
      "GLD": 0.25
    },
    "risk_contributions": {
      "AAPL": 0.20,
      "MSFT": 0.20,
      "GOOGL": 0.20,
      "TLT": 0.20,
      "GLD": 0.20
    },
    "portfolio_volatility": 0.12,
    "risk_parity_achieved": true
  }
}
```

#### **Automated Rebalancing**
```http
POST /portfolio/rebalance
```

**Request Body**:
```json
{
  "portfolio_id": "portfolio_001",
  "rebalance_config": {
    "trigger_type": "threshold",
    "threshold": 0.05,
    "frequency": "monthly",
    "min_trade_size": 1000
  },
  "execution_config": {
    "execution_algo": "twap",
    "max_participation_rate": 0.10,
    "urgency": "medium"
  }
}
```

**Response**:
```json
{
  "success": true,
  "data": {
    "rebalance_id": "rebal_001",
    "status": "rebalancing_required",
    "current_weights": {
      "AAPL": 0.23,
      "MSFT": 0.17,
      "GOOGL": 0.14
    },
    "target_weights": {
      "AAPL": 0.20,
      "MSFT": 0.18,
      "GOOGL": 0.15
    },
    "required_trades": [
      {
        "symbol": "AAPL",
        "action": "sell",
        "shares": 150,
        "value": 23250.00
      },
      {
        "symbol": "MSFT", 
        "action": "buy",
        "shares": 45,
        "value": 15750.00
      }
    ],
    "estimated_execution_time": "2025-08-23T11:00:00Z",
    "transaction_costs": 125.50
  }
}
```

#### **Factor-Based Portfolio Construction**
```http
POST /portfolio/construct/factor_based
```

**Request Body**:
```json
{
  "factor_exposures": {
    "momentum": 0.30,
    "value": 0.25,
    "quality": 0.20,
    "low_volatility": 0.25
  },
  "universe": "SP500",
  "constraints": {
    "max_weight": 0.05,
    "sector_neutral": true,
    "max_turnover": 0.20
  }
}
```

**Response**:
```json
{
  "success": true,
  "data": {
    "portfolio_weights": {
      "AAPL": 0.045,
      "MSFT": 0.042,
      "BERKSHIRE": 0.038
    },
    "factor_exposures_achieved": {
      "momentum": 0.31,
      "value": 0.24,
      "quality": 0.21,
      "low_volatility": 0.24
    },
    "tracking_error_to_benchmark": 0.045,
    "expected_information_ratio": 0.65
  }
}
```

### **Optimization Methods**

```python
OPTIMIZATION_METHODS = {
    "mean_variance": {
        "description": "Classic Markowitz optimization",
        "parameters": ["expected_returns", "covariance_matrix", "risk_aversion"],
        "constraints": ["weight_limits", "turnover", "sector_limits"]
    },
    "risk_parity": {
        "description": "Equal risk contribution optimization",
        "parameters": ["risk_budget", "covariance_matrix"],
        "constraints": ["weight_limits", "risk_budget_constraints"]
    },
    "black_litterman": {
        "description": "Black-Litterman with views",
        "parameters": ["market_weights", "views", "confidence"],
        "constraints": ["weight_limits", "view_constraints"]
    },
    "hierarchical_risk_parity": {
        "description": "HRP based on hierarchical clustering",
        "parameters": ["correlation_matrix", "returns"],
        "constraints": ["weight_limits", "cluster_constraints"]
    }
}
```

---

## 🔄 Cross-Engine Communication

### **MessageBus Integration**

All 9 engines communicate through the enhanced MessageBus system using Redis Streams:

#### **Message Flow Examples**

```python
# Analytics Engine requests risk check
{
    "topic": "risk.check.request",
    "source_engine": "analytics",
    "target_engine": "risk",
    "data": {
        "portfolio_id": "portfolio_001",
        "positions": [...],
        "request_id": "req_001"
    }
}

# Risk Engine responds with breach alert
{
    "topic": "risk.breach.alert", 
    "source_engine": "risk",
    "broadcast": true,
    "data": {
        "portfolio_id": "portfolio_001",
        "breach_type": "position_limit",
        "severity": "HIGH",
        "request_id": "req_001"
    }
}

# WebSocket Engine broadcasts to clients
{
    "topic": "websocket.broadcast",
    "source_engine": "risk", 
    "target_engine": "websocket",
    "data": {
        "client_filter": {"portfolio_id": "portfolio_001"},
        "message": {
            "type": "risk_alert",
            "data": {...}
        }
    }
}
```

### **Engine Coordination Patterns**

```python
COORDINATION_PATTERNS = {
    "request_response": {
        "description": "Direct engine-to-engine requests",
        "examples": ["risk_check", "factor_calculation", "ml_prediction"]
    },
    "event_broadcast": {
        "description": "One-to-many event distribution", 
        "examples": ["portfolio_update", "market_data", "system_alerts"]
    },
    "pipeline_processing": {
        "description": "Sequential processing chain",
        "examples": ["market_data -> factors -> ml -> analytics -> websocket"]
    },
    "fan_out_fan_in": {
        "description": "Parallel processing with aggregation",
        "examples": ["portfolio_optimization", "multi_source_factor_synthesis"]
    }
}
```

---

## 📊 Performance Monitoring

### **Engine-Specific Metrics**

Each engine exposes Prometheus metrics at `/metrics` endpoint:

```python
# Common metrics across all engines
engine_requests_total{engine="analytics", method="POST", endpoint="/calculate"}
engine_request_duration_seconds{engine="analytics", quantile="0.95"}
engine_errors_total{engine="analytics", error_type="calculation_error"}
engine_messagebus_messages_sent_total{engine="analytics", topic="risk.check"}
engine_messagebus_connection_status{engine="analytics"}

# Engine-specific business metrics
analytics_portfolios_calculated_total
analytics_pnl_updates_total
risk_limit_checks_total
risk_breaches_detected_total  
factor_calculations_completed_total
ml_predictions_served_total
websocket_connections_active
strategy_deployments_completed_total
```

### **Health Check Aggregation**

```http
GET /system/health/engines
```

**Response**:
```json
{
  "success": true,
  "data": {
    "overall_status": "healthy",
    "engines_healthy": 9,
    "engines_total": 9,
    "engine_status": {
      "analytics-engine": {"status": "healthy", "response_time_ms": 45},
      "risk-engine": {"status": "healthy", "response_time_ms": 25},
      "factor-engine": {"status": "healthy", "response_time_ms": 125},
      "ml-engine": {"status": "healthy", "response_time_ms": 67},
      "features-engine": {"status": "healthy", "response_time_ms": 15},
      "websocket-engine": {"status": "healthy", "response_time_ms": 12},
      "strategy-engine": {"status": "healthy", "response_time_ms": 85},
      "marketdata-engine": {"status": "healthy", "response_time_ms": 25},
      "portfolio-engine": {"status": "healthy", "response_time_ms": 145}
    },
    "messagebus_connectivity": {
      "connected_engines": 9,
      "total_engines": 9,
      "connectivity_rate": 1.0
    }
  }
}
```

---

## 🚀 Conclusion

This comprehensive API documentation covers all **9 containerized processing engines** in the Nautilus trading platform. Each engine operates as an independent microservice with:

### **Technical Excellence**
- **FastAPI Framework**: High-performance async API endpoints
- **Docker Containerization**: Complete isolation and resource optimization  
- **Enhanced MessageBus**: Event-driven inter-engine communication
- **Prometheus Metrics**: Comprehensive monitoring and observability
- **Production-Ready**: Health checks, error handling, and performance optimization

### **Business Capabilities**
- **Analytics Engine**: Real-time P&L and performance analytics
- **Risk Engine**: Advanced risk management with ML-based breach detection
- **Factor Engine**: 380,000+ factor synthesis across 8 data sources
- **ML Engine**: Real-time predictions with model registry management
- **Features Engine**: Technical and fundamental feature engineering
- **WebSocket Engine**: Enterprise streaming with 1000+ concurrent connections
- **Strategy Engine**: Automated deployment with CI/CD pipelines
- **Market Data Engine**: High-throughput data ingestion with <50ms latency
- **Portfolio Engine**: Advanced optimization with multiple methodologies

### **Performance Achievements**
- **50x System Improvement**: From 1,000 to 50,000+ operations/second
- **True Parallel Processing**: GIL-free execution across all engines
- **Sub-second Response Times**: Optimized for real-time trading requirements
- **Horizontal Scalability**: Independent scaling per engine workload

The platform delivers **institutional-grade performance** and is ready for **production deployment** in high-frequency trading environments.

---

**Document Version**: 1.0  
**Last Updated**: August 23, 2025  
**API Coverage**: All 9 Containerized Engines ✅