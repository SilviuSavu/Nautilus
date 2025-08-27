# VPIN Engine API Reference

**Service**: VPIN Market Microstructure Engine  
**Port**: 10000  
**Status**: ✅ Production Ready  
**Performance**: <2ms toxicity calculations, GPU-accelerated  
**Documentation**: http://localhost:10000/docs (Interactive Swagger UI)

## Overview

The VPIN (Volume-Synchronized Probability of Informed Trading) Engine provides real-time market microstructure analysis using IBKR Level 2 order book data. This engine detects informed trading activity and adverse selection through sophisticated volume bucket synchronization and neural pattern recognition.

## Core Capabilities

- **Level 2 Data Integration**: Full 10-level IBKR order book depth with exchange attribution
- **GPU-Accelerated VPIN**: Metal GPU optimization for real-time toxicity calculations (<2ms)
- **Neural Pattern Recognition**: M4 Max Neural Engine for market regime detection
- **Volume Synchronization**: Advanced trade classification with Lee-Ready algorithm
- **Informed Trading Detection**: Real-time probability scoring for adverse selection
- **Smart Order Flow Analysis**: Exchange routing patterns and liquidity provider behavior

## API Endpoints

### System Health & Status

#### `GET /health`
Get VPIN engine health status and component availability.

**Response**:
```json
{
  "status": "healthy",
  "timestamp": "2025-08-24T12:00:00Z",
  "components": {
    "level2_collector": "active",
    "volume_synchronizer": "active", 
    "gpu_calculator": "active",
    "neural_analyzer": "active"
  },
  "performance": {
    "vpin_calculation_time_ms": 1.8,
    "neural_analysis_time_ms": 4.2,
    "hardware_utilization": {
      "metal_gpu": 0.85,
      "neural_engine": 0.72
    }
  }
}
```

#### `GET /metrics`
Detailed performance metrics and hardware utilization.

### Real-time VPIN Analysis

#### `GET /realtime/{symbol}`
Get current VPIN reading with real-time toxicity analysis.

**Parameters**:
- `symbol` (path): Trading symbol (e.g., "AAPL", "TSLA")

**Response**:
```json
{
  "symbol": "AAPL",
  "timestamp": "2025-08-24T12:00:00Z",
  "vpin_value": 0.42,
  "toxicity_level": "MODERATE",
  "confidence": 0.87,
  "volume_bucket": {
    "bucket_id": 1523,
    "target_volume": 100000.0,
    "buy_volume": 58400.0,
    "sell_volume": 41600.0,
    "order_imbalance": 0.168
  },
  "market_regime": "NORMAL_VOLATILITY",
  "neural_patterns": {
    "informed_trading_probability": 0.42,
    "adverse_selection_risk": "MODERATE",
    "predicted_direction": "BULLISH"
  }
}
```

#### `GET /realtime/batch`
Get real-time VPIN data for multiple symbols.

**Query Parameters**:
- `symbols`: Comma-separated list of symbols (max 8)

**Response**: Array of realtime VPIN objects

### Historical VPIN Data

#### `GET /history/{symbol}`
Get historical VPIN values and patterns.

**Parameters**:
- `symbol` (path): Trading symbol
- `start_date` (query): Start date (ISO format)
- `end_date` (query): End date (ISO format)
- `bucket_size` (query): Volume bucket size (default: 100000)

**Response**:
```json
{
  "symbol": "AAPL",
  "period": {
    "start": "2025-08-20T09:30:00Z",
    "end": "2025-08-24T16:00:00Z"
  },
  "bucket_size": 100000,
  "data_points": 1247,
  "vpin_history": [
    {
      "timestamp": "2025-08-20T09:30:15Z",
      "vpin_value": 0.38,
      "toxicity_level": "LOW",
      "volume_bucket_id": 1450
    }
  ],
  "statistics": {
    "mean_vpin": 0.41,
    "volatility": 0.15,
    "high_toxicity_periods": 23,
    "informed_trading_events": 7
  }
}
```

### Volume Bucket Management

#### `GET /buckets/{symbol}/current`
Get current volume bucket state for a symbol.

**Response**:
```json
{
  "symbol": "AAPL",
  "current_bucket": {
    "bucket_id": 1523,
    "target_volume": 100000.0,
    "accumulated_volume": 76800.0,
    "completion_percentage": 76.8,
    "buy_volume": 44320.0,
    "sell_volume": 32480.0,
    "trades_count": 127,
    "time_in_bucket_ms": 2340
  },
  "next_bucket_eta_ms": 890
}
```

#### `POST /buckets/{symbol}/configure`
Configure volume bucket parameters for a symbol.

**Request Body**:
```json
{
  "target_volume": 50000.0,
  "bucket_count": 50,
  "enable_smart_sizing": true
}
```

### Toxicity Alerts & Monitoring

#### `GET /alerts/{symbol}`
Get active toxicity alerts for a symbol.

**Response**:
```json
{
  "symbol": "AAPL",
  "active_alerts": [
    {
      "alert_id": "alert_1692891234",
      "type": "HIGH_TOXICITY",
      "severity": "WARNING",
      "vpin_threshold": 0.7,
      "current_vpin": 0.73,
      "triggered_at": "2025-08-24T11:45:23Z",
      "duration_ms": 15000,
      "description": "VPIN above 0.7 indicating high informed trading probability"
    }
  ],
  "alert_history_24h": 3
}
```

#### `POST /alerts/{symbol}/subscribe`
Subscribe to real-time toxicity alerts via WebSocket.

**Request Body**:
```json
{
  "thresholds": {
    "high_toxicity": 0.65,
    "extreme_toxicity": 0.8
  },
  "alert_types": ["HIGH_TOXICITY", "REGIME_CHANGE", "INFORMED_TRADING"],
  "notification_method": "WEBSOCKET"
}
```

### Neural Pattern Analysis

#### `GET /patterns/{symbol}/analysis`
Get detailed neural pattern analysis for market regime detection.

**Response**:
```json
{
  "symbol": "AAPL",
  "timestamp": "2025-08-24T12:00:00Z",
  "pattern_analysis": {
    "market_regime": "NORMAL_VOLATILITY",
    "regime_confidence": 0.89,
    "regime_duration_ms": 45000,
    "pattern_features": {
      "volatility_cluster": false,
      "informed_trading_cluster": true,
      "order_flow_imbalance": "MODERATE_SELL",
      "liquidity_provision_pattern": "NORMAL"
    },
    "predictions": {
      "next_regime_probability": {
        "HIGH_VOLATILITY": 0.23,
        "LOW_VOLATILITY": 0.12,
        "NORMAL_VOLATILITY": 0.65
      },
      "price_direction": {
        "bullish_probability": 0.31,
        "bearish_probability": 0.69
      }
    }
  },
  "neural_engine_metrics": {
    "inference_time_ms": 4.1,
    "model_confidence": 0.87,
    "hardware_acceleration": "NEURAL_ENGINE"
  }
}
```

### Level 2 Order Book Integration

#### `GET /orderbook/{symbol}/depth`
Get current Level 2 order book depth with exchange attribution.

**Response**:
```json
{
  "symbol": "AAPL",
  "timestamp": "2025-08-24T12:00:00.123Z",
  "bid_depth": [
    {
      "level": 1,
      "price": 150.45,
      "size": 500,
      "exchange": "NYSE",
      "market_maker": "CDRG"
    }
  ],
  "ask_depth": [
    {
      "level": 1,
      "price": 150.46,
      "size": 300,
      "exchange": "NASDAQ",
      "market_maker": "ARCA"
    }
  ],
  "spread": 0.01,
  "depth_analysis": {
    "bid_depth_quality": "GOOD",
    "ask_depth_quality": "EXCELLENT", 
    "liquidity_imbalance": 0.25,
    "smart_routing_available": true
  }
}
```

### Configuration & Management

#### `GET /config`
Get current VPIN engine configuration.

#### `POST /config/update`
Update VPIN engine configuration parameters.

**Request Body**:
```json
{
  "default_bucket_size": 100000,
  "enable_neural_analysis": true,
  "gpu_acceleration": true,
  "alert_thresholds": {
    "moderate_toxicity": 0.5,
    "high_toxicity": 0.7,
    "extreme_toxicity": 0.85
  },
  "supported_symbols": ["AAPL", "TSLA", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "NFLX"]
}
```

#### `POST /symbols/{symbol}/subscribe`
Subscribe to Level 2 data and start VPIN calculation for a symbol.

#### `DELETE /symbols/{symbol}/unsubscribe`
Stop VPIN calculation and unsubscribe from Level 2 data.

## WebSocket Endpoints

### Real-time VPIN Streaming
- **Endpoint**: `ws://localhost:10000/ws/vpin/{symbol}`
- **Messages**: Real-time VPIN values, toxicity alerts, regime changes

### Level 2 Order Book Stream
- **Endpoint**: `ws://localhost:10000/ws/orderbook/{symbol}`
- **Messages**: Real-time order book updates with VPIN context

## Hardware Acceleration

The VPIN Engine leverages M4 Max hardware acceleration for optimal performance:

- **Metal GPU**: 40-core GPU for VPIN calculations (51x speedup)
- **Neural Engine**: 16-core Neural Engine for pattern recognition (7.3x speedup)
- **Unified Memory**: Zero-copy operations for high-frequency data processing
- **CPU Optimization**: 12 Performance cores for I/O and coordination

## Performance Metrics

**Validated Performance** (Production Environment):
- **VPIN Calculation**: <2ms real-time processing
- **Neural Analysis**: <5ms pattern recognition
- **Level 2 Processing**: <1ms order book updates
- **Trade Classification**: 95%+ accuracy
- **GPU Utilization**: 85% Metal GPU, 72% Neural Engine
- **Supported Symbols**: 8 concurrent Tier 1 symbols
- **Throughput**: 1000+ VPIN calculations/second

## Error Codes

- `VPIN_001`: Insufficient volume data for bucket completion
- `VPIN_002`: Level 2 data subscription failed
- `VPIN_003`: GPU acceleration unavailable
- `VPIN_004`: Neural Engine inference failed
- `VPIN_005`: Symbol not supported or not subscribed
- `VPIN_006`: Volume bucket configuration invalid
- `VPIN_007`: Trade classification algorithm failed

## Integration Examples

### Python Client
```python
import requests
import asyncio
import websockets

# Get real-time VPIN
response = requests.get("http://localhost:10000/realtime/AAPL")
vpin_data = response.json()
print(f"AAPL VPIN: {vpin_data['vpin_value']}, Toxicity: {vpin_data['toxicity_level']}")

# WebSocket streaming
async def stream_vpin():
    uri = "ws://localhost:10000/ws/vpin/AAPL"
    async with websockets.connect(uri) as websocket:
        async for message in websocket:
            vpin_update = json.loads(message)
            print(f"Real-time VPIN: {vpin_update}")
```

### cURL Examples
```bash
# Get real-time VPIN for AAPL
curl "http://localhost:10000/realtime/AAPL"

# Subscribe to Level 2 data
curl -X POST "http://localhost:10000/symbols/AAPL/subscribe" \
  -H "Content-Type: application/json"

# Get historical VPIN data
curl "http://localhost:10000/history/AAPL?start_date=2025-08-20&end_date=2025-08-24"

# Configure toxicity alerts
curl -X POST "http://localhost:10000/alerts/AAPL/subscribe" \
  -H "Content-Type: application/json" \
  -d '{"thresholds": {"high_toxicity": 0.65}, "alert_types": ["HIGH_TOXICITY"]}'
```

## Production Deployment

The VPIN Engine is production-ready with:
- ✅ Docker containerization with M4 Max optimizations
- ✅ Comprehensive error handling and fallback mechanisms
- ✅ Real-time monitoring and health checks
- ✅ Hardware acceleration with graceful CPU fallback
- ✅ Integration with existing risk management systems
- ✅ Professional API documentation and client libraries