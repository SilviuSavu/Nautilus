# Order Book API Documentation

**Feature**: Order Book Depth Visualization  
**Version**: 2.4  
**Last Updated**: August 19, 2025  

---

## Overview

This document covers the API endpoints and data flows for the Order Book functionality, including instrument search, real-time data subscriptions, and order book visualization.

---

## API Endpoints

### 1. Instrument Search

**Endpoint**: `GET /api/v1/ib/instruments/search/{query}`

**Description**: Search for tradeable instruments across multiple asset classes

**Parameters**:
- `query` (path, required): Symbol to search for (e.g., "AAPL", "MSFT")
- `sec_type` (query, optional): Security type filter
  - `STK` - Stocks (default when omitted)
  - `FUT` - Futures ⚠️ *Known limitation - timeouts*
  - `CASH` - Forex
  - `OPT` - Options  
  - `IND` - Indexes ⚠️ *Known limitation - timeouts*
- `exchange` (query, optional): Exchange filter (default: "SMART")
- `currency` (query, optional): Currency filter (default: "USD")
- `max_results` (query, optional): Maximum results to return (default: 50)

**Example Requests**:
```bash
# Search for Apple stock
GET /api/v1/ib/instruments/search/AAPL?sec_type=STK

# Search for EUR/USD forex
GET /api/v1/ib/instruments/search/EUR?sec_type=CASH

# Search across all asset classes (auto-detects best matches)
GET /api/v1/ib/instruments/search/AAPL
```

**Response Format**:
```json
{
  "instruments": [
    {
      "contract_id": 265598,
      "symbol": "AAPL",
      "name": "Apple Inc",
      "sec_type": "STK",
      "exchange": "NASDAQ",
      "currency": "USD",
      "local_symbol": "AAPL",
      "trading_class": "NMS",
      "multiplier": null,
      "expiry": null,
      "strike": null,
      "right": null,
      "primary_exchange": "NASDAQ",
      "description": "Apple Inc",
      "min_tick": 0.01,
      "price_magnifier": 1,
      "order_types": ["LIMIT", "MARKET", "STOP"],
      "valid_exchanges": ["SMART", "NASDAQ", "NYSE"],
      "market_hours": "America/New_York",
      "liquid_hours": "20231201:0930-20231201:1600",
      "timezone": "America/New_York"
    }
  ],
  "total": 1,
  "query": "AAPL",
  "timestamp": "2025-08-19T00:15:27.123456"
}
```

**Success Codes**:
- `200 OK` - Search completed successfully
- `200 OK` - Search completed with no results (empty instruments array)

**Error Codes**:
- `500 Internal Server Error` - Search timeout or IB Gateway connection issue
- `503 Service Unavailable` - IB Gateway not connected

---

### 2. IB Gateway Connection Status

**Endpoint**: `GET /api/v1/ib/connection/status`

**Description**: Check IB Gateway connection status

**Response Format**:
```json
{
  "connected": true,
  "gateway_type": "IB Gateway",
  "account_id": "DU7925702", 
  "connection_time": "2025-08-18T02:08:48.397731",
  "next_valid_order_id": 1,
  "host": "localhost",
  "port": 7496,
  "client_id": 1
}
```

---

## WebSocket Data Flows

### Order Book Subscription

**Message Type**: `order_book_subscription`

**Client to Server**:
```json
{
  "type": "order_book_subscription", 
  "action": "subscribe",
  "instrument": {
    "id": "AAPL-STK",
    "symbol": "AAPL",
    "venue": "NASDAQ",
    "assetClass": "STK",
    "currency": "USD"
  }
}
```

**Server to Client - Order Book Updates**:
```json
{
  "type": "order_book_update",
  "instrument": {
    "id": "AAPL-STK", 
    "symbol": "AAPL",
    "venue": "NASDAQ"
  },
  "data": {
    "bids": [
      {
        "price": 150.00,
        "quantity": 1000,
        "orderCount": 5,
        "level": 0
      },
      {
        "price": 149.99,
        "quantity": 500, 
        "orderCount": 3,
        "level": 1
      }
    ],
    "asks": [
      {
        "price": 150.01,
        "quantity": 800,
        "orderCount": 4,
        "level": 0  
      },
      {
        "price": 150.02,
        "quantity": 600,
        "orderCount": 2,
        "level": 1
      }
    ],
    "spread": {
      "absolute": 0.01,
      "percentage": 0.0067,
      "bestBid": 150.00,
      "bestAsk": 150.01
    },
    "timestamp": 1692419727123,
    "lastUpdateTime": "2025-08-19T00:15:27.123Z"
  }
}
```

### Unsubscribe from Order Book

**Client to Server**:
```json
{
  "type": "order_book_subscription",
  "action": "unsubscribe", 
  "instrument": {
    "id": "AAPL-STK"
  }
}
```

---

## Data Processing Pipeline

### 1. Frontend Search Flow

```
User Input (Symbol) 
    ↓
Frontend Search Modal
    ↓  
API Call: GET /api/v1/ib/instruments/search/{symbol}?sec_type=STK
    ↓
Backend: ib_routes.py:search_instruments()
    ↓
Backend: ib_instrument_provider.py:search_contracts()
    ↓
IB Gateway: Contract resolution
    ↓
Response: Instrument list with contract details
    ↓
Frontend: Display search results with Order Book buttons
```

### 2. Order Book Data Flow

```
User Clicks "Order Book" Button
    ↓
Frontend: Convert search result to Instrument object
    ↓
Frontend: Set selectedInstrument state
    ↓
OrderBookDisplay Component renders
    ↓
useOrderBookData Hook: WebSocket subscription
    ↓
WebSocket: Send order_book_subscription message
    ↓
Backend: Subscribe to IB Gateway market data
    ↓
IB Gateway: Stream real-time order book data
    ↓
Backend: Process and forward via WebSocket
    ↓
Frontend: Receive order_book_update messages
    ↓
orderBookService: Process and aggregate data
    ↓
OrderBookDisplay: Render real-time visualization
```

---

## Component Architecture

### Frontend Components

**OrderBookDisplay** (`src/components/OrderBook/OrderBookDisplay.tsx`)
- Main container component
- Manages instrument subscription
- Handles loading/error states
- Integrates all sub-components

**OrderBookLevel** (`src/components/OrderBook/OrderBookLevel.tsx`)
- Individual price level row
- Quantity bar visualization
- Click handlers for price levels
- Color coding (green bids, red asks)

**OrderBookHeader** (`src/components/OrderBook/OrderBookHeader.tsx`)
- Spread calculation and display
- Best bid/ask highlighting
- Market statistics

**OrderBookControls** (`src/components/OrderBook/OrderBookControls.tsx`)
- Aggregation level settings
- Display preferences
- Performance metrics

### Data Layer

**useOrderBookData Hook** (`src/hooks/useOrderBookData.ts`)
- WebSocket subscription management
- Real-time data state
- Connection status tracking
- Error handling

**orderBookService** (`src/services/orderBookService.ts`)
- Data processing and validation
- Price level aggregation
- Performance optimization
- Market data normalization

**TypeScript Interfaces** (`src/types/orderBook.ts`)
- OrderBookData, OrderBookLevel
- ProcessedOrderBookData
- OrderBookSpread, OrderBookAggregationSettings
- WebSocket message types

---

## Performance Characteristics

### Latency Requirements
- **Order Book Updates**: < 100ms from IB Gateway to UI display
- **Search Response**: < 2 seconds for stock symbols
- **WebSocket Reconnection**: < 5 seconds

### Throughput
- **Update Frequency**: Up to 10 updates/second per symbol (throttled)
- **Concurrent Instruments**: Up to 5 order books simultaneously
- **Price Levels**: Up to 20 levels displayed (aggregated from deeper book)

### Memory Usage
- **Order Book Data**: < 2MB per active instrument
- **Total Memory**: < 10MB for all order book components
- **WebSocket Buffer**: < 1MB for message queuing

---

## Error Handling

### Search Errors

**Timeout Errors (Futures/Indexes)**:
```json
{
  "detail": "Error searching instruments: Contract search timeout for ES"
}
```
- **Cause**: Futures require specific contract months
- **User Experience**: Shows error message, search remains usable
- **Resolution**: Use stock symbols instead

**No Results**:
```json
{
  "instruments": [],
  "total": 0,
  "query": "INVALID",
  "timestamp": "2025-08-19T00:15:27.123456"
}
```
- **User Experience**: "No results found" message
- **Resolution**: User can try different symbol

### WebSocket Errors

**Connection Loss**:
- **Detection**: Automatic WebSocket disconnection events
- **Recovery**: Automatic reconnection with exponential backoff
- **User Experience**: "Reconnecting..." indicator

**Subscription Errors**:
- **Cause**: Invalid instrument or IB Gateway permission issues
- **User Experience**: Error message in order book area
- **Recovery**: User can try different instrument

---

## Testing Endpoints

### Health Check
```bash
curl http://localhost:8000/health
# Expected: {"status": "healthy"}
```

### IB Gateway Status
```bash  
curl http://localhost:8000/api/v1/ib/connection/status
# Expected: Connection details with "connected": true
```

### Stock Search (Reliable)
```bash
curl "http://localhost:8000/api/v1/ib/instruments/search/AAPL?sec_type=STK"
# Expected: AAPL instrument details
```

### Futures Search (Known Issue)
```bash
curl "http://localhost:8000/api/v1/ib/instruments/search/ES?sec_type=FUT"
# Expected: Timeout after 2 minutes (known limitation)
```

---

## Security Considerations

### API Security
- No authentication required for read-only operations
- IB Gateway credentials managed separately
- WebSocket connections use same-origin policy

### Data Privacy
- No sensitive user data transmitted
- Market data subject to IB Gateway permissions
- Real-time data limited to subscribed instruments

### Rate Limiting
- Search endpoint: No rate limiting implemented
- WebSocket updates: Throttled to 10 updates/second per symbol
- IB Gateway: Subject to IB's own rate limiting

---

## Monitoring & Debugging

### Client-Side Debugging
```javascript
// Enable debug logging
localStorage.setItem('debug', 'orderbook:*');

// Monitor WebSocket messages
page.on('console', msg => console.log('BROWSER:', msg.text()));

// Check order book state
console.log(useOrderBookData.getState());
```

### Server-Side Monitoring
```bash
# Check backend logs  
tail -f backend/logs/app.log

# Monitor IB Gateway connection
curl http://localhost:8000/api/v1/ib/connection/status

# Check instrument provider cache
# (Requires backend debugging endpoint)
```

### Performance Monitoring
- Browser DevTools Network tab for API response times
- Performance tab for rendering performance
- Memory tab for memory leak detection
- WebSocket frame inspection for data throughput

---

**Documentation Status**: ✅ Complete  
**API Version**: v1  
**Last Tested**: August 19, 2025  
**Known Issues**: See [known-issues.md](../qa/known-issues.md)