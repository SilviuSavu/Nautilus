# Interactive Brokers Integration for NautilusTrader Dashboard

This document describes the Interactive Brokers (IB) integration implementation for the NautilusTrader web dashboard.

## Overview

The IB integration provides real-time connection to Interactive Brokers through the existing NautilusTrader IB adapter, enabling:

1. **Real-time account monitoring** (margin, buying power, cash balances)
2. **Position tracking** with live P&L updates
3. **Order management** (place, modify, cancel orders)
4. **Connection status monitoring** (TWS/Gateway health)
5. **Market data streaming** from IB feeds

## Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   React Frontend │    │   FastAPI Backend │    │ NautilusTrader  │
│                 │    │                  │    │                 │
│ - IBDashboard   │◄──►│ - IB API endpoints│◄──►│ - IB Adapter    │
│ - Order Form    │    │ - WebSocket       │    │ - MessageBus    │
│ - Real-time UI  │    │ - IB Service      │    │ - TWS/Gateway   │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

## Implementation Components

### Backend Components

#### 1. IB Integration Service (`ib_integration_service.py`)
- **Purpose**: Core service that handles IB-specific MessageBus communication
- **Features**:
  - Subscribes to IB MessageBus topics
  - Manages connection status, account data, positions, and orders
  - Provides async API for data access and order operations
  - Broadcasts updates via WebSocket

#### 2. FastAPI API Endpoints (`main.py`)
New IB-specific endpoints added:

```
GET    /api/v1/ib/connection/status     # Get IB connection status
GET    /api/v1/ib/account               # Get account data
GET    /api/v1/ib/positions             # Get current positions  
GET    /api/v1/ib/orders                # Get orders
POST   /api/v1/ib/account/refresh       # Request account data refresh
POST   /api/v1/ib/positions/refresh     # Request positions refresh
POST   /api/v1/ib/orders/refresh        # Request orders refresh
POST   /api/v1/ib/orders/place          # Place new order
POST   /api/v1/ib/orders/{id}/cancel    # Cancel order
PUT    /api/v1/ib/orders/{id}/modify    # Modify order
```

#### 3. WebSocket Message Broadcasting
Real-time WebSocket messages for IB events:
- `ib_connection`: Connection status updates
- `ib_account`: Account data updates
- `ib_positions`: Position updates
- `ib_order`: Order status updates

### Frontend Components

#### 1. IBDashboard Component (`IBDashboard.tsx`)
- **Purpose**: Main IB dashboard displaying account overview
- **Features**:
  - Connection status with gateway type and health
  - Account summary (net liquidation, buying power, margins)
  - Live positions table with P&L
  - Active orders table with status tracking
  - Real-time WebSocket data updates
  - Manual refresh buttons

#### 2. IBOrderPlacement Component (`IBOrderPlacement.tsx`)
- **Purpose**: Order placement form with validation
- **Features**:
  - Symbol input with validation
  - Order types: Market, Limit, Stop, Stop-Limit
  - Quantity and price inputs
  - Time-in-force options (DAY, GTC, IOC, FOK)
  - Pre-submission validation
  - Order summary preview
  - Risk warnings

#### 3. Dashboard Integration
- Added tabbed interface with "Interactive Brokers" tab
- Floating action button for quick order placement
- Real-time message handling for IB data

## Setup Instructions

### Prerequisites

1. **NautilusTrader with IB Adapter**: Ensure NautilusTrader is configured with the Interactive Brokers adapter
2. **TWS or IB Gateway**: Have TWS (Trader Workstation) or IB Gateway running
3. **API Permissions**: Enable API trading in TWS/Gateway settings

### Configuration

1. **IB Connection Settings** (in NautilusTrader config):
```python
{
    "ibg_host": "127.0.0.1",          # IB Gateway host
    "ibg_port": 7497,                 # Paper trading port (4001 for live)
    "ibg_client_id": 1,               # Client ID
    "account_id": "DU12345",          # IB account ID
    "trading_mode": "paper"           # paper/live
}
```

2. **Environment Variables** (optional):
```bash
TWS_USERNAME=your_username
TWS_PASSWORD=your_password
TWS_ACCOUNT=DU12345
```

### MessageBus Topic Mapping

The integration subscribes to these NautilusTrader MessageBus topics:

```
adapter.interactive_brokers.connection  → Connection status
adapter.interactive_brokers.account     → Account updates
adapter.interactive_brokers.position    → Position updates
adapter.interactive_brokers.order       → Order updates
data.quotes.IB                         → Quote data
data.trades.IB                         → Trade data
data.bars.IB                           → Bar data
```

## Usage Guide

### 1. Monitoring Connection Status
- Dashboard shows real-time connection status
- Displays gateway type (TWS/IBG), host, port, client ID
- Shows last heartbeat and error messages
- Connection indicators update automatically

### 2. Account Monitoring
- Real-time account values (net liquidation, cash, buying power)
- Margin requirements (initial, maintenance)
- Excess liquidity calculations
- Currency support (USD default)
- Manual refresh available

### 3. Position Tracking
- Live position display with quantities
- Average cost and current market price
- Real-time market value calculation
- Unrealized and realized P&L
- Automatic updates via WebSocket

### 4. Order Management
- View all orders with real-time status
- Place new orders with validation
- Modify existing orders (quantity, price)
- Cancel orders
- Order history and execution details

### 5. Placing Orders
1. Click floating action button or go to IB tab
2. Select symbol (e.g., AAPL, MSFT, SPY)
3. Choose action (BUY/SELL)
4. Enter quantity
5. Select order type:
   - **Market (MKT)**: Execute at current market price
   - **Limit (LMT)**: Execute at specified price or better
   - **Stop (STP)**: Stop-loss at specified price
   - **Stop-Limit (STP_LMT)**: Stop with limit price
6. Set time-in-force (DAY/GTC/IOC/FOK)
7. Review order summary
8. Submit order

## Testing

### Manual Testing
1. Start NautilusTrader with IB adapter configured
2. Start the backend: `python backend/main.py`
3. Start the frontend: `npm run dev` in `frontend/`
4. Navigate to IB tab in dashboard
5. Verify connection status, account data, positions

### Automated Testing
Run the integration test suite:
```bash
cd backend
python test_ib_integration.py
```

Test coverage includes:
- API endpoint health checks
- IB service functionality
- WebSocket communication
- Order validation
- Error handling

## Data Models

### IBConnectionStatus
```typescript
interface IBConnectionStatus {
  connected: boolean;
  gateway_type: string;        // "TWS" or "IBG"
  host: string;               // Gateway host
  port: number;               // Gateway port
  client_id: number;          // Client ID
  account_id?: string;        // IB account ID
  connection_time?: string;   // ISO timestamp
  last_heartbeat?: string;    // ISO timestamp
  error_message?: string;     // Error details
}
```

### IBAccountData
```typescript
interface IBAccountData {
  account_id: string;
  net_liquidation?: number;    // Total account value
  total_cash_value?: number;   // Available cash
  buying_power?: number;       // Buying power
  maintenance_margin?: number; // Maintenance margin
  initial_margin?: number;     // Initial margin
  excess_liquidity?: number;   // Excess liquidity
  currency: string;           // Account currency
  timestamp?: string;         // Last update
}
```

### IBPosition
```typescript
interface IBPosition {
  account_id: string;
  contract_id: string;       // IB contract ID
  symbol: string;           // Symbol name
  position: number;         // Position quantity
  avg_cost?: number;        // Average cost
  market_price?: number;    // Current market price
  market_value?: number;    // Current market value
  unrealized_pnl?: number;  // Unrealized P&L
  realized_pnl?: number;    // Realized P&L
  timestamp?: string;       // Last update
}
```

### IBOrder
```typescript
interface IBOrder {
  order_id: string;
  client_id: number;
  account_id: string;
  contract_id: string;
  symbol: string;
  action: string;             // "BUY" or "SELL"
  order_type: string;         // "MKT", "LMT", "STP", etc.
  total_quantity: number;
  filled_quantity: number;
  remaining_quantity: number;
  limit_price?: number;       // For limit orders
  stop_price?: number;        // For stop orders
  status: string;            // Order status
  avg_fill_price?: number;   // Average fill price
  commission?: number;       // Commission paid
  timestamp?: string;        // Last update
}
```

## Error Handling

### Common Issues

1. **IB Service Not Initialized**
   - Error: `503 Service Unavailable - IB service not initialized`
   - Solution: Ensure NautilusTrader is running with IB adapter

2. **Connection Timeout**
   - Error: IB connection shows disconnected
   - Solution: Check TWS/Gateway is running, verify host/port

3. **Authentication Required**
   - Error: `401 Unauthorized`
   - Solution: Implement authentication or disable auth for development

4. **Invalid Order Parameters**
   - Error: `400 Bad Request` with validation details
   - Solution: Check order parameters (symbol, quantity, prices)

### Debug Mode
Enable debug logging in backend:
```python
logging.basicConfig(level=logging.DEBUG)
```

View WebSocket messages in browser console:
```javascript
// In browser developer tools
localStorage.setItem('debug', 'websocket:*');
```

## Security Considerations

1. **API Authentication**: Production deployments should require authentication
2. **Order Validation**: All orders validated before submission to IB
3. **Risk Warnings**: UI displays trading risk warnings
4. **Error Logging**: Sensitive data excluded from logs
5. **HTTPS**: Use HTTPS in production for WebSocket connections

## Performance

### Optimization Features
- **WebSocket Streaming**: Real-time data updates without polling
- **Efficient Rendering**: React components optimized for frequent updates
- **Data Caching**: IB service caches data to reduce API calls
- **Rate Limiting**: Built-in protection against API rate limits

### Performance Metrics
- Order placement latency: < 100ms (network dependent)
- UI update frequency: Real-time WebSocket updates
- Data refresh: On-demand or automatic via MessageBus

## Troubleshooting

### Common Commands

**Check IB adapter status in NautilusTrader:**
```python
# In NautilusTrader Python console
print(trader.data_engine.is_connected)
print(trader.exec_engine.is_connected)
```

**Test WebSocket connection:**
```bash
# Install wscat: npm install -g wscat
wscat -c ws://localhost:8000/ws
```

**Check backend logs:**
```bash
# If using Docker
docker logs nautilus-backend

# If running directly
tail -f backend.log
```

### Support

For issues specific to:
- **NautilusTrader IB Adapter**: See NautilusTrader documentation
- **Interactive Brokers API**: See IB API documentation
- **Dashboard Integration**: Check this implementation's logs and tests

## Future Enhancements

### Planned Features
1. **Market Data Visualization**: Real-time charts integration
2. **Advanced Order Types**: Bracket orders, trailing stops
3. **Risk Management**: Position size limits, portfolio risk metrics
4. **Historical Analysis**: Trade history and performance analytics
5. **Multi-Account Support**: Handle multiple IB accounts
6. **Mobile Optimization**: Responsive design improvements

### Extension Points
The IB integration is designed to be extensible:
- Add new MessageBus topic handlers in `IBIntegrationService`
- Create new API endpoints in `main.py`
- Build additional React components for specific features
- Extend WebSocket message types for custom data

## API Reference

### REST Endpoints

All IB endpoints require authentication (if enabled) and return JSON responses.

#### Connection Status
```
GET /api/v1/ib/connection/status
Response: IBConnectionStatus
```

#### Account Data
```
GET /api/v1/ib/account
Response: IBAccountData

POST /api/v1/ib/account/refresh
Response: {"message": "refresh requested"}
```

#### Positions
```
GET /api/v1/ib/positions
Response: {"positions": IBPosition[]}

POST /api/v1/ib/positions/refresh
Response: {"message": "refresh requested"}
```

#### Orders
```
GET /api/v1/ib/orders
Response: {"orders": IBOrder[]}

POST /api/v1/ib/orders/place
Body: IBOrderRequest
Response: {"order_id": string, "message": string}

POST /api/v1/ib/orders/{order_id}/cancel
Response: {"message": "cancellation requested"}

PUT /api/v1/ib/orders/{order_id}/modify
Body: IBOrderModification
Response: {"message": "modification requested"}

POST /api/v1/ib/orders/refresh
Response: {"message": "refresh requested"}
```

### WebSocket Messages

#### Connection Updates
```json
{
  "type": "ib_connection",
  "data": IBConnectionStatus,
  "timestamp": number
}
```

#### Account Updates
```json
{
  "type": "ib_account", 
  "data": IBAccountData,
  "timestamp": number
}
```

#### Position Updates
```json
{
  "type": "ib_positions",
  "data": {[position_key]: IBPosition},
  "timestamp": number
}
```

#### Order Updates
```json
{
  "type": "ib_order",
  "data": IBOrder,
  "timestamp": number
}
```

---

**Note**: This integration is designed for paper trading and testing. For live trading, ensure proper risk management, testing, and compliance with your broker's requirements.