# 🔧 Interactive Brokers Connector Restoration Guide (Sprint 1 Fixes)

## Overview
This document captures the key fixes implemented during Sprint 1 to restore Interactive Brokers Gateway connectivity and resolve critical integration issues. Use this guide to restore IB functionality when it gets broken again.

## 🚨 Critical Fixes from Sprint 1

### 1. "EtradeOnly" Order Attribute Error Fix

**Problem**: IB Gateway was rejecting orders with error `"EtradeOnly" order attribute is not supported`

**Location**: `/backend/ib_routes.py` lines 648-651

**Fix Applied**:
```python
# Map common IB Gateway errors to user-friendly messages
if "EtradeOnly" in error_msg:
    raise HTTPException(
        status_code=400, detail="Order attributes not supported by your IB account type. Please contact your broker."
    )
```

**Root Cause**: IB Gateway was receiving order attributes that are only supported for specific account types. The fix provides user-friendly error mapping.

### 2. Enhanced IBOrderRequest Model

**Location**: `/backend/ib_routes.py` lines 513-535

**Fix Applied**: Comprehensive order request model with advanced order parameters:
```python
class IBOrderRequest(BaseModel):
    symbol: str
    action: str  # BUY/SELL
    quantity: float
    order_type: str  # MKT/LMT/STP/STP_LMT/TRAIL/BRACKET/OCA
    asset_class: str = "STK"
    exchange: str = "SMART"
    currency: str = "USD"
    limit_price: float | None = None
    stop_price: float | None = None
    time_in_force: str = "DAY"
    account_id: str | None = None
    # Advanced order fields for enhanced frontend compatibility
    trail_amount: float | None = None
    trail_percent: float | None = None
    take_profit_price: float | None = None
    stop_loss_price: float | None = None
    outside_rth: bool | None = False
    hidden: bool | None = False
    discretionary_amount: float | None = None
    parent_order_id: str | None = None
    oca_group: str | None = None
```

### 3. Order Validation System

**Location**: `/backend/ib_routes.py` lines 537-582

**Fix Applied**: Comprehensive pre-submission validation:
```python
def validate_order_request(request: IBOrderRequest) -> dict[str, str]:
    """Validate order request and return any validation errors"""
    errors = {}
    
    # Basic validation
    if not request.symbol or not request.symbol.strip():
        errors["symbol"] = "Symbol is required"
    
    if request.quantity <= 0:
        errors["quantity"] = "Quantity must be greater than 0"
    
    if request.action not in ["BUY", "SELL"]:
        errors["action"] = "Action must be BUY or SELL"
    
    # Order type specific validation
    if request.order_type == "LMT" and not request.limit_price:
        errors["limit_price"] = "Limit price is required for limit orders"
    
    if request.order_type == "STP" and not request.stop_price:
        errors["stop_price"] = "Stop price is required for stop orders"
    
    if request.order_type == "STP_LMT":
        if not request.limit_price:
            errors["limit_price"] = "Limit price is required for stop-limit orders"
        if not request.stop_price:
            errors["stop_price"] = "Stop price is required for stop-limit orders"
```

### 4. Comprehensive Error Handling

**Location**: `/backend/ib_routes.py` lines 642-678

**Fix Applied**: Detailed error mapping for common IB Gateway issues:
```python
# Map common IB Gateway errors to user-friendly messages
if "EtradeOnly" in error_msg:
    raise HTTPException(
        status_code=400, detail="Order attributes not supported by your IB account type. Please contact your broker."
    )
elif "Invalid order type" in error_msg:
    raise HTTPException(
        status_code=400, detail=f"Order type '{request.order_type}' is not valid for this instrument"
    )
elif "Not connected" in error_msg:
    raise HTTPException(
        status_code=503, detail="IB Gateway connection lost. Please reconnect."
    )
elif "client id is already in use" in error_msg.lower():
    raise HTTPException(
        status_code=503, detail="IB Gateway client ID conflict. Please try again."
    )
elif "upgrade to a minimum version" in error_msg:
    raise HTTPException(
        status_code=503, detail="IB Gateway version is outdated. Please upgrade to version 163 or higher."
    )
```

### 5. Nautilus Trading Node Compatibility Fix

**Location**: `/backend/ib_routes.py` lines 17-22

**Fix Applied**: Temporary mock to prevent import errors:
```python
# from nautilus_trading_node import get_nautilus_node_manager  # Temporarily disabled due to IB API compatibility

def get_nautilus_node_manager():
    """Temporary mock for compatibility"""
    return None
```

**Note**: This is a temporary fix to prevent blocking issues. May need real implementation for full functionality.

### 6. **CRITICAL**: IBAPI Import Compatibility Fix (Official NautilusTrader Pattern)

**Problem**: Import errors for `FundAssetType` and other new IBAPI fields causing startup failures

**Location**: Based on official NautilusTrader adapter pattern in `common.py` lines 20-29

**Official Fix Pattern**:
```python
try:
    from ibapi.contract import FundAssetType
except ImportError:
    # FundAssetType not available in this version of ibapi
    FundAssetType = None
try:
    from ibapi.contract import FundDistributionPolicyIndicator
except ImportError:
    # FundDistributionPolicyIndicator not available in this version of ibapi
    FundDistributionPolicyIndicator = None
```

**Why This Matters**: The official NautilusTrader adapter uses defensive imports to handle IBAPI version compatibility issues. This prevents the `ImportError: cannot import name 'FundAssetType' from 'ibapi.contract'` error that breaks the entire backend startup.

**Implementation**: Apply this pattern to any file that imports from `ibapi.contract` to ensure compatibility across different IBAPI versions.

## 🏗️ Official NautilusTrader Architecture Patterns

### Comparison with Official Adapter Structure

**Official NautilusTrader IB Adapter Structure** (from github.com/nautechsystems/nautilus_trader):
```
nautilus_trader/adapters/interactive_brokers/
├── client/          # Client connection management
├── historical/      # Historical data processing
├── parsing/         # Data parsing utilities
├── gateway.py       # Docker-based gateway management
├── execution.py     # Order execution handling
├── data.py          # Market data client
├── config.py        # Configuration management
├── providers.py     # Instrument and data providers
└── common.py        # Shared utilities and constants
```

**Key Architectural Differences**:

1. **Modular Design**: Official adapter separates concerns across multiple modules vs. our monolithic `ib_routes.py`

2. **Docker Gateway Management**: Official uses `DockerizedIBGateway` class with:
   - Container lifecycle management
   - Automatic status tracking  
   - Configurable trading modes (paper/live)
   - Robust error handling with custom exceptions

3. **Configuration Management**: Official uses Pydantic-based configs with:
   - Environment variable integration
   - Sensitive data masking
   - Flexible instrument provider settings
   - Security-first approach

4. **Error Handling**: Official implements:
   - Custom exception hierarchy (`ContainerExists`, `GatewayLoginFailure`)
   - Context manager support for cleanup
   - Comprehensive logging integration

### Recent Updates (2024-2025)

**Recent Official Improvements**:
- **Order Book Deltas Support** (July 2025): Enhanced market data capabilities
- **Refined Bar Subscriptions** (August 2025): Better historical data handling
- **IBAPI Compatibility Fixes**: Defensive imports for version compatibility

### Recommended Architecture Migration

To align with official patterns, consider restructuring:

```python
# Instead of monolithic ib_routes.py, consider:
backend/
├── ib_adapter/
│   ├── gateway_manager.py      # Docker gateway management
│   ├── execution_client.py     # Order execution
│   ├── data_client.py          # Market data handling
│   ├── config_manager.py       # Configuration
│   ├── error_handlers.py       # Error management
│   └── common.py               # Shared utilities
└── routes/
    └── ib_routes.py            # Simplified API endpoints
```

## 🔄 IB Connector Restoration Procedure

### Step 1: Verify Core Files
Ensure these critical files are in place with the fixes:
- `/backend/ib_routes.py` - Main API routes with error handling
- `/backend/ib_gateway_client.py` - IB Gateway connection management
- `/backend/ib_order_manager.py` - Order management logic
- `/backend/ib_asset_classes.py` - Asset class support

### Step 2: Check Environment Configuration
```bash
# Backend configuration
DATABASE_URL=postgresql://nautilus:nautilus123@localhost:5432/nautilus
REDIS_URL=redis://localhost:6379
CORS_ORIGINS=http://localhost:3000,http://localhost:3001

# IB Gateway settings
IB_HOST=host.docker.internal  # or localhost
IB_PORT=4002  # Paper trading
IB_CLIENT_ID=1001  # Unique client ID
```

### Step 3: Verify IB Gateway Connection
```python
# Test IB Gateway connection
from ib_gateway_client import get_ib_gateway_client

client = get_ib_gateway_client()
if client.is_connected():
    print("✅ IB Gateway connected successfully")
else:
    print("❌ IB Gateway connection failed")
```

### Step 4: Test Order Placement API
```bash
# Test order placement endpoint
curl -X POST http://localhost:8001/api/v1/ib/orders/place \
  -H "Content-Type: application/json" \
  -d '{
    "symbol": "AAPL",
    "action": "BUY",
    "quantity": 100,
    "order_type": "MKT",
    "asset_class": "STK",
    "exchange": "SMART",
    "currency": "USD",
    "time_in_force": "DAY"
  }'
```

### Step 5: Verify Frontend Integration
Check that frontend components can successfully:
1. Connect to IB Gateway via `/api/v1/ib/connection/status`
2. Place orders via `/api/v1/ib/orders/place`
3. Retrieve market data via `/api/v1/ib/market-data/subscribe`

## 🛠️ Common Issues & Solutions

### Issue: Import Error for nautilus_trading_node
**Symptom**: `ImportError: cannot import name 'get_nautilus_node_manager'`
**Solution**: Use the mock function provided in the fix (lines 19-22 in ib_routes.py)

### Issue: IBAPI Import Compatibility Errors
**Symptom**: `ImportError: cannot import name 'FundAssetType' from 'ibapi.contract'`
**Root Cause**: Different IBAPI versions have different available imports
**Solution**: Apply defensive import pattern from official NautilusTrader:
```python
try:
    from ibapi.contract import FundAssetType
except ImportError:
    FundAssetType = None
try:
    from ibapi.contract import FundDistributionPolicyIndicator  
except ImportError:
    FundDistributionPolicyIndicator = None
```
**Where to Apply**: Any file importing from `ibapi.contract` or other IBAPI modules

### Issue: EtradeOnly Order Attribute Error
**Symptom**: Orders rejected with "EtradeOnly order attribute is not supported"
**Solution**: The error mapping in lines 648-651 provides user-friendly feedback. Check account type with IB.

### Issue: Invalid Order Type Errors
**Symptom**: "Invalid order type was entered" for advanced orders
**Solution**: Ensure order type validation in lines 552-563 is working and order types match IB Gateway expectations.

### Issue: IB Gateway Version Warnings
**Symptom**: "Please upgrade to a minimum version 163"
**Solution**: The error handling in lines 664-667 catches this. Upgrade IB Gateway to version 163+.

### Issue: Client ID Conflicts
**Symptom**: "client id is already in use"
**Solution**: Use unique client IDs or restart IB Gateway. Error handled in lines 660-663.

## 📋 Health Check Commands

### Backend Health Check
```bash
curl http://localhost:8001/api/v1/ib/connection/status
```

### Order System Test
```bash
# Test with AAPL market order
curl -X POST http://localhost:8001/api/v1/ib/orders/place \
  -H "Content-Type: application/json" \
  -d '{
    "symbol": "AAPL",
    "action": "BUY", 
    "quantity": 1,
    "order_type": "MKT",
    "time_in_force": "DAY"
  }'
```

### Market Data Test
```bash
curl -X POST http://localhost:8001/api/v1/ib/market-data/subscribe \
  -H "Content-Type: application/json" \
  -d '{"symbols": ["AAPL"]}'
```

## 🔧 File Restore Checklist

If IB functionality is completely broken, restore these key components:

### Critical Backend Files
- [ ] `/backend/ib_routes.py` - API endpoints with Sprint 1 fixes
- [ ] `/backend/ib_gateway_client.py` - Connection management  
- [ ] `/backend/ib_order_manager.py` - Order processing
- [ ] `/backend/ib_asset_classes.py` - Asset class definitions

### Frontend Integration Files
- [ ] `/frontend/src/components/IBOrderPlacement.tsx` - Order entry form
- [ ] `/frontend/src/components/IBDashboard.tsx` - Main IB dashboard
- [ ] `/frontend/src/services/ibService.ts` - API integration

### Configuration Files
- [ ] `docker-compose.yml` - Container configuration
- [ ] `/backend/main.py` - Route registration
- [ ] Environment variables properly set

## 🎯 Success Criteria

IB connector is fully restored when:
1. ✅ Backend starts without import errors
2. ✅ IB Gateway connection status returns successfully
3. ✅ Order placement API accepts and processes orders
4. ✅ Frontend can connect and display IB data
5. ✅ Error handling provides user-friendly messages
6. ✅ All advanced order types work without gateway errors

## 🔧 Version Compatibility

### IBAPI Version Management
**Current Compatibility**: The fixes handle multiple IBAPI versions using defensive imports

**Supported IBAPI Versions**:
- **Minimum**: IBAPI 9.81.x (basic functionality)
- **Recommended**: IBAPI 10.x+ (full feature support)
- **Latest Tested**: IBAPI 10.29.x (with all recent fixes)

**Version-Specific Issues**:
- `FundAssetType` availability varies by version
- `FundDistributionPolicyIndicator` only in newer versions
- Use defensive imports to ensure compatibility

### IB Gateway Version Requirements
- **Minimum**: Version 163+ (as detected by error handling)
- **Paper Trading**: Both TWS and IB Gateway supported
- **Live Trading**: IB Gateway recommended for production

### NautilusTrader Integration
- **Local Implementation**: Custom Sprint 1 fixes
- **Official Adapter**: Based on NautilusTrader 1.219.0+ patterns
- **Migration Path**: Consider adopting official architecture patterns

## 📚 Reference Documentation

### Internal Documentation
- **Sprint 1 QA Summary**: `/nautilus_trader/frontend/docs/QA_SUMMARY_IB_IMPLEMENTATION.md`
- **Order Placement Story**: `/nautilus_trader/frontend/docs/stories/3.1.order-placement.md`
- **Trade History Integration**: `/nautilus_trader/frontend/docs/stories/3.3.trade-history.md`
- **Technical Implementation**: `/docs/TECHNICAL-IMPLEMENTATION-SUMMARY.md`

### Official NautilusTrader References
- **Official IB Adapter**: https://github.com/nautechsystems/nautilus_trader/tree/develop/nautilus_trader/adapters/interactive_brokers
- **Gateway Management**: https://github.com/nautechsystems/nautilus_trader/blob/develop/nautilus_trader/adapters/interactive_brokers/gateway.py
- **Execution Client**: https://github.com/nautechsystems/nautilus_trader/blob/develop/nautilus_trader/adapters/interactive_brokers/execution.py
- **Configuration**: https://github.com/nautechsystems/nautilus_trader/blob/develop/nautilus_trader/adapters/interactive_brokers/config.py

---

**Created**: Based on Sprint 1 fixes and IB implementation documentation
**Last Updated**: From Sprint 2 context analysis
**Status**: Ready for use when IB connector breaks again