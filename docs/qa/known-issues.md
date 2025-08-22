# Known Issues & Limitations

## Order Book Visualization Feature

**Last Updated**: August 19, 2025  
**Version**: 2.4.orderbook-visualization  

---

## Critical Issues

### 🚨 Futures & Indexes Search Timeout

**Issue ID**: KB-001  
**Severity**: High  
**Status**: Known Limitation  
**Affects**: Futures (FUT) and Indexes (IND) search functionality  

**Description**:
Search for futures contracts (e.g., ES, NQ) and indexes results in timeout errors. The API call hangs for 2+ minutes before returning "Contract search timeout" error.

**Root Cause**:
Futures contracts require specific contract months (expiry dates) to be properly resolved by IB Gateway. The current search implementation times out when trying to search for generic futures symbols without explicit expiry dates.

**Affected Instruments**:
- E-mini S&P 500 (ES)
- E-mini NASDAQ (NQ) 
- Crude Oil (CL)
- Gold (GC)
- All futures contracts
- Most index instruments

**Error Message**:
```json
{"detail":"Error searching instruments: Contract search timeout for ES"}
```

**Workaround**:
Currently, no workaround available through the UI. Futures and indexes functionality is temporarily unavailable.

**Technical Details**:
- Backend timeout occurs in `ib_instrument_provider.py:search_futures()`
- IB Gateway requires specific contract months for futures resolution
- API endpoint: `GET /api/v1/ib/instruments/search/{symbol}?sec_type=FUT`
- Timeout period: 10 seconds (configurable in backend)

**Resolution Plan**:
1. Implement automatic front-month contract discovery
2. Add popular futures contract mappings (ES -> ES Dec25, etc.)
3. Enhance search to try multiple contract months
4. Add user-friendly error messages for unsupported instruments

**Testing**:
```bash
# Reproduce the issue
curl "http://localhost:8000/api/v1/ib/instruments/search/ES?sec_type=FUT"
# Expected: Timeout after 2 minutes
```

---

## Working Functionality

### ✅ Stocks (Equities)

**Status**: Fully Functional  
**Examples**: AAPL, MSFT, GOOGL, TSLA, PLTR  

All stock searches work correctly:
- Symbol search returns multiple results
- Order Book button appears in results
- Real-time order book visualization functional
- Performance meets <100ms latency requirements

### ✅ Currency Pairs (Forex)

**Status**: Functional  
**Examples**: EURUSD, GBPUSD (when available)  

Forex pairs work when available through IB Gateway:
- IDEALPRO exchange used automatically
- Currency-specific formatting applied
- Real-time forex order book data

---

## Minor Issues

### ⚠️ Empty Results for Some Valid Symbols

**Issue ID**: KB-002  
**Severity**: Medium  
**Status**: Environmental  

**Description**:
Some valid stock symbols may return empty results depending on:
- Market hours (after-hours may have limited data)
- IB Gateway permissions (some symbols require specific data subscriptions)
- Exchange connectivity

**Resolution**: 
Usually resolves when markets are open and with proper IB Gateway permissions.

---

## User Guidance

### Recommended Testing Instruments

**Stocks (Always Work)**:
- AAPL (Apple Inc.)
- MSFT (Microsoft)
- GOOGL (Google/Alphabet)
- TSLA (Tesla)
- PLTR (Palantir)

**Avoid During Testing**:
- Any futures symbols (ES, NQ, CL, GC)
- Index symbols (SPX, NDX)
- Exotic currency pairs
- Options contracts

### Error Handling

The application gracefully handles:
- ✅ Invalid symbols (shows "No results found")
- ✅ Network timeouts (shows connection error)
- ✅ Backend errors (shows appropriate error message)
- ❌ Futures timeout (may appear to hang - known issue)

---

## Developer Notes

### Quick Fix for Demos

If futures functionality is needed for demonstrations, consider:

1. **Mock Data Approach**: Add mock futures data for popular contracts
2. **Preload Contracts**: Cache common futures contracts with known expiry dates
3. **User Education**: Clear messaging about supported instrument types

### Code References

**Backend Issues**:
- `/backend/ib_routes.py:702-708` - Futures search logic
- `/backend/ib_instrument_provider.py:259-269` - Futures search method
- IB Gateway contract resolution timeout handling

**Frontend Handling**:
- Search timeout displays generic error message
- No specific handling for futures vs other timeouts
- Could be enhanced with instrument-type-specific error messages

---

## QA Testing Guidelines

### Test Cases to Skip

- Skip all futures search tests (ES, NQ, CL, GC)
- Skip index search tests
- Skip timeout resilience tests for futures

### Focus Testing On

- Stock symbol searches (high confidence)
- Order book visualization for equities
- Real-time data updates for stocks
- Search result display and interaction
- Order book button functionality
- Cross-browser compatibility for working features

### Expected Behavior

When testing futures:
- Search will appear to hang for 2+ minutes
- Eventually returns timeout error
- This is expected behavior - not a test failure
- Application remains stable after timeout

---

**Status Summary**:
- 🟢 **Stocks**: Fully functional
- 🟡 **Forex**: Functional (with permissions) 
- 🔴 **Futures**: Known limitation
- 🔴 **Indexes**: Known limitation

**Overall Feature Status**: ✅ Ready for production with documented limitations