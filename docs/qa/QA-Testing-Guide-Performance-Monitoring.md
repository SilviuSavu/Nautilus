# QA Testing Guide - Performance Monitoring System

**Story**: 4.2 Real-Time Strategy Performance Monitoring  
**Status**: ✅ Ready for QA Testing  
**Date**: 2025-08-20  

## 🎯 Quick Start Testing

### Prerequisites
1. **Backend Running**: Port 8000 (http://localhost:8000)
2. **Frontend Running**: Port 3000 (http://localhost:3000)  
3. **Database**: PostgreSQL with 4 real trades loaded

### 1-Minute Smoke Test
1. Open http://localhost:3000
2. ✅ Dashboard should load completely (no blank page)
3. ✅ Click "Performance Monitoring" tab
4. ✅ Performance Dashboard opens with real data
5. ✅ See "Total Trades: 4" displayed
6. ✅ No error notifications or red error messages

## 📋 Comprehensive Test Plan

### Test Case 1: Frontend Loading
**Expected**: Dashboard loads with Performance tab
- [ ] Navigate to http://localhost:3000
- [ ] Dashboard loads within 3 seconds
- [ ] "Performance Monitoring" tab visible in navigation
- [ ] No JavaScript errors in browser console
- [ ] No blank page or loading forever

### Test Case 2: Performance Dashboard Access
**Expected**: Performance features accessible
- [ ] Click "Performance Monitoring" tab
- [ ] Performance Dashboard loads with overview tab
- [ ] Multiple sub-tabs visible: Overview, Real-Time Monitor, Strategy Comparison, Execution Analytics, Alert System
- [ ] Real data displayed (not "No data" messages)

### Test Case 3: Real Data Display
**Expected**: Actual database data shown
- [ ] Performance metrics show: Total Trades = 4
- [ ] P&L values calculated from real trade data
- [ ] Win rate calculated from actual trades
- [ ] All metrics display numbers (not 0 or null)

### Test Case 4: API Integration
**Expected**: Backend APIs working
- [ ] Test `/api/v1/performance/aggregate` returns 200 OK
- [ ] Test `/api/v1/execution/metrics` returns real data
- [ ] No 404 or 500 errors in network tab
- [ ] Real-time updates working (data refreshes every 5 seconds)

### Test Case 5: Component Functionality
**Expected**: All tabs and features work
- [ ] Click "Overview" tab - displays performance summary
- [ ] Click "Real-Time Monitor" tab - shows live monitoring
- [ ] Click "Strategy Comparison" tab - comparison tools load
- [ ] Click "Execution Analytics" tab - execution data displayed
- [ ] Click "Alert System" tab - alert configuration accessible

### Test Case 6: Error Handling
**Expected**: Graceful error handling
- [ ] If backend stops, frontend shows error notifications
- [ ] Error boundaries prevent full page crashes
- [ ] Loading states shown during API calls
- [ ] User-friendly error messages (not technical stack traces)

## 🔧 Backend API Testing

### API Endpoints to Test

**Performance APIs:**
```bash
curl http://localhost:8000/api/v1/performance/aggregate
curl http://localhost:8000/api/v1/performance/history
curl http://localhost:8000/api/v1/performance/compare
curl http://localhost:8000/api/v1/performance/alerts
```

**Execution APIs:**
```bash
curl http://localhost:8000/api/v1/execution/metrics
curl http://localhost:8000/api/v1/execution/analytics
```

**Expected Results:**
- All endpoints return 200 OK status
- JSON responses with actual data
- No 404 "Not Found" errors
- No 500 "Internal Server Error" responses

## 🗄️ Database Verification

### Check Real Data
```sql
-- Should return 4 trades
SELECT COUNT(*) FROM trades;

-- Should show real trade data
SELECT symbol, side, quantity, price, execution_time FROM trades LIMIT 5;
```

**Expected**:
- 4 trades in database
- Real symbols, prices, and timestamps
- No mock or fake data

## ⚠️ Known Issues/Limitations

1. **Historical Data**: Performance history returns empty arrays (TODO for future implementation)
2. **Alert Configuration**: Basic structure in place, full configuration UI pending
3. **Ant Design Warning**: `Tabs.TabPane` deprecation warning (cosmetic only)

## 🚫 Test Failures - Contact Dev Team If:

1. ❌ Frontend shows blank page
2. ❌ "Performance Data Error" notifications constantly appearing
3. ❌ Any API returning 404 or 500 errors
4. ❌ Performance Dashboard tab not clickable
5. ❌ Total Trades showing 0 instead of 4
6. ❌ JavaScript errors preventing navigation

## 📞 Success Criteria

**✅ QA PASS Requirements:**
- [ ] Frontend loads completely without errors
- [ ] Performance Monitoring tab accessible
- [ ] Real data (4 trades) displayed correctly
- [ ] All API endpoints return 200 OK
- [ ] Performance Dashboard shows calculated metrics
- [ ] Real-time updates working
- [ ] No critical errors or crashes

## 🎭 User Scenarios to Test

### Scenario 1: Trader Checking Performance
1. Trader opens dashboard
2. Clicks Performance Monitoring
3. Reviews P&L and trade statistics
4. Checks different time periods
5. Navigates between different tabs

### Scenario 2: Real-Time Monitoring
1. Open Performance Dashboard
2. Leave browser open for 30 seconds
3. Verify data refreshes automatically
4. Check that real-time updates don't cause errors

### Scenario 3: Multi-Tab Navigation
1. Navigate between all Performance sub-tabs
2. Return to main Dashboard
3. Go back to Performance Monitoring
4. Verify state preserved and no data loss

## 📊 Test Data

**Current Database State:**
- **Trades Table**: 4 real trades
- **Symbols**: Real stock symbols from actual trades
- **P&L**: Calculated from actual price * quantity - commission
- **Timestamps**: Real execution times

## 🔄 Regression Testing

If making changes, verify:
- [ ] Main Dashboard still loads
- [ ] Other tabs (IB Dashboard, etc.) unaffected
- [ ] Performance tab integration doesn't break existing features
- [ ] Backend startup sequence works correctly

## 📝 Bug Report Template

If issues found:

```
**Bug Title**: [Descriptive title]
**Priority**: High/Medium/Low
**Steps to Reproduce**:
1. 
2. 
3. 

**Expected Result**: 
**Actual Result**: 
**Browser**: Chrome/Safari/Firefox
**Console Errors**: [Any JavaScript errors]
**API Errors**: [Any network errors]
**Screenshot**: [Attach if visual issue]
```

---

**🎯 Ready for QA - All Systems Operational**

*Last Updated: 2025-08-20 by Claude AI Development Team*