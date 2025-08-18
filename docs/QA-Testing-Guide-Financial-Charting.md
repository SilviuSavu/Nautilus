# QA Testing Guide - Financial Charting Implementation

## 🎯 Testing Scope: Story 2.2 Financial Charting Component

**Version**: Development Build  
**Test Environment**: Local development setup  
**Focus**: Chart rendering, data integration, user experience  

## 🔧 Test Environment Setup

### Prerequisites
- Chrome/Firefox/Safari browser with developer tools
- Local development environment running
- Interactive Brokers Gateway connection (already configured)

### Environment Verification
Before testing, verify the environment is properly set up:

#### 1. Backend Service Check
```bash
# Test backend health
curl http://localhost:8000/health

# Expected Response:
{"status": "ok", "timestamp": "2025-08-17T11:XX:XX.XXXZ"}
```

#### 2. Frontend Service Check
```bash
# Test frontend accessibility
curl -I http://localhost:3000

# Expected Response:
HTTP/1.1 200 OK
```

#### 3. API Data Verification
```bash
# Test historical data API
curl "http://localhost:8000/api/v1/market-data/historical/bars?symbol=AAPL&timeframe=4h&asset_class=STK&exchange=NASDAQ&currency=USD"

# Expected: JSON response with 124+ AAPL candles
# Verify: symbol="AAPL", candles array with OHLCV data
```

## 📋 Critical Test Cases

### Test Case 1: Chart Tab Navigation
**Objective**: Verify chart tab is accessible and loads properly

**Steps**:
1. Open browser to http://localhost:3000
2. Navigate to "Financial Chart" tab
3. Wait for page to load completely

**Expected Results**:
- Chart tab loads without errors
- Instrument selector displays with dropdown options
- Timeframe selector shows available intervals
- Chart container area is visible

**Critical Issues to Check**:
- [ ] Tab navigation works
- [ ] No JavaScript console errors
- [ ] UI components render properly

### Test Case 2: Browser Console Investigation
**Objective**: Identify chart rendering issues through developer tools

**Steps**:
1. Open browser developer tools (F12)
2. Navigate to Console tab
3. Clear console log
4. Navigate to Financial Chart tab
5. Document all console messages

**Expected Results**:
- Debug messages showing data flow (if working)
- No critical JavaScript errors
- TradingView library loads successfully

**Debug Messages to Look For**:
```
✅ Working Console Output:
🔍 loadChartData called with instrument: {symbol: "AAPL", ...}
📡 Making API request: /api/v1/market-data/historical/bars?...
📊 API Response received: {symbol: "AAPL", candleCount: 124, ...}
🚀 Initializing chart after timeout
📊 Converted candle data: {originalCount: 124, convertedCount: 124}

❌ Error Indicators:
- JavaScript errors in console
- Failed network requests
- TradingView library errors
- Canvas rendering failures
```

### Test Case 3: Network Request Analysis
**Objective**: Verify API calls are successful and data is retrieved

**Steps**:
1. Open browser developer tools
2. Navigate to Network tab
3. Clear network log
4. Navigate to Financial Chart tab
5. Monitor network requests

**Expected Results**:
- API call to `/api/v1/market-data/historical/bars` returns 200 OK
- Response contains valid JSON with candles array
- No failed network requests

**API Response Validation**:
```json
{
  "symbol": "AAPL",
  "timeframe": "4h", 
  "candles": [
    {
      "time": "20250519  15:30:00",
      "open": 207.78,
      "high": 208.21,
      "low": 204.26,
      "close": 207.64,
      "volume": 39042
    },
    // ... more candles
  ],
  "total": 124
}
```

### Test Case 4: Chart Container Inspection
**Objective**: Verify chart container DOM structure and styling

**Steps**:
1. Open browser developer tools
2. Navigate to Elements/Inspector tab
3. Navigate to Financial Chart tab
4. Inspect chart container element
5. Check for canvas elements

**Expected Results**:
- Chart container div has proper dimensions
- TradingView canvas elements are present
- No CSS styling conflicts
- Container has visible border (temporary debug styling)

**DOM Structure to Verify**:
```html
<div style="width: 100%; height: 400px; border: 1px solid #e1e1e1;">
  <!-- TradingView chart elements should be here -->
  <div class="tv-lightweight-charts">
    <canvas>...</canvas>
  </div>
</div>
```

### Test Case 5: Instrument Selection Testing
**Objective**: Test different instrument selections

**Steps**:
1. Navigate to Financial Chart tab
2. Click instrument selector dropdown
3. Select different instruments from each category
4. Monitor chart updates

**Test Instruments**:
- **Stocks**: AAPL, MSFT, GOOGL
- **Forex**: EURUSD, GBPUSD (may have connection issues)
- **Futures**: ES, NQ, CL
- **Indices**: SPX, VIX
- **ETFs**: SPY, QQQ

**Expected Results**:
- Dropdown shows all instrument categories
- Selecting instrument triggers new API call
- Chart updates (when rendering is fixed)

### Test Case 6: Timeframe Selection Testing
**Objective**: Test different timeframe options

**Steps**:
1. Navigate to Financial Chart tab
2. Select AAPL instrument
3. Test each timeframe option
4. Monitor API calls and data changes

**Timeframes to Test**:
- 1m (1 minute)
- 5m (5 minutes)
- 15m (15 minutes)
- 1h (1 hour) - default
- 4h (4 hours)
- 1d (1 day)

**Expected Results**:
- Each timeframe triggers new API call
- Different data ranges returned
- Chart updates accordingly (when rendering works)

## 🐛 Known Issues Testing

### Issue: Chart Black Screen
**Current Status**: Critical issue preventing chart display

**Testing Focus**:
1. **Browser Compatibility**: Test Chrome, Firefox, Safari
2. **Canvas Rendering**: Check if canvas elements are created
3. **Data Validation**: Verify data format matches TradingView requirements
4. **Timing Issues**: Check if chart initializes before data loads

**Debugging Steps**:
1. Check console for TradingView library errors
2. Verify canvas element creation in DOM
3. Validate timestamp conversion (IB format to Unix timestamp)
4. Test with minimal data set

**Common Causes to Investigate**:
- Invalid timestamp format in chart data
- Container sizing issues (width/height = 0)
- TradingView library configuration errors
- Data format mismatches
- CSS styling conflicts

## 📊 Performance Testing

### Chart Load Time Testing
**Steps**:
1. Navigate to Financial Chart tab
2. Measure time from navigation to chart display
3. Test with different instruments and timeframes

**Performance Targets**:
- Initial load: < 3 seconds
- Instrument switch: < 2 seconds
- Timeframe change: < 2 seconds

### Memory Usage Testing
**Steps**:
1. Open browser developer tools
2. Navigate to Performance/Memory tab
3. Navigate to chart multiple times
4. Switch between instruments repeatedly
5. Monitor memory usage for leaks

## 🌐 Browser Compatibility Testing

### Primary Browsers
Test the following browsers with identical test cases:

1. **Chrome** (latest version)
   - Primary development target
   - Best TradingView support expected

2. **Firefox** (latest version)
   - Canvas rendering differences possible
   - JavaScript engine variations

3. **Safari** (latest version)
   - WebKit rendering engine
   - Potential canvas/WebGL differences

### Browser-Specific Issues to Check
- Canvas rendering differences
- JavaScript engine compatibility
- Memory management variations
- Network request handling

## 📝 Test Reporting

### For Each Test Case
Document the following information:

#### Test Results Template
```
Test Case: [Name]
Browser: [Chrome/Firefox/Safari]
Status: [PASS/FAIL/BLOCKED]
Date/Time: [Timestamp]

Steps Executed:
1. [Step 1]
2. [Step 2]
...

Expected Result:
[Description]

Actual Result:
[What actually happened]

Console Errors:
[Copy any JavaScript errors]

Network Issues:
[Copy any failed requests]

Screenshots:
[Attach relevant screenshots]

Additional Notes:
[Any other observations]
```

### Critical Issues Reporting
For the chart black screen issue specifically:

#### Required Information
1. **Browser and version**
2. **Complete console log** (copy all messages)
3. **Network tab** (all requests and responses)
4. **DOM inspection** (chart container structure)
5. **Canvas elements** (present/absent)
6. **JavaScript errors** (full stack traces)

#### Debug Data Collection
```javascript
// Run in browser console for additional debugging
console.log('Chart container:', document.querySelector('[ref=chartContainerRef]'));
console.log('TradingView elements:', document.querySelectorAll('[class*="tv-"]'));
console.log('Canvas elements:', document.querySelectorAll('canvas'));
```

## 🔄 Regression Testing

### After Chart Fix
Once the chart rendering issue is resolved, retest:

1. **All instrument types** work correctly
2. **All timeframes** display proper data
3. **Performance targets** are met
4. **Browser compatibility** maintained
5. **Error handling** for invalid instruments
6. **Loading states** display appropriately

### Full Integration Testing
1. **Dashboard navigation** to chart tab
2. **Authentication** persists during chart usage
3. **WebSocket connections** (for future real-time updates)
4. **Multi-tab behavior** (chart state persistence)

## ✅ Test Sign-off Criteria

### Must Pass Before Approval
- [ ] Chart displays candlestick data for AAPL
- [ ] Instrument selection works for all asset classes
- [ ] Timeframe selection updates chart correctly
- [ ] No critical JavaScript console errors
- [ ] Performance meets targets (< 3 second load)
- [ ] Compatible with Chrome, Firefox, Safari
- [ ] Loading states and error handling functional

### Nice to Have
- [ ] Real-time data updates (future enhancement)
- [ ] Advanced chart features (indicators, etc.)
- [ ] Mobile responsiveness
- [ ] Export functionality

---

**Testing Focus**: The primary goal is to identify and resolve the chart black screen issue while validating that all supporting infrastructure (API, data flow, UI components) functions correctly. The backend and data integration are working properly - the issue is specifically with chart rendering in the browser.