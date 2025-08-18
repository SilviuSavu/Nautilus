# Development Policies

## CRITICAL RULES - NO EXCEPTIONS

### 🚫 NO MOCK DATA POLICY
- **NEVER implement mock data fallbacks**
- **NEVER use fake/dummy/test/placeholder data**
- **NEVER mask backend issues with synthetic data**
- **ALWAYS surface real backend problems with proper error messages**
- **ALWAYS fix root causes, not symptoms**

### 🎭 PLAYWRIGHT TESTING MANDATORY
- **ALWAYS use Playwright for testing functionality**
- **NEVER assume code changes work without browser verification**
- **ALWAYS test actual user interactions and API calls**
- **ALWAYS capture console logs and error states**
- **ALWAYS take screenshots for visual verification**

### 🔧 REAL PROBLEM SOLVING
- **ALWAYS investigate and fix backend/API issues**
- **ALWAYS show meaningful error messages to users**
- **ALWAYS log detailed debugging information**
- **NEVER hide system failures behind fake success states**

## Enforcement
These policies are absolute. Any violation must be immediately corrected.

## Examples of WRONG vs RIGHT

### ❌ WRONG: Mock Data Fallback
```javascript
if (!data.candles.length) {
  const mockData = generateMockData(timeframe, 100)
  return mockData
}
```

### ✅ RIGHT: Proper Error Handling
```javascript
if (!data.candles.length) {
  console.error('No data from backend for', symbol, timeframe)
  throw new Error(`No historical data available. Backend issue.`)
}
```

### ❌ WRONG: Assuming Code Works
```javascript
// Made changes to chart component
console.log('Chart should work now')
```

### ✅ RIGHT: Playwright Verification
```javascript
// Run Playwright test to verify chart functionality
await page.click('text=1M')
await page.waitForTimeout(3000)
const errorCount = await page.locator('.ant-alert-error').count()
expect(errorCount).toBe(0)
```