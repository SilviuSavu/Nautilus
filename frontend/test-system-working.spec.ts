import { test, expect } from '@playwright/test';

test.describe('System End-to-End Verification', () => {
  test('Backend and Frontend are working with real data', async ({ page }) => {
    // Enable console logging to verify API calls
    const consoleMessages: string[] = [];
    page.on('console', msg => {
      const message = `BROWSER: ${msg.text()}`;
      console.log(message);
      consoleMessages.push(message);
    });

    // Navigate to frontend
    await page.goto('http://localhost:3000');
    
    // Wait for page to load
    await page.waitForTimeout(2000);
    
    // Check that we can reach the frontend without errors
    const title = await page.title();
    console.log(`Page title: ${title}`);
    
    // Test API connectivity by making a direct request to backend
    const healthResponse = await page.evaluate(async () => {
      try {
        const response = await fetch('http://localhost:8000/health');
        return await response.json();
      } catch (error) {
        return { error: error.message };
      }
    });
    
    console.log('Health check response:', JSON.stringify(healthResponse));
    expect(healthResponse.status).toBe('healthy');
    
    // Test IB Gateway status
    const ibStatusResponse = await page.evaluate(async () => {
      try {
        const response = await fetch('http://localhost:8000/api/v1/ib/status');
        return await response.json();
      } catch (error) {
        return { error: error.message };
      }
    });
    
    console.log('IB Status response:', JSON.stringify(ibStatusResponse));
    expect(ibStatusResponse.connected).toBe(true);
    expect(ibStatusResponse.client_id).toBe(2);
    
    // Test real market data retrieval
    const marketDataResponse = await page.evaluate(async () => {
      try {
        const response = await fetch('http://localhost:8000/api/v1/market-data/historical/bars?symbol=AAPL&timeframe=5m');
        return await response.json();
      } catch (error) {
        return { error: error.message };
      }
    });
    
    console.log('Market data response summary:', {
      symbol: marketDataResponse.symbol,
      timeframe: marketDataResponse.timeframe,
      candleCount: marketDataResponse.candles?.length || 0,
      source: marketDataResponse.source,
      hasData: !!(marketDataResponse.candles && marketDataResponse.candles.length > 0)
    });
    
    expect(marketDataResponse.symbol).toBe('AAPL');
    expect(marketDataResponse.timeframe).toBe('5m');
    expect(marketDataResponse.source).toBe('IB Gateway');
    expect(marketDataResponse.candles).toBeDefined();
    expect(Array.isArray(marketDataResponse.candles)).toBe(true);
    expect(marketDataResponse.candles.length).toBeGreaterThan(0);
    
    // Verify candle data structure
    const firstCandle = marketDataResponse.candles[0];
    expect(firstCandle).toHaveProperty('time');
    expect(firstCandle).toHaveProperty('open');
    expect(firstCandle).toHaveProperty('high');
    expect(firstCandle).toHaveProperty('low');
    expect(firstCandle).toHaveProperty('close');
    expect(firstCandle).toHaveProperty('volume');
    
    // Test daily timeframe that was previously failing
    const dailyDataResponse = await page.evaluate(async () => {
      try {
        const response = await fetch('http://localhost:8000/api/v1/market-data/historical/bars?symbol=AAPL&timeframe=1d');
        return await response.json();
      } catch (error) {
        return { error: error.message };
      }
    });
    
    console.log('Daily data response summary:', {
      symbol: dailyDataResponse.symbol,
      timeframe: dailyDataResponse.timeframe,
      candleCount: dailyDataResponse.candles?.length || 0,
      source: dailyDataResponse.source,
      hasData: !!(dailyDataResponse.candles && dailyDataResponse.candles.length > 0)
    });
    
    expect(dailyDataResponse.symbol).toBe('AAPL');
    expect(dailyDataResponse.timeframe).toBe('1d');
    expect(dailyDataResponse.source).toBe('IB Gateway');
    expect(dailyDataResponse.candles).toBeDefined();
    expect(Array.isArray(dailyDataResponse.candles)).toBe(true);
    expect(dailyDataResponse.candles.length).toBeGreaterThan(100); // Should have lots of daily data
    
    // Take screenshot for verification
    await page.screenshot({ path: 'test-system-working.png', fullPage: true });
    
    console.log('✅ SYSTEM IS FULLY FUNCTIONAL:');
    console.log('- Backend connected to IB Gateway with client ID 2');
    console.log('- Real market data flowing from IB Gateway');
    console.log('- 5m timeframe working with', marketDataResponse.candles.length, 'candles');
    console.log('- 1d timeframe working with', dailyDataResponse.candles.length, 'candles');
    console.log('- Frontend accessible and can call backend APIs');
    console.log('- No mock data - all real IB Gateway data');
  });
});