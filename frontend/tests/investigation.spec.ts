import { test, expect } from '@playwright/test';

test('Current Application State Investigation', async ({ page }) => {
  // Capture all console messages
  const consoleMessages: string[] = [];
  page.on('console', msg => {
    const message = `[${msg.type()}] ${msg.text()}`;
    consoleMessages.push(message);
    console.log('BROWSER:', message);
  });

  // Capture network requests
  const apiCalls: string[] = [];
  page.on('request', request => {
    if (request.url().includes('localhost:8000')) {
      apiCalls.push(`${request.method()} ${request.url()}`);
      console.log('API CALL:', `${request.method()} ${request.url()}`);
    }
  });

  // Navigate to the application
  console.log('=== NAVIGATING TO APPLICATION ===');
  await page.goto('http://localhost:3000');
  
  // Take initial screenshot
  await page.screenshot({ path: '/Users/savusilviu/Desktop/SilviuCorneliuSavu/Nautilus/frontend/tests/screenshots/initial-state.png', fullPage: true });
  
  // Wait for the page to load
  await page.waitForTimeout(3000);
  
  // Check for any immediate errors
  const errorElements = await page.locator('.error, [class*="error"], .alert-danger').count();
  console.log(`=== INITIAL ERROR COUNT: ${errorElements} ===`);
  
  // Take screenshot after load
  await page.screenshot({ path: '/Users/savusilviu/Desktop/SilviuCorneliuSavu/Nautilus/frontend/tests/screenshots/after-load.png', fullPage: true });
  
  // Check if charts are present
  const chartElements = await page.locator('canvas, svg, [class*="chart"], [class*="Chart"]').count();
  console.log(`=== CHART ELEMENTS FOUND: ${chartElements} ===`);
  
  // Look for timeframe buttons/selectors
  const timeframeElements = await page.locator('button:has-text("1M"), button:has-text("1W"), button:has-text("1D"), button:has-text("1H"), button:has-text("15m"), button:has-text("5m"), button:has-text("1m"), [class*="timeframe"], [class*="interval"]').count();
  console.log(`=== TIMEFRAME CONTROLS FOUND: ${timeframeElements} ===`);
  
  // Test clicking on different timeframes if they exist
  if (timeframeElements > 0) {
    console.log('=== TESTING TIMEFRAME INTERACTIONS ===');
    
    const timeframes = ['1m', '5m', '15m', '1H', '1D', '1W', '1M'];
    
    for (const tf of timeframes) {
      try {
        const button = page.locator(`button:has-text("${tf}")`).first();
        if (await button.isVisible()) {
          console.log(`Clicking timeframe: ${tf}`);
          await button.click();
          await page.waitForTimeout(2000);
          
          // Take screenshot after clicking
          await page.screenshot({ 
            path: `/Users/savusilviu/Desktop/SilviuCorneliuSavu/Nautilus/frontend/tests/screenshots/timeframe-${tf}.png`, 
            fullPage: true 
          });
          
          // Check for loading states or data
          const loadingElements = await page.locator('[class*="loading"], .spinner, .loader').count();
          console.log(`Loading indicators after ${tf} click: ${loadingElements}`);
        }
      } catch (e) {
        console.log(`Error clicking ${tf}: ${e.message}`);
      }
    }
  }
  
  // Wait for any async operations to complete
  await page.waitForTimeout(5000);
  
  // Final screenshot
  await page.screenshot({ path: '/Users/savusilviu/Desktop/SilviuCorneliuSavu/Nautilus/frontend/tests/screenshots/final-state.png', fullPage: true });
  
  // Output summary
  console.log('=== INVESTIGATION SUMMARY ===');
  console.log(`Total Console Messages: ${consoleMessages.length}`);
  console.log(`Total API Calls: ${apiCalls.length}`);
  console.log(`Chart Elements: ${chartElements}`);
  console.log(`Timeframe Controls: ${timeframeElements}`);
  console.log(`Error Elements: ${errorElements}`);
  
  console.log('\n=== API CALLS MADE ===');
  apiCalls.forEach(call => console.log(call));
  
  console.log('\n=== CONSOLE MESSAGES ===');
  consoleMessages.slice(-20).forEach(msg => console.log(msg)); // Last 20 messages
  
  // Basic assertions
  expect(chartElements).toBeGreaterThanOrEqual(0);
  expect(errorElements).toBeLessThan(10); // Allow some minor errors but not excessive
});

test('Backend Health Check', async ({ page }) => {
  console.log('=== TESTING BACKEND HEALTH ===');
  
  // Test health endpoint directly
  const response = await page.request.get('http://localhost:8000/health');
  console.log(`Health endpoint status: ${response.status()}`);
  
  if (response.ok()) {
    const healthData = await response.json();
    console.log('Health data:', JSON.stringify(healthData, null, 2));
  }
  
  // Test other endpoints
  const endpoints = [
    '/symbols',
    '/candles/AAPL?timeframe=1m&limit=10',
    '/candles/AAPL?timeframe=1H&limit=10',
    '/candles/AAPL?timeframe=1D&limit=10'
  ];
  
  for (const endpoint of endpoints) {
    try {
      const resp = await page.request.get(`http://localhost:8000${endpoint}`);
      console.log(`${endpoint}: ${resp.status()}`);
      
      if (resp.ok()) {
        const data = await resp.json();
        console.log(`${endpoint} data preview:`, JSON.stringify(data).substring(0, 200) + '...');
      } else {
        const errorText = await resp.text();
        console.log(`${endpoint} error:`, errorText.substring(0, 200));
      }
    } catch (e) {
      console.log(`${endpoint} failed:`, e.message);
    }
  }
});