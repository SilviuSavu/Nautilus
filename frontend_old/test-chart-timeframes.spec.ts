import { test, expect } from '@playwright/test';

test.describe('Chart Timeframe Testing', () => {
  test.beforeEach(async ({ page }) => {
    // Listen to console logs
    page.on('console', msg => console.log('BROWSER:', msg.text()));
    
    // Navigate to the application
    await page.goto('http://localhost:3000');
    await page.waitForLoadState('networkidle');
  });

  test('test all timeframes with AAPL', async ({ page }) => {
    console.log('🧪 Starting comprehensive timeframe test...');
    
    // Wait for and click Financial tab
    await page.waitForSelector('[data-testid="financial-tab"], text="Financial"', { timeout: 10000 });
    const financialTab = page.locator('[data-testid="financial-tab"], text="Financial"').first();
    await financialTab.click();
    await page.waitForTimeout(2000);
    
    // Select AAPL instrument
    const instrumentDropdown = page.locator('.ant-select-selector').first();
    await instrumentDropdown.click();
    await page.waitForSelector('.ant-select-dropdown', { timeout: 5000 });
    
    const aaplOption = page.locator('.ant-select-item').filter({ hasText: 'AAPL' }).first();
    await aaplOption.click();
    await page.waitForTimeout(3000);
    
    console.log('📍 AAPL selected, testing timeframes...');
    
    // Test all timeframes
    const timeframes = ['1m', '5m', '15m', '1h', '1d', '1w', '1M'];
    const results: Array<{timeframe: string, success: boolean, candleCount: number, error?: string}> = [];
    
    for (const timeframe of timeframes) {
      console.log(`\n🔍 Testing timeframe: ${timeframe}`);
      
      try {
        // Click timeframe button
        const timeframeButton = page.locator('button', { hasText: new RegExp(`^${timeframe.replace('M', 'M')}$`, 'i') });
        
        if (await timeframeButton.count() === 0) {
          console.log(`❌ Timeframe button not found for ${timeframe}`);
          results.push({timeframe, success: false, candleCount: 0, error: 'Button not found'});
          continue;
        }
        
        await timeframeButton.click();
        console.log(`✅ Clicked ${timeframe} button`);
        
        // Wait for API request and response
        await page.waitForTimeout(5000);
        
        // Check for error messages
        const errorElement = await page.locator('.ant-alert-error, [class*="error"]').count();
        if (errorElement > 0) {
          const errorText = await page.locator('.ant-alert-error, [class*="error"]').first().textContent();
          console.log(`❌ Error found for ${timeframe}: ${errorText}`);
          results.push({timeframe, success: false, candleCount: 0, error: errorText || 'Unknown error'});
          continue;
        }
        
        // Check if "No Market Data Available" message is shown
        const noDataMessage = await page.locator('text="No Market Data Available"').count();
        if (noDataMessage > 0) {
          console.log(`❌ No data available for ${timeframe}`);
          results.push({timeframe, success: false, candleCount: 0, error: 'No market data available'});
          continue;
        }
        
        // Check if chart has rendered with data
        const chartContainer = page.locator('[data-testid="chart-container"], .tv-lightweight-charts').first();
        const hasChart = await chartContainer.count() > 0;
        
        if (!hasChart) {
          console.log(`❌ Chart not rendered for ${timeframe}`);
          results.push({timeframe, success: false, candleCount: 0, error: 'Chart not rendered'});
          continue;
        }
        
        // Look for bar count in the UI
        const barCountText = await page.locator('text=*bars', { hasText: /\d+\s*bars/ }).textContent();
        const candleCount = barCountText ? parseInt(barCountText.match(/(\d+)/)?.[1] || '0') : 0;
        
        if (candleCount > 0) {
          console.log(`✅ ${timeframe}: ${candleCount} candles loaded successfully`);
          results.push({timeframe, success: true, candleCount});
        } else {
          console.log(`❌ ${timeframe}: No candles loaded`);
          results.push({timeframe, success: false, candleCount: 0, error: 'No candles loaded'});
        }
        
      } catch (error) {
        console.log(`❌ Exception testing ${timeframe}: ${error}`);
        results.push({timeframe, success: false, candleCount: 0, error: String(error)});
      }
    }
    
    // Generate summary report
    console.log('\n📊 TIMEFRAME TEST RESULTS:');
    console.log('============================');
    
    const successful = results.filter(r => r.success);
    const failed = results.filter(r => !r.success);
    
    console.log(`✅ Successful: ${successful.length}/${results.length}`);
    successful.forEach(r => console.log(`  ${r.timeframe}: ${r.candleCount} candles`));
    
    console.log(`❌ Failed: ${failed.length}/${results.length}`);
    failed.forEach(r => console.log(`  ${r.timeframe}: ${r.error}`));
    
    // Take screenshot of final state
    await page.screenshot({ path: 'timeframe-test-results.png', fullPage: true });
    
    // At least 1m and 5m should work for basic validation
    const criticalTimeframes = results.filter(r => ['1m', '5m'].includes(r.timeframe));
    const criticalSuccess = criticalTimeframes.filter(r => r.success).length;
    
    expect(criticalSuccess).toBeGreaterThan(0); // At least one critical timeframe should work
  });
  
  test('analyze backend API mapping issues', async ({ page }) => {
    console.log('🔍 Analyzing backend API mapping...');
    
    // Test direct API calls for different timeframes
    const timeframes = ['1m', '5m', '15m', '1h', '1d', '1w', '1M'];
    
    for (const timeframe of timeframes) {
      console.log(`\n🌐 Testing API for timeframe: ${timeframe}`);
      
      const response = await page.request.get(`http://localhost:8000/api/v1/market-data/historical/bars?symbol=AAPL&timeframe=${timeframe}&asset_class=STK&exchange=SMART&currency=USD`);
      
      if (response.ok()) {
        const data = await response.json();
        console.log(`✅ ${timeframe} API: ${data.candles?.length || 0} candles, total: ${data.total || 0}`);
        
        if (data.candles?.length === 0) {
          console.log(`❌ ${timeframe} API returned empty candles array`);
          console.log(`   Start: ${data.start_date}, End: ${data.end_date}`);
          console.log(`   Source: ${data.source}`);
        }
      } else {
        const errorText = await response.text();
        console.log(`❌ ${timeframe} API failed: ${response.status()} - ${errorText}`);
      }
    }
  });
});