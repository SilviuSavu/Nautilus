import { test, expect } from '@playwright/test';

test('Test all timeframes including longer ones like 1M', async ({ page }) => {
  // Enable console logging to see API calls and responses
  page.on('console', msg => console.log('BROWSER:', msg.text()));
  page.on('pageerror', err => console.log('BROWSER ERROR:', err.message));

  // Go to the dashboard
  await page.goto('http://localhost:3000');
  await page.waitForTimeout(2000);
  
  // Click on Financial Chart tab
  console.log('🔄 Clicking Financial Chart tab...');
  await page.click('text=Financial Chart');
  await page.waitForTimeout(3000);

  // Wait for initial chart load (should default to AAPL 1h)
  await page.waitForTimeout(3000);

  // Test all timeframes, especially longer ones
  const timeframes = [
    // Minutes
    '1m', '2m', '5m', '10m', '15m', '30m',
    // Hours  
    '1H', '2H', '4H',
    // Daily and longer (these were problematic)
    '1D', '1W', '1M'
  ];
  
  for (const tf of timeframes) {
    console.log(`\n🔄 Testing timeframe: ${tf}`);
    
    try {
      // Click timeframe button
      await page.click(`text=${tf}`);
      console.log(`✅ Clicked ${tf} button`);
      
      // Wait for API call and chart update
      await page.waitForTimeout(4000);
      
      // Check for successful data load (no error alerts)
      const errorCount = await page.locator('.ant-alert-error').count();
      if (errorCount > 0) {
        const errorText = await page.locator('.ant-alert-error').first().textContent();
        console.log(`❌ Error for ${tf}: ${errorText}`);
      } else {
        console.log(`✅ ${tf} loaded successfully - no errors`);
      }
      
      // Check if market data is displayed (price should be visible)
      const priceElements = await page.locator('[title*="AAPL"]').count();
      if (priceElements > 0) {
        console.log(`✅ ${tf} market data displayed`);
      } else {
        console.log(`⚠️ ${tf} market data not visible`);
      }
      
      // Check for canvas elements (chart should be rendered)
      const canvasCount = await page.locator('canvas').count();
      console.log(`📊 ${tf} canvas elements: ${canvasCount}`);
      
      // Take screenshot for visual verification
      await page.screenshot({ path: `test-timeframe-${tf.replace('/', '_')}.png` });
      
    } catch (error) {
      console.log(`❌ Error testing ${tf}: ${error.message}`);
      await page.screenshot({ path: `test-error-${tf.replace('/', '_')}.png` });
    }
  }

  // Final verification - test the problematic 1M timeframe specifically
  console.log('\n🎯 Final verification of 1M timeframe...');
  await page.click('text=1M');
  await page.waitForTimeout(6000); // Give more time for monthly data
  
  const finalErrorCount = await page.locator('.ant-alert-error').count();
  const finalCanvasCount = await page.locator('canvas').count();
  
  console.log(`🎯 1M Final check - Errors: ${finalErrorCount}, Canvas: ${finalCanvasCount}`);
  
  // Should have no errors and should have canvas elements
  expect(finalErrorCount).toBe(0);
  expect(finalCanvasCount).toBeGreaterThan(0);
  
  await page.screenshot({ path: 'test-1M-final.png' });
});