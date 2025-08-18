import { test, expect } from '@playwright/test';

test('Test chart timeframe changes work', async ({ page }) => {
  // Enable console logging
  page.on('console', msg => console.log('BROWSER:', msg.text()));
  page.on('pageerror', err => console.log('BROWSER ERROR:', err.message));

  // Go to the dashboard
  await page.goto('http://localhost:3000');
  await page.waitForTimeout(2000);
  
  // Click on Financial Chart tab
  console.log('🔄 Clicking Financial Chart tab...');
  await page.click('text=Financial Chart');
  await page.waitForTimeout(3000);

  // Wait for chart to be ready (this should show AAPL data by default)
  await page.waitForTimeout(5000);

  // Try clicking different timeframes
  const timeframes = ['5M', '1H', '4H', '1D'];
  
  for (const tf of timeframes) {
    console.log(`🔄 Testing timeframe: ${tf}`);
    
    // Click timeframe button
    await page.click(`text=${tf}`);
    console.log(`✅ Clicked ${tf} button`);
    
    // Wait for chart update
    await page.waitForTimeout(4000);
    
    // Take screenshot
    await page.screenshot({ path: `test-tf-${tf}.png` });
    
    // Check for canvas elements (chart should be visible)
    const canvasCount = await page.locator('canvas').count();
    console.log(`📊 Canvas elements after ${tf}: ${canvasCount}`);
    
    // Check for error messages
    const errorCount = await page.locator('.ant-alert-error').count();
    if (errorCount > 0) {
      const errorText = await page.locator('.ant-alert-error').first().textContent();
      console.log(`❌ Error after ${tf}: ${errorText}`);
    } else {
      console.log(`✅ No errors after ${tf}`);
    }
  }

  await page.screenshot({ path: 'test-final.png' });
});