import { test, expect } from '@playwright/test';

test('Debug chart timeframe changes', async ({ page }) => {
  // Enable console logging
  page.on('console', msg => console.log('BROWSER:', msg.text()));
  page.on('pageerror', err => console.log('BROWSER ERROR:', err.message));

  // Go to the dashboard
  await page.goto('http://localhost:3000');
  
  // Wait for the page to load
  await page.waitForTimeout(2000);
  
  // Click on Financial Chart tab
  console.log('🔄 Clicking Financial Chart tab...');
  await page.click('text=Financial Chart');
  await page.waitForTimeout(1000);

  // Select an instrument (AAPL)
  console.log('🔄 Selecting AAPL instrument...');
  const instrumentSelector = page.locator('.ant-select').first();
  await instrumentSelector.click();
  await page.click('text=AAPL');
  await page.waitForTimeout(2000);

  // Take screenshot of initial state
  await page.screenshot({ path: 'debug-initial.png' });

  // Try different timeframes and see what happens
  const timeframes = ['5M', '1H', '4H', '1D'];
  
  for (const tf of timeframes) {
    console.log(`🔄 Clicking timeframe: ${tf}`);
    
    // Click the timeframe button
    await page.click(`text=${tf}`);
    
    // Wait a bit
    await page.waitForTimeout(3000);
    
    // Take screenshot
    await page.screenshot({ path: `debug-${tf}.png` });
    
    // Check if chart container has content
    const chartContainer = page.locator('div[style*="width: 800px"]');
    const hasContent = await chartContainer.isVisible();
    console.log(`📊 Chart visible after ${tf}: ${hasContent}`);
    
    // Check for any error messages
    const errorElements = page.locator('.ant-alert-error');
    const errorCount = await errorElements.count();
    if (errorCount > 0) {
      const errorText = await errorElements.first().textContent();
      console.log(`❌ Error found after ${tf}: ${errorText}`);
    }
    
    // Check for loading spinners
    const loadingElements = page.locator('.ant-spin');
    const loadingCount = await loadingElements.count();
    console.log(`⏳ Loading spinners after ${tf}: ${loadingCount}`);
  }

  // Wait at the end to see final state
  await page.waitForTimeout(5000);
  await page.screenshot({ path: 'debug-final.png' });
});