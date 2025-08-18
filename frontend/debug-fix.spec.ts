import { test, expect } from '@playwright/test';

test('Debug and fix chart issue step by step', async ({ page }) => {
  // Enable console logging
  page.on('console', msg => console.log('BROWSER:', msg.text()));
  page.on('pageerror', err => console.log('BROWSER ERROR:', err.message));

  // Go to the dashboard
  await page.goto('http://localhost:3000');
  await page.waitForTimeout(2000);
  
  // Click on Financial Chart tab
  console.log('🔄 Clicking Financial Chart tab...');
  await page.click('text=Financial Chart');
  await page.waitForTimeout(1000);

  // Wait for instrument selector to be visible
  await page.waitForSelector('[data-testid="instrument-selector"], .ant-select', { timeout: 10000 });
  
  // Select AAPL
  console.log('🔄 Selecting AAPL...');
  await page.click('.ant-select >> nth=0');
  await page.waitForTimeout(500);
  await page.click('text=AAPL');
  await page.waitForTimeout(3000);

  // Take screenshot after selecting instrument
  await page.screenshot({ path: 'debug-after-instrument.png' });

  // Check if chart is visible
  const chartElement = page.locator('canvas, svg, [class*="chart"], [class*="Chart"]');
  const chartVisible = await chartElement.isVisible();
  console.log(`📊 Chart visible after instrument selection: ${chartVisible}`);

  // Try clicking different timeframes
  const timeframes = ['1M', '5M', '15M', '1H', '4H', '1D'];
  
  for (let i = 0; i < timeframes.length; i++) {
    const tf = timeframes[i];
    console.log(`\n🔄 Testing timeframe: ${tf}`);
    
    try {
      // Click timeframe button
      await page.click(`text=${tf}`);
      console.log(`✅ Clicked ${tf} button`);
      
      // Wait for any loading/updates
      await page.waitForTimeout(4000);
      
      // Take screenshot
      await page.screenshot({ path: `debug-timeframe-${tf}.png` });
      
      // Check for chart content
      const canvasElements = await page.locator('canvas').count();
      console.log(`📊 Canvas elements after ${tf}: ${canvasElements}`);
      
      // Check for any error messages
      const errorElements = await page.locator('.ant-alert-error').count();
      if (errorElements > 0) {
        const errorText = await page.locator('.ant-alert-error').first().textContent();
        console.log(`❌ Error after ${tf}: ${errorText}`);
      }
      
      // Check for loading indicators
      const loadingElements = await page.locator('.ant-spin').count();
      console.log(`⏳ Loading indicators after ${tf}: ${loadingElements}`);
      
      // Log DOM structure around chart area
      const chartContainers = await page.locator('[style*="width: 800px"], [style*="height: 400px"], [class*="chart"]').count();
      console.log(`📦 Chart containers after ${tf}: ${chartContainers}`);
      
    } catch (error) {
      console.log(`❌ Error clicking ${tf}: ${error.message}`);
      await page.screenshot({ path: `debug-error-${tf}.png` });
    }
  }

  // Final analysis - let's inspect the DOM structure
  console.log('\n🔍 Final DOM Analysis:');
  
  const allDivs = await page.locator('div').count();
  console.log(`Total divs: ${allDivs}`);
  
  const chartDivs = await page.locator('div[style*="800px"]').count();
  console.log(`Chart-sized divs: ${chartDivs}`);
  
  const canvases = await page.locator('canvas').count();
  console.log(`Canvas elements: ${canvases}`);
  
  // Try to get the actual HTML content of chart area
  try {
    const chartHTML = await page.locator('div[style*="800px"]').first().innerHTML();
    console.log(`Chart container HTML: ${chartHTML.substring(0, 200)}...`);
  } catch (e) {
    console.log('Could not get chart HTML');
  }
  
  await page.screenshot({ path: 'debug-final-analysis.png' });
});