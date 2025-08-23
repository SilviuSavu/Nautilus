/**
 * Final test to check if chart works with default instrument
 */

import { chromium } from 'playwright';

(async () => {
  console.log('🚀 Testing chart with default SPY instrument...');
  
  const browser = await chromium.launch({ 
    headless: false,
    slowMo: 500
  });
  
  const page = await browser.newPage();
  
  // Listen for specific console messages
  page.on('console', msg => {
    const text = msg.text();
    if (text.includes('📊 ChartComponent state') || 
        text.includes('📡 Making API request') ||
        text.includes('📊 API Response received') ||
        text.includes('Canvas') ||
        text.includes('currentInstrument')) {
      console.log('BROWSER:', text);
    }
  });
  
  try {
    await page.goto('http://localhost:3000', { waitUntil: 'networkidle' });
    await page.waitForSelector('[data-testid="dashboard"]', { timeout: 15000 });
    
    console.log('📍 Clicking Chart tab...');
    await page.click('text=Chart');
    await page.waitForTimeout(3000);
    
    console.log('📍 Clicking 5m timeframe...');
    await page.click('button:has-text("5m")');
    await page.waitForTimeout(5000);
    
    // Check for canvas (chart rendered)
    const canvasCount = await page.locator('canvas').count();
    console.log(`📊 Canvas elements: ${canvasCount}`);
    
    // Check for error states
    const noDataElements = await page.locator('text=No Market Data Available').count();
    const errorElements = await page.locator('.ant-alert-error').count();
    
    console.log(`📭 "No Market Data" messages: ${noDataElements}`);
    console.log(`❌ Error alerts: ${errorElements}`);
    
    // Look for chart data
    const chartContainer = await page.locator('.chart-container').count();
    console.log(`📈 Chart containers: ${chartContainer}`);
    
    await page.screenshot({ path: 'chart-test-final.png', fullPage: true });
    
    if (canvasCount > 0) {
      console.log('✅ SUCCESS: Chart canvas found - chart is rendering!');
    } else if (noDataElements > 0) {
      console.log('📭 INFO: Chart shows "No Market Data Available"');
    } else {
      console.log('❓ UNKNOWN: Chart state unclear');
    }
    
  } catch (error) {
    console.error('❌ Test failed:', error.message);
  }
  
  await page.waitForTimeout(3000);
  await browser.close();
})();