/**
 * Custom test to reproduce chart flashing/disappearing issue
 */

import { chromium } from 'playwright';

(async () => {
  console.log('🚀 Starting chart timeframe test...');
  
  const browser = await chromium.launch({ 
    headless: false, // Show browser so we can observe the issue
    slowMo: 1000     // Slow down actions to see what happens
  });
  
  const page = await browser.newPage();
  
  // Listen for console messages
  page.on('console', msg => console.log('BROWSER:', msg.text()));
  
  try {
    console.log('📍 Navigating to localhost:3000...');
    await page.goto('http://localhost:3000', { waitUntil: 'networkidle' });
    
    console.log('📍 Waiting for page to load...');
    await page.waitForTimeout(3000); // Just wait for page to load
    
    console.log('📍 Looking for Chart tab...');
    // Try multiple selectors for the Chart tab
    let chartTabFound = false;
    
    // Try different possible selectors
    const selectors = [
      'text=Chart',
      '[data-testid="main-dashboard-tabs"] >> text=Chart',
      '.ant-tabs-tab >> text=Chart',
      'div:has-text("Chart")'
    ];
    
    for (const selector of selectors) {
      try {
        const element = await page.locator(selector).first();
        if (await element.isVisible({ timeout: 1000 })) {
          console.log(`✅ Found Chart tab with selector: ${selector}`);
          await element.click();
          await page.waitForTimeout(2000);
          chartTabFound = true;
          break;
        }
      } catch (e) {
        // Continue to next selector
      }
    }
    
    if (!chartTabFound) {
      console.log('❌ Chart tab not found with any selector, checking what tabs are available...');
      try {
        const allText = await page.textContent('body');
        console.log('Page loaded, checking for tab text...');
        if (allText.includes('Chart')) {
          console.log('✅ Found "Chart" text on page');
        }
        // Just continue with the test anyway
      } catch (e) {
        console.log('Could not check page content');
      }
    }
    
    console.log('📍 Looking for timeframe selectors...');
    
    // Look for timeframe buttons
    const timeframes = ['1m', '5m', '15m', '1H', '4H', '1D'];
    
    for (const tf of timeframes) {
      console.log(`📍 Testing timeframe: ${tf}`);
      
      // Look for the timeframe button
      const timeframeButton = page.locator(`button:has-text("${tf}")`).first();
      
      if (await timeframeButton.isVisible()) {
        console.log(`✅ Found ${tf} button, clicking...`);
        
        // Take screenshot before click
        await page.screenshot({ path: `chart-before-${tf}.png`, fullPage: true });
        
        // Click the timeframe
        await timeframeButton.click();
        
        console.log(`⏱️ Waiting 3 seconds to observe chart behavior after ${tf} click...`);
        await page.waitForTimeout(3000);
        
        // Take screenshot after click
        await page.screenshot({ path: `chart-after-${tf}.png`, fullPage: true });
        
        // Check if chart canvas exists
        const canvas = await page.locator('canvas').count();
        console.log(`📊 Canvas elements found: ${canvas}`);
        
        // Check for any error messages
        const errorElements = await page.locator('.error-message, .ant-alert-error').count();
        if (errorElements > 0) {
          console.log(`❌ Error elements found: ${errorElements}`);
        }
        
        // Look for "No Market Data Available" message
        const noDataMessage = await page.locator('text=No Market Data Available').count();
        if (noDataMessage > 0) {
          console.log(`📭 "No Market Data Available" message found`);
        }
        
        break; // Test just one timeframe to observe the issue
      } else {
        console.log(`❌ ${tf} timeframe button not found`);
      }
    }
    
    console.log('📍 Final screenshot...');
    await page.screenshot({ path: 'chart-final-state.png', fullPage: true });
    
  } catch (error) {
    console.error('❌ Test failed:', error.message);
    await page.screenshot({ path: 'chart-error-state.png', fullPage: true });
  }
  
  console.log('📍 Test completed. Check screenshots for chart behavior.');
  console.log('🔍 Screenshots saved:');
  console.log('  - chart-before-[timeframe].png');
  console.log('  - chart-after-[timeframe].png');
  console.log('  - chart-final-state.png');
  
  await page.waitForTimeout(5000); // Keep browser open for 5 seconds to observe
  await browser.close();
})();