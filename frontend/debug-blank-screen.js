/**
 * Debug blank screen issue by capturing all console errors
 */

import { chromium } from 'playwright';

(async () => {
  console.log('🔍 Debugging blank screen issue...');
  
  const browser = await chromium.launch({ 
    headless: false,
    slowMo: 500
  });
  
  const page = await browser.newPage();
  
  // Capture all console messages
  page.on('console', msg => {
    console.log(`CONSOLE ${msg.type().toUpperCase()}:`, msg.text());
  });
  
  // Capture page errors
  page.on('pageerror', error => {
    console.log('PAGE ERROR:', error.message);
  });
  
  // Capture request failures
  page.on('requestfailed', request => {
    console.log('REQUEST FAILED:', request.url(), request.failure()?.errorText);
  });
  
  try {
    console.log('📍 Loading page...');
    await page.goto('http://localhost:3000', { waitUntil: 'networkidle', timeout: 30000 });
    
    console.log('📍 Waiting for content to load...');
    await page.waitForTimeout(5000);
    
    // Check if page has any content
    const bodyText = await page.textContent('body');
    console.log('📄 Body text length:', bodyText?.length || 0);
    
    if ((bodyText?.length || 0) < 100) {
      console.log('❌ Page appears to be blank or nearly empty');
      console.log('📄 Body content:', bodyText);
    } else {
      console.log('✅ Page has content');
    }
    
    // Check for common React error indicators
    const reactErrors = await page.locator('.react-error-overlay, [data-testid="error-boundary"]').count();
    if (reactErrors > 0) {
      console.log('❌ React error overlay detected');
    }
    
    // Check for missing resources
    const missingResources = await page.locator('link[rel="stylesheet"]:not([href*="data:"]), script:not([src*="data:"])').count();
    console.log('📦 External resources loaded:', missingResources);
    
    // Take screenshot
    await page.screenshot({ path: 'debug-blank-screen.png', fullPage: true });
    console.log('📸 Screenshot saved: debug-blank-screen.png');
    
  } catch (error) {
    console.error('❌ Debug failed:', error.message);
  }
  
  await page.waitForTimeout(3000);
  await browser.close();
})();