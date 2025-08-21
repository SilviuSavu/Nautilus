import { test, expect } from '@playwright/test';

test('Capture console errors from frontend', async ({ page }) => {
  // Collect console messages
  const consoleMessages: string[] = [];
  const errors: string[] = [];
  
  page.on('console', msg => {
    const text = `${msg.type()}: ${msg.text()}`;
    consoleMessages.push(text);
    console.log(`🔍 Console: ${text}`);
  });

  page.on('pageerror', error => {
    const errorText = `PAGE ERROR: ${error.message}`;
    errors.push(errorText);
    console.log(`❌ ${errorText}`);
  });

  // Navigate to the main page
  console.log('🚀 Navigating to http://localhost:3001');
  await page.goto('http://localhost:3001');
  
  // Wait a bit for scripts to load and execute
  await page.waitForTimeout(5000);
  
  // Check what's in the DOM
  const bodyText = await page.textContent('body');
  console.log(`📄 Body text content: "${bodyText}"`);
  
  // Check if the red screen is visible
  const redScreenVisible = await page.isVisible('div:has-text("MAIN.TSX SCRIPT EXECUTED")');
  console.log(`🔴 Red screen visible: ${redScreenVisible}`);
  
  // Get all script tags
  const scripts = await page.$$eval('script', scripts => 
    scripts.map(script => ({
      src: script.src,
      type: script.type,
      hasContent: script.innerHTML.length > 0
    }))
  );
  
  console.log(`📜 Scripts found:`, scripts);
  
  // Check network failures
  const failedRequests: string[] = [];
  page.on('requestfailed', request => {
    const failed = `FAILED REQUEST: ${request.url()} - ${request.failure()?.errorText}`;
    failedRequests.push(failed);
    console.log(`🌐 ${failed}`);
  });
  
  // Summary
  console.log('\n=== SUMMARY ===');
  console.log(`Console messages: ${consoleMessages.length}`);
  console.log(`Page errors: ${errors.length}`);
  console.log(`Failed requests: ${failedRequests.length}`);
  console.log(`Red screen visible: ${redScreenVisible}`);
  
  if (errors.length > 0) {
    console.log('\n❌ ERRORS:');
    errors.forEach(error => console.log(error));
  }
  
  if (failedRequests.length > 0) {
    console.log('\n🌐 FAILED REQUESTS:');
    failedRequests.forEach(req => console.log(req));
  }
  
  // Take a screenshot for debugging
  await page.screenshot({ path: 'frontend-debug-screenshot.png', fullPage: true });
  console.log('📸 Screenshot saved as frontend-debug-screenshot.png');
});