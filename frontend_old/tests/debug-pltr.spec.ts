import { test, expect } from '@playwright/test';

test('Debug PLTR Search', async ({ page }) => {
  // Capture ALL console messages
  page.on('console', msg => {
    console.log(`BROWSER [${msg.type()}]:`, msg.text());
  });

  // Capture network requests
  page.on('request', request => {
    if (request.url().includes('localhost:8000')) {
      console.log(`NETWORK REQUEST: ${request.method()} ${request.url()}`);
    }
  });

  page.on('response', async response => {
    if (response.url().includes('localhost:8000')) {
      console.log(`NETWORK RESPONSE: ${response.status()} ${response.url()}`);
      if (response.url().includes('PLTR')) {
        try {
          const text = await response.text();
          console.log(`PLTR RESPONSE BODY:`, text.substring(0, 500));
        } catch (e) {
          console.log(`Could not read response body:`, e.message);
        }
      }
    }
  });

  console.log('=== NAVIGATING TO APPLICATION ===');
  await page.goto('http://localhost:3000');
  
  await page.waitForTimeout(2000);
  
  // Find search input
  const searchInput = page.locator('input[placeholder*="Search"]').first();
  
  if (await searchInput.isVisible()) {
    console.log('=== TYPING PLTR ===');
    await searchInput.fill('PLTR');
    
    // Wait longer for API call
    await page.waitForTimeout(5000);
    
    console.log('=== CHECKING RESULTS ===');
    const results = await page.locator('[class*="ant-list-item"]').count();
    console.log(`Found ${results} result elements in UI`);
    
  } else {
    console.log('Could not find search input');
  }
  
  await page.waitForTimeout(2000);
});