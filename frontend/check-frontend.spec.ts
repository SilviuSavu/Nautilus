import { test, expect } from '@playwright/test';

test('Check frontend dashboard errors', async ({ page }) => {
  const consoleMessages: string[] = [];
  const networkErrors: string[] = [];

  // Capture console messages
  page.on('console', msg => {
    consoleMessages.push(`${msg.type()}: ${msg.text()}`);
  });

  // Capture network errors
  page.on('requestfailed', request => {
    networkErrors.push(`Failed: ${request.method()} ${request.url()} - ${request.failure()?.errorText}`);
  });

  console.log('🚀 Navigating to http://localhost:3000');
  
  try {
    await page.goto('http://localhost:3000', { 
      waitUntil: 'domcontentloaded',
      timeout: 10000 
    });

    // Wait for the page to load
    await page.waitForTimeout(3000);

    // Take a screenshot
    await page.screenshot({ 
      path: 'frontend-current-state.png', 
      fullPage: true 
    });

    console.log('\n=== PAGE TITLE ===');
    const title = await page.title();
    console.log(`Title: ${title}`);

    console.log('\n=== CONSOLE MESSAGES ===');
    consoleMessages.forEach(msg => console.log(msg));

    console.log('\n=== NETWORK ERRORS ===');
    networkErrors.forEach(error => console.log(error));

    console.log('\n=== PAGE CONTENT CHECK ===');
    const bodyText = await page.textContent('body');
    if (bodyText?.includes('Error')) {
      console.log('❌ Found "Error" in page content');
    }
    if (bodyText?.includes('Failed')) {
      console.log('❌ Found "Failed" in page content');  
    }
    if (bodyText?.includes('Cannot read properties')) {
      console.log('❌ Found React/JavaScript errors in page content');
    }

    // Check for specific error elements
    const errorElements = await page.locator('.ant-alert-error, .error, [class*="error"]').count();
    console.log(`Found ${errorElements} error elements on page`);

    // Check if the page loaded successfully
    const isPageLoaded = await page.locator('body').isVisible();
    console.log(`Page loaded successfully: ${isPageLoaded}`);

  } catch (error) {
    console.error('❌ Error loading page:', error);
  }
});