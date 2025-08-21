import { test, expect } from '@playwright/test';

test('Investigate Engine Tab Errors', async ({ page }) => {
  // Set up console error tracking
  const consoleErrors: string[] = [];
  const networkErrors: any[] = [];
  
  page.on('console', (msg) => {
    if (msg.type() === 'error') {
      consoleErrors.push(msg.text());
    }
  });

  page.on('response', (response) => {
    if (!response.ok()) {
      networkErrors.push({
        url: response.url(),
        status: response.status(),
        statusText: response.statusText()
      });
    }
  });

  // Navigate to the application
  console.log('Navigating to http://localhost:3000');
  await page.goto('http://localhost:3000');
  
  // Wait for the page to load
  await page.waitForLoadState('networkidle');
  
  // Take initial screenshot
  await page.screenshot({ path: 'frontend/test-results/engine-investigation-01-initial-load.png', fullPage: true });
  
  // Look for navigation tabs
  console.log('Looking for navigation tabs...');
  
  // Try different selectors for the Engine tab
  const engineTabSelectors = [
    'text="Engine"',
    '[data-testid="engine-tab"]',
    '.ant-tabs-tab:has-text("Engine")',
    'button:has-text("Engine")',
    'a:has-text("Engine")',
    '[role="tab"]:has-text("Engine")'
  ];
  
  let engineTabFound = false;
  let engineTab = null;
  
  for (const selector of engineTabSelectors) {
    try {
      engineTab = await page.locator(selector).first();
      if (await engineTab.isVisible()) {
        console.log(`Found Engine tab with selector: ${selector}`);
        engineTabFound = true;
        break;
      }
    } catch (e) {
      // Continue to next selector
    }
  }
  
  if (!engineTabFound) {
    console.log('Engine tab not found, checking all available tabs...');
    
    // Look for any tab-like elements
    const tabSelectors = [
      '.ant-tabs-tab',
      '[role="tab"]',
      'button[data-node-key]',
      '.tab',
      'nav button',
      'nav a'
    ];
    
    for (const selector of tabSelectors) {
      try {
        const tabs = await page.locator(selector);
        const count = await tabs.count();
        console.log(`Found ${count} elements with selector: ${selector}`);
        
        for (let i = 0; i < count; i++) {
          const tab = tabs.nth(i);
          const text = await tab.textContent();
          console.log(`Tab ${i}: "${text}"`);
        }
      } catch (e) {
        // Continue
      }
    }
    
    await page.screenshot({ path: 'frontend/test-results/engine-investigation-02-tabs-not-found.png', fullPage: true });
  } else {
    // Click on the Engine tab
    console.log('Clicking on Engine tab...');
    await engineTab.click();
    
    // Wait a moment for any content to load
    await page.waitForTimeout(2000);
    
    // Take screenshot after clicking
    await page.screenshot({ path: 'frontend/test-results/engine-investigation-03-engine-tab-clicked.png', fullPage: true });
    
    // Check for error messages in the UI
    const errorSelectors = [
      '.ant-message-error',
      '.error',
      '.alert-danger',
      '[class*="error"]',
      'text="Error"',
      'text="Failed"',
      'text="Cannot"'
    ];
    
    for (const selector of errorSelectors) {
      try {
        const errorElements = await page.locator(selector);
        const count = await errorElements.count();
        if (count > 0) {
          console.log(`Found ${count} error elements with selector: ${selector}`);
          for (let i = 0; i < count; i++) {
            const errorText = await errorElements.nth(i).textContent();
            console.log(`Error ${i}: ${errorText}`);
          }
        }
      } catch (e) {
        // Continue
      }
    }
    
    // Check for loading states
    const loadingSelectors = [
      '.ant-spin',
      '.loading',
      '.spinner',
      '[class*="loading"]'
    ];
    
    let hasLoadingState = false;
    for (const selector of loadingSelectors) {
      try {
        const loadingElements = await page.locator(selector);
        if (await loadingElements.count() > 0) {
          console.log(`Found loading state with selector: ${selector}`);
          hasLoadingState = true;
        }
      } catch (e) {
        // Continue
      }
    }
    
    if (hasLoadingState) {
      console.log('Waiting for loading to complete...');
      await page.waitForTimeout(5000);
      await page.screenshot({ path: 'frontend/test-results/engine-investigation-04-after-loading.png', fullPage: true });
    }
  }
  
  // Log all console errors
  console.log('\n=== CONSOLE ERRORS ===');
  consoleErrors.forEach((error, index) => {
    console.log(`Console Error ${index + 1}: ${error}`);
  });
  
  // Log all network errors
  console.log('\n=== NETWORK ERRORS ===');
  networkErrors.forEach((error, index) => {
    console.log(`Network Error ${index + 1}: ${error.status} ${error.statusText} - ${error.url}`);
  });
  
  // Take final screenshot
  await page.screenshot({ path: 'frontend/test-results/engine-investigation-05-final-state.png', fullPage: true });
  
  // Wait a bit more and check for any delayed errors
  await page.waitForTimeout(3000);
});