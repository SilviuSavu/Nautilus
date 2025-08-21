import { test, expect } from '@playwright/test';

test('Verify Engine Tab Fix', async ({ page }) => {
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
  await page.screenshot({ path: 'frontend/test-results/engine-fix-01-initial-load.png', fullPage: true });
  
  // Click on Engine tab
  console.log('Clicking Engine tab...');
  await page.click('text="Engine"');
  
  // Wait for content to load
  await page.waitForTimeout(3000);
  
  // Take screenshot after clicking Engine tab
  await page.screenshot({ path: 'frontend/test-results/engine-fix-02-engine-tab-clicked.png', fullPage: true });
  
  // Check if the error message is gone
  const errorMessage = page.locator('text="Unable to load engine status"');
  const errorExists = await errorMessage.isVisible();
  
  console.log(`Error message visible: ${errorExists}`);
  
  // Look for success indicators
  const engineManagerTitle = page.locator('text="NautilusTrader Engine Manager"');
  const engineManagerVisible = await engineManagerTitle.isVisible();
  
  console.log(`Engine Manager title visible: ${engineManagerVisible}`);
  
  // Check for status indicators
  const statusBadge = page.locator('.ant-badge');
  const statusBadgeCount = await statusBadge.count();
  
  console.log(`Found ${statusBadgeCount} status badges`);
  
  // Log console errors
  console.log('\n=== CONSOLE ERRORS ===');
  if (consoleErrors.length === 0) {
    console.log('No console errors found!');
  } else {
    consoleErrors.forEach((error, index) => {
      console.log(`Console Error ${index + 1}: ${error}`);
    });
  }
  
  // Log network errors
  console.log('\n=== NETWORK ERRORS ===');
  if (networkErrors.length === 0) {
    console.log('No network errors found!');
  } else {
    networkErrors.forEach((error, index) => {
      console.log(`Network Error ${index + 1}: ${error.status} ${error.statusText} - ${error.url}`);
    });
  }
  
  // Take final screenshot
  await page.screenshot({ path: 'frontend/test-results/engine-fix-03-final-state.png', fullPage: true });
  
  // Assert that the fix worked
  expect(consoleErrors.filter(e => e.includes('Failed to fetch engine status')).length).toBe(0);
  expect(networkErrors.filter(e => e.url.includes('/api/v1/nautilus/engine/status')).length).toBe(0);
});