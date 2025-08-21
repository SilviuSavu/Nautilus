import { test, expect } from '@playwright/test';

test('Debug localhost:3000 frontend issues', async ({ page }) => {
  console.log('Starting frontend debug session...');
  
  // Navigate to the frontend
  await page.goto('http://localhost:3000');
  
  // Wait for page to load
  await page.waitForLoadState('networkidle');
  
  // Take a screenshot to see current state
  await page.screenshot({ path: 'frontend/debug-current-state.png', fullPage: true });
  
  // Check for console errors
  const consoleLogs: string[] = [];
  const consoleErrors: string[] = [];
  
  page.on('console', msg => {
    if (msg.type() === 'error') {
      consoleErrors.push(msg.text());
    } else {
      consoleLogs.push(`${msg.type()}: ${msg.text()}`);
    }
  });
  
  // Check if page title loads
  const title = await page.title();
  console.log('Page title:', title);
  
  // Check if React app is mounted
  const reactRoot = await page.locator('#root').count();
  console.log('React root found:', reactRoot > 0);
  
  // Check for any visible content
  const bodyText = await page.locator('body').textContent();
  console.log('Body content length:', bodyText?.length || 0);
  console.log('Body content preview:', bodyText?.substring(0, 200));
  
  // Check for common React/Vite elements
  const viteClient = await page.locator('script[type="module"][src*="@vite/client"]').count();
  console.log('Vite client script found:', viteClient > 0);
  
  // Look for any error messages on the page
  const errorMessages = await page.locator('text=/error|Error|ERROR/i').all();
  console.log('Error messages found:', errorMessages.length);
  
  for (const error of errorMessages) {
    const errorText = await error.textContent();
    console.log('Error on page:', errorText);
  }
  
  // Check network requests
  let networkErrors: string[] = [];
  page.on('response', response => {
    if (!response.ok()) {
      networkErrors.push(`${response.status()} - ${response.url()}`);
    }
  });
  
  // Reload page to catch any network issues
  await page.reload();
  await page.waitForLoadState('networkidle');
  
  // Wait a bit for any async loading
  await page.waitForTimeout(2000);
  
  // Final screenshot
  await page.screenshot({ path: 'frontend/debug-final-state.png', fullPage: true });
  
  // Report findings
  console.log('\n=== DEBUG REPORT ===');
  console.log('Page Title:', title);
  console.log('React Root Present:', reactRoot > 0);
  console.log('Vite Client Present:', viteClient > 0);
  console.log('Page Content Length:', bodyText?.length || 0);
  console.log('Console Errors:', consoleErrors.length);
  console.log('Network Errors:', networkErrors.length);
  
  if (consoleErrors.length > 0) {
    console.log('\nConsole Errors:');
    consoleErrors.forEach(error => console.log('  -', error));
  }
  
  if (networkErrors.length > 0) {
    console.log('\nNetwork Errors:');
    networkErrors.forEach(error => console.log('  -', error));
  }
  
  // Try to identify specific issues
  if (bodyText?.includes('Cannot GET')) {
    console.log('\nISSUE: Server routing issue detected');
  }
  
  if (bodyText && bodyText.length < 100) {
    console.log('\nISSUE: Minimal content suggests loading failure');
  }
  
  if (reactRoot === 0) {
    console.log('\nISSUE: React root element not found');
  }
  
  if (viteClient === 0) {
    console.log('\nISSUE: Vite development client not loaded');
  }
});