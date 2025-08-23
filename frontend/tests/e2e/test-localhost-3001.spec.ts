import { test, expect } from '@playwright/test';

test('Test localhost:3000 frontend working', async ({ page }) => {
  console.log('Testing localhost:3000...');
  
  // Navigate to the frontend on the new port
  await page.goto('http://localhost:3000');
  
  // Wait for page to load
  await page.waitForLoadState('networkidle', { timeout: 10000 });
  
  // Take a screenshot to see current state
  await page.screenshot({ path: 'frontend/debug-localhost-3001-working.png', fullPage: true });
  
  // Check for console errors
  const consoleErrors: string[] = [];
  
  page.on('console', msg => {
    if (msg.type() === 'error') {
      consoleErrors.push(msg.text());
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
  
  // Check for navigation
  const tabsContainer = await page.locator('.ant-tabs, [role="tablist"]').count();
  console.log('Tabs found:', tabsContainer > 0);
  
  // Look for main dashboard components
  const dashboardElements = await page.locator('text=/Dashboard|Interactive Brokers|Instruments|Portfolio/i').count();
  console.log('Dashboard elements found:', dashboardElements);
  
  // Wait a bit for any async loading
  await page.waitForTimeout(3000);
  
  // Final screenshot
  await page.screenshot({ path: 'frontend/debug-3001-final-state.png', fullPage: true });
  
  // Report findings
  console.log('\n=== LOCALHOST:3001 TEST REPORT ===');
  console.log('Page Title:', title);
  console.log('React Root Present:', reactRoot > 0);
  console.log('Page Content Length:', bodyText?.length || 0);
  console.log('Console Errors:', consoleErrors.length);
  console.log('Dashboard Elements:', dashboardElements);
  
  if (consoleErrors.length > 0) {
    console.log('\nConsole Errors:');
    consoleErrors.forEach(error => console.log('  -', error));
  }
  
  // Assertions to ensure the frontend is working
  expect(title).toContain('Nautilus');
  expect(reactRoot).toBeGreaterThan(0);
  expect(bodyText?.length || 0).toBeGreaterThan(50);
});

test('Test navigation and basic functionality on 3001', async ({ page }) => {
  await page.goto('http://localhost:3000');
  await page.waitForLoadState('networkidle');
  
  // Try clicking on different tabs if they exist
  const ibTab = await page.locator('text=/Interactive Brokers|IB/i').first();
  if (await ibTab.count() > 0) {
    await ibTab.click();
    await page.waitForTimeout(1000);
    console.log('IB tab clicked successfully');
  }
  
  const instrumentsTab = await page.locator('text=/Instruments/i').first();
  if (await instrumentsTab.count() > 0) {
    await instrumentsTab.click();
    await page.waitForTimeout(1000);
    console.log('Instruments tab clicked successfully');
  }
  
  await page.screenshot({ path: 'frontend/navigation-test-3001.png', fullPage: true });
  
  console.log('Navigation test completed');
});