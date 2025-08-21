import { test, expect } from '@playwright/test';

test('Verify localhost:3000 is working after fix', async ({ page }) => {
  console.log('Testing fixed localhost:3000...');
  
  // Navigate to the frontend
  await page.goto('http://localhost:3000');
  
  // Wait for page to load
  await page.waitForLoadState('networkidle', { timeout: 10000 });
  
  // Take a screenshot to confirm it's working
  await page.screenshot({ path: 'frontend/localhost-3000-fixed.png', fullPage: true });
  
  // Check basic functionality
  const title = await page.title();
  const reactRoot = await page.locator('#root').count();
  const bodyText = await page.locator('body').textContent();
  
  console.log('=== LOCALHOST:3000 FIXED REPORT ===');
  console.log('Page Title:', title);
  console.log('React Root Present:', reactRoot > 0);
  console.log('Page Content Length:', bodyText?.length || 0);
  
  // Verify it's working
  expect(title).toContain('Nautilus');
  expect(reactRoot).toBeGreaterThan(0);
  expect(bodyText?.length || 0).toBeGreaterThan(100);
  
  console.log('✅ localhost:3000 is now working correctly!');
});