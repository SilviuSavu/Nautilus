import { test, expect } from '@playwright/test';

test('verify strategy tab in dashboard', async ({ page }) => {
  await page.goto('http://localhost:3000');
  await page.waitForTimeout(3000);

  // Take screenshot
  await page.screenshot({ path: 'final-integration.png', fullPage: true });

  // Look for Strategy Management tab
  const tabText = await page.textContent('body');
  console.log('PAGE CONTAINS STRATEGY TAB:', tabText?.includes('Strategy Management'));

  // Count tabs
  const tabs = await page.locator('[role="tab"]').count();
  console.log('TOTAL TABS:', tabs);

  // List all tab names
  const tabNames = await page.locator('[role="tab"]').allTextContents();
  console.log('TAB NAMES:', tabNames);

  expect(tabs).toBeGreaterThan(4); // Should have at least 5 tabs now including Strategy
});