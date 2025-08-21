import { test, expect } from '@playwright/test';

test('frontend loads without blank page', async ({ page }) => {
  // Listen for console logs and errors
  page.on('console', msg => console.log('BROWSER CONSOLE:', msg.type(), msg.text()));
  page.on('pageerror', error => console.log('BROWSER ERROR:', error.message));

  // Navigate to the application
  await page.goto('http://localhost:3000');

  // Wait for the app to load
  await page.waitForTimeout(3000);

  // Check if the main title is visible
  await expect(page.getByText('NautilusTrader Dashboard')).toBeVisible();

  // Check if the main content is not blank
  const dashboardElement = page.getByTestId('dashboard');
  await expect(dashboardElement).toBeVisible();

  // Take a screenshot for verification
  await page.screenshot({ path: 'frontend-loaded.png' });

  // Check if any tabs are visible
  const systemTab = page.getByText('System Overview');
  await expect(systemTab).toBeVisible();

  console.log('✅ Frontend is successfully loading with content');
});