import { test, expect } from '@playwright/test';

test('Risk Dashboard Functionality Test', async ({ page }) => {
  // Navigate to the application
  await page.goto('http://localhost:3000');
  
  // Wait for the page to load
  await page.waitForTimeout(2000);
  
  // Click on the Risk tab
  await page.click('text=Risk');
  
  // Wait for the Risk dashboard to load
  await page.waitForTimeout(3000);
  
  // Check if Risk dashboard content is visible
  await expect(page.locator('text=Portfolio Value')).toBeVisible();
  await expect(page.locator('text=1-Day VaR (95%)')).toBeVisible();
  await expect(page.locator('text=Expected Shortfall')).toBeVisible();
  await expect(page.locator('text=Portfolio Beta')).toBeVisible();
  
  // Check for error messages (should not be present)
  await expect(page.locator('text=Error Loading Risk Data')).not.toBeVisible();
  await expect(page.locator('text=Request failed with status code 500')).not.toBeVisible();
  await expect(page.locator('text=Request failed with status code 404')).not.toBeVisible();
  
  // Take a screenshot to verify the dashboard is working
  await page.screenshot({ path: 'risk-dashboard-working.png', fullPage: true });
  
  console.log('✅ Risk Dashboard is functioning correctly');
});