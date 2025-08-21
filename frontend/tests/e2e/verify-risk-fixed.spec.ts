import { test, expect } from '@playwright/test';

test('Verify Risk Dashboard is Working', async ({ page }) => {
  await page.goto('http://localhost:3000');
  await page.waitForTimeout(2000);
  
  // Click on the Risk tab
  await page.click('text=Risk');
  await page.waitForTimeout(5000);
  
  // Check that no error messages are present
  await expect(page.locator('text=Error Loading Risk Data')).not.toBeVisible();
  await expect(page.locator('text=Network Error')).not.toBeVisible();
  
  // Check that key dashboard elements are visible
  await expect(page.locator('text=Portfolio Value').first()).toBeVisible();
  await expect(page.locator('text=Expected Shortfall').first()).toBeVisible();
  
  // Take a screenshot of the working dashboard
  await page.screenshot({ path: 'risk-dashboard-fixed.png', fullPage: true });
  
  console.log('✅ Risk Dashboard is working correctly - no errors found');
});