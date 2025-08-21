import { test, expect } from '@playwright/test';

test('Risk tab loads successfully', async ({ page }) => {
  // Go to localhost:3000
  await page.goto('http://localhost:3000');
  
  // Wait for the page to load
  await page.waitForSelector('[data-testid="dashboard"]');
  
  // Click on the Risk tab
  await page.click('text=Risk');
  
  // Wait for risk dashboard to load
  await page.waitForTimeout(2000);
  
  // Check if we can see risk-related content
  await expect(page.locator('text=Portfolio Value')).toBeVisible({ timeout: 10000 });
  
  // Take a screenshot for verification
  await page.screenshot({ path: 'frontend/risk-tab-test-success.png' });
  
  console.log('✅ Risk tab test completed successfully');
});