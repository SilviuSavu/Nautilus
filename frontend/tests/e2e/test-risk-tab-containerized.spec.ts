import { test, expect } from '@playwright/test';

test('Risk tab works with containerized backend on port 8001', async ({ page }) => {
  // Go to localhost:3000 (containerized frontend)
  await page.goto('http://localhost:3000');
  
  // Wait for the page to load
  await page.waitForSelector('[data-testid="dashboard"]', { timeout: 15000 });
  
  // Click on the Risk tab
  await page.click('text=Risk');
  
  // Wait for risk dashboard to load
  await page.waitForTimeout(3000);
  
  // Check if we can see risk-related content
  await expect(page.locator('text=Portfolio Value')).toBeVisible({ timeout: 15000 });
  
  // Take a screenshot for verification
  await page.screenshot({ path: 'frontend/risk-tab-containerized-success.png' });
  
  console.log('✅ Risk tab with containerized backend test completed successfully');
});