import { test, expect } from '@playwright/test';

test('Risk tab with containerized backend', async ({ page }) => {
  // Go to localhost:3000 (containerized frontend)
  await page.goto('http://localhost:3000');
  
  // Wait for the page to load
  await page.waitForSelector('[data-testid="dashboard"]', { timeout: 15000 });
  
  // Take initial screenshot
  await page.screenshot({ path: 'frontend/containerized-initial.png' });
  
  // Click on the Risk tab
  await page.click('text=Risk');
  
  // Wait for risk dashboard to load
  await page.waitForTimeout(3000);
  
  // Take screenshot of Risk tab
  await page.screenshot({ path: 'frontend/containerized-risk-tab.png' });
  
  // Check if we can see risk-related content
  const portfolioValueVisible = await page.locator('text=Portfolio Value').isVisible();
  
  if (portfolioValueVisible) {
    console.log('✅ Risk tab with containerized backend test successful');
  } else {
    console.log('⚠️ Risk tab loaded but Portfolio Value not visible - may still be loading');
  }
  
  expect(portfolioValueVisible).toBeTruthy();
});