import { test, expect } from '@playwright/test';

test('Verify PLTR Search Fix', async ({ page }) => {
  // Navigate to application
  await page.goto('http://localhost:3000');
  await page.waitForTimeout(2000);
  
  // Find search input
  const searchInput = page.locator('input[placeholder*="Search"]').first();
  await expect(searchInput).toBeVisible();
  
  // Search for PLTR
  await searchInput.fill('PLTR');
  await page.waitForTimeout(3000);
  
  // Check that we have search results displayed
  const resultItems = page.locator('[class*="ant-list-item"]');
  const resultCount = await resultItems.count();
  
  console.log(`PLTR search shows ${resultCount} results`);
  
  // Verify we have results (should be 25: 1 stock + 24 options)
  expect(resultCount).toBeGreaterThan(20);
  
  // Verify the first result contains PLTR
  const firstResult = resultItems.first();
  await expect(firstResult).toContainText('PLTR');
  
  // Take screenshot as proof
  await page.screenshot({ path: '/Users/savusilviu/Desktop/SilviuCorneliuSavu/Nautilus/frontend/tests/screenshots/pltr-search-working.png', fullPage: true });
});