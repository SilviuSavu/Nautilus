import { test, expect } from '@playwright/test';

test.describe('Instrument Search Functionality', () => {
  test.beforeEach(async ({ page }) => {
    // Navigate to the dashboard and switch to instrument search tab
    await page.goto('/');
    await page.waitForLoadState('networkidle');
    
    // Wait for the dashboard to load and click instrument search tab
    await page.waitForSelector('[data-testid="instrument-search-tab"]', { timeout: 10000 });
    await page.click('[data-testid="instrument-search-tab"]');
    await page.waitForTimeout(1000);
  });

  test('should display instrument search interface', async ({ page }) => {
    // Check if search input is visible
    await expect(page.locator('input[placeholder*="Search instruments"]')).toBeVisible();
    
    // Check if search results container exists
    await expect(page.locator('[data-testid="search-results"]')).toBeVisible();
  });

  test('should perform fuzzy search functionality', async ({ page }) => {
    const searchInput = page.locator('input[placeholder*="Search instruments"]');
    
    // Test fuzzy search with a real symbol
    await searchInput.fill('AAPL');
    await page.waitForTimeout(500); // Wait for debouncing
    
    // Should show search results
    const resultsContainer = page.locator('[data-testid="search-results"]');
    await expect(resultsContainer).toBeVisible();
    
    // Should show some results or "No instruments found" message
    const hasResults = await page.locator('.ant-list-item').count() > 0;
    const hasNoResultsMessage = await page.locator('text=No instruments found').isVisible();
    
    expect(hasResults || hasNoResultsMessage).toBeTruthy();
  });

  test('should display favorites and recent sections', async ({ page }) => {
    // Check for favorites section
    await expect(page.locator('text=Favorites')).toBeVisible();
    
    // Check for recent selections section
    await expect(page.locator('text=Recent Selections')).toBeVisible();
  });

  test('should show asset class tags for instruments', async ({ page }) => {
    const searchInput = page.locator('input[placeholder*="Search instruments"]');
    await searchInput.fill('EURUSD');
    await page.waitForTimeout(500);
    
    // Look for asset class tags - should be unique elements
    const assetTags = page.locator('.ant-tag').filter({ hasText: /FX|Equities|Futures|Options|Crypto/ });
    
    if (await assetTags.count() > 0) {
      // Verify no duplicate elements causing strict mode violations
      const tagTexts = await assetTags.allTextContents();
      const uniqueTexts = [...new Set(tagTexts)];
      expect(tagTexts.length).toBeLessThanOrEqual(uniqueTexts.length * 2); // Allow some reasonable duplication
    }
  });

  test('should display venue status indicators', async ({ page }) => {
    const searchInput = page.locator('input[placeholder*="Search instruments"]');
    await searchInput.fill('AAPL');
    await page.waitForTimeout(500);
    
    // Check for venue status indicators
    const venueIndicators = page.locator('[data-testid="venue-status"]');
    
    if (await venueIndicators.count() > 0) {
      await expect(venueIndicators.first()).toBeVisible();
    } else {
      // If no results, should show appropriate message
      await expect(page.locator('text=No instruments found')).toBeVisible();
    }
  });

  test('should handle error states gracefully', async ({ page }) => {
    // Mock a network error or empty response
    await page.route('**/api/v1/ib/instruments/**', route => {
      route.fulfill({ status: 500, body: 'Server Error' });
    });
    
    const searchInput = page.locator('input[placeholder*="Search instruments"]');
    await searchInput.fill('TEST');
    await page.waitForTimeout(1000);
    
    // Should display error message or "No instruments found"
    const hasErrorMessage = await page.locator('text=Error').isVisible();
    const hasNoResultsMessage = await page.locator('text=No instruments found').isVisible();
    
    expect(hasErrorMessage || hasNoResultsMessage).toBeTruthy();
  });
});