import { test, expect } from '@playwright/test';

test.describe('Nautilus Frontend Verification', () => {
  test('Dashboard loads and displays content', async ({ page }) => {
    // Navigate to the frontend
    await page.goto('http://localhost:3000');
    
    // Wait for the page to load
    await page.waitForLoadState('networkidle');
    
    // Check that the title is correct
    await expect(page).toHaveTitle(/Nautilus Trader Dashboard/);
    
    // Check that the main dashboard element exists
    await expect(page.locator('[data-testid="dashboard"]')).toBeVisible({ timeout: 10000 });
    
    // Check that the main heading exists
    await expect(page.locator('h2').filter({ hasText: 'NautilusTrader Dashboard' })).toBeVisible();
    
    // Check that the System tab is visible (default tab)
    await expect(page.locator('.ant-tabs-tab').filter({ hasText: 'System' })).toBeVisible();
    
    // Check that backend status alert is visible
    await expect(page.locator('.ant-alert').first()).toBeVisible();
    
    // Take a screenshot to verify visual rendering
    await page.screenshot({ path: 'dashboard-screenshot.png', fullPage: true });
    
    console.log('✅ Dashboard loaded successfully with visible content');
  });

  test('Tab navigation works', async ({ page }) => {
    await page.goto('http://localhost:3000');
    await page.waitForLoadState('networkidle');
    
    // Wait for dashboard to be visible
    await expect(page.locator('[data-testid="dashboard"]')).toBeVisible({ timeout: 10000 });
    
    // Test clicking different tabs
    const tabs = ['Engine', 'Search', 'Chart', 'IB'];
    
    for (const tab of tabs) {
      const tabLocator = page.locator('.ant-tabs-tab').filter({ hasText: tab });
      if (await tabLocator.isVisible()) {
        await tabLocator.click();
        await page.waitForTimeout(500); // Wait for tab content to load
        console.log(`✅ ${tab} tab clicked successfully`);
      }
    }
  });

  test('Backend connection status is displayed', async ({ page }) => {
    await page.goto('http://localhost:3000');
    await page.waitForLoadState('networkidle');
    
    // Check for backend status
    const statusAlert = page.locator('.ant-alert').first();
    await expect(statusAlert).toBeVisible();
    
    // The alert should contain either "Connected" or "Disconnected"
    const alertText = await statusAlert.textContent();
    expect(alertText).toMatch(/(Connected|Disconnected|Checking)/);
    
    console.log(`✅ Backend status displayed: ${alertText}`);
  });

  test('Page is not blank - has visible content', async ({ page }) => {
    await page.goto('http://localhost:3000');
    await page.waitForLoadState('networkidle');
    
    // Check that there's actual content, not just a blank page
    const bodyText = await page.locator('body').textContent();
    expect(bodyText).toBeTruthy();
    expect(bodyText.length).toBeGreaterThan(100); // Should have substantial content
    
    // Check for specific dashboard elements
    await expect(page.locator('.ant-card')).toHaveCount({ min: 1 });
    await expect(page.locator('.ant-tabs')).toBeVisible();
    
    console.log('✅ Page has substantial content and is not blank');
  });
});