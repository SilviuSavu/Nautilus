import { test, expect } from '@playwright/test';

test('strategy tab integrated into main dashboard', async ({ page }) => {
  // Navigate to main dashboard
  await page.goto('http://localhost:3000/dashboard');
  
  // Wait for tabs to load
  await page.waitForTimeout(3000);

  // Take screenshot of main dashboard with tabs
  await page.screenshot({ path: 'dashboard-with-strategy-tab.png', fullPage: true });

  // Check that Strategy Management tab exists
  const strategyTab = await page.locator('div[role="tab"]:has-text("Strategy Management")').count();
  console.log('STRATEGY TAB FOUND:', strategyTab);

  // Click on Strategy Management tab
  if (strategyTab > 0) {
    await page.click('div[role="tab"]:has-text("Strategy Management")');
    await page.waitForTimeout(2000);
    
    // Take screenshot of strategy tab content
    await page.screenshot({ path: 'strategy-tab-content.png', fullPage: true });
    
    // Check for strategy content
    const strategyContent = await page.textContent('body');
    console.log('STRATEGY TAB CONTENT (first 300 chars):', strategyContent?.substring(0, 300));
    
    // Verify strategy dashboard content is loaded
    const hasStrategyDashboard = await page.locator('text=Strategy Management Dashboard').count();
    console.log('STRATEGY DASHBOARD CONTENT:', hasStrategyDashboard);
    
    expect(hasStrategyDashboard).toBeGreaterThan(0);
  }

  // Check that old separate routes no longer work
  await page.goto('http://localhost:3000/strategy');
  await page.waitForTimeout(1000);
  
  // Should redirect to dashboard or show 404, not show strategy page separately
  const currentUrl = page.url();
  console.log('STRATEGY ROUTE URL:', currentUrl);
  
  // Verify tab structure
  expect(strategyTab).toBeGreaterThan(0);
});