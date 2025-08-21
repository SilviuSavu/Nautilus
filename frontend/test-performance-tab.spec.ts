import { test, expect } from '@playwright/test';

test.describe('Performance Tab - Fixed', () => {
  test('should load Performance tab without errors', async ({ page }) => {
    // Go to the frontend
    await page.goto('http://localhost:3001');
    
    // Wait for the dashboard to load
    await page.waitForSelector('[data-testid="dashboard"]', { timeout: 10000 });
    
    // Click on the Performance tab
    await page.click('text=Perform');
    
    // Wait for the Performance Dashboard to load
    await page.waitForSelector('text=Performance Dashboard', { timeout: 10000 });
    
    // Check that the key components are present
    await expect(page.locator('text=Performance Dashboard')).toBeVisible();
    await expect(page.locator('text=Real-time strategy performance monitoring and analytics')).toBeVisible();
    
    // Check that the main tabs are present
    await expect(page.locator('text=Overview')).toBeVisible();
    await expect(page.locator('text=Real-Time Monitor')).toBeVisible();
    await expect(page.locator('text=Strategy Comparison')).toBeVisible();
    
    // Check that no console errors related to missing components
    const logs = [];
    page.on('console', msg => {
      if (msg.type() === 'error') {
        logs.push(msg.text());
      }
    });
    
    // Refresh the page to capture any console errors
    await page.reload();
    await page.waitForSelector('text=Performance Dashboard', { timeout: 10000 });
    
    // Filter out unrelated errors and focus on component import errors
    const componentErrors = logs.filter(log => 
      log.includes('Cannot resolve module') || 
      log.includes('DataExportDashboard') ||
      log.includes('SystemPerformanceDashboard')
    );
    
    expect(componentErrors).toHaveLength(0);
    
    console.log('✅ Performance tab loaded successfully');
    console.log('✅ No component import errors detected');
  });
  
  test('should display performance metrics', async ({ page }) => {
    await page.goto('http://localhost:3001');
    await page.waitForSelector('[data-testid="dashboard"]');
    
    // Click Performance tab
    await page.click('text=Perform');
    await page.waitForSelector('text=Performance Dashboard');
    
    // Check that metrics are displayed (even if they show loading or default values)
    await expect(page.locator('text=Total P&L')).toBeVisible();
    await expect(page.locator('text=Sharpe Ratio')).toBeVisible();
    await expect(page.locator('text=Win Rate')).toBeVisible();
    
    console.log('✅ Performance metrics are visible');
  });
  
  test('should switch between performance tabs', async ({ page }) => {
    await page.goto('http://localhost:3001');
    await page.waitForSelector('[data-testid="dashboard"]');
    
    // Click Performance tab
    await page.click('text=Perform');
    await page.waitForSelector('text=Performance Dashboard');
    
    // Test switching to different sub-tabs
    await page.click('text=Real-Time Monitor');
    await page.waitForTimeout(1000); // Allow tab to load
    
    await page.click('text=Strategy Comparison');
    await page.waitForTimeout(1000);
    
    await page.click('text=System Monitoring');
    await expect(page.locator('text=System Performance Dashboard')).toBeVisible();
    
    await page.click('text=Data Export');
    await expect(page.locator('text=Data Export Dashboard')).toBeVisible();
    
    console.log('✅ All performance sub-tabs are working');
  });
});