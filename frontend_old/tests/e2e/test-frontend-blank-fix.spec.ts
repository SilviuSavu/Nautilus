import { test, expect } from '@playwright/test';

test.describe('Frontend Blank Page Fix', () => {
  test('should load Dashboard with content (not blank)', async ({ page }) => {
    // Enable console logging
    page.on('console', msg => console.log('BROWSER:', msg.text()));
    
    // Navigate to frontend
    await page.goto('http://localhost:3000');
    
    // Wait for React to load
    await page.waitForTimeout(3000);
    
    // Verify page title
    await expect(page).toHaveTitle(/Nautilus Trader Dashboard/);
    
    // Verify Dashboard header loads
    await expect(page.locator('h2')).toContainText('NautilusTrader Dashboard');
    
    // Verify tabs are present (not blank page)
    const tabs = page.locator('.ant-tabs-tab');
    await expect(tabs).toHaveCount(7); // System, Instruments, Watchlists, Chart, Strategy, Performance, IB
    
    // Verify Performance tab exists
    await expect(page.locator('text=Performance Monitoring')).toBeVisible();
    
    // Click Performance tab to verify it loads
    await page.click('text=Performance Monitoring');
    await page.waitForTimeout(2000);
    
    // Verify Performance Dashboard components render
    const performanceContent = page.locator('[data-testid="performance-dashboard"]');
    await expect(performanceContent).toBeVisible({ timeout: 5000 });
    
    // Check for any error messages or blank states
    const errorMessages = await page.locator('.ant-alert-error').count();
    console.log(`Error messages found: ${errorMessages}`);
    
    // Take screenshot for evidence
    await page.screenshot({ path: 'frontend-dashboard-verification.png', fullPage: true });
    
    console.log('✅ Frontend Dashboard loaded successfully with Performance tab');
  });

  test('should verify backend API connectivity', async ({ page }) => {
    page.on('console', msg => console.log('BROWSER:', msg.text()));
    
    await page.goto('http://localhost:3000');
    await page.waitForTimeout(2000);
    
    // Check System Overview tab for backend status
    await page.click('text=System Overview');
    await page.waitForTimeout(1000);
    
    // Look for backend connected status
    const backendStatus = page.locator('text=Backend Connected');
    await expect(backendStatus).toBeVisible({ timeout: 5000 });
    
    console.log('✅ Backend connectivity verified from frontend');
  });
});