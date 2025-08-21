import { test, expect } from '@playwright/test';

test('test YFinance backfill functionality', async ({ page }) => {
  // Capture console logs for debugging
  page.on('console', msg => console.log('BROWSER:', msg.text()));
  
  // Navigate to dashboard
  await page.goto('http://localhost:3000');
  
  // Wait for dashboard to load
  await page.waitForTimeout(3000);
  
  // Check if YFinance section is present
  const yfinanceSection = page.locator('text=YFinance Data Source');
  await expect(yfinanceSection).toBeVisible();
  
  // Initialize YFinance first
  console.log('🔧 Initializing YFinance service...');
  const initializeBtn = page.locator('text=Initialize YFinance');
  await initializeBtn.click();
  
  // Wait for initialization
  await page.waitForTimeout(5000);
  
  // Check if YFinance is operational
  const operationalText = page.locator('text=operational').first();
  await expect(operationalText).toBeVisible({ timeout: 10000 });
  
  // Test YFinance data fetch
  console.log('📊 Testing YFinance data fetch...');
  const testBtn = page.locator('text=Test YFinance Data');
  await testBtn.click();
  
  // Wait for response
  await page.waitForTimeout(3000);
  
  // Test YFinance backfill
  console.log('🚀 Starting YFinance backfill...');
  const backfillBtn = page.locator('text=Start YFinance Backfill');
  await backfillBtn.click();
  
  // Wait for backfill to start
  await page.waitForTimeout(5000);
  
  // Take a screenshot
  await page.screenshot({ path: 'yfinance-backfill-test.png', fullPage: true });
  
  console.log('✅ YFinance backfill test completed');
});

test('verify YFinance status and functionality', async ({ page }) => {
  // Capture console logs for debugging
  page.on('console', msg => console.log('BROWSER:', msg.text()));
  
  // Navigate to dashboard
  await page.goto('http://localhost:3000');
  
  // Wait for dashboard to load
  await page.waitForTimeout(3000);
  
  // Verify YFinance status display
  const serviceStatus = page.locator('text=Service Status:');
  await expect(serviceStatus).toBeVisible();
  
  const initialized = page.locator('text=Initialized:');
  await expect(initialized).toBeVisible();
  
  const rateLimit = page.locator('text=Rate Limit:');
  await expect(rateLimit).toBeVisible();
  
  // Verify all YFinance buttons are present
  const initBtn = page.locator('text=Initialize YFinance');
  await expect(initBtn).toBeVisible();
  
  const testBtn = page.locator('text=Test YFinance Data');
  await expect(testBtn).toBeVisible();
  
  const backfillBtn = page.locator('text=Start YFinance Backfill');
  await expect(backfillBtn).toBeVisible();
  
  const refreshBtn = page.locator('button:has-text("Refresh YFinance Status")');
  await expect(refreshBtn).toBeVisible();
  
  // Take a screenshot
  await page.screenshot({ path: 'yfinance-status-verification.png', fullPage: true });
  
  console.log('✅ YFinance status verification completed');
});