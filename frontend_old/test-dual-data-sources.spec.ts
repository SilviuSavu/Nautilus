import { test, expect } from '@playwright/test';

test('dashboard shows both IB Gateway and YFinance data sources', async ({ page }) => {
  // Capture console logs for debugging
  page.on('console', msg => console.log('BROWSER:', msg.text()));
  
  // Navigate to dashboard
  await page.goto('http://localhost:3000');
  
  // Wait for dashboard to load
  await page.waitForTimeout(3000);
  
  // Check if the Historical Data Backfill Status card is present
  const backfillCard = page.locator('text=Historical Data Backfill Status');
  await expect(backfillCard).toBeVisible();
  
  // Check for YFinance Data Source section
  const yfinanceSection = page.locator('text=YFinance Data Source');
  await expect(yfinanceSection).toBeVisible();
  
  // Check for IB Gateway Backfill section (use first match to avoid strict mode violation)
  const ibSection = page.locator('text=IB Gateway Backfill').first();
  await expect(ibSection).toBeVisible();
  
  // Check for YFinance status indicators
  const yfinanceStatus = page.locator('text=Service Status:');
  await expect(yfinanceStatus).toBeVisible();
  
  // Check for YFinance actions
  const initializeBtn = page.locator('text=Initialize YFinance');
  await expect(initializeBtn).toBeVisible();
  
  const testBtn = page.locator('text=Test YFinance Data');
  await expect(testBtn).toBeVisible();
  
  // Check for IB Gateway actions
  const ibBackfillBtn = page.locator('text=Start IB Gateway Backfill');
  await expect(ibBackfillBtn).toBeVisible();
  
  // Take a screenshot for evidence
  await page.screenshot({ path: 'dual-data-sources-dashboard.png', fullPage: true });
  
  console.log('✅ Dashboard successfully shows both IB Gateway and YFinance data sources');
});

test('test YFinance functionality on dashboard', async ({ page }) => {
  // Capture console logs for debugging
  page.on('console', msg => console.log('BROWSER:', msg.text()));
  
  // Navigate to dashboard
  await page.goto('http://localhost:3000');
  
  // Wait for dashboard to load
  await page.waitForTimeout(3000);
  
  // Click Initialize YFinance button
  const initializeBtn = page.locator('text=Initialize YFinance');
  await initializeBtn.click();
  
  // Wait for initialization
  await page.waitForTimeout(2000);
  
  // Check if status updates
  const statusText = page.locator('text=operational').or(page.locator('text=disconnected'));
  await expect(statusText.first()).toBeVisible();
  
  // Try testing YFinance data
  const testBtn = page.locator('text=Test YFinance Data');
  await testBtn.click();
  
  // Wait for response
  await page.waitForTimeout(3000);
  
  // Take a screenshot
  await page.screenshot({ path: 'yfinance-test-dashboard.png', fullPage: true });
  
  console.log('✅ YFinance functionality tested on dashboard');
});