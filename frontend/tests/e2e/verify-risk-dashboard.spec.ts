import { test, expect } from '@playwright/test';

test('Risk Management Dashboard Integration Test', async ({ page }) => {
  console.log('🔍 Testing Risk Management Dashboard visibility...');
  
  // Capture console errors and logs
  page.on('console', msg => console.log('BROWSER:', msg.text()));
  page.on('pageerror', error => console.log('PAGE ERROR:', error.message));
  
  // Navigate to the dashboard
  await page.goto('http://localhost:3000');
  await page.waitForTimeout(3000);
  
  // Check if the Risk Management tab is present
  const riskTab = page.locator('text=Risk Management');
  
  try {
    await expect(riskTab).toBeVisible({ timeout: 10000 });
    console.log('✅ Risk Management tab is visible');
    
    // Click on the Risk Management tab
    await riskTab.click();
    await page.waitForTimeout(2000);
    
    // Check for risk dashboard components (basic structure should always be present)
    const riskOverviewCard = page.locator('text=Portfolio Risk Overview');
    const refreshButton = page.locator('button[aria-label*="Refresh"], button:has([data-icon="reload"])');
    
    // The dashboard should either show data or an error message, but basic structure should be there
    await expect(riskOverviewCard.or(page.locator('text=Error Loading Risk Data'))).toBeVisible({ timeout: 10000 });
    
    // Check that the basic layout is rendered
    const tabContent = page.locator('[role="tabpanel"]');
    await expect(tabContent).toBeVisible({ timeout: 5000 });
    
    console.log('✅ Risk Dashboard components are visible');
    
    // Take a screenshot
    await page.screenshot({ path: 'risk-dashboard-integration.png', fullPage: true });
    console.log('📸 Screenshot saved: risk-dashboard-integration.png');
    
    console.log('🎉 Risk Management Dashboard successfully integrated!');
    
  } catch (error) {
    console.log('❌ Risk Management tab not found or not functional');
    console.log('Available tabs:');
    
    // List available tabs for debugging
    const tabs = await page.locator('[role="tab"]').allTextContents();
    tabs.forEach((tab, index) => {
      console.log(`  ${index + 1}. ${tab}`);
    });
    
    // Take a screenshot for debugging
    await page.screenshot({ path: 'dashboard-debug.png', fullPage: true });
    console.log('📸 Debug screenshot saved: dashboard-debug.png');
    
    throw error;
  }
});