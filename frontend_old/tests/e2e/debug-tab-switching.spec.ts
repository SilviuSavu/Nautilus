import { test, expect } from '@playwright/test';

test('Debug Tab Switching', async ({ page }) => {
  console.log('🔍 Debugging tab switching...');
  
  // Capture all browser output
  page.on('console', msg => console.log('BROWSER:', msg.text()));
  page.on('pageerror', error => console.log('PAGE ERROR:', error.message));
  
  // Navigate to dashboard
  await page.goto('http://localhost:3000');
  console.log('✅ Page loaded');
  
  // Wait for page to load
  await page.waitForTimeout(3000);
  
  // Check all available tabs
  const allTabs = await page.locator('[role="tab"]').allTextContents();
  console.log('📋 Available tabs:', allTabs);
  
  // Try to click Performance Monitoring tab first to test tab switching
  const performanceTab = page.locator('text=Performance Monitoring');
  if (await performanceTab.isVisible()) {
    console.log('🔄 Clicking Performance Monitoring tab...');
    await performanceTab.click();
    await page.waitForTimeout(2000);
    await page.screenshot({ path: 'performance-tab.png' });
  }
  
  // Now try Risk Management tab
  const riskTab = page.locator('text=Risk Management');
  if (await riskTab.isVisible()) {
    console.log('🔄 Clicking Risk Management tab...');
    await riskTab.click();
    await page.waitForTimeout(3000);
    
    // Check if tab content changed
    const tabContent = await page.locator('[role="tabpanel"]').innerHTML();
    console.log('📄 Tab content length:', tabContent.length);
    
    await page.screenshot({ path: 'risk-tab-clicked.png' });
    console.log('📸 Screenshots saved');
  }
  
  // Try clicking the tab multiple times
  console.log('🔄 Trying multiple clicks...');
  for (let i = 0; i < 3; i++) {
    await riskTab.click();
    await page.waitForTimeout(1000);
  }
  
  await page.screenshot({ path: 'after-multiple-clicks.png' });
});