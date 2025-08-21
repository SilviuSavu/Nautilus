import { test, expect } from '@playwright/test';

test('Simple Risk Tab Test', async ({ page }) => {
  console.log('🔍 Quick risk tab test...');
  
  // Capture all browser output
  page.on('console', msg => console.log('BROWSER:', msg.text()));
  page.on('pageerror', error => console.log('PAGE ERROR:', error.message));
  
  // Navigate to dashboard
  await page.goto('http://localhost:3000');
  console.log('✅ Page loaded');
  
  // Wait for page to load
  await page.waitForTimeout(5000);
  
  // Find and click Risk Management tab
  const riskTab = page.locator('text=Risk Management');
  if (await riskTab.isVisible()) {
    console.log('✅ Risk Management tab is visible');
    await riskTab.click();
    console.log('✅ Clicked Risk Management tab');
    
    // Wait a bit for content to load
    await page.waitForTimeout(5000);
    
    // Take screenshot
    await page.screenshot({ path: 'simple-risk-test.png', fullPage: true });
    console.log('📸 Screenshot saved');
  } else {
    console.log('❌ Risk Management tab not found');
  }
});