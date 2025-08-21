import { test, expect } from '@playwright/test';

test('Force Risk Tab Activation', async ({ page }) => {
  console.log('🔍 Force activating risk tab...');
  
  // Capture all browser output
  page.on('console', msg => console.log('BROWSER:', msg.text()));
  page.on('pageerror', error => console.log('PAGE ERROR:', error.message));
  
  // Navigate to dashboard
  await page.goto('http://localhost:3000');
  await page.waitForTimeout(3000);
  
  // Force the tab to change via JavaScript
  console.log('🔧 Manually triggering tab change to risk...');
  await page.evaluate(() => {
    // Look for any setActiveTab or similar function
    console.log('🔍 Looking for tab management...');
    
    // Try to trigger a manual tab change by finding the React component instance
    const tabElements = document.querySelectorAll('[role="tab"]');
    console.log('Found tabs:', Array.from(tabElements).map(el => el.textContent));
    
    // Find Risk Management tab and try to trigger its click event
    const riskTab = Array.from(tabElements).find(tab => 
      tab.textContent?.includes('Risk Management')
    );
    
    if (riskTab) {
      console.log('📍 Found Risk Management tab, triggering events...');
      riskTab.dispatchEvent(new Event('click', { bubbles: true }));
      riskTab.dispatchEvent(new Event('mousedown', { bubbles: true }));
      riskTab.dispatchEvent(new Event('mouseup', { bubbles: true }));
    }
  });
  
  await page.waitForTimeout(3000);
  
  // Check what content is displayed
  await page.evaluate(() => {
    const panels = document.querySelectorAll('[role="tabpanel"]');
    console.log('Active panels:', panels.length);
    panels.forEach((panel, index) => {
      if (!panel.hasAttribute('aria-hidden') || panel.getAttribute('aria-hidden') === 'false') {
        console.log(`Active panel ${index}:`, panel.id, panel.className);
      }
    });
  });
  
  await page.screenshot({ path: 'force-risk-tab.png', fullPage: true });
  console.log('📸 Screenshot saved');
});