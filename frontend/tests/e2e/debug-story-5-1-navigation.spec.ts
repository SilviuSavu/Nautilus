/**
 * Debug Story 5.1 Frontend Navigation
 */
import { test, expect } from '@playwright/test';

test.describe('Debug Story 5.1 Navigation', () => {
  test('Check what tabs are actually available', async ({ page }) => {
    console.log('🔍 Debugging frontend navigation for Story 5.1');
    
    page.on('console', msg => console.log('BROWSER:', msg.text()));
    page.on('pageerror', error => console.log('PAGE ERROR:', error.message));
    
    await page.goto('http://localhost:3000');
    await page.waitForTimeout(5000);
    
    // Take screenshot of initial state
    await page.screenshot({ path: 'debug-initial-state.png' });
    
    // List all visible text content
    const pageText = await page.textContent('body');
    console.log('📄 Page content preview:', pageText?.substring(0, 500));
    
    // Look for all tab elements
    const tabs = await page.locator('[role="tab"], .ant-tabs-tab').allTextContents();
    console.log('🏷️ Available tabs:', tabs);
    
    // Look for Performance-related elements
    const performanceElements = await page.locator('text*=Performance').count();
    console.log('📊 Performance elements found:', performanceElements);
    
    // Look for any navigation elements
    const navElements = await page.locator('nav, .navigation, .menu, .tabs').count();
    console.log('🧭 Navigation elements found:', navElements);
    
    // Check if we need to navigate to a different route
    const currentUrl = page.url();
    console.log('🌐 Current URL:', currentUrl);
    
    // Try to find the Performance tab or similar
    if (performanceElements === 0) {
      console.log('❌ No Performance tab found in main navigation');
      console.log('🔍 Looking for other ways to access Performance...');
      
      // Look for dashboard-related links
      const dashboardLinks = await page.locator('text*=Dashboard').count();
      console.log('📋 Dashboard links found:', dashboardLinks);
      
      // Check if Performance is in a different section
      const menuItems = await page.locator('.ant-menu-item, .menu-item, .nav-item').allTextContents();
      console.log('📋 Menu items:', menuItems);
    }
    
    await page.screenshot({ path: 'debug-navigation-analysis.png' });
  });

  test('Try to access Performance Dashboard directly', async ({ page }) => {
    console.log('🔍 Attempting direct access to Performance Dashboard');
    
    page.on('console', msg => console.log('BROWSER:', msg.text()));
    
    await page.goto('http://localhost:3000');
    await page.waitForTimeout(3000);
    
    // Try clicking on different tabs to see if Performance is nested
    const allTabs = await page.locator('[role="tab"], .ant-tabs-tab').all();
    
    for (let i = 0; i < allTabs.length; i++) {
      const tab = allTabs[i];
      const tabText = await tab.textContent();
      console.log(`🔍 Checking tab ${i}: "${tabText}"`);
      
      await tab.click();
      await page.waitForTimeout(2000);
      
      // Check if Performance Dashboard appears after clicking this tab
      const perfDashboard = await page.locator('.performance-dashboard').count();
      if (perfDashboard > 0) {
        console.log(`✅ Found Performance Dashboard under tab: "${tabText}"`);
        await page.screenshot({ path: `found-performance-under-${tabText?.replace(/\s+/g, '-')}.png` });
        break;
      }
      
      // Check if there are sub-tabs
      const subTabs = await page.locator('[role="tab"], .ant-tabs-tab').allTextContents();
      console.log(`📋 Sub-tabs after clicking "${tabText}":`, subTabs.filter(t => !tabs.includes(t)));
      
      await page.screenshot({ path: `debug-tab-${i}-${tabText?.replace(/\s+/g, '-')}.png` });
    }
    
    console.log('🏁 Navigation debug complete');
  });
});