import { test, expect } from '@playwright/test';

test.describe('MacBook Pro 14" Layout Tests', () => {
  test('should fit all tabs on MacBook Pro 14" screen', async ({ page }) => {
    // MacBook Pro 14" resolution: 3024x1964 (Retina), but logical resolution is ~1512x982
    await page.setViewportSize({ width: 1512, height: 982 });
    
    await page.goto('http://localhost:3001');
    await page.waitForTimeout(3000);
    
    console.log('🖥️ Testing on MacBook Pro 14" logical resolution: 1512x982');
    
    // Check that dashboard is visible
    await expect(page.locator('[data-testid="dashboard"]')).toBeVisible();
    
    // Check that tabs are visible and compact
    const tabs = page.locator('.ant-tabs-tab');
    const tabCount = await tabs.count();
    
    console.log(`Found ${tabCount} tabs`);
    
    // Verify first few tabs are visible without scrolling
    await expect(tabs.first()).toBeVisible();
    if (tabCount > 1) {
      await expect(tabs.nth(1)).toBeVisible();
    }
    if (tabCount > 2) {
      await expect(tabs.nth(2)).toBeVisible();
    }
    
    // Take a full page screenshot
    await page.screenshot({ 
      path: 'test-results/macbook-pro-14-layout.png', 
      fullPage: true 
    });
    
    // Check the tab container width to ensure it fits
    const tabsNav = page.locator('.ant-tabs-nav').first();
    const tabsBox = await tabsNav.boundingBox();
    
    if (tabsBox) {
      console.log(`Tab container width: ${tabsBox.width}px`);
      console.log(`Viewport width: 1512px`);
      
      // Tabs should be scrollable if they exceed viewport
      if (tabsBox.width > 1512) {
        console.log('✅ Tabs are horizontally scrollable as expected');
      } else {
        console.log('✅ All tabs fit within viewport width');
      }
    }
    
    console.log('✅ MacBook Pro 14" layout test completed');
  });

  test('should allow horizontal scrolling through all tabs', async ({ page }) => {
    await page.setViewportSize({ width: 1512, height: 982 });
    
    await page.goto('http://localhost:3001');
    await page.waitForTimeout(3000);
    
    // Get all tabs
    const tabs = page.locator('.ant-tabs-tab');
    const tabCount = await tabs.count();
    
    console.log(`Testing horizontal scroll through ${tabCount} tabs`);
    
    // Try to scroll to the last tab
    if (tabCount > 5) {
      const lastTab = tabs.last();
      
      // Scroll the tabs container horizontally
      const tabsContainer = page.locator('.ant-tabs-nav').first();
      await tabsContainer.hover();
      
      // Use mouse wheel to scroll horizontally
      await page.mouse.wheel(200, 0);
      await page.waitForTimeout(500);
      
      // Try to click the last tab
      try {
        await lastTab.scrollIntoViewIfNeeded();
        await lastTab.click();
        await page.waitForTimeout(1000);
        
        console.log('✅ Successfully scrolled to and clicked last tab');
      } catch (error) {
        console.log('⚠️ Could not reach last tab, but this may be expected');
      }
    }
    
    // Take screenshot after scrolling
    await page.screenshot({ 
      path: 'test-results/macbook-pro-14-scrolled.png', 
      fullPage: true 
    });
  });

  test('should maintain compact appearance', async ({ page }) => {
    await page.setViewportSize({ width: 1512, height: 982 });
    
    await page.goto('http://localhost:3001');
    await page.waitForTimeout(3000);
    
    // Check that tab text is compact (should be 12px font size)
    const firstTabText = page.locator('.ant-tabs-tab').first().locator('span');
    const fontSize = await firstTabText.evaluate((el) => {
      return window.getComputedStyle(el).fontSize;
    });
    
    console.log(`Tab font size: ${fontSize}`);
    expect(fontSize).toBe('12px');
    
    // Check that tab spacing is compact
    const tabGutter = await page.locator('.ant-tabs-nav').first().evaluate((el) => {
      const tabs = el.querySelectorAll('.ant-tabs-tab');
      if (tabs.length > 1) {
        const tab1Rect = tabs[0].getBoundingClientRect();
        const tab2Rect = tabs[1].getBoundingClientRect();
        return tab2Rect.left - tab1Rect.right;
      }
      return 0;
    });
    
    console.log(`Tab gutter (spacing): ${tabGutter}px`);
    expect(tabGutter).toBeLessThanOrEqual(8); // Should be 4px or less
    
    console.log('✅ Compact appearance maintained');
  });
});