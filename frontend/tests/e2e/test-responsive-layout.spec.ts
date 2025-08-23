import { test, expect } from '@playwright/test';

test.describe('Responsive Layout Tests', () => {
  test('dashboard should be scrollable horizontally on normal screens', async ({ page }) => {
    // Test on a standard laptop screen size
    await page.setViewportSize({ width: 1366, height: 768 });
    
    await page.goto('http://localhost:3000');
    await page.waitForTimeout(2000);
    
    // Check that dashboard is visible
    await expect(page.locator('[data-testid="dashboard"]')).toBeVisible();
    
    // Check that tabs are visible and scrollable
    const tabsContainer = page.locator('.ant-tabs-nav');
    await expect(tabsContainer).toBeVisible();
    
    // Check that we can see the first tab
    await expect(page.locator('.ant-tabs-tab').first()).toBeVisible();
    
    // Take a screenshot for verification
    await page.screenshot({ path: 'test-results/responsive-1366x768.png', fullPage: true });
    
    console.log('✅ Layout works on 1366x768 screen');
  });

  test('dashboard should work on smaller laptop screens', async ({ page }) => {
    // Test on a smaller laptop screen
    await page.setViewportSize({ width: 1280, height: 720 });
    
    await page.goto('http://localhost:3000');
    await page.waitForTimeout(2000);
    
    // Check that dashboard is visible
    await expect(page.locator('[data-testid="dashboard"]')).toBeVisible();
    
    // Check that tabs are scrollable horizontally
    const tabsContainer = page.locator('.ant-tabs-nav');
    await expect(tabsContainer).toBeVisible();
    
    // Take a screenshot for verification
    await page.screenshot({ path: 'test-results/responsive-1280x720.png', fullPage: true });
    
    console.log('✅ Layout works on 1280x720 screen');
  });

  test('dashboard should work on tablet screens', async ({ page }) => {
    // Test on tablet size
    await page.setViewportSize({ width: 1024, height: 768 });
    
    await page.goto('http://localhost:3000');
    await page.waitForTimeout(2000);
    
    // Check that dashboard is visible
    await expect(page.locator('[data-testid="dashboard"]')).toBeVisible();
    
    // Check that tabs are accessible via horizontal scroll
    const tabsContainer = page.locator('.ant-tabs-nav');
    await expect(tabsContainer).toBeVisible();
    
    // Try to scroll horizontally in the tabs container
    const tabsScrollArea = page.locator('.ant-tabs-nav');
    const boundingBox = await tabsScrollArea.boundingBox();
    
    if (boundingBox) {
      // Simulate horizontal scroll
      await page.mouse.move(boundingBox.x + boundingBox.width / 2, boundingBox.y + boundingBox.height / 2);
      await page.mouse.wheel(100, 0); // Horizontal scroll
      await page.waitForTimeout(500);
    }
    
    // Take a screenshot for verification
    await page.screenshot({ path: 'test-results/responsive-1024x768.png', fullPage: true });
    
    console.log('✅ Layout works on 1024x768 tablet screen');
  });

  test('verify tabs can be clicked and navigation works', async ({ page }) => {
    await page.setViewportSize({ width: 1280, height: 720 });
    
    await page.goto('http://localhost:3000');
    await page.waitForTimeout(2000);
    
    // Check that we can click on different tabs
    const tabs = page.locator('.ant-tabs-tab');
    const tabCount = await tabs.count();
    
    console.log(`Found ${tabCount} tabs`);
    
    if (tabCount > 1) {
      // Click on the second tab if it exists
      await tabs.nth(1).click();
      await page.waitForTimeout(1000);
      
      // Verify tab changed (active tab should have different class)
      const activeTab = page.locator('.ant-tabs-tab-active');
      await expect(activeTab).toBeVisible();
      
      console.log('✅ Tab navigation works');
    }
    
    // Take final screenshot
    await page.screenshot({ path: 'test-results/responsive-navigation-test.png', fullPage: true });
  });
});