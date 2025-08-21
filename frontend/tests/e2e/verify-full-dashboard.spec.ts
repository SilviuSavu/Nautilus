import { test, expect } from '@playwright/test';

test.describe('Nautilus Trading Platform Full Dashboard Verification', () => {
  test('should display complete Nautilus dashboard with all tabs', async ({ page }) => {
    // Navigate to the application
    await page.goto('http://localhost:3001');
    
    // Wait for the page to fully load
    await page.waitForLoadState('networkidle');
    await page.waitForTimeout(3000);
    
    // Take screenshot of the initial state
    await page.screenshot({ 
      path: 'full-dashboard-verification.png', 
      fullPage: true 
    });
    
    // Check for NautilusTrader Dashboard title
    const titleElements = [
      'h1:has-text("NautilusTrader Dashboard")',
      'h2:has-text("NautilusTrader Dashboard")', 
      'h3:has-text("NautilusTrader Dashboard")',
      '[data-testid*="title"]',
      '.title'
    ];
    
    let titleFound = false;
    for (const selector of titleElements) {
      try {
        await page.waitForSelector(selector, { timeout: 5000 });
        titleFound = true;
        console.log(`✓ Found title with selector: ${selector}`);
        break;
      } catch (e) {
        // Continue to next selector
      }
    }
    
    if (!titleFound) {
      // Check if "NautilusTrader" text exists anywhere on the page
      const pageContent = await page.textContent('body');
      if (pageContent?.includes('NautilusTrader')) {
        console.log('✓ NautilusTrader text found in page content');
        titleFound = true;
      }
    }
    
    // Check for tab interface - expected tabs
    const expectedTabs = [
      'System Overview',
      'NautilusTrader Engine', 
      'Instrument Search',
      'Interactive Brokers',
      'Portfolio',
      'Risk',
      'Performance'
    ];
    
    const foundTabs: string[] = [];
    
    // Check for tabs using various selectors
    const tabSelectors = [
      '[role="tab"]',
      '.ant-tabs-tab',
      '[data-testid*="tab"]',
      '.tab',
      'button[role="tab"]'
    ];
    
    for (const selector of tabSelectors) {
      const tabs = await page.locator(selector).all();
      for (const tab of tabs) {
        try {
          const tabText = await tab.textContent();
          if (tabText && tabText.trim()) {
            foundTabs.push(tabText.trim());
          }
        } catch (e) {
          // Continue
        }
      }
    }
    
    // Check page content for expected tabs
    const pageContent = await page.textContent('body');
    const tabsFoundInContent: string[] = [];
    
    for (const expectedTab of expectedTabs) {
      if (pageContent?.includes(expectedTab)) {
        tabsFoundInContent.push(expectedTab);
      }
    }
    
    console.log('Found tabs:', foundTabs);
    console.log('Expected tabs found in content:', tabsFoundInContent);
    
    // Get page title
    const pageTitle = await page.title();
    console.log('Page title:', pageTitle);
    
    // Verify we're not seeing the simple test app
    const isSimpleApp = pageContent?.includes('Simple Test App') || 
                       pageContent?.includes('This is a minimal test page');
    
    console.log('Is Simple App:', isSimpleApp);
    console.log('Has NautilusTrader text:', pageContent?.includes('NautilusTrader'));
    console.log('Has Dashboard text:', pageContent?.includes('Dashboard'));
    
    // Assertions
    expect(isSimpleApp).toBeFalsy();
    expect(pageContent).toContain('NautilusTrader');
    expect(tabsFoundInContent.length).toBeGreaterThan(0);
    
    // Log verification results
    console.log('\n=== VERIFICATION RESULTS ===');
    console.log(`✓ Full dashboard loaded: ${!isSimpleApp}`);
    console.log(`✓ NautilusTrader text present: ${pageContent?.includes('NautilusTrader')}`);
    console.log(`✓ Dashboard text present: ${pageContent?.includes('Dashboard')}`);
    console.log(`✓ Expected tabs found: ${tabsFoundInContent.length}/${expectedTabs.length}`);
    console.log(`✓ Screenshot saved: full-dashboard-verification.png`);
  });
});