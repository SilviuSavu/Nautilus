import { test, expect } from '@playwright/test';

test.describe('Position and Account Monitoring', () => {
  test.beforeEach(async ({ page }) => {
    // Enable console logging
    page.on('console', msg => console.log('BROWSER:', msg.text()));
    
    // Navigate to the application
    await page.goto('http://localhost:3000');
    await page.waitForTimeout(2000); // Allow app to load
  });

  test('should display position monitoring components', async ({ page }) => {
    // Check if there are any position-related components or tabs
    const positionElements = await page.locator('text=/position/i').count();
    console.log('Position elements found:', positionElements);
    
    // Take screenshot for visual verification
    await page.screenshot({ path: 'position-monitoring-test.png', fullPage: true });
    
    // Check for account monitoring components
    const accountElements = await page.locator('[data-testid*="account"], [class*="account"], text=/account/i').count();
    console.log('Account elements found:', accountElements);
    
    // Try to find P&L related components
    const pnlElements = await page.locator('text=/P&L|PnL|profit|loss/i').count();
    console.log('P&L elements found:', pnlElements);
    
    // Check if order monitoring is available
    const orderElements = await page.locator('text=/order/i').count();
    console.log('Order elements found:', orderElements);
  });

  test('should test order placement functionality', async ({ page }) => {
    // Look for any order placement forms or buttons
    const orderButtons = await page.locator('button').filter({ hasText: /buy|sell|order|place/i }).count();
    console.log('Order buttons found:', orderButtons);
    
    if (orderButtons > 0) {
      // Click on an order button if found
      const firstOrderButton = page.locator('button').filter({ hasText: /buy|sell|order|place/i }).first();
      await firstOrderButton.click();
      await page.waitForTimeout(1000);
      
      // Take screenshot after clicking
      await page.screenshot({ path: 'order-placement-test.png' });
    }
    
    // Check for any order forms or modals
    const orderForms = await page.locator('form, .modal, .drawer').count();
    console.log('Order forms/modals found:', orderForms);
  });

  test('should verify API connectivity for positions and orders', async ({ page }) => {
    // Check if we can access the IB Dashboard or trading interface
    const tabs = page.locator('.ant-tabs-tab');
    const tabCount = await tabs.count();
    console.log('Tabs found:', tabCount);
    
    // If there are tabs, try clicking on trading/IB related ones
    for (let i = 0; i < tabCount; i++) {
      const tabText = await tabs.nth(i).textContent();
      console.log(`Tab ${i}:`, tabText);
      
      if (tabText?.toLowerCase().includes('ib') || 
          tabText?.toLowerCase().includes('trading') ||
          tabText?.toLowerCase().includes('order')) {
        await tabs.nth(i).click();
        await page.waitForTimeout(2000);
        
        // Take screenshot of the trading interface
        await page.screenshot({ 
          path: `trading-interface-${i}.png`,
          fullPage: true 
        });
        
        // Check for any API calls or errors in console
        const errors = await page.locator('.error, .ant-alert-error').count();
        console.log('Errors found:', errors);
        
        break;
      }
    }
  });

  test('should check for order book and market data', async ({ page }) => {
    // Look for order book or market data components
    const orderBookElements = await page.locator('text=/order book|orderbook|market data|bid|ask/i').count();
    console.log('Order book elements found:', orderBookElements);
    
    // Check for price data
    const priceElements = await page.locator('text=/\\$|price|bid|ask/').count();
    console.log('Price elements found:', priceElements);
    
    // Look for instrument search
    const searchElements = await page.locator('input[placeholder*="search"], input[placeholder*="symbol"]').count();
    console.log('Search elements found:', searchElements);
    
    if (searchElements > 0) {
      // Try searching for AAPL
      const searchInput = page.locator('input[placeholder*="search"], input[placeholder*="symbol"]').first();
      await searchInput.fill('AAPL');
      await page.waitForTimeout(2000);
      
      // Take screenshot of search results
      await page.screenshot({ path: 'search-results-test.png' });
    }
  });
});