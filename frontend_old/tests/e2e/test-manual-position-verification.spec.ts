import { test, expect } from '@playwright/test';

test.describe('Position Monitoring Manual Verification', () => {
  test('should verify position monitoring components are accessible', async ({ page }) => {
    // Enable console logging
    page.on('console', msg => console.log('BROWSER:', msg.text()));
    
    // Navigate to the application
    await page.goto('http://localhost:3000');
    await page.waitForTimeout(3000);
    
    // Take initial screenshot
    await page.screenshot({ path: 'app-initial-load.png', fullPage: true });
    
    // Check for tabs and click on Interactive Brokers tab
    const ibTab = page.locator('.ant-tabs-tab').filter({ hasText: /Interactive Brokers|IB/i });
    const ibTabExists = await ibTab.count();
    console.log('IB Tab exists:', ibTabExists > 0);
    
    if (ibTabExists > 0) {
      await ibTab.click();
      await page.waitForTimeout(2000);
      await page.screenshot({ path: 'ib-tab-clicked.png', fullPage: true });
      
      // Look for order buttons or forms
      const orderButtons = await page.locator('button').filter({ hasText: /buy|sell|place.*order/i }).count();
      console.log('Order buttons found:', orderButtons);
      
      // Look for account/position information
      const accountInfo = await page.locator('text=/account|balance|position|portfolio/i').count();
      console.log('Account/Position info elements:', accountInfo);
      
      // Check for any tables (might contain orders or positions)
      const tables = await page.locator('table, .ant-table').count();
      console.log('Tables found:', tables);
      
      if (tables > 0) {
        await page.screenshot({ path: 'tables-found.png', fullPage: true });
      }
    }
    
    // Test simple order placement through UI if available
    const placeOrderBtn = page.locator('button').filter({ hasText: /place.*order|buy|sell/i }).first();
    const placeOrderExists = await placeOrderBtn.count();
    
    if (placeOrderExists > 0) {
      console.log('Found order placement button, testing...');
      await placeOrderBtn.click();
      await page.waitForTimeout(1000);
      await page.screenshot({ path: 'order-form-opened.png', fullPage: true });
      
      // Look for symbol input
      const symbolInput = page.locator('input[placeholder*="symbol"], input[placeholder*="instrument"]');
      const symbolInputExists = await symbolInput.count();
      
      if (symbolInputExists > 0) {
        await symbolInput.fill('AAPL');
        await page.waitForTimeout(500);
        await page.screenshot({ path: 'symbol-entered.png', fullPage: true });
      }
    }
    
    // Final screenshot
    await page.screenshot({ path: 'position-monitoring-final.png', fullPage: true });
  });
});