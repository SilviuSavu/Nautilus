import { test, expect } from '@playwright/test';

test.describe('Order Placement IB Gateway Integration Debugging', () => {
  test.beforeEach(async ({ page }) => {
    // Capture console logs and network traffic
    page.on('console', msg => console.log('BROWSER:', msg.text()));
    page.on('response', response => {
      if (response.url().includes('/api/v1/ib/orders/place')) {
        console.log(`API Response: ${response.status()} ${response.url()}`);
      }
    });
    
    await page.goto('http://localhost:3000');
    await page.waitForLoadState('networkidle');
  });

  test('Debug IB Gateway order attribute errors', async ({ page }) => {
    // Navigate to IB Dashboard
    await page.click('text=IB Dashboard');
    await page.waitForTimeout(2000);

    // Open order placement modal
    await page.click('text=Place Order');
    await page.waitForSelector('[data-testid="order-form"]');
    
    // Test basic market order first (should work)
    await page.fill('[data-testid="symbol-input"]', 'AAPL');
    await page.selectOption('[data-testid="order-type-select"]', 'MKT');
    await page.fill('[data-testid="quantity-input"]', '100');
    await page.selectOption('[data-testid="action-select"]', 'BUY');
    
    // Submit basic order and check for errors
    await page.click('[data-testid="submit-order-button"]');
    await page.waitForTimeout(3000);
    
    // Check for success or error messages
    const errorMessage = await page.locator('.ant-message-error').textContent().catch(() => null);
    const successMessage = await page.locator('.ant-message-success').textContent().catch(() => null);
    
    console.log('Basic Market Order Result:', { errorMessage, successMessage });
    
    // Take screenshot of result
    await page.screenshot({ path: 'order-placement-basic-test.png' });
    
    // Reset form
    await page.click('[data-testid="cancel-button"]');
    await page.waitForTimeout(1000);
    
    // Test advanced order type (TRAIL) - this should expose the error
    await page.click('text=Place Order');
    await page.waitForSelector('[data-testid="order-form"]');
    
    await page.fill('[data-testid="symbol-input"]', 'AAPL');
    await page.selectOption('[data-testid="order-type-select"]', 'TRAIL');
    await page.fill('[data-testid="quantity-input"]', '100');
    await page.selectOption('[data-testid="action-select"]', 'BUY');
    
    // Add trailing stop parameters
    const trailAmountInput = page.locator('[data-testid="trail-amount-input"]');
    if (await trailAmountInput.isVisible()) {
      await trailAmountInput.fill('1.00');
    }
    
    // Submit advanced order and capture detailed error
    await page.click('[data-testid="submit-order-button"]');
    await page.waitForTimeout(5000);
    
    // Check for IB Gateway specific errors
    const advancedErrorMessage = await page.locator('.ant-message-error').textContent().catch(() => null);
    const advancedSuccessMessage = await page.locator('.ant-message-success').textContent().catch(() => null);
    
    console.log('Advanced TRAIL Order Result:', { errorMessage: advancedErrorMessage, successMessage: advancedSuccessMessage });
    
    // Take screenshot of advanced order result
    await page.screenshot({ path: 'order-placement-trail-test.png' });
    
    // Log final status
    if (advancedErrorMessage?.includes('EtradeOnly') || advancedErrorMessage?.includes('Invalid order type')) {
      console.log('✗ CONFIRMED: IB Gateway attribute mapping issue found');
      console.log('Error details:', advancedErrorMessage);
    } else if (advancedSuccessMessage) {
      console.log('✓ Advanced order placed successfully');
    } else {
      console.log('? Unexpected result - need to investigate further');
    }
  });

  test('Test all advanced order types for IB compatibility', async ({ page }) => {
    const orderTypes = ['MKT', 'LMT', 'STP', 'STP_LMT', 'TRAIL'];
    const results: Record<string, any> = {};
    
    for (const orderType of orderTypes) {
      console.log(`Testing order type: ${orderType}`);
      
      // Navigate to order form
      await page.click('text=IB Dashboard');
      await page.waitForTimeout(1000);
      await page.click('text=Place Order');
      await page.waitForSelector('[data-testid="order-form"]');
      
      // Fill basic order details
      await page.fill('[data-testid="symbol-input"]', 'AAPL');
      await page.selectOption('[data-testid="order-type-select"]', orderType);
      await page.fill('[data-testid="quantity-input"]', '10');
      await page.selectOption('[data-testid="action-select"]', 'BUY');
      
      // Add type-specific parameters
      if (orderType === 'LMT') {
        await page.fill('[data-testid="limit-price-input"]', '150.00');
      } else if (orderType === 'STP') {
        await page.fill('[data-testid="stop-price-input"]', '140.00');
      } else if (orderType === 'STP_LMT') {
        await page.fill('[data-testid="limit-price-input"]', '150.00');
        await page.fill('[data-testid="stop-price-input"]', '140.00');
      } else if (orderType === 'TRAIL') {
        const trailAmountInput = page.locator('[data-testid="trail-amount-input"]');
        if (await trailAmountInput.isVisible()) {
          await trailAmountInput.fill('1.00');
        }
      }
      
      // Submit order
      await page.click('[data-testid="submit-order-button"]');
      await page.waitForTimeout(3000);
      
      // Capture result
      const errorMessage = await page.locator('.ant-message-error').textContent().catch(() => null);
      const successMessage = await page.locator('.ant-message-success').textContent().catch(() => null);
      
      results[orderType] = {
        error: errorMessage,
        success: successMessage,
        timestamp: new Date().toISOString()
      };
      
      console.log(`${orderType} Result:`, results[orderType]);
      
      // Cancel/close current order
      await page.click('[data-testid="cancel-button"]').catch(() => {});
      await page.waitForTimeout(1000);
    }
    
    // Log summary
    console.log('\n=== ORDER TYPE COMPATIBILITY SUMMARY ===');
    for (const [type, result] of Object.entries(results)) {
      const status = result.success ? '✓ SUCCESS' : (result.error ? '✗ ERROR' : '? UNKNOWN');
      console.log(`${type}: ${status}`);
      if (result.error) {
        console.log(`  Error: ${result.error}`);
      }
    }
  });
});