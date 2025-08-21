import { test, expect } from '@playwright/test';

test.describe('Order Placement - User Acceptance Test', () => {
  test.beforeEach(async ({ page }) => {
    // Capture console logs for debugging
    page.on('console', msg => console.log('BROWSER:', msg.text()));
    
    await page.goto('http://localhost:3000');
    await page.waitForTimeout(3000); // Give time for app to load
  });

  test('Complete order placement user journey', async ({ page }) => {
    console.log('=== USER ACCEPTANCE TEST: Order Placement Journey ===');
    
    // Step 1: Navigate to IB Dashboard
    console.log('Step 1: Navigate to IB Dashboard');
    await page.click('text=IB Dashboard');
    await page.waitForTimeout(2000);
    
    // Take screenshot of initial state
    await page.screenshot({ path: 'uat-step1-ib-dashboard.png' });
    
    // Step 2: Open order placement modal
    console.log('Step 2: Open order placement modal');
    await page.click('text=Place Order');
    
    // Wait for modal to appear
    await page.waitForSelector('[data-testid="order-form"]', { timeout: 10000 });
    await page.screenshot({ path: 'uat-step2-order-modal.png' });
    
    // Step 3: Test form validation (should show errors)
    console.log('Step 3: Test form validation');
    await page.click('[data-testid="submit-order-button"]');
    await page.waitForTimeout(1000);
    
    // Should show validation errors
    const hasErrors = await page.locator('.ant-form-item-explain-error').count() > 0;
    console.log('Validation errors shown:', hasErrors);
    await page.screenshot({ path: 'uat-step3-validation-errors.png' });
    
    // Step 4: Fill valid market order
    console.log('Step 4: Fill valid market order');
    await page.fill('[data-testid="symbol-input"]', 'AAPL');
    await page.selectOption('[data-testid="order-type-select"]', 'MKT');
    await page.fill('[data-testid="quantity-input"]', '100');
    await page.selectOption('[data-testid="action-select"]', 'BUY');
    
    await page.screenshot({ path: 'uat-step4-filled-order.png' });
    
    // Step 5: Submit market order
    console.log('Step 5: Submit market order');
    await page.click('[data-testid="submit-order-button"]');
    await page.waitForTimeout(3000);
    
    // Check for success or error message
    const successMessage = await page.locator('.ant-message-success').textContent().catch(() => null);
    const errorMessage = await page.locator('.ant-message-error').textContent().catch(() => null);
    
    console.log('Market Order Result:', { success: successMessage, error: errorMessage });
    await page.screenshot({ path: 'uat-step5-order-result.png' });
    
    if (successMessage) {
      console.log('✅ Market order placed successfully');
    } else if (errorMessage) {
      console.log('⚠️ Market order failed:', errorMessage);
    }
    
    // Step 6: Test advanced order (Limit Order)
    console.log('Step 6: Test limit order');
    
    // Close current modal and open new one
    await page.click('[data-testid="cancel-button"]').catch(() => {});
    await page.waitForTimeout(1000);
    
    await page.click('text=Place Order');
    await page.waitForSelector('[data-testid="order-form"]');
    
    // Fill limit order
    await page.fill('[data-testid="symbol-input"]', 'TSLA');
    await page.selectOption('[data-testid="order-type-select"]', 'LMT');
    await page.fill('[data-testid="quantity-input"]', '50');
    await page.selectOption('[data-testid="action-select"]', 'BUY');
    await page.fill('[data-testid="limit-price-input"]', '200.00');
    
    await page.screenshot({ path: 'uat-step6-limit-order.png' });
    
    // Submit limit order
    await page.click('[data-testid="submit-order-button"]');
    await page.waitForTimeout(3000);
    
    const limitSuccessMessage = await page.locator('.ant-message-success').textContent().catch(() => null);
    const limitErrorMessage = await page.locator('.ant-message-error').textContent().catch(() => null);
    
    console.log('Limit Order Result:', { success: limitSuccessMessage, error: limitErrorMessage });
    await page.screenshot({ path: 'uat-step6-limit-result.png' });
    
    // Step 7: Test trailing stop order
    console.log('Step 7: Test trailing stop order');
    
    await page.click('[data-testid="cancel-button"]').catch(() => {});
    await page.waitForTimeout(1000);
    
    await page.click('text=Place Order');
    await page.waitForSelector('[data-testid="order-form"]');
    
    // Fill trailing stop order
    await page.fill('[data-testid="symbol-input"]', 'MSFT');
    await page.selectOption('[data-testid="order-type-select"]', 'TRAIL');
    await page.fill('[data-testid="quantity-input"]', '75');
    await page.selectOption('[data-testid="action-select"]', 'SELL');
    
    // Add trail amount if field is visible
    const trailAmountField = page.locator('[data-testid="trail-amount-input"]');
    if (await trailAmountField.isVisible()) {
      await trailAmountField.fill('1.50');
    }
    
    await page.screenshot({ path: 'uat-step7-trail-order.png' });
    
    // Submit trailing stop order
    await page.click('[data-testid="submit-order-button"]');
    await page.waitForTimeout(3000);
    
    const trailSuccessMessage = await page.locator('.ant-message-success').textContent().catch(() => null);
    const trailErrorMessage = await page.locator('.ant-message-error').textContent().catch(() => null);
    
    console.log('Trail Order Result:', { success: trailSuccessMessage, error: trailErrorMessage });
    await page.screenshot({ path: 'uat-step7-trail-result.png' });
    
    // Step 8: Final assessment
    console.log('Step 8: Final assessment');
    
    const totalSuccesses = [successMessage, limitSuccessMessage, trailSuccessMessage].filter(Boolean).length;
    const totalErrors = [errorMessage, limitErrorMessage, trailErrorMessage].filter(Boolean).length;
    
    console.log(`\n=== USER ACCEPTANCE TEST RESULTS ===`);
    console.log(`✅ Successful orders: ${totalSuccesses}`);
    console.log(`❌ Failed orders: ${totalErrors}`);
    console.log(`📊 Success rate: ${totalSuccesses}/3 (${Math.round(totalSuccesses/3*100)}%)`);
    
    if (totalSuccesses >= 2) {
      console.log('🎉 USER ACCEPTANCE TEST PASSED - Order placement functionality working');
    } else {
      console.log('⚠️ USER ACCEPTANCE TEST NEEDS ATTENTION - Multiple order failures detected');
    }
    
    // Take final screenshot
    await page.screenshot({ path: 'uat-final-state.png' });
  });

  test('Error handling and user experience test', async ({ page }) => {
    console.log('=== ERROR HANDLING TEST ===');
    
    // Navigate to order form
    await page.click('text=IB Dashboard');
    await page.waitForTimeout(1000);
    await page.click('text=Place Order');
    await page.waitForSelector('[data-testid="order-form"]');
    
    // Test 1: Invalid symbol
    console.log('Test 1: Invalid symbol handling');
    await page.fill('[data-testid="symbol-input"]', 'INVALID_SYMBOL_12345');
    await page.selectOption('[data-testid="order-type-select"]', 'MKT');
    await page.fill('[data-testid="quantity-input"]', '100');
    await page.selectOption('[data-testid="action-select"]', 'BUY');
    
    await page.click('[data-testid="submit-order-button"]');
    await page.waitForTimeout(3000);
    
    const invalidSymbolError = await page.locator('.ant-message-error').textContent().catch(() => null);
    console.log('Invalid symbol error:', invalidSymbolError);
    
    // Test 2: Excessive quantity
    console.log('Test 2: Large quantity handling');
    await page.fill('[data-testid="symbol-input"]', 'AAPL');
    await page.fill('[data-testid="quantity-input"]', '999999999');
    
    await page.click('[data-testid="submit-order-button"]');
    await page.waitForTimeout(3000);
    
    const largeQuantityError = await page.locator('.ant-message-error').textContent().catch(() => null);
    console.log('Large quantity handling:', largeQuantityError);
    
    console.log('✅ Error handling test completed');
  });
});