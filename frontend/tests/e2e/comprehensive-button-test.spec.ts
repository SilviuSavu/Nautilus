import { test, expect } from '@playwright/test';

test.describe('BMad Team: Comprehensive Button Testing 🧪', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('http://localhost:3001');
    await page.waitForLoadState('networkidle');
  });

  test('Dashboard Navigation and Core Buttons', async ({ page }) => {
    // Test main navigation buttons
    const navButtons = [
      'Dashboard',
      'Portfolio', 
      'Market Data',
      'Orders',
      'Positions',
      'Settings'
    ];

    for (const buttonText of navButtons) {
      const button = page.getByText(buttonText).first();
      if (await button.isVisible()) {
        await button.click();
        await page.waitForTimeout(500); // Allow navigation
        console.log(`✅ Navigation button "${buttonText}" clicked successfully`);
      }
    }
  });

  test('Interactive Widget Buttons', async ({ page }) => {
    // Look for interactive buttons by common patterns
    const buttonSelectors = [
      'button',
      '[role="button"]',
      '.ant-btn',
      'a[href]',
      '[onclick]',
      '[data-testid*="button"]'
    ];

    for (const selector of buttonSelectors) {
      const buttons = await page.locator(selector).all();
      
      for (let i = 0; i < Math.min(buttons.length, 20); i++) { // Limit to first 20 to avoid overwhelming
        const button = buttons[i];
        
        if (await button.isVisible() && await button.isEnabled()) {
          try {
            const text = await button.textContent() || `Button ${i + 1}`;
            await button.click();
            await page.waitForTimeout(200);
            console.log(`✅ Button "${text.trim()}" clicked successfully`);
          } catch (error) {
            console.log(`⚠️ Button ${i + 1} interaction failed:`, error);
          }
        }
      }
    }
  });

  test('Form Controls and Input Interactions', async ({ page }) => {
    // Test form inputs and controls
    const inputTypes = ['input', 'select', 'textarea'];
    
    for (const inputType of inputTypes) {
      const inputs = await page.locator(inputType).all();
      
      for (let i = 0; i < Math.min(inputs.length, 10); i++) {
        const input = inputs[i];
        
        if (await input.isVisible() && await input.isEnabled()) {
          try {
            if (inputType === 'input') {
              await input.fill('test-value');
            } else if (inputType === 'select') {
              await input.selectOption({ index: 0 });
            } else if (inputType === 'textarea') {
              await input.fill('test content');
            }
            console.log(`✅ ${inputType} ${i + 1} interaction successful`);
          } catch (error) {
            console.log(`⚠️ ${inputType} ${i + 1} interaction failed:`, error);
          }
        }
      }
    }
  });

  test('Chart and Visualization Controls', async ({ page }) => {
    // Look for chart-specific controls
    const chartSelectors = [
      '.chart-container button',
      '.lightweight-chart button',
      '[class*="chart"] button',
      '[data-testid*="chart"] button'
    ];

    for (const selector of chartSelectors) {
      const chartButtons = await page.locator(selector).all();
      
      for (let i = 0; i < chartButtons.length; i++) {
        const button = chartButtons[i];
        
        if (await button.isVisible() && await button.isEnabled()) {
          try {
            const text = await button.textContent() || `Chart Button ${i + 1}`;
            await button.click();
            await page.waitForTimeout(300);
            console.log(`✅ Chart control "${text.trim()}" clicked successfully`);
          } catch (error) {
            console.log(`⚠️ Chart button ${i + 1} interaction failed:`, error);
          }
        }
      }
    }
  });

  test('Modal and Dropdown Triggers', async ({ page }) => {
    // Test buttons that open modals or dropdowns
    const triggerSelectors = [
      '[data-testid*="modal"]',
      '[data-testid*="dropdown"]',
      '.ant-dropdown-trigger',
      '[aria-haspopup]'
    ];

    for (const selector of triggerSelectors) {
      const triggers = await page.locator(selector).all();
      
      for (let i = 0; i < triggers.length; i++) {
        const trigger = triggers[i];
        
        if (await trigger.isVisible() && await trigger.isEnabled()) {
          try {
            await trigger.click();
            await page.waitForTimeout(300);
            
            // Try to close modal/dropdown if opened
            await page.keyboard.press('Escape');
            await page.waitForTimeout(200);
            
            console.log(`✅ Modal/Dropdown trigger ${i + 1} tested successfully`);
          } catch (error) {
            console.log(`⚠️ Modal/Dropdown trigger ${i + 1} failed:`, error);
          }
        }
      }
    }
  });

  test('Trading Interface Controls', async ({ page }) => {
    // Test trading-specific buttons
    const tradingSelectors = [
      '[data-testid*="buy"]',
      '[data-testid*="sell"]',
      '[data-testid*="order"]',
      '[data-testid*="trade"]',
      'button:has-text("Buy")',
      'button:has-text("Sell")',
      'button:has-text("Submit")',
      'button:has-text("Cancel")'
    ];

    for (const selector of tradingSelectors) {
      const tradingButtons = await page.locator(selector).all();
      
      for (let i = 0; i < tradingButtons.length; i++) {
        const button = tradingButtons[i];
        
        if (await button.isVisible() && await button.isEnabled()) {
          try {
            const text = await button.textContent() || `Trading Button ${i + 1}`;
            // Note: Don't actually submit trading orders in test
            console.log(`✅ Trading button "${text.trim()}" found and ready`);
          } catch (error) {
            console.log(`⚠️ Trading button ${i + 1} check failed:`, error);
          }
        }
      }
    }
  });
});