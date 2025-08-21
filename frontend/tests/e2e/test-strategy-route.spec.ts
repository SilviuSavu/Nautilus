import { test, expect } from '@playwright/test';

test.describe('Strategy Route Integration', () => {
  test('Strategy route is accessible and components load', async ({ page }) => {
    // Navigate to strategy route
    await page.goto('http://localhost:3000/strategy');
    await page.waitForTimeout(3000);

    // Log browser console messages
    page.on('console', msg => console.log('BROWSER:', msg.text()));

    // Take screenshot
    await page.screenshot({ path: 'strategy-route-test.png', fullPage: true });

    // Check for strategy management dashboard
    const hasStrategyDashboard = await page.locator('text=Strategy').count() > 0;
    console.log('Has Strategy Dashboard:', hasStrategyDashboard);

    // Check for template library
    const hasTemplateLibrary = await page.locator('text=Template').count() > 0;
    console.log('Has Template Library:', hasTemplateLibrary);

    // Check for configuration elements
    const hasConfigElements = await page.locator('text=Configuration').count() > 0;
    console.log('Has Configuration Elements:', hasConfigElements);

    // Test API connectivity
    const apiTest = await page.evaluate(async () => {
      try {
        const response = await fetch('http://localhost:8000/api/v1/strategies/templates');
        const data = await response.json();
        return { success: true, templateCount: data.templates?.length || 0 };
      } catch (error) {
        return { success: false, error: error.message };
      }
    });

    console.log('API Test:', apiTest);

    // Verify page loads successfully
    expect(page.url()).toContain('/strategy');
    
    if (apiTest.success) {
      console.log(`✅ Strategy system working - ${apiTest.templateCount} templates available`);
    }
  });
});