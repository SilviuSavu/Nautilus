import { test, expect } from '@playwright/test';

test('access Strategy UI for story 4.1', async ({ page }) => {
  // Navigate to strategy management dashboard
  await page.goto('http://localhost:3000/strategy');
  
  // Wait for content to load
  await page.waitForTimeout(3000);

  // Take screenshot of Strategy UI
  await page.screenshot({ path: 'strategy-ui-access.png', fullPage: true });

  // Check page title and content
  const title = await page.title();
  console.log('PAGE TITLE:', title);

  // Look for strategy-specific elements
  const strategyElements = await page.locator('text=Strategy, text=Template, text=Configuration, text=Deploy').count();
  const buttons = await page.locator('button').count();
  const forms = await page.locator('form, .ant-form').count();

  console.log('STRATEGY ELEMENTS FOUND:', strategyElements);
  console.log('BUTTONS:', buttons);
  console.log('FORMS:', forms);

  // Check if strategy builder is accessible
  await page.goto('http://localhost:3000/strategy-builder');
  await page.waitForTimeout(2000);
  
  await page.screenshot({ path: 'strategy-builder-access.png', fullPage: true });

  // Get visible text on strategy pages
  const strategyText = await page.textContent('body');
  console.log('STRATEGY PAGE CONTENT (first 300 chars):', strategyText?.substring(0, 300));

  // Check for main strategy components
  const hasStrategyContent = await page.locator('text=Strategy Management, text=Template Library, text=Visual Builder').count();
  console.log('MAIN STRATEGY COMPONENTS:', hasStrategyContent);

  // Verify strategy functionality exists
  expect(hasStrategyContent).toBeGreaterThan(0);
});