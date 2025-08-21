import { test, expect } from '@playwright/test';

test('Debug Risk Dashboard', async ({ page }) => {
  await page.goto('http://localhost:3000');
  await page.waitForTimeout(2000);
  
  // Click on the Risk tab
  await page.click('text=Risk');
  await page.waitForTimeout(5000);
  
  // Take a screenshot to see what's happening
  await page.screenshot({ path: 'debug-risk-dashboard.png', fullPage: true });
  
  // Log any error messages
  const errorElement = page.locator('text=Error Loading Risk Data');
  if (await errorElement.isVisible()) {
    console.log('Error message found: Error Loading Risk Data');
    
    // Look for more detailed error information
    const errorDescription = page.locator('.ant-alert-description');
    if (await errorDescription.isVisible()) {
      const errorText = await errorDescription.textContent();
      console.log('Error description:', errorText);
    }
  }
  
  // Check network tab in console
  page.on('response', response => {
    if (response.url().includes('/risk/')) {
      console.log(`Risk API Response: ${response.url()} - Status: ${response.status()}`);
    }
  });
});