import { test, expect } from '@playwright/test';

test('Debug IB Dashboard Issues', async ({ page }) => {
  // Enable console logging
  page.on('console', msg => {
    console.log(`BROWSER ${msg.type()}: ${msg.text()}`);
  });
  
  // Enable error logging
  page.on('pageerror', error => {
    console.log(`PAGE ERROR: ${error.message}`);
    console.log(`Stack: ${error.stack}`);
  });
  
  // Navigate to the frontend
  await page.goto('http://localhost:3000');
  
  // Wait for page to load
  await page.waitForLoadState('networkidle');
  
  // Click on Interactive Brokers tab
  console.log('Clicking on Interactive Brokers tab...');
  await page.click('text=Interactive Brokers');
  
  // Wait a moment for the component to load
  await page.waitForTimeout(3000);
  
  // Take screenshot of what we see
  await page.screenshot({ 
    path: 'ib-dashboard-debug.png',
    fullPage: true 
  });
  
  // Check for debug message
  const debugMessage = page.locator('text=Debug: IBDashboard loading');
  if (await debugMessage.isVisible()) {
    console.log('✓ Debug message visible');
  }
  
  // Check for error messages
  const errorAlert = page.locator('.ant-alert-error');
  if (await errorAlert.isVisible()) {
    const errorText = await errorAlert.textContent();
    console.log(`❌ Error alert found: ${errorText}`);
  }
  
  // Check for IBDashboard content
  const ibDashboard = page.locator('text=Interactive Brokers Dashboard');
  if (await ibDashboard.isVisible()) {
    console.log('✓ IBDashboard title found');
  }
  
  // Check for orders
  const ordersSection = page.locator('text=Orders');
  if (await ordersSection.isVisible()) {
    console.log('✓ Orders section found');
    
    // Check for SPY order
    const spyOrder = page.locator('text=SPY');
    if (await spyOrder.isVisible()) {
      console.log('✓ SPY order visible');
    } else {
      console.log('❌ SPY order not visible');
    }
  }
  
  // Check network requests
  const requests = [];
  page.on('request', request => {
    if (request.url().includes('/api/v1/ib/')) {
      requests.push({
        url: request.url(),
        method: request.method()
      });
    }
  });
  
  // Wait for any API calls
  await page.waitForTimeout(2000);
  
  console.log('API Requests made:');
  requests.forEach(req => {
    console.log(`  ${req.method} ${req.url}`);
  });
});