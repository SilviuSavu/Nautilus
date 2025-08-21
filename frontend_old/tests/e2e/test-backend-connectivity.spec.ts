import { test, expect } from '@playwright/test';

test('Verify backend services are working', async ({ page }) => {
  console.log('Testing backend connectivity...');
  
  // Track console messages
  const consoleErrors: string[] = [];
  const consoleWarnings: string[] = [];
  
  page.on('console', msg => {
    if (msg.type() === 'error') {
      consoleErrors.push(msg.text());
    } else if (msg.type() === 'warning') {
      consoleWarnings.push(msg.text());
    }
  });
  
  // Navigate to the frontend
  await page.goto('http://localhost:3000');
  await page.waitForLoadState('networkidle');
  
  // Wait for initial API calls to complete
  await page.waitForTimeout(5000);
  
  // Check for backend disconnection messages
  const backendDisconnected = await page.locator('text=/Backend.*Disconnected/i').count();
  const messagebusDisconnected = await page.locator('text=/MessageBus.*Disconnected/i').count();
  
  console.log('Backend disconnected indicators:', backendDisconnected);
  console.log('MessageBus disconnected indicators:', messagebusDisconnected);
  
  // Look for successful API responses in network tab
  const apiSuccessMessages = await page.locator('text=/✓|Success|Connected/i').count();
  console.log('Success indicators found:', apiSuccessMessages);
  
  // Check WebSocket connection errors
  const wsErrors = consoleErrors.filter(error => 
    error.includes('WebSocket') || error.includes('ws://')
  );
  
  console.log('\n=== BACKEND CONNECTIVITY REPORT ===');
  console.log('Backend Disconnected Indicators:', backendDisconnected);
  console.log('MessageBus Disconnected Indicators:', messagebusDisconnected);
  console.log('Success Indicators:', apiSuccessMessages);
  console.log('WebSocket Errors:', wsErrors.length);
  console.log('Total Console Errors:', consoleErrors.length);
  
  if (wsErrors.length > 0) {
    console.log('\nWebSocket Errors:');
    wsErrors.forEach(error => console.log('  -', error));
  }
  
  // Take screenshot
  await page.screenshot({ path: 'frontend/backend-connectivity-test.png', fullPage: true });
});

test('Test API endpoints directly', async ({ page, request }) => {
  console.log('Testing API endpoints...');
  
  // Test health endpoint
  const healthResponse = await request.get('http://localhost:8000/health');
  console.log('Health endpoint status:', healthResponse.status());
  console.log('Health response:', await healthResponse.text());
  
  // Test strategies endpoint
  const strategiesResponse = await request.get('http://localhost:8000/api/v1/strategies/active');
  console.log('Strategies endpoint status:', strategiesResponse.status());
  
  // Test performance endpoint
  const performanceResponse = await request.get('http://localhost:8000/api/v1/performance/aggregate?start_date=2025-07-22T04:13:07.279Z&end_date=2025-08-21T04:13:07.279Z');
  console.log('Performance endpoint status:', performanceResponse.status());
  
  // Test engine status
  const engineResponse = await request.get('http://localhost:8000/api/v1/nautilus/engine/status');
  console.log('Engine endpoint status:', engineResponse.status());
  
  expect(healthResponse.status()).toBe(200);
  expect(strategiesResponse.status()).toBe(200);
  expect(performanceResponse.status()).toBe(200);
  expect(engineResponse.status()).toBe(200);
  
  console.log('✅ All API endpoints are working!');
});