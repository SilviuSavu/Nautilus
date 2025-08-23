import { test, expect } from '@playwright/test';

test.describe('Backend Health Check and Recovery', () => {
  test('diagnose and fix backend connectivity', async ({ page }) => {
    console.log('🔍 Starting backend health diagnostics...');
    
    // Test backend health endpoint
    try {
      const healthResponse = await page.request.get('http://localhost:8001/health');
      console.log(`Health endpoint status: ${healthResponse.status()}`);
      
      if (healthResponse.ok()) {
        const healthData = await healthResponse.json();
        console.log('✅ Backend is responding:', healthData);
      } else {
        console.log('❌ Backend health check failed');
      }
    } catch (error) {
      console.log('❌ Cannot connect to backend health endpoint:', error);
    }

    // Test specific API endpoints
    const endpoints = [
      '/api/v1/portfolio/positions',
      '/api/v1/market-data/status',
      '/api/v1/ib/connection',
      '/api/v1/monitoring/status'
    ];

    for (const endpoint of endpoints) {
      try {
        const response = await page.request.get(`http://localhost:8001${endpoint}`);
        console.log(`${endpoint}: ${response.status()} ${response.statusText()}`);
        
        if (response.ok()) {
          const data = await response.text();
          console.log(`✅ ${endpoint} working - response length: ${data.length}`);
        } else {
          console.log(`❌ ${endpoint} failed with status ${response.status()}`);
        }
      } catch (error) {
        console.log(`❌ ${endpoint} error:`, error);
      }
    }

    // Test frontend can reach backend
    await page.goto('http://localhost:3000');
    await page.waitForTimeout(2000);
    
    // Check for console errors
    const logs: string[] = [];
    page.on('console', msg => {
      logs.push(`${msg.type()}: ${msg.text()}`);
    });

    // Try to load dashboard and check network requests
    await page.goto('http://localhost:3000');
    await page.waitForTimeout(3000);
    
    console.log('Console logs:', logs);
    
    // Check if backend is accessible from frontend
    const networkRequests: string[] = [];
    page.on('response', response => {
      if (response.url().includes('localhost:8080')) {
        networkRequests.push(`${response.status()} ${response.url()}`);
      }
    });

    // Navigate to trigger API calls
    if (await page.locator('[data-testid="ib-tab"], text="IB"').first().isVisible()) {
      await page.locator('[data-testid="ib-tab"], text="IB"').first().click();
      await page.waitForTimeout(2000);
    }

    console.log('Network requests to backend:', networkRequests);
    
    // Final assessment
    if (networkRequests.some(req => req.startsWith('200'))) {
      console.log('✅ Backend is accessible from frontend');
    } else {
      console.log('❌ Frontend cannot reach backend');
    }
  });

  test('attempt backend restart if needed', async ({ page }) => {
    console.log('🔄 Testing backend restart capability...');
    
    // First check if backend is responsive
    let backendWorking = false;
    try {
      const response = await page.request.get('http://localhost:8080/health');
      backendWorking = response.ok();
    } catch (error) {
      console.log('Backend not responding, restart needed');
    }

    if (!backendWorking) {
      console.log('❌ Backend needs restart - please check backend process');
      
      // Log current process status
      const processCheck = await page.evaluate(async () => {
        try {
          const response = await fetch('http://localhost:8080/health');
          return { status: response.status, working: true };
        } catch (error) {
          return { error: error.toString(), working: false };
        }
      });
      
      console.log('Process check result:', processCheck);
    } else {
      console.log('✅ Backend is working properly');
    }
  });
});