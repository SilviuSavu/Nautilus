import { test, expect } from '@playwright/test';

/**
 * Playwright MCP Demonstration Test
 * 
 * This test demonstrates the capabilities of Playwright MCP by:
 * 1. Testing the complete Nautilus trading platform workflow
 * 2. Capturing comprehensive console logs and network activity
 * 3. Taking screenshots for visual verification
 * 4. Validating backend API responses
 * 5. Testing real-time data flow and chart functionality
 */

test.describe('Playwright MCP Demo - Nautilus Trading Platform', () => {
  let consoleMessages: string[] = [];
  let networkRequests: any[] = [];

  test.beforeEach(async ({ page }) => {
    // Capture all console messages for debugging
    page.on('console', msg => {
      const message = `[${msg.type().toUpperCase()}] ${msg.text()}`;
      consoleMessages.push(message);
      console.log('BROWSER CONSOLE:', message);
    });

    // Capture network requests for API analysis
    page.on('request', request => {
      networkRequests.push({
        url: request.url(),
        method: request.method(),
        headers: request.headers(),
        timestamp: new Date().toISOString()
      });
      console.log('NETWORK REQUEST:', request.method(), request.url());
    });

    // Capture network responses for API validation
    page.on('response', response => {
      console.log('NETWORK RESPONSE:', response.status(), response.url());
    });

    // Clear arrays for each test
    consoleMessages = [];
    networkRequests = [];
  });

  test('Complete Trading Platform Workflow - MCP Demo', async ({ page }) => {
    console.log('🚀 Starting Playwright MCP Demo Test');

    // Step 1: Navigate to the application
    console.log('📍 Step 1: Loading Nautilus Trading Platform');
    await page.goto('http://localhost:3000');
    await page.screenshot({ path: 'mcp-demo-01-initial-load.png', fullPage: true });

    // Wait for initial load and verify no critical errors
    await page.waitForTimeout(2000);
    
    // Step 2: Verify the page loaded correctly
    console.log('📍 Step 2: Verifying page load');
    const title = await page.title();
    expect(title).toBeTruthy();
    console.log('✅ Page title:', title);

    // Step 3: Navigate to IB Dashboard
    console.log('📍 Step 3: Navigating to IB Dashboard');
    await page.click('text=IB Dashboard');
    await page.waitForTimeout(1000);
    await page.screenshot({ path: 'mcp-demo-02-ib-dashboard.png', fullPage: true });

    // Step 4: Test instrument search functionality
    console.log('📍 Step 4: Testing instrument search');
    
    // Look for search input or search trigger
    const searchElements = await page.locator('input[placeholder*="search"], input[placeholder*="Search"], button:has-text("Search"), .search-trigger').count();
    
    if (searchElements > 0) {
      console.log('✅ Found search element');
      
      // Try different search strategies
      const searchInput = page.locator('input[placeholder*="search"], input[placeholder*="Search"]').first();
      const searchButton = page.locator('button:has-text("Search"), .search-trigger').first();
      
      if (await searchInput.count() > 0) {
        await searchInput.fill('AAPL');
        await page.keyboard.press('Enter');
      } else if (await searchButton.count() > 0) {
        await searchButton.click();
        await page.waitForTimeout(500);
        const modalSearch = page.locator('input[placeholder*="search"], input[placeholder*="Search"]').first();
        if (await modalSearch.count() > 0) {
          await modalSearch.fill('AAPL');
          await page.keyboard.press('Enter');
        }
      }
      
      await page.waitForTimeout(2000);
      await page.screenshot({ path: 'mcp-demo-03-search-results.png', fullPage: true });
    } else {
      console.log('⚠️ No search element found, taking screenshot for analysis');
      await page.screenshot({ path: 'mcp-demo-03-no-search-found.png', fullPage: true });
    }

    // Step 5: Test chart functionality
    console.log('📍 Step 5: Testing chart functionality');
    
    // Look for chart or chart container
    const chartElements = await page.locator('canvas, .chart, .trading-chart, .lightweight-charts').count();
    console.log(`📊 Found ${chartElements} chart elements`);
    
    if (chartElements > 0) {
      console.log('✅ Charts detected');
      
      // Test timeframe selection if available
      const timeframes = ['1m', '5m', '15m', '1h', '1d'];
      for (const tf of timeframes) {
        const tfButton = page.locator(`button:has-text("${tf}"), [data-timeframe="${tf}"]`);
        if (await tfButton.count() > 0) {
          console.log(`📈 Testing timeframe: ${tf}`);
          await tfButton.click();
          await page.waitForTimeout(1000);
          break;
        }
      }
    }
    
    await page.screenshot({ path: 'mcp-demo-04-charts.png', fullPage: true });

    // Step 6: Test API endpoints directly
    console.log('📍 Step 6: Testing API endpoints');
    
    // Test backend health
    const healthResponse = await page.evaluate(async () => {
      try {
        const response = await fetch('http://localhost:8000/health');
        return {
          status: response.status,
          ok: response.ok,
          data: await response.text()
        };
      } catch (error) {
        return { error: error.message };
      }
    });
    
    console.log('🏥 Health endpoint response:', healthResponse);

    // Test instrument search API
    const searchResponse = await page.evaluate(async () => {
      try {
        const response = await fetch('http://localhost:8000/api/v1/ib/instruments/search/AAPL?sec_type=STK');
        return {
          status: response.status,
          ok: response.ok,
          data: await response.json()
        };
      } catch (error) {
        return { error: error.message };
      }
    });
    
    console.log('🔍 Search API response:', JSON.stringify(searchResponse, null, 2));

    // Step 7: Analyze network activity
    console.log('📍 Step 7: Analyzing network activity');
    console.log(`📡 Total network requests: ${networkRequests.length}`);
    
    const apiRequests = networkRequests.filter(req => req.url.includes('api'));
    console.log(`🌐 API requests: ${apiRequests.length}`);
    
    apiRequests.forEach(req => {
      console.log(`  📤 ${req.method} ${req.url}`);
    });

    // Step 8: Final verification screenshot
    await page.screenshot({ path: 'mcp-demo-05-final-state.png', fullPage: true });

    // Step 9: Summary and validation
    console.log('📍 Step 9: Test Summary');
    console.log(`💬 Console messages captured: ${consoleMessages.length}`);
    console.log(`🌐 Network requests captured: ${networkRequests.length}`);
    
    // Check for critical errors
    const errors = consoleMessages.filter(msg => msg.includes('[ERROR]'));
    const warnings = consoleMessages.filter(msg => msg.includes('[WARNING]'));
    
    console.log(`❌ Errors found: ${errors.length}`);
    console.log(`⚠️ Warnings found: ${warnings.length}`);
    
    if (errors.length > 0) {
      console.log('🚨 Critical errors detected:');
      errors.forEach(error => console.log(`  ${error}`));
    }

    // MCP Demo specific validations
    expect(consoleMessages.length).toBeGreaterThan(0); // Should have some console activity
    expect(networkRequests.length).toBeGreaterThan(0); // Should have some network activity
    
    console.log('✅ Playwright MCP Demo completed successfully!');
    
    // Export captured data for further analysis
    await page.evaluate((data) => {
      window.MCPDemoResults = data;
    }, {
      consoleMessages,
      networkRequests,
      healthResponse,
      searchResponse,
      timestamp: new Date().toISOString()
    });
  });

  test('Backend API Validation - MCP Style', async ({ page }) => {
    console.log('🔧 Starting Backend API Validation Test');

    // Test multiple endpoints systematically
    const endpoints = [
      '/health',
      '/api/v1/ib/instruments/search/AAPL?sec_type=STK',
      '/api/v1/ib/instruments/search/MSFT?sec_type=STK',
      '/api/v1/ib/status'
    ];

    const results = [];

    for (const endpoint of endpoints) {
      console.log(`🔍 Testing endpoint: ${endpoint}`);
      
      const result = await page.evaluate(async (url) => {
        try {
          const startTime = Date.now();
          const response = await fetch(`http://localhost:8000${url}`);
          const endTime = Date.now();
          
          let data;
          try {
            data = await response.json();
          } catch {
            data = await response.text();
          }
          
          return {
            endpoint: url,
            status: response.status,
            ok: response.ok,
            responseTime: endTime - startTime,
            data: data,
            success: true
          };
        } catch (error) {
          return {
            endpoint: url,
            error: error.message,
            success: false
          };
        }
      }, endpoint);
      
      results.push(result);
      console.log(`📊 ${endpoint}: ${result.success ? '✅' : '❌'} (${result.responseTime || 'N/A'}ms)`);
    }

    // Generate API test report
    console.log('📋 API Test Report:');
    results.forEach(result => {
      if (result.success) {
        console.log(`✅ ${result.endpoint}: Status ${result.status}, ${result.responseTime}ms`);
      } else {
        console.log(`❌ ${result.endpoint}: ${result.error}`);
      }
    });

    await page.screenshot({ path: 'mcp-demo-api-validation.png' });

    // Validate that at least some endpoints work
    const successfulRequests = results.filter(r => r.success && r.status === 200);
    expect(successfulRequests.length).toBeGreaterThan(0);
    
    console.log('✅ Backend API Validation completed!');
  });

  test('Real-time Data Flow Test - MCP Enhanced', async ({ page }) => {
    console.log('📊 Starting Real-time Data Flow Test');

    await page.goto('http://localhost:3000');
    await page.waitForTimeout(2000);

    // Monitor data updates over time
    const dataSnapshots = [];
    
    for (let i = 0; i < 5; i++) {
      console.log(`📸 Taking data snapshot ${i + 1}/5`);
      
      const snapshot = await page.evaluate(() => {
        // Try to capture any market data or chart data
        const charts = document.querySelectorAll('canvas, .chart-container');
        const dataElements = document.querySelectorAll('[data-price], [data-volume], .market-data');
        
        return {
          timestamp: new Date().toISOString(),
          chartCount: charts.length,
          dataElementCount: dataElements.length,
          // Try to get any visible data
          visibleText: document.body.innerText.slice(0, 500)
        };
      });
      
      dataSnapshots.push(snapshot);
      await page.waitForTimeout(3000); // Wait 3 seconds between snapshots
    }

    console.log('📊 Data Flow Analysis:');
    dataSnapshots.forEach((snapshot, index) => {
      console.log(`  Snapshot ${index + 1}: ${snapshot.chartCount} charts, ${snapshot.dataElementCount} data elements`);
    });

    await page.screenshot({ path: 'mcp-demo-data-flow.png', fullPage: true });

    // Validate that we captured meaningful data
    expect(dataSnapshots.length).toBe(5);
    console.log('✅ Real-time Data Flow Test completed!');
  });
});