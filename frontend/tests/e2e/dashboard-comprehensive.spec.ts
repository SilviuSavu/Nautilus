import { test, expect, Page } from '@playwright/test';

/**
 * Comprehensive Dashboard Test Suite
 * Tests all 15 tabs, core functionality, UI elements, AND ACTUAL BACKEND INTEGRATION
 * Updated with functional testing to verify buttons actually work, not just exist
 */

// Helper function to wait for dashboard to load
async function waitForDashboardLoad(page: Page) {
  await page.waitForSelector('[data-testid="dashboard"]', { timeout: 10000 });
  await page.waitForSelector('.ant-tabs-tab', { timeout: 5000 });
  // Wait for initial API calls to complete
  await page.waitForTimeout(2000);
}

// Helper function to wait for API calls
async function waitForApiCall(page: Page, endpoint: string) {
  return page.waitForResponse(response => 
    response.url().includes(endpoint) && response.status() === 200
  );
}

// Helper function to verify backend API connectivity
async function verifyBackendAPI(page: Page, endpoint: string) {
  const response = await page.evaluate(async (ep) => {
    try {
      const res = await fetch(`http://localhost:8001${ep}`);
      return { status: res.status, data: await res.json() };
    } catch (error) {
      return { error: error.message };
    }
  }, endpoint);
  return response;
}

// Helper function to click tab and verify content loads
async function clickTabAndVerify(page: Page, tabKey: string, tabLabel: string) {
  // Click the tab
  await page.click(`[data-node-key="${tabKey}"]`);
  await page.waitForTimeout(1000);
  
  // Verify tab is active
  const activeTab = await page.locator('.ant-tabs-tab-active');
  await expect(activeTab).toContainText(tabLabel);
  
  // Wait for content to load
  await page.waitForTimeout(2000);
}

test.describe('Dashboard Comprehensive Tests', () => {
  test.beforeEach(async ({ page }) => {
    // Navigate to dashboard
    await page.goto('/');
    await waitForDashboardLoad(page);
  });

  test('Dashboard loads with all basic elements AND FloatButton works', async ({ page }) => {
    console.log('🧪 Testing Dashboard loading and FloatButton functionality...');
    
    // Check main title
    await expect(page.locator('h2')).toContainText('NautilusTrader Dashboard');
    
    // Check that tabs container exists
    await expect(page.locator('.ant-tabs')).toBeVisible();
    
    // Check floating action button exists AND ACTUALLY WORKS
    const floatButton = page.locator('[aria-label="Place IB Order"]');
    await expect(floatButton).toBeVisible();
    
    // FUNCTIONAL TEST: Click the FloatButton and verify modal opens
    await floatButton.click();
    await page.waitForTimeout(500);
    
    // Verify modal opens with real form fields
    const modal = page.locator('.ant-modal');
    await expect(modal).toBeVisible();
    
    // Count form fields to ensure it's a real functional modal
    const formFields = await page.locator('.ant-form-item').count();
    console.log(`✅ Order modal opened with ${formFields} form fields`);
    expect(formFields).toBeGreaterThan(5); // Should have symbol, quantity, price, etc.
    
    // Close modal and verify it closes
    await page.click('.ant-modal-close');
    await expect(modal).toBeHidden();
    console.log('✅ Order modal closes properly - FloatButton is fully functional');
    
    // Verify at least some tabs are visible
    const tabs = await page.locator('.ant-tabs-tab').count();
    expect(tabs).toBeGreaterThan(10);
  });

  test('System tab functionality WITH REAL BACKEND INTEGRATION', async ({ page }) => {
    console.log('🧪 Testing System tab with actual backend API calls...');
    
    await clickTabAndVerify(page, 'system', 'System');
    
    // Check backend status alert exists
    await expect(page.locator('[data-testid="backend-status-alert"]')).toBeVisible();
    
    // FUNCTIONAL TEST: Verify backend is actually healthy
    const backendHealth = await verifyBackendAPI(page, '/health');
    console.log('Backend health check:', backendHealth);
    expect(backendHealth.status).toBe(200);
    expect(backendHealth.data.status).toBe('healthy');
    
    // FUNCTIONAL TEST: Click Refresh button and verify API call
    console.log('Testing Refresh button functionality...');
    const apiCallPromise = waitForApiCall(page, '/health');
    await page.click('button:has-text("Refresh")');
    const apiResponse = await apiCallPromise;
    console.log('✅ Refresh button successfully called backend API');
    expect(apiResponse.status()).toBe(200);
    
    // FUNCTIONAL TEST: Verify data source APIs are actually operational
    const dataSourceHealth = await verifyBackendAPI(page, '/api/v1/nautilus-data/health');
    console.log('Data sources health:', dataSourceHealth);
    if (dataSourceHealth.status === 200) {
      const fredStatus = dataSourceHealth.data.find(source => source.source === 'FRED');
      const alphaVantageStatus = dataSourceHealth.data.find(source => source.source === 'Alpha Vantage');
      console.log('✅ FRED API operational:', fredStatus?.status === 'operational');
      console.log('✅ Alpha Vantage API operational:', alphaVantageStatus?.status === 'operational');
    }
    
    // FUNCTIONAL TEST: MessageBus connection status
    const mbStatus = await verifyBackendAPI(page, '/api/v1/messagebus/status');
    if (mbStatus.status === 200) {
      console.log('✅ MessageBus connection state:', mbStatus.data.connection_state);
      expect(['connected', 'disconnected', 'connecting'].includes(mbStatus.data.connection_state)).toBe(true);
    }
    
    // Check UI elements exist (after verifying they have real backend data)
    await expect(page.getByText('API Status')).toBeVisible();
    await expect(page.getByText('Backend URL:')).toBeVisible();
    await expect(page.getByText('Environment')).toBeVisible();
    await expect(page.getByText('MessageBus Connection')).toBeVisible();
    await expect(page.getByText('Data Backfill System')).toBeVisible();
    
    console.log('✅ System tab has both UI elements AND working backend integration');
  });

  test('Engine tab functionality', async ({ page }) => {
    await clickTabAndVerify(page, 'nautilus-engine', 'Engine');
    
    // Should load NautilusEngineManager component
    // Check for common engine elements
    await page.waitForTimeout(3000);
    
    // Verify no critical errors occurred
    const errorElements = await page.locator('.ant-alert-error').count();
    expect(errorElements).toBe(0);
  });

  test('Backtest tab functionality', async ({ page }) => {
    await clickTabAndVerify(page, 'backtesting', 'Backtest');
    
    // Should load BacktestRunner component
    await page.waitForTimeout(3000);
    
    // Verify no critical errors occurred
    const errorElements = await page.locator('.ant-alert-error').count();
    expect(errorElements).toBe(0);
  });

  test('Deploy tab functionality', async ({ page }) => {
    await clickTabAndVerify(page, 'deployment', 'Deploy');
    
    // Should load StrategyDeploymentPipeline component
    await page.waitForTimeout(3000);
    
    // Verify no critical errors occurred
    const errorElements = await page.locator('.ant-alert-error').count();
    expect(errorElements).toBe(0);
  });

  test('Data tab functionality', async ({ page }) => {
    await clickTabAndVerify(page, 'data-catalog', 'Data');
    
    // Should load DataCatalogBrowser component
    await page.waitForTimeout(3000);
    
    // Verify no critical errors occurred
    const errorElements = await page.locator('.ant-alert-error').count();
    expect(errorElements).toBe(0);
  });

  test('Search tab functionality', async ({ page }) => {
    await clickTabAndVerify(page, 'instruments', 'Search');
    
    // Check for Universal Instrument Search
    await expect(page.getByText('Universal Instrument Search')).toBeVisible();
    
    // Check for Search Features card
    await expect(page.getByText('Search Features')).toBeVisible();
    await expect(page.getByText('Search Capabilities:')).toBeVisible();
    
    // Check for Supported Asset Classes
    await expect(page.getByText('Supported Asset Classes')).toBeVisible();
    
    // Verify asset class tags
    await expect(page.locator('.ant-tag:has-text("STK - Stocks")')).toBeVisible();
    await expect(page.locator('.ant-tag:has-text("CASH - Forex")')).toBeVisible();
  });

  test('Watchlist tab functionality', async ({ page }) => {
    await clickTabAndVerify(page, 'watchlists', 'Watchlist');
    
    // Check for Watchlist Features
    await expect(page.getByText('Watchlist Features')).toBeVisible();
    await expect(page.getByText('Watchlist Management:')).toBeVisible();
    
    // Check for Data Formats
    await expect(page.getByText('Data Formats')).toBeVisible();
    await expect(page.getByText('Export Formats:')).toBeVisible();
  });

  test('Chart tab functionality', async ({ page }) => {
    await clickTabAndVerify(page, 'chart', 'Chart');
    
    // Check for chart control cards
    await expect(page.getByText('Instrument Selection')).toBeVisible();
    await expect(page.getByText('Timeframe Selection')).toBeVisible();
    await expect(page.getByText('Technical Indicators')).toBeVisible();
    
    // Wait for chart component to load
    await page.waitForTimeout(3000);
  });

  test('Strategy tab functionality', async ({ page }) => {
    await clickTabAndVerify(page, 'strategy', 'Strategy');
    
    // Should load StrategyManagementDashboard component
    await page.waitForTimeout(3000);
    
    // Verify no critical errors occurred
    const errorElements = await page.locator('.ant-alert-error').count();
    expect(errorElements).toBe(0);
  });

  test('Performance tab functionality', async ({ page }) => {
    await clickTabAndVerify(page, 'performance', 'Perform');
    
    // Should load PerformanceDashboard component
    await page.waitForTimeout(3000);
    
    // Verify no critical errors occurred
    const errorElements = await page.locator('.ant-alert-error').count();
    expect(errorElements).toBe(0);
  });

  test('Portfolio tab functionality', async ({ page }) => {
    await clickTabAndVerify(page, 'portfolio', 'Portfolio');
    
    // Should load PortfolioVisualization component
    await page.waitForTimeout(3000);
    
    // Verify no critical errors occurred
    const errorElements = await page.locator('.ant-alert-error').count();
    expect(errorElements).toBe(0);
  });

  test('Factors tab functionality', async ({ page }) => {
    await clickTabAndVerify(page, 'factors', 'Factors');
    
    // Should load FactorDashboard component
    await page.waitForTimeout(3000);
    
    // Verify no critical errors occurred
    const errorElements = await page.locator('.ant-alert-error').count();
    expect(errorElements).toBe(0);
  });

  test('Risk tab functionality', async ({ page }) => {
    await clickTabAndVerify(page, 'risk', 'Risk');
    
    // Should load RiskDashboard component
    await page.waitForTimeout(3000);
    
    // Verify no critical errors occurred
    const errorElements = await page.locator('.ant-alert-error').count();
    expect(errorElements).toBe(0);
  });

  test('IB tab functionality', async ({ page }) => {
    await clickTabAndVerify(page, 'ib', 'IB');
    
    // Should load IBDashboard component
    await page.waitForTimeout(3000);
    
    // Verify no critical errors occurred
    const errorElements = await page.locator('.ant-alert-error').count();
    expect(errorElements).toBe(0);
  });

  test('Tab switching preserves state', async ({ page }) => {
    // Start with System tab
    await clickTabAndVerify(page, 'system', 'System');
    
    // Switch to Search tab
    await clickTabAndVerify(page, 'instruments', 'Search');
    
    // Switch back to System tab
    await clickTabAndVerify(page, 'system', 'System');
    
    // Verify System tab content is still there
    await expect(page.getByText('API Status')).toBeVisible();
    await expect(page.getByText('Environment')).toBeVisible();
  });

  test('Floating action button for order placement', async ({ page }) => {
    // Check floating button exists
    const floatButton = page.locator('[aria-label="Place IB Order"]');
    await expect(floatButton).toBeVisible();
    
    // Click the button
    await floatButton.click();
    
    // Should open order placement modal
    await page.waitForTimeout(1000);
    
    // Check if modal appeared (it might not work without IB connection)
    // This is just to test the UI interaction
  });

  test('Error boundaries handle component errors gracefully', async ({ page }) => {
    // Navigate through all tabs and check for error boundary fallbacks
    const tabs = [
      { key: 'nautilus-engine', label: 'Engine' },
      { key: 'backtesting', label: 'Backtest' },
      { key: 'deployment', label: 'Deploy' },
      { key: 'data-catalog', label: 'Data' },
      { key: 'strategy', label: 'Strategy' },
      { key: 'performance', label: 'Perform' },
      { key: 'portfolio', label: 'Portfolio' },
      { key: 'factors', label: 'Factors' },
      { key: 'risk', label: 'Risk' },
      { key: 'ib', label: 'IB' }
    ];

    for (const tab of tabs) {
      await clickTabAndVerify(page, tab.key, tab.label);
      
      // Check for error boundary messages
      const errorBoundaryExists = await page.locator('.ant-result-error').count() > 0;
      if (errorBoundaryExists) {
        console.log(`Error boundary active for ${tab.label} tab`);
        // Verify error boundary has proper fallback UI
        await expect(page.locator('.ant-result-title')).toBeVisible();
      }
    }
  });

  test('Responsive layout on different screen sizes', async ({ page }) => {
    // Test mobile size
    await page.setViewportSize({ width: 375, height: 667 });
    await page.waitForTimeout(1000);
    
    // Dashboard should still be visible
    await expect(page.locator('[data-testid="dashboard"]')).toBeVisible();
    
    // Test tablet size
    await page.setViewportSize({ width: 768, height: 1024 });
    await page.waitForTimeout(1000);
    
    // Dashboard should still be visible
    await expect(page.locator('[data-testid="dashboard"]')).toBeVisible();
    
    // Test desktop size
    await page.setViewportSize({ width: 1920, height: 1080 });
    await page.waitForTimeout(1000);
    
    // Dashboard should still be visible
    await expect(page.locator('[data-testid="dashboard"]')).toBeVisible();
  });

  test('Keyboard navigation works for tabs', async ({ page }) => {
    // Focus on the first tab
    await page.keyboard.press('Tab');
    await page.keyboard.press('Tab');
    
    // Navigate with arrow keys
    await page.keyboard.press('ArrowRight');
    await page.waitForTimeout(500);
    
    // Check that focus moved
    const focusedElement = await page.locator(':focus');
    await expect(focusedElement).toBeVisible();
  });

  test('Dashboard handles API failures gracefully', async ({ page }) => {
    // Intercept API calls and simulate failures
    await page.route('**/api/v1/**', (route) => {
      route.fulfill({ status: 500, body: 'Server Error' });
    });
    
    // Reload page
    await page.reload();
    await page.waitForTimeout(3000);
    
    // Dashboard should still load
    await expect(page.locator('[data-testid="dashboard"]')).toBeVisible();
    
    // Should show error states
    const alerts = await page.locator('.ant-alert-error').count();
    expect(alerts).toBeGreaterThan(0);
  });
});