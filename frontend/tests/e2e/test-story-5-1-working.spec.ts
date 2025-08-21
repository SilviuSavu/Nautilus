/**
 * Story 5.1 Working Test Suite
 * Tests both backend API functionality and frontend integration with correct navigation
 */
import { test, expect } from '@playwright/test';

test.describe('Story 5.1 Advanced Analytics - Working Tests', () => {
  test.beforeEach(async ({ page }) => {
    page.on('console', msg => console.log('BROWSER:', msg.text()));
    page.on('pageerror', error => console.log('PAGE ERROR:', error.message));
    
    await page.goto('http://localhost:3000');
    await page.waitForTimeout(3000);
  });

  test('Backend API Endpoints - All Story 5.1 Analytics', async ({ page }) => {
    console.log('🔍 Testing Story 5.1 Backend API Endpoints');
    
    // Test 1: Performance Analytics Endpoint
    const performanceResponse = await page.evaluate(async () => {
      const response = await fetch('/api/v1/analytics/performance/test_portfolio');
      return { status: response.status, data: await response.json() };
    });
    
    expect(performanceResponse.status).toBe(200);
    expect(performanceResponse.data).toHaveProperty('alpha');
    expect(performanceResponse.data).toHaveProperty('beta');
    expect(performanceResponse.data).toHaveProperty('sharpe_ratio');
    console.log('✅ Performance Analytics API working');

    // Test 2: Monte Carlo Simulation Endpoint
    const monteCarloResponse = await page.evaluate(async () => {
      const response = await fetch('/api/v1/analytics/monte-carlo', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          portfolio_id: 'test_portfolio',
          scenarios: 100,
          time_horizon_days: 30
        })
      });
      return { status: response.status, data: await response.json() };
    });
    
    expect(monteCarloResponse.status).toBe(200);
    expect(monteCarloResponse.data).toHaveProperty('scenarios_run');
    expect(monteCarloResponse.data).toHaveProperty('confidence_intervals');
    expect(monteCarloResponse.data).toHaveProperty('value_at_risk_5');
    console.log('✅ Monte Carlo API working');

    // Test 3: Attribution Analysis Endpoint
    const attributionResponse = await page.evaluate(async () => {
      const response = await fetch('/api/v1/analytics/attribution/test_portfolio?attribution_type=sector');
      return { status: response.status, data: await response.json() };
    });
    
    expect(attributionResponse.status).toBe(200);
    expect(attributionResponse.data).toHaveProperty('attribution_type');
    expect(attributionResponse.data).toHaveProperty('attribution_breakdown');
    console.log('✅ Attribution Analysis API working');

    // Test 4: Statistical Tests Endpoint
    const statsResponse = await page.evaluate(async () => {
      const response = await fetch('/api/v1/analytics/statistical-tests/test_portfolio?test_type=sharpe');
      return { status: response.status, data: await response.json() };
    });
    
    expect(statsResponse.status).toBe(200);
    expect(statsResponse.data).toHaveProperty('sharpe_ratio_test');
    expect(statsResponse.data).toHaveProperty('alpha_significance_test');
    expect(statsResponse.data).toHaveProperty('bootstrap_results');
    console.log('✅ Statistical Tests API working');

    // Test 5: Benchmarks Endpoint
    const benchmarksResponse = await page.evaluate(async () => {
      const response = await fetch('/api/v1/analytics/benchmarks');
      return { status: response.status, data: await response.json() };
    });
    
    expect(benchmarksResponse.status).toBe(200);
    expect(benchmarksResponse.data).toHaveProperty('benchmarks');
    expect(benchmarksResponse.data.benchmarks).toBeInstanceOf(Array);
    console.log('✅ Benchmarks API working');

    await page.screenshot({ path: 'story-5-1-api-success.png' });
  });

  test('Frontend Integration - Navigate to Story 5.1 Analytics', async ({ page }) => {
    console.log('🔍 Testing Story 5.1 Frontend Navigation');
    
    // Click on Performance Monitoring tab (correct name from debug output)
    await page.click('text=Performance Monitoring');
    await page.waitForTimeout(3000);
    
    // Look for performance dashboard content
    await expect(page.locator('.performance-dashboard')).toBeVisible({ timeout: 10000 });
    console.log('✅ Performance dashboard loaded');
    
    // Look for Story 5.1 Analytics tab
    const story51Tab = page.locator('text=Story 5.1 Analytics');
    await expect(story51Tab).toBeVisible({ timeout: 10000 });
    console.log('✅ Story 5.1 Analytics tab found');
    
    // Click Story 5.1 Analytics tab
    await story51Tab.click();
    await page.waitForTimeout(3000);
    
    // Verify Story 5.1 Analytics content loads
    const analyticsContent = page.locator('.story5-advanced-analytics-dashboard');
    await expect(analyticsContent).toBeVisible({ timeout: 10000 });
    console.log('✅ Story 5.1 Analytics dashboard rendered');
    
    // Check for main sections
    await expect(page.locator('text=Performance Metrics')).toBeVisible();
    await expect(page.locator('text=Monte Carlo Simulation')).toBeVisible();
    await expect(page.locator('text=Attribution Analysis')).toBeVisible();
    await expect(page.locator('text=Statistical Tests')).toBeVisible();
    console.log('✅ All Story 5.1 sections visible');
    
    await page.screenshot({ path: 'story-5-1-frontend-success.png' });
  });

  test('Story 5.1 Tab Navigation within Analytics Dashboard', async ({ page }) => {
    console.log('🔍 Testing Story 5.1 Internal Tab Navigation');
    
    // Navigate to Story 5.1 Analytics
    await page.click('text=Performance Monitoring');
    await page.waitForTimeout(2000);
    await page.click('text=Story 5.1 Analytics');
    await page.waitForTimeout(3000);
    
    // Test Monte Carlo tab
    await page.click('text=Monte Carlo');
    await page.waitForTimeout(2000);
    
    await expect(page.locator('text=Run Simulation')).toBeVisible();
    await expect(page.locator('text=Confidence Intervals')).toBeVisible();
    console.log('✅ Monte Carlo tab working');
    
    // Test Attribution tab
    await page.click('text=Attribution');
    await page.waitForTimeout(2000);
    
    await expect(page.locator('text=Attribution Type')).toBeVisible();
    console.log('✅ Attribution tab working');
    
    // Test Statistical Tests tab
    await page.click('text=Statistical');
    await page.waitForTimeout(2000);
    
    await expect(page.locator('text=Sharpe Ratio Test')).toBeVisible();
    console.log('✅ Statistical Tests tab working');
    
    await page.screenshot({ path: 'story-5-1-tabs-success.png' });
  });

  test('Story 5.1 Component Integration Test', async ({ page }) => {
    console.log('🔍 Testing Story 5.1 React Component Integration');
    
    // Navigate to Story 5.1
    await page.click('text=Performance Monitoring');
    await page.waitForTimeout(2000);
    await page.click('text=Story 5.1 Analytics');
    await page.waitForTimeout(3000);
    
    // Monitor network requests
    const apiRequests: string[] = [];
    page.on('response', response => {
      if (response.url().includes('/api/v1/analytics/')) {
        apiRequests.push(response.url());
        console.log('📡 API Call:', response.url(), response.status());
      }
    });
    
    // Look for refresh/reload button and click it to trigger API calls
    const refreshButtons = page.locator('button:has-text("Refresh"), button:has-text("Load"), button:has-text("Update")');
    const refreshCount = await refreshButtons.count();
    
    if (refreshCount > 0) {
      await refreshButtons.first().click();
      await page.waitForTimeout(2000);
    }
    
    // Check for error states (should not have errors)
    const errorMessages = await page.locator('.ant-alert-error').count();
    expect(errorMessages).toBe(0);
    console.log('✅ No error states detected');
    
    // Verify content is loading or loaded
    const loadingSpinners = await page.locator('.ant-spin').count();
    console.log(`📊 Loading indicators: ${loadingSpinners}`);
    
    await page.screenshot({ path: 'story-5-1-integration-success.png' });
  });
});