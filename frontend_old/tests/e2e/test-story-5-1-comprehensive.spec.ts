/**
 * Story 5.1 Comprehensive Test Suite
 * Tests both backend API functionality and frontend integration
 */
import { test, expect } from '@playwright/test';

test.describe('Story 5.1 Advanced Analytics - Comprehensive Testing', () => {
  test.beforeEach(async ({ page }) => {
    // Set up console logging
    page.on('console', msg => console.log('BROWSER:', msg.text()));
    page.on('pageerror', error => console.log('PAGE ERROR:', error.message));
    
    await page.goto('http://localhost:3000');
    await page.waitForTimeout(2000);
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

    await page.screenshot({ path: 'story-5-1-api-tests-complete.png' });
  });

  test('Frontend Integration - Story 5.1 Analytics Dashboard', async ({ page }) => {
    console.log('🔍 Testing Story 5.1 Frontend Integration');
    
    // Wait for main content to load
    await page.waitForSelector('.performance-dashboard', { timeout: 10000 });
    
    // Navigate to Performance tab
    await page.click('text=Performance');
    await page.waitForTimeout(2000);
    
    // Look for Story 5.1 Analytics tab
    const story51Tab = page.locator('text=Story 5.1 Analytics');
    await expect(story51Tab).toBeVisible();
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
    
    // Test tab navigation within Story 5.1 component
    await page.click('text=Monte Carlo');
    await page.waitForTimeout(2000);
    
    // Look for Monte Carlo specific content
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
    
    await page.screenshot({ path: 'story-5-1-frontend-complete.png' });
  });

  test('Story 5.1 API Integration in Browser', async ({ page }) => {
    console.log('🔍 Testing Story 5.1 React Hook API Integration');
    
    await page.waitForSelector('.performance-dashboard');
    await page.click('text=Performance');
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
    
    // Trigger API calls by interacting with components
    await page.click('text=Refresh Analytics');
    await page.waitForTimeout(2000);
    
    // Verify API calls were made
    expect(apiRequests.length).toBeGreaterThan(0);
    console.log('✅ Story 5.1 API integration working');
    
    // Check for error states (should not have errors)
    const errorMessages = await page.locator('.ant-alert-error').count();
    expect(errorMessages).toBe(0);
    console.log('✅ No error states detected');
    
    await page.screenshot({ path: 'story-5-1-integration-final.png' });
  });

  test('Story 5.1 Component Lifecycle and State Management', async ({ page }) => {
    console.log('🔍 Testing Story 5.1 Component Lifecycle');
    
    await page.waitForSelector('.performance-dashboard');
    await page.click('text=Performance');
    await page.click('text=Story 5.1 Analytics');
    await page.waitForTimeout(3000);
    
    // Test component mounting and initialization
    const dashboardContent = await page.textContent('.story5-advanced-analytics-dashboard');
    expect(dashboardContent).toContain('Performance Metrics');
    console.log('✅ Component properly mounted');
    
    // Test state management by switching tabs
    await page.click('text=Monte Carlo');
    await page.waitForTimeout(1000);
    
    const monteCarloVisible = await page.isVisible('text=Run Simulation');
    expect(monteCarloVisible).toBe(true);
    console.log('✅ Tab state management working');
    
    // Test data loading states
    const loadingSpinners = await page.locator('.ant-spin').count();
    console.log(`📊 Loading indicators: ${loadingSpinners}`);
    
    await page.screenshot({ path: 'story-5-1-lifecycle-test.png' });
  });
});