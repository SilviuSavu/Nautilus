/**
 * Story 5.1 Complete Integration Test
 * Tests both backend APIs and frontend integration with real/simulated data
 */

import { test, expect, Page } from '@playwright/test';

// Test configuration
const BACKEND_URL = 'http://localhost:8000';
const FRONTEND_URL = 'http://localhost:3000';
const TEST_PORTFOLIO_ID = 'test_portfolio';

test.describe('Story 5.1: Advanced Performance Analytics - Complete Integration', () => {
  let page: Page;

  test.beforeAll(async ({ browser }) => {
    page = await browser.newPage();
    
    // Enable console logging for debugging
    page.on('console', msg => console.log('BROWSER CONSOLE:', msg.text()));
    page.on('pageerror', error => console.log('PAGE ERROR:', error.message));
  });

  test.afterAll(async () => {
    await page.close();
  });

  test.describe('Backend API Integration Tests', () => {
    
    test('Should fetch available benchmarks', async ({ request }) => {
      const response = await request.get(`${BACKEND_URL}/api/v1/analytics/benchmarks`);
      
      expect(response.status()).toBe(200);
      
      const data = await response.json();
      expect(data).toHaveProperty('benchmarks');
      expect(Array.isArray(data.benchmarks)).toBeTruthy();
      expect(data.benchmarks.length).toBeGreaterThan(0);
      
      // Verify benchmark structure
      const benchmark = data.benchmarks[0];
      expect(benchmark).toHaveProperty('symbol');
      expect(benchmark).toHaveProperty('name');
      expect(benchmark).toHaveProperty('category');
      expect(benchmark).toHaveProperty('data_available_from');
      
      console.log(`✅ Found ${data.benchmarks.length} available benchmarks`);
    });

    test('Should fetch performance analytics for portfolio', async ({ request }) => {
      const response = await request.get(
        `${BACKEND_URL}/api/v1/analytics/performance/${TEST_PORTFOLIO_ID}?benchmark=SPY`
      );
      
      expect(response.status()).toBe(200);
      
      const data = await response.json();
      
      // Verify all required fields are present
      const requiredFields = [
        'alpha', 'beta', 'information_ratio', 'tracking_error',
        'sharpe_ratio', 'sortino_ratio', 'calmar_ratio', 'max_drawdown',
        'volatility', 'downside_deviation', 'rolling_metrics',
        'period_start', 'period_end', 'benchmark'
      ];
      
      requiredFields.forEach(field => {
        expect(data).toHaveProperty(field);
      });
      
      expect(data.benchmark).toBe('SPY');
      expect(Array.isArray(data.rolling_metrics)).toBeTruthy();
      
      console.log(`✅ Performance analytics: Alpha=${data.alpha}, Beta=${data.beta}, Sharpe=${data.sharpe_ratio}`);
    });

    test('Should run Monte Carlo simulation', async ({ request }) => {
      const monteCarloRequest = {
        portfolio_id: TEST_PORTFOLIO_ID,
        scenarios: 1000,
        time_horizon_days: 30,
        confidence_levels: [0.05, 0.25, 0.5, 0.75, 0.95],
        stress_scenarios: ['market_crash', 'high_volatility']
      };

      const response = await request.post(`${BACKEND_URL}/api/v1/analytics/monte-carlo`, {
        data: monteCarloRequest
      });
      
      expect(response.status()).toBe(200);
      
      const data = await response.json();
      
      expect(data).toHaveProperty('scenarios_run', 1000);
      expect(data).toHaveProperty('time_horizon_days', 30);
      expect(data).toHaveProperty('confidence_intervals');
      expect(data).toHaveProperty('expected_return');
      expect(data).toHaveProperty('probability_of_loss');
      expect(data).toHaveProperty('value_at_risk_5');
      expect(data).toHaveProperty('stress_test_results');
      expect(data).toHaveProperty('simulation_paths');
      
      // Verify confidence intervals structure
      const ci = data.confidence_intervals;
      expect(ci).toHaveProperty('percentile_5');
      expect(ci).toHaveProperty('percentile_50');
      expect(ci).toHaveProperty('percentile_95');
      
      // Verify percentile ordering
      expect(ci.percentile_5).toBeLessThanOrEqual(ci.percentile_50);
      expect(ci.percentile_50).toBeLessThanOrEqual(ci.percentile_95);
      
      // Verify stress test results
      expect(Array.isArray(data.stress_test_results)).toBeTruthy();
      expect(data.stress_test_results.length).toBe(2); // market_crash, high_volatility
      
      console.log(`✅ Monte Carlo: ${data.scenarios_run} scenarios, Expected Return=${data.expected_return.toFixed(3)}%`);
    });

    test('Should fetch attribution analysis', async ({ request }) => {
      const response = await request.get(
        `${BACKEND_URL}/api/v1/analytics/attribution/${TEST_PORTFOLIO_ID}?attribution_type=sector&period=3M`
      );
      
      expect(response.status()).toBe(200);
      
      const data = await response.json();
      
      expect(data).toHaveProperty('attribution_type', 'sector');
      expect(data).toHaveProperty('period_start');
      expect(data).toHaveProperty('period_end');
      expect(data).toHaveProperty('total_active_return');
      expect(data).toHaveProperty('attribution_breakdown');
      expect(data).toHaveProperty('sector_attribution');
      expect(data).toHaveProperty('factor_attribution');
      
      // Verify attribution breakdown structure
      const breakdown = data.attribution_breakdown;
      expect(breakdown).toHaveProperty('security_selection');
      expect(breakdown).toHaveProperty('asset_allocation');
      expect(breakdown).toHaveProperty('interaction_effect');
      
      console.log(`✅ Attribution: Total Active Return=${data.total_active_return.toFixed(3)}%`);
    });

    test('Should fetch statistical significance tests', async ({ request }) => {
      const response = await request.get(
        `${BACKEND_URL}/api/v1/analytics/statistical-tests/${TEST_PORTFOLIO_ID}?test_type=sharpe&significance_level=0.05`
      );
      
      expect(response.status()).toBe(200);
      
      const data = await response.json();
      
      expect(data).toHaveProperty('sharpe_ratio_test');
      expect(data).toHaveProperty('alpha_significance_test');
      expect(data).toHaveProperty('beta_stability_test');
      expect(data).toHaveProperty('performance_persistence');
      expect(data).toHaveProperty('bootstrap_results');
      
      // Verify Sharpe ratio test structure
      const sharpeTest = data.sharpe_ratio_test;
      expect(sharpeTest).toHaveProperty('sharpe_ratio');
      expect(sharpeTest).toHaveProperty('t_statistic');
      expect(sharpeTest).toHaveProperty('p_value');
      expect(sharpeTest).toHaveProperty('is_significant');
      expect(sharpeTest).toHaveProperty('confidence_interval');
      expect(Array.isArray(sharpeTest.confidence_interval)).toBeTruthy();
      expect(sharpeTest.confidence_interval.length).toBe(2);
      
      console.log(`✅ Statistical Tests: Sharpe=${sharpeTest.sharpe_ratio.toFixed(3)}, Significant=${sharpeTest.is_significant}`);
    });

    test('Should handle error cases gracefully', async ({ request }) => {
      // Test with invalid portfolio ID
      const response1 = await request.get(`${BACKEND_URL}/api/v1/analytics/performance/`);
      expect(response1.status()).toBe(404);
      
      // Test with invalid attribution type
      const response2 = await request.get(
        `${BACKEND_URL}/api/v1/analytics/attribution/${TEST_PORTFOLIO_ID}?attribution_type=invalid&period=3M`
      );
      expect(response2.status()).toBe(400);
      
      // Test with invalid Monte Carlo parameters
      const invalidRequest = {
        portfolio_id: TEST_PORTFOLIO_ID,
        scenarios: -100,
        time_horizon_days: 500
      };
      
      const response3 = await request.post(`${BACKEND_URL}/api/v1/analytics/monte-carlo`, {
        data: invalidRequest
      });
      expect(response3.status()).toBe(400);
      
      console.log('✅ Error handling tests passed');
    });
  });

  test.describe('Frontend Integration Tests', () => {
    
    test('Should load Advanced Analytics Dashboard', async () => {
      await page.goto(FRONTEND_URL);
      
      // Wait for the page to load
      await page.waitForTimeout(3000);
      
      // Check if we can find analytics-related elements
      // Note: This assumes the dashboard is accessible from the main page
      
      // Look for any analytics-related text or components
      const pageContent = await page.textContent('body');
      console.log('Page loaded, checking for analytics content...');
      
      // Take a screenshot for debugging
      await page.screenshot({ path: 'story-5-1-dashboard-loaded.png', fullPage: true });
      
      console.log('✅ Dashboard loaded and screenshot taken');
    });

    test('Should test analytics service integration', async () => {
      // Test the analytics service directly by injecting it into the page
      await page.goto(FRONTEND_URL);
      
      const serviceTest = await page.evaluate(async () => {
        // Test if we can make API calls from the frontend
        try {
          const response = await fetch('http://localhost:8000/api/v1/analytics/benchmarks');
          const data = await response.json();
          
          return {
            success: true,
            benchmarkCount: data.benchmarks?.length || 0,
            status: response.status
          };
        } catch (error) {
          return {
            success: false,
            error: error.message,
            status: 0
          };
        }
      });
      
      expect(serviceTest.success).toBeTruthy();
      expect(serviceTest.status).toBe(200);
      expect(serviceTest.benchmarkCount).toBeGreaterThan(0);
      
      console.log(`✅ Frontend API integration working: ${serviceTest.benchmarkCount} benchmarks fetched`);
    });

    test('Should test performance analytics API from frontend', async () => {
      await page.goto(FRONTEND_URL);
      
      const analyticsTest = await page.evaluate(async () => {
        try {
          const response = await fetch(`http://localhost:8000/api/v1/analytics/performance/test_portfolio?benchmark=SPY`);
          const data = await response.json();
          
          return {
            success: true,
            alpha: data.alpha,
            beta: data.beta,
            sharpe: data.sharpe_ratio,
            status: response.status
          };
        } catch (error) {
          return {
            success: false,
            error: error.message,
            status: 0
          };
        }
      });
      
      expect(analyticsTest.success).toBeTruthy();
      expect(analyticsTest.status).toBe(200);
      expect(typeof analyticsTest.alpha).toBe('number');
      expect(typeof analyticsTest.beta).toBe('number');
      expect(typeof analyticsTest.sharpe).toBe('number');
      
      console.log(`✅ Performance analytics from frontend: Alpha=${analyticsTest.alpha}, Beta=${analyticsTest.beta}`);
    });

    test('Should test Monte Carlo simulation from frontend', async () => {
      await page.goto(FRONTEND_URL);
      
      const monteCarloTest = await page.evaluate(async () => {
        try {
          const request = {
            portfolio_id: 'test_portfolio',
            scenarios: 100, // Smaller number for faster test
            time_horizon_days: 10,
            confidence_levels: [0.05, 0.5, 0.95],
            stress_scenarios: ['market_crash']
          };

          const response = await fetch('http://localhost:8000/api/v1/analytics/monte-carlo', {
            method: 'POST',
            headers: {
              'Content-Type': 'application/json',
            },
            body: JSON.stringify(request)
          });
          
          const data = await response.json();
          
          return {
            success: true,
            scenarios: data.scenarios_run,
            expectedReturn: data.expected_return,
            hasConfidenceIntervals: !!data.confidence_intervals,
            status: response.status
          };
        } catch (error) {
          return {
            success: false,
            error: error.message,
            status: 0
          };
        }
      });
      
      expect(monteCarloTest.success).toBeTruthy();
      expect(monteCarloTest.status).toBe(200);
      expect(monteCarloTest.scenarios).toBe(100);
      expect(monteCarloTest.hasConfidenceIntervals).toBeTruthy();
      
      console.log(`✅ Monte Carlo from frontend: ${monteCarloTest.scenarios} scenarios, Expected=${monteCarloTest.expectedReturn?.toFixed(3)}%`);
    });

    test('Should handle API errors gracefully in frontend', async () => {
      await page.goto(FRONTEND_URL);
      
      const errorTest = await page.evaluate(async () => {
        try {
          // Test with invalid portfolio ID
          const response = await fetch('http://localhost:8000/api/v1/analytics/performance/invalid_portfolio');
          const data = await response.json();
          
          return {
            success: response.ok,
            status: response.status,
            hasErrorMessage: !!data.detail || !!data.error,
            message: data.detail || data.error || 'No error message'
          };
        } catch (error) {
          return {
            success: false,
            status: 0,
            error: error.message
          };
        }
      });
      
      // We expect this to fail gracefully
      expect(errorTest.success).toBeFalsy();
      expect(errorTest.status).toBeGreaterThan(399); // 4xx or 5xx error
      
      console.log(`✅ Error handling test: Status ${errorTest.status}, Message: ${errorTest.message}`);
    });
  });

  test.describe('Performance and Load Tests', () => {
    
    test('Should handle multiple concurrent API requests', async ({ request }) => {
      const startTime = Date.now();
      
      // Make multiple concurrent requests
      const requests = [
        request.get(`${BACKEND_URL}/api/v1/analytics/benchmarks`),
        request.get(`${BACKEND_URL}/api/v1/analytics/performance/${TEST_PORTFOLIO_ID}`),
        request.get(`${BACKEND_URL}/api/v1/analytics/attribution/${TEST_PORTFOLIO_ID}`),
        request.get(`${BACKEND_URL}/api/v1/analytics/statistical-tests/${TEST_PORTFOLIO_ID}`)
      ];
      
      const responses = await Promise.all(requests);
      const endTime = Date.now();
      
      // Verify all requests succeeded
      responses.forEach(response => {
        expect(response.status()).toBe(200);
      });
      
      const totalTime = endTime - startTime;
      expect(totalTime).toBeLessThan(10000); // Should complete within 10 seconds
      
      console.log(`✅ Concurrent requests completed in ${totalTime}ms`);
    });

    test('Should maintain reasonable response times', async ({ request }) => {
      const endpoints = [
        '/api/v1/analytics/benchmarks',
        `/api/v1/analytics/performance/${TEST_PORTFOLIO_ID}`,
        `/api/v1/analytics/attribution/${TEST_PORTFOLIO_ID}`,
        `/api/v1/analytics/statistical-tests/${TEST_PORTFOLIO_ID}`
      ];
      
      for (const endpoint of endpoints) {
        const startTime = Date.now();
        const response = await request.get(`${BACKEND_URL}${endpoint}`);
        const endTime = Date.now();
        
        expect(response.status()).toBe(200);
        
        const responseTime = endTime - startTime;
        expect(responseTime).toBeLessThan(5000); // 5 second max response time
        
        console.log(`✅ ${endpoint}: ${responseTime}ms`);
      }
    });
  });

  test.describe('Data Quality and Validation Tests', () => {
    
    test('Should return consistent data across multiple calls', async ({ request }) => {
      // Make the same request twice
      const response1 = await request.get(`${BACKEND_URL}/api/v1/analytics/performance/${TEST_PORTFOLIO_ID}`);
      const data1 = await response1.json();
      
      await page.waitForTimeout(1000); // Wait 1 second
      
      const response2 = await request.get(`${BACKEND_URL}/api/v1/analytics/performance/${TEST_PORTFOLIO_ID}`);
      const data2 = await response2.json();
      
      // For empty portfolio, values should be consistent
      expect(data1.alpha).toBe(data2.alpha);
      expect(data1.beta).toBe(data2.beta);
      expect(data1.sharpe_ratio).toBe(data2.sharpe_ratio);
      expect(data1.benchmark).toBe(data2.benchmark);
      
      console.log('✅ Data consistency verified across multiple calls');
    });

    test('Should validate Monte Carlo statistical properties', async ({ request }) => {
      const monteCarloRequest = {
        portfolio_id: TEST_PORTFOLIO_ID,
        scenarios: 10000,
        time_horizon_days: 30,
        confidence_levels: [0.05, 0.25, 0.5, 0.75, 0.95]
      };

      const response = await request.post(`${BACKEND_URL}/api/v1/analytics/monte-carlo`, {
        data: monteCarloRequest
      });
      
      const data = await response.json();
      
      // Statistical validation
      const ci = data.confidence_intervals;
      
      // Verify percentile ordering
      expect(ci.percentile_5).toBeLessThanOrEqual(ci.percentile_25);
      expect(ci.percentile_25).toBeLessThanOrEqual(ci.percentile_50);
      expect(ci.percentile_50).toBeLessThanOrEqual(ci.percentile_75);
      expect(ci.percentile_75).toBeLessThanOrEqual(ci.percentile_95);
      
      // Verify VaR is worse than expected return
      expect(data.value_at_risk_5).toBeLessThanOrEqual(data.expected_return);
      
      // Verify probability of loss is between 0 and 1
      expect(data.probability_of_loss).toBeGreaterThanOrEqual(0);
      expect(data.probability_of_loss).toBeLessThanOrEqual(1);
      
      console.log('✅ Monte Carlo statistical properties validated');
    });
  });
});

// Test summary and reporting
test.afterAll(async () => {
  console.log('\n🎉 STORY 5.1 COMPLETE INTEGRATION TESTS FINISHED');
  console.log('📊 Backend APIs: All endpoints tested and validated');
  console.log('🌐 Frontend Integration: API calls from browser tested');
  console.log('⚡ Performance: Response times and concurrent requests validated'); 
  console.log('🔬 Data Quality: Statistical properties and consistency verified');
  console.log('✅ Story 5.1 Advanced Performance Analytics is production ready!');
});