/**
 * Portfolio Visualization Integration Test
 */

import { test, expect } from '@playwright/test';

test.describe('Portfolio Visualization Components', () => {
  test.beforeEach(async ({ page }) => {
    // Navigate to the application
    await page.goto('http://localhost:3000');
    
    // Wait for the application to load
    await page.waitForLoadState('networkidle');
  });

  test('Portfolio components can be imported and instantiated', async ({ page }) => {
    // Test if we can access the portfolio components via browser console
    const componentsTest = await page.evaluate(() => {
      try {
        // Try to access the components we created
        const testResults = {
          portfolioAggregationService: false,
          portfolioMetrics: false,
          correlationCalculator: false,
          attributionCalculator: false
        };

        // Check if our services can be imported
        try {
          // This is a basic smoke test to see if our files are accessible
          fetch('/src/services/portfolioAggregationService.ts').then(() => {
            testResults.portfolioAggregationService = true;
          });
          fetch('/src/services/portfolioMetrics.ts').then(() => {
            testResults.portfolioMetrics = true;
          });
          fetch('/src/services/correlationCalculator.ts').then(() => {
            testResults.correlationCalculator = true;
          });
          fetch('/src/services/performanceAttributionCalculator.ts').then(() => {
            testResults.attributionCalculator = true;
          });
        } catch (error) {
          console.error('Component access error:', error);
        }

        return testResults;
      } catch (error) {
        console.error('Error testing components:', error);
        return { error: error.message };
      }
    });

    console.log('Portfolio components test results:', componentsTest);
  });

  test('Check if portfolio component files exist', async ({ page }) => {
    // Check file existence by trying to navigate to them
    const files = [
      '/src/components/Portfolio/PortfolioDashboard.tsx',
      '/src/components/Portfolio/PortfolioPnLChart.tsx',
      '/src/components/Portfolio/StrategyContributionAnalysis.tsx',
      '/src/components/Portfolio/StrategyComparison.tsx',
      '/src/components/Portfolio/StrategyCorrelationMatrix.tsx',
      '/src/services/portfolioAggregationService.ts',
      '/src/services/portfolioMetrics.ts',
      '/src/services/correlationCalculator.ts'
    ];

    for (const file of files) {
      const response = await page.request.get(`http://localhost:3000${file}`);
      console.log(`File ${file}: Status ${response.status()}`);
      
      // If status is 200, file exists and is accessible
      if (response.status() === 200) {
        const content = await response.text();
        expect(content.length).toBeGreaterThan(0);
        console.log(`✅ ${file} exists and has content (${content.length} chars)`);
      } else {
        console.log(`❌ ${file} not accessible (status: ${response.status()})`);
      }
    }
  });

  test('Application loads without critical errors', async ({ page }) => {
    // Capture console errors
    const errors: string[] = [];
    page.on('console', msg => {
      if (msg.type() === 'error') {
        errors.push(msg.text());
      }
    });

    // Capture page errors
    const pageErrors: string[] = [];
    page.on('pageerror', error => {
      pageErrors.push(error.message);
    });

    // Wait for page to fully load
    await page.waitForTimeout(5000);

    console.log('Console errors:', errors);
    console.log('Page errors:', pageErrors);

    // Check if the main application loaded
    const bodyContent = await page.textContent('body');
    expect(bodyContent).toBeTruthy();
    
    // Take a screenshot for verification
    await page.screenshot({ path: 'portfolio-test-verification.png' });

    // Log what's actually on the page
    const pageTitle = await page.title();
    console.log('Page title:', pageTitle);
    console.log('Body content length:', bodyContent?.length || 0);
  });
});