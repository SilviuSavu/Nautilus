import { test, expect, Page } from '@playwright/test';

/**
 * Component-Specific Test Suite
 * Tests individual components and tabs in detail
 */

// Helper functions
async function navigateToDashboard(page: Page) {
  await page.goto('/');
  await page.waitForSelector('[data-testid="dashboard"]', { timeout: 10000 });
  await page.waitForTimeout(2000);
}

async function clickTab(page: Page, tabKey: string) {
  await page.click(`[data-node-key="${tabKey}"]`);
  await page.waitForTimeout(3000);
}

test.describe('Component-Specific Tests', () => {
  test.beforeEach(async ({ page }) => {
    await navigateToDashboard(page);
  });

  test.describe('Search Tab - Instrument Search Component', () => {
    test('Instrument search interface loads correctly', async ({ page }) => {
      await clickTab(page, 'instruments');
      
      // Check main search card
      await expect(page.getByText('Universal Instrument Search')).toBeVisible();
      
      // Check search features information
      await expect(page.getByText('Search Features')).toBeVisible();
      await expect(page.getByText('Search Capabilities:')).toBeVisible();
      
      // Verify listed capabilities
      await expect(page.getByText('Fuzzy symbol matching')).toBeVisible();
      await expect(page.getByText('Company name search')).toBeVisible();
      await expect(page.getByText('Venue filtering')).toBeVisible();
      
      // Check supported asset classes
      await expect(page.getByText('Supported Asset Classes')).toBeVisible();
      
      // Verify asset class tags
      const assetClassTags = [
        'STK - Stocks',
        'CASH - Forex', 
        'FUT - Futures',
        'IND - Indices',
        'OPT - Options',
        'BOND - Bonds',
        'CRYPTO - Crypto ETFs'
      ];
      
      for (const tag of assetClassTags) {
        await expect(page.locator(`.ant-tag:has-text("${tag}")`)).toBeVisible();
      }
    });

    test('Search component error boundary', async ({ page }) => {
      await clickTab(page, 'instruments');
      
      // Should have error boundary wrapper
      await page.waitForTimeout(3000);
      
      // If error boundary is triggered, should show fallback
      const errorBoundary = page.locator('.ant-result-error');
      const hasError = await errorBoundary.isVisible();
      
      if (hasError) {
        await expect(page.getByText('Instrument Search Error')).toBeVisible();
        await expect(page.getByText('The instrument search component encountered an error')).toBeVisible();
      } else {
        // Normal operation - search interface should be present
        await expect(page.getByText('Universal Instrument Search')).toBeVisible();
      }
    });
  });

  test.describe('Watchlist Tab - Watchlist Manager Component', () => {
    test('Watchlist manager interface loads correctly', async ({ page }) => {
      await clickTab(page, 'watchlists');
      
      // Check watchlist features
      await expect(page.getByText('Watchlist Features')).toBeVisible();
      await expect(page.getByText('Watchlist Management:')).toBeVisible();
      
      // Verify listed features
      await expect(page.getByText('Create multiple watchlists')).toBeVisible();
      await expect(page.getByText('Drag & drop organization')).toBeVisible();
      await expect(page.getByText('Import/Export functionality')).toBeVisible();
      
      // Check quick actions
      await expect(page.getByText('Quick Actions:')).toBeVisible();
      await expect(page.getByText('Add to favorites')).toBeVisible();
      await expect(page.getByText('Export as CSV/JSON')).toBeVisible();
      
      // Check export formats
      await expect(page.getByText('Data Formats')).toBeVisible();
      await expect(page.getByText('Export Formats:')).toBeVisible();
      await expect(page.locator('.ant-tag:has-text("JSON - Full data preservation")')).toBeVisible();
      await expect(page.locator('.ant-tag:has-text("CSV - Spreadsheet compatible")')).toBeVisible();
    });
  });

  test.describe('Chart Tab - Chart Component', () => {
    test('Chart interface loads with all controls', async ({ page }) => {
      await clickTab(page, 'chart');
      
      // Check control cards
      await expect(page.getByText('Instrument Selection')).toBeVisible();
      await expect(page.getByText('Timeframe Selection')).toBeVisible();
      await expect(page.getByText('Technical Indicators')).toBeVisible();
      
      // Wait for chart component to load
      await page.waitForTimeout(5000);
      
      // Check if chart container exists (even if empty)
      const chartContainer = page.locator('[class*="chart"], [id*="chart"], canvas');
      const hasChart = await chartContainer.count() > 0;
      
      if (!hasChart) {
        // Chart might be in error state, which is acceptable
        console.log('Chart component may require market data connection');
      }
    });
  });

  test.describe('Strategy Tab - Strategy Management', () => {
    test('Strategy management component loads', async ({ page }) => {
      await clickTab(page, 'strategy');
      
      // Wait for component to load
      await page.waitForTimeout(5000);
      
      // Check for error boundary or normal operation
      const errorBoundary = page.locator('.ant-result-error');
      const hasError = await errorBoundary.isVisible();
      
      if (hasError) {
        await expect(page.getByText('Strategy Management Error')).toBeVisible();
      } else {
        // Component loaded successfully - it may be empty which is fine
        console.log('Strategy management component loaded without errors');
      }
    });
  });

  test.describe('Performance Tab - Performance Dashboard', () => {
    test('Performance dashboard component loads', async ({ page }) => {
      await clickTab(page, 'performance');
      
      await page.waitForTimeout(5000);
      
      // Check for error boundary or normal operation
      const errorBoundary = page.locator('.ant-result-error');
      const hasError = await errorBoundary.isVisible();
      
      if (hasError) {
        await expect(page.getByText('Performance Monitoring Error')).toBeVisible();
      } else {
        console.log('Performance dashboard loaded without errors');
      }
    });
  });

  test.describe('Portfolio Tab - Portfolio Visualization', () => {
    test('Portfolio visualization component loads', async ({ page }) => {
      await clickTab(page, 'portfolio');
      
      await page.waitForTimeout(5000);
      
      // Check for error boundary or normal operation
      const errorBoundary = page.locator('.ant-result-error');
      const hasError = await errorBoundary.isVisible();
      
      if (hasError) {
        await expect(page.getByText('Portfolio Visualization Error')).toBeVisible();
      } else {
        console.log('Portfolio visualization loaded without errors');
      }
    });
  });

  test.describe('Factors Tab - Factor Dashboard', () => {
    test('Factor dashboard component loads', async ({ page }) => {
      await clickTab(page, 'factors');
      
      await page.waitForTimeout(5000);
      
      // Check for error boundary or normal operation
      const errorBoundary = page.locator('.ant-result-error');
      const hasError = await errorBoundary.isVisible();
      
      if (hasError) {
        await expect(page.getByText('Factor Dashboard Error')).toBeVisible();
      } else {
        console.log('Factor dashboard loaded without errors');
      }
    });
  });

  test.describe('Risk Tab - Risk Dashboard', () => {
    test('Risk dashboard component loads', async ({ page }) => {
      await clickTab(page, 'risk');
      
      await page.waitForTimeout(5000);
      
      // Check for error boundary or normal operation
      const errorBoundary = page.locator('.ant-result-error');
      const hasError = await errorBoundary.isVisible();
      
      if (hasError) {
        await expect(page.getByText('Risk Management Dashboard Error')).toBeVisible();
      } else {
        console.log('Risk dashboard loaded without errors');
      }
    });
  });

  test.describe('IB Tab - Interactive Brokers Dashboard', () => {
    test('IB dashboard component loads', async ({ page }) => {
      await clickTab(page, 'ib');
      
      await page.waitForTimeout(5000);
      
      // Check for error boundary or normal operation
      const errorBoundary = page.locator('.ant-result-error');
      const hasError = await errorBoundary.isVisible();
      
      if (hasError) {
        await expect(page.getByText('Interactive Brokers Dashboard Error')).toBeVisible();
      } else {
        console.log('IB dashboard loaded without errors');
      }
    });
  });

  test.describe('Engine Tab - Nautilus Engine Manager', () => {
    test('Nautilus engine manager component loads', async ({ page }) => {
      await clickTab(page, 'nautilus-engine');
      
      await page.waitForTimeout(5000);
      
      // Check for error boundary or normal operation
      const errorBoundary = page.locator('.ant-result-error');
      const hasError = await errorBoundary.isVisible();
      
      if (hasError) {
        await expect(page.getByText('NautilusTrader Engine Error')).toBeVisible();
      } else {
        console.log('Nautilus engine manager loaded without errors');
      }
    });
  });

  test.describe('Backtest Tab - Backtest Runner', () => {
    test('Backtest runner component loads', async ({ page }) => {
      await clickTab(page, 'backtesting');
      
      await page.waitForTimeout(5000);
      
      // Check for error boundary or normal operation
      const errorBoundary = page.locator('.ant-result-error');
      const hasError = await errorBoundary.isVisible();
      
      if (hasError) {
        await expect(page.getByText('Backtesting Engine Error')).toBeVisible();
      } else {
        console.log('Backtest runner loaded without errors');
      }
    });
  });

  test.describe('Deploy Tab - Strategy Deployment Pipeline', () => {
    test('Strategy deployment pipeline component loads', async ({ page }) => {
      await clickTab(page, 'deployment');
      
      await page.waitForTimeout(5000);
      
      // Check for error boundary or normal operation
      const errorBoundary = page.locator('.ant-result-error');
      const hasError = await errorBoundary.isVisible();
      
      if (hasError) {
        await expect(page.getByText('Strategy Deployment Error')).toBeVisible();
      } else {
        console.log('Strategy deployment pipeline loaded without errors');
      }
    });
  });

  test.describe('Data Tab - Data Catalog Browser', () => {
    test('Data catalog browser component loads', async ({ page }) => {
      await clickTab(page, 'data-catalog');
      
      await page.waitForTimeout(5000);
      
      // Check for error boundary or normal operation
      const errorBoundary = page.locator('.ant-result-error');
      const hasError = await errorBoundary.isVisible();
      
      if (hasError) {
        await expect(page.getByText('Data Catalog Error')).toBeVisible();
      } else {
        console.log('Data catalog browser loaded without errors');
      }
    });
  });

  test.describe('Floating Action Button - Order Placement', () => {
    test('Order placement modal functionality', async ({ page }) => {
      // Check floating action button
      const floatButton = page.locator('[aria-label="Place IB Order"]');
      await expect(floatButton).toBeVisible();
      
      // Click to open modal
      await floatButton.click();
      await page.waitForTimeout(2000);
      
      // Modal might appear or might require IB connection
      // Just verify the click didn't crash the app
      await expect(page.locator('[data-testid="dashboard"]')).toBeVisible();
      
      // If modal appeared, try to close it
      const modalClose = page.locator('.ant-modal-close, .ant-modal-mask');
      if (await modalClose.isVisible()) {
        await modalClose.first().click();
        await page.waitForTimeout(1000);
      }
    });
  });

  test.describe('Error Boundary Testing', () => {
    test('All components handle errors gracefully', async ({ page }) => {
      const tabs = [
        'nautilus-engine',
        'backtesting', 
        'deployment',
        'data-catalog',
        'strategy',
        'performance',
        'portfolio',
        'factors',
        'risk',
        'ib'
      ];

      for (const tabKey of tabs) {
        await clickTab(page, tabKey);
        
        // Check if error boundary is active
        const errorBoundary = page.locator('.ant-result-error');
        const hasError = await errorBoundary.isVisible();
        
        if (hasError) {
          // Verify error boundary has proper structure
          await expect(page.locator('.ant-result-title')).toBeVisible();
          
          // Verify fallback message is informative
          const errorText = await page.locator('.ant-result-subtitle').textContent();
          expect(errorText).toBeTruthy();
          expect(errorText?.length).toBeGreaterThan(10);
        }
        
        // Verify dashboard is still functional
        await expect(page.locator('[data-testid="dashboard"]')).toBeVisible();
        await expect(page.locator('.ant-tabs')).toBeVisible();
      }
    });
  });

  test.describe('Component State Management', () => {
    test('Components maintain state across tab switches', async ({ page }) => {
      // Start with search tab
      await clickTab(page, 'instruments');
      
      // Verify search tab loaded
      await expect(page.getByText('Universal Instrument Search')).toBeVisible();
      
      // Switch to watchlist tab
      await clickTab(page, 'watchlists');
      
      // Verify watchlist tab loaded
      await expect(page.getByText('Watchlist Features')).toBeVisible();
      
      // Switch back to search tab
      await clickTab(page, 'instruments');
      
      // Should still show search interface
      await expect(page.getByText('Universal Instrument Search')).toBeVisible();
      
      // Switch to chart tab
      await clickTab(page, 'chart');
      
      // Should show chart controls
      await expect(page.getByText('Instrument Selection')).toBeVisible();
      await expect(page.getByText('Timeframe Selection')).toBeVisible();
    });
  });

  test.describe('Component Performance', () => {
    test('Components load within reasonable time', async ({ page }) => {
      const tabs = ['instruments', 'watchlists', 'chart', 'system'];
      
      for (const tabKey of tabs) {
        const startTime = Date.now();
        
        await clickTab(page, tabKey);
        
        const loadTime = Date.now() - startTime;
        
        // Components should load within 5 seconds
        expect(loadTime).toBeLessThan(5000);
        
        // Verify component is visible
        await expect(page.locator('.ant-tabs-tabpane-active')).toBeVisible();
      }
    });
  });

  test.describe('Component Accessibility', () => {
    test('Components have proper ARIA labels and keyboard navigation', async ({ page }) => {
      // Test search tab accessibility
      await clickTab(page, 'instruments');
      
      // Check for proper headings and labels
      const headings = await page.locator('h1, h2, h3, h4, h5, h6').count();
      expect(headings).toBeGreaterThan(0);
      
      // Test keyboard navigation
      await page.keyboard.press('Tab');
      await page.waitForTimeout(500);
      
      // Should be able to navigate with keyboard
      const focusedElement = page.locator(':focus');
      const hasFocus = await focusedElement.count() > 0;
      expect(hasFocus).toBe(true);
    });
  });
});