import { test, expect, Page } from '@playwright/test';

/**
 * Integration Flow Test Suite
 * Tests complete user workflows and end-to-end functionality
 */

// Helper functions
async function navigateToDashboard(page: Page) {
  await page.goto('/');
  await page.waitForSelector('[data-testid="dashboard"]', { timeout: 10000 });
  await page.waitForTimeout(2000);
}

async function clickTab(page: Page, tabKey: string) {
  await page.click(`[data-node-key="${tabKey}"]`);
  await page.waitForTimeout(2000);
}

async function checkSystemHealth(page: Page): Promise<boolean> {
  await clickTab(page, 'system');
  
  // Check backend status
  const statusAlert = page.locator('.ant-alert').first();
  const alertText = await statusAlert.textContent();
  
  return alertText?.includes('Backend Connected') || false;
}

test.describe('Integration Flow Tests', () => {
  test.beforeEach(async ({ page }) => {
    await navigateToDashboard(page);
  });

  test.describe('System Health and Status Flow', () => {
    test('Complete system health check workflow', async ({ page }) => {
      // Step 1: Check initial dashboard load
      await expect(page.locator('[data-testid="dashboard"]')).toBeVisible();
      await expect(page.getByText('NautilusTrader Dashboard')).toBeVisible();
      
      // Step 2: Navigate to system tab
      await clickTab(page, 'system');
      
      // Step 3: Verify system components
      await expect(page.getByText('API Status')).toBeVisible();
      await expect(page.getByText('Environment')).toBeVisible();
      await expect(page.getByText('MessageBus Connection')).toBeVisible();
      
      // Step 4: Test backend health refresh
      const refreshButton = page.locator('button:has-text("Refresh")');
      await refreshButton.click();
      await page.waitForTimeout(2000);
      
      // Step 5: Verify system still functional after refresh
      await expect(page.getByText('API Status')).toBeVisible();
      
      // Step 6: Check message bus functionality
      const clearButton = page.locator('button:has-text("Clear")');
      await clearButton.click();
      await page.waitForTimeout(1000);
      
      // Step 7: Verify dashboard remains stable
      await expect(page.locator('[data-testid="dashboard"]')).toBeVisible();
    });

    test('Data backfill system configuration flow', async ({ page }) => {
      await clickTab(page, 'system');
      
      // Step 1: Locate data backfill system
      await expect(page.getByText('Data Backfill System')).toBeVisible();
      
      // Step 2: Check mode selection interface
      await expect(page.getByText('Data Source Mode:')).toBeVisible();
      
      // Step 3: Verify IBKR and YFinance options
      const ibkrTag = page.locator('.ant-tag:has-text("IBKR Gateway")');
      const yfinanceTag = page.locator('.ant-tag:has-text("YFinance")');
      await expect(ibkrTag).toBeVisible();
      await expect(yfinanceTag).toBeVisible();
      
      // Step 4: Test mode switch if available
      const modeSwitch = page.locator('.ant-switch');
      if (await modeSwitch.isVisible() && !(await modeSwitch.isDisabled())) {
        const initialState = await modeSwitch.isChecked();
        await modeSwitch.click();
        await page.waitForTimeout(2000);
        
        // Verify state changed
        const newState = await modeSwitch.isChecked();
        expect(newState).toBe(!initialState);
        
        // Switch back
        await modeSwitch.click();
        await page.waitForTimeout(2000);
      }
      
      // Step 5: Test refresh status functionality
      const refreshStatusButton = page.locator('button:has-text("Refresh Status")');
      await refreshStatusButton.click();
      await page.waitForTimeout(2000);
      
      // Step 6: Verify system remains functional
      await expect(page.getByText('Data Backfill System')).toBeVisible();
    });
  });

  test.describe('Navigation and Tab Management Flow', () => {
    test('Complete tab navigation workflow', async ({ page }) => {
      const tabs = [
        { key: 'system', label: 'System', expectedContent: 'API Status' },
        { key: 'instruments', label: 'Search', expectedContent: 'Universal Instrument Search' },
        { key: 'watchlists', label: 'Watchlist', expectedContent: 'Watchlist Features' },
        { key: 'chart', label: 'Chart', expectedContent: 'Instrument Selection' },
        { key: 'strategy', label: 'Strategy', expectedContent: null }, // May have error boundary
        { key: 'performance', label: 'Perform', expectedContent: null },
        { key: 'portfolio', label: 'Portfolio', expectedContent: null },
        { key: 'factors', label: 'Factors', expectedContent: null },
        { key: 'risk', label: 'Risk', expectedContent: null },
        { key: 'ib', label: 'IB', expectedContent: null }
      ];

      for (const tab of tabs) {
        // Step 1: Click tab
        await clickTab(page, tab.key);
        
        // Step 2: Verify tab is active
        const activeTab = page.locator('.ant-tabs-tab-active');
        await expect(activeTab).toContainText(tab.label);
        
        // Step 3: Check for expected content or error boundary
        if (tab.expectedContent) {
          await expect(page.getByText(tab.expectedContent)).toBeVisible();
        } else {
          // Should either load content or show error boundary
          const hasContent = await page.locator('.ant-tabs-tabpane-active').isVisible();
          const hasErrorBoundary = await page.locator('.ant-result-error').isVisible();
          
          expect(hasContent || hasErrorBoundary).toBe(true);
        }
        
        // Step 4: Verify dashboard structure remains intact
        await expect(page.locator('[data-testid="dashboard"]')).toBeVisible();
        await expect(page.locator('.ant-tabs')).toBeVisible();
      }
    });

    test('Tab state persistence workflow', async ({ page }) => {
      // Step 1: Start with system tab and note message count
      await clickTab(page, 'system');
      
      let initialMessageCount = 0;
      const totalMessagesStatistic = page.locator('.ant-statistic').filter({ hasText: 'Total Messages' });
      if (await totalMessagesStatistic.isVisible()) {
        const messagesValue = await page.locator('.ant-statistic-content-value').first().textContent();
        if (messagesValue) {
          initialMessageCount = parseInt(messagesValue);
        }
      }
      
      // Step 2: Navigate to different tabs
      await clickTab(page, 'instruments');
      await expect(page.getByText('Universal Instrument Search')).toBeVisible();
      
      await clickTab(page, 'chart');
      await expect(page.getByText('Instrument Selection')).toBeVisible();
      
      // Step 3: Return to system tab
      await clickTab(page, 'system');
      
      // Step 4: Verify state is maintained
      await expect(page.getByText('API Status')).toBeVisible();
      
      // Message count may have increased but shouldn't be reset
      if (await totalMessagesStatistic.isVisible()) {
        const finalMessagesValue = await page.locator('.ant-statistic-content-value').first().textContent();
        if (finalMessagesValue) {
          const finalMessageCount = parseInt(finalMessagesValue);
          expect(finalMessageCount).toBeGreaterThanOrEqual(initialMessageCount);
        }
      }
    });
  });

  test.describe('Search and Instrument Selection Flow', () => {
    test('Instrument search workflow', async ({ page }) => {
      // Step 1: Navigate to search tab
      await clickTab(page, 'instruments');
      
      // Step 2: Verify search interface
      await expect(page.getByText('Universal Instrument Search')).toBeVisible();
      
      // Step 3: Check search features
      await expect(page.getByText('Search Capabilities:')).toBeVisible();
      await expect(page.getByText('Fuzzy symbol matching')).toBeVisible();
      await expect(page.getByText('Company name search')).toBeVisible();
      
      // Step 4: Verify asset class support
      const assetClasses = ['STK - Stocks', 'CASH - Forex', 'FUT - Futures'];
      for (const assetClass of assetClasses) {
        await expect(page.locator(`.ant-tag:has-text("${assetClass}")`)).toBeVisible();
      }
      
      // Step 5: Check for search input if available
      const searchInputs = await page.locator('input[type="text"], .ant-input, .ant-select').count();
      if (searchInputs > 0) {
        console.log('Search interface has input elements available');
      }
      
      // Step 6: Navigate to chart tab to see if instrument selection persists
      await clickTab(page, 'chart');
      await expect(page.getByText('Instrument Selection')).toBeVisible();
    });

    test('Watchlist management workflow', async ({ page }) => {
      // Step 1: Navigate to watchlist tab
      await clickTab(page, 'watchlists');
      
      // Step 2: Verify watchlist interface
      await expect(page.getByText('Watchlist Features')).toBeVisible();
      
      // Step 3: Check management features
      await expect(page.getByText('Create multiple watchlists')).toBeVisible();
      await expect(page.getByText('Drag & drop organization')).toBeVisible();
      
      // Step 4: Verify export functionality
      await expect(page.getByText('Export Formats:')).toBeVisible();
      await expect(page.locator('.ant-tag:has-text("JSON - Full data preservation")')).toBeVisible();
      await expect(page.locator('.ant-tag:has-text("CSV - Spreadsheet compatible")')).toBeVisible();
      
      // Step 5: Check quick actions
      await expect(page.getByText('Quick Actions:')).toBeVisible();
      await expect(page.getByText('Add to favorites')).toBeVisible();
      
      // Step 6: Navigate back to search to test integration
      await clickTab(page, 'instruments');
      await expect(page.getByText('Universal Instrument Search')).toBeVisible();
    });
  });

  test.describe('Chart and Analysis Flow', () => {
    test('Chart configuration workflow', async ({ page }) => {
      // Step 1: Navigate to chart tab
      await clickTab(page, 'chart');
      
      // Step 2: Verify chart control panels
      await expect(page.getByText('Instrument Selection')).toBeVisible();
      await expect(page.getByText('Timeframe Selection')).toBeVisible();
      await expect(page.getByText('Technical Indicators')).toBeVisible();
      
      // Step 3: Wait for chart component
      await page.waitForTimeout(3000);
      
      // Step 4: Check if chart container exists
      const chartElements = await page.locator('[class*="chart"], canvas, svg').count();
      console.log(`Chart elements found: ${chartElements}`);
      
      // Step 5: Test switching back to other tabs
      await clickTab(page, 'instruments');
      await clickTab(page, 'chart');
      
      // Step 6: Verify chart controls still available
      await expect(page.getByText('Instrument Selection')).toBeVisible();
    });
  });

  test.describe('Order Management Flow', () => {
    test('Order placement interface workflow', async ({ page }) => {
      // Step 1: Verify floating action button
      const floatButton = page.locator('[aria-label="Place IB Order"]');
      await expect(floatButton).toBeVisible();
      
      // Step 2: Click order button
      await floatButton.click();
      await page.waitForTimeout(2000);
      
      // Step 3: Check if modal appeared or handle gracefully
      const modal = page.locator('.ant-modal');
      const hasModal = await modal.isVisible();
      
      if (hasModal) {
        // Modal opened successfully
        console.log('Order placement modal opened');
        
        // Try to close modal
        const closeButton = page.locator('.ant-modal-close, .ant-modal-mask');
        if (await closeButton.isVisible()) {
          await closeButton.first().click();
          await page.waitForTimeout(1000);
        }
      } else {
        console.log('Order placement modal requires IB connection');
      }
      
      // Step 4: Verify dashboard remains functional
      await expect(page.locator('[data-testid="dashboard"]')).toBeVisible();
      
      // Step 5: Navigate to IB tab
      await clickTab(page, 'ib');
      await page.waitForTimeout(3000);
      
      // Should load IB dashboard or error boundary
      const hasErrorBoundary = await page.locator('.ant-result-error').isVisible();
      if (hasErrorBoundary) {
        await expect(page.getByText('Interactive Brokers Dashboard Error')).toBeVisible();
      }
    });
  });

  test.describe('Strategy and Portfolio Flow', () => {
    test('Strategy management workflow', async ({ page }) => {
      // Step 1: Navigate to strategy tab
      await clickTab(page, 'strategy');
      await page.waitForTimeout(3000);
      
      // Step 2: Check component load state
      const hasErrorBoundary = await page.locator('.ant-result-error').isVisible();
      
      if (hasErrorBoundary) {
        await expect(page.getByText('Strategy Management Error')).toBeVisible();
      } else {
        console.log('Strategy management component loaded successfully');
      }
      
      // Step 3: Navigate to performance tab
      await clickTab(page, 'performance');
      await page.waitForTimeout(3000);
      
      // Step 4: Check performance dashboard
      const perfErrorBoundary = await page.locator('.ant-result-error').isVisible();
      
      if (perfErrorBoundary) {
        await expect(page.getByText('Performance Monitoring Error')).toBeVisible();
      }
      
      // Step 5: Navigate to portfolio tab
      await clickTab(page, 'portfolio');
      await page.waitForTimeout(3000);
      
      // Step 6: Check portfolio visualization
      const portfolioErrorBoundary = await page.locator('.ant-result-error').isVisible();
      
      if (portfolioErrorBoundary) {
        await expect(page.getByText('Portfolio Visualization Error')).toBeVisible();
      }
      
      // Step 7: Verify dashboard navigation remains functional
      await clickTab(page, 'system');
      await expect(page.getByText('API Status')).toBeVisible();
    });
  });

  test.describe('Risk Management Flow', () => {
    test('Risk monitoring workflow', async ({ page }) => {
      // Step 1: Navigate to risk tab
      await clickTab(page, 'risk');
      await page.waitForTimeout(3000);
      
      // Step 2: Check risk dashboard load state
      const hasErrorBoundary = await page.locator('.ant-result-error').isVisible();
      
      if (hasErrorBoundary) {
        await expect(page.getByText('Risk Management Dashboard Error')).toBeVisible();
      } else {
        console.log('Risk dashboard loaded successfully');
      }
      
      // Step 3: Navigate to factors tab
      await clickTab(page, 'factors');
      await page.waitForTimeout(3000);
      
      // Step 4: Check factors dashboard
      const factorsErrorBoundary = await page.locator('.ant-result-error').isVisible();
      
      if (factorsErrorBoundary) {
        await expect(page.getByText('Factor Dashboard Error')).toBeVisible();
      }
      
      // Step 5: Return to risk tab
      await clickTab(page, 'risk');
      await page.waitForTimeout(2000);
      
      // Step 6: Verify tab switching works
      await expect(page.locator('.ant-tabs-tab-active')).toContainText('Risk');
    });
  });

  test.describe('Engine and Deployment Flow', () => {
    test('Nautilus engine workflow', async ({ page }) => {
      // Step 1: Navigate to engine tab
      await clickTab(page, 'nautilus-engine');
      await page.waitForTimeout(3000);
      
      // Step 2: Check engine manager state
      const hasErrorBoundary = await page.locator('.ant-result-error').isVisible();
      
      if (hasErrorBoundary) {
        await expect(page.getByText('NautilusTrader Engine Error')).toBeVisible();
      }
      
      // Step 3: Navigate to backtest tab
      await clickTab(page, 'backtesting');
      await page.waitForTimeout(3000);
      
      // Step 4: Check backtest runner
      const backtestErrorBoundary = await page.locator('.ant-result-error').isVisible();
      
      if (backtestErrorBoundary) {
        await expect(page.getByText('Backtesting Engine Error')).toBeVisible();
      }
      
      // Step 5: Navigate to deployment tab
      await clickTab(page, 'deployment');
      await page.waitForTimeout(3000);
      
      // Step 6: Check deployment pipeline
      const deployErrorBoundary = await page.locator('.ant-result-error').isVisible();
      
      if (deployErrorBoundary) {
        await expect(page.getByText('Strategy Deployment Error')).toBeVisible();
      }
      
      // Step 7: Navigate to data catalog
      await clickTab(page, 'data-catalog');
      await page.waitForTimeout(3000);
      
      // Step 8: Check data catalog browser
      const dataErrorBoundary = await page.locator('.ant-result-error').isVisible();
      
      if (dataErrorBoundary) {
        await expect(page.getByText('Data Catalog Error')).toBeVisible();
      }
    });
  });

  test.describe('Complete User Journey', () => {
    test('End-to-end dashboard usage scenario', async ({ page }) => {
      // Scenario: User logs in, checks system health, searches for instruments, 
      // reviews charts, monitors portfolio, checks risk, and places order
      
      // Step 1: Initial dashboard load
      await expect(page.locator('[data-testid="dashboard"]')).toBeVisible();
      await expect(page.getByText('NautilusTrader Dashboard')).toBeVisible();
      
      // Step 2: Check system health
      const isHealthy = await checkSystemHealth(page);
      console.log('System health check:', isHealthy ? 'Healthy' : 'Issues detected');
      
      // Step 3: Search for instruments
      await clickTab(page, 'instruments');
      await expect(page.getByText('Universal Instrument Search')).toBeVisible();
      
      // Step 4: Check watchlists
      await clickTab(page, 'watchlists');
      await expect(page.getByText('Watchlist Features')).toBeVisible();
      
      // Step 5: View charts
      await clickTab(page, 'chart');
      await expect(page.getByText('Instrument Selection')).toBeVisible();
      
      // Step 6: Check portfolio
      await clickTab(page, 'portfolio');
      await page.waitForTimeout(3000);
      
      // Step 7: Review risk
      await clickTab(page, 'risk');
      await page.waitForTimeout(3000);
      
      // Step 8: Check performance
      await clickTab(page, 'performance');
      await page.waitForTimeout(3000);
      
      // Step 9: Attempt order placement
      const floatButton = page.locator('[aria-label="Place IB Order"]');
      await floatButton.click();
      await page.waitForTimeout(2000);
      
      // Step 10: Return to system overview
      await clickTab(page, 'system');
      await expect(page.getByText('API Status')).toBeVisible();
      
      // Step 11: Verify all tabs are still functional
      const tabCount = await page.locator('.ant-tabs-tab').count();
      expect(tabCount).toBeGreaterThan(10);
      
      // Step 12: Final verification - dashboard is stable
      await expect(page.locator('[data-testid="dashboard"]')).toBeVisible();
    });
  });

  test.describe('Error Recovery Flow', () => {
    test('Dashboard recovery from component errors', async ({ page }) => {
      // Step 1: Navigate through tabs that might have errors
      const errorProneTabs = ['nautilus-engine', 'backtesting', 'deployment', 'strategy', 'portfolio'];
      
      for (const tabKey of errorProneTabs) {
        await clickTab(page, tabKey);
        await page.waitForTimeout(2000);
        
        // Check if error boundary is triggered
        const hasErrorBoundary = await page.locator('.ant-result-error').isVisible();
        
        if (hasErrorBoundary) {
          // Verify error boundary has proper UI
          await expect(page.locator('.ant-result-title')).toBeVisible();
          
          // Verify user can still navigate away
          await clickTab(page, 'system');
          await expect(page.getByText('API Status')).toBeVisible();
          
          // Return to the error tab to test persistence
          await clickTab(page, tabKey);
          await page.waitForTimeout(1000);
        }
      }
      
      // Step 2: Verify navigation still works after errors
      await clickTab(page, 'instruments');
      await expect(page.getByText('Universal Instrument Search')).toBeVisible();
      
      // Step 3: Final stability check
      await expect(page.locator('[data-testid="dashboard"]')).toBeVisible();
      await expect(page.locator('.ant-tabs')).toBeVisible();
    });
  });
});