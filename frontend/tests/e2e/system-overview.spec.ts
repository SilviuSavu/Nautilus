import { test, expect, Page } from '@playwright/test';

/**
 * System Overview Test Suite
 * Tests system status, backend health, API endpoints, and data backfill functionality
 */

// Helper function to wait for dashboard and navigate to system tab
async function navigateToSystemTab(page: Page) {
  await page.goto('/');
  await page.waitForSelector('[data-testid="dashboard"]', { timeout: 10000 });
  await page.click('[data-node-key="system"]');
  await page.waitForTimeout(2000);
}

// Helper function to check API endpoint status
async function checkAPIEndpoint(page: Page, endpoint: string): Promise<boolean> {
  try {
    const response = await page.request.get(`http://localhost:8001${endpoint}`);
    return response.ok();
  } catch (error) {
    return false;
  }
}

test.describe('System Overview Tests', () => {
  test.beforeEach(async ({ page }) => {
    await navigateToSystemTab(page);
  });

  test('Backend status alert functionality', async ({ page }) => {
    // Check backend status alert exists
    const statusAlert = page.locator('.ant-alert').first();
    await expect(statusAlert).toBeVisible();
    
    // Should show backend connection status
    const alertText = await statusAlert.textContent();
    expect(alertText).toMatch(/(Backend Connected|Backend Disconnected|Checking Backend)/);
    
    // Test refresh button
    const refreshButton = page.locator('button:has-text("Refresh")');
    await expect(refreshButton).toBeVisible();
    
    await refreshButton.click();
    await page.waitForTimeout(2000);
    
    // Should still have status alert after refresh
    await expect(statusAlert).toBeVisible();
  });

  test('API Status card displays correct information', async ({ page }) => {
    // Check API Status card
    await expect(page.getByText('API Status')).toBeVisible();
    
    // Should show backend URL
    const backendUrlText = page.locator('text="Backend URL:"');
    await expect(backendUrlText).toBeVisible();
    
    // URL should be localhost:8001
    const urlValue = await page.locator('text="Backend URL:"').locator('..').textContent();
    expect(urlValue).toContain('localhost:8001');
    
    // Should show status
    const statusText = page.locator('text="Status:"');
    await expect(statusText).toBeVisible();
  });

  test('Environment card shows configuration', async ({ page }) => {
    // Check Environment card
    await expect(page.getByText('Environment')).toBeVisible();
    
    // Should show mode
    const modeText = page.locator('text="Mode:"');
    await expect(modeText).toBeVisible();
    
    // Should show debug setting
    const debugText = page.locator('text="Debug:"');
    await expect(debugText).toBeVisible();
  });

  test('MessageBus Connection card functionality', async ({ page }) => {
    // Check MessageBus Connection card
    await expect(page.getByText('MessageBus Connection')).toBeVisible();
    
    // Should show status with badge
    const statusBadge = page.locator('.ant-badge-status');
    await expect(statusBadge).toBeVisible();
    
    // Should have connection controls
    const connectButton = page.locator('button:has-text("Connect")');
    const disconnectButton = page.locator('button:has-text("Disconnect")');
    const clearButton = page.locator('button:has-text("Clear")');
    
    await expect(clearButton).toBeVisible();
    
    // At least one connection button should be visible
    const hasConnectionButton = await connectButton.isVisible() || await disconnectButton.isVisible();
    expect(hasConnectionButton).toBe(true);
  });

  test('Message Statistics card displays metrics', async ({ page }) => {
    // Check Message Statistics card
    await expect(page.getByText('Message Statistics')).toBeVisible();
    
    // Should show total messages statistic
    const totalMessagesStatistic = page.locator('.ant-statistic').filter({ hasText: 'Total Messages' });
    await expect(totalMessagesStatistic).toBeVisible();
    
    // Should show unique topics statistic  
    const uniqueTopicsStatistic = page.locator('.ant-statistic').filter({ hasText: 'Unique Topics' });
    await expect(uniqueTopicsStatistic).toBeVisible();
    
    // Values should be numeric
    const statisticValues = await page.locator('.ant-statistic-content-value').all();
    for (const value of statisticValues) {
      const text = await value.textContent();
      if (text) {
        expect(text).toMatch(/^\d+$/);
      }
    }
  });

  test('Data Backfill System functionality', async ({ page }) => {
    // Check Data Backfill System card
    await expect(page.getByText('Data Backfill System')).toBeVisible();
    
    // Should show backfill mode section
    await expect(page.getByText('Backfill Mode')).toBeVisible();
    
    // Should show data source mode toggle
    await expect(page.getByText('Data Source Mode:')).toBeVisible();
    
    // Should have IBKR and YFinance tags
    const ibkrTag = page.locator('.ant-tag:has-text("IBKR Gateway")');
    const yfinanceTag = page.locator('.ant-tag:has-text("YFinance")');
    
    await expect(ibkrTag).toBeVisible();
    await expect(yfinanceTag).toBeVisible();
    
    // Should have mode switch
    const modeSwitch = page.locator('.ant-switch');
    await expect(modeSwitch).toBeVisible();
  });

  test('Backfill mode switching', async ({ page }) => {
    // Find the mode switch
    const modeSwitch = page.locator('.ant-switch');
    
    if (await modeSwitch.isVisible()) {
      // Get initial state
      const isInitiallyChecked = await modeSwitch.isChecked();
      
      // Only test switching if not disabled
      const isDisabled = await modeSwitch.isDisabled();
      
      if (!isDisabled) {
        // Try to switch mode
        await modeSwitch.click();
        await page.waitForTimeout(2000);
        
        // Verify switch state changed
        const isNowChecked = await modeSwitch.isChecked();
        expect(isNowChecked).toBe(!isInitiallyChecked);
        
        // Switch back
        await modeSwitch.click();
        await page.waitForTimeout(2000);
      } else {
        console.log('Mode switch is disabled (backfill may be running)');
      }
    }
  });

  test('Backfill control buttons functionality', async ({ page }) => {
    // Check for backfill control buttons
    const startButton = page.locator('button:has-text("Start")').filter({ hasText: /Start.*Backfill/ });
    const stopButton = page.locator('button:has-text("Stop Backfill")');
    const refreshButton = page.locator('button:has-text("Refresh Status")');
    
    // At least start button should be visible
    await expect(startButton).toBeVisible();
    await expect(refreshButton).toBeVisible();
    
    // Test refresh status button
    await refreshButton.click();
    await page.waitForTimeout(2000);
    
    // Page should still be functional after refresh
    await expect(page.getByText('Data Backfill System')).toBeVisible();
  });

  test('Service status display for IBKR mode', async ({ page }) => {
    // Look for IBKR Gateway Status section
    const ibkrStatusCard = page.locator('text="IBKR Gateway Status"');
    const hasIBKRStatus = await ibkrStatusCard.isVisible();
    
    if (hasIBKRStatus) {
      // Should show various metrics
      const metricsCards = page.locator('.ant-card').filter({ hasText: /Queue Size|Active|Completed|Total Bars|Instruments|Database Size/ });
      const metricsCount = await metricsCards.count();
      
      expect(metricsCount).toBeGreaterThan(0);
      
      // Check for statistic values
      const statisticElements = await page.locator('.ant-statistic-content-value').all();
      for (const element of statisticElements) {
        const value = await element.textContent();
        if (value) {
          // Should be numeric or include units (like GB)
          expect(value).toMatch(/^[\d,.]+(GB)?$/);
        }
      }
    }
  });

  test('Service status display for YFinance mode', async ({ page }) => {
    // Switch to YFinance mode if possible
    const modeSwitch = page.locator('.ant-switch');
    
    if (await modeSwitch.isVisible() && !(await modeSwitch.isDisabled())) {
      // Ensure we're in YFinance mode
      const isChecked = await modeSwitch.isChecked();
      if (!isChecked) {
        await modeSwitch.click();
        await page.waitForTimeout(3000);
      }
      
      // Look for YFinance Service Status
      const yfinanceStatusCard = page.locator('text="YFinance Service Status"');
      const hasYFinanceStatus = await yfinanceStatusCard.isVisible();
      
      if (hasYFinanceStatus) {
        // Should show service metrics
        const serviceStatusStatistic = page.locator('.ant-statistic').filter({ hasText: 'Service Status' });
        const instrumentsLoadedStatistic = page.locator('.ant-statistic').filter({ hasText: 'Instruments Loaded' });
        const connectionStatusStatistic = page.locator('.ant-statistic').filter({ hasText: 'Connection Status' });
        
        await expect(serviceStatusStatistic).toBeVisible();
        await expect(instrumentsLoadedStatistic).toBeVisible();
        await expect(connectionStatusStatistic).toBeVisible();
        
        // Should show YFinance configuration alert
        const configAlert = page.locator('.ant-alert').filter({ hasText: 'YFinance Service Configuration' });
        await expect(configAlert).toBeVisible();
      }
    }
  });

  test('Latest message display when available', async ({ page }) => {
    // Wait for potential messages
    await page.waitForTimeout(5000);
    
    // Check if latest message card is displayed
    const latestMessageCard = page.locator('text="Latest Message"');
    const hasLatestMessage = await latestMessageCard.isVisible();
    
    if (hasLatestMessage) {
      // Verify message structure
      await expect(page.locator('text="Type:"')).toBeVisible();
      await expect(page.locator('text="Timestamp:"')).toBeVisible();
      
      // Should have JSON payload display
      const jsonPayload = page.locator('.ant-typography code');
      await expect(jsonPayload).toBeVisible();
      
      // Timestamp should be in valid format
      const timestampElement = page.locator('text="Timestamp:"').locator('..');
      const timestampText = await timestampElement.textContent();
      expect(timestampText).toMatch(/\d+:\d+:\d+/); // Should contain time format
    } else {
      console.log('No latest message displayed - normal when no messages received');
    }
  });

  test('Backend health endpoint accessibility', async ({ page }) => {
    // Test that the health endpoint is accessible
    const healthEndpointAccessible = await checkAPIEndpoint(page, '/health');
    
    if (healthEndpointAccessible) {
      // Should show connected status
      const statusAlert = page.locator('.ant-alert').first();
      const alertText = await statusAlert.textContent();
      expect(alertText).toContain('Backend Connected');
    } else {
      // Should show error status
      const statusAlert = page.locator('.ant-alert').first();
      const alertText = await statusAlert.textContent();
      expect(alertText).toMatch(/(Backend Disconnected|Checking Backend)/);
    }
  });

  test('System status indicators color coding', async ({ page }) => {
    // Check backend status alert color
    const statusAlert = page.locator('.ant-alert').first();
    const alertClass = await statusAlert.getAttribute('class');
    
    // Should have success, error, or info class
    expect(alertClass).toMatch(/(ant-alert-success|ant-alert-error|ant-alert-info)/);
    
    // Check MessageBus status badge color
    const statusBadge = page.locator('.ant-badge-status');
    if (await statusBadge.isVisible()) {
      const badgeClass = await statusBadge.getAttribute('class');
      expect(badgeClass).toMatch(/(ant-badge-status-success|ant-badge-status-error|ant-badge-status-default|ant-badge-status-processing)/);
    }
  });

  test('System tab handles API errors gracefully', async ({ page }) => {
    // Intercept API calls and return errors
    await page.route('**/api/v1/**', (route) => {
      route.fulfill({ status: 500, body: 'Internal Server Error' });
    });
    
    // Refresh the page to trigger API calls with errors
    await page.reload();
    await navigateToSystemTab(page);
    
    // Should still show the system interface
    await expect(page.getByText('API Status')).toBeVisible();
    await expect(page.getByText('Environment')).toBeVisible();
    
    // Should show error state
    const statusAlert = page.locator('.ant-alert').first();
    const alertText = await statusAlert.textContent();
    expect(alertText).toMatch(/(Backend Disconnected|Backend Error|Error)/);
  });

  test('System information persistence across refreshes', async ({ page }) => {
    // Get initial backend URL
    const urlElement = page.locator('text="Backend URL:"').locator('..');
    const initialUrl = await urlElement.textContent();
    
    // Refresh page
    await page.reload();
    await navigateToSystemTab(page);
    
    // URL should be the same
    const newUrlElement = page.locator('text="Backend URL:"').locator('..');
    const newUrl = await newUrlElement.textContent();
    
    expect(newUrl).toBe(initialUrl);
  });

  test('Responsive layout in system tab', async ({ page }) => {
    // Test mobile layout
    await page.setViewportSize({ width: 375, height: 667 });
    await page.waitForTimeout(1000);
    
    // All main cards should still be visible
    await expect(page.getByText('API Status')).toBeVisible();
    await expect(page.getByText('Environment')).toBeVisible();
    await expect(page.getByText('MessageBus Connection')).toBeVisible();
    
    // Test tablet layout
    await page.setViewportSize({ width: 768, height: 1024 });
    await page.waitForTimeout(1000);
    
    await expect(page.getByText('Data Backfill System')).toBeVisible();
    
    // Test desktop layout
    await page.setViewportSize({ width: 1920, height: 1080 });
    await page.waitForTimeout(1000);
    
    await expect(page.getByText('Message Statistics')).toBeVisible();
  });
});