import { test, expect, Page } from '@playwright/test';

/**
 * Performance and Load Test Suite
 * Tests dashboard performance, load times, memory usage, and responsiveness
 */

// Helper functions
async function navigateToDashboard(page: Page) {
  await page.goto('/');
  await page.waitForSelector('[data-testid="dashboard"]', { timeout: 10000 });
  await page.waitForTimeout(2000);
}

async function measurePageLoad(page: Page): Promise<number> {
  const startTime = Date.now();
  await page.goto('/');
  await page.waitForSelector('[data-testid="dashboard"]', { timeout: 10000 });
  return Date.now() - startTime;
}

async function measureTabSwitch(page: Page, tabKey: string): Promise<number> {
  const startTime = Date.now();
  await page.click(`[data-node-key="${tabKey}"]`);
  await page.waitForTimeout(1000);
  return Date.now() - startTime;
}

async function getMemoryUsage(page: Page): Promise<any> {
  return await page.evaluate(() => {
    if ('memory' in performance) {
      return (performance as any).memory;
    }
    return null;
  });
}

async function setupHighVolumeWebSocket(page: Page, messagesPerSecond: number = 10) {
  await page.addInitScript((mps) => {
    (window as any).WebSocket = class MockWebSocket {
      static CONNECTING = 0;
      static OPEN = 1;
      static CLOSING = 2;
      static CLOSED = 3;
      
      readyState = MockWebSocket.CONNECTING;
      onopen: any = null;
      onclose: any = null;
      onerror: any = null;
      onmessage: any = null;
      
      constructor(url: string) {
        setTimeout(() => {
          this.readyState = MockWebSocket.OPEN;
          if (this.onopen) this.onopen({ type: 'open' });
          
          // Send messages at specified rate
          let messageId = 0;
          const interval = 1000 / mps; // Convert to milliseconds per message
          
          const sendMessage = () => {
            if (this.readyState !== MockWebSocket.OPEN) return;
            
            const message = {
              type: 'messagebus',
              topic: `performance.test.${messageId % 10}`, // Cycle through 10 topics
              payload: {
                id: messageId,
                timestamp: Date.now(),
                data: `Performance test message ${messageId}`,
                randomValue: Math.random()
              },
              timestamp: Date.now() * 1000000
            };
            
            if (this.onmessage) {
              this.onmessage({ data: JSON.stringify(message) });
            }
            
            messageId++;
            setTimeout(sendMessage, interval);
          };
          
          setTimeout(sendMessage, 100);
        }, 100);
      }
      
      close() {
        this.readyState = MockWebSocket.CLOSED;
        if (this.onclose) this.onclose({ type: 'close', code: 1000 });
      }
      
      send(data: string) {}
    };
  }, messagesPerSecond);
}

test.describe('Performance and Load Tests', () => {
  test.describe('Page Load Performance', () => {
    test('Dashboard loads within acceptable time', async ({ page }) => {
      // Test initial load time
      const loadTime = await measurePageLoad(page);
      
      // Should load within 5 seconds
      expect(loadTime).toBeLessThan(5000);
      
      // Verify dashboard is fully loaded
      await expect(page.locator('[data-testid="dashboard"]')).toBeVisible();
      await expect(page.getByText('NautilusTrader Dashboard')).toBeVisible();
      
      console.log(`Dashboard load time: ${loadTime}ms`);
    });

    test('Dashboard loads consistently across multiple attempts', async ({ page }) => {
      const loadTimes: number[] = [];
      const attempts = 3;
      
      for (let i = 0; i < attempts; i++) {
        const loadTime = await measurePageLoad(page);
        loadTimes.push(loadTime);
        
        // Wait a bit between attempts
        await page.waitForTimeout(1000);
      }
      
      // Calculate average load time
      const avgLoadTime = loadTimes.reduce((a, b) => a + b, 0) / loadTimes.length;
      
      // All load times should be reasonable
      for (const time of loadTimes) {
        expect(time).toBeLessThan(7000);
      }
      
      // Variance shouldn't be too high (no load time should be 3x average)
      for (const time of loadTimes) {
        expect(time).toBeLessThan(avgLoadTime * 3);
      }
      
      console.log(`Load times: ${loadTimes.join(', ')}ms (avg: ${avgLoadTime.toFixed(1)}ms)`);
    });
  });

  test.describe('Tab Switching Performance', () => {
    test('Tab switching is responsive', async ({ page }) => {
      await navigateToDashboard(page);
      
      const tabs = ['system', 'instruments', 'chart', 'watchlists'];
      const switchTimes: number[] = [];
      
      for (const tabKey of tabs) {
        const switchTime = await measureTabSwitch(page, tabKey);
        switchTimes.push(switchTime);
        
        // Each tab switch should be quick
        expect(switchTime).toBeLessThan(2000);
        
        // Verify tab is active
        const activeTab = page.locator('.ant-tabs-tab-active');
        await expect(activeTab).toBeVisible();
      }
      
      const avgSwitchTime = switchTimes.reduce((a, b) => a + b, 0) / switchTimes.length;
      console.log(`Tab switch times: ${switchTimes.join(', ')}ms (avg: ${avgSwitchTime.toFixed(1)}ms)`);
    });

    test('Rapid tab switching performance', async ({ page }) => {
      await navigateToDashboard(page);
      
      const tabs = ['system', 'instruments', 'chart', 'watchlists', 'system'];
      const startTime = Date.now();
      
      // Switch tabs rapidly
      for (const tabKey of tabs) {
        await page.click(`[data-node-key="${tabKey}"]`);
        await page.waitForTimeout(200); // Minimal wait
      }
      
      const totalTime = Date.now() - startTime;
      
      // Should complete rapid switching within reasonable time
      expect(totalTime).toBeLessThan(3000);
      
      // Dashboard should still be functional
      await expect(page.locator('[data-testid="dashboard"]')).toBeVisible();
      
      console.log(`Rapid tab switching total time: ${totalTime}ms`);
    });
  });

  test.describe('Message Bus Performance', () => {
    test('High-volume message handling performance', async ({ page }) => {
      // Setup high-volume message flow (50 messages per second)
      await setupHighVolumeWebSocket(page, 50);
      
      await page.reload();
      await navigateToDashboard(page);
      await page.click('[data-node-key="system"]');
      
      // Let it run for 10 seconds
      await page.waitForTimeout(10000);
      
      // Dashboard should still be responsive
      await expect(page.locator('[data-testid="dashboard"]')).toBeVisible();
      
      // Should be able to switch tabs
      const switchStartTime = Date.now();
      await page.click('[data-node-key="instruments"]');
      await page.waitForTimeout(1000);
      const switchTime = Date.now() - switchStartTime;
      
      // Tab switch should still be reasonably fast despite high message volume
      expect(switchTime).toBeLessThan(3000);
      
      // Check message statistics
      await page.click('[data-node-key="system"]');
      const totalMessagesStatistic = page.locator('.ant-statistic').filter({ hasText: 'Total Messages' });
      
      if (await totalMessagesStatistic.isVisible()) {
        const messagesValue = await page.locator('.ant-statistic-content-value').first().textContent();
        if (messagesValue) {
          const messageCount = parseInt(messagesValue);
          // Should have received many messages
          expect(messageCount).toBeGreaterThan(100);
          console.log(`Processed ${messageCount} high-volume messages`);
        }
      }
    });

    test('Message bus memory usage under load', async ({ page }) => {
      const initialMemory = await getMemoryUsage(page);
      
      // Setup moderate message flow
      await setupHighVolumeWebSocket(page, 20);
      
      await page.reload();
      await navigateToDashboard(page);
      await page.click('[data-node-key="system"]');
      
      // Let it run for 15 seconds
      await page.waitForTimeout(15000);
      
      const finalMemory = await getMemoryUsage(page);
      
      if (initialMemory && finalMemory) {
        const memoryIncrease = finalMemory.usedJSHeapSize - initialMemory.usedJSHeapSize;
        const memoryIncreaseMB = memoryIncrease / (1024 * 1024);
        
        // Memory increase should be reasonable (less than 50MB)
        expect(memoryIncreaseMB).toBeLessThan(50);
        
        console.log(`Memory usage increase: ${memoryIncreaseMB.toFixed(2)}MB`);
      }
      
      // Clear messages and check if memory is released
      const clearButton = page.locator('button:has-text("Clear")');
      await clearButton.click();
      await page.waitForTimeout(2000);
      
      // Dashboard should remain functional
      await expect(page.locator('[data-testid="dashboard"]')).toBeVisible();
    });
  });

  test.describe('Component Rendering Performance', () => {
    test('Large dataset rendering performance', async ({ page }) => {
      await navigateToDashboard(page);
      
      // Test components that might render large datasets
      const performanceTestTabs = [
        { key: 'instruments', name: 'Search' },
        { key: 'watchlists', name: 'Watchlist' },
        { key: 'chart', name: 'Chart' }
      ];
      
      for (const tab of performanceTestTabs) {
        const startTime = Date.now();
        
        await page.click(`[data-node-key="${tab.key}"]`);
        await page.waitForTimeout(3000); // Wait for full load
        
        const renderTime = Date.now() - startTime;
        
        // Component should render within reasonable time
        expect(renderTime).toBeLessThan(5000);
        
        // Verify component is visible
        const activeTabPane = page.locator('.ant-tabs-tabpane-active');
        await expect(activeTabPane).toBeVisible();
        
        console.log(`${tab.name} tab render time: ${renderTime}ms`);
      }
    });

    test('Error boundary performance impact', async ({ page }) => {
      await navigateToDashboard(page);
      
      // Test tabs that might trigger error boundaries
      const errorProneTabs = ['nautilus-engine', 'backtesting', 'strategy', 'portfolio'];
      
      for (const tabKey of errorProneTabs) {
        const startTime = Date.now();
        
        await page.click(`[data-node-key="${tabKey}"]`);
        await page.waitForTimeout(3000);
        
        const loadTime = Date.now() - startTime;
        
        // Even with error boundaries, should load reasonably fast
        expect(loadTime).toBeLessThan(6000);
        
        // Check if error boundary was triggered
        const hasErrorBoundary = await page.locator('.ant-result-error').isVisible();
        
        if (hasErrorBoundary) {
          // Error boundary should render quickly
          await expect(page.locator('.ant-result-title')).toBeVisible();
        }
        
        console.log(`${tabKey} tab load time: ${loadTime}ms (error boundary: ${hasErrorBoundary})`);
      }
    });
  });

  test.describe('Responsive Performance', () => {
    test('Performance across different viewport sizes', async ({ page }) => {
      const viewportSizes = [
        { width: 375, height: 667, name: 'Mobile' },
        { width: 768, height: 1024, name: 'Tablet' },
        { width: 1920, height: 1080, name: 'Desktop' }
      ];
      
      for (const viewport of viewportSizes) {
        await page.setViewportSize(viewport);
        
        const loadStartTime = Date.now();
        await navigateToDashboard(page);
        const loadTime = Date.now() - loadStartTime;
        
        // Should load quickly on all devices
        expect(loadTime).toBeLessThan(6000);
        
        // Test tab switching performance
        const switchStartTime = Date.now();
        await page.click('[data-node-key="instruments"]');
        await page.waitForTimeout(1000);
        const switchTime = Date.now() - switchStartTime;
        
        // Tab switching should be responsive
        expect(switchTime).toBeLessThan(2500);
        
        // Verify UI is still functional
        await expect(page.locator('[data-testid="dashboard"]')).toBeVisible();
        
        console.log(`${viewport.name} (${viewport.width}x${viewport.height}): load ${loadTime}ms, switch ${switchTime}ms`);
      }
    });
  });

  test.describe('Concurrent Operations Performance', () => {
    test('Multiple operations performance', async ({ page }) => {
      await setupHighVolumeWebSocket(page, 10);
      
      await page.reload();
      await navigateToDashboard(page);
      
      // Start with system tab (receiving messages)
      await page.click('[data-node-key="system"]');
      await page.waitForTimeout(2000);
      
      // Perform multiple operations concurrently
      const startTime = Date.now();
      
      // Tab switches
      await page.click('[data-node-key="instruments"]');
      await page.waitForTimeout(500);
      
      await page.click('[data-node-key="chart"]');
      await page.waitForTimeout(500);
      
      // System tab interaction while messages are flowing
      await page.click('[data-node-key="system"]');
      await page.waitForTimeout(500);
      
      // Click refresh button
      const refreshButton = page.locator('button:has-text("Refresh")');
      if (await refreshButton.isVisible()) {
        await refreshButton.click();
        await page.waitForTimeout(1000);
      }
      
      // Clear messages
      const clearButton = page.locator('button:has-text("Clear")');
      if (await clearButton.isVisible()) {
        await clearButton.click();
        await page.waitForTimeout(500);
      }
      
      const totalTime = Date.now() - startTime;
      
      // All operations should complete within reasonable time
      expect(totalTime).toBeLessThan(7000);
      
      // Dashboard should still be functional
      await expect(page.locator('[data-testid="dashboard"]')).toBeVisible();
      
      console.log(`Concurrent operations completed in: ${totalTime}ms`);
    });
  });

  test.describe('API Call Performance', () => {
    test('Backend API response time impact', async ({ page }) => {
      // Mock slow API responses
      await page.route('**/api/v1/**', async (route) => {
        // Simulate slow network
        await new Promise(resolve => setTimeout(resolve, 1000));
        route.continue();
      });
      
      const loadStartTime = Date.now();
      await navigateToDashboard(page);
      
      // Navigate to system tab (which makes API calls)
      await page.click('[data-node-key="system"]');
      await page.waitForTimeout(5000); // Wait for API calls
      
      const totalTime = Date.now() - loadStartTime;
      
      // Should handle slow APIs gracefully
      expect(totalTime).toBeLessThan(15000);
      
      // UI should still be responsive
      await expect(page.locator('[data-testid="dashboard"]')).toBeVisible();
      await expect(page.getByText('API Status')).toBeVisible();
      
      console.log(`Dashboard with slow API responses: ${totalTime}ms`);
    });
  });

  test.describe('Memory Leak Detection', () => {
    test('Extended usage memory stability', async ({ page }) => {
      const initialMemory = await getMemoryUsage(page);
      
      await navigateToDashboard(page);
      
      // Simulate extended usage - multiple tab switches and operations
      const tabs = ['system', 'instruments', 'chart', 'watchlists', 'portfolio', 'risk'];
      
      for (let cycle = 0; cycle < 5; cycle++) {
        for (const tabKey of tabs) {
          await page.click(`[data-node-key="${tabKey}"]`);
          await page.waitForTimeout(300);
        }
        
        // Perform some operations
        await page.click('[data-node-key="system"]');
        
        const clearButton = page.locator('button:has-text("Clear")');
        if (await clearButton.isVisible()) {
          await clearButton.click();
          await page.waitForTimeout(200);
        }
        
        const refreshButton = page.locator('button:has-text("Refresh")');
        if (await refreshButton.isVisible()) {
          await refreshButton.click();
          await page.waitForTimeout(200);
        }
      }
      
      const finalMemory = await getMemoryUsage(page);
      
      if (initialMemory && finalMemory) {
        const memoryIncrease = finalMemory.usedJSHeapSize - initialMemory.usedJSHeapSize;
        const memoryIncreaseMB = memoryIncrease / (1024 * 1024);
        
        // Memory increase should be reasonable after extended usage
        expect(memoryIncreaseMB).toBeLessThan(100);
        
        console.log(`Memory increase after extended usage: ${memoryIncreaseMB.toFixed(2)}MB`);
      }
      
      // Dashboard should still be fully functional
      await expect(page.locator('[data-testid="dashboard"]')).toBeVisible();
    });
  });

  test.describe('Stress Testing', () => {
    test('Dashboard stability under stress', async ({ page }) => {
      // Combine multiple stress factors
      await setupHighVolumeWebSocket(page, 30); // High message volume
      
      // Mock some API delays
      await page.route('**/api/v1/health', async (route) => {
        await new Promise(resolve => setTimeout(resolve, 500));
        route.continue();
      });
      
      await page.reload();
      await navigateToDashboard(page);
      
      // Rapid operations under stress
      const operations = [
        () => page.click('[data-node-key="system"]'),
        () => page.click('[data-node-key="instruments"]'),
        () => page.click('[data-node-key="chart"]'),
        () => page.click('[data-node-key="watchlists"]'),
        () => page.click('button:has-text("Clear")').catch(() => {}),
        () => page.click('button:has-text("Refresh")').catch(() => {})
      ];
      
      // Perform rapid operations
      for (let i = 0; i < 20; i++) {
        const operation = operations[i % operations.length];
        await operation();
        await page.waitForTimeout(100);
      }
      
      // Final stability check
      await page.waitForTimeout(3000);
      await expect(page.locator('[data-testid="dashboard"]')).toBeVisible();
      
      // Should still be able to navigate
      await page.click('[data-node-key="system"]');
      await expect(page.getByText('API Status')).toBeVisible();
      
      console.log('Dashboard survived stress test');
    });
  });
});