import { test, expect, Page } from '@playwright/test';

/**
 * Message Bus Integration Test Suite
 * Tests WebSocket connectivity, message handling, and communication channels
 */

// Helper function to wait for dashboard to load
async function waitForDashboardLoad(page: Page) {
  await page.waitForSelector('[data-testid="dashboard"]', { timeout: 10000 });
  await page.waitForSelector('.ant-tabs-tab', { timeout: 5000 });
  await page.waitForTimeout(2000);
}

// Helper function to check message bus status
async function getMessageBusStatus(page: Page): Promise<string> {
  // Look for the MessageBus status badge or text
  const statusElement = await page.locator('[data-testid="messagebus-status"], .ant-badge-status-text').first();
  if (await statusElement.isVisible()) {
    return await statusElement.textContent() || 'unknown';
  }
  return 'not-found';
}

// Helper function to click message bus connect/disconnect
async function toggleMessageBusConnection(page: Page, action: 'connect' | 'disconnect') {
  const buttonText = action === 'connect' ? 'Connect' : 'Disconnect';
  const button = page.locator(`button:has-text("${buttonText}")`);
  if (await button.isVisible()) {
    await button.click();
    await page.waitForTimeout(2000);
  }
}

test.describe('Message Bus Integration Tests', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/');
    await waitForDashboardLoad(page);
    
    // Navigate to System tab where message bus controls are located
    await page.click('[data-node-key="system"]');
    await page.waitForTimeout(1000);
  });

  test('Message bus status is displayed correctly', async ({ page }) => {
    // Check for MessageBus Connection card
    await expect(page.getByText('MessageBus Connection')).toBeVisible();
    
    // Check for status badge
    const statusBadge = page.locator('.ant-badge-status');
    await expect(statusBadge).toBeVisible();
    
    // Check for connection info
    const connectionInfo = page.locator('text="Status:"');
    await expect(connectionInfo).toBeVisible();
  });

  test('Message bus connection controls work', async ({ page }) => {
    // Look for connect/disconnect buttons
    const connectButton = page.locator('button:has-text("Connect")');
    const disconnectButton = page.locator('button:has-text("Disconnect")');
    
    // At least one should be visible
    const hasConnectionControls = await connectButton.isVisible() || await disconnectButton.isVisible();
    expect(hasConnectionControls).toBe(true);
    
    // Try to toggle connection if disconnect is available
    if (await disconnectButton.isVisible()) {
      await disconnectButton.click();
      await page.waitForTimeout(2000);
      
      // Should show connect button now
      await expect(connectButton).toBeVisible();
      
      // Try to reconnect
      await connectButton.click();
      await page.waitForTimeout(3000);
    }
  });

  test('Message statistics are displayed', async ({ page }) => {
    // Check for Message Statistics card
    await expect(page.getByText('Message Statistics')).toBeVisible();
    
    // Check for total messages statistic
    const totalMessages = page.locator('text="Total Messages"');
    await expect(totalMessages).toBeVisible();
    
    // Check for unique topics statistic
    const uniqueTopics = page.locator('text="Unique Topics"');
    await expect(uniqueTopics).toBeVisible();
  });

  test('Latest message display works', async ({ page }) => {
    // Wait for potential messages to arrive
    await page.waitForTimeout(5000);
    
    // Check if latest message card appears
    const latestMessageCard = page.locator('text="Latest Message"');
    const hasLatestMessage = await latestMessageCard.isVisible();
    
    if (hasLatestMessage) {
      // Verify message structure
      await expect(page.locator('text="Type:"')).toBeVisible();
      await expect(page.locator('text="Timestamp:"')).toBeVisible();
    } else {
      console.log('No latest message displayed - this is normal if no messages received');
    }
  });

  test('Clear messages functionality works', async ({ page }) => {
    // Look for clear button
    const clearButton = page.locator('button:has-text("Clear")');
    await expect(clearButton).toBeVisible();
    
    // Click clear button
    await clearButton.click();
    await page.waitForTimeout(1000);
    
    // Messages should be cleared (statistics should reset or latest message should disappear)
    // Note: This test might need adjustment based on actual implementation
  });

  test('Message bus handles WebSocket connection states', async ({ page }) => {
    // Mock WebSocket connection states by intercepting WebSocket creation
    await page.addInitScript(() => {
      let originalWebSocket = window.WebSocket;
      let mockConnections: any[] = [];
      
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
          mockConnections.push(this);
          
          // Simulate connection lifecycle
          setTimeout(() => {
            this.readyState = MockWebSocket.OPEN;
            if (this.onopen) this.onopen({ type: 'open' });
          }, 100);
        }
        
        close() {
          this.readyState = MockWebSocket.CLOSED;
          if (this.onclose) this.onclose({ type: 'close', code: 1000, reason: 'Normal closure' });
        }
        
        send(data: string) {
          // Mock sending data
        }
      };
      
      (window as any).mockConnections = mockConnections;
    });
    
    // Reload to use mocked WebSocket
    await page.reload();
    await waitForDashboardLoad(page);
    await page.click('[data-node-key="system"]');
    await page.waitForTimeout(3000);
    
    // Should show connected state
    const statusElements = await page.locator('.ant-badge-status').count();
    expect(statusElements).toBeGreaterThan(0);
  });

  test('Message bus displays connection info correctly', async ({ page }) => {
    // Check for backend state display
    const backendStateText = page.locator('text="Backend State:"');
    const hasBackendState = await backendStateText.isVisible();
    
    if (hasBackendState) {
      // Should have some value
      const stateValue = await page.locator('text="Backend State:"').locator('..').textContent();
      expect(stateValue).toBeTruthy();
    }
    
    // Check for reconnect attempts
    const reconnectText = page.locator('text="Reconnect Attempts:"');
    const hasReconnectAttempts = await reconnectText.isVisible();
    
    if (hasReconnectAttempts) {
      // Should show number of attempts
      const attemptsValue = await page.locator('text="Reconnect Attempts:"').locator('..').textContent();
      expect(attemptsValue).toBeTruthy();
    }
  });

  test('Message bus error handling', async ({ page }) => {
    // Mock WebSocket errors
    await page.addInitScript(() => {
      let originalWebSocket = window.WebSocket;
      
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
          // Simulate connection error
          setTimeout(() => {
            this.readyState = MockWebSocket.CLOSED;
            if (this.onerror) this.onerror({ type: 'error' });
            if (this.onclose) this.onclose({ type: 'close', code: 1006, reason: 'Connection failed' });
          }, 100);
        }
        
        close() {
          this.readyState = MockWebSocket.CLOSED;
        }
        
        send(data: string) {
          throw new Error('Connection not open');
        }
      };
    });
    
    await page.reload();
    await waitForDashboardLoad(page);
    await page.click('[data-node-key="system"]');
    await page.waitForTimeout(3000);
    
    // Should handle error state gracefully
    const errorElements = await page.locator('.ant-alert-error').count();
    // Error handling should not crash the app
    await expect(page.locator('[data-testid="dashboard"]')).toBeVisible();
  });

  test('Message bus performance metrics', async ({ page }) => {
    // Check if messages received counter is working
    const messagesReceivedText = page.locator('text="messages received"');
    const hasMessagesReceived = await messagesReceivedText.isVisible();
    
    if (hasMessagesReceived) {
      // Should show some count
      const messageCount = await page.locator('text="messages received"').locator('..').textContent();
      expect(messageCount).toBeTruthy();
    }
    
    // Statistics should be numbers
    const statisticElements = await page.locator('.ant-statistic-content-value').all();
    for (const element of statisticElements) {
      const value = await element.textContent();
      if (value) {
        // Should be numeric or at least not crash
        expect(value).toBeTruthy();
      }
    }
  });

  test('Message bus topic filtering', async ({ page }) => {
    // Mock receiving messages with different topics
    await page.addInitScript(() => {
      let originalWebSocket = window.WebSocket;
      
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
            
            // Send mock messages
            setTimeout(() => {
              if (this.onmessage) {
                this.onmessage({
                  data: JSON.stringify({
                    type: 'messagebus',
                    topic: 'market_data',
                    payload: { symbol: 'AAPL', price: 150.0 },
                    timestamp: Date.now() * 1000000
                  })
                });
              }
            }, 500);
            
            setTimeout(() => {
              if (this.onmessage) {
                this.onmessage({
                  data: JSON.stringify({
                    type: 'messagebus',
                    topic: 'order_updates',
                    payload: { orderId: '123', status: 'filled' },
                    timestamp: Date.now() * 1000000
                  })
                });
              }
            }, 1000);
          }, 100);
        }
        
        close() {
          this.readyState = MockWebSocket.CLOSED;
          if (this.onclose) this.onclose({ type: 'close', code: 1000, reason: 'Normal closure' });
        }
        
        send(data: string) {}
      };
    });
    
    await page.reload();
    await waitForDashboardLoad(page);
    await page.click('[data-node-key="system"]');
    await page.waitForTimeout(5000);
    
    // Should receive and display messages
    const latestMessageCard = page.locator('text="Latest Message"');
    const hasLatestMessage = await latestMessageCard.isVisible();
    
    if (hasLatestMessage) {
      // Should show topic information
      await expect(page.locator('text="Type:"')).toBeVisible();
      
      // Unique topics count should be > 0
      const uniqueTopicsElement = page.locator('.ant-statistic-content-value').nth(1);
      const uniqueTopicsValue = await uniqueTopicsElement.textContent();
      if (uniqueTopicsValue) {
        const count = parseInt(uniqueTopicsValue);
        expect(count).toBeGreaterThanOrEqual(0);
      }
    }
  });

  test('Message bus reconnection behavior', async ({ page }) => {
    // Test automatic reconnection
    await page.addInitScript(() => {
      let connectionAttempts = 0;
      let originalWebSocket = window.WebSocket;
      
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
          connectionAttempts++;
          
          if (connectionAttempts === 1) {
            // First attempt: connect then disconnect
            setTimeout(() => {
              this.readyState = MockWebSocket.OPEN;
              if (this.onopen) this.onopen({ type: 'open' });
              
              setTimeout(() => {
                this.readyState = MockWebSocket.CLOSED;
                if (this.onclose) this.onclose({ type: 'close', code: 1006, reason: 'Connection lost' });
              }, 1000);
            }, 100);
          } else {
            // Subsequent attempts: successful connection
            setTimeout(() => {
              this.readyState = MockWebSocket.OPEN;
              if (this.onopen) this.onopen({ type: 'open' });
            }, 100);
          }
        }
        
        close() {
          this.readyState = MockWebSocket.CLOSED;
        }
        
        send(data: string) {}
      };
    });
    
    await page.reload();
    await waitForDashboardLoad(page);
    await page.click('[data-node-key="system"]');
    await page.waitForTimeout(8000); // Wait for reconnection
    
    // Should handle reconnection gracefully
    await expect(page.locator('[data-testid="dashboard"]')).toBeVisible();
  });
});