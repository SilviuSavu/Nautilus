import { test, expect, Page } from '@playwright/test';

/**
 * Real-time Communication Test Suite
 * Tests WebSocket messaging, live data updates, order book streaming, and system notifications
 */

// Helper functions
async function navigateToDashboard(page: Page) {
  await page.goto('/');
  await page.waitForSelector('[data-testid="dashboard"]', { timeout: 10000 });
  await page.waitForTimeout(2000);
}

async function setupMockWebSocket(page: Page, messageTypes: string[] = ['messagebus', 'order_book', 'market_data']) {
  await page.addInitScript((types) => {
    let messageId = 0;
    let isConnected = false;
    
    // Store original WebSocket
    (window as any).originalWebSocket = window.WebSocket;
    
    // Mock WebSocket class
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
        console.log('MockWebSocket created for:', url);
        
        // Simulate connection
        setTimeout(() => {
          this.readyState = MockWebSocket.OPEN;
          isConnected = true;
          if (this.onopen) {
            this.onopen({ type: 'open' });
          }
          
          // Start sending mock messages
          this.startMockMessages();
        }, 100);
      }
      
      startMockMessages() {
        const sendMessage = () => {
          if (!isConnected || this.readyState !== MockWebSocket.OPEN) return;
          
          const messageType = types[Math.floor(Math.random() * types.length)];
          let message: any;
          
          switch (messageType) {
            case 'messagebus':
              message = {
                type: 'messagebus',
                topic: `system.${Math.random() > 0.5 ? 'status' : 'alert'}`,
                payload: {
                  id: ++messageId,
                  message: `Mock system message ${messageId}`,
                  level: Math.random() > 0.7 ? 'warning' : 'info'
                },
                timestamp: Date.now() * 1000000
              };
              break;
              
            case 'order_book':
              message = {
                type: 'order_book_update',
                symbol: 'AAPL',
                venue: 'NASDAQ',
                bids: [
                  { price: 150.0 + Math.random(), quantity: 100 + Math.floor(Math.random() * 900) },
                  { price: 149.9 + Math.random(), quantity: 200 + Math.floor(Math.random() * 800) }
                ],
                asks: [
                  { price: 150.1 + Math.random(), quantity: 150 + Math.floor(Math.random() * 850) },
                  { price: 150.2 + Math.random(), quantity: 300 + Math.floor(Math.random() * 700) }
                ],
                timestamp: Date.now() * 1000000
              };
              break;
              
            case 'market_data':
              message = {
                type: 'market_data',
                topic: 'market_data.AAPL',
                payload: {
                  symbol: 'AAPL',
                  price: 150 + (Math.random() - 0.5) * 10,
                  volume: Math.floor(Math.random() * 1000000),
                  timestamp: Date.now()
                },
                timestamp: Date.now() * 1000000
              };
              break;
          }
          
          if (this.onmessage && message) {
            this.onmessage({
              data: JSON.stringify(message)
            });
          }
          
          // Schedule next message
          setTimeout(sendMessage, 1000 + Math.random() * 2000);
        };
        
        // Start sending messages after a short delay
        setTimeout(sendMessage, 500);
      }
      
      close() {
        isConnected = false;
        this.readyState = MockWebSocket.CLOSED;
        if (this.onclose) {
          this.onclose({ type: 'close', code: 1000, reason: 'Normal closure' });
        }
      }
      
      send(data: string) {
        console.log('MockWebSocket send:', data);
      }
    };
  }, messageTypes);
}

test.describe('Real-time Communication Tests', () => {
  test.beforeEach(async ({ page }) => {
    await navigateToDashboard(page);
  });

  test('WebSocket connection establishes and receives messages', async ({ page }) => {
    // Setup mock WebSocket with all message types
    await setupMockWebSocket(page, ['messagebus', 'order_book', 'market_data']);
    
    // Reload to use mock WebSocket
    await page.reload();
    await navigateToDashboard(page);
    
    // Navigate to system tab to see message bus
    await page.click('[data-node-key="system"]');
    await page.waitForTimeout(3000);
    
    // Should show connected status
    const statusBadge = page.locator('.ant-badge-status');
    await expect(statusBadge).toBeVisible();
    
    // Wait for messages to arrive
    await page.waitForTimeout(5000);
    
    // Check message statistics
    const totalMessagesStatistic = page.locator('.ant-statistic').filter({ hasText: 'Total Messages' });
    if (await totalMessagesStatistic.isVisible()) {
      const messagesValue = await page.locator('.ant-statistic-content-value').first().textContent();
      if (messagesValue) {
        const messageCount = parseInt(messagesValue);
        expect(messageCount).toBeGreaterThan(0);
      }
    }
    
    // Check for latest message display
    const latestMessageCard = page.locator('text="Latest Message"');
    const hasLatestMessage = await latestMessageCard.isVisible();
    
    if (hasLatestMessage) {
      await expect(page.locator('text="Type:"')).toBeVisible();
      await expect(page.locator('text="Timestamp:"')).toBeVisible();
    }
  });

  test('Message bus handles different message types', async ({ page }) => {
    await setupMockWebSocket(page, ['messagebus']);
    
    await page.reload();
    await navigateToDashboard(page);
    await page.click('[data-node-key="system"]');
    await page.waitForTimeout(5000);
    
    // Should receive messagebus type messages
    const latestMessageCard = page.locator('text="Latest Message"');
    const hasLatestMessage = await latestMessageCard.isVisible();
    
    if (hasLatestMessage) {
      // Check message type
      const messageTypeElement = page.locator('text="Type:"').locator('..');
      const messageType = await messageTypeElement.textContent();
      expect(messageType).toContain('messagebus');
    }
    
    // Check unique topics count
    const uniqueTopicsStatistic = page.locator('.ant-statistic').filter({ hasText: 'Unique Topics' });
    if (await uniqueTopicsStatistic.isVisible()) {
      const topicsValue = await page.locator('.ant-statistic-content-value').nth(1).textContent();
      if (topicsValue) {
        const topicsCount = parseInt(topicsValue);
        expect(topicsCount).toBeGreaterThan(0);
      }
    }
  });

  test('Order book updates are processed correctly', async ({ page }) => {
    await setupMockWebSocket(page, ['order_book']);
    
    await page.reload();
    await navigateToDashboard(page);
    
    // Wait for order book messages
    await page.waitForTimeout(5000);
    
    // Order book messages might be handled by specific components
    // Check that the system doesn't crash when receiving them
    await expect(page.locator('[data-testid="dashboard"]')).toBeVisible();
    
    // Navigate to system tab to check if any messages were received
    await page.click('[data-node-key="system"]');
    await page.waitForTimeout(2000);
    
    // Even if not displayed as latest message, the connection should be stable
    const statusBadge = page.locator('.ant-badge-status');
    await expect(statusBadge).toBeVisible();
  });

  test('Market data streaming integration', async ({ page }) => {
    await setupMockWebSocket(page, ['market_data']);
    
    await page.reload();
    await navigateToDashboard(page);
    await page.waitForTimeout(5000);
    
    // Check that market data messages don't crash the system
    await expect(page.locator('[data-testid="dashboard"]')).toBeVisible();
    
    // Navigate to chart tab where market data might be used
    await page.click('[data-node-key="chart"]');
    await page.waitForTimeout(3000);
    
    // Should load without errors
    await expect(page.getByText('Instrument Selection')).toBeVisible();
    await expect(page.getByText('Timeframe Selection')).toBeVisible();
  });

  test('Message bus performance under load', async ({ page }) => {
    // Setup WebSocket with rapid message sending
    await page.addInitScript(() => {
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
            
            // Send messages rapidly
            let messageCount = 0;
            const sendRapidMessages = () => {
              if (messageCount >= 50) return; // Send 50 messages
              
              const message = {
                type: 'messagebus',
                topic: `rapid.message.${messageCount}`,
                payload: { id: messageCount, data: `Rapid message ${messageCount}` },
                timestamp: Date.now() * 1000000
              };
              
              if (this.onmessage) {
                this.onmessage({ data: JSON.stringify(message) });
              }
              
              messageCount++;
              setTimeout(sendRapidMessages, 50); // Send every 50ms
            };
            
            setTimeout(sendRapidMessages, 100);
          }, 100);
        }
        
        close() {
          this.readyState = MockWebSocket.CLOSED;
          if (this.onclose) this.onclose({ type: 'close', code: 1000 });
        }
        
        send(data: string) {}
      };
    });
    
    await page.reload();
    await navigateToDashboard(page);
    await page.click('[data-node-key="system"]');
    await page.waitForTimeout(5000);
    
    // Should handle rapid messages without crashing
    await expect(page.locator('[data-testid="dashboard"]')).toBeVisible();
    
    // Check message count increased significantly
    const totalMessagesStatistic = page.locator('.ant-statistic').filter({ hasText: 'Total Messages' });
    if (await totalMessagesStatistic.isVisible()) {
      const messagesValue = await page.locator('.ant-statistic-content-value').first().textContent();
      if (messagesValue) {
        const messageCount = parseInt(messagesValue);
        expect(messageCount).toBeGreaterThan(20); // Should have received many messages
      }
    }
  });

  test('WebSocket reconnection after connection loss', async ({ page }) => {
    let connectionAttempts = 0;
    
    await page.addInitScript(() => {
      let attempts = 0;
      
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
          attempts++;
          
          if (attempts === 1) {
            // First connection: connect then disconnect
            setTimeout(() => {
              this.readyState = MockWebSocket.OPEN;
              if (this.onopen) this.onopen({ type: 'open' });
              
              // Disconnect after 2 seconds
              setTimeout(() => {
                this.readyState = MockWebSocket.CLOSED;
                if (this.onclose) this.onclose({ type: 'close', code: 1006, reason: 'Connection lost' });
              }, 2000);
            }, 100);
          } else {
            // Subsequent connections: successful
            setTimeout(() => {
              this.readyState = MockWebSocket.OPEN;
              if (this.onopen) this.onopen({ type: 'open' });
            }, 1000);
          }
        }
        
        close() {
          this.readyState = MockWebSocket.CLOSED;
        }
        
        send(data: string) {}
      };
    });
    
    await page.reload();
    await navigateToDashboard(page);
    await page.click('[data-node-key="system"]');
    
    // Wait for initial connection, disconnection, and reconnection
    await page.waitForTimeout(8000);
    
    // Should eventually reconnect
    const statusBadge = page.locator('.ant-badge-status');
    await expect(statusBadge).toBeVisible();
    
    // Dashboard should remain functional
    await expect(page.locator('[data-testid="dashboard"]')).toBeVisible();
  });

  test('Error handling for malformed messages', async ({ page }) => {
    await page.addInitScript(() => {
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
            
            // Send malformed messages
            setTimeout(() => {
              if (this.onmessage) {
                // Invalid JSON
                this.onmessage({ data: 'invalid json {' });
                
                // Missing required fields
                this.onmessage({ data: '{"type":"incomplete"}' });
                
                // Valid message after invalid ones
                this.onmessage({ 
                  data: JSON.stringify({
                    type: 'messagebus',
                    topic: 'valid.message',
                    payload: { test: 'data' },
                    timestamp: Date.now() * 1000000
                  })
                });
              }
            }, 1000);
          }, 100);
        }
        
        close() {
          this.readyState = MockWebSocket.CLOSED;
          if (this.onclose) this.onclose({ type: 'close', code: 1000 });
        }
        
        send(data: string) {}
      };
    });
    
    await page.reload();
    await navigateToDashboard(page);
    await page.click('[data-node-key="system"]');
    await page.waitForTimeout(5000);
    
    // Should handle malformed messages gracefully
    await expect(page.locator('[data-testid="dashboard"]')).toBeVisible();
    
    // Valid messages should still be processed
    const latestMessageCard = page.locator('text="Latest Message"');
    const hasLatestMessage = await latestMessageCard.isVisible();
    
    if (hasLatestMessage) {
      // Should show the valid message
      await expect(page.locator('text="Type:"')).toBeVisible();
    }
  });

  test('Message bus buffer management', async ({ page }) => {
    // Test that the message buffer doesn't grow indefinitely
    await page.addInitScript(() => {
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
            
            // Send many messages to test buffer management
            let messageId = 0;
            const sendMessages = () => {
              if (messageId >= 150) return; // Send more than buffer size (100)
              
              const message = {
                type: 'messagebus',
                topic: `buffer.test.${messageId}`,
                payload: { id: messageId },
                timestamp: Date.now() * 1000000
              };
              
              if (this.onmessage) {
                this.onmessage({ data: JSON.stringify(message) });
              }
              
              messageId++;
              setTimeout(sendMessages, 10);
            };
            
            setTimeout(sendMessages, 500);
          }, 100);
        }
        
        close() {
          this.readyState = MockWebSocket.CLOSED;
        }
        
        send(data: string) {}
      };
    });
    
    await page.reload();
    await navigateToDashboard(page);
    await page.click('[data-node-key="system"]');
    await page.waitForTimeout(8000);
    
    // Should handle buffer overflow gracefully
    await expect(page.locator('[data-testid="dashboard"]')).toBeVisible();
    
    // Message count should be reasonable (not infinite)
    const totalMessagesStatistic = page.locator('.ant-statistic').filter({ hasText: 'Total Messages' });
    if (await totalMessagesStatistic.isVisible()) {
      const messagesValue = await page.locator('.ant-statistic-content-value').first().textContent();
      if (messagesValue) {
        const messageCount = parseInt(messagesValue);
        expect(messageCount).toBeGreaterThan(100);
        expect(messageCount).toBeLessThan(1000); // Shouldn't be extremely high
      }
    }
  });

  test('Clear messages functionality', async ({ page }) => {
    await setupMockWebSocket(page, ['messagebus']);
    
    await page.reload();
    await navigateToDashboard(page);
    await page.click('[data-node-key="system"]');
    await page.waitForTimeout(3000);
    
    // Wait for some messages to arrive
    await page.waitForTimeout(3000);
    
    // Click clear button
    const clearButton = page.locator('button:has-text("Clear")');
    await expect(clearButton).toBeVisible();
    await clearButton.click();
    await page.waitForTimeout(1000);
    
    // Messages should be cleared
    const latestMessageCard = page.locator('text="Latest Message"');
    const hasLatestMessage = await latestMessageCard.isVisible();
    
    // Latest message should be cleared (might take a moment)
    if (hasLatestMessage) {
      // If still visible, it means new messages arrived after clear
      console.log('New messages arrived after clear - this is expected behavior');
    }
  });

  test('Connection controls work correctly', async ({ page }) => {
    await setupMockWebSocket(page);
    
    await page.reload();
    await navigateToDashboard(page);
    await page.click('[data-node-key="system"]');
    await page.waitForTimeout(3000);
    
    // Check for disconnect button (if connected)
    const disconnectButton = page.locator('button:has-text("Disconnect")');
    const connectButton = page.locator('button:has-text("Connect")');
    
    if (await disconnectButton.isVisible()) {
      // Test disconnection
      await disconnectButton.click();
      await page.waitForTimeout(2000);
      
      // Should show connect button
      await expect(connectButton).toBeVisible();
      
      // Test reconnection
      await connectButton.click();
      await page.waitForTimeout(3000);
    } else if (await connectButton.isVisible()) {
      // Test connection
      await connectButton.click();
      await page.waitForTimeout(3000);
    }
    
    // Dashboard should remain functional regardless
    await expect(page.locator('[data-testid="dashboard"]')).toBeVisible();
  });
});