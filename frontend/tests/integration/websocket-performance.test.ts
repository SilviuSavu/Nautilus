/**
 * Integration tests for WebSocket performance monitoring and end-to-end communication
 */

import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest'
import { WebSocketService, PerformanceMetrics } from '../../src/services/websocket'

// Mock WebSocket for testing
class MockWebSocket {
  static CONNECTING = 0
  static OPEN = 1
  static CLOSING = 2
  static CLOSED = 3

  readyState = MockWebSocket.CONNECTING
  onopen: ((event: Event) => void) | null = null
  onmessage: ((event: MessageEvent) => void) | null = null
  onclose: ((event: CloseEvent) => void) | null = null
  onerror: ((event: Event) => void) | null = null

  constructor(public url: string) {
    setTimeout(() => {
      this.readyState = MockWebSocket.OPEN
      this.onopen?.(new Event('open'))
    }, 10)
  }

  send(data: string) {
    // Mock send functionality
  }

  close() {
    this.readyState = MockWebSocket.CLOSED
    this.onclose?.(new CloseEvent('close'))
  }

  // Helper method to simulate receiving a message
  simulateMessage(data: any) {
    if (this.onmessage && this.readyState === MockWebSocket.OPEN) {
      this.onmessage(new MessageEvent('message', { data: JSON.stringify(data) }))
    }
  }
}

// Mock performance.now() for consistent testing
const mockPerformanceNow = vi.fn()
Object.defineProperty(window, 'performance', {
  value: { now: mockPerformanceNow },
  writable: true
})

describe('WebSocket Performance Integration Tests', () => {
  let wsService: WebSocketService
  let mockWebSocket: MockWebSocket

  beforeEach(() => {
    // Reset performance.now mock
    mockPerformanceNow.mockReturnValue(1000)
    
    // Mock WebSocket constructor
    global.WebSocket = vi.fn().mockImplementation((url) => {
      mockWebSocket = new MockWebSocket(url)
      return mockWebSocket
    }) as any

    wsService = new WebSocketService({
      url: 'ws://localhost:8000/ws',
      reconnectInterval: 1000,
      maxReconnectAttempts: 3
    })
  })

  afterEach(() => {
    wsService.disconnect()
    vi.clearAllMocks()
  })

  describe('Performance Metrics Collection', () => {
    it('should initialize with default performance metrics', () => {
      const metrics = wsService.getPerformanceMetrics()
      
      expect(metrics.messageLatency).toEqual([])
      expect(metrics.averageLatency).toBe(0)
      expect(metrics.maxLatency).toBe(0)
      expect(metrics.minLatency).toBe(Infinity)
      expect(metrics.messagesProcessed).toBe(0)
      expect(metrics.messagesPerSecond).toBe(0)
    })

    it('should calculate latency correctly for messages with timestamps', async () => {
      wsService.connect()
      
      // Wait for connection
      await new Promise(resolve => setTimeout(resolve, 20))
      
      // Mock current time for latency calculation
      const messageTimestamp = 500_000_000 // 500ms in nanoseconds  
      const receiveTime = 600 // 600ms
      mockPerformanceNow.mockReturnValue(receiveTime)
      
      // Simulate receiving a message with timestamp
      mockWebSocket.simulateMessage({
        type: 'messagebus',
        topic: 'test.performance',
        timestamp: messageTimestamp,
        payload: { test: 'data' }
      })
      
      const metrics = wsService.getPerformanceMetrics()
      expect(metrics.messagesProcessed).toBe(1)
      expect(metrics.messageLatency).toHaveLength(1)
      
      // Expected latency: 600ms - (500_000_000 / 1_000_000)ms = 600 - 500 = 100ms
      expect(metrics.messageLatency[0]).toBe(100)
      expect(metrics.averageLatency).toBe(100)
      expect(metrics.maxLatency).toBe(100)
      expect(metrics.minLatency).toBe(100)
    })

    it('should handle multiple messages and calculate statistics correctly', async () => {
      wsService.connect()
      await new Promise(resolve => setTimeout(resolve, 20))
      
      const latencies = [50, 75, 100, 125, 150] // Expected latencies in ms
      
      for (let i = 0; i < latencies.length; i++) {
        const messageTime = 1000 - latencies[i] // Calculate message timestamp
        const receiveTime = 1000
        
        mockPerformanceNow.mockReturnValue(receiveTime)
        
        mockWebSocket.simulateMessage({
          type: 'messagebus',
          topic: 'test.performance',
          timestamp: messageTime * 1_000_000, // Convert to nanoseconds
          payload: { message: i }
        })
      }
      
      const metrics = wsService.getPerformanceMetrics()
      expect(metrics.messagesProcessed).toBe(5)
      expect(metrics.messageLatency).toHaveLength(5)
      
      // Check average calculation
      const expectedAverage = latencies.reduce((a, b) => a + b, 0) / latencies.length
      expect(metrics.averageLatency).toBe(expectedAverage)
      
      // Check min/max
      expect(metrics.minLatency).toBe(Math.min(...latencies))
      expect(metrics.maxLatency).toBe(Math.max(...latencies))
    })

    it('should limit latency buffer to maximum samples', async () => {
      wsService.connect()
      await new Promise(resolve => setTimeout(resolve, 20))
      
      // Send more than 100 messages (buffer limit)
      for (let i = 0; i < 120; i++) {
        mockPerformanceNow.mockReturnValue(1000 + i)
        
        mockWebSocket.simulateMessage({
          type: 'messagebus',
          topic: 'test.performance',
          timestamp: (1000 + i - 50) * 1_000_000, // 50ms latency
          payload: { message: i }
        })
      }
      
      const metrics = wsService.getPerformanceMetrics()
      expect(metrics.messagesProcessed).toBe(120)
      expect(metrics.messageLatency).toHaveLength(100) // Should be limited to 100
    })
  })

  describe('Performance Requirements Validation', () => {
    it('should meet <100ms latency requirement for normal operation', async () => {
      wsService.connect()
      await new Promise(resolve => setTimeout(resolve, 20))
      
      // Simulate messages with latencies under 100ms
      const goodLatencies = [25, 45, 60, 80, 95]
      
      for (const latency of goodLatencies) {
        mockPerformanceNow.mockReturnValue(1000)
        
        mockWebSocket.simulateMessage({
          type: 'messagebus',
          topic: 'test.performance',
          timestamp: (1000 - latency) * 1_000_000,
          payload: { test: 'data' }
        })
      }
      
      const metrics = wsService.getPerformanceMetrics()
      expect(metrics.averageLatency).toBeLessThan(100)
      expect(metrics.maxLatency).toBeLessThan(100)
    })

    it('should detect when latency exceeds 100ms requirement', async () => {
      wsService.connect()
      await new Promise(resolve => setTimeout(resolve, 20))
      
      // Simulate messages with high latencies
      const badLatencies = [120, 150, 200, 250]
      
      for (const latency of badLatencies) {
        mockPerformanceNow.mockReturnValue(1000)
        
        mockWebSocket.simulateMessage({
          type: 'messagebus',
          topic: 'test.performance',
          timestamp: (1000 - latency) * 1_000_000,
          payload: { test: 'data' }
        })
      }
      
      const metrics = wsService.getPerformanceMetrics()
      expect(metrics.averageLatency).toBeGreaterThan(100)
      expect(metrics.maxLatency).toBeGreaterThan(100)
    })

    it('should handle high-frequency messages (100+ per second)', async () => {
      wsService.connect()
      await new Promise(resolve => setTimeout(resolve, 20))
      
      const messageCount = 150
      const timeInterval = 1000 // 1 second
      
      // Simulate rapid message sending
      for (let i = 0; i < messageCount; i++) {
        const currentTime = 1000 + (i * timeInterval / messageCount)
        mockPerformanceNow.mockReturnValue(currentTime)
        
        mockWebSocket.simulateMessage({
          type: 'messagebus',
          topic: 'test.performance',
          timestamp: (currentTime - 10) * 1_000_000, // 10ms latency
          payload: { message: i }
        })
      }
      
      const metrics = wsService.getPerformanceMetrics()
      expect(metrics.messagesProcessed).toBe(messageCount)
      expect(metrics.averageLatency).toBeLessThan(100) // Should maintain good latency
    })
  })

  describe('Connection Performance', () => {
    it('should establish connection within 2 seconds', async () => {
      const startTime = Date.now()
      
      wsService.connect()
      
      // Wait for connection establishment
      return new Promise<void>((resolve) => {
        wsService.addStatusHandler((status) => {
          if (status === 'connected') {
            const connectionTime = Date.now() - startTime
            expect(connectionTime).toBeLessThan(2000)
            resolve()
          }
        })
      })
    })

    it('should reconnect within 5 seconds after disconnection', async () => {
      wsService.connect()
      await new Promise(resolve => setTimeout(resolve, 20))
      
      // Force disconnection
      mockWebSocket.close()
      
      const reconnectStartTime = Date.now()
      
      return new Promise<void>((resolve) => {
        wsService.addStatusHandler((status) => {
          if (status === 'connected') {
            const reconnectionTime = Date.now() - reconnectStartTime
            expect(reconnectionTime).toBeLessThan(5000)
            resolve()
          }
        })
        
        // Simulate reconnection after delay
        setTimeout(() => {
          const newMockWS = new MockWebSocket('ws://localhost:8000/ws')
          mockWebSocket = newMockWS
        }, 1000)
      })
    })
  })

  describe('Message Queue Performance', () => {
    it('should handle message queuing without dropping messages', async () => {
      wsService.connect()
      await new Promise(resolve => setTimeout(resolve, 20))
      
      const messageCount = 50
      const messageHandlerResults: any[] = []
      
      // Add message handler to track received messages
      wsService.addMessageHandler((message) => {
        messageHandlerResults.push(message)
      })
      
      // Send rapid burst of messages
      for (let i = 0; i < messageCount; i++) {
        mockPerformanceNow.mockReturnValue(1000 + i)
        
        mockWebSocket.simulateMessage({
          type: 'messagebus',
          topic: 'test.queue',
          timestamp: (1000 + i - 5) * 1_000_000,
          payload: { messageId: i }
        })
      }
      
      // Allow time for message processing
      await new Promise(resolve => setTimeout(resolve, 100))
      
      expect(messageHandlerResults).toHaveLength(messageCount)
      
      // Verify all messages were processed in order
      for (let i = 0; i < messageCount; i++) {
        expect(messageHandlerResults[i].payload.messageId).toBe(i)
      }
    })
  })

  describe('Error Handling Performance', () => {
    it('should handle malformed messages without affecting performance', async () => {
      wsService.connect()
      await new Promise(resolve => setTimeout(resolve, 20))
      
      // Send mix of good and bad messages
      for (let i = 0; i < 10; i++) {
        if (i % 2 === 0) {
          // Good message
          mockPerformanceNow.mockReturnValue(1000 + i)
          mockWebSocket.simulateMessage({
            type: 'messagebus',
            topic: 'test.good',
            timestamp: (1000 + i - 10) * 1_000_000,
            payload: { good: true }
          })
        } else {
          // Send malformed JSON (should be handled gracefully)
          if (mockWebSocket.onmessage) {
            mockWebSocket.onmessage(new MessageEvent('message', { 
              data: 'invalid json{' 
            }))
          }
        }
      }
      
      const metrics = wsService.getPerformanceMetrics()
      expect(metrics.messagesProcessed).toBe(5) // Only good messages counted
      expect(metrics.averageLatency).toBeGreaterThan(0) // Should have valid latency data
    })
  })
})