/**
 * Performance benchmark tests for WebSocket latency requirements
 */

import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest'
import { WebSocketService } from '../../src/services/websocket'

describe('WebSocket Performance Benchmarks', () => {
  let wsService: WebSocketService
  let mockWebSocket: any

  beforeEach(() => {
    // Mock WebSocket
    mockWebSocket = {
      readyState: WebSocket.OPEN,
      onopen: null,
      onmessage: null,
      onclose: null,
      onerror: null,
      send: vi.fn(),
      close: vi.fn(),
      addEventListener: vi.fn(),
      removeEventListener: vi.fn(),
    }

    global.WebSocket = vi.fn().mockImplementation(() => mockWebSocket)

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

  describe('Latency Benchmarks', () => {
    it('should process messages within 100ms latency requirement', async () => {
      // Benchmark test for the <100ms requirement
      const testDuration = 1000 // 1 second test
      const messageInterval = 10 // Send message every 10ms
      const expectedMessages = testDuration / messageInterval
      
      wsService.connect()
      
      // Mock connection establishment
      setTimeout(() => {
        mockWebSocket.onopen?.()
      }, 10)

      const startTime = performance.now()
      const latencies: number[] = []
      let messagesReceived = 0

      // Add message handler to track processing time
      wsService.addMessageHandler((message) => {
        if (message.timestamp) {
          const receiveTime = performance.now()
          const messageTime = message.timestamp / 1000000 // Convert from nanoseconds
          const latency = receiveTime - messageTime
          latencies.push(latency)
        }
        messagesReceived++
      })

      // Simulate high-frequency message stream
      return new Promise<void>((resolve) => {
        let messageCount = 0
        const interval = setInterval(() => {
          if (messageCount >= expectedMessages) {
            clearInterval(interval)
            
            // Analyze results
            const avgLatency = latencies.reduce((a, b) => a + b, 0) / latencies.length
            const maxLatency = Math.max(...latencies)
            const minLatency = Math.min(...latencies)
            const totalTime = performance.now() - startTime
            
            // Performance assertions
            expect(avgLatency).toBeLessThan(100) // <100ms average latency
            expect(maxLatency).toBeLessThan(200) // Max latency shouldn't exceed 200ms
            expect(messagesReceived).toBe(expectedMessages) // No dropped messages
            expect(totalTime).toBeLessThan(testDuration + 100) // Test completed in time
            
            console.log(`Benchmark Results:
              - Messages processed: ${messagesReceived}
              - Average latency: ${avgLatency.toFixed(2)}ms
              - Max latency: ${maxLatency.toFixed(2)}ms
              - Min latency: ${minLatency.toFixed(2)}ms
              - Test duration: ${totalTime.toFixed(2)}ms`)
            
            resolve()
          }

          // Simulate message with realistic timestamp
          const messageTime = performance.now()
          mockWebSocket.onmessage?.({
            data: JSON.stringify({
              type: 'messagebus',
              topic: 'benchmark.test',
              timestamp: messageTime * 1000000, // Convert to nanoseconds
              payload: { messageId: messageCount }
            })
          })
          
          messageCount++
        }, messageInterval)
      })
    }, 10000) // 10 second timeout for benchmark

    it('should handle burst load of 100+ messages per second', async () => {
      wsService.connect()
      setTimeout(() => mockWebSocket.onopen?.(), 10)

      const burstSize = 150 // Messages
      const burstDuration = 1000 // 1 second
      const latencies: number[] = []
      let processedCount = 0

      wsService.addMessageHandler((message) => {
        if (message.timestamp) {
          const receiveTime = performance.now()
          const messageTime = message.timestamp / 1000000
          latencies.push(receiveTime - messageTime)
        }
        processedCount++
      })

      return new Promise<void>((resolve) => {
        const startTime = performance.now()
        
        // Send burst of messages
        for (let i = 0; i < burstSize; i++) {
          setTimeout(() => {
            const messageTime = performance.now()
            mockWebSocket.onmessage?.({
              data: JSON.stringify({
                type: 'messagebus',
                topic: 'burst.test',
                timestamp: messageTime * 1000000,
                payload: { burstId: i }
              })
            })
            
            // Check if burst is complete
            if (i === burstSize - 1) {
              setTimeout(() => {
                const totalTime = performance.now() - startTime
                const avgLatency = latencies.reduce((a, b) => a + b, 0) / latencies.length
                const messagesPerSecond = (processedCount / totalTime) * 1000
                
                // Performance assertions for burst load
                expect(processedCount).toBe(burstSize)
                expect(messagesPerSecond).toBeGreaterThan(100) // >100 msg/sec
                expect(avgLatency).toBeLessThan(100) // Maintain <100ms even under load
                expect(totalTime).toBeLessThan(burstDuration + 500) // Complete within reasonable time
                
                console.log(`Burst Load Results:
                  - Messages processed: ${processedCount}
                  - Messages per second: ${messagesPerSecond.toFixed(1)}
                  - Average latency: ${avgLatency.toFixed(2)}ms
                  - Total time: ${totalTime.toFixed(2)}ms`)
                
                resolve()
              }, 100) // Allow time for all messages to process
            }
          }, (i / burstSize) * burstDuration)
        }
      })
    })

    it('should maintain performance during connection stress test', async () => {
      const connectionCycles = 5
      const messagesPerCycle = 20
      const allLatencies: number[] = []
      let totalMessages = 0

      for (let cycle = 0; cycle < connectionCycles; cycle++) {
        wsService.connect()
        
        // Mock connection
        setTimeout(() => mockWebSocket.onopen?.(), 10)
        
        await new Promise<void>((resolve) => {
          let cycleMessages = 0
          
          wsService.addMessageHandler((message) => {
            if (message.timestamp) {
              const receiveTime = performance.now()
              const messageTime = message.timestamp / 1000000
              allLatencies.push(receiveTime - messageTime)
            }
            cycleMessages++
            totalMessages++
            
            if (cycleMessages === messagesPerCycle) {
              resolve()
            }
          })
          
          // Send messages for this cycle
          for (let i = 0; i < messagesPerCycle; i++) {
            setTimeout(() => {
              const messageTime = performance.now()
              mockWebSocket.onmessage?.({
                data: JSON.stringify({
                  type: 'messagebus',
                  topic: 'stress.test',
                  timestamp: messageTime * 1000000,
                  payload: { cycle, message: i }
                })
              })
            }, i * 10)
          }
        })
        
        // Disconnect and prepare for next cycle
        wsService.disconnect()
        mockWebSocket.onclose?.()
        await new Promise(resolve => setTimeout(resolve, 50))
      }
      
      const avgLatency = allLatencies.reduce((a, b) => a + b, 0) / allLatencies.length
      const maxLatency = Math.max(...allLatencies)
      
      // Performance should remain consistent across connection cycles
      expect(totalMessages).toBe(connectionCycles * messagesPerCycle)
      expect(avgLatency).toBeLessThan(100)
      expect(maxLatency).toBeLessThan(300) // Allow some tolerance for connection overhead
      
      console.log(`Connection Stress Test Results:
        - Total messages: ${totalMessages}
        - Connection cycles: ${connectionCycles}
        - Average latency: ${avgLatency.toFixed(2)}ms
        - Max latency: ${maxLatency.toFixed(2)}ms`)
    })
  })

  describe('Memory Performance', () => {
    it('should limit memory usage with large message volumes', async () => {
      wsService.connect()
      setTimeout(() => mockWebSocket.onopen?.(), 10)

      const largeMessageCount = 1000
      let processedCount = 0

      wsService.addMessageHandler(() => {
        processedCount++
      })

      // Send large volume of messages
      for (let i = 0; i < largeMessageCount; i++) {
        const messageTime = performance.now()
        mockWebSocket.onmessage?.({
          data: JSON.stringify({
            type: 'messagebus',
            topic: 'memory.test',
            timestamp: messageTime * 1000000,
            payload: { 
              id: i,
              // Add some data to make messages larger
              data: new Array(100).fill('x').join('')
            }
          })
        })
      }

      await new Promise(resolve => setTimeout(resolve, 100))

      const metrics = wsService.getPerformanceMetrics()
      
      // Should process all messages
      expect(processedCount).toBe(largeMessageCount)
      
      // Should limit latency buffer size to prevent memory growth
      expect(metrics.messageLatency.length).toBeLessThanOrEqual(100)
      
      // Should still maintain performance
      expect(metrics.averageLatency).toBeLessThan(100)
      
      console.log(`Memory Test Results:
        - Messages processed: ${processedCount}
        - Latency buffer size: ${metrics.messageLatency.length}
        - Average latency: ${metrics.averageLatency.toFixed(2)}ms`)
    })
  })

  describe('Real-world Scenario Benchmarks', () => {
    it('should handle market data stream simulation', async () => {
      // Simulate realistic market data stream
      wsService.connect()
      setTimeout(() => mockWebSocket.onopen?.(), 10)

      const symbols = ['EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD', 'USDCAD']
      const updateFrequency = 50 // Update every 50ms per symbol
      const testDuration = 2000 // 2 seconds
      const latencies: number[] = []
      let marketDataCount = 0

      wsService.addMessageHandler((message) => {
        if (message.topic?.startsWith('market.') && message.timestamp) {
          const receiveTime = performance.now()
          const messageTime = message.timestamp / 1000000
          latencies.push(receiveTime - messageTime)
          marketDataCount++
        }
      })

      return new Promise<void>((resolve) => {
        const intervals: NodeJS.Timeout[] = []
        
        // Start market data streams for each symbol
        symbols.forEach((symbol, symbolIndex) => {
          const interval = setInterval(() => {
            const messageTime = performance.now()
            mockWebSocket.onmessage?.({
              data: JSON.stringify({
                type: 'messagebus',
                topic: `market.${symbol.toLowerCase()}.quote`,
                timestamp: messageTime * 1000000,
                payload: {
                  symbol,
                  bid: 1.0000 + Math.random() * 0.1,
                  ask: 1.0010 + Math.random() * 0.1,
                  timestamp: messageTime
                }
              })
            })
          }, updateFrequency)
          
          intervals.push(interval)
        })

        // Stop after test duration
        setTimeout(() => {
          intervals.forEach(clearInterval)
          
          const avgLatency = latencies.reduce((a, b) => a + b, 0) / latencies.length
          const maxLatency = Math.max(...latencies)
          const expectedMessages = Math.floor(testDuration / updateFrequency) * symbols.length
          
          // Market data performance requirements
          expect(marketDataCount).toBeGreaterThan(expectedMessages * 0.9) // Allow 10% tolerance
          expect(avgLatency).toBeLessThan(100) // Critical for trading
          expect(maxLatency).toBeLessThan(200) // No spikes above 200ms
          
          console.log(`Market Data Simulation Results:
            - Symbols: ${symbols.length}
            - Messages received: ${marketDataCount}
            - Average latency: ${avgLatency.toFixed(2)}ms
            - Max latency: ${maxLatency.toFixed(2)}ms
            - Messages per second: ${(marketDataCount / (testDuration / 1000)).toFixed(1)}`)
          
          resolve()
        }, testDuration)
      })
    })
  })
})