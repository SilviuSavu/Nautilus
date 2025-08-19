import React from 'react'
import { render, act } from '@testing-library/react'
import { describe, it, expect, vi, beforeEach } from 'vitest'
import { orderBookService } from '../../../services/orderBookService'
import { OrderBookDisplay } from '../OrderBookDisplay'
import { 
  OrderBookData, 
  OrderBookAggregationSettings,
  ProcessedOrderBookLevel 
} from '../../../types/orderBook'
import { Instrument } from '../../Chart/types/chartTypes'

describe('OrderBook Performance Tests', () => {
  const mockInstrument: Instrument = {
    id: 'AAPL-STK',
    symbol: 'AAPL',
    name: 'Apple Inc.',
    venue: 'NASDAQ',
    assetClass: 'STK',
    currency: 'USD'
  }

  // Generate large order book data for performance testing
  const generateLargeOrderBookData = (levels: number): OrderBookData => {
    const bids = []
    const asks = []
    const basePrice = 150.00

    for (let i = 0; i < levels; i++) {
      bids.push({
        price: basePrice - (i * 0.01),
        quantity: Math.floor(Math.random() * 1000) + 100,
        orderCount: Math.floor(Math.random() * 10) + 1
      })

      asks.push({
        price: basePrice + 0.01 + (i * 0.01),
        quantity: Math.floor(Math.random() * 1000) + 100,
        orderCount: Math.floor(Math.random() * 10) + 1
      })
    }

    return {
      symbol: 'AAPL',
      venue: 'NASDAQ',
      bids,
      asks,
      timestamp: Date.now()
    }
  }

  beforeEach(() => {
    vi.clearAllMocks()
    orderBookService.resetPerformanceMetrics()
  })

  describe('Data Processing Performance', () => {
    it('should process order book data within 10ms for 20 levels', () => {
      const data = generateLargeOrderBookData(20)
      const aggregationSettings: OrderBookAggregationSettings = {
        enabled: false,
        increment: 0.01,
        maxLevels: 20
      }

      const startTime = performance.now()
      const result = orderBookService.processOrderBookData(data, aggregationSettings)
      const endTime = performance.now()

      const processingTime = endTime - startTime
      expect(processingTime).toBeLessThan(10) // Should complete within 10ms
      expect(result.bids).toHaveLength(20)
      expect(result.asks).toHaveLength(20)
    })

    it('should process order book data within 50ms for 100 levels', () => {
      const data = generateLargeOrderBookData(100)
      const aggregationSettings: OrderBookAggregationSettings = {
        enabled: false,
        increment: 0.01,
        maxLevels: 100
      }

      const startTime = performance.now()
      const result = orderBookService.processOrderBookData(data, aggregationSettings)
      const endTime = performance.now()

      const processingTime = endTime - startTime
      expect(processingTime).toBeLessThan(50) // Should complete within 50ms
      expect(result.bids).toHaveLength(100)
      expect(result.asks).toHaveLength(100)
    })

    it('should handle aggregation efficiently for deep books', () => {
      const data = generateLargeOrderBookData(200) // Deep book
      const aggregationSettings: OrderBookAggregationSettings = {
        enabled: true,
        increment: 0.05,
        maxLevels: 20
      }

      const startTime = performance.now()
      const result = orderBookService.processOrderBookData(data, aggregationSettings)
      const endTime = performance.now()

      const processingTime = endTime - startTime
      expect(processingTime).toBeLessThan(30) // Aggregation should still be fast
      expect(result.bids.length).toBeLessThanOrEqual(20)
      expect(result.asks.length).toBeLessThanOrEqual(20)
    })

    it('should maintain performance with rapid successive processing', () => {
      const data = generateLargeOrderBookData(50)
      const aggregationSettings: OrderBookAggregationSettings = {
        enabled: false,
        increment: 0.01,
        maxLevels: 50
      }

      const iterations = 100
      const startTime = performance.now()

      for (let i = 0; i < iterations; i++) {
        orderBookService.processOrderBookData(data, aggregationSettings)
      }

      const endTime = performance.now()
      const totalTime = endTime - startTime
      const averageTime = totalTime / iterations

      expect(averageTime).toBeLessThan(5) // Average under 5ms per operation
      expect(totalTime).toBeLessThan(500) // Total under 500ms for 100 operations
    })
  })

  describe('Memory Performance', () => {
    it('should not accumulate memory with repeated processing', () => {
      const data = generateLargeOrderBookData(50)
      const aggregationSettings: OrderBookAggregationSettings = {
        enabled: false,
        increment: 0.01,
        maxLevels: 50
      }

      // Initial memory baseline (approximation)
      const initialHeap = (performance as any).memory?.usedJSHeapSize || 0

      // Process many updates
      for (let i = 0; i < 1000; i++) {
        const result = orderBookService.processOrderBookData(data, aggregationSettings)
        // Ensure result is used to prevent optimization
        expect(result.bids).toBeDefined()
      }

      // Force garbage collection if available
      if (global.gc) {
        global.gc()
      }

      const finalHeap = (performance as any).memory?.usedJSHeapSize || 0
      
      // Memory should not grow significantly (allow for some variance)
      if (initialHeap > 0 && finalHeap > 0) {
        const memoryGrowth = finalHeap - initialHeap
        const memoryGrowthMB = memoryGrowth / (1024 * 1024)
        expect(memoryGrowthMB).toBeLessThan(10) // Less than 10MB growth
      }
    })

    it('should limit performance metrics array size', () => {
      const data = generateLargeOrderBookData(20)
      const aggregationSettings: OrderBookAggregationSettings = {
        enabled: false,
        increment: 0.01,
        maxLevels: 20
      }

      // Generate many metrics
      for (let i = 0; i < 200; i++) {
        orderBookService.processOrderBookData(data, aggregationSettings)
      }

      const metrics = orderBookService.getPerformanceMetrics()
      
      // Should limit array size to prevent memory bloat
      expect(metrics.updateLatency.length).toBeLessThanOrEqual(100)
    })
  })

  describe('Update Frequency Performance', () => {
    it('should handle 100+ updates per second', async () => {
      const data = generateLargeOrderBookData(20)
      const aggregationSettings: OrderBookAggregationSettings = {
        enabled: false,
        increment: 0.01,
        maxLevels: 20
      }

      const updatesPerSecond = 100
      const testDuration = 1000 // 1 second
      const expectedUpdates = (updatesPerSecond * testDuration) / 1000

      let processedUpdates = 0
      const startTime = Date.now()

      const intervalId = setInterval(() => {
        if (Date.now() - startTime >= testDuration) {
          clearInterval(intervalId)
          return
        }

        const result = orderBookService.processOrderBookData(data, aggregationSettings)
        if (result) processedUpdates++
      }, 1000 / updatesPerSecond) // 10ms intervals for 100 updates/sec

      // Wait for test to complete
      await new Promise(resolve => setTimeout(resolve, testDuration + 100))

      expect(processedUpdates).toBeGreaterThanOrEqual(expectedUpdates * 0.9) // Allow 10% tolerance
    })

    it('should throttle updates effectively', () => {
      let callCount = 0
      const throttledFn = orderBookService.throttleUpdates(
        () => { callCount++ },
        10 // 10 calls per second
      )

      // Make 50 rapid calls
      for (let i = 0; i < 50; i++) {
        throttledFn('test')
      }

      // Should only execute once immediately
      expect(callCount).toBe(1)
    })
  })

  describe('Rendering Performance', () => {
    it('should render order book within 16ms for smooth 60fps', async () => {
      const mockData = {
        symbol: 'AAPL',
        venue: 'NASDAQ',
        bids: Array.from({ length: 20 }, (_, i) => ({
          id: `bid-${i}`,
          price: 150 - i * 0.01,
          quantity: 100 + i * 10,
          cumulative: (100 + i * 10) * (i + 1),
          percentage: 100 / 20,
          orderCount: 5
        })),
        asks: Array.from({ length: 20 }, (_, i) => ({
          id: `ask-${i}`,
          price: 150.01 + i * 0.01,
          quantity: 100 + i * 10,
          cumulative: (100 + i * 10) * (i + 1),
          percentage: 100 / 20,
          orderCount: 5
        })),
        spread: {
          bestBid: 150.00,
          bestAsk: 150.01,
          spread: 0.01,
          spreadPercentage: 0.0067
        },
        timestamp: Date.now(),
        totalBidVolume: 2000,
        totalAskVolume: 2000
      }

      // Mock the hook to return our test data
      vi.doMock('../../../hooks/useOrderBookData', () => ({
        useOrderBookData: () => ({
          orderBookData: mockData,
          isLoading: false,
          error: null,
          connectionStatus: 'connected' as const,
          subscriptions: [],
          aggregationSettings: { enabled: false, increment: 0.01, maxLevels: 20 },
          displaySettings: { showSpread: true, showOrderCount: false, colorScheme: 'default' as const, decimals: 2 },
          performanceMetrics: null,
          subscribeToOrderBook: vi.fn(),
          unsubscribeFromOrderBook: vi.fn(),
          updateAggregationSettings: vi.fn(),
          updateDisplaySettings: vi.fn(),
          clearOrderBook: vi.fn()
        })
      }))

      const startTime = performance.now()
      
      act(() => {
        render(<OrderBookDisplay instrument={mockInstrument} />)
      })

      const endTime = performance.now()
      const renderTime = endTime - startTime

      // Should render within 16ms for 60fps
      expect(renderTime).toBeLessThan(16)
    })

    it('should maintain stable memory usage during repeated renders', () => {
      const { rerender } = render(<OrderBookDisplay instrument={null} />)

      const initialHeap = (performance as any).memory?.usedJSHeapSize || 0

      // Perform multiple re-renders
      for (let i = 0; i < 100; i++) {
        rerender(<OrderBookDisplay instrument={mockInstrument} />)
        rerender(<OrderBookDisplay instrument={null} />)
      }

      // Force garbage collection if available
      if (global.gc) {
        global.gc()
      }

      const finalHeap = (performance as any).memory?.usedJSHeapSize || 0

      if (initialHeap > 0 && finalHeap > 0) {
        const memoryGrowth = finalHeap - initialHeap
        const memoryGrowthMB = memoryGrowth / (1024 * 1024)
        expect(memoryGrowthMB).toBeLessThan(5) // Less than 5MB growth
      }
    })
  })

  describe('Large Dataset Performance', () => {
    it('should handle order books with 50+ levels efficiently', () => {
      const data = generateLargeOrderBookData(50)
      const aggregationSettings: OrderBookAggregationSettings = {
        enabled: false,
        increment: 0.01,
        maxLevels: 50
      }

      const startTime = performance.now()
      const result = orderBookService.processOrderBookData(data, aggregationSettings)
      const endTime = performance.now()

      const processingTime = endTime - startTime

      expect(processingTime).toBeLessThan(20) // Should complete within 20ms
      expect(result.bids).toHaveLength(50)
      expect(result.asks).toHaveLength(50)
      
      // Verify cumulative calculations are correct
      expect(result.bids[0].cumulative).toBe(result.bids[0].quantity)
      expect(result.bids[1].cumulative).toBe(result.bids[0].quantity + result.bids[1].quantity)
    })

    it('should optimize aggregation for very deep books', () => {
      const data = generateLargeOrderBookData(500) // Very deep book
      const aggregationSettings: OrderBookAggregationSettings = {
        enabled: true,
        increment: 0.1, // Aggressive aggregation
        maxLevels: 20
      }

      const startTime = performance.now()
      const result = orderBookService.processOrderBookData(data, aggregationSettings)
      const endTime = performance.now()

      const processingTime = endTime - startTime

      expect(processingTime).toBeLessThan(100) // Should complete within 100ms even for 500 levels
      expect(result.bids.length).toBeLessThanOrEqual(20)
      expect(result.asks.length).toBeLessThanOrEqual(20)
    })
  })

  describe('Performance Metrics Tracking', () => {
    it('should track metrics with minimal overhead', () => {
      const data = generateLargeOrderBookData(20)
      const aggregationSettings: OrderBookAggregationSettings = {
        enabled: false,
        increment: 0.01,
        maxLevels: 20
      }

      const iterations = 50
      const startTime = performance.now()

      for (let i = 0; i < iterations; i++) {
        orderBookService.processOrderBookData(data, aggregationSettings)
      }

      const endTime = performance.now()
      const totalTime = endTime - startTime

      const metrics = orderBookService.getPerformanceMetrics()

      // Metrics should be tracked
      expect(metrics.updateLatency).toHaveLength(iterations)
      expect(metrics.averageLatency).toBeGreaterThan(0)
      expect(metrics.maxLatency).toBeGreaterThan(0)

      // Tracking overhead should be minimal (< 20% overhead)
      const averageTimeWithMetrics = totalTime / iterations
      expect(averageTimeWithMetrics).toBeLessThan(2) // Should still be very fast
    })

    it('should calculate accurate performance statistics', () => {
      const data = generateLargeOrderBookData(10)
      const aggregationSettings: OrderBookAggregationSettings = {
        enabled: false,
        increment: 0.01,
        maxLevels: 10
      }

      // Process data multiple times to generate metrics
      for (let i = 0; i < 10; i++) {
        orderBookService.processOrderBookData(data, aggregationSettings)
      }

      const metrics = orderBookService.getPerformanceMetrics()

      // Verify metrics calculations
      expect(metrics.updateLatency).toHaveLength(10)
      expect(metrics.averageLatency).toBeGreaterThan(0)
      expect(metrics.maxLatency).toBeGreaterThanOrEqual(metrics.averageLatency)
      expect(metrics.lastUpdateTime).toBeGreaterThan(0)

      // Average should be reasonable
      const sum = metrics.updateLatency.reduce((a, b) => a + b, 0)
      const expectedAverage = sum / metrics.updateLatency.length
      expect(Math.abs(metrics.averageLatency - expectedAverage)).toBeLessThan(0.01)
    })
  })
})