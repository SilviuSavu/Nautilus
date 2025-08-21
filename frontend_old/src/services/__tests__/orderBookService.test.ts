import { describe, it, expect, beforeEach } from 'vitest'
import { orderBookService } from '../orderBookService'
import { 
  OrderBookData, 
  OrderBookAggregationSettings, 
  ProcessedOrderBookLevel 
} from '../../types/orderBook'

describe('OrderBookService', () => {
  let mockOrderBookData: OrderBookData

  beforeEach(() => {
    mockOrderBookData = {
      symbol: 'AAPL',
      venue: 'NASDAQ',
      bids: [
        { price: 150.25, quantity: 100, orderCount: 5 },
        { price: 150.24, quantity: 200, orderCount: 8 },
        { price: 150.23, quantity: 150, orderCount: 3 },
        { price: 150.22, quantity: 300, orderCount: 12 },
        { price: 150.21, quantity: 75, orderCount: 2 }
      ],
      asks: [
        { price: 150.26, quantity: 80, orderCount: 4 },
        { price: 150.27, quantity: 120, orderCount: 6 },
        { price: 150.28, quantity: 200, orderCount: 9 },
        { price: 150.29, quantity: 150, orderCount: 7 },
        { price: 150.30, quantity: 100, orderCount: 3 }
      ],
      timestamp: Date.now()
    }
  })

  describe('processOrderBookData', () => {
    it('should process order book data correctly', () => {
      const aggregationSettings: OrderBookAggregationSettings = {
        enabled: false,
        increment: 0.01,
        maxLevels: 20
      }

      const result = orderBookService.processOrderBookData(mockOrderBookData, aggregationSettings)

      expect(result.symbol).toBe('AAPL')
      expect(result.venue).toBe('NASDAQ')
      expect(result.bids).toHaveLength(5)
      expect(result.asks).toHaveLength(5)
      expect(result.totalBidVolume).toBe(825) // 100+200+150+300+75
      expect(result.totalAskVolume).toBe(650) // 80+120+200+150+100
    })

    it('should sort bids in descending order and asks in ascending order', () => {
      const aggregationSettings: OrderBookAggregationSettings = {
        enabled: false,
        increment: 0.01,
        maxLevels: 20
      }

      const result = orderBookService.processOrderBookData(mockOrderBookData, aggregationSettings)

      // Bids should be sorted highest to lowest
      expect(result.bids[0].price).toBe(150.25)
      expect(result.bids[4].price).toBe(150.21)

      // Asks should be sorted lowest to highest
      expect(result.asks[0].price).toBe(150.26)
      expect(result.asks[4].price).toBe(150.30)
    })

    it('should calculate spread correctly', () => {
      const aggregationSettings: OrderBookAggregationSettings = {
        enabled: false,
        increment: 0.01,
        maxLevels: 20
      }

      const result = orderBookService.processOrderBookData(mockOrderBookData, aggregationSettings)

      expect(result.spread.bestBid).toBe(150.25)
      expect(result.spread.bestAsk).toBe(150.26)
      expect(result.spread.spread).toBe(0.01)
      expect(result.spread.spreadPercentage).toBeCloseTo(0.0067, 4) // (0.01 / 150.255) * 100
    })

    it('should calculate cumulative quantities correctly', () => {
      const aggregationSettings: OrderBookAggregationSettings = {
        enabled: false,
        increment: 0.01,
        maxLevels: 20
      }

      const result = orderBookService.processOrderBookData(mockOrderBookData, aggregationSettings)

      // Check bid cumulative
      expect(result.bids[0].cumulative).toBe(100)
      expect(result.bids[1].cumulative).toBe(300) // 100 + 200
      expect(result.bids[2].cumulative).toBe(450) // 100 + 200 + 150

      // Check ask cumulative
      expect(result.asks[0].cumulative).toBe(80)
      expect(result.asks[1].cumulative).toBe(200) // 80 + 120
      expect(result.asks[2].cumulative).toBe(400) // 80 + 120 + 200
    })

    it('should calculate percentages correctly', () => {
      const aggregationSettings: OrderBookAggregationSettings = {
        enabled: false,
        increment: 0.01,
        maxLevels: 20
      }

      const result = orderBookService.processOrderBookData(mockOrderBookData, aggregationSettings)

      // Total bid volume is 825
      expect(result.bids[0].percentage).toBeCloseTo(12.12, 2) // 100/825 * 100
      expect(result.bids[1].percentage).toBeCloseTo(24.24, 2) // 200/825 * 100

      // Total ask volume is 650
      expect(result.asks[0].percentage).toBeCloseTo(12.31, 2) // 80/650 * 100
      expect(result.asks[1].percentage).toBeCloseTo(18.46, 2) // 120/650 * 100
    })

    it('should handle aggregation when enabled', () => {
      const aggregationSettings: OrderBookAggregationSettings = {
        enabled: true,
        increment: 0.05,
        maxLevels: 20
      }

      const result = orderBookService.processOrderBookData(mockOrderBookData, aggregationSettings)

      // With 0.05 increment, some levels should be aggregated
      expect(result.bids.length).toBeLessThan(5)
      expect(result.asks.length).toBeLessThan(5)
    })

    it('should limit levels to maxLevels', () => {
      const aggregationSettings: OrderBookAggregationSettings = {
        enabled: false,
        increment: 0.01,
        maxLevels: 3
      }

      const result = orderBookService.processOrderBookData(mockOrderBookData, aggregationSettings)

      expect(result.bids).toHaveLength(3)
      expect(result.asks).toHaveLength(3)
    })
  })

  describe('validateOrderBookData', () => {
    it('should validate correct order book data', () => {
      const isValid = orderBookService.validateOrderBookData(mockOrderBookData)
      expect(isValid).toBe(true)
    })

    it('should reject invalid data structures', () => {
      expect(orderBookService.validateOrderBookData(null as any)).toBe(false)
      expect(orderBookService.validateOrderBookData(undefined as any)).toBe(false)
      expect(orderBookService.validateOrderBookData('invalid' as any)).toBe(false)
    })

    it('should reject data missing required fields', () => {
      const invalidData = { ...mockOrderBookData }
      delete (invalidData as any).symbol
      expect(orderBookService.validateOrderBookData(invalidData)).toBe(false)
    })

    it('should reject data with invalid bids/asks', () => {
      const invalidData = {
        ...mockOrderBookData,
        bids: 'not an array' as any
      }
      expect(orderBookService.validateOrderBookData(invalidData)).toBe(false)
    })

    it('should reject levels with invalid price/quantity', () => {
      const invalidData = {
        ...mockOrderBookData,
        bids: [
          { price: -100, quantity: 100 } // Negative price
        ]
      }
      expect(orderBookService.validateOrderBookData(invalidData)).toBe(false)
    })
  })

  describe('normalizeIBOrderBookData', () => {
    it('should normalize valid IB data', () => {
      const ibData = {
        symbol: 'AAPL',
        exchange: 'NASDAQ',
        marketDepth: {
          bids: [
            { price: '150.25', size: '100', count: '5' },
            { price: '150.24', size: '200', count: '8' }
          ],
          asks: [
            { price: '150.26', size: '80', count: '4' },
            { price: '150.27', size: '120', count: '6' }
          ]
        },
        timestamp: Date.now()
      }

      const result = orderBookService.normalizeIBOrderBookData(ibData)

      expect(result).not.toBeNull()
      expect(result!.symbol).toBe('AAPL')
      expect(result!.venue).toBe('NASDAQ')
      expect(result!.bids).toHaveLength(2)
      expect(result!.asks).toHaveLength(2)
      expect(result!.bids[0].price).toBe(150.25)
      expect(result!.bids[0].quantity).toBe(100)
      expect(result!.bids[0].orderCount).toBe(5)
    })

    it('should handle direct bid/ask arrays', () => {
      const ibData = {
        symbol: 'AAPL',
        exchange: 'NASDAQ',
        bids: [
          { price: 150.25, size: 100 },
          { price: 150.24, size: 200 }
        ],
        asks: [
          { price: 150.26, size: 80 },
          { price: 150.27, size: 120 }
        ],
        timestamp: Date.now()
      }

      const result = orderBookService.normalizeIBOrderBookData(ibData)

      expect(result).not.toBeNull()
      expect(result!.bids).toHaveLength(2)
      expect(result!.asks).toHaveLength(2)
    })

    it('should return null for invalid IB data', () => {
      expect(orderBookService.normalizeIBOrderBookData(null)).toBeNull()
      expect(orderBookService.normalizeIBOrderBookData({})).toBeNull()
      expect(orderBookService.normalizeIBOrderBookData({ invalid: 'data' })).toBeNull()
    })
  })

  describe('throttleUpdates', () => {
    it('should throttle function calls', async () => {
      let callCount = 0
      const throttledFn = orderBookService.throttleUpdates(
        () => { callCount++ },
        2 // 2 calls per second
      )

      // Make multiple rapid calls
      throttledFn('data1')
      throttledFn('data2')
      throttledFn('data3')
      throttledFn('data4')

      // Only first call should be executed immediately
      expect(callCount).toBe(1)

      // Wait for throttle period to pass
      await new Promise(resolve => setTimeout(resolve, 600))

      throttledFn('data5')
      expect(callCount).toBe(2)
    })
  })

  describe('getOptimalAggregationIncrement', () => {
    it('should calculate appropriate increment for given levels', () => {
      const bids = mockOrderBookData.bids
      const asks = mockOrderBookData.asks

      const increment = orderBookService.getOptimalAggregationIncrement(bids, asks)

      expect(increment).toBeGreaterThan(0)
      expect(increment).toBeLessThanOrEqual(1)
    })

    it('should return 0 for small level counts', () => {
      const bids = [{ price: 150.25, quantity: 100 }]
      const asks = [{ price: 150.26, quantity: 80 }]

      const increment = orderBookService.getOptimalAggregationIncrement(bids, asks)

      expect(increment).toBe(0)
    })

    it('should return default increment for empty levels', () => {
      const increment = orderBookService.getOptimalAggregationIncrement([], [])
      expect(increment).toBe(0.01)
    })
  })

  describe('performance metrics', () => {
    it('should track performance metrics', () => {
      const aggregationSettings: OrderBookAggregationSettings = {
        enabled: false,
        increment: 0.01,
        maxLevels: 20
      }

      // Process some data to generate metrics
      orderBookService.processOrderBookData(mockOrderBookData, aggregationSettings)
      orderBookService.processOrderBookData(mockOrderBookData, aggregationSettings)

      const metrics = orderBookService.getPerformanceMetrics()

      expect(metrics.updateLatency).toHaveLength(2)
      expect(metrics.averageLatency).toBeGreaterThan(0)
      expect(metrics.maxLatency).toBeGreaterThan(0)
      expect(metrics.lastUpdateTime).toBeGreaterThan(0)
    })

    it('should reset performance metrics', () => {
      const aggregationSettings: OrderBookAggregationSettings = {
        enabled: false,
        increment: 0.01,
        maxLevels: 20
      }

      // Generate some metrics
      orderBookService.processOrderBookData(mockOrderBookData, aggregationSettings)

      orderBookService.resetPerformanceMetrics()

      const metrics = orderBookService.getPerformanceMetrics()
      expect(metrics.updateLatency).toHaveLength(0)
      expect(metrics.averageLatency).toBe(0)
      expect(metrics.maxLatency).toBe(0)
    })
  })
})