import {
  OrderBookData,
  OrderBookLevel,
  ProcessedOrderBookData,
  ProcessedOrderBookLevel,
  OrderBookSpread,
  OrderBookAggregationSettings,
  OrderBookPerformanceMetrics,
  OrderBookMessage
} from '../types/orderBook'

class OrderBookService {
  private performanceMetrics: OrderBookPerformanceMetrics = {
    updateLatency: [],
    averageLatency: 0,
    maxLatency: 0,
    updatesPerSecond: 0,
    lastUpdateTime: Date.now()
  }
  
  private updateCount = 0
  private lastSecondTimestamp = Date.now()

  processOrderBookData(
    data: OrderBookData,
    aggregationSettings: OrderBookAggregationSettings
  ): ProcessedOrderBookData {
    const startTime = performance.now()

    try {
      // Process bids and asks
      const processedBids = this.processLevels(data.bids, 'bid', aggregationSettings)
      const processedAsks = this.processLevels(data.asks, 'ask', aggregationSettings)

      // Calculate spread
      const spread = this.calculateSpread(processedBids, processedAsks)

      // Calculate total volumes
      const totalBidVolume = processedBids.reduce((sum, level) => sum + level.quantity, 0)
      const totalAskVolume = processedAsks.reduce((sum, level) => sum + level.quantity, 0)

      const processedData: ProcessedOrderBookData = {
        symbol: data.symbol,
        venue: data.venue,
        bids: processedBids,
        asks: processedAsks,
        spread,
        timestamp: data.timestamp,
        totalBidVolume,
        totalAskVolume
      }

      // Update performance metrics
      const processingTime = performance.now() - startTime
      this.updatePerformanceMetrics(processingTime)

      return processedData
    } catch (error) {
      console.error('Error processing order book data:', error)
      throw error
    }
  }

  private processLevels(
    levels: OrderBookLevel[],
    side: 'bid' | 'ask',
    aggregationSettings: OrderBookAggregationSettings
  ): ProcessedOrderBookLevel[] {
    if (!levels || levels.length === 0) {
      return []
    }

    // Sort levels (bids descending, asks ascending)
    const sortedLevels = [...levels].sort((a, b) => 
      side === 'bid' ? b.price - a.price : a.price - b.price
    )

    // Apply aggregation if enabled
    const aggregatedLevels = aggregationSettings.enabled
      ? this.aggregateLevels(sortedLevels, aggregationSettings.increment, side)
      : sortedLevels

    // Limit to max levels
    const limitedLevels = aggregatedLevels.slice(0, aggregationSettings.maxLevels)

    // Calculate cumulative quantities and percentages
    let cumulativeQuantity = 0
    const totalQuantity = limitedLevels.reduce((sum, level) => sum + level.quantity, 0)

    return limitedLevels.map((level, index) => {
      cumulativeQuantity += level.quantity
      const percentage = totalQuantity > 0 ? (level.quantity / totalQuantity) * 100 : 0

      return {
        ...level,
        cumulative: cumulativeQuantity,
        percentage,
        id: `${side}-${level.price}-${index}`
      }
    })
  }

  private aggregateLevels(
    levels: OrderBookLevel[],
    increment: number,
    side: 'bid' | 'ask'
  ): OrderBookLevel[] {
    if (increment <= 0) return levels

    const aggregatedMap = new Map<number, OrderBookLevel>()

    levels.forEach(level => {
      // Calculate aggregation bucket
      const bucket = side === 'bid' 
        ? Math.floor(level.price / increment) * increment
        : Math.ceil(level.price / increment) * increment

      const existing = aggregatedMap.get(bucket)
      if (existing) {
        existing.quantity += level.quantity
        if (level.orderCount && existing.orderCount) {
          existing.orderCount += level.orderCount
        }
      } else {
        aggregatedMap.set(bucket, {
          price: bucket,
          quantity: level.quantity,
          orderCount: level.orderCount
        })
      }
    })

    return Array.from(aggregatedMap.values())
  }

  private calculateSpread(
    bids: ProcessedOrderBookLevel[],
    asks: ProcessedOrderBookLevel[]
  ): OrderBookSpread {
    const bestBid = bids.length > 0 ? bids[0].price : null
    const bestAsk = asks.length > 0 ? asks[0].price : null

    if (bestBid === null || bestAsk === null) {
      return {
        bestBid,
        bestAsk,
        spread: null,
        spreadPercentage: null
      }
    }

    const spread = bestAsk - bestBid
    const midPrice = (bestBid + bestAsk) / 2
    const spreadPercentage = midPrice > 0 ? (spread / midPrice) * 100 : null

    return {
      bestBid,
      bestAsk,
      spread,
      spreadPercentage
    }
  }

  validateOrderBookData(data: OrderBookData): boolean {
    if (!data || typeof data !== 'object') {
      console.warn('Order book data is not a valid object')
      return false
    }

    if (!data.symbol || typeof data.symbol !== 'string') {
      console.warn('Order book data missing valid symbol')
      return false
    }

    if (!data.venue || typeof data.venue !== 'string') {
      console.warn('Order book data missing valid venue')
      return false
    }

    if (!Array.isArray(data.bids)) {
      console.warn('Order book bids is not an array')
      return false
    }

    if (!Array.isArray(data.asks)) {
      console.warn('Order book asks is not an array')
      return false
    }

    // Validate level structure
    const validateLevel = (level: any): level is OrderBookLevel => {
      return level &&
        typeof level.price === 'number' &&
        typeof level.quantity === 'number' &&
        level.price > 0 &&
        level.quantity > 0
    }

    const validBids = data.bids.every(validateLevel)
    const validAsks = data.asks.every(validateLevel)

    if (!validBids) {
      console.warn('Invalid bid levels in order book data')
      return false
    }

    if (!validAsks) {
      console.warn('Invalid ask levels in order book data')
      return false
    }

    return true
  }

  normalizeIBOrderBookData(ibData: any): OrderBookData | null {
    try {
      if (!ibData || typeof ibData !== 'object') {
        console.warn('IB order book data is not a valid object')
        return null
      }

      // Handle different IB data formats
      const symbol = ibData.symbol || ibData.contract?.symbol
      const venue = ibData.exchange || ibData.contract?.exchange || 'SMART'

      if (!symbol) {
        console.warn('IB order book data missing symbol')
        return null
      }

      // Extract bid/ask levels from IB format
      let bids: OrderBookLevel[] = []
      let asks: OrderBookLevel[] = []

      // Handle IB market depth format
      if (ibData.marketDepth || ibData.market_depth) {
        const depth = ibData.marketDepth || ibData.market_depth
        
        if (depth.bids && Array.isArray(depth.bids)) {
          bids = depth.bids.map((bid: any) => ({
            price: parseFloat(bid.price),
            quantity: parseFloat(bid.size || bid.quantity || bid.volume),
            orderCount: bid.count ? parseInt(bid.count) : undefined
          })).filter((bid: OrderBookLevel) => bid.price > 0 && bid.quantity > 0)
        }

        if (depth.asks && Array.isArray(depth.asks)) {
          asks = depth.asks.map((ask: any) => ({
            price: parseFloat(ask.price),
            quantity: parseFloat(ask.size || ask.quantity || ask.volume),
            orderCount: ask.count ? parseInt(ask.count) : undefined
          })).filter((ask: OrderBookLevel) => ask.price > 0 && ask.quantity > 0)
        }
      }

      // Handle direct bid/ask arrays
      if (ibData.bids && Array.isArray(ibData.bids)) {
        bids = ibData.bids.map((bid: any) => ({
          price: parseFloat(bid.price || bid[0]),
          quantity: parseFloat(bid.size || bid.quantity || bid[1]),
          orderCount: bid.count ? parseInt(bid.count) : undefined
        })).filter((bid: OrderBookLevel) => bid.price > 0 && bid.quantity > 0)
      }

      if (ibData.asks && Array.isArray(ibData.asks)) {
        asks = ibData.asks.map((ask: any) => ({
          price: parseFloat(ask.price || ask[0]),
          quantity: parseFloat(ask.size || ask.quantity || ask[1]),
          orderCount: ask.count ? parseInt(ask.count) : undefined
        })).filter((ask: OrderBookLevel) => ask.price > 0 && ask.quantity > 0)
      }

      const normalizedData: OrderBookData = {
        symbol,
        venue,
        bids,
        asks,
        timestamp: ibData.timestamp || Date.now()
      }

      // Validate the normalized data
      if (!this.validateOrderBookData(normalizedData)) {
        console.warn('Normalized IB order book data failed validation')
        return null
      }

      return normalizedData
    } catch (error) {
      console.error('Error normalizing IB order book data:', error)
      return null
    }
  }

  throttleUpdates<T>(
    callback: (data: T) => void,
    maxUpdatesPerSecond: number = 10
  ): (data: T) => void {
    let lastCallTime = 0
    const minInterval = 1000 / maxUpdatesPerSecond

    return (data: T) => {
      const now = Date.now()
      if (now - lastCallTime >= minInterval) {
        lastCallTime = now
        callback(data)
      }
    }
  }

  private updatePerformanceMetrics(latency: number): void {
    const maxSamples = 100
    const currentTime = Date.now()

    // Track latency
    this.performanceMetrics.updateLatency.push(latency)
    if (this.performanceMetrics.updateLatency.length > maxSamples) {
      this.performanceMetrics.updateLatency.shift()
    }

    // Update max latency
    this.performanceMetrics.maxLatency = Math.max(
      this.performanceMetrics.maxLatency,
      latency
    )

    // Calculate average latency
    const sum = this.performanceMetrics.updateLatency.reduce((a, b) => a + b, 0)
    this.performanceMetrics.averageLatency = sum / this.performanceMetrics.updateLatency.length

    // Calculate updates per second
    this.updateCount++
    const timeDiff = currentTime - this.lastSecondTimestamp

    if (timeDiff >= 1000) {
      this.performanceMetrics.updatesPerSecond = this.updateCount / (timeDiff / 1000)
      this.performanceMetrics.lastUpdateTime = currentTime
      this.updateCount = 0
      this.lastSecondTimestamp = currentTime
    }
  }

  getPerformanceMetrics(): OrderBookPerformanceMetrics {
    return { ...this.performanceMetrics }
  }

  resetPerformanceMetrics(): void {
    this.performanceMetrics = {
      updateLatency: [],
      averageLatency: 0,
      maxLatency: 0,
      updatesPerSecond: 0,
      lastUpdateTime: Date.now()
    }
    this.updateCount = 0
    this.lastSecondTimestamp = Date.now()
  }

  createOrderBookMessage(data: OrderBookData): OrderBookMessage {
    return {
      type: 'order_book_update',
      symbol: data.symbol,
      venue: data.venue,
      bids: data.bids,
      asks: data.asks,
      timestamp: data.timestamp
    }
  }

  getOptimalAggregationIncrement(
    bids: OrderBookLevel[],
    asks: OrderBookLevel[]
  ): number {
    if (bids.length === 0 && asks.length === 0) {
      return 0.01 // Default increment
    }

    const allPrices = [...bids, ...asks].map(level => level.price)
    const minPrice = Math.min(...allPrices)
    const maxPrice = Math.max(...allPrices)
    const priceRange = maxPrice - minPrice

    // Calculate increment based on price range and level count
    const totalLevels = bids.length + asks.length
    const targetLevels = 20 // Target number of levels after aggregation

    if (totalLevels <= targetLevels) {
      return 0 // No aggregation needed
    }

    // Use 1% of price range as base increment
    let increment = priceRange * 0.01

    // Round to sensible values
    if (increment < 0.01) increment = 0.01
    else if (increment < 0.05) increment = 0.05
    else if (increment < 0.1) increment = 0.1
    else if (increment < 0.25) increment = 0.25
    else if (increment < 0.5) increment = 0.5
    else if (increment < 1) increment = 1
    else increment = Math.round(increment)

    return increment
  }
}

export const orderBookService = new OrderBookService()