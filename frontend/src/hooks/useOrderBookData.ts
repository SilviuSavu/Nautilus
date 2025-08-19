import { useState, useEffect, useCallback, useRef } from 'react'
import { webSocketService, WebSocketMessage } from '../services/websocket'
import { orderBookService } from '../services/orderBookService'
import {
  OrderBookData,
  ProcessedOrderBookData,
  OrderBookMessage,
  OrderBookAggregationSettings,
  OrderBookDisplaySettings,
  OrderBookSubscription,
  OrderBookPerformanceMetrics
} from '../types/orderBook'
import { Instrument } from '../components/Chart/types/chartTypes'

interface UseOrderBookDataOptions {
  maxUpdatesPerSecond?: number
  autoSubscribe?: boolean
  enablePerformanceTracking?: boolean
}

interface UseOrderBookDataReturn {
  orderBookData: ProcessedOrderBookData | null
  subscriptions: OrderBookSubscription[]
  aggregationSettings: OrderBookAggregationSettings
  displaySettings: OrderBookDisplaySettings
  isLoading: boolean
  error: string | null
  connectionStatus: 'connected' | 'disconnected' | 'error'
  performanceMetrics: OrderBookPerformanceMetrics | null
  
  // Actions
  subscribeToOrderBook: (instrument: Instrument) => void
  unsubscribeFromOrderBook: (instrument: Instrument) => void
  updateAggregationSettings: (settings: Partial<OrderBookAggregationSettings>) => void
  updateDisplaySettings: (settings: Partial<OrderBookDisplaySettings>) => void
  clearOrderBook: () => void
  getPerformanceMetrics: () => OrderBookPerformanceMetrics | null
}

const defaultAggregationSettings: OrderBookAggregationSettings = {
  enabled: false,
  increment: 0.01,
  maxLevels: 20
}

const defaultDisplaySettings: OrderBookDisplaySettings = {
  showSpread: true,
  showOrderCount: false,
  colorScheme: 'default',
  decimals: 2
}

export const useOrderBookData = (
  options: UseOrderBookDataOptions = {}
): UseOrderBookDataReturn => {
  const {
    maxUpdatesPerSecond = 10,
    autoSubscribe = true,
    enablePerformanceTracking = true
  } = options

  // State
  const [orderBookData, setOrderBookData] = useState<ProcessedOrderBookData | null>(null)
  const [subscriptions, setSubscriptions] = useState<OrderBookSubscription[]>([])
  const [aggregationSettings, setAggregationSettings] = useState<OrderBookAggregationSettings>(defaultAggregationSettings)
  const [displaySettings, setDisplaySettings] = useState<OrderBookDisplaySettings>(defaultDisplaySettings)
  const [isLoading, setIsLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [connectionStatus, setConnectionStatus] = useState<'connected' | 'disconnected' | 'error'>('disconnected')
  const [performanceMetrics, setPerformanceMetrics] = useState<OrderBookPerformanceMetrics | null>(null)

  // Refs
  const messageHandlerRef = useRef<((message: WebSocketMessage) => void) | null>(null)
  const statusHandlerRef = useRef<((status: 'connecting' | 'connected' | 'disconnected' | 'error') => void) | null>(null)
  const throttledUpdateRef = useRef<((data: OrderBookData) => void) | null>(null)

  // Initialize throttled update function
  useEffect(() => {
    throttledUpdateRef.current = orderBookService.throttleUpdates(
      (data: OrderBookData) => {
        try {
          if (!orderBookService.validateOrderBookData(data)) {
            console.warn('Invalid order book data received:', data)
            setError('Invalid order book data format')
            return
          }

          const processedData = orderBookService.processOrderBookData(data, aggregationSettings)
          setOrderBookData(processedData)
          setError(null)

          // Update performance metrics if tracking is enabled
          if (enablePerformanceTracking) {
            const metrics = orderBookService.getPerformanceMetrics()
            setPerformanceMetrics(metrics)
          }
        } catch (err) {
          console.error('Error processing order book data:', err)
          setError(err instanceof Error ? err.message : 'Failed to process order book data')
        }
      },
      maxUpdatesPerSecond
    )
  }, [aggregationSettings, enablePerformanceTracking, maxUpdatesPerSecond])

  // Message handler for WebSocket messages
  const handleWebSocketMessage = useCallback((message: WebSocketMessage) => {
    if (message.type === 'order_book_update') {
      const orderBookMessage = message as OrderBookMessage
      
      // Check if this update is for a subscribed symbol
      const isSubscribed = subscriptions.some(sub => 
        sub.symbol === orderBookMessage.symbol && 
        sub.venue === orderBookMessage.venue && 
        sub.active
      )

      if (isSubscribed && throttledUpdateRef.current) {
        const orderBookData: OrderBookData = {
          symbol: orderBookMessage.symbol,
          venue: orderBookMessage.venue,
          bids: orderBookMessage.bids,
          asks: orderBookMessage.asks,
          timestamp: orderBookMessage.timestamp
        }

        throttledUpdateRef.current(orderBookData)

        // Update subscription last update time
        setSubscriptions(prev => prev.map(sub => 
          sub.symbol === orderBookMessage.symbol && sub.venue === orderBookMessage.venue
            ? { ...sub, lastUpdate: orderBookMessage.timestamp }
            : sub
        ))
      }
    }
  }, [subscriptions])

  // Status handler for connection status
  const handleConnectionStatus = useCallback((status: 'connecting' | 'connected' | 'disconnected' | 'error') => {
    const mappedStatus = status === 'connecting' ? 'disconnected' : status
    setConnectionStatus(mappedStatus)

    if (status === 'disconnected' || status === 'error') {
      setOrderBookData(null)
      setError(status === 'error' ? 'WebSocket connection error' : null)
    }
  }, [])

  // Initialize WebSocket handlers
  useEffect(() => {
    messageHandlerRef.current = handleWebSocketMessage
    statusHandlerRef.current = handleConnectionStatus

    webSocketService.addMessageHandler(handleWebSocketMessage)
    webSocketService.addStatusHandler(handleConnectionStatus)

    // Set initial connection status
    setConnectionStatus(webSocketService.getConnectionState())

    return () => {
      if (messageHandlerRef.current) {
        webSocketService.removeMessageHandler(messageHandlerRef.current)
      }
      if (statusHandlerRef.current) {
        webSocketService.removeStatusHandler(statusHandlerRef.current)
      }
    }
  }, [handleWebSocketMessage, handleConnectionStatus])

  // Connect to WebSocket if auto-subscribe is enabled
  useEffect(() => {
    if (autoSubscribe) {
      webSocketService.connect()
    }
  }, [autoSubscribe])

  // Subscribe to order book for an instrument
  const subscribeToOrderBook = useCallback((instrument: Instrument) => {
    setIsLoading(true)
    setError(null)

    try {
      // Check if already subscribed
      const existingSubscription = subscriptions.find(sub => 
        sub.symbol === instrument.symbol && sub.venue === instrument.venue
      )

      if (existingSubscription && existingSubscription.active) {
        console.log(`Already subscribed to order book for ${instrument.symbol}`)
        setIsLoading(false)
        return
      }

      // Add or update subscription
      const newSubscription: OrderBookSubscription = {
        symbol: instrument.symbol,
        venue: instrument.venue,
        active: true,
        lastUpdate: Date.now()
      }

      setSubscriptions(prev => {
        const filtered = prev.filter(sub => 
          !(sub.symbol === instrument.symbol && sub.venue === instrument.venue)
        )
        return [...filtered, newSubscription]
      })

      // Send subscription message via WebSocket
      const subscriptionMessage = {
        type: 'subscribe_order_book',
        symbol: instrument.symbol,
        venue: instrument.venue,
        asset_class: instrument.assetClass,
        currency: instrument.currency
      }

      webSocketService.send(subscriptionMessage)
      
      console.log(`Subscribed to order book for ${instrument.symbol} on ${instrument.venue}`)
    } catch (err) {
      console.error('Failed to subscribe to order book:', err)
      setError(err instanceof Error ? err.message : 'Failed to subscribe to order book')
    } finally {
      setIsLoading(false)
    }
  }, [subscriptions])

  // Unsubscribe from order book for an instrument
  const unsubscribeFromOrderBook = useCallback((instrument: Instrument) => {
    try {
      // Update subscription status
      setSubscriptions(prev => prev.map(sub => 
        sub.symbol === instrument.symbol && sub.venue === instrument.venue
          ? { ...sub, active: false }
          : sub
      ))

      // Send unsubscription message via WebSocket
      const unsubscriptionMessage = {
        type: 'unsubscribe_order_book',
        symbol: instrument.symbol,
        venue: instrument.venue
      }

      webSocketService.send(unsubscriptionMessage)

      // Clear order book data if this was the current subscription
      if (orderBookData?.symbol === instrument.symbol && orderBookData?.venue === instrument.venue) {
        setOrderBookData(null)
      }

      console.log(`Unsubscribed from order book for ${instrument.symbol} on ${instrument.venue}`)
    } catch (err) {
      console.error('Failed to unsubscribe from order book:', err)
      setError(err instanceof Error ? err.message : 'Failed to unsubscribe from order book')
    }
  }, [orderBookData])

  // Update aggregation settings
  const updateAggregationSettings = useCallback((settings: Partial<OrderBookAggregationSettings>) => {
    setAggregationSettings(prev => ({ ...prev, ...settings }))

    // Reprocess current data with new settings
    if (orderBookData) {
      const rawData: OrderBookData = {
        symbol: orderBookData.symbol,
        venue: orderBookData.venue,
        bids: orderBookData.bids.map(level => ({
          price: level.price,
          quantity: level.quantity,
          orderCount: level.orderCount
        })),
        asks: orderBookData.asks.map(level => ({
          price: level.price,
          quantity: level.quantity,
          orderCount: level.orderCount
        })),
        timestamp: orderBookData.timestamp
      }

      const newAggregationSettings = { ...aggregationSettings, ...settings }
      const reprocessedData = orderBookService.processOrderBookData(rawData, newAggregationSettings)
      setOrderBookData(reprocessedData)
    }
  }, [aggregationSettings, orderBookData])

  // Update display settings
  const updateDisplaySettings = useCallback((settings: Partial<OrderBookDisplaySettings>) => {
    setDisplaySettings(prev => ({ ...prev, ...settings }))
  }, [])

  // Clear order book data
  const clearOrderBook = useCallback(() => {
    setOrderBookData(null)
    setError(null)
  }, [])

  // Get performance metrics
  const getPerformanceMetrics = useCallback((): OrderBookPerformanceMetrics | null => {
    return enablePerformanceTracking ? orderBookService.getPerformanceMetrics() : null
  }, [enablePerformanceTracking])

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      // Unsubscribe from all active subscriptions
      subscriptions.forEach(sub => {
        if (sub.active) {
          const unsubscriptionMessage = {
            type: 'unsubscribe_order_book',
            symbol: sub.symbol,
            venue: sub.venue
          }
          webSocketService.send(unsubscriptionMessage)
        }
      })
    }
  }, []) // Empty dependency array to run only on unmount

  return {
    orderBookData,
    subscriptions,
    aggregationSettings,
    displaySettings,
    isLoading,
    error,
    connectionStatus,
    performanceMetrics,
    subscribeToOrderBook,
    unsubscribeFromOrderBook,
    updateAggregationSettings,
    updateDisplaySettings,
    clearOrderBook,
    getPerformanceMetrics
  }
}