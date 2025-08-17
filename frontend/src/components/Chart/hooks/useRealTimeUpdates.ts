import { useEffect, useCallback, useRef } from 'react'
import { useChartStore } from './useChartStore'
import { useMessageBus } from '../../../hooks/useMessageBus'
import { MessageBusMessage } from '../../../types/messagebus'
import { OHLCVData, PriceUpdate } from '../types/chartTypes'

export const useRealTimeUpdates = () => {
  const {
    currentInstrument,
    realTimeUpdates,
    chartData,
    setChartData,
    setError
  } = useChartStore()

  const {
    connectionStatus,
    getLatestMessageByTopic
  } = useMessageBus()

  const lastUpdateRef = useRef<number>(0)
  const priceBufferRef = useRef<PriceUpdate[]>([])

  // Process market data messages and convert to chart updates
  const processMarketDataMessage = useCallback((message: MessageBusMessage): PriceUpdate | null => {
    try {
      const { payload, topic } = message

      // Handle different types of market data messages
      if (topic.includes('QuoteTick') && payload) {
        return {
          symbol: payload.instrument_id || payload.symbol,
          price: payload.bid || payload.price,
          volume: 0, // Quote ticks don't have volume
          timestamp: new Date(payload.ts_event / 1000000).toISOString() // Convert nanoseconds to milliseconds
        }
      }

      if (topic.includes('TradeTick') && payload) {
        return {
          symbol: payload.instrument_id || payload.symbol,
          price: payload.price,
          volume: payload.size || 0,
          timestamp: new Date(payload.ts_event / 1000000).toISOString()
        }
      }

      if (topic.includes('Bar') && payload) {
        // This is already aggregated bar data, which is perfect for our charts
        return {
          symbol: payload.instrument_id || payload.symbol,
          price: payload.close,
          volume: payload.volume,
          timestamp: new Date(payload.ts_event / 1000000).toISOString()
        }
      }

      return null
    } catch (error) {
      console.error('Failed to process market data message:', error)
      return null
    }
  }, [])

  // Update chart data with new price information
  const updateChartWithRealTimeData = useCallback((updates: PriceUpdate[]) => {
    if (!currentInstrument || updates.length === 0) return

    const currentSymbol = currentInstrument.symbol
    const relevantUpdates = updates.filter(update => 
      update.symbol === currentSymbol || 
      update.symbol.includes(currentSymbol)
    )

    if (relevantUpdates.length === 0) return

    const latestUpdate = relevantUpdates[relevantUpdates.length - 1]

    // Get current chart data
    const currentCandles = [...chartData.candles]
    if (currentCandles.length === 0) return

    // Get the latest candle
    const lastCandle = currentCandles[currentCandles.length - 1]

    // Determine if we should update the last candle or create a new one
    // For simplicity, we'll update the last candle with new price data
    const updatedCandle: OHLCVData = {
      ...lastCandle,
      close: latestUpdate.price,
      high: Math.max(lastCandle.high, latestUpdate.price),
      low: Math.min(lastCandle.low, latestUpdate.price),
      volume: lastCandle.volume + latestUpdate.volume,
      time: lastCandle.time // Keep the original time
    }

    // Update the last candle
    currentCandles[currentCandles.length - 1] = updatedCandle

    // Update volume data
    const updatedVolume = chartData.volume.map((vol, index) => {
      if (index === chartData.volume.length - 1) {
        return {
          ...vol,
          value: updatedCandle.volume,
          color: updatedCandle.close > updatedCandle.open ? '#26a69a80' : '#ef535080'
        }
      }
      return vol
    })

    setChartData({
      candles: currentCandles,
      volume: updatedVolume
    })

  }, [currentInstrument, chartData, setChartData])

  // Process buffered price updates
  const processBufferedUpdates = useCallback(() => {
    if (priceBufferRef.current.length === 0) return

    const updates = [...priceBufferRef.current]
    priceBufferRef.current = []

    updateChartWithRealTimeData(updates)
  }, [updateChartWithRealTimeData])

  // Buffer price updates to avoid overwhelming the chart
  const bufferPriceUpdate = useCallback((update: PriceUpdate) => {
    priceBufferRef.current.push(update)

    const now = Date.now()
    if (now - lastUpdateRef.current > 100) { // Update chart max once per 100ms
      lastUpdateRef.current = now
      setTimeout(processBufferedUpdates, 0) // Process on next tick
    }
  }, [processBufferedUpdates])

  // Monitor MessageBus for relevant market data
  useEffect(() => {
    if (!realTimeUpdates || connectionStatus !== 'connected' || !currentInstrument) {
      return
    }

    const checkForNewMarketData = () => {
      try {
        // Look for messages related to our current instrument
        const currentSymbol = currentInstrument.symbol
        
        // Check for different types of market data topics
        const topicsToCheck = [
          `data.quotes.${currentSymbol}`,
          `data.trades.${currentSymbol}`,
          `data.bars.${currentSymbol}`,
          'data.quotes',
          'data.trades', 
          'data.bars'
        ]

        for (const topic of topicsToCheck) {
          const latestMessage = getLatestMessageByTopic(topic)
          if (latestMessage) {
            const priceUpdate = processMarketDataMessage(latestMessage)
            if (priceUpdate) {
              bufferPriceUpdate(priceUpdate)
            }
          }
        }
      } catch (error) {
        console.error('Error processing real-time market data:', error)
        setError({
          type: 'connection',
          message: 'Failed to process real-time market data',
          timestamp: new Date().toISOString()
        })
      }
    }

    // Check for new data every 50ms when real-time updates are enabled
    const interval = setInterval(checkForNewMarketData, 50)

    return () => {
      clearInterval(interval)
    }
  }, [
    realTimeUpdates,
    connectionStatus,
    currentInstrument,
    getLatestMessageByTopic,
    processMarketDataMessage,
    bufferPriceUpdate,
    setError
  ])

  // Subscribe to specific market data topics when instrument changes
  const subscribeToMarketData = useCallback(async (instrument: string) => {
    // This would typically send a subscription message to the backend
    // For now, we'll just log the subscription request
    console.log(`Subscribing to market data for instrument: ${instrument}`)
    
    // In a real implementation, you might send something like:
    // webSocketService.send({
    //   type: 'subscribe',
    //   topic: `data.quotes.${instrument}`,
    //   timestamp: Date.now()
    // })
  }, [])

  // Subscribe to market data when instrument changes
  useEffect(() => {
    if (currentInstrument && realTimeUpdates && connectionStatus === 'connected') {
      subscribeToMarketData(currentInstrument.symbol)
    }
  }, [currentInstrument, realTimeUpdates, connectionStatus, subscribeToMarketData])

  return {
    connectionStatus,
    realTimeUpdates,
    subscribeToMarketData
  }
}