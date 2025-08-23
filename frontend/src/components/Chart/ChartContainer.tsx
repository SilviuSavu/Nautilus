import React, { useEffect, useRef, useCallback } from 'react'
import { createChart, IChartApi, ISeriesApi, CandlestickData, HistogramData, Time } from 'lightweight-charts'
import { OHLCVData, ChartData, ChartSettings } from './types/chartTypes'

interface ChartContainerProps {
  data: ChartData
  settings: ChartSettings
  onPriceChange?: (price: number) => void
  width?: number
  height?: number
}

export const ChartContainer: React.FC<ChartContainerProps> = ({
  data,
  settings,
  onPriceChange,
  width = 800,
  height = 400
}) => {
  const chartContainerRef = useRef<HTMLDivElement>(null)
  const chartRef = useRef<IChartApi | null>(null)
  const candlestickSeriesRef = useRef<ISeriesApi<'Candlestick'> | null>(null)
  const volumeSeriesRef = useRef<ISeriesApi<'Histogram'> | null>(null)

  // Initialize chart only once - avoid dependency on frequently changing props
  const initChart = useCallback(() => {
    if (!chartContainerRef.current) return

    // Clean up existing chart
    if (chartRef.current) {
      chartRef.current.remove()
    }

    // Get actual container dimensions
    const containerWidth = chartContainerRef.current.clientWidth || width
    const containerHeight = chartContainerRef.current.clientHeight || height
    
    console.log('📏 Chart dimensions:', { containerWidth, containerHeight, propWidth: width, propHeight: height })
    
    // Create new chart with basic configuration (settings will be applied separately)
    const chart = createChart(chartContainerRef.current, {
      width: containerWidth,
      height: containerHeight,
      layout: {
        background: { color: '#ffffff' },
        textColor: '#333',
      },
      grid: {
        vertLines: { color: '#e1e1e1' },
        horzLines: { color: '#e1e1e1' },
      },
      crosshair: {
        mode: 1, // Normal crosshair mode
        vertLine: {
          color: '#C3BCDB44',
          labelBackgroundColor: '#9B7DFF',
          width: 1,
          style: 1, // Solid line
        },
        horzLine: {
          color: '#C3BCDB44',
          labelBackgroundColor: '#9B7DFF',
          width: 1,
          style: 1, // Solid line
        },
      },
      timeScale: {
        timeVisible: true,
        secondsVisible: false,
        borderColor: '#d1d1d1',
      },
      rightPriceScale: {
        borderColor: '#d1d1d1',
        scaleMargins: {
          top: 0.1,
          bottom: 0.1,
        },
      },
      handleScroll: {
        mouseWheel: true,
        pressedMouseMove: true,
        horzTouchDrag: true,
        vertTouchDrag: true,
      },
      handleScale: {
        axisPressedMouseMove: true,
        mouseWheel: true,
        pinch: true,
      },
    })

    // Add candlestick series
    const candlestickSeries = chart.addCandlestickSeries({
      upColor: '#26a69a',
      downColor: '#ef5350',
      borderVisible: false,
      wickUpColor: '#26a69a',
      wickDownColor: '#ef5350',
      priceFormat: {
        type: 'price',
        precision: 4,
        minMove: 0.0001,
      },
    })

    // Store refs (volume series will be added dynamically based on settings)
    chartRef.current = chart
    candlestickSeriesRef.current = candlestickSeries
    volumeSeriesRef.current = null

    return chart
  }, [width, height]) // Only depend on stable props

  // Convert data format for TradingView with robust time parsing
  const convertCandleData = useCallback((candles: OHLCVData[]): CandlestickData[] => {
    console.log('🔄 Converting candle data, input count:', candles.length)
    
    const converted = candles.map((candle, index) => {
      try {
        let timestamp: number
        
        // Handle different time formats
        const timeStr = candle.time.toString().trim()
        
        if (timeStr.match(/^\d{4}-\d{2}-\d{2}$/)) {
          // Format: "2024-08-22" (already in ISO date format)
          timestamp = new Date(timeStr + 'T00:00:00').getTime() / 1000
        } else if (timeStr.match(/^\d{8}$/)) {
          // Format: "20240822" (IB Gateway daily format)
          const year = timeStr.substring(0, 4)
          const month = timeStr.substring(4, 6)
          const day = timeStr.substring(6, 8)
          timestamp = new Date(`${year}-${month}-${day}T00:00:00`).getTime() / 1000
        } else if (timeStr.match(/^\d{8}\s+\d{2}:\d{2}:\d{2}$/)) {
          // Format: "20240822  15:30:00" (IB Gateway intraday format)
          const parts = timeStr.replace(/\s+/g, ' ').split(' ')
          const datePart = parts[0]
          const timePart = parts[1]
          const year = datePart.substring(0, 4)
          const month = datePart.substring(4, 6)
          const day = datePart.substring(6, 8)
          timestamp = new Date(`${year}-${month}-${day}T${timePart}`).getTime() / 1000
        } else {
          // Try parsing as-is (fallback)
          const date = new Date(timeStr)
          if (isNaN(date.getTime())) {
            throw new Error(`Unrecognized time format: ${timeStr}`)
          }
          timestamp = date.getTime() / 1000
        }
        
        // Validate timestamp
        if (isNaN(timestamp) || timestamp <= 0) {
          throw new Error(`Invalid timestamp generated: ${timestamp}`)
        }
        
        // Validate OHLC values
        const { open, high, low, close } = candle
        if (isNaN(open) || isNaN(high) || isNaN(low) || isNaN(close)) {
          throw new Error(`Invalid OHLC values: O:${open} H:${high} L:${low} C:${close}`)
        }
        
        if (high < Math.max(open, close) || low > Math.min(open, close)) {
          console.warn(`Suspicious OHLC values for ${timeStr}:`, { open, high, low, close })
        }
        
        return {
          time: timestamp as Time,
          open: Number(open),
          high: Number(high),
          low: Number(low),
          close: Number(close),
        }
      } catch (error) {
        console.error(`Failed to convert candle ${index}:`, candle, error)
        return null
      }
    }).filter(Boolean) as CandlestickData[]
    
    console.log('✅ Converted candles:', {
      original: candles.length,
      converted: converted.length,
      sample: converted[0]
    })
    
    // CRITICAL: Sort by time ascending (lightweight-charts requirement)
    converted.sort((a, b) => (a.time as number) - (b.time as number))
    
    // CRITICAL: Remove duplicates - lightweight-charts requires unique timestamps
    const uniqueConverted = []
    const seenTimes = new Set()
    
    for (const item of converted) {
      if (!seenTimes.has(item.time)) {
        seenTimes.add(item.time)
        uniqueConverted.push(item)
      }
    }
    
    // Log only if there were duplicates removed
    const duplicatesRemoved = converted.length - uniqueConverted.length
    if (duplicatesRemoved > 0) {
      console.log('📊 Removed', duplicatesRemoved, 'duplicate timestamps from', converted.length, 'candles')
    }
    
    return uniqueConverted
  }, [])

  const convertVolumeData = useCallback((candles: OHLCVData[]): HistogramData[] => {
    const converted = candles.map(candle => {
      // Parse IB Gateway time format: "20250519  15:30:00" or "20250519" for daily
      const timeStr = candle.time.replace(/\s+/g, ' ').trim()
      const [datePart, timePart] = timeStr.split(' ')
      const year = datePart.substring(0, 4)
      const month = datePart.substring(4, 6)
      const day = datePart.substring(6, 8)
      // For daily data, timePart might be undefined
      const formattedTime = timePart ? `${year}-${month}-${day}T${timePart}` : `${year}-${month}-${day}T00:00:00`
      const timestamp = new Date(formattedTime).getTime() / 1000
      
      return {
        time: timestamp as Time,
        value: candle.volume,
        color: candle.close > candle.open ? '#26a69a80' : '#ef535080',
      }
    })
    
    // CRITICAL: Sort by time ascending (lightweight-charts requirement)
    converted.sort((a, b) => (a.time as number) - (b.time as number))
    
    // CRITICAL: Remove duplicates - lightweight-charts requires unique timestamps
    const uniqueConverted = []
    const seenTimes = new Set()
    
    for (const item of converted) {
      if (!seenTimes.has(item.time)) {
        seenTimes.add(item.time)
        uniqueConverted.push(item)
      }
    }
    
    return uniqueConverted
  }, [])

  // Initialize chart on mount with proper dimension checking
  useEffect(() => {
    const attemptInitialization = () => {
      if (!chartContainerRef.current) {
        console.log('⚠️ Chart container ref not available, retrying...')
        return false
      }

      const container = chartContainerRef.current
      const containerWidth = container.clientWidth || width
      const containerHeight = container.clientHeight || height

      console.log('📏 Container check:', {
        clientWidth: container.clientWidth,
        clientHeight: container.clientHeight,
        offsetWidth: container.offsetWidth,
        offsetHeight: container.offsetHeight,
        computedWidth: containerWidth,
        computedHeight: containerHeight
      })

      if (containerWidth === 0 || containerHeight === 0) {
        console.log('⚠️ Container has zero dimensions, retrying...')
        return false
      }

      console.log('🚀 Initializing chart with valid dimensions')
      initChart()
      return true
    }

    // Try immediate initialization
    if (!attemptInitialization()) {
      // If immediate fails, use retries with increasing delays
      let attempts = 0
      const maxAttempts = 10
      
      const retryInit = () => {
        attempts++
        console.log(`🔄 Retry attempt ${attempts}/${maxAttempts}`)
        
        if (attemptInitialization() || attempts >= maxAttempts) {
          if (attempts >= maxAttempts) {
            console.error('❌ Failed to initialize chart after', maxAttempts, 'attempts')
          }
          return
        }
        
        // Exponential backoff: 50ms, 100ms, 200ms, etc.
        const delay = 50 * Math.pow(2, attempts - 1)
        setTimeout(retryInit, delay)
      }
      
      // Start retrying after a small delay
      setTimeout(retryInit, 50)
    }
    
    return () => {
      if (chartRef.current) {
        console.log('🧹 Cleaning up chart')
        chartRef.current.remove()
        chartRef.current = null
      }
    }
  }, [initChart, width, height])

  // Update data when it changes - but ensure chart is ready
  useEffect(() => {
    console.log('🎯 ChartContainer data update:', { 
      hasCandlestickSeries: !!candlestickSeriesRef.current,
      candlesLength: data.candles.length,
      firstCandle: data.candles[0]
    })
    
    if (!data.candles.length) {
      console.log('⚠️ No candles data')
      return
    }

    // If chart isn't ready yet, wait for it
    if (!candlestickSeriesRef.current) {
      console.log('📊 Chart not ready, waiting...')
      const checkChart = () => {
        if (candlestickSeriesRef.current && data.candles.length) {
          console.log('📊 Chart now ready, setting data')
          const candleData = convertCandleData(data.candles)
          candlestickSeriesRef.current.setData(candleData)

          if (volumeSeriesRef.current && settings.showVolume) {
            const volumeData = convertVolumeData(data.candles)
            volumeSeriesRef.current.setData(volumeData)
          }

          if (chartRef.current) {
            chartRef.current.timeScale().fitContent()
          }
        } else {
          // Keep checking until chart is ready
          setTimeout(checkChart, 50)
        }
      }
      checkChart()
      return
    }

    const candleData = convertCandleData(data.candles)
    console.log('📊 Converted candle data:', { 
      originalCount: data.candles.length,
      convertedCount: candleData.length,
      firstConverted: candleData[0]
    })
    candlestickSeriesRef.current.setData(candleData)

    if (volumeSeriesRef.current && settings.showVolume) {
      const volumeData = convertVolumeData(data.candles)
      volumeSeriesRef.current.setData(volumeData)
    }

    // Fit content to visible area
    if (chartRef.current) {
      chartRef.current.timeScale().fitContent()
    }
  }, [data, convertCandleData, convertVolumeData, settings.showVolume])

  // Handle resize
  useEffect(() => {
    if (!chartRef.current) return

    const resizeObserver = new ResizeObserver(entries => {
      const { width: newWidth, height: newHeight } = entries[0].contentRect
      chartRef.current?.applyOptions({
        width: newWidth,
        height: newHeight,
      })
    })

    if (chartContainerRef.current) {
      resizeObserver.observe(chartContainerRef.current)
    }

    return () => {
      resizeObserver.disconnect()
    }
  }, [])

  // Handle settings changes separately to avoid reinitializing chart
  useEffect(() => {
    if (!chartRef.current) return

    console.log('⚙️ Updating chart settings:', settings)

    // Update grid settings
    chartRef.current.applyOptions({
      grid: {
        vertLines: { color: settings.grid ? '#e1e1e1' : 'transparent' },
        horzLines: { color: settings.grid ? '#e1e1e1' : 'transparent' },
      },
      crosshair: {
        mode: settings.crosshair ? 1 : 0,
      },
    })

    // Handle volume series
    if (settings.showVolume && !volumeSeriesRef.current) {
      // Add volume series
      const volumeSeries = chartRef.current.addHistogramSeries({
        priceFormat: { type: 'volume' },
        priceScaleId: 'volume',
      })
      
      chartRef.current.priceScale('volume').applyOptions({
        scaleMargins: { top: 0.8, bottom: 0 },
      })
      
      chartRef.current.applyOptions({
        rightPriceScale: {
          scaleMargins: { top: 0.1, bottom: 0.4 },
        },
      })
      
      volumeSeriesRef.current = volumeSeries
      
      // Set volume data if available
      if (data.candles.length > 0) {
        const volumeData = convertVolumeData(data.candles)
        volumeSeries.setData(volumeData)
      }
    } else if (!settings.showVolume && volumeSeriesRef.current) {
      // Remove volume series
      chartRef.current.removeSeries(volumeSeriesRef.current)
      volumeSeriesRef.current = null
      
      chartRef.current.applyOptions({
        rightPriceScale: {
          scaleMargins: { top: 0.1, bottom: 0.1 },
        },
      })
    }
  }, [settings, data.candles, convertVolumeData])

  // Handle price change subscription separately
  useEffect(() => {
    if (!chartRef.current || !candlestickSeriesRef.current || !onPriceChange) return

    console.log('📈 Setting up price change subscription')

    const handleCrosshairMove = (param: any) => {
      if (param.point && param.seriesData.has(candlestickSeriesRef.current!)) {
        const data = param.seriesData.get(candlestickSeriesRef.current!) as CandlestickData
        if (data && typeof data.close === 'number') {
          onPriceChange(data.close)
        }
      }
    }

    chartRef.current.subscribeCrosshairMove(handleCrosshairMove)

    return () => {
      if (chartRef.current) {
        chartRef.current.unsubscribeCrosshairMove(handleCrosshairMove)
      }
    }
  }, [onPriceChange])

  return (
    <div 
      ref={chartContainerRef}
      style={{ 
        width: '100%', 
        height: height,
        minHeight: height,
        position: 'relative',
        border: '1px solid #e1e1e1',  // Visual border to confirm container
        backgroundColor: '#ffffff',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center'
      }}
    >
      {/* Fallback content while chart initializes */}
      {!chartRef.current && (
        <div style={{ 
          color: '#888', 
          fontSize: '14px',
          textAlign: 'center',
          position: 'absolute',
          zIndex: 1
        }}>
          <div>📊 Initializing TradingView Chart...</div>
          <div style={{ fontSize: '12px', marginTop: '8px' }}>
            Container: {width} × {height}px
          </div>
        </div>
      )}
    </div>
  )
}