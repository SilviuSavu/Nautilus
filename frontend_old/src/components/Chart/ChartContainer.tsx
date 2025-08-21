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

  // Initialize chart
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
    
    // Create new chart
    const chart = createChart(chartContainerRef.current, {
      width: containerWidth,
      height: containerHeight,
      layout: {
        background: { color: '#ffffff' },
        textColor: '#333',
      },
      grid: {
        vertLines: { color: settings.grid ? '#e1e1e1' : 'transparent' },
        horzLines: { color: settings.grid ? '#e1e1e1' : 'transparent' },
      },
      crosshair: {
        mode: settings.crosshair ? 1 : 0, // Normal crosshair mode
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
          bottom: settings.showVolume ? 0.4 : 0.1,
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

    // Add volume series if enabled
    let volumeSeries: ISeriesApi<'Histogram'> | null = null
    if (settings.showVolume) {
      volumeSeries = chart.addHistogramSeries({
        priceFormat: {
          type: 'volume',
        },
        priceScaleId: 'volume',
      })
      
      // Configure volume price scale
      chart.priceScale('volume').applyOptions({
        scaleMargins: {
          top: 0.8,
          bottom: 0,
        },
      })
    }

    // Store refs
    chartRef.current = chart
    candlestickSeriesRef.current = candlestickSeries
    volumeSeriesRef.current = volumeSeries

    // Subscribe to price changes
    if (onPriceChange) {
      chart.subscribeCrosshairMove((param) => {
        if (param.point && param.seriesData.has(candlestickSeries)) {
          const data = param.seriesData.get(candlestickSeries) as CandlestickData
          if (data && typeof data.close === 'number') {
            onPriceChange(data.close)
          }
        }
      })
    }

    return chart
  }, [width, height, settings, onPriceChange])

  // Convert data format for TradingView
  const convertCandleData = useCallback((candles: OHLCVData[]): CandlestickData[] => {
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
      
      // Validate timestamp
      if (isNaN(timestamp)) {
        console.error('Invalid timestamp for candle:', candle.time, 'formatted:', formattedTime)
        return null
      }
      
      return {
        time: timestamp as Time,
        open: candle.open,
        high: candle.high,
        low: candle.low,
        close: candle.close,
      }
    }).filter(Boolean) as CandlestickData[]
    
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

  // Initialize chart on mount - delay until container is ready
  useEffect(() => {
    const timeoutId = setTimeout(() => {
      if (chartContainerRef.current) {
        console.log('🚀 Initializing chart after timeout')
        initChart()
      }
    }, 100) // Small delay to ensure container is fully rendered
    
    return () => {
      clearTimeout(timeoutId)
      if (chartRef.current) {
        chartRef.current.remove()
        chartRef.current = null
      }
    }
  }, [initChart])

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

  return (
    <div 
      ref={chartContainerRef}
      style={{ 
        width: '100%', 
        height: height,
        minHeight: height,
        position: 'relative',
        border: '1px solid #e1e1e1',  // Temporary border to see the container
        backgroundColor: '#ffffff'
      }}
    />
  )
}