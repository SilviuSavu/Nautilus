import React, { useEffect, useRef, useState, useMemo } from 'react'
import { createChart, IChartApi, ISeriesApi, ColorType, CrosshairMode } from 'lightweight-charts'
import { Spin, Alert, Card, Statistic, Row, Col, Typography } from 'antd'
import { ArrowUpOutlined, ArrowDownOutlined } from '@ant-design/icons'
import { useChartStore } from './Chart/hooks/useChartStore'

const { Text } = Typography

export const SimpleChart: React.FC = () => {
  const containerRef = useRef<HTMLDivElement>(null)
  const chartRef = useRef<IChartApi | null>(null)
  const seriesRef = useRef<ISeriesApi<'Candlestick'> | null>(null)
  const volumeSeriesRef = useRef<ISeriesApi<'Histogram'> | null>(null)
  
  const { currentInstrument, timeframe } = useChartStore()
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [marketData, setMarketData] = useState<{
    currentPrice: number
    priceChange: number
    percentChange: number
    volume: number
    high: number
    low: number
  } | null>(null)

  // Initialize chart once
  useEffect(() => {
    if (!containerRef.current) return

    // Get container dimensions
    const containerWidth = containerRef.current.clientWidth || 800
    const containerHeight = 500

    const chart = createChart(containerRef.current, {
      width: containerWidth,
      height: containerHeight,
      layout: {
        background: { type: ColorType.Solid, color: '#ffffff' },
        textColor: '#333333',
        fontSize: 12,
      },
      grid: {
        vertLines: { color: '#f0f0f0', style: 2, visible: true },
        horzLines: { color: '#f0f0f0', style: 2, visible: true },
      },
      crosshair: {
        mode: CrosshairMode.Normal,
        vertLine: {
          color: '#9B7DFF',
          width: 1,
          style: 0,
          labelBackgroundColor: '#9B7DFF',
        },
        horzLine: {
          color: '#9B7DFF',
          width: 1,
          style: 0,
          labelBackgroundColor: '#9B7DFF',
        },
      },
      rightPriceScale: {
        borderColor: '#cccccc',
        scaleMargins: {
          top: 0.05,
          bottom: 0.4,
        },
      },
      timeScale: {
        borderColor: '#cccccc',
        timeVisible: true,
        secondsVisible: false,
        tickMarkFormatter: (time: any, tickMarkType: any, locale: string) => {
          const date = new Date(time * 1000)
          const now = new Date()
          const diffDays = Math.floor((now.getTime() - date.getTime()) / (1000 * 60 * 60 * 24))
          
          if (diffDays === 0) {
            return date.toLocaleTimeString(locale, { hour: '2-digit', minute: '2-digit' })
          } else if (diffDays < 7) {
            return date.toLocaleDateString(locale, { weekday: 'short', hour: '2-digit', minute: '2-digit' })
          } else {
            return date.toLocaleDateString(locale, { month: 'short', day: 'numeric' })
          }
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

    const series = chart.addCandlestickSeries({
      upColor: '#00C896',
      downColor: '#FF4976',
      borderVisible: false,
      wickUpColor: '#00C896',
      wickDownColor: '#FF4976',
      priceFormat: {
        type: 'price',
        precision: 4,
        minMove: 0.0001,
      },
    })

    // Add volume series
    const volumeSeries = chart.addHistogramSeries({
      priceFormat: {
        type: 'volume',
      },
      priceScaleId: 'volume',
      color: '#E0E0E0',
    })

    // Configure volume price scale
    chart.priceScale('volume').applyOptions({
      scaleMargins: {
        top: 0.8,
        bottom: 0,
      },
      borderColor: '#cccccc',
    })

    chartRef.current = chart
    seriesRef.current = series
    volumeSeriesRef.current = volumeSeries

    // Add resize observer to handle container size changes
    const resizeObserver = new ResizeObserver(entries => {
      const { width, height } = entries[0].contentRect
      chart.applyOptions({
        width: width,
        height: height,
      })
    })

    resizeObserver.observe(containerRef.current)

    return () => {
      resizeObserver.disconnect()
      chart.remove()
    }
  }, [])

  // Load data when selection changes
  useEffect(() => {
    if (!currentInstrument || !seriesRef.current) return

    const loadData = async () => {
      setLoading(true)
      setError(null)

      try {
        // Build comprehensive API parameters like the ChartComponent does
        const params = new URLSearchParams({
          symbol: currentInstrument.symbol,
          timeframe,
          asset_class: currentInstrument.assetClass,
          exchange: currentInstrument.venue,
          currency: currentInstrument.currency || 'USD'
        })

        console.log('📡 SimpleChart API request:', `http://localhost:8000/api/v1/market-data/historical/bars?${params}`)

        const response = await fetch(
          `http://localhost:8000/api/v1/market-data/historical/bars?${params}`
        )

        if (!response.ok) {
          throw new Error(`HTTP ${response.status}: ${response.statusText}`)
        }

        const data = await response.json()
        console.log('📊 SimpleChart API response:', { 
          symbol: data.symbol, 
          candleCount: data.candles?.length || 0,
          timeframe: data.timeframe 
        })

        if (data.candles && data.candles.length > 0) {
          const chartData = data.candles.map((candle: any) => {
            let time: number
            
            // Enhanced timestamp parsing for different formats
            try {
              if (candle.time.includes('  ')) {
                // Format: "20250519  15:30:00" 
                const dateStr = candle.time.replace(/\s+/g, ' ').trim()
                const [datePart, timePart] = dateStr.split(' ')
                const year = datePart.substring(0, 4)
                const month = datePart.substring(4, 6)
                const day = datePart.substring(6, 8)
                const formattedTime = timePart ? `${year}-${month}-${day}T${timePart}` : `${year}-${month}-${day}T00:00:00`
                time = Math.floor(new Date(formattedTime).getTime() / 1000)
              } else if (/^\d{8}$/.test(candle.time)) {
                // Format: "20250519" (daily/weekly/monthly)
                const year = candle.time.substring(0, 4)
                const month = candle.time.substring(4, 6)
                const day = candle.time.substring(6, 8)
                time = Math.floor(new Date(`${year}-${month}-${day}T00:00:00`).getTime() / 1000)
              } else {
                // Try standard ISO format or other formats
                time = Math.floor(new Date(candle.time).getTime() / 1000)
              }
              
              // Validate timestamp
              if (isNaN(time) || time <= 0) {
                console.warn('Invalid timestamp for candle:', candle.time)
                return null
              }
            } catch (error) {
              console.error('Error parsing timestamp:', candle.time, error)
              return null
            }

            return {
              time,
              open: candle.open,
              high: candle.high,
              low: candle.low,
              close: candle.close,
            }
          }).filter(Boolean)

          // Sort and remove duplicates
          chartData.sort((a, b) => a.time - b.time)
          const uniqueData = []
          const seenTimes = new Set()
          
          for (const item of chartData) {
            if (!seenTimes.has(item.time)) {
              seenTimes.add(item.time)
              uniqueData.push(item)
            }
          }

          // Create volume data with improved mapping
          const volumeData = uniqueData.map(item => {
            const originalCandle = data.candles.find((c: any) => {
              let time: number
              try {
                if (c.time.includes('  ')) {
                  const dateStr = c.time.replace(/\s+/g, ' ').trim()
                  const [datePart, timePart] = dateStr.split(' ')
                  const year = datePart.substring(0, 4)
                  const month = datePart.substring(4, 6)
                  const day = datePart.substring(6, 8)
                  const formattedTime = timePart ? `${year}-${month}-${day}T${timePart}` : `${year}-${month}-${day}T00:00:00`
                  time = Math.floor(new Date(formattedTime).getTime() / 1000)
                } else if (/^\d{8}$/.test(c.time)) {
                  const year = c.time.substring(0, 4)
                  const month = c.time.substring(4, 6)
                  const day = c.time.substring(6, 8)
                  time = Math.floor(new Date(`${year}-${month}-${day}T00:00:00`).getTime() / 1000)
                } else {
                  time = Math.floor(new Date(c.time).getTime() / 1000)
                }
                return time === item.time
              } catch {
                return false
              }
            })
            
            return {
              time: item.time,
              value: originalCandle?.volume || 0,
              color: item.close > item.open ? '#00C89650' : '#FF497650',
            }
          })

          // Calculate market statistics
          if (uniqueData.length > 0) {
            const currentCandle = uniqueData[uniqueData.length - 1]
            const previousCandle = uniqueData.length > 1 ? uniqueData[uniqueData.length - 2] : uniqueData[0]
            const priceChange = currentCandle.close - previousCandle.close
            const percentChange = (priceChange / previousCandle.close) * 100
            const high = Math.max(...uniqueData.map(d => d.high))
            const low = Math.min(...uniqueData.map(d => d.low))
            const totalVolume = volumeData.reduce((sum, v) => sum + v.value, 0)

            setMarketData({
              currentPrice: currentCandle.close,
              priceChange,
              percentChange,
              volume: totalVolume,
              high,
              low,
            })
          }

          seriesRef.current?.setData(uniqueData)
          volumeSeriesRef.current?.setData(volumeData)
        } else {
          console.error('📊 No candles data received from API for', currentInstrument.symbol, timeframe)
          throw new Error(`No historical data available for ${currentInstrument.symbol} with ${timeframe} timeframe. Backend issue.`)
        }
      } catch (err) {
        setError(err instanceof Error ? err.message : 'Failed to load data')
      } finally {
        setLoading(false)
      }
    }

    loadData()
  }, [currentInstrument, timeframe])

  // Memoize market data display for performance
  const marketDataDisplay = useMemo(() => {
    if (!marketData || !currentInstrument) return null

    const isPositive = marketData.priceChange >= 0
    const priceColor = isPositive ? '#00C896' : '#FF4976'
    const icon = isPositive ? <ArrowUpOutlined /> : <ArrowDownOutlined />

    return (
      <Row gutter={16} style={{ marginBottom: 16 }}>
        <Col span={6}>
          <Card size="small" style={{ textAlign: 'center' }}>
            <Statistic
              title={currentInstrument.symbol}
              value={marketData.currentPrice}
              precision={4}
              valueStyle={{ color: priceColor, fontSize: '24px', fontWeight: 'bold' }}
              prefix={icon}
            />
            <Text style={{ color: priceColor, fontSize: '14px', fontWeight: 'bold' }}>
              {isPositive ? '+' : ''}{marketData.priceChange.toFixed(4)} ({isPositive ? '+' : ''}{marketData.percentChange.toFixed(2)}%)
            </Text>
          </Card>
        </Col>
        <Col span={4}>
          <Card size="small" style={{ textAlign: 'center' }}>
            <Statistic
              title="24h High"
              value={marketData.high}
              precision={4}
              valueStyle={{ fontSize: '16px' }}
            />
          </Card>
        </Col>
        <Col span={4}>
          <Card size="small" style={{ textAlign: 'center' }}>
            <Statistic
              title="24h Low"
              value={marketData.low}
              precision={4}
              valueStyle={{ fontSize: '16px' }}
            />
          </Card>
        </Col>
        <Col span={5}>
          <Card size="small" style={{ textAlign: 'center' }}>
            <Statistic
              title="Volume"
              value={marketData.volume}
              formatter={(value) => {
                const num = Number(value)
                if (num >= 1e9) return `${(num / 1e9).toFixed(1)}B`
                if (num >= 1e6) return `${(num / 1e6).toFixed(1)}M`
                if (num >= 1e3) return `${(num / 1e3).toFixed(1)}K`
                return num.toLocaleString()
              }}
              valueStyle={{ fontSize: '16px' }}
            />
          </Card>
        </Col>
        <Col span={5}>
          <Card size="small" style={{ textAlign: 'center' }}>
            <Statistic
              title="Timeframe"
              value={timeframe.toUpperCase()}
              valueStyle={{ fontSize: '16px', color: '#666' }}
            />
          </Card>
        </Col>
      </Row>
    )
  }, [marketData, currentInstrument, timeframe])

  return (
    <div>
      {marketDataDisplay}
      
      {error && (
        <Alert type="error" message={error} style={{ marginBottom: 16 }} />
      )}
      
      <Card 
        bodyStyle={{ padding: 0 }}
        style={{ 
          position: 'relative',
          border: '1px solid #e8e8e8',
          borderRadius: '8px',
          overflow: 'hidden'
        }}
      >
        {loading && (
          <div style={{
            position: 'absolute',
            top: 0,
            left: 0,
            right: 0,
            bottom: 0,
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            backgroundColor: 'rgba(255,255,255,0.9)',
            zIndex: 1000,
            borderRadius: '8px'
          }}>
            <Spin size="large" />
          </div>
        )}
        
        <div ref={containerRef} style={{ width: '100%', height: '500px' }} />
      </Card>
    </div>
  )
}