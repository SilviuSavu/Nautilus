/**
 * Advanced Chart Container
 * Main container for rendering different chart types with lightweight-charts
 */

import React, { useRef, useEffect, useState, useMemo } from 'react'
import { createChart, IChartApi, ISeriesApi, ColorType, LineStyle } from 'lightweight-charts'
import { Alert, Spin } from 'antd'
import { OHLCVData } from '../Chart/types/chartTypes'
import { ChartType } from '../../types/charting'
import { chartDataProcessor, RenkoConfig, PointFigureConfig, VolumeProfileConfig } from '../../services/chartDataProcessors'
import { indicatorEngine, IndicatorResult } from '../../services/indicatorEngine'

interface AdvancedChartContainerProps {
  data: OHLCVData[]
  chartType: ChartType
  width?: number
  height?: number
  indicators?: Array<{ id: string; params: Record<string, any> }>
  onPriceChange?: (price: number) => void
  theme?: 'light' | 'dark'
  autoSize?: boolean
}

export const AdvancedChartContainer: React.FC<AdvancedChartContainerProps> = ({
  data,
  chartType,
  width = 800,
  height = 400,
  indicators = [],
  onPriceChange,
  theme = 'light',
  autoSize = true
}) => {
  const chartContainerRef = useRef<HTMLDivElement>(null)
  const chartRef = useRef<IChartApi | null>(null)
  const seriesRef = useRef<ISeriesApi<any> | null>(null)
  const indicatorSeriesRefs = useRef<Map<string, ISeriesApi<any>>>(new Map())
  
  const [isLoading, setIsLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [containerSize, setContainerSize] = useState({ width, height })

  // Process chart data based on type
  const processedData = useMemo(() => {
    if (!data || data.length === 0) return null

    setIsLoading(true)
    setError(null)

    try {
      switch (chartType) {
        case 'candlestick':
        case 'line':
        case 'area':
          return data.map(item => ({
            time: item.time,
            open: item.open,
            high: item.high,
            low: item.low,
            close: item.close,
            volume: item.volume
          }))

        case 'renko':
          const renkoConfig: RenkoConfig = {
            autoCalculateBrickSize: true,
            source: 'close'
          }
          const renkoData = chartDataProcessor.processRenkoData(data, renkoConfig)
          return renkoData.map(brick => ({
            time: brick.time,
            open: brick.open,
            high: Math.max(brick.open, brick.close),
            low: Math.min(brick.open, brick.close),
            close: brick.close,
            color: brick.trend === 'up' ? '#26a69a' : '#ef5350'
          }))

        case 'point_figure':
          const pfConfig: PointFigureConfig = {
            autoCalculateBoxSize: true,
            reversalAmount: 3,
            source: 'close'
          }
          const pfData = chartDataProcessor.processPointFigureData(data, pfConfig)
          // Convert P&F to line series for now (could be enhanced with custom renderer)
          return pfData.map(column => ({
            time: column.time,
            value: column.boxes[column.boxes.length - 1]?.price || 0
          }))

        case 'heikin_ashi':
          const haData = chartDataProcessor.processHeikinAshiData(data)
          return haData.map(item => ({
            time: item.time,
            open: item.open,
            high: item.high,
            low: item.low,
            close: item.close,
            volume: item.volume
          }))

        case 'volume_profile':
          const vpConfig: VolumeProfileConfig = {
            priceLevels: 50,
            sessionType: 'daily',
            showPOC: true
          }
          const vpData = chartDataProcessor.processVolumeProfileData(data, vpConfig)
          // Convert to histogram-like data
          return vpData.map(level => ({
            time: data[Math.floor(data.length / 2)].time, // Use middle time
            value: level.priceLevel,
            volume: level.volume
          }))

        default:
          return data
      }
    } catch (err) {
      setError(`Error processing ${chartType} data: ${err instanceof Error ? err.message : String(err)}`)
      return null
    } finally {
      setIsLoading(false)
    }
  }, [data, chartType])

  // Calculate indicator values
  const indicatorResults = useMemo(() => {
    if (!data || data.length === 0 || indicators.length === 0) return []

    return indicators.map(({ id, params }) => {
      const result = indicatorEngine.calculate(id, data, params)
      return result ? { ...result, instanceId: `${id}_${JSON.stringify(params)}` } : null
    }).filter((result): result is IndicatorResult & { instanceId: string } => result !== null)
  }, [data, indicators])

  // Handle container resize
  useEffect(() => {
    if (!autoSize) return

    const handleResize = () => {
      if (chartContainerRef.current) {
        const rect = chartContainerRef.current.getBoundingClientRect()
        setContainerSize({
          width: rect.width,
          height: rect.height
        })
      }
    }

    const resizeObserver = new ResizeObserver(handleResize)
    if (chartContainerRef.current) {
      resizeObserver.observe(chartContainerRef.current)
    }

    return () => {
      resizeObserver.disconnect()
    }
  }, [autoSize])

  // Initialize chart
  useEffect(() => {
    if (!chartContainerRef.current || !processedData) return

    // Create chart
    const chart = createChart(chartContainerRef.current, {
      width: containerSize.width,
      height: containerSize.height,
      layout: {
        background: { type: ColorType.Solid, color: theme === 'dark' ? '#1a1a1a' : '#ffffff' },
        textColor: theme === 'dark' ? '#d1d4dc' : '#333333',
      },
      grid: {
        vertLines: { color: theme === 'dark' ? '#2B2B43' : '#e1e1e1' },
        horzLines: { color: theme === 'dark' ? '#2B2B43' : '#e1e1e1' },
      },
      crosshair: {
        mode: 1,
        vertLine: {
          width: 1,
          color: theme === 'dark' ? '#758696' : '#9B7DFF',
          style: LineStyle.Solid,
        },
        horzLine: {
          width: 1,
          color: theme === 'dark' ? '#758696' : '#9B7DFF',
          style: LineStyle.Solid,
        },
      },
      timeScale: {
        borderColor: theme === 'dark' ? '#485c7b' : '#cccccc',
        timeVisible: true,
        secondsVisible: false,
      },
      rightPriceScale: {
        borderColor: theme === 'dark' ? '#485c7b' : '#cccccc',
      },
    })

    chartRef.current = chart

    // Create main series based on chart type
    let series: ISeriesApi<any>

    switch (chartType) {
      case 'candlestick':
      case 'renko':
      case 'heikin_ashi':
        series = chart.addCandlestickSeries({
          upColor: '#26a69a',
          downColor: '#ef5350',
          borderVisible: false,
          wickUpColor: '#26a69a',
          wickDownColor: '#ef5350',
        })
        series.setData(processedData)
        break

      case 'line':
      case 'point_figure':
        series = chart.addLineSeries({
          color: '#2196F3',
          lineWidth: 2,
        })
        series.setData(processedData.map(d => ({
          time: d.time,
          value: 'close' in d ? d.close : 'value' in d ? d.value : 0
        })))
        break

      case 'area':
        series = chart.addAreaSeries({
          topColor: 'rgba(33, 150, 243, 0.56)',
          bottomColor: 'rgba(33, 150, 243, 0.04)',
          lineColor: 'rgba(33, 150, 243, 1)',
          lineWidth: 2,
        })
        series.setData(processedData.map(d => ({
          time: d.time,
          value: 'close' in d ? d.close : 'value' in d ? d.value : 0
        })))
        break

      case 'volume_profile':
        // For volume profile, we'll use a histogram series
        series = chart.addHistogramSeries({
          color: '#26a69a',
          priceFormat: {
            type: 'volume',
          },
        })
        series.setData(processedData.map(d => ({
          time: d.time,
          value: 'volume' in d ? d.volume : 0,
          color: '#26a69a'
        })))
        break

      default:
        series = chart.addCandlestickSeries()
        series.setData(processedData)
    }

    seriesRef.current = series

    // Add price change listener
    if (onPriceChange) {
      chart.subscribeCrosshairMove((param) => {
        if (param.time && param.point && seriesRef.current) {
          const price = seriesRef.current.coordinateToPrice(param.point.y)
          if (price !== null) {
            onPriceChange(price)
          }
        }
      })
    }

    // Cleanup function
    return () => {
      if (chartRef.current) {
        chartRef.current.remove()
        chartRef.current = null
      }
    }
  }, [processedData, containerSize, theme, chartType, onPriceChange])

  // Add indicators
  useEffect(() => {
    if (!chartRef.current || !indicatorResults) return

    // Clear existing indicator series
    indicatorSeriesRefs.current.forEach(series => {
      chartRef.current?.removeSeries(series)
    })
    indicatorSeriesRefs.current.clear()

    // Add new indicator series
    indicatorResults.forEach(result => {
      if (!chartRef.current) return

      const series = chartRef.current.addLineSeries({
        color: result.metadata.color,
        lineWidth: result.metadata.lineWidth as any, // Type assertion for lineWidth compatibility
        lineStyle: result.metadata.style === 'dashed' ? LineStyle.Dashed : LineStyle.Solid,
        priceScaleId: result.metadata.overlay ? 'right' : 'left',
      })

      const data = result.values
        .filter(v => v.value !== null && isFinite(v.value!))
        .map(v => ({
          time: v.time,
          value: v.value!
        }))

      series.setData(data)
      indicatorSeriesRefs.current.set(result.instanceId, series)
    })
  }, [indicatorResults])

  // Update chart size
  useEffect(() => {
    if (chartRef.current) {
      chartRef.current.applyOptions({
        width: containerSize.width,
        height: containerSize.height,
      })
    }
  }, [containerSize])

  if (error) {
    return (
      <Alert
        type="error"
        message="Chart Error"
        description={error}
        showIcon
        style={{ margin: '20px' }}
      />
    )
  }

  return (
    <div style={{ position: 'relative', width: '100%', height: '100%' }}>
      {isLoading && (
        <div style={{
          position: 'absolute',
          top: '50%',
          left: '50%',
          transform: 'translate(-50%, -50%)',
          zIndex: 1000,
        }}>
          <Spin size="large" />
        </div>
      )}
      <div
        ref={chartContainerRef}
        style={{
          width: autoSize ? '100%' : width,
          height: autoSize ? '100%' : height,
          opacity: isLoading ? 0.5 : 1,
        }}
      />
    </div>
  )
}

export default AdvancedChartContainer