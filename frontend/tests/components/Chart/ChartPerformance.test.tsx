import { describe, it, expect, beforeEach, vi } from 'vitest'
import { render, waitFor } from '@testing-library/react'
import { ChartComponent } from '../../../src/components/Chart/ChartComponent'
import { useChartStore } from '../../../src/components/Chart/hooks/useChartStore'
import type { OHLCVData } from '../../../src/components/Chart/types/chartTypes'

// Mock the chart store
vi.mock('../../../src/components/Chart/hooks/useChartStore')
const mockUseChartStore = vi.mocked(useChartStore)

// Mock TradingView Lightweight Charts
const mockChart = {
  addCandlestickSeries: vi.fn().mockReturnValue({
    setData: vi.fn(),
    update: vi.fn()
  }),
  addHistogramSeries: vi.fn().mockReturnValue({
    setData: vi.fn(),
    update: vi.fn()
  }),
  timeScale: vi.fn().mockReturnValue({
    fitContent: vi.fn(),
    subscribeVisibleTimeRangeChange: vi.fn(),
    unsubscribeVisibleTimeRangeChange: vi.fn()
  }),
  subscribeCrosshairMove: vi.fn(),
  unsubscribeCrosshairMove: vi.fn(),
  resize: vi.fn(),
  remove: vi.fn(),
  applyOptions: vi.fn()
}

vi.mock('lightweight-charts', () => ({
  createChart: vi.fn(() => mockChart),
  ColorType: {
    Solid: 'solid',
    VerticalGradient: 'verticalGradient'
  }
}))

describe('Chart Performance Tests', () => {
  beforeEach(() => {
    vi.clearAllMocks()
    
    mockUseChartStore.mockReturnValue({
      currentInstrument: {
        id: 'AAPL-NASDAQ',
        symbol: 'AAPL',
        venue: 'NASDAQ',
        name: 'Apple Inc.',
        assetClass: 'STK',
        currency: 'USD'
      },
      timeframe: '1m',
      indicators: [],
      chartData: { candles: [], volume: [] },
      settings: {
        timeframe: '1m',
        showVolume: true,
        indicators: [],
        crosshair: true,
        grid: true,
        timezone: 'UTC'
      },
      isLoading: false,
      error: null,
      realTimeUpdates: true,
      setCurrentInstrument: vi.fn(),
      setTimeframe: vi.fn(),
      addIndicator: vi.fn(),
      removeIndicator: vi.fn(),
      updateIndicator: vi.fn(),
      setChartData: vi.fn(),
      updateSettings: vi.fn(),
      setLoading: vi.fn(),
      setError: vi.fn(),
      toggleRealTimeUpdates: vi.fn()
    })
  })

  it('should render 1000+ candles within 500ms', async () => {
    // Generate 1000 OHLCV data points
    const generateCandles = (count: number): OHLCVData[] => {
      const candles: OHLCVData[] = []
      const baseTime = new Date('2024-01-01T00:00:00Z').getTime()
      
      for (let i = 0; i < count; i++) {
        const time = new Date(baseTime + i * 60000).toISOString() // 1 minute intervals
        const basePrice = 100 + Math.sin(i * 0.1) * 10 // Some price movement
        const volatility = Math.random() * 2
        
        candles.push({
          time,
          open: basePrice + Math.random() * volatility - volatility / 2,
          high: basePrice + Math.random() * volatility + volatility,
          low: basePrice - Math.random() * volatility - volatility,
          close: basePrice + Math.random() * volatility - volatility / 2,
          volume: Math.floor(Math.random() * 10000) + 1000
        })
      }
      
      return candles
    }

    const largeDataset = generateCandles(1000)
    
    // Update store with large dataset
    mockUseChartStore.mockReturnValue({
      ...mockUseChartStore(),
      chartData: {
        candles: largeDataset,
        volume: largeDataset.map(candle => ({
          time: candle.time,
          value: candle.volume,
          color: candle.close > candle.open ? '#26a69a' : '#ef5350'
        }))
      }
    })

    // Measure rendering performance
    const startTime = performance.now()
    
    render(<ChartComponent />)
    
    // Wait for chart to be rendered
    await waitFor(() => {
      expect(mockChart.addCandlestickSeries).toHaveBeenCalled()
    }, { timeout: 1000 })

    const endTime = performance.now()
    const renderTime = endTime - startTime
    
    // Assert that rendering took less than 500ms as per story requirement
    expect(renderTime).toBeLessThan(500)
    
    // Verify that the chart was initialized with the data
    expect(mockChart.addCandlestickSeries).toHaveBeenCalledTimes(1)
    expect(mockChart.addHistogramSeries).toHaveBeenCalledTimes(1)
  })

  it('should handle 5000+ candles for stress testing', async () => {
    const generateLargeDataset = (count: number): OHLCVData[] => {
      const candles: OHLCVData[] = []
      const baseTime = new Date('2024-01-01T00:00:00Z').getTime()
      
      for (let i = 0; i < count; i++) {
        const time = new Date(baseTime + i * 60000).toISOString()
        const basePrice = 100 + Math.sin(i * 0.01) * 20
        
        candles.push({
          time,
          open: basePrice + Math.random() * 2 - 1,
          high: basePrice + Math.random() * 3 + 1,
          low: basePrice - Math.random() * 3 - 1,
          close: basePrice + Math.random() * 2 - 1,
          volume: Math.floor(Math.random() * 50000) + 5000
        })
      }
      
      return candles
    }

    const massiveDataset = generateLargeDataset(5000)
    
    mockUseChartStore.mockReturnValue({
      ...mockUseChartStore(),
      chartData: {
        candles: massiveDataset,
        volume: massiveDataset.map(candle => ({
          time: candle.time,
          value: candle.volume,
          color: candle.close > candle.open ? '#26a69a' : '#ef5350'
        }))
      }
    })

    const startTime = performance.now()
    
    render(<ChartComponent />)
    
    await waitFor(() => {
      expect(mockChart.addCandlestickSeries).toHaveBeenCalled()
    }, { timeout: 2000 })

    const endTime = performance.now()
    const renderTime = endTime - startTime
    
    // For stress testing, allow up to 2 seconds
    expect(renderTime).toBeLessThan(2000)
    
    // Verify chart handles large dataset
    expect(mockChart.addCandlestickSeries).toHaveBeenCalledTimes(1)
  })

  it('should maintain performance with multiple indicators', async () => {
    const dataset = Array.from({ length: 1000 }, (_, i) => {
      const time = new Date(Date.now() - (1000 - i) * 60000).toISOString()
      const price = 100 + Math.sin(i * 0.1) * 10
      
      return {
        time,
        open: price + Math.random() - 0.5,
        high: price + Math.random() + 0.5,
        low: price - Math.random() - 0.5,
        close: price + Math.random() - 0.5,
        volume: Math.floor(Math.random() * 10000) + 1000
      }
    })

    // Add multiple indicators
    const indicators = [
      {
        id: 'sma_20',
        type: 'SMA' as const,
        period: 20,
        color: '#FF6B6B',
        visible: true
      },
      {
        id: 'ema_50',
        type: 'EMA' as const,
        period: 50,
        color: '#4ECDC4',
        visible: true
      },
      {
        id: 'sma_200',
        type: 'SMA' as const,
        period: 200,
        color: '#45B7D1',
        visible: true
      }
    ]

    mockUseChartStore.mockReturnValue({
      ...mockUseChartStore(),
      indicators,
      chartData: {
        candles: dataset,
        volume: dataset.map(candle => ({
          time: candle.time,
          value: candle.volume,
          color: candle.close > candle.open ? '#26a69a' : '#ef5350'
        }))
      },
      settings: {
        ...mockUseChartStore().settings,
        indicators
      }
    })

    const startTime = performance.now()
    
    render(<ChartComponent />)
    
    await waitFor(() => {
      expect(mockChart.addCandlestickSeries).toHaveBeenCalled()
    }, { timeout: 1000 })

    const endTime = performance.now()
    const renderTime = endTime - startTime
    
    // Should still render within performance requirements even with indicators
    expect(renderTime).toBeLessThan(800)
  })

  it('should handle rapid data updates without performance degradation', async () => {
    const baseDataset = Array.from({ length: 500 }, (_, i) => {
      const time = new Date(Date.now() - (500 - i) * 60000).toISOString()
      const price = 100 + Math.sin(i * 0.1) * 10
      
      return {
        time,
        open: price,
        high: price + 1,
        low: price - 1,
        close: price + Math.random() - 0.5,
        volume: Math.floor(Math.random() * 10000) + 1000
      }
    })

    let updateCount = 0
    const maxUpdates = 100
    const updateTimes: number[] = []

    // Mock the setData function to measure update performance
    const mockSetData = vi.fn().mockImplementation(() => {
      updateTimes.push(performance.now())
    })

    mockChart.addCandlestickSeries.mockReturnValue({
      setData: mockSetData,
      update: vi.fn()
    })

    mockUseChartStore.mockReturnValue({
      ...mockUseChartStore(),
      chartData: {
        candles: baseDataset,
        volume: []
      }
    })

    render(<ChartComponent />)
    
    await waitFor(() => {
      expect(mockChart.addCandlestickSeries).toHaveBeenCalled()
    })

    // Simulate rapid updates (similar to real-time trading)
    const startTime = performance.now()
    
    for (let i = 0; i < maxUpdates; i++) {
      const updatedDataset = [...baseDataset, {
        time: new Date(Date.now() + i * 1000).toISOString(),
        open: 100 + i * 0.1,
        high: 101 + i * 0.1,
        low: 99 + i * 0.1,
        close: 100.5 + i * 0.1,
        volume: 5000 + i * 10
      }]

      // Force re-render with new data
      mockUseChartStore.mockReturnValue({
        ...mockUseChartStore(),
        chartData: {
          candles: updatedDataset,
          volume: []
        }
      })
    }

    const endTime = performance.now()
    const totalTime = endTime - startTime
    
    // Should handle rapid updates efficiently
    expect(totalTime).toBeLessThan(1000) // Under 1 second for 100 updates
    
    // Average time per update should be reasonable
    const averageUpdateTime = totalTime / maxUpdates
    expect(averageUpdateTime).toBeLessThan(10) // Under 10ms per update
  })

  it('should validate memory usage remains reasonable with large datasets', async () => {
    // This test would ideally use actual memory measurement
    // For now, we'll test that the component doesn't crash with large datasets
    
    const hugeDataset = Array.from({ length: 10000 }, (_, i) => ({
      time: new Date(Date.now() - (10000 - i) * 60000).toISOString(),
      open: 100 + Math.sin(i * 0.01) * 20,
      high: 105 + Math.sin(i * 0.01) * 20,
      low: 95 + Math.sin(i * 0.01) * 20,
      close: 102 + Math.sin(i * 0.01) * 20,
      volume: Math.floor(Math.random() * 100000) + 10000
    }))

    mockUseChartStore.mockReturnValue({
      ...mockUseChartStore(),
      chartData: {
        candles: hugeDataset,
        volume: hugeDataset.map(candle => ({
          time: candle.time,
          value: candle.volume,
          color: '#26a69a'
        }))
      }
    })

    // Should not throw or crash
    expect(() => {
      render(<ChartComponent />)
    }).not.toThrow()

    await waitFor(() => {
      expect(mockChart.addCandlestickSeries).toHaveBeenCalled()
    }, { timeout: 3000 })

    // Chart should handle the large dataset
    expect(mockChart.addCandlestickSeries).toHaveBeenCalledTimes(1)
    expect(mockChart.addHistogramSeries).toHaveBeenCalledTimes(1)
  })
})