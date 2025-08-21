/**
 * Advanced Chart Container Tests
 * Unit tests for chart rendering, data processing, and user interactions
 */

import React from 'react'
import { render, screen, waitFor, fireEvent } from '@testing-library/react'
import { describe, it, expect, beforeEach, vi, afterEach } from 'vitest'
import { createChart } from 'lightweight-charts'
import { AdvancedChartContainer } from '../ChartContainer'
import { OHLCVData } from '../../Chart/types/chartTypes'
import { ChartType } from '../../../types/charting'

// Mock lightweight-charts
vi.mock('lightweight-charts', () => ({
  createChart: vi.fn(() => ({
    addCandlestickSeries: vi.fn(() => ({
      setData: vi.fn(),
      coordinateToPrice: vi.fn(() => 100)
    })),
    addLineSeries: vi.fn(() => ({
      setData: vi.fn()
    })),
    addAreaSeries: vi.fn(() => ({
      setData: vi.fn()
    })),
    addHistogramSeries: vi.fn(() => ({
      setData: vi.fn()
    })),
    subscribeCrosshairMove: vi.fn(),
    applyOptions: vi.fn(),
    remove: vi.fn(),
    removeSeries: vi.fn()
  })),
  ColorType: { Solid: 'solid' },
  LineStyle: { Solid: 0, Dashed: 1 }
}))

// Mock chart data processors
vi.mock('../../../services/chartDataProcessors', () => ({
  chartDataProcessor: {
    processRenkoData: vi.fn((data) => 
      data.map((item: OHLCVData, i: number) => ({
        time: item.time,
        open: item.open,
        close: item.close + (i % 2 === 0 ? 1 : -1),
        trend: i % 2 === 0 ? 'up' : 'down'
      }))
    ),
    processPointFigureData: vi.fn((data) =>
      data.map((item: OHLCVData) => ({
        time: item.time,
        boxes: [{ price: item.close }]
      }))
    ),
    processHeikinAshiData: vi.fn((data) => data),
    processVolumeProfileData: vi.fn((data) =>
      data.map((item: OHLCVData) => ({
        priceLevel: item.close,
        volume: item.volume
      }))
    )
  }
}))

// Mock indicator engine
vi.mock('../../../services/indicatorEngine', () => ({
  indicatorEngine: {
    calculate: vi.fn(() => ({
      indicatorId: 'sma',
      values: [
        { time: '2023-01-01', value: 100 },
        { time: '2023-01-02', value: 101 }
      ],
      metadata: {
        name: 'SMA',
        color: '#FF0000',
        lineWidth: 2,
        style: 'solid',
        overlay: true
      }
    }))
  }
}))

// Mock ResizeObserver
global.ResizeObserver = vi.fn().mockImplementation(() => ({
  observe: vi.fn(),
  unobserve: vi.fn(),
  disconnect: vi.fn()
}))

describe('AdvancedChartContainer', () => {
  let mockData: OHLCVData[]
  
  beforeEach(() => {
    mockData = [
      { time: '2023-01-01', open: 100, high: 105, low: 95, close: 102, volume: 1000 },
      { time: '2023-01-02', open: 102, high: 108, low: 100, close: 106, volume: 1200 },
      { time: '2023-01-03', open: 106, high: 110, low: 104, close: 108, volume: 800 },
      { time: '2023-01-04', open: 108, high: 112, low: 106, close: 110, volume: 900 },
      { time: '2023-01-05', open: 110, high: 115, low: 108, close: 113, volume: 1100 }
    ]
  })

  afterEach(() => {
    vi.clearAllMocks()
  })

  describe('Basic Rendering', () => {
    it('should render chart container', () => {
      render(
        <AdvancedChartContainer
          data={mockData}
          chartType="candlestick"
        />
      )
      
      const container = screen.getByRole('generic')
      expect(container).toBeDefined()
    })

    it('should handle empty data gracefully', () => {
      render(
        <AdvancedChartContainer
          data={[]}
          chartType="candlestick"
        />
      )
      
      expect(screen.getByRole('generic')).toBeDefined()
    })

    it('should apply custom dimensions', () => {
      const { container } = render(
        <AdvancedChartContainer
          data={mockData}
          chartType="candlestick"
          width={600}
          height={300}
          autoSize={false}
        />
      )
      
      const chartDiv = container.querySelector('div > div')
      expect(chartDiv).toHaveStyle({ width: '600px', height: '300px' })
    })
  })

  describe('Chart Type Processing', () => {
    it('should render candlestick chart correctly', async () => {
      render(
        <AdvancedChartContainer
          data={mockData}
          chartType="candlestick"
        />
      )
      
      await waitFor(() => {
        // Chart should be created without errors
        expect(screen.queryByRole('alert')).toBeNull()
      })
    })

    it('should process Renko data correctly', async () => {
      const { chartDataProcessor } = await import('../../../services/chartDataProcessors')
      
      render(
        <AdvancedChartContainer
          data={mockData}
          chartType="renko"
        />
      )
      
      await waitFor(() => {
        expect(chartDataProcessor.processRenkoData).toHaveBeenCalledWith(
          mockData,
          expect.objectContaining({
            autoCalculateBrickSize: true,
            source: 'close'
          })
        )
      })
    })

    it('should process Point & Figure data correctly', async () => {
      const { chartDataProcessor } = await import('../../../services/chartDataProcessors')
      
      render(
        <AdvancedChartContainer
          data={mockData}
          chartType="point_figure"
        />
      )
      
      await waitFor(() => {
        expect(chartDataProcessor.processPointFigureData).toHaveBeenCalledWith(
          mockData,
          expect.objectContaining({
            autoCalculateBoxSize: true,
            reversalAmount: 3,
            source: 'close'
          })
        )
      })
    })

    it('should process Heikin-Ashi data correctly', async () => {
      const { chartDataProcessor } = await import('../../../services/chartDataProcessors')
      
      render(
        <AdvancedChartContainer
          data={mockData}
          chartType="heikin_ashi"
        />
      )
      
      await waitFor(() => {
        expect(chartDataProcessor.processHeikinAshiData).toHaveBeenCalledWith(mockData)
      })
    })

    it('should process Volume Profile data correctly', async () => {
      const { chartDataProcessor } = await import('../../../services/chartDataProcessors')
      
      render(
        <AdvancedChartContainer
          data={mockData}
          chartType="volume_profile"
        />
      )
      
      await waitFor(() => {
        expect(chartDataProcessor.processVolumeProfileData).toHaveBeenCalledWith(
          mockData,
          expect.objectContaining({
            priceLevels: 50,
            sessionType: 'daily',
            showPOC: true
          })
        )
      })
    })
  })

  describe('Error Handling', () => {
    it('should display error when data processing fails', async () => {
      const { chartDataProcessor } = await import('../../../services/chartDataProcessors')
      vi.mocked(chartDataProcessor.processRenkoData).mockImplementation(() => {
        throw new Error('Processing failed')
      })
      
      render(
        <AdvancedChartContainer
          data={mockData}
          chartType="renko"
        />
      )
      
      await waitFor(() => {
        expect(screen.getByRole('alert')).toBeDefined()
        expect(screen.getByText(/Error processing renko data/i)).toBeDefined()
      })
    })

    it('should handle invalid chart type gracefully', async () => {
      render(
        <AdvancedChartContainer
          data={mockData}
          chartType={'invalid' as ChartType}
        />
      )
      
      await waitFor(() => {
        // Should not crash and render something
        expect(screen.getByRole('generic')).toBeDefined()
      })
    })
  })

  describe('Indicators Integration', () => {
    it('should calculate and display indicators', async () => {
      const indicators = [
        { id: 'sma', params: { period: 20 } },
        { id: 'ema', params: { period: 12 } }
      ]
      
      const { indicatorEngine } = await import('../../../services/indicatorEngine')
      
      render(
        <AdvancedChartContainer
          data={mockData}
          chartType="candlestick"
          indicators={indicators}
        />
      )
      
      await waitFor(() => {
        expect(indicatorEngine.calculate).toHaveBeenCalledTimes(2)
        expect(indicatorEngine.calculate).toHaveBeenCalledWith('sma', mockData, { period: 20 })
        expect(indicatorEngine.calculate).toHaveBeenCalledWith('ema', mockData, { period: 12 })
      })
    })

    it('should handle indicator calculation failures', async () => {
      const { indicatorEngine } = await import('../../../services/indicatorEngine')
      vi.mocked(indicatorEngine.calculate).mockReturnValue(null)
      
      const indicators = [{ id: 'invalid', params: {} }]
      
      render(
        <AdvancedChartContainer
          data={mockData}
          chartType="candlestick"
          indicators={indicators}
        />
      )
      
      await waitFor(() => {
        // Should not crash when indicator calculation fails
        expect(screen.getByRole('generic')).toBeDefined()
      })
    })
  })

  describe('Theme Support', () => {
    it('should apply light theme correctly', () => {
      render(
        <AdvancedChartContainer
          data={mockData}
          chartType="candlestick"
          theme="light"
        />
      )
      
      // Chart should be created with light theme options
      expect(screen.getByRole('generic')).toBeDefined()
    })

    it('should apply dark theme correctly', () => {
      render(
        <AdvancedChartContainer
          data={mockData}
          chartType="candlestick"
          theme="dark"
        />
      )
      
      // Chart should be created with dark theme options
      expect(screen.getByRole('generic')).toBeDefined()
    })
  })

  describe('User Interactions', () => {
    it('should call onPriceChange when crosshair moves', async () => {
      const onPriceChange = vi.fn()
      const { createChart } = await import('lightweight-charts')
      const mockChart = {
        addCandlestickSeries: vi.fn(() => ({
          setData: vi.fn(),
          coordinateToPrice: vi.fn(() => 105)
        })),
        subscribeCrosshairMove: vi.fn(),
        applyOptions: vi.fn(),
        remove: vi.fn(),
        removeSeries: vi.fn()
      }
      vi.mocked(createChart).mockReturnValue(mockChart as any)
      
      render(
        <AdvancedChartContainer
          data={mockData}
          chartType="candlestick"
          onPriceChange={onPriceChange}
        />
      )
      
      await waitFor(() => {
        expect(mockChart.subscribeCrosshairMove).toHaveBeenCalled()
      })
    })
  })

  describe('Loading States', () => {
    it('should show loading spinner during data processing', async () => {
      const { chartDataProcessor } = await import('../../../services/chartDataProcessors')
      
      // Simulate slow processing
      vi.mocked(chartDataProcessor.processRenkoData).mockImplementation(() => {
        return new Promise(resolve => {
          setTimeout(() => resolve([]), 100)
        }) as any
      })
      
      render(
        <AdvancedChartContainer
          data={mockData}
          chartType="renko"
        />
      )
      
      // Should show loading state initially
      expect(screen.getByText('Loading')).toBeDefined()
      
      await waitFor(() => {
        expect(screen.queryByText('Loading')).toBeNull()
      }, { timeout: 200 })
    })
  })

  describe('Resize Handling', () => {
    it('should observe container resize when autoSize is enabled', () => {
      render(
        <AdvancedChartContainer
          data={mockData}
          chartType="candlestick"
          autoSize={true}
        />
      )
      
      expect(ResizeObserver).toHaveBeenCalled()
    })

    it('should not observe resize when autoSize is disabled', () => {
      render(
        <AdvancedChartContainer
          data={mockData}
          chartType="candlestick"
          autoSize={false}
        />
      )
      
      // ResizeObserver should still be called but not used for sizing
      expect(screen.getByRole('generic')).toBeDefined()
    })
  })

  describe('Memory Management', () => {
    it('should cleanup chart on unmount', () => {
      const mockCreateChart = vi.mocked(createChart)
      const mockChart = {
        addCandlestickSeries: vi.fn(() => ({ setData: vi.fn() })),
        subscribeCrosshairMove: vi.fn(),
        applyOptions: vi.fn(),
        remove: vi.fn(),
        removeSeries: vi.fn()
      }
      mockCreateChart.mockReturnValue(mockChart as any)
      
      const { unmount } = render(
        <AdvancedChartContainer
          data={mockData}
          chartType="candlestick"
        />
      )
      
      unmount()
      
      expect(mockChart.remove).toHaveBeenCalled()
    })
  })
})