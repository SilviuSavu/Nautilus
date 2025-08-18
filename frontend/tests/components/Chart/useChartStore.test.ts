import { describe, it, expect, beforeEach, vi } from 'vitest'
import { renderHook, act } from '@testing-library/react'
import { useChartStore } from '../../../src/components/Chart/hooks/useChartStore'
import type { Instrument, IndicatorConfig, ChartData, ChartError } from '../../../src/components/Chart/types/chartTypes'

// Mock zustand persist
vi.mock('zustand/middleware', () => ({
  persist: vi.fn((fn) => fn)
}))

describe('useChartStore', () => {
  beforeEach(() => {
    // Reset store state before each test
    const { result } = renderHook(() => useChartStore())
    act(() => {
      result.current.setCurrentInstrument(null)
      result.current.setTimeframe('1h')
      result.current.setChartData({ candles: [], volume: [] })
      result.current.setError(null)
      result.current.setLoading(false)
      // Clear indicators
      result.current.indicators.forEach(indicator => {
        result.current.removeIndicator(indicator.id)
      })
    })
  })

  it('initializes with default values', () => {
    const { result } = renderHook(() => useChartStore())
    
    expect(result.current.currentInstrument).toBeNull()
    expect(result.current.timeframe).toBe('1h')
    expect(result.current.indicators).toEqual([])
    expect(result.current.chartData).toEqual({ candles: [], volume: [] })
    expect(result.current.isLoading).toBe(false)
    expect(result.current.error).toBeNull()
    expect(result.current.realTimeUpdates).toBe(true)
    expect(result.current.settings.timeframe).toBe('1h')
    expect(result.current.settings.showVolume).toBe(true)
    expect(result.current.settings.crosshair).toBe(true)
    expect(result.current.settings.grid).toBe(true)
    expect(result.current.settings.timezone).toBe('UTC')
  })

  it('sets current instrument', () => {
    const { result } = renderHook(() => useChartStore())
    
    const instrument: Instrument = {
      id: 'AAPL-NASDAQ',
      symbol: 'AAPL',
      venue: 'NASDAQ',
      name: 'Apple Inc.',
      assetClass: 'STK',
      currency: 'USD'
    }
    
    act(() => {
      result.current.setCurrentInstrument(instrument)
    })
    
    expect(result.current.currentInstrument).toEqual(instrument)
  })

  it('sets timeframe and updates settings', () => {
    const { result } = renderHook(() => useChartStore())
    
    act(() => {
      result.current.setTimeframe('5m')
    })
    
    expect(result.current.timeframe).toBe('5m')
    expect(result.current.settings.timeframe).toBe('5m')
  })

  it('adds indicator', () => {
    const { result } = renderHook(() => useChartStore())
    
    const indicator: IndicatorConfig = {
      id: 'sma_20',
      type: 'SMA',
      period: 20,
      color: '#FF6B6B',
      visible: true
    }
    
    act(() => {
      result.current.addIndicator(indicator)
    })
    
    expect(result.current.indicators).toContain(indicator)
    expect(result.current.settings.indicators).toContain(indicator)
  })

  it('does not add duplicate indicator', () => {
    const { result } = renderHook(() => useChartStore())
    
    const indicator: IndicatorConfig = {
      id: 'sma_20',
      type: 'SMA',
      period: 20,
      color: '#FF6B6B',
      visible: true
    }
    
    act(() => {
      result.current.addIndicator(indicator)
      result.current.addIndicator(indicator) // Try to add same indicator again
    })
    
    expect(result.current.indicators).toHaveLength(1)
    expect(result.current.indicators[0]).toEqual(indicator)
  })

  it('removes indicator', () => {
    const { result } = renderHook(() => useChartStore())
    
    const indicator: IndicatorConfig = {
      id: 'sma_20',
      type: 'SMA',
      period: 20,
      color: '#FF6B6B',
      visible: true
    }
    
    act(() => {
      result.current.addIndicator(indicator)
    })
    
    expect(result.current.indicators).toHaveLength(1)
    
    act(() => {
      result.current.removeIndicator('sma_20')
    })
    
    expect(result.current.indicators).toHaveLength(0)
    expect(result.current.settings.indicators).toHaveLength(0)
  })

  it('updates indicator', () => {
    const { result } = renderHook(() => useChartStore())
    
    const indicator: IndicatorConfig = {
      id: 'sma_20',
      type: 'SMA',
      period: 20,
      color: '#FF6B6B',
      visible: true
    }
    
    act(() => {
      result.current.addIndicator(indicator)
    })
    
    act(() => {
      result.current.updateIndicator('sma_20', { period: 50, visible: false })
    })
    
    const updatedIndicator = result.current.indicators.find(ind => ind.id === 'sma_20')
    expect(updatedIndicator).toEqual({
      id: 'sma_20',
      type: 'SMA',
      period: 50,
      color: '#FF6B6B',
      visible: false
    })
    
    const settingsIndicator = result.current.settings.indicators.find(ind => ind.id === 'sma_20')
    expect(settingsIndicator).toEqual(updatedIndicator)
  })

  it('does not update non-existent indicator', () => {
    const { result } = renderHook(() => useChartStore())
    
    const originalIndicators = result.current.indicators
    
    act(() => {
      result.current.updateIndicator('non_existent', { period: 50 })
    })
    
    expect(result.current.indicators).toEqual(originalIndicators)
  })

  it('sets chart data', () => {
    const { result } = renderHook(() => useChartStore())
    
    const chartData: ChartData = {
      candles: [
        {
          time: '2024-01-01T10:00:00Z',
          open: 100,
          high: 110,
          low: 95,
          close: 105,
          volume: 1000
        }
      ],
      volume: [
        {
          time: '2024-01-01T10:00:00Z',
          value: 1000,
          color: '#26a69a'
        }
      ]
    }
    
    act(() => {
      result.current.setChartData(chartData)
    })
    
    expect(result.current.chartData).toEqual(chartData)
  })

  it('updates settings', () => {
    const { result } = renderHook(() => useChartStore())
    
    act(() => {
      result.current.updateSettings({
        showVolume: false,
        timezone: 'America/New_York'
      })
    })
    
    expect(result.current.settings.showVolume).toBe(false)
    expect(result.current.settings.timezone).toBe('America/New_York')
    // Other settings should remain unchanged
    expect(result.current.settings.crosshair).toBe(true)
    expect(result.current.settings.grid).toBe(true)
  })

  it('sets loading state', () => {
    const { result } = renderHook(() => useChartStore())
    
    act(() => {
      result.current.setLoading(true)
    })
    
    expect(result.current.isLoading).toBe(true)
    
    act(() => {
      result.current.setLoading(false)
    })
    
    expect(result.current.isLoading).toBe(false)
  })

  it('sets error state', () => {
    const { result } = renderHook(() => useChartStore())
    
    const error: ChartError = {
      type: 'connection',
      message: 'WebSocket connection failed',
      timestamp: '2024-01-01T10:00:00Z'
    }
    
    act(() => {
      result.current.setError(error)
    })
    
    expect(result.current.error).toEqual(error)
    
    act(() => {
      result.current.setError(null)
    })
    
    expect(result.current.error).toBeNull()
  })

  it('toggles real-time updates', () => {
    const { result } = renderHook(() => useChartStore())
    
    expect(result.current.realTimeUpdates).toBe(true)
    
    act(() => {
      result.current.toggleRealTimeUpdates()
    })
    
    expect(result.current.realTimeUpdates).toBe(false)
    
    act(() => {
      result.current.toggleRealTimeUpdates()
    })
    
    expect(result.current.realTimeUpdates).toBe(true)
  })

  it('maintains multiple indicators independently', () => {
    const { result } = renderHook(() => useChartStore())
    
    const smaIndicator: IndicatorConfig = {
      id: 'sma_20',
      type: 'SMA',
      period: 20,
      color: '#FF6B6B',
      visible: true
    }
    
    const emaIndicator: IndicatorConfig = {
      id: 'ema_50',
      type: 'EMA',
      period: 50,
      color: '#4ECDC4',
      visible: false
    }
    
    act(() => {
      result.current.addIndicator(smaIndicator)
      result.current.addIndicator(emaIndicator)
    })
    
    expect(result.current.indicators).toHaveLength(2)
    expect(result.current.indicators).toContain(smaIndicator)
    expect(result.current.indicators).toContain(emaIndicator)
    
    // Update one indicator
    act(() => {
      result.current.updateIndicator('sma_20', { color: '#00FF00' })
    })
    
    const updatedSma = result.current.indicators.find(ind => ind.id === 'sma_20')
    const unchangedEma = result.current.indicators.find(ind => ind.id === 'ema_50')
    
    expect(updatedSma?.color).toBe('#00FF00')
    expect(unchangedEma).toEqual(emaIndicator) // Should remain unchanged
  })

  it('syncs indicators between main state and settings', () => {
    const { result } = renderHook(() => useChartStore())
    
    const indicator: IndicatorConfig = {
      id: 'sma_20',
      type: 'SMA',
      period: 20,
      color: '#FF6B6B',
      visible: true
    }
    
    act(() => {
      result.current.addIndicator(indicator)
    })
    
    expect(result.current.indicators).toEqual(result.current.settings.indicators)
    
    act(() => {
      result.current.updateIndicator('sma_20', { visible: false })
    })
    
    expect(result.current.indicators).toEqual(result.current.settings.indicators)
    
    act(() => {
      result.current.removeIndicator('sma_20')
    })
    
    expect(result.current.indicators).toEqual(result.current.settings.indicators)
    expect(result.current.indicators).toHaveLength(0)
  })
})