import { create } from 'zustand'
import { persist } from 'zustand/middleware'
import { ChartStore, Timeframe, Instrument, IndicatorConfig, ChartData, ChartSettings, ChartError } from '../types/chartTypes'

const defaultSettings: ChartSettings = {
  timeframe: '1h',
  showVolume: true,
  indicators: [],
  crosshair: true,
  grid: true,
  timezone: 'UTC'
}

const defaultChartData: ChartData = {
  candles: [],
  volume: []
}

// No default instrument - user must select one explicitly

export const useChartStore = create<ChartStore>()(
  persist(
    (set, get) => ({
      currentInstrument: null,
      timeframe: '1h',
      indicators: [],
      chartData: defaultChartData,
      settings: defaultSettings,
      isLoading: false,
      error: null,
      realTimeUpdates: true,

      setCurrentInstrument: (instrument: Instrument | null) => {
        set({ currentInstrument: instrument })
      },

      setTimeframe: (timeframe: Timeframe) => {
        set({ 
          timeframe,
          settings: { ...get().settings, timeframe }
        })
      },

      addIndicator: (indicator: IndicatorConfig) => {
        const currentIndicators = get().indicators
        const exists = currentIndicators.find(ind => ind.id === indicator.id)
        if (!exists) {
          set({ 
            indicators: [...currentIndicators, indicator],
            settings: { ...get().settings, indicators: [...currentIndicators, indicator] }
          })
        }
      },

      removeIndicator: (indicatorId: string) => {
        const filteredIndicators = get().indicators.filter(ind => ind.id !== indicatorId)
        set({ 
          indicators: filteredIndicators,
          settings: { ...get().settings, indicators: filteredIndicators }
        })
      },

      updateIndicator: (indicatorId: string, config: Partial<IndicatorConfig>) => {
        const updatedIndicators = get().indicators.map(ind => 
          ind.id === indicatorId ? { ...ind, ...config } : ind
        )
        set({ 
          indicators: updatedIndicators,
          settings: { ...get().settings, indicators: updatedIndicators }
        })
      },

      setChartData: (data: ChartData) => {
        set({ chartData: data })
      },

      updateSettings: (newSettings: Partial<ChartSettings>) => {
        set({ 
          settings: { ...get().settings, ...newSettings }
        })
      },

      setLoading: (isLoading: boolean) => {
        set({ isLoading })
      },

      setError: (error: ChartError | null) => {
        set({ error })
      },

      toggleRealTimeUpdates: () => {
        set({ realTimeUpdates: !get().realTimeUpdates })
      }
    }),
    {
      name: 'chart-store',
      partialize: (state) => ({
        timeframe: state.timeframe,
        settings: state.settings,
        indicators: state.indicators,
        currentInstrument: state.currentInstrument
      })
    }
  )
)