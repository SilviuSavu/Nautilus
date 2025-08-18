// Chart Types for TradingView Lightweight Charts Integration

export interface OHLCVData {
  time: string
  open: number
  high: number
  low: number
  close: number
  volume: number
}

export interface Instrument {
  id: string
  symbol: string
  venue: string
  name: string
  assetClass: string
  currency: string
  secType?: string
  exchange?: string
}

export type Timeframe = '1m' | '2m' | '5m' | '10m' | '15m' | '30m' | '1h' | '2h' | '4h' | '1d' | '1w' | '1M'

export interface IndicatorConfig {
  id: string
  type: 'SMA' | 'EMA'
  period: number
  color: string
  visible: boolean
}

export interface ChartSettings {
  timeframe: Timeframe
  showVolume: boolean
  indicators: IndicatorConfig[]
  crosshair: boolean
  grid: boolean
  timezone: string
}

export interface ChartData {
  candles: OHLCVData[]
  volume: Array<{ time: string; value: number; color: string }>
}

export interface PriceUpdate {
  symbol: string
  price: number
  volume: number
  timestamp: string
}

export interface ChartError {
  type: 'connection' | 'data' | 'rendering'
  message: string
  timestamp: string
}

export interface ChartStore {
  currentInstrument: Instrument | null
  timeframe: Timeframe
  indicators: IndicatorConfig[]
  chartData: ChartData
  settings: ChartSettings
  isLoading: boolean
  error: ChartError | null
  realTimeUpdates: boolean
  
  // Actions
  setCurrentInstrument: (instrument: Instrument | null) => void
  setTimeframe: (timeframe: Timeframe) => void
  addIndicator: (indicator: IndicatorConfig) => void
  removeIndicator: (indicatorId: string) => void
  updateIndicator: (indicatorId: string, config: Partial<IndicatorConfig>) => void
  setChartData: (data: ChartData) => void
  updateSettings: (settings: Partial<ChartSettings>) => void
  setLoading: (isLoading: boolean) => void
  setError: (error: ChartError | null) => void
  toggleRealTimeUpdates: () => void
}