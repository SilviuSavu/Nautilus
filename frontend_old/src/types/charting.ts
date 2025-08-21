/**
 * Advanced Charting Types
 * Comprehensive type definitions for advanced charting features
 */

import { OHLCVData, Instrument } from '../components/Chart/types/chartTypes'

// Re-export basic types
export type { OHLCVData, Instrument }

// Advanced Chart Types
export type ChartType = 'candlestick' | 'line' | 'area' | 'renko' | 'point_figure' | 'volume_profile' | 'heikin_ashi'

// Technical Indicator Types
export interface IndicatorParameter {
  name: string
  type: 'number' | 'string' | 'boolean' | 'color'
  defaultValue: any
  min?: number
  max?: number
  options?: string[]
}

export interface TechnicalIndicator {
  id: string
  name: string
  type: 'built_in' | 'custom' | 'scripted'
  parameters: IndicatorParameter[]
  calculation: {
    script?: string
    function?: Function
    period: number
    source: 'close' | 'open' | 'high' | 'low' | 'volume'
  }
  display: {
    color: string
    lineWidth: number
    style: 'solid' | 'dashed' | 'dotted'
    overlay: boolean
  }
  alerts?: AlertCondition[]
}

export interface AlertCondition {
  id: string
  type: 'crossover' | 'crossunder' | 'greater_than' | 'less_than'
  value?: number
  compareIndicatorId?: string
  enabled: boolean
}

// Chart Layout Types
export interface Point {
  x: number
  y: number
  time?: string
  price?: number
}

export interface ChartPosition {
  chartId: string
  row: number
  column: number
  rowSpan: number
  columnSpan: number
}

export interface ChartConfig {
  id: string
  instrument: Instrument
  chartType: ChartType
  timeframe: string
  indicators: string[] // Indicator IDs
  drawings: string[] // Drawing IDs
  theme: string
}

export interface ChartLayout {
  id: string
  name: string
  charts: ChartConfig[]
  layout: {
    rows: number
    columns: number
    chartPositions: ChartPosition[]
  }
  synchronization: {
    crosshair: boolean
    zoom: boolean
    timeRange: boolean
  }
  theme: ChartTheme
}

export interface ChartTheme {
  id: string
  name: string
  colors: {
    background: string
    grid: string
    crosshair: string
    upCandle: string
    downCandle: string
    volume: string
    text: string
  }
  fonts: {
    family: string
    sizes: {
      small: number
      medium: number
      large: number
    }
  }
}

// Drawing Tools Types
export interface DrawingObject {
  id: string
  type: 'trend_line' | 'rectangle' | 'circle' | 'channel' | 'text' | 'fibonacci' | 'arrow'
  coordinates: Point[]
  style: {
    color: string
    lineWidth: number
    fillColor?: string
    transparency?: number
    fontSize?: number
  }
  text?: string
  anchored: boolean
  persistent: boolean
  chartId: string
  timeframe: string
}

export interface DrawingTool {
  id: string
  name: string
  icon: string
  type: DrawingObject['type']
  category: 'lines' | 'shapes' | 'text' | 'fibonacci'
  cursor: string
}

// Pattern Recognition Types
export interface PatternCoordinates {
  points: Point[]
  boundingBox: {
    left: number
    top: number
    right: number
    bottom: number
  }
}

export interface ChartPattern {
  id: string
  name: string
  type: 'head_shoulders' | 'triangle' | 'flag' | 'wedge' | 'double_top' | 'double_bottom' | 'cup_handle' | 'custom'
  confidence: number
  coordinates: PatternCoordinates
  timeframe: string
  status: 'forming' | 'completed' | 'broken'
  projectedTarget?: number
  stopLoss?: number
  detectedAt: string
  instrument: string
}

export interface PatternDefinition {
  id: string
  name: string
  type: ChartPattern['type']
  rules: PatternRule[]
  minBars: number
  maxBars: number
  minConfidence: number
}

export interface PatternRule {
  type: 'price_action' | 'volume' | 'indicator'
  condition: string
  parameters: Record<string, any>
}

// Advanced Chart Data Types
export interface RenkoData {
  time: string
  open: number
  close: number
  trend: 'up' | 'down'
  brickSize: number
}

export interface PointFigureData {
  column: number
  boxes: Array<{
    price: number
    type: 'X' | 'O'
  }>
  time: string
}

export interface VolumeProfileData {
  priceLevel: number
  volume: number
  buyVolume: number
  sellVolume: number
  pocLevel: boolean // Point of Control
}

// Indicator Results
export interface IndicatorResult {
  indicatorId: string
  values: Array<{
    time: string
    value: number | null
    metadata?: Record<string, any>
  }>
  metadata: {
    name: string
    color: string
    lineWidth: number
    style: string
    overlay: boolean
  }
}

// Chart Store Types
export interface AdvancedChartStore {
  // Basic Chart State
  currentInstrument: Instrument | null
  chartType: ChartType
  timeframe: string
  isLoading: boolean
  error: ChartError | null
  
  // Data
  ohlcvData: OHLCVData[]
  renkoData: RenkoData[]
  pointFigureData: PointFigureData[]
  volumeProfileData: VolumeProfileData[]
  
  // Indicators
  indicators: Map<string, TechnicalIndicator>
  activeIndicators: string[]
  indicatorResults: Map<string, IndicatorResult>
  
  // Layouts and Multi-Chart
  currentLayout: ChartLayout | null
  layouts: ChartLayout[]
  
  // Drawing Tools
  drawingMode: string | null
  drawings: Map<string, DrawingObject>
  drawingTools: DrawingTool[]
  
  // Pattern Recognition
  patterns: ChartPattern[]
  patternDefinitions: PatternDefinition[]
  patternDetectionEnabled: boolean
  
  // Display Settings
  theme: ChartTheme
  synchronization: {
    enabled: boolean
    crosshair: boolean
    zoom: boolean
    timeRange: boolean
  }
  
  // Actions
  setInstrument: (instrument: Instrument) => void
  setChartType: (type: ChartType) => void
  setTimeframe: (timeframe: string) => void
  
  // Indicator Actions
  addIndicator: (indicatorId: string, params: Record<string, any>) => void
  removeIndicator: (instanceId: string) => void
  updateIndicatorParams: (instanceId: string, params: Record<string, any>) => void
  
  // Layout Actions
  createLayout: (layout: Omit<ChartLayout, 'id'>) => string
  loadLayout: (layoutId: string) => void
  updateLayout: (layoutId: string, changes: Partial<ChartLayout>) => void
  deleteLayout: (layoutId: string) => void
  
  // Drawing Actions
  startDrawing: (toolId: string) => void
  finishDrawing: () => void
  addDrawing: (drawing: Omit<DrawingObject, 'id'>) => string
  updateDrawing: (drawingId: string, changes: Partial<DrawingObject>) => void
  deleteDrawing: (drawingId: string) => void
  
  // Pattern Actions
  enablePatternDetection: (enabled: boolean) => void
  addPatternDefinition: (definition: Omit<PatternDefinition, 'id'>) => string
  updatePatternDefinition: (id: string, changes: Partial<PatternDefinition>) => void
  deletePatternDefinition: (id: string) => void
}

export interface ChartError {
  type: 'connection' | 'data' | 'rendering' | 'indicator' | 'pattern'
  message: string
  timestamp: string
  context?: Record<string, any>
}

// API Types
export interface ChartDataRequest {
  instrument: Instrument
  timeframe: string
  chartType: ChartType
  startDate?: string
  endDate?: string
  indicators?: string[]
}

export interface ChartDataResponse {
  ohlcv: OHLCVData[]
  indicators: Record<string, IndicatorResult>
  patterns?: ChartPattern[]
  metadata: {
    instrument: Instrument
    timeframe: string
    totalBars: number
    startDate: string
    endDate: string
  }
}

// Utility Types
export type ChartEventType = 
  | 'price_update' 
  | 'indicator_update' 
  | 'pattern_detected' 
  | 'drawing_created' 
  | 'alert_triggered'

export interface ChartEvent {
  type: ChartEventType
  timestamp: string
  data: any
  chartId?: string
}

// Main types are already exported above