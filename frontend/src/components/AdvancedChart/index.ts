/**
 * Advanced Chart Module
 * 
 * This module provides advanced charting capabilities including:
 * - Multiple chart types (Renko, Point & Figure, Volume Profile, Heikin Ashi)
 * - Chart type selection interface
 * - Advanced chart container with lightweight-charts integration
 * - Data processing for exotic chart types
 */

export { default as ChartContainer } from './ChartContainer'
export { default as ChartTypeSelector } from './ChartTypeSelector'

// Re-export advanced charting types
export type {
  ChartType,
  RenkoData,
  PointFigureData,
  VolumeProfileData
} from '../../types/charting'

// Re-export data processors
export { 
  chartDataProcessor,
  type RenkoConfig,
  type PointFigureConfig,
  type VolumeProfileConfig
} from '../../services/chartDataProcessors'