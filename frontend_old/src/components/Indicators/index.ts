/**
 * Indicators Module - Advanced Technical Indicators
 * 
 * This module provides components and utilities for managing technical indicators
 * including built-in indicators (SMA, EMA, RSI, MACD, Bollinger Bands) and 
 * custom scripted indicators.
 */

export { default as IndicatorBuilder } from './IndicatorBuilder'
export { default as ParameterConfig } from './ParameterConfig'
export { default as IndicatorLibrary } from './IndicatorLibrary'

// Re-export types and services
export type { 
  TechnicalIndicator, 
  IndicatorParameter, 
  IndicatorResult,
  AlertCondition
} from '../../services/indicatorEngine'

export { indicatorEngine } from '../../services/indicatorEngine'