/**
 * Pattern Recognition Module
 * 
 * This module provides comprehensive pattern recognition capabilities including:
 * - Custom pattern definition framework for creating user-defined patterns
 * - Pattern-based alert system with configurable notifications
 * - Historical pattern analysis with performance metrics
 * - AI-powered pattern recognition using machine learning models
 * - Pattern management dashboard for organizing and configuring patterns
 */

export { default as CustomPatternBuilder } from './CustomPatternBuilder'
export { default as PatternManagementDashboard } from './PatternManagementDashboard'
export { default as PatternAlertSystem } from './PatternAlertSystem'
export { default as HistoricalPatternAnalysis } from './HistoricalPatternAnalysis'
export { default as AIPatternRecognition } from './AIPatternRecognition'

// Re-export pattern recognition service and types
export { 
  patternRecognition,
  type PatternDetectionConfig,
  type PatternAlert
} from '../../services/patternRecognition'

export type {
  ChartPattern,
  PatternDefinition,
  PatternRule,
  PatternCoordinates
} from '../../types/charting'