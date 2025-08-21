/**
 * Layout Module
 * 
 * This module provides comprehensive chart layout management including:
 * - Flexible multi-chart grid layouts
 * - Layout templates and persistence
 * - Chart synchronization groups
 * - Responsive chart resizing
 * - Layout management interface
 */

export { default as LayoutManager } from './LayoutManager'
export { default as MultiChartView } from './MultiChartView'
export { default as LayoutTemplates } from './LayoutTemplates'

// Re-export layout service and types
export { 
  chartLayoutService,
  type LayoutTemplate,
  type SynchronizationGroup 
} from '../../services/chartLayoutService'

export type {
  ChartLayout,
  ChartConfig,
  ChartPosition
} from '../../types/charting'