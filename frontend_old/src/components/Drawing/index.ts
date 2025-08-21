/**
 * Drawing Module
 * 
 * This module provides comprehensive drawing and annotation capabilities including:
 * - Interactive drawing canvas with trend lines, shapes, and annotations
 * - Drawing tools toolbar with style configuration
 * - Geometric shape tools (rectangles, circles, channels, fibonacci)
 * - Advanced annotation system with text formatting
 * - Drawing template library for reuse and sharing
 * - Drawing persistence and export/import functionality
 */

export { default as DrawingToolbar } from './DrawingToolbar'
export { default as DrawingCanvas } from './DrawingCanvas'
export { default as ShapeTools } from './ShapeTools'
export { default as AnnotationTools } from './AnnotationTools'
export { default as DrawingTemplateLibrary } from './DrawingTemplateLibrary'

// Re-export drawing service and types
export { 
  drawingService,
  type DrawingState,
  type DrawingEventHandlers
} from '../../services/drawingService'

export type {
  DrawingObject,
  DrawingTool,
  Point
} from '../../types/charting'