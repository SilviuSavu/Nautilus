/**
 * Chart Layout Service
 * Manages multiple chart layouts, synchronization, and persistence
 */

import { ChartLayout, ChartConfig, ChartPosition, Instrument } from '../types/charting'

export interface LayoutTemplate {
  id: string
  name: string
  description: string
  thumbnail?: string
  layout: Omit<ChartLayout, 'id' | 'name'>
  category: 'trading' | 'analysis' | 'monitoring' | 'custom'
  isBuiltIn: boolean
}

export interface SynchronizationGroup {
  id: string
  chartIds: string[]
  syncSettings: {
    crosshair: boolean
    zoom: boolean
    timeRange: boolean
    instrument: boolean
  }
}

interface LayoutEventHandlers {
  onLayoutChange?: (layout: ChartLayout) => void
  onChartAdd?: (chartConfig: ChartConfig) => void
  onChartRemove?: (chartId: string) => void
  onSynchronizationChange?: (groups: SynchronizationGroup[]) => void
}

class ChartLayoutService {
  private layouts: Map<string, ChartLayout> = new Map()
  private templates: Map<string, LayoutTemplate> = new Map()
  private synchronizationGroups: Map<string, SynchronizationGroup> = new Map()
  private activeLayoutId: string | null = null
  private eventHandlers: LayoutEventHandlers = {}

  constructor() {
    this.initializeBuiltInTemplates()
  }

  // Event handling
  setEventHandlers(handlers: LayoutEventHandlers) {
    this.eventHandlers = { ...this.eventHandlers, ...handlers }
  }

  // Layout management
  createLayout(config: Omit<ChartLayout, 'id'>): string {
    const id = `layout_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`
    const layout: ChartLayout = {
      id,
      ...config
    }
    
    this.layouts.set(id, layout)
    this.eventHandlers.onLayoutChange?.(layout)
    return id
  }

  getLayout(id: string): ChartLayout | null {
    return this.layouts.get(id) || null
  }

  getAllLayouts(): ChartLayout[] {
    return Array.from(this.layouts.values())
  }

  updateLayout(id: string, changes: Partial<ChartLayout>): boolean {
    const layout = this.layouts.get(id)
    if (!layout) return false

    const updatedLayout = { ...layout, ...changes }
    this.layouts.set(id, updatedLayout)
    this.eventHandlers.onLayoutChange?.(updatedLayout)
    return true
  }

  deleteLayout(id: string): boolean {
    const deleted = this.layouts.delete(id)
    if (deleted && this.activeLayoutId === id) {
      this.activeLayoutId = null
    }
    return deleted
  }

  setActiveLayout(id: string): boolean {
    if (!this.layouts.has(id)) return false
    this.activeLayoutId = id
    return true
  }

  getActiveLayout(): ChartLayout | null {
    return this.activeLayoutId ? this.layouts.get(this.activeLayoutId) || null : null
  }

  // Chart management within layouts
  addChartToLayout(layoutId: string, chartConfig: Omit<ChartConfig, 'id'>): string | null {
    const layout = this.layouts.get(layoutId)
    if (!layout) return null

    const chartId = `chart_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`
    const newChart: ChartConfig = {
      id: chartId,
      ...chartConfig
    }

    // Find available position
    const availablePosition = this.findAvailablePosition(layout)
    if (!availablePosition) {
      // Expand layout if needed
      this.expandLayout(layout)
    }

    layout.charts.push(newChart)
    layout.layout.chartPositions.push({
      chartId,
      ...availablePosition || { row: 0, column: 0, rowSpan: 1, columnSpan: 1 }
    })

    this.layouts.set(layoutId, layout)
    this.eventHandlers.onChartAdd?.(newChart)
    return chartId
  }

  removeChartFromLayout(layoutId: string, chartId: string): boolean {
    const layout = this.layouts.get(layoutId)
    if (!layout) return false

    layout.charts = layout.charts.filter(chart => chart.id !== chartId)
    layout.layout.chartPositions = layout.layout.chartPositions.filter(pos => pos.chartId !== chartId)

    this.layouts.set(layoutId, layout)
    this.eventHandlers.onChartRemove?.(chartId)
    return true
  }

  updateChartInLayout(layoutId: string, chartId: string, changes: Partial<ChartConfig>): boolean {
    const layout = this.layouts.get(layoutId)
    if (!layout) return false

    const chartIndex = layout.charts.findIndex(chart => chart.id === chartId)
    if (chartIndex === -1) return false

    layout.charts[chartIndex] = { ...layout.charts[chartIndex], ...changes }
    this.layouts.set(layoutId, layout)
    return true
  }

  moveChart(layoutId: string, chartId: string, newPosition: Omit<ChartPosition, 'chartId'>): boolean {
    const layout = this.layouts.get(layoutId)
    if (!layout) return false

    const positionIndex = layout.layout.chartPositions.findIndex(pos => pos.chartId === chartId)
    if (positionIndex === -1) return false

    layout.layout.chartPositions[positionIndex] = {
      chartId,
      ...newPosition
    }

    this.layouts.set(layoutId, layout)
    return true
  }

  // Layout templates
  createTemplate(name: string, description: string, layout: ChartLayout, category: LayoutTemplate['category'] = 'custom'): string {
    const id = `template_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`
    
    const template: LayoutTemplate = {
      id,
      name,
      description,
      layout: {
        charts: layout.charts,
        layout: layout.layout,
        synchronization: layout.synchronization,
        theme: layout.theme
      },
      category,
      isBuiltIn: false
    }

    this.templates.set(id, template)
    return id
  }

  getTemplate(id: string): LayoutTemplate | null {
    return this.templates.get(id) || null
  }

  getAllTemplates(): LayoutTemplate[] {
    return Array.from(this.templates.values())
  }

  getTemplatesByCategory(category: LayoutTemplate['category']): LayoutTemplate[] {
    return Array.from(this.templates.values()).filter(template => template.category === category)
  }

  createLayoutFromTemplate(templateId: string, name: string): string | null {
    const template = this.templates.get(templateId)
    if (!template) return null

    return this.createLayout({
      name,
      ...template.layout
    })
  }

  // Synchronization management
  createSynchronizationGroup(chartIds: string[], syncSettings: SynchronizationGroup['syncSettings']): string {
    const id = `sync_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`
    
    const group: SynchronizationGroup = {
      id,
      chartIds: [...chartIds],
      syncSettings: { ...syncSettings }
    }

    this.synchronizationGroups.set(id, group)
    this.eventHandlers.onSynchronizationChange?.(Array.from(this.synchronizationGroups.values()))
    return id
  }

  addChartToSyncGroup(groupId: string, chartId: string): boolean {
    const group = this.synchronizationGroups.get(groupId)
    if (!group || group.chartIds.includes(chartId)) return false

    group.chartIds.push(chartId)
    this.synchronizationGroups.set(groupId, group)
    this.eventHandlers.onSynchronizationChange?.(Array.from(this.synchronizationGroups.values()))
    return true
  }

  removeChartFromSyncGroup(groupId: string, chartId: string): boolean {
    const group = this.synchronizationGroups.get(groupId)
    if (!group) return false

    group.chartIds = group.chartIds.filter(id => id !== chartId)
    
    if (group.chartIds.length === 0) {
      this.synchronizationGroups.delete(groupId)
    } else {
      this.synchronizationGroups.set(groupId, group)
    }

    this.eventHandlers.onSynchronizationChange?.(Array.from(this.synchronizationGroups.values()))
    return true
  }

  getSynchronizationGroups(): SynchronizationGroup[] {
    return Array.from(this.synchronizationGroups.values())
  }

  getSyncGroupForChart(chartId: string): SynchronizationGroup | null {
    for (const group of this.synchronizationGroups.values()) {
      if (group.chartIds.includes(chartId)) {
        return group
      }
    }
    return null
  }

  // Layout persistence
  saveLayoutsToStorage(): void {
    try {
      const layoutsData = {
        layouts: Array.from(this.layouts.entries()),
        templates: Array.from(this.templates.entries()).filter(([, template]) => !template.isBuiltIn),
        synchronizationGroups: Array.from(this.synchronizationGroups.entries()),
        activeLayoutId: this.activeLayoutId
      }
      localStorage.setItem('chartLayouts', JSON.stringify(layoutsData))
    } catch (error) {
      console.error('Failed to save layouts to storage:', error)
    }
  }

  loadLayoutsFromStorage(): void {
    try {
      const stored = localStorage.getItem('chartLayouts')
      if (!stored) return

      const data = JSON.parse(stored)
      
      // Load layouts
      if (data.layouts) {
        this.layouts = new Map(data.layouts)
      }

      // Load custom templates (don't override built-in templates)
      if (data.templates) {
        data.templates.forEach(([id, template]: [string, LayoutTemplate]) => {
          if (!template.isBuiltIn) {
            this.templates.set(id, template)
          }
        })
      }

      // Load synchronization groups
      if (data.synchronizationGroups) {
        this.synchronizationGroups = new Map(data.synchronizationGroups)
      }

      // Set active layout
      if (data.activeLayoutId && this.layouts.has(data.activeLayoutId)) {
        this.activeLayoutId = data.activeLayoutId
      }
    } catch (error) {
      console.error('Failed to load layouts from storage:', error)
    }
  }

  // Utility methods
  private findAvailablePosition(layout: ChartLayout): ChartPosition | null {
    const { rows, columns, chartPositions } = layout.layout
    
    // Create a grid to track occupied positions
    const occupied: boolean[][] = Array(rows).fill(null).map(() => Array(columns).fill(false))
    
    // Mark occupied positions
    chartPositions.forEach(pos => {
      for (let r = pos.row; r < pos.row + pos.rowSpan; r++) {
        for (let c = pos.column; c < pos.column + pos.columnSpan; c++) {
          if (r < rows && c < columns) {
            occupied[r][c] = true
          }
        }
      }
    })

    // Find first available position
    for (let r = 0; r < rows; r++) {
      for (let c = 0; c < columns; c++) {
        if (!occupied[r][c]) {
          return {
            chartId: '', // Will be set by caller
            row: r,
            column: c,
            rowSpan: 1,
            columnSpan: 1
          }
        }
      }
    }

    return null
  }

  private expandLayout(layout: ChartLayout): void {
    // Add a new row if no space available
    layout.layout.rows += 1
  }

  private initializeBuiltInTemplates(): void {
    // Single Chart Template
    this.templates.set('single', {
      id: 'single',
      name: 'Single Chart',
      description: 'Single full-screen chart',
      layout: {
        charts: [],
        layout: {
          rows: 1,
          columns: 1,
          chartPositions: []
        },
        synchronization: {
          crosshair: false,
          zoom: false,
          timeRange: false
        },
        theme: {
          id: 'default',
          name: 'Default',
          colors: {
            background: '#ffffff',
            grid: '#e1e1e1',
            crosshair: '#9B7DFF',
            upCandle: '#26a69a',
            downCandle: '#ef5350',
            volume: '#26a69a',
            text: '#333333'
          },
          fonts: {
            family: 'Arial, sans-serif',
            sizes: { small: 10, medium: 12, large: 14 }
          }
        }
      },
      category: 'trading',
      isBuiltIn: true
    })

    // Dual Chart Template
    this.templates.set('dual_horizontal', {
      id: 'dual_horizontal',
      name: 'Dual Charts (Horizontal)',
      description: 'Two charts side by side',
      layout: {
        charts: [],
        layout: {
          rows: 1,
          columns: 2,
          chartPositions: []
        },
        synchronization: {
          crosshair: true,
          zoom: true,
          timeRange: true
        },
        theme: {
          id: 'default',
          name: 'Default',
          colors: {
            background: '#ffffff',
            grid: '#e1e1e1',
            crosshair: '#9B7DFF',
            upCandle: '#26a69a',
            downCandle: '#ef5350',
            volume: '#26a69a',
            text: '#333333'
          },
          fonts: {
            family: 'Arial, sans-serif',
            sizes: { small: 10, medium: 12, large: 14 }
          }
        }
      },
      category: 'analysis',
      isBuiltIn: true
    })

    // Quad Chart Template
    this.templates.set('quad', {
      id: 'quad',
      name: 'Quad Charts',
      description: '2x2 grid of charts',
      layout: {
        charts: [],
        layout: {
          rows: 2,
          columns: 2,
          chartPositions: []
        },
        synchronization: {
          crosshair: true,
          zoom: false,
          timeRange: true
        },
        theme: {
          id: 'default',
          name: 'Default',
          colors: {
            background: '#ffffff',
            grid: '#e1e1e1',
            crosshair: '#9B7DFF',
            upCandle: '#26a69a',
            downCandle: '#ef5350',
            volume: '#26a69a',
            text: '#333333'
          },
          fonts: {
            family: 'Arial, sans-serif',
            sizes: { small: 10, medium: 12, large: 14 }
          }
        }
      },
      category: 'monitoring',
      isBuiltIn: true
    })
  }
}

// Export singleton instance
export const chartLayoutService = new ChartLayoutService()

export default chartLayoutService