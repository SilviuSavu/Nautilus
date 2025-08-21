/**
 * Chart Layout Service Tests  
 * Comprehensive tests for layout management, synchronization, and persistence
 */

import { describe, it, expect, beforeEach, vi, afterEach } from 'vitest'
import { chartLayoutService } from '../chartLayoutService'
import { ChartLayout, ChartConfig, Instrument } from '../../types/charting'

// Mock localStorage
const localStorageMock = (() => {
  let store: Record<string, string> = {}
  
  return {
    getItem: (key: string) => store[key] || null,
    setItem: (key: string, value: string) => {
      store[key] = value.toString()
    },
    removeItem: (key: string) => {
      delete store[key]
    },
    clear: () => {
      store = {}
    }
  }
})()

Object.defineProperty(window, 'localStorage', {
  value: localStorageMock
})

describe('ChartLayoutService', () => {
  let mockEventHandlers: any

  beforeEach(() => {
    // Reset the service state for each test
    chartLayoutService.getAllLayouts().forEach(layout => {
      chartLayoutService.deleteLayout(layout.id)
    })
    mockEventHandlers = {
      onLayoutChange: vi.fn(),
      onChartAdd: vi.fn(),
      onChartRemove: vi.fn(),
      onSynchronizationChange: vi.fn()
    }
    chartLayoutService.setEventHandlers(mockEventHandlers)
    localStorageMock.clear()
  })

  afterEach(() => {
    vi.clearAllMocks()
  })

  describe('Layout Management', () => {
    it('should create a new layout', () => {
      const layoutConfig = {
        name: 'Test Layout',
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
          name: 'dark',
          colors: {}
        }
      }

      const layoutId = chartLayoutService.createLayout(layoutConfig)
      
      expect(layoutId).toBeDefined()
      expect(layoutId).toMatch(/^layout_\d+_[a-z0-9]+$/)
      expect(mockEventHandlers.onLayoutChange).toHaveBeenCalledWith(
        expect.objectContaining({
          id: layoutId,
          name: 'Test Layout'
        })
      )
    })

    it('should retrieve layout by ID', () => {
      const layoutConfig = {
        name: 'Test Layout',
        charts: [],
        layout: { rows: 1, columns: 1, chartPositions: [] },
        synchronization: { crosshair: false, zoom: false, timeRange: false },
        theme: { name: 'light', colors: {} }
      }

      const layoutId = chartLayoutService.createLayout(layoutConfig)
      const retrieved = chartLayoutService.getLayout(layoutId)
      
      expect(retrieved).toBeDefined()
      expect(retrieved!.id).toBe(layoutId)
      expect(retrieved!.name).toBe('Test Layout')
    })

    it('should return null for non-existent layout', () => {
      const result = chartLayoutService.getLayout('non-existent-id')
      expect(result).toBe(null)
    })

    it('should get all layouts', () => {
      const layout1 = chartLayoutService.createLayout({
        name: 'Layout 1',
        charts: [],
        layout: { rows: 1, columns: 1, chartPositions: [] },
        synchronization: { crosshair: false, zoom: false, timeRange: false },
        theme: { name: 'light', colors: {} }
      })

      const layout2 = chartLayoutService.createLayout({
        name: 'Layout 2', 
        charts: [],
        layout: { rows: 2, columns: 2, chartPositions: [] },
        synchronization: { crosshair: true, zoom: true, timeRange: true },
        theme: { name: 'dark', colors: {} }
      })

      const allLayouts = chartLayoutService.getAllLayouts()
      expect(allLayouts).toHaveLength(2)
      expect(allLayouts.some(l => l.id === layout1)).toBe(true)
      expect(allLayouts.some(l => l.id === layout2)).toBe(true)
    })

    it('should update existing layout', () => {
      const layoutId = chartLayoutService.createLayout({
        name: 'Original Layout',
        charts: [],
        layout: { rows: 1, columns: 1, chartPositions: [] },
        synchronization: { crosshair: false, zoom: false, timeRange: false },
        theme: { name: 'light', colors: {} }
      })

      const updateResult = chartLayoutService.updateLayout(layoutId, {
        name: 'Updated Layout',
        layout: { rows: 2, columns: 3, chartPositions: [] }
      })

      expect(updateResult).toBe(true)
      expect(mockEventHandlers.onLayoutChange).toHaveBeenCalledTimes(2) // Create + Update
      
      const updated = chartLayoutService.getLayout(layoutId)
      expect(updated!.name).toBe('Updated Layout')
      expect(updated!.layout.rows).toBe(2)
      expect(updated!.layout.columns).toBe(3)
    })

    it('should fail to update non-existent layout', () => {
      const result = chartLayoutService.updateLayout('non-existent', { name: 'New Name' })
      expect(result).toBe(false)
    })

    it('should delete layout', () => {
      const layoutId = chartLayoutService.createLayout({
        name: 'To Delete',
        charts: [],
        layout: { rows: 1, columns: 1, chartPositions: [] },
        synchronization: { crosshair: false, zoom: false, timeRange: false },
        theme: { name: 'light', colors: {} }
      })

      const deleteResult = chartLayoutService.deleteLayout(layoutId)
      expect(deleteResult).toBe(true)
      
      const retrieved = chartLayoutService.getLayout(layoutId)
      expect(retrieved).toBe(null)
    })
  })

  describe('Active Layout Management', () => {
    it('should set and get active layout', () => {
      const layoutId = chartLayoutService.createLayout({
        name: 'Active Layout',
        charts: [],
        layout: { rows: 1, columns: 1, chartPositions: [] },
        synchronization: { crosshair: false, zoom: false, timeRange: false },
        theme: { name: 'light', colors: {} }
      })

      chartLayoutService.setActiveLayout(layoutId)
      const activeLayout = chartLayoutService.getActiveLayout()
      
      expect(activeLayout).toBeDefined()
      expect(activeLayout!.id).toBe(layoutId)
    })

    it('should return null when no active layout is set', () => {
      const activeLayout = chartLayoutService.getActiveLayout()
      expect(activeLayout).toBe(null)
    })

    it('should fail to set non-existent layout as active', () => {
      const result = chartLayoutService.setActiveLayout('non-existent')
      expect(result).toBe(false)
      
      const activeLayout = chartLayoutService.getActiveLayout()
      expect(activeLayout).toBe(null)
    })
  })

  describe('Chart Management', () => {
    let layoutId: string

    beforeEach(() => {
      layoutId = chartLayoutService.createLayout({
        name: 'Test Layout',
        charts: [],
        layout: { rows: 2, columns: 2, chartPositions: [] },
        synchronization: { crosshair: false, zoom: false, timeRange: false },
        theme: { name: 'light', colors: {} }
      })
    })

    it('should add chart to layout', () => {
      const chartConfig: ChartConfig = {
        id: 'chart1',
        type: 'candlestick',
        instrument: { symbol: 'AAPL', exchange: 'NASDAQ', name: 'Apple Inc.' },
        timeframe: '1H',
        indicators: [],
        position: { row: 0, col: 0, width: 1, height: 1 }
      }

      const result = chartLayoutService.addChartToLayout(layoutId, chartConfig)
      expect(result).toBe(true)
      expect(mockEventHandlers.onChartAdd).toHaveBeenCalledWith(chartConfig)
      
      const layout = chartLayoutService.getLayout(layoutId)
      expect(layout!.charts).toHaveLength(1)
      expect(layout!.charts[0]).toEqual(chartConfig)
    })

    it('should remove chart from layout', () => {
      const chartConfig: ChartConfig = {
        id: 'chart1',
        type: 'line',
        instrument: { symbol: 'MSFT', exchange: 'NASDAQ', name: 'Microsoft Corp.' },
        timeframe: '5M',
        indicators: [],
        position: { row: 0, col: 0, width: 1, height: 1 }
      }

      chartLayoutService.addChartToLayout(layoutId, chartConfig)
      const removeResult = chartLayoutService.removeChartFromLayout(layoutId, 'chart1')
      
      expect(removeResult).toBe(true)
      expect(mockEventHandlers.onChartRemove).toHaveBeenCalledWith('chart1')
      
      const layout = chartLayoutService.getLayout(layoutId)
      expect(layout!.charts).toHaveLength(0)
    })

    it('should update chart configuration', () => {
      const chartConfig: ChartConfig = {
        id: 'chart1',
        type: 'candlestick',
        instrument: { symbol: 'GOOGL', exchange: 'NASDAQ', name: 'Alphabet Inc.' },
        timeframe: '1D',
        indicators: [],
        position: { row: 0, col: 0, width: 1, height: 1 }
      }

      chartLayoutService.addChartToLayout(layoutId, chartConfig)
      
      const updateResult = chartLayoutService.updateChartInLayout(layoutId, 'chart1', {
        type: 'line',
        timeframe: '4H'
      })

      expect(updateResult).toBe(true)
      
      const layout = chartLayoutService.getLayout(layoutId)
      const updatedChart = layout!.charts.find(c => c.id === 'chart1')
      expect(updatedChart!.type).toBe('line')
      expect(updatedChart!.timeframe).toBe('4H')
      expect(updatedChart!.instrument.symbol).toBe('GOOGL') // Unchanged
    })
  })

  describe('Synchronization Groups', () => {
    it('should create synchronization group', () => {
      const groupId = chartLayoutService.createSynchronizationGroup(['chart1', 'chart2'], {
        crosshair: true,
        zoom: false,
        timeRange: true,
        instrument: false
      })

      expect(groupId).toBeDefined()
      expect(groupId).toMatch(/^sync_\d+_[a-z0-9]+$/)
      expect(mockEventHandlers.onSynchronizationChange).toHaveBeenCalled()
    })

    it('should get synchronization groups', () => {
      const group1Id = chartLayoutService.createSynchronizationGroup(['chart1', 'chart2'], {
        crosshair: true,
        zoom: true,
        timeRange: false,
        instrument: false
      })

      const group2Id = chartLayoutService.createSynchronizationGroup(['chart3', 'chart4'], {
        crosshair: false,
        zoom: false,
        timeRange: true,
        instrument: true
      })

      const groups = chartLayoutService.getSynchronizationGroups()
      expect(groups).toHaveLength(2)
      expect(groups.some(g => g.id === group1Id)).toBe(true)
      expect(groups.some(g => g.id === group2Id)).toBe(true)
    })

    it('should remove synchronization group', () => {
      const groupId = chartLayoutService.createSynchronizationGroup(['chart1', 'chart2'], {
        crosshair: true,
        zoom: false,
        timeRange: false,
        instrument: false
      })

      const removeResult = chartLayoutService.removeSynchronizationGroup(groupId)
      expect(removeResult).toBe(true)
      
      const groups = chartLayoutService.getSynchronizationGroups()
      expect(groups.some(g => g.id === groupId)).toBe(false)
    })

    it('should update synchronization group', () => {
      const groupId = chartLayoutService.createSynchronizationGroup(['chart1', 'chart2'], {
        crosshair: true,
        zoom: false,
        timeRange: false,
        instrument: false
      })

      const updateResult = chartLayoutService.updateSynchronizationGroup(groupId, {
        chartIds: ['chart1', 'chart2', 'chart3'],
        syncSettings: {
          crosshair: false,
          zoom: true,
          timeRange: true,
          instrument: false
        }
      })

      expect(updateResult).toBe(true)
      
      const groups = chartLayoutService.getSynchronizationGroups()
      const updatedGroup = groups.find(g => g.id === groupId)
      expect(updatedGroup!.chartIds).toContain('chart3')
      expect(updatedGroup!.syncSettings.zoom).toBe(true)
    })
  })

  describe('Template Management', () => {
    it('should get built-in templates', () => {
      const templates = chartLayoutService.getAllTemplates()
      expect(templates.length).toBeGreaterThan(0)
      
      // Should have at least standard templates
      const templateNames = templates.map(t => t.name)
      expect(templateNames).toContain('Single Chart')
      expect(templateNames).toContain('Four Chart Grid')
    })

    it('should create layout from template', () => {
      const templates = chartLayoutService.getAllTemplates()
      const singleChartTemplate = templates.find(t => t.name === 'Single Chart')
      
      expect(singleChartTemplate).toBeDefined()
      
      const layoutId = chartLayoutService.createLayoutFromTemplate(singleChartTemplate!.id, {
        name: 'My Single Chart Layout'
      })

      expect(layoutId).toBeDefined()
      
      const createdLayout = chartLayoutService.getLayout(layoutId)
      expect(createdLayout!.name).toBe('My Single Chart Layout')
      expect(createdLayout!.layout.rows).toBe(singleChartTemplate!.layout.layout.rows)
    })

    it('should save layout as template', () => {
      const layoutId = chartLayoutService.createLayout({
        name: 'Custom Layout',
        charts: [],
        layout: { rows: 3, columns: 2, chartPositions: [] },
        synchronization: { crosshair: true, zoom: true, timeRange: false },
        theme: { name: 'dark', colors: {} }
      })

      const templateId = chartLayoutService.saveLayoutAsTemplate(layoutId, {
        name: 'My Custom Template',
        description: 'A custom 3x2 layout',
        category: 'custom'
      })

      expect(templateId).toBeDefined()
      
      const templates = chartLayoutService.getAllTemplates()
      const customTemplate = templates.find(t => t.id === templateId)
      expect(customTemplate).toBeDefined()
      expect(customTemplate!.name).toBe('My Custom Template')
      expect(customTemplate!.category).toBe('custom')
      expect(customTemplate!.isBuiltIn).toBe(false)
    })
  })

  describe('Persistence', () => {
    it('should save layouts to storage', () => {
      const layoutId = chartLayoutService.createLayout({
        name: 'Persistent Layout',
        charts: [],
        layout: { rows: 1, columns: 1, chartPositions: [] },
        synchronization: { crosshair: false, zoom: false, timeRange: false },
        theme: { name: 'light', colors: {} }
      })

      chartLayoutService.saveLayoutsToStorage()
      
      const saved = localStorage.getItem('nautilus_chart_layouts')
      expect(saved).toBeDefined()
      
      const parsedLayouts = JSON.parse(saved!)
      expect(Array.isArray(parsedLayouts)).toBe(true)
      expect(parsedLayouts.some((l: any) => l.id === layoutId)).toBe(true)
    })

    it('should load layouts from storage', () => {
      // Manually add layout to storage
      const testLayout = {
        id: 'test-layout-id',
        name: 'Loaded Layout',
        charts: [],
        layout: { rows: 2, columns: 1, chartPositions: [] },
        synchronization: { crosshair: true, zoom: false, timeRange: false },
        theme: { name: 'dark', colors: {} }
      }

      localStorage.setItem('nautilus_chart_layouts', JSON.stringify([testLayout]))
      
      chartLayoutService.loadLayoutsFromStorage()
      
      const loadedLayout = chartLayoutService.getLayout('test-layout-id')
      expect(loadedLayout).toBeDefined()
      expect(loadedLayout!.name).toBe('Loaded Layout')
      expect(loadedLayout!.layout.rows).toBe(2)
    })

    it('should handle corrupted storage data gracefully', () => {
      localStorage.setItem('nautilus_chart_layouts', 'invalid-json')
      
      // Should not throw error
      expect(() => chartLayoutService.loadLayoutsFromStorage()).not.toThrow()
      
      // Should have no layouts loaded
      const layouts = chartLayoutService.getAllLayouts()
      expect(layouts).toHaveLength(0)
    })
  })

  describe('Event Handling', () => {
    it('should trigger events when layouts change', () => {
      const layoutConfig = {
        name: 'Event Test Layout',
        charts: [],
        layout: { rows: 1, columns: 1, chartPositions: [] },
        synchronization: { crosshair: false, zoom: false, timeRange: false },
        theme: { name: 'light', colors: {} }
      }

      chartLayoutService.createLayout(layoutConfig)
      
      expect(mockEventHandlers.onLayoutChange).toHaveBeenCalledWith(
        expect.objectContaining({ name: 'Event Test Layout' })
      )
    })

    it('should allow updating event handlers', () => {
      const newHandlers = {
        onLayoutChange: vi.fn(),
        onChartAdd: vi.fn()
      }

      chartLayoutService.setEventHandlers(newHandlers)
      
      chartLayoutService.createLayout({
        name: 'Handler Test',
        charts: [],
        layout: { rows: 1, columns: 1, chartPositions: [] },
        synchronization: { crosshair: false, zoom: false, timeRange: false },
        theme: { name: 'light', colors: {} }
      })

      expect(newHandlers.onLayoutChange).toHaveBeenCalled()
      expect(mockEventHandlers.onLayoutChange).not.toHaveBeenCalled()
    })
  })

  describe('Validation', () => {
    it('should validate layout structure', () => {
      expect(() => {
        chartLayoutService.createLayout({
          name: '',
          charts: [],
          layout: { rows: 0, columns: 0, chartPositions: [] },
          synchronization: { crosshair: false, zoom: false, timeRange: false },
          theme: { name: 'light', colors: {} }
        })
      }).not.toThrow()
    })

    it('should handle edge cases in chart positioning', () => {
      const layoutId = chartLayoutService.createLayout({
        name: 'Edge Case Layout',
        charts: [],
        layout: { rows: 2, columns: 2, chartPositions: [] },
        synchronization: { crosshair: false, zoom: false, timeRange: false },
        theme: { name: 'light', colors: {} }
      })

      const chartConfig: ChartConfig = {
        id: 'edge-chart',
        type: 'candlestick',
        instrument: { symbol: 'TEST', exchange: 'TEST', name: 'Test Symbol' },
        timeframe: '1M',
        indicators: [],
        position: { row: 10, col: 10, width: 5, height: 5 } // Invalid position
      }

      // Should handle gracefully
      expect(() => chartLayoutService.addChartToLayout(layoutId, chartConfig)).not.toThrow()
    })
  })
})