/**
 * Drawing Service
 * Manages chart drawings, annotations, and interactive drawing tools
 */

import { DrawingObject, Point, DrawingTool } from '../types/charting'

export interface DrawingState {
  isDrawing: boolean
  currentTool: string | null
  activeDrawing: Partial<DrawingObject> | null
  selectedDrawings: string[]
}

export interface DrawingEventHandlers {
  onDrawingStart?: (tool: DrawingTool) => void
  onDrawingUpdate?: (drawing: Partial<DrawingObject>) => void
  onDrawingComplete?: (drawing: DrawingObject) => void
  onDrawingSelect?: (drawingIds: string[]) => void
  onDrawingDelete?: (drawingIds: string[]) => void
}

class DrawingService {
  private drawings: Map<string, DrawingObject> = new Map()
  private drawingTools: Map<string, DrawingTool> = new Map()
  private state: DrawingState = {
    isDrawing: false,
    currentTool: null,
    activeDrawing: null,
    selectedDrawings: []
  }
  private eventHandlers: DrawingEventHandlers = {}

  constructor() {
    this.initializeDrawingTools()
  }

  // Event handling
  setEventHandlers(handlers: DrawingEventHandlers) {
    this.eventHandlers = { ...this.eventHandlers, ...handlers }
  }

  // Drawing tools management
  private initializeDrawingTools() {
    const tools: DrawingTool[] = [
      {
        id: 'trend_line',
        name: 'Trend Line',
        icon: '📈',
        type: 'trend_line',
        category: 'lines',
        cursor: 'crosshair'
      },
      {
        id: 'horizontal_line',
        name: 'Horizontal Line',
        icon: '➖',
        type: 'trend_line',
        category: 'lines',
        cursor: 'crosshair'
      },
      {
        id: 'vertical_line',
        name: 'Vertical Line',
        icon: '|',
        type: 'trend_line',
        category: 'lines',
        cursor: 'crosshair'
      },
      {
        id: 'rectangle',
        name: 'Rectangle',
        icon: '⬛',
        type: 'rectangle',
        category: 'shapes',
        cursor: 'crosshair'
      },
      {
        id: 'circle',
        name: 'Circle',
        icon: '⭕',
        type: 'circle',
        category: 'shapes',
        cursor: 'crosshair'
      },
      {
        id: 'channel',
        name: 'Channel',
        icon: '📊',
        type: 'channel',
        category: 'lines',
        cursor: 'crosshair'
      },
      {
        id: 'fibonacci',
        name: 'Fibonacci',
        icon: '🔢',
        type: 'fibonacci',
        category: 'fibonacci',
        cursor: 'crosshair'
      },
      {
        id: 'text',
        name: 'Text',
        icon: '📝',
        type: 'text',
        category: 'text',
        cursor: 'text'
      },
      {
        id: 'arrow',
        name: 'Arrow',
        icon: '➡️',
        type: 'arrow',
        category: 'text',
        cursor: 'crosshair'
      }
    ]

    tools.forEach(tool => {
      this.drawingTools.set(tool.id, tool)
    })
  }

  getDrawingTools(category?: DrawingTool['category']): DrawingTool[] {
    const tools = Array.from(this.drawingTools.values())
    return category ? tools.filter(tool => tool.category === category) : tools
  }

  getDrawingTool(id: string): DrawingTool | null {
    return this.drawingTools.get(id) || null
  }

  // Drawing state management
  getState(): DrawingState {
    return { ...this.state }
  }

  setCurrentTool(toolId: string | null): boolean {
    const tool = toolId ? this.drawingTools.get(toolId) : null
    
    if (toolId && !tool) return false

    this.state.currentTool = toolId
    this.state.isDrawing = false
    this.state.activeDrawing = null

    if (tool) {
      this.eventHandlers.onDrawingStart?.(tool)
    }

    return true
  }

  startDrawing(point: Point, chartId: string, timeframe: string): boolean {
    if (!this.state.currentTool) return false

    const tool = this.drawingTools.get(this.state.currentTool)
    if (!tool) return false

    this.state.isDrawing = true
    this.state.activeDrawing = {
      type: tool.type,
      coordinates: [point],
      style: {
        color: '#1890ff',
        lineWidth: 2,
        fillColor: 'rgba(24, 144, 255, 0.1)',
        transparency: 0.1
      },
      anchored: true,
      persistent: true,
      chartId,
      timeframe
    }

    this.eventHandlers.onDrawingUpdate?.(this.state.activeDrawing)
    return true
  }

  updateDrawing(point: Point): boolean {
    if (!this.state.isDrawing || !this.state.activeDrawing) return false

    const drawing = this.state.activeDrawing
    
    switch (drawing.type) {
      case 'trend_line':
        drawing.coordinates = [drawing.coordinates![0], point]
        break
        
      case 'rectangle':
        drawing.coordinates = [drawing.coordinates![0], point]
        break
        
      case 'circle':
        const center = drawing.coordinates![0]
        const radius = Math.sqrt(
          Math.pow(point.x - center.x, 2) + Math.pow(point.y - center.y, 2)
        )
        drawing.coordinates = [center, { ...point, x: center.x + radius, y: center.y }]
        break
        
      case 'channel':
        if (drawing.coordinates!.length === 1) {
          drawing.coordinates = [drawing.coordinates![0], point]
        } else if (drawing.coordinates!.length === 2) {
          drawing.coordinates = [...drawing.coordinates!, point]
        }
        break
        
      default:
        drawing.coordinates!.push(point)
    }

    this.eventHandlers.onDrawingUpdate?.(drawing)
    return true
  }

  finishDrawing(): DrawingObject | null {
    if (!this.state.isDrawing || !this.state.activeDrawing) return null

    const drawingId = `drawing_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`
    const completedDrawing: DrawingObject = {
      id: drawingId,
      ...this.state.activeDrawing
    } as DrawingObject

    // Validate drawing has minimum required points
    if (!this.validateDrawing(completedDrawing)) {
      this.cancelDrawing()
      return null
    }

    this.drawings.set(drawingId, completedDrawing)
    
    this.state.isDrawing = false
    this.state.activeDrawing = null
    
    this.eventHandlers.onDrawingComplete?.(completedDrawing)
    return completedDrawing
  }

  cancelDrawing(): void {
    this.state.isDrawing = false
    this.state.activeDrawing = null
  }

  // Drawing management
  getDrawing(id: string): DrawingObject | null {
    return this.drawings.get(id) || null
  }

  getAllDrawings(chartId?: string, timeframe?: string): DrawingObject[] {
    let drawings = Array.from(this.drawings.values())
    
    if (chartId) {
      drawings = drawings.filter(d => d.chartId === chartId)
    }
    
    if (timeframe) {
      drawings = drawings.filter(d => d.timeframe === timeframe)
    }
    
    return drawings
  }

  addDrawing(drawing: Omit<DrawingObject, 'id'>): string {
    const id = `drawing_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`
    const newDrawing: DrawingObject = {
      id,
      ...drawing
    }
    
    this.drawings.set(id, newDrawing)
    this.eventHandlers.onDrawingComplete?.(newDrawing)
    return id
  }

  updateDrawing(id: string, changes: Partial<DrawingObject>): boolean {
    const drawing = this.drawings.get(id)
    if (!drawing) return false

    const updatedDrawing = { ...drawing, ...changes, id } // Preserve ID
    this.drawings.set(id, updatedDrawing)
    return true
  }

  deleteDrawing(id: string): boolean {
    const deleted = this.drawings.delete(id);
    if (deleted) {
      this.state.selectedDrawings = this.state.selectedDrawings.filter(sid => sid !== id);
      this.eventHandlers.onDrawingDelete?.([id]);
    }
    return deleted;
  }

  deleteDrawings(ids: string[]): number {
    let deletedCount = 0
    
    ids.forEach(id => {
      if (this.drawings.delete(id)) {
        deletedCount++
      }
    })
    
    if (deletedCount > 0) {
      this.state.selectedDrawings = this.state.selectedDrawings.filter(sid => !ids.includes(sid))
      this.eventHandlers.onDrawingDelete?.(ids)
    }
    
    return deletedCount
  }

  // Selection management
  selectDrawing(id: string, multiSelect = false): boolean {
    if (!this.drawings.has(id)) return false

    if (multiSelect) {
      if (!this.state.selectedDrawings.includes(id)) {
        this.state.selectedDrawings.push(id)
      }
    } else {
      this.state.selectedDrawings = [id]
    }

    this.eventHandlers.onDrawingSelect?.(this.state.selectedDrawings)
    return true
  }

  deselectDrawing(id: string): boolean {
    const index = this.state.selectedDrawings.indexOf(id)
    if (index === -1) return false

    this.state.selectedDrawings.splice(index, 1)
    this.eventHandlers.onDrawingSelect?.(this.state.selectedDrawings)
    return true
  }

  selectAll(chartId?: string): void {
    const drawings = this.getAllDrawings(chartId)
    this.state.selectedDrawings = drawings.map(d => d.id)
    this.eventHandlers.onDrawingSelect?.(this.state.selectedDrawings)
  }

  clearSelection(): void {
    this.state.selectedDrawings = []
    this.eventHandlers.onDrawingSelect?.(this.state.selectedDrawings)
  }

  getSelectedDrawings(): DrawingObject[] {
    return this.state.selectedDrawings
      .map(id => this.drawings.get(id))
      .filter((drawing): drawing is DrawingObject => drawing !== undefined)
  }

  // Drawing validation
  private validateDrawing(drawing: DrawingObject): boolean {
    if (!drawing.coordinates || drawing.coordinates.length === 0) {
      return false
    }

    switch (drawing.type) {
      case 'trend_line':
        return drawing.coordinates.length >= 2
      case 'rectangle':
        return drawing.coordinates.length >= 2
      case 'circle':
        return drawing.coordinates.length >= 2
      case 'text':
        return drawing.coordinates.length >= 1 && !!drawing.text
      case 'channel':
        return drawing.coordinates.length >= 3
      case 'fibonacci':
        return drawing.coordinates.length >= 2
      default:
        return drawing.coordinates.length >= 1
    }
  }

  // Coordinate transformation utilities
  screenToPrice(screenPoint: Point, chart: any): Point {
    // This would convert screen coordinates to price/time coordinates
    // Implementation depends on the chart library being used
    const timeScale = chart.timeScale()
    const priceScale = chart.priceScale('right')
    
    return {
      x: screenPoint.x,
      y: screenPoint.y,
      time: timeScale.coordinateToTime(screenPoint.x),
      price: priceScale.coordinateToPrice(screenPoint.y)
    }
  }

  priceToScreen(pricePoint: Point, chart: any): Point {
    // This would convert price/time coordinates to screen coordinates
    const timeScale = chart.timeScale()
    const priceScale = chart.priceScale('right')
    
    return {
      x: pricePoint.time ? timeScale.timeToCoordinate(pricePoint.time) : pricePoint.x,
      y: pricePoint.price ? priceScale.priceToCoordinate(pricePoint.price) : pricePoint.y
    }
  }

  // Drawing templates
  createTemplate(name: string, drawings: DrawingObject[]): void {
    // Store drawing template for reuse
    localStorage.setItem(`drawing_template_${name}`, JSON.stringify(drawings))
  }

  getTemplates(): string[] {
    const templates: string[] = []
    for (let i = 0; i < localStorage.length; i++) {
      const key = localStorage.key(i)
      if (key?.startsWith('drawing_template_')) {
        templates.push(key.replace('drawing_template_', ''))
      }
    }
    return templates
  }

  loadTemplate(name: string): DrawingObject[] {
    const stored = localStorage.getItem(`drawing_template_${name}`)
    return stored ? JSON.parse(stored) : []
  }

  // Persistence
  saveDrawings(): void {
    const drawingsData = Array.from(this.drawings.entries())
    localStorage.setItem('chart_drawings', JSON.stringify(drawingsData))
  }

  loadDrawings(): void {
    try {
      const stored = localStorage.getItem('chart_drawings')
      if (stored) {
        const drawingsData = JSON.parse(stored)
        this.drawings = new Map(drawingsData)
      }
    } catch (error) {
      console.error('Failed to load drawings:', error)
    }
  }

  // Export/Import
  exportDrawings(format: 'json' | 'csv' = 'json'): string {
    const drawings = Array.from(this.drawings.values())
    
    if (format === 'json') {
      return JSON.stringify(drawings, null, 2)
    } else {
      // CSV format for basic drawing data
      const headers = ['id', 'type', 'chartId', 'timeframe', 'coordinates', 'style']
      const rows = drawings.map(d => [
        d.id,
        d.type,
        d.chartId,
        d.timeframe,
        JSON.stringify(d.coordinates),
        JSON.stringify(d.style)
      ])
      
      return [headers, ...rows].map(row => row.join(',')).join('\n')
    }
  }

  importDrawings(data: string, format: 'json' | 'csv' = 'json'): number {
    try {
      let drawings: DrawingObject[]
      
      if (format === 'json') {
        drawings = JSON.parse(data)
      } else {
        // Simple CSV parsing (would need more robust implementation)
        const lines = data.split('\n')
        const headers = lines[0].split(',')
        drawings = lines.slice(1).map(line => {
          const values = line.split(',')
          return {
            id: values[0],
            type: values[1] as DrawingObject['type'],
            chartId: values[2],
            timeframe: values[3],
            coordinates: JSON.parse(values[4]),
            style: JSON.parse(values[5]),
            anchored: true,
            persistent: true
          }
        })
      }
      
      let importedCount = 0
      drawings.forEach(drawing => {
        if (this.validateDrawing(drawing)) {
          this.drawings.set(drawing.id, drawing)
          importedCount++
        }
      })
      
      return importedCount
    } catch (error) {
      console.error('Failed to import drawings:', error)
      return 0
    }
  }
}

// Export singleton instance
export const drawingService = new DrawingService()

export default drawingService