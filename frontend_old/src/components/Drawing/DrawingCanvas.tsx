/**
 * Drawing Canvas Component
 * Handles interactive drawing on chart overlays
 */

import React, { useRef, useEffect, useState, useCallback } from 'react'
import { drawingService } from '../../services/drawingService'
import { DrawingObject, Point } from '../../types/charting'

interface DrawingCanvasProps {
  width: number
  height: number
  chart?: any // Chart API instance
  onDrawingComplete?: (drawing: DrawingObject) => void
  onDrawingSelect?: (drawingIds: string[]) => void
  drawings?: DrawingObject[]
  className?: string
}

export const DrawingCanvas: React.FC<DrawingCanvasProps> = ({
  width,
  height,
  chart,
  onDrawingComplete,
  onDrawingSelect,
  drawings = [],
  className
}) => {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const overlayRef = useRef<HTMLDivElement>(null)
  const [isDrawing, setIsDrawing] = useState(false)
  const [dragStart, setDragStart] = useState<Point | null>(null)
  const [hoveredDrawing, setHoveredDrawing] = useState<string | null>(null)

  // Drawing state
  const drawingState = drawingService.getState()

  useEffect(() => {
    if (canvasRef.current) {
      const canvas = canvasRef.current
      canvas.width = width
      canvas.height = height
      redrawCanvas()
    }
  }, [width, height, drawings, drawingState.activeDrawing])

  const redrawCanvas = useCallback(() => {
    const canvas = canvasRef.current
    if (!canvas) return

    const ctx = canvas.getContext('2d')
    if (!ctx) return

    // Clear canvas
    ctx.clearRect(0, 0, width, height)

    // Draw completed drawings
    drawings.forEach(drawing => {
      drawDrawing(ctx, drawing, drawing.id === hoveredDrawing)
    })

    // Draw active drawing (if any)
    if (drawingState.activeDrawing) {
      drawDrawing(ctx, drawingState.activeDrawing as DrawingObject, false, true)
    }
  }, [drawings, drawingState.activeDrawing, hoveredDrawing, width, height])

  const drawDrawing = (
    ctx: CanvasRenderingContext2D, 
    drawing: Partial<DrawingObject>, 
    isHovered: boolean = false,
    isActive: boolean = false
  ) => {
    if (!drawing.coordinates || drawing.coordinates.length === 0) return

    const style = drawing.style || {
      color: '#1890ff',
      lineWidth: 2,
      fillColor: 'rgba(24, 144, 255, 0.1)',
      transparency: 0.1
    }

    // Adjust style for hover/active states
    const lineColor = isHovered ? '#ff4d4f' : isActive ? '#52c41a' : style.color
    const lineWidth = (style.lineWidth || 2) * (isHovered ? 1.5 : 1)

    ctx.strokeStyle = lineColor
    ctx.lineWidth = lineWidth
    ctx.fillStyle = style.fillColor || 'rgba(24, 144, 255, 0.1)'
    ctx.globalAlpha = 1 - (style.transparency || 0)

    switch (drawing.type) {
      case 'trend_line':
        drawTrendLine(ctx, drawing.coordinates)
        break
      case 'rectangle':
        drawRectangle(ctx, drawing.coordinates)
        break
      case 'circle':
        drawCircle(ctx, drawing.coordinates)
        break
      case 'channel':
        drawChannel(ctx, drawing.coordinates)
        break
      case 'fibonacci':
        drawFibonacci(ctx, drawing.coordinates)
        break
      case 'text':
        drawText(ctx, drawing.coordinates, drawing.text || '')
        break
      case 'arrow':
        drawArrow(ctx, drawing.coordinates)
        break
    }

    ctx.globalAlpha = 1
  }

  const drawTrendLine = (ctx: CanvasRenderingContext2D, coordinates: Point[]) => {
    if (coordinates.length < 2) return

    ctx.beginPath()
    ctx.moveTo(coordinates[0].x, coordinates[0].y)
    ctx.lineTo(coordinates[1].x, coordinates[1].y)
    ctx.stroke()

    // Draw control points
    coordinates.forEach(point => {
      drawControlPoint(ctx, point)
    })
  }

  const drawRectangle = (ctx: CanvasRenderingContext2D, coordinates: Point[]) => {
    if (coordinates.length < 2) return

    const [start, end] = coordinates
    const width = end.x - start.x
    const height = end.y - start.y

    ctx.beginPath()
    ctx.rect(start.x, start.y, width, height)
    ctx.fill()
    ctx.stroke()

    // Draw control points
    coordinates.forEach(point => {
      drawControlPoint(ctx, point)
    })
  }

  const drawCircle = (ctx: CanvasRenderingContext2D, coordinates: Point[]) => {
    if (coordinates.length < 2) return

    const [center, edge] = coordinates
    const radius = Math.sqrt(
      Math.pow(edge.x - center.x, 2) + Math.pow(edge.y - center.y, 2)
    )

    ctx.beginPath()
    ctx.arc(center.x, center.y, radius, 0, 2 * Math.PI)
    ctx.fill()
    ctx.stroke()

    // Draw control points
    drawControlPoint(ctx, center)
    drawControlPoint(ctx, edge)
  }

  const drawChannel = (ctx: CanvasRenderingContext2D, coordinates: Point[]) => {
    if (coordinates.length < 3) return

    const [start, end, parallel] = coordinates

    // Draw main trend line
    ctx.beginPath()
    ctx.moveTo(start.x, start.y)
    ctx.lineTo(end.x, end.y)
    ctx.stroke()

    // Draw parallel line
    const dx = end.x - start.x
    const dy = end.y - start.y
    const parallelEnd = {
      x: parallel.x + dx,
      y: parallel.y + dy
    }

    ctx.beginPath()
    ctx.moveTo(parallel.x, parallel.y)
    ctx.lineTo(parallelEnd.x, parallelEnd.y)
    ctx.stroke()

    // Draw control points
    coordinates.forEach(point => {
      drawControlPoint(ctx, point)
    })
  }

  const drawFibonacci = (ctx: CanvasRenderingContext2D, coordinates: Point[]) => {
    if (coordinates.length < 2) return

    const [start, end] = coordinates
    const fibLevels = [0, 0.236, 0.382, 0.5, 0.618, 0.786, 1]
    
    const dx = end.x - start.x
    const dy = end.y - start.y

    fibLevels.forEach((level, index) => {
      const x = start.x + dx * level
      const y = start.y + dy * level

      ctx.beginPath()
      ctx.moveTo(start.x, y)
      ctx.lineTo(end.x, y)
      ctx.stroke()

      // Draw level label
      ctx.fillStyle = ctx.strokeStyle
      ctx.font = '10px Arial'
      ctx.fillText(`${(level * 100).toFixed(1)}%`, end.x + 5, y + 3)
    })

    // Draw control points
    coordinates.forEach(point => {
      drawControlPoint(ctx, point)
    })
  }

  const drawText = (ctx: CanvasRenderingContext2D, coordinates: Point[], text: string) => {
    if (coordinates.length === 0 || !text) return

    const point = coordinates[0]
    
    ctx.fillStyle = ctx.strokeStyle
    ctx.font = '12px Arial'
    ctx.fillText(text, point.x, point.y)

    drawControlPoint(ctx, point)
  }

  const drawArrow = (ctx: CanvasRenderingContext2D, coordinates: Point[]) => {
    if (coordinates.length < 2) return

    const [start, end] = coordinates
    const headLength = 15
    const angle = Math.atan2(end.y - start.y, end.x - start.x)

    // Draw line
    ctx.beginPath()
    ctx.moveTo(start.x, start.y)
    ctx.lineTo(end.x, end.y)
    ctx.stroke()

    // Draw arrowhead
    ctx.beginPath()
    ctx.moveTo(end.x, end.y)
    ctx.lineTo(
      end.x - headLength * Math.cos(angle - Math.PI / 6),
      end.y - headLength * Math.sin(angle - Math.PI / 6)
    )
    ctx.moveTo(end.x, end.y)
    ctx.lineTo(
      end.x - headLength * Math.cos(angle + Math.PI / 6),
      end.y - headLength * Math.sin(angle + Math.PI / 6)
    )
    ctx.stroke()

    // Draw control points
    coordinates.forEach(point => {
      drawControlPoint(ctx, point)
    })
  }

  const drawControlPoint = (ctx: CanvasRenderingContext2D, point: Point) => {
    const size = 4
    ctx.fillStyle = '#ffffff'
    ctx.strokeStyle = '#1890ff'
    ctx.lineWidth = 1

    ctx.beginPath()
    ctx.rect(point.x - size/2, point.y - size/2, size, size)
    ctx.fill()
    ctx.stroke()
  }

  const getMousePoint = (event: React.MouseEvent): Point => {
    const canvas = canvasRef.current
    if (!canvas) return { x: 0, y: 0 }

    const rect = canvas.getBoundingClientRect()
    return {
      x: event.clientX - rect.left,
      y: event.clientY - rect.top
    }
  }

  const findDrawingAtPoint = (point: Point): string | null => {
    // Simple hit testing - could be improved with more sophisticated algorithms
    for (const drawing of drawings) {
      if (isPointInDrawing(point, drawing)) {
        return drawing.id
      }
    }
    return null
  }

  const isPointInDrawing = (point: Point, drawing: DrawingObject): boolean => {
    if (!drawing.coordinates) return false

    const tolerance = 5 // pixels

    switch (drawing.type) {
      case 'trend_line':
        if (drawing.coordinates.length < 2) return false
        return isPointOnLine(point, drawing.coordinates[0], drawing.coordinates[1], tolerance)
      
      case 'rectangle':
        if (drawing.coordinates.length < 2) return false
        return isPointInRectangle(point, drawing.coordinates[0], drawing.coordinates[1])
      
      case 'circle':
        if (drawing.coordinates.length < 2) return false
        const center = drawing.coordinates[0]
        const edge = drawing.coordinates[1]
        const radius = Math.sqrt(
          Math.pow(edge.x - center.x, 2) + Math.pow(edge.y - center.y, 2)
        )
        const distance = Math.sqrt(
          Math.pow(point.x - center.x, 2) + Math.pow(point.y - center.y, 2)
        )
        return Math.abs(distance - radius) <= tolerance
      
      case 'text':
        if (drawing.coordinates.length === 0) return false
        const textPoint = drawing.coordinates[0]
        return Math.abs(point.x - textPoint.x) <= 50 && Math.abs(point.y - textPoint.y) <= 20
      
      default:
        return false
    }
  }

  const isPointOnLine = (point: Point, start: Point, end: Point, tolerance: number): boolean => {
    const distance = distanceFromPointToLine(point, start, end)
    return distance <= tolerance
  }

  const distanceFromPointToLine = (point: Point, start: Point, end: Point): number => {
    const A = point.x - start.x
    const B = point.y - start.y
    const C = end.x - start.x
    const D = end.y - start.y

    const dot = A * C + B * D
    const lenSq = C * C + D * D
    
    if (lenSq === 0) return Math.sqrt(A * A + B * B)

    const param = dot / lenSq
    let xx, yy

    if (param < 0) {
      xx = start.x
      yy = start.y
    } else if (param > 1) {
      xx = end.x
      yy = end.y
    } else {
      xx = start.x + param * C
      yy = start.y + param * D
    }

    const dx = point.x - xx
    const dy = point.y - yy
    return Math.sqrt(dx * dx + dy * dy)
  }

  const isPointInRectangle = (point: Point, start: Point, end: Point): boolean => {
    const minX = Math.min(start.x, end.x)
    const maxX = Math.max(start.x, end.x)
    const minY = Math.min(start.y, end.y)
    const maxY = Math.max(start.y, end.y)

    return point.x >= minX && point.x <= maxX && point.y >= minY && point.y <= maxY
  }

  const handleMouseDown = (event: React.MouseEvent) => {
    const point = getMousePoint(event)
    setDragStart(point)

    if (drawingState.currentTool) {
      // Start new drawing
      setIsDrawing(true)
      drawingService.startDrawing(point, 'chart', '1d') // TODO: Get actual chart and timeframe
    } else {
      // Selection mode
      const drawingId = findDrawingAtPoint(point)
      if (drawingId) {
        const multiSelect = event.ctrlKey || event.metaKey
        drawingService.selectDrawing(drawingId, multiSelect)
        onDrawingSelect?.(drawingService.getState().selectedDrawings)
      } else {
        drawingService.clearSelection()
        onDrawingSelect?.([])
      }
    }
  }

  const handleMouseMove = (event: React.MouseEvent) => {
    const point = getMousePoint(event)

    if (isDrawing && drawingState.currentTool) {
      // Update active drawing
      drawingService.updateDrawing(point)
      redrawCanvas()
    } else {
      // Handle hover
      const drawingId = findDrawingAtPoint(point)
      if (drawingId !== hoveredDrawing) {
        setHoveredDrawing(drawingId)
        redrawCanvas()
      }
    }
  }

  const handleMouseUp = (event: React.MouseEvent) => {
    if (isDrawing) {
      const completedDrawing = drawingService.finishDrawing()
      if (completedDrawing) {
        onDrawingComplete?.(completedDrawing)
      }
      setIsDrawing(false)
    }
    setDragStart(null)
  }

  const handleDoubleClick = (event: React.MouseEvent) => {
    const point = getMousePoint(event)
    const drawingId = findDrawingAtPoint(point)
    
    if (drawingId) {
      // Open drawing properties dialog
      console.log('Edit drawing:', drawingId)
    }
  }

  return (
    <div
      ref={overlayRef}
      className={className}
      style={{
        position: 'relative',
        width,
        height,
        cursor: drawingState.currentTool ? drawingService.getDrawingTool(drawingState.currentTool)?.cursor || 'crosshair' : 'default'
      }}
    >
      <canvas
        ref={canvasRef}
        width={width}
        height={height}
        style={{
          position: 'absolute',
          top: 0,
          left: 0,
          pointerEvents: 'all'
        }}
        onMouseDown={handleMouseDown}
        onMouseMove={handleMouseMove}
        onMouseUp={handleMouseUp}
        onDoubleClick={handleDoubleClick}
      />
    </div>
  )
}

export default DrawingCanvas