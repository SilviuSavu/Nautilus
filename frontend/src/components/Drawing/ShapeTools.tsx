/**
 * Shape Tools Component
 * Provides advanced geometric drawing tools
 */

import React, { useState } from 'react'
import { Card, Space, Button, InputNumber, Select, ColorPicker, Switch, Tooltip, Divider } from 'antd'
import { 
  BorderOutlined, 
  RadiusUprightOutlined, 
  FormatPainterOutlined,
  CopyOutlined,
  RotateLeftOutlined
} from '@ant-design/icons'
import { drawingService } from '../../services/drawingService'
import { DrawingObject, Point } from '../../types/charting'

const { Option } = Select

interface ShapeToolsProps {
  onShapeCreate?: (shape: DrawingObject) => void
  selectedDrawings?: string[]
  className?: string
}

interface ShapeConfig {
  type: 'rectangle' | 'circle' | 'ellipse' | 'polygon' | 'arc'
  fillStyle: 'none' | 'solid' | 'gradient' | 'pattern'
  strokeStyle: 'solid' | 'dashed' | 'dotted'
  cornerRadius?: number
  rotation?: number
  proportional?: boolean
}

export const ShapeTools: React.FC<ShapeToolsProps> = ({
  onShapeCreate,
  selectedDrawings = [],
  className
}) => {
  const [shapeConfig, setShapeConfig] = useState<ShapeConfig>({
    type: 'rectangle',
    fillStyle: 'none',
    strokeStyle: 'solid',
    cornerRadius: 0,
    rotation: 0,
    proportional: false
  })

  const [activePresets, setActivePresets] = useState<string[]>([])

  const handleShapeConfigChange = (changes: Partial<ShapeConfig>) => {
    const newConfig = { ...shapeConfig, ...changes }
    setShapeConfig(newConfig)
  }

  const createPredefinedShape = (shapeType: 'square' | 'circle' | 'triangle' | 'diamond') => {
    const baseStyle = {
      color: '#1890ff',
      lineWidth: 2,
      fillColor: shapeConfig.fillStyle === 'none' ? 'transparent' : 'rgba(24, 144, 255, 0.2)',
      transparency: 0.2
    }

    let coordinates: Point[] = []
    let type: DrawingObject['type'] = 'rectangle'

    switch (shapeType) {
      case 'square':
        coordinates = [
          { x: 100, y: 100 },
          { x: 200, y: 200 }
        ]
        type = 'rectangle'
        break
      
      case 'circle':
        coordinates = [
          { x: 150, y: 150 }, // center
          { x: 200, y: 150 }  // edge
        ]
        type = 'circle'
        break
      
      case 'triangle':
        coordinates = [
          { x: 150, y: 100 }, // top
          { x: 100, y: 200 }, // bottom left
          { x: 200, y: 200 }  // bottom right
        ]
        type = 'rectangle' // Will be drawn as custom shape
        break
      
      case 'diamond':
        coordinates = [
          { x: 150, y: 100 }, // top
          { x: 200, y: 150 }, // right
          { x: 150, y: 200 }, // bottom
          { x: 100, y: 150 }  // left
        ]
        type = 'rectangle' // Will be drawn as custom shape
        break
    }

    const shape: Omit<DrawingObject, 'id'> = {
      type,
      coordinates,
      style: baseStyle,
      anchored: true,
      persistent: true,
      chartId: 'current', // Will be updated by drawing service
      timeframe: '1d'
    }

    const shapeId = drawingService.addDrawing(shape)
    const createdShape = drawingService.getDrawing(shapeId)
    
    if (createdShape) {
      onShapeCreate?.(createdShape)
    }
  }

  const applyShapePreset = (preset: string) => {
    const presets: Record<string, Partial<ShapeConfig>> = {
      support_resistance: {
        type: 'rectangle',
        fillStyle: 'solid',
        strokeStyle: 'dashed'
      },
      price_channel: {
        type: 'rectangle',
        fillStyle: 'gradient',
        strokeStyle: 'solid'
      },
      consolidation_zone: {
        type: 'rectangle',
        fillStyle: 'pattern',
        strokeStyle: 'dotted'
      },
      target_zone: {
        type: 'circle',
        fillStyle: 'solid',
        strokeStyle: 'solid'
      }
    }

    const presetConfig = presets[preset]
    if (presetConfig) {
      handleShapeConfigChange(presetConfig)
    }

    // Toggle preset active state
    setActivePresets(prev => {
      if (prev.includes(preset)) {
        return prev.filter(p => p !== preset)
      } else {
        return [...prev, preset]
      }
    })
  }

  const duplicateSelected = () => {
    selectedDrawings.forEach(drawingId => {
      const drawing = drawingService.getDrawing(drawingId)
      if (drawing) {
        const duplicated = {
          ...drawing,
          coordinates: drawing.coordinates.map(coord => ({
            ...coord,
            x: coord.x + 20,
            y: coord.y + 20
          }))
        }
        delete (duplicated as any).id // Remove ID so new one is generated
        drawingService.addDrawing(duplicated)
      }
    })
  }

  const rotateSelected = (degrees: number) => {
    selectedDrawings.forEach(drawingId => {
      const drawing = drawingService.getDrawing(drawingId)
      if (drawing && drawing.coordinates.length >= 2) {
        // Calculate center point
        const centerX = drawing.coordinates.reduce((sum, coord) => sum + coord.x, 0) / drawing.coordinates.length
        const centerY = drawing.coordinates.reduce((sum, coord) => sum + coord.y, 0) / drawing.coordinates.length
        
        // Rotate coordinates around center
        const radians = (degrees * Math.PI) / 180
        const cos = Math.cos(radians)
        const sin = Math.sin(radians)

        const rotatedCoordinates = drawing.coordinates.map(coord => {
          const x = coord.x - centerX
          const y = coord.y - centerY
          
          return {
            ...coord,
            x: centerX + x * cos - y * sin,
            y: centerY + x * sin + y * cos
          }
        })

        drawingService.updateDrawing(drawingId, { coordinates: rotatedCoordinates })
      }
    })
  }

  const shapePresets = [
    { id: 'support_resistance', name: 'Support/Resistance', icon: '📊' },
    { id: 'price_channel', name: 'Price Channel', icon: '📈' },
    { id: 'consolidation_zone', name: 'Consolidation', icon: '📋' },
    { id: 'target_zone', name: 'Target Zone', icon: '🎯' }
  ]

  return (
    <Card className={className} title="Shape Tools" size="small">
      {/* Quick Shape Buttons */}
      <div style={{ marginBottom: 12 }}>
        <div style={{ fontSize: '11px', color: '#666', marginBottom: 4 }}>
          Quick Shapes:
        </div>
        <Space size="small">
          <Tooltip title="Square">
            <Button
              size="small"
              icon={<BorderOutlined />}
              onClick={() => createPredefinedShape('square')}
            />
          </Tooltip>
          
          <Tooltip title="Circle">
            <Button
              size="small"
              icon={<RadiusUprightOutlined />}
              onClick={() => createPredefinedShape('circle')}
            />
          </Tooltip>
          
          <Tooltip title="Triangle">
            <Button
              size="small"
              onClick={() => createPredefinedShape('triangle')}
            >
              △
            </Button>
          </Tooltip>
          
          <Tooltip title="Diamond">
            <Button
              size="small"
              onClick={() => createPredefinedShape('diamond')}
            >
              ◇
            </Button>
          </Tooltip>
        </Space>
      </div>

      <Divider style={{ margin: '8px 0' }} />

      {/* Shape Presets */}
      <div style={{ marginBottom: 12 }}>
        <div style={{ fontSize: '11px', color: '#666', marginBottom: 4 }}>
          Trading Presets:
        </div>
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 4 }}>
          {shapePresets.map(preset => (
            <Button
              key={preset.id}
              size="small"
              type={activePresets.includes(preset.id) ? 'primary' : 'default'}
              onClick={() => applyShapePreset(preset.id)}
              style={{ fontSize: '10px', textAlign: 'left' }}
            >
              <span>{preset.icon} {preset.name}</span>
            </Button>
          ))}
        </div>
      </div>

      <Divider style={{ margin: '8px 0' }} />

      {/* Shape Configuration */}
      <div style={{ marginBottom: 12 }}>
        <div style={{ fontSize: '11px', color: '#666', marginBottom: 6 }}>
          Shape Settings:
        </div>
        
        <div style={{ display: 'grid', gap: 8 }}>
          <div>
            <label style={{ fontSize: '10px', display: 'block', marginBottom: 2 }}>
              Shape Type:
            </label>
            <Select
              size="small"
              value={shapeConfig.type}
              onChange={(value) => handleShapeConfigChange({ type: value })}
              style={{ width: '100%' }}
            >
              <Option value="rectangle">Rectangle</Option>
              <Option value="circle">Circle</Option>
              <Option value="ellipse">Ellipse</Option>
              <Option value="polygon">Polygon</Option>
              <Option value="arc">Arc</Option>
            </Select>
          </div>

          <div>
            <label style={{ fontSize: '10px', display: 'block', marginBottom: 2 }}>
              Fill Style:
            </label>
            <Select
              size="small"
              value={shapeConfig.fillStyle}
              onChange={(value) => handleShapeConfigChange({ fillStyle: value })}
              style={{ width: '100%' }}
            >
              <Option value="none">None</Option>
              <Option value="solid">Solid</Option>
              <Option value="gradient">Gradient</Option>
              <Option value="pattern">Pattern</Option>
            </Select>
          </div>

          <div>
            <label style={{ fontSize: '10px', display: 'block', marginBottom: 2 }}>
              Stroke Style:
            </label>
            <Select
              size="small"
              value={shapeConfig.strokeStyle}
              onChange={(value) => handleShapeConfigChange({ strokeStyle: value })}
              style={{ width: '100%' }}
            >
              <Option value="solid">Solid</Option>
              <Option value="dashed">Dashed</Option>
              <Option value="dotted">Dotted</Option>
            </Select>
          </div>

          {shapeConfig.type === 'rectangle' && (
            <div>
              <label style={{ fontSize: '10px', display: 'block', marginBottom: 2 }}>
                Corner Radius:
              </label>
              <InputNumber
                size="small"
                min={0}
                max={50}
                value={shapeConfig.cornerRadius}
                onChange={(value) => handleShapeConfigChange({ cornerRadius: value || 0 })}
                style={{ width: '100%' }}
              />
            </div>
          )}

          <div>
            <label style={{ fontSize: '10px', display: 'block', marginBottom: 2 }}>
              Rotation (degrees):
            </label>
            <InputNumber
              size="small"
              min={-180}
              max={180}
              value={shapeConfig.rotation}
              onChange={(value) => handleShapeConfigChange({ rotation: value || 0 })}
              style={{ width: '100%' }}
            />
          </div>

          <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
            <Switch
              size="small"
              checked={shapeConfig.proportional}
              onChange={(checked) => handleShapeConfigChange({ proportional: checked })}
            />
            <span style={{ fontSize: '10px' }}>Lock Proportions</span>
          </div>
        </div>
      </div>

      {/* Selected Shapes Actions */}
      {selectedDrawings.length > 0 && (
        <>
          <Divider style={{ margin: '8px 0' }} />
          <div>
            <div style={{ fontSize: '11px', color: '#666', marginBottom: 4 }}>
              Selected Actions:
            </div>
            <Space size="small">
              <Tooltip title="Duplicate">
                <Button
                  size="small"
                  icon={<CopyOutlined />}
                  onClick={duplicateSelected}
                />
              </Tooltip>
              
              <Tooltip title="Rotate 90°">
                <Button
                  size="small"
                  icon={<RotateLeftOutlined />}
                  onClick={() => rotateSelected(90)}
                />
              </Tooltip>
              
              <Tooltip title="Apply current style">
                <Button
                  size="small"
                  icon={<FormatPainterOutlined />}
                  onClick={() => {
                    // Apply current shape config to selected shapes
                    selectedDrawings.forEach(drawingId => {
                      drawingService.updateDrawing(drawingId, {
                        style: {
                          ...drawingService.getDrawing(drawingId)?.style,
                          // Apply relevant style changes based on config
                        }
                      })
                    })
                  }}
                />
              </Tooltip>
            </Space>
          </div>
        </>
      )}
    </Card>
  )
}

export default ShapeTools