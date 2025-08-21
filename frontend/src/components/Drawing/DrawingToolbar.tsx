/**
 * Drawing Toolbar Component
 * Provides interface for selecting and configuring drawing tools
 */

import React, { useState, useEffect } from 'react'
import { Card, Button, Space, Tooltip, Divider, ColorPicker, InputNumber, Select, Popover } from 'antd'
import { 
  LineOutlined, 
  BorderOutlined, 
  RadiusUprightOutlined,
  FontSizeOutlined,
  ArrowRightOutlined,
  SettingOutlined,
  DeleteOutlined,
  CopyOutlined,
  SaveOutlined
} from '@ant-design/icons'
import { drawingService, DrawingEventHandlers } from '../../services/drawingService'
import { DrawingTool, DrawingObject } from '../../types/charting'

const { Option } = Select

interface DrawingToolbarProps {
  onToolSelect?: (toolId: string | null) => void
  onDrawingStyleChange?: (style: Partial<DrawingObject['style']>) => void
  selectedDrawings?: string[]
  className?: string
}

export const DrawingToolbar: React.FC<DrawingToolbarProps> = ({
  onToolSelect,
  onDrawingStyleChange,
  selectedDrawings = [],
  className
}) => {
  const [tools, setTools] = useState<DrawingTool[]>([])
  const [currentTool, setCurrentTool] = useState<string | null>(null)
  const [drawingStyle, setDrawingStyle] = useState({
    color: '#1890ff',
    lineWidth: 2,
    fillColor: 'rgba(24, 144, 255, 0.1)',
    transparency: 0.1
  })
  const [isExpanded, setIsExpanded] = useState(false)

  useEffect(() => {
    // Load drawing tools
    setTools(drawingService.getDrawingTools())

    // Set up event handlers
    const handlers: DrawingEventHandlers = {
      onDrawingStart: (tool) => {
        setCurrentTool(tool.id)
      },
      onDrawingComplete: () => {
        // Optionally auto-deselect tool after drawing
        // setCurrentTool(null)
      }
    }

    drawingService.setEventHandlers(handlers)
  }, [])

  const handleToolSelect = (toolId: string) => {
    const newTool = currentTool === toolId ? null : toolId
    setCurrentTool(newTool)
    drawingService.setCurrentTool(newTool)
    onToolSelect?.(newTool)
  }

  const handleStyleChange = (changes: Partial<typeof drawingStyle>) => {
    const newStyle = { ...drawingStyle, ...changes }
    setDrawingStyle(newStyle)
    onDrawingStyleChange?.(newStyle)

    // Apply to selected drawings
    selectedDrawings.forEach(drawingId => {
      drawingService.updateDrawing(drawingId, { style: newStyle })
    })
  }

  const handleDeleteSelected = () => {
    if (selectedDrawings.length > 0) {
      drawingService.deleteDrawings(selectedDrawings)
    }
  }

  const handleCopySelected = () => {
    // Copy selected drawings to clipboard
    const drawings = drawingService.getSelectedDrawings()
    if (drawings.length > 0) {
      navigator.clipboard.writeText(JSON.stringify(drawings))
    }
  }

  const handleSaveTemplate = () => {
    const drawings = drawingService.getSelectedDrawings()
    if (drawings.length > 0) {
      const templateName = `Template_${new Date().toISOString().slice(0, 10)}`
      drawingService.createTemplate(templateName, drawings)
    }
  }

  const getToolIcon = (tool: DrawingTool) => {
    switch (tool.id) {
      case 'trend_line':
      case 'horizontal_line':
      case 'vertical_line':
        return <LineOutlined />
      case 'rectangle':
        return <BorderOutlined />
      case 'circle':
        return <RadiusUprightOutlined />
      case 'text':
        return <FontSizeOutlined />
      case 'arrow':
        return <ArrowRightOutlined />
      default:
        return <span>{tool.icon}</span>
    }
  }

  const toolsByCategory = tools.reduce((acc, tool) => {
    if (!acc[tool.category]) {
      acc[tool.category] = []
    }
    acc[tool.category].push(tool)
    return acc
  }, {} as Record<string, DrawingTool[]>)

  const styleConfigPopover = (
    <div style={{ padding: 8, minWidth: 250 }}>
      <div style={{ marginBottom: 12 }}>
        <label style={{ display: 'block', marginBottom: 4, fontSize: '12px' }}>
          Line Color:
        </label>
        <ColorPicker
          value={drawingStyle.color}
          onChange={(color) => handleStyleChange({ color: color.toHexString() })}
        />
      </div>

      <div style={{ marginBottom: 12 }}>
        <label style={{ display: 'block', marginBottom: 4, fontSize: '12px' }}>
          Line Width:
        </label>
        <InputNumber
          min={1}
          max={10}
          value={drawingStyle.lineWidth}
          onChange={(value) => handleStyleChange({ lineWidth: value || 1 })}
          size="small"
          style={{ width: '100%' }}
        />
      </div>

      <div style={{ marginBottom: 12 }}>
        <label style={{ display: 'block', marginBottom: 4, fontSize: '12px' }}>
          Fill Color:
        </label>
        <ColorPicker
          value={drawingStyle.fillColor}
          onChange={(color) => handleStyleChange({ fillColor: color.toHexString() })}
        />
      </div>

      <div>
        <label style={{ display: 'block', marginBottom: 4, fontSize: '12px' }}>
          Transparency:
        </label>
        <InputNumber
          min={0}
          max={1}
          step={0.1}
          value={drawingStyle.transparency}
          onChange={(value) => handleStyleChange({ transparency: value || 0 })}
          size="small"
          style={{ width: '100%' }}
        />
      </div>
    </div>
  )

  return (
    <Card 
      className={className}
      title="Drawing Tools" 
      size="small"
      extra={
        <Button
          size="small"
          type="text"
          onClick={() => setIsExpanded(!isExpanded)}
        >
          {isExpanded ? '−' : '+'}
        </Button>
      }
    >
      {/* Tool Selection */}
      <div style={{ marginBottom: 12 }}>
        {Object.entries(toolsByCategory).map(([category, categoryTools]) => (
          <div key={category} style={{ marginBottom: 8 }}>
            <div style={{ fontSize: '11px', color: '#666', marginBottom: 4, textTransform: 'capitalize' }}>
              {category}:
            </div>
            <Space size="small" wrap>
              {categoryTools.map(tool => (
                <Tooltip key={tool.id} title={tool.name}>
                  <Button
                    size="small"
                    type={currentTool === tool.id ? 'primary' : 'default'}
                    icon={getToolIcon(tool)}
                    onClick={() => handleToolSelect(tool.id)}
                    style={{ minWidth: 32 }}
                  />
                </Tooltip>
              ))}
            </Space>
          </div>
        ))}
      </div>

      {isExpanded && (
        <>
          <Divider style={{ margin: '8px 0' }} />

          {/* Style Configuration */}
          <div style={{ marginBottom: 12 }}>
            <div style={{ fontSize: '11px', color: '#666', marginBottom: 4 }}>
              Style Settings:
            </div>
            <Space size="small">
              <Popover
                content={styleConfigPopover}
                title="Drawing Style"
                trigger="click"
                placement="right"
              >
                <Button size="small" icon={<SettingOutlined />}>
                  Style
                </Button>
              </Popover>

              <div style={{ 
                width: 20, 
                height: 20, 
                backgroundColor: drawingStyle.color,
                border: '1px solid #ccc',
                borderRadius: 2
              }} />
            </Space>
          </div>

          {/* Selected Drawings Actions */}
          {selectedDrawings.length > 0 && (
            <>
              <Divider style={{ margin: '8px 0' }} />
              <div style={{ marginBottom: 8 }}>
                <div style={{ fontSize: '11px', color: '#666', marginBottom: 4 }}>
                  Selected ({selectedDrawings.length}):
                </div>
                <Space size="small">
                  <Tooltip title="Delete selected">
                    <Button
                      size="small"
                      danger
                      icon={<DeleteOutlined />}
                      onClick={handleDeleteSelected}
                    />
                  </Tooltip>
                  
                  <Tooltip title="Copy selected">
                    <Button
                      size="small"
                      icon={<CopyOutlined />}
                      onClick={handleCopySelected}
                    />
                  </Tooltip>
                  
                  <Tooltip title="Save as template">
                    <Button
                      size="small"
                      icon={<SaveOutlined />}
                      onClick={handleSaveTemplate}
                    />
                  </Tooltip>
                </Space>
              </div>
            </>
          )}

          {/* Current Tool Info */}
          {currentTool && (
            <>
              <Divider style={{ margin: '8px 0' }} />
              <div style={{ fontSize: '11px', color: '#666' }}>
                Active: {drawingService.getDrawingTool(currentTool)?.name}
              </div>
            </>
          )}
        </>
      )}

      {/* Quick Actions - Always Visible */}
      <div style={{ marginTop: 8 }}>
        <Space size="small">
          <Button
            size="small"
            type={currentTool ? 'default' : 'primary'}
            onClick={() => handleToolSelect('')}
          >
            Select
          </Button>
          
          <Button
            size="small"
            onClick={() => drawingService.clearSelection()}
            disabled={selectedDrawings.length === 0}
          >
            Clear
          </Button>
        </Space>
      </div>
    </Card>
  )
}

export default DrawingToolbar