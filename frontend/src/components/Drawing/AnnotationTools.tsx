/**
 * Annotation Tools Component
 * Provides text annotation and labeling capabilities
 */

import React, { useState, useRef } from 'react'
import { Card, Input, Button, Space, Select, ColorPicker, Slider, Switch, Modal, List, Tooltip } from 'antd'
import { 
  FontSizeOutlined, 
  CommentOutlined, 
  TagOutlined,
  PushpinOutlined,
  MessageOutlined,
  EditOutlined,
  DeleteOutlined,
  CopyOutlined
} from '@ant-design/icons'
import { drawingService } from '../../services/drawingService'
import { DrawingObject, Point } from '../../types/charting'

const { TextArea } = Input
const { Option } = Select

interface AnnotationToolsProps {
  onAnnotationCreate?: (annotation: DrawingObject) => void
  selectedDrawings?: string[]
  className?: string
}

interface AnnotationConfig {
  fontSize: number
  fontFamily: string
  textColor: string
  backgroundColor: string
  borderColor: string
  padding: number
  borderRadius: number
  showArrow: boolean
  arrowPosition: 'top' | 'bottom' | 'left' | 'right'
  opacity: number
  bold: boolean
  italic: boolean
  underline: boolean
}

interface AnnotationTemplate {
  id: string
  name: string
  text: string
  config: AnnotationConfig
  category: 'trading' | 'analysis' | 'notes' | 'alerts'
}

export const AnnotationTools: React.FC<AnnotationToolsProps> = ({
  onAnnotationCreate,
  selectedDrawings = [],
  className
}) => {
  const [annotationConfig, setAnnotationConfig] = useState<AnnotationConfig>({
    fontSize: 12,
    fontFamily: 'Arial',
    textColor: '#000000',
    backgroundColor: '#ffffff',
    borderColor: '#1890ff',
    padding: 8,
    borderRadius: 4,
    showArrow: false,
    arrowPosition: 'bottom',
    opacity: 1,
    bold: false,
    italic: false,
    underline: false
  })

  const [currentText, setCurrentText] = useState('')
  const [annotationMode, setAnnotationMode] = useState<'text' | 'label' | 'callout' | 'note'>('text')
  const [isTemplateModalVisible, setIsTemplateModalVisible] = useState(false)
  const [templates, setTemplates] = useState<AnnotationTemplate[]>([])
  const textInputRef = useRef<any>(null)

  const handleConfigChange = (changes: Partial<AnnotationConfig>) => {
    const newConfig = { ...annotationConfig, ...changes }
    setAnnotationConfig(newConfig)
  }

  const createAnnotation = (position: Point = { x: 100, y: 100 }) => {
    if (!currentText.trim()) {
      Modal.warning({
        title: 'No Text',
        content: 'Please enter text for the annotation.'
      })
      return
    }

    const annotation: Omit<DrawingObject, 'id'> = {
      type: 'text',
      coordinates: [position],
      style: {
        color: annotationConfig.textColor,
        lineWidth: 1,
        fillColor: annotationConfig.backgroundColor,
        transparency: 1 - annotationConfig.opacity,
        fontSize: annotationConfig.fontSize
      },
      text: currentText,
      anchored: true,
      persistent: true,
      chartId: 'current',
      timeframe: '1d'
    }

    const annotationId = drawingService.addDrawing(annotation)
    const createdAnnotation = drawingService.getDrawing(annotationId)
    
    if (createdAnnotation) {
      onAnnotationCreate?.(createdAnnotation)
      setCurrentText('')
    }
  }

  const createQuickNote = (noteType: 'buy' | 'sell' | 'watch' | 'warning') => {
    const quickNotes = {
      buy: { text: '🟢 BUY', color: '#52c41a' },
      sell: { text: '🔴 SELL', color: '#ff4d4f' },
      watch: { text: '👁️ WATCH', color: '#1890ff' },
      warning: { text: '⚠️ WARNING', color: '#faad14' }
    }

    const note = quickNotes[noteType]
    setCurrentText(note.text)
    handleConfigChange({ textColor: note.color, bold: true })
    
    // Auto-create the annotation
    createAnnotation()
  }

  const createPriceLabel = (price?: number) => {
    const labelText = price ? `$${price.toFixed(2)}` : '$XXX.XX'
    setCurrentText(labelText)
    handleConfigChange({
      backgroundColor: '#000000',
      textColor: '#ffffff',
      showArrow: true,
      padding: 6,
      borderRadius: 2
    })
    createAnnotation()
  }

  const editSelectedAnnotations = () => {
    const selectedAnnotations = selectedDrawings
      .map(id => drawingService.getDrawing(id))
      .filter((drawing): drawing is DrawingObject => 
        drawing !== null && drawing.type === 'text'
      )

    if (selectedAnnotations.length === 0) return

    // For multiple selections, show batch edit modal
    if (selectedAnnotations.length > 1) {
      Modal.confirm({
        title: 'Batch Edit Annotations',
        content: `Apply current style to ${selectedAnnotations.length} annotations?`,
        onOk: () => {
          selectedAnnotations.forEach(annotation => {
            drawingService.updateDrawing(annotation.id, {
              style: {
                ...annotation.style,
                color: annotationConfig.textColor,
                fillColor: annotationConfig.backgroundColor,
                fontSize: annotationConfig.fontSize
              }
            })
          })
        }
      })
    } else {
      // For single selection, populate the editor
      const annotation = selectedAnnotations[0]
      setCurrentText(annotation.text || '')
      handleConfigChange({
        textColor: annotation.style?.color || '#000000',
        backgroundColor: annotation.style?.fillColor || '#ffffff',
        fontSize: annotation.style?.fontSize || 12
      })
    }
  }

  const saveTemplate = () => {
    if (!currentText.trim()) return

    const template: AnnotationTemplate = {
      id: `template_${Date.now()}`,
      name: currentText.slice(0, 20) + (currentText.length > 20 ? '...' : ''),
      text: currentText,
      config: annotationConfig,
      category: 'notes'
    }

    setTemplates(prev => [...prev, template])
    
    // Save to localStorage
    const existingTemplates = JSON.parse(localStorage.getItem('annotationTemplates') || '[]')
    existingTemplates.push(template)
    localStorage.setItem('annotationTemplates', JSON.stringify(existingTemplates))
  }

  const loadTemplate = (template: AnnotationTemplate) => {
    setCurrentText(template.text)
    setAnnotationConfig(template.config)
    setIsTemplateModalVisible(false)
  }

  const fontFamilies = [
    'Arial', 'Helvetica', 'Times New Roman', 'Courier New', 'Georgia', 'Verdana'
  ]

  return (
    <Card className={className} title="Annotations" size="small">
      {/* Annotation Mode */}
      <div style={{ marginBottom: 12 }}>
        <div style={{ fontSize: '11px', color: '#666', marginBottom: 4 }}>
          Annotation Type:
        </div>
        <Select
          size="small"
          value={annotationMode}
          onChange={setAnnotationMode}
          style={{ width: '100%' }}
        >
          <Option value="text">Text</Option>
          <Option value="label">Price Label</Option>
          <Option value="callout">Callout</Option>
          <Option value="note">Note</Option>
        </Select>
      </div>

      {/* Quick Actions */}
      <div style={{ marginBottom: 12 }}>
        <div style={{ fontSize: '11px', color: '#666', marginBottom: 4 }}>
          Quick Notes:
        </div>
        <Space size="small" wrap>
          <Button size="small" onClick={() => createQuickNote('buy')}>🟢 BUY</Button>
          <Button size="small" onClick={() => createQuickNote('sell')}>🔴 SELL</Button>
          <Button size="small" onClick={() => createQuickNote('watch')}>👁️ WATCH</Button>
          <Button size="small" onClick={() => createQuickNote('warning')}>⚠️ WARN</Button>
        </Space>
      </div>

      {/* Text Input */}
      <div style={{ marginBottom: 12 }}>
        <div style={{ fontSize: '11px', color: '#666', marginBottom: 4 }}>
          Text Content:
        </div>
        <TextArea
          ref={textInputRef}
          value={currentText}
          onChange={(e) => setCurrentText(e.target.value)}
          placeholder="Enter annotation text..."
          rows={3}
          style={{ fontSize: '12px' }}
        />
      </div>

      {/* Style Configuration */}
      <div style={{ marginBottom: 12 }}>
        <div style={{ fontSize: '11px', color: '#666', marginBottom: 6 }}>
          Text Style:
        </div>
        
        <div style={{ display: 'grid', gap: 6, gridTemplateColumns: '1fr 1fr' }}>
          <div>
            <label style={{ fontSize: '10px', display: 'block', marginBottom: 2 }}>
              Font:
            </label>
            <Select
              size="small"
              value={annotationConfig.fontFamily}
              onChange={(value) => handleConfigChange({ fontFamily: value })}
              style={{ width: '100%' }}
            >
              {fontFamilies.map(font => (
                <Option key={font} value={font}>{font}</Option>
              ))}
            </Select>
          </div>

          <div>
            <label style={{ fontSize: '10px', display: 'block', marginBottom: 2 }}>
              Size:
            </label>
            <Select
              size="small"
              value={annotationConfig.fontSize}
              onChange={(value) => handleConfigChange({ fontSize: value })}
              style={{ width: '100%' }}
            >
              {[8, 10, 12, 14, 16, 18, 20, 24, 28, 32].map(size => (
                <Option key={size} value={size}>{size}px</Option>
              ))}
            </Select>
          </div>

          <div>
            <label style={{ fontSize: '10px', display: 'block', marginBottom: 2 }}>
              Text Color:
            </label>
            <ColorPicker
              value={annotationConfig.textColor}
              onChange={(color) => handleConfigChange({ textColor: color.toHexString() })}
              size="small"
            />
          </div>

          <div>
            <label style={{ fontSize: '10px', display: 'block', marginBottom: 2 }}>
              Background:
            </label>
            <ColorPicker
              value={annotationConfig.backgroundColor}
              onChange={(color) => handleConfigChange({ backgroundColor: color.toHexString() })}
              size="small"
            />
          </div>
        </div>

        {/* Text Formatting */}
        <div style={{ marginTop: 8 }}>
          <Space size="small">
            <Button
              size="small"
              type={annotationConfig.bold ? 'primary' : 'default'}
              onClick={() => handleConfigChange({ bold: !annotationConfig.bold })}
            >
              <strong>B</strong>
            </Button>
            <Button
              size="small"
              type={annotationConfig.italic ? 'primary' : 'default'}
              onClick={() => handleConfigChange({ italic: !annotationConfig.italic })}
            >
              <em>I</em>
            </Button>
            <Button
              size="small"
              type={annotationConfig.underline ? 'primary' : 'default'}
              onClick={() => handleConfigChange({ underline: !annotationConfig.underline })}
            >
              <u>U</u>
            </Button>
          </Space>
        </div>
      </div>

      {/* Advanced Options */}
      <div style={{ marginBottom: 12 }}>
        <div style={{ fontSize: '11px', color: '#666', marginBottom: 6 }}>
          Layout:
        </div>
        
        <div style={{ display: 'grid', gap: 6 }}>
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
            <span style={{ fontSize: '10px' }}>Padding:</span>
            <Slider
              min={0}
              max={20}
              value={annotationConfig.padding}
              onChange={(value) => handleConfigChange({ padding: value })}
              style={{ width: 100 }}
            />
          </div>

          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
            <span style={{ fontSize: '10px' }}>Corner Radius:</span>
            <Slider
              min={0}
              max={10}
              value={annotationConfig.borderRadius}
              onChange={(value) => handleConfigChange({ borderRadius: value })}
              style={{ width: 100 }}
              size="small"
            />
          </div>

          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
            <span style={{ fontSize: '10px' }}>Opacity:</span>
            <Slider
              min={0}
              max={1}
              step={0.1}
              value={annotationConfig.opacity}
              onChange={(value) => handleConfigChange({ opacity: value })}
              style={{ width: 100 }}
              size="small"
            />
          </div>

          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
            <span style={{ fontSize: '10px' }}>Show Arrow:</span>
            <Switch
              size="small"
              checked={annotationConfig.showArrow}
              onChange={(checked) => handleConfigChange({ showArrow: checked })}
            />
          </div>
        </div>
      </div>

      {/* Action Buttons */}
      <div style={{ marginBottom: 12 }}>
        <Space size="small" direction="vertical" style={{ width: '100%' }}>
          <Button
            type="primary"
            size="small"
            block
            icon={<FontSizeOutlined />}
            onClick={() => createAnnotation()}
            disabled={!currentText.trim()}
          >
            Add Text
          </Button>
          
          <Space size="small">
            <Button
              size="small"
              icon={<TagOutlined />}
              onClick={() => createPriceLabel()}
            >
              Price Label
            </Button>
            
            <Button
              size="small"
              icon={<MessageOutlined />}
              onClick={() => setIsTemplateModalVisible(true)}
            >
              Templates
            </Button>
            
            <Button
              size="small"
              icon={<CopyOutlined />}
              onClick={saveTemplate}
              disabled={!currentText.trim()}
            >
              Save
            </Button>
          </Space>
        </Space>
      </div>

      {/* Selected Annotations Actions */}
      {selectedDrawings.length > 0 && (
        <div>
          <div style={{ fontSize: '11px', color: '#666', marginBottom: 4 }}>
            Selected Actions:
          </div>
          <Space size="small">
            <Tooltip title="Edit selected annotations">
              <Button
                size="small"
                icon={<EditOutlined />}
                onClick={editSelectedAnnotations}
              />
            </Tooltip>
            
            <Tooltip title="Delete selected">
              <Button
                size="small"
                danger
                icon={<DeleteOutlined />}
                onClick={() => drawingService.deleteDrawings(selectedDrawings)}
              />
            </Tooltip>
          </Space>
        </div>
      )}

      {/* Templates Modal */}
      <Modal
        title="Annotation Templates"
        open={isTemplateModalVisible}
        onCancel={() => setIsTemplateModalVisible(false)}
        footer={null}
        width={400}
      >
        <List
          size="small"
          dataSource={templates}
          renderItem={(template) => (
            <List.Item
              actions={[
                <Button 
                  type="link" 
                  size="small"
                  onClick={() => loadTemplate(template)}
                >
                  Use
                </Button>
              ]}
            >
              <List.Item.Meta
                title={template.name}
                description={template.text}
              />
            </List.Item>
          )}
        />
      </Modal>
    </Card>
  )
}

export default AnnotationTools