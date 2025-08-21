/**
 * Drawing Template Library Component
 * Manages drawing templates for reuse and sharing
 */

import React, { useState, useEffect } from 'react'
import { Card, Button, Space, Modal, Input, List, Tag, Upload, message, Tooltip, Popconfirm } from 'antd'
import { 
  SaveOutlined, 
  FolderOpenOutlined, 
  ShareAltOutlined,
  DownloadOutlined,
  UploadOutlined,
  DeleteOutlined,
  CopyOutlined,
  StarOutlined
} from '@ant-design/icons'
import { drawingService } from '../../services/drawingService'
import { DrawingObject } from '../../types/charting'

const { TextArea } = Input

interface DrawingTemplate {
  id: string
  name: string
  description: string
  category: 'trading' | 'analysis' | 'patterns' | 'annotations' | 'custom'
  drawings: DrawingObject[]
  createdAt: string
  updatedAt: string
  tags: string[]
  author: string
  isPublic: boolean
  usageCount: number
  rating: number
}

interface DrawingTemplateLibraryProps {
  selectedDrawings?: string[]
  onTemplateApply?: (drawings: DrawingObject[]) => void
  className?: string
}

export const DrawingTemplateLibrary: React.FC<DrawingTemplateLibraryProps> = ({
  selectedDrawings = [],
  onTemplateApply,
  className
}) => {
  const [templates, setTemplates] = useState<DrawingTemplate[]>([])
  const [filteredTemplates, setFilteredTemplates] = useState<DrawingTemplate[]>([])
  const [selectedCategory, setSelectedCategory] = useState<string>('all')
  const [searchTerm, setSearchTerm] = useState('')
  const [isCreateModalVisible, setIsCreateModalVisible] = useState(false)
  const [isShareModalVisible, setIsShareModalVisible] = useState(false)
  const [selectedTemplate, setSelectedTemplate] = useState<DrawingTemplate | null>(null)
  const [templateName, setTemplateName] = useState('')
  const [templateDescription, setTemplateDescription] = useState('')
  const [templateTags, setTemplateTags] = useState('')
  const [templateCategory, setTemplateCategory] = useState<DrawingTemplate['category']>('custom')

  useEffect(() => {
    loadTemplates()
  }, [])

  useEffect(() => {
    filterTemplates()
  }, [templates, selectedCategory, searchTerm])

  const loadTemplates = () => {
    try {
      // Load from localStorage
      const stored = localStorage.getItem('drawingTemplates')
      if (stored) {
        const loadedTemplates = JSON.parse(stored)
        setTemplates(loadedTemplates)
      }

      // Load built-in templates
      const builtInTemplates = getBuiltInTemplates()
      setTemplates(prev => [...builtInTemplates, ...prev])
    } catch (error) {
      console.error('Failed to load templates:', error)
    }
  }

  const saveTemplates = (newTemplates: DrawingTemplate[]) => {
    try {
      // Only save custom templates to localStorage (exclude built-in)
      const customTemplates = newTemplates.filter(t => t.author !== 'system')
      localStorage.setItem('drawingTemplates', JSON.stringify(customTemplates))
      setTemplates(newTemplates)
    } catch (error) {
      console.error('Failed to save templates:', error)
    }
  }

  const filterTemplates = () => {
    let filtered = templates

    // Filter by category
    if (selectedCategory !== 'all') {
      filtered = filtered.filter(template => template.category === selectedCategory)
    }

    // Filter by search term
    if (searchTerm) {
      filtered = filtered.filter(template =>
        template.name.toLowerCase().includes(searchTerm.toLowerCase()) ||
        template.description.toLowerCase().includes(searchTerm.toLowerCase()) ||
        template.tags.some(tag => tag.toLowerCase().includes(searchTerm.toLowerCase()))
      )
    }

    setFilteredTemplates(filtered)
  }

  const createTemplate = () => {
    if (selectedDrawings.length === 0) {
      message.error('Please select drawings to create a template')
      return
    }

    if (!templateName.trim()) {
      message.error('Please enter a template name')
      return
    }

    const drawings = selectedDrawings
      .map(id => drawingService.getDrawing(id))
      .filter((drawing): drawing is DrawingObject => drawing !== null)

    const template: DrawingTemplate = {
      id: `template_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
      name: templateName.trim(),
      description: templateDescription.trim(),
      category: templateCategory,
      drawings: drawings.map(drawing => ({
        ...drawing,
        id: '', // Reset IDs for template
        chartId: '',
        timeframe: ''
      })),
      createdAt: new Date().toISOString(),
      updatedAt: new Date().toISOString(),
      tags: templateTags.split(',').map(tag => tag.trim()).filter(tag => tag),
      author: 'user',
      isPublic: false,
      usageCount: 0,
      rating: 0
    }

    const newTemplates = [...templates, template]
    saveTemplates(newTemplates)
    
    setIsCreateModalVisible(false)
    resetTemplateForm()
    message.success('Template created successfully')
  }

  const applyTemplate = (template: DrawingTemplate) => {
    const drawings = template.drawings.map(drawing => ({
      ...drawing,
      id: `drawing_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
      chartId: 'current',
      timeframe: '1d'
    }))

    drawings.forEach(drawing => {
      drawingService.addDrawing(drawing)
    })

    // Update usage count
    const updatedTemplate = {
      ...template,
      usageCount: template.usageCount + 1,
      updatedAt: new Date().toISOString()
    }
    
    const updatedTemplates = templates.map(t => 
      t.id === template.id ? updatedTemplate : t
    )
    saveTemplates(updatedTemplates)

    onTemplateApply?.(drawings)
    message.success(`Applied template: ${template.name}`)
  }

  const deleteTemplate = (templateId: string) => {
    const updatedTemplates = templates.filter(t => t.id !== templateId)
    saveTemplates(updatedTemplates)
    message.success('Template deleted')
  }

  const duplicateTemplate = (template: DrawingTemplate) => {
    const duplicate: DrawingTemplate = {
      ...template,
      id: `template_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
      name: `${template.name} (Copy)`,
      createdAt: new Date().toISOString(),
      updatedAt: new Date().toISOString(),
      usageCount: 0
    }

    const newTemplates = [...templates, duplicate]
    saveTemplates(newTemplates)
    message.success('Template duplicated')
  }

  const exportTemplate = (template: DrawingTemplate) => {
    const dataStr = JSON.stringify(template, null, 2)
    const dataBlob = new Blob([dataStr], { type: 'application/json' })
    const url = URL.createObjectURL(dataBlob)
    const link = document.createElement('a')
    link.href = url
    link.download = `${template.name.replace(/[^a-z0-9]/gi, '_')}.json`
    link.click()
    URL.revokeObjectURL(url)
  }

  const importTemplate = (file: File) => {
    const reader = new FileReader()
    reader.onload = (e) => {
      try {
        const template = JSON.parse(e.target?.result as string) as DrawingTemplate
        template.id = `template_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`
        template.createdAt = new Date().toISOString()
        template.updatedAt = new Date().toISOString()
        template.author = 'imported'

        const newTemplates = [...templates, template]
        saveTemplates(newTemplates)
        message.success(`Imported template: ${template.name}`)
      } catch (error) {
        message.error('Failed to import template')
      }
    }
    reader.readAsText(file)
    return false // Prevent default upload
  }

  const resetTemplateForm = () => {
    setTemplateName('')
    setTemplateDescription('')
    setTemplateTags('')
    setTemplateCategory('custom')
  }

  const getBuiltInTemplates = (): DrawingTemplate[] => [
    {
      id: 'builtin_support_resistance',
      name: 'Support & Resistance Lines',
      description: 'Horizontal support and resistance level lines',
      category: 'trading',
      drawings: [
        {
          id: '',
          type: 'trend_line',
          coordinates: [{ x: 0, y: 100 }, { x: 200, y: 100 }],
          style: { color: '#52c41a', lineWidth: 2, fillColor: 'transparent', transparency: 0 },
          anchored: true,
          persistent: true,
          chartId: '',
          timeframe: ''
        },
        {
          id: '',
          type: 'trend_line',
          coordinates: [{ x: 0, y: 150 }, { x: 200, y: 150 }],
          style: { color: '#ff4d4f', lineWidth: 2, fillColor: 'transparent', transparency: 0 },
          anchored: true,
          persistent: true,
          chartId: '',
          timeframe: ''
        }
      ],
      createdAt: '2025-08-20T00:00:00.000Z',
      updatedAt: '2025-08-20T00:00:00.000Z',
      tags: ['support', 'resistance', 'levels'],
      author: 'system',
      isPublic: true,
      usageCount: 0,
      rating: 5
    },
    {
      id: 'builtin_price_channel',
      name: 'Price Channel',
      description: 'Parallel trend lines forming a price channel',
      category: 'patterns',
      drawings: [
        {
          id: '',
          type: 'channel',
          coordinates: [
            { x: 50, y: 50 }, 
            { x: 150, y: 100 }, 
            { x: 50, y: 150 }
          ],
          style: { color: '#1890ff', lineWidth: 2, fillColor: 'rgba(24, 144, 255, 0.1)', transparency: 0.1 },
          anchored: true,
          persistent: true,
          chartId: '',
          timeframe: ''
        }
      ],
      createdAt: '2025-08-20T00:00:00.000Z',
      updatedAt: '2025-08-20T00:00:00.000Z',
      tags: ['channel', 'trend', 'parallel'],
      author: 'system',
      isPublic: true,
      usageCount: 0,
      rating: 4
    }
  ]

  const categories = [
    { id: 'all', name: 'All', count: templates.length },
    { id: 'trading', name: 'Trading', count: templates.filter(t => t.category === 'trading').length },
    { id: 'analysis', name: 'Analysis', count: templates.filter(t => t.category === 'analysis').length },
    { id: 'patterns', name: 'Patterns', count: templates.filter(t => t.category === 'patterns').length },
    { id: 'annotations', name: 'Annotations', count: templates.filter(t => t.category === 'annotations').length },
    { id: 'custom', name: 'Custom', count: templates.filter(t => t.category === 'custom').length }
  ]

  return (
    <Card className={className} title="Drawing Templates" size="small">
      {/* Search and Filters */}
      <div style={{ marginBottom: 12 }}>
        <Input
          placeholder="Search templates..."
          value={searchTerm}
          onChange={(e) => setSearchTerm(e.target.value)}
          style={{ marginBottom: 8 }}
          allowClear
        />
        
        <div style={{ display: 'flex', gap: 4, flexWrap: 'wrap' }}>
          {categories.map(category => (
            <Button
              key={category.id}
              size="small"
              type={selectedCategory === category.id ? 'primary' : 'default'}
              onClick={() => setSelectedCategory(category.id)}
            >
              {category.name} ({category.count})
            </Button>
          ))}
        </div>
      </div>

      {/* Action Buttons */}
      <div style={{ marginBottom: 12 }}>
        <Space size="small">
          <Button
            type="primary"
            size="small"
            icon={<SaveOutlined />}
            onClick={() => setIsCreateModalVisible(true)}
            disabled={selectedDrawings.length === 0}
          >
            Create Template
          </Button>
          
          <Upload
            beforeUpload={importTemplate}
            showUploadList={false}
            accept=".json"
          >
            <Button size="small" icon={<UploadOutlined />}>
              Import
            </Button>
          </Upload>
        </Space>
      </div>

      {/* Templates List */}
      <div style={{ maxHeight: 400, overflowY: 'auto' }}>
        <List
          size="small"
          dataSource={filteredTemplates}
          renderItem={(template) => (
            <List.Item
              actions={[
                <Tooltip title="Apply template">
                  <Button
                    type="link"
                    size="small"
                    icon={<FolderOpenOutlined />}
                    onClick={() => applyTemplate(template)}
                  />
                </Tooltip>,
                <Tooltip title="Export template">
                  <Button
                    type="link"
                    size="small"
                    icon={<DownloadOutlined />}
                    onClick={() => exportTemplate(template)}
                  />
                </Tooltip>,
                <Tooltip title="Duplicate template">
                  <Button
                    type="link"
                    size="small"
                    icon={<CopyOutlined />}
                    onClick={() => duplicateTemplate(template)}
                  />
                </Tooltip>,
                ...(template.author !== 'system' ? [
                  <Popconfirm
                    key="delete"
                    title="Delete this template?"
                    onConfirm={() => deleteTemplate(template.id)}
                  >
                    <Button
                      type="link"
                      size="small"
                      danger
                      icon={<DeleteOutlined />}
                    />
                  </Popconfirm>
                ] : [])
              ]}
            >
              <List.Item.Meta
                title={
                  <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                    <span>{template.name}</span>
                    <Tag color="blue">{template.category}</Tag>
                    {template.author === 'system' && <Tag color="gold">Built-in</Tag>}
                    {template.usageCount > 0 && (
                      <span style={{ fontSize: '10px', color: '#999' }}>
                        Used {template.usageCount} times
                      </span>
                    )}
                  </div>
                }
                description={
                  <div>
                    <div style={{ marginBottom: 4 }}>{template.description}</div>
                    <div>
                      <Space size="small">
                        {template.tags.map(tag => (
                          <Tag key={tag} size="small">{tag}</Tag>
                        ))}
                      </Space>
                    </div>
                    <div style={{ fontSize: '10px', color: '#999', marginTop: 4 }}>
                      {template.drawings.length} drawing{template.drawings.length !== 1 ? 's' : ''} • 
                      Created {new Date(template.createdAt).toLocaleDateString()}
                    </div>
                  </div>
                }
              />
            </List.Item>
          )}
        />
      </div>

      {/* Create Template Modal */}
      <Modal
        title="Create Drawing Template"
        open={isCreateModalVisible}
        onOk={createTemplate}
        onCancel={() => {
          setIsCreateModalVisible(false)
          resetTemplateForm()
        }}
        width={500}
      >
        <div style={{ display: 'grid', gap: 16 }}>
          <div>
            <label style={{ display: 'block', marginBottom: 4 }}>Template Name:</label>
            <Input
              value={templateName}
              onChange={(e) => setTemplateName(e.target.value)}
              placeholder="Enter template name"
            />
          </div>

          <div>
            <label style={{ display: 'block', marginBottom: 4 }}>Description:</label>
            <TextArea
              value={templateDescription}
              onChange={(e) => setTemplateDescription(e.target.value)}
              placeholder="Describe what this template contains"
              rows={3}
            />
          </div>

          <div>
            <label style={{ display: 'block', marginBottom: 4 }}>Category:</label>
            <select
              value={templateCategory}
              onChange={(e) => setTemplateCategory(e.target.value as DrawingTemplate['category'])}
              style={{ width: '100%', padding: '6px 11px', border: '1px solid #d9d9d9', borderRadius: 6 }}
            >
              <option value="trading">Trading</option>
              <option value="analysis">Analysis</option>
              <option value="patterns">Patterns</option>
              <option value="annotations">Annotations</option>
              <option value="custom">Custom</option>
            </select>
          </div>

          <div>
            <label style={{ display: 'block', marginBottom: 4 }}>Tags (comma-separated):</label>
            <Input
              value={templateTags}
              onChange={(e) => setTemplateTags(e.target.value)}
              placeholder="support, resistance, trend"
            />
          </div>

          <div style={{ padding: 8, background: '#f5f5f5', borderRadius: 4, fontSize: '12px' }}>
            <strong>Selected Drawings:</strong> {selectedDrawings.length}
            <div style={{ marginTop: 4, color: '#666' }}>
              This template will contain {selectedDrawings.length} drawing{selectedDrawings.length !== 1 ? 's' : ''} 
              from your current selection.
            </div>
          </div>
        </div>
      </Modal>
    </Card>
  )
}

export default DrawingTemplateLibrary