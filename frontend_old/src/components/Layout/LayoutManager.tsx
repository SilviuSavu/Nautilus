/**
 * Layout Manager Component
 * Provides interface for managing chart layouts and templates
 */

import React, { useState, useEffect } from 'react'
import { Card, Button, Space, Select, Modal, Form, Input, Radio, Grid, Tooltip, message } from 'antd'
import { PlusOutlined, SaveOutlined, DeleteOutlined, CopyOutlined, LayoutOutlined } from '@ant-design/icons'
import { chartLayoutService, LayoutTemplate } from '../../services/chartLayoutService'
import { ChartLayout } from '../../types/charting'

const { Option } = Select
const { TextArea } = Input
const { useBreakpoint } = Grid

interface LayoutManagerProps {
  currentLayout?: ChartLayout
  onLayoutChange?: (layout: ChartLayout) => void
  onLayoutCreate?: (layoutId: string) => void
  onLayoutDelete?: (layoutId: string) => void
}

export const LayoutManager: React.FC<LayoutManagerProps> = ({
  currentLayout,
  onLayoutChange,
  onLayoutCreate,
  onLayoutDelete
}) => {
  const [layouts, setLayouts] = useState<ChartLayout[]>([])
  const [templates, setTemplates] = useState<LayoutTemplate[]>([])
  const [selectedLayoutId, setSelectedLayoutId] = useState<string>('')
  const [isCreateModalVisible, setIsCreateModalVisible] = useState(false)
  const [isTemplateModalVisible, setIsTemplateModalVisible] = useState(false)
  const [createForm] = Form.useForm()
  const [templateForm] = Form.useForm()
  const screens = useBreakpoint()

  useEffect(() => {
    loadLayoutsAndTemplates()
    
    // Set up event handlers
    chartLayoutService.setEventHandlers({
      onLayoutChange: handleLayoutServiceChange
    })

    // Load saved layouts from storage
    chartLayoutService.loadLayoutsFromStorage()
    loadLayoutsAndTemplates()
  }, [])

  useEffect(() => {
    if (currentLayout) {
      setSelectedLayoutId(currentLayout.id)
    }
  }, [currentLayout])

  const loadLayoutsAndTemplates = () => {
    setLayouts(chartLayoutService.getAllLayouts())
    setTemplates(chartLayoutService.getAllTemplates())
  }

  const handleLayoutServiceChange = (layout: ChartLayout) => {
    loadLayoutsAndTemplates()
    onLayoutChange?.(layout)
  }

  const handleLayoutSelect = (layoutId: string) => {
    setSelectedLayoutId(layoutId)
    const layout = chartLayoutService.getLayout(layoutId)
    if (layout) {
      chartLayoutService.setActiveLayout(layoutId)
      onLayoutChange?.(layout)
    }
  }

  const handleCreateLayout = async () => {
    try {
      const values = await createForm.validateFields()
      const layoutId = chartLayoutService.createLayout({
        name: values.name,
        charts: [],
        layout: {
          rows: values.rows || 1,
          columns: values.columns || 1,
          chartPositions: []
        },
        synchronization: {
          crosshair: values.synchronization?.includes('crosshair') || false,
          zoom: values.synchronization?.includes('zoom') || false,
          timeRange: values.synchronization?.includes('timeRange') || false
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
      })

      setIsCreateModalVisible(false)
      createForm.resetFields()
      loadLayoutsAndTemplates()
      onLayoutCreate?.(layoutId)
      message.success('Layout created successfully')
    } catch (error) {
      console.error('Failed to create layout:', error)
    }
  }

  const handleCreateFromTemplate = (templateId: string) => {
    const template = chartLayoutService.getTemplate(templateId)
    if (template) {
      const layoutId = chartLayoutService.createLayoutFromTemplate(templateId, `${template.name} - ${Date.now()}`)
      if (layoutId) {
        loadLayoutsAndTemplates()
        onLayoutCreate?.(layoutId)
        message.success(`Created layout from ${template.name} template`)
      }
    }
  }

  const handleSaveAsTemplate = async () => {
    if (!currentLayout) {
      message.error('No active layout to save as template')
      return
    }

    try {
      const values = await templateForm.validateFields()
      chartLayoutService.createTemplate(
        values.name,
        values.description || '',
        currentLayout,
        values.category || 'custom'
      )
      
      setIsTemplateModalVisible(false)
      templateForm.resetFields()
      loadLayoutsAndTemplates()
      message.success('Template saved successfully')
    } catch (error) {
      console.error('Failed to save template:', error)
    }
  }

  const handleDeleteLayout = (layoutId: string) => {
    Modal.confirm({
      title: 'Delete Layout',
      content: 'Are you sure you want to delete this layout? This action cannot be undone.',
      onOk: () => {
        if (chartLayoutService.deleteLayout(layoutId)) {
          loadLayoutsAndTemplates()
          if (selectedLayoutId === layoutId) {
            setSelectedLayoutId('')
          }
          onLayoutDelete?.(layoutId)
          message.success('Layout deleted successfully')
        }
      }
    })
  }

  const handleDuplicateLayout = (layoutId: string) => {
    const layout = chartLayoutService.getLayout(layoutId)
    if (layout) {
      const newLayoutId = chartLayoutService.createLayout({
        ...layout,
        name: `${layout.name} (Copy)`,
        id: undefined as any // Will be generated
      })
      loadLayoutsAndTemplates()
      onLayoutCreate?.(newLayoutId)
      message.success('Layout duplicated successfully')
    }
  }

  const getTemplatesByCategory = (category: LayoutTemplate['category']) => {
    return templates.filter(template => template.category === category)
  }

  const getLayoutGridDescription = (rows: number, columns: number) => {
    if (rows === 1 && columns === 1) return 'Single Chart'
    if (rows === 1 && columns === 2) return 'Dual Horizontal'
    if (rows === 2 && columns === 1) return 'Dual Vertical'
    if (rows === 2 && columns === 2) return '2×2 Grid'
    if (rows === 3 && columns === 3) return '3×3 Grid'
    return `${rows}×${columns} Grid`
  }

  return (
    <div>
      <Card title="Layout Manager" size="small">
        {/* Current Layout Selection */}
        <div style={{ marginBottom: 16 }}>
          <Space direction="vertical" style={{ width: '100%' }}>
            <div>
              <label style={{ display: 'block', marginBottom: 8, fontWeight: 500 }}>
                Active Layout:
              </label>
              <Select
                value={selectedLayoutId}
                onChange={handleLayoutSelect}
                style={{ width: '100%' }}
                placeholder="Select a layout"
                allowClear
              >
                {layouts.map(layout => (
                  <Option key={layout.id} value={layout.id}>
                    <Space>
                      <LayoutOutlined />
                      <span>{layout.name}</span>
                      <span style={{ color: '#999', fontSize: '12px' }}>
                        ({getLayoutGridDescription(layout.layout.rows, layout.layout.columns)})
                      </span>
                    </Space>
                  </Option>
                ))}
              </Select>
            </div>

            {/* Layout Actions */}
            <Space wrap>
              <Button
                type="primary"
                icon={<PlusOutlined />}
                onClick={() => setIsCreateModalVisible(true)}
                size="small"
              >
                New Layout
              </Button>
              
              {currentLayout && (
                <>
                  <Button
                    icon={<SaveOutlined />}
                    onClick={() => setIsTemplateModalVisible(true)}
                    size="small"
                  >
                    Save as Template
                  </Button>
                  
                  <Button
                    icon={<CopyOutlined />}
                    onClick={() => handleDuplicateLayout(currentLayout.id)}
                    size="small"
                  >
                    Duplicate
                  </Button>
                  
                  <Button
                    danger
                    icon={<DeleteOutlined />}
                    onClick={() => handleDeleteLayout(currentLayout.id)}
                    size="small"
                  >
                    Delete
                  </Button>
                </>
              )}
            </Space>
          </Space>
        </div>

        {/* Layout Templates */}
        <div>
          <label style={{ display: 'block', marginBottom: 8, fontWeight: 500 }}>
            Quick Start Templates:
          </label>
          
          {/* Built-in Templates */}
          <div style={{ marginBottom: 12 }}>
            <div style={{ fontSize: '12px', color: '#666', marginBottom: 6 }}>Built-in Templates:</div>
            <Space wrap size="small">
              {getTemplatesByCategory('trading').concat(
                getTemplatesByCategory('analysis'),
                getTemplatesByCategory('monitoring')
              ).filter(template => template.isBuiltIn).map(template => (
                <Tooltip key={template.id} title={template.description}>
                  <Button
                    size="small"
                    onClick={() => handleCreateFromTemplate(template.id)}
                    style={{ fontSize: '11px' }}
                  >
                    {template.name}
                  </Button>
                </Tooltip>
              ))}
            </Space>
          </div>

          {/* Custom Templates */}
          {getTemplatesByCategory('custom').length > 0 && (
            <div>
              <div style={{ fontSize: '12px', color: '#666', marginBottom: 6 }}>Custom Templates:</div>
              <Space wrap size="small">
                {getTemplatesByCategory('custom').map(template => (
                  <Tooltip key={template.id} title={template.description}>
                    <Button
                      size="small"
                      onClick={() => handleCreateFromTemplate(template.id)}
                      style={{ fontSize: '11px' }}
                    >
                      {template.name}
                    </Button>
                  </Tooltip>
                ))}
              </Space>
            </div>
          )}
        </div>

        {/* Current Layout Info */}
        {currentLayout && (
          <div style={{ 
            marginTop: 16,
            padding: 8,
            background: '#f5f5f5',
            borderRadius: 4,
            fontSize: '12px'
          }}>
            <div><strong>Layout:</strong> {currentLayout.name}</div>
            <div><strong>Grid:</strong> {getLayoutGridDescription(currentLayout.layout.rows, currentLayout.layout.columns)}</div>
            <div><strong>Charts:</strong> {currentLayout.charts.length}</div>
            <div><strong>Sync:</strong> {
              [
                currentLayout.synchronization.crosshair && 'Crosshair',
                currentLayout.synchronization.zoom && 'Zoom',
                currentLayout.synchronization.timeRange && 'Time'
              ].filter(Boolean).join(', ') || 'None'
            }</div>
          </div>
        )}
      </Card>

      {/* Create Layout Modal */}
      <Modal
        title="Create New Layout"
        open={isCreateModalVisible}
        onOk={handleCreateLayout}
        onCancel={() => setIsCreateModalVisible(false)}
        width={500}
      >
        <Form form={createForm} layout="vertical">
          <Form.Item
            name="name"
            label="Layout Name"
            rules={[{ required: true, message: 'Please enter a layout name' }]}
          >
            <Input placeholder="Enter layout name" />
          </Form.Item>

          <Space direction="vertical" style={{ width: '100%' }}>
            <Form.Item label="Grid Size">
              <Space>
                <Form.Item name="rows" style={{ marginBottom: 0 }}>
                  <Select placeholder="Rows" style={{ width: 80 }}>
                    {[1, 2, 3, 4].map(n => (
                      <Option key={n} value={n}>{n}</Option>
                    ))}
                  </Select>
                </Form.Item>
                <span>×</span>
                <Form.Item name="columns" style={{ marginBottom: 0 }}>
                  <Select placeholder="Cols" style={{ width: 80 }}>
                    {[1, 2, 3, 4].map(n => (
                      <Option key={n} value={n}>{n}</Option>
                    ))}
                  </Select>
                </Form.Item>
              </Space>
            </Form.Item>

            <Form.Item
              name="synchronization"
              label="Chart Synchronization"
            >
              <Select mode="multiple" placeholder="Select synchronization options">
                <Option value="crosshair">Crosshair</Option>
                <Option value="zoom">Zoom</Option>
                <Option value="timeRange">Time Range</Option>
              </Select>
            </Form.Item>
          </Space>
        </Form>
      </Modal>

      {/* Save as Template Modal */}
      <Modal
        title="Save as Template"
        open={isTemplateModalVisible}
        onOk={handleSaveAsTemplate}
        onCancel={() => setIsTemplateModalVisible(false)}
        width={500}
      >
        <Form form={templateForm} layout="vertical">
          <Form.Item
            name="name"
            label="Template Name"
            rules={[{ required: true, message: 'Please enter a template name' }]}
          >
            <Input placeholder="Enter template name" />
          </Form.Item>

          <Form.Item name="description" label="Description">
            <TextArea 
              placeholder="Enter template description (optional)" 
              rows={3}
            />
          </Form.Item>

          <Form.Item name="category" label="Category">
            <Radio.Group>
              <Radio value="trading">Trading</Radio>
              <Radio value="analysis">Analysis</Radio>
              <Radio value="monitoring">Monitoring</Radio>
              <Radio value="custom">Custom</Radio>
            </Radio.Group>
          </Form.Item>
        </Form>
      </Modal>
    </div>
  )
}

export default LayoutManager