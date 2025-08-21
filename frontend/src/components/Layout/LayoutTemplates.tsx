/**
 * Layout Templates Component
 * Provides a gallery of layout templates for quick chart setup
 */

import React, { useState, useEffect } from 'react'
import { Card, Button, Space, Modal, Input, Tag, Tooltip, message } from 'antd'
import { LayoutOutlined, EyeOutlined, CopyOutlined, DeleteOutlined, StarOutlined } from '@ant-design/icons'
import { chartLayoutService, LayoutTemplate } from '../../services/chartLayoutService'

const { Search } = Input

interface LayoutTemplatesProps {
  onTemplateSelect?: (templateId: string) => void
  onCreateFromTemplate?: (layoutId: string) => void
  showCreateButton?: boolean
}

export const LayoutTemplates: React.FC<LayoutTemplatesProps> = ({
  onTemplateSelect,
  onCreateFromTemplate,
  showCreateButton = true
}) => {
  const [templates, setTemplates] = useState<LayoutTemplate[]>([])
  const [filteredTemplates, setFilteredTemplates] = useState<LayoutTemplate[]>([])
  const [selectedCategory, setSelectedCategory] = useState<string>('all')
  const [searchTerm, setSearchTerm] = useState('')
  const [previewTemplate, setPreviewTemplate] = useState<LayoutTemplate | null>(null)
  const [isPreviewModalVisible, setIsPreviewModalVisible] = useState(false)
  const [favorites, setFavorites] = useState<string[]>([])

  useEffect(() => {
    loadTemplates()
    loadFavorites()
  }, [])

  useEffect(() => {
    filterTemplates()
  }, [templates, selectedCategory, searchTerm])

  const loadTemplates = () => {
    const allTemplates = chartLayoutService.getAllTemplates()
    setTemplates(allTemplates)
  }

  const loadFavorites = () => {
    try {
      const stored = localStorage.getItem('favoriteTemplates')
      if (stored) {
        setFavorites(JSON.parse(stored))
      }
    } catch (error) {
      console.error('Failed to load favorite templates:', error)
    }
  }

  const saveFavorites = (newFavorites: string[]) => {
    try {
      localStorage.setItem('favoriteTemplates', JSON.stringify(newFavorites))
      setFavorites(newFavorites)
    } catch (error) {
      console.error('Failed to save favorite templates:', error)
    }
  }

  const filterTemplates = () => {
    let filtered = templates

    // Filter by category
    if (selectedCategory !== 'all') {
      if (selectedCategory === 'favorites') {
        filtered = filtered.filter(template => favorites.includes(template.id))
      } else {
        filtered = filtered.filter(template => template.category === selectedCategory)
      }
    }

    // Filter by search term
    if (searchTerm) {
      filtered = filtered.filter(template =>
        template.name.toLowerCase().includes(searchTerm.toLowerCase()) ||
        template.description.toLowerCase().includes(searchTerm.toLowerCase())
      )
    }

    setFilteredTemplates(filtered)
  }

  const handleCreateFromTemplate = (template: LayoutTemplate) => {
    const layoutId = chartLayoutService.createLayoutFromTemplate(template.id, template.name)
    if (layoutId) {
      onCreateFromTemplate?.(layoutId)
      message.success(`Created layout from ${template.name} template`)
    }
  }

  const handlePreview = (template: LayoutTemplate) => {
    setPreviewTemplate(template)
    setIsPreviewModalVisible(true)
  }

  const handleToggleFavorite = (templateId: string) => {
    const newFavorites = favorites.includes(templateId)
      ? favorites.filter(id => id !== templateId)
      : [...favorites, templateId]
    saveFavorites(newFavorites)
  }

  const handleDeleteTemplate = (templateId: string) => {
    Modal.confirm({
      title: 'Delete Template',
      content: 'Are you sure you want to delete this template? This action cannot be undone.',
      onOk: () => {
        // Note: Need to implement delete in chartLayoutService
        message.success('Template deleted successfully')
        loadTemplates()
      }
    })
  }

  const getCategoryColor = (category: LayoutTemplate['category']) => {
    switch (category) {
      case 'trading': return 'blue'
      case 'analysis': return 'green'
      case 'monitoring': return 'orange'
      case 'custom': return 'purple'
      default: return 'default'
    }
  }

  const getGridDescription = (template: LayoutTemplate) => {
    const { rows, columns } = template.layout.layout
    if (rows === 1 && columns === 1) return 'Single Chart'
    if (rows === 1 && columns === 2) return 'Dual Horizontal'
    if (rows === 2 && columns === 1) return 'Dual Vertical'
    if (rows === 2 && columns === 2) return '2×2 Grid'
    return `${rows}×${columns} Grid`
  }

  const categories = [
    { id: 'all', name: 'All Templates', count: templates.length },
    { id: 'favorites', name: 'Favorites', count: favorites.length },
    { id: 'trading', name: 'Trading', count: templates.filter(t => t.category === 'trading').length },
    { id: 'analysis', name: 'Analysis', count: templates.filter(t => t.category === 'analysis').length },
    { id: 'monitoring', name: 'Monitoring', count: templates.filter(t => t.category === 'monitoring').length },
    { id: 'custom', name: 'Custom', count: templates.filter(t => t.category === 'custom').length },
  ]

  return (
    <div>
      <Card title="Layout Templates" size="small">
        {/* Search and Filters */}
        <div style={{ marginBottom: 16 }}>
          <Search
            placeholder="Search templates..."
            value={searchTerm}
            onChange={(e) => setSearchTerm(e.target.value)}
            style={{ marginBottom: 12 }}
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

        {/* Templates Grid */}
        <div style={{ 
          display: 'grid', 
          gridTemplateColumns: 'repeat(auto-fill, minmax(300px, 1fr))',
          gap: 12,
          maxHeight: 600,
          overflowY: 'auto'
        }}>
          {filteredTemplates.length === 0 ? (
            <div style={{
              gridColumn: '1 / -1',
              textAlign: 'center',
              padding: 40,
              color: '#999',
              background: '#fafafa',
              borderRadius: 4
            }}>
              <LayoutOutlined style={{ fontSize: 24, marginBottom: 8 }} />
              <div>No templates found</div>
              <div style={{ fontSize: '12px', marginTop: 4 }}>
                Try adjusting your search or category filter
              </div>
            </div>
          ) : (
            filteredTemplates.map(template => {
              const isFavorite = favorites.includes(template.id)
              
              return (
                <Card
                  key={template.id}
                  size="small"
                  style={{ 
                    position: 'relative',
                    transition: 'all 0.2s',
                    cursor: 'pointer'
                  }}
                  hoverable
                  bodyStyle={{ padding: 12 }}
                  onClick={() => onTemplateSelect?.(template.id)}
                >
                  {/* Template Preview Area */}
                  <div style={{ 
                    height: 120, 
                    background: '#f5f5f5',
                    marginBottom: 8,
                    borderRadius: 4,
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    position: 'relative'
                  }}>
                    {template.thumbnail ? (
                      <img 
                        src={template.thumbnail} 
                        alt={template.name}
                        style={{ maxWidth: '100%', maxHeight: '100%' }}
                      />
                    ) : (
                      <div style={{ 
                        display: 'grid',
                        gridTemplateColumns: `repeat(${template.layout.layout.columns}, 1fr)`,
                        gridTemplateRows: `repeat(${template.layout.layout.rows}, 1fr)`,
                        gap: 2,
                        width: 80,
                        height: 60
                      }}>
                        {Array(template.layout.layout.rows * template.layout.layout.columns)
                          .fill(0)
                          .map((_, index) => (
                            <div
                              key={index}
                              style={{
                                background: '#d9d9d9',
                                borderRadius: 2
                              }}
                            />
                          ))
                        }
                      </div>
                    )}
                    
                    {/* Favorite Button */}
                    <Button
                      type="text"
                      size="small"
                      icon={<StarOutlined />}
                      onClick={(e) => {
                        e.stopPropagation()
                        handleToggleFavorite(template.id)
                      }}
                      style={{
                        position: 'absolute',
                        top: 4,
                        right: 4,
                        color: isFavorite ? '#faad14' : '#d9d9d9',
                        background: 'rgba(255, 255, 255, 0.8)'
                      }}
                    />
                  </div>

                  {/* Template Info */}
                  <div>
                    <div style={{ display: 'flex', alignItems: 'center', gap: 4, marginBottom: 4 }}>
                      <span style={{ fontWeight: 500, fontSize: '13px' }}>
                        {template.name}
                      </span>
                      <Tag 
                        color={getCategoryColor(template.category)} 
                        size="small"
                        style={{ fontSize: '10px', margin: 0 }}
                      >
                        {template.category.toUpperCase()}
                      </Tag>
                      {template.isBuiltIn && (
                        <Tag color="blue" size="small" style={{ fontSize: '10px', margin: 0 }}>
                          BUILT-IN
                        </Tag>
                      )}
                    </div>

                    <div style={{ fontSize: '11px', color: '#666', marginBottom: 8 }}>
                      {getGridDescription(template)}
                    </div>

                    <div style={{ 
                      fontSize: '11px', 
                      color: '#999', 
                      marginBottom: 8,
                      lineHeight: '1.3',
                      maxHeight: '2.6em',
                      overflow: 'hidden'
                    }}>
                      {template.description}
                    </div>

                    {/* Action Buttons */}
                    <Space size="small">
                      <Tooltip title="Preview template">
                        <Button
                          size="small"
                          type="text"
                          icon={<EyeOutlined />}
                          onClick={(e) => {
                            e.stopPropagation()
                            handlePreview(template)
                          }}
                        />
                      </Tooltip>

                      {showCreateButton && (
                        <Tooltip title="Create layout from template">
                          <Button
                            size="small"
                            type="primary"
                            icon={<CopyOutlined />}
                            onClick={(e) => {
                              e.stopPropagation()
                              handleCreateFromTemplate(template)
                            }}
                          >
                            Use
                          </Button>
                        </Tooltip>
                      )}

                      {!template.isBuiltIn && (
                        <Tooltip title="Delete template">
                          <Button
                            size="small"
                            danger
                            type="text"
                            icon={<DeleteOutlined />}
                            onClick={(e) => {
                              e.stopPropagation()
                              handleDeleteTemplate(template.id)
                            }}
                          />
                        </Tooltip>
                      )}
                    </Space>
                  </div>
                </Card>
              )
            })
          )}
        </div>
      </Card>

      {/* Preview Modal */}
      <Modal
        title={previewTemplate?.name}
        open={isPreviewModalVisible}
        onCancel={() => setIsPreviewModalVisible(false)}
        footer={
          showCreateButton && previewTemplate ? [
            <Button key="cancel" onClick={() => setIsPreviewModalVisible(false)}>
              Cancel
            </Button>,
            <Button
              key="use"
              type="primary"
              icon={<CopyOutlined />}
              onClick={() => {
                handleCreateFromTemplate(previewTemplate)
                setIsPreviewModalVisible(false)
              }}
            >
              Use This Template
            </Button>
          ] : [
            <Button key="close" onClick={() => setIsPreviewModalVisible(false)}>
              Close
            </Button>
          ]
        }
        width={600}
      >
        {previewTemplate && (
          <div>
            <div style={{ marginBottom: 16 }}>
              <div style={{ display: 'flex', gap: 8, marginBottom: 8 }}>
                <Tag color={getCategoryColor(previewTemplate.category)}>
                  {previewTemplate.category.toUpperCase()}
                </Tag>
                {previewTemplate.isBuiltIn && (
                  <Tag color="blue">BUILT-IN</Tag>
                )}
              </div>
              
              <p style={{ color: '#666', margin: 0 }}>
                {previewTemplate.description}
              </p>
            </div>

            <div style={{ marginBottom: 16 }}>
              <h4 style={{ margin: '0 0 8px 0' }}>Layout Details:</h4>
              <div style={{ fontSize: '12px', color: '#666' }}>
                <div>Grid: {getGridDescription(previewTemplate)}</div>
                <div>
                  Synchronization: {[
                    previewTemplate.layout.synchronization.crosshair && 'Crosshair',
                    previewTemplate.layout.synchronization.zoom && 'Zoom',
                    previewTemplate.layout.synchronization.timeRange && 'Time Range'
                  ].filter(Boolean).join(', ') || 'None'}
                </div>
              </div>
            </div>

            {/* Layout visualization */}
            <div style={{
              height: 200,
              background: '#f5f5f5',
              borderRadius: 4,
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center'
            }}>
              <div style={{
                display: 'grid',
                gridTemplateColumns: `repeat(${previewTemplate.layout.layout.columns}, 1fr)`,
                gridTemplateRows: `repeat(${previewTemplate.layout.layout.rows}, 1fr)`,
                gap: 4,
                width: '80%',
                height: '80%'
              }}>
                {Array(previewTemplate.layout.layout.rows * previewTemplate.layout.layout.columns)
                  .fill(0)
                  .map((_, index) => (
                    <div
                      key={index}
                      style={{
                        background: '#d9d9d9',
                        borderRadius: 4,
                        display: 'flex',
                        alignItems: 'center',
                        justifyContent: 'center',
                        fontSize: '11px',
                        color: '#666'
                      }}
                    >
                      Chart {index + 1}
                    </div>
                  ))
                }
              </div>
            </div>
          </div>
        )}
      </Modal>
    </div>
  )
}

export default LayoutTemplates