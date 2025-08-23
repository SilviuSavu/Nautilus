/**
 * Indicator Library Component
 * Displays available indicators with categories and search functionality
 */

import React, { useState, useMemo } from 'react'
import { Card, Input, Tag, Button, Space, Tooltip, Badge, Modal, Typography } from 'antd'
import { SearchOutlined, InfoCircleOutlined, PlusOutlined, StarOutlined } from '@ant-design/icons'
import { indicatorEngine, TechnicalIndicator } from '../../services/indicatorEngine'

const { Search } = Input
const { Text, Paragraph } = Typography

interface IndicatorLibraryProps {
  onAddIndicator: (indicatorId: string) => void
  favoriteIndicators?: string[]
  onToggleFavorite?: (indicatorId: string) => void
}

interface IndicatorCategory {
  id: string
  name: string
  description: string
  color: string
}

const INDICATOR_CATEGORIES: IndicatorCategory[] = [
  {
    id: 'trend',
    name: 'Trend Following',
    description: 'Indicators that identify market trends',
    color: '#1890ff'
  },
  {
    id: 'momentum',
    name: 'Momentum',
    description: 'Indicators that measure price momentum',
    color: '#52c41a'
  },
  {
    id: 'volatility',
    name: 'Volatility',
    description: 'Indicators that measure market volatility',
    color: '#faad14'
  },
  {
    id: 'volume',
    name: 'Volume',
    description: 'Volume-based indicators',
    color: '#722ed1'
  },
  {
    id: 'support_resistance',
    name: 'Support/Resistance',
    description: 'Indicators for support and resistance levels',
    color: '#eb2f96'
  },
  {
    id: 'custom',
    name: 'Custom',
    description: 'User-created custom indicators',
    color: '#13c2c2'
  }
]

const INDICATOR_DESCRIPTIONS: Record<string, { description: string; category: string; useCase: string }> = {
  sma: {
    description: 'Simple Moving Average - Average price over N periods',
    category: 'trend',
    useCase: 'Trend identification and support/resistance levels'
  },
  ema: {
    description: 'Exponential Moving Average - Weighted average giving more weight to recent prices',
    category: 'trend',
    useCase: 'Trend following with faster response to price changes'
  },
  rsi: {
    description: 'Relative Strength Index - Momentum oscillator (0-100)',
    category: 'momentum',
    useCase: 'Overbought/oversold conditions and divergence analysis'
  },
  macd: {
    description: 'Moving Average Convergence Divergence - Trend and momentum indicator',
    category: 'momentum',
    useCase: 'Trend changes and momentum shifts'
  },
  bollinger: {
    description: 'Bollinger Bands - Volatility bands around moving average',
    category: 'volatility',
    useCase: 'Volatility measurement and mean reversion trades'
  }
}

export const IndicatorLibrary: React.FC<IndicatorLibraryProps> = ({
  onAddIndicator,
  favoriteIndicators = [],
  onToggleFavorite
}) => {
  const [searchTerm, setSearchTerm] = useState('')
  const [selectedCategory, setSelectedCategory] = useState<string>('all')
  const [selectedIndicator, setSelectedIndicator] = useState<TechnicalIndicator | null>(null)
  const [isInfoModalVisible, setIsInfoModalVisible] = useState(false)

  const availableIndicators = useMemo(() => {
    return indicatorEngine.getAvailableIndicators()
  }, [])

  const filteredIndicators = useMemo(() => {
    let filtered = availableIndicators

    // Filter by search term
    if (searchTerm) {
      filtered = filtered.filter(indicator =>
        indicator.name.toLowerCase().includes(searchTerm.toLowerCase()) ||
        indicator.id.toLowerCase().includes(searchTerm.toLowerCase())
      )
    }

    // Filter by category
    if (selectedCategory !== 'all') {
      filtered = filtered.filter(indicator => {
        const info = INDICATOR_DESCRIPTIONS[indicator.id]
        if (selectedCategory === 'favorites') {
          return favoriteIndicators.includes(indicator.id)
        }
        return info?.category === selectedCategory || 
               (selectedCategory === 'custom' && indicator.type !== 'built_in')
      })
    }

    return filtered
  }, [availableIndicators, searchTerm, selectedCategory, favoriteIndicators])

  const handleShowInfo = (indicator: TechnicalIndicator) => {
    setSelectedIndicator(indicator)
    setIsInfoModalVisible(true)
  }

  const handleAddIndicator = (indicatorId: string) => {
    onAddIndicator(indicatorId)
  }

  const getIndicatorInfo = (indicatorId: string) => {
    return INDICATOR_DESCRIPTIONS[indicatorId] || {
      description: 'Custom or advanced indicator',
      category: 'custom',
      useCase: 'User-defined analysis'
    }
  }

  const categoryTabs = [
    { id: 'all', name: 'All', count: availableIndicators.length },
    { id: 'favorites', name: 'Favorites', count: favoriteIndicators.length },
    ...INDICATOR_CATEGORIES.map(cat => ({
      ...cat,
      count: availableIndicators.filter(ind => {
        const info = getIndicatorInfo(ind.id)
        return info.category === cat.id || (cat.id === 'custom' && ind.type !== 'built_in')
      }).length
    }))
  ]

  return (
    <div>
      <Card title="Indicator Library" size="small">
        {/* Search */}
        <div style={{ marginBottom: 16 }}>
          <Search
            placeholder="Search indicators..."
            value={searchTerm}
            onChange={(e) => setSearchTerm(e.target.value)}
            prefix={<SearchOutlined />}
            allowClear
          />
        </div>

        {/* Category Tabs */}
        <div style={{ marginBottom: 16 }}>
          <div style={{ display: 'flex', gap: 4, flexWrap: 'wrap' }}>
            {categoryTabs.map(category => (
              <Button
                key={category.id}
                size="small"
                type={selectedCategory === category.id ? 'primary' : 'default'}
                onClick={() => setSelectedCategory(category.id)}
                style={{ 
                  borderColor: (category as any).color || '#1890ff',
                  ...(selectedCategory === category.id && { backgroundColor: (category as any).color || '#1890ff' })
                }}
              >
                {category.name}
                <Badge 
                  count={category.count} 
                  size="small" 
                  style={{ marginLeft: 4 }}
                  showZero={false}
                />
              </Button>
            ))}
          </div>
        </div>

        {/* Indicators Grid */}
        <div style={{ maxHeight: 400, overflowY: 'auto' }}>
          {filteredIndicators.length === 0 ? (
            <div style={{ 
              textAlign: 'center', 
              padding: 40, 
              color: '#999',
              background: '#fafafa',
              borderRadius: 4
            }}>
              <SearchOutlined style={{ fontSize: 24, marginBottom: 8 }} />
              <div>No indicators found</div>
              <div style={{ fontSize: '12px', marginTop: 4 }}>
                Try adjusting your search or category filter
              </div>
            </div>
          ) : (
            <div style={{ display: 'grid', gap: 8 }}>
              {filteredIndicators.map(indicator => {
                const info = getIndicatorInfo(indicator.id)
                const isFavorite = favoriteIndicators.includes(indicator.id)
                const category = INDICATOR_CATEGORIES.find(cat => cat.id === info.category)

                return (
                  <div
                    key={indicator.id}
                    style={{
                      border: '1px solid #e8e8e8',
                      borderRadius: 4,
                      padding: 12,
                      background: '#fff',
                      transition: 'all 0.2s'
                    }}
                    onMouseEnter={(e) => {
                      e.currentTarget.style.borderColor = '#1890ff'
                      e.currentTarget.style.boxShadow = '0 2px 8px rgba(0,0,0,0.1)'
                    }}
                    onMouseLeave={(e) => {
                      e.currentTarget.style.borderColor = '#e8e8e8'
                      e.currentTarget.style.boxShadow = 'none'
                    }}
                  >
                    <div style={{ display: 'flex', alignItems: 'flex-start', justifyContent: 'space-between' }}>
                      <div style={{ flex: 1 }}>
                        <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 4 }}>
                          <Text strong>{indicator.name}</Text>
                          {category && (
                            <Tag 
                              color={category.color} 
                              style={{ fontSize: '10px', padding: '0 4px', margin: 0 }}
                            >
                              {category.name.toUpperCase()}
                            </Tag>
                          )}
                          {indicator.type !== 'built_in' && (
                            <Tag 
                              color="purple" 
                              style={{ fontSize: '10px', padding: '0 4px', margin: 0 }}
                            >
                              {indicator.type.toUpperCase()}
                            </Tag>
                          )}
                        </div>
                        
                        <Paragraph 
                          style={{ 
                            fontSize: '12px', 
                            color: '#666', 
                            margin: '4px 0 8px 0',
                            lineHeight: '1.4'
                          }}
                          ellipsis={{ rows: 2 }}
                        >
                          {info.description}
                        </Paragraph>

                        <div style={{ fontSize: '11px', color: '#999' }}>
                          <strong>Use case:</strong> {info.useCase}
                        </div>
                      </div>

                      <div style={{ display: 'flex', flexDirection: 'column', gap: 4, marginLeft: 8 }}>
                        {onToggleFavorite && (
                          <Button
                            size="small"
                            type="text"
                            icon={<StarOutlined />}
                            onClick={() => onToggleFavorite(indicator.id)}
                            style={{ 
                              color: isFavorite ? '#faad14' : '#d9d9d9',
                              padding: '2px 4px'
                            }}
                          />
                        )}
                        
                        <Tooltip title={`View ${indicator.name} details`}>
                          <Button
                            size="small"
                            type="text"
                            icon={<InfoCircleOutlined />}
                            onClick={() => handleShowInfo(indicator)}
                            style={{ padding: '2px 4px' }}
                          />
                        </Tooltip>

                        <Button
                          size="small"
                          type="primary"
                          icon={<PlusOutlined />}
                          onClick={() => handleAddIndicator(indicator.id)}
                          style={{ padding: '2px 4px' }}
                        >
                          Add
                        </Button>
                      </div>
                    </div>
                  </div>
                )
              })}
            </div>
          )}
        </div>
      </Card>

      {/* Indicator Info Modal */}
      <Modal
        title={selectedIndicator?.name}
        open={isInfoModalVisible}
        onCancel={() => setIsInfoModalVisible(false)}
        footer={[
          <Button key="cancel" onClick={() => setIsInfoModalVisible(false)}>
            Close
          </Button>,
          <Button
            key="add"
            type="primary"
            icon={<PlusOutlined />}
            onClick={() => {
              if (selectedIndicator) {
                handleAddIndicator(selectedIndicator.id)
                setIsInfoModalVisible(false)
              }
            }}
          >
            Add to Chart
          </Button>
        ]}
        width={600}
      >
        {selectedIndicator && (
          <div>
            <div style={{ marginBottom: 16 }}>
              <Text strong>Description:</Text>
              <Paragraph style={{ marginTop: 8 }}>
                {getIndicatorInfo(selectedIndicator.id).description}
              </Paragraph>
            </div>

            <div style={{ marginBottom: 16 }}>
              <Text strong>Parameters:</Text>
              <div style={{ marginTop: 8 }}>
                {selectedIndicator.parameters.map(param => (
                  <div key={param.name} style={{ marginBottom: 8 }}>
                    <Tag color="blue">{param.name}</Tag>
                    <Text style={{ fontSize: '12px' }}>
                      {param.type} 
                      {param.defaultValue !== undefined && ` (default: ${param.defaultValue})`}
                      {param.min !== undefined && param.max !== undefined && 
                        ` (range: ${param.min}-${param.max})`
                      }
                    </Text>
                  </div>
                ))}
              </div>
            </div>

            <div style={{ marginBottom: 16 }}>
              <Text strong>Display:</Text>
              <div style={{ marginTop: 8 }}>
                <Tag color={selectedIndicator.display.overlay ? 'green' : 'orange'}>
                  {selectedIndicator.display.overlay ? 'Price Overlay' : 'Separate Panel'}
                </Tag>
                <Tag color="blue">
                  Line Width: {selectedIndicator.display.lineWidth}
                </Tag>
                <Tag color="purple">
                  Style: {selectedIndicator.display.style}
                </Tag>
              </div>
            </div>

            <div>
              <Text strong>Use Case:</Text>
              <Paragraph style={{ marginTop: 8 }}>
                {getIndicatorInfo(selectedIndicator.id).useCase}
              </Paragraph>
            </div>
          </div>
        )}
      </Modal>
    </div>
  )
}

export default IndicatorLibrary