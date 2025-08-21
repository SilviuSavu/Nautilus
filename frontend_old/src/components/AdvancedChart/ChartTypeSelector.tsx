/**
 * Chart Type Selector Component
 * Provides interface for selecting different chart types including advanced types
 */

import React from 'react'
import { Select, Card, Space, Tag, Tooltip } from 'antd'
import { BarChartOutlined, LineChartOutlined, AreaChartOutlined } from '@ant-design/icons'
import { ChartType } from '../../types/charting'

const { Option, OptGroup } = Select

interface ChartTypeOption {
  value: ChartType
  label: string
  description: string
  icon?: React.ReactNode
  category: 'basic' | 'advanced' | 'volume'
  complexity: 'simple' | 'intermediate' | 'advanced'
}

const CHART_TYPE_OPTIONS: ChartTypeOption[] = [
  {
    value: 'candlestick',
    label: 'Candlestick',
    description: 'Traditional OHLC candlestick chart',
    icon: <BarChartOutlined />,
    category: 'basic',
    complexity: 'simple'
  },
  {
    value: 'line',
    label: 'Line Chart',
    description: 'Simple line chart showing closing prices',
    icon: <LineChartOutlined />,
    category: 'basic',
    complexity: 'simple'
  },
  {
    value: 'area',
    label: 'Area Chart',
    description: 'Filled area chart based on closing prices',
    icon: <AreaChartOutlined />,
    category: 'basic',
    complexity: 'simple'
  },
  {
    value: 'renko',
    label: 'Renko Bricks',
    description: 'Fixed-size bricks that ignore time, focusing only on price movement',
    category: 'advanced',
    complexity: 'intermediate'
  },
  {
    value: 'point_figure',
    label: 'Point & Figure',
    description: 'X and O pattern chart filtering out small price movements',
    category: 'advanced',
    complexity: 'intermediate'
  },
  {
    value: 'heikin_ashi',
    label: 'Heikin Ashi',
    description: 'Modified candlestick chart that filters noise and shows trend clearly',
    category: 'advanced',
    complexity: 'intermediate'
  },
  {
    value: 'volume_profile',
    label: 'Volume Profile',
    description: 'Horizontal histogram showing volume distribution at price levels',
    category: 'volume',
    complexity: 'advanced'
  }
]

interface ChartTypeSelectorProps {
  selectedType: ChartType
  onTypeChange: (type: ChartType) => void
  disabled?: boolean
  size?: 'small' | 'middle' | 'large'
  showDetails?: boolean
}

export const ChartTypeSelector: React.FC<ChartTypeSelectorProps> = ({
  selectedType,
  onTypeChange,
  disabled = false,
  size = 'middle',
  showDetails = false
}) => {
  const selectedOption = CHART_TYPE_OPTIONS.find(option => option.value === selectedType)

  const getComplexityColor = (complexity: string) => {
    switch (complexity) {
      case 'simple': return 'green'
      case 'intermediate': return 'orange'
      case 'advanced': return 'red'
      default: return 'default'
    }
  }

  const getCategoryColor = (category: string) => {
    switch (category) {
      case 'basic': return 'blue'
      case 'advanced': return 'purple'
      case 'volume': return 'cyan'
      default: return 'default'
    }
  }

  if (showDetails) {
    return (
      <Card title="Chart Type" size="small">
        <div style={{ marginBottom: 16 }}>
          <Select
            value={selectedType}
            onChange={onTypeChange}
            disabled={disabled}
            size={size}
            style={{ width: '100%' }}
            placeholder="Select chart type"
          >
            <OptGroup label="Basic Charts">
              {CHART_TYPE_OPTIONS.filter(opt => opt.category === 'basic').map(option => (
                <Option key={option.value} value={option.value}>
                  <Space>
                    {option.icon}
                    {option.label}
                  </Space>
                </Option>
              ))}
            </OptGroup>
            
            <OptGroup label="Advanced Charts">
              {CHART_TYPE_OPTIONS.filter(opt => opt.category === 'advanced').map(option => (
                <Option key={option.value} value={option.value}>
                  <Space>
                    {option.label}
                    <Tag color={getComplexityColor(option.complexity)}>
                      {option.complexity}
                    </Tag>
                  </Space>
                </Option>
              ))}
            </OptGroup>
            
            <OptGroup label="Volume Charts">
              {CHART_TYPE_OPTIONS.filter(opt => opt.category === 'volume').map(option => (
                <Option key={option.value} value={option.value}>
                  <Space>
                    {option.label}
                    <Tag color={getComplexityColor(option.complexity)}>
                      {option.complexity}
                    </Tag>
                  </Space>
                </Option>
              ))}
            </OptGroup>
          </Select>
        </div>

        {/* Selected Chart Info */}
        {selectedOption && (
          <div style={{ 
            padding: 12, 
            background: '#f5f5f5', 
            borderRadius: 4,
            border: '1px solid #e8e8e8'
          }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 8 }}>
              <span style={{ fontWeight: 500 }}>{selectedOption.label}</span>
              <Tag color={getCategoryColor(selectedOption.category)}>
                {selectedOption.category}
              </Tag>
              <Tag color={getComplexityColor(selectedOption.complexity)}>
                {selectedOption.complexity}
              </Tag>
            </div>
            <div style={{ fontSize: '12px', color: '#666', lineHeight: '1.4' }}>
              {selectedOption.description}
            </div>
          </div>
        )}
      </Card>
    )
  }

  return (
    <Tooltip title={selectedOption?.description || 'Select chart type'}>
      <Select
        value={selectedType}
        onChange={onTypeChange}
        disabled={disabled}
        size={size}
        style={{ minWidth: 150 }}
        placeholder="Chart Type"
      >
        <OptGroup label="Basic">
          {CHART_TYPE_OPTIONS.filter(opt => opt.category === 'basic').map(option => (
            <Option key={option.value} value={option.value}>
              <Space size="small">
                {option.icon}
                {option.label}
              </Space>
            </Option>
          ))}
        </OptGroup>
        
        <OptGroup label="Advanced">
          {CHART_TYPE_OPTIONS.filter(opt => opt.category === 'advanced').map(option => (
            <Option key={option.value} value={option.value}>
              {option.label}
            </Option>
          ))}
        </OptGroup>
        
        <OptGroup label="Volume">
          {CHART_TYPE_OPTIONS.filter(opt => opt.category === 'volume').map(option => (
            <Option key={option.value} value={option.value}>
              {option.label}
            </Option>
          ))}
        </OptGroup>
      </Select>
    </Tooltip>
  )
}

export default ChartTypeSelector