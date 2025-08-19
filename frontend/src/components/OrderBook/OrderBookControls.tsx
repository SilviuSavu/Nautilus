import React from 'react'
import { 
  Space, 
  Switch, 
  Select, 
  InputNumber, 
  Tooltip, 
  Divider,
  Button,
  Popover
} from 'antd'
import { 
  SettingOutlined, 
  InfoCircleOutlined,
  CompressOutlined,
  ExpandOutlined
} from '@ant-design/icons'
import { 
  OrderBookAggregationSettings, 
  OrderBookDisplaySettings,
  OrderBookPerformanceMetrics
} from '../../types/orderBook'

const { Option } = Select

interface OrderBookControlsProps {
  aggregationSettings: OrderBookAggregationSettings
  displaySettings: OrderBookDisplaySettings
  onAggregationChange: (settings: Partial<OrderBookAggregationSettings>) => void
  onDisplayChange: (settings: Partial<OrderBookDisplaySettings>) => void
  performanceMetrics?: OrderBookPerformanceMetrics | null
  className?: string
}

export const OrderBookControls: React.FC<OrderBookControlsProps> = ({
  aggregationSettings,
  displaySettings,
  onAggregationChange,
  onDisplayChange,
  performanceMetrics,
  className
}) => {
  const { enabled, increment, maxLevels } = aggregationSettings
  const { showSpread, showOrderCount, colorScheme, decimals } = displaySettings

  const incrementOptions = [
    { value: 0.001, label: '0.001' },
    { value: 0.01, label: '0.01' },
    { value: 0.05, label: '0.05' },
    { value: 0.1, label: '0.1' },
    { value: 0.25, label: '0.25' },
    { value: 0.5, label: '0.5' },
    { value: 1, label: '1.0' },
    { value: 5, label: '5.0' },
    { value: 10, label: '10.0' }
  ]

  const levelOptions = [
    { value: 5, label: '5 levels' },
    { value: 10, label: '10 levels' },
    { value: 15, label: '15 levels' },
    { value: 20, label: '20 levels' },
    { value: 30, label: '30 levels' },
    { value: 50, label: '50 levels' }
  ]

  const colorSchemeOptions = [
    { value: 'default', label: 'Default' },
    { value: 'dark', label: 'Dark' },
    { value: 'light', label: 'Light' }
  ]

  const decimalsOptions = [
    { value: 0, label: '0' },
    { value: 1, label: '1' },
    { value: 2, label: '2' },
    { value: 3, label: '3' },
    { value: 4, label: '4' },
    { value: 5, label: '5' }
  ]

  const formatLatency = (latency: number) => {
    return `${latency.toFixed(1)}ms`
  }

  const performanceContent = performanceMetrics && (
    <div style={{ minWidth: '200px' }}>
      <h4 style={{ margin: '0 0 8px 0' }}>Performance Metrics</h4>
      <div style={{ fontSize: '12px', lineHeight: '18px' }}>
        <div><strong>Avg Latency:</strong> {formatLatency(performanceMetrics.averageLatency)}</div>
        <div><strong>Max Latency:</strong> {formatLatency(performanceMetrics.maxLatency)}</div>
        <div><strong>Updates/sec:</strong> {performanceMetrics.updatesPerSecond.toFixed(1)}</div>
        <div><strong>Last Update:</strong> {new Date(performanceMetrics.lastUpdateTime).toLocaleTimeString()}</div>
      </div>
    </div>
  )

  const settingsContent = (
    <div style={{ minWidth: '280px', maxWidth: '400px' }}>
      <h4 style={{ margin: '0 0 12px 0' }}>Order Book Settings</h4>
      
      {/* Aggregation Settings */}
      <div style={{ marginBottom: '16px' }}>
        <h5 style={{ margin: '0 0 8px 0', fontSize: '13px' }}>Price Aggregation</h5>
        
        <div style={{ marginBottom: '8px' }}>
          <Space>
            <Switch
              checked={enabled}
              onChange={(checked) => onAggregationChange({ enabled: checked })}
              size="small"
            />
            <span style={{ fontSize: '12px' }}>Enable Aggregation</span>
            <Tooltip title="Group price levels to reduce visual noise">
              <InfoCircleOutlined style={{ fontSize: '12px', color: '#999' }} />
            </Tooltip>
          </Space>
        </div>

        {enabled && (
          <Space direction="vertical" size="small" style={{ width: '100%' }}>
            <div>
              <span style={{ fontSize: '12px', display: 'block', marginBottom: '4px' }}>
                Increment:
              </span>
              <Select
                value={increment}
                onChange={(value) => onAggregationChange({ increment: value })}
                size="small"
                style={{ width: '100px' }}
              >
                {incrementOptions.map(option => (
                  <Option key={option.value} value={option.value}>
                    {option.label}
                  </Option>
                ))}
              </Select>
            </div>
            
            <div>
              <span style={{ fontSize: '12px', display: 'block', marginBottom: '4px' }}>
                Max Levels:
              </span>
              <Select
                value={maxLevels}
                onChange={(value) => onAggregationChange({ maxLevels: value })}
                size="small"
                style={{ width: '120px' }}
              >
                {levelOptions.map(option => (
                  <Option key={option.value} value={option.value}>
                    {option.label}
                  </Option>
                ))}
              </Select>
            </div>
          </Space>
        )}
      </div>

      <Divider style={{ margin: '12px 0' }} />

      {/* Display Settings */}
      <div style={{ marginBottom: '16px' }}>
        <h5 style={{ margin: '0 0 8px 0', fontSize: '13px' }}>Display Options</h5>
        
        <Space direction="vertical" size="small" style={{ width: '100%' }}>
          <div>
            <Space>
              <Switch
                checked={showSpread}
                onChange={(checked) => onDisplayChange({ showSpread: checked })}
                size="small"
              />
              <span style={{ fontSize: '12px' }}>Show Spread</span>
            </Space>
          </div>
          
          <div>
            <Space>
              <Switch
                checked={showOrderCount}
                onChange={(checked) => onDisplayChange({ showOrderCount: checked })}
                size="small"
              />
              <span style={{ fontSize: '12px' }}>Show Order Count</span>
            </Space>
          </div>
          
          <div>
            <span style={{ fontSize: '12px', display: 'block', marginBottom: '4px' }}>
              Color Scheme:
            </span>
            <Select
              value={colorScheme}
              onChange={(value) => onDisplayChange({ colorScheme: value })}
              size="small"
              style={{ width: '100px' }}
            >
              {colorSchemeOptions.map(option => (
                <Option key={option.value} value={option.value}>
                  {option.label}
                </Option>
              ))}
            </Select>
          </div>
          
          <div>
            <span style={{ fontSize: '12px', display: 'block', marginBottom: '4px' }}>
              Price Decimals:
            </span>
            <Select
              value={decimals}
              onChange={(value) => onDisplayChange({ decimals: value })}
              size="small"
              style={{ width: '80px' }}
            >
              {decimalsOptions.map(option => (
                <Option key={option.value} value={option.value}>
                  {option.label}
                </Option>
              ))}
            </Select>
          </div>
        </Space>
      </div>
    </div>
  )

  return (
    <div className={className} style={{ 
      padding: '8px 16px', 
      borderBottom: '1px solid #f0f0f0',
      backgroundColor: '#fafafa',
      display: 'flex',
      justifyContent: 'space-between',
      alignItems: 'center'
    }}>
      {/* Quick Controls */}
      <Space size="small">
        <Tooltip title="Toggle price aggregation">
          <Button
            type={enabled ? 'primary' : 'default'}
            size="small"
            icon={enabled ? <CompressOutlined /> : <ExpandOutlined />}
            onClick={() => onAggregationChange({ enabled: !enabled })}
          >
            {enabled ? 'Grouped' : 'Raw'}
          </Button>
        </Tooltip>

        {enabled && (
          <Select
            value={increment}
            onChange={(value) => onAggregationChange({ increment: value })}
            size="small"
            style={{ width: '70px' }}
            bordered={false}
          >
            {incrementOptions.map(option => (
              <Option key={option.value} value={option.value}>
                {option.label}
              </Option>
            ))}
          </Select>
        )}

        <Select
          value={maxLevels}
          onChange={(value) => onAggregationChange({ maxLevels: value })}
          size="small"
          style={{ width: '60px' }}
          bordered={false}
        >
          {levelOptions.map(option => (
            <Option key={option.value} value={option.value}>
              {option.value}
            </Option>
          ))}
        </Select>
      </Space>

      {/* Settings and Performance */}
      <Space size="small">
        {performanceMetrics && (
          <Popover 
            content={performanceContent} 
            title={null}
            trigger="hover"
            placement="bottomRight"
          >
            <Button
              type="text"
              size="small"
              style={{ fontSize: '11px', padding: '0 4px', color: '#666' }}
            >
              {formatLatency(performanceMetrics.averageLatency)} avg
            </Button>
          </Popover>
        )}

        <Popover 
          content={settingsContent} 
          title={null}
          trigger="click"
          placement="bottomRight"
        >
          <Button
            type="text"
            size="small"
            icon={<SettingOutlined />}
          />
        </Popover>
      </Space>
    </div>
  )
}