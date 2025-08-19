import React from 'react'
import { Space, Statistic, Tag, Tooltip } from 'antd'
import { ArrowUpOutlined, ArrowDownOutlined, InfoCircleOutlined } from '@ant-design/icons'
import { OrderBookSpread, OrderBookDisplaySettings } from '../../types/orderBook'

interface OrderBookHeaderProps {
  symbol: string
  venue: string
  spread: OrderBookSpread
  displaySettings: OrderBookDisplaySettings
  connectionStatus: 'connected' | 'disconnected' | 'error'
  lastUpdateTime?: number
  className?: string
}

export const OrderBookHeader: React.FC<OrderBookHeaderProps> = ({
  symbol,
  venue,
  spread,
  displaySettings,
  connectionStatus,
  lastUpdateTime,
  className
}) => {
  const { decimals } = displaySettings

  const formatPrice = (price: number | null) => {
    if (price === null) return 'N/A'
    return price.toFixed(decimals)
  }

  const formatSpread = (spreadValue: number | null) => {
    if (spreadValue === null) return 'N/A'
    return spreadValue.toFixed(decimals)
  }

  const formatSpreadPercentage = (percentage: number | null) => {
    if (percentage === null) return 'N/A'
    return `${percentage.toFixed(4)}%`
  }

  const getConnectionStatusColor = () => {
    switch (connectionStatus) {
      case 'connected': return 'green'
      case 'disconnected': return 'orange'
      case 'error': return 'red'
      default: return 'default'
    }
  }

  const getConnectionStatusText = () => {
    switch (connectionStatus) {
      case 'connected': return 'Live'
      case 'disconnected': return 'Offline'
      case 'error': return 'Error'
      default: return 'Unknown'
    }
  }

  const formatLastUpdate = (timestamp?: number) => {
    if (!timestamp) return 'Never'
    const now = Date.now()
    const diff = now - timestamp
    
    if (diff < 1000) return 'Just now'
    if (diff < 60000) return `${Math.floor(diff / 1000)}s ago`
    if (diff < 3600000) return `${Math.floor(diff / 60000)}m ago`
    return new Date(timestamp).toLocaleTimeString()
  }

  return (
    <div className={className} style={{ 
      padding: '12px 16px', 
      borderBottom: '1px solid #f0f0f0',
      backgroundColor: '#fafafa'
    }}>
      {/* Title and Connection Status */}
      <div style={{ 
        display: 'flex', 
        justifyContent: 'space-between', 
        alignItems: 'center',
        marginBottom: '12px'
      }}>
        <div>
          <h4 style={{ margin: 0, fontSize: '16px', fontWeight: 'bold' }}>
            {symbol}
            <span style={{ fontSize: '12px', color: '#888', marginLeft: '8px' }}>
              {venue}
            </span>
          </h4>
        </div>
        
        <Space size="small">
          <Tag color={getConnectionStatusColor()}>
            {getConnectionStatusText()}
          </Tag>
          {lastUpdateTime && (
            <span style={{ fontSize: '11px', color: '#999' }}>
              {formatLastUpdate(lastUpdateTime)}
            </span>
          )}
        </Space>
      </div>

      {/* Spread Information */}
      <div style={{ 
        display: 'flex', 
        justifyContent: 'space-between', 
        alignItems: 'center',
        gap: '16px'
      }}>
        {/* Best Bid */}
        <div style={{ flex: 1, textAlign: 'center' }}>
          <Statistic
            title={
              <span style={{ fontSize: '11px', color: '#666' }}>
                Best Bid
                <Tooltip title="Highest price buyers are willing to pay">
                  <InfoCircleOutlined style={{ marginLeft: '4px', fontSize: '10px' }} />
                </Tooltip>
              </span>
            }
            value={formatPrice(spread.bestBid)}
            precision={0}
            valueStyle={{ 
              fontSize: '14px', 
              fontWeight: 'bold',
              color: '#52c41a'
            }}
            prefix={<ArrowUpOutlined style={{ fontSize: '12px' }} />}
          />
        </div>

        {/* Spread */}
        <div style={{ flex: 1, textAlign: 'center' }}>
          <Statistic
            title={
              <span style={{ fontSize: '11px', color: '#666' }}>
                Spread
                <Tooltip title="Difference between best ask and best bid">
                  <InfoCircleOutlined style={{ marginLeft: '4px', fontSize: '10px' }} />
                </Tooltip>
              </span>
            }
            value={formatSpread(spread.spread)}
            precision={0}
            valueStyle={{ 
              fontSize: '12px',
              fontWeight: 'bold',
              color: '#1890ff'
            }}
            suffix={
              spread.spreadPercentage !== null && (
                <span style={{ fontSize: '10px', color: '#999' }}>
                  ({formatSpreadPercentage(spread.spreadPercentage)})
                </span>
              )
            }
          />
        </div>

        {/* Best Ask */}
        <div style={{ flex: 1, textAlign: 'center' }}>
          <Statistic
            title={
              <span style={{ fontSize: '11px', color: '#666' }}>
                Best Ask
                <Tooltip title="Lowest price sellers are willing to accept">
                  <InfoCircleOutlined style={{ marginLeft: '4px', fontSize: '10px' }} />
                </Tooltip>
              </span>
            }
            value={formatPrice(spread.bestAsk)}
            precision={0}
            valueStyle={{ 
              fontSize: '14px', 
              fontWeight: 'bold',
              color: '#ff4d4f'
            }}
            prefix={<ArrowDownOutlined style={{ fontSize: '12px' }} />}
          />
        </div>
      </div>

      {/* Column Headers */}
      <div style={{ 
        display: 'flex', 
        justifyContent: 'space-between',
        marginTop: '16px',
        paddingTop: '8px',
        borderTop: '1px solid #e8e8e8',
        fontSize: '11px',
        fontWeight: 'bold',
        color: '#666'
      }}>
        {/* Bid side headers */}
        <div style={{ 
          flex: 1, 
          display: 'flex', 
          justifyContent: 'space-between',
          paddingRight: '16px'
        }}>
          <span style={{ color: '#52c41a' }}>Quantity</span>
          <span style={{ color: '#52c41a' }}>Bid Price</span>
        </div>

        {/* Ask side headers */}
        <div style={{ 
          flex: 1, 
          display: 'flex', 
          justifyContent: 'space-between',
          paddingLeft: '16px'
        }}>
          <span style={{ color: '#ff4d4f' }}>Ask Price</span>
          <span style={{ color: '#ff4d4f' }}>Quantity</span>
        </div>
      </div>
    </div>
  )
}