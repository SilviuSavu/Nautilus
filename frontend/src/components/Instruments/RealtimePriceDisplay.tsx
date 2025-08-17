import React from 'react'
import { Space, Tag, Tooltip, Typography, Badge, Statistic } from 'antd'
import { 
  ArrowUpOutlined, 
  ArrowDownOutlined, 
  DashOutlined,
  WifiOutlined,
  DisconnectOutlined,
  ClockCircleOutlined
} from '@ant-design/icons'
import { useRealtimePrice } from './hooks/useRealtime'
import { Instrument } from './types/instrumentTypes'

const { Text } = Typography

interface RealtimePriceDisplayProps {
  instrument: Instrument
  showVolume?: boolean
  showBidAsk?: boolean
  compact?: boolean
  className?: string
}

export const RealtimePriceDisplay: React.FC<RealtimePriceDisplayProps> = ({
  instrument,
  showVolume = true,
  showBidAsk = false,
  compact = false,
  className
}) => {
  const { price, lastUpdate } = useRealtimePrice(instrument.id)

  const formatPrice = (value: number) => {
    return new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency: instrument.currency,
      minimumFractionDigits: 2,
      maximumFractionDigits: 4
    }).format(value)
  }

  const formatVolume = (volume: number) => {
    if (volume >= 1000000) {
      return `${(volume / 1000000).toFixed(1)}M`
    }
    if (volume >= 1000) {
      return `${(volume / 1000).toFixed(1)}K`
    }
    return volume.toString()
  }

  const getChangeColor = (change: number) => {
    if (change > 0) return '#52c41a'
    if (change < 0) return '#ff4d4f'
    return '#8c8c8c'
  }

  const getChangeIcon = (change: number) => {
    if (change > 0) return <ArrowUpOutlined />
    if (change < 0) return <ArrowDownOutlined />
    return <DashOutlined />
  }

  const formatTime = (timestamp: string) => {
    return new Date(timestamp).toLocaleTimeString('en-US', {
      hour12: false,
      hour: '2-digit',
      minute: '2-digit',
      second: '2-digit'
    })
  }

  const getConnectionIndicator = () => {
    const isStale = lastUpdate ? 
      (Date.now() - new Date(lastUpdate).getTime()) > 30000 : true // 30 seconds

    if (!price || isStale) {
      return (
        <Tooltip title="No real-time data available">
          <DisconnectOutlined style={{ color: '#8c8c8c' }} />
        </Tooltip>
      )
    }

    return (
      <Tooltip title={`Live data • Last: ${formatTime(lastUpdate!)}`}>
        <WifiOutlined style={{ color: '#52c41a' }} />
      </Tooltip>
    )
  }

  if (compact) {
    return (
      <div className={className}>
        <Space size="small" style={{ fontSize: '12px' }}>
          {getConnectionIndicator()}
          {price ? (
            <>
              <Text strong style={{ color: getChangeColor(price.change) }}>
                {formatPrice(price.last)}
              </Text>
              <Text style={{ color: getChangeColor(price.change) }}>
                {getChangeIcon(price.change)}
                {price.changePercent.toFixed(2)}%
              </Text>
            </>
          ) : (
            <Text type="secondary">No data</Text>
          )}
        </Space>
      </div>
    )
  }

  if (!price) {
    return (
      <div className={className}>
        <Space direction="vertical" size="small" style={{ width: '100%' }}>
          <Space>
            {getConnectionIndicator()}
            <Text type="secondary">No real-time data available</Text>
          </Space>
          <Text type="secondary" style={{ fontSize: '11px' }}>
            {instrument.symbol} • {instrument.venue}
          </Text>
        </Space>
      </div>
    )
  }

  return (
    <div className={className}>
      <Space direction="vertical" size="small" style={{ width: '100%' }}>
        {/* Header with connection status */}
        <Space style={{ width: '100%', justifyContent: 'space-between' }}>
          <Space>
            {getConnectionIndicator()}
            <Text strong>{instrument.symbol}</Text>
            <Tag size="small">{instrument.venue}</Tag>
          </Space>
          {lastUpdate && (
            <Text type="secondary" style={{ fontSize: '11px' }}>
              <ClockCircleOutlined /> {formatTime(lastUpdate)}
            </Text>
          )}
        </Space>

        {/* Main price display */}
        <Space align="center">
          <Statistic
            value={price.last}
            precision={2}
            prefix={instrument.currency === 'USD' ? '$' : ''}
            suffix={instrument.currency !== 'USD' ? instrument.currency : ''}
            valueStyle={{ 
              color: getChangeColor(price.change),
              fontSize: '20px',
              fontWeight: 'bold'
            }}
          />
          <Space direction="vertical" size={0}>
            <Text style={{ color: getChangeColor(price.change) }}>
              {getChangeIcon(price.change)}
              {price.change >= 0 ? '+' : ''}{price.change.toFixed(2)}
            </Text>
            <Text style={{ color: getChangeColor(price.change), fontSize: '12px' }}>
              ({price.changePercent >= 0 ? '+' : ''}{price.changePercent.toFixed(2)}%)
            </Text>
          </Space>
        </Space>

        {/* Bid/Ask spread */}
        {showBidAsk && (
          <Space>
            <Space direction="vertical" size={0}>
              <Text type="secondary" style={{ fontSize: '11px' }}>Bid</Text>
              <Text style={{ fontSize: '13px', color: '#ff4d4f' }}>
                {formatPrice(price.bid)}
              </Text>
            </Space>
            <Space direction="vertical" size={0}>
              <Text type="secondary" style={{ fontSize: '11px' }}>Ask</Text>
              <Text style={{ fontSize: '13px', color: '#52c41a' }}>
                {formatPrice(price.ask)}
              </Text>
            </Space>
            <Space direction="vertical" size={0}>
              <Text type="secondary" style={{ fontSize: '11px' }}>Spread</Text>
              <Text style={{ fontSize: '13px' }}>
                {formatPrice(price.ask - price.bid)}
              </Text>
            </Space>
          </Space>
        )}

        {/* Volume */}
        {showVolume && (
          <Space>
            <Text type="secondary" style={{ fontSize: '12px' }}>
              Volume: <Text strong>{formatVolume(price.volume)}</Text>
            </Text>
          </Space>
        )}
      </Space>
    </div>
  )
}

// Simplified component for use in lists and tables
interface PriceChangeIndicatorProps {
  instrument: Instrument
  showPercentage?: boolean
}

export const PriceChangeIndicator: React.FC<PriceChangeIndicatorProps> = ({
  instrument,
  showPercentage = true
}) => {
  const { price } = useRealtimePrice(instrument.id)

  if (!price) {
    return <Badge status="default" text="No data" />
  }

  const color = getChangeColor(price.change)
  const icon = getChangeIcon(price.change)

  return (
    <Space size="small">
      <span style={{ color }}>
        {icon}
        {price.change >= 0 ? '+' : ''}{price.change.toFixed(2)}
      </span>
      {showPercentage && (
        <span style={{ color, fontSize: '11px' }}>
          ({price.changePercent >= 0 ? '+' : ''}{price.changePercent.toFixed(1)}%)
        </span>
      )}
    </Space>
  )
}

// Component for real-time price in watchlist items
interface WatchlistPriceProps {
  instrument: Instrument
}

export const WatchlistPrice: React.FC<WatchlistPriceProps> = ({ instrument }) => {
  const { price, lastUpdate } = useRealtimePrice(instrument.id)

  if (!price) {
    return (
      <Space direction="vertical" size={0}>
        <Text type="secondary" style={{ fontSize: '12px' }}>No data</Text>
        <Badge status="default" />
      </Space>
    )
  }

  const isStale = lastUpdate ? 
    (Date.now() - new Date(lastUpdate).getTime()) > 30000 : true

  return (
    <Space direction="vertical" size={0} style={{ textAlign: 'right' }}>
      <Text strong style={{ color: getChangeColor(price.change) }}>
        {new Intl.NumberFormat('en-US', {
          style: 'currency',
          currency: instrument.currency,
          minimumFractionDigits: 2,
          maximumFractionDigits: 2
        }).format(price.last)}
      </Text>
      <Space size="small">
        <span style={{ color: getChangeColor(price.change), fontSize: '11px' }}>
          {getChangeIcon(price.change)}
          {price.changePercent >= 0 ? '+' : ''}{price.changePercent.toFixed(1)}%
        </span>
        <Badge 
          status={isStale ? "default" : "success"} 
          title={isStale ? "Data may be stale" : "Live data"}
        />
      </Space>
    </Space>
  )
}

function getChangeColor(change: number): string {
  if (change > 0) return '#52c41a'
  if (change < 0) return '#ff4d4f'
  return '#8c8c8c'
}

function getChangeIcon(change: number) {
  if (change > 0) return <ArrowUpOutlined />
  if (change < 0) return <ArrowDownOutlined />
  return <DashOutlined />
}