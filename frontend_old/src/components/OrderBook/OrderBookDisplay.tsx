import React, { useMemo } from 'react'
import { Card, Empty, Alert, Spin, Space } from 'antd'
import { WifiOutlined, DisconnectOutlined } from '@ant-design/icons'
import { OrderBookHeader } from './OrderBookHeader'
import { OrderBookLevel } from './OrderBookLevel'
import { OrderBookControls } from './OrderBookControls'
import { useOrderBookData } from '../../hooks/useOrderBookData'
import { Instrument } from '../Chart/types/chartTypes'
import { ProcessedOrderBookLevel } from '../../types/orderBook'

interface OrderBookDisplayProps {
  instrument: Instrument | null
  height?: number
  maxUpdatesPerSecond?: number
  autoSubscribe?: boolean
  onLevelClick?: (level: ProcessedOrderBookLevel, side: 'bid' | 'ask') => void
  className?: string
}

export const OrderBookDisplay: React.FC<OrderBookDisplayProps> = ({
  instrument,
  height = 600,
  maxUpdatesPerSecond = 10,
  autoSubscribe = true,
  onLevelClick,
  className
}) => {
  const {
    orderBookData,
    aggregationSettings,
    displaySettings,
    isLoading,
    error,
    connectionStatus,
    performanceMetrics,
    subscribeToOrderBook,
    unsubscribeFromOrderBook,
    updateAggregationSettings,
    updateDisplaySettings,
    clearOrderBook
  } = useOrderBookData({
    maxUpdatesPerSecond,
    autoSubscribe,
    enablePerformanceTracking: true
  })

  // Subscribe/unsubscribe when instrument changes
  React.useEffect(() => {
    if (instrument && autoSubscribe) {
      subscribeToOrderBook(instrument)
    } else if (!instrument) {
      clearOrderBook()
    }

    // Cleanup on unmount or instrument change
    return () => {
      if (instrument) {
        unsubscribeFromOrderBook(instrument)
      }
    }
  }, [instrument, autoSubscribe, subscribeToOrderBook, unsubscribeFromOrderBook, clearOrderBook])

  // Calculate max quantities for depth visualization
  const maxQuantities = useMemo(() => {
    if (!orderBookData) {
      return { bid: 0, ask: 0 }
    }

    const maxBidQuantity = orderBookData.bids.length > 0 
      ? Math.max(...orderBookData.bids.map(level => level.quantity))
      : 0

    const maxAskQuantity = orderBookData.asks.length > 0
      ? Math.max(...orderBookData.asks.map(level => level.quantity))
      : 0

    return {
      bid: maxBidQuantity,
      ask: maxAskQuantity
    }
  }, [orderBookData])

  // Handle error display
  if (error) {
    return (
      <Card className={className} title="Order Book">
        <Alert
          type="error"
          message="Order Book Error"
          description={error}
          showIcon
        />
      </Card>
    )
  }

  // Handle no instrument selected
  if (!instrument) {
    return (
      <Card 
        className={className} 
        title="Order Book"
        style={{ height }}
      >
        <div style={{ 
          display: 'flex', 
          justifyContent: 'center', 
          alignItems: 'center', 
          height: height - 100,
          flexDirection: 'column'
        }}>
          <Empty
            description="Select an instrument to view order book"
            image={Empty.PRESENTED_IMAGE_SIMPLE}
          />
        </div>
      </Card>
    )
  }

  // Calculate content height for scrolling
  const headerHeight = 140
  const controlsHeight = 50
  const contentHeight = height - headerHeight - controlsHeight

  return (
    <Card 
      className={className}
      bodyStyle={{ padding: 0 }}
      style={{ height }}
    >
      {/* Header with spread information */}
      <OrderBookHeader
        symbol={instrument.symbol}
        venue={instrument.venue}
        spread={orderBookData?.spread || { bestBid: null, bestAsk: null, spread: null, spreadPercentage: null }}
        displaySettings={displaySettings}
        connectionStatus={connectionStatus}
        lastUpdateTime={orderBookData?.timestamp}
      />

      {/* Controls */}
      <OrderBookControls
        aggregationSettings={aggregationSettings}
        displaySettings={displaySettings}
        onAggregationChange={updateAggregationSettings}
        onDisplayChange={updateDisplaySettings}
        performanceMetrics={performanceMetrics}
      />

      {/* Order Book Content */}
      <div style={{ height: contentHeight, position: 'relative' }}>
        {isLoading && !orderBookData ? (
          <div style={{ 
            display: 'flex', 
            justifyContent: 'center', 
            alignItems: 'center', 
            height: '100%',
            background: '#fafafa'
          }}>
            <Space direction="vertical" align="center">
              <Spin size="large" />
              <span style={{ fontSize: '12px', color: '#999' }}>
                Loading order book data...
              </span>
            </Space>
          </div>
        ) : !orderBookData || (orderBookData.bids.length === 0 && orderBookData.asks.length === 0) ? (
          <div style={{ 
            display: 'flex', 
            justifyContent: 'center', 
            alignItems: 'center', 
            height: '100%',
            flexDirection: 'column'
          }}>
            <Empty
              description={
                connectionStatus === 'connected' 
                  ? "No order book data available" 
                  : "Waiting for connection..."
              }
              image={Empty.PRESENTED_IMAGE_SIMPLE}
            />
            {connectionStatus !== 'connected' && (
              <div style={{ marginTop: '16px', fontSize: '12px', color: '#999' }}>
                <DisconnectOutlined style={{ marginRight: '4px' }} />
                Connection status: {connectionStatus}
              </div>
            )}
          </div>
        ) : (
          <div style={{ 
            display: 'flex', 
            height: '100%',
            fontFamily: 'monospace'
          }}>
            {/* Bid Side (Left) */}
            <div style={{ 
              flex: 1, 
              borderRight: '1px solid #f0f0f0',
              display: 'flex',
              flexDirection: 'column'
            }}>
              <div style={{ 
                flex: 1, 
                overflowY: 'auto',
                display: 'flex',
                flexDirection: 'column-reverse' // Highest bids at bottom
              }}>
                {orderBookData.bids.map((level, index) => (
                  <OrderBookLevel
                    key={level.id}
                    level={level}
                    side="bid"
                    displaySettings={displaySettings}
                    maxQuantity={maxQuantities.bid}
                    onLevelClick={onLevelClick}
                  />
                ))}
              </div>
            </div>

            {/* Ask Side (Right) */}
            <div style={{ 
              flex: 1,
              display: 'flex',
              flexDirection: 'column'
            }}>
              <div style={{ 
                flex: 1, 
                overflowY: 'auto'
              }}>
                {orderBookData.asks.map((level, index) => (
                  <OrderBookLevel
                    key={level.id}
                    level={level}
                    side="ask"
                    displaySettings={displaySettings}
                    maxQuantity={maxQuantities.ask}
                    onLevelClick={onLevelClick}
                  />
                ))}
              </div>
            </div>
          </div>
        )}

        {/* Loading overlay */}
        {isLoading && orderBookData && (
          <div style={{
            position: 'absolute',
            top: 0,
            left: 0,
            right: 0,
            height: '2px',
            background: 'linear-gradient(90deg, transparent, #1890ff, transparent)',
            animation: 'loading 1s infinite'
          }}>
            <style>
              {`
                @keyframes loading {
                  0% { transform: translateX(-100%); }
                  100% { transform: translateX(100%); }
                }
              `}
            </style>
          </div>
        )}
      </div>

      {/* Footer with summary info */}
      {orderBookData && (
        <div style={{
          padding: '8px 16px',
          borderTop: '1px solid #f0f0f0',
          background: '#fafafa',
          fontSize: '11px',
          color: '#666',
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center'
        }}>
          <Space size="large">
            <span>
              Bid Volume: {orderBookData.totalBidVolume.toLocaleString()}
            </span>
            <span>
              Ask Volume: {orderBookData.totalAskVolume.toLocaleString()}
            </span>
            <span>
              Levels: {orderBookData.bids.length + orderBookData.asks.length}
            </span>
          </Space>
          
          {connectionStatus === 'connected' && (
            <Space size="small">
              <WifiOutlined style={{ color: '#52c41a' }} />
              <span style={{ color: '#52c41a' }}>Live Data</span>
            </Space>
          )}
        </div>
      )}
    </Card>
  )
}