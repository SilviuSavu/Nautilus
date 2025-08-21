import React, { useState } from 'react'
import { Card, Spin, Alert, Tag, Space } from 'antd'
import { WifiOutlined, DisconnectOutlined } from '@ant-design/icons'
import { ChartContainer } from './ChartContainer'
import { useChartStore } from './hooks/useChartStore'
import { useChartData } from './hooks/useChartData'
import { useRealTimeUpdates } from './hooks/useRealTimeUpdates'

interface ChartComponentProps {
  className?: string
  height?: number
}


export const ChartComponent: React.FC<ChartComponentProps> = ({
  className,
  height = 600
}) => {
  console.log('🎬 ChartComponent rendering...')
  
  const {
    chartData,
    settings,
    isLoading,
    error,
    currentInstrument,
    realTimeUpdates
  } = useChartStore()
  
  console.log('📊 ChartComponent state:', { 
    candlesCount: chartData.candles.length,
    isLoading, 
    error: error?.message,
    currentInstrument: currentInstrument?.symbol 
  })

  // Initialize hooks for data management
  useChartData() // Hook manages data loading automatically
  const { connectionStatus } = useRealTimeUpdates()
  
  const [currentPrice, setCurrentPrice] = useState<number | null>(null)

  // No default instrument - wait for user selection or real data

  const handlePriceChange = (price: number) => {
    setCurrentPrice(price)
  }

  if (error) {
    return (
      <Card className={className}>
        <Alert
          type="error"
          message="Chart Error"
          description={error.message}
          showIcon
        />
      </Card>
    )
  }

  // Show message when no data is available
  if (!isLoading && chartData.candles.length === 0) {
    return (
      <Card 
        className={className}
        title={
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
            <Space>
              <span>
                {currentInstrument?.symbol || 'Financial Chart'} 
                {currentInstrument?.venue && ` (${currentInstrument.venue})`}
              </span>
              {realTimeUpdates && (
                <Tag 
                  icon={connectionStatus === 'connected' ? <WifiOutlined /> : <DisconnectOutlined />}
                  color={connectionStatus === 'connected' ? 'green' : 'red'}
                >
                  {connectionStatus === 'connected' ? 'Live' : 'Offline'}
                </Tag>
              )}
            </Space>
          </div>
        }
        extra={
          <span style={{ fontSize: '12px', color: '#888' }}>
            {settings.timeframe.toUpperCase()}
          </span>
        }
      >
        <div style={{ 
          display: 'flex', 
          justifyContent: 'center', 
          alignItems: 'center', 
          height,
          flexDirection: 'column',
          color: '#888',
          background: '#fafafa'
        }}>
          <h3 style={{ color: '#666', marginBottom: 16 }}>No Market Data Available</h3>
          <p style={{ color: '#999', textAlign: 'center', maxWidth: 400 }}>
            Historical data for {currentInstrument?.symbol || 'this instrument'} is not available. 
            Please check your data connection or try a different instrument.
          </p>
        </div>
      </Card>
    )
  }

  return (
    <Card 
      className={className}
      title={
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <Space>
            <span>
              {currentInstrument?.symbol || 'Financial Chart'} 
              {currentInstrument?.venue && ` (${currentInstrument.venue})`}
            </span>
            {realTimeUpdates && (
              <Tag 
                icon={connectionStatus === 'connected' ? <WifiOutlined /> : <DisconnectOutlined />}
                color={connectionStatus === 'connected' ? 'green' : 'red'}
              >
                {connectionStatus === 'connected' ? 'Live' : 'Offline'}
              </Tag>
            )}
          </Space>
          {currentPrice && (
            <span style={{ 
              fontSize: '16px', 
              fontWeight: 'bold',
              color: '#1890ff'
            }}>
              ${currentPrice.toLocaleString(undefined, { 
                minimumFractionDigits: 2,
                maximumFractionDigits: 4 
              })}
            </span>
          )}
        </div>
      }
      extra={
        <Space size="small">
          <span style={{ fontSize: '12px', color: '#888' }}>
            {settings.timeframe.toUpperCase()}
          </span>
          <span style={{ fontSize: '12px', color: '#888' }}>
            {settings.showVolume ? 'Volume On' : 'Volume Off'}
          </span>
          {chartData.candles.length > 0 && (
            <span style={{ fontSize: '12px', color: '#888' }}>
              {chartData.candles.length} bars
            </span>
          )}
        </Space>
      }
      bodyStyle={{ padding: 0 }}
    >
      <div style={{ position: 'relative', height }}>
        {isLoading ? (
          <div style={{ 
            display: 'flex', 
            justifyContent: 'center', 
            alignItems: 'center', 
            height: '100%',
            background: '#fafafa'
          }}>
            <Spin size="large" />
          </div>
        ) : (
          <ChartContainer
            data={chartData}
            settings={settings}
            onPriceChange={handlePriceChange}
            height={height}
          />
        )}
      </div>
    </Card>
  )
}