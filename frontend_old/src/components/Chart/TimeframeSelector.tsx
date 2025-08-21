import React from 'react'
import { Button, Space, Divider, Typography } from 'antd'
import { useChartStore } from './hooks/useChartStore'
import { Timeframe } from './types/chartTypes'

const { Text } = Typography

interface TimeframeSelectorProps {
  className?: string
}

const timeframes: Array<{ value: Timeframe; label: string; category: string }> = [
  // Intraday - Minutes
  { value: '1m', label: '1m', category: 'minutes' },
  { value: '2m', label: '2m', category: 'minutes' },
  { value: '5m', label: '5m', category: 'minutes' },
  { value: '10m', label: '10m', category: 'minutes' },
  { value: '15m', label: '15m', category: 'minutes' },
  { value: '30m', label: '30m', category: 'minutes' },
  
  // Intraday - Hours
  { value: '1h', label: '1H', category: 'hours' },
  { value: '2h', label: '2H', category: 'hours' },
  { value: '4h', label: '4H', category: 'hours' },
  
  // Daily and above
  { value: '1d', label: '1D', category: 'daily' },
  { value: '1w', label: '1W', category: 'daily' },
  { value: '1M', label: '1M', category: 'daily' },
]

export const TimeframeSelector: React.FC<TimeframeSelectorProps> = ({ className }) => {
  const { timeframe, setTimeframe, setLoading } = useChartStore()

  const handleTimeframeChange = (newTimeframe: Timeframe) => {
    if (newTimeframe === timeframe) return
    
    setLoading(true)
    setTimeframe(newTimeframe)
    
    // Simulate loading delay for timeframe switch
    setTimeout(() => {
      setLoading(false)
    }, 200)
  }

  const groupedTimeframes = {
    minutes: timeframes.filter(tf => tf.category === 'minutes'),
    hours: timeframes.filter(tf => tf.category === 'hours'),
    daily: timeframes.filter(tf => tf.category === 'daily'),
  }

  const renderTimeframeGroup = (groupName: string, items: Array<{ value: Timeframe; label: string; category: string }>) => (
    <Space size="small" key={groupName}>
      {items.map(({ value, label }) => (
        <Button
          key={value}
          type={timeframe === value ? 'primary' : 'default'}
          size="small"
          onClick={() => handleTimeframeChange(value)}
          style={{
            borderRadius: '4px',
            fontWeight: timeframe === value ? 'bold' : 'normal',
            minWidth: '38px',
            height: '28px',
            fontSize: '12px'
          }}
        >
          {label}
        </Button>
      ))}
    </Space>
  )

  return (
    <div className={className}>
      <Space direction="vertical" size="small" style={{ width: '100%' }}>
        <Space split={<Divider type="vertical" />} size="small" wrap>
          <div>
            <Text type="secondary" style={{ fontSize: '11px', marginRight: '8px' }}>Minutes:</Text>
            {renderTimeframeGroup('minutes', groupedTimeframes.minutes)}
          </div>
          <div>
            <Text type="secondary" style={{ fontSize: '11px', marginRight: '8px' }}>Hours:</Text>
            {renderTimeframeGroup('hours', groupedTimeframes.hours)}
          </div>
          <div>
            <Text type="secondary" style={{ fontSize: '11px', marginRight: '8px' }}>Daily+:</Text>
            {renderTimeframeGroup('daily', groupedTimeframes.daily)}
          </div>
        </Space>
      </Space>
    </div>
  )
}