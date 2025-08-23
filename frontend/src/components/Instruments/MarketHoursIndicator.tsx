import React, { useState, useEffect } from 'react'
import { Card, Space, Badge, Typography, Tooltip, Progress, Row, Col, Divider, Tag } from 'antd'
import { 
  ClockCircleOutlined, 
  CalendarOutlined, 
  GlobalOutlined,
  CheckCircleOutlined,
  CloseCircleOutlined,
  ExclamationCircleOutlined,
  MinusCircleOutlined
} from '@ant-design/icons'
import { useMarketSession } from './hooks/useRealtime'

const { Text, Title } = Typography

interface MarketHoursIndicatorProps {
  venue: string
  showProgress?: boolean
  showDetails?: boolean
  compact?: boolean
  className?: string
}

interface MarketProgress {
  percentage: number
  timeElapsed: string
  timeRemaining: string
  totalHours: number
}

export const MarketHoursIndicator: React.FC<MarketHoursIndicatorProps> = ({
  venue,
  showProgress = true,
  showDetails = false,
  compact = false,
  className
}) => {
  const { sessionInfo, loading } = useMarketSession(venue)
  const [currentTime, setCurrentTime] = useState(new Date())
  const [marketProgress, setMarketProgress] = useState<MarketProgress | null>(null)

  // Update current time every minute
  useEffect(() => {
    const interval = setInterval(() => {
      setCurrentTime(new Date())
    }, 60000)

    return () => clearInterval(interval)
  }, [])

  // Calculate market progress
  useEffect(() => {
    if (!sessionInfo?.marketHours || !sessionInfo.isOpen) {
      setMarketProgress(null)
      return
    }

    const progress = calculateMarketProgress(currentTime, sessionInfo)
    setMarketProgress(progress)
  }, [sessionInfo, currentTime])

  const calculateMarketProgress = (now: Date, session: any): MarketProgress | null => {
    if (!session.marketHours || !session.isOpen) return null

    const { open, close } = session.marketHours
    const [openHour, openMin] = open.split(':').map(Number)
    const [closeHour, closeMin] = close.split(':').map(Number)

    const marketOpen = new Date(now)
    marketOpen.setHours(openHour, openMin, 0, 0)

    const marketClose = new Date(now)
    marketClose.setHours(closeHour, closeMin, 0, 0)

    const totalDuration = marketClose.getTime() - marketOpen.getTime()
    const elapsed = now.getTime() - marketOpen.getTime()
    const remaining = marketClose.getTime() - now.getTime()

    const percentage = Math.max(0, Math.min(100, (elapsed / totalDuration) * 100))
    const totalHours = totalDuration / (1000 * 60 * 60)

    return {
      percentage,
      timeElapsed: formatDuration(elapsed),
      timeRemaining: formatDuration(remaining),
      totalHours
    }
  }

  const formatDuration = (milliseconds: number): string => {
    if (milliseconds <= 0) return '0h 0m'
    
    const hours = Math.floor(milliseconds / (1000 * 60 * 60))
    const minutes = Math.floor((milliseconds % (1000 * 60 * 60)) / (1000 * 60))
    
    return `${hours}h ${minutes}m`
  }

  const getMarketStatus = () => {
    if (!sessionInfo) {
      return {
        status: 'unknown' as const,
        icon: <MinusCircleOutlined />,
        text: 'Unknown',
        color: '#d9d9d9'
      }
    }

    if (sessionInfo.isOpen) {
      return {
        status: 'open' as const,
        icon: <CheckCircleOutlined />,
        text: 'Market Open',
        color: '#52c41a'
      }
    }

    // Check if it's a trading day
    const currentDay = currentTime.toLocaleDateString('en-US', { weekday: 'long' })
    const isTradingDay = sessionInfo.marketHours?.days.includes(currentDay)

    if (!isTradingDay) {
      return {
        status: 'holiday' as const,
        icon: <CalendarOutlined />,
        text: 'Holiday/Weekend',
        color: '#d9d9d9'
      }
    }

    // Determine if pre-market or after-hours
    const { open, close } = sessionInfo.marketHours
    const [openHour, openMin] = open.split(':').map(Number)
    const [closeHour, closeMin] = close.split(':').map(Number)

    const marketOpen = new Date(currentTime)
    marketOpen.setHours(openHour, openMin, 0, 0)

    const marketClose = new Date(currentTime)
    marketClose.setHours(closeHour, closeMin, 0, 0)

    if (currentTime < marketOpen) {
      return {
        status: 'pre-market' as const,
        icon: <ExclamationCircleOutlined />,
        text: 'Pre-Market',
        color: '#faad14'
      }
    } else {
      return {
        status: 'after-hours' as const,
        icon: <CloseCircleOutlined />,
        text: 'After Hours',
        color: '#faad14'
      }
    }
  }

  const getNextChange = () => {
    if (!sessionInfo?.marketHours) return null

    const { open, close } = sessionInfo.marketHours
    const [openHour, openMin] = open.split(':').map(Number)
    const [closeHour, closeMin] = close.split(':').map(Number)

    if (sessionInfo.isOpen) {
      // Market is open, next change is close
      const marketClose = new Date(currentTime)
      marketClose.setHours(closeHour, closeMin, 0, 0)
      
      const timeUntil = marketClose.getTime() - currentTime.getTime()
      return {
        event: 'closes',
        time: marketClose,
        duration: formatDuration(timeUntil)
      }
    } else {
      // Market is closed, next change is open
      let marketOpen = new Date(currentTime)
      marketOpen.setHours(openHour, openMin, 0, 0)
      
      // If we're past today's close, move to next trading day
      if (currentTime > marketOpen) {
        marketOpen.setDate(marketOpen.getDate() + 1)
        // Skip weekends if needed
        while (!sessionInfo.marketHours.days.includes(marketOpen.toLocaleDateString('en-US', { weekday: 'long' }))) {
          marketOpen.setDate(marketOpen.getDate() + 1)
        }
      }
      
      const timeUntil = marketOpen.getTime() - currentTime.getTime()
      return {
        event: 'opens',
        time: marketOpen,
        duration: formatDuration(timeUntil)
      }
    }
  }

  if (loading) {
    return (
      <Card className={className} size={compact ? 'small' : 'default'} loading>
        <div style={{ height: compact ? '40px' : '80px' }} />
      </Card>
    )
  }

  const status = getMarketStatus()
  const nextChange = getNextChange()

  if (compact) {
    return (
      <div className={className}>
        <Space size="small">
          <Badge 
            status={status.status === 'open' ? 'success' : status.status === 'unknown' ? 'error' : 'default'} 
            text={
              <Space size="small">
                {status.icon}
                <Text style={{ fontSize: '12px' }}>{status.text}</Text>
              </Space>
            }
          />
          {nextChange && (
            <Text type="secondary" style={{ fontSize: '11px' }}>
              {status.status === 'open' ? 'Closes' : 'Opens'} in {nextChange.duration}
            </Text>
          )}
        </Space>
      </div>
    )
  }

  return (
    <Card 
      className={className}
      title={
        <Space>
          <ClockCircleOutlined />
          Market Hours - {venue}
        </Space>
      }
          >
      <Space direction="vertical" style={{ width: '100%' }} size="middle">
        {/* Current Status */}
        <Row align="middle" justify="space-between">
          <Col>
            <Space>
              <span style={{ color: status.color }}>{status.icon}</span>
              <Text strong style={{ color: status.color }}>{status.text}</Text>
            </Space>
          </Col>
          <Col>
            <Text type="secondary" style={{ fontSize: '12px' }}>
              <GlobalOutlined /> {sessionInfo?.timezone || 'UTC'}
            </Text>
          </Col>
        </Row>

        {/* Market Progress (only when market is open) */}
        {showProgress && marketProgress && (
          <>
            <Divider style={{ margin: '8px 0' }} />
            <div>
              <Row align="middle" justify="space-between" style={{ marginBottom: '8px' }}>
                <Col>
                  <Text style={{ fontSize: '12px' }}>Market Progress</Text>
                </Col>
                <Col>
                  <Text style={{ fontSize: '12px' }}>{marketProgress.percentage.toFixed(1)}%</Text>
                </Col>
              </Row>
              <Progress 
                percent={marketProgress.percentage} 
                size="small" 
                status="active"
                strokeColor={{
                  '0%': '#52c41a',
                  '70%': '#faad14',
                  '90%': '#ff4d4f'
                }}
              />
              <Row align="middle" justify="space-between" style={{ marginTop: '4px' }}>
                <Col>
                  <Text type="secondary" style={{ fontSize: '11px' }}>
                    Elapsed: {marketProgress.timeElapsed}
                  </Text>
                </Col>
                <Col>
                  <Text type="secondary" style={{ fontSize: '11px' }}>
                    Remaining: {marketProgress.timeRemaining}
                  </Text>
                </Col>
              </Row>
            </div>
          </>
        )}

        {/* Next Change */}
        {nextChange && (
          <>
            <Divider style={{ margin: '8px 0' }} />
            <Row align="middle" justify="space-between">
              <Col>
                <Text style={{ fontSize: '12px' }}>
                  Market {nextChange.event} in:
                </Text>
              </Col>
              <Col>
                <Tag color="blue">
                  <ClockCircleOutlined /> {nextChange.duration}
                </Tag>
              </Col>
            </Row>
          </>
        )}

        {/* Market Hours Details */}
        {showDetails && sessionInfo?.marketHours && (
          <>
            <Divider style={{ margin: '8px 0' }} />
            <Row gutter={[8, 4]}>
              <Col span={12}>
                <Space direction="vertical" size={0}>
                  <Text type="secondary" style={{ fontSize: '11px' }}>Opens</Text>
                  <Text style={{ fontSize: '12px' }}>{sessionInfo.marketHours.open}</Text>
                </Space>
              </Col>
              <Col span={12}>
                <Space direction="vertical" size={0}>
                  <Text type="secondary" style={{ fontSize: '11px' }}>Closes</Text>
                  <Text style={{ fontSize: '12px' }}>{sessionInfo.marketHours.close}</Text>
                </Space>
              </Col>
            </Row>
            <div style={{ marginTop: '8px' }}>
              <Text type="secondary" style={{ fontSize: '11px' }}>Trading Days:</Text>
              <div style={{ marginTop: '4px' }}>
                <Space wrap size="small">
                  {sessionInfo.marketHours.days.map((day: string) => (
                    <Tag 
                      key={day}
                                            color={currentTime.toLocaleDateString('en-US', { weekday: 'long' }) === day ? 'blue' : 'default'}
                    >
                      {day.substring(0, 3)}
                    </Tag>
                  ))}
                </Space>
              </div>
            </div>
          </>
        )}
      </Space>
    </Card>
  )
}

// Multi-venue market hours overview
interface MultiVenueMarketHoursProps {
  venues: string[]
  className?: string
}

export const MultiVenueMarketHours: React.FC<MultiVenueMarketHoursProps> = ({
  venues,
  className
}) => {
  return (
    <Card 
      className={className}
      title={
        <Space>
          <GlobalOutlined />
          Global Market Hours
        </Space>
      }
          >
      <Space direction="vertical" style={{ width: '100%' }} size="small">
        {venues.map(venue => (
          <Row key={venue} align="middle" justify="space-between">
            <Col span={8}>
              <Text strong style={{ fontSize: '12px' }}>{venue}</Text>
            </Col>
            <Col span={16}>
              <MarketHoursIndicator venue={venue} compact />
            </Col>
          </Row>
        ))}
      </Space>
    </Card>
  )
}