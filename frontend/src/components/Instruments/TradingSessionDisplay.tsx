import React, { useState, useEffect } from 'react'
import { Card, Space, Badge, Typography, Tooltip, Tag, Row, Col, Divider, Statistic } from 'antd'
import { 
  ClockCircleOutlined, 
  CalendarOutlined, 
  GlobalOutlined,
  InfoCircleOutlined,
  CheckCircleOutlined,
  CloseCircleOutlined,
  MinusCircleOutlined
} from '@ant-design/icons'
import { useMarketSession } from './hooks/useRealtime'
import { Instrument } from './types/instrumentTypes'

const { Text, Title } = Typography

interface TradingSessionDisplayProps {
  instrument: Instrument
  showDetails?: boolean
  compact?: boolean
  className?: string
}

interface SessionStatus {
  status: 'open' | 'closed' | 'pre-market' | 'after-hours' | 'holiday' | 'unknown'
  nextChange?: string
  timeUntilChange?: string
}

export const TradingSessionDisplay: React.FC<TradingSessionDisplayProps> = ({
  instrument,
  showDetails = true,
  compact = false,
  className
}) => {
  const { sessionInfo, loading } = useMarketSession(instrument.venue)
  const [currentTime, setCurrentTime] = useState(new Date())
  const [sessionStatus, setSessionStatus] = useState<SessionStatus>({ status: 'unknown' })

  // Update current time every minute
  useEffect(() => {
    const interval = setInterval(() => {
      setCurrentTime(new Date())
    }, 60000) // Update every minute

    return () => clearInterval(interval)
  }, [])

  // Calculate session status based on current time and market hours
  useEffect(() => {
    if (!sessionInfo) {
      setSessionStatus({ status: 'unknown' })
      return
    }

    const now = new Date()
    const status = calculateSessionStatus(now, sessionInfo)
    setSessionStatus(status)
  }, [sessionInfo, currentTime])

  const calculateSessionStatus = (now: Date, session: any): SessionStatus => {
    if (!session.marketHours) {
      return { status: 'unknown' }
    }

    const { open, close, days } = session.marketHours
    const currentDay = now.toLocaleDateString('en-US', { weekday: 'long' })
    const isTrainingDay = days.includes(currentDay)

    if (!isTrainingDay) {
      const nextTrainingDay = getNextTradingDay(now, days)
      return {
        status: 'holiday',
        nextChange: nextTrainingDay,
        timeUntilChange: calculateTimeUntil(nextTrainingDay)
      }
    }

    // Parse market hours
    const [openHour, openMin] = open.split(':').map(Number)
    const [closeHour, closeMin] = close.split(':').map(Number)

    const marketOpen = new Date(now)
    marketOpen.setHours(openHour, openMin, 0, 0)

    const marketClose = new Date(now)
    marketClose.setHours(closeHour, closeMin, 0, 0)

    if (now < marketOpen) {
      return {
        status: 'pre-market',
        nextChange: marketOpen.toISOString(),
        timeUntilChange: calculateTimeUntil(marketOpen.toISOString())
      }
    } else if (now >= marketOpen && now < marketClose) {
      return {
        status: 'open',
        nextChange: marketClose.toISOString(),
        timeUntilChange: calculateTimeUntil(marketClose.toISOString())
      }
    } else {
      // After market close
      const nextMarketOpen = new Date(now)
      nextMarketOpen.setDate(nextMarketOpen.getDate() + 1)
      nextMarketOpen.setHours(openHour, openMin, 0, 0)

      return {
        status: 'after-hours',
        nextChange: nextMarketOpen.toISOString(),
        timeUntilChange: calculateTimeUntil(nextMarketOpen.toISOString())
      }
    }
  }

  const getNextTradingDay = (currentDate: Date, tradingDays: string[]): string => {
    let nextDate = new Date(currentDate)
    nextDate.setDate(nextDate.getDate() + 1)

    while (!tradingDays.includes(nextDate.toLocaleDateString('en-US', { weekday: 'long' }))) {
      nextDate.setDate(nextDate.getDate() + 1)
    }

    return nextDate.toISOString()
  }

  const calculateTimeUntil = (targetTime: string): string => {
    const target = new Date(targetTime)
    const now = new Date()
    const diff = target.getTime() - now.getTime()

    if (diff <= 0) return 'Now'

    const hours = Math.floor(diff / (1000 * 60 * 60))
    const minutes = Math.floor((diff % (1000 * 60 * 60)) / (1000 * 60))

    if (hours > 24) {
      const days = Math.floor(hours / 24)
      const remainingHours = hours % 24
      return `${days}d ${remainingHours}h`
    } else if (hours > 0) {
      return `${hours}h ${minutes}m`
    } else {
      return `${minutes}m`
    }
  }

  const getStatusBadge = () => {
    const statusConfig = {
      'open': { status: 'success' as const, text: 'Market Open', icon: <CheckCircleOutlined /> },
      'closed': { status: 'default' as const, text: 'Market Closed', icon: <CloseCircleOutlined /> },
      'pre-market': { status: 'warning' as const, text: 'Pre-Market', icon: <ClockCircleOutlined /> },
      'after-hours': { status: 'warning' as const, text: 'After Hours', icon: <ClockCircleOutlined /> },
      'holiday': { status: 'default' as const, text: 'Holiday/Weekend', icon: <CalendarOutlined /> },
      'unknown': { status: 'error' as const, text: 'Unknown', icon: <MinusCircleOutlined /> }
    }

    const config = statusConfig[sessionStatus.status]
    return (
      <Badge 
        status={config.status} 
        text={
          <Space size="small">
            {config.icon}
            {config.text}
          </Space>
        } 
      />
    )
  }

  const formatTime = (time: string, timezone?: string) => {
    const date = new Date(time)
    return date.toLocaleTimeString('en-US', {
      hour12: false,
      hour: '2-digit',
      minute: '2-digit',
      timeZone: timezone
    })
  }

  if (loading) {
    return (
      <Card className={className} size={compact ? 'small' : 'default'} loading>
        <div style={{ height: compact ? '40px' : '120px' }} />
      </Card>
    )
  }

  if (compact) {
    return (
      <Card className={className} size="small" bodyStyle={{ padding: '8px 12px' }}>
        <Row align="middle" justify="space-between">
          <Col>
            {getStatusBadge()}
          </Col>
          <Col>
            <Space direction="vertical" size={0} align="end">
              <Text style={{ fontSize: '11px' }}>
                {sessionInfo?.timezone || 'UTC'}
              </Text>
              {sessionStatus.timeUntilChange && (
                <Text type="secondary" style={{ fontSize: '10px' }}>
                  {sessionStatus.status === 'open' ? 'Closes in' : 'Opens in'} {sessionStatus.timeUntilChange}
                </Text>
              )}
            </Space>
          </Col>
        </Row>
      </Card>
    )
  }

  if (!sessionInfo) {
    return (
      <Card className={className} title="Trading Session">
        <div style={{ textAlign: 'center', padding: '20px' }}>
          <Space direction="vertical">
            <InfoCircleOutlined style={{ fontSize: '24px', color: '#d9d9d9' }} />
            <Text type="secondary">Session information not available for {instrument.venue}</Text>
          </Space>
        </div>
      </Card>
    )
  }

  return (
    <Card 
      className={className}
      title={
        <Space>
          <CalendarOutlined />
          Trading Session - {instrument.venue}
        </Space>
      }
      extra={
        <Tooltip title={`Current time in ${sessionInfo.timezone}`}>
          <Space>
            <GlobalOutlined />
            <Text style={{ fontSize: '12px' }}>
              {currentTime.toLocaleTimeString('en-US', {
                hour12: false,
                hour: '2-digit',
                minute: '2-digit',
                timeZone: sessionInfo.timezone
              })}
            </Text>
          </Space>
        </Tooltip>
      }
    >
      <Space direction="vertical" style={{ width: '100%' }} size="middle">
        {/* Current Status */}
        <div>
          <Title level={5} style={{ margin: 0, marginBottom: '8px' }}>Current Status</Title>
          <Row align="middle" justify="space-between">
            <Col>
              {getStatusBadge()}
            </Col>
            <Col>
              {sessionStatus.timeUntilChange && (
                <Tooltip title={`Next change: ${new Date(sessionStatus.nextChange!).toLocaleString()}`}>
                  <Tag color="blue">
                    <ClockCircleOutlined /> 
                    {sessionStatus.status === 'open' ? 'Closes in' : 'Opens in'} {sessionStatus.timeUntilChange}
                  </Tag>
                </Tooltip>
              )}
            </Col>
          </Row>
        </div>

        {showDetails && sessionInfo.marketHours && (
          <>
            <Divider style={{ margin: '12px 0' }} />
            
            {/* Market Hours */}
            <div>
              <Title level={5} style={{ margin: 0, marginBottom: '12px' }}>Market Hours</Title>
              <Row gutter={[16, 8]}>
                <Col xs={12}>
                  <Statistic
                    title="Opens"
                    value={sessionInfo.marketHours.open}
                    valueStyle={{ fontSize: '16px' }}
                    prefix={<ClockCircleOutlined />}
                  />
                </Col>
                <Col xs={12}>
                  <Statistic
                    title="Closes"
                    value={sessionInfo.marketHours.close}
                    valueStyle={{ fontSize: '16px' }}
                    prefix={<ClockCircleOutlined />}
                  />
                </Col>
              </Row>
            </div>

            <Divider style={{ margin: '12px 0' }} />

            {/* Trading Days */}
            <div>
              <Title level={5} style={{ margin: 0, marginBottom: '8px' }}>Trading Days</Title>
              <Space wrap>
                {sessionInfo.marketHours.days.map((day: string) => (
                  <Tag 
                    key={day} 
                    color={currentTime.toLocaleDateString('en-US', { weekday: 'long' }) === day ? 'blue' : 'default'}
                  >
                    {day}
                  </Tag>
                ))}
              </Space>
            </div>

            <Divider style={{ margin: '12px 0' }} />

            {/* Additional Info */}
            <Row gutter={[16, 8]}>
              <Col xs={12}>
                <Space direction="vertical" size={0}>
                  <Text type="secondary" style={{ fontSize: '12px' }}>Venue</Text>
                  <Text strong>{instrument.venue}</Text>
                </Space>
              </Col>
              <Col xs={12}>
                <Space direction="vertical" size={0}>
                  <Text type="secondary" style={{ fontSize: '12px' }}>Timezone</Text>
                  <Text strong>{sessionInfo.timezone}</Text>
                </Space>
              </Col>
            </Row>
          </>
        )}
      </Space>
    </Card>
  )
}

// Simple component for showing session status in instrument lists
interface SessionStatusIndicatorProps {
  instrument: Instrument
  showText?: boolean
}

export const SessionStatusIndicator: React.FC<SessionStatusIndicatorProps> = ({
  instrument,
  showText = true
}) => {
  const { sessionInfo } = useMarketSession(instrument.venue)
  
  if (!sessionInfo) {
    return (
      <Badge 
        status="default" 
        text={showText ? "Unknown" : undefined} 
      />
    )
  }

  const now = new Date()
  const isOpen = sessionInfo.isOpen

  return (
    <Tooltip title={`Market ${isOpen ? 'Open' : 'Closed'} • ${instrument.venue}`}>
      <Badge 
        status={isOpen ? "success" : "default"} 
        text={showText ? (isOpen ? "Open" : "Closed") : undefined}
      />
    </Tooltip>
  )
}