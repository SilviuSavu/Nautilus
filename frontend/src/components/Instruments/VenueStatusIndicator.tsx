import React, { useState, useEffect } from 'react'
import { Badge, Tooltip, Card, Space, Typography, Row, Col, Tag, Divider, Statistic } from 'antd'
import { 
  WifiOutlined, 
  DisconnectOutlined, 
  LoadingOutlined, 
  ExclamationCircleOutlined,
  CheckCircleOutlined,
  ClockCircleOutlined,
  WarningOutlined
} from '@ant-design/icons'
import { VenueInfo, VenueConnectionStatus } from './types/instrumentTypes'
import { useRealtimeVenueStatus } from './hooks/useRealtime'

const { Text, Title } = Typography

interface VenueStatusIndicatorProps {
  venue: string
  status?: VenueInfo
  showName?: boolean
  size?: 'small' | 'default'
}

export const VenueStatusIndicator: React.FC<VenueStatusIndicatorProps> = ({
  venue,
  status,
  showName = true,
  size = 'default'
}) => {
  const getStatusColor = (connectionStatus?: VenueConnectionStatus) => {
    switch (connectionStatus) {
      case 'connected':
        return 'success'
      case 'connecting':
        return 'processing'
      case 'error':
        return 'error'
      case 'maintenance':
        return 'warning'
      case 'disconnected':
      default:
        return 'default'
    }
  }

  const getStatusText = (connectionStatus?: VenueConnectionStatus) => {
    switch (connectionStatus) {
      case 'connected':
        return 'Connected'
      case 'connecting':
        return 'Connecting...'
      case 'error':
        return 'Connection Error'
      case 'maintenance':
        return 'Maintenance'
      case 'disconnected':
        return 'Disconnected'
      default:
        return 'Unknown'
    }
  }

  const getTooltipContent = () => {
    if (!status) {
      return `${venue}: Status unknown`
    }

    return (
      <div>
        <div><strong>{status.name || venue}</strong></div>
        <div>Status: {getStatusText(status.connectionStatus)}</div>
        {status.country && <div>Country: {status.country}</div>}
        {status.timezone && <div>Timezone: {status.timezone}</div>}
        {status.lastHeartbeat && (
          <div>Last update: {new Date(status.lastHeartbeat).toLocaleTimeString()}</div>
        )}
        {status.errorMessage && (
          <div style={{ color: '#ff4d4f' }}>Error: {status.errorMessage}</div>
        )}
        {status.assetClasses && status.assetClasses.length > 0 && (
          <div>
            Asset classes: {status.assetClasses.join(', ')}
          </div>
        )}
      </div>
    )
  }

  return (
    <Tooltip title={getTooltipContent()}>
      <Badge
        status={getStatusColor(status?.connectionStatus)}
        text={showName ? venue : undefined}
        size={size}
      />
    </Tooltip>
  )
}

// Real-time venue connection monitoring component
interface RealtimeVenueMonitorProps {
  venues: string[]
  showDetails?: boolean
  compact?: boolean
  className?: string
}

export const RealtimeVenueMonitor: React.FC<RealtimeVenueMonitorProps> = ({
  venues,
  showDetails = true,
  compact = false,
  className
}) => {
  const { venueStatuses, lastUpdate } = useRealtimeVenueStatus(venues)
  const [connectionStats, setConnectionStats] = useState({
    connected: 0,
    disconnected: 0,
    error: 0,
    connecting: 0
  })

  useEffect(() => {
    const stats = { connected: 0, disconnected: 0, error: 0, connecting: 0 }
    
    venues.forEach(venue => {
      const status = venueStatuses.get(venue)
      if (status) {
        stats[status.status] = (stats[status.status] || 0) + 1
      } else {
        stats.disconnected++
      }
    })
    
    setConnectionStats(stats)
  }, [venueStatuses, venues])

  const getStatusIcon = (status: VenueConnectionStatus) => {
    switch (status) {
      case 'connected':
        return <CheckCircleOutlined style={{ color: '#52c41a' }} />
      case 'connecting':
        return <LoadingOutlined style={{ color: '#1890ff' }} />
      case 'error':
        return <ExclamationCircleOutlined style={{ color: '#ff4d4f' }} />
      case 'maintenance':
        return <WarningOutlined style={{ color: '#faad14' }} />
      default:
        return <DisconnectOutlined style={{ color: '#d9d9d9' }} />
    }
  }

  const formatLastHeartbeat = (timestamp: string) => {
    const now = Date.now()
    const heartbeatTime = new Date(timestamp).getTime()
    const diff = now - heartbeatTime

    if (diff < 60000) { // Less than 1 minute
      return 'Just now'
    } else if (diff < 3600000) { // Less than 1 hour
      const minutes = Math.floor(diff / 60000)
      return `${minutes}m ago`
    } else {
      const hours = Math.floor(diff / 3600000)
      return `${hours}h ago`
    }
  }

  if (compact) {
    return (
      <Card className={className} size="small" bodyStyle={{ padding: '8px 12px' }}>
        <Row align="middle" justify="space-between">
          <Col>
            <Space size="small">
              <WifiOutlined />
              <Text style={{ fontSize: '12px' }}>
                {connectionStats.connected}/{venues.length} Connected
              </Text>
            </Space>
          </Col>
          <Col>
            <Space size="small">
              {connectionStats.error > 0 && (
                <Badge count={connectionStats.error} status="error" size="small" />
              )}
              {connectionStats.connecting > 0 && (
                <Badge count={connectionStats.connecting} status="processing" size="small" />
              )}
              {lastUpdate && (
                <Text type="secondary" style={{ fontSize: '10px' }}>
                  {formatLastHeartbeat(lastUpdate)}
                </Text>
              )}
            </Space>
          </Col>
        </Row>
      </Card>
    )
  }

  return (
    <Card 
      className={className}
      title={
        <Space>
          <WifiOutlined />
          Venue Connection Status
        </Space>
      }
      extra={
        lastUpdate && (
          <Tooltip title={`Last update: ${new Date(lastUpdate).toLocaleString()}`}>
            <Space size="small">
              <ClockCircleOutlined />
              <Text type="secondary" style={{ fontSize: '12px' }}>
                {formatLastHeartbeat(lastUpdate)}
              </Text>
            </Space>
          </Tooltip>
        )
      }
    >
      <Space direction="vertical" style={{ width: '100%' }} size="middle">
        {/* Connection Overview */}
        <Row gutter={[16, 8]}>
          <Col span={6}>
            <Statistic
              title="Connected"
              value={connectionStats.connected}
              valueStyle={{ color: '#52c41a', fontSize: '16px' }}
              prefix={<CheckCircleOutlined />}
            />
          </Col>
          <Col span={6}>
            <Statistic
              title="Connecting"
              value={connectionStats.connecting}
              valueStyle={{ color: '#1890ff', fontSize: '16px' }}
              prefix={<LoadingOutlined />}
            />
          </Col>
          <Col span={6}>
            <Statistic
              title="Errors"
              value={connectionStats.error}
              valueStyle={{ color: '#ff4d4f', fontSize: '16px' }}
              prefix={<ExclamationCircleOutlined />}
            />
          </Col>
          <Col span={6}>
            <Statistic
              title="Offline"
              value={connectionStats.disconnected}
              valueStyle={{ color: '#d9d9d9', fontSize: '16px' }}
              prefix={<DisconnectOutlined />}
            />
          </Col>
        </Row>

        {showDetails && (
          <>
            <Divider style={{ margin: '12px 0' }} />
            
            {/* Individual Venue Status */}
            <div>
              <Title level={5} style={{ margin: 0, marginBottom: '12px' }}>Individual Venues</Title>
              <Space direction="vertical" style={{ width: '100%' }} size="small">
                {venues.map(venue => {
                  const venueStatus = venueStatuses.get(venue)
                  
                  return (
                    <Card key={venue} size="small" bodyStyle={{ padding: '8px 12px' }}>
                      <Row align="middle" justify="space-between">
                        <Col span={8}>
                          <Space>
                            {getStatusIcon(venueStatus?.status || 'disconnected')}
                            <Text strong style={{ fontSize: '13px' }}>{venue}</Text>
                          </Space>
                        </Col>
                        <Col span={8}>
                          <Space direction="vertical" size={0} align="center">
                            <Tag 
                              color={venueStatus?.status === 'connected' ? 'success' : 
                                     venueStatus?.status === 'error' ? 'error' : 
                                     venueStatus?.status === 'connecting' ? 'processing' : 'default'}
                              style={{ fontSize: '11px' }}
                            >
                              {venueStatus?.status || 'disconnected'}
                            </Tag>
                            {venueStatus?.connectedInstruments !== undefined && (
                              <Text type="secondary" style={{ fontSize: '10px' }}>
                                {venueStatus.connectedInstruments} instruments
                              </Text>
                            )}
                          </Space>
                        </Col>
                        <Col span={8}>
                          <Space direction="vertical" size={0} align="end">
                            {venueStatus?.lastHeartbeat && (
                              <Text type="secondary" style={{ fontSize: '11px' }}>
                                {formatLastHeartbeat(venueStatus.lastHeartbeat)}
                              </Text>
                            )}
                            {venueStatus?.errorMessage && (
                              <Tooltip title={venueStatus.errorMessage}>
                                <Text type="danger" style={{ fontSize: '10px' }}>
                                  <ExclamationCircleOutlined /> Error
                                </Text>
                              </Tooltip>
                            )}
                          </Space>
                        </Col>
                      </Row>
                    </Card>
                  )
                })}
              </Space>
            </div>
          </>
        )}
      </Space>
    </Card>
  )
}

// Simple venue status list for sidebars
interface VenueStatusListProps {
  venues: string[]
  className?: string
}

export const VenueStatusList: React.FC<VenueStatusListProps> = ({
  venues,
  className
}) => {
  const { venueStatuses } = useRealtimeVenueStatus(venues)

  return (
    <div className={className}>
      <Space direction="vertical" style={{ width: '100%' }} size="small">
        {venues.map(venue => {
          const status = venueStatuses.get(venue)
          return (
            <Row key={venue} align="middle" justify="space-between">
              <Col>
                <Text style={{ fontSize: '12px' }}>{venue}</Text>
              </Col>
              <Col>
                <VenueStatusIndicator
                  venue={venue}
                  status={status as any}
                  showName={false}
                  size="small"
                />
              </Col>
            </Row>
          )
        })}
      </Space>
    </div>
  )
}