/**
 * MessageBus connection status and health monitoring component
 */

import React from 'react'
import { Card, Space, Badge, Statistic, Button, Alert, Typography, Row, Col } from 'antd'
import { WifiOutlined, PlayCircleOutlined, StopOutlined, ReloadOutlined } from '@ant-design/icons'
import { useMessageBus } from '../hooks/useMessageBus'

const { Text, Title } = Typography

interface MessageBusStatusProps {
  showControls?: boolean;
  compact?: boolean;
}

const MessageBusStatus: React.FC<MessageBusStatusProps> = ({ 
  showControls = true,
  compact = false 
}) => {
  const {
    connectionStatus,
    connectionInfo,
    messagesReceived,
    connect,
    disconnect,
    latestMessage
  } = useMessageBus()

  const getStatusColor = () => {
    switch (connectionStatus) {
      case 'connected': return 'success'
      case 'connecting': return 'processing'
      case 'error': return 'error'
      default: return 'default'
    }
  }

  const getStatusText = () => {
    switch (connectionStatus) {
      case 'connected': return 'Connected'
      case 'connecting': return 'Connecting...'
      case 'disconnected': return 'Disconnected'
      case 'error': return 'Connection Error'
      default: return 'Unknown'
    }
  }

  const getAlertType = () => {
    switch (connectionStatus) {
      case 'connected': return 'success'
      case 'connecting': return 'info'
      case 'error': return 'error'
      default: return 'warning'
    }
  }

  if (compact) {
    return (
      <Alert
        message={
          <Space>
            <Badge status={getStatusColor()} />
            MessageBus: {getStatusText()}
            {connectionStatus === 'connected' && messagesReceived > 0 && (
              <Text type="secondary">({messagesReceived} messages)</Text>
            )}
          </Space>
        }
        type={getAlertType()}
        showIcon
        action={
          showControls && (
            <Space>
              {connectionStatus === 'connected' ? (
                <Button size="small" icon={<StopOutlined />} onClick={disconnect}>
                  Disconnect
                </Button>
              ) : (
                <Button size="small" icon={<PlayCircleOutlined />} onClick={connect}>
                  Connect
                </Button>
              )}
            </Space>
          )
        }
      />
    )
  }

  return (
    <Card
      title={
        <Space>
          <WifiOutlined />
          <Title level={4} style={{ margin: 0 }}>MessageBus Status</Title>
        </Space>
      }
      extra={
        showControls && (
          <Space>
            {connectionStatus === 'connected' ? (
              <Button icon={<StopOutlined />} onClick={disconnect} size="small">
                Disconnect
              </Button>
            ) : (
              <Button 
                icon={<PlayCircleOutlined />} 
                onClick={connect} 
                type="primary" 
                size="small"
              >
                Connect
              </Button>
            )}
            <Button icon={<ReloadOutlined />} size="small" onClick={() => window.location.reload()}>
              Refresh
            </Button>
          </Space>
        )
      }
    >
      <Space direction="vertical" style={{ width: '100%' }}>
        {/* Connection Status */}
        <Alert
          message={
            <Space>
              <Badge status={getStatusColor()} />
              WebSocket Status: {getStatusText()}
            </Space>
          }
          type={getAlertType()}
          showIcon
        />

        {/* Statistics */}
        <Row gutter={16}>
          <Col span={8}>
            <Statistic
              title="Messages Received"
              value={messagesReceived}
              valueStyle={{ color: connectionStatus === 'connected' ? '#3f8600' : '#999' }}
            />
          </Col>
          <Col span={8}>
            <Statistic
              title="Connection Status"
              value={connectionStatus}
              valueStyle={{ 
                color: connectionStatus === 'connected' ? '#3f8600' : 
                       connectionStatus === 'error' ? '#cf1322' : '#1890ff'
              }}
            />
          </Col>
          <Col span={8}>
            {latestMessage && (
              <Statistic
                title="Last Message"
                value={new Date(latestMessage.timestamp / 1000000).toLocaleTimeString()}
                valueStyle={{ fontSize: '14px' }}
              />
            )}
          </Col>
        </Row>

        {/* Backend MessageBus Info */}
        {connectionInfo && (
          <div>
            <Title level={5}>Backend MessageBus Info</Title>
            <Space direction="vertical" style={{ width: '100%' }}>
              <Row>
                <Col span={12}>
                  <Text><strong>Backend State:</strong> {connectionInfo.connection_state}</Text>
                </Col>
                <Col span={12}>
                  <Text><strong>Reconnect Attempts:</strong> {connectionInfo.reconnect_attempts}</Text>
                </Col>
              </Row>
              
              {connectionInfo.connected_at && (
                <Text><strong>Connected At:</strong> {new Date(connectionInfo.connected_at).toLocaleString()}</Text>
              )}
              
              {connectionInfo.last_message_at && (
                <Text><strong>Last Backend Message:</strong> {new Date(connectionInfo.last_message_at).toLocaleString()}</Text>
              )}
              
              <Text><strong>Backend Messages Received:</strong> {connectionInfo.messages_received}</Text>
              
              {connectionInfo.error_message && (
                <Alert
                  message="Backend Error"
                  description={connectionInfo.error_message}
                  type="error"
                  showIcon
                />
              )}
            </Space>
          </div>
        )}
      </Space>
    </Card>
  )
}

export default MessageBusStatus