/**
 * WebSocket Connection Manager Component
 * Sprint 3: Advanced WebSocket Infrastructure
 * 
 * Provides real-time connection status, management, and controls for WebSocket connections
 * with comprehensive monitoring and automatic reconnection capabilities.
 */

import React, { useState, useEffect } from 'react';
import { Card, Badge, Button, Progress, Tooltip, Alert, Space, Typography, Row, Col, Divider, Switch } from 'antd';
import { 
  WifiOutlined, 
  DisconnectOutlined, 
  ReloadOutlined, 
  SettingOutlined,
  CheckCircleOutlined,
  ExclamationCircleOutlined,
  CloseCircleOutlined,
  SyncOutlined
} from '@ant-design/icons';
import { useWebSocketManager } from '../../hooks/useWebSocketManager';
import { ConnectionStatistics } from './ConnectionStatistics';

const { Title, Text } = Typography;

interface WebSocketConnectionManagerProps {
  className?: string;
  showDetailedStats?: boolean;
  showReconnectControls?: boolean;
  showSubscriptionCount?: boolean;
  onConnectionChange?: (status: string) => void;
}

export const WebSocketConnectionManager: React.FC<WebSocketConnectionManagerProps> = ({
  className,
  showDetailedStats = true,
  showReconnectControls = true,
  showSubscriptionCount = true,
  onConnectionChange
}) => {
  const {
    connectionState,
    connectionError,
    connectionAttempts,
    isReconnecting,
    messageLatency,
    messagesReceived,
    subscriptionCount,
    connect,
    disconnect,
    reconnect,
    getConnectionInfo,
    enableAutoReconnect,
    disableAutoReconnect,
    autoReconnectEnabled
  } = useWebSocketManager();

  const [connectionInfo, setConnectionInfo] = useState<any>(null);
  const [showAdvancedStats, setShowAdvancedStats] = useState(false);

  // Update connection info periodically
  useEffect(() => {
    const updateInfo = () => {
      const info = getConnectionInfo();
      setConnectionInfo(info);
    };

    updateInfo();
    const interval = setInterval(updateInfo, 1000);

    return () => clearInterval(interval);
  }, [getConnectionInfo]);

  // Notify parent of connection changes
  useEffect(() => {
    if (onConnectionChange) {
      onConnectionChange(connectionState);
    }
  }, [connectionState, onConnectionChange]);

  const getConnectionStatusIcon = () => {
    switch (connectionState) {
      case 'connected':
        return <CheckCircleOutlined style={{ color: '#52c41a' }} />;
      case 'connecting':
      case 'reconnecting':
        return <SyncOutlined spin style={{ color: '#1890ff' }} />;
      case 'disconnected':
        return <DisconnectOutlined style={{ color: '#d9d9d9' }} />;
      case 'error':
        return <CloseCircleOutlined style={{ color: '#ff4d4f' }} />;
      default:
        return <ExclamationCircleOutlined style={{ color: '#faad14' }} />;
    }
  };

  const getConnectionStatusColor = (): 'success' | 'processing' | 'default' | 'error' | 'warning' => {
    switch (connectionState) {
      case 'connected':
        return 'success';
      case 'connecting':
      case 'reconnecting':
        return 'processing';
      case 'disconnected':
        return 'default';
      case 'error':
        return 'error';
      default:
        return 'warning';
    }
  };

  const getConnectionHealthProgress = () => {
    switch (connectionState) {
      case 'connected':
        return messageLatency < 100 ? 100 : messageLatency < 500 ? 75 : 50;
      case 'connecting':
      case 'reconnecting':
        return 30;
      case 'disconnected':
        return 0;
      case 'error':
        return 10;
      default:
        return 25;
    }
  };

  const formatLatency = (latency: number): string => {
    if (latency === 0) return 'N/A';
    return latency < 1000 ? `${latency.toFixed(0)}ms` : `${(latency / 1000).toFixed(1)}s`;
  };

  const handleConnect = async () => {
    try {
      await connect();
    } catch (error) {
      console.error('Failed to connect:', error);
    }
  };

  const handleDisconnect = async () => {
    try {
      await disconnect();
    } catch (error) {
      console.error('Failed to disconnect:', error);
    }
  };

  const handleReconnect = async () => {
    try {
      await reconnect();
    } catch (error) {
      console.error('Failed to reconnect:', error);
    }
  };

  const handleAutoReconnectToggle = (enabled: boolean) => {
    if (enabled) {
      enableAutoReconnect();
    } else {
      disableAutoReconnect();
    }
  };

  return (
    <div className={className}>
      <Card 
        title={
          <Space>
            <WifiOutlined />
            <span>WebSocket Connection</span>
            {getConnectionStatusIcon()}
          </Space>
        }
        size="small"
        extra={
          showReconnectControls && (
            <Space>
              {connectionState === 'disconnected' || connectionState === 'error' ? (
                <Button 
                  type="primary" 
                  size="small" 
                  icon={<WifiOutlined />}
                  onClick={handleConnect}
                  loading={connectionState === 'connecting'}
                >
                  Connect
                </Button>
              ) : (
                <Button 
                  size="small" 
                  icon={<DisconnectOutlined />}
                  onClick={handleDisconnect}
                >
                  Disconnect
                </Button>
              )}
              <Button 
                size="small" 
                icon={<ReloadOutlined />}
                onClick={handleReconnect}
                loading={isReconnecting}
              >
                Reconnect
              </Button>
              <Tooltip title="Toggle advanced statistics">
                <Button 
                  size="small" 
                  icon={<SettingOutlined />}
                  onClick={() => setShowAdvancedStats(!showAdvancedStats)}
                />
              </Tooltip>
            </Space>
          )
        }
      >
        {/* Connection Status */}
        <Row gutter={[16, 16]}>
          <Col span={24}>
            <Space size="large">
              <Badge 
                status={getConnectionStatusColor()} 
                text={
                  <span style={{ textTransform: 'capitalize' }}>
                    {connectionState}
                    {isReconnecting && ' (Reconnecting...)'}
                  </span>
                } 
              />
              
              {showSubscriptionCount && (
                <Text>
                  <strong>Subscriptions:</strong> {subscriptionCount}
                </Text>
              )}
              
              <Text>
                <strong>Messages:</strong> {messagesReceived.toLocaleString()}
              </Text>
              
              <Text>
                <strong>Latency:</strong> {formatLatency(messageLatency)}
              </Text>
            </Space>
          </Col>
        </Row>

        {/* Connection Health Progress */}
        <Row gutter={[16, 8]} style={{ marginTop: 12 }}>
          <Col span={24}>
            <Text strong>Connection Health</Text>
            <Progress 
              percent={getConnectionHealthProgress()}
              strokeColor={{
                from: '#108ee9',
                to: '#87d068',
              }}
              status={connectionState === 'error' ? 'exception' : 'active'}
              showInfo={false}
              size="small"
            />
          </Col>
        </Row>

        {/* Auto-reconnect Setting */}
        <Row style={{ marginTop: 12 }}>
          <Col span={24}>
            <Space>
              <Switch 
                checked={autoReconnectEnabled}
                onChange={handleAutoReconnectToggle}
                size="small"
              />
              <Text>Auto-reconnect</Text>
              {connectionAttempts > 0 && (
                <Text type="secondary">
                  (Attempt {connectionAttempts})
                </Text>
              )}
            </Space>
          </Col>
        </Row>

        {/* Connection Error */}
        {connectionError && (
          <Alert
            message="Connection Error"
            description={connectionError}
            type="error"
            showIcon
            style={{ marginTop: 12 }}
            action={
              <Button size="small" type="text" onClick={handleReconnect}>
                Retry
              </Button>
            }
          />
        )}

        {/* Advanced Statistics */}
        {showAdvancedStats && showDetailedStats && connectionInfo && (
          <>
            <Divider />
            <Title level={5}>Connection Details</Title>
            <ConnectionStatistics connectionInfo={connectionInfo} />
            
            <Row gutter={[16, 8]} style={{ marginTop: 12 }}>
              <Col span={12}>
                <Text strong>Connected Since:</Text>
                <br />
                <Text type="secondary">
                  {connectionInfo.connectedAt ? 
                    new Date(connectionInfo.connectedAt).toLocaleTimeString() : 
                    'N/A'
                  }
                </Text>
              </Col>
              
              <Col span={12}>
                <Text strong>Last Activity:</Text>
                <br />
                <Text type="secondary">
                  {connectionInfo.lastActivity ? 
                    new Date(connectionInfo.lastActivity).toLocaleTimeString() : 
                    'N/A'
                  }
                </Text>
              </Col>
            </Row>

            <Row gutter={[16, 8]} style={{ marginTop: 8 }}>
              <Col span={12}>
                <Text strong>Protocol Version:</Text>
                <br />
                <Text type="secondary">{connectionInfo.protocolVersion || '2.0'}</Text>
              </Col>
              
              <Col span={12}>
                <Text strong>Session ID:</Text>
                <br />
                <Text type="secondary" copyable>
                  {connectionInfo.sessionId || 'N/A'}
                </Text>
              </Col>
            </Row>
          </>
        )}
      </Card>
    </div>
  );
};

export default WebSocketConnectionManager;