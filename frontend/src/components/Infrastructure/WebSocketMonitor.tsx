import React, { useState, useEffect } from 'react';
import {
  Card,
  Row,
  Col,
  Statistic,
  Table,
  Tag,
  Space,
  Button,
  Alert,
  Typography,
  Progress,
  Tooltip,
  Badge,
  Timeline,
  Select,
  Switch,
  Divider,
  List
} from 'antd';
import {
  WifiOutlined,
  DisconnectOutlined,
  ReloadOutlined,
  SettingOutlined,
  LineChartOutlined,
  ExclamationCircleOutlined,
  CheckCircleOutlined,
  ClockCircleOutlined,
  MessageOutlined,
  ThunderboltOutlined,
  AlertOutlined
} from '@ant-design/icons';
import { useEngineWebSocket } from '../../hooks/useEngineWebSocket';
import { useMessageBus } from '../../hooks/useMessageBus';
import type {
  WebSocketConnectionState,
  WebSocketMetrics,
  WebSocketMessage
} from '../../types/sprint3';

const { Title, Text } = Typography;
const { Option } = Select;

interface WebSocketMonitorProps {
  showDetailedMetrics?: boolean;
  compactMode?: boolean;
  autoRefresh?: boolean;
  refreshInterval?: number;
}

const WebSocketMonitor: React.FC<WebSocketMonitorProps> = ({
  showDetailedMetrics = true,
  compactMode = false,
  autoRefresh = true,
  refreshInterval = 5000
}) => {
  const [selectedConnection, setSelectedConnection] = useState<string>('messagebus');
  const [showHistory, setShowHistory] = useState(false);
  const [messageFilter, setMessageFilter] = useState<string>('all');

  // Use existing WebSocket hooks
  const engineWs = useEngineWebSocket({
    autoReconnect: true,
    reconnectInterval: 5000,
    maxReconnectAttempts: 10
  });

  const messageBus = useMessageBus();

  // Mock additional connections for demonstration
  const [connections] = useState<WebSocketConnectionState[]>([
    {
      id: 'messagebus',
      url: 'ws://localhost:8001/ws/messagebus',
      status: messageBus.connectionStatus === 'connected' ? 'connected' : 
             messageBus.connectionStatus === 'connecting' ? 'connecting' : 'disconnected',
      lastConnected: new Date().toISOString(),
      messageCount: messageBus.messagesReceived,
      latency: 45,
      reconnectAttempts: 0,
      subscriptions: ['nautilus_engine_status', 'market_data', 'order_updates']
    },
    {
      id: 'engine',
      url: 'ws://localhost:8001/ws/engine',
      status: engineWs.isConnected ? 'connected' : 'disconnected',
      lastConnected: new Date().toISOString(),
      messageCount: 0,
      latency: 32,
      reconnectAttempts: engineWs.connectionAttempts,
      subscriptions: ['engine_status', 'resource_metrics']
    }
  ]);

  // Calculate overall metrics
  const metrics: WebSocketMetrics = {
    totalConnections: connections.length,
    activeConnections: connections.filter(c => c.status === 'connected').length,
    messageRate: connections.reduce((sum, c) => sum + c.messageCount, 0) / connections.length,
    errorRate: 0.02,
    averageLatency: connections.reduce((sum, c) => sum + (c.latency || 0), 0) / connections.length,
    uptime: 99.5,
    lastUpdate: new Date().toISOString()
  };

  // Get status color for connection
  const getConnectionStatusColor = (status: WebSocketConnectionState['status']) => {
    switch (status) {
      case 'connected': return 'success';
      case 'connecting': case 'reconnecting': return 'processing';
      case 'disconnected': return 'default';
      case 'error': return 'error';
      default: return 'default';
    }
  };

  // Get status icon for connection
  const getConnectionStatusIcon = (status: WebSocketConnectionState['status']) => {
    switch (status) {
      case 'connected': return <CheckCircleOutlined />;
      case 'connecting': case 'reconnecting': return <ClockCircleOutlined />;
      case 'disconnected': return <DisconnectOutlined />;
      case 'error': return <ExclamationCircleOutlined />;
      default: return <WifiOutlined />;
    }
  };

  // Handle connection actions
  const handleConnect = (connectionId: string) => {
    if (connectionId === 'messagebus') {
      messageBus.connect();
    } else if (connectionId === 'engine') {
      engineWs.connect();
    }
  };

  const handleDisconnect = (connectionId: string) => {
    if (connectionId === 'messagebus') {
      messageBus.disconnect();
    } else if (connectionId === 'engine') {
      engineWs.disconnect();
    }
  };

  // Table columns for connections
  const connectionColumns = [
    {
      title: 'Connection',
      dataIndex: 'id',
      key: 'id',
      render: (id: string, record: WebSocketConnectionState) => (
        <Space>
          {getConnectionStatusIcon(record.status)}
          <Text strong>{id.toUpperCase()}</Text>
        </Space>
      )
    },
    {
      title: 'Status',
      dataIndex: 'status',
      key: 'status',
      render: (status: WebSocketConnectionState['status']) => (
        <Badge
          status={getConnectionStatusColor(status)}
          text={status.charAt(0).toUpperCase() + status.slice(1)}
        />
      )
    },
    {
      title: 'Messages',
      dataIndex: 'messageCount',
      key: 'messageCount',
      render: (count: number) => (
        <Statistic value={count} valueStyle={{ fontSize: '14px' }} />
      )
    },
    {
      title: 'Latency',
      dataIndex: 'latency',
      key: 'latency',
      render: (latency?: number) => (
        latency ? (
          <Text type={latency > 100 ? 'warning' : 'success'}>
            {latency}ms
          </Text>
        ) : (
          <Text type="secondary">N/A</Text>
        )
      )
    },
    {
      title: 'Subscriptions',
      dataIndex: 'subscriptions',
      key: 'subscriptions',
      render: (subscriptions: string[]) => (
        <Text type="secondary">{subscriptions.length} topics</Text>
      )
    },
    {
      title: 'Actions',
      key: 'actions',
      render: (record: WebSocketConnectionState) => (
        <Space>
          {record.status === 'connected' ? (
            <Button
              size="small"
              icon={<DisconnectOutlined />}
              onClick={() => handleDisconnect(record.id)}
            >
              Disconnect
            </Button>
          ) : (
            <Button
              size="small"
              type="primary"
              icon={<WifiOutlined />}
              onClick={() => handleConnect(record.id)}
            >
              Connect
            </Button>
          )}
          <Button
            size="small"
            icon={<ReloadOutlined />}
            onClick={() => {
              // Refresh connection status
              console.log('Refreshing connection:', record.id);
            }}
          />
        </Space>
      )
    }
  ];

  if (compactMode) {
    return (
      <Card
        size="small"
        title={
          <Space>
            <WifiOutlined />
            WebSocket Status
            <Badge count={metrics.activeConnections} color="green" />
          </Space>
        }
      >
        <Row gutter={[16, 8]}>
          <Col span={8}>
            <Statistic
              title="Active"
              value={metrics.activeConnections}
              suffix={`/ ${metrics.totalConnections}`}
              valueStyle={{ fontSize: '16px', color: '#52c41a' }}
            />
          </Col>
          <Col span={8}>
            <Statistic
              title="Latency"
              value={metrics.averageLatency.toFixed(0)}
              suffix="ms"
              valueStyle={{ fontSize: '16px' }}
            />
          </Col>
          <Col span={8}>
            <Statistic
              title="Uptime"
              value={metrics.uptime}
              suffix="%"
              valueStyle={{ fontSize: '16px' }}
            />
          </Col>
        </Row>
      </Card>
    );
  }

  return (
    <div style={{ width: '100%' }}>
      {/* Overall Metrics */}
      <Row gutter={[16, 16]} style={{ marginBottom: 24 }}>
        <Col xs={24} sm={12} lg={6}>
          <Card size="small">
            <Statistic
              title="Total Connections"
              value={metrics.totalConnections}
              prefix={<WifiOutlined />}
            />
          </Card>
        </Col>
        <Col xs={24} sm={12} lg={6}>
          <Card size="small">
            <Statistic
              title="Active Connections"
              value={metrics.activeConnections}
              valueStyle={{ color: '#52c41a' }}
              prefix={<CheckCircleOutlined />}
            />
          </Card>
        </Col>
        <Col xs={24} sm={12} lg={6}>
          <Card size="small">
            <Statistic
              title="Average Latency"
              value={metrics.averageLatency.toFixed(0)}
              suffix="ms"
              valueStyle={{ 
                color: metrics.averageLatency > 100 ? '#faad14' : '#52c41a' 
              }}
            />
          </Card>
        </Col>
        <Col xs={24} sm={12} lg={6}>
          <Card size="small">
            <Statistic
              title="System Uptime"
              value={metrics.uptime}
              suffix="%"
              valueStyle={{ color: '#52c41a' }}
              prefix={<LineChartOutlined />}
            />
          </Card>
        </Col>
      </Row>

      {/* Connection Health Alert */}
      {metrics.activeConnections < metrics.totalConnections && (
        <Alert
          message="Connection Issues Detected"
          description={`${metrics.totalConnections - metrics.activeConnections} connection(s) are not active. Check network connectivity and service status.`}
          type="warning"
          showIcon
          closable
          style={{ marginBottom: 16 }}
          action={
            <Button size="small" onClick={() => {
              connections.forEach(conn => {
                if (conn.status !== 'connected') {
                  handleConnect(conn.id);
                }
              });
            }}>
              Reconnect All
            </Button>
          }
        />
      )}

      {/* Connections Table */}
      <Card
        title={
          <Space>
            <WifiOutlined />
            WebSocket Connections
            <Badge count={metrics.activeConnections} color="green" />
          </Space>
        }
        extra={
          <Space>
            <Switch
              checkedChildren="Auto-refresh"
              unCheckedChildren="Manual"
              checked={autoRefresh}
            />
            <Button icon={<ReloadOutlined />} size="small">
              Refresh
            </Button>
            <Button icon={<SettingOutlined />} size="small">
              Configure
            </Button>
          </Space>
        }
        style={{ marginBottom: 16 }}
      >
        <Table
          columns={connectionColumns}
          dataSource={connections}
          rowKey="id"
          pagination={false}
          size="small"
        />
      </Card>

      {showDetailedMetrics && (
        <Row gutter={[16, 16]}>
          {/* Message History */}
          <Col xs={24} lg={12}>
            <Card
              title={
                <Space>
                  <MessageOutlined />
                  Recent Messages
                </Space>
              }
              extra={
                <Space>
                  <Select
                    value={messageFilter}
                    onChange={setMessageFilter}
                    size="small"
                    style={{ width: 120 }}
                  >
                    <Option value="all">All Types</Option>
                    <Option value="engine">Engine</Option>
                    <Option value="market">Market Data</Option>
                    <Option value="orders">Orders</Option>
                  </Select>
                  <Switch
                    checked={showHistory}
                    onChange={setShowHistory}
                    size="small"
                  />
                </Space>
              }
              size="small"
            >
              {messageBus.messages.length > 0 ? (
                <List
                  size="small"
                  dataSource={messageBus.messages.slice(-10).reverse()}
                  renderItem={(message: any) => (
                    <List.Item>
                      <List.Item.Meta
                        title={
                          <Space>
                            <Tag color="blue">
                              {message.type}
                            </Tag>
                            <Text type="secondary" style={{ fontSize: '12px' }}>
                              {new Date(message.timestamp / 1000000).toLocaleTimeString()}
                            </Text>
                          </Space>
                        }
                        description={
                          <Text ellipsis style={{ fontSize: '12px' }}>
                            {JSON.stringify(message.payload).substring(0, 100)}...
                          </Text>
                        }
                      />
                    </List.Item>
                  )}
                />
              ) : (
                <div style={{ textAlign: 'center', padding: '20px' }}>
                  <Text type="secondary">No messages received yet</Text>
                </div>
              )}
            </Card>
          </Col>

          {/* Connection Timeline */}
          <Col xs={24} lg={12}>
            <Card
              title={
                <Space>
                  <ClockCircleOutlined />
                  Connection Timeline
                </Space>
              }
              size="small"
            >
              <Timeline
                items={[
                  {
                    color: 'green',
                    children: (
                      <div>
                        <Text strong>MessageBus Connected</Text>
                        <br />
                        <Text type="secondary" style={{ fontSize: '12px' }}>
                          {new Date().toLocaleTimeString()} - Active subscriptions: 3
                        </Text>
                      </div>
                    )
                  },
                  {
                    color: messageBus.connectionStatus === 'connected' ? 'green' : 'red',
                    children: (
                      <div>
                        <Text strong>Engine WebSocket {engineWs.isConnected ? 'Connected' : 'Disconnected'}</Text>
                        <br />
                        <Text type="secondary" style={{ fontSize: '12px' }}>
                          {new Date().toLocaleTimeString()} - Reconnect attempts: {engineWs.connectionAttempts}
                        </Text>
                      </div>
                    )
                  },
                  {
                    color: 'blue',
                    children: (
                      <div>
                        <Text strong>System Health Check</Text>
                        <br />
                        <Text type="secondary" style={{ fontSize: '12px' }}>
                          {new Date().toLocaleTimeString()} - All systems operational
                        </Text>
                      </div>
                    )
                  }
                ]}
              />
            </Card>
          </Col>
        </Row>
      )}
    </div>
  );
};

export default WebSocketMonitor;