import React, { useState, useEffect } from 'react';
import {
  Card,
  Row,
  Col,
  Statistic,
  Badge,
  Space,
  Typography,
  Progress,
  Tooltip,
  Tag,
  Alert,
  Button,
  Dropdown
} from 'antd';
import {
  ThunderboltOutlined,
  WifiOutlined,
  LineChartOutlined,
  SafetyCertificateOutlined,
  RocketOutlined,
  SettingOutlined,
  ReloadOutlined,
  InfoCircleOutlined,
  CheckCircleOutlined,
  ExclamationCircleOutlined,
  CloseCircleOutlined,
  MoreOutlined
} from '@ant-design/icons';
import { useEngineWebSocket } from '../../hooks/useEngineWebSocket';
import { useMessageBus } from '../../hooks/useMessageBus';
import type { 
  Sprint3StatusResponse,
  SystemHealthMetrics,
  ComponentHealth
} from '../../types/sprint3';

const { Text, Title } = Typography;

interface Sprint3StatusWidgetProps {
  size?: 'small' | 'default' | 'large';
  showDetails?: boolean;
  refreshInterval?: number;
  onFeatureClick?: (feature: string) => void;
}

const Sprint3StatusWidget: React.FC<Sprint3StatusWidgetProps> = ({
  size = 'default',
  showDetails = true,
  refreshInterval = 30000,
  onFeatureClick
}) => {
  const [status, setStatus] = useState<Sprint3StatusResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [lastUpdate, setLastUpdate] = useState<string>(new Date().toISOString());

  // Use existing hooks for real data
  const engineWs = useEngineWebSocket();
  const messageBus = useMessageBus();

  // Mock Sprint 3 status (in real implementation, this would come from API)
  useEffect(() => {
    const mockStatus: Sprint3StatusResponse = {
      websocket: {
        totalConnections: 3,
        activeConnections: messageBus.connectionStatus === 'connected' ? 3 : 2,
        messageRate: 45.2,
        errorRate: 0.01,
        averageLatency: 32,
        uptime: 99.8,
        lastUpdate: new Date().toISOString()
      },
      analytics: {
        enabled: true,
        activeStreams: 8,
        metricsCount: 156,
        lastUpdate: new Date().toISOString()
      },
      risk: {
        activeLimits: 12,
        breachCount: 2,
        alertsCount: 0,
        lastCheck: new Date().toISOString()
      },
      deployment: {
        activePipelines: 3,
        pendingApprovals: 1,
        lastDeployment: new Date(Date.now() - 3600000).toISOString() // 1 hour ago
      },
      system: {
        overall: messageBus.connectionStatus === 'connected' ? 'healthy' : 'warning',
        components: [
          {
            id: 'websocket',
            name: 'WebSocket Infrastructure',
            type: 'websocket',
            status: messageBus.connectionStatus === 'connected' ? 'healthy' : 'warning',
            metrics: [
              { name: 'connections', value: 3, unit: 'count', status: 'normal' },
              { name: 'latency', value: 32, unit: 'ms', status: 'normal' },
              { name: 'uptime', value: 99.8, unit: '%', status: 'normal' }
            ],
            lastCheck: new Date().toISOString(),
            dependencies: ['messagebus', 'engine-ws']
          },
          {
            id: 'analytics',
            name: 'Real-time Analytics',
            type: 'service',
            status: 'healthy',
            metrics: [
              { name: 'streams', value: 8, unit: 'count', status: 'normal' },
              { name: 'throughput', value: 156, unit: 'msg/s', status: 'normal' }
            ],
            lastCheck: new Date().toISOString(),
            dependencies: ['websocket']
          },
          {
            id: 'risk',
            name: 'Risk Management',
            type: 'service',
            status: 'warning',
            metrics: [
              { name: 'limits', value: 12, unit: 'count', status: 'normal' },
              { name: 'breaches', value: 2, unit: 'count', status: 'warning' }
            ],
            lastCheck: new Date().toISOString(),
            dependencies: ['analytics']
          },
          {
            id: 'deployment',
            name: 'Strategy Deployment',
            type: 'service',
            status: 'healthy',
            metrics: [
              { name: 'pipelines', value: 3, unit: 'count', status: 'normal' },
              { name: 'pending', value: 1, unit: 'count', status: 'normal' }
            ],
            lastCheck: new Date().toISOString(),
            dependencies: ['risk']
          }
        ],
        lastUpdate: new Date().toISOString(),
        uptime: 99.8,
        alerts: [
          ...(messageBus.connectionStatus !== 'connected' ? [{
            id: 'ws-1',
            type: 'system' as const,
            severity: 'warning' as const,
            message: 'MessageBus WebSocket connection unstable',
            source: 'websocket',
            timestamp: new Date().toISOString(),
            acknowledged: false
          }] : []),
          {
            id: 'risk-1',
            type: 'performance' as const,
            severity: 'info' as const,
            message: '2 risk limit breaches detected - within normal thresholds',
            source: 'risk-engine',
            timestamp: new Date(Date.now() - 300000).toISOString(), // 5 min ago
            acknowledged: true
          }
        ]
      }
    };

    setStatus(mockStatus);
    setLoading(false);
    setLastUpdate(new Date().toISOString());
  }, [messageBus.connectionStatus, messageBus.messagesReceived]);

  // Auto-refresh
  useEffect(() => {
    const interval = setInterval(() => {
      // Trigger refresh
      setLastUpdate(new Date().toISOString());
    }, refreshInterval);

    return () => clearInterval(interval);
  }, [refreshInterval]);

  if (loading || !status) {
    return (
      <Card loading={loading} title="Sprint 3 Status" size={size}>
        <div style={{ height: 200 }} />
      </Card>
    );
  }

  // Get overall health status
  const getOverallStatus = () => {
    const { system } = status;
    const criticalCount = system.components.filter(c => c.status === 'critical').length;
    const warningCount = system.components.filter(c => c.status === 'warning').length;
    
    if (criticalCount > 0) return { status: 'critical', color: '#ff4d4f', text: 'Critical Issues' };
    if (warningCount > 0) return { status: 'warning', color: '#faad14', text: 'Warnings Present' };
    return { status: 'healthy', color: '#52c41a', text: 'All Systems Operational' };
  };

  const overallStatus = getOverallStatus();

  // Feature cards data
  const features = [
    {
      key: 'websocket',
      title: 'WebSocket',
      icon: <WifiOutlined />,
      value: status.websocket.activeConnections,
      total: status.websocket.totalConnections,
      status: status.websocket.activeConnections === status.websocket.totalConnections ? 'success' : 'warning',
      details: `${status.websocket.averageLatency.toFixed(0)}ms avg latency`
    },
    {
      key: 'analytics',
      title: 'Analytics',
      icon: <LineChartOutlined />,
      value: status.analytics.activeStreams,
      total: null,
      status: status.analytics.enabled ? 'success' : 'error',
      details: `${status.analytics.metricsCount} metrics/min`
    },
    {
      key: 'risk',
      title: 'Risk Mgmt',
      icon: <SafetyCertificateOutlined />,
      value: status.risk.activeLimits,
      total: null,
      status: status.risk.breachCount === 0 ? 'success' : 'warning',
      details: `${status.risk.breachCount} breaches`
    },
    {
      key: 'deployment',
      title: 'Deployment',
      icon: <RocketOutlined />,
      value: status.deployment.activePipelines,
      total: null,
      status: status.deployment.pendingApprovals === 0 ? 'success' : 'processing',
      details: `${status.deployment.pendingApprovals} pending`
    }
  ];

  // Dropdown menu for actions
  const actionMenu = {
    items: [
      {
        key: 'refresh',
        label: 'Refresh Status',
        icon: <ReloadOutlined />
      },
      {
        key: 'configure',
        label: 'Configure Features',
        icon: <SettingOutlined />
      },
      {
        key: 'details',
        label: 'View Details',
        icon: <InfoCircleOutlined />
      }
    ],
    onClick: ({ key }: { key: string }) => {
      switch (key) {
        case 'refresh':
          setLastUpdate(new Date().toISOString());
          break;
        case 'configure':
          onFeatureClick?.('settings');
          break;
        case 'details':
          onFeatureClick?.('dashboard');
          break;
      }
    }
  };

  if (size === 'small') {
    return (
      <Card size="small" style={{ width: '100%' }}>
        <Space direction="vertical" style={{ width: '100%' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
            <Space>
              <ThunderboltOutlined style={{ color: overallStatus.color }} />
              <Text strong>Sprint 3</Text>
              <Badge status={overallStatus.status === 'healthy' ? 'success' : 'warning'} />
            </Space>
            <Text type="secondary" style={{ fontSize: '12px' }}>
              {new Date(lastUpdate).toLocaleTimeString()}
            </Text>
          </div>
          <Row gutter={8}>
            {features.map(feature => (
              <Col span={6} key={feature.key}>
                <div style={{ textAlign: 'center' }}>
                  <div style={{ 
                    color: feature.status === 'success' ? '#52c41a' : 
                           feature.status === 'warning' ? '#faad14' : '#ff4d4f',
                    fontSize: '18px' 
                  }}>
                    {feature.icon}
                  </div>
                  <Text style={{ fontSize: '12px' }}>{feature.value}</Text>
                </div>
              </Col>
            ))}
          </Row>
        </Space>
      </Card>
    );
  }

  return (
    <Card
      title={
        <Space>
          <ThunderboltOutlined style={{ color: overallStatus.color }} />
          <Title level={4} style={{ margin: 0 }}>
            Sprint 3 System Status
          </Title>
          <Tag color={overallStatus.status === 'healthy' ? 'success' : 'warning'}>
            {overallStatus.text}
          </Tag>
        </Space>
      }
      extra={
        <Space>
          <Text type="secondary" style={{ fontSize: '12px' }}>
            Last updated: {new Date(lastUpdate).toLocaleTimeString()}
          </Text>
          <Dropdown menu={actionMenu} placement="bottomRight">
            <Button size="small" icon={<MoreOutlined />} />
          </Dropdown>
        </Space>
      }
      size={size}
    >
      {/* System Health Alert */}
      {overallStatus.status !== 'healthy' && (
        <Alert
          message={overallStatus.text}
          description={
            status.system.alerts
              .filter(alert => !alert.acknowledged && alert.severity !== 'info')
              .map(alert => alert.message)
              .join(', ') || 'Some components require attention'
          }
          type={overallStatus.status === 'critical' ? 'error' : 'warning'}
          showIcon
          closable
          style={{ marginBottom: 16 }}
          action={
            <Button size="small" onClick={() => onFeatureClick?.('system-health')}>
              View Details
            </Button>
          }
        />
      )}

      {/* Feature Status Grid */}
      <Row gutter={[16, 16]}>
        {features.map(feature => (
          <Col xs={12} sm={6} key={feature.key}>
            <Card
              size="small"
              hoverable
              onClick={() => onFeatureClick?.(feature.key)}
              style={{
                borderColor: feature.status === 'success' ? '#52c41a' : 
                           feature.status === 'warning' ? '#faad14' : '#ff4d4f',
                cursor: 'pointer'
              }}
            >
              <Space direction="vertical" style={{ width: '100%', textAlign: 'center' }}>
                <div style={{ 
                  fontSize: '24px',
                  color: feature.status === 'success' ? '#52c41a' : 
                         feature.status === 'warning' ? '#faad14' : '#ff4d4f'
                }}>
                  {feature.icon}
                </div>
                <div>
                  <div style={{ fontSize: '18px', fontWeight: 'bold' }}>
                    {feature.value}
                    {feature.total && <span style={{ fontSize: '14px', color: '#999' }}>/{feature.total}</span>}
                  </div>
                  <Text type="secondary" style={{ fontSize: '12px' }}>
                    {feature.title}
                  </Text>
                </div>
                {showDetails && (
                  <Text type="secondary" style={{ fontSize: '11px' }}>
                    {feature.details}
                  </Text>
                )}
              </Space>
            </Card>
          </Col>
        ))}
      </Row>

      {showDetails && (
        <>
          {/* System Uptime Progress */}
          <div style={{ marginTop: 16 }}>
            <Row gutter={[16, 8]}>
              <Col span={12}>
                <Text strong>System Uptime</Text>
                <Progress
                  percent={status.system.uptime}
                  size="small"
                  status="active"
                  format={(percent) => `${percent}%`}
                />
              </Col>
              <Col span={12}>
                <Text strong>WebSocket Health</Text>
                <Progress
                  percent={(status.websocket.activeConnections / status.websocket.totalConnections) * 100}
                  size="small"
                  status={status.websocket.activeConnections === status.websocket.totalConnections ? 'success' : 'exception'}
                  format={(percent) => `${status.websocket.activeConnections}/${status.websocket.totalConnections}`}
                />
              </Col>
            </Row>
          </div>

          {/* Recent Activity Summary */}
          <div style={{ marginTop: 16, padding: 12, backgroundColor: '#fafafa', borderRadius: 6 }}>
            <Text strong style={{ fontSize: '12px' }}>Recent Activity:</Text>
            <div style={{ marginTop: 8 }}>
              <Space direction="vertical" size="small" style={{ width: '100%' }}>
                <Space>
                  <Badge status="success" />
                  <Text style={{ fontSize: '11px' }}>
                    Analytics processing {status.analytics.metricsCount} metrics/min
                  </Text>
                </Space>
                <Space>
                  <Badge status="processing" />
                  <Text style={{ fontSize: '11px' }}>
                    {status.deployment.activePipelines} deployment pipelines active
                  </Text>
                </Space>
                {status.risk.breachCount > 0 && (
                  <Space>
                    <Badge status="warning" />
                    <Text style={{ fontSize: '11px' }}>
                      {status.risk.breachCount} risk limit breaches detected
                    </Text>
                  </Space>
                )}
              </Space>
            </div>
          </div>
        </>
      )}
    </Card>
  );
};

export default Sprint3StatusWidget;