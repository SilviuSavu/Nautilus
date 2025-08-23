import React, { useState, useEffect } from 'react';
import {
  Card,
  Row,
  Col,
  Typography,
  Progress,
  Timeline,
  Alert,
  Table,
  Tag,
  Space,
  Button,
  Statistic,
  Badge,
  Tooltip,
  Tabs,
  List,
  Divider,
  Switch,
  Select
} from 'antd';
import {
  MonitorOutlined,
  CheckCircleOutlined,
  ExclamationCircleOutlined,
  CloseCircleOutlined,
  ReloadOutlined,
  SettingOutlined,
  AlertOutlined,
  DatabaseOutlined,
  WifiOutlined,
  ApiOutlined,
  ThunderboltOutlined,
  LineChartOutlined,
  SafetyCertificateOutlined,
  RocketOutlined,
  InfoCircleOutlined,
  WarningOutlined
} from '@ant-design/icons';
import { useEngineWebSocket } from '../../hooks/useEngineWebSocket';
import { useMessageBus } from '../../hooks/useMessageBus';
import type {
  SystemHealthMetrics,
  ComponentHealth,
  SystemAlert,
  ComponentMetric
} from '../../types/sprint3';

const { Title, Text } = Typography;
const { TabPane } = Tabs;
const { Option } = Select;

interface SystemHealthOverviewProps {
  autoRefresh?: boolean;
  refreshInterval?: number;
  showAlerts?: boolean;
  compactMode?: boolean;
}

const SystemHealthOverview: React.FC<SystemHealthOverviewProps> = ({
  autoRefresh = true,
  refreshInterval = 15000,
  showAlerts = true,
  compactMode = false
}) => {
  const [healthMetrics, setHealthMetrics] = useState<SystemHealthMetrics | null>(null);
  const [loading, setLoading] = useState(true);
  const [selectedComponent, setSelectedComponent] = useState<string>('all');
  const [alertFilter, setAlertFilter] = useState<'all' | 'unacknowledged' | 'critical'>('unacknowledged');

  // Use existing hooks for real system status
  const engineWs = useEngineWebSocket();
  const messageBus = useMessageBus();

  useEffect(() => {
    // Mock system health data (in real implementation, this would come from API)
    const mockHealthMetrics: SystemHealthMetrics = {
      overall: messageBus.connectionStatus === 'connected' && engineWs.isConnected ? 'healthy' : 'warning',
      uptime: 99.7,
      lastUpdate: new Date().toISOString(),
      components: [
        {
          id: 'messagebus',
          name: 'MessageBus WebSocket',
          type: 'websocket',
          status: messageBus.connectionStatus === 'connected' ? 'healthy' : 'warning',
          metrics: [
            { name: 'connection_status', value: messageBus.connectionStatus === 'connected' ? 1 : 0, unit: 'bool', status: messageBus.connectionStatus === 'connected' ? 'normal' : 'critical' },
            { name: 'messages_received', value: messageBus.messagesReceived, unit: 'count', status: 'normal' },
            { name: 'uptime', value: 98.5, unit: '%', status: 'normal' },
            { name: 'latency', value: 45, unit: 'ms', status: 'normal' }
          ],
          lastCheck: new Date().toISOString(),
          dependencies: ['backend-api']
        },
        {
          id: 'engine-websocket',
          name: 'Engine WebSocket',
          type: 'websocket',
          status: engineWs.isConnected ? 'healthy' : 'critical',
          metrics: [
            { name: 'connection_status', value: engineWs.isConnected ? 1 : 0, unit: 'bool', status: engineWs.isConnected ? 'normal' : 'critical' },
            { name: 'reconnect_attempts', value: engineWs.connectionAttempts, unit: 'count', status: engineWs.connectionAttempts > 5 ? 'warning' : 'normal' },
            { name: 'last_message', value: engineWs.lastMessage ? 1 : 0, unit: 'bool', status: 'normal' }
          ],
          lastCheck: new Date().toISOString(),
          dependencies: ['messagebus']
        },
        {
          id: 'backend-api',
          name: 'Backend API',
          type: 'api',
          status: 'healthy',
          metrics: [
            { name: 'response_time', value: 120, unit: 'ms', status: 'normal' },
            { name: 'success_rate', value: 99.8, unit: '%', status: 'normal' },
            { name: 'active_connections', value: 24, unit: 'count', status: 'normal' },
            { name: 'cpu_usage', value: 35, unit: '%', status: 'normal' },
            { name: 'memory_usage', value: 2.1, unit: 'GB', status: 'normal' }
          ],
          lastCheck: new Date().toISOString(),
          dependencies: ['database']
        },
        {
          id: 'database',
          name: 'PostgreSQL Database',
          type: 'database',
          status: 'healthy',
          metrics: [
            { name: 'connection_pool', value: 18, unit: 'count', status: 'normal', threshold: 50 },
            { name: 'query_time', value: 25, unit: 'ms', status: 'normal', threshold: 100 },
            { name: 'disk_usage', value: 67, unit: '%', status: 'normal', threshold: 80 },
            { name: 'active_queries', value: 3, unit: 'count', status: 'normal' }
          ],
          lastCheck: new Date().toISOString(),
          dependencies: []
        },
        {
          id: 'nautilus-engine',
          name: 'NautilusTrader Engine',
          type: 'engine',
          status: 'healthy',
          metrics: [
            { name: 'engine_status', value: 1, unit: 'bool', status: 'normal' },
            { name: 'strategies_active', value: 3, unit: 'count', status: 'normal' },
            { name: 'orders_per_sec', value: 12, unit: 'rate', status: 'normal' },
            { name: 'memory_usage', value: 1.8, unit: 'GB', status: 'normal' }
          ],
          lastCheck: new Date().toISOString(),
          dependencies: ['backend-api']
        },
        {
          id: 'analytics-engine',
          name: 'Analytics Engine',
          type: 'service',
          status: 'healthy',
          metrics: [
            { name: 'metrics_per_sec', value: 156, unit: 'rate', status: 'normal' },
            { name: 'active_streams', value: 8, unit: 'count', status: 'normal' },
            { name: 'buffer_size', value: 45, unit: '%', status: 'normal' },
            { name: 'processing_lag', value: 2.3, unit: 'ms', status: 'normal' }
          ],
          lastCheck: new Date().toISOString(),
          dependencies: ['messagebus']
        },
        {
          id: 'risk-engine',
          name: 'Risk Management',
          type: 'service',
          status: messageBus.messagesReceived > 50 ? 'healthy' : 'warning',
          metrics: [
            { name: 'active_limits', value: 12, unit: 'count', status: 'normal' },
            { name: 'breach_count', value: 2, unit: 'count', status: 'warning', threshold: 5 },
            { name: 'check_frequency', value: 100, unit: 'Hz', status: 'normal' },
            { name: 'response_time', value: 15, unit: 'ms', status: 'normal' }
          ],
          lastCheck: new Date().toISOString(),
          dependencies: ['analytics-engine']
        }
      ],
      alerts: [
        ...(messageBus.connectionStatus !== 'connected' ? [{
          id: 'alert-1',
          type: 'system' as const,
          severity: 'warning' as const,
          message: 'MessageBus WebSocket connection unstable - intermittent disconnections detected',
          source: 'messagebus',
          timestamp: new Date().toISOString(),
          acknowledged: false
        }] : []),
        ...(engineWs.connectionAttempts > 3 ? [{
          id: 'alert-2',
          type: 'system' as const,
          severity: 'critical' as const,
          message: `Engine WebSocket failed ${engineWs.connectionAttempts} reconnection attempts`,
          source: 'engine-websocket',
          timestamp: new Date(Date.now() - 600000).toISOString(), // 10 minutes ago
          acknowledged: false
        }] : []),
        {
          id: 'alert-3',
          type: 'performance' as const,
          severity: 'info' as const,
          message: 'Risk engine detected 2 limit breaches - within acceptable thresholds',
          source: 'risk-engine',
          timestamp: new Date(Date.now() - 300000).toISOString(), // 5 minutes ago
          acknowledged: true,
          resolvedAt: new Date().toISOString()
        },
        {
          id: 'alert-4',
          type: 'deployment' as const,
          severity: 'info' as const,
          message: 'Strategy deployment pipeline completed successfully',
          source: 'deployment-system',
          timestamp: new Date(Date.now() - 1800000).toISOString(), // 30 minutes ago
          acknowledged: true
        }
      ]
    };

    setHealthMetrics(mockHealthMetrics);
    setLoading(false);
  }, [messageBus.connectionStatus, messageBus.messagesReceived, engineWs.isConnected, engineWs.connectionAttempts]);

  // Auto-refresh
  useEffect(() => {
    if (!autoRefresh) return;

    const interval = setInterval(() => {
      // Trigger refresh - in real implementation, refetch from API
      setHealthMetrics(prev => prev ? {
        ...prev,
        lastUpdate: new Date().toISOString()
      } : null);
    }, refreshInterval);

    return () => clearInterval(interval);
  }, [autoRefresh, refreshInterval]);

  if (loading || !healthMetrics) {
    return <Card loading={loading} title="System Health" style={{ height: 400 }} />;
  }

  // Get component icon
  const getComponentIcon = (type: ComponentHealth['type']) => {
    switch (type) {
      case 'websocket': return <WifiOutlined />;
      case 'api': return <ApiOutlined />;
      case 'database': return <DatabaseOutlined />;
      case 'engine': return <ThunderboltOutlined />;
      case 'service': return <SettingOutlined />;
      default: return <MonitorOutlined />;
    }
  };

  // Get status color and icon
  const getStatusDisplay = (status: ComponentHealth['status']) => {
    switch (status) {
      case 'healthy':
        return { color: 'success', icon: <CheckCircleOutlined />, text: 'Healthy' };
      case 'warning':
        return { color: 'warning', icon: <ExclamationCircleOutlined />, text: 'Warning' };
      case 'critical':
        return { color: 'error', icon: <CloseCircleOutlined />, text: 'Critical' };
      case 'offline':
        return { color: 'default', icon: <MonitorOutlined />, text: 'Offline' };
      default:
        return { color: 'default', icon: <MonitorOutlined />, text: 'Unknown' };
    }
  };

  // Filter components
  const filteredComponents = selectedComponent === 'all' 
    ? healthMetrics.components 
    : healthMetrics.components.filter(c => c.id === selectedComponent);

  // Filter alerts
  const filteredAlerts = healthMetrics.alerts.filter(alert => {
    switch (alertFilter) {
      case 'unacknowledged': return !alert.acknowledged;
      case 'critical': return alert.severity === 'critical';
      default: return true;
    }
  });

  // Component table columns
  const componentColumns = [
    {
      title: 'Component',
      key: 'component',
      render: (record: ComponentHealth) => (
        <Space>
          {getComponentIcon(record.type)}
          <div>
            <Text strong>{record.name}</Text>
            <br />
            <Text type="secondary" style={{ fontSize: '12px' }}>
              {record.type.charAt(0).toUpperCase() + record.type.slice(1)}
            </Text>
          </div>
        </Space>
      )
    },
    {
      title: 'Status',
      dataIndex: 'status',
      key: 'status',
      render: (status: ComponentHealth['status']) => {
        const display = getStatusDisplay(status);
        return (
          <Badge
            status={display.color}
            text={display.text}
          />
        );
      }
    },
    {
      title: 'Key Metrics',
      key: 'metrics',
      render: (record: ComponentHealth) => (
        <Space direction="vertical" size="small">
          {record.metrics.slice(0, 2).map(metric => (
            <div key={metric.name} style={{ fontSize: '12px' }}>
              <Text type="secondary">{metric.name}: </Text>
              <Text 
                type={metric.status === 'critical' ? 'danger' : 
                     metric.status === 'warning' ? 'warning' : 'success'}
              >
                {metric.value} {metric.unit}
              </Text>
            </div>
          ))}
        </Space>
      )
    },
    {
      title: 'Last Check',
      dataIndex: 'lastCheck',
      key: 'lastCheck',
      render: (timestamp: string) => (
        <Text type="secondary" style={{ fontSize: '12px' }}>
          {new Date(timestamp).toLocaleTimeString()}
        </Text>
      )
    },
    {
      title: 'Dependencies',
      dataIndex: 'dependencies',
      key: 'dependencies',
      render: (deps: string[]) => (
        <Space>
          {deps.slice(0, 2).map(dep => (
            <Tag key={dep} size="small">{dep}</Tag>
          ))}
          {deps.length > 2 && <Tag size="small">+{deps.length - 2}</Tag>}
        </Space>
      )
    }
  ];

  if (compactMode) {
    const healthyCount = healthMetrics.components.filter(c => c.status === 'healthy').length;
    const warningCount = healthMetrics.components.filter(c => c.status === 'warning').length;
    const criticalCount = healthMetrics.components.filter(c => c.status === 'critical').length;

    return (
      <Card size="small" title="System Health">
        <Row gutter={[8, 8]}>
          <Col span={8}>
            <Statistic
              title="Healthy"
              value={healthyCount}
              valueStyle={{ color: '#52c41a', fontSize: '16px' }}
              prefix={<CheckCircleOutlined />}
            />
          </Col>
          <Col span={8}>
            <Statistic
              title="Issues"
              value={warningCount + criticalCount}
              valueStyle={{ 
                color: criticalCount > 0 ? '#ff4d4f' : warningCount > 0 ? '#faad14' : '#52c41a',
                fontSize: '16px'
              }}
              prefix={criticalCount > 0 ? <CloseCircleOutlined /> : <ExclamationCircleOutlined />}
            />
          </Col>
          <Col span={8}>
            <Statistic
              title="Uptime"
              value={healthMetrics.uptime}
              suffix="%"
              valueStyle={{ fontSize: '16px' }}
              prefix={<LineChartOutlined />}
            />
          </Col>
        </Row>
      </Card>
    );
  }

  return (
    <div style={{ width: '100%' }}>
      {/* Overall Health Status */}
      <Alert
        message={
          <Space>
            {getStatusDisplay(healthMetrics.overall).icon}
            <Text strong>
              System Status: {getStatusDisplay(healthMetrics.overall).text}
            </Text>
            <Text type="secondary">
              ({healthMetrics.components.filter(c => c.status === 'healthy').length}/{healthMetrics.components.length} components healthy)
            </Text>
          </Space>
        }
        type={healthMetrics.overall === 'healthy' ? 'success' : 
             healthMetrics.overall === 'warning' ? 'warning' : 'error'}
        showIcon
        style={{ marginBottom: 16 }}
        action={
          <Space>
            <Text type="secondary" style={{ fontSize: '12px' }}>
              Last updated: {new Date(healthMetrics.lastUpdate).toLocaleTimeString()}
            </Text>
            <Button size="small" icon={<ReloadOutlined />}>
              Refresh
            </Button>
          </Space>
        }
      />

      {/* Critical Alerts */}
      {showAlerts && filteredAlerts.filter(a => a.severity === 'critical' && !a.acknowledged).length > 0 && (
        <Alert
          message="Critical System Alerts"
          description={
            <div>
              {filteredAlerts
                .filter(a => a.severity === 'critical' && !a.acknowledged)
                .map(alert => (
                  <div key={alert.id} style={{ marginBottom: '8px' }}>
                    <Text strong>{alert.message}</Text>
                    <br />
                    <Text type="secondary" style={{ fontSize: '12px' }}>
                      Source: {alert.source} • {new Date(alert.timestamp).toLocaleString()}
                    </Text>
                  </div>
                ))}
            </div>
          }
          type="error"
          showIcon
          closable
          style={{ marginBottom: 16 }}
        />
      )}

      <Tabs defaultActiveKey="overview" size="small">
        <TabPane tab={<Space><MonitorOutlined />Overview</Space>} key="overview">
          <Row gutter={[16, 16]}>
            {/* System Metrics */}
            <Col xs={24} lg={8}>
              <Card title="System Metrics" size="small">
                <Space direction="vertical" style={{ width: '100%' }}>
                  <div>
                    <Text type="secondary">Overall Health</Text>
                    <Progress
                      percent={Math.round(
                        (healthMetrics.components.filter(c => c.status === 'healthy').length / 
                         healthMetrics.components.length) * 100
                      )}
                      status={healthMetrics.overall === 'healthy' ? 'success' : 'exception'}
                      size="small"
                    />
                  </div>
                  <div>
                    <Text type="secondary">System Uptime</Text>
                    <Progress
                      percent={healthMetrics.uptime}
                      status="active"
                      size="small"
                    />
                  </div>
                  <Divider style={{ margin: '12px 0' }} />
                  <Row gutter={16}>
                    <Col span={8}>
                      <Statistic
                        title="Healthy"
                        value={healthMetrics.components.filter(c => c.status === 'healthy').length}
                        valueStyle={{ fontSize: '14px', color: '#52c41a' }}
                      />
                    </Col>
                    <Col span={8}>
                      <Statistic
                        title="Warning"
                        value={healthMetrics.components.filter(c => c.status === 'warning').length}
                        valueStyle={{ fontSize: '14px', color: '#faad14' }}
                      />
                    </Col>
                    <Col span={8}>
                      <Statistic
                        title="Critical"
                        value={healthMetrics.components.filter(c => c.status === 'critical').length}
                        valueStyle={{ fontSize: '14px', color: '#ff4d4f' }}
                      />
                    </Col>
                  </Row>
                </Space>
              </Card>
            </Col>

            {/* Recent Activity */}
            <Col xs={24} lg={16}>
              <Card title="System Timeline" size="small">
                <Timeline
                  size="small"
                  items={[
                    {
                      color: 'green',
                      children: (
                        <div>
                          <Text strong>System Health Check Complete</Text>
                          <br />
                          <Text type="secondary" style={{ fontSize: '11px' }}>
                            {new Date().toLocaleTimeString()} - All components checked
                          </Text>
                        </div>
                      )
                    },
                    ...(healthMetrics.alerts.slice(0, 3).map(alert => ({
                      color: alert.severity === 'critical' ? 'red' : 
                            alert.severity === 'warning' ? 'orange' : 'blue',
                      children: (
                        <div key={alert.id}>
                          <Text strong>{alert.message}</Text>
                          <br />
                          <Text type="secondary" style={{ fontSize: '11px' }}>
                            {new Date(alert.timestamp).toLocaleTimeString()} - {alert.source}
                            {alert.acknowledged && ' (Acknowledged)'}
                          </Text>
                        </div>
                      )
                    })))
                  ]}
                />
              </Card>
            </Col>
          </Row>
        </TabPane>

        <TabPane tab={<Space><SettingOutlined />Components</Space>} key="components">
          <div style={{ marginBottom: 16 }}>
            <Space>
              <Text>Filter by component:</Text>
              <Select
                value={selectedComponent}
                onChange={setSelectedComponent}
                style={{ width: 200 }}
                size="small"
              >
                <Option value="all">All Components</Option>
                {healthMetrics.components.map(component => (
                  <Option key={component.id} value={component.id}>
                    {component.name}
                  </Option>
                ))}
              </Select>
            </Space>
          </div>
          
          <Table
            columns={componentColumns}
            dataSource={filteredComponents}
            rowKey="id"
            size="small"
            pagination={false}
          />
        </TabPane>

        {showAlerts && (
          <TabPane 
            tab={
              <Space>
                <AlertOutlined />
                Alerts
                {filteredAlerts.filter(a => !a.acknowledged).length > 0 && (
                  <Badge count={filteredAlerts.filter(a => !a.acknowledged).length} />
                )}
              </Space>
            } 
            key="alerts"
          >
            <div style={{ marginBottom: 16 }}>
              <Space>
                <Text>Show:</Text>
                <Select
                  value={alertFilter}
                  onChange={setAlertFilter}
                  style={{ width: 150 }}
                  size="small"
                >
                  <Option value="all">All Alerts</Option>
                  <Option value="unacknowledged">Unacknowledged</Option>
                  <Option value="critical">Critical Only</Option>
                </Select>
              </Space>
            </div>

            <List
              size="small"
              dataSource={filteredAlerts}
              renderItem={(alert) => (
                <List.Item
                  actions={[
                    !alert.acknowledged && (
                      <Button size="small" type="link">
                        Acknowledge
                      </Button>
                    ),
                    alert.severity === 'critical' && (
                      <Button size="small" type="link" danger>
                        Escalate
                      </Button>
                    )
                  ].filter(Boolean)}
                >
                  <List.Item.Meta
                    avatar={
                      <div style={{
                        color: alert.severity === 'critical' ? '#ff4d4f' :
                               alert.severity === 'warning' ? '#faad14' : '#1890ff'
                      }}>
                        {alert.severity === 'critical' ? <CloseCircleOutlined /> :
                         alert.severity === 'warning' ? <ExclamationCircleOutlined /> : 
                         <InfoCircleOutlined />}
                      </div>
                    }
                    title={
                      <Space>
                        <Text strong>{alert.message}</Text>
                        <Tag 
                          color={
                            alert.severity === 'critical' ? 'error' :
                            alert.severity === 'warning' ? 'warning' : 'default'
                          }
                          size="small"
                        >
                          {alert.severity}
                        </Tag>
                        {alert.acknowledged && <Tag color="success" size="small">Acknowledged</Tag>}
                        {alert.resolvedAt && <Tag color="success" size="small">Resolved</Tag>}
                      </Space>
                    }
                    description={
                      <Text type="secondary" style={{ fontSize: '12px' }}>
                        {alert.source} • {new Date(alert.timestamp).toLocaleString()}
                        {alert.resolvedAt && ` • Resolved: ${new Date(alert.resolvedAt).toLocaleString()}`}
                      </Text>
                    }
                  />
                </List.Item>
              )}
            />
          </TabPane>
        )}
      </Tabs>
    </div>
  );
};

export default SystemHealthOverview;