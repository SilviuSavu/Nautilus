/**
 * Sprint 3: Component Status Matrix
 * Visual grid display of all Sprint 3 component statuses
 * Real-time health monitoring with dependency visualization
 */

import React, { useState, useEffect } from 'react';
import {
  Card,
  Row,
  Col,
  Typography,
  Space,
  Button,
  Badge,
  Tooltip,
  Progress,
  Tag,
  Avatar,
  Popover,
  Modal,
  Descriptions,
  Timeline,
  Alert,
  Spin,
  Switch,
  notification
} from 'antd';
import {
  CheckCircleOutlined,
  ExclamationCircleOutlined,
  CloseCircleOutlined,
  WarningOutlined,
  ReloadOutlined,
  ApiOutlined,
  DatabaseOutlined,
  ThunderboltOutlined,
  DashboardOutlined,
  ClusterOutlined,
  EyeOutlined,
  SafetyCertificateOutlined,
  ToolOutlined,
  HeartOutlined,
  BugOutlined
} from '@ant-design/icons';

const { Title, Text } = Typography;

interface ComponentStatus {
  component_id: string;
  name: string;
  type: 'service' | 'infrastructure' | 'database' | 'external';
  status: 'online' | 'degraded' | 'offline' | 'maintenance' | 'starting' | 'stopping';
  health_score: number;
  response_time_ms: number;
  uptime_percentage: number;
  error_rate_percentage: number;
  last_check: string;
  version?: string;
  dependencies: string[];
  dependent_services: string[];
  metrics: {
    cpu_usage?: number;
    memory_usage?: number;
    connections?: number;
    throughput?: number;
    errors_per_minute?: number;
  };
  alerts: ComponentAlert[];
  recent_events: ComponentEvent[];
}

interface ComponentAlert {
  id: string;
  severity: 'critical' | 'high' | 'medium' | 'low';
  message: string;
  timestamp: string;
  acknowledged: boolean;
}

interface ComponentEvent {
  id: string;
  event_type: 'status_change' | 'deployment' | 'restart' | 'error' | 'recovery';
  message: string;
  timestamp: string;
  metadata?: Record<string, any>;
}

interface ComponentStatusMatrixProps {
  className?: string;
  compact?: boolean;
  showDependencies?: boolean;
}

export const ComponentStatusMatrix: React.FC<ComponentStatusMatrixProps> = ({
  className,
  compact = false,
  showDependencies = true
}) => {
  const [components, setComponents] = useState<ComponentStatus[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [selectedComponent, setSelectedComponent] = useState<string | null>(null);
  const [detailModalVisible, setDetailModalVisible] = useState(false);
  const [autoRefresh, setAutoRefresh] = useState(true);
  const [lastUpdate, setLastUpdate] = useState<Date | null>(null);

  // Sprint 3 Component Configuration
  const componentConfig = {
    websocket_infrastructure: {
      name: 'WebSocket Infrastructure',
      icon: <ApiOutlined />,
      color: '#1890ff',
      description: 'Real-time messaging, 1000+ connections',
      category: 'Infrastructure'
    },
    analytics_engine: {
      name: 'Analytics Engine',
      icon: <DashboardOutlined />,
      color: '#722ed1',
      description: 'P&L calculations, performance metrics',
      category: 'Analytics'
    },
    risk_management: {
      name: 'Risk Management',
      icon: <SafetyCertificateOutlined />,
      color: '#f5222d',
      description: 'Dynamic limits, ML breach detection',
      category: 'Risk'
    },
    strategy_deployment: {
      name: 'Strategy Deployment',
      icon: <ThunderboltOutlined />,
      color: '#fa541c',
      description: 'CI/CD pipeline, automated testing',
      category: 'Deployment'
    },
    database_layer: {
      name: 'TimescaleDB',
      icon: <DatabaseOutlined />,
      color: '#13c2c2',
      description: 'Time-series data, hypertables',
      category: 'Database'
    },
    nautilus_engine: {
      name: 'Trading Engine',
      icon: <ClusterOutlined />,
      color: '#52c41a',
      description: 'NautilusTrader containers',
      category: 'Trading'
    },
    monitoring_stack: {
      name: 'Monitoring Stack',
      icon: <EyeOutlined />,
      color: '#faad14',
      description: 'Prometheus + Grafana',
      category: 'Observability'
    }
  };

  const fetchComponentStatuses = async () => {
    try {
      const response = await fetch(`${import.meta.env.VITE_API_BASE_URL}/api/v1/system/components/status`);
      if (!response.ok) {
        throw new Error(`Failed to fetch component statuses: ${response.statusText}`);
      }
      const data: ComponentStatus[] = await response.json();
      setComponents(data);
      setError(null);
      setLastUpdate(new Date());
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to fetch component statuses');
      console.error('Component status fetch error:', err);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchComponentStatuses();
  }, []);

  useEffect(() => {
    if (autoRefresh) {
      const interval = setInterval(fetchComponentStatuses, 5000); // Refresh every 5 seconds
      return () => clearInterval(interval);
    }
  }, [autoRefresh]);

  const handleRefresh = async () => {
    setLoading(true);
    await fetchComponentStatuses();
    notification.success({
      message: 'Component Status Updated',
      description: 'All component statuses have been refreshed',
      duration: 2
    });
  };

  const getStatusIcon = (status: string, size: number = 16) => {
    const style = { fontSize: size, color: getStatusColor(status) };
    switch (status) {
      case 'online':
        return <CheckCircleOutlined style={style} />;
      case 'degraded':
        return <ExclamationCircleOutlined style={style} />;
      case 'offline':
        return <CloseCircleOutlined style={style} />;
      case 'maintenance':
        return <ToolOutlined style={style} />;
      case 'starting':
        return <HeartOutlined style={style} />;
      case 'stopping':
        return <WarningOutlined style={style} />;
      default:
        return <WarningOutlined style={style} />;
    }
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'online': return '#52c41a';
      case 'degraded': return '#faad14';
      case 'offline': return '#ff4d4f';
      case 'maintenance': return '#1890ff';
      case 'starting': return '#722ed1';
      case 'stopping': return '#fa541c';
      default: return '#d9d9d9';
    }
  };

  const getHealthScoreColor = (score: number) => {
    if (score >= 90) return '#52c41a';
    if (score >= 70) return '#faad14';
    if (score >= 50) return '#fa8c16';
    return '#ff4d4f';
  };

  const handleComponentClick = (componentId: string) => {
    setSelectedComponent(componentId);
    setDetailModalVisible(true);
  };

  const renderComponentCard = (component: ComponentStatus) => {
    const config = componentConfig[component.component_id as keyof typeof componentConfig];
    if (!config) return null;

    const criticalAlerts = component.alerts.filter(alert => alert.severity === 'critical').length;
    const totalAlerts = component.alerts.length;

    return (
      <Card
        key={component.component_id}
        size={compact ? 'small' : 'default'}
        hoverable
        onClick={() => handleComponentClick(component.component_id)}
        style={{
          border: `2px solid ${getStatusColor(component.status)}`,
          cursor: 'pointer',
          height: compact ? '120px' : '160px'
        }}
      >
        <div style={{ textAlign: 'center' }}>
          {/* Component Icon and Status */}
          <div style={{ marginBottom: compact ? 4 : 8 }}>
            <Badge
              count={totalAlerts}
              offset={[8, -8]}
              style={{ backgroundColor: criticalAlerts > 0 ? '#ff4d4f' : '#faad14' }}
            >
              <Avatar
                size={compact ? 32 : 48}
                style={{ backgroundColor: config.color }}
                icon={config.icon}
              />
            </Badge>
          </div>

          {/* Component Name */}
          <div style={{ marginBottom: compact ? 2 : 4 }}>
            <Text strong style={{ fontSize: compact ? '12px' : '14px' }}>
              {compact ? config.name.split(' ')[0] : config.name}
            </Text>
          </div>

          {/* Status Badge */}
          <div style={{ marginBottom: compact ? 2 : 4 }}>
            <Badge
              status={component.status === 'online' ? 'success' : component.status === 'degraded' ? 'warning' : 'error'}
              text={
                <Text style={{ fontSize: compact ? '10px' : '12px' }}>
                  {component.status.toUpperCase()}
                </Text>
              }
            />
          </div>

          {/* Health Score */}
          <div style={{ marginBottom: compact ? 2 : 4 }}>
            <Progress
              percent={component.health_score}
              size="small"
              strokeColor={getHealthScoreColor(component.health_score)}
              format={() => compact ? `${component.health_score}` : `${component.health_score}/100`}
              style={{ width: '100%' }}
            />
          </div>

          {/* Quick Metrics */}
          {!compact && (
            <div style={{ display: 'flex', justifyContent: 'space-around', fontSize: '10px' }}>
              <Tooltip title="Response Time">
                <Text type="secondary">{component.response_time_ms}ms</Text>
              </Tooltip>
              <Tooltip title="Uptime">
                <Text type="secondary">{component.uptime_percentage.toFixed(0)}%</Text>
              </Tooltip>
              {component.error_rate_percentage > 0 && (
                <Tooltip title="Error Rate">
                  <Text type="danger">{component.error_rate_percentage.toFixed(1)}%</Text>
                </Tooltip>
              )}
            </div>
          )}
        </div>
      </Card>
    );
  };

  const renderComponentDetails = () => {
    const component = components.find(c => c.component_id === selectedComponent);
    if (!component) return null;

    const config = componentConfig[component.component_id as keyof typeof componentConfig];

    return (
      <Modal
        title={
          <Space>
            {config?.icon}
            <span>{config?.name || component.name}</span>
            {getStatusIcon(component.status)}
          </Space>
        }
        open={detailModalVisible}
        onCancel={() => setDetailModalVisible(false)}
        footer={null}
        width={800}
      >
        <Row gutter={[16, 16]}>
          <Col span={24}>
            <Descriptions bordered size="small" column={2}>
              <Descriptions.Item label="Status" span={1}>
                <Space>
                  {getStatusIcon(component.status)}
                  <Tag color={getStatusColor(component.status)}>
                    {component.status.toUpperCase()}
                  </Tag>
                </Space>
              </Descriptions.Item>
              <Descriptions.Item label="Health Score" span={1}>
                <Progress
                  percent={component.health_score}
                  size="small"
                  strokeColor={getHealthScoreColor(component.health_score)}
                />
              </Descriptions.Item>
              <Descriptions.Item label="Response Time" span={1}>
                <Text style={{ color: component.response_time_ms > 1000 ? '#ff4d4f' : '#52c41a' }}>
                  {component.response_time_ms}ms
                </Text>
              </Descriptions.Item>
              <Descriptions.Item label="Uptime" span={1}>
                {component.uptime_percentage.toFixed(2)}%
              </Descriptions.Item>
              <Descriptions.Item label="Error Rate" span={1}>
                <Text style={{ color: component.error_rate_percentage > 1 ? '#ff4d4f' : '#52c41a' }}>
                  {component.error_rate_percentage.toFixed(2)}%
                </Text>
              </Descriptions.Item>
              <Descriptions.Item label="Last Check" span={1}>
                {new Date(component.last_check).toLocaleString()}
              </Descriptions.Item>
            </Descriptions>
          </Col>

          {/* Metrics */}
          {Object.keys(component.metrics).length > 0 && (
            <Col span={12}>
              <Card title="Performance Metrics" size="small">
                {Object.entries(component.metrics).map(([key, value]) => (
                  <div key={key} style={{ marginBottom: 8 }}>
                    <Text type="secondary">{key.replace('_', ' ').toUpperCase()}: </Text>
                    <Text strong>{value}</Text>
                  </div>
                ))}
              </Card>
            </Col>
          )}

          {/* Dependencies */}
          {showDependencies && (
            <Col span={12}>
              <Card title="Dependencies" size="small">
                <div>
                  <Text strong>Depends On:</Text>
                  <div style={{ marginTop: 4 }}>
                    {component.dependencies.map(dep => (
                      <Tag key={dep} size="small" style={{ margin: 2 }}>
                        {dep}
                      </Tag>
                    ))}
                  </div>
                </div>
                <div style={{ marginTop: 12 }}>
                  <Text strong>Provides To:</Text>
                  <div style={{ marginTop: 4 }}>
                    {component.dependent_services.map(service => (
                      <Tag key={service} size="small" color="blue" style={{ margin: 2 }}>
                        {service}
                      </Tag>
                    ))}
                  </div>
                </div>
              </Card>
            </Col>
          )}

          {/* Active Alerts */}
          {component.alerts.length > 0 && (
            <Col span={24}>
              <Card title="Active Alerts" size="small">
                {component.alerts.map(alert => (
                  <Alert
                    key={alert.id}
                    message={alert.message}
                    type={alert.severity === 'critical' ? 'error' : alert.severity === 'high' ? 'warning' : 'info'}
                    style={{ marginBottom: 8 }}
                    action={
                      <Tag color={alert.severity === 'critical' ? 'red' : alert.severity === 'high' ? 'orange' : 'blue'}>
                        {alert.severity.toUpperCase()}
                      </Tag>
                    }
                  />
                ))}
              </Card>
            </Col>
          )}

          {/* Recent Events */}
          {component.recent_events.length > 0 && (
            <Col span={24}>
              <Card title="Recent Events" size="small">
                <Timeline size="small">
                  {component.recent_events.slice(0, 5).map(event => (
                    <Timeline.Item
                      key={event.id}
                      color={
                        event.event_type === 'error' ? 'red' :
                        event.event_type === 'recovery' ? 'green' :
                        event.event_type === 'deployment' ? 'blue' : 'gray'
                      }
                    >
                      <div>
                        <Text strong>{event.event_type.replace('_', ' ').toUpperCase()}</Text>
                        <div>
                          <Text>{event.message}</Text>
                        </div>
                        <div>
                          <Text type="secondary" style={{ fontSize: '11px' }}>
                            {new Date(event.timestamp).toLocaleString()}
                          </Text>
                        </div>
                      </div>
                    </Timeline.Item>
                  ))}
                </Timeline>
              </Card>
            </Col>
          )}
        </Row>
      </Modal>
    );
  };

  if (loading && components.length === 0) {
    return (
      <div style={{ textAlign: 'center', padding: compact ? 20 : 40 }}>
        <Spin size={compact ? 'default' : 'large'} />
        <div style={{ marginTop: 8 }}>
          <Text type="secondary">Loading component statuses...</Text>
        </div>
      </div>
    );
  }

  const onlineComponents = components.filter(c => c.status === 'online').length;
  const degradedComponents = components.filter(c => c.status === 'degraded').length;
  const offlineComponents = components.filter(c => c.status === 'offline').length;
  const totalAlerts = components.reduce((sum, c) => sum + c.alerts.length, 0);

  return (
    <div className={`component-status-matrix ${className || ''}`}>
      {/* Header */}
      {!compact && (
        <div style={{ marginBottom: 16 }}>
          <Row justify="space-between" align="middle">
            <Col>
              <Space>
                <Title level={4} style={{ margin: 0 }}>
                  Component Status Matrix
                </Title>
                <Badge count={onlineComponents} style={{ backgroundColor: '#52c41a' }} />
                {degradedComponents > 0 && (
                  <Badge count={degradedComponents} style={{ backgroundColor: '#faad14' }} />
                )}
                {offlineComponents > 0 && (
                  <Badge count={offlineComponents} style={{ backgroundColor: '#ff4d4f' }} />
                )}
              </Space>
            </Col>
            <Col>
              <Space>
                <Switch
                  checked={autoRefresh}
                  onChange={setAutoRefresh}
                  size="small"
                  checkedChildren="Auto"
                  unCheckedChildren="Manual"
                />
                <Button
                  size="small"
                  icon={<ReloadOutlined />}
                  onClick={handleRefresh}
                  loading={loading}
                />
              </Space>
            </Col>
          </Row>
        </div>
      )}

      {/* Error Display */}
      {error && (
        <Alert
          message="Component Status Error"
          description={error}
          type="error"
          style={{ marginBottom: 16 }}
          action={
            <Button size="small" onClick={handleRefresh}>
              Retry
            </Button>
          }
        />
      )}

      {/* Status Summary */}
      {!compact && totalAlerts > 0 && (
        <Alert
          message={`${totalAlerts} Active Alerts Across Components`}
          description={`${components.filter(c => c.alerts.some(a => a.severity === 'critical')).length} components have critical alerts`}
          type="warning"
          style={{ marginBottom: 16 }}
          showIcon
        />
      )}

      {/* Component Grid */}
      <Row gutter={[compact ? 8 : 16, compact ? 8 : 16]}>
        {components.map(component => (
          <Col 
            key={component.component_id}
            xs={compact ? 8 : 12} 
            sm={compact ? 6 : 8} 
            md={compact ? 4 : 6} 
            lg={compact ? 4 : 6}
            xl={compact ? 3 : 4}
          >
            {renderComponentCard(component)}
          </Col>
        ))}
      </Row>

      {/* Last Update */}
      {lastUpdate && !compact && (
        <div style={{ marginTop: 16, textAlign: 'center' }}>
          <Text type="secondary" style={{ fontSize: '11px' }}>
            Last updated: {lastUpdate.toLocaleTimeString()} | 
            Auto-refresh: {autoRefresh ? '5s' : 'Off'}
          </Text>
        </div>
      )}

      {/* Component Details Modal */}
      {renderComponentDetails()}
    </div>
  );
};

export default ComponentStatusMatrix;