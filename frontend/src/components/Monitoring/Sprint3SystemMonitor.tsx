/**
 * Sprint 3: Advanced System Monitoring Dashboard
 * Enterprise-grade comprehensive system monitoring with real-time metrics
 * Integration with Prometheus, Grafana, and all Sprint 3 components
 */

import React, { useState, useEffect, memo, useCallback, useMemo } from 'react';
import {
  Card,
  Row,
  Col,
  Typography,
  Space,
  Button,
  Tabs,
  Spin,
  Alert,
  Select,
  Statistic,
  Progress,
  Badge,
  Tag,
  notification,
  Tooltip,
  Timeline,
  Divider,
  Switch,
  List,
  Avatar
} from 'antd';
import {
  MonitorOutlined,
  AlertOutlined,
  DashboardOutlined,
  ThunderboltOutlined,
  CloudServerOutlined,
  DatabaseOutlined,
  ReloadOutlined,
  SettingOutlined,
  ApiOutlined,
  SafetyCertificateOutlined,
  PlayCircleOutlined,
  PauseCircleOutlined,
  EyeOutlined,
  CheckCircleOutlined,
  ExclamationCircleOutlined,
  WarningOutlined,
  CloseCircleOutlined
} from '@ant-design/icons';

import { SystemHealthAggregator } from './SystemHealthAggregator';
import { PrometheusMetricsDashboard } from './PrometheusMetricsDashboard';
import { AlertRulesManager } from './AlertRulesManager';
import { ComponentStatusMatrix } from './ComponentStatusMatrix';
import { SystemResourceMonitor } from './SystemResourceMonitor';
import { PerformanceTrendAnalyzer } from './PerformanceTrendAnalyzer';

const { Title, Text } = Typography;
const { TabPane } = Tabs;
const { Option } = Select;

interface Sprint3SystemHealth {
  overall_status: 'healthy' | 'degraded' | 'critical' | 'unknown';
  overall_score: number;
  last_check: string;
  components: {
    websocket_infrastructure: ComponentHealth;
    analytics_engine: ComponentHealth;
    risk_management: ComponentHealth;
    strategy_deployment: ComponentHealth;
    database_layer: ComponentHealth;
    nautilus_engine: ComponentHealth;
    monitoring_stack: ComponentHealth;
  };
  alerts: SystemAlert[];
  metrics_summary: {
    websocket_connections: number;
    messages_per_second: number;
    risk_checks_per_second: number;
    active_strategies: number;
    database_performance: number;
    system_load: number;
  };
}

interface ComponentHealth {
  name: string;
  status: 'online' | 'degraded' | 'offline' | 'maintenance';
  health_score: number;
  response_time_ms: number;
  uptime_percentage: number;
  last_check: string;
  details: {
    [key: string]: any;
  };
  sub_components?: ComponentHealth[];
}

interface SystemAlert {
  id: string;
  component: string;
  severity: 'critical' | 'high' | 'medium' | 'low';
  title: string;
  description: string;
  timestamp: string;
  acknowledged: boolean;
  auto_resolve: boolean;
}

interface Sprint3SystemMonitorProps {
  className?: string;
  autoRefresh?: boolean;
  refreshInterval?: number;
  showAlerts?: boolean;
  compactMode?: boolean;
  showAdvancedMetrics?: boolean;
}

export const Sprint3SystemMonitor: React.FC<Sprint3SystemMonitorProps> = memo(({
  className,
  autoRefresh: propAutoRefresh = true,
  refreshInterval: propRefreshInterval = 5000,
  showAlerts = true,
  compactMode = false,
  showAdvancedMetrics = false
}) => {
  const [activeTab, setActiveTab] = useState('overview');
  const [systemHealth, setSystemHealth] = useState<Sprint3SystemHealth | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [autoRefresh, setAutoRefresh] = useState(propAutoRefresh);
  const [refreshInterval, setRefreshInterval] = useState(propRefreshInterval);
  const [lastUpdate, setLastUpdate] = useState<Date | null>(null);

  const fetchSystemHealth = async () => {
    try {
      const response = await fetch(`${import.meta.env.VITE_API_BASE_URL}/api/v1/system/health`);
      if (!response.ok) {
        throw new Error(`System health fetch failed: ${response.statusText}`);
      }
      const data: Sprint3SystemHealth = await response.json();
      setSystemHealth(data);
      setError(null);
      setLastUpdate(new Date());
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Unknown error occurred');
      console.error('System health fetch error:', err);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchSystemHealth();
  }, []);

  useEffect(() => {
    let intervalId: NodeJS.Timeout;
    
    if (autoRefresh && refreshInterval > 0) {
      intervalId = setInterval(fetchSystemHealth, refreshInterval);
    }

    return () => {
      if (intervalId) {
        clearInterval(intervalId);
      }
    };
  }, [autoRefresh, refreshInterval]);

  const handleRefresh = async () => {
    setLoading(true);
    await fetchSystemHealth();
    notification.success({
      message: 'System Health Updated',
      description: 'All Sprint 3 components have been checked',
      duration: 2
    });
  };

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'online':
      case 'healthy':
        return <CheckCircleOutlined style={{ color: '#52c41a' }} />;
      case 'degraded':
        return <ExclamationCircleOutlined style={{ color: '#faad14' }} />;
      case 'offline':
      case 'critical':
        return <CloseCircleOutlined style={{ color: '#ff4d4f' }} />;
      case 'maintenance':
        return <WarningOutlined style={{ color: '#1890ff' }} />;
      default:
        return <ExclamationCircleOutlined style={{ color: '#d9d9d9' }} />;
    }
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'online':
      case 'healthy':
        return '#52c41a';
      case 'degraded':
        return '#faad14';
      case 'offline':
      case 'critical':
        return '#ff4d4f';
      case 'maintenance':
        return '#1890ff';
      default:
        return '#d9d9d9';
    }
  };

  const getSeverityColor = (severity: string) => {
    switch (severity) {
      case 'critical':
        return '#ff4d4f';
      case 'high':
        return '#fa8c16';
      case 'medium':
        return '#faad14';
      case 'low':
        return '#52c41a';
      default:
        return '#d9d9d9';
    }
  };

  const acknowledgeAlert = async (alertId: string) => {
    try {
      const response = await fetch(`${import.meta.env.VITE_API_BASE_URL}/api/v1/system/alerts/${alertId}/acknowledge`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        }
      });

      if (response.ok) {
        notification.success({
          message: 'Alert Acknowledged',
          description: 'Alert has been marked as acknowledged',
          duration: 2
        });
        await fetchSystemHealth();
      }
    } catch (err) {
      notification.error({
        message: 'Failed to Acknowledge Alert',
        description: 'Could not acknowledge the alert',
        duration: 4
      });
    }
  };

  const renderOverviewTab = () => {
    if (!systemHealth) return null;

    const criticalAlerts = systemHealth.alerts.filter(alert => alert.severity === 'critical');
    const highAlerts = systemHealth.alerts.filter(alert => alert.severity === 'high');

    return (
      <div>
        {/* Critical System Alerts */}
        {criticalAlerts.length > 0 && (
          <Alert
            message="Critical System Alerts"
            description={`${criticalAlerts.length} critical alerts require immediate attention`}
            type="error"
            icon={<AlertOutlined />}
            style={{ marginBottom: 16 }}
            showIcon
            action={
              <Button 
                size="small" 
                danger 
                onClick={() => setActiveTab('alerts')}
              >
                View Alerts
              </Button>
            }
          />
        )}

        {/* Overall System Status */}
        <Row gutter={[16, 16]} style={{ marginBottom: 24 }}>
          <Col xs={24} sm={12} md={6}>
            <Card>
              <Statistic
                title="System Health Score"
                value={systemHealth.overall_score}
                suffix="/100"
                precision={1}
                valueStyle={{ color: getStatusColor(systemHealth.overall_status) }}
                prefix={getStatusIcon(systemHealth.overall_status)}
              />
              <Progress
                percent={systemHealth.overall_score}
                size="small"
                status={systemHealth.overall_score > 90 ? 'success' : systemHealth.overall_score > 70 ? 'normal' : 'exception'}
                showInfo={false}
              />
            </Card>
          </Col>

          <Col xs={24} sm={12} md={6}>
            <Card>
              <Statistic
                title="WebSocket Connections"
                value={systemHealth.metrics_summary.websocket_connections}
                valueStyle={{ color: '#1890ff' }}
                prefix={<ApiOutlined />}
              />
              <Text type="secondary" style={{ fontSize: '12px' }}>
                Target: 1000+ concurrent
              </Text>
            </Card>
          </Col>

          <Col xs={24} sm={12} md={6}>
            <Card>
              <Statistic
                title="Messages/Second"
                value={systemHealth.metrics_summary.messages_per_second}
                valueStyle={{ color: '#722ed1' }}
                prefix={<ThunderboltOutlined />}
              />
              <Text type="secondary" style={{ fontSize: '12px' }}>
                Target: 50,000+ msgs/sec
              </Text>
            </Card>
          </Col>

          <Col xs={24} sm={12} md={6}>
            <Card>
              <Statistic
                title="Active Strategies"
                value={systemHealth.metrics_summary.active_strategies}
                valueStyle={{ color: '#13c2c2' }}
                prefix={<DashboardOutlined />}
              />
              <Text type="secondary" style={{ fontSize: '12px' }}>
                Deployed & running
              </Text>
            </Card>
          </Col>
        </Row>

        {/* Sprint 3 Component Status */}
        <Row gutter={[16, 16]} style={{ marginBottom: 24 }}>
          <Col xs={24} lg={16}>
            <Card title="Sprint 3 Component Health Matrix" extra={<ComponentStatusMatrix />}>
              <List
                dataSource={Object.entries(systemHealth.components)}
                renderItem={([key, component]) => (
                  <List.Item
                    key={key}
                    actions={[
                      <Badge
                        status={component.status === 'online' ? 'success' : component.status === 'degraded' ? 'warning' : 'error'}
                        text={component.status.toUpperCase()}
                      />,
                      <Tooltip title="Response Time">
                        <Text type="secondary">{component.response_time_ms}ms</Text>
                      </Tooltip>,
                      <Tooltip title="Uptime">
                        <Text type="secondary">{component.uptime_percentage.toFixed(1)}%</Text>
                      </Tooltip>
                    ]}
                  >
                    <List.Item.Meta
                      avatar={<Avatar icon={getStatusIcon(component.status)} />}
                      title={component.name}
                      description={
                        <div>
                          <Text>Health Score: {component.health_score}/100</Text>
                          {component.sub_components && (
                            <div style={{ marginTop: 4 }}>
                              <Text type="secondary" style={{ fontSize: '12px' }}>
                                Sub-components: {component.sub_components.length}
                              </Text>
                            </div>
                          )}
                        </div>
                      }
                    />
                  </List.Item>
                )}
              />
            </Card>
          </Col>

          <Col xs={24} lg={8}>
            <Card title="Recent System Events">
              <Timeline
                items={systemHealth.alerts
                  .slice(0, 5)
                  .map(alert => ({
                    color: getSeverityColor(alert.severity),
                    children: (
                      <div>
                        <div>
                          <Text strong>{alert.title}</Text>
                          <Tag color={getSeverityColor(alert.severity)} style={{ marginLeft: 8 }}>
                            {alert.severity.toUpperCase()}
                          </Tag>
                        </div>
                        <Text type="secondary" style={{ fontSize: '12px' }}>
                          {alert.component} - {new Date(alert.timestamp).toLocaleTimeString()}
                        </Text>
                      </div>
                    )
                  }))}
              />
            </Card>
          </Col>
        </Row>

        {/* Performance Metrics Summary */}
        <Row gutter={[16, 16]}>
          <Col xs={24} md={12}>
            <Card title="Risk Management Performance" extra={<SafetyCertificateOutlined />}>
              <Row gutter={16}>
                <Col span={12}>
                  <Statistic
                    title="Risk Checks/Second"
                    value={systemHealth.metrics_summary.risk_checks_per_second}
                    valueStyle={{ color: '#f5222d' }}
                  />
                  <Text type="secondary" style={{ fontSize: '12px' }}>
                    Target: 5-second intervals
                  </Text>
                </Col>
                <Col span={12}>
                  <Statistic
                    title="System Load"
                    value={systemHealth.metrics_summary.system_load}
                    suffix="%"
                    valueStyle={{ color: systemHealth.metrics_summary.system_load > 80 ? '#f5222d' : '#52c41a' }}
                  />
                </Col>
              </Row>
            </Card>
          </Col>

          <Col xs={24} md={12}>
            <Card title="Database Performance" extra={<DatabaseOutlined />}>
              <Row gutter={16}>
                <Col span={12}>
                  <Statistic
                    title="DB Performance Score"
                    value={systemHealth.metrics_summary.database_performance}
                    suffix="/100"
                    valueStyle={{ color: '#722ed1' }}
                  />
                  <Text type="secondary" style={{ fontSize: '12px' }}>
                    TimescaleDB optimized
                  </Text>
                </Col>
                <Col span={12}>
                  <div>
                    <Text type="secondary">Query Response</Text>
                    <div>
                      <Text strong>
                        {systemHealth.components.database_layer.response_time_ms}ms
                      </Text>
                    </div>
                  </div>
                </Col>
              </Row>
            </Card>
          </Col>
        </Row>
      </div>
    );
  };

  return (
    <div className={`sprint3-system-monitor ${className || ''}`}>
      <Card>
        {/* Header */}
        <div style={{ marginBottom: 24 }}>
          <Row justify="space-between" align="middle">
            <Col>
              <Title level={2} style={{ margin: 0 }}>
                <MonitorOutlined style={{ marginRight: 8, color: '#1890ff' }} />
                Sprint 3 System Monitor
              </Title>
              <Text type="secondary">
                Enterprise monitoring for advanced trading infrastructure
              </Text>
            </Col>
            <Col>
              <Space>
                <Select
                  value={refreshInterval}
                  onChange={setRefreshInterval}
                  style={{ width: 120 }}
                  size="small"
                >
                  <Option value={1000}>1s</Option>
                  <Option value={5000}>5s</Option>
                  <Option value={10000}>10s</Option>
                  <Option value={30000}>30s</Option>
                </Select>

                <Switch
                  checked={autoRefresh}
                  onChange={setAutoRefresh}
                  checkedChildren={<PlayCircleOutlined />}
                  unCheckedChildren={<PauseCircleOutlined />}
                />

                <Button
                  type="primary"
                  icon={<ReloadOutlined />}
                  onClick={handleRefresh}
                  loading={loading}
                  size="small"
                >
                  Refresh
                </Button>
              </Space>
            </Col>
          </Row>

          {lastUpdate && (
            <div style={{ marginTop: 8 }}>
              <Text type="secondary" style={{ fontSize: 12 }}>
                Last updated: {lastUpdate.toLocaleTimeString()} | 
                Auto-refresh: {autoRefresh ? `${refreshInterval/1000}s` : 'Off'}
              </Text>
            </div>
          )}
        </div>

        {/* Error Display */}
        {error && (
          <Alert
            message="System Health Error"
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

        {/* Main Content */}
        <Tabs activeKey={activeTab} onChange={setActiveTab}>
          <TabPane 
            tab={
              <span>
                <DashboardOutlined />
                Overview
              </span>
            } 
            key="overview"
          >
            {loading ? (
              <div style={{ textAlign: 'center', padding: 60 }}>
                <Spin size="large" />
                <div style={{ marginTop: 16 }}>
                  <Text type="secondary">Loading Sprint 3 system health...</Text>
                </div>
              </div>
            ) : (
              renderOverviewTab()
            )}
          </TabPane>

          <TabPane 
            tab={
              <span>
                <CloudServerOutlined />
                Prometheus
              </span>
            } 
            key="prometheus"
          >
            <PrometheusMetricsDashboard />
          </TabPane>

          <TabPane 
            tab={
              <span>
                <EyeOutlined />
                Grafana
              </span>
            } 
            key="grafana"
          >
            <div style={{ height: '600px' }}>
              <iframe
                src={`${import.meta.env.VITE_GRAFANA_URL || 'http://localhost:3002'}/d/nautilus-overview/nautilus-trading-platform`}
                width="100%"
                height="100%"
                frameBorder="0"
                title="Grafana Dashboard"
              />
            </div>
          </TabPane>

          <TabPane 
            tab={
              <span>
                <AlertOutlined />
                Alerts
                {systemHealth?.alerts.length && systemHealth.alerts.length > 0 && (
                  <Badge 
                    count={systemHealth.alerts.filter(a => !a.acknowledged).length} 
                    offset={[8, -4]} 
                  />
                )}
              </span>
            } 
            key="alerts"
          >
            <AlertRulesManager />
          </TabPane>

          <TabPane 
            tab={
              <span>
                <SafetyCertificateOutlined />
                Health Aggregator
              </span>
            } 
            key="health"
          >
            <SystemHealthAggregator />
          </TabPane>

          <TabPane 
            tab={
              <span>
                <ThunderboltOutlined />
                Performance
              </span>
            } 
            key="performance"
          >
            <PerformanceTrendAnalyzer />
          </TabPane>

          <TabPane 
            tab={
              <span>
                <DatabaseOutlined />
                Resources
              </span>
            } 
            key="resources"
          >
            <SystemResourceMonitor />
          </TabPane>

          <TabPane 
            tab={
              <span>
                <SettingOutlined />
                Configuration
              </span>
            } 
            key="config"
          >
            <div style={{ textAlign: 'center', padding: 60 }}>
              <Text type="secondary">
                System configuration and tuning options will be implemented here
              </Text>
            </div>
          </TabPane>
        </Tabs>
      </Card>
    </div>
  );
});
Sprint3SystemMonitor.displayName = 'Sprint3SystemMonitor';

export default Sprint3SystemMonitor;