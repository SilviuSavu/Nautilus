/**
 * Story 5.2: System Performance Dashboard
 * Main dashboard component for system performance monitoring
 */

import React, { useState } from 'react';
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
  notification
} from 'antd';
import {
  ReloadOutlined,
  DashboardOutlined,
  AlertOutlined,
  MonitorOutlined,
  ApiOutlined,
  PlayCircleOutlined,
  PauseCircleOutlined,
  SettingOutlined
} from '@ant-design/icons';

import { useSystemMonitoring } from '../../hooks/monitoring/useSystemMonitoring';
import { systemMonitoringService } from '../../services/monitoring/SystemMonitoringService';

const { Title, Text } = Typography;
const { TabPane } = Tabs;
const { Option } = Select;

interface SystemPerformanceDashboardProps {
  className?: string;
}

export const SystemPerformanceDashboard: React.FC<SystemPerformanceDashboardProps> = ({
  className
}) => {
  const [activeTab, setActiveTab] = useState('overview');
  const [selectedVenue, setSelectedVenue] = useState<string>('all');
  const [timeframe, setTimeframe] = useState<string>('1h');

  const {
    latencyMetrics,
    systemMetrics,
    connectionQuality,
    activeAlerts,
    performanceTrends,
    loading,
    error,
    lastUpdate,
    realTimeLatency,
    realTimeCpuUsage,
    realTimeMemoryUsage,
    refreshAllMetrics,
    refreshLatencyMetrics,
    refreshSystemMetrics,
    refreshConnectionMetrics,
    refreshAlerts,
    startAutoRefresh,
    stopAutoRefresh,
    isAutoRefreshActive
  } = useSystemMonitoring({
    refreshInterval: 5000, // 5 seconds for real-time monitoring
    autoRefresh: true
  });

  const handleRefresh = async () => {
    try {
      await refreshAllMetrics();
      notification.success({
        message: 'Metrics Updated',
        description: 'All monitoring metrics have been refreshed',
        duration: 2
      });
    } catch (error) {
      notification.error({
        message: 'Refresh Failed',
        description: 'Failed to refresh monitoring metrics',
        duration: 4
      });
    }
  };

  const toggleAutoRefresh = () => {
    if (isAutoRefreshActive) {
      stopAutoRefresh();
      notification.info({
        message: 'Auto-refresh Disabled',
        description: 'Real-time updates have been paused',
        duration: 2
      });
    } else {
      startAutoRefresh();
      notification.success({
        message: 'Auto-refresh Enabled',
        description: 'Real-time updates have been resumed',
        duration: 2
      });
    }
  };

  const getAlertSeverityColor = (severity: string): string => {
    switch (severity) {
      case 'critical': return '#ff4d4f';
      case 'high': return '#fa8c16';
      case 'medium': return '#faad14';
      case 'low': return '#52c41a';
      default: return '#d9d9d9';
    }
  };

  const getConnectionStatusColor = (status: string): string => {
    switch (status) {
      case 'connected': return '#52c41a';
      case 'degraded': return '#faad14';
      case 'disconnected': return '#ff4d4f';
      case 'reconnecting': return '#1890ff';
      default: return '#d9d9d9';
    }
  };

  const renderOverviewTab = () => (
    <div>
      {/* Error Alert */}
      {error && (
        <Alert
          message="Monitoring Error"
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

      {/* Critical Alerts */}
      {activeAlerts.filter(alert => alert.severity === 'critical').length > 0 && (
        <Alert
          message="Critical System Alerts"
          description={`${activeAlerts.filter(alert => alert.severity === 'critical').length} critical alerts require immediate attention`}
          type="error"
          icon={<AlertOutlined />}
          style={{ marginBottom: 16 }}
          showIcon
        />
      )}

      {/* Real-time Metrics Cards */}
      <Row gutter={[16, 16]} style={{ marginBottom: 24 }}>
        <Col xs={24} sm={12} md={6}>
          <Card>
            <Statistic
              title="Avg Order Latency"
              value={realTimeLatency || 0}
              suffix="ms"
              precision={1}
              valueStyle={{ 
                color: systemMonitoringService.getStatusColor('latency', realTimeLatency || 0)
              }}
            />
            <Progress 
              percent={Math.min((realTimeLatency || 0) / 100 * 100, 100)} 
              size="small" 
              status={realTimeLatency && realTimeLatency > 50 ? 'exception' : 'success'}
              showInfo={false}
            />
          </Card>
        </Col>

        <Col xs={24} sm={12} md={6}>
          <Card>
            <Statistic
              title="CPU Usage"
              value={realTimeCpuUsage || 0}
              suffix="%"
              precision={1}
              valueStyle={{ 
                color: systemMonitoringService.getStatusColor('cpu', realTimeCpuUsage || 0)
              }}
            />
            <Progress 
              percent={realTimeCpuUsage || 0} 
              size="small" 
              status={realTimeCpuUsage && realTimeCpuUsage > 80 ? 'exception' : 'success'}
              showInfo={false}
            />
          </Card>
        </Col>

        <Col xs={24} sm={12} md={6}>
          <Card>
            <Statistic
              title="Memory Usage"
              value={realTimeMemoryUsage || 0}
              suffix="%"
              precision={1}
              valueStyle={{ 
                color: systemMonitoringService.getStatusColor('memory', realTimeMemoryUsage || 0)
              }}
            />
            <Progress 
              percent={realTimeMemoryUsage || 0} 
              size="small" 
              status={realTimeMemoryUsage && realTimeMemoryUsage > 85 ? 'exception' : 'success'}
              showInfo={false}
            />
          </Card>
        </Col>

        <Col xs={24} sm={12} md={6}>
          <Card>
            <Statistic
              title="Connected Venues"
              value={connectionQuality.filter(c => c.status === 'connected').length}
              suffix={`/ ${connectionQuality.length}`}
              valueStyle={{ 
                color: connectionQuality.filter(c => c.status === 'connected').length === connectionQuality.length ? '#52c41a' : '#faad14'
              }}
            />
          </Card>
        </Col>
      </Row>

      {/* Latency Overview */}
      <Row gutter={[16, 16]} style={{ marginBottom: 24 }}>
        <Col xs={24} lg={16}>
          <Card title="Venue Latency Overview">
            {latencyMetrics.length > 0 ? (
              <div>
                {latencyMetrics.map((venue, index) => (
                  <div key={venue.venue_name} style={{ marginBottom: 12 }}>
                    <Row justify="space-between" align="middle">
                      <Col>
                        <Text strong>{venue.venue_name}</Text>
                      </Col>
                      <Col>
                        <Space>
                          <Tag color={systemMonitoringService.getStatusColor('latency', venue.order_execution_latency.avg_ms)}>
                            Avg: {venue.order_execution_latency.avg_ms}ms
                          </Tag>
                          <Tag>P95: {venue.order_execution_latency.p95_ms}ms</Tag>
                          <Tag>P99: {venue.order_execution_latency.p99_ms}ms</Tag>
                        </Space>
                      </Col>
                    </Row>
                    <Progress
                      percent={Math.min(venue.order_execution_latency.avg_ms / 100 * 100, 100)}
                      size="small"
                      status={venue.order_execution_latency.avg_ms > 50 ? 'exception' : 'success'}
                      showInfo={false}
                    />
                  </div>
                ))}
              </div>
            ) : (
              <div style={{ textAlign: 'center', padding: 20 }}>
                <Text type="secondary">No latency data available</Text>
              </div>
            )}
          </Card>
        </Col>

        <Col xs={24} lg={8}>
          <Card title="Connection Status">
            {connectionQuality.length > 0 ? (
              <div>
                {connectionQuality.map((connection) => (
                  <div key={connection.venue_name} style={{ marginBottom: 12 }}>
                    <Row justify="space-between" align="middle">
                      <Col>
                        <Text strong>{connection.venue_name}</Text>
                      </Col>
                      <Col>
                        <Badge
                          status={connection.status === 'connected' ? 'success' : 'error'}
                          text={connection.status}
                        />
                      </Col>
                    </Row>
                    <div style={{ marginTop: 4 }}>
                      <Text type="secondary" style={{ fontSize: 12 }}>
                        Quality: {connection.quality_score}/100 | 
                        Uptime: {connection.uptime_percent_24h.toFixed(1)}%
                      </Text>
                    </div>
                  </div>
                ))}
              </div>
            ) : (
              <div style={{ textAlign: 'center', padding: 20 }}>
                <Text type="secondary">No connection data available</Text>
              </div>
            )}
          </Card>
        </Col>
      </Row>

      {/* Active Alerts */}
      {activeAlerts.length > 0 && (
        <Card title="Active Performance Alerts" style={{ marginBottom: 24 }}>
          <div>
            {activeAlerts.map((alert) => (
              <Alert
                key={alert.alert_id}
                message={`${alert.metric_name.toUpperCase()} Alert`}
                description={`${alert.description} (Current: ${alert.current_value}, Threshold: ${alert.threshold_value})`}
                type={alert.severity === 'critical' ? 'error' : alert.severity === 'high' ? 'warning' : 'info'}
                style={{ marginBottom: 8 }}
                action={
                  <Tag color={getAlertSeverityColor(alert.severity)}>
                    {alert.severity.toUpperCase()}
                  </Tag>
                }
              />
            ))}
          </div>
        </Card>
      )}

      {/* System Resource Summary */}
      {systemMetrics && (
        <Card title="System Resource Summary">
          <Row gutter={[16, 16]}>
            <Col xs={24} sm={12} md={6}>
              <div>
                <Text type="secondary">CPU Cores</Text>
                <div><Text strong>{systemMetrics.cpu_metrics.core_count}</Text></div>
              </div>
            </Col>
            <Col xs={24} sm={12} md={6}>
              <div>
                <Text type="secondary">Total Memory</Text>
                <div><Text strong>{systemMetrics.memory_metrics.total_gb.toFixed(1)} GB</Text></div>
              </div>
            </Col>
            <Col xs={24} sm={12} md={6}>
              <div>
                <Text type="secondary">Disk Space</Text>
                <div><Text strong>{systemMetrics.disk_metrics.total_space_gb.toFixed(0)} GB</Text></div>
              </div>
            </Col>
            <Col xs={24} sm={12} md={6}>
              <div>
                <Text type="secondary">Active Processes</Text>
                <div><Text strong>{systemMetrics.process_metrics.total_processes}</Text></div>
              </div>
            </Col>
          </Row>
        </Card>
      )}
    </div>
  );

  return (
    <div className={`system-performance-dashboard ${className || ''}`}>
      <Card>
        <div style={{ marginBottom: 24 }}>
          <Row justify="space-between" align="middle">
            <Col>
              <Title level={2} style={{ margin: 0 }}>
                <MonitorOutlined style={{ marginRight: 8 }} />
                System Performance Monitoring
              </Title>
              <Text type="secondary">
                Real-time system performance, latency, and connection monitoring
              </Text>
            </Col>
            <Col>
              <Space>
                <Select
                  value={selectedVenue}
                  onChange={setSelectedVenue}
                  style={{ width: 120 }}
                  size="small"
                >
                  <Option value="all">All Venues</Option>
                  <Option value="IB">IB</Option>
                  <Option value="Alpaca">Alpaca</Option>
                  <Option value="Binance">Binance</Option>
                </Select>
                
                <Select
                  value={timeframe}
                  onChange={setTimeframe}
                  style={{ width: 80 }}
                  size="small"
                >
                  <Option value="1h">1H</Option>
                  <Option value="6h">6H</Option>
                  <Option value="24h">24H</Option>
                  <Option value="7d">7D</Option>
                </Select>

                <Button
                  size="small"
                  icon={isAutoRefreshActive ? <PauseCircleOutlined /> : <PlayCircleOutlined />}
                  onClick={toggleAutoRefresh}
                  type={isAutoRefreshActive ? 'primary' : 'default'}
                >
                  Auto-refresh
                </Button>
                
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
                Last updated: {lastUpdate.toLocaleTimeString()}
              </Text>
            </div>
          )}
        </div>

        <Tabs activeKey={activeTab} onChange={setActiveTab}>
          <TabPane tab="Overview" key="overview" icon={<DashboardOutlined />}>
            {renderOverviewTab()}
          </TabPane>
          
          <TabPane tab="Latency Details" key="latency" icon={<ApiOutlined />}>
            <div style={{ textAlign: 'center', padding: 60 }}>
              <Text type="secondary">Detailed latency monitoring charts will be implemented here</Text>
            </div>
          </TabPane>
          
          <TabPane tab="System Resources" key="system" icon={<MonitorOutlined />}>
            <div style={{ textAlign: 'center', padding: 60 }}>
              <Text type="secondary">System resource monitoring charts will be implemented here</Text>
            </div>
          </TabPane>
          
          <TabPane tab="Connections" key="connections" icon={<ApiOutlined />}>
            <div style={{ textAlign: 'center', padding: 60 }}>
              <Text type="secondary">Connection quality monitoring will be implemented here</Text>
            </div>
          </TabPane>
          
          <TabPane tab="Alerts" key="alerts" icon={<AlertOutlined />}>
            <div style={{ textAlign: 'center', padding: 60 }}>
              <Text type="secondary">Alert configuration and management will be implemented here</Text>
            </div>
          </TabPane>

          <TabPane tab="Trends" key="trends" icon={<SettingOutlined />}>
            <div style={{ textAlign: 'center', padding: 60 }}>
              <Text type="secondary">Performance trends and capacity planning will be implemented here</Text>
            </div>
          </TabPane>
        </Tabs>

        {loading && (
          <div style={{ textAlign: 'center', padding: 20 }}>
            <Spin size="large" />
          </div>
        )}
      </Card>
    </div>
  );
};

export default SystemPerformanceDashboard;