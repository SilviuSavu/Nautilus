/**
 * WebSocket Monitoring Suite Component
 * Sprint 3: Advanced WebSocket Infrastructure
 * 
 * Comprehensive monitoring dashboard providing real-time visibility into WebSocket
 * connections, performance metrics, subscription management, and system health.
 */

import React, { useState, useEffect, useMemo, useCallback, memo } from 'react';
import {
  Card,
  Row,
  Col,
  Tabs,
  Typography,
  Space,
  Button,
  Switch,
  Select,
  Badge,
  Progress,
  Alert,
  Statistic,
  Table,
  Tooltip,
  Divider,
  Modal,
  notification,
  Input,
  DatePicker,
  Tag
} from 'antd';
import {
  MonitorOutlined,
  DashboardOutlined,
  LineChartOutlined,
  SettingOutlined,
  ReloadOutlined,
  ExportOutlined,
  FilterOutlined,
  BellOutlined,
  CloudServerOutlined,
  ThunderboltOutlined,
  WifiOutlined,
  DisconnectOutlined,
  WarningOutlined,
  CheckCircleOutlined,
  ClockCircleOutlined,
  ApiOutlined
} from '@ant-design/icons';
import {
  LineChart,
  Line,
  AreaChart,
  Area,
  BarChart,
  Bar,
  PieChart,
  Pie,
  Cell,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip as RechartsTooltip,
  ResponsiveContainer,
  ReferenceLine,
  Legend
} from 'recharts';
import dayjs from 'dayjs';
import { useWebSocketManager } from '../../hooks/useWebSocketManager';
import { ConnectionHealthDashboard } from './ConnectionHealthDashboard';
import { MessageThroughputAnalyzer } from './MessageThroughputAnalyzer';
import { WebSocketScalabilityMonitor } from './WebSocketScalabilityMonitor';
import { StreamingPerformanceTracker } from './StreamingPerformanceTracker';

const { Title, Text } = Typography;
const { TabPane } = Tabs;
const { Option } = Select;
const { RangePicker } = DatePicker;

interface WebSocketEndpoint {
  id: string;
  name: string;
  url: string;
  status: 'connected' | 'disconnected' | 'connecting' | 'error';
  latency: number;
  messagesPerSecond: number;
  subscriptions: number;
  lastActivity: string;
  uptime: number;
  errorCount: number;
  quality: number;
}

interface AlertRule {
  id: string;
  name: string;
  type: 'latency' | 'throughput' | 'error_rate' | 'connection' | 'subscription';
  threshold: number;
  comparison: 'gt' | 'lt' | 'eq';
  enabled: boolean;
  severity: 'low' | 'medium' | 'high' | 'critical';
  endpoint?: string;
}

interface WebSocketMonitoringSuiteProps {
  className?: string;
  defaultTab?: string;
  showAlerts?: boolean;
  showExport?: boolean;
  refreshInterval?: number;
  maxHistoryPoints?: number;
  enableNotifications?: boolean;
  customEndpoints?: string[];
}

export const WebSocketMonitoringSuite: React.FC<WebSocketMonitoringSuiteProps> = memo(({
  className,
  defaultTab = 'overview',
  showAlerts = true,
  showExport = true,
  refreshInterval = 1000,
  maxHistoryPoints = 100,
  enableNotifications = true,
  customEndpoints = []
}) => {
  const {
    connectionState,
    messageLatency,
    messagesReceived,
    messagesSent,
    subscriptionCount,
    getConnectionInfo,
    getMessageStats
  } = useWebSocketManager();

  // State management
  const [activeTab, setActiveTab] = useState<string>(defaultTab);
  const [isMonitoring, setIsMonitoring] = useState<boolean>(true);
  const [timeRange, setTimeRange] = useState<string>('1h');
  const [selectedEndpoints, setSelectedEndpoints] = useState<string[]>([]);
  const [alertRules, setAlertRules] = useState<AlertRule[]>([]);
  const [activeAlerts, setActiveAlerts] = useState<any[]>([]);
  const [showSettings, setShowSettings] = useState<boolean>(false);
  const [autoRefresh, setAutoRefresh] = useState<boolean>(true);
  const [historicalData, setHistoricalData] = useState<any[]>([]);
  const [performanceThresholds, setPerformanceThresholds] = useState({
    latencyWarning: 100,
    latencyError: 500,
    throughputWarning: 10,
    throughputError: 5,
    errorRateWarning: 5,
    errorRateError: 10
  });

  // Mock WebSocket endpoints for demonstration
  const [endpoints, setEndpoints] = useState<WebSocketEndpoint[]>([
    {
      id: 'engine-status',
      name: 'Engine Status',
      url: '/ws/engine/status',
      status: 'connected',
      latency: 45,
      messagesPerSecond: 12,
      subscriptions: 3,
      lastActivity: new Date().toISOString(),
      uptime: 125000,
      errorCount: 0,
      quality: 98
    },
    {
      id: 'market-data',
      name: 'Market Data',
      url: '/ws/market-data/AAPL',
      status: 'connected',
      latency: 23,
      messagesPerSecond: 156,
      subscriptions: 8,
      lastActivity: new Date().toISOString(),
      uptime: 89000,
      errorCount: 2,
      quality: 94
    },
    {
      id: 'trade-updates',
      name: 'Trade Updates',
      url: '/ws/trades/updates',
      status: 'connecting',
      latency: 78,
      messagesPerSecond: 45,
      subscriptions: 5,
      lastActivity: new Date().toISOString(),
      uptime: 45000,
      errorCount: 1,
      quality: 87
    },
    {
      id: 'system-health',
      name: 'System Health',
      url: '/ws/system/health',
      status: 'connected',
      latency: 12,
      messagesPerSecond: 8,
      subscriptions: 2,
      lastActivity: new Date().toISOString(),
      uptime: 185000,
      errorCount: 0,
      quality: 100
    }
  ]);

  // Calculate system overview statistics
  const systemStats = useMemo(() => {
    const totalEndpoints = endpoints.length;
    const connectedEndpoints = endpoints.filter(ep => ep.status === 'connected').length;
    const totalMessages = endpoints.reduce((sum, ep) => sum + ep.messagesPerSecond, 0);
    const avgLatency = endpoints.reduce((sum, ep) => sum + ep.latency, 0) / totalEndpoints;
    const totalSubscriptions = endpoints.reduce((sum, ep) => sum + ep.subscriptions, 0);
    const totalErrors = endpoints.reduce((sum, ep) => sum + ep.errorCount, 0);
    const avgQuality = endpoints.reduce((sum, ep) => sum + ep.quality, 0) / totalEndpoints;
    const connectionHealth = (connectedEndpoints / totalEndpoints) * 100;

    return {
      totalEndpoints,
      connectedEndpoints,
      totalMessages,
      avgLatency,
      totalSubscriptions,
      totalErrors,
      avgQuality,
      connectionHealth
    };
  }, [endpoints]);

  // Update historical data
  useEffect(() => {
    if (isMonitoring && autoRefresh) {
      const interval = setInterval(() => {
        const dataPoint = {
          timestamp: Date.now(),
          time: new Date().toISOString(),
          latency: systemStats.avgLatency,
          throughput: systemStats.totalMessages,
          connections: systemStats.connectedEndpoints,
          quality: systemStats.avgQuality,
          errors: systemStats.totalErrors
        };

        setHistoricalData(prev => {
          const newData = [...prev, dataPoint];
          return newData.slice(-maxHistoryPoints);
        });
      }, refreshInterval);

      return () => clearInterval(interval);
    }
  }, [isMonitoring, autoRefresh, refreshInterval, maxHistoryPoints, systemStats]);

  // Alert processing
  useEffect(() => {
    const processAlerts = () => {
      const newAlerts: any[] = [];

      alertRules.forEach(rule => {
        if (!rule.enabled) return;

        const relevantEndpoints = rule.endpoint 
          ? endpoints.filter(ep => ep.id === rule.endpoint)
          : endpoints;

        relevantEndpoints.forEach(endpoint => {
          let value: number;
          let description: string;

          switch (rule.type) {
            case 'latency':
              value = endpoint.latency;
              description = `Latency ${value}ms exceeds threshold ${rule.threshold}ms`;
              break;
            case 'throughput':
              value = endpoint.messagesPerSecond;
              description = `Throughput ${value} msg/s below threshold ${rule.threshold} msg/s`;
              break;
            case 'error_rate':
              const totalMessages = endpoint.messagesPerSecond * (endpoint.uptime / 1000);
              value = totalMessages > 0 ? (endpoint.errorCount / totalMessages) * 100 : 0;
              description = `Error rate ${value.toFixed(2)}% exceeds threshold ${rule.threshold}%`;
              break;
            case 'connection':
              value = endpoint.status === 'connected' ? 1 : 0;
              description = `Connection to ${endpoint.name} is ${endpoint.status}`;
              break;
            case 'subscription':
              value = endpoint.subscriptions;
              description = `Subscription count ${value} differs from threshold ${rule.threshold}`;
              break;
            default:
              return;
          }

          let triggered = false;
          switch (rule.comparison) {
            case 'gt':
              triggered = value > rule.threshold;
              break;
            case 'lt':
              triggered = value < rule.threshold;
              break;
            case 'eq':
              triggered = value === rule.threshold;
              break;
          }

          if (triggered) {
            newAlerts.push({
              id: `${rule.id}-${endpoint.id}-${Date.now()}`,
              ruleId: rule.id,
              ruleName: rule.name,
              endpointId: endpoint.id,
              endpointName: endpoint.name,
              severity: rule.severity,
              description,
              value,
              threshold: rule.threshold,
              timestamp: new Date().toISOString()
            });
          }
        });
      });

      if (newAlerts.length > 0 && enableNotifications) {
        newAlerts.forEach(alert => {
          const notificationType = alert.severity === 'critical' || alert.severity === 'high' 
            ? 'error' : 'warning';
          
          notification[notificationType]({
            message: `WebSocket Alert: ${alert.ruleName}`,
            description: alert.description,
            placement: 'topRight',
            duration: alert.severity === 'critical' ? 0 : 4.5
          });
        });
      }

      setActiveAlerts(prev => [...prev, ...newAlerts].slice(-50));
    };

    if (alertRules.length > 0) {
      processAlerts();
    }
  }, [endpoints, alertRules, enableNotifications]);

  // Export data functionality
  const handleExportData = useCallback(async () => {
    const exportData = {
      timestamp: new Date().toISOString(),
      systemStats,
      endpoints,
      historicalData: historicalData.slice(-1000), // Last 1000 points
      alerts: activeAlerts.slice(-100), // Last 100 alerts
      configuration: {
        alertRules,
        performanceThresholds,
        timeRange
      }
    };

    const dataStr = JSON.stringify(exportData, null, 2);
    const dataBlob = new Blob([dataStr], { type: 'application/json' });
    const url = URL.createObjectURL(dataBlob);
    
    const link = document.createElement('a');
    link.href = url;
    link.download = `websocket-monitoring-${Date.now()}.json`;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(url);

    notification.success({
      message: 'Data Exported',
      description: 'WebSocket monitoring data has been exported successfully',
      placement: 'topRight'
    });
  }, [systemStats, endpoints, historicalData, activeAlerts, alertRules, performanceThresholds, timeRange]);

  // Status indicators
  const getStatusColor = (status: string) => {
    switch (status) {
      case 'connected': return '#52c41a';
      case 'connecting': return '#1890ff';
      case 'disconnected': return '#d9d9d9';
      case 'error': return '#ff4d4f';
      default: return '#faad14';
    }
  };

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'connected': return <CheckCircleOutlined style={{ color: '#52c41a' }} />;
      case 'connecting': return <ClockCircleOutlined style={{ color: '#1890ff' }} />;
      case 'disconnected': return <DisconnectOutlined style={{ color: '#d9d9d9' }} />;
      case 'error': return <WarningOutlined style={{ color: '#ff4d4f' }} />;
      default: return <ApiOutlined style={{ color: '#faad14' }} />;
    }
  };

  const getQualityColor = (quality: number) => {
    if (quality >= 95) return '#52c41a';
    if (quality >= 85) return '#faad14';
    if (quality >= 70) return '#fa8c16';
    return '#ff4d4f';
  };

  // Table columns for endpoints
  const endpointColumns = [
    {
      title: 'Endpoint',
      dataIndex: 'name',
      key: 'name',
      render: (text: string, record: WebSocketEndpoint) => (
        <Space>
          {getStatusIcon(record.status)}
          <div>
            <Text strong>{text}</Text>
            <br />
            <Text type="secondary" style={{ fontSize: '12px' }}>
              {record.url}
            </Text>
          </div>
        </Space>
      )
    },
    {
      title: 'Status',
      dataIndex: 'status',
      key: 'status',
      render: (status: string) => (
        <Badge 
          color={getStatusColor(status)} 
          text={status.charAt(0).toUpperCase() + status.slice(1)} 
        />
      )
    },
    {
      title: 'Quality',
      dataIndex: 'quality',
      key: 'quality',
      render: (quality: number) => (
        <div>
          <Progress
            percent={quality}
            size="small"
            strokeColor={getQualityColor(quality)}
            format={(percent) => `${percent}%`}
          />
        </div>
      )
    },
    {
      title: 'Latency',
      dataIndex: 'latency',
      key: 'latency',
      render: (latency: number) => (
        <Text style={{ color: latency > performanceThresholds.latencyWarning ? '#ff4d4f' : '#52c41a' }}>
          {latency}ms
        </Text>
      ),
      sorter: (a: WebSocketEndpoint, b: WebSocketEndpoint) => a.latency - b.latency
    },
    {
      title: 'Throughput',
      dataIndex: 'messagesPerSecond',
      key: 'throughput',
      render: (rate: number) => `${rate} msg/s`,
      sorter: (a: WebSocketEndpoint, b: WebSocketEndpoint) => a.messagesPerSecond - b.messagesPerSecond
    },
    {
      title: 'Subscriptions',
      dataIndex: 'subscriptions',
      key: 'subscriptions',
      sorter: (a: WebSocketEndpoint, b: WebSocketEndpoint) => a.subscriptions - b.subscriptions
    },
    {
      title: 'Errors',
      dataIndex: 'errorCount',
      key: 'errors',
      render: (count: number) => (
        <Text type={count > 0 ? 'danger' : 'success'}>
          {count}
        </Text>
      ),
      sorter: (a: WebSocketEndpoint, b: WebSocketEndpoint) => a.errorCount - b.errorCount
    },
    {
      title: 'Uptime',
      dataIndex: 'uptime',
      key: 'uptime',
      render: (uptime: number) => {
        const hours = Math.floor(uptime / 3600000);
        const minutes = Math.floor((uptime % 3600000) / 60000);
        return `${hours}h ${minutes}m`;
      }
    }
  ];

  // Chart data preparation
  const chartData = historicalData.slice(-50).map((point, index) => ({
    ...point,
    index,
    time: new Date(point.timestamp).toLocaleTimeString()
  }));

  return (
    <div className={className}>
      <Card
        title={
          <Space>
            <MonitorOutlined />
            <Title level={4} style={{ margin: 0 }}>
              WebSocket Monitoring Suite
            </Title>
            <Badge 
              count={systemStats.connectedEndpoints} 
              showZero 
              style={{ backgroundColor: '#52c41a' }} 
            />
            <Badge 
              count={activeAlerts.length} 
              showZero 
              style={{ backgroundColor: '#ff4d4f' }} 
            />
          </Space>
        }
        extra={
          <Space>
            <Switch
              checkedChildren="Monitoring"
              unCheckedChildren="Stopped"
              checked={isMonitoring}
              onChange={setIsMonitoring}
            />
            <Switch
              checkedChildren="Auto-refresh"
              unCheckedChildren="Manual"
              checked={autoRefresh}
              onChange={setAutoRefresh}
              size="small"
            />
            <Select 
              value={timeRange}
              onChange={setTimeRange}
              size="small"
              style={{ width: 80 }}
            >
              <Option value="5m">5m</Option>
              <Option value="15m">15m</Option>
              <Option value="1h">1h</Option>
              <Option value="6h">6h</Option>
              <Option value="24h">24h</Option>
            </Select>
            {showExport && (
              <Button
                type="text"
                icon={<ExportOutlined />}
                onClick={handleExportData}
                size="small"
              >
                Export
              </Button>
            )}
            <Button
              type="text"
              icon={<SettingOutlined />}
              onClick={() => setShowSettings(true)}
              size="small"
            />
          </Space>
        }
      >
        {/* System Overview Stats */}
        <Row gutter={[16, 16]} style={{ marginBottom: 24 }}>
          <Col span={6}>
            <Statistic
              title="Connection Health"
              value={systemStats.connectionHealth}
              precision={1}
              suffix="%"
              valueStyle={{ color: systemStats.connectionHealth > 90 ? '#52c41a' : '#ff4d4f' }}
              prefix={<WifiOutlined />}
            />
          </Col>
          <Col span={6}>
            <Statistic
              title="Avg Latency"
              value={systemStats.avgLatency}
              precision={0}
              suffix="ms"
              valueStyle={{ 
                color: systemStats.avgLatency < performanceThresholds.latencyWarning ? '#52c41a' : '#ff4d4f' 
              }}
              prefix={<ThunderboltOutlined />}
            />
          </Col>
          <Col span={6}>
            <Statistic
              title="Total Throughput"
              value={systemStats.totalMessages}
              precision={0}
              suffix="msg/s"
              valueStyle={{ color: '#1890ff' }}
              prefix={<LineChartOutlined />}
            />
          </Col>
          <Col span={6}>
            <Statistic
              title="System Quality"
              value={systemStats.avgQuality}
              precision={1}
              suffix="%"
              valueStyle={{ color: getQualityColor(systemStats.avgQuality) }}
              prefix={<DashboardOutlined />}
            />
          </Col>
        </Row>

        {/* Active Alerts */}
        {showAlerts && activeAlerts.length > 0 && (
          <Alert
            message={`${activeAlerts.length} Active Alert${activeAlerts.length > 1 ? 's' : ''}`}
            description={
              <Space direction="vertical" size="small">
                {activeAlerts.slice(-3).map((alert, index) => (
                  <div key={index}>
                    <Tag color={alert.severity === 'critical' ? 'red' : alert.severity === 'high' ? 'orange' : 'yellow'}>
                      {alert.severity.toUpperCase()}
                    </Tag>
                    <Text>{alert.endpointName}: {alert.description}</Text>
                  </div>
                ))}
                {activeAlerts.length > 3 && (
                  <Text type="secondary">... and {activeAlerts.length - 3} more alerts</Text>
                )}
              </Space>
            }
            type="warning"
            showIcon
            style={{ marginBottom: 16 }}
          />
        )}

        {/* Main Content Tabs */}
        <Tabs
          activeKey={activeTab}
          onChange={setActiveTab}
          tabBarExtraContent={
            <Space>
              <Text type="secondary">
                Last updated: {new Date().toLocaleTimeString()}
              </Text>
              <Button
                type="text"
                icon={<ReloadOutlined />}
                onClick={() => window.location.reload()}
                size="small"
              />
            </Space>
          }
        >
          <TabPane tab="Overview" key="overview">
            <Row gutter={[16, 16]}>
              <Col span={24}>
                <Card title="System Performance Trends" size="small">
                  <ResponsiveContainer width="100%" height={300}>
                    <LineChart data={chartData}>
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis dataKey="time" />
                      <YAxis yAxisId="left" />
                      <YAxis yAxisId="right" orientation="right" />
                      <RechartsTooltip />
                      <Legend />
                      <Line 
                        yAxisId="left"
                        type="monotone" 
                        dataKey="latency" 
                        stroke="#ff4d4f" 
                        name="Latency (ms)"
                        strokeWidth={2}
                      />
                      <Line 
                        yAxisId="right"
                        type="monotone" 
                        dataKey="throughput" 
                        stroke="#1890ff" 
                        name="Throughput (msg/s)"
                        strokeWidth={2}
                      />
                      <Line 
                        yAxisId="left"
                        type="monotone" 
                        dataKey="quality" 
                        stroke="#52c41a" 
                        name="Quality (%)"
                        strokeWidth={2}
                      />
                      <ReferenceLine 
                        yAxisId="left"
                        y={performanceThresholds.latencyWarning} 
                        stroke="#faad14" 
                        strokeDasharray="5 5" 
                        label="Latency Warning"
                      />
                    </LineChart>
                  </ResponsiveContainer>
                </Card>
              </Col>
              
              <Col span={24}>
                <Card title="WebSocket Endpoints" size="small">
                  <Table
                    dataSource={endpoints}
                    columns={endpointColumns}
                    rowKey="id"
                    size="small"
                    pagination={false}
                    scroll={{ x: 1000 }}
                  />
                </Card>
              </Col>
            </Row>
          </TabPane>

          <TabPane tab="Connection Health" key="health">
            <ConnectionHealthDashboard 
              endpoints={endpoints}
              performanceThresholds={performanceThresholds}
              timeRange={timeRange}
              refreshInterval={refreshInterval}
            />
          </TabPane>

          <TabPane tab="Message Analysis" key="messages">
            <MessageThroughputAnalyzer
              endpoints={endpoints}
              historicalData={historicalData}
              timeRange={timeRange}
              showRealtime={autoRefresh}
            />
          </TabPane>

          <TabPane tab="Scalability" key="scalability">
            <WebSocketScalabilityMonitor
              endpoints={endpoints}
              connectionCount={systemStats.connectedEndpoints}
              targetConnections={1000}
              performanceMetrics={systemStats}
            />
          </TabPane>

          <TabPane tab="Performance" key="performance">
            <StreamingPerformanceTracker
              endpoints={endpoints}
              historicalData={historicalData}
              performanceThresholds={performanceThresholds}
              enableBenchmarking={true}
            />
          </TabPane>
        </Tabs>

        {/* Settings Modal */}
        <Modal
          title="Monitoring Configuration"
          open={showSettings}
          onCancel={() => setShowSettings(false)}
          width={800}
          footer={[
            <Button key="cancel" onClick={() => setShowSettings(false)}>
              Cancel
            </Button>,
            <Button 
              key="save" 
              type="primary" 
              onClick={() => {
                setShowSettings(false);
                notification.success({
                  message: 'Settings Saved',
                  description: 'Monitoring configuration has been updated',
                  placement: 'topRight'
                });
              }}
            >
              Save Settings
            </Button>
          ]}
        >
          <Tabs defaultActiveKey="general">
            <TabPane tab="General" key="general">
              <Space direction="vertical" style={{ width: '100%' }}>
                <Row gutter={16}>
                  <Col span={12}>
                    <Text>Refresh Interval (ms):</Text>
                    <Input
                      type="number"
                      value={refreshInterval}
                      onChange={(e) => setRefreshInterval(Number(e.target.value))}
                      min={500}
                      max={10000}
                    />
                  </Col>
                  <Col span={12}>
                    <Text>Max History Points:</Text>
                    <Input
                      type="number"
                      value={maxHistoryPoints}
                      onChange={(e) => setMaxHistoryPoints(Number(e.target.value))}
                      min={50}
                      max={1000}
                    />
                  </Col>
                </Row>
              </Space>
            </TabPane>
            
            <TabPane tab="Thresholds" key="thresholds">
              <Space direction="vertical" style={{ width: '100%' }}>
                <Row gutter={16}>
                  <Col span={12}>
                    <Text>Latency Warning (ms):</Text>
                    <Input
                      type="number"
                      value={performanceThresholds.latencyWarning}
                      onChange={(e) => setPerformanceThresholds(prev => ({
                        ...prev,
                        latencyWarning: Number(e.target.value)
                      }))}
                    />
                  </Col>
                  <Col span={12}>
                    <Text>Latency Error (ms):</Text>
                    <Input
                      type="number"
                      value={performanceThresholds.latencyError}
                      onChange={(e) => setPerformanceThresholds(prev => ({
                        ...prev,
                        latencyError: Number(e.target.value)
                      }))}
                    />
                  </Col>
                </Row>
              </Space>
            </TabPane>
          </Tabs>
        </Modal>
      </Card>
    </div>
  );
});
WebSocketMonitoringSuite.displayName = 'WebSocketMonitoringSuite';

export default WebSocketMonitoringSuite;