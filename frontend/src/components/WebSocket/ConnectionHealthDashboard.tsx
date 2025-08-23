/**
 * Connection Health Dashboard Component
 * Sprint 3: Advanced WebSocket Infrastructure
 * 
 * Comprehensive connection health monitoring with real-time metrics, quality scoring,
 * connection stability analysis, and proactive health alerts.
 */

import React, { useState, useEffect, useMemo, useCallback } from 'react';
import {
  Card,
  Row,
  Col,
  Typography,
  Space,
  Progress,
  Statistic,
  Alert,
  Table,
  Tag,
  Tooltip,
  Button,
  Select,
  Switch,
  Badge,
  Drawer,
  Timeline,
  List,
  Avatar,
  Divider
} from 'antd';
import {
  HeartOutlined,
  ThunderboltOutlined,
  WifiOutlined,
  DisconnectOutlined,
  WarningOutlined,
  CheckCircleOutlined,
  ClockCircleOutlined,
  LineChartOutlined,
  BarChartOutlined,
  DashboardOutlined,
  SettingOutlined,
  ReloadOutlined,
  ApiOutlined,
  CloudServerOutlined,
  MonitorOutlined
} from '@ant-design/icons';
import {
  LineChart,
  Line,
  AreaChart,
  Area,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip as RechartsTooltip,
  ResponsiveContainer,
  ReferenceLine,
  Legend,
  RadialBarChart,
  RadialBar,
  PieChart,
  Pie,
  Cell
} from 'recharts';

const { Title, Text } = Typography;
const { Option } = Select;

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
  reconnectCount?: number;
  packetLoss?: number;
  stabilityScore?: number;
  healthHistory?: HealthDataPoint[];
}

interface HealthDataPoint {
  timestamp: number;
  latency: number;
  quality: number;
  connectionState: number; // 1 for connected, 0 for disconnected
  errorRate: number;
  throughput: number;
}

interface ConnectionHealthDashboardProps {
  endpoints: WebSocketEndpoint[];
  performanceThresholds: {
    latencyWarning: number;
    latencyError: number;
    throughputWarning: number;
    throughputError: number;
    errorRateWarning: number;
    errorRateError: number;
  };
  timeRange: string;
  refreshInterval?: number;
  showDetailedAnalysis?: boolean;
  className?: string;
}

interface HealthAnalysis {
  overallHealth: number;
  healthyConnections: number;
  degradedConnections: number;
  criticalConnections: number;
  averageLatency: number;
  packetLossRate: number;
  stabilityScore: number;
  recommendations: string[];
}

export const ConnectionHealthDashboard: React.FC<ConnectionHealthDashboardProps> = ({
  endpoints,
  performanceThresholds,
  timeRange,
  refreshInterval = 1000,
  showDetailedAnalysis = true,
  className
}) => {
  const [selectedEndpoint, setSelectedEndpoint] = useState<string>('all');
  const [healthHistoryData, setHealthHistoryData] = useState<HealthDataPoint[]>([]);
  const [showHealthDetail, setShowHealthDetail] = useState<boolean>(false);
  const [detailEndpointId, setDetailEndpointId] = useState<string>('');
  const [alertThreshold, setAlertThreshold] = useState<number>(80);
  const [enablePredictiveAlerts, setEnablePredictiveAlerts] = useState<boolean>(true);

  // Enhanced health analysis
  const healthAnalysis: HealthAnalysis = useMemo(() => {
    const totalEndpoints = endpoints.length;
    if (totalEndpoints === 0) {
      return {
        overallHealth: 0,
        healthyConnections: 0,
        degradedConnections: 0,
        criticalConnections: 0,
        averageLatency: 0,
        packetLossRate: 0,
        stabilityScore: 0,
        recommendations: ['No active connections detected']
      };
    }

    let healthyCount = 0;
    let degradedCount = 0;
    let criticalCount = 0;
    let totalLatency = 0;
    let totalPacketLoss = 0;
    let totalStability = 0;

    const recommendations: string[] = [];

    endpoints.forEach(endpoint => {
      // Categorize health
      if (endpoint.quality >= 90 && endpoint.status === 'connected') {
        healthyCount++;
      } else if (endpoint.quality >= 70 || endpoint.status === 'connecting') {
        degradedCount++;
      } else {
        criticalCount++;
      }

      totalLatency += endpoint.latency;
      totalPacketLoss += endpoint.packetLoss || 0;
      totalStability += endpoint.stabilityScore || endpoint.quality;

      // Generate recommendations
      if (endpoint.latency > performanceThresholds.latencyError) {
        recommendations.push(`High latency detected on ${endpoint.name} (${endpoint.latency}ms)`);
      }
      if (endpoint.errorCount > 5) {
        recommendations.push(`Frequent errors on ${endpoint.name} (${endpoint.errorCount} errors)`);
      }
      if (endpoint.quality < 70) {
        recommendations.push(`Poor connection quality on ${endpoint.name} (${endpoint.quality}%)`);
      }
      if ((endpoint.reconnectCount || 0) > 3) {
        recommendations.push(`Unstable connection on ${endpoint.name} (${endpoint.reconnectCount} reconnects)`);
      }
    });

    const overallHealth = (healthyCount / totalEndpoints) * 100;
    const averageLatency = totalLatency / totalEndpoints;
    const packetLossRate = totalPacketLoss / totalEndpoints;
    const stabilityScore = totalStability / totalEndpoints;

    // System-wide recommendations
    if (overallHealth < 70) {
      recommendations.push('System health is degraded - consider investigating network connectivity');
    }
    if (averageLatency > performanceThresholds.latencyWarning) {
      recommendations.push('Average latency is high - check server performance and network conditions');
    }
    if (packetLossRate > 2) {
      recommendations.push('Packet loss detected - verify network stability');
    }

    return {
      overallHealth,
      healthyConnections: healthyCount,
      degradedConnections: degradedCount,
      criticalConnections: criticalCount,
      averageLatency,
      packetLossRate,
      stabilityScore,
      recommendations: recommendations.slice(0, 5) // Limit to top 5 recommendations
    };
  }, [endpoints, performanceThresholds]);

  // Generate health history data
  useEffect(() => {
    const generateHealthData = () => {
      const now = Date.now();
      const dataPoint: HealthDataPoint = {
        timestamp: now,
        latency: healthAnalysis.averageLatency,
        quality: healthAnalysis.overallHealth,
        connectionState: healthAnalysis.healthyConnections / endpoints.length,
        errorRate: endpoints.reduce((sum, ep) => sum + ep.errorCount, 0) / Math.max(endpoints.length, 1),
        throughput: endpoints.reduce((sum, ep) => sum + ep.messagesPerSecond, 0)
      };

      setHealthHistoryData(prev => {
        const newData = [...prev, dataPoint];
        // Keep last 100 points based on time range
        const maxPoints = timeRange === '5m' ? 50 : timeRange === '15m' ? 60 : 100;
        return newData.slice(-maxPoints);
      });
    };

    if (endpoints.length > 0) {
      generateHealthData();
      const interval = setInterval(generateHealthData, refreshInterval);
      return () => clearInterval(interval);
    }
  }, [endpoints, healthAnalysis, timeRange, refreshInterval]);

  // Health score calculation
  const calculateConnectionHealth = useCallback((endpoint: WebSocketEndpoint): number => {
    let score = 100;

    // Latency impact (30%)
    if (endpoint.latency > performanceThresholds.latencyError) {
      score -= 30;
    } else if (endpoint.latency > performanceThresholds.latencyWarning) {
      score -= 15;
    }

    // Connection status impact (25%)
    switch (endpoint.status) {
      case 'connected':
        break; // No penalty
      case 'connecting':
        score -= 10;
        break;
      case 'disconnected':
        score -= 25;
        break;
      case 'error':
        score -= 25;
        break;
    }

    // Error rate impact (20%)
    const errorRate = endpoint.errorCount / Math.max(endpoint.messagesPerSecond * (endpoint.uptime / 1000), 1);
    if (errorRate > 0.1) score -= 20;
    else if (errorRate > 0.05) score -= 10;

    // Throughput impact (15%)
    if (endpoint.messagesPerSecond < performanceThresholds.throughputError) {
      score -= 15;
    } else if (endpoint.messagesPerSecond < performanceThresholds.throughputWarning) {
      score -= 7;
    }

    // Stability impact (10%)
    const reconnectPenalty = Math.min((endpoint.reconnectCount || 0) * 2, 10);
    score -= reconnectPenalty;

    return Math.max(0, Math.min(100, score));
  }, [performanceThresholds]);

  // Get health color
  const getHealthColor = (health: number): string => {
    if (health >= 90) return '#52c41a';
    if (health >= 80) return '#faad14';
    if (health >= 60) return '#fa8c16';
    return '#ff4d4f';
  };

  // Get status badge
  const getStatusBadge = (status: string) => {
    const colors = {
      connected: 'success',
      connecting: 'processing',
      disconnected: 'default',
      error: 'error'
    };
    return <Badge status={colors[status] as any} text={status.charAt(0).toUpperCase() + status.slice(1)} />;
  };

  // Table columns for detailed endpoint health
  const healthColumns = [
    {
      title: 'Endpoint',
      dataIndex: 'name',
      key: 'name',
      render: (text: string, record: WebSocketEndpoint) => (
        <Space>
          <Avatar 
            size="small" 
            icon={<ApiOutlined />} 
            style={{ backgroundColor: getHealthColor(record.quality) }}
          />
          <div>
            <Text strong>{text}</Text>
            <br />
            <Text type="secondary" style={{ fontSize: '11px' }}>
              {record.url}
            </Text>
          </div>
        </Space>
      )
    },
    {
      title: 'Health Score',
      key: 'health',
      render: (record: WebSocketEndpoint) => {
        const health = calculateConnectionHealth(record);
        return (
          <div>
            <Progress
              percent={health}
              size="small"
              strokeColor={getHealthColor(health)}
              format={(percent) => `${percent?.toFixed(0)}%`}
            />
          </div>
        );
      },
      sorter: (a: WebSocketEndpoint, b: WebSocketEndpoint) => 
        calculateConnectionHealth(a) - calculateConnectionHealth(b)
    },
    {
      title: 'Status',
      dataIndex: 'status',
      key: 'status',
      render: (status: string) => getStatusBadge(status),
      filters: [
        { text: 'Connected', value: 'connected' },
        { text: 'Connecting', value: 'connecting' },
        { text: 'Disconnected', value: 'disconnected' },
        { text: 'Error', value: 'error' }
      ],
      onFilter: (value: any, record: WebSocketEndpoint) => record.status === value
    },
    {
      title: 'Latency',
      dataIndex: 'latency',
      key: 'latency',
      render: (latency: number) => (
        <Text style={{ 
          color: latency > performanceThresholds.latencyError ? '#ff4d4f' : 
                latency > performanceThresholds.latencyWarning ? '#faad14' : '#52c41a' 
        }}>
          {latency}ms
        </Text>
      ),
      sorter: (a: WebSocketEndpoint, b: WebSocketEndpoint) => a.latency - b.latency
    },
    {
      title: 'Stability',
      key: 'stability',
      render: (record: WebSocketEndpoint) => (
        <Space direction="vertical" size="small">
          <Text style={{ fontSize: '12px' }}>
            Uptime: {Math.floor(record.uptime / 60000)}m
          </Text>
          <Text style={{ fontSize: '12px' }}>
            Reconnects: {record.reconnectCount || 0}
          </Text>
        </Space>
      )
    },
    {
      title: 'Actions',
      key: 'actions',
      render: (record: WebSocketEndpoint) => (
        <Button
          type="link"
          size="small"
          icon={<MonitorOutlined />}
          onClick={() => {
            setDetailEndpointId(record.id);
            setShowHealthDetail(true);
          }}
        >
          Details
        </Button>
      )
    }
  ];

  // Chart data preparation
  const healthTrendData = healthHistoryData.slice(-30).map((point, index) => ({
    time: new Date(point.timestamp).toLocaleTimeString(),
    health: point.quality,
    latency: point.latency,
    connections: point.connectionState * 100,
    throughput: point.throughput,
    errors: point.errorRate
  }));

  // Health distribution data for radial chart
  const healthDistributionData = [
    { name: 'Healthy', value: healthAnalysis.healthyConnections, color: '#52c41a' },
    { name: 'Degraded', value: healthAnalysis.degradedConnections, color: '#faad14' },
    { name: 'Critical', value: healthAnalysis.criticalConnections, color: '#ff4d4f' }
  ].filter(item => item.value > 0);

  return (
    <div className={className}>
      <Space direction="vertical" size="large" style={{ width: '100%' }}>
        {/* Health Overview Cards */}
        <Row gutter={[16, 16]}>
          <Col span={6}>
            <Card size="small">
              <Statistic
                title="Overall Health"
                value={healthAnalysis.overallHealth}
                precision={1}
                suffix="%"
                valueStyle={{ color: getHealthColor(healthAnalysis.overallHealth) }}
                prefix={<HeartOutlined />}
              />
              <Progress
                percent={healthAnalysis.overallHealth}
                strokeColor={getHealthColor(healthAnalysis.overallHealth)}
                size="small"
                showInfo={false}
              />
            </Card>
          </Col>
          
          <Col span={6}>
            <Card size="small">
              <Statistic
                title="Healthy Connections"
                value={healthAnalysis.healthyConnections}
                suffix={`/ ${endpoints.length}`}
                valueStyle={{ color: '#52c41a' }}
                prefix={<CheckCircleOutlined />}
              />
            </Card>
          </Col>
          
          <Col span={6}>
            <Card size="small">
              <Statistic
                title="Avg Latency"
                value={healthAnalysis.averageLatency}
                precision={0}
                suffix="ms"
                valueStyle={{ 
                  color: healthAnalysis.averageLatency > performanceThresholds.latencyWarning ? '#ff4d4f' : '#52c41a' 
                }}
                prefix={<ThunderboltOutlined />}
              />
            </Card>
          </Col>
          
          <Col span={6}>
            <Card size="small">
              <Statistic
                title="Stability Score"
                value={healthAnalysis.stabilityScore}
                precision={1}
                suffix="%"
                valueStyle={{ color: getHealthColor(healthAnalysis.stabilityScore) }}
                prefix={<DashboardOutlined />}
              />
            </Card>
          </Col>
        </Row>

        {/* Health Alerts */}
        {healthAnalysis.recommendations.length > 0 && (
          <Alert
            message="Health Recommendations"
            description={
              <List
                size="small"
                dataSource={healthAnalysis.recommendations}
                renderItem={(item, index) => (
                  <List.Item key={index}>
                    <WarningOutlined style={{ color: '#faad14', marginRight: 8 }} />
                    {item}
                  </List.Item>
                )}
              />
            }
            type="warning"
            showIcon
          />
        )}

        {/* Health Trends and Distribution */}
        <Row gutter={[16, 16]}>
          <Col span={16}>
            <Card 
              title="Health Trends" 
              size="small"
              extra={
                <Space>
                  <Select
                    value={selectedEndpoint}
                    onChange={setSelectedEndpoint}
                    style={{ width: 150 }}
                    size="small"
                  >
                    <Option value="all">All Endpoints</Option>
                    {endpoints.map(ep => (
                      <Option key={ep.id} value={ep.id}>{ep.name}</Option>
                    ))}
                  </Select>
                  <Switch
                    checkedChildren="Predictive"
                    unCheckedChildren="Current"
                    checked={enablePredictiveAlerts}
                    onChange={setEnablePredictiveAlerts}
                    size="small"
                  />
                </Space>
              }
            >
              <ResponsiveContainer width="100%" height={250}>
                <LineChart data={healthTrendData}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="time" />
                  <YAxis yAxisId="left" domain={[0, 100]} />
                  <YAxis yAxisId="right" orientation="right" />
                  <RechartsTooltip />
                  <Legend />
                  <Line
                    yAxisId="left"
                    type="monotone"
                    dataKey="health"
                    stroke="#52c41a"
                    name="Health (%)"
                    strokeWidth={2}
                  />
                  <Line
                    yAxisId="right"
                    type="monotone"
                    dataKey="latency"
                    stroke="#ff4d4f"
                    name="Latency (ms)"
                    strokeWidth={2}
                  />
                  <Line
                    yAxisId="left"
                    type="monotone"
                    dataKey="connections"
                    stroke="#1890ff"
                    name="Connection Rate (%)"
                    strokeWidth={2}
                  />
                  <ReferenceLine 
                    yAxisId="left"
                    y={alertThreshold} 
                    stroke="#faad14" 
                    strokeDasharray="5 5" 
                    label="Alert Threshold"
                  />
                </LineChart>
              </ResponsiveContainer>
            </Card>
          </Col>
          
          <Col span={8}>
            <Card title="Health Distribution" size="small">
              <ResponsiveContainer width="100%" height={250}>
                <PieChart>
                  <Pie
                    data={healthDistributionData}
                    dataKey="value"
                    nameKey="name"
                    cx="50%"
                    cy="50%"
                    outerRadius={80}
                    label={({ name, percent }) => `${name}: ${(percent * 100).toFixed(0)}%`}
                  >
                    {healthDistributionData.map((entry, index) => (
                      <Cell key={`cell-${index}`} fill={entry.color} />
                    ))}
                  </Pie>
                  <RechartsTooltip />
                </PieChart>
              </ResponsiveContainer>
              
              <Divider />
              
              <Row gutter={16}>
                <Col span={24}>
                  <Space direction="vertical" style={{ width: '100%' }}>
                    <div>
                      <Text strong style={{ color: '#52c41a' }}>Healthy: </Text>
                      <Text>{healthAnalysis.healthyConnections}</Text>
                    </div>
                    <div>
                      <Text strong style={{ color: '#faad14' }}>Degraded: </Text>
                      <Text>{healthAnalysis.degradedConnections}</Text>
                    </div>
                    <div>
                      <Text strong style={{ color: '#ff4d4f' }}>Critical: </Text>
                      <Text>{healthAnalysis.criticalConnections}</Text>
                    </div>
                  </Space>
                </Col>
              </Row>
            </Card>
          </Col>
        </Row>

        {/* Detailed Endpoint Health Table */}
        <Card 
          title="Endpoint Health Details" 
          size="small"
          extra={
            <Space>
              <Text type="secondary">Alert Threshold:</Text>
              <Select
                value={alertThreshold}
                onChange={setAlertThreshold}
                size="small"
                style={{ width: 80 }}
              >
                <Option value={70}>70%</Option>
                <Option value={80}>80%</Option>
                <Option value={90}>90%</Option>
              </Select>
              <Button
                type="text"
                icon={<ReloadOutlined />}
                size="small"
                onClick={() => window.location.reload()}
              />
            </Space>
          }
        >
          <Table
            dataSource={endpoints}
            columns={healthColumns}
            rowKey="id"
            size="small"
            pagination={false}
            rowClassName={(record) => {
              const health = calculateConnectionHealth(record);
              if (health < 60) return 'health-critical';
              if (health < 80) return 'health-warning';
              return 'health-good';
            }}
          />
        </Card>

        {/* Health Detail Drawer */}
        <Drawer
          title={`Connection Health Details`}
          placement="right"
          onClose={() => setShowHealthDetail(false)}
          open={showHealthDetail}
          width={600}
        >
          {detailEndpointId && (() => {
            const endpoint = endpoints.find(ep => ep.id === detailEndpointId);
            if (!endpoint) return null;

            const health = calculateConnectionHealth(endpoint);
            
            return (
              <Space direction="vertical" size="large" style={{ width: '100%' }}>
                <Card size="small">
                  <Row gutter={16}>
                    <Col span={12}>
                      <Statistic
                        title="Health Score"
                        value={health}
                        precision={1}
                        suffix="%"
                        valueStyle={{ color: getHealthColor(health) }}
                      />
                    </Col>
                    <Col span={12}>
                      <div>
                        <Text strong>Status: </Text>
                        {getStatusBadge(endpoint.status)}
                      </div>
                    </Col>
                  </Row>
                </Card>

                <Card title="Performance Metrics" size="small">
                  <Row gutter={[16, 16]}>
                    <Col span={8}>
                      <Statistic
                        title="Latency"
                        value={endpoint.latency}
                        suffix="ms"
                        valueStyle={{ 
                          color: endpoint.latency > performanceThresholds.latencyWarning ? '#ff4d4f' : '#52c41a' 
                        }}
                      />
                    </Col>
                    <Col span={8}>
                      <Statistic
                        title="Throughput"
                        value={endpoint.messagesPerSecond}
                        suffix="msg/s"
                      />
                    </Col>
                    <Col span={8}>
                      <Statistic
                        title="Quality"
                        value={endpoint.quality}
                        suffix="%"
                        valueStyle={{ color: getHealthColor(endpoint.quality) }}
                      />
                    </Col>
                  </Row>
                </Card>

                <Card title="Connection History" size="small">
                  <Timeline>
                    <Timeline.Item color="green">
                      <Text>Connected at {new Date(Date.now() - endpoint.uptime).toLocaleTimeString()}</Text>
                    </Timeline.Item>
                    {endpoint.reconnectCount && endpoint.reconnectCount > 0 && (
                      <Timeline.Item color="orange">
                        <Text>Reconnected {endpoint.reconnectCount} times</Text>
                      </Timeline.Item>
                    )}
                    {endpoint.errorCount > 0 && (
                      <Timeline.Item color="red">
                        <Text>{endpoint.errorCount} errors recorded</Text>
                      </Timeline.Item>
                    )}
                    <Timeline.Item color="blue">
                      <Text>Last activity: {new Date(endpoint.lastActivity).toLocaleTimeString()}</Text>
                    </Timeline.Item>
                  </Timeline>
                </Card>
              </Space>
            );
          })()}
        </Drawer>
      </Space>

      <style jsx>{`
        .health-critical {
          background-color: #fff2f0 !important;
        }
        .health-warning {
          background-color: #fffbe6 !important;
        }
        .health-good {
          background-color: #f6ffed !important;
        }
      `}</style>
    </div>
  );
};

export default ConnectionHealthDashboard;