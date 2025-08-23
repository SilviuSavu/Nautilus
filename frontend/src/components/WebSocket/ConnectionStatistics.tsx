/**
 * Connection Statistics Component
 * Sprint 3: Advanced WebSocket Infrastructure
 * 
 * Displays comprehensive WebSocket performance metrics, connection health statistics,
 * and real-time monitoring data with interactive charts and analytics.
 */

import React, { useState, useEffect, useMemo } from 'react';
import { 
  Card, 
  Row, 
  Col, 
  Statistic, 
  Progress, 
  Typography, 
  Space, 
  Badge, 
  Tooltip,
  Select,
  Switch,
  Divider,
  Alert,
  Table
} from 'antd';
import { 
  LineChart, 
  Line, 
  XAxis, 
  YAxis, 
  CartesianGrid, 
  Tooltip as RechartsTooltip, 
  ResponsiveContainer,
  AreaChart,
  Area,
  BarChart,
  Bar,
  PieChart,
  Pie,
  Cell
} from 'recharts';
import { 
  ThunderboltOutlined, 
  CloudServerOutlined,
  BarChartOutlined,
  DashboardOutlined,
  WarningOutlined,
  CheckCircleOutlined
} from '@ant-design/icons';
import { useConnectionHealth } from '../../hooks/useConnectionHealth';

const { Title, Text } = Typography;
const { Option } = Select;

interface ConnectionInfo {
  connectedAt: string;
  lastActivity: string;
  protocolVersion: string;
  sessionId: string;
  uptime: number;
  messagesSent: number;
  messagesReceived: number;
  bytesTransferred: number;
  errorCount: number;
  latencyHistory: number[];
  throughputHistory: Array<{
    timestamp: number;
    messagesPerSecond: number;
    bytesPerSecond: number;
  }>;
}

interface ConnectionStatisticsProps {
  connectionInfo: ConnectionInfo;
  className?: string;
  showAdvancedMetrics?: boolean;
  updateInterval?: number;
  chartHeight?: number;
}

export const ConnectionStatistics: React.FC<ConnectionStatisticsProps> = ({
  connectionInfo,
  className,
  showAdvancedMetrics = true,
  updateInterval = 1000,
  chartHeight = 200
}) => {
  const {
    connectionHealth,
    qualityScore,
    stabilityMetrics,
    performanceMetrics,
    alertSummary
  } = useConnectionHealth();

  const [selectedTimeRange, setSelectedTimeRange] = useState<string>('5m');
  const [showRealtime, setShowRealtime] = useState(true);
  const [selectedMetric, setSelectedMetric] = useState<string>('latency');

  // Calculate derived statistics
  const statistics = useMemo(() => {
    const uptime = connectionInfo.uptime || 0;
    const totalMessages = connectionInfo.messagesSent + connectionInfo.messagesReceived;
    const avgLatency = connectionInfo.latencyHistory?.length > 0 
      ? connectionInfo.latencyHistory.reduce((a, b) => a + b, 0) / connectionInfo.latencyHistory.length 
      : 0;
    
    const errorRate = totalMessages > 0 ? (connectionInfo.errorCount / totalMessages) * 100 : 0;
    const throughput = uptime > 0 ? totalMessages / (uptime / 1000) : 0;
    
    return {
      uptime: Math.floor(uptime / 1000), // Convert to seconds
      totalMessages,
      avgLatency,
      errorRate,
      throughput,
      dataTransferred: connectionInfo.bytesTransferred || 0,
    };
  }, [connectionInfo]);

  // Format uptime display
  const formatUptime = (seconds: number): string => {
    if (seconds < 60) return `${seconds}s`;
    if (seconds < 3600) return `${Math.floor(seconds / 60)}m ${seconds % 60}s`;
    const hours = Math.floor(seconds / 3600);
    const minutes = Math.floor((seconds % 3600) / 60);
    return `${hours}h ${minutes}m`;
  };

  // Format bytes
  const formatBytes = (bytes: number): string => {
    if (bytes === 0) return '0 B';
    const k = 1024;
    const sizes = ['B', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
  };

  // Get connection quality color
  const getQualityColor = (score: number) => {
    if (score >= 90) return '#52c41a';
    if (score >= 70) return '#faad14';
    if (score >= 50) return '#fa8c16';
    return '#ff4d4f';
  };

  // Prepare chart data
  const latencyChartData = connectionInfo.latencyHistory?.slice(-50).map((latency, index) => ({
    time: index,
    latency,
  })) || [];

  const throughputChartData = connectionInfo.throughputHistory?.slice(-50).map((point, index) => ({
    time: new Date(point.timestamp).toLocaleTimeString(),
    messages: point.messagesPerSecond,
    bytes: point.bytesPerSecond / 1024, // Convert to KB/s
  })) || [];

  // Message type distribution data
  const messageTypeData = [
    { name: 'Market Data', value: 45, color: '#1890ff' },
    { name: 'Trade Updates', value: 20, color: '#52c41a' },
    { name: 'Risk Alerts', value: 10, color: '#ff4d4f' },
    { name: 'Engine Status', value: 15, color: '#722ed1' },
    { name: 'Other', value: 10, color: '#fa8c16' }
  ];

  // Performance thresholds
  const performanceThresholds = {
    latency: { excellent: 50, good: 100, poor: 500 },
    throughput: { excellent: 100, good: 50, poor: 10 },
    errorRate: { excellent: 1, good: 5, poor: 10 }
  };

  const getPerformanceStatus = (metric: string, value: number) => {
    const thresholds = performanceThresholds[metric as keyof typeof performanceThresholds];
    if (!thresholds) return 'unknown';
    
    if (metric === 'errorRate') {
      if (value <= thresholds.excellent) return 'excellent';
      if (value <= thresholds.good) return 'good';
      return 'poor';
    } else {
      if (value >= thresholds.excellent) return 'excellent';
      if (value >= thresholds.good) return 'good';
      return 'poor';
    }
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'excellent': return '#52c41a';
      case 'good': return '#faad14';
      case 'poor': return '#ff4d4f';
      default: return '#d9d9d9';
    }
  };

  return (
    <div className={className}>
      <Space direction="vertical" size="large" style={{ width: '100%' }}>
        {/* Key Metrics Overview */}
        <Row gutter={[16, 16]}>
          <Col span={6}>
            <Statistic
              title="Connection Quality"
              value={qualityScore || 85}
              precision={0}
              suffix="%"
              valueStyle={{ color: getQualityColor(qualityScore || 85) }}
              prefix={<DashboardOutlined />}
            />
            <Progress 
              percent={qualityScore || 85} 
              strokeColor={getQualityColor(qualityScore || 85)}
              size="small"
              showInfo={false}
            />
          </Col>
          
          <Col span={6}>
            <Statistic
              title="Avg Latency"
              value={statistics.avgLatency}
              precision={0}
              suffix="ms"
              valueStyle={{ color: getStatusColor(getPerformanceStatus('latency', statistics.avgLatency)) }}
              prefix={<ThunderboltOutlined />}
            />
          </Col>
          
          <Col span={6}>
            <Statistic
              title="Throughput"
              value={statistics.throughput}
              precision={1}
              suffix="msg/s"
              valueStyle={{ color: getStatusColor(getPerformanceStatus('throughput', statistics.throughput)) }}
              prefix={<BarChartOutlined />}
            />
          </Col>
          
          <Col span={6}>
            <Statistic
              title="Uptime"
              value={formatUptime(statistics.uptime)}
              valueStyle={{ color: '#52c41a' }}
              prefix={<CloudServerOutlined />}
            />
          </Col>
        </Row>

        {/* Connection Health Alerts */}
        {alertSummary && alertSummary.length > 0 && (
          <Alert
            message="Connection Health Alerts"
            description={
              <Space direction="vertical" size="small">
                {alertSummary.map((alert: any, index: number) => (
                  <Text key={index} type={alert.severity === 'high' ? 'danger' : 'warning'}>
                    <WarningOutlined /> {alert.message}
                  </Text>
                ))}
              </Space>
            }
            type="warning"
            showIcon
            style={{ marginBottom: 16 }}
          />
        )}

        {/* Detailed Statistics */}
        <Row gutter={[16, 16]}>
          <Col span={8}>
            <Card size="small" title="Message Statistics">
              <Space direction="vertical" size="small" style={{ width: '100%' }}>
                <Row justify="space-between">
                  <Text>Total Messages:</Text>
                  <Text strong>{statistics.totalMessages.toLocaleString()}</Text>
                </Row>
                <Row justify="space-between">
                  <Text>Messages Sent:</Text>
                  <Text>{connectionInfo.messagesSent.toLocaleString()}</Text>
                </Row>
                <Row justify="space-between">
                  <Text>Messages Received:</Text>
                  <Text>{connectionInfo.messagesReceived.toLocaleString()}</Text>
                </Row>
                <Row justify="space-between">
                  <Text>Error Count:</Text>
                  <Text type={connectionInfo.errorCount > 0 ? 'danger' : 'success'}>
                    {connectionInfo.errorCount}
                  </Text>
                </Row>
                <Row justify="space-between">
                  <Text>Error Rate:</Text>
                  <Text type={statistics.errorRate > 5 ? 'danger' : 'success'}>
                    {statistics.errorRate.toFixed(2)}%
                  </Text>
                </Row>
              </Space>
            </Card>
          </Col>

          <Col span={8}>
            <Card size="small" title="Performance Metrics">
              <Space direction="vertical" size="small" style={{ width: '100%' }}>
                <Row justify="space-between">
                  <Text>Data Transferred:</Text>
                  <Text strong>{formatBytes(statistics.dataTransferred)}</Text>
                </Row>
                <Row justify="space-between">
                  <Text>Protocol Version:</Text>
                  <Text>{connectionInfo.protocolVersion || '2.0'}</Text>
                </Row>
                <Row justify="space-between">
                  <Text>Session ID:</Text>
                  <Text style={{ fontSize: '12px', fontFamily: 'monospace' }}>
                    {connectionInfo.sessionId?.substring(0, 12)}...
                  </Text>
                </Row>
                <Row justify="space-between">
                  <Text>Connected Since:</Text>
                  <Text style={{ fontSize: '12px' }}>
                    {connectionInfo.connectedAt ? 
                      new Date(connectionInfo.connectedAt).toLocaleTimeString() : 
                      'Unknown'
                    }
                  </Text>
                </Row>
                <Row justify="space-between">
                  <Text>Last Activity:</Text>
                  <Text style={{ fontSize: '12px' }}>
                    {connectionInfo.lastActivity ? 
                      new Date(connectionInfo.lastActivity).toLocaleTimeString() : 
                      'Unknown'
                    }
                  </Text>
                </Row>
              </Space>
            </Card>
          </Col>

          <Col span={8}>
            <Card size="small" title="Connection Health">
              {connectionHealth && (
                <Space direction="vertical" size="small" style={{ width: '100%' }}>
                  <Row justify="space-between">
                    <Text>Connection Status:</Text>
                    <Badge 
                      status={connectionHealth.isHealthy ? 'success' : 'error'} 
                      text={connectionHealth.isHealthy ? 'Healthy' : 'Degraded'} 
                    />
                  </Row>
                  <Row justify="space-between">
                    <Text>Stability Score:</Text>
                    <Text strong style={{ color: getQualityColor(connectionHealth.stabilityScore || 0) }}>
                      {connectionHealth.stabilityScore || 0}%
                    </Text>
                  </Row>
                  <Row justify="space-between">
                    <Text>Reconnections:</Text>
                    <Text>{connectionHealth.reconnectCount || 0}</Text>
                  </Row>
                  <Row justify="space-between">
                    <Text>Packet Loss:</Text>
                    <Text type={connectionHealth.packetLoss > 1 ? 'danger' : 'success'}>
                      {connectionHealth.packetLoss || 0}%
                    </Text>
                  </Row>
                </Space>
              )}
            </Card>
          </Col>
        </Row>

        {/* Performance Charts */}
        {showAdvancedMetrics && (
          <>
            <Divider />
            
            <Row gutter={[16, 16]}>
              <Col span={24}>
                <Card 
                  title="Performance Charts" 
                  size="small"
                  extra={
                    <Space>
                      <Select 
                        value={selectedMetric} 
                        onChange={setSelectedMetric}
                        size="small"
                        style={{ width: 120 }}
                      >
                        <Option value="latency">Latency</Option>
                        <Option value="throughput">Throughput</Option>
                      </Select>
                      
                      <Switch 
                        checked={showRealtime}
                        onChange={setShowRealtime}
                        size="small"
                      />
                      <Text style={{ fontSize: '12px' }}>Real-time</Text>
                    </Space>
                  }
                >
                  <Row gutter={[16, 16]}>
                    <Col span={selectedMetric === 'latency' ? 24 : 12}>
                      {selectedMetric === 'latency' && (
                        <div>
                          <Title level={5}>Latency History</Title>
                          <ResponsiveContainer width="100%" height={chartHeight}>
                            <AreaChart data={latencyChartData}>
                              <CartesianGrid strokeDasharray="3 3" />
                              <XAxis dataKey="time" />
                              <YAxis />
                              <RechartsTooltip formatter={(value) => [`${value}ms`, 'Latency']} />
                              <Area 
                                type="monotone" 
                                dataKey="latency" 
                                stroke="#1890ff" 
                                fill="#1890ff"
                                fillOpacity={0.3}
                              />
                            </AreaChart>
                          </ResponsiveContainer>
                        </div>
                      )}
                      
                      {selectedMetric === 'throughput' && (
                        <div>
                          <Title level={5}>Throughput History</Title>
                          <ResponsiveContainer width="100%" height={chartHeight}>
                            <BarChart data={throughputChartData}>
                              <CartesianGrid strokeDasharray="3 3" />
                              <XAxis dataKey="time" />
                              <YAxis />
                              <RechartsTooltip />
                              <Bar dataKey="messages" fill="#52c41a" name="Messages/sec" />
                              <Bar dataKey="bytes" fill="#1890ff" name="KB/sec" />
                            </BarChart>
                          </ResponsiveContainer>
                        </div>
                      )}
                    </Col>

                    {selectedMetric === 'throughput' && (
                      <Col span={12}>
                        <Title level={5}>Message Type Distribution</Title>
                        <ResponsiveContainer width="100%" height={chartHeight}>
                          <PieChart>
                            <Pie
                              data={messageTypeData}
                              dataKey="value"
                              nameKey="name"
                              cx="50%"
                              cy="50%"
                              outerRadius={80}
                              label={({ name, percent }) => `${name} ${(percent * 100).toFixed(0)}%`}
                            >
                              {messageTypeData.map((entry, index) => (
                                <Cell key={`cell-${index}`} fill={entry.color} />
                              ))}
                            </Pie>
                            <RechartsTooltip />
                          </PieChart>
                        </ResponsiveContainer>
                      </Col>
                    )}
                  </Row>
                </Card>
              </Col>
            </Row>
          </>
        )}
      </Space>
    </div>
  );
};

export default ConnectionStatistics;