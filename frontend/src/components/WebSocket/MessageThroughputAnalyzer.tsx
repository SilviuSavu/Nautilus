/**
 * Message Throughput Analyzer Component
 * Sprint 3: Advanced WebSocket Infrastructure
 * 
 * Real-time message analysis with throughput monitoring, message type classification,
 * bandwidth analysis, and performance optimization insights.
 */

import React, { useState, useEffect, useMemo, useCallback } from 'react';
import {
  Card,
  Row,
  Col,
  Typography,
  Space,
  Table,
  Tag,
  Progress,
  Statistic,
  Select,
  Switch,
  Button,
  Tooltip,
  Alert,
  Divider,
  Badge,
  Tabs,
  List,
  Avatar,
  TreeSelect,
  InputNumber,
  Modal,
  notification
} from 'antd';
import {
  BarChartOutlined,
  LineChartOutlined,
  ThunderboltOutlined,
  CloudDownloadOutlined,
  CloudUploadOutlined,
  MessageOutlined,
  DashboardOutlined,
  FilterOutlined,
  SettingOutlined,
  ExportOutlined,
  ReloadOutlined,
  WarningOutlined,
  CheckCircleOutlined,
  ClockCircleOutlined,
  ApiOutlined,
  FunnelPlotOutlined,
  ArrowUpOutlined,
  ArrowDownOutlined
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
  ComposedChart,
  PieChart,
  Pie,
  Cell,
  Treemap
} from 'recharts';

const { Title, Text } = Typography;
const { Option } = Select;
const { TabPane } = Tabs;

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

interface MessageMetrics {
  timestamp: number;
  messageType: string;
  endpoint: string;
  size: number;
  latency: number;
  direction: 'inbound' | 'outbound';
  priority: number;
  processed: boolean;
  errorCount: number;
}

interface ThroughputData {
  timestamp: number;
  time: string;
  totalMessages: number;
  inboundMessages: number;
  outboundMessages: number;
  totalBandwidth: number; // bytes per second
  inboundBandwidth: number;
  outboundBandwidth: number;
  averageLatency: number;
  errorRate: number;
  endpointBreakdown: Record<string, number>;
  messageTypeBreakdown: Record<string, number>;
}

interface MessageThroughputAnalyzerProps {
  endpoints: WebSocketEndpoint[];
  historicalData: any[];
  timeRange: string;
  showRealtime?: boolean;
  className?: string;
  maxDataPoints?: number;
  enablePrediction?: boolean;
  alertThresholds?: {
    throughputMin: number;
    throughputMax: number;
    bandwidthMax: number;
    latencyMax: number;
    errorRateMax: number;
  };
}

interface MessageTypeStats {
  type: string;
  count: number;
  totalSize: number;
  averageSize: number;
  averageLatency: number;
  errorCount: number;
  percentage: number;
  trend: 'up' | 'down' | 'stable';
}

export const MessageThroughputAnalyzer: React.FC<MessageThroughputAnalyzerProps> = ({
  endpoints,
  historicalData,
  timeRange,
  showRealtime = true,
  className,
  maxDataPoints = 200,
  enablePrediction = true,
  alertThresholds = {
    throughputMin: 10,
    throughputMax: 50000,
    bandwidthMax: 10485760, // 10 MB/s
    latencyMax: 1000,
    errorRateMax: 5
  }
}) => {
  const [activeTab, setActiveTab] = useState<string>('throughput');
  const [selectedEndpoints, setSelectedEndpoints] = useState<string[]>([]);
  const [messageTypeFilter, setMessageTypeFilter] = useState<string[]>([]);
  const [timeWindow, setTimeWindow] = useState<number>(60); // seconds
  const [throughputData, setThroughputData] = useState<ThroughputData[]>([]);
  const [messageStats, setMessageStats] = useState<MessageMetrics[]>([]);
  const [showBandwidth, setShowBandwidth] = useState<boolean>(true);
  const [showPrediction, setShowPrediction] = useState<boolean>(true);
  const [alertsEnabled, setAlertsEnabled] = useState<boolean>(true);
  const [detailModalVisible, setDetailModalVisible] = useState<boolean>(false);
  const [selectedTimePoint, setSelectedTimePoint] = useState<ThroughputData | null>(null);

  // Mock message types for realistic data
  const messageTypes = [
    'market_data',
    'trade_update',
    'risk_alert',
    'engine_status',
    'heartbeat',
    'subscription',
    'error',
    'system_health',
    'performance_metric',
    'user_action'
  ];

  // Generate realistic throughput data
  useEffect(() => {
    const generateThroughputData = () => {
      const now = Date.now();
      const totalMessages = endpoints.reduce((sum, ep) => sum + ep.messagesPerSecond, 0);
      
      // Simulate message distribution
      const inboundRatio = 0.7; // 70% inbound
      const inboundMessages = Math.floor(totalMessages * inboundRatio);
      const outboundMessages = totalMessages - inboundMessages;
      
      // Simulate bandwidth (average 1KB per message with variation)
      const avgMessageSize = 800 + Math.random() * 400; // 800-1200 bytes
      const totalBandwidth = totalMessages * avgMessageSize;
      const inboundBandwidth = inboundMessages * avgMessageSize;
      const outboundBandwidth = outboundMessages * avgMessageSize;
      
      // Calculate metrics
      const averageLatency = endpoints.reduce((sum, ep) => sum + ep.latency, 0) / Math.max(endpoints.length, 1);
      const errorRate = endpoints.reduce((sum, ep) => sum + ep.errorCount, 0) / Math.max(totalMessages, 1) * 100;
      
      // Endpoint breakdown
      const endpointBreakdown: Record<string, number> = {};
      endpoints.forEach(ep => {
        endpointBreakdown[ep.id] = ep.messagesPerSecond;
      });
      
      // Message type breakdown (simulated distribution)
      const messageTypeBreakdown: Record<string, number> = {};
      const distributions = {
        market_data: 0.4,
        trade_update: 0.15,
        risk_alert: 0.05,
        engine_status: 0.1,
        heartbeat: 0.1,
        subscription: 0.05,
        error: 0.02,
        system_health: 0.08,
        performance_metric: 0.03,
        user_action: 0.02
      };
      
      Object.entries(distributions).forEach(([type, ratio]) => {
        messageTypeBreakdown[type] = Math.floor(totalMessages * ratio);
      });

      const dataPoint: ThroughputData = {
        timestamp: now,
        time: new Date(now).toLocaleTimeString(),
        totalMessages,
        inboundMessages,
        outboundMessages,
        totalBandwidth,
        inboundBandwidth,
        outboundBandwidth,
        averageLatency,
        errorRate,
        endpointBreakdown,
        messageTypeBreakdown
      };

      setThroughputData(prev => {
        const newData = [...prev, dataPoint];
        return newData.slice(-maxDataPoints);
      });
    };

    if (showRealtime && endpoints.length > 0) {
      generateThroughputData();
      const interval = setInterval(generateThroughputData, 1000);
      return () => clearInterval(interval);
    }
  }, [showRealtime, endpoints, maxDataPoints]);

  // Process message type statistics
  const messageTypeStats: MessageTypeStats[] = useMemo(() => {
    if (throughputData.length === 0) return [];
    
    const latestData = throughputData[throughputData.length - 1];
    const totalMessages = latestData.totalMessages;
    
    const stats: MessageTypeStats[] = [];
    
    Object.entries(latestData.messageTypeBreakdown).forEach(([type, count]) => {
      // Calculate trend based on last 5 data points
      let trend: 'up' | 'down' | 'stable' = 'stable';
      if (throughputData.length >= 5) {
        const recentData = throughputData.slice(-5);
        const oldCount = recentData[0].messageTypeBreakdown[type] || 0;
        const newCount = count;
        const change = (newCount - oldCount) / Math.max(oldCount, 1);
        
        if (change > 0.1) trend = 'up';
        else if (change < -0.1) trend = 'down';
      }
      
      stats.push({
        type,
        count,
        totalSize: count * (800 + Math.random() * 400), // Simulate size
        averageSize: 800 + Math.random() * 400,
        averageLatency: 20 + Math.random() * 80,
        errorCount: Math.floor(count * 0.01), // 1% error rate
        percentage: (count / Math.max(totalMessages, 1)) * 100,
        trend
      });
    });
    
    return stats.sort((a, b) => b.count - a.count);
  }, [throughputData]);

  // Calculate performance insights
  const performanceInsights = useMemo(() => {
    if (throughputData.length < 10) return [];
    
    const insights: string[] = [];
    const recent = throughputData.slice(-10);
    const latest = recent[recent.length - 1];
    
    // Throughput analysis
    const avgThroughput = recent.reduce((sum, d) => sum + d.totalMessages, 0) / recent.length;
    if (latest.totalMessages > avgThroughput * 1.5) {
      insights.push('High message throughput detected - system under heavy load');
    }
    if (latest.totalMessages < alertThresholds.throughputMin) {
      insights.push('Low message throughput - possible connection issues');
    }
    
    // Bandwidth analysis
    const avgBandwidth = recent.reduce((sum, d) => sum + d.totalBandwidth, 0) / recent.length;
    if (latest.totalBandwidth > alertThresholds.bandwidthMax) {
      insights.push('High bandwidth usage - consider message compression');
    }
    
    // Latency analysis
    const avgLatency = recent.reduce((sum, d) => sum + d.averageLatency, 0) / recent.length;
    if (latest.averageLatency > avgLatency * 2) {
      insights.push('Latency spike detected - check network conditions');
    }
    
    // Error rate analysis
    if (latest.errorRate > alertThresholds.errorRateMax) {
      insights.push('High error rate detected - investigate message processing');
    }
    
    // Imbalance detection
    const inboundRatio = latest.inboundMessages / Math.max(latest.totalMessages, 1);
    if (inboundRatio > 0.9) {
      insights.push('High inbound message ratio - system may be overwhelmed');
    } else if (inboundRatio < 0.3) {
      insights.push('High outbound message ratio - check subscription efficiency');
    }
    
    return insights.slice(0, 3); // Limit to top 3 insights
  }, [throughputData, alertThresholds]);

  // Format data size
  const formatBytes = useCallback((bytes: number): string => {
    if (bytes === 0) return '0 B';
    const k = 1024;
    const sizes = ['B', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return `${parseFloat((bytes / Math.pow(k, i)).toFixed(1))} ${sizes[i]}`;
  }, []);

  // Get trend icon
  const getTrendIcon = (trend: 'up' | 'down' | 'stable') => {
    switch (trend) {
      case 'up': return <ArrowUpOutlined style={{ color: '#52c41a' }} />;
      case 'down': return <ArrowDownOutlined style={{ color: '#ff4d4f' }} />;
      case 'stable': return <LineChartOutlined style={{ color: '#1890ff' }} />;
    }
  };

  // Table columns for message type stats
  const messageTypeColumns = [
    {
      title: 'Message Type',
      dataIndex: 'type',
      key: 'type',
      render: (type: string, record: MessageTypeStats) => (
        <Space>
          <Avatar size="small" icon={<MessageOutlined />} />
          <div>
            <Text strong>{type.replace('_', ' ').toUpperCase()}</Text>
            <br />
            <Text type="secondary" style={{ fontSize: '11px' }}>
              {record.percentage.toFixed(1)}% of total
            </Text>
          </div>
        </Space>
      )
    },
    {
      title: 'Count',
      dataIndex: 'count',
      key: 'count',
      render: (count: number, record: MessageTypeStats) => (
        <Space direction="vertical" size="small">
          <Text strong>{count.toLocaleString()}</Text>
          <Progress
            percent={(count / Math.max(messageTypeStats[0]?.count || 1, 1)) * 100}
            size="small"
            showInfo={false}
            strokeColor="#1890ff"
          />
        </Space>
      ),
      sorter: (a: MessageTypeStats, b: MessageTypeStats) => a.count - b.count
    },
    {
      title: 'Size',
      key: 'size',
      render: (record: MessageTypeStats) => (
        <Space direction="vertical" size="small">
          <Text>{formatBytes(record.totalSize)}</Text>
          <Text type="secondary" style={{ fontSize: '11px' }}>
            Avg: {formatBytes(record.averageSize)}
          </Text>
        </Space>
      )
    },
    {
      title: 'Latency',
      dataIndex: 'averageLatency',
      key: 'latency',
      render: (latency: number) => (
        <Text style={{ 
          color: latency > alertThresholds.latencyMax ? '#ff4d4f' : 
                latency > alertThresholds.latencyMax * 0.7 ? '#faad14' : '#52c41a' 
        }}>
          {latency.toFixed(0)}ms
        </Text>
      ),
      sorter: (a: MessageTypeStats, b: MessageTypeStats) => a.averageLatency - b.averageLatency
    },
    {
      title: 'Errors',
      dataIndex: 'errorCount',
      key: 'errors',
      render: (count: number) => (
        <Badge 
          count={count} 
          showZero 
          style={{ backgroundColor: count > 0 ? '#ff4d4f' : '#52c41a' }} 
        />
      )
    },
    {
      title: 'Trend',
      dataIndex: 'trend',
      key: 'trend',
      render: (trend: 'up' | 'down' | 'stable') => (
        <Tooltip title={`Trend: ${trend}`}>
          {getTrendIcon(trend)}
        </Tooltip>
      )
    }
  ];

  // Chart data preparation
  const chartData = throughputData.slice(-50).map((point, index) => ({
    ...point,
    index,
    bandwidthMB: point.totalBandwidth / 1048576 // Convert to MB
  }));

  // Message type pie chart data
  const messageTypePieData = messageTypeStats.slice(0, 6).map(stat => ({
    name: stat.type.replace('_', ' ').toUpperCase(),
    value: stat.count,
    percentage: stat.percentage
  }));

  return (
    <div className={className}>
      <Space direction="vertical" size="large" style={{ width: '100%' }}>
        {/* Performance Insights */}
        {performanceInsights.length > 0 && (
          <Alert
            message="Throughput Analysis Insights"
            description={
              <List
                size="small"
                dataSource={performanceInsights}
                renderItem={(item, index) => (
                  <List.Item key={index}>
                    <WarningOutlined style={{ color: '#faad14', marginRight: 8 }} />
                    {item}
                  </List.Item>
                )}
              />
            }
            type="info"
            showIcon
          />
        )}

        {/* Key Metrics */}
        <Row gutter={[16, 16]}>
          <Col span={6}>
            <Card size="small">
              <Statistic
                title="Total Throughput"
                value={throughputData.length > 0 ? throughputData[throughputData.length - 1].totalMessages : 0}
                suffix="msg/s"
                valueStyle={{ color: '#1890ff' }}
                prefix={<BarChartOutlined />}
              />
            </Card>
          </Col>
          
          <Col span={6}>
            <Card size="small">
              <Statistic
                title="Bandwidth Usage"
                value={throughputData.length > 0 ? 
                  formatBytes(throughputData[throughputData.length - 1].totalBandwidth) : '0 B'}
                suffix="/s"
                valueStyle={{ color: '#52c41a' }}
                prefix={<CloudDownloadOutlined />}
              />
            </Card>
          </Col>
          
          <Col span={6}>
            <Card size="small">
              <Statistic
                title="Avg Latency"
                value={throughputData.length > 0 ? 
                  throughputData[throughputData.length - 1].averageLatency : 0}
                precision={0}
                suffix="ms"
                valueStyle={{ 
                  color: throughputData.length > 0 && 
                         throughputData[throughputData.length - 1].averageLatency > alertThresholds.latencyMax 
                         ? '#ff4d4f' : '#52c41a' 
                }}
                prefix={<ThunderboltOutlined />}
              />
            </Card>
          </Col>
          
          <Col span={6}>
            <Card size="small">
              <Statistic
                title="Error Rate"
                value={throughputData.length > 0 ? 
                  throughputData[throughputData.length - 1].errorRate : 0}
                precision={2}
                suffix="%"
                valueStyle={{ 
                  color: throughputData.length > 0 && 
                         throughputData[throughputData.length - 1].errorRate > alertThresholds.errorRateMax 
                         ? '#ff4d4f' : '#52c41a' 
                }}
                prefix={<WarningOutlined />}
              />
            </Card>
          </Col>
        </Row>

        {/* Main Analysis Tabs */}
        <Card
          title={
            <Space>
              <FunnelPlotOutlined />
              <Title level={4} style={{ margin: 0 }}>Message Throughput Analysis</Title>
            </Space>
          }
          extra={
            <Space>
              <Switch
                checkedChildren="Bandwidth"
                unCheckedChildren="Messages"
                checked={showBandwidth}
                onChange={setShowBandwidth}
                size="small"
              />
              <Switch
                checkedChildren="Prediction"
                unCheckedChildren="Current"
                checked={showPrediction && enablePrediction}
                onChange={setShowPrediction}
                size="small"
                disabled={!enablePrediction}
              />
              <Select
                placeholder="Filter endpoints"
                mode="multiple"
                style={{ width: 200 }}
                value={selectedEndpoints}
                onChange={setSelectedEndpoints}
                size="small"
              >
                {endpoints.map(ep => (
                  <Option key={ep.id} value={ep.id}>{ep.name}</Option>
                ))}
              </Select>
              <Button
                type="text"
                icon={<ExportOutlined />}
                size="small"
                onClick={() => {
                  // Export functionality would go here
                  notification.success({
                    message: 'Data Exported',
                    description: 'Throughput data has been exported successfully'
                  });
                }}
              >
                Export
              </Button>
            </Space>
          }
        >
          <Tabs activeKey={activeTab} onChange={setActiveTab}>
            <TabPane tab="Throughput Trends" key="throughput">
              <ResponsiveContainer width="100%" height={400}>
                <ComposedChart data={chartData}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="time" />
                  <YAxis yAxisId="left" />
                  <YAxis yAxisId="right" orientation="right" />
                  <RechartsTooltip 
                    formatter={(value, name) => [
                      name.includes('Bandwidth') ? formatBytes(value as number) + '/s' : 
                      name.includes('Latency') ? `${value}ms` : value,
                      name
                    ]}
                  />
                  <Legend />
                  
                  {showBandwidth ? (
                    <>
                      <Area
                        yAxisId="right"
                        type="monotone"
                        dataKey="bandwidthMB"
                        stackId="1"
                        stroke="#52c41a"
                        fill="#52c41a"
                        fillOpacity={0.3}
                        name="Bandwidth (MB/s)"
                      />
                    </>
                  ) : (
                    <>
                      <Area
                        yAxisId="left"
                        type="monotone"
                        dataKey="inboundMessages"
                        stackId="1"
                        stroke="#1890ff"
                        fill="#1890ff"
                        fillOpacity={0.6}
                        name="Inbound Messages"
                      />
                      <Area
                        yAxisId="left"
                        type="monotone"
                        dataKey="outboundMessages"
                        stackId="1"
                        stroke="#ff4d4f"
                        fill="#ff4d4f"
                        fillOpacity={0.6}
                        name="Outbound Messages"
                      />
                    </>
                  )}
                  
                  <Line
                    yAxisId="right"
                    type="monotone"
                    dataKey="averageLatency"
                    stroke="#faad14"
                    strokeWidth={2}
                    name="Avg Latency (ms)"
                    dot={false}
                  />
                  
                  <ReferenceLine 
                    yAxisId="left"
                    y={alertThresholds.throughputMax} 
                    stroke="#ff4d4f" 
                    strokeDasharray="5 5" 
                    label="Max Throughput"
                  />
                </ComposedChart>
              </ResponsiveContainer>
            </TabPane>

            <TabPane tab="Message Types" key="types">
              <Row gutter={[16, 16]}>
                <Col span={16}>
                  <Table
                    dataSource={messageTypeStats}
                    columns={messageTypeColumns}
                    rowKey="type"
                    size="small"
                    pagination={{ pageSize: 10 }}
                  />
                </Col>
                
                <Col span={8}>
                  <Card title="Message Distribution" size="small">
                    <ResponsiveContainer width="100%" height={300}>
                      <PieChart>
                        <Pie
                          data={messageTypePieData}
                          dataKey="value"
                          nameKey="name"
                          cx="50%"
                          cy="50%"
                          outerRadius={80}
                          label={({ name, percentage }) => `${percentage.toFixed(1)}%`}
                        >
                          {messageTypePieData.map((entry, index) => (
                            <Cell 
                              key={`cell-${index}`} 
                              fill={`hsl(${(index * 137.5) % 360}, 70%, 50%)`} 
                            />
                          ))}
                        </Pie>
                        <RechartsTooltip 
                          formatter={(value, name) => [`${value} messages`, name]}
                        />
                      </PieChart>
                    </ResponsiveContainer>
                  </Card>
                </Col>
              </Row>
            </TabPane>

            <TabPane tab="Bandwidth Analysis" key="bandwidth">
              <ResponsiveContainer width="100%" height={400}>
                <AreaChart data={chartData}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="time" />
                  <YAxis />
                  <RechartsTooltip 
                    formatter={(value) => [formatBytes(value as number * 1048576) + '/s', 'Bandwidth']}
                  />
                  <Legend />
                  <Area
                    type="monotone"
                    dataKey="bandwidthMB"
                    stackId="1"
                    stroke="#1890ff"
                    fill="#1890ff"
                    fillOpacity={0.6}
                    name="Total Bandwidth (MB/s)"
                  />
                  <ReferenceLine 
                    y={alertThresholds.bandwidthMax / 1048576} 
                    stroke="#ff4d4f" 
                    strokeDasharray="5 5" 
                    label="Max Bandwidth"
                  />
                </AreaChart>
              </ResponsiveContainer>
            </TabPane>
          </Tabs>
        </Card>
      </Space>
    </div>
  );
};

export default MessageThroughputAnalyzer;