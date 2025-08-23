/**
 * Real-Time Streaming Dashboard Component
 * Sprint 3: Advanced WebSocket Infrastructure
 * 
 * Comprehensive real-time data streaming dashboard with live data feeds,
 * subscription management, and performance monitoring for all Sprint 3 message types.
 */

import React, { useState, useEffect, useMemo } from 'react';
import { 
  Card, 
  Row, 
  Col, 
  Typography, 
  Space, 
  Tabs, 
  Table, 
  Tag, 
  Badge, 
  Button,
  Switch,
  Tooltip,
  Alert,
  Statistic,
  Progress,
  List,
  Avatar,
  Empty,
  Spin,
  Select,
  Input
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
  Area
} from 'recharts';
import { 
  PlayCircleOutlined,
  PauseCircleOutlined,
  SettingOutlined,
  BarChartOutlined,
  RiseOutlined,
  AlertOutlined,
  DollarOutlined,
  LineChartOutlined,
  UserOutlined,
  DatabaseOutlined,
  ThunderboltOutlined,
  WarningOutlined,
  CheckCircleOutlined,
  InfoCircleOutlined
} from '@ant-design/icons';
import { useRealTimeData } from '../../hooks/useRealTimeData';
import { useWebSocketManager } from '../../hooks/useWebSocketManager';

const { Title, Text } = Typography;
const { TabPane } = Tabs;
const { Option } = Select;

interface RealTimeStreamingProps {
  className?: string;
  defaultActiveTab?: string;
  showPerformanceMetrics?: boolean;
  maxDataPoints?: number;
  updateInterval?: number;
}

interface StreamingData {
  marketData: Array<{
    symbol: string;
    price: number;
    change: number;
    volume: number;
    timestamp: string;
  }>;
  tradeUpdates: Array<{
    id: string;
    symbol: string;
    side: string;
    quantity: number;
    price: number;
    status: string;
    timestamp: string;
  }>;
  riskAlerts: Array<{
    id: string;
    type: string;
    severity: string;
    message: string;
    portfolio: string;
    timestamp: string;
  }>;
  performanceUpdates: Array<{
    portfolio: string;
    pnl: number;
    returns: number;
    sharpe: number;
    maxDrawdown: number;
    timestamp: string;
  }>;
  engineStatus: Array<{
    engine: string;
    state: string;
    uptime: number;
    memory: number;
    cpu: number;
    timestamp: string;
  }>;
}

export const RealTimeStreaming: React.FC<RealTimeStreamingProps> = ({
  className,
  defaultActiveTab = 'market_data',
  showPerformanceMetrics = true,
  maxDataPoints = 100,
  updateInterval = 1000
}) => {
  const {
    marketData,
    tradeUpdates,
    riskAlerts,
    performanceData,
    orderUpdates,
    positionUpdates,
    systemHealth,
    isStreaming,
    startStreaming,
    stopStreaming,
    subscribeToStream,
    unsubscribeFromStream,
    getStreamStatistics
  } = useRealTimeData();

  const { 
    connectionState, 
    messagesReceived, 
    messageLatency 
  } = useWebSocketManager();

  const [activeTab, setActiveTab] = useState(defaultActiveTab);
  const [streamingEnabled, setStreamingEnabled] = useState(false);
  const [selectedSymbols, setSelectedSymbols] = useState<string[]>(['AAPL', 'GOOGL', 'TSLA']);
  const [priceChartData, setPriceChartData] = useState<any[]>([]);
  const [filterSeverity, setFilterSeverity] = useState<string>('all');

  // Toggle streaming
  const handleStreamingToggle = (enabled: boolean) => {
    setStreamingEnabled(enabled);
    if (enabled) {
      startStreaming();
      // Subscribe to default streams
      subscribeToStream('market_data', { symbols: selectedSymbols });
      subscribeToStream('trade_updates', {});
      subscribeToStream('risk_alerts', {});
      subscribeToStream('performance_updates', {});
    } else {
      stopStreaming();
    }
  };

  // Update price chart data for selected symbols
  useEffect(() => {
    if (marketData && selectedSymbols.length > 0) {
      const chartData = marketData
        .filter(data => selectedSymbols.includes(data.symbol))
        .slice(-maxDataPoints)
        .map(data => ({
          time: new Date(data.timestamp).toLocaleTimeString(),
          ...selectedSymbols.reduce((acc, symbol) => {
            const symbolData = marketData.find(d => d.symbol === symbol);
            acc[symbol] = symbolData?.price || 0;
            return acc;
          }, {} as Record<string, number>)
        }));
      
      setPriceChartData(chartData);
    }
  }, [marketData, selectedSymbols, maxDataPoints]);

  // Streaming statistics
  const streamStats = useMemo(() => {
    const stats = getStreamStatistics ? getStreamStatistics() : null;
    return {
      totalStreams: 6,
      activeStreams: isStreaming ? 6 : 0,
      messagesPerSecond: stats?.messagesPerSecond || 0,
      totalMessages: messagesReceived || 0,
      averageLatency: messageLatency || 0,
      connectionQuality: connectionState === 'connected' ? 100 : 0
    };
  }, [isStreaming, messagesReceived, messageLatency, connectionState, getStreamStatistics]);

  // Filter risk alerts by severity
  const filteredRiskAlerts = useMemo(() => {
    if (!riskAlerts) return [];
    if (filterSeverity === 'all') return riskAlerts;
    return riskAlerts.filter(alert => alert.severity === filterSeverity);
  }, [riskAlerts, filterSeverity]);

  // Market Data columns
  const marketDataColumns = [
    {
      title: 'Symbol',
      dataIndex: 'symbol',
      key: 'symbol',
      render: (symbol: string) => <Text strong>{symbol}</Text>
    },
    {
      title: 'Price',
      dataIndex: 'price',
      key: 'price',
      render: (price: number) => `$${price.toFixed(2)}`
    },
    {
      title: 'Change',
      dataIndex: 'change',
      key: 'change',
      render: (change: number) => (
        <Text type={change >= 0 ? 'success' : 'danger'}>
          {change >= 0 ? '+' : ''}{change.toFixed(2)}%
        </Text>
      )
    },
    {
      title: 'Volume',
      dataIndex: 'volume',
      key: 'volume',
      render: (volume: number) => volume.toLocaleString()
    },
    {
      title: 'Time',
      dataIndex: 'timestamp',
      key: 'timestamp',
      render: (timestamp: string) => new Date(timestamp).toLocaleTimeString()
    }
  ];

  // Trade Updates columns
  const tradeColumns = [
    {
      title: 'Symbol',
      dataIndex: 'symbol',
      key: 'symbol'
    },
    {
      title: 'Side',
      dataIndex: 'side',
      key: 'side',
      render: (side: string) => (
        <Tag color={side === 'buy' ? 'green' : 'red'}>{side.toUpperCase()}</Tag>
      )
    },
    {
      title: 'Quantity',
      dataIndex: 'quantity',
      key: 'quantity'
    },
    {
      title: 'Price',
      dataIndex: 'price',
      key: 'price',
      render: (price: number) => `$${price.toFixed(2)}`
    },
    {
      title: 'Status',
      dataIndex: 'status',
      key: 'status',
      render: (status: string) => {
        const color = status === 'filled' ? 'green' : status === 'pending' ? 'orange' : 'red';
        return <Badge color={color} text={status} />;
      }
    }
  ];

  // Get alert icon and color
  const getAlertProps = (severity: string) => {
    switch (severity) {
      case 'critical':
        return { icon: <WarningOutlined />, color: '#ff4d4f' };
      case 'high':
        return { icon: <AlertOutlined />, color: '#fa8c16' };
      case 'medium':
        return { icon: <InfoCircleOutlined />, color: '#faad14' };
      case 'low':
        return { icon: <CheckCircleOutlined />, color: '#52c41a' };
      default:
        return { icon: <InfoCircleOutlined />, color: '#d9d9d9' };
    }
  };

  return (
    <div className={className}>
      <Card
        title={
          <Space>
            <LineChartOutlined />
            <span>Real-Time Streaming Dashboard</span>
            <Badge 
              count={streamStats.activeStreams} 
              style={{ 
                backgroundColor: streamingEnabled ? '#52c41a' : '#d9d9d9' 
              }} 
            />
          </Space>
        }
        extra={
          <Space>
            <Switch
              checked={streamingEnabled}
              onChange={handleStreamingToggle}
              checkedChildren={<PlayCircleOutlined />}
              unCheckedChildren={<PauseCircleOutlined />}
              disabled={connectionState !== 'connected'}
            />
            <Text>Streaming {streamingEnabled ? 'On' : 'Off'}</Text>
          </Space>
        }
      >
        {/* Connection Status Alert */}
        {connectionState !== 'connected' && (
          <Alert
            message="WebSocket Disconnected"
            description="Real-time streaming requires an active WebSocket connection"
            type="warning"
            showIcon
            style={{ marginBottom: 16 }}
          />
        )}

        {/* Performance Overview */}
        {showPerformanceMetrics && (
          <Row gutter={[16, 16]} style={{ marginBottom: 16 }}>
            <Col span={4}>
              <Statistic
                title="Active Streams"
                value={streamStats.activeStreams}
                suffix={`/ ${streamStats.totalStreams}`}
                prefix={<DatabaseOutlined />}
                valueStyle={{ color: streamingEnabled ? '#52c41a' : '#d9d9d9' }}
              />
            </Col>
            
            <Col span={5}>
              <Statistic
                title="Messages/sec"
                value={streamStats.messagesPerSecond}
                precision={1}
                prefix={<ThunderboltOutlined />}
                valueStyle={{ color: '#1890ff' }}
              />
            </Col>
            
            <Col span={5}>
              <Statistic
                title="Total Messages"
                value={streamStats.totalMessages}
                prefix={<BarChartOutlined />}
              />
            </Col>
            
            <Col span={5}>
              <Statistic
                title="Avg Latency"
                value={streamStats.averageLatency}
                precision={0}
                suffix="ms"
                prefix={<RiseOutlined />}
              />
            </Col>
            
            <Col span={5}>
              <div style={{ textAlign: 'center' }}>
                <Text>Connection Quality</Text>
                <Progress
                  type="circle"
                  percent={streamStats.connectionQuality}
                  width={60}
                  strokeColor={{
                    '0%': '#ff4d4f',
                    '50%': '#faad14',
                    '100%': '#52c41a',
                  }}
                />
              </div>
            </Col>
          </Row>
        )}

        {/* Real-Time Data Tabs */}
        <Tabs activeKey={activeTab} onChange={setActiveTab}>
          {/* Market Data Tab */}
          <TabPane
            tab={
              <Space>
                <DollarOutlined />
                Market Data
                <Badge count={marketData?.length || 0} size="small" />
              </Space>
            }
            key="market_data"
          >
            <Row gutter={[16, 16]}>
              <Col span={16}>
                <Card 
                  title="Price Chart" 
                  size="small"
                  extra={
                    <Select
                      mode="multiple"
                      value={selectedSymbols}
                      onChange={setSelectedSymbols}
                      style={{ width: 200 }}
                      size="small"
                    >
                      <Option value="AAPL">AAPL</Option>
                      <Option value="GOOGL">GOOGL</Option>
                      <Option value="TSLA">TSLA</Option>
                      <Option value="MSFT">MSFT</Option>
                      <Option value="AMZN">AMZN</Option>
                    </Select>
                  }
                >
                  <ResponsiveContainer width="100%" height={300}>
                    <LineChart data={priceChartData}>
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis dataKey="time" />
                      <YAxis />
                      <RechartsTooltip />
                      {selectedSymbols.map((symbol, index) => (
                        <Line
                          key={symbol}
                          type="monotone"
                          dataKey={symbol}
                          stroke={['#1890ff', '#52c41a', '#fa8c16', '#722ed1', '#eb2f96'][index % 5]}
                          strokeWidth={2}
                          dot={false}
                        />
                      ))}
                    </LineChart>
                  </ResponsiveContainer>
                </Card>
              </Col>
              
              <Col span={8}>
                <Card title="Live Prices" size="small">
                  {streamingEnabled ? (
                    <Table
                      dataSource={marketData?.slice(-10) || []}
                      columns={marketDataColumns}
                      size="small"
                      pagination={false}
                      scroll={{ y: 300 }}
                    />
                  ) : (
                    <Empty description="Start streaming to see live data" />
                  )}
                </Card>
              </Col>
            </Row>
          </TabPane>

          {/* Trade Updates Tab */}
          <TabPane
            tab={
              <Space>
                <UserOutlined />
                Trade Updates
                <Badge count={tradeUpdates?.length || 0} size="small" />
              </Space>
            }
            key="trade_updates"
          >
            {streamingEnabled ? (
              <Table
                dataSource={tradeUpdates?.slice(-20) || []}
                columns={tradeColumns}
                size="small"
                pagination={{ pageSize: 10 }}
                loading={!tradeUpdates}
              />
            ) : (
              <Empty description="Start streaming to see trade updates" />
            )}
          </TabPane>

          {/* Risk Alerts Tab */}
          <TabPane
            tab={
              <Space>
                <AlertOutlined />
                Risk Alerts
                <Badge 
                  count={riskAlerts?.filter(alert => alert.severity === 'critical').length || 0} 
                  size="small"
                  style={{ backgroundColor: '#ff4d4f' }}
                />
              </Space>
            }
            key="risk_alerts"
          >
            <Row gutter={[16, 8]} style={{ marginBottom: 16 }}>
              <Col>
                <Text>Filter by severity:</Text>
              </Col>
              <Col>
                <Select
                  value={filterSeverity}
                  onChange={setFilterSeverity}
                  size="small"
                  style={{ width: 120 }}
                >
                  <Option value="all">All</Option>
                  <Option value="critical">Critical</Option>
                  <Option value="high">High</Option>
                  <Option value="medium">Medium</Option>
                  <Option value="low">Low</Option>
                </Select>
              </Col>
            </Row>

            {streamingEnabled ? (
              <List
                dataSource={filteredRiskAlerts?.slice(-20) || []}
                renderItem={(alert: any) => {
                  const { icon, color } = getAlertProps(alert.severity);
                  return (
                    <List.Item>
                      <List.Item.Meta
                        avatar={
                          <Avatar 
                            icon={icon} 
                            style={{ backgroundColor: color }}
                          />
                        }
                        title={
                          <Space>
                            <Text strong>{alert.type}</Text>
                            <Tag color={color}>{alert.severity}</Tag>
                          </Space>
                        }
                        description={
                          <Space direction="vertical" size="small">
                            <Text>{alert.message}</Text>
                            <Text type="secondary">
                              Portfolio: {alert.portfolio} | 
                              {new Date(alert.timestamp).toLocaleTimeString()}
                            </Text>
                          </Space>
                        }
                      />
                    </List.Item>
                  );
                }}
                pagination={{ pageSize: 10 }}
              />
            ) : (
              <Empty description="Start streaming to see risk alerts" />
            )}
          </TabPane>

          {/* Performance Updates Tab */}
          <TabPane
            tab={
              <Space>
                <BarChartOutlined />
                Performance
                <Badge count={performanceData?.length || 0} size="small" />
              </Space>
            }
            key="performance"
          >
            {streamingEnabled ? (
              performanceData && performanceData.length > 0 ? (
                <Row gutter={[16, 16]}>
                  <Col span={12}>
                    <Card title="Portfolio Performance" size="small">
                      <Space direction="vertical" size="large" style={{ width: '100%' }}>
                        {performanceData.slice(-5).map((perf: any, index: number) => (
                          <div key={index}>
                            <Row justify="space-between" align="middle">
                              <Col>
                                <Text strong>{perf.portfolio}</Text>
                              </Col>
                              <Col>
                                <Text type="secondary">
                                  {new Date(perf.timestamp).toLocaleTimeString()}
                                </Text>
                              </Col>
                            </Row>
                            <Row gutter={[16, 8]}>
                              <Col span={6}>
                                <Statistic
                                  title="P&L"
                                  value={perf.pnl}
                                  precision={2}
                                  valueStyle={{ 
                                    color: perf.pnl >= 0 ? '#52c41a' : '#ff4d4f',
                                    fontSize: '14px'
                                  }}
                                  prefix="$"
                                />
                              </Col>
                              <Col span={6}>
                                <Statistic
                                  title="Returns"
                                  value={perf.returns}
                                  precision={2}
                                  suffix="%"
                                  valueStyle={{ 
                                    color: perf.returns >= 0 ? '#52c41a' : '#ff4d4f',
                                    fontSize: '14px'
                                  }}
                                />
                              </Col>
                              <Col span={6}>
                                <Statistic
                                  title="Sharpe"
                                  value={perf.sharpe}
                                  precision={2}
                                  valueStyle={{ fontSize: '14px' }}
                                />
                              </Col>
                              <Col span={6}>
                                <Statistic
                                  title="Max DD"
                                  value={perf.maxDrawdown}
                                  precision={2}
                                  suffix="%"
                                  valueStyle={{ 
                                    color: '#ff4d4f',
                                    fontSize: '14px'
                                  }}
                                />
                              </Col>
                            </Row>
                          </div>
                        ))}
                      </Space>
                    </Card>
                  </Col>
                  
                  <Col span={12}>
                    <Card title="Performance Trend" size="small">
                      <ResponsiveContainer width="100%" height={300}>
                        <AreaChart data={performanceData.slice(-20)}>
                          <CartesianGrid strokeDasharray="3 3" />
                          <XAxis 
                            dataKey="timestamp" 
                            tickFormatter={(timestamp) => 
                              new Date(timestamp).toLocaleTimeString()
                            }
                          />
                          <YAxis />
                          <RechartsTooltip />
                          <Area
                            type="monotone"
                            dataKey="pnl"
                            stroke="#1890ff"
                            fill="#1890ff"
                            fillOpacity={0.3}
                          />
                        </AreaChart>
                      </ResponsiveContainer>
                    </Card>
                  </Col>
                </Row>
              ) : (
                <Spin tip="Loading performance data..." />
              )
            ) : (
              <Empty description="Start streaming to see performance updates" />
            )}
          </TabPane>

          {/* System Health Tab */}
          <TabPane
            tab={
              <Space>
                <SettingOutlined />
                System Health
                <Badge count={systemHealth?.length || 0} size="small" />
              </Space>
            }
            key="system_health"
          >
            {streamingEnabled ? (
              systemHealth && systemHealth.length > 0 ? (
                <Row gutter={[16, 16]}>
                  {systemHealth.map((health: any, index: number) => (
                    <Col span={8} key={index}>
                      <Card size="small" title={health.component}>
                        <Space direction="vertical" size="small" style={{ width: '100%' }}>
                          <Row justify="space-between">
                            <Text>Status:</Text>
                            <Badge 
                              status={health.status === 'healthy' ? 'success' : 'error'} 
                              text={health.status} 
                            />
                          </Row>
                          <Row justify="space-between">
                            <Text>CPU:</Text>
                            <Progress 
                              percent={health.cpu} 
                              size="small" 
                              strokeColor={health.cpu > 80 ? '#ff4d4f' : '#52c41a'}
                            />
                          </Row>
                          <Row justify="space-between">
                            <Text>Memory:</Text>
                            <Progress 
                              percent={health.memory} 
                              size="small"
                              strokeColor={health.memory > 80 ? '#ff4d4f' : '#52c41a'}
                            />
                          </Row>
                          <Row justify="space-between">
                            <Text>Uptime:</Text>
                            <Text>{Math.floor(health.uptime / 60)}m</Text>
                          </Row>
                        </Space>
                      </Card>
                    </Col>
                  ))}
                </Row>
              ) : (
                <Spin tip="Loading system health data..." />
              )
            ) : (
              <Empty description="Start streaming to see system health" />
            )}
          </TabPane>
        </Tabs>
      </Card>
    </div>
  );
};

export default RealTimeStreaming;