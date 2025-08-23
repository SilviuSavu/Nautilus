import React, { useState, useEffect, useMemo } from 'react';
import {
  Card,
  Row,
  Col,
  Statistic,
  Typography,
  Space,
  Badge,
  Progress,
  Table,
  Tag,
  Select,
  Button,
  Alert,
  Tooltip,
  Switch,
  Divider,
  Timeline,
  List
} from 'antd';
import {
  LineChartOutlined,
  RiseOutlined,
  FallOutlined,
  DashboardOutlined,
  ThunderboltOutlined,
  AlertOutlined,
  ReloadOutlined,
  SettingOutlined,
  PlayCircleOutlined,
  PauseCircleOutlined,
  FireOutlined
} from '@ant-design/icons';
import { Line, Area } from '@ant-design/charts';
import { useMessageBus } from '../../hooks/useMessageBus';
import { useEngineWebSocket } from '../../hooks/useEngineWebSocket';
import type { PerformanceMetric, AnalyticsStreamData } from '../../types/sprint3';

const { Text, Title } = Typography;
const { Option } = Select;

interface RealTimeAnalyticsProps {
  portfolioId: string;
  showStreaming?: boolean;
  updateInterval?: number;
  compactMode?: boolean;
}

const RealTimeAnalytics: React.FC<RealTimeAnalyticsProps> = ({
  portfolioId,
  showStreaming = true,
  updateInterval = 1000,
  compactMode = false
}) => {
  const [isStreaming, setIsStreaming] = useState(showStreaming);
  const [metrics, setMetrics] = useState<PerformanceMetric[]>([]);
  const [streamData, setStreamData] = useState<AnalyticsStreamData[]>([]);
  const [selectedMetric, setSelectedMetric] = useState<string>('pnl');
  const [timeRange, setTimeRange] = useState<'1m' | '5m' | '15m' | '1h'>('5m');

  const messageBus = useMessageBus();
  const engineWs = useEngineWebSocket();

  // Generate mock real-time data
  useEffect(() => {
    if (!isStreaming) return;

    const interval = setInterval(() => {
      const now = new Date().toISOString();
      
      // Generate mock performance metrics
      const newMetrics: PerformanceMetric[] = [
        {
          id: 'pnl',
          name: 'Unrealized P&L',
          value: 12450.75 + (Math.random() - 0.5) * 1000,
          change: (Math.random() - 0.5) * 500,
          changePercent: (Math.random() - 0.5) * 5,
          trend: Math.random() > 0.5 ? 'up' : 'down',
          timestamp: now,
          category: 'returns'
        },
        {
          id: 'realized_pnl',
          name: 'Realized P&L',
          value: 8234.50 + (Math.random() - 0.5) * 200,
          change: (Math.random() - 0.5) * 100,
          changePercent: (Math.random() - 0.5) * 2,
          trend: Math.random() > 0.5 ? 'up' : 'down',
          timestamp: now,
          category: 'returns'
        },
        {
          id: 'sharpe',
          name: 'Sharpe Ratio',
          value: 1.85 + (Math.random() - 0.5) * 0.2,
          change: (Math.random() - 0.5) * 0.1,
          changePercent: (Math.random() - 0.5) * 5,
          trend: Math.random() > 0.5 ? 'up' : 'down',
          timestamp: now,
          category: 'risk'
        },
        {
          id: 'drawdown',
          name: 'Max Drawdown',
          value: -2.34 + (Math.random() - 0.5) * 1,
          change: (Math.random() - 0.5) * 0.5,
          changePercent: (Math.random() - 0.5) * 10,
          trend: Math.random() > 0.5 ? 'up' : 'down',
          timestamp: now,
          category: 'risk'
        },
        {
          id: 'trades_count',
          name: 'Total Trades',
          value: 1247 + Math.floor(Math.random() * 5),
          change: Math.floor(Math.random() * 3),
          changePercent: (Math.random()) * 2,
          trend: 'up',
          timestamp: now,
          category: 'execution'
        },
        {
          id: 'win_rate',
          name: 'Win Rate',
          value: 68.5 + (Math.random() - 0.5) * 5,
          change: (Math.random() - 0.5) * 2,
          changePercent: (Math.random() - 0.5) * 3,
          trend: Math.random() > 0.5 ? 'up' : 'down',
          timestamp: now,
          category: 'execution'
        },
        {
          id: 'latency',
          name: 'Avg Latency',
          value: 45.2 + (Math.random() - 0.5) * 10,
          change: (Math.random() - 0.5) * 5,
          changePercent: (Math.random() - 0.5) * 10,
          trend: Math.random() > 0.5 ? 'down' : 'up', // Lower is better for latency
          timestamp: now,
          category: 'system'
        },
        {
          id: 'messages_sec',
          name: 'Messages/sec',
          value: 156.8 + (Math.random() - 0.5) * 50,
          change: (Math.random() - 0.5) * 20,
          changePercent: (Math.random() - 0.5) * 15,
          trend: Math.random() > 0.5 ? 'up' : 'down',
          timestamp: now,
          category: 'system'
        }
      ];

      setMetrics(newMetrics);

      // Add to stream data for charts
      const selectedMetricData = newMetrics.find(m => m.id === selectedMetric);
      if (selectedMetricData) {
        setStreamData(prev => {
          const newData = [...prev, {
            timestamp: now,
            metric: selectedMetric,
            value: selectedMetricData.value,
            metadata: { category: selectedMetricData.category }
          }];
          
          // Keep only last N points based on time range
          const maxPoints = timeRange === '1m' ? 60 : timeRange === '5m' ? 300 : timeRange === '15m' ? 900 : 3600;
          return newData.slice(-maxPoints);
        });
      }
    }, updateInterval);

    return () => clearInterval(interval);
  }, [isStreaming, updateInterval, selectedMetric, timeRange]);

  // Prepare chart data
  const chartData = useMemo(() => {
    return streamData.map((item, index) => ({
      time: new Date(item.timestamp).toLocaleTimeString(),
      value: item.value,
      index
    }));
  }, [streamData]);

  // Get metric color and icon
  const getMetricDisplay = (metric: PerformanceMetric) => {
    const isPositive = metric.change >= 0;
    const color = isPositive ? '#52c41a' : '#ff4d4f';
    const icon = isPositive ? <RiseOutlined /> : <FallOutlined />;
    
    return { color, icon, isPositive };
  };

  // Table columns for metrics
  const metricColumns = [
    {
      title: 'Metric',
      key: 'metric',
      render: (record: PerformanceMetric) => (
        <Space>
          <div style={{ color: getMetricDisplay(record).color }}>
            {getMetricDisplay(record).icon}
          </div>
          <div>
            <Text strong>{record.name}</Text>
            <br />
            <Tag size="small" color={
              record.category === 'returns' ? 'blue' :
              record.category === 'risk' ? 'orange' :
              record.category === 'execution' ? 'green' : 'purple'
            }>
              {record.category}
            </Tag>
          </div>
        </Space>
      )
    },
    {
      title: 'Current Value',
      key: 'value',
      render: (record: PerformanceMetric) => (
        <div>
          <Text strong style={{ fontSize: '16px' }}>
            {record.category === 'execution' && record.id === 'trades_count' 
              ? record.value.toFixed(0)
              : record.id === 'win_rate' 
              ? `${record.value.toFixed(1)}%`
              : record.id === 'latency' 
              ? `${record.value.toFixed(1)}ms`
              : record.id === 'messages_sec'
              ? `${record.value.toFixed(1)}/s`
              : `$${record.value.toFixed(2)}`
            }
          </Text>
        </div>
      )
    },
    {
      title: 'Change',
      key: 'change',
      render: (record: PerformanceMetric) => {
        const display = getMetricDisplay(record);
        return (
          <div>
            <Text style={{ color: display.color }}>
              {display.isPositive ? '+' : ''}{record.change.toFixed(2)}
            </Text>
            <br />
            <Text style={{ color: display.color, fontSize: '12px' }}>
              ({display.isPositive ? '+' : ''}{record.changePercent.toFixed(2)}%)
            </Text>
          </div>
        );
      }
    },
    {
      title: 'Trend',
      dataIndex: 'trend',
      key: 'trend',
      render: (trend: PerformanceMetric['trend']) => (
        <Progress
          percent={trend === 'up' ? 75 : trend === 'down' ? 25 : 50}
          status={trend === 'up' ? 'success' : trend === 'down' ? 'exception' : 'normal'}
          size="small"
          showInfo={false}
        />
      )
    }
  ];

  if (compactMode) {
    const keyMetrics = metrics.slice(0, 4);
    return (
      <Card
        title={
          <Space>
            <ThunderboltOutlined />
            Real-time Analytics
            {isStreaming && <Badge status="processing" text="Live" />}
          </Space>
        }
        size="small"
        extra={
          <Switch
            checked={isStreaming}
            onChange={setIsStreaming}
            size="small"
            checkedChildren="Live"
            unCheckedChildren="Off"
          />
        }
      >
        <Row gutter={[8, 8]}>
          {keyMetrics.map(metric => {
            const display = getMetricDisplay(metric);
            return (
              <Col span={6} key={metric.id}>
                <div style={{ textAlign: 'center' }}>
                  <div style={{ color: display.color, fontSize: '18px' }}>
                    {display.icon}
                  </div>
                  <Text strong style={{ fontSize: '14px' }}>
                    {metric.id === 'pnl' ? `$${metric.value.toFixed(0)}` : 
                     metric.id === 'sharpe' ? metric.value.toFixed(2) :
                     metric.id === 'win_rate' ? `${metric.value.toFixed(0)}%` :
                     metric.value.toFixed(1)}
                  </Text>
                  <br />
                  <Text type="secondary" style={{ fontSize: '10px' }}>
                    {metric.name.split(' ')[0]}
                  </Text>
                </div>
              </Col>
            );
          })}
        </Row>
      </Card>
    );
  }

  return (
    <div style={{ width: '100%' }}>
      {/* Control Panel */}
      <Card size="small" style={{ marginBottom: 16 }}>
        <Row gutter={[16, 8]} align="middle">
          <Col>
            <Space>
              <ThunderboltOutlined style={{ color: isStreaming ? '#52c41a' : '#999' }} />
              <Text strong>Real-time Analytics</Text>
              {isStreaming && <Badge status="processing" text="Streaming" />}
            </Space>
          </Col>
          <Col>
            <Space>
              <Text type="secondary">Status:</Text>
              <Badge 
                status={messageBus.connectionStatus === 'connected' ? 'success' : 'warning'} 
                text={messageBus.connectionStatus} 
              />
            </Space>
          </Col>
          <Col flex="auto" />
          <Col>
            <Space>
              <Select
                value={selectedMetric}
                onChange={setSelectedMetric}
                size="small"
                style={{ width: 120 }}
              >
                <Option value="pnl">P&L</Option>
                <Option value="sharpe">Sharpe</Option>
                <Option value="drawdown">Drawdown</Option>
                <Option value="win_rate">Win Rate</Option>
                <Option value="latency">Latency</Option>
              </Select>
              
              <Select
                value={timeRange}
                onChange={setTimeRange}
                size="small"
                style={{ width: 80 }}
              >
                <Option value="1m">1m</Option>
                <Option value="5m">5m</Option>
                <Option value="15m">15m</Option>
                <Option value="1h">1h</Option>
              </Select>
              
              <Button
                size="small"
                type={isStreaming ? "default" : "primary"}
                icon={isStreaming ? <PauseCircleOutlined /> : <PlayCircleOutlined />}
                onClick={() => setIsStreaming(!isStreaming)}
              >
                {isStreaming ? 'Pause' : 'Start'}
              </Button>
              
              <Button
                size="small"
                icon={<ReloadOutlined />}
                onClick={() => setStreamData([])}
              >
                Clear
              </Button>
            </Space>
          </Col>
        </Row>
      </Card>

      {/* Streaming Alert */}
      {isStreaming && messageBus.connectionStatus !== 'connected' && (
        <Alert
          message="WebSocket Disconnected"
          description="Real-time analytics requires active WebSocket connection. Some data may be simulated."
          type="warning"
          showIcon
          closable
          style={{ marginBottom: 16 }}
        />
      )}

      <Row gutter={[16, 16]}>
        {/* Real-time Chart */}
        <Col xs={24} lg={16}>
          <Card 
            title={
              <Space>
                <LineChartOutlined />
                Real-time {metrics.find(m => m.id === selectedMetric)?.name || 'Metric'}
                <Tag color="blue">{timeRange} window</Tag>
              </Space>
            }
            size="small"
          >
            {chartData.length > 0 ? (
              <Area
                data={chartData}
                xField="time"
                yField="value"
                height={300}
                smooth
                color="#1890ff"
                areaStyle={{
                  fill: 'l(270) 0:#ffffff 1:#1890ff',
                  fillOpacity: 0.3
                }}
                line={{
                  size: 2
                }}
                animation={{
                  appear: {
                    animation: 'wave-in',
                    duration: 1000
                  }
                }}
                xAxis={{
                  tickCount: 5,
                  label: {
                    style: { fontSize: 10 }
                  }
                }}
                yAxis={{
                  label: {
                    formatter: (value: string) => {
                      const num = parseFloat(value);
                      return selectedMetric === 'pnl' || selectedMetric === 'realized_pnl' 
                        ? `$${num.toFixed(0)}` 
                        : selectedMetric === 'win_rate' 
                        ? `${num.toFixed(0)}%`
                        : selectedMetric === 'latency'
                        ? `${num.toFixed(0)}ms`
                        : num.toFixed(1);
                    },
                    style: { fontSize: 10 }
                  }
                }}
              />
            ) : (
              <div style={{ 
                height: 300, 
                display: 'flex', 
                alignItems: 'center', 
                justifyContent: 'center',
                color: '#999'
              }}>
                <Space direction="vertical" align="center">
                  <LineChartOutlined style={{ fontSize: '48px' }} />
                  <Text type="secondary">
                    {isStreaming ? 'Collecting data...' : 'Start streaming to view real-time chart'}
                  </Text>
                </Space>
              </div>
            )}
          </Card>
        </Col>

        {/* Metrics Summary */}
        <Col xs={24} lg={8}>
          <Card 
            title={
              <Space>
                <DashboardOutlined />
                Live Metrics
                <Badge count={metrics.length} color="blue" />
              </Space>
            }
            size="small"
          >
            <List
              size="small"
              dataSource={metrics.slice(0, 6)}
              renderItem={(metric) => {
                const display = getMetricDisplay(metric);
                return (
                  <List.Item
                    style={{ 
                      cursor: 'pointer',
                      backgroundColor: selectedMetric === metric.id ? '#f0f0f0' : 'transparent',
                      padding: '8px 12px',
                      borderRadius: '4px'
                    }}
                    onClick={() => setSelectedMetric(metric.id)}
                  >
                    <List.Item.Meta
                      avatar={
                        <div style={{ color: display.color, fontSize: '16px' }}>
                          {display.icon}
                        </div>
                      }
                      title={
                        <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                          <Text strong style={{ fontSize: '12px' }}>
                            {metric.name}
                          </Text>
                          <Text style={{ fontSize: '12px', color: display.color }}>
                            {display.isPositive ? '+' : ''}{metric.changePercent.toFixed(1)}%
                          </Text>
                        </div>
                      }
                      description={
                        <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                          <Text style={{ fontSize: '14px', fontWeight: 'bold' }}>
                            {metric.id === 'trades_count' 
                              ? metric.value.toFixed(0)
                              : metric.id === 'win_rate' 
                              ? `${metric.value.toFixed(1)}%`
                              : metric.id === 'latency' 
                              ? `${metric.value.toFixed(1)}ms`
                              : metric.id === 'messages_sec'
                              ? `${metric.value.toFixed(1)}/s`
                              : `$${metric.value.toFixed(2)}`
                            }
                          </Text>
                          <Text type="secondary" style={{ fontSize: '11px' }}>
                            {new Date(metric.timestamp).toLocaleTimeString()}
                          </Text>
                        </div>
                      }
                    />
                  </List.Item>
                );
              }}
            />
          </Card>
        </Col>
      </Row>

      {/* Detailed Metrics Table */}
      <Card
        title={
          <Space>
            <FireOutlined />
            Performance Metrics Detail
          </Space>
        }
        size="small"
        style={{ marginTop: 16 }}
      >
        <Table
          columns={metricColumns}
          dataSource={metrics}
          rowKey="id"
          size="small"
          pagination={false}
          scroll={{ x: 'max-content' }}
        />
      </Card>
    </div>
  );
};

export default RealTimeAnalytics;