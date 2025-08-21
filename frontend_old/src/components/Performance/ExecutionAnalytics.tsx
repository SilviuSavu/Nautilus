import React, { useState, useEffect } from 'react';
import {
  Card,
  Row,
  Col,
  Typography,
  Table,
  Statistic,
  Progress,
  Space,
  Select,
  DatePicker,
  Button,
  Alert,
  Tooltip,
  Tag,
  Tabs
} from 'antd';
import {
  ThunderboltOutlined,
  ClockCircleOutlined,
  DollarOutlined,
  PercentageOutlined,
  BarChartOutlined,
  ReloadOutlined,
  RiseOutlined,
  FallOutlined
} from '@ant-design/icons';
import type { ColumnsType } from 'antd/es/table';
import dayjs from 'dayjs';

const { Title, Text } = Typography;
const { RangePicker } = DatePicker;
const { Option } = Select;

interface ExecutionAnalyticsProps {
  className?: string;
}

interface ExecutionMetrics {
  strategy_id: string;
  total_trades: number;
  avg_execution_time_ms: number;
  avg_slippage_bps: number;
  avg_commission: number;
  fill_rate: number;
  latency_p50: number;
  latency_p95: number;
  latency_p99: number;
  market_impact_bps: number;
  implementation_shortfall: number;
  vwap_performance: number;
  execution_cost_bps: number;
  rejected_orders: number;
  partial_fills: number;
  successful_fills: number;
}

interface TradeExecution {
  id: string;
  strategy_id: string;
  instrument: string;
  side: 'buy' | 'sell';
  order_type: 'market' | 'limit' | 'stop' | 'stop_limit';
  quantity_requested: number;
  quantity_filled: number;
  requested_price?: number;
  executed_price: number;
  commission: number;
  slippage_bps: number;
  execution_time_ms: number;
  market_impact_bps: number;
  vwap_benchmark: number;
  timestamp: Date;
  venue: string;
  order_id: string;
  fill_quality: 'excellent' | 'good' | 'fair' | 'poor';
}

interface SlippageAnalysis {
  time_period: string;
  avg_slippage_bps: number;
  slippage_volatility: number;
  worst_slippage_bps: number;
  best_slippage_bps: number;
  trade_count: number;
  slippage_distribution: {
    excellent: number; // < 1 bps
    good: number;      // 1-5 bps
    fair: number;      // 5-15 bps
    poor: number;      // > 15 bps
  };
}

interface LatencyAnalysis {
  strategy_id: string;
  avg_latency_ms: number;
  p50_latency_ms: number;
  p95_latency_ms: number;
  p99_latency_ms: number;
  max_latency_ms: number;
  timeout_rate: number;
  latency_trend: 'improving' | 'stable' | 'degrading';
}

export const ExecutionAnalytics: React.FC<ExecutionAnalyticsProps> = ({ className }) => {
  const [loading, setLoading] = useState(false);
  const [executionMetrics, setExecutionMetrics] = useState<ExecutionMetrics[]>([]);
  const [recentTrades, setRecentTrades] = useState<TradeExecution[]>([]);
  const [slippageAnalysis, setSlippageAnalysis] = useState<SlippageAnalysis[]>([]);
  const [latencyAnalysis, setLatencyAnalysis] = useState<LatencyAnalysis[]>([]);
  const [selectedStrategy, setSelectedStrategy] = useState<string>('all');
  const [timeRange, setTimeRange] = useState<[dayjs.Dayjs, dayjs.Dayjs]>([
    dayjs().subtract(7, 'days'),
    dayjs()
  ]);
  const [activeTab, setActiveTab] = useState('overview');

  useEffect(() => {
    loadExecutionData();
  }, [selectedStrategy, timeRange]);

  const loadExecutionData = async () => {
    try {
      setLoading(true);
      
      const params = new URLSearchParams({
        strategy_id: selectedStrategy,
        start_date: timeRange[0].toISOString(),
        end_date: timeRange[1].toISOString()
      });

      // Load execution metrics
      const metricsResponse = await fetch(`/api/v1/execution/metrics?${params}`);
      if (metricsResponse.ok) {
        const data = await metricsResponse.json();
        setExecutionMetrics(data.metrics || []);
      }

      // Load recent trades
      const tradesResponse = await fetch(`/api/v1/execution/trades?${params}&limit=100`);
      if (tradesResponse.ok) {
        const data = await tradesResponse.json();
        setRecentTrades(data.trades || []);
      }

      // Load slippage analysis
      const slippageResponse = await fetch(`/api/v1/execution/slippage-analysis?${params}`);
      if (slippageResponse.ok) {
        const data = await slippageResponse.json();
        setSlippageAnalysis(data.analysis || []);
      }

      // Load latency analysis
      const latencyResponse = await fetch(`/api/v1/execution/latency-analysis?${params}`);
      if (latencyResponse.ok) {
        const data = await latencyResponse.json();
        setLatencyAnalysis(data.analysis || []);
      }

    } catch (error: any) {
      console.error('Failed to load execution data:', error);
    } finally {
      setLoading(false);
    }
  };

  const getFillQualityColor = (quality: string): string => {
    switch (quality) {
      case 'excellent': return '#52c41a';
      case 'good': return '#73d13d';
      case 'fair': return '#fa8c16';
      case 'poor': return '#ff4d4f';
      default: return '#8c8c8c';
    }
  };

  const getLatencyTrendIcon = (trend: string) => {
    switch (trend) {
      case 'improving': return <FallOutlined style={{ color: '#52c41a' }} />;
      case 'degrading': return <RiseOutlined style={{ color: '#ff4d4f' }} />;
      default: return <BarChartOutlined style={{ color: '#8c8c8c' }} />;
    }
  };

  const formatBasisPoints = (bps: number): string => {
    return `${bps.toFixed(1)} bps`;
  };

  const calculateOverallMetrics = () => {
    if (executionMetrics.length === 0) return null;
    
    const totalTrades = executionMetrics.reduce((sum, m) => sum + m.total_trades, 0);
    const avgFillRate = executionMetrics.reduce((sum, m) => sum + m.fill_rate * m.total_trades, 0) / totalTrades;
    const avgSlippage = executionMetrics.reduce((sum, m) => sum + m.avg_slippage_bps * m.total_trades, 0) / totalTrades;
    const avgLatency = executionMetrics.reduce((sum, m) => sum + m.avg_execution_time_ms * m.total_trades, 0) / totalTrades;
    const avgCost = executionMetrics.reduce((sum, m) => sum + m.execution_cost_bps * m.total_trades, 0) / totalTrades;

    return {
      totalTrades,
      avgFillRate,
      avgSlippage,
      avgLatency,
      avgCost
    };
  };

  const metricsColumns: ColumnsType<ExecutionMetrics> = [
    {
      title: 'Strategy',
      dataIndex: 'strategy_id',
      key: 'strategy_id',
      render: (id: string) => <Text strong>{id}</Text>
    },
    {
      title: 'Trades',
      dataIndex: 'total_trades',
      key: 'total_trades',
      render: (count: number) => count.toLocaleString()
    },
    {
      title: 'Fill Rate',
      dataIndex: 'fill_rate',
      key: 'fill_rate',
      render: (rate: number) => (
        <div>
          <Text style={{ color: rate > 0.95 ? '#52c41a' : rate > 0.9 ? '#fa8c16' : '#ff4d4f' }}>
            {(rate * 100).toFixed(1)}%
          </Text>
          <Progress 
            percent={rate * 100} 
            size="small" 
            strokeColor={rate > 0.95 ? '#52c41a' : rate > 0.9 ? '#fa8c16' : '#ff4d4f'}
            showInfo={false}
          />
        </div>
      )
    },
    {
      title: 'Avg Slippage',
      dataIndex: 'avg_slippage_bps',
      key: 'avg_slippage_bps',
      render: (slippage: number) => (
        <Text style={{ color: slippage < 5 ? '#52c41a' : slippage < 15 ? '#fa8c16' : '#ff4d4f' }}>
          {formatBasisPoints(slippage)}
        </Text>
      )
    },
    {
      title: 'Avg Latency',
      dataIndex: 'avg_execution_time_ms',
      key: 'avg_execution_time_ms',
      render: (latency: number) => (
        <Text style={{ color: latency < 100 ? '#52c41a' : latency < 500 ? '#fa8c16' : '#ff4d4f' }}>
          {latency.toFixed(0)}ms
        </Text>
      )
    },
    {
      title: 'Market Impact',
      dataIndex: 'market_impact_bps',
      key: 'market_impact_bps',
      render: (impact: number) => (
        <Text style={{ color: impact < 2 ? '#52c41a' : impact < 10 ? '#fa8c16' : '#ff4d4f' }}>
          {formatBasisPoints(impact)}
        </Text>
      )
    },
    {
      title: 'Total Cost',
      dataIndex: 'execution_cost_bps',
      key: 'execution_cost_bps',
      render: (cost: number) => (
        <Text style={{ color: cost < 10 ? '#52c41a' : cost < 25 ? '#fa8c16' : '#ff4d4f' }}>
          {formatBasisPoints(cost)}
        </Text>
      )
    },
    {
      title: 'Latency P95',
      dataIndex: 'latency_p95',
      key: 'latency_p95',
      render: (latency: number) => `${latency.toFixed(0)}ms`
    }
  ];

  const tradesColumns: ColumnsType<TradeExecution> = [
    {
      title: 'Time',
      dataIndex: 'timestamp',
      key: 'timestamp',
      render: (date: Date) => new Date(date).toLocaleTimeString()
    },
    {
      title: 'Instrument',
      dataIndex: 'instrument',
      key: 'instrument'
    },
    {
      title: 'Side',
      dataIndex: 'side',
      key: 'side',
      render: (side: string) => (
        <Tag color={side === 'buy' ? 'green' : 'red'}>
          {side.toUpperCase()}
        </Tag>
      )
    },
    {
      title: 'Quantity',
      key: 'quantity',
      render: (_, record: TradeExecution) => (
        <div>
          <Text>{record.quantity_filled.toLocaleString()}</Text>
          {record.quantity_filled < record.quantity_requested && (
            <div style={{ fontSize: 12, color: '#fa8c16' }}>
              Partial: {record.quantity_requested.toLocaleString()} req
            </div>
          )}
        </div>
      )
    },
    {
      title: 'Price',
      dataIndex: 'executed_price',
      key: 'executed_price',
      render: (price: number) => `$${price.toFixed(4)}`
    },
    {
      title: 'Slippage',
      dataIndex: 'slippage_bps',
      key: 'slippage_bps',
      render: (slippage: number) => (
        <Text style={{ color: Math.abs(slippage) < 5 ? '#52c41a' : Math.abs(slippage) < 15 ? '#fa8c16' : '#ff4d4f' }}>
          {formatBasisPoints(slippage)}
        </Text>
      )
    },
    {
      title: 'Execution Time',
      dataIndex: 'execution_time_ms',
      key: 'execution_time_ms',
      render: (time: number) => (
        <Text style={{ color: time < 100 ? '#52c41a' : time < 500 ? '#fa8c16' : '#ff4d4f' }}>
          {time}ms
        </Text>
      )
    },
    {
      title: 'Quality',
      dataIndex: 'fill_quality',
      key: 'fill_quality',
      render: (quality: string) => (
        <Tag color={getFillQualityColor(quality)}>
          {quality.toUpperCase()}
        </Tag>
      )
    }
  ];

  const renderOverview = () => {
    const overallMetrics = calculateOverallMetrics();
    if (!overallMetrics) return null;

    return (
      <div>
        <Row gutter={[16, 16]} style={{ marginBottom: 24 }}>
          <Col xs={24} sm={6}>
            <Card>
              <Statistic
                title="Total Trades"
                value={overallMetrics.totalTrades}
                prefix={<BarChartOutlined />}
                valueStyle={{ color: '#1890ff' }}
              />
            </Card>
          </Col>
          <Col xs={24} sm={6}>
            <Card>
              <Statistic
                title="Avg Fill Rate"
                value={overallMetrics.avgFillRate * 100}
                precision={1}
                suffix="%"
                prefix={<PercentageOutlined />}
                valueStyle={{ 
                  color: overallMetrics.avgFillRate > 0.95 ? '#52c41a' : 
                         overallMetrics.avgFillRate > 0.9 ? '#fa8c16' : '#ff4d4f' 
                }}
              />
            </Card>
          </Col>
          <Col xs={24} sm={6}>
            <Card>
              <Statistic
                title="Avg Slippage"
                value={overallMetrics.avgSlippage}
                precision={1}
                suffix=" bps"
                prefix={<DollarOutlined />}
                valueStyle={{ 
                  color: overallMetrics.avgSlippage < 5 ? '#52c41a' : 
                         overallMetrics.avgSlippage < 15 ? '#fa8c16' : '#ff4d4f' 
                }}
              />
            </Card>
          </Col>
          <Col xs={24} sm={6}>
            <Card>
              <Statistic
                title="Avg Latency"
                value={overallMetrics.avgLatency}
                precision={0}
                suffix=" ms"
                prefix={<ClockCircleOutlined />}
                valueStyle={{ 
                  color: overallMetrics.avgLatency < 100 ? '#52c41a' : 
                         overallMetrics.avgLatency < 500 ? '#fa8c16' : '#ff4d4f' 
                }}
              />
            </Card>
          </Col>
        </Row>

        {/* Execution Quality Alert */}
        {overallMetrics.avgFillRate < 0.9 && (
          <Alert
            message="Low Fill Rate Warning"
            description={`Average fill rate is ${(overallMetrics.avgFillRate * 100).toFixed(1)}%, which is below the 90% threshold. Consider reviewing order routing and timing.`}
            type="warning"
            style={{ marginBottom: 16 }}
            showIcon
          />
        )}

        <Card title="Strategy Execution Metrics">
          <Table
            columns={metricsColumns}
            dataSource={executionMetrics}
            rowKey="strategy_id"
            loading={loading}
            pagination={false}
            scroll={{ x: 1000 }}
          />
        </Card>
      </div>
    );
  };

  const renderTradeQuality = () => (
    <Card title="Recent Trade Executions">
      <Table
        columns={tradesColumns}
        dataSource={recentTrades}
        rowKey="id"
        loading={loading}
        pagination={{ pageSize: 20 }}
        scroll={{ x: 1000 }}
      />
    </Card>
  );

  const renderSlippageAnalysis = () => (
    <Row gutter={[16, 16]}>
      {slippageAnalysis.map((analysis, index) => (
        <Col xs={24} lg={12} key={index}>
          <Card title={`Slippage Analysis - ${analysis.time_period}`}>
            <Space direction="vertical" style={{ width: '100%' }}>
              <Row gutter={16}>
                <Col span={12}>
                  <Statistic
                    title="Average Slippage"
                    value={analysis.avg_slippage_bps}
                    precision={1}
                    suffix=" bps"
                    valueStyle={{ 
                      color: analysis.avg_slippage_bps < 5 ? '#52c41a' : '#fa8c16' 
                    }}
                  />
                </Col>
                <Col span={12}>
                  <Statistic
                    title="Worst Slippage"
                    value={analysis.worst_slippage_bps}
                    precision={1}
                    suffix=" bps"
                    valueStyle={{ color: '#ff4d4f' }}
                  />
                </Col>
              </Row>
              
              <div>
                <Text strong>Quality Distribution:</Text>
                <div style={{ marginTop: 8 }}>
                  <Space direction="vertical" style={{ width: '100%' }}>
                    <div>
                      <Text>Excellent (&lt;1 bps): </Text>
                      <Progress 
                        percent={(analysis.slippage_distribution.excellent / analysis.trade_count) * 100}
                        strokeColor="#52c41a"
                        size="small"
                      />
                    </div>
                    <div>
                      <Text>Good (1-5 bps): </Text>
                      <Progress 
                        percent={(analysis.slippage_distribution.good / analysis.trade_count) * 100}
                        strokeColor="#73d13d"
                        size="small"
                      />
                    </div>
                    <div>
                      <Text>Fair (5-15 bps): </Text>
                      <Progress 
                        percent={(analysis.slippage_distribution.fair / analysis.trade_count) * 100}
                        strokeColor="#fa8c16"
                        size="small"
                      />
                    </div>
                    <div>
                      <Text>Poor (&gt;15 bps): </Text>
                      <Progress 
                        percent={(analysis.slippage_distribution.poor / analysis.trade_count) * 100}
                        strokeColor="#ff4d4f"
                        size="small"
                      />
                    </div>
                  </Space>
                </div>
              </div>
            </Space>
          </Card>
        </Col>
      ))}
    </Row>
  );

  const renderLatencyAnalysis = () => (
    <Row gutter={[16, 16]}>
      {latencyAnalysis.map((analysis, index) => (
        <Col xs={24} lg={8} key={index}>
          <Card 
            title={`${analysis.strategy_id} Latency`}
            extra={getLatencyTrendIcon(analysis.latency_trend)}
          >
            <Space direction="vertical" style={{ width: '100%' }}>
              <Statistic
                title="P50 Latency"
                value={analysis.p50_latency_ms}
                suffix=" ms"
                valueStyle={{ 
                  color: analysis.p50_latency_ms < 50 ? '#52c41a' : '#fa8c16' 
                }}
              />
              <Statistic
                title="P95 Latency"
                value={analysis.p95_latency_ms}
                suffix=" ms"
                valueStyle={{ 
                  color: analysis.p95_latency_ms < 200 ? '#52c41a' : '#fa8c16' 
                }}
              />
              <Statistic
                title="P99 Latency"
                value={analysis.p99_latency_ms}
                suffix=" ms"
                valueStyle={{ 
                  color: analysis.p99_latency_ms < 500 ? '#52c41a' : '#ff4d4f' 
                }}
              />
              {analysis.timeout_rate > 0 && (
                <div>
                  <Text type="danger">
                    Timeout Rate: {(analysis.timeout_rate * 100).toFixed(2)}%
                  </Text>
                </div>
              )}
            </Space>
          </Card>
        </Col>
      ))}
    </Row>
  );

  return (
    <div className={`execution-analytics ${className || ''}`}>
      <Card>
        <Row justify="space-between" align="middle" style={{ marginBottom: 24 }}>
          <Col>
            <Title level={2} style={{ margin: 0 }}>
              <ThunderboltOutlined style={{ marginRight: 8 }} />
              Execution Analytics
            </Title>
            <Text type="secondary">
              Trade execution quality and performance analysis
            </Text>
          </Col>
          <Col>
            <Space>
              <Select
                value={selectedStrategy}
                onChange={setSelectedStrategy}
                style={{ width: 150 }}
              >
                <Option value="all">All Strategies</Option>
                {executionMetrics.map(metric => (
                  <Option key={metric.strategy_id} value={metric.strategy_id}>
                    {metric.strategy_id}
                  </Option>
                ))}
              </Select>
              
              <RangePicker
                value={timeRange}
                onChange={(dates) => dates && setTimeRange(dates)}
                style={{ width: 240 }}
              />
              
              <Button
                icon={<ReloadOutlined />}
                onClick={loadExecutionData}
                loading={loading}
              >
                Refresh
              </Button>
            </Space>
          </Col>
        </Row>

        <Tabs 
          activeKey={activeTab} 
          onChange={setActiveTab}
          items={[
            {
              key: 'overview',
              label: 'Overview',
              children: renderOverview()
            },
            {
              key: 'quality',
              label: 'Trade Quality',
              children: renderTradeQuality()
            },
            {
              key: 'slippage',
              label: 'Slippage Analysis',
              children: renderSlippageAnalysis()
            },
            {
              key: 'latency',
              label: 'Latency Analysis',
              children: renderLatencyAnalysis()
            }
          ]}
        />
      </Card>
    </div>
  );
};