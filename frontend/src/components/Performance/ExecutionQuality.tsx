/**
 * ExecutionQuality Component - Sprint 3 Integration
 * Trade execution analysis with slippage monitoring and market impact
 */

import React, { useState, useEffect, useMemo } from 'react';
import { 
  Card, Row, Col, Table, Tabs, Select, DatePicker, Button, Space, Typography, 
  Statistic, Progress, Tag, Tooltip, Alert, Modal, Descriptions, Switch
} from 'antd';
import { 
  ThunderboltOutlined,
  LineChartOutlined,
  TableOutlined,
  BarChartOutlined,
  DotChartOutlined,
  SettingOutlined,
  PlayCircleOutlined,
  StopOutlined,
  DownloadOutlined,
  InfoCircleOutlined,
  RiseOutlined,
  FallOutlined
} from '@ant-design/icons';
import { Line, Column, Scatter, Heatmap, Histogram } from '@ant-design/plots';
import dayjs from 'dayjs';
import useExecutionAnalytics from '../../hooks/analytics/useExecutionAnalytics';

const { Title, Text } = Typography;
const { TabPane } = Tabs;
const { Option } = Select;
const { RangePicker } = DatePicker;

interface ExecutionQualityProps {
  portfolioId?: string;
  strategyId?: string;
  className?: string;
  height?: number;
  showRealTimeTracking?: boolean;
  defaultBenchmark?: 'arrival' | 'vwap' | 'twap' | 'close';
}

const ExecutionQuality: React.FC<ExecutionQualityProps> = ({
  portfolioId,
  strategyId,
  className,
  height = 900,
  showRealTimeTracking = true,
  defaultBenchmark = 'arrival',
}) => {
  // State
  const [activeTab, setActiveTab] = useState('overview');
  const [dateRange, setDateRange] = useState<[dayjs.Dayjs, dayjs.Dayjs] | null>([
    dayjs().subtract(30, 'days'),
    dayjs(),
  ]);
  const [selectedSymbol, setSelectedSymbol] = useState<string>('');
  const [selectedVenue, setSelectedVenue] = useState<string>('');
  const [benchmarkType, setBenchmarkType] = useState<'arrival' | 'vwap' | 'twap' | 'close'>(defaultBenchmark);
  const [showSettings, setShowSettings] = useState(false);
  const [includeMarketImpact, setIncludeMarketImpact] = useState(true);
  const [includeTimingAnalysis, setIncludeTimingAnalysis] = useState(true);
  
  const {
    result,
    isAnalyzing,
    error,
    currentTrade,
    executionAlerts,
    isTracking,
    analyzeExecution,
    startRealTimeTracking,
    stopRealTimeTracking,
    analyzeSlippage,
    getMarketImpactAnalysis,
    getExecutionAlpha,
    runTCA,
  } = useExecutionAnalytics({
    autoRefresh: true,
    refreshInterval: 10000,
    enableRealTimeTracking: showRealTimeTracking,
    alertThresholds: {
      slippage_threshold: 0.005, // 50bps
      fill_rate_threshold: 0.90,
      execution_time_threshold: 10, // 10 seconds
    },
  });
  
  // Fetch execution analysis when parameters change
  useEffect(() => {
    if (portfolioId || strategyId) {
      analyzeExecution({
        portfolio_id: portfolioId,
        strategy_id: strategyId,
        start_date: dateRange?.[0].format('YYYY-MM-DD'),
        end_date: dateRange?.[1].format('YYYY-MM-DD'),
        symbol: selectedSymbol || undefined,
        venue: selectedVenue || undefined,
        include_market_impact: includeMarketImpact,
        include_timing_analysis: includeTimingAnalysis,
        benchmark_type: benchmarkType,
      });
    }
  }, [portfolioId, strategyId, dateRange, selectedSymbol, selectedVenue, benchmarkType, includeMarketImpact, includeTimingAnalysis]);
  
  // Chart data preparation
  const slippageDistributionData = useMemo(() => {
    if (!result || !result.slippage_analysis.slippage_distribution) return [];
    
    return result.slippage_analysis.slippage_distribution.map(sd => ({
      range: `${(sd.bucket_range[0] * 100).toFixed(1)}% to ${(sd.bucket_range[1] * 100).toFixed(1)}%`,
      count: sd.trade_count,
      volume_pct: sd.volume_percentage * 100,
      avg_slippage: sd.avg_slippage * 100,
    }));
  }, [result]);
  
  const timingAnalysisData = useMemo(() => {
    if (!result || !result.timing_analysis.intraday_patterns) return [];
    
    return result.timing_analysis.intraday_patterns.map(pattern => ({
      hour: pattern.hour,
      avg_slippage: pattern.avg_slippage * 10000, // in bps
      avg_market_impact: pattern.avg_market_impact * 10000, // in bps
      trade_count: pattern.trade_count,
      volume_participation: pattern.volume_participation * 100,
    }));
  }, [result]);
  
  const venueAnalysisData = useMemo(() => {
    if (!result || !result.venue_analysis) return [];
    
    return result.venue_analysis.map(venue => ({
      venue: venue.venue_name,
      trade_count: venue.trade_count,
      volume_pct: venue.volume_percentage * 100,
      avg_slippage: venue.avg_slippage * 10000, // in bps
      avg_market_impact: venue.avg_market_impact * 10000, // in bps
      fill_rate: venue.fill_rate * 100,
      avg_execution_time: venue.avg_execution_time,
    }));
  }, [result]);
  
  const orderTypeData = useMemo(() => {
    if (!result || !result.order_type_analysis) return [];
    
    return result.order_type_analysis.map(orderType => ({
      type: orderType.order_type,
      count: orderType.trade_count,
      avg_slippage: orderType.avg_slippage * 10000,
      fill_rate: orderType.fill_rate * 100,
      avg_size: orderType.avg_size,
      market_impact: orderType.market_impact * 10000,
    }));
  }, [result]);
  
  const tradeScatterData = useMemo(() => {
    if (!result || !result.trade_details) return [];
    
    return result.trade_details.map(trade => ({
      size: trade.quantity,
      slippage: trade.slippage * 10000, // in bps
      market_impact: trade.market_impact * 10000, // in bps
      execution_time: trade.execution_time,
      venue: trade.venue,
      symbol: trade.symbol,
      side: trade.side,
    }));
  }, [result]);
  
  // Chart configurations
  const slippageHistogramConfig = {
    data: slippageDistributionData,
    xField: 'range',
    yField: 'count',
    height: 300,
    color: '#1890ff',
    label: {
      position: 'top' as const,
    },
    xAxis: {
      label: {
        autoRotate: true,
      },
    },
    tooltip: {
      formatter: (data: any) => [
        { name: 'Trade Count', value: data.count },
        { name: 'Volume %', value: `${data.volume_pct.toFixed(1)}%` },
        { name: 'Avg Slippage', value: `${data.avg_slippage.toFixed(1)}%` },
      ],
    },
  };
  
  const timingPatternConfig = {
    data: timingAnalysisData,
    xField: 'hour',
    yField: ['avg_slippage', 'avg_market_impact'],
    height: 300,
    color: ['#1890ff', '#52c41a'],
    yAxis: {
      label: {
        formatter: (v: string) => `${v}bps`,
      },
    },
    tooltip: {
      formatter: (data: any) => [
        { name: 'Avg Slippage', value: `${data.avg_slippage.toFixed(1)}bps` },
        { name: 'Market Impact', value: `${data.avg_market_impact.toFixed(1)}bps` },
        { name: 'Trade Count', value: data.trade_count },
        { name: 'Volume %', value: `${data.volume_participation.toFixed(1)}%` },
      ],
    },
  };
  
  const venueComparisonConfig = {
    data: venueAnalysisData,
    xField: 'venue',
    yField: 'avg_slippage',
    height: 300,
    color: '#faad14',
    label: {
      position: 'top' as const,
      formatter: (data: any) => `${data.avg_slippage.toFixed(1)}bps`,
    },
    xAxis: {
      label: {
        autoRotate: true,
      },
    },
    yAxis: {
      label: {
        formatter: (v: string) => `${v}bps`,
      },
    },
  };
  
  const tradeScatterConfig = {
    data: tradeScatterData,
    xField: 'size',
    yField: 'slippage',
    colorField: 'venue',
    sizeField: 'execution_time',
    height: 400,
    size: [4, 20],
    tooltip: {
      formatter: (data: any) => [
        { name: 'Symbol', value: data.symbol },
        { name: 'Size', value: data.size.toLocaleString() },
        { name: 'Slippage', value: `${data.slippage.toFixed(1)}bps` },
        { name: 'Market Impact', value: `${data.market_impact.toFixed(1)}bps` },
        { name: 'Execution Time', value: `${data.execution_time.toFixed(1)}s` },
        { name: 'Venue', value: data.venue },
      ],
    },
    xAxis: {
      label: {
        formatter: (v: string) => Number(v).toLocaleString(),
      },
    },
    yAxis: {
      label: {
        formatter: (v: string) => `${v}bps`,
      },
    },
  };
  
  // Table columns
  const tradeDetailColumns = [
    {
      title: 'Trade ID',
      dataIndex: 'trade_id',
      key: 'trade_id',
      width: 100,
      fixed: 'left' as const,
    },
    {
      title: 'Symbol',
      dataIndex: 'symbol',
      key: 'symbol',
      width: 80,
    },
    {
      title: 'Side',
      dataIndex: 'side',
      key: 'side',
      render: (side: string) => (
        <Tag color={side === 'buy' ? 'green' : 'red'}>
          {side.toUpperCase()}
        </Tag>
      ),
      width: 60,
    },
    {
      title: 'Quantity',
      dataIndex: 'quantity',
      key: 'quantity',
      render: (value: number) => value.toLocaleString(),
      sorter: (a: any, b: any) => b.quantity - a.quantity,
    },
    {
      title: 'Filled',
      dataIndex: 'filled_quantity',
      key: 'filled_quantity',
      render: (value: number) => value.toLocaleString(),
    },
    {
      title: 'Avg Price',
      dataIndex: 'avg_price',
      key: 'avg_price',
      render: (value: number) => `$${value.toFixed(4)}`,
    },
    {
      title: 'Slippage',
      dataIndex: 'slippage',
      key: 'slippage',
      render: (value: number) => (
        <span style={{ color: value < 0 ? '#52c41a' : '#ff4d4f' }}>
          {(value * 10000).toFixed(1)}bps
        </span>
      ),
      sorter: (a: any, b: any) => Math.abs(b.slippage) - Math.abs(a.slippage),
    },
    {
      title: 'Market Impact',
      dataIndex: 'market_impact',
      key: 'market_impact',
      render: (value: number) => `${(value * 10000).toFixed(1)}bps`,
      sorter: (a: any, b: any) => b.market_impact - a.market_impact,
    },
    {
      title: 'Execution Time',
      dataIndex: 'execution_time',
      key: 'execution_time',
      render: (value: number) => `${value.toFixed(1)}s`,
      sorter: (a: any, b: any) => b.execution_time - a.execution_time,
    },
    {
      title: 'Venue',
      dataIndex: 'venue',
      key: 'venue',
      width: 80,
    },
  ];
  
  const venueAnalysisColumns = [
    {
      title: 'Venue',
      dataIndex: 'venue',
      key: 'venue',
    },
    {
      title: 'Trade Count',
      dataIndex: 'trade_count',
      key: 'trade_count',
      sorter: (a: any, b: any) => b.trade_count - a.trade_count,
    },
    {
      title: 'Volume %',
      dataIndex: 'volume_pct',
      key: 'volume_pct',
      render: (value: number) => `${value.toFixed(1)}%`,
      sorter: (a: any, b: any) => b.volume_pct - a.volume_pct,
    },
    {
      title: 'Avg Slippage',
      dataIndex: 'avg_slippage',
      key: 'avg_slippage',
      render: (value: number) => (
        <span style={{ color: value < 0 ? '#52c41a' : '#ff4d4f' }}>
          {value.toFixed(1)}bps
        </span>
      ),
      sorter: (a: any, b: any) => Math.abs(b.avg_slippage) - Math.abs(a.avg_slippage),
    },
    {
      title: 'Market Impact',
      dataIndex: 'avg_market_impact',
      key: 'avg_market_impact',
      render: (value: number) => `${value.toFixed(1)}bps`,
      sorter: (a: any, b: any) => b.avg_market_impact - a.avg_market_impact,
    },
    {
      title: 'Fill Rate',
      dataIndex: 'fill_rate',
      key: 'fill_rate',
      render: (value: number) => (
        <span>
          {value.toFixed(1)}%
          <Progress 
            percent={value} 
            size="small" 
            showInfo={false}
            strokeColor={value > 95 ? '#52c41a' : value > 85 ? '#faad14' : '#ff4d4f'}
          />
        </span>
      ),
      sorter: (a: any, b: any) => b.fill_rate - a.fill_rate,
    },
    {
      title: 'Avg Exec Time',
      dataIndex: 'avg_execution_time',
      key: 'avg_execution_time',
      render: (value: number) => `${value.toFixed(1)}s`,
      sorter: (a: any, b: any) => b.avg_execution_time - a.avg_execution_time,
    },
  ];
  
  return (
    <div className={className} style={{ height }}>
      <Row gutter={[16, 16]}>
        {/* Header with controls */}
        <Col span={24}>
          <Card size="small">
            <Row justify="space-between" align="middle">
              <Col>
                <Space>
                  <ThunderboltOutlined style={{ color: '#1890ff', fontSize: '18px' }} />
                  <Title level={4} style={{ margin: 0 }}>
                    Execution Quality Analysis
                  </Title>
                  {showRealTimeTracking && (
                    <Tag color={isTracking ? 'green' : 'default'}>
                      {isTracking ? 'Live Tracking' : 'Static Analysis'}
                    </Tag>
                  )}
                  {executionAlerts.length > 0 && (
                    <Tag color="red">
                      {executionAlerts.length} Alert{executionAlerts.length > 1 ? 's' : ''}
                    </Tag>
                  )}
                </Space>
              </Col>
              <Col>
                <Space>
                  <RangePicker
                    value={dateRange}
                    onChange={setDateRange}
                    size="small"
                    style={{ width: 240 }}
                  />
                  <Select
                    placeholder="Symbol"
                    value={selectedSymbol}
                    onChange={setSelectedSymbol}
                    style={{ width: 100 }}
                    size="small"
                    allowClear
                  >
                    {/* Dynamic symbol options based on available data */}
                  </Select>
                  <Button 
                    icon={<SettingOutlined />} 
                    size="small"
                    onClick={() => setShowSettings(true)}
                  >
                    Settings
                  </Button>
                  {showRealTimeTracking && (
                    <>
                      {!isTracking ? (
                        <Button 
                          type="primary"
                          icon={<PlayCircleOutlined />} 
                          size="small"
                          onClick={() => startRealTimeTracking(portfolioId || strategyId || '')}
                        >
                          Start Tracking
                        </Button>
                      ) : (
                        <Button 
                          danger
                          icon={<StopOutlined />} 
                          size="small"
                          onClick={stopRealTimeTracking}
                        >
                          Stop Tracking
                        </Button>
                      )}
                    </>
                  )}
                  <Button 
                    icon={<DownloadOutlined />} 
                    size="small"
                    onClick={() => {
                      const blob = new Blob([JSON.stringify(result, null, 2)], { type: 'application/json' });
                      const url = URL.createObjectURL(blob);
                      const a = document.createElement('a');
                      a.href = url;
                      a.download = `execution_analysis_${portfolioId || strategyId}.json`;
                      a.click();
                      URL.revokeObjectURL(url);
                    }}
                  >
                    Export
                  </Button>
                </Space>
              </Col>
            </Row>
          </Card>
        </Col>
        
        {/* Execution Alerts */}
        {executionAlerts.length > 0 && (
          <Col span={24}>
            <Card size="small">
              <Space direction="vertical" style={{ width: '100%' }}>
                {executionAlerts.slice(0, 3).map((alert) => (
                  <Alert
                    key={alert.id}
                    message={alert.message}
                    type={alert.severity === 'high' ? 'error' : 'warning'}
                    showIcon
                    closable
                    style={{ marginBottom: 8 }}
                    description={
                      <Text type="secondary">
                        Trade: {alert.trade_id} | 
                        Current: {alert.current_value.toFixed(3)} | 
                        Threshold: {alert.threshold_value.toFixed(3)}
                      </Text>
                    }
                  />
                ))}
              </Space>
            </Card>
          </Col>
        )}
        
        {/* Current Trade (Real-time) */}
        {currentTrade && (
          <Col span={24}>
            <Card title="Current Trade" size="small">
              <Descriptions size="small" column={6}>
                <Descriptions.Item label="Symbol">{currentTrade.symbol}</Descriptions.Item>
                <Descriptions.Item label="Side">
                  <Tag color={currentTrade.side === 'buy' ? 'green' : 'red'}>
                    {currentTrade.side.toUpperCase()}
                  </Tag>
                </Descriptions.Item>
                <Descriptions.Item label="Quantity">{currentTrade.quantity.toLocaleString()}</Descriptions.Item>
                <Descriptions.Item label="Price">${currentTrade.price.toFixed(4)}</Descriptions.Item>
                <Descriptions.Item label="Status">
                  <Tag color={currentTrade.status === 'filled' ? 'green' : 'processing'}>
                    {currentTrade.status.toUpperCase()}
                  </Tag>
                </Descriptions.Item>
                <Descriptions.Item label="Slippage">
                  <span style={{ color: currentTrade.slippage < 0 ? '#52c41a' : '#ff4d4f' }}>
                    {(currentTrade.slippage * 10000).toFixed(1)}bps
                  </span>
                </Descriptions.Item>
              </Descriptions>
            </Card>
          </Col>
        )}
        
        {/* Summary Statistics */}
        {result && (
          <Col span={24}>
            <Row gutter={[16, 16]}>
              <Col xs={12} sm={6}>
                <Card size="small">
                  <Statistic
                    title="Total Trades"
                    value={result.summary_metrics.total_trades}
                    valueStyle={{ fontSize: '20px' }}
                  />
                </Card>
              </Col>
              <Col xs={12} sm={6}>
                <Card size="small">
                  <Statistic
                    title="Fill Rate"
                    value={result.summary_metrics.fill_rate * 100}
                    precision={1}
                    suffix="%"
                    valueStyle={{ 
                      fontSize: '20px',
                      color: result.summary_metrics.fill_rate > 0.95 ? '#52c41a' : 
                             result.summary_metrics.fill_rate > 0.85 ? '#faad14' : '#ff4d4f'
                    }}
                  />
                </Card>
              </Col>
              <Col xs={12} sm={6}>
                <Card size="small">
                  <Statistic
                    title="Avg Slippage"
                    value={result.slippage_analysis.benchmark_comparison.arrival_price_slippage * 10000}
                    precision={1}
                    suffix="bps"
                    valueStyle={{ 
                      fontSize: '20px',
                      color: Math.abs(result.slippage_analysis.benchmark_comparison.arrival_price_slippage) < 0.005 ? '#52c41a' : '#faad14'
                    }}
                    prefix={result.slippage_analysis.benchmark_comparison.arrival_price_slippage < 0 ? 
                      <RiseOutlined style={{ color: '#52c41a' }} /> : 
                      <FallOutlined style={{ color: '#ff4d4f' }} />}
                  />
                </Card>
              </Col>
              <Col xs={12} sm={6}>
                <Card size="small">
                  <Statistic
                    title="Avg Exec Time"
                    value={result.summary_metrics.avg_execution_time}
                    precision={1}
                    suffix="s"
                    valueStyle={{ fontSize: '20px' }}
                  />
                </Card>
              </Col>
            </Row>
          </Col>
        )}
        
        {/* Main Content Tabs */}
        <Col span={24}>
          <Card>
            <Tabs activeKey={activeTab} onChange={setActiveTab}>
              <TabPane tab={<Space><BarChartOutlined />Overview</Space>} key="overview">
                <Row gutter={[16, 16]}>
                  {/* Implementation Shortfall Breakdown */}
                  {result && result.slippage_analysis.implementation_shortfall && (
                    <Col xs={24} lg={12}>
                      <Card title="Implementation Shortfall Breakdown" size="small">
                        <Space direction="vertical" style={{ width: '100%' }}>
                          <div>
                            <Text>Total IS: </Text>
                            <Text strong style={{ color: '#1890ff' }}>
                              {(result.slippage_analysis.implementation_shortfall.total_is * 10000).toFixed(1)}bps
                            </Text>
                          </div>
                          <Progress 
                            percent={100} 
                            success={{ 
                              percent: (Math.abs(result.slippage_analysis.implementation_shortfall.market_impact) / Math.abs(result.slippage_analysis.implementation_shortfall.total_is)) * 100 
                            }}
                            format={() => 'Market Impact'}
                            strokeColor="#ff4d4f"
                          />
                          <Progress 
                            percent={(Math.abs(result.slippage_analysis.implementation_shortfall.timing_risk) / Math.abs(result.slippage_analysis.implementation_shortfall.total_is)) * 100}
                            format={() => 'Timing Risk'}
                            strokeColor="#faad14"
                          />
                          <Progress 
                            percent={(Math.abs(result.slippage_analysis.implementation_shortfall.delay_cost) / Math.abs(result.slippage_analysis.implementation_shortfall.total_is)) * 100}
                            format={() => 'Delay Cost'}
                            strokeColor="#1890ff"
                          />
                        </Space>
                      </Card>
                    </Col>
                  )}
                  
                  {/* Performance Attribution */}
                  {result && result.performance_attribution && (
                    <Col xs={24} lg={12}>
                      <Card title="Execution Performance Impact" size="small">
                        <Space direction="vertical" style={{ width: '100%' }}>
                          <Statistic
                            title="Execution Alpha"
                            value={result.performance_attribution.alpha_from_execution * 10000}
                            precision={1}
                            suffix="bps"
                            valueStyle={{ color: result.performance_attribution.alpha_from_execution >= 0 ? '#52c41a' : '#ff4d4f' }}
                          />
                          <Statistic
                            title="Cost Impact on Returns"
                            value={result.performance_attribution.execution_cost_as_pct_of_returns * 100}
                            precision={2}
                            suffix="%"
                            valueStyle={{ color: '#faad14' }}
                          />
                          <Statistic
                            title="Sharpe Impact"
                            value={result.performance_attribution.execution_sharpe_impact}
                            precision={3}
                            valueStyle={{ color: result.performance_attribution.execution_sharpe_impact >= 0 ? '#52c41a' : '#ff4d4f' }}
                          />
                        </Space>
                      </Card>
                    </Col>
                  )}
                  
                  <Col span={24}>
                    <Card title="Slippage Distribution" size="small">
                      {slippageDistributionData.length > 0 ? (
                        <Column {...slippageHistogramConfig} />
                      ) : (
                        <div style={{ height: 300, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                          No slippage data available
                        </div>
                      )}
                    </Card>
                  </Col>
                </Row>
              </TabPane>
              
              <TabPane tab={<Space><LineChartOutlined />Timing Analysis</Space>} key="timing">
                <Row gutter={[16, 16]}>
                  <Col span={24}>
                    <Card title="Intraday Execution Patterns" size="small">
                      {timingAnalysisData.length > 0 ? (
                        <Column {...timingPatternConfig} />
                      ) : (
                        <div style={{ height: 300, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                          No timing data available
                        </div>
                      )}
                    </Card>
                  </Col>
                  
                  {result && result.timing_analysis.time_to_fill && (
                    <Col span={24}>
                      <Card title="Time to Fill Statistics" size="small">
                        <Row gutter={[16, 16]}>
                          <Col xs={12} sm={6}>
                            <Statistic
                              title="Mean"
                              value={result.timing_analysis.time_to_fill.mean}
                              precision={1}
                              suffix="s"
                            />
                          </Col>
                          <Col xs={12} sm={6}>
                            <Statistic
                              title="Median"
                              value={result.timing_analysis.time_to_fill.median}
                              precision={1}
                              suffix="s"
                            />
                          </Col>
                          <Col xs={12} sm={6}>
                            <Statistic
                              title="75th Percentile"
                              value={result.timing_analysis.time_to_fill.percentile_75}
                              precision={1}
                              suffix="s"
                            />
                          </Col>
                          <Col xs={12} sm={6}>
                            <Statistic
                              title="95th Percentile"
                              value={result.timing_analysis.time_to_fill.percentile_95}
                              precision={1}
                              suffix="s"
                            />
                          </Col>
                        </Row>
                      </Card>
                    </Col>
                  )}
                </Row>
              </TabPane>
              
              <TabPane tab={<Space><DotChartOutlined />Market Impact</Space>} key="impact">
                <Row gutter={[16, 16]}>
                  <Col span={24}>
                    <Card title="Trade Size vs Slippage Analysis" size="small">
                      {tradeScatterData.length > 0 ? (
                        <Scatter {...tradeScatterConfig} />
                      ) : (
                        <div style={{ height: 400, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                          No trade data available
                        </div>
                      )}
                    </Card>
                  </Col>
                  
                  {result && result.slippage_analysis.price_impact && (
                    <Col span={24}>
                      <Card title="Price Impact Decomposition" size="small">
                        <Row gutter={[16, 16]}>
                          <Col xs={8}>
                            <Statistic
                              title="Temporary Impact"
                              value={result.slippage_analysis.price_impact.temporary_impact * 10000}
                              precision={1}
                              suffix="bps"
                              valueStyle={{ color: '#1890ff' }}
                            />
                          </Col>
                          <Col xs={8}>
                            <Statistic
                              title="Permanent Impact"
                              value={result.slippage_analysis.price_impact.permanent_impact * 10000}
                              precision={1}
                              suffix="bps"
                              valueStyle={{ color: '#ff4d4f' }}
                            />
                          </Col>
                          <Col xs={8}>
                            <Statistic
                              title="Total Impact"
                              value={result.slippage_analysis.price_impact.total_impact * 10000}
                              precision={1}
                              suffix="bps"
                              valueStyle={{ color: '#faad14' }}
                            />
                          </Col>
                        </Row>
                      </Card>
                    </Col>
                  )}
                </Row>
              </TabPane>
              
              <TabPane tab={<Space><BarChartOutlined />Venue Analysis</Space>} key="venue">
                <Row gutter={[16, 16]}>
                  <Col span={24}>
                    <Card title="Venue Performance Comparison" size="small">
                      {venueAnalysisData.length > 0 ? (
                        <Column {...venueComparisonConfig} />
                      ) : (
                        <div style={{ height: 300, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                          No venue data available
                        </div>
                      )}
                    </Card>
                  </Col>
                  
                  <Col span={24}>
                    <Table
                      dataSource={venueAnalysisData}
                      columns={venueAnalysisColumns}
                      rowKey="venue"
                      size="small"
                      pagination={false}
                      scroll={{ x: 800 }}
                    />
                  </Col>
                </Row>
              </TabPane>
              
              <TabPane tab={<Space><TableOutlined />Trade Details</Space>} key="trades">
                {result && result.trade_details && (
                  <Table
                    dataSource={result.trade_details}
                    columns={tradeDetailColumns}
                    rowKey="trade_id"
                    size="small"
                    scroll={{ x: 1200, y: 500 }}
                    pagination={{
                      pageSize: 50,
                      showSizeChanger: true,
                      showQuickJumper: true,
                    }}
                    loading={isAnalyzing}
                  />
                )}
              </TabPane>
            </Tabs>
          </Card>
        </Col>
      </Row>
      
      {/* Settings Modal */}
      <Modal
        title="Execution Analysis Settings"
        open={showSettings}
        onOk={() => setShowSettings(false)}
        onCancel={() => setShowSettings(false)}
        width={500}
      >
        <Space direction="vertical" style={{ width: '100%' }}>
          <div>
            <Text strong>Benchmark Type</Text>
            <br />
            <Select 
              value={benchmarkType} 
              onChange={setBenchmarkType}
              style={{ width: '100%', marginTop: 8 }}
            >
              <Option value="arrival">Arrival Price</Option>
              <Option value="vwap">VWAP</Option>
              <Option value="twap">TWAP</Option>
              <Option value="close">Close Price</Option>
            </Select>
          </div>
          
          <div>
            <Space>
              <Switch 
                checked={includeMarketImpact} 
                onChange={setIncludeMarketImpact}
              />
              <Text>Include Market Impact Analysis</Text>
            </Space>
          </div>
          
          <div>
            <Space>
              <Switch 
                checked={includeTimingAnalysis} 
                onChange={setIncludeTimingAnalysis}
              />
              <Text>Include Timing Analysis</Text>
            </Space>
          </div>
        </Space>
      </Modal>
    </div>
  );
};

export default ExecutionQuality;