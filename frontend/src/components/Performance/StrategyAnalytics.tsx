/**
 * StrategyAnalytics Component - Sprint 3 Integration
 * Strategy performance comparison and benchmarking with advanced analytics
 */

import React, { useState, useEffect, useMemo } from 'react';
import { 
  Card, Row, Col, Table, Tabs, Select, Button, Space, Typography, Statistic, 
  Progress, Tag, Tooltip, Alert, Modal, Switch, InputNumber, Transfer, List
} from 'antd';
import { 
  TrophyOutlined,
  BarChartOutlined,
  LineChartOutlined,
  RadarChartOutlined,
  TableOutlined,
  DiffOutlined,
  SettingOutlined,
  PlayCircleOutlined,
  StopOutlined,
  DownloadOutlined,
  RiseOutlined,
  FallOutlined,
  MinusOutlined,
  EyeOutlined,
  StarOutlined
} from '@ant-design/icons';
import { Line, Column, Radar, Heatmap, Waterfall, Treemap } from '@ant-design/plots';
import useStrategyAnalytics from '../../hooks/analytics/useStrategyAnalytics';

const { Title, Text } = Typography;
const { TabPane } = Tabs;
const { Option } = Select;

interface StrategyAnalyticsProps {
  initialStrategyIds?: string[];
  className?: string;
  height?: number;
  showLiveTracking?: boolean;
  defaultBenchmark?: string;
  enablePeerComparison?: boolean;
}

const StrategyAnalytics: React.FC<StrategyAnalyticsProps> = ({
  initialStrategyIds = [],
  className,
  height = 1000,
  showLiveTracking = true,
  defaultBenchmark = 'SPY',
  enablePeerComparison = true,
}) => {
  // State
  const [activeTab, setActiveTab] = useState('overview');
  const [selectedStrategies, setSelectedStrategies] = useState<string[]>(initialStrategyIds);
  const [benchmark, setBenchmark] = useState(defaultBenchmark);
  const [availableStrategies, setAvailableStrategies] = useState<string[]>([]);
  const [showStrategySelector, setShowStrategySelector] = useState(false);
  const [showSettings, setShowSettings] = useState(false);
  const [includeFactor, setIncludeFactor] = useState(true);
  const [includeRegime, setIncludeRegime] = useState(true);
  const [includeAttribution, setIncludeAttribution] = useState(true);
  const [comparisonType, setComparisonType] = useState<'peer' | 'benchmark' | 'historical'>('benchmark');
  const [rollingWindow, setRollingWindow] = useState(252);
  
  const {
    result,
    isAnalyzing,
    error,
    livePerformance,
    strategyAlerts,
    isTracking,
    analyzeStrategies,
    startLiveTracking,
    stopLiveTracking,
    compareStrategies,
    getPeerAnalysis,
    forecastPerformance,
    getStyleAnalysis,
  } = useStrategyAnalytics({
    autoRefresh: true,
    refreshInterval: 60000,
    enableLiveTracking: showLiveTracking,
    benchmarkStrategy: benchmark,
    alertThresholds: {
      drawdown_threshold: 0.15,
      correlation_threshold: 0.8,
      performance_threshold: 0.5,
    },
  });
  
  // Fetch strategy analysis when parameters change
  useEffect(() => {
    if (selectedStrategies.length > 0) {
      analyzeStrategies({
        strategy_ids: selectedStrategies,
        benchmark,
        include_attribution: includeAttribution,
        include_factor_analysis: includeFactor,
        include_regime_analysis: includeRegime,
        comparison_type: comparisonType,
        rolling_window: rollingWindow,
      });
    }
  }, [selectedStrategies, benchmark, includeAttribution, includeFactor, includeRegime, comparisonType, rollingWindow]);
  
  // Performance comparison data
  const performanceComparisonData = useMemo(() => {
    if (!result || !result.strategy_summaries) return [];
    
    return result.strategy_summaries.map(strategy => ({
      strategy: strategy.strategy_name,
      total_return: strategy.total_return * 100,
      sharpe_ratio: strategy.sharpe_ratio,
      max_drawdown: Math.abs(strategy.max_drawdown) * 100,
      volatility: strategy.volatility * 100,
      win_rate: strategy.win_rate * 100,
      profit_factor: strategy.profit_factor,
      status: strategy.status,
    }));
  }, [result]);
  
  // Rolling metrics data for time series
  const rollingMetricsData = useMemo(() => {
    if (!result || !result.rolling_analytics) return [];
    
    return result.rolling_analytics.map(point => {
      const strategyReturns: any = { date: point.date };
      selectedStrategies.forEach((strategyId, index) => {
        const strategyName = result.strategy_summaries[index]?.strategy_name || strategyId;
        strategyReturns[strategyName] = (point.strategy_returns[strategyId] || 0) * 100;
      });
      strategyReturns.benchmark = point.benchmark_return * 100;
      return strategyReturns;
    });
  }, [result, selectedStrategies]);
  
  // Drawdown analysis data
  const drawdownData = useMemo(() => {
    if (!result || !result.drawdown_analysis) return [];
    
    return result.drawdown_analysis.map(dd => ({
      strategy: result.strategy_summaries.find(s => s.strategy_id === dd.strategy_id)?.strategy_name || dd.strategy_id,
      current_drawdown: Math.abs(dd.current_drawdown) * 100,
      max_drawdown: Math.abs(dd.max_drawdown) * 100,
      avg_drawdown: Math.abs(dd.avg_drawdown) * 100,
      max_duration: dd.drawdown_duration.max,
      avg_recovery_time: dd.recovery_analysis.avg_recovery_time,
      recovery_rate: dd.recovery_analysis.recovery_rate * 100,
    }));
  }, [result]);
  
  // Factor exposure radar chart data
  const factorExposureData = useMemo(() => {
    if (!result || !result.factor_analysis) return [];
    
    return result.factor_analysis.map(fa => {
      const strategy = result.strategy_summaries.find(s => s.strategy_id === fa.strategy_id);
      return {
        strategy: strategy?.strategy_name || fa.strategy_id,
        market_beta: fa.factor_exposures.market_beta,
        size_factor: fa.factor_exposures.size_factor,
        value_factor: fa.factor_exposures.value_factor,
        momentum_factor: fa.factor_exposures.momentum_factor,
        quality_factor: fa.factor_exposures.quality_factor,
        volatility_factor: fa.factor_exposures.volatility_factor,
      };
    });
  }, [result]);
  
  // Correlation heatmap data
  const correlationData = useMemo(() => {
    if (!result || !result.comparative_analysis.correlation_matrix) return [];
    
    const { strategies, correlations } = result.comparative_analysis.correlation_matrix;
    const heatmapData = [];
    
    for (let i = 0; i < strategies.length; i++) {
      for (let j = 0; j < strategies.length; j++) {
        heatmapData.push({
          x: strategies[i],
          y: strategies[j],
          value: correlations[i][j],
        });
      }
    }
    
    return heatmapData;
  }, [result]);
  
  // Chart configurations
  const performanceChartConfig = {
    data: performanceComparisonData,
    xField: 'strategy',
    yField: 'total_return',
    height: 300,
    color: '#1890ff',
    label: {
      position: 'top' as const,
      formatter: (data: any) => `${data.total_return.toFixed(1)}%`,
    },
    xAxis: {
      label: {
        autoRotate: true,
      },
    },
    yAxis: {
      label: {
        formatter: (v: string) => `${v}%`,
      },
    },
    tooltip: {
      formatter: (data: any) => [
        { name: 'Total Return', value: `${data.total_return.toFixed(2)}%` },
        { name: 'Sharpe Ratio', value: data.sharpe_ratio.toFixed(3) },
        { name: 'Max Drawdown', value: `${data.max_drawdown.toFixed(1)}%` },
        { name: 'Win Rate', value: `${data.win_rate.toFixed(1)}%` },
      ],
    },
  };
  
  const rollingMetricsConfig = {
    data: rollingMetricsData,
    xField: 'date',
    yField: 'value',
    seriesField: 'strategy',
    height: 400,
    smooth: true,
    xAxis: {
      type: 'time' as const,
    },
    yAxis: {
      label: {
        formatter: (v: string) => `${v}%`,
      },
    },
    slider: {
      start: 0.8,
      end: 1.0,
    },
  };
  
  const factorRadarConfig = {
    data: factorExposureData.length > 0 ? factorExposureData[0] : {},
    xField: 'factor',
    yField: 'exposure',
    height: 400,
    area: {
      visible: true,
      alpha: 0.3,
    },
    point: {
      visible: true,
      size: 4,
    },
  };
  
  const correlationHeatmapConfig = {
    data: correlationData,
    xField: 'x',
    yField: 'y',
    colorField: 'value',
    height: 400,
    color: ['#1890ff', '#ffffff', '#ff4d4f'],
    meta: {
      value: {
        min: -1,
        max: 1,
      },
    },
    tooltip: {
      formatter: (data: any) => ({
        name: `${data.x} vs ${data.y}`,
        value: data.value.toFixed(3),
      }),
    },
  };
  
  // Table columns
  const strategySummaryColumns = [
    {
      title: 'Strategy',
      dataIndex: 'strategy_name',
      key: 'strategy_name',
      width: 150,
      fixed: 'left' as const,
    },
    {
      title: 'Status',
      dataIndex: 'status',
      key: 'status',
      render: (status: string) => (
        <Tag color={status === 'active' ? 'green' : status === 'paused' ? 'orange' : 'red'}>
          {status.toUpperCase()}
        </Tag>
      ),
      width: 80,
    },
    {
      title: 'Total Return',
      dataIndex: 'total_return',
      key: 'total_return',
      render: (value: number) => (
        <span style={{ color: value >= 0 ? '#52c41a' : '#ff4d4f', fontWeight: 'bold' }}>
          {value >= 0 ? '+' : ''}{(value * 100).toFixed(2)}%
        </span>
      ),
      sorter: (a: any, b: any) => b.total_return - a.total_return,
    },
    {
      title: 'Annualized',
      dataIndex: 'annualized_return',
      key: 'annualized_return',
      render: (value: number) => (
        <span style={{ color: value >= 0 ? '#52c41a' : '#ff4d4f' }}>
          {value >= 0 ? '+' : ''}{(value * 100).toFixed(2)}%
        </span>
      ),
      sorter: (a: any, b: any) => b.annualized_return - a.annualized_return,
    },
    {
      title: 'Volatility',
      dataIndex: 'volatility',
      key: 'volatility',
      render: (value: number) => `${(value * 100).toFixed(2)}%`,
      sorter: (a: any, b: any) => b.volatility - a.volatility,
    },
    {
      title: 'Sharpe',
      dataIndex: 'sharpe_ratio',
      key: 'sharpe_ratio',
      render: (value: number) => (
        <span style={{ 
          color: value > 1 ? '#52c41a' : value > 0.5 ? '#faad14' : '#ff4d4f',
          fontWeight: 'bold'
        }}>
          {value.toFixed(3)}
        </span>
      ),
      sorter: (a: any, b: any) => b.sharpe_ratio - a.sharpe_ratio,
    },
    {
      title: 'Sortino',
      dataIndex: 'sortino_ratio',
      key: 'sortino_ratio',
      render: (value: number) => value.toFixed(3),
      sorter: (a: any, b: any) => b.sortino_ratio - a.sortino_ratio,
    },
    {
      title: 'Calmar',
      dataIndex: 'calmar_ratio',
      key: 'calmar_ratio',
      render: (value: number) => value.toFixed(3),
      sorter: (a: any, b: any) => b.calmar_ratio - a.calmar_ratio,
    },
    {
      title: 'Max DD',
      dataIndex: 'max_drawdown',
      key: 'max_drawdown',
      render: (value: number) => (
        <span style={{ color: Math.abs(value) > 0.2 ? '#ff4d4f' : Math.abs(value) > 0.1 ? '#faad14' : '#52c41a' }}>
          {(Math.abs(value) * 100).toFixed(2)}%
        </span>
      ),
      sorter: (a: any, b: any) => Math.abs(a.max_drawdown) - Math.abs(b.max_drawdown),
    },
    {
      title: 'Win Rate',
      dataIndex: 'win_rate',
      key: 'win_rate',
      render: (value: number) => `${(value * 100).toFixed(1)}%`,
      sorter: (a: any, b: any) => b.win_rate - a.win_rate,
    },
    {
      title: 'Profit Factor',
      dataIndex: 'profit_factor',
      key: 'profit_factor',
      render: (value: number) => value.toFixed(2),
      sorter: (a: any, b: any) => b.profit_factor - a.profit_factor,
    },
    {
      title: 'AUM',
      dataIndex: 'current_aum',
      key: 'current_aum',
      render: (value: number) => `$${(value / 1000000).toFixed(1)}M`,
      sorter: (a: any, b: any) => b.current_aum - a.current_aum,
    },
  ];
  
  const rankingColumns = [
    {
      title: 'Rank',
      dataIndex: 'rank',
      key: 'rank',
      render: (rank: number) => (
        <Tag color={rank <= 3 ? 'gold' : rank <= 5 ? 'silver' : 'default'}>
          #{rank}
        </Tag>
      ),
      width: 70,
    },
    {
      title: 'Strategy',
      dataIndex: 'strategy_id',
      key: 'strategy_id',
      render: (strategyId: string) => {
        const strategy = result?.strategy_summaries.find(s => s.strategy_id === strategyId);
        return strategy?.strategy_name || strategyId;
      },
    },
    {
      title: 'Percentile',
      dataIndex: 'percentile',
      key: 'percentile',
      render: (percentile: number) => `${(percentile * 100).toFixed(1)}th`,
      sorter: (a: any, b: any) => b.percentile - a.percentile,
    },
    {
      title: 'Score',
      dataIndex: 'score',
      key: 'score',
      render: (score: number) => (
        <Progress 
          percent={score * 100} 
          size="small" 
          format={percent => `${percent?.toFixed(0)}`}
          strokeColor={score > 0.8 ? '#52c41a' : score > 0.6 ? '#faad14' : '#ff4d4f'}
        />
      ),
      sorter: (a: any, b: any) => b.score - a.score,
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
                  <TrophyOutlined style={{ color: '#faad14', fontSize: '18px' }} />
                  <Title level={4} style={{ margin: 0 }}>
                    Strategy Analytics
                  </Title>
                  {showLiveTracking && (
                    <Tag color={isTracking ? 'green' : 'default'}>
                      {isTracking ? 'Live Tracking' : 'Static Analysis'}
                    </Tag>
                  )}
                  {strategyAlerts.length > 0 && (
                    <Tag color="red">
                      {strategyAlerts.length} Alert{strategyAlerts.length > 1 ? 's' : ''}
                    </Tag>
                  )}
                  <Text type="secondary">
                    ({selectedStrategies.length} strategies selected)
                  </Text>
                </Space>
              </Col>
              <Col>
                <Space>
                  <Button 
                    icon={<DiffOutlined />} 
                    size="small"
                    onClick={() => setShowStrategySelector(true)}
                  >
                    Select Strategies
                  </Button>
                  <Select
                    value={benchmark}
                    onChange={setBenchmark}
                    style={{ width: 100 }}
                    size="small"
                  >
                    <Option value="SPY">SPY</Option>
                    <Option value="QQQ">QQQ</Option>
                    <Option value="IWM">IWM</Option>
                    <Option value="VTI">VTI</Option>
                  </Select>
                  <Button 
                    icon={<SettingOutlined />} 
                    size="small"
                    onClick={() => setShowSettings(true)}
                  >
                    Settings
                  </Button>
                  {showLiveTracking && (
                    <>
                      {!isTracking ? (
                        <Button 
                          type="primary"
                          icon={<PlayCircleOutlined />} 
                          size="small"
                          onClick={() => startLiveTracking(selectedStrategies)}
                        >
                          Start Tracking
                        </Button>
                      ) : (
                        <Button 
                          danger
                          icon={<StopOutlined />} 
                          size="small"
                          onClick={stopLiveTracking}
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
                      a.download = `strategy_analysis_${selectedStrategies.join('_')}.json`;
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
        
        {/* Strategy Alerts */}
        {strategyAlerts.length > 0 && (
          <Col span={24}>
            <Card size="small">
              <Space direction="vertical" style={{ width: '100%' }}>
                {strategyAlerts.slice(0, 3).map((alert) => (
                  <Alert
                    key={alert.id}
                    message={alert.message}
                    type={alert.severity === 'high' ? 'error' : 'warning'}
                    showIcon
                    closable
                    style={{ marginBottom: 8 }}
                    description={
                      <Text type="secondary">
                        Strategy: {alert.strategy_id} | 
                        Current: {alert.current_value.toFixed(3)} | 
                        Threshold: {alert.threshold_value.toFixed(3)} | 
                        Time: {alert.timestamp.toLocaleTimeString()}
                      </Text>
                    }
                  />
                ))}
              </Space>
            </Card>
          </Col>
        )}
        
        {/* Live Performance Summary */}
        {Object.keys(livePerformance).length > 0 && (
          <Col span={24}>
            <Card title="Live Performance" size="small">
              <Row gutter={[16, 8]}>
                {Object.values(livePerformance).map((perf) => (
                  <Col xs={24} sm={12} md={8} lg={6} key={perf.strategy_id}>
                    <Card size="small" style={{ textAlign: 'center' }}>
                      <div>
                        <Text strong>{perf.strategy_id}</Text>
                        <br />
                        <Statistic
                          value={perf.current_return * 100}
                          precision={2}
                          suffix="%"
                          valueStyle={{ 
                            fontSize: '14px', 
                            color: perf.current_return >= 0 ? '#52c41a' : '#ff4d4f' 
                          }}
                          prefix={
                            perf.current_return > 0 ? <RiseOutlined /> :
                            perf.current_return < 0 ? <FallOutlined /> : <MinusOutlined />
                          }
                        />
                        <Text type="secondary" style={{ fontSize: '12px' }}>
                          DD: {(Math.abs(perf.current_drawdown) * 100).toFixed(1)}% | 
                          SR: {perf.current_sharpe.toFixed(2)}
                        </Text>
                      </div>
                    </Card>
                  </Col>
                ))}
              </Row>
            </Card>
          </Col>
        )}
        
        {/* Main Content Tabs */}
        <Col span={24}>
          <Card>
            <Tabs activeKey={activeTab} onChange={setActiveTab}>
              <TabPane tab={<Space><BarChartOutlined />Overview</Space>} key="overview">
                <Row gutter={[16, 16]}>
                  <Col span={24}>
                    <Card title="Performance Comparison" size="small">
                      {performanceComparisonData.length > 0 ? (
                        <Column {...performanceChartConfig} />
                      ) : (
                        <div style={{ height: 300, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                          No performance data available
                        </div>
                      )}
                    </Card>
                  </Col>
                  
                  {result && result.comparative_analysis.performance_ranking && (
                    <Col xs={24} lg={12}>
                      <Card title="Performance Ranking" size="small">
                        <Table
                          dataSource={result.comparative_analysis.performance_ranking}
                          columns={rankingColumns}
                          size="small"
                          pagination={false}
                          rowKey="strategy_id"
                        />
                      </Card>
                    </Col>
                  )}
                  
                  {result && result.comparative_analysis.correlation_matrix && (
                    <Col xs={24} lg={12}>
                      <Card title="Strategy Correlation" size="small">
                        <div>
                          <Text>Average Correlation: </Text>
                          <Text strong style={{ color: result.comparative_analysis.correlation_matrix.avg_correlation > 0.7 ? '#ff4d4f' : '#52c41a' }}>
                            {(result.comparative_analysis.correlation_matrix.avg_correlation * 100).toFixed(1)}%
                          </Text>
                          <br />
                          <Text>Diversification Benefit: </Text>
                          <Text strong style={{ color: '#1890ff' }}>
                            {(result.comparative_analysis.correlation_matrix.diversification_benefit * 100).toFixed(1)}%
                          </Text>
                        </div>
                        {correlationData.length > 0 && (
                          <div style={{ marginTop: 16 }}>
                            <Heatmap {...correlationHeatmapConfig} />
                          </div>
                        )}
                      </Card>
                    </Col>
                  )}
                </Row>
              </TabPane>
              
              <TabPane tab={<Space><LineChartOutlined />Time Series</Space>} key="timeseries">
                <Row gutter={[16, 16]}>
                  <Col span={24}>
                    <Card title="Rolling Performance Metrics" size="small">
                      {rollingMetricsData.length > 0 ? (
                        <Line {...rollingMetricsConfig} />
                      ) : (
                        <div style={{ height: 400, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                          No time series data available
                        </div>
                      )}
                    </Card>
                  </Col>
                  
                  {drawdownData.length > 0 && (
                    <Col span={24}>
                      <Card title="Drawdown Analysis" size="small">
                        <Table
                          dataSource={drawdownData}
                          size="small"
                          pagination={false}
                          columns={[
                            {
                              title: 'Strategy',
                              dataIndex: 'strategy',
                              key: 'strategy',
                            },
                            {
                              title: 'Current DD',
                              dataIndex: 'current_drawdown',
                              key: 'current_drawdown',
                              render: (value: number) => (
                                <span style={{ color: value > 10 ? '#ff4d4f' : value > 5 ? '#faad14' : '#52c41a' }}>
                                  {value.toFixed(2)}%
                                </span>
                              ),
                              sorter: (a: any, b: any) => b.current_drawdown - a.current_drawdown,
                            },
                            {
                              title: 'Max DD',
                              dataIndex: 'max_drawdown',
                              key: 'max_drawdown',
                              render: (value: number) => `${value.toFixed(2)}%`,
                              sorter: (a: any, b: any) => b.max_drawdown - a.max_drawdown,
                            },
                            {
                              title: 'Avg DD',
                              dataIndex: 'avg_drawdown',
                              key: 'avg_drawdown',
                              render: (value: number) => `${value.toFixed(2)}%`,
                            },
                            {
                              title: 'Max Duration',
                              dataIndex: 'max_duration',
                              key: 'max_duration',
                              render: (value: number) => `${value} days`,
                            },
                            {
                              title: 'Avg Recovery',
                              dataIndex: 'avg_recovery_time',
                              key: 'avg_recovery_time',
                              render: (value: number) => `${value.toFixed(0)} days`,
                            },
                            {
                              title: 'Recovery Rate',
                              dataIndex: 'recovery_rate',
                              key: 'recovery_rate',
                              render: (value: number) => `${value.toFixed(1)}%`,
                            },
                          ]}
                        />
                      </Card>
                    </Col>
                  )}
                </Row>
              </TabPane>
              
              <TabPane tab={<Space><RadarChartOutlined />Factor Analysis</Space>} key="factors">
                <Row gutter={[16, 16]}>
                  {factorExposureData.length > 0 && (
                    <Col span={24}>
                      <Card title="Factor Exposures" size="small">
                        <Row gutter={[16, 16]}>
                          {factorExposureData.map((strategy, index) => (
                            <Col xs={24} lg={12} key={index}>
                              <Card size="small" title={strategy.strategy}>
                                <Radar
                                  data={[
                                    { factor: 'Market Beta', exposure: strategy.market_beta },
                                    { factor: 'Size', exposure: strategy.size_factor },
                                    { factor: 'Value', exposure: strategy.value_factor },
                                    { factor: 'Momentum', exposure: strategy.momentum_factor },
                                    { factor: 'Quality', exposure: strategy.quality_factor },
                                    { factor: 'Volatility', exposure: strategy.volatility_factor },
                                  ]}
                                  xField="factor"
                                  yField="exposure"
                                  height={250}
                                  area={{ visible: true, alpha: 0.3 }}
                                  point={{ visible: true, size: 4 }}
                                />
                              </Card>
                            </Col>
                          ))}
                        </Row>
                      </Card>
                    </Col>
                  )}
                  
                  {result && result.factor_analysis && (
                    <Col span={24}>
                      <Card title="Factor Attribution" size="small">
                        {result.factor_analysis.map((fa, index) => {
                          const strategy = result.strategy_summaries.find(s => s.strategy_id === fa.strategy_id);
                          return (
                            <div key={index} style={{ marginBottom: 24 }}>
                              <Title level={5}>{strategy?.strategy_name || fa.strategy_id}</Title>
                              <Row gutter={[16, 8]}>
                                <Col xs={8}>
                                  <Statistic
                                    title="Pure Alpha"
                                    value={fa.alpha_decomposition.pure_alpha * 100}
                                    precision={3}
                                    suffix="%"
                                    valueStyle={{ color: fa.alpha_decomposition.pure_alpha >= 0 ? '#52c41a' : '#ff4d4f' }}
                                  />
                                </Col>
                                <Col xs={8}>
                                  <Statistic
                                    title="Factor Alpha"
                                    value={fa.alpha_decomposition.factor_alpha * 100}
                                    precision={3}
                                    suffix="%"
                                    valueStyle={{ color: fa.alpha_decomposition.factor_alpha >= 0 ? '#52c41a' : '#ff4d4f' }}
                                  />
                                </Col>
                                <Col xs={8}>
                                  <Statistic
                                    title="Total Alpha"
                                    value={fa.alpha_decomposition.total_alpha * 100}
                                    precision={3}
                                    suffix="%"
                                    valueStyle={{ color: fa.alpha_decomposition.total_alpha >= 0 ? '#52c41a' : '#ff4d4f' }}
                                  />
                                </Col>
                              </Row>
                            </div>
                          );
                        })}
                      </Card>
                    </Col>
                  )}
                </Row>
              </TabPane>
              
              <TabPane tab={<Space><TableOutlined />Detailed View</Space>} key="detailed">
                {result && result.strategy_summaries && (
                  <Table
                    dataSource={result.strategy_summaries}
                    columns={strategySummaryColumns}
                    rowKey="strategy_id"
                    size="small"
                    scroll={{ x: 1500, y: 500 }}
                    pagination={false}
                    loading={isAnalyzing}
                  />
                )}
              </TabPane>
            </Tabs>
          </Card>
        </Col>
      </Row>
      
      {/* Strategy Selector Modal */}
      <Modal
        title="Select Strategies"
        open={showStrategySelector}
        onOk={() => setShowStrategySelector(false)}
        onCancel={() => setShowStrategySelector(false)}
        width={600}
      >
        <List
          dataSource={availableStrategies}
          renderItem={(strategyId) => (
            <List.Item>
              <Button
                type={selectedStrategies.includes(strategyId) ? 'primary' : 'default'}
                onClick={() => {
                  if (selectedStrategies.includes(strategyId)) {
                    setSelectedStrategies(selectedStrategies.filter(id => id !== strategyId));
                  } else {
                    setSelectedStrategies([...selectedStrategies, strategyId]);
                  }
                }}
                style={{ width: '100%', textAlign: 'left' }}
              >
                {strategyId}
              </Button>
            </List.Item>
          )}
        />
      </Modal>
      
      {/* Settings Modal */}
      <Modal
        title="Strategy Analytics Settings"
        open={showSettings}
        onOk={() => setShowSettings(false)}
        onCancel={() => setShowSettings(false)}
        width={500}
      >
        <Space direction="vertical" style={{ width: '100%' }}>
          <div>
            <Text strong>Comparison Type</Text>
            <br />
            <Select 
              value={comparisonType} 
              onChange={setComparisonType}
              style={{ width: '100%', marginTop: 8 }}
            >
              <Option value="benchmark">Benchmark</Option>
              <Option value="peer">Peer Comparison</Option>
              <Option value="historical">Historical</Option>
            </Select>
          </div>
          
          <div>
            <Text strong>Rolling Window (days)</Text>
            <br />
            <InputNumber 
              value={rollingWindow} 
              onChange={(value) => setRollingWindow(value || 252)}
              min={30}
              max={1260}
              style={{ width: '100%', marginTop: 8 }}
            />
          </div>
          
          <div>
            <Space direction="vertical">
              <div>
                <Switch 
                  checked={includeAttribution} 
                  onChange={setIncludeAttribution}
                />
                <Text style={{ marginLeft: 8 }}>Include Performance Attribution</Text>
              </div>
              
              <div>
                <Switch 
                  checked={includeFactor} 
                  onChange={setIncludeFactor}
                />
                <Text style={{ marginLeft: 8 }}>Include Factor Analysis</Text>
              </div>
              
              <div>
                <Switch 
                  checked={includeRegime} 
                  onChange={setIncludeRegime}
                />
                <Text style={{ marginLeft: 8 }}>Include Regime Analysis</Text>
              </div>
            </Space>
          </div>
        </Space>
      </Modal>
    </div>
  );
};

export default StrategyAnalytics;