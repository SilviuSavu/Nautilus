import React, { useState, useEffect, useRef, useMemo, useCallback, memo } from 'react';
import {
  Card,
  Space,
  Typography,
  Select,
  Button,
  Tooltip,
  Badge,
  Row,
  Col,
  Statistic,
  Switch,
  Dropdown,
  Menu,
  Progress,
  Table,
  Tag,
  Alert,
  Tabs,
  DatePicker,
  Divider
} from 'antd';
import {
  LineChartOutlined,
  BarChartOutlined,
  TrophyOutlined,
  SettingOutlined,
  ExportOutlined,
  FullscreenOutlined,
  PlayCircleOutlined,
  PauseCircleOutlined,
  ReloadOutlined,
  CompareOutlined
} from '@ant-design/icons';
import * as d3 from 'd3';
import moment from 'moment';
import { useStrategyAnalytics } from '../../hooks/analytics/useStrategyAnalytics';

const { Text, Title } = Typography;
const { RangePicker } = DatePicker;
const { TabPane } = Tabs;

interface StrategyMetrics {
  strategyId: string;
  name: string;
  totalReturn: number;
  annualizedReturn: number;
  volatility: number;
  sharpeRatio: number;
  maxDrawdown: number;
  winRate: number;
  profitFactor: number;
  sortino: number;
  calmar: number;
  var95: number;
  beta: number;
  alpha: number;
  treynor: number;
  informationRatio: number;
  trackingError: number;
  returns: Array<{
    date: string;
    value: number;
    cumulativeReturn: number;
    drawdown: number;
  }>;
  trades: Array<{
    id: string;
    timestamp: number;
    symbol: string;
    side: 'buy' | 'sell';
    quantity: number;
    price: number;
    pnl: number;
    commission: number;
  }>;
  attribution: {
    alpha: number;
    beta: number;
    sectors: Array<{
      name: string;
      contribution: number;
      weight: number;
    }>;
    factors: Array<{
      name: string;
      exposure: number;
      contribution: number;
    }>;
  };
  riskMetrics: {
    var: number;
    expectedShortfall: number;
    maxDrawdown: number;
    drawdownDuration: number;
    volatility: number;
    downVolatility: number;
  };
  benchmark: {
    name: string;
    totalReturn: number;
    volatility: number;
    correlation: number;
    beta: number;
  };
}

interface StrategyPerformanceWidgetProps {
  strategyIds: string[];
  benchmarkId?: string;
  timeRange?: [moment.Moment, moment.Moment];
  showComparison?: boolean;
  showAttribution?: boolean;
  showRiskMetrics?: boolean;
  showTrades?: boolean;
  compactMode?: boolean;
  height?: number;
  autoRefresh?: boolean;
  refreshInterval?: number;
  theme?: 'light' | 'dark';
  onStrategyClick?: (strategyId: string) => void;
  onTradeClick?: (trade: any) => void;
  exportEnabled?: boolean;
}

const StrategyPerformanceWidget: React.FC<StrategyPerformanceWidgetProps> = memo(({
  strategyIds,
  benchmarkId = 'SPY',
  timeRange,
  showComparison = true,
  showAttribution = true,
  showRiskMetrics = true,
  showTrades = false,
  compactMode = false,
  height = 700,
  autoRefresh = true,
  refreshInterval = 10000,
  theme = 'light',
  onStrategyClick,
  onTradeClick,
  exportEnabled = true
}) => {
  const [selectedStrategy, setSelectedStrategy] = useState<string>(strategyIds[0] || '');
  const [activeTab, setActiveTab] = useState('overview');
  const [viewMode, setViewMode] = useState<'absolute' | 'relative' | 'risk-adjusted'>('absolute');
  const [chartType, setChartType] = useState<'line' | 'area' | 'candlestick'>('line');
  const [isPlaying, setIsPlaying] = useState(autoRefresh);
  const [fullscreen, setFullscreen] = useState(false);

  const svgRef = useRef<SVGSVGElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);

  // Use strategy analytics hook
  const {
    strategies,
    isLoading,
    error,
    lastUpdate
  } = useStrategyAnalytics({
    strategyIds,
    benchmarkId,
    timeRange: timeRange ? [timeRange[0].toISOString(), timeRange[1].toISOString()] : undefined,
    autoRefresh: isPlaying,
    refreshInterval
  });

  // Process strategy data
  const processedData = useMemo(() => {
    if (!strategies) return null;

    const selectedStrategyData = strategies.find((s: StrategyMetrics) => s.strategyId === selectedStrategy);
    
    return {
      selectedStrategy: selectedStrategyData,
      allStrategies: strategies,
      comparison: strategies.map((strategy: StrategyMetrics) => ({
        ...strategy,
        relativeReturn: strategy.totalReturn - (selectedStrategyData?.totalReturn || 0),
        outperformance: strategy.totalReturn - (strategy.benchmark?.totalReturn || 0)
      }))
    };
  }, [strategies, selectedStrategy]);

  // Performance metrics table columns
  const performanceColumns = [
    {
      title: 'Metric',
      dataIndex: 'metric',
      key: 'metric',
      width: 150,
      render: (text: string) => <Text strong>{text}</Text>
    },
    {
      title: 'Value',
      dataIndex: 'value',
      key: 'value',
      width: 120,
      render: (value: any, record: any) => {
        if (record.type === 'percentage') {
          return <Text style={{ color: value >= 0 ? '#52c41a' : '#ff4d4f' }}>{(value * 100).toFixed(2)}%</Text>;
        } else if (record.type === 'ratio') {
          return <Text>{value.toFixed(3)}</Text>;
        } else if (record.type === 'currency') {
          return <Text>${value.toLocaleString()}</Text>;
        }
        return <Text>{value.toFixed(2)}</Text>;
      }
    },
    {
      title: 'Benchmark',
      dataIndex: 'benchmark',
      key: 'benchmark',
      width: 120,
      render: (value: any, record: any) => {
        if (!value && value !== 0) return <Text>-</Text>;
        if (record.type === 'percentage') {
          return <Text type="secondary">{(value * 100).toFixed(2)}%</Text>;
        } else if (record.type === 'ratio') {
          return <Text type="secondary">{value.toFixed(3)}</Text>;
        }
        return <Text type="secondary">{value.toFixed(2)}</Text>;
      }
    },
    {
      title: 'Difference',
      dataIndex: 'difference',
      key: 'difference',
      render: (value: any, record: any) => {
        if (!value && value !== 0) return <Text>-</Text>;
        const color = value >= 0 ? '#52c41a' : '#ff4d4f';
        if (record.type === 'percentage') {
          return <Text style={{ color }}>{value >= 0 ? '+' : ''}{(value * 100).toFixed(2)}%</Text>;
        }
        return <Text style={{ color }}>{value >= 0 ? '+' : ''}{value.toFixed(3)}</Text>;
      }
    }
  ];

  // Render performance chart
  const renderPerformanceChart = useCallback(() => {
    if (!processedData?.selectedStrategy || !svgRef.current) return;

    const svg = d3.select(svgRef.current);
    svg.selectAll('*').remove();

    const margin = { top: 20, right: 30, bottom: 40, left: 60 };
    const width = (containerRef.current?.clientWidth || 800) - margin.left - margin.right;
    const chartHeight = 300 - margin.top - margin.bottom;

    const g = svg.append('g')
      .attr('transform', `translate(${margin.left},${margin.top})`);

    const strategy = processedData.selectedStrategy;
    const returns = strategy.returns || [];

    // Scales
    const xScale = d3.scaleTime()
      .domain(d3.extent(returns, d => new Date(d.date)) as [Date, Date])
      .range([0, width]);

    const yScale = d3.scaleLinear()
      .domain(d3.extent(returns, d => 
        viewMode === 'absolute' ? d.cumulativeReturn :
        viewMode === 'relative' ? d.cumulativeReturn - (strategy.benchmark?.totalReturn || 0) :
        d.cumulativeReturn / Math.max(strategy.volatility, 0.01)
      ) as [number, number])
      .range([chartHeight, 0]);

    // Line generator
    const line = d3.line<any>()
      .x(d => xScale(new Date(d.date)))
      .y(d => yScale(
        viewMode === 'absolute' ? d.cumulativeReturn :
        viewMode === 'relative' ? d.cumulativeReturn - (strategy.benchmark?.totalReturn || 0) :
        d.cumulativeReturn / Math.max(strategy.volatility, 0.01)
      ))
      .curve(d3.curveMonotoneX);

    // Performance line
    g.append('path')
      .datum(returns)
      .attr('fill', 'none')
      .attr('stroke', '#1890ff')
      .attr('stroke-width', 2)
      .attr('d', line);

    // Benchmark line (if showing comparison)
    if (showComparison && strategy.benchmark) {
      const benchmarkLine = d3.line<any>()
        .x(d => xScale(new Date(d.date)))
        .y(d => yScale(viewMode === 'absolute' ? strategy.benchmark.totalReturn : 0))
        .curve(d3.curveMonotoneX);

      g.append('path')
        .datum(returns)
        .attr('fill', 'none')
        .attr('stroke', '#999')
        .attr('stroke-width', 1)
        .attr('stroke-dasharray', '3,3')
        .attr('d', benchmarkLine);
    }

    // Drawdown area
    if (viewMode === 'absolute') {
      const area = d3.area<any>()
        .x(d => xScale(new Date(d.date)))
        .y0(yScale(0))
        .y1(d => yScale(d.drawdown))
        .curve(d3.curveMonotoneX);

      g.append('path')
        .datum(returns)
        .attr('fill', '#ff4d4f')
        .attr('opacity', 0.3)
        .attr('d', area);
    }

    // Add zero line
    g.append('line')
      .attr('x1', 0)
      .attr('x2', width)
      .attr('y1', yScale(0))
      .attr('y2', yScale(0))
      .attr('stroke', '#333')
      .attr('stroke-width', 1)
      .attr('opacity', 0.5);

    // Axes
    const xAxis = d3.axisBottom(xScale).tickFormat(d3.timeFormat('%b %Y'));
    g.append('g')
      .attr('transform', `translate(0,${chartHeight})`)
      .call(xAxis);

    const yAxis = d3.axisLeft(yScale).tickFormat(d => 
      viewMode === 'absolute' ? `${(d * 100).toFixed(1)}%` :
      viewMode === 'relative' ? `${(d * 100).toFixed(1)}%` :
      d.toFixed(2)
    );
    g.append('g').call(yAxis);

    // Title
    g.append('text')
      .attr('x', width / 2)
      .attr('y', -5)
      .attr('text-anchor', 'middle')
      .attr('font-size', '14px')
      .attr('font-weight', 'bold')
      .text(`${strategy.name} - ${viewMode.replace('-', ' ').toUpperCase()} Performance`);

    // Interactive dots for recent performance
    g.selectAll('.performance-dot')
      .data(returns.slice(-10))
      .enter()
      .append('circle')
      .attr('class', 'performance-dot')
      .attr('cx', d => xScale(new Date(d.date)))
      .attr('cy', d => yScale(
        viewMode === 'absolute' ? d.cumulativeReturn :
        viewMode === 'relative' ? d.cumulativeReturn - (strategy.benchmark?.totalReturn || 0) :
        d.cumulativeReturn / Math.max(strategy.volatility, 0.01)
      ))
      .attr('r', 3)
      .attr('fill', '#1890ff')
      .style('cursor', 'pointer')
      .on('mouseover', function(event, d) {
        const tooltip = g.append('g')
          .attr('class', 'tooltip')
          .attr('transform', `translate(${xScale(new Date(d.date))}, ${yScale(d.cumulativeReturn) - 20})`);

        tooltip.append('rect')
          .attr('x', -60)
          .attr('y', -15)
          .attr('width', 120)
          .attr('height', 12)
          .attr('fill', '#333')
          .attr('opacity', 0.9)
          .attr('rx', 2);

        tooltip.append('text')
          .attr('text-anchor', 'middle')
          .attr('fill', '#fff')
          .attr('font-size', '10px')
          .text(`${moment(d.date).format('MMM DD')}: ${(d.cumulativeReturn * 100).toFixed(2)}%`);
      })
      .on('mouseout', function() {
        g.selectAll('.tooltip').remove();
      });

  }, [processedData, viewMode, showComparison]);

  // Update chart when data changes
  useEffect(() => {
    renderPerformanceChart();
  }, [renderPerformanceChart]);

  // Handle export
  const handleExport = useCallback(() => {
    if (!svgRef.current || !exportEnabled) return;

    const svgData = new XMLSerializer().serializeToString(svgRef.current);
    const canvas = document.createElement('canvas');
    const ctx = canvas.getContext('2d');
    const img = new Image();

    img.onload = () => {
      canvas.width = img.width;
      canvas.height = img.height;
      ctx?.drawImage(img, 0, 0);
      
      const link = document.createElement('a');
      link.download = `strategy-performance-${selectedStrategy}-${Date.now()}.png`;
      link.href = canvas.toDataURL();
      link.click();
    };

    img.src = `data:image/svg+xml;base64,${btoa(svgData)}`;
  }, [selectedStrategy, exportEnabled]);

  // Settings menu
  const settingsMenu = (
    <Menu>
      <Menu.SubMenu key="view" title="View Mode">
        <Menu.Item
          key="absolute"
          onClick={() => setViewMode('absolute')}
          style={{ backgroundColor: viewMode === 'absolute' ? '#e6f7ff' : 'transparent' }}
        >
          Absolute Returns
        </Menu.Item>
        <Menu.Item
          key="relative"
          onClick={() => setViewMode('relative')}
          style={{ backgroundColor: viewMode === 'relative' ? '#e6f7ff' : 'transparent' }}
        >
          Relative to Benchmark
        </Menu.Item>
        <Menu.Item
          key="risk-adjusted"
          onClick={() => setViewMode('risk-adjusted')}
          style={{ backgroundColor: viewMode === 'risk-adjusted' ? '#e6f7ff' : 'transparent' }}
        >
          Risk-Adjusted
        </Menu.Item>
      </Menu.SubMenu>
      <Menu.SubMenu key="chart" title="Chart Type">
        <Menu.Item key="line">Line Chart</Menu.Item>
        <Menu.Item key="area">Area Chart</Menu.Item>
        <Menu.Item key="candlestick">Candlestick</Menu.Item>
      </Menu.SubMenu>
    </Menu>
  );

  if (error) {
    return (
      <Card title="Strategy Performance - Error" style={{ height }}>
        <Text type="danger">{error}</Text>
      </Card>
    );
  }

  return (
    <div ref={containerRef} style={{ position: 'relative' }}>
      <Card
        title={
          <Space>
            <TrophyOutlined />
            <Text strong>Strategy Performance</Text>
            <Badge status={isLoading ? 'processing' : 'success'} />
          </Space>
        }
        extra={
          <Space>
            {/* Strategy Selector */}
            <Select
              value={selectedStrategy}
              onChange={(value) => {
                setSelectedStrategy(value);
                onStrategyClick?.(value);
              }}
              style={{ width: 200 }}
              size="small"
            >
              {strategyIds.map(id => (
                <Select.Option key={id} value={id}>
                  {processedData?.allStrategies.find((s: StrategyMetrics) => s.strategyId === id)?.name || id}
                </Select.Option>
              ))}
            </Select>

            {/* View Mode */}
            <Select
              value={viewMode}
              onChange={setViewMode}
              size="small"
              style={{ width: 130 }}
            >
              <Select.Option value="absolute">Absolute</Select.Option>
              <Select.Option value="relative">Relative</Select.Option>
              <Select.Option value="risk-adjusted">Risk-Adj</Select.Option>
            </Select>

            {/* Controls */}
            <Button
              size="small"
              icon={isPlaying ? <PauseCircleOutlined /> : <PlayCircleOutlined />}
              onClick={() => setIsPlaying(!isPlaying)}
            />

            <Button size="small" icon={<ReloadOutlined />} onClick={renderPerformanceChart} />

            <Dropdown overlay={settingsMenu} trigger={['click']}>
              <Button size="small" icon={<SettingOutlined />} />
            </Dropdown>

            {exportEnabled && (
              <Button size="small" icon={<ExportOutlined />} onClick={handleExport} />
            )}

            <Button
              size="small"
              icon={<FullscreenOutlined />}
              onClick={() => setFullscreen(!fullscreen)}
            />
          </Space>
        }
        size={compactMode ? 'small' : 'default'}
        style={{ 
          height: fullscreen ? '100vh' : height,
          position: fullscreen ? 'fixed' : 'relative',
          top: fullscreen ? 0 : 'auto',
          left: fullscreen ? 0 : 'auto',
          width: fullscreen ? '100vw' : '100%',
          zIndex: fullscreen ? 1000 : 'auto'
        }}
      >
        <Tabs activeKey={activeTab} onChange={setActiveTab} size="small">
          {/* Overview Tab */}
          <TabPane
            tab={
              <Space>
                <LineChartOutlined />
                Overview
              </Space>
            }
            key="overview"
          >
            {/* Key Performance Metrics */}
            {!compactMode && processedData?.selectedStrategy && (
              <Row gutter={[16, 16]} style={{ marginBottom: 16 }}>
                <Col span={6}>
                  <Statistic
                    title="Total Return"
                    value={processedData.selectedStrategy.totalReturn * 100}
                    precision={2}
                    suffix="%"
                    valueStyle={{ 
                      color: processedData.selectedStrategy.totalReturn >= 0 ? '#52c41a' : '#ff4d4f',
                      fontSize: '18px'
                    }}
                  />
                </Col>
                <Col span={6}>
                  <Statistic
                    title="Sharpe Ratio"
                    value={processedData.selectedStrategy.sharpeRatio}
                    precision={2}
                    valueStyle={{ fontSize: '18px' }}
                  />
                </Col>
                <Col span={6}>
                  <Statistic
                    title="Max Drawdown"
                    value={Math.abs(processedData.selectedStrategy.maxDrawdown) * 100}
                    precision={2}
                    suffix="%"
                    valueStyle={{ color: '#ff4d4f', fontSize: '18px' }}
                  />
                </Col>
                <Col span={6}>
                  <Statistic
                    title="Win Rate"
                    value={processedData.selectedStrategy.winRate * 100}
                    precision={1}
                    suffix="%"
                    valueStyle={{ fontSize: '18px' }}
                  />
                </Col>
              </Row>
            )}

            {/* Performance Chart */}
            <div style={{ 
              width: '100%', 
              height: compactMode ? 250 : 300,
              marginBottom: 16
            }}>
              <svg
                ref={svgRef}
                width="100%"
                height="100%"
                style={{ background: theme === 'dark' ? '#1f1f1f' : '#ffffff' }}
              />
            </div>

            {/* Performance Table */}
            {processedData?.selectedStrategy && (
              <Table
                dataSource={[
                  { 
                    key: '1', 
                    metric: 'Annualized Return', 
                    value: processedData.selectedStrategy.annualizedReturn,
                    benchmark: processedData.selectedStrategy.benchmark?.totalReturn,
                    difference: processedData.selectedStrategy.annualizedReturn - (processedData.selectedStrategy.benchmark?.totalReturn || 0),
                    type: 'percentage'
                  },
                  { 
                    key: '2', 
                    metric: 'Volatility', 
                    value: processedData.selectedStrategy.volatility,
                    benchmark: processedData.selectedStrategy.benchmark?.volatility,
                    difference: processedData.selectedStrategy.volatility - (processedData.selectedStrategy.benchmark?.volatility || 0),
                    type: 'percentage'
                  },
                  { 
                    key: '3', 
                    metric: 'Sharpe Ratio', 
                    value: processedData.selectedStrategy.sharpeRatio,
                    type: 'ratio'
                  },
                  { 
                    key: '4', 
                    metric: 'Sortino Ratio', 
                    value: processedData.selectedStrategy.sortino,
                    type: 'ratio'
                  },
                  { 
                    key: '5', 
                    metric: 'Calmar Ratio', 
                    value: processedData.selectedStrategy.calmar,
                    type: 'ratio'
                  },
                  { 
                    key: '6', 
                    metric: 'Alpha', 
                    value: processedData.selectedStrategy.alpha,
                    type: 'percentage'
                  },
                  { 
                    key: '7', 
                    metric: 'Beta', 
                    value: processedData.selectedStrategy.beta,
                    type: 'ratio'
                  },
                  { 
                    key: '8', 
                    metric: 'Information Ratio', 
                    value: processedData.selectedStrategy.informationRatio,
                    type: 'ratio'
                  }
                ]}
                columns={performanceColumns}
                pagination={false}
                size="small"
                style={{ maxHeight: 200, overflow: 'auto' }}
              />
            )}
          </TabPane>

          {/* Strategy Comparison Tab */}
          {showComparison && (
            <TabPane
              tab={
                <Space>
                  <CompareOutlined />
                  Comparison
                </Space>
              }
              key="comparison"
            >
              {processedData?.comparison && (
                <Table
                  dataSource={processedData.comparison.map((strategy: any, index: number) => ({
                    ...strategy,
                    key: index,
                    rank: index + 1
                  }))}
                  columns={[
                    {
                      title: 'Rank',
                      dataIndex: 'rank',
                      key: 'rank',
                      width: 60,
                      render: (rank: number) => (
                        <Badge 
                          count={rank} 
                          style={{ 
                            backgroundColor: rank <= 3 ? '#52c41a' : '#faad14'
                          }} 
                        />
                      )
                    },
                    {
                      title: 'Strategy',
                      dataIndex: 'name',
                      key: 'name',
                      render: (name: string, record: any) => (
                        <Space>
                          <Text strong>{name}</Text>
                          {record.strategyId === selectedStrategy && (
                            <Tag color="blue">Selected</Tag>
                          )}
                        </Space>
                      )
                    },
                    {
                      title: 'Total Return',
                      dataIndex: 'totalReturn',
                      key: 'totalReturn',
                      sorter: (a: any, b: any) => a.totalReturn - b.totalReturn,
                      render: (value: number) => (
                        <Text style={{ color: value >= 0 ? '#52c41a' : '#ff4d4f' }}>
                          {(value * 100).toFixed(2)}%
                        </Text>
                      )
                    },
                    {
                      title: 'Sharpe Ratio',
                      dataIndex: 'sharpeRatio',
                      key: 'sharpeRatio',
                      sorter: (a: any, b: any) => a.sharpeRatio - b.sharpeRatio,
                      render: (value: number) => value.toFixed(2)
                    },
                    {
                      title: 'Max Drawdown',
                      dataIndex: 'maxDrawdown',
                      key: 'maxDrawdown',
                      sorter: (a: any, b: any) => Math.abs(a.maxDrawdown) - Math.abs(b.maxDrawdown),
                      render: (value: number) => (
                        <Text style={{ color: '#ff4d4f' }}>
                          {(Math.abs(value) * 100).toFixed(2)}%
                        </Text>
                      )
                    },
                    {
                      title: 'Outperformance',
                      dataIndex: 'outperformance',
                      key: 'outperformance',
                      sorter: (a: any, b: any) => a.outperformance - b.outperformance,
                      render: (value: number) => (
                        <Text style={{ color: value >= 0 ? '#52c41a' : '#ff4d4f' }}>
                          {value >= 0 ? '+' : ''}{(value * 100).toFixed(2)}%
                        </Text>
                      )
                    }
                  ]}
                  pagination={false}
                  size="small"
                />
              )}
            </TabPane>
          )}

          {/* Risk Metrics Tab */}
          {showRiskMetrics && processedData?.selectedStrategy && (
            <TabPane
              tab={
                <Space>
                  <BarChartOutlined />
                  Risk Metrics
                </Space>
              }
              key="risk"
            >
              <Row gutter={[16, 16]}>
                <Col span={12}>
                  <Card size="small" title="Risk Measures">
                    <Space direction="vertical" style={{ width: '100%' }}>
                      <Row justify="space-between">
                        <Text>Value at Risk (95%)</Text>
                        <Text strong>${processedData.selectedStrategy.var95?.toLocaleString() || 'N/A'}</Text>
                      </Row>
                      <Row justify="space-between">
                        <Text>Expected Shortfall</Text>
                        <Text strong>${processedData.selectedStrategy.riskMetrics?.expectedShortfall?.toLocaleString() || 'N/A'}</Text>
                      </Row>
                      <Row justify="space-between">
                        <Text>Volatility</Text>
                        <Text strong>{(processedData.selectedStrategy.volatility * 100).toFixed(2)}%</Text>
                      </Row>
                      <Row justify="space-between">
                        <Text>Downside Volatility</Text>
                        <Text strong>{((processedData.selectedStrategy.riskMetrics?.downVolatility || 0) * 100).toFixed(2)}%</Text>
                      </Row>
                    </Space>
                  </Card>
                </Col>
                <Col span={12}>
                  <Card size="small" title="Drawdown Analysis">
                    <Space direction="vertical" style={{ width: '100%' }}>
                      <Row justify="space-between">
                        <Text>Max Drawdown</Text>
                        <Text strong style={{ color: '#ff4d4f' }}>
                          {(Math.abs(processedData.selectedStrategy.maxDrawdown) * 100).toFixed(2)}%
                        </Text>
                      </Row>
                      <Row justify="space-between">
                        <Text>Drawdown Duration</Text>
                        <Text strong>{processedData.selectedStrategy.riskMetrics?.drawdownDuration || 'N/A'} days</Text>
                      </Row>
                      <Row justify="space-between">
                        <Text>Recovery Factor</Text>
                        <Text strong>{(processedData.selectedStrategy.totalReturn / Math.abs(processedData.selectedStrategy.maxDrawdown)).toFixed(2)}</Text>
                      </Row>
                      <Progress
                        percent={Math.min(Math.abs(processedData.selectedStrategy.maxDrawdown) * 100, 100)}
                        status="exception"
                        size="small"
                        format={percent => `${percent?.toFixed(1)}%`}
                      />
                    </Space>
                  </Card>
                </Col>
              </Row>
            </TabPane>
          )}
        </Tabs>

        {/* Footer */}
        {!compactMode && (
          <div style={{ 
            marginTop: 16, 
            padding: '8px 0',
            borderTop: '1px solid #f0f0f0',
            display: 'flex',
            justifyContent: 'space-between',
            alignItems: 'center'
          }}>
            <Space>
              <Text type="secondary" style={{ fontSize: '11px' }}>
                Last Update: {lastUpdate ? new Date(lastUpdate).toLocaleTimeString() : 'Never'}
              </Text>
              <Text type="secondary" style={{ fontSize: '11px' }}>
                View: {viewMode.replace('-', ' ')}
              </Text>
              {processedData?.selectedStrategy?.benchmark && (
                <Text type="secondary" style={{ fontSize: '11px' }}>
                  Benchmark: {processedData.selectedStrategy.benchmark.name}
                </Text>
              )}
            </Space>
            
            <Space>
              {isPlaying && (
                <Badge status="processing" text="Live" />
              )}
              <Text type="secondary" style={{ fontSize: '11px' }}>
                Strategies: {strategyIds.length}
              </Text>
            </Space>
          </div>
        )}
      </Card>
    </div>
  );
});

StrategyPerformanceWidget.displayName = 'StrategyPerformanceWidget';

export default StrategyPerformanceWidget;