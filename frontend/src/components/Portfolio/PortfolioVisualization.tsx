import React, { useState, useEffect } from 'react';
import {
  Card,
  Row,
  Col,
  Typography,
  Space,
  Button,
  Select,
  DatePicker,
  Tabs,
  Table,
  Tag,
  Progress,
  Spin,
  Alert,
  Tooltip,
  Switch,
  Modal,
  notification,
  Statistic,
  Divider
} from 'antd';
import {
  PieChartOutlined,
  BarChartOutlined,
  LineChartOutlined,
  TrophyOutlined,
  AlertOutlined,
  ReloadOutlined,
  SettingOutlined,
  EyeOutlined,
  InfoCircleOutlined,
  RiseOutlined,
  FallOutlined
} from '@ant-design/icons';
import { Line, Pie, Bar, Treemap } from '@ant-design/charts';
import dayjs from 'dayjs';

const { Title, Text } = Typography;
const { RangePicker } = DatePicker;
const { Option } = Select;

interface PortfolioVisualizationProps {
  className?: string;
}

interface StrategyAllocation {
  strategy_id: string;
  strategy_name: string;
  allocation_percentage: number;
  current_value: number;
  pnl: number;
  pnl_percentage: number;
  sharpe_ratio: number;
  max_drawdown: number;
  win_rate: number;
  last_trade_time: string;
  status: 'active' | 'paused' | 'stopped';
  risk_level: 'low' | 'medium' | 'high';
}

interface PortfolioPerformance {
  timestamp: string;
  total_value: number;
  total_pnl: number;
  unrealized_pnl: number;
  daily_return: number;
  cumulative_return: number;
  sharpe_ratio: number;
  sortino_ratio: number;
  calmar_ratio: number;
  max_drawdown: number;
  volatility: number;
}

interface AssetAllocation {
  asset_class: string;
  symbol: string;
  value: number;
  percentage: number;
  pnl: number;
  pnl_percentage: number;
  risk_contribution: number;
}

interface CorrelationMatrix {
  strategy_pairs: {
    strategy_1: string;
    strategy_2: string;
    correlation: number;
    significance: number;
  }[];
}

const PortfolioVisualization: React.FC<PortfolioVisualizationProps> = ({ className }) => {
  const [loading, setLoading] = useState(false);
  const [activeTab, setActiveTab] = useState('overview');
  const [timeRange, setTimeRange] = useState<[dayjs.Dayjs, dayjs.Dayjs]>([
    dayjs().subtract(30, 'days'),
    dayjs()
  ]);
  const [portfolioId, setPortfolioId] = useState<string>('default');
  const [realTimeEnabled, setRealTimeEnabled] = useState(false);
  const [autoRefresh, setAutoRefresh] = useState<NodeJS.Timeout | null>(null);

  // Data states
  const [strategyAllocations, setStrategyAllocations] = useState<StrategyAllocation[]>([]);
  const [portfolioPerformance, setPortfolioPerformance] = useState<PortfolioPerformance[]>([]);
  const [currentPerformance, setCurrentPerformance] = useState<PortfolioPerformance | null>(null);
  const [assetAllocations, setAssetAllocations] = useState<AssetAllocation[]>([]);
  const [correlationMatrix, setCorrelationMatrix] = useState<CorrelationMatrix | null>(null);
  const [benchmarkData, setBenchmarkData] = useState<any[]>([]);

  const [lastUpdate, setLastUpdate] = useState<Date | null>(null);

  useEffect(() => {
    loadPortfolioData();
    if (realTimeEnabled) {
      startAutoRefresh();
    }
    return () => stopAutoRefresh();
  }, [portfolioId, timeRange, realTimeEnabled]);

  const startAutoRefresh = () => {
    const interval = setInterval(() => {
      loadPortfolioData();
    }, 10000); // Update every 10 seconds
    setAutoRefresh(interval);
  };

  const stopAutoRefresh = () => {
    if (autoRefresh) {
      clearInterval(autoRefresh);
      setAutoRefresh(null);
    }
  };

  const loadPortfolioData = async () => {
    try {
      setLoading(true);
      
      // Load strategy allocations
      const allocationsResponse = await fetch(
        `/api/v1/portfolio/${portfolioId}/strategy-allocations?start_date=${timeRange[0].toISOString()}&end_date=${timeRange[1].toISOString()}`
      );
      if (allocationsResponse.ok) {
        const allocationsData = await allocationsResponse.json();
        // Map API response to expected component data structure
        const strategies = allocationsData.allocations?.strategies || [];
        const mappedAllocations = strategies.map((strategy: any) => ({
          strategy_id: strategy.name?.toLowerCase().replace(/\s+/g, '_') || 'unknown',
          strategy_name: strategy.name || 'Unknown Strategy',
          allocation_percentage: strategy.allocation || 0,
          current_value: strategy.value || 0,
          pnl: strategy.pnl || 0,
          pnl_percentage: strategy.value > 0 ? (strategy.pnl / strategy.value) * 100 : 0,
          sharpe_ratio: 1.2, // Default value - should come from API
          max_drawdown: 0.05, // Default value - should come from API
          win_rate: 0.65, // Default value - should come from API
          last_trade_time: new Date().toISOString(),
          status: 'active' as const,
          risk_level: 'medium' as const
        }));
        setStrategyAllocations(mappedAllocations);
      }

      // Load portfolio performance history
      const performanceResponse = await fetch(
        `/api/v1/portfolio/${portfolioId}/performance-history?start_date=${timeRange[0].toISOString()}&end_date=${timeRange[1].toISOString()}`
      );
      if (performanceResponse.ok) {
        const performanceData = await performanceResponse.json();
        // Map API response to expected component data structure
        const history = performanceData.history?.performance || [];
        const mappedHistory = history.map((point: any) => ({
          timestamp: point.date,
          total_value: point.portfolio_value || 0,
          total_pnl: (point.portfolio_value || 1000000) - 1000000, // Assuming 1M starting value
          unrealized_pnl: 0, // Default value
          daily_return: point.daily_return / 100 || 0, // Convert percentage to decimal
          cumulative_return: point.cumulative_return / 100 || 0, // Convert percentage to decimal
          sharpe_ratio: 1.2, // Default value
          sortino_ratio: 1.0, // Default value
          calmar_ratio: 0.8, // Default value
          max_drawdown: 0.05, // Default value
          volatility: 0.15 // Default value
        }));
        setPortfolioPerformance(mappedHistory);
        if (mappedHistory.length > 0) {
          setCurrentPerformance(mappedHistory[mappedHistory.length - 1]);
        }
      }

      // Load asset allocations
      const assetsResponse = await fetch(`/api/v1/portfolio/${portfolioId}/asset-allocations`);
      if (assetsResponse.ok) {
        const assetsData = await assetsResponse.json();
        // Map API response to expected component data structure
        const allocations = assetsData.allocations?.allocations || {};
        const mappedAssets = Object.entries(allocations).map(([assetClass, data]: [string, any]) => ({
          asset_class: assetClass,
          symbol: assetClass.toUpperCase(),
          value: data.value || 0,
          percentage: data.percentage || 0,
          pnl: (data.value || 0) * 0.02, // Approximate 2% gain as default
          pnl_percentage: 2.0, // Default 2% as placeholder
          risk_contribution: data.percentage ? data.percentage / 100 * 0.1 : 0 // Risk contribution approximation
        }));
        setAssetAllocations(mappedAssets);
      }

      // Load correlation matrix
      const correlationResponse = await fetch(`/api/v1/portfolio/${portfolioId}/strategy-correlations`);
      if (correlationResponse.ok) {
        const correlationData = await correlationResponse.json();
        setCorrelationMatrix(correlationData);
      }

      // Load benchmark comparison
      const benchmarkResponse = await fetch(
        `/api/v1/portfolio/${portfolioId}/benchmark-comparison?start_date=${timeRange[0].toISOString()}&end_date=${timeRange[1].toISOString()}`
      );
      if (benchmarkResponse.ok) {
        const benchmarkData = await benchmarkResponse.json();
        // Map API response to expected component data structure
        const comparison = benchmarkData.comparison || {};
        const portfolioPerf = comparison.portfolio?.performance || [];
        const benchmarkPerf = comparison.benchmark?.performance || [];
        
        const mappedBenchmarkData = portfolioPerf.map((point: any, index: number) => {
          const benchmarkPoint = benchmarkPerf[index] || {};
          return {
            timestamp: dayjs(point.date).format('MM/DD'),
            value: point.return || 0,
            series: 'Portfolio'
          };
        }).concat(
          benchmarkPerf.map((point: any) => ({
            timestamp: dayjs(point.date).format('MM/DD'),
            value: point.return || 0,
            series: comparison.benchmark?.name || 'Benchmark'
          }))
        );
        
        setBenchmarkData(mappedBenchmarkData);
      }

      setLastUpdate(new Date());

    } catch (error: any) {
      console.error('Failed to load portfolio data:', error);
      notification.error({
        message: 'Portfolio Data Error',
        description: error.message || 'Failed to load portfolio visualization data',
        duration: 4
      });
    } finally {
      setLoading(false);
    }
  };

  const calculatePortfolioMetrics = () => {
    if (!currentPerformance) return null;

    const totalValue = currentPerformance.total_value;
    const totalPnL = currentPerformance.total_pnl;
    const pnlPercentage = totalValue > 0 ? (totalPnL / (totalValue - totalPnL)) * 100 : 0;

    return {
      totalValue,
      totalPnL,
      pnlPercentage,
      sharpeRatio: currentPerformance.sharpe_ratio,
      maxDrawdown: currentPerformance.max_drawdown,
      volatility: currentPerformance.volatility
    };
  };

  const strategyAllocationColumns = [
    {
      title: 'Strategy',
      dataIndex: 'strategy_name',
      key: 'strategy_name',
      render: (name: string, record: StrategyAllocation) => (
        <div>
          <Text strong>{name}</Text>
          <br />
          <Tag color={record.status === 'active' ? 'green' : record.status === 'paused' ? 'orange' : 'red'}>
            {record.status.toUpperCase()}
          </Tag>
          <Tag color={record.risk_level === 'low' ? 'green' : record.risk_level === 'medium' ? 'orange' : 'red'}>
            {record.risk_level.toUpperCase()} RISK
          </Tag>
        </div>
      )
    },
    {
      title: 'Allocation',
      dataIndex: 'allocation_percentage',
      key: 'allocation_percentage',
      render: (percentage: number, record: StrategyAllocation) => (
        <div>
          <Progress
            percent={percentage}
            size="small"
            showInfo={false}
            strokeColor={percentage > 30 ? '#faad14' : '#52c41a'}
          />
          <div style={{ display: 'flex', justifyContent: 'space-between', marginTop: 4 }}>
            <Text type="secondary">{percentage.toFixed(1)}%</Text>
            <Text>${record.current_value.toLocaleString()}</Text>
          </div>
        </div>
      )
    },
    {
      title: 'P&L',
      key: 'pnl',
      render: (_, record: StrategyAllocation) => (
        <div>
          <Text style={{ color: record.pnl >= 0 ? '#3f8600' : '#cf1322' }}>
            {record.pnl >= 0 ? <RiseOutlined /> : <FallOutlined />}
            ${Math.abs(record.pnl).toLocaleString()}
          </Text>
          <br />
          <Text type="secondary" style={{ color: record.pnl_percentage >= 0 ? '#3f8600' : '#cf1322' }}>
            ({record.pnl_percentage >= 0 ? '+' : ''}{record.pnl_percentage.toFixed(2)}%)
          </Text>
        </div>
      )
    },
    {
      title: 'Metrics',
      key: 'metrics',
      render: (_, record: StrategyAllocation) => (
        <Space direction="vertical" size="small">
          <div>
            <Text type="secondary">Sharpe: </Text>
            <Text>{record.sharpe_ratio.toFixed(2)}</Text>
          </div>
          <div>
            <Text type="secondary">Win Rate: </Text>
            <Text>{(record.win_rate * 100).toFixed(1)}%</Text>
          </div>
          <div>
            <Text type="secondary">Drawdown: </Text>
            <Text style={{ color: record.max_drawdown > 0.1 ? '#cf1322' : '#666' }}>
              {(record.max_drawdown * 100).toFixed(1)}%
            </Text>
          </div>
        </Space>
      )
    }
  ];

  const getPerformanceChartData = () => {
    return portfolioPerformance.map(point => ({
      timestamp: dayjs(point.timestamp).format('MM/DD'),
      'Portfolio Value': point.total_value,
      'Cumulative Return': point.cumulative_return * 100,
      'Daily Return': point.daily_return * 100
    }));
  };

  const getAllocationPieData = () => {
    return strategyAllocations.map(allocation => ({
      type: allocation.strategy_name,
      value: allocation.allocation_percentage,
      pnl: allocation.pnl
    }));
  };

  const getAssetAllocationTreemapData = () => {
    return {
      name: 'Portfolio',
      children: assetAllocations.map(asset => ({
        name: asset.symbol,
        value: asset.value,
        category: asset.asset_class
      }))
    };
  };

  const portfolioMetrics = calculatePortfolioMetrics();

  const renderOverview = () => (
    <div>
      {/* Key Metrics Cards */}
      {portfolioMetrics && (
        <Row gutter={[16, 16]} style={{ marginBottom: 24 }}>
          <Col xs={24} sm={12} lg={6}>
            <Card>
              <Statistic
                title="Total Portfolio Value"
                value={portfolioMetrics.totalValue}
                precision={2}
                prefix="$"
                valueStyle={{ color: '#1890ff' }}
              />
            </Card>
          </Col>
          <Col xs={24} sm={12} lg={6}>
            <Card>
              <Statistic
                title="Total P&L"
                value={portfolioMetrics.totalPnL}
                precision={2}
                prefix="$"
                valueStyle={{ color: portfolioMetrics.totalPnL >= 0 ? '#3f8600' : '#cf1322' }}
                suffix={
                  <Text type="secondary">
                    ({portfolioMetrics.pnlPercentage >= 0 ? '+' : ''}{portfolioMetrics.pnlPercentage.toFixed(2)}%)
                  </Text>
                }
              />
            </Card>
          </Col>
          <Col xs={24} sm={12} lg={6}>
            <Card>
              <Statistic
                title="Sharpe Ratio"
                value={portfolioMetrics.sharpeRatio}
                precision={2}
                valueStyle={{ color: portfolioMetrics.sharpeRatio > 1 ? '#3f8600' : '#666' }}
              />
            </Card>
          </Col>
          <Col xs={24} sm={12} lg={6}>
            <Card>
              <Statistic
                title="Max Drawdown"
                value={portfolioMetrics.maxDrawdown * 100}
                precision={1}
                suffix="%"
                valueStyle={{ color: portfolioMetrics.maxDrawdown > 0.1 ? '#cf1322' : '#666' }}
              />
            </Card>
          </Col>
        </Row>
      )}

      {/* Performance Chart */}
      <Row gutter={[16, 16]} style={{ marginBottom: 24 }}>
        <Col xs={24} lg={16}>
          <Card title="Portfolio Performance" extra={
            <Space>
              <Text type="secondary" style={{ fontSize: 12 }}>
                Last updated: {lastUpdate?.toLocaleTimeString()}
              </Text>
            </Space>
          }>
            <Line
              data={getPerformanceChartData()}
              xField="timestamp"
              yField="Portfolio Value"
              height={300}
              smooth={true}
              point={{ size: 3 }}
              color="#1890ff"
            />
          </Card>
        </Col>
        <Col xs={24} lg={8}>
          <Card title="Strategy Allocation">
            <Pie
              data={getAllocationPieData()}
              angleField="value"
              colorField="type"
              radius={0.8}
              height={300}
              label={{
                type: 'spider',
                labelHeight: 28,
                content: '{name}\n{percentage}'
              }}
            />
          </Card>
        </Col>
      </Row>

      {/* Strategy Allocations Table */}
      <Card title="Strategy Breakdown">
        <Table
          columns={strategyAllocationColumns}
          dataSource={strategyAllocations}
          rowKey="strategy_id"
          loading={loading}
          pagination={false}
          size="small"
        />
      </Card>
    </div>
  );

  const renderPerformanceAnalysis = () => (
    <div>
      {/* Performance vs Benchmark */}
      <Row gutter={[16, 16]} style={{ marginBottom: 24 }}>
        <Col span={24}>
          <Card title="Performance vs Benchmark">
            <Line
              data={benchmarkData}
              xField="timestamp"
              yField="value"
              seriesField="series"
              height={400}
              smooth={true}
              legend={{ position: 'top' }}
            />
          </Card>
        </Col>
      </Row>

      {/* Risk-Adjusted Metrics */}
      <Row gutter={[16, 16]}>
        <Col xs={24} lg={12}>
          <Card title="Risk-Adjusted Returns">
            <Bar
              data={strategyAllocations.map(s => ({
                strategy: s.strategy_name,
                sharpe: s.sharpe_ratio,
                sortino: s.sharpe_ratio * 1.2, // Approximation
                calmar: s.sharpe_ratio * 0.8   // Approximation
              }))}
              xField="sharpe"
              yField="strategy"
              seriesField="metric"
              height={300}
            />
          </Card>
        </Col>
        <Col xs={24} lg={12}>
          <Card title="Drawdown Analysis">
            {/* Drawdown visualization would go here */}
            <div style={{ textAlign: 'center', padding: '60px 0' }}>
              <Text type="secondary">Drawdown analysis chart</Text>
            </div>
          </Card>
        </Col>
      </Row>
    </div>
  );

  const renderAssetAllocation = () => (
    <div>
      <Row gutter={[16, 16]} style={{ marginBottom: 24 }}>
        <Col span={24}>
          <Card title="Asset Allocation Treemap">
            <Treemap
              data={getAssetAllocationTreemapData()}
              colorField="category"
              height={400}
            />
          </Card>
        </Col>
      </Row>

      <Row gutter={[16, 16]}>
        <Col span={24}>
          <Card title="Asset Breakdown">
            <Table
              columns={[
                {
                  title: 'Asset',
                  dataIndex: 'symbol',
                  key: 'symbol'
                },
                {
                  title: 'Asset Class',
                  dataIndex: 'asset_class',
                  key: 'asset_class',
                  render: (assetClass: string) => <Tag>{assetClass}</Tag>
                },
                {
                  title: 'Value',
                  dataIndex: 'value',
                  key: 'value',
                  render: (value: number) => `$${value.toLocaleString()}`
                },
                {
                  title: 'Allocation %',
                  dataIndex: 'percentage',
                  key: 'percentage',
                  render: (percentage: number) => `${percentage.toFixed(1)}%`
                },
                {
                  title: 'P&L',
                  key: 'pnl',
                  render: (_, record: AssetAllocation) => (
                    <Text style={{ color: record.pnl >= 0 ? '#3f8600' : '#cf1322' }}>
                      ${record.pnl.toLocaleString()} ({record.pnl_percentage.toFixed(2)}%)
                    </Text>
                  )
                }
              ]}
              dataSource={assetAllocations}
              rowKey="symbol"
              loading={loading}
              size="small"
            />
          </Card>
        </Col>
      </Row>
    </div>
  );

  return (
    <div className={`portfolio-visualization ${className || ''}`}>
      <Card>
        <div style={{ marginBottom: 24 }}>
          <Row justify="space-between" align="middle">
            <Col>
              <Title level={2} style={{ margin: 0 }}>
                <PieChartOutlined style={{ marginRight: 8 }} />
                Portfolio Visualization
              </Title>
              <Text type="secondary">
                Multi-strategy portfolio performance and allocation analysis
              </Text>
            </Col>
            <Col>
              <Space>
                <Select
                  value={portfolioId}
                  onChange={setPortfolioId}
                  style={{ width: 150 }}
                >
                  <Option value="default">Main Portfolio</Option>
                  <Option value="aggressive">Aggressive</Option>
                  <Option value="conservative">Conservative</Option>
                </Select>
                
                <RangePicker
                  value={timeRange}
                  onChange={(dates) => dates && setTimeRange(dates)}
                  style={{ width: 240 }}
                />

                <Tooltip title={realTimeEnabled ? 'Disable real-time updates' : 'Enable real-time updates'}>
                  <Switch
                    checked={realTimeEnabled}
                    onChange={setRealTimeEnabled}
                    checkedChildren="Live"
                    unCheckedChildren="Manual"
                  />
                </Tooltip>
                
                <Button
                  type="primary"
                  icon={<ReloadOutlined />}
                  onClick={loadPortfolioData}
                  loading={loading}
                >
                  Refresh
                </Button>
              </Space>
            </Col>
          </Row>
        </div>

        <Tabs 
          activeKey={activeTab} 
          onChange={setActiveTab}
          className="portfolio-internal-tabs"
          items={[
            {
              key: 'overview',
              label: 'Overview',
              children: renderOverview()
            },
            {
              key: 'performance',
              label: 'Performance Analysis',
              children: renderPerformanceAnalysis()
            },
            {
              key: 'allocation',
              label: 'Asset Allocation', 
              children: renderAssetAllocation()
            },
            {
              key: 'correlation',
              label: 'Risk Correlation',
              children: (
                <Card title="Strategy Correlation Matrix">
                  {correlationMatrix ? (
                    <div style={{ textAlign: 'center', padding: '60px 0' }}>
                      <Text type="secondary">Strategy correlation heatmap would be displayed here</Text>
                    </div>
                  ) : (
                    <Spin />
                  )}
                </Card>
              )
            }
          ]}
        />
      </Card>
    </div>
  );
};

export default PortfolioVisualization;