/**
 * Strategy Comparison Component with performance metrics table
 */

import React, { useState, useEffect, useMemo } from 'react';
import { 
  Card, 
  Table, 
  Select, 
  Radio, 
  Space, 
  Tooltip, 
  Alert, 
  Badge,
  Button,
  Dropdown,
  Menu
} from 'antd';
import { ColumnType } from 'antd/es/table';
import { 
  InfoCircleOutlined, 
  SortAscendingOutlined, 
  ExportOutlined,
  DiffOutlined,
  BarChartOutlined
} from '@ant-design/icons';
import { 
  portfolioAggregationService, 
  PortfolioAggregation, 
  StrategyPnL 
} from '../../services/portfolioAggregationService';
import {
  portfolioMetricsService,
  PerformanceMetrics,
  TimeWeightedReturn
} from '../../services/portfolioMetrics';

const { Option } = Select;

interface StrategyComparisonProps {
  period?: '1D' | '1W' | '1M' | '3M' | '6M' | '1Y';
  comparison?: 'absolute' | 'relative' | 'benchmark';
  maxStrategies?: number;
}

interface StrategyMetrics extends StrategyPnL {
  performance_metrics: PerformanceMetrics;
  time_weighted_return: TimeWeightedReturn;
  rank: number;
  percentile: number;
  risk_score: number;
}

const StrategyComparison: React.FC<StrategyComparisonProps> = ({
  period = '1M',
  comparison = 'absolute',
  maxStrategies = 10
}) => {
  const [portfolioData, setPortfolioData] = useState<PortfolioAggregation | null>(null);
  const [strategyMetrics, setStrategyMetrics] = useState<StrategyMetrics[]>([]);
  const [selectedPeriod, setSelectedPeriod] = useState(period);
  const [comparisonType, setComparisonType] = useState(comparison);
  const [sortMetric, setSortMetric] = useState<string>('total_return');
  const [loading, setLoading] = useState(true);
  const [selectedStrategies, setSelectedStrategies] = useState<string[]>([]);

  useEffect(() => {
    const handleAggregationUpdate = (aggregation: PortfolioAggregation) => {
      setPortfolioData(aggregation);
      calculateStrategyMetrics(aggregation);
      setLoading(false);
    };

    portfolioAggregationService.addAggregationHandler(handleAggregationUpdate);

    const initialData = portfolioAggregationService.getPortfolioAggregation();
    if (initialData) {
      handleAggregationUpdate(initialData);
    }

    return () => {
      portfolioAggregationService.removeAggregationHandler(handleAggregationUpdate);
    };
  }, [selectedPeriod]);

  // Calculate comprehensive metrics for each strategy
  const calculateStrategyMetrics = (aggregation: PortfolioAggregation) => {
    const metrics: StrategyMetrics[] = [];
    
    aggregation.strategies.forEach((strategy, index) => {
      // Generate mock historical data for metrics calculation
      const historicalValues = generateMockHistoricalData(strategy, selectedPeriod);
      
      // Calculate performance metrics
      const performanceMetrics = portfolioMetricsService.calculatePerformanceMetrics(
        aggregation,
        historicalValues,
        selectedPeriod
      );
      
      // Calculate time-weighted return
      const timeWeightedReturn = portfolioMetricsService.calculateTimeWeightedReturn(
        aggregation,
        historicalValues,
        selectedPeriod
      );
      
      // Calculate risk score (0-100, higher = riskier)
      const riskScore = calculateRiskScore(performanceMetrics);
      
      metrics.push({
        ...strategy,
        performance_metrics: performanceMetrics,
        time_weighted_return: timeWeightedReturn,
        rank: index + 1, // Will be recalculated after sorting
        percentile: 0, // Will be calculated after sorting
        risk_score: riskScore
      });
    });
    
    // Sort by selected metric and assign ranks
    const sortedMetrics = sortMetrics(metrics, sortMetric);
    sortedMetrics.forEach((metric, index) => {
      metric.rank = index + 1;
      metric.percentile = ((sortedMetrics.length - index) / sortedMetrics.length) * 100;
    });
    
    setStrategyMetrics(sortedMetrics.slice(0, maxStrategies));
  };

  // Generate mock historical data for strategy
  const generateMockHistoricalData = (strategy: StrategyPnL, period: string): number[] => {
    const days = getPeriodDays(period);
    const data: number[] = [];
    const startValue = 100000; // $100k starting value
    const finalValue = startValue + strategy.total_pnl;
    
    let currentValue = startValue;
    
    for (let i = 0; i <= days; i++) {
      const progress = i / days;
      const randomVariation = (Math.random() - 0.5) * startValue * 0.02; // 2% daily volatility
      currentValue = startValue + (finalValue - startValue) * progress + randomVariation;
      data.push(Math.max(currentValue, startValue * 0.5)); // Prevent going below 50% of start
    }
    
    return data;
  };

  // Calculate risk score based on performance metrics
  const calculateRiskScore = (metrics: PerformanceMetrics): number => {
    let score = 0;
    
    // Volatility component (0-30 points)
    score += Math.min(metrics.volatility * 100, 30);
    
    // Max drawdown component (0-25 points)
    score += Math.min(metrics.max_drawdown * 100, 25);
    
    // Negative Sharpe ratio penalty (0-20 points)
    if (metrics.sharpe_ratio < 0) {
      score += Math.min(Math.abs(metrics.sharpe_ratio) * 10, 20);
    }
    
    // VaR component (0-15 points)
    score += Math.min(Math.abs(metrics.var_95) * 100, 15);
    
    // Beta component (0-10 points)
    score += Math.min(Math.abs(metrics.beta - 1) * 20, 10);
    
    return Math.min(score, 100);
  };

  // Sort strategies by metric
  const sortMetrics = (metrics: StrategyMetrics[], metric: string): StrategyMetrics[] => {
    const sorted = [...metrics];
    
    sorted.sort((a, b) => {
      switch (metric) {
        case 'total_return':
          return b.performance_metrics.total_return - a.performance_metrics.total_return;
        case 'sharpe_ratio':
          return b.performance_metrics.sharpe_ratio - a.performance_metrics.sharpe_ratio;
        case 'sortino_ratio':
          return b.performance_metrics.sortino_ratio - a.performance_metrics.sortino_ratio;
        case 'max_drawdown':
          return a.performance_metrics.max_drawdown - b.performance_metrics.max_drawdown; // Lower is better
        case 'volatility':
          return a.performance_metrics.volatility - b.performance_metrics.volatility; // Lower is better
        case 'calmar_ratio':
          return b.performance_metrics.calmar_ratio - a.performance_metrics.calmar_ratio;
        case 'information_ratio':
          return b.performance_metrics.information_ratio - a.performance_metrics.information_ratio;
        case 'risk_score':
          return a.risk_score - b.risk_score; // Lower is better
        case 'total_pnl':
          return b.total_pnl - a.total_pnl;
        case 'contribution_percent':
          return Math.abs(b.contribution_percent) - Math.abs(a.contribution_percent);
        default:
          return b.total_pnl - a.total_pnl;
      }
    });
    
    return sorted;
  };

  const getPeriodDays = (period: string): number => {
    switch (period) {
      case '1D': return 1;
      case '1W': return 7;
      case '1M': return 30;
      case '3M': return 90;
      case '6M': return 180;
      case '1Y': return 365;
      default: return 30;
    }
  };

  // Format percentage with color
  const formatPercentage = (value: number, decimals: number = 2, showPlus: boolean = true): JSX.Element => {
    const color = value >= 0 ? '#52c41a' : '#ff4d4f';
    const prefix = showPlus && value >= 0 ? '+' : '';
    return (
      <span style={{ color, fontWeight: 'bold' }}>
        {prefix}{(value * 100).toFixed(decimals)}%
      </span>
    );
  };

  // Format currency with color
  const formatCurrency = (value: number): JSX.Element => {
    const color = value >= 0 ? '#52c41a' : '#ff4d4f';
    const prefix = value >= 0 ? '+' : '';
    return (
      <span style={{ color, fontWeight: 'bold' }}>
        {prefix}${Math.abs(value).toLocaleString()}
      </span>
    );
  };

  // Get risk badge
  const getRiskBadge = (riskScore: number): JSX.Element => {
    if (riskScore < 30) {
      return <Badge status="success" text="Low Risk" />;
    } else if (riskScore < 60) {
      return <Badge status="warning" text="Medium Risk" />;
    } else {
      return <Badge status="error" text="High Risk" />;
    }
  };

  // Table columns configuration
  const columns: ColumnType<StrategyMetrics>[] = [
    {
      title: 'Rank',
      dataIndex: 'rank',
      key: 'rank',
      width: 60,
      fixed: 'left',
      render: (rank: number) => (
        <div style={{ textAlign: 'center', fontWeight: 'bold' }}>
          {rank}
        </div>
      ),
    },
    {
      title: 'Strategy',
      dataIndex: 'strategy_name',
      key: 'strategy_name',
      width: 140,
      fixed: 'left',
      render: (name: string, record: StrategyMetrics) => (
        <div>
          <div style={{ fontWeight: 'bold' }}>{name}</div>
          <div style={{ fontSize: '11px', color: '#666' }}>
            {record.positions_count} positions • {record.weight.toFixed(1)}%
          </div>
        </div>
      ),
    },
    {
      title: 'P&L',
      dataIndex: 'total_pnl',
      key: 'total_pnl',
      width: 100,
      render: (pnl: number) => formatCurrency(pnl),
    },
    {
      title: (
        <Tooltip title="Total Return">
          <span>Total Return <InfoCircleOutlined /></span>
        </Tooltip>
      ),
      key: 'total_return',
      width: 100,
      render: (_, record: StrategyMetrics) => formatPercentage(record.performance_metrics.total_return),
    },
    {
      title: (
        <Tooltip title="Annualized Return">
          <span>Ann. Return <InfoCircleOutlined /></span>
        </Tooltip>
      ),
      key: 'annualized_return',
      width: 100,
      render: (_, record: StrategyMetrics) => formatPercentage(record.performance_metrics.annualized_return),
    },
    {
      title: (
        <Tooltip title="Sharpe Ratio - Risk-adjusted return">
          <span>Sharpe <InfoCircleOutlined /></span>
        </Tooltip>
      ),
      key: 'sharpe_ratio',
      width: 80,
      render: (_, record: StrategyMetrics) => (
        <span style={{ 
          color: record.performance_metrics.sharpe_ratio > 1 ? '#52c41a' : 
                 record.performance_metrics.sharpe_ratio > 0 ? '#faad14' : '#ff4d4f'
        }}>
          {record.performance_metrics.sharpe_ratio.toFixed(2)}
        </span>
      ),
    },
    {
      title: (
        <Tooltip title="Sortino Ratio - Downside risk-adjusted return">
          <span>Sortino <InfoCircleOutlined /></span>
        </Tooltip>
      ),
      key: 'sortino_ratio',
      width: 80,
      render: (_, record: StrategyMetrics) => (
        <span style={{ 
          color: record.performance_metrics.sortino_ratio > 1 ? '#52c41a' : 
                 record.performance_metrics.sortino_ratio > 0 ? '#faad14' : '#ff4d4f'
        }}>
          {record.performance_metrics.sortino_ratio.toFixed(2)}
        </span>
      ),
    },
    {
      title: (
        <Tooltip title="Maximum Drawdown">
          <span>Max DD <InfoCircleOutlined /></span>
        </Tooltip>
      ),
      key: 'max_drawdown',
      width: 80,
      render: (_, record: StrategyMetrics) => (
        <span style={{ color: '#ff4d4f' }}>
          -{(record.performance_metrics.max_drawdown * 100).toFixed(1)}%
        </span>
      ),
    },
    {
      title: (
        <Tooltip title="Volatility (Annualized)">
          <span>Volatility <InfoCircleOutlined /></span>
        </Tooltip>
      ),
      key: 'volatility',
      width: 80,
      render: (_, record: StrategyMetrics) => (
        <span style={{ color: '#1890ff' }}>
          {(record.performance_metrics.volatility * 100).toFixed(1)}%
        </span>
      ),
    },
    {
      title: (
        <Tooltip title="Calmar Ratio - Annual return / Max drawdown">
          <span>Calmar <InfoCircleOutlined /></span>
        </Tooltip>
      ),
      key: 'calmar_ratio',
      width: 80,
      render: (_, record: StrategyMetrics) => (
        <span style={{ 
          color: record.performance_metrics.calmar_ratio > 1 ? '#52c41a' : 
                 record.performance_metrics.calmar_ratio > 0 ? '#faad14' : '#ff4d4f'
        }}>
          {record.performance_metrics.calmar_ratio.toFixed(2)}
        </span>
      ),
    },
    {
      title: (
        <Tooltip title="Beta vs Benchmark">
          <span>Beta <InfoCircleOutlined /></span>
        </Tooltip>
      ),
      key: 'beta',
      width: 70,
      render: (_, record: StrategyMetrics) => (
        <span style={{ 
          color: Math.abs(record.performance_metrics.beta - 1) < 0.2 ? '#52c41a' : '#faad14'
        }}>
          {record.performance_metrics.beta.toFixed(2)}
        </span>
      ),
    },
    {
      title: (
        <Tooltip title="Win Rate">
          <span>Win Rate <InfoCircleOutlined /></span>
        </Tooltip>
      ),
      key: 'win_rate',
      width: 80,
      render: (_, record: StrategyMetrics) => (
        <span style={{ 
          color: record.performance_metrics.win_rate > 0.6 ? '#52c41a' : 
                 record.performance_metrics.win_rate > 0.4 ? '#faad14' : '#ff4d4f'
        }}>
          {(record.performance_metrics.win_rate * 100).toFixed(0)}%
        </span>
      ),
    },
    {
      title: 'Risk Score',
      key: 'risk_score',
      width: 100,
      render: (_, record: StrategyMetrics) => getRiskBadge(record.risk_score),
    },
    {
      title: 'Percentile',
      dataIndex: 'percentile',
      key: 'percentile',
      width: 80,
      render: (percentile: number) => (
        <span style={{ 
          color: percentile > 75 ? '#52c41a' : 
                 percentile > 50 ? '#faad14' : 
                 percentile > 25 ? '#ff7a45' : '#ff4d4f'
        }}>
          {percentile.toFixed(0)}th
        </span>
      ),
    },
  ];

  const exportMenu = (
    <Menu>
      <Menu.Item key="csv">Export to CSV</Menu.Item>
      <Menu.Item key="pdf">Export to PDF</Menu.Item>
      <Menu.Item key="excel">Export to Excel</Menu.Item>
    </Menu>
  );

  if (loading) {
    return (
      <Card title="Strategy Performance Comparison" loading={true}>
        <div style={{ height: 400 }}>Loading strategy comparison data...</div>
      </Card>
    );
  }

  if (!portfolioData || portfolioData.strategies.length === 0) {
    return (
      <Card title="Strategy Performance Comparison">
        <Alert
          message="No Strategy Data"
          description="No active strategies found for comparison."
          type="info"
          showIcon
        />
      </Card>
    );
  }

  return (
    <Card
      title="Strategy Performance Comparison"
      extra={
        <Space>
          <Select 
            value={selectedPeriod} 
            onChange={setSelectedPeriod}
            style={{ width: 80 }}
          >
            <Option value="1D">1D</Option>
            <Option value="1W">1W</Option>
            <Option value="1M">1M</Option>
            <Option value="3M">3M</Option>
            <Option value="6M">6M</Option>
            <Option value="1Y">1Y</Option>
          </Select>
          
          <Radio.Group value={comparisonType} onChange={(e) => setComparisonType(e.target.value)}>
            <Radio.Button value="absolute">Absolute</Radio.Button>
            <Radio.Button value="relative">Relative</Radio.Button>
            <Radio.Button value="benchmark">vs Benchmark</Radio.Button>
          </Radio.Group>
          
          <Select 
            value={sortMetric} 
            onChange={setSortMetric}
            style={{ width: 140 }}
            prefix={<SortAscendingOutlined />}
          >
            <Option value="total_return">Total Return</Option>
            <Option value="sharpe_ratio">Sharpe Ratio</Option>
            <Option value="sortino_ratio">Sortino Ratio</Option>
            <Option value="max_drawdown">Max Drawdown</Option>
            <Option value="calmar_ratio">Calmar Ratio</Option>
            <Option value="volatility">Volatility</Option>
            <Option value="risk_score">Risk Score</Option>
            <Option value="total_pnl">P&L</Option>
          </Select>
          
          <Dropdown overlay={exportMenu} trigger={['click']}>
            <Button icon={<ExportOutlined />}>Export</Button>
          </Dropdown>
        </Space>
      }
    >
      <Table
        columns={columns}
        dataSource={strategyMetrics}
        rowKey="strategy_id"
        pagination={false}
        scroll={{ x: 1400 }}
        size="small"
        bordered
        rowSelection={{
          selectedRowKeys: selectedStrategies,
          onChange: setSelectedStrategies,
          type: 'checkbox',
        }}
        rowClassName={(record, index) => {
          if (index === 0) return 'top-performer';
          if (record.risk_score > 70) return 'high-risk';
          if (record.performance_metrics.total_return < 0) return 'negative-performance';
          return '';
        }}
      />

      <style jsx>{`
        .top-performer {
          background-color: rgba(82, 196, 26, 0.1);
          border-left: 3px solid #52c41a;
        }
        .high-risk {
          background-color: rgba(255, 77, 79, 0.05);
        }
        .negative-performance {
          background-color: rgba(255, 77, 79, 0.03);
        }
      `}</style>
    </Card>
  );
};

export default StrategyComparison;