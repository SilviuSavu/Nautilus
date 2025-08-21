import React, { useState, useEffect } from 'react';
import {
  Card,
  Row,
  Col,
  Table,
  Typography,
  Space,
  Select,
  Button,
  Spin,
  notification,
  Tag,
  Tooltip,
  Progress
} from 'antd';
import {
  BarChartOutlined,
  LineChartOutlined,
  SwapOutlined,
  FilterOutlined,
  ReloadOutlined
} from '@ant-design/icons';
import type { ColumnsType } from 'antd/es/table';
import dayjs from 'dayjs';

import { StrategyInstance, PerformanceMetrics } from '../Strategy/types/strategyTypes';

const { Title, Text } = Typography;
const { Option } = Select;

interface StrategyComparisonProps {
  strategies: StrategyInstance[];
  timeRange: [dayjs.Dayjs, dayjs.Dayjs];
  className?: string;
}

interface ComparisonData {
  strategy_id: string;
  strategy_name: string;
  total_pnl: number;
  win_rate: number;
  sharpe_ratio: number;
  max_drawdown: number;
  total_trades: number;
  avg_trade_pnl: number;
  volatility: number;
  calmar_ratio: number;
  sortino_ratio: number;
  beta: number;
  correlation_to_benchmark: number;
  state: string;
  uptime_percentage: number;
  last_update: Date;
}

interface BenchmarkData {
  name: string;
  symbol: string;
  return_period: number;
  volatility: number;
  sharpe_ratio: number;
}

export const StrategyComparison: React.FC<StrategyComparisonProps> = ({
  strategies,
  timeRange,
  className
}) => {
  const [loading, setLoading] = useState(false);
  const [comparisonData, setComparisonData] = useState<ComparisonData[]>([]);
  const [selectedBenchmark, setSelectedBenchmark] = useState<string>('SPY');
  const [benchmarkData, setBenchmarkData] = useState<BenchmarkData | null>(null);
  const [sortBy, setSortBy] = useState<string>('total_pnl');
  const [sortOrder, setSortOrder] = useState<'asc' | 'desc'>('desc');

  const benchmarkOptions = [
    { value: 'SPY', label: 'S&P 500 (SPY)' },
    { value: 'QQQ', label: 'NASDAQ 100 (QQQ)' },
    { value: 'IWM', label: 'Russell 2000 (IWM)' },
    { value: 'VTI', label: 'Total Stock Market (VTI)' },
    { value: 'CASH', label: 'Cash (Risk-free)' }
  ];

  useEffect(() => {
    if (strategies.length > 0) {
      loadComparisonData();
      loadBenchmarkData();
    }
  }, [strategies, timeRange, selectedBenchmark]);

  const loadComparisonData = async () => {
    try {
      setLoading(true);
      
      const strategyIds = strategies.map(s => s.id);
      const response = await fetch('/api/v1/performance/compare', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          strategy_ids: strategyIds,
          start_date: timeRange[0].toISOString(),
          end_date: timeRange[1].toISOString(),
          benchmark: selectedBenchmark
        })
      });

      if (response.ok) {
        const data = await response.json();
        setComparisonData(data.comparisons || []);
      } else {
        throw new Error('Failed to load comparison data');
      }
    } catch (error: any) {
      console.error('Failed to load comparison data:', error);
      notification.error({
        message: 'Comparison Error',
        description: error.message || 'Failed to load strategy comparison',
        duration: 4
      });
    } finally {
      setLoading(false);
    }
  };

  const loadBenchmarkData = async () => {
    try {
      const response = await fetch(
        `/api/v1/performance/benchmarks/${selectedBenchmark}?start_date=${timeRange[0].toISOString()}&end_date=${timeRange[1].toISOString()}`
      );
      
      if (response.ok) {
        const data = await response.json();
        setBenchmarkData(data);
      }
    } catch (error: any) {
      console.error('Failed to load benchmark data:', error);
    }
  };

  const getPerformanceColor = (value: number, isPercentage: boolean = false): string => {
    const threshold = isPercentage ? 0 : 0;
    if (value > threshold) return '#3f8600';
    if (value < threshold) return '#cf1322';
    return '#666666';
  };

  const getStateColor = (state: string): string => {
    switch (state.toLowerCase()) {
      case 'running': return 'success';
      case 'paused': return 'warning';
      case 'stopped': return 'default';
      case 'error': return 'error';
      default: return 'default';
    }
  };

  const getRankingIcon = (rank: number) => {
    if (rank === 1) return '🥇';
    if (rank === 2) return '🥈';
    if (rank === 3) return '🥉';
    return `#${rank}`;
  };

  const calculateRelativePerformance = (strategyValue: number, benchmarkValue: number): number => {
    if (!benchmarkData || benchmarkValue === 0) return 0;
    return ((strategyValue - benchmarkValue) / Math.abs(benchmarkValue)) * 100;
  };

  const sortedData = [...comparisonData].sort((a, b) => {
    const aValue = a[sortBy as keyof ComparisonData] as number;
    const bValue = b[sortBy as keyof ComparisonData] as number;
    
    if (sortOrder === 'asc') return aValue - bValue;
    return bValue - aValue;
  });

  const columns: ColumnsType<ComparisonData> = [
    {
      title: 'Rank',
      key: 'rank',
      width: 60,
      render: (_, __, index) => (
        <Text strong>{getRankingIcon(index + 1)}</Text>
      )
    },
    {
      title: 'Strategy',
      dataIndex: 'strategy_name',
      key: 'strategy_name',
      render: (name: string, record: ComparisonData) => (
        <div>
          <Text strong>{name || record.strategy_id}</Text>
          <br />
          <Space size="small">
            <Tag color={getStateColor(record.state)}>{record.state}</Tag>
            <Text type="secondary" style={{ fontSize: 12 }}>
              {record.uptime_percentage.toFixed(1)}% uptime
            </Text>
          </Space>
        </div>
      )
    },
    {
      title: 'Total P&L',
      dataIndex: 'total_pnl',
      key: 'total_pnl',
      sorter: true,
      render: (pnl: number, record: ComparisonData) => {
        const relativeToBenchmark = benchmarkData ? 
          calculateRelativePerformance(pnl, benchmarkData.return_period) : 0;
        
        return (
          <div>
            <Text style={{ color: getPerformanceColor(pnl) }} strong>
              ${pnl.toFixed(2)}
            </Text>
            {benchmarkData && (
              <div style={{ fontSize: 12 }}>
                <Text type="secondary">
                  vs {selectedBenchmark}: 
                  <span style={{ color: getPerformanceColor(relativeToBenchmark) }}>
                    {relativeToBenchmark > 0 ? '+' : ''}{relativeToBenchmark.toFixed(1)}%
                  </span>
                </Text>
              </div>
            )}
          </div>
        );
      }
    },
    {
      title: 'Win Rate',
      dataIndex: 'win_rate',
      key: 'win_rate',
      sorter: true,
      render: (rate: number) => (
        <div>
          <Text style={{ color: getPerformanceColor(rate - 0.5) }}>
            {(rate * 100).toFixed(1)}%
          </Text>
          <div style={{ marginTop: 4 }}>
            <Progress 
              percent={rate * 100} 
              size="small" 
              strokeColor={rate > 0.5 ? '#3f8600' : '#cf1322'}
              showInfo={false}
            />
          </div>
        </div>
      )
    },
    {
      title: 'Sharpe Ratio',
      dataIndex: 'sharpe_ratio',
      key: 'sharpe_ratio',
      sorter: true,
      render: (ratio: number) => {
        const color = ratio > 1 ? '#3f8600' : ratio > 0 ? '#fa8c16' : '#cf1322';
        return <Text style={{ color }}>{ratio.toFixed(2)}</Text>;
      }
    },
    {
      title: 'Max Drawdown',
      dataIndex: 'max_drawdown',
      key: 'max_drawdown',
      sorter: true,
      render: (drawdown: number) => {
        const color = drawdown > 10 ? '#cf1322' : drawdown > 5 ? '#fa8c16' : '#3f8600';
        return <Text style={{ color }}>{drawdown.toFixed(2)}%</Text>;
      }
    },
    {
      title: 'Trades',
      dataIndex: 'total_trades',
      key: 'total_trades',
      sorter: true,
      render: (trades: number, record: ComparisonData) => (
        <div>
          <Text strong>{trades}</Text>
          <div style={{ fontSize: 12, color: '#666' }}>
            Avg: ${record.avg_trade_pnl.toFixed(2)}
          </div>
        </div>
      )
    },
    {
      title: 'Risk Metrics',
      key: 'risk_metrics',
      render: (_, record: ComparisonData) => (
        <Space direction="vertical" size="small">
          <Tooltip title="Volatility">
            <Text type="secondary">Vol: {record.volatility.toFixed(2)}%</Text>
          </Tooltip>
          <Tooltip title="Calmar Ratio">
            <Text type="secondary">Calmar: {record.calmar_ratio.toFixed(2)}</Text>
          </Tooltip>
          <Tooltip title="Sortino Ratio">
            <Text type="secondary">Sortino: {record.sortino_ratio.toFixed(2)}</Text>
          </Tooltip>
        </Space>
      )
    },
    {
      title: 'Benchmark Correlation',
      key: 'benchmark_correlation',
      render: (_, record: ComparisonData) => (
        <Space direction="vertical" size="small">
          <Tooltip title={`Beta relative to ${selectedBenchmark}`}>
            <Text type="secondary">β: {record.beta.toFixed(2)}</Text>
          </Tooltip>
          <Tooltip title={`Correlation to ${selectedBenchmark}`}>
            <Text type="secondary">ρ: {record.correlation_to_benchmark.toFixed(2)}</Text>
          </Tooltip>
        </Space>
      )
    }
  ];

  return (
    <div className={`strategy-comparison ${className || ''}`}>
      <Row gutter={[16, 16]}>
        {/* Controls */}
        <Col span={24}>
          <Card>
            <Row justify="space-between" align="middle">
              <Col>
                <Title level={4} style={{ margin: 0 }}>
                  <SwapOutlined style={{ marginRight: 8 }} />
                  Strategy Comparison
                </Title>
              </Col>
              <Col>
                <Space>
                  <Select
                    value={selectedBenchmark}
                    onChange={setSelectedBenchmark}
                    style={{ width: 200 }}
                  >
                    {benchmarkOptions.map(option => (
                      <Option key={option.value} value={option.value}>
                        {option.label}
                      </Option>
                    ))}
                  </Select>
                  
                  <Select
                    value={`${sortBy}-${sortOrder}`}
                    onChange={(value) => {
                      const [field, order] = value.split('-');
                      setSortBy(field);
                      setSortOrder(order as 'asc' | 'desc');
                    }}
                    style={{ width: 160 }}
                  >
                    <Option value="total_pnl-desc">P&L (High to Low)</Option>
                    <Option value="total_pnl-asc">P&L (Low to High)</Option>
                    <Option value="sharpe_ratio-desc">Sharpe (High to Low)</Option>
                    <Option value="win_rate-desc">Win Rate (High to Low)</Option>
                    <Option value="max_drawdown-asc">Drawdown (Low to High)</Option>
                  </Select>
                  
                  <Button
                    icon={<ReloadOutlined />}
                    onClick={loadComparisonData}
                    loading={loading}
                  >
                    Refresh
                  </Button>
                </Space>
              </Col>
            </Row>
          </Card>
        </Col>

        {/* Benchmark Information */}
        {benchmarkData && (
          <Col span={24}>
            <Card size="small">
              <Row gutter={16}>
                <Col>
                  <Text strong>{selectedBenchmark} Benchmark:</Text>
                </Col>
                <Col>
                  <Text>Return: {benchmarkData.return_period.toFixed(2)}%</Text>
                </Col>
                <Col>
                  <Text>Volatility: {benchmarkData.volatility.toFixed(2)}%</Text>
                </Col>
                <Col>
                  <Text>Sharpe: {benchmarkData.sharpe_ratio.toFixed(2)}</Text>
                </Col>
              </Row>
            </Card>
          </Col>
        )}

        {/* Comparison Table */}
        <Col span={24}>
          <Card>
            <Table
              columns={columns}
              dataSource={sortedData}
              rowKey="strategy_id"
              loading={loading}
              pagination={false}
              scroll={{ x: 1200 }}
              size="middle"
            />
          </Card>
        </Col>
      </Row>
    </div>
  );
};