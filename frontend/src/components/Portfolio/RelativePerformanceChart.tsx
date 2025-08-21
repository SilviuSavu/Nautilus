/**
 * Relative Performance Chart Component for strategy vs benchmark comparison
 */

import React, { useState, useEffect, useMemo } from 'react';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  BarElement,
  Title,
  Tooltip,
  Legend,
  Filler,
} from 'chart.js';
import { Line, Bar } from 'react-chartjs-2';
import { Card, Select, Switch, Row, Col, Statistic, Alert, Space, Button } from 'antd';
import { LineChartOutlined, BarChartOutlined, InfoCircleOutlined } from '@ant-design/icons';
import { 
  portfolioAggregationService, 
  PortfolioAggregation, 
  StrategyPnL 
} from '../../services/portfolioAggregationService';

ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  BarElement,
  Title,
  Tooltip,
  Legend,
  Filler
);

const { Option } = Select;

interface BenchmarkData {
  name: string;
  symbol: string;
  returns: number[];
  color: string;
}

interface RelativePerformanceChartProps {
  height?: number;
  timeframe?: '1M' | '3M' | '6M' | '1Y';
  benchmark?: 'SP500' | 'NASDAQ' | 'PORTFOLIO' | 'CUSTOM';
}

const RelativePerformanceChart: React.FC<RelativePerformanceChartProps> = ({
  height = 400,
  timeframe = '3M',
  benchmark = 'SP500'
}) => {
  const [portfolioData, setPortfolioData] = useState<PortfolioAggregation | null>(null);
  const [selectedTimeframe, setSelectedTimeframe] = useState(timeframe);
  const [selectedBenchmark, setSelectedBenchmark] = useState(benchmark);
  const [chartType, setChartType] = useState<'line' | 'bar'>('line');
  const [showCumulative, setShowCumulative] = useState(true);
  const [selectedStrategies, setSelectedStrategies] = useState<string[]>([]);
  const [loading, setLoading] = useState(true);

  // Benchmark definitions
  const benchmarks: Record<string, BenchmarkData> = {
    SP500: {
      name: 'S&P 500',
      symbol: 'SPY',
      returns: [],
      color: '#1890ff'
    },
    NASDAQ: {
      name: 'NASDAQ',
      symbol: 'QQQ',
      returns: [],
      color: '#52c41a'
    },
    PORTFOLIO: {
      name: 'Portfolio Weighted',
      symbol: 'PORTFOLIO',
      returns: [],
      color: '#722ed1'
    },
    CUSTOM: {
      name: 'Custom Benchmark',
      symbol: 'CUSTOM',
      returns: [],
      color: '#fa8c16'
    }
  };

  useEffect(() => {
    // Generate benchmark data
    generateBenchmarkData();

    const handleAggregationUpdate = (aggregation: PortfolioAggregation) => {
      setPortfolioData(aggregation);
      
      // Auto-select top 3 strategies by contribution
      const topStrategies = aggregation.strategies
        .sort((a, b) => Math.abs(b.contribution_percent) - Math.abs(a.contribution_percent))
        .slice(0, 3)
        .map(s => s.strategy_id);
      setSelectedStrategies(topStrategies);
      
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
  }, [selectedTimeframe]);

  // Generate mock benchmark data
  const generateBenchmarkData = () => {
    const periods = getTimeframePeriods(selectedTimeframe);
    
    // S&P 500 returns (mock)
    benchmarks.SP500.returns = generateMockReturns(0.0008, 0.012, periods); // ~8% annual, 12% vol
    
    // NASDAQ returns (mock)
    benchmarks.NASDAQ.returns = generateMockReturns(0.001, 0.015, periods); // ~10% annual, 15% vol
    
    // Custom benchmark returns (mock)
    benchmarks.CUSTOM.returns = generateMockReturns(0.0006, 0.010, periods); // ~6% annual, 10% vol
  };

  // Generate mock returns with specified mean and volatility
  const generateMockReturns = (dailyMean: number, annualVol: number, periods: number): number[] => {
    const returns: number[] = [];
    const dailyVol = annualVol / Math.sqrt(252);
    
    for (let i = 0; i < periods; i++) {
      const randomReturn = dailyMean + (Math.random() - 0.5) * dailyVol * 2;
      returns.push(randomReturn);
    }
    
    return returns;
  };

  // Generate strategy returns based on P&L
  const generateStrategyReturns = (strategy: StrategyPnL): number[] => {
    const periods = getTimeframePeriods(selectedTimeframe);
    const totalReturn = strategy.total_pnl / 100000; // Assume $100k base
    const dailyReturn = totalReturn / periods;
    const volatility = Math.max(0.01, Math.abs(totalReturn) * 0.5); // Higher vol for higher returns
    
    const returns: number[] = [];
    for (let i = 0; i < periods; i++) {
      const randomComponent = (Math.random() - 0.5) * volatility * 0.1;
      returns.push(dailyReturn + randomComponent);
    }
    
    return returns;
  };

  const getTimeframePeriods = (timeframe: string): number => {
    switch (timeframe) {
      case '1M': return 30;
      case '3M': return 90;
      case '6M': return 180;
      case '1Y': return 365;
      default: return 90;
    }
  };

  // Calculate cumulative returns
  const calculateCumulative = (returns: number[]): number[] => {
    const cumulative: number[] = [];
    let cumulativeValue = 0;
    
    for (const ret of returns) {
      cumulativeValue = (1 + cumulativeValue) * (1 + ret) - 1;
      cumulative.push(cumulativeValue * 100); // Convert to percentage
    }
    
    return cumulative;
  };

  // Calculate relative performance (strategy - benchmark)
  const calculateRelativePerformance = (strategyReturns: number[], benchmarkReturns: number[]): number[] => {
    return strategyReturns.map((ret, i) => (ret - (benchmarkReturns[i] || 0)) * 100);
  };

  // Generate date labels
  const generateDateLabels = (periods: number): string[] => {
    const labels: string[] = [];
    const now = new Date();
    
    for (let i = periods - 1; i >= 0; i--) {
      const date = new Date(now);
      date.setDate(date.getDate() - i);
      labels.push(date.toLocaleDateString('en-US', { month: 'short', day: 'numeric' }));
    }
    
    return labels;
  };

  // Chart data preparation
  const chartData = useMemo(() => {
    if (!portfolioData || selectedStrategies.length === 0) {
      return { labels: [], datasets: [] };
    }

    const periods = getTimeframePeriods(selectedTimeframe);
    const labels = generateDateLabels(periods);
    const benchmarkReturns = benchmarks[selectedBenchmark].returns;
    
    // Calculate portfolio weighted benchmark if selected
    if (selectedBenchmark === 'PORTFOLIO') {
      const weightedReturns = new Array(periods).fill(0);
      portfolioData.strategies.forEach(strategy => {
        const strategyReturns = generateStrategyReturns(strategy);
        const weight = strategy.weight / 100;
        strategyReturns.forEach((ret, i) => {
          weightedReturns[i] += ret * weight;
        });
      });
      benchmarks.PORTFOLIO.returns = weightedReturns;
    }

    const datasets = [];

    // Add benchmark line
    const benchmarkCumulative = showCumulative 
      ? calculateCumulative(benchmarkReturns)
      : benchmarkReturns.map(r => r * 100);

    datasets.push({
      label: benchmarks[selectedBenchmark].name,
      data: benchmarkCumulative,
      borderColor: benchmarks[selectedBenchmark].color,
      backgroundColor: `${benchmarks[selectedBenchmark].color}20`,
      borderWidth: 3,
      borderDash: [5, 5],
      fill: false,
      pointRadius: 0,
      pointHoverRadius: 4,
    });

    // Add selected strategies
    const strategyColors = ['#ff4d4f', '#52c41a', '#faad14', '#722ed1', '#13c2c2'];
    
    selectedStrategies.forEach((strategyId, index) => {
      const strategy = portfolioData.strategies.find(s => s.strategy_id === strategyId);
      if (!strategy) return;

      const strategyReturns = generateStrategyReturns(strategy);
      const strategyData = showCumulative 
        ? calculateCumulative(strategyReturns)
        : calculateRelativePerformance(strategyReturns, benchmarkReturns);

      datasets.push({
        label: strategy.strategy_name,
        data: strategyData,
        borderColor: strategyColors[index % strategyColors.length],
        backgroundColor: `${strategyColors[index % strategyColors.length]}10`,
        borderWidth: 2,
        fill: false,
        tension: 0.1,
        pointRadius: 1,
        pointHoverRadius: 4,
      });
    });

    return { labels, datasets };
  }, [portfolioData, selectedStrategies, selectedTimeframe, selectedBenchmark, showCumulative]);

  // Chart options
  const chartOptions = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: {
        position: 'top' as const,
      },
      title: {
        display: true,
        text: showCumulative 
          ? `Cumulative Performance vs ${benchmarks[selectedBenchmark].name}`
          : `Relative Performance vs ${benchmarks[selectedBenchmark].name}`,
      },
      tooltip: {
        callbacks: {
          label: function(context: any) {
            const value = context.parsed.y;
            return `${context.dataset.label}: ${value >= 0 ? '+' : ''}${value.toFixed(2)}%`;
          }
        }
      }
    },
    scales: {
      y: {
        beginAtZero: !showCumulative,
        title: {
          display: true,
          text: showCumulative ? 'Cumulative Return (%)' : 'Relative Return (%)'
        },
        ticks: {
          callback: function(value: any) {
            return value.toFixed(1) + '%';
          }
        },
        grid: {
          color: 'rgba(0, 0, 0, 0.1)',
        }
      },
      x: {
        title: {
          display: true,
          text: 'Date'
        },
        grid: {
          display: false,
        }
      }
    },
    interaction: {
      intersect: false,
      mode: 'index' as const,
    },
  };

  // Calculate performance statistics
  const performanceStats = useMemo(() => {
    if (!portfolioData || selectedStrategies.length === 0) return null;

    const benchmarkReturns = benchmarks[selectedBenchmark].returns;
    const benchmarkCumReturn = calculateCumulative(benchmarkReturns);
    const finalBenchmarkReturn = benchmarkCumReturn[benchmarkCumReturn.length - 1] || 0;

    const strategyStats = selectedStrategies.map(strategyId => {
      const strategy = portfolioData.strategies.find(s => s.strategy_id === strategyId);
      if (!strategy) return null;

      const strategyReturns = generateStrategyReturns(strategy);
      const strategyCumReturn = calculateCumulative(strategyReturns);
      const finalStrategyReturn = strategyCumReturn[strategyCumReturn.length - 1] || 0;
      const relativeReturn = finalStrategyReturn - finalBenchmarkReturn;

      // Calculate beta
      const correlation = calculateCorrelation(strategyReturns, benchmarkReturns);
      const strategyVol = calculateVolatility(strategyReturns);
      const benchmarkVol = calculateVolatility(benchmarkReturns);
      const beta = benchmarkVol > 0 ? correlation * (strategyVol / benchmarkVol) : 1;

      return {
        strategy,
        totalReturn: finalStrategyReturn,
        relativeReturn,
        beta,
        correlation
      };
    }).filter(Boolean);

    return {
      benchmarkReturn: finalBenchmarkReturn,
      strategies: strategyStats
    };
  }, [portfolioData, selectedStrategies, selectedBenchmark]);

  const calculateCorrelation = (x: number[], y: number[]): number => {
    if (x.length !== y.length || x.length < 2) return 0;
    
    const n = x.length;
    const sumX = x.reduce((a, b) => a + b, 0);
    const sumY = y.reduce((a, b) => a + b, 0);
    const sumXY = x.reduce((sum, xi, i) => sum + xi * y[i], 0);
    const sumX2 = x.reduce((sum, xi) => sum + xi * xi, 0);
    const sumY2 = y.reduce((sum, yi) => sum + yi * yi, 0);
    
    const numerator = n * sumXY - sumX * sumY;
    const denominator = Math.sqrt((n * sumX2 - sumX * sumX) * (n * sumY2 - sumY * sumY));
    
    return denominator === 0 ? 0 : numerator / denominator;
  };

  const calculateVolatility = (returns: number[]): number => {
    if (returns.length < 2) return 0;
    const mean = returns.reduce((sum, r) => sum + r, 0) / returns.length;
    const variance = returns.reduce((sum, r) => sum + Math.pow(r - mean, 2), 0) / (returns.length - 1);
    return Math.sqrt(variance);
  };

  if (loading) {
    return (
      <Card title="Relative Performance Analysis" loading={true}>
        <div style={{ height }}>Loading performance data...</div>
      </Card>
    );
  }

  return (
    <Card
      title="Relative Performance Analysis"
      extra={
        <Space>
          <Select 
            value={selectedTimeframe} 
            onChange={setSelectedTimeframe}
            style={{ width: 80 }}
          >
            <Option value="1M">1M</Option>
            <Option value="3M">3M</Option>
            <Option value="6M">6M</Option>
            <Option value="1Y">1Y</Option>
          </Select>
          
          <Select 
            value={selectedBenchmark} 
            onChange={setSelectedBenchmark}
            style={{ width: 120 }}
          >
            <Option value="SP500">S&P 500</Option>
            <Option value="NASDAQ">NASDAQ</Option>
            <Option value="PORTFOLIO">Portfolio</Option>
            <Option value="CUSTOM">Custom</Option>
          </Select>
          
          <Switch
            checked={showCumulative}
            onChange={setShowCumulative}
            checkedChildren="Cumulative"
            unCheckedChildren="Relative"
          />
          
          <Button
            icon={chartType === 'line' ? <BarChartOutlined /> : <LineChartOutlined />}
            onClick={() => setChartType(chartType === 'line' ? 'bar' : 'line')}
          >
            {chartType === 'line' ? 'Bar' : 'Line'}
          </Button>
        </Space>
      }
    >
      {/* Strategy Selection */}
      {portfolioData && (
        <div style={{ marginBottom: 16 }}>
          <span style={{ marginRight: 8, fontWeight: 'bold' }}>Strategies:</span>
          <Select
            mode="multiple"
            value={selectedStrategies}
            onChange={setSelectedStrategies}
            style={{ minWidth: 300 }}
            placeholder="Select strategies to compare"
            maxTagCount={3}
          >
            {portfolioData.strategies.map(strategy => (
              <Option key={strategy.strategy_id} value={strategy.strategy_id}>
                {strategy.strategy_name} ({strategy.contribution_percent.toFixed(1)}%)
              </Option>
            ))}
          </Select>
        </div>
      )}

      {/* Performance Statistics */}
      {performanceStats && (
        <Row gutter={16} style={{ marginBottom: 16 }}>
          <Col span={6}>
            <Statistic
              title={`${benchmarks[selectedBenchmark].name} Return`}
              value={performanceStats.benchmarkReturn}
              precision={2}
              suffix="%"
              valueStyle={{ color: '#1890ff' }}
            />
          </Col>
          {performanceStats.strategies.slice(0, 3).map((stat: any, index) => (
            <Col span={6} key={index}>
              <Statistic
                title={`${stat.strategy.strategy_name} vs Benchmark`}
                value={stat.relativeReturn}
                precision={2}
                suffix="%"
                valueStyle={{ 
                  color: stat.relativeReturn >= 0 ? '#52c41a' : '#ff4d4f'
                }}
                prefix={stat.relativeReturn >= 0 ? '+' : ''}
              />
              <div style={{ fontSize: '12px', color: '#666' }}>
                Beta: {stat.beta.toFixed(2)} | Corr: {stat.correlation.toFixed(2)}
              </div>
            </Col>
          ))}
        </Row>
      )}

      {/* Chart */}
      <div style={{ height }}>
        {chartType === 'line' ? (
          <Line data={chartData} options={chartOptions} />
        ) : (
          <Bar data={chartData} options={chartOptions} />
        )}
      </div>

      {/* Analysis Notes */}
      <Alert
        style={{ marginTop: 16 }}
        message="Performance Analysis"
        description={
          <div>
            <p><strong>Relative Performance:</strong> Shows strategy returns minus benchmark returns. Positive values indicate outperformance.</p>
            <p><strong>Beta:</strong> Measures sensitivity to benchmark movements. Beta &gt; 1 indicates higher volatility than benchmark.</p>
            <p><strong>Correlation:</strong> Measures how closely strategy moves with benchmark. Higher correlation means less diversification benefit.</p>
          </div>
        }
        type="info"
        showIcon
        icon={<InfoCircleOutlined />}
      />
    </Card>
  );
};

export default RelativePerformanceChart;