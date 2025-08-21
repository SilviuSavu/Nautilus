/**
 * Portfolio P&L Chart Component with breakdown visualization
 */

import React, { useState, useEffect, useMemo, useRef } from 'react';
import { createChart, ColorType, IChartApi, ISeriesApi } from 'lightweight-charts';
import { Card, Select, Switch, Row, Col, Statistic, Alert, Progress } from 'antd';
import { portfolioAggregationService, PortfolioAggregation, StrategyPnL } from '../../services/portfolioAggregationService';

const { Option } = Select;

interface PortfolioPnLChartProps {
  height?: number;
  showStrategyBreakdown?: boolean;
  timeframe?: '1D' | '1W' | '1M' | '3M' | '6M' | '1Y';
}

const PortfolioPnLChart: React.FC<PortfolioPnLChartProps> = ({
  height = 400,
  showStrategyBreakdown = true,
  timeframe = '1M'
}) => {
  const [portfolioData, setPortfolioData] = useState<PortfolioAggregation | null>(null);
  const [viewType, setViewType] = useState<'line' | 'bar' | 'breakdown'>('line');
  const [showRealizedOnly, setShowRealizedOnly] = useState(false);
  const [loading, setLoading] = useState(true);
  const [selectedTimeframe, setSelectedTimeframe] = useState(timeframe);

  useEffect(() => {
    // Subscribe to portfolio aggregation updates
    const handleAggregationUpdate = (aggregation: PortfolioAggregation) => {
      setPortfolioData(aggregation);
      setLoading(false);
    };

    portfolioAggregationService.addAggregationHandler(handleAggregationUpdate);

    // Get initial data
    const initialData = portfolioAggregationService.getPortfolioAggregation();
    if (initialData) {
      setPortfolioData(initialData);
      setLoading(false);
    }

    return () => {
      portfolioAggregationService.removeAggregationHandler(handleAggregationUpdate);
    };
  }, []);

  // Generate mock historical data for demonstration
  const historicalData = useMemo(() => {
    const days = selectedTimeframe === '1D' ? 1 : 
                 selectedTimeframe === '1W' ? 7 :
                 selectedTimeframe === '1M' ? 30 :
                 selectedTimeframe === '3M' ? 90 :
                 selectedTimeframe === '6M' ? 180 : 365;

    const dates: string[] = [];
    const portfolioValues: number[] = [];
    const realizedValues: number[] = [];
    const unrealizedValues: number[] = [];

    const baseValue = portfolioData?.total_pnl || 0;
    let currentValue = baseValue - (baseValue * 0.1); // Start 10% lower

    for (let i = days; i >= 0; i--) {
      const date = new Date();
      date.setDate(date.getDate() - i);
      dates.push(date.toLocaleDateString());

      // Simulate gradual progression to current P&L
      const progress = (days - i) / days;
      const randomVariation = (Math.random() - 0.5) * baseValue * 0.02; // 2% random variation
      currentValue = baseValue * progress + randomVariation;

      portfolioValues.push(currentValue);
      realizedValues.push(currentValue * 0.6); // 60% realized
      unrealizedValues.push(currentValue * 0.4); // 40% unrealized
    }

    return { dates, portfolioValues, realizedValues, unrealizedValues };
  }, [portfolioData, selectedTimeframe]);

  const chartRef = useRef<HTMLDivElement>(null);
  const chartInstanceRef = useRef<IChartApi | null>(null);
  
  useEffect(() => {
    if (!chartRef.current || !portfolioData) return;
    
    if (chartInstanceRef.current) {
      chartInstanceRef.current.remove();
    }
    
    const chart = createChart(chartRef.current, {
      width: chartRef.current.clientWidth,
      height: height,
      layout: {
        backgroundColor: '#ffffff',
        textColor: 'rgba(33, 56, 77, 1)',
      },
      grid: {
        vertLines: {
          color: 'rgba(197, 203, 206, 0.5)',
        },
        horzLines: {
          color: 'rgba(197, 203, 206, 0.5)',
        },
      },
      crosshair: {
        mode: 1,
      },
      rightPriceScale: {
        borderColor: 'rgba(197, 203, 206, 0.8)',
      },
      timeScale: {
        borderColor: 'rgba(197, 203, 206, 0.8)',
      },
    });
    
    const lineSeries = chart.addLineSeries({
      color: portfolioData.total_pnl >= 0 ? '#52c41a' : '#ff4d4f',
      lineWidth: 2,
    });
    
    const lineData = historicalData.dates.map((date, index) => ({
      time: new Date(date).getTime() / 1000,
      value: historicalData.portfolioValues[index]
    }));
    
    lineSeries.setData(lineData);
    chartInstanceRef.current = chart;
    
    return () => {
      if (chartInstanceRef.current) {
        chartInstanceRef.current.remove();
      }
    };
  }, [portfolioData, historicalData, height]);

  const renderStrategyBreakdown = () => {
    if (!portfolioData || !showStrategyBreakdown) return null;
    
    return (
      <div style={{ marginTop: 16 }}>
        <h4>Strategy Breakdown</h4>
        {portfolioData.strategies.map((strategy, index) => (
          <div key={strategy.strategy_id} style={{ marginBottom: 8 }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 4 }}>
              <span>{strategy.strategy_name}</span>
              <span style={{ 
                color: strategy.total_pnl >= 0 ? '#52c41a' : '#ff4d4f',
                fontWeight: 'bold'
              }}>
                ${strategy.total_pnl.toLocaleString()}
              </span>
            </div>
            <Progress 
              percent={Math.abs(strategy.contribution_percent)}
              strokeColor={strategy.total_pnl >= 0 ? '#52c41a' : '#ff4d4f'}
              showInfo={false}
              size="small"
            />
          </div>
        ))}
      </div>
    );
  };

  const renderChart = () => {
    if (!portfolioData) {
      return <Alert message="No portfolio data available" type="info" />;
    }

    return (
      <div>
        <div ref={chartRef} style={{ width: '100%', height: `${height}px` }} />
        {renderStrategyBreakdown()}
      </div>
    );
  };

  const formatCurrency = (value: number) => {
    const sign = value >= 0 ? '+' : '';
    return `${sign}$${Math.abs(value).toLocaleString()}`;
  };

  if (loading) {
    return (
      <Card title="Portfolio P&L Chart" loading={loading}>
        <div style={{ height }}>Loading portfolio data...</div>
      </Card>
    );
  }

  return (
    <Card 
      title="Portfolio P&L Analysis"
      extra={
        <div style={{ display: 'flex', gap: '12px', alignItems: 'center' }}>
          <Select 
            value={selectedTimeframe} 
            onChange={setSelectedTimeframe}
            style={{ width: 80 }}
          >
            <Option value="1D">1D</Option>
            <Option value="1W">1W</Option>
            <Option value="1M">1M</Option>
            <Option value="3M">3M</Option>
            <Option value="6M">6M</Option>
            <Option value="1Y">1Y</Option>
          </Select>
          <Select 
            value={viewType} 
            onChange={setViewType}
            style={{ width: 120 }}
          >
            <Option value="line">Time Series</Option>
            <Option value="bar">By Strategy</Option>
            <Option value="breakdown">Breakdown</Option>
          </Select>
          <Switch 
            checked={showRealizedOnly}
            onChange={setShowRealizedOnly}
            checkedChildren="All P&L"
            unCheckedChildren="Realized"
          />
        </div>
      }
    >
      <Row gutter={16} style={{ marginBottom: 16 }}>
        <Col span={6}>
          <Statistic
            title="Total P&L"
            value={portfolioData?.total_pnl || 0}
            precision={2}
            valueStyle={{ 
              color: (portfolioData?.total_pnl || 0) >= 0 ? '#3f8600' : '#cf1322'
            }}
            formatter={(value) => formatCurrency(value as number)}
          />
        </Col>
        <Col span={6}>
          <Statistic
            title="Unrealized P&L"
            value={portfolioData?.unrealized_pnl || 0}
            precision={2}
            valueStyle={{ color: '#fa8c16' }}
            formatter={(value) => formatCurrency(value as number)}
          />
        </Col>
        <Col span={6}>
          <Statistic
            title="Realized P&L"
            value={portfolioData?.realized_pnl || 0}
            precision={2}
            valueStyle={{ color: '#1890ff' }}
            formatter={(value) => formatCurrency(value as number)}
          />
        </Col>
        <Col span={6}>
          <Statistic
            title="Active Strategies"
            value={portfolioData?.strategies.length || 0}
            suffix="strategies"
            valueStyle={{ color: '#722ed1' }}
          />
        </Col>
      </Row>

      <div style={{ height }}>
        {renderChart()}
      </div>

      {showStrategyBreakdown && portfolioData && portfolioData.strategies.length > 0 && (
        <div style={{ marginTop: 16 }}>
          <h4>Strategy Contribution Breakdown</h4>
          <Row gutter={8}>
            {portfolioData.strategies.slice(0, 4).map((strategy, index) => (
              <Col span={6} key={strategy.strategy_id}>
                <Card size="small">
                  <Statistic
                    title={strategy.strategy_name}
                    value={strategy.contribution_percent}
                    precision={1}
                    suffix="%"
                    valueStyle={{ 
                      color: strategy.total_pnl >= 0 ? '#3f8600' : '#cf1322',
                      fontSize: '14px'
                    }}
                  />
                  <div style={{ fontSize: '12px', color: '#666' }}>
                    {formatCurrency(strategy.total_pnl)}
                  </div>
                </Card>
              </Col>
            ))}
          </Row>
        </div>
      )}
    </Card>
  );
};

export default PortfolioPnLChart;