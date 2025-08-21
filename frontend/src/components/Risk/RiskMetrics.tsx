import React, { useState, useEffect } from 'react';
import { Card, Row, Col, Statistic, Progress, Table, Spin, Alert, Tag, Tooltip } from 'antd';
import { Line, Heatmap } from '@ant-design/plots';
import { InfoCircleOutlined, RiseOutlined, FallOutlined } from '@ant-design/icons';
import { RiskMetrics as RiskMetricsData, RiskChartData } from './types/riskTypes';
import { riskService } from './services/riskService';

interface RiskMetricsProps {
  portfolioId: string;
  refreshInterval?: number;
}

const RiskMetrics: React.FC<RiskMetricsProps> = ({ 
  portfolioId, 
  refreshInterval = 30000 
}) => {
  const [metricsData, setMetricsData] = useState<RiskMetricsData | null>(null);
  const [chartData, setChartData] = useState<RiskChartData | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const fetchData = async () => {
    try {
      setError(null);
      const [metrics, varHistory, correlationMatrix] = await Promise.all([
        riskService.getRiskMetrics(portfolioId),
        riskService.getRiskChartData(portfolioId, 'var_history'),
        riskService.getRiskChartData(portfolioId, 'correlation_matrix')
      ]);
      
      setMetricsData(metrics);
      setChartData({
        var_history: varHistory.data || [],
        correlation_matrix: correlationMatrix.data || [],
        concentration_pie: [],
        exposure_timeline: []
      });
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to fetch risk metrics');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchData();
    
    if (refreshInterval > 0) {
      const interval = setInterval(fetchData, refreshInterval);
      return () => clearInterval(interval);
    }
  }, [portfolioId, refreshInterval]);

  const getRiskLevel = (value: number, thresholds: [number, number, number]): string => {
    const [low, medium, high] = thresholds;
    if (value >= high) return 'High';
    if (value >= medium) return 'Medium';
    if (value >= low) return 'Low';
    return 'Minimal';
  };

  const getRiskColor = (value: number, thresholds: [number, number, number]): string => {
    const [low, medium, high] = thresholds;
    if (value >= high) return '#ff4d4f';
    if (value >= medium) return '#faad14';
    if (value >= low) return '#1890ff';
    return '#52c41a';
  };

  const formatCurrency = (value: string | number): string => {
    const num = typeof value === 'string' ? parseFloat(value) : value;
    return `$${num.toLocaleString()}`;
  };

  const formatPercentage = (value: number): string => {
    return `${(value * 100).toFixed(2)}%`;
  };

  if (loading) {
    return (
      <Card title="Risk Metrics">
        <div style={{ textAlign: 'center', padding: '50px 0' }}>
          <Spin size="large" />
          <p style={{ marginTop: 16 }}>Loading risk metrics...</p>
        </div>
      </Card>
    );
  }

  if (error) {
    return (
      <Card title="Risk Metrics">
        <Alert
          message="Error Loading Data"
          description={error}
          type="error"
          showIcon
          action={<button onClick={fetchData}>Retry</button>}
        />
      </Card>
    );
  }

  if (!metricsData) {
    return (
      <Card title="Risk Metrics">
        <Alert
          message="No Data Available"
          description="No risk metrics found for this portfolio."
          type="info"
          showIcon
        />
      </Card>
    );
  }

  const varData = [
    {
      title: '1-Day VaR (95%)',
      value: metricsData.var_1d_95,
      suffix: '',
      precision: 0,
      prefix: '$',
      tooltip: 'Expected maximum loss over 1 day with 95% confidence'
    },
    {
      title: '1-Day VaR (99%)',
      value: metricsData.var_1d_99,
      suffix: '',
      precision: 0,
      prefix: '$',
      tooltip: 'Expected maximum loss over 1 day with 99% confidence'
    },
    {
      title: '1-Week VaR (95%)',
      value: metricsData.var_1w_95,
      suffix: '',
      precision: 0,
      prefix: '$',
      tooltip: 'Expected maximum loss over 1 week with 95% confidence'
    },
    {
      title: '1-Month VaR (95%)',
      value: metricsData.var_1m_95,
      suffix: '',
      precision: 0,
      prefix: '$',
      tooltip: 'Expected maximum loss over 1 month with 95% confidence'
    }
  ];

  const portfolioMetrics = [
    {
      title: 'Portfolio Beta',
      value: metricsData.beta_vs_market,
      precision: 2,
      tooltip: 'Sensitivity to market movements (1.0 = market neutral)'
    },
    {
      title: 'Portfolio Volatility',
      value: metricsData.portfolio_volatility * 100,
      suffix: '%',
      precision: 1,
      tooltip: 'Annualized portfolio volatility'
    },
    {
      title: 'Sharpe Ratio',
      value: metricsData.sharpe_ratio,
      precision: 2,
      tooltip: 'Risk-adjusted return measure'
    },
    {
      title: 'Max Drawdown',
      value: metricsData.max_drawdown,
      suffix: '',
      precision: 2,
      prefix: '$',
      tooltip: 'Largest peak-to-trough decline'
    }
  ];

  const lineConfig = {
    data: chartData?.var_history || [],
    xField: 'date',
    yField: 'var_95',
    seriesField: 'type',
    smooth: true,
    animation: {
      appear: {
        animation: 'path-in',
        duration: 1000,
      },
    },
    tooltip: {
      formatter: (datum: any) => ({
        name: 'VaR 95%',
        value: formatCurrency(datum.var_95)
      })
    }
  };

  const correlationData = chartData?.correlation_matrix.map(item => ({
    x: item.symbol1,
    y: item.symbol2,
    value: item.correlation
  })) || [];

  const heatmapConfig = {
    data: correlationData,
    xField: 'x',
    yField: 'y',
    colorField: 'value',
    color: ['#313695', '#4575b4', '#74add1', '#abd9e9', '#e0f3f8', '#ffffcc', '#fee090', '#fdae61', '#f46d43', '#d73027', '#a50026'],
    meta: {
      value: {
        min: -1,
        max: 1,
      },
    },
    tooltip: {
      formatter: (datum: any) => ({
        name: `${datum.x} vs ${datum.y}`,
        value: datum.value.toFixed(3)
      })
    }
  };

  return (
    <div>
      <Row gutter={[16, 16]}>
        {/* Value at Risk Metrics */}
        <Col span={24}>
          <Card 
            title={
              <span>
                Value at Risk (VaR)
                <Tooltip title="Maximum expected loss at different confidence levels">
                  <InfoCircleOutlined style={{ marginLeft: 8, color: '#1890ff' }} />
                </Tooltip>
              </span>
            }
            size="small"
          >
            <Row gutter={16}>
              {varData.map((metric, index) => (
                <Col span={6} key={index}>
                  <Tooltip title={metric.tooltip}>
                    <Statistic
                      title={metric.title}
                      value={parseFloat(metric.value)}
                      precision={metric.precision}
                      prefix={metric.prefix}
                      suffix={metric.suffix}
                    />
                  </Tooltip>
                </Col>
              ))}
            </Row>
          </Card>
        </Col>

        {/* Portfolio Risk Metrics */}
        <Col span={24}>
          <Card 
            title={
              <span>
                Portfolio Risk Metrics
                <Tooltip title="Key risk and performance indicators">
                  <InfoCircleOutlined style={{ marginLeft: 8, color: '#1890ff' }} />
                </Tooltip>
              </span>
            }
            size="small"
          >
            <Row gutter={16}>
              {portfolioMetrics.map((metric, index) => (
                <Col span={6} key={index}>
                  <Tooltip title={metric.tooltip}>
                    <Statistic
                      title={metric.title}
                      value={metric.value}
                      precision={metric.precision}
                      prefix={metric.prefix}
                      suffix={metric.suffix}
                      valueStyle={{
                        color: metric.title === 'Sharpe Ratio' && metric.value > 1 ? '#52c41a' :
                               metric.title === 'Portfolio Beta' && Math.abs(metric.value - 1) < 0.2 ? '#52c41a' :
                               undefined
                      }}
                    />
                  </Tooltip>
                </Col>
              ))}
            </Row>
          </Card>
        </Col>

        {/* Expected Shortfall and Advanced Metrics */}
        <Col span={12}>
          <Card title="Expected Shortfall" size="small">
            <Row gutter={16}>
              <Col span={12}>
                <Statistic
                  title="ES 95%"
                  value={parseFloat(metricsData.expected_shortfall_95)}
                  precision={0}
                  prefix="$"
                  valueStyle={{ color: '#ff4d4f' }}
                />
                <p style={{ fontSize: '12px', color: '#666', marginTop: 4 }}>
                  Average loss beyond VaR
                </p>
              </Col>
              <Col span={12}>
                <Statistic
                  title="ES 99%"
                  value={parseFloat(metricsData.expected_shortfall_99)}
                  precision={0}
                  prefix="$"
                  valueStyle={{ color: '#ff4d4f' }}
                />
                <p style={{ fontSize: '12px', color: '#666', marginTop: 4 }}>
                  Tail risk measure
                </p>
              </Col>
            </Row>
          </Card>
        </Col>

        {/* Market Relationship */}
        <Col span={12}>
          <Card title="Market Relationship" size="small">
            <Row gutter={16}>
              <Col span={12}>
                <Statistic
                  title="Tracking Error"
                  value={metricsData.tracking_error * 100}
                  precision={1}
                  suffix="%"
                />
                <p style={{ fontSize: '12px', color: '#666', marginTop: 4 }}>
                  Deviation from benchmark
                </p>
              </Col>
              <Col span={12}>
                <Statistic
                  title="Information Ratio"
                  value={metricsData.information_ratio}
                  precision={2}
                  valueStyle={{
                    color: metricsData.information_ratio > 0.5 ? '#52c41a' : 
                           metricsData.information_ratio > 0 ? '#1890ff' : '#ff4d4f'
                  }}
                />
                <p style={{ fontSize: '12px', color: '#666', marginTop: 4 }}>
                  Risk-adjusted alpha
                </p>
              </Col>
            </Row>
          </Card>
        </Col>

        {/* VaR History Chart */}
        {chartData?.var_history && chartData.var_history.length > 0 && (
          <Col span={12}>
            <Card title="VaR History (30 Days)" size="small">
              <Line {...lineConfig} height={250} />
            </Card>
          </Col>
        )}

        {/* Correlation Heatmap */}
        {correlationData.length > 0 && (
          <Col span={12}>
            <Card title="Asset Correlation Matrix" size="small">
              <Heatmap {...heatmapConfig} height={250} />
            </Card>
          </Col>
        )}
      </Row>
    </div>
  );
};

export default RiskMetrics;