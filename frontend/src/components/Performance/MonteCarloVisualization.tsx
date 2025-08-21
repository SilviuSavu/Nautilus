/**
 * Monte Carlo Visualization Component - Story 5.1
 * Advanced visualization for Monte Carlo simulation results
 * 
 * Features:
 * - Distribution histograms
 * - Confidence interval charts
 * - Simulation path visualization
 * - Stress test scenario results
 * - Risk metrics display
 */

import React, { useMemo } from 'react';
import {
  Card,
  Row,
  Col,
  Statistic,
  Progress,
  Table,
  Tag,
  Space,
  Typography,
  Tooltip,
  Alert
} from 'antd';
import {
  ResponsiveContainer,
  LineChart,
  Line,
  BarChart,
  Bar,
  AreaChart,
  Area,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip as RechartsTooltip,
  Legend,
  ComposedChart,
  Scatter,
  ScatterChart
} from 'recharts';
import {
  ThunderboltOutlined,
  AlertOutlined,
  RiseOutlined,
  FallOutlined,
  TrophyOutlined,
  ExperimentOutlined
} from '@ant-design/icons';
import { MonteCarloResponse, StressTestResult } from '../../types/analytics';

const { Title, Text } = Typography;

interface Props {
  data: MonteCarloResponse;
  loading?: boolean;
  height?: number;
}

const MonteCarloVisualization: React.FC<Props> = ({ data, loading, height = 600 }) => {
  // Process simulation paths for visualization
  const simulationPathsData = useMemo(() => {
    if (!data.simulation_paths || data.simulation_paths.length === 0) {
      return [];
    }

    // Take first 10 paths for visualization to avoid performance issues
    const samplePaths = data.simulation_paths.slice(0, 10);
    const maxLength = Math.max(...samplePaths.map(path => path.length));
    
    return Array.from({ length: maxLength }, (_, day) => {
      const pathData: any = { day };
      samplePaths.forEach((path, index) => {
        pathData[`path_${index}`] = path[day] || null;
      });
      return pathData;
    });
  }, [data.simulation_paths]);

  // Process confidence intervals for chart
  const confidenceData = useMemo(() => {
    const intervals = data.confidence_intervals;
    return [
      { percentile: '5th', value: intervals.percentile_5, label: 'P5' },
      { percentile: '25th', value: intervals.percentile_25, label: 'P25' },
      { percentile: '50th', value: intervals.percentile_50, label: 'Median' },
      { percentile: '75th', value: intervals.percentile_75, label: 'P75' },
      { percentile: '95th', value: intervals.percentile_95, label: 'P95' }
    ];
  }, [data.confidence_intervals]);

  // Generate histogram data for return distribution
  const histogramData = useMemo(() => {
    if (!data.simulation_paths || data.simulation_paths.length === 0) {
      return [];
    }

    // Calculate final returns from simulation paths
    const finalReturns = data.simulation_paths.map(path => 
      path.length > 0 ? path[path.length - 1] : 0
    );

    // Create histogram bins
    const minReturn = Math.min(...finalReturns);
    const maxReturn = Math.max(...finalReturns);
    const binCount = 20;
    const binWidth = (maxReturn - minReturn) / binCount;

    const bins = Array.from({ length: binCount }, (_, i) => ({
      binStart: minReturn + i * binWidth,
      binEnd: minReturn + (i + 1) * binWidth,
      count: 0,
      frequency: 0
    }));

    // Count returns in each bin
    finalReturns.forEach(ret => {
      const binIndex = Math.min(
        Math.floor((ret - minReturn) / binWidth),
        binCount - 1
      );
      bins[binIndex].count++;
    });

    // Calculate frequencies
    const totalSamples = finalReturns.length;
    bins.forEach(bin => {
      bin.frequency = (bin.count / totalSamples) * 100;
    });

    return bins.map(bin => ({
      range: `${bin.binStart.toFixed(1)}-${bin.binEnd.toFixed(1)}%`,
      count: bin.count,
      frequency: bin.frequency,
      midpoint: (bin.binStart + bin.binEnd) / 2
    }));
  }, [data.simulation_paths]);

  const formatNumber = (value: number, precision: number = 2, suffix: string = ''): string => {
    if (value === null || value === undefined || isNaN(value)) return 'N/A';
    return `${value.toFixed(precision)}${suffix}`;
  };

  const getValueColor = (value: number): string => {
    return value >= 0 ? '#52c41a' : '#f5222d';
  };

  const getRiskColor = (probability: number): string => {
    if (probability <= 0.1) return '#52c41a';
    if (probability <= 0.3) return '#faad14';
    return '#f5222d';
  };

  const renderConfidenceIntervals = () => (
    <Card title="Confidence Intervals" extra={<ExperimentOutlined />}>
      <Row gutter={[16, 16]}>
        <Col xs={24} lg={14}>
          <ResponsiveContainer width="100%" height={250}>
            <BarChart data={confidenceData} margin={{ top: 20, right: 30, left: 20, bottom: 5 }}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="label" />
              <YAxis />
              <RechartsTooltip 
                formatter={(value: any) => [`${formatNumber(value as number, 2)}%`, 'Return']}
                labelFormatter={(label) => `${label} Percentile`}
              />
              <Bar 
                dataKey="value" 
                fill={(entry: any) => {
                  const value = entry.value;
                  return value >= 0 ? '#52c41a' : '#f5222d';
                }}
                radius={[4, 4, 0, 0]}
              />
            </BarChart>
          </ResponsiveContainer>
        </Col>
        
        <Col xs={24} lg={10}>
          <Space direction="vertical" style={{ width: '100%' }}>
            {confidenceData.map((item, index) => (
              <div key={index} style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <Text strong>{item.percentile} Percentile:</Text>
                <Text style={{ color: getValueColor(item.value), fontSize: '16px', fontWeight: 'bold' }}>
                  {formatNumber(item.value, 2, '%')}
                </Text>
              </div>
            ))}
          </Space>
        </Col>
      </Row>
    </Card>
  );

  const renderRiskMetrics = () => (
    <Row gutter={[16, 16]}>
      <Col xs={12} sm={8} md={6}>
        <Card>
          <Statistic
            title="Expected Return"
            value={data.expected_return}
            precision={2}
            suffix="%"
            valueStyle={{ color: getValueColor(data.expected_return) }}
            prefix={data.expected_return >= 0 ? <RiseOutlined /> : <FallOutlined />}
          />
        </Card>
      </Col>
      
      <Col xs={12} sm={8} md={6}>
        <Card>
          <Statistic
            title="Probability of Loss"
            value={data.probability_of_loss * 100}
            precision={1}
            suffix="%"
            valueStyle={{ color: getRiskColor(data.probability_of_loss) }}
            prefix={<AlertOutlined />}
          />
        </Card>
      </Col>
      
      <Col xs={12} sm={8} md={6}>
        <Card>
          <Statistic
            title="Value at Risk (5%)"
            value={data.value_at_risk_5}
            precision={2}
            suffix="%"
            valueStyle={{ color: '#f5222d' }}
          />
        </Card>
      </Col>
      
      <Col xs={12} sm={8} md={6}>
        <Card>
          <Statistic
            title="Expected Shortfall (5%)"
            value={data.expected_shortfall_5}
            precision={2}
            suffix="%"
            valueStyle={{ color: '#f5222d' }}
          />
        </Card>
      </Col>

      <Col xs={12} sm={8} md={6}>
        <Card>
          <Statistic
            title="Best Case"
            value={data.best_case_scenario}
            precision={2}
            suffix="%"
            valueStyle={{ color: '#52c41a' }}
            prefix={<TrophyOutlined />}
          />
        </Card>
      </Col>
      
      <Col xs={12} sm={8} md={6}>
        <Card>
          <Statistic
            title="Worst Case"
            value={data.worst_case_scenario}
            precision={2}
            suffix="%"
            valueStyle={{ color: '#f5222d' }}
            prefix={<AlertOutlined />}
          />
        </Card>
      </Col>
    </Row>
  );

  const renderReturnDistribution = () => (
    <Card title="Return Distribution Histogram">
      <ResponsiveContainer width="100%" height={300}>
        <ComposedChart data={histogramData} margin={{ top: 20, right: 30, left: 20, bottom: 5 }}>
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis dataKey="range" angle={-45} textAnchor="end" height={70} />
          <YAxis yAxisId="left" />
          <YAxis yAxisId="right" orientation="right" />
          <RechartsTooltip 
            formatter={(value: any, name: string) => [
              name === 'count' ? value : `${formatNumber(value as number, 1)}%`,
              name === 'count' ? 'Occurrences' : 'Frequency'
            ]}
          />
          <Legend />
          <Bar 
            yAxisId="left"
            dataKey="count" 
            fill="#1890ff"
            name="Count"
            opacity={0.8}
          />
          <Line 
            yAxisId="right"
            type="monotone" 
            dataKey="frequency" 
            stroke="#52c41a"
            strokeWidth={3}
            name="Frequency %"
            dot={{ r: 4 }}
          />
        </ComposedChart>
      </ResponsiveContainer>
    </Card>
  );

  const renderSimulationPaths = () => (
    <Card title="Sample Simulation Paths" extra={<Text type="secondary">{data.scenarios_run} total scenarios</Text>}>
      {simulationPathsData.length > 0 ? (
        <ResponsiveContainer width="100%" height={350}>
          <LineChart data={simulationPathsData} margin={{ top: 20, right: 30, left: 20, bottom: 5 }}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="day" />
            <YAxis />
            <RechartsTooltip 
              formatter={(value: any, name: string) => [
                formatNumber(value as number, 2, '%'),
                `Path ${name.split('_')[1]}`
              ]}
              labelFormatter={(label) => `Day ${label}`}
            />
            {Array.from({ length: 10 }, (_, i) => (
              <Line
                key={`path_${i}`}
                type="monotone"
                dataKey={`path_${i}`}
                stroke={`hsl(${i * 36}, 70%, 50%)`}
                strokeWidth={1.5}
                dot={false}
                connectNulls={false}
                opacity={0.6}
              />
            ))}
          </LineChart>
        </ResponsiveContainer>
      ) : (
        <Alert 
          message="No simulation paths data available" 
          type="info" 
          description="Simulation paths are needed for path visualization"
        />
      )}
    </Card>
  );

  const renderStressTests = () => {
    if (!data.stress_test_results || data.stress_test_results.length === 0) {
      return null;
    }

    const stressColumns = [
      {
        title: 'Scenario',
        dataIndex: 'scenario_name',
        key: 'scenario_name',
        render: (text: string) => {
          const scenarioColors: { [key: string]: string } = {
            market_crash: 'red',
            high_volatility: 'orange', 
            recession: 'purple',
            interest_rate_shock: 'blue'
          };
          
          return (
            <Tag color={scenarioColors[text] || 'default'}>
              {text.replace(/_/g, ' ').toUpperCase()}
            </Tag>
          );
        }
      },
      {
        title: 'Loss Probability',
        dataIndex: 'probability_of_loss',
        key: 'probability_of_loss',
        render: (value: number) => (
          <Progress
            percent={value * 100}
            format={() => formatNumber(value * 100, 1, '%')}
            strokeColor={getRiskColor(value)}
            size="small"
          />
        ),
        sorter: (a: StressTestResult, b: StressTestResult) => a.probability_of_loss - b.probability_of_loss
      },
      {
        title: 'Expected Loss',
        dataIndex: 'expected_loss',
        key: 'expected_loss',
        render: (value: number) => (
          <Text type="danger" strong>
            {formatNumber(value, 2, '%')}
          </Text>
        ),
        sorter: (a: StressTestResult, b: StressTestResult) => a.expected_loss - b.expected_loss
      },
      {
        title: 'VaR (95%)',
        dataIndex: 'var_95',
        key: 'var_95',
        render: (value: number) => (
          <Text type="danger">
            {formatNumber(value, 2, '%')}
          </Text>
        ),
        sorter: (a: StressTestResult, b: StressTestResult) => a.var_95 - b.var_95
      }
    ];

    return (
      <Card title="Stress Test Results" extra={<AlertOutlined style={{ color: '#f5222d' }} />}>
        <Table
          dataSource={data.stress_test_results}
          columns={stressColumns}
          rowKey="scenario_name"
          size="small"
          pagination={false}
          scroll={{ x: 600 }}
        />
      </Card>
    );
  };

  const renderSummaryStats = () => (
    <Card title="Simulation Summary">
      <Row gutter={[16, 8]}>
        <Col span={8}>
          <Statistic
            title="Scenarios Run"
            value={data.scenarios_run}
            formatter={(value) => value?.toLocaleString()}
          />
        </Col>
        <Col span={8}>
          <Statistic
            title="Time Horizon"
            value={data.time_horizon_days}
            suffix="days"
          />
        </Col>
        <Col span={8}>
          <Statistic
            title="Confidence Range"
            value={`${data.confidence_intervals.percentile_5.toFixed(1)}% to ${data.confidence_intervals.percentile_95.toFixed(1)}%`}
            valueStyle={{ fontSize: '14px' }}
          />
        </Col>
      </Row>
    </Card>
  );

  return (
    <div style={{ height }}>
      <Space direction="vertical" style={{ width: '100%' }} size="large">
        {/* Summary Statistics */}
        {renderSummaryStats()}
        
        {/* Risk Metrics */}
        {renderRiskMetrics()}

        {/* Main Visualizations */}
        <Row gutter={[16, 16]}>
          <Col xs={24} lg={12}>
            {renderConfidenceIntervals()}
          </Col>
          <Col xs={24} lg={12}>
            {renderReturnDistribution()}
          </Col>
        </Row>

        {/* Simulation Paths */}
        {renderSimulationPaths()}

        {/* Stress Test Results */}
        {renderStressTests()}
      </Space>
    </div>
  );
};

export default MonteCarloVisualization;