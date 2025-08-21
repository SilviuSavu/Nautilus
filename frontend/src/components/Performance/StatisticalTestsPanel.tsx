/**
 * Statistical Tests Panel Component - Story 5.1
 * Advanced statistical significance testing for portfolio performance
 * 
 * Features:
 * - Sharpe ratio significance testing
 * - Alpha significance testing with t-statistics
 * - Beta stability analysis
 * - Performance persistence testing
 * - Bootstrap confidence intervals
 * - Regime change detection
 */

import React, { useState, useMemo } from 'react';
import {
  Card,
  Row,
  Col,
  Statistic,
  Table,
  Tag,
  Space,
  Typography,
  Progress,
  Alert,
  Button,
  Tooltip,
  Badge,
  Select,
  Form,
  InputNumber,
  Switch
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
  ScatterChart,
  Scatter
} from 'recharts';
import {
  ExperimentOutlined,
  TrophyOutlined,
  AlertOutlined,
  CheckCircleOutlined,
  CloseCircleOutlined,
  LineChartOutlined,
  BarChartOutlined,
  ThunderboltOutlined,
  StarOutlined
} from '@ant-design/icons';
import { 
  StatisticalTestsResponse,
  BootstrapResult
} from '../../types/analytics';

const { Title, Text } = Typography;
const { Option } = Select;

interface Props {
  data: StatisticalTestsResponse;
  loading?: boolean;
  onRunTests?: (testType: string, significanceLevel: number) => void;
  height?: number;
}

const StatisticalTestsPanel: React.FC<Props> = ({ 
  data, 
  loading, 
  onRunTests,
  height = 700 
}) => {
  const [selectedTest, setSelectedTest] = useState<string>('sharpe');
  const [significanceLevel, setSignificanceLevel] = useState<number>(0.05);
  const [showConfidenceIntervals, setShowConfidenceIntervals] = useState(true);

  const formatNumber = (value: number, precision: number = 2, suffix: string = ''): string => {
    if (value === null || value === undefined || isNaN(value)) return 'N/A';
    return `${value.toFixed(precision)}${suffix}`;
  };

  const getSignificanceColor = (pValue: number, alpha: number = 0.05): string => {
    if (pValue < alpha / 2) return '#52c41a'; // Highly significant
    if (pValue < alpha) return '#faad14';     // Significant  
    return '#f5222d';                        // Not significant
  };

  const getSignificanceLevel = (pValue: number, alpha: number = 0.05): string => {
    if (pValue < 0.001) return '***';
    if (pValue < 0.01) return '**';
    if (pValue < alpha) return '*';
    return '';
  };

  const getConsistencyColor = (rating: string): string => {
    switch (rating) {
      case 'High': return '#52c41a';
      case 'Medium': return '#faad14';
      case 'Low': return '#f5222d';
      default: return '#666';
    }
  };

  const getStabilityColor = (score: number): string => {
    if (score >= 0.8) return '#52c41a';
    if (score >= 0.6) return '#faad14';
    return '#f5222d';
  };

  // Bootstrap results for chart visualization
  const bootstrapChartData = useMemo(() => {
    if (!data.bootstrap_results || data.bootstrap_results.length === 0) {
      return [];
    }

    return data.bootstrap_results.map(result => ({
      ...result,
      lower_bound: result.confidence_interval_95[0],
      upper_bound: result.confidence_interval_95[1],
      range: result.confidence_interval_95[1] - result.confidence_interval_95[0]
    }));
  }, [data.bootstrap_results]);

  const renderSharpeRatioTest = () => (
    <Card title="Sharpe Ratio Significance Test" extra={<TrophyOutlined />}>
      <Row gutter={[16, 16]}>
        <Col xs={24} lg={12}>
          <Space direction="vertical" style={{ width: '100%' }}>
            <Statistic
              title="Sharpe Ratio"
              value={data.sharpe_ratio_test.sharpe_ratio}
              precision={4}
              valueStyle={{ 
                fontSize: '24px',
                color: data.sharpe_ratio_test.sharpe_ratio > 1 ? '#52c41a' : '#faad14'
              }}
            />
            
            <div>
              <Text strong>Statistical Significance:</Text>
              <div style={{ marginTop: 4 }}>
                <Badge 
                  status={data.sharpe_ratio_test.is_significant ? 'success' : 'error'}
                  text={
                    <Space>
                      <Text strong>
                        {data.sharpe_ratio_test.is_significant ? 'Significant' : 'Not Significant'}
                      </Text>
                      <Text type="secondary">
                        {getSignificanceLevel(data.sharpe_ratio_test.p_value)}
                      </Text>
                    </Space>
                  }
                />
              </div>
            </div>

            <div>
              <Text strong>95% Confidence Interval:</Text>
              <div style={{ marginTop: 4 }}>
                <Tag color="blue">
                  [{formatNumber(data.sharpe_ratio_test.confidence_interval[0], 4)}, {formatNumber(data.sharpe_ratio_test.confidence_interval[1], 4)}]
                </Tag>
              </div>
            </div>
          </Space>
        </Col>
        
        <Col xs={24} lg={12}>
          <Space direction="vertical" style={{ width: '100%' }}>
            <Statistic
              title="T-Statistic"
              value={data.sharpe_ratio_test.t_statistic}
              precision={3}
              valueStyle={{ color: '#1890ff' }}
            />
            
            <Statistic
              title="P-Value"
              value={data.sharpe_ratio_test.p_value}
              precision={4}
              valueStyle={{ 
                color: getSignificanceColor(data.sharpe_ratio_test.p_value, significanceLevel)
              }}
              suffix={
                <Tooltip title={`Significance level: ${significanceLevel}`}>
                  <Tag color={getSignificanceColor(data.sharpe_ratio_test.p_value, significanceLevel)}>
                    {data.sharpe_ratio_test.p_value < significanceLevel ? 'Significant' : 'Not Significant'}
                  </Tag>
                </Tooltip>
              }
            />

            <div>
              <Text strong>Interpretation:</Text>
              <div style={{ marginTop: 4 }}>
                <Alert
                  type={data.sharpe_ratio_test.is_significant ? 'success' : 'warning'}
                  message={
                    data.sharpe_ratio_test.is_significant 
                      ? 'The Sharpe ratio is statistically significant, indicating genuine risk-adjusted outperformance.'
                      : 'The Sharpe ratio is not statistically significant, suggesting performance may be due to chance.'
                  }
                  showIcon
                  size="small"
                />
              </div>
            </div>
          </Space>
        </Col>
      </Row>
    </Card>
  );

  const renderAlphaSignificanceTest = () => (
    <Card title="Alpha Significance Test" extra={<StarOutlined />}>
      <Row gutter={[16, 16]}>
        <Col xs={24} lg={12}>
          <Space direction="vertical" style={{ width: '100%' }}>
            <Statistic
              title="Alpha"
              value={data.alpha_significance_test.alpha}
              precision={4}
              suffix="%"
              valueStyle={{ 
                fontSize: '24px',
                color: data.alpha_significance_test.alpha > 0 ? '#52c41a' : '#f5222d'
              }}
              prefix={data.alpha_significance_test.alpha > 0 ? 
                <CheckCircleOutlined /> : <CloseCircleOutlined />
              }
            />
            
            <div>
              <Text strong>Statistical Significance:</Text>
              <div style={{ marginTop: 4 }}>
                <Badge 
                  status={data.alpha_significance_test.is_significant ? 'success' : 'error'}
                  text={
                    <Space>
                      <Text strong>
                        {data.alpha_significance_test.is_significant ? 'Significant Alpha' : 'No Significant Alpha'}
                      </Text>
                      <Text type="secondary">
                        {getSignificanceLevel(data.alpha_significance_test.p_value)}
                      </Text>
                    </Space>
                  }
                />
              </div>
            </div>
          </Space>
        </Col>
        
        <Col xs={24} lg={12}>
          <Space direction="vertical" style={{ width: '100%' }}>
            <Statistic
              title="T-Statistic"
              value={data.alpha_significance_test.t_statistic}
              precision={3}
              valueStyle={{ color: '#1890ff' }}
            />
            
            <Statistic
              title="P-Value"
              value={data.alpha_significance_test.p_value}
              precision={4}
              valueStyle={{ 
                color: getSignificanceColor(data.alpha_significance_test.p_value, significanceLevel)
              }}
            />

            <div>
              <Text strong>95% Confidence Interval:</Text>
              <div style={{ marginTop: 4 }}>
                <Tag color={data.alpha_significance_test.alpha > 0 ? 'green' : 'red'}>
                  [{formatNumber(data.alpha_significance_test.confidence_interval[0], 4, '%')}, {formatNumber(data.alpha_significance_test.confidence_interval[1], 4, '%')}]
                </Tag>
              </div>
            </div>
          </Space>
        </Col>
      </Row>
    </Card>
  );

  const renderBetaStabilityTest = () => (
    <Card title="Beta Stability Analysis" extra={<LineChartOutlined />}>
      <Row gutter={[16, 16]}>
        <Col xs={24} lg={8}>
          <Statistic
            title="Current Beta"
            value={data.beta_stability_test.beta}
            precision={3}
            valueStyle={{ 
              color: data.beta_stability_test.beta <= 1.2 && data.beta_stability_test.beta >= 0.8 ? '#52c41a' : '#faad14'
            }}
          />
        </Col>
        
        <Col xs={24} lg={8}>
          <Statistic
            title="Rolling Beta Std Dev"
            value={data.beta_stability_test.rolling_beta_std}
            precision={4}
            valueStyle={{ 
              color: data.beta_stability_test.rolling_beta_std < 0.2 ? '#52c41a' : '#f5222d'
            }}
          />
        </Col>
        
        <Col xs={24} lg={8}>
          <Statistic
            title="Regime Changes"
            value={data.beta_stability_test.regime_changes_detected}
            valueStyle={{ 
              color: data.beta_stability_test.regime_changes_detected <= 2 ? '#52c41a' : '#f5222d'
            }}
            prefix={<ThunderboltOutlined />}
          />
        </Col>

        <Col xs={24}>
          <div>
            <Text strong>Stability Score</Text>
            <Progress
              percent={data.beta_stability_test.stability_score * 100}
              strokeColor={getStabilityColor(data.beta_stability_test.stability_score)}
              format={(percent) => 
                <Text strong style={{ color: getStabilityColor(data.beta_stability_test.stability_score) }}>
                  {formatNumber(data.beta_stability_test.stability_score, 3)} 
                  ({percent?.toFixed(1)}%)
                </Text>
              }
            />
          </div>
        </Col>

        <Col xs={24}>
          <Alert
            type={data.beta_stability_test.stability_score > 0.7 ? 'success' : 'warning'}
            message={
              data.beta_stability_test.stability_score > 0.7 
                ? 'Beta is stable over time, indicating consistent market sensitivity.'
                : 'Beta shows instability, suggesting varying market sensitivity across different periods.'
            }
            showIcon
          />
        </Col>
      </Row>
    </Card>
  );

  const renderPerformancePersistence = () => (
    <Card title="Performance Persistence Analysis" extra={<TrophyOutlined />}>
      <Row gutter={[16, 16]}>
        <Col xs={24} lg={8}>
          <div style={{ textAlign: 'center' }}>
            <Progress
              type="circle"
              percent={data.performance_persistence.persistence_score * 100}
              format={() => data.performance_persistence.consistency_rating}
              strokeColor={getConsistencyColor(data.performance_persistence.consistency_rating)}
              width={120}
            />
            <div style={{ marginTop: 8 }}>
              <Text strong>Consistency Rating</Text>
            </div>
          </div>
        </Col>
        
        <Col xs={24} lg={16}>
          <Space direction="vertical" style={{ width: '100%' }}>
            <Statistic
              title="Persistence Score"
              value={data.performance_persistence.persistence_score}
              precision={3}
              valueStyle={{ 
                color: getConsistencyColor(data.performance_persistence.consistency_rating),
                fontSize: '20px'
              }}
            />
            
            <Statistic
              title="Consecutive Winning Periods"
              value={data.performance_persistence.consecutive_winning_periods}
              valueStyle={{ color: '#52c41a' }}
              prefix={<TrophyOutlined />}
            />

            <div>
              <Text strong>Performance Consistency:</Text>
              <div style={{ marginTop: 4 }}>
                <Tag color={getConsistencyColor(data.performance_persistence.consistency_rating)}>
                  {data.performance_persistence.consistency_rating} Consistency
                </Tag>
              </div>
            </div>

            <Alert
              type={
                data.performance_persistence.consistency_rating === 'High' ? 'success' :
                data.performance_persistence.consistency_rating === 'Medium' ? 'info' : 'warning'
              }
              message={
                data.performance_persistence.consistency_rating === 'High' 
                  ? 'Strong performance persistence indicates sustainable strategy effectiveness.'
                  : data.performance_persistence.consistency_rating === 'Medium'
                  ? 'Moderate performance persistence suggests generally consistent results with some variation.'
                  : 'Low performance persistence indicates inconsistent results and potential strategy issues.'
              }
              showIcon
              size="small"
            />
          </Space>
        </Col>
      </Row>
    </Card>
  );

  const renderBootstrapResults = () => {
    if (!data.bootstrap_results || data.bootstrap_results.length === 0) {
      return null;
    }

    const columns = [
      {
        title: 'Metric',
        dataIndex: 'metric',
        key: 'metric',
        render: (text: string) => <Text strong>{text}</Text>
      },
      {
        title: 'Bootstrap Mean',
        dataIndex: 'bootstrap_mean',
        key: 'bootstrap_mean',
        render: (value: number) => formatNumber(value, 4),
        sorter: (a: BootstrapResult, b: BootstrapResult) => a.bootstrap_mean - b.bootstrap_mean
      },
      {
        title: 'Bootstrap Std Dev',
        dataIndex: 'bootstrap_std',
        key: 'bootstrap_std',
        render: (value: number) => formatNumber(value, 4),
        sorter: (a: BootstrapResult, b: BootstrapResult) => a.bootstrap_std - b.bootstrap_std
      },
      {
        title: '95% Confidence Interval',
        dataIndex: 'confidence_interval_95',
        key: 'confidence_interval_95',
        render: (interval: [number, number]) => (
          <Tag color="blue">
            [{formatNumber(interval[0], 4)}, {formatNumber(interval[1], 4)}]
          </Tag>
        )
      }
    ];

    return (
      <Card title="Bootstrap Confidence Intervals" extra={<ExperimentOutlined />}>
        <Row gutter={[16, 16]}>
          <Col xs={24} lg={16}>
            <Table
              dataSource={data.bootstrap_results}
              columns={columns}
              rowKey="metric"
              size="small"
              pagination={false}
            />
          </Col>
          
          <Col xs={24} lg={8}>
            <ResponsiveContainer width="100%" height={200}>
              <BarChart data={bootstrapChartData} margin={{ top: 20, right: 30, left: 20, bottom: 5 }}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="metric" angle={-45} textAnchor="end" height={70} />
                <YAxis />
                <RechartsTooltip 
                  formatter={(value: any, name: string) => [formatNumber(value as number, 4), name]}
                />
                <Bar dataKey="range" fill="#1890ff" name="95% CI Range" />
              </BarChart>
            </ResponsiveContainer>
          </Col>
        </Row>
      </Card>
    );
  };

  const renderTestControls = () => (
    <Card title="Test Configuration" size="small">
      <Row gutter={[16, 8]} align="middle">
        <Col>
          <Space>
            <Text strong>Test Type:</Text>
            <Select
              value={selectedTest}
              onChange={setSelectedTest}
              style={{ width: 120 }}
            >
              <Option value="sharpe">Sharpe Ratio</Option>
              <Option value="alpha">Alpha</Option>
              <Option value="beta">Beta Stability</Option>
              <Option value="persistence">Persistence</Option>
            </Select>
          </Space>
        </Col>
        
        <Col>
          <Space>
            <Text strong>Significance Level:</Text>
            <InputNumber
              value={significanceLevel}
              onChange={(value) => setSignificanceLevel(value || 0.05)}
              min={0.001}
              max={0.1}
              step={0.01}
              precision={3}
              style={{ width: 80 }}
            />
          </Space>
        </Col>
        
        <Col>
          <Space>
            <Switch
              checked={showConfidenceIntervals}
              onChange={setShowConfidenceIntervals}
              size="small"
            />
            <Text>Show CI</Text>
          </Space>
        </Col>
        
        <Col>
          <Button
            type="primary"
            icon={<ExperimentOutlined />}
            onClick={() => onRunTests?.(selectedTest, significanceLevel)}
            loading={loading}
            size="small"
          >
            Run Tests
          </Button>
        </Col>
      </Row>
    </Card>
  );

  return (
    <div style={{ height }}>
      <Space direction="vertical" style={{ width: '100%' }} size="large">
        {/* Test Controls */}
        {renderTestControls()}

        {/* Statistical Tests Results */}
        {renderSharpeRatioTest()}
        
        {renderAlphaSignificanceTest()}
        
        <Row gutter={[16, 16]}>
          <Col xs={24} lg={12}>
            {renderBetaStabilityTest()}
          </Col>
          <Col xs={24} lg={12}>
            {renderPerformancePersistence()}
          </Col>
        </Row>

        {/* Bootstrap Results */}
        {showConfidenceIntervals && renderBootstrapResults()}
      </Space>
    </div>
  );
};

export default StatisticalTestsPanel;