/**
 * Sprint 3: Performance Trend Analyzer
 * Advanced system performance trend analysis with ML insights
 * Resource utilization forecasting and capacity planning
 */

import React, { useState, useEffect } from 'react';
import {
  Card,
  Row,
  Col,
  Typography,
  Space,
  Button,
  Select,
  Statistic,
  Alert,
  Spin,
  Tag,
  Tooltip,
  Progress,
  Table,
  Timeline,
  Tabs,
  notification,
  DatePicker,
  Switch,
  Divider
} from 'antd';
import {
  ArrowUpOutlined,
  ArrowDownOutlined,
  LineChartOutlined,
  BarChartOutlined,
  PieChartOutlined,
  ReloadOutlined,
  ThunderboltOutlined,
  DatabaseOutlined,
  ApiOutlined,
  RobotOutlined,
  WarningOutlined,
  CheckCircleOutlined,
  ExclamationCircleOutlined,
  ClockCircleOutlined,
  FundOutlined
} from '@ant-design/icons';
import type { ColumnsType } from 'antd/es/table';
import dayjs from 'dayjs';

const { Title, Text } = Typography;
const { Option } = Select;
const { TabPane } = Tabs;
const { RangePicker } = DatePicker;

interface PerformanceTrend {
  metric_name: string;
  component: string;
  current_value: number;
  previous_value: number;
  trend_direction: 'improving' | 'degrading' | 'stable';
  change_percent_24h: number;
  change_percent_7d: number;
  change_percent_30d: number;
  predicted_value_24h: number;
  predicted_value_7d: number;
  confidence_score: number;
  anomaly_detected: boolean;
  last_updated: string;
}

interface CapacityPrediction {
  resource_type: 'cpu' | 'memory' | 'disk' | 'network' | 'connections' | 'throughput';
  current_utilization: number;
  predicted_exhaustion_days: number;
  recommended_actions: string[];
  severity: 'low' | 'medium' | 'high' | 'critical';
  trend_analysis: {
    daily_growth_rate: number;
    weekly_growth_rate: number;
    seasonal_pattern: boolean;
  };
}

interface PerformanceAnomaly {
  id: string;
  metric_name: string;
  component: string;
  anomaly_type: 'spike' | 'dip' | 'pattern_break' | 'trend_reversal';
  severity: 'low' | 'medium' | 'high' | 'critical';
  detected_at: string;
  anomaly_score: number;
  description: string;
  probable_causes: string[];
  recommended_actions: string[];
  resolved: boolean;
  resolution_time?: string;
}

interface MLInsight {
  insight_type: 'performance_optimization' | 'resource_allocation' | 'capacity_planning' | 'anomaly_prediction';
  title: string;
  description: string;
  confidence: number;
  impact_level: 'low' | 'medium' | 'high' | 'critical';
  recommended_actions: string[];
  time_horizon: string;
  related_metrics: string[];
}

interface PerformanceTrendAnalyzerProps {
  className?: string;
}

export const PerformanceTrendAnalyzer: React.FC<PerformanceTrendAnalyzerProps> = ({
  className
}) => {
  const [activeTab, setActiveTab] = useState('trends');
  const [performanceTrends, setPerformanceTrends] = useState<PerformanceTrend[]>([]);
  const [capacityPredictions, setCapacityPredictions] = useState<CapacityPrediction[]>([]);
  const [anomalies, setAnomalies] = useState<PerformanceAnomaly[]>([]);
  const [mlInsights, setMlInsights] = useState<MLInsight[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [timeRange, setTimeRange] = useState<string>('7d');
  const [selectedComponent, setSelectedComponent] = useState<string>('all');
  const [mlEnabled, setMlEnabled] = useState(true);
  const [autoRefresh, setAutoRefresh] = useState(true);
  const [lastUpdate, setLastUpdate] = useState<Date | null>(null);

  const fetchPerformanceData = async () => {
    try {
      setLoading(true);
      
      const [trendsResponse, capacityResponse, anomaliesResponse, insightsResponse] = await Promise.all([
        fetch(`${import.meta.env.VITE_API_BASE_URL}/api/v1/system/performance/trends?range=${timeRange}&component=${selectedComponent}`),
        fetch(`${import.meta.env.VITE_API_BASE_URL}/api/v1/system/performance/capacity`),
        fetch(`${import.meta.env.VITE_API_BASE_URL}/api/v1/system/performance/anomalies?range=${timeRange}`),
        mlEnabled ? fetch(`${import.meta.env.VITE_API_BASE_URL}/api/v1/system/performance/ml-insights`) : Promise.resolve(null)
      ]);

      if (!trendsResponse.ok || !capacityResponse.ok || !anomaliesResponse.ok) {
        throw new Error('Failed to fetch performance data');
      }

      const [trendsData, capacityData, anomaliesData, insightsData] = await Promise.all([
        trendsResponse.json(),
        capacityResponse.json(),
        anomaliesResponse.json(),
        insightsResponse ? insightsResponse.json() : []
      ]);

      setPerformanceTrends(trendsData);
      setCapacityPredictions(capacityData);
      setAnomalies(anomaliesData);
      setMlInsights(insightsData || []);
      setError(null);
      setLastUpdate(new Date());
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to fetch performance data');
      console.error('Performance data fetch error:', err);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchPerformanceData();
  }, [timeRange, selectedComponent, mlEnabled]);

  useEffect(() => {
    if (autoRefresh) {
      const interval = setInterval(fetchPerformanceData, 30000); // Refresh every 30 seconds
      return () => clearInterval(interval);
    }
  }, [autoRefresh, timeRange, selectedComponent, mlEnabled]);

  const handleRefresh = async () => {
    await fetchPerformanceData();
    notification.success({
      message: 'Performance Data Updated',
      description: 'All performance trends and predictions have been refreshed',
      duration: 2
    });
  };

  const getTrendIcon = (direction: string) => {
    switch (direction) {
      case 'improving':
        return <LineChartOutlined style={{ color: '#52c41a' }} />;
      case 'degrading':
        return <TrendingDownOutlined style={{ color: '#ff4d4f' }} />;
      default:
        return <LineChartOutlined style={{ color: '#1890ff' }} />;
    }
  };

  const getTrendColor = (direction: string) => {
    switch (direction) {
      case 'improving': return '#52c41a';
      case 'degrading': return '#ff4d4f';
      default: return '#1890ff';
    }
  };

  const getSeverityColor = (severity: string) => {
    switch (severity) {
      case 'critical': return '#ff4d4f';
      case 'high': return '#fa8c16';
      case 'medium': return '#faad14';
      case 'low': return '#52c41a';
      default: return '#d9d9d9';
    }
  };

  const formatChange = (change: number, suffix: string = '%') => {
    const color = change > 0 ? '#52c41a' : change < 0 ? '#ff4d4f' : '#1890ff';
    const prefix = change > 0 ? '+' : '';
    return (
      <Text style={{ color }}>{prefix}{change.toFixed(1)}{suffix}</Text>
    );
  };

  const trendsColumns: ColumnsType<PerformanceTrend> = [
    {
      title: 'Metric',
      dataIndex: 'metric_name',
      key: 'metric',
      render: (text, record) => (
        <Space>
          {getTrendIcon(record.trend_direction)}
          <div>
            <div>{text}</div>
            <Text type="secondary" style={{ fontSize: '11px' }}>
              {record.component}
            </Text>
          </div>
        </Space>
      )
    },
    {
      title: 'Current Value',
      dataIndex: 'current_value',
      key: 'current',
      render: (value) => <Text strong>{value.toLocaleString()}</Text>
    },
    {
      title: '24h Change',
      dataIndex: 'change_percent_24h',
      key: 'change_24h',
      render: (change) => formatChange(change),
      sorter: (a, b) => a.change_percent_24h - b.change_percent_24h
    },
    {
      title: '7d Change',
      dataIndex: 'change_percent_7d',
      key: 'change_7d',
      render: (change) => formatChange(change),
      sorter: (a, b) => a.change_percent_7d - b.change_percent_7d
    },
    {
      title: 'Predicted (24h)',
      dataIndex: 'predicted_value_24h',
      key: 'predicted',
      render: (value, record) => (
        <Tooltip title={`Confidence: ${(record.confidence_score * 100).toFixed(0)}%`}>
          <Space>
            <Text>{value.toLocaleString()}</Text>
            <Progress
              percent={record.confidence_score * 100}
              size="small"
              showInfo={false}
              style={{ width: 40 }}
            />
          </Space>
        </Tooltip>
      )
    },
    {
      title: 'Trend',
      dataIndex: 'trend_direction',
      key: 'trend',
      render: (direction, record) => (
        <Space>
          <Tag color={getTrendColor(direction)}>{direction.toUpperCase()}</Tag>
          {record.anomaly_detected && (
            <Tag color="red" icon={<WarningOutlined />}>ANOMALY</Tag>
          )}
        </Space>
      )
    }
  ];

  const capacityColumns: ColumnsType<CapacityPrediction> = [
    {
      title: 'Resource',
      dataIndex: 'resource_type',
      key: 'resource',
      render: (type) => (
        <Space>
          {type === 'cpu' && <ThunderboltOutlined />}
          {type === 'memory' && <DatabaseOutlined />}
          {type === 'disk' && <DatabaseOutlined />}
          {type === 'network' && <ApiOutlined />}
          {type === 'connections' && <ApiOutlined />}
          {type === 'throughput' && <BarChartOutlined />}
          <Text>{type.toUpperCase()}</Text>
        </Space>
      )
    },
    {
      title: 'Current Usage',
      dataIndex: 'current_utilization',
      key: 'current',
      render: (utilization) => (
        <div style={{ width: 80 }}>
          <Progress
            percent={utilization}
            size="small"
            format={() => `${utilization}%`}
            strokeColor={utilization > 80 ? '#ff4d4f' : utilization > 60 ? '#faad14' : '#52c41a'}
          />
        </div>
      )
    },
    {
      title: 'Exhaustion Prediction',
      dataIndex: 'predicted_exhaustion_days',
      key: 'exhaustion',
      render: (days, record) => {
        if (days === -1) {
          return <Text type="secondary">No exhaustion predicted</Text>;
        }
        const color = days < 7 ? '#ff4d4f' : days < 30 ? '#faad14' : '#52c41a';
        return (
          <Text style={{ color }}>
            {days} days
          </Text>
        );
      },
      sorter: (a, b) => a.predicted_exhaustion_days - b.predicted_exhaustion_days
    },
    {
      title: 'Growth Rate',
      key: 'growth',
      render: (_, record) => (
        <div>
          <div>Daily: {formatChange(record.trend_analysis.daily_growth_rate)}</div>
          <div style={{ fontSize: '11px' }}>Weekly: {formatChange(record.trend_analysis.weekly_growth_rate)}</div>
        </div>
      )
    },
    {
      title: 'Severity',
      dataIndex: 'severity',
      key: 'severity',
      render: (severity) => (
        <Tag color={getSeverityColor(severity)}>{severity.toUpperCase()}</Tag>
      )
    },
    {
      title: 'Actions',
      dataIndex: 'recommended_actions',
      key: 'actions',
      render: (actions) => (
        <Tooltip title={actions.join(', ')}>
          <Tag>{actions.length} recommendations</Tag>
        </Tooltip>
      )
    }
  ];

  const renderTrendsTab = () => (
    <div>
      {/* Trend Summary Cards */}
      <Row gutter={[16, 16]} style={{ marginBottom: 24 }}>
        <Col xs={24} sm={6}>
          <Card size="small">
            <Statistic
              title="Improving Trends"
              value={performanceTrends.filter(t => t.trend_direction === 'improving').length}
              valueStyle={{ color: '#52c41a' }}
              prefix={<LineChartOutlined />}
            />
          </Card>
        </Col>
        <Col xs={24} sm={6}>
          <Card size="small">
            <Statistic
              title="Degrading Trends"
              value={performanceTrends.filter(t => t.trend_direction === 'degrading').length}
              valueStyle={{ color: '#ff4d4f' }}
              prefix={<TrendingDownOutlined />}
            />
          </Card>
        </Col>
        <Col xs={24} sm={6}>
          <Card size="small">
            <Statistic
              title="Anomalies Detected"
              value={performanceTrends.filter(t => t.anomaly_detected).length}
              valueStyle={{ color: '#fa8c16' }}
              prefix={<WarningOutlined />}
            />
          </Card>
        </Col>
        <Col xs={24} sm={6}>
          <Card size="small">
            <Statistic
              title="Avg Confidence"
              value={(performanceTrends.reduce((sum, t) => sum + t.confidence_score, 0) / Math.max(performanceTrends.length, 1) * 100)}
              suffix="%"
              precision={0}
              valueStyle={{ color: '#1890ff' }}
              prefix={<RobotOutlined />}
            />
          </Card>
        </Col>
      </Row>

      <Table
        columns={trendsColumns}
        dataSource={performanceTrends}
        rowKey={(record) => `${record.metric_name}-${record.component}`}
        size="small"
        loading={loading}
        pagination={{ pageSize: 10 }}
      />
    </div>
  );

  const renderCapacityTab = () => (
    <div>
      {/* Critical Capacity Alerts */}
      {capacityPredictions.filter(c => c.severity === 'critical' || c.predicted_exhaustion_days < 7).length > 0 && (
        <Alert
          message="Critical Capacity Issues"
          description={`${capacityPredictions.filter(c => c.severity === 'critical').length} resources require immediate attention`}
          type="error"
          style={{ marginBottom: 16 }}
          showIcon
        />
      )}

      <Table
        columns={capacityColumns}
        dataSource={capacityPredictions}
        rowKey="resource_type"
        size="small"
        loading={loading}
        pagination={false}
      />

      {/* Capacity Planning Recommendations */}
      <Card title="Capacity Planning Recommendations" size="small" style={{ marginTop: 16 }}>
        <Timeline
          items={capacityPredictions
            .filter(c => c.recommended_actions.length > 0)
            .slice(0, 5)
            .map(prediction => ({
              color: getSeverityColor(prediction.severity),
              children: (
                <div>
                  <div>
                    <Text strong>{prediction.resource_type.toUpperCase()} Resource Planning</Text>
                    <Tag color={getSeverityColor(prediction.severity)} style={{ marginLeft: 8 }}>
                      {prediction.severity}
                    </Tag>
                  </div>
                  <ul style={{ marginTop: 8, marginBottom: 0 }}>
                    {prediction.recommended_actions.map((action, index) => (
                      <li key={index}>
                        <Text type="secondary">{action}</Text>
                      </li>
                    ))}
                  </ul>
                </div>
              )
            }))}
        />
      </Card>
    </div>
  );

  const renderAnomaliesTab = () => (
    <div>
      <Row gutter={[16, 16]} style={{ marginBottom: 16 }}>
        <Col xs={24} sm={6}>
          <Card size="small">
            <div style={{ textAlign: 'center' }}>
              <div style={{ fontSize: '24px', color: '#ff4d4f' }}>
                {anomalies.filter(a => a.severity === 'critical').length}
              </div>
              <Text type="secondary">Critical</Text>
            </div>
          </Card>
        </Col>
        <Col xs={24} sm={6}>
          <Card size="small">
            <div style={{ textAlign: 'center' }}>
              <div style={{ fontSize: '24px', color: '#fa8c16' }}>
                {anomalies.filter(a => a.severity === 'high').length}
              </div>
              <Text type="secondary">High</Text>
            </div>
          </Card>
        </Col>
        <Col xs={24} sm={6}>
          <Card size="small">
            <div style={{ textAlign: 'center' }}>
              <div style={{ fontSize: '24px', color: '#52c41a' }}>
                {anomalies.filter(a => a.resolved).length}
              </div>
              <Text type="secondary">Resolved</Text>
            </div>
          </Card>
        </Col>
        <Col xs={24} sm={6}>
          <Card size="small">
            <div style={{ textAlign: 'center' }}>
              <div style={{ fontSize: '24px', color: '#1890ff' }}>
                {anomalies.length}
              </div>
              <Text type="secondary">Total</Text>
            </div>
          </Card>
        </Col>
      </Row>

      {anomalies.map(anomaly => (
        <Card key={anomaly.id} size="small" style={{ marginBottom: 12 }}>
          <Row>
            <Col span={18}>
              <Space>
                <Tag color={getSeverityColor(anomaly.severity)}>
                  {anomaly.severity.toUpperCase()}
                </Tag>
                <Text strong>{anomaly.metric_name}</Text>
                <Text type="secondary">- {anomaly.component}</Text>
                {anomaly.resolved && <Tag color="green">RESOLVED</Tag>}
              </Space>
              <div style={{ marginTop: 8 }}>
                <Text>{anomaly.description}</Text>
              </div>
              <div style={{ marginTop: 4 }}>
                <Text type="secondary" style={{ fontSize: '11px' }}>
                  Detected: {new Date(anomaly.detected_at).toLocaleString()} | 
                  Score: {anomaly.anomaly_score.toFixed(2)}
                </Text>
              </div>
            </Col>
            <Col span={6} style={{ textAlign: 'right' }}>
              <div style={{ marginBottom: 8 }}>
                <Tag color={anomaly.anomaly_type === 'spike' ? 'red' : 'blue'}>
                  {anomaly.anomaly_type.replace('_', ' ').toUpperCase()}
                </Tag>
              </div>
              {anomaly.recommended_actions.length > 0 && (
                <Tooltip title={anomaly.recommended_actions.join(', ')}>
                  <Button size="small">
                    {anomaly.recommended_actions.length} Actions
                  </Button>
                </Tooltip>
              )}
            </Col>
          </Row>
        </Card>
      ))}
    </div>
  );

  const renderInsightsTab = () => (
    <div>
      {mlInsights.map((insight, index) => (
        <Card key={index} size="small" style={{ marginBottom: 12 }}>
          <Row>
            <Col span={20}>
              <Space style={{ marginBottom: 8 }}>
                <RobotOutlined style={{ color: '#722ed1' }} />
                <Text strong>{insight.title}</Text>
                <Tag color="purple">{insight.insight_type.replace('_', ' ').toUpperCase()}</Tag>
                <Tag color={getSeverityColor(insight.impact_level)}>
                  {insight.impact_level.toUpperCase()} IMPACT
                </Tag>
              </Space>
              <div>
                <Text>{insight.description}</Text>
              </div>
              <div style={{ marginTop: 8 }}>
                <Text strong>Recommendations:</Text>
                <ul style={{ marginTop: 4, marginBottom: 0 }}>
                  {insight.recommended_actions.map((action, actionIndex) => (
                    <li key={actionIndex}>
                      <Text type="secondary">{action}</Text>
                    </li>
                  ))}
                </ul>
              </div>
            </Col>
            <Col span={4} style={{ textAlign: 'center' }}>
              <div>
                <Progress
                  type="circle"
                  percent={insight.confidence * 100}
                  size={60}
                  format={() => `${(insight.confidence * 100).toFixed(0)}%`}
                />
                <div style={{ marginTop: 4 }}>
                  <Text type="secondary" style={{ fontSize: '11px' }}>
                    Confidence
                  </Text>
                </div>
              </div>
            </Col>
          </Row>
        </Card>
      ))}
    </div>
  );

  return (
    <div className={`performance-trend-analyzer ${className || ''}`}>
      {/* Header */}
      <div style={{ marginBottom: 24 }}>
        <Row justify="space-between" align="middle">
          <Col>
            <Title level={3} style={{ margin: 0 }}>
              <FundOutlined style={{ marginRight: 8, color: '#722ed1' }} />
              Performance Trend Analyzer
            </Title>
            <Text type="secondary">
              ML-powered performance analysis and capacity planning
            </Text>
          </Col>
          <Col>
            <Space>
              <Select
                value={selectedComponent}
                onChange={setSelectedComponent}
                style={{ width: 150 }}
                size="small"
              >
                <Option value="all">All Components</Option>
                <Option value="websocket">WebSocket</Option>
                <Option value="trading">Trading</Option>
                <Option value="risk">Risk</Option>
                <Option value="database">Database</Option>
                <Option value="system">System</Option>
              </Select>

              <Select
                value={timeRange}
                onChange={setTimeRange}
                style={{ width: 100 }}
                size="small"
              >
                <Option value="24h">24H</Option>
                <Option value="7d">7D</Option>
                <Option value="30d">30D</Option>
                <Option value="90d">90D</Option>
              </Select>

              <Switch
                checked={mlEnabled}
                onChange={setMlEnabled}
                checkedChildren="ML"
                unCheckedChildren="ML"
                size="small"
              />

              <Switch
                checked={autoRefresh}
                onChange={setAutoRefresh}
                checkedChildren="Auto"
                unCheckedChildren="Manual"
                size="small"
              />

              <Button
                type="primary"
                size="small"
                icon={<ReloadOutlined />}
                onClick={handleRefresh}
                loading={loading}
              >
                Refresh
              </Button>
            </Space>
          </Col>
        </Row>

        {lastUpdate && (
          <div style={{ marginTop: 8 }}>
            <Text type="secondary" style={{ fontSize: 12 }}>
              Last updated: {lastUpdate.toLocaleTimeString()} | 
              ML Insights: {mlEnabled ? 'Enabled' : 'Disabled'} | 
              Time Range: {timeRange}
            </Text>
          </div>
        )}
      </div>

      {/* Error Display */}
      {error && (
        <Alert
          message="Performance Analysis Error"
          description={error}
          type="error"
          style={{ marginBottom: 16 }}
          action={
            <Button size="small" onClick={handleRefresh}>
              Retry
            </Button>
          }
        />
      )}

      {/* Main Content */}
      <Tabs activeKey={activeTab} onChange={setActiveTab}>
        <TabPane 
          tab={
            <span>
              <LineChartOutlined />
              Performance Trends
            </span>
          } 
          key="trends"
        >
          {renderTrendsTab()}
        </TabPane>

        <TabPane 
          tab={
            <span>
              <BarChartOutlined />
              Capacity Planning
            </span>
          } 
          key="capacity"
        >
          {renderCapacityTab()}
        </TabPane>

        <TabPane 
          tab={
            <span>
              <WarningOutlined />
              Anomaly Detection
            </span>
          } 
          key="anomalies"
        >
          {renderAnomaliesTab()}
        </TabPane>

        <TabPane 
          tab={
            <span>
              <RobotOutlined />
              ML Insights
            </span>
          } 
          key="insights"
        >
          {renderInsightsTab()}
        </TabPane>
      </Tabs>

      {/* Loading State */}
      {loading && (
        <div style={{ textAlign: 'center', padding: 40 }}>
          <Spin size="large" />
        </div>
      )}
    </div>
  );
};

export default PerformanceTrendAnalyzer;