/**
 * Sprint 3: Prometheus Metrics Dashboard
 * Real-time integration with Prometheus metrics for comprehensive monitoring
 * Displays custom trading platform metrics and system performance
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
  Table,
  Alert,
  Spin,
  Tag,
  Tooltip,
  Progress,
  List,
  Divider,
  notification,
  DatePicker,
  InputNumber
} from 'antd';
import {
  ReloadOutlined,
  CloudServerOutlined,
  LineChartOutlined,
  ThunderboltOutlined,
  DatabaseOutlined,
  ApiOutlined,
  SettingOutlined,
  ExportOutlined,
  FilterOutlined
} from '@ant-design/icons';
import type { ColumnsType } from 'antd/es/table';
import dayjs from 'dayjs';

const { Title, Text } = Typography;
const { Option } = Select;
const { RangePicker } = DatePicker;

interface PrometheusMetric {
  metric_name: string;
  value: number;
  timestamp: number;
  labels: Record<string, string>;
  help: string;
  type: 'counter' | 'gauge' | 'histogram' | 'summary';
}

interface PrometheusQuery {
  query: string;
  result: PrometheusMetric[];
  result_type: 'matrix' | 'vector' | 'scalar' | 'string';
  execution_time_ms: number;
}

interface TradingMetrics {
  websocket_connections_active: number;
  websocket_messages_sent_total: number;
  websocket_messages_received_total: number;
  websocket_errors_total: number;
  
  trading_orders_total: number;
  trading_orders_filled_total: number;
  trading_orders_cancelled_total: number;
  trading_latency_seconds: number;
  
  risk_checks_total: number;
  risk_violations_total: number;
  risk_processing_seconds: number;
  
  strategy_deployments_total: number;
  strategy_errors_total: number;
  strategy_performance_ratio: number;
  
  database_queries_total: number;
  database_query_duration_seconds: number;
  database_connections_active: number;
  
  system_cpu_usage_ratio: number;
  system_memory_usage_ratio: number;
  system_disk_usage_ratio: number;
  system_network_bytes_total: number;
}

interface PrometheusMetricsDashboardProps {
  className?: string;
}

export const PrometheusMetricsDashboard: React.FC<PrometheusMetricsDashboardProps> = ({
  className
}) => {
  const [metrics, setMetrics] = useState<TradingMetrics | null>(null);
  const [customQueries, setCustomQueries] = useState<PrometheusQuery[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [selectedMetricType, setSelectedMetricType] = useState<string>('all');
  const [timeRange, setTimeRange] = useState<string>('1h');
  const [refreshInterval, setRefreshInterval] = useState<number>(30);
  const [lastUpdate, setLastUpdate] = useState<Date | null>(null);

  // Pre-defined critical trading metrics queries
  const predefinedQueries = [
    {
      name: 'WebSocket Connection Rate',
      query: 'rate(websocket_connections_total[5m])',
      description: 'New WebSocket connections per second'
    },
    {
      name: 'Order Fill Rate',
      query: 'rate(trading_orders_filled_total[1m])',
      description: 'Orders filled per second'
    },
    {
      name: 'Trading Latency P95',
      query: 'histogram_quantile(0.95, rate(trading_latency_seconds_bucket[5m]))',
      description: '95th percentile trading latency'
    },
    {
      name: 'Risk Violation Rate',
      query: 'rate(risk_violations_total[5m])',
      description: 'Risk violations per second'
    },
    {
      name: 'Database Query Rate',
      query: 'rate(database_queries_total[1m])',
      description: 'Database queries per second'
    },
    {
      name: 'System Resource Utilization',
      query: 'avg(system_cpu_usage_ratio) * 100',
      description: 'Average CPU utilization percentage'
    }
  ];

  const fetchPrometheusMetrics = async () => {
    try {
      setLoading(true);
      
      // Fetch aggregated trading metrics
      const metricsResponse = await fetch(
        `${import.meta.env.VITE_API_BASE_URL}/api/v1/system/metrics/prometheus/trading?range=${timeRange}`
      );
      
      if (!metricsResponse.ok) {
        throw new Error(`Prometheus metrics fetch failed: ${metricsResponse.statusText}`);
      }
      
      const metricsData: TradingMetrics = await metricsResponse.json();
      setMetrics(metricsData);

      // Execute predefined queries
      const queryPromises = predefinedQueries.map(async (queryDef) => {
        try {
          const response = await fetch(
            `${import.meta.env.VITE_API_BASE_URL}/api/v1/system/metrics/prometheus/query`,
            {
              method: 'POST',
              headers: {
                'Content-Type': 'application/json'
              },
              body: JSON.stringify({
                query: queryDef.query,
                time_range: timeRange
              })
            }
          );

          if (response.ok) {
            const queryData: PrometheusQuery = await response.json();
            return {
              ...queryData,
              name: queryDef.name,
              description: queryDef.description
            };
          }
          return null;
        } catch (err) {
          console.warn(`Query failed: ${queryDef.name}`, err);
          return null;
        }
      });

      const queryResults = await Promise.all(queryPromises);
      setCustomQueries(queryResults.filter(result => result !== null) as PrometheusQuery[]);
      
      setError(null);
      setLastUpdate(new Date());
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to fetch Prometheus metrics');
      console.error('Prometheus metrics fetch error:', err);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchPrometheusMetrics();
  }, [timeRange]);

  useEffect(() => {
    const interval = setInterval(() => {
      if (refreshInterval > 0) {
        fetchPrometheusMetrics();
      }
    }, refreshInterval * 1000);

    return () => clearInterval(interval);
  }, [refreshInterval, timeRange]);

  const handleRefresh = async () => {
    await fetchPrometheusMetrics();
    notification.success({
      message: 'Prometheus Metrics Updated',
      description: 'All metrics have been refreshed from Prometheus',
      duration: 2
    });
  };

  const exportMetrics = () => {
    if (!metrics) return;
    
    const exportData = {
      timestamp: new Date().toISOString(),
      time_range: timeRange,
      metrics: metrics,
      custom_queries: customQueries
    };
    
    const blob = new Blob([JSON.stringify(exportData, null, 2)], {
      type: 'application/json'
    });
    
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `prometheus-metrics-${new Date().getTime()}.json`;
    a.click();
    URL.revokeObjectURL(url);
  };

  const getMetricColor = (value: number, type: 'percentage' | 'rate' | 'latency') => {
    if (type === 'percentage') {
      if (value > 90) return '#ff4d4f';
      if (value > 70) return '#faad14';
      return '#52c41a';
    } else if (type === 'latency') {
      if (value > 100) return '#ff4d4f';
      if (value > 50) return '#faad14';
      return '#52c41a';
    } else if (type === 'rate') {
      return '#1890ff';
    }
    return '#52c41a';
  };

  const queryColumns: ColumnsType<any> = [
    {
      title: 'Query Name',
      dataIndex: 'name',
      key: 'name',
      render: (text: string) => <Text strong>{text}</Text>
    },
    {
      title: 'Description',
      dataIndex: 'description',
      key: 'description',
      render: (text: string) => <Text type="secondary">{text}</Text>
    },
    {
      title: 'Result Count',
      dataIndex: 'result',
      key: 'result_count',
      render: (result: PrometheusMetric[]) => (
        <Tag color="blue">{result?.length || 0} metrics</Tag>
      )
    },
    {
      title: 'Execution Time',
      dataIndex: 'execution_time_ms',
      key: 'execution_time',
      render: (time: number) => (
        <Text style={{ color: getMetricColor(time, 'latency') }}>
          {time}ms
        </Text>
      )
    },
    {
      title: 'Type',
      dataIndex: 'result_type',
      key: 'result_type',
      render: (type: string) => <Tag>{type}</Tag>
    }
  ];

  if (loading && !metrics) {
    return (
      <div style={{ textAlign: 'center', padding: 60 }}>
        <Spin size="large" />
        <div style={{ marginTop: 16 }}>
          <Text type="secondary">Loading Prometheus metrics...</Text>
        </div>
      </div>
    );
  }

  return (
    <div className={`prometheus-metrics-dashboard ${className || ''}`}>
      {/* Header */}
      <div style={{ marginBottom: 24 }}>
        <Row justify="space-between" align="middle">
          <Col>
            <Title level={3} style={{ margin: 0 }}>
              <CloudServerOutlined style={{ marginRight: 8, color: '#e6522c' }} />
              Prometheus Metrics
            </Title>
            <Text type="secondary">
              Real-time metrics from Prometheus monitoring stack
            </Text>
          </Col>
          <Col>
            <Space>
              <Select
                value={timeRange}
                onChange={setTimeRange}
                style={{ width: 100 }}
                size="small"
              >
                <Option value="5m">5m</Option>
                <Option value="15m">15m</Option>
                <Option value="1h">1h</Option>
                <Option value="6h">6h</Option>
                <Option value="24h">24h</Option>
              </Select>

              <InputNumber
                value={refreshInterval}
                onChange={(value) => setRefreshInterval(value || 30)}
                min={5}
                max={300}
                step={5}
                suffix="s"
                size="small"
                style={{ width: 80 }}
              />

              <Button
                size="small"
                icon={<ExportOutlined />}
                onClick={exportMetrics}
                disabled={!metrics}
              >
                Export
              </Button>

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
              Auto-refresh: {refreshInterval}s | 
              Time range: {timeRange}
            </Text>
          </div>
        )}
      </div>

      {/* Error Display */}
      {error && (
        <Alert
          message="Prometheus Connection Error"
          description={error}
          type="error"
          style={{ marginBottom: 16 }}
          action={
            <Button size="small" onClick={handleRefresh}>
              Retry Connection
            </Button>
          }
        />
      )}

      {/* Trading Metrics Overview */}
      {metrics && (
        <>
          <Row gutter={[16, 16]} style={{ marginBottom: 24 }}>
            <Col xs={24} sm={12} md={6}>
              <Card size="small">
                <Statistic
                  title="WebSocket Connections"
                  value={metrics.websocket_connections_active}
                  valueStyle={{ color: '#1890ff' }}
                  prefix={<ApiOutlined />}
                />
                <Progress 
                  percent={Math.min((metrics.websocket_connections_active / 1000) * 100, 100)}
                  size="small"
                  showInfo={false}
                />
                <Text type="secondary" style={{ fontSize: '11px' }}>
                  Target: 1000+ concurrent
                </Text>
              </Card>
            </Col>

            <Col xs={24} sm={12} md={6}>
              <Card size="small">
                <Statistic
                  title="Messages/sec"
                  value={metrics.websocket_messages_sent_total + metrics.websocket_messages_received_total}
                  valueStyle={{ color: '#722ed1' }}
                  prefix={<ThunderboltOutlined />}
                />
                <Text type="secondary" style={{ fontSize: '11px' }}>
                  Total WebSocket throughput
                </Text>
              </Card>
            </Col>

            <Col xs={24} sm={12} md={6}>
              <Card size="small">
                <Statistic
                  title="Trading Latency"
                  value={metrics.trading_latency_seconds * 1000}
                  suffix="ms"
                  precision={1}
                  valueStyle={{ color: getMetricColor(metrics.trading_latency_seconds * 1000, 'latency') }}
                  prefix={<LineChartOutlined />}
                />
                <Text type="secondary" style={{ fontSize: '11px' }}>
                  Average execution latency
                </Text>
              </Card>
            </Col>

            <Col xs={24} sm={12} md={6}>
              <Card size="small">
                <Statistic
                  title="Fill Rate"
                  value={((metrics.trading_orders_filled_total / Math.max(metrics.trading_orders_total, 1)) * 100)}
                  suffix="%"
                  precision={1}
                  valueStyle={{ color: '#52c41a' }}
                />
                <Text type="secondary" style={{ fontSize: '11px' }}>
                  Orders successfully filled
                </Text>
              </Card>
            </Col>
          </Row>

          {/* Detailed Metrics Cards */}
          <Row gutter={[16, 16]} style={{ marginBottom: 24 }}>
            <Col xs={24} lg={12}>
              <Card title="Risk Management Metrics" size="small">
                <Row gutter={16}>
                  <Col span={8}>
                    <Statistic
                      title="Risk Checks"
                      value={metrics.risk_checks_total}
                      valueStyle={{ color: '#fa541c' }}
                    />
                  </Col>
                  <Col span={8}>
                    <Statistic
                      title="Violations"
                      value={metrics.risk_violations_total}
                      valueStyle={{ color: metrics.risk_violations_total > 0 ? '#ff4d4f' : '#52c41a' }}
                    />
                  </Col>
                  <Col span={8}>
                    <Statistic
                      title="Processing Time"
                      value={metrics.risk_processing_seconds * 1000}
                      suffix="ms"
                      precision={1}
                      valueStyle={{ color: getMetricColor(metrics.risk_processing_seconds * 1000, 'latency') }}
                    />
                  </Col>
                </Row>
              </Card>
            </Col>

            <Col xs={24} lg={12}>
              <Card title="Database Performance" size="small">
                <Row gutter={16}>
                  <Col span={8}>
                    <Statistic
                      title="Queries/sec"
                      value={metrics.database_queries_total}
                      valueStyle={{ color: '#13c2c2' }}
                    />
                  </Col>
                  <Col span={8}>
                    <Statistic
                      title="Query Duration"
                      value={metrics.database_query_duration_seconds * 1000}
                      suffix="ms"
                      precision={1}
                      valueStyle={{ color: getMetricColor(metrics.database_query_duration_seconds * 1000, 'latency') }}
                    />
                  </Col>
                  <Col span={8}>
                    <Statistic
                      title="Active Connections"
                      value={metrics.database_connections_active}
                      valueStyle={{ color: '#722ed1' }}
                    />
                  </Col>
                </Row>
              </Card>
            </Col>
          </Row>

          {/* System Resources */}
          <Card title="System Resource Utilization" size="small" style={{ marginBottom: 24 }}>
            <Row gutter={[16, 16]}>
              <Col xs={24} md={8}>
                <div>
                  <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 4 }}>
                    <Text>CPU Usage</Text>
                    <Text strong style={{ color: getMetricColor(metrics.system_cpu_usage_ratio * 100, 'percentage') }}>
                      {(metrics.system_cpu_usage_ratio * 100).toFixed(1)}%
                    </Text>
                  </div>
                  <Progress 
                    percent={metrics.system_cpu_usage_ratio * 100} 
                    size="small"
                    status={metrics.system_cpu_usage_ratio > 0.8 ? 'exception' : 'success'}
                  />
                </div>
              </Col>

              <Col xs={24} md={8}>
                <div>
                  <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 4 }}>
                    <Text>Memory Usage</Text>
                    <Text strong style={{ color: getMetricColor(metrics.system_memory_usage_ratio * 100, 'percentage') }}>
                      {(metrics.system_memory_usage_ratio * 100).toFixed(1)}%
                    </Text>
                  </div>
                  <Progress 
                    percent={metrics.system_memory_usage_ratio * 100} 
                    size="small"
                    status={metrics.system_memory_usage_ratio > 0.85 ? 'exception' : 'success'}
                  />
                </div>
              </Col>

              <Col xs={24} md={8}>
                <div>
                  <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 4 }}>
                    <Text>Disk Usage</Text>
                    <Text strong style={{ color: getMetricColor(metrics.system_disk_usage_ratio * 100, 'percentage') }}>
                      {(metrics.system_disk_usage_ratio * 100).toFixed(1)}%
                    </Text>
                  </div>
                  <Progress 
                    percent={metrics.system_disk_usage_ratio * 100} 
                    size="small"
                    status={metrics.system_disk_usage_ratio > 0.9 ? 'exception' : 'success'}
                  />
                </div>
              </Col>
            </Row>
          </Card>
        </>
      )}

      {/* Custom Query Results */}
      {customQueries.length > 0 && (
        <Card title="Query Execution Results" size="small">
          <Table
            columns={queryColumns}
            dataSource={customQueries}
            size="small"
            pagination={false}
            rowKey="name"
          />
        </Card>
      )}
    </div>
  );
};

export default PrometheusMetricsDashboard;