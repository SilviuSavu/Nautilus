/**
 * Streaming Performance Tracker Component
 * Sprint 3: Advanced WebSocket Infrastructure
 * 
 * Advanced performance tracking for WebSocket streaming with real-time benchmarking,
 * SLA monitoring, performance regression detection, and optimization insights.
 */

import React, { useState, useEffect, useMemo, useCallback } from 'react';
import {
  Card,
  Row,
  Col,
  Typography,
  Space,
  Statistic,
  Progress,
  Table,
  Tag,
  Select,
  Switch,
  Button,
  Alert,
  Tooltip,
  Badge,
  Modal,
  InputNumber,
  Slider,
  Timeline,
  List,
  Avatar,
  Divider,
  notification,
  Tabs
} from 'antd';
import {
  DashboardOutlined,
  ThunderboltOutlined,
  BarChartOutlined,
  LineChartOutlined,
  RocketOutlined,
  WarningOutlined,
  CheckCircleOutlined,
  TrophyOutlined,
  ExperimentOutlined,
  ClockCircleOutlined,
  FireOutlined,
  BugOutlined,
  SettingOutlined,
  PlayCircleOutlined,
  StopOutlined,
  ReloadOutlined,
  ExportOutlined,
  MonitorOutlined
} from '@ant-design/icons';
import {
  LineChart,
  Line,
  AreaChart,
  Area,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip as RechartsTooltip,
  ResponsiveContainer,
  ReferenceLine,
  Legend,
  ScatterChart,
  Scatter,
  RadialBarChart,
  RadialBar,
  ComposedChart
} from 'recharts';

const { Title, Text } = Typography;
const { Option } = Select;
const { TabPane } = Tabs;

interface WebSocketEndpoint {
  id: string;
  name: string;
  url: string;
  status: 'connected' | 'disconnected' | 'connecting' | 'error';
  latency: number;
  messagesPerSecond: number;
  subscriptions: number;
  lastActivity: string;
  uptime: number;
  errorCount: number;
  quality: number;
}

interface PerformanceBenchmark {
  id: string;
  name: string;
  timestamp: number;
  duration: number; // seconds
  messageCount: number;
  dataVolume: number; // bytes
  averageLatency: number;
  minLatency: number;
  maxLatency: number;
  p95Latency: number;
  p99Latency: number;
  throughput: number; // messages per second
  errorRate: number;
  cpuUsage: number;
  memoryUsage: number;
  jitterScore: number;
  stabilityScore: number;
  overallScore: number;
  status: 'running' | 'completed' | 'failed';
}

interface PerformanceMetrics {
  currentThroughput: number;
  averageLatency: number;
  latencyPercentiles: {
    p50: number;
    p90: number;
    p95: number;
    p99: number;
  };
  errorRate: number;
  dataRate: number; // bytes per second
  connectionStability: number;
  resourceEfficiency: number;
  performanceIndex: number; // 0-100 composite score
}

interface SLAThresholds {
  maxLatency: number;
  maxErrorRate: number;
  minThroughput: number;
  minUptime: number;
  maxJitter: number;
}

interface StreamingPerformanceTrackerProps {
  endpoints: WebSocketEndpoint[];
  historicalData: any[];
  performanceThresholds: {
    latencyWarning: number;
    latencyError: number;
    throughputWarning: number;
    throughputError: number;
    errorRateWarning: number;
    errorRateError: number;
  };
  enableBenchmarking?: boolean;
  className?: string;
  slaThresholds?: SLAThresholds;
}

interface PerformanceAlert {
  id: string;
  timestamp: number;
  type: 'latency' | 'throughput' | 'error_rate' | 'stability' | 'sla_violation';
  severity: 'low' | 'medium' | 'high' | 'critical';
  message: string;
  endpoint?: string;
  value: number;
  threshold: number;
  resolved: boolean;
}

export const StreamingPerformanceTracker: React.FC<StreamingPerformanceTrackerProps> = ({
  endpoints,
  historicalData,
  performanceThresholds,
  enableBenchmarking = true,
  className,
  slaThresholds = {
    maxLatency: 100,
    maxErrorRate: 1,
    minThroughput: 50,
    minUptime: 99.9,
    maxJitter: 50
  }
}) => {
  const [activeTab, setActiveTab] = useState<string>('realtime');
  const [benchmarkResults, setBenchmarkResults] = useState<PerformanceBenchmark[]>([]);
  const [currentBenchmark, setCurrentBenchmark] = useState<PerformanceBenchmark | null>(null);
  const [isBenchmarkRunning, setIsBenchmarkRunning] = useState<boolean>(false);
  const [performanceAlerts, setPerformanceAlerts] = useState<PerformanceAlert[]>([]);
  const [performanceHistory, setPerformanceHistory] = useState<any[]>([]);
  const [showBenchmarkModal, setShowBenchmarkModal] = useState<boolean>(false);
  const [benchmarkConfig, setBenchmarkConfig] = useState({
    duration: 300, // 5 minutes
    targetThroughput: 1000,
    messageSize: 1024,
    endpoint: 'all',
    testType: 'sustained_load'
  });
  const [selectedTimeRange, setSelectedTimeRange] = useState<string>('1h');
  const [autoOptimization, setAutoOptimization] = useState<boolean>(false);

  // Calculate current performance metrics
  const currentPerformance: PerformanceMetrics = useMemo(() => {
    if (endpoints.length === 0) {
      return {
        currentThroughput: 0,
        averageLatency: 0,
        latencyPercentiles: { p50: 0, p90: 0, p95: 0, p99: 0 },
        errorRate: 0,
        dataRate: 0,
        connectionStability: 0,
        resourceEfficiency: 0,
        performanceIndex: 0
      };
    }

    const totalThroughput = endpoints.reduce((sum, ep) => sum + ep.messagesPerSecond, 0);
    const avgLatency = endpoints.reduce((sum, ep) => sum + ep.latency, 0) / endpoints.length;
    const totalErrors = endpoints.reduce((sum, ep) => sum + ep.errorCount, 0);
    const connectedCount = endpoints.filter(ep => ep.status === 'connected').length;
    
    // Simulate latency percentiles based on average
    const latencyPercentiles = {
      p50: avgLatency * 0.8,
      p90: avgLatency * 1.2,
      p95: avgLatency * 1.4,
      p99: avgLatency * 1.8
    };
    
    const errorRate = totalThroughput > 0 ? (totalErrors / (totalThroughput * 60)) * 100 : 0;
    const dataRate = totalThroughput * 1024; // Assume 1KB per message
    const connectionStability = (connectedCount / endpoints.length) * 100;
    const avgUptime = endpoints.reduce((sum, ep) => sum + ep.uptime, 0) / endpoints.length;
    const resourceEfficiency = Math.min(100, (totalThroughput / 1000) * 100); // Efficiency based on 1000 msg/s baseline
    
    // Calculate composite performance index (0-100)
    let performanceIndex = 100;
    
    // Latency impact (30%)
    if (avgLatency > slaThresholds.maxLatency * 2) performanceIndex -= 30;
    else if (avgLatency > slaThresholds.maxLatency) performanceIndex -= 15;
    
    // Throughput impact (25%)
    if (totalThroughput < slaThresholds.minThroughput * 0.5) performanceIndex -= 25;
    else if (totalThroughput < slaThresholds.minThroughput) performanceIndex -= 12;
    
    // Error rate impact (25%)
    if (errorRate > slaThresholds.maxErrorRate * 2) performanceIndex -= 25;
    else if (errorRate > slaThresholds.maxErrorRate) performanceIndex -= 12;
    
    // Stability impact (20%)
    if (connectionStability < 95) performanceIndex -= 20;
    else if (connectionStability < 98) performanceIndex -= 10;
    
    performanceIndex = Math.max(0, performanceIndex);
    
    return {
      currentThroughput: totalThroughput,
      averageLatency: avgLatency,
      latencyPercentiles,
      errorRate,
      dataRate,
      connectionStability,
      resourceEfficiency,
      performanceIndex
    };
  }, [endpoints, slaThresholds]);

  // Generate performance history
  useEffect(() => {
    const interval = setInterval(() => {
      if (endpoints.length > 0) {
        const dataPoint = {
          timestamp: Date.now(),
          time: new Date().toLocaleTimeString(),
          throughput: currentPerformance.currentThroughput,
          latency: currentPerformance.averageLatency,
          p95Latency: currentPerformance.latencyPercentiles.p95,
          p99Latency: currentPerformance.latencyPercentiles.p99,
          errorRate: currentPerformance.errorRate,
          stability: currentPerformance.connectionStability,
          performanceIndex: currentPerformance.performanceIndex,
          dataRate: currentPerformance.dataRate / 1024 // Convert to KB/s
        };
        
        setPerformanceHistory(prev => {
          const newHistory = [...prev, dataPoint];
          return newHistory.slice(-100); // Keep last 100 points
        });
      }
    }, 2000);
    
    return () => clearInterval(interval);
  }, [currentPerformance, endpoints.length]);

  // Monitor for performance alerts
  useEffect(() => {
    const checkPerformanceAlerts = () => {
      const newAlerts: PerformanceAlert[] = [];
      const now = Date.now();
      
      // Latency alerts
      if (currentPerformance.averageLatency > slaThresholds.maxLatency) {
        newAlerts.push({
          id: `latency-${now}`,
          timestamp: now,
          type: 'latency',
          severity: currentPerformance.averageLatency > slaThresholds.maxLatency * 2 ? 'critical' : 'high',
          message: `High latency detected: ${currentPerformance.averageLatency.toFixed(0)}ms`,
          value: currentPerformance.averageLatency,
          threshold: slaThresholds.maxLatency,
          resolved: false
        });
      }
      
      // Throughput alerts
      if (currentPerformance.currentThroughput < slaThresholds.minThroughput) {
        newAlerts.push({
          id: `throughput-${now}`,
          timestamp: now,
          type: 'throughput',
          severity: currentPerformance.currentThroughput < slaThresholds.minThroughput * 0.5 ? 'critical' : 'medium',
          message: `Low throughput: ${currentPerformance.currentThroughput} msg/s`,
          value: currentPerformance.currentThroughput,
          threshold: slaThresholds.minThroughput,
          resolved: false
        });
      }
      
      // Error rate alerts
      if (currentPerformance.errorRate > slaThresholds.maxErrorRate) {
        newAlerts.push({
          id: `error-rate-${now}`,
          timestamp: now,
          type: 'error_rate',
          severity: currentPerformance.errorRate > slaThresholds.maxErrorRate * 3 ? 'critical' : 'high',
          message: `High error rate: ${currentPerformance.errorRate.toFixed(2)}%`,
          value: currentPerformance.errorRate,
          threshold: slaThresholds.maxErrorRate,
          resolved: false
        });
      }
      
      // Stability alerts
      if (currentPerformance.connectionStability < slaThresholds.minUptime) {
        newAlerts.push({
          id: `stability-${now}`,
          timestamp: now,
          type: 'stability',
          severity: currentPerformance.connectionStability < 95 ? 'critical' : 'high',
          message: `Connection instability: ${currentPerformance.connectionStability.toFixed(1)}% uptime`,
          value: currentPerformance.connectionStability,
          threshold: slaThresholds.minUptime,
          resolved: false
        });
      }
      
      // Add new alerts
      if (newAlerts.length > 0) {
        setPerformanceAlerts(prev => [...prev, ...newAlerts].slice(-50));
        
        // Show notifications for critical alerts
        newAlerts.filter(alert => alert.severity === 'critical').forEach(alert => {
          notification.error({
            message: 'Critical Performance Alert',
            description: alert.message,
            placement: 'topRight',
            duration: 0
          });
        });
      }
    };
    
    const interval = setInterval(checkPerformanceAlerts, 5000);
    return () => clearInterval(interval);
  }, [currentPerformance, slaThresholds]);

  // Run performance benchmark
  const runBenchmark = useCallback(async () => {
    if (isBenchmarkRunning) return;
    
    const benchmarkId = `benchmark-${Date.now()}`;
    setIsBenchmarkRunning(true);
    
    const initialBenchmark: PerformanceBenchmark = {
      id: benchmarkId,
      name: `${benchmarkConfig.testType} - ${benchmarkConfig.targetThroughput} msg/s`,
      timestamp: Date.now(),
      duration: 0,
      messageCount: 0,
      dataVolume: 0,
      averageLatency: 0,
      minLatency: 0,
      maxLatency: 0,
      p95Latency: 0,
      p99Latency: 0,
      throughput: 0,
      errorRate: 0,
      cpuUsage: 0,
      memoryUsage: 0,
      jitterScore: 0,
      stabilityScore: 0,
      overallScore: 0,
      status: 'running'
    };
    
    setCurrentBenchmark(initialBenchmark);
    setBenchmarkResults(prev => [...prev, initialBenchmark]);
    
    // Simulate benchmark execution
    const totalSteps = Math.ceil(benchmarkConfig.duration / 10); // Update every 10 seconds
    let currentStep = 0;
    
    const benchmarkInterval = setInterval(() => {
      currentStep++;
      const progress = currentStep / totalSteps;
      
      // Simulate realistic benchmark progression
      const targetThroughput = benchmarkConfig.targetThroughput;
      const actualThroughput = Math.min(
        targetThroughput,
        targetThroughput * Math.min(1, progress * 1.2) + Math.random() * targetThroughput * 0.1
      );
      
      // Simulate performance characteristics
      const baseLatency = 25;
      const loadLatency = (actualThroughput / targetThroughput) * 50;
      const avgLatency = baseLatency + loadLatency + Math.random() * 20;
      const minLatency = avgLatency * 0.3;
      const maxLatency = avgLatency * 2.5;
      const p95Latency = avgLatency * 1.4;
      const p99Latency = avgLatency * 1.8;
      
      const errorRate = Math.max(0, (actualThroughput / targetThroughput - 0.8) * 5 + Math.random() * 2);
      const jitterScore = Math.random() * 30 + 10;
      const stabilityScore = Math.max(70, 100 - (errorRate * 5) - (jitterScore * 0.5));
      
      const messageCount = Math.floor(actualThroughput * currentStep * 10);
      const dataVolume = messageCount * benchmarkConfig.messageSize;
      
      // Calculate overall score
      let overallScore = 100;
      if (avgLatency > 100) overallScore -= 20;
      if (actualThroughput < targetThroughput * 0.9) overallScore -= 15;
      if (errorRate > 1) overallScore -= 25;
      if (jitterScore > 40) overallScore -= 10;
      if (stabilityScore < 90) overallScore -= 15;
      overallScore = Math.max(0, overallScore);
      
      const updatedBenchmark: PerformanceBenchmark = {
        ...initialBenchmark,
        duration: currentStep * 10,
        messageCount,
        dataVolume,
        averageLatency: avgLatency,
        minLatency,
        maxLatency,
        p95Latency,
        p99Latency,
        throughput: actualThroughput,
        errorRate,
        cpuUsage: Math.min(90, 20 + (actualThroughput / targetThroughput) * 40 + Math.random() * 20),
        memoryUsage: Math.min(90, 30 + (messageCount / 100000) * 20 + Math.random() * 15),
        jitterScore,
        stabilityScore,
        overallScore,
        status: currentStep >= totalSteps ? 'completed' : 'running'
      };
      
      setCurrentBenchmark(updatedBenchmark);
      setBenchmarkResults(prev =>
        prev.map(bench => bench.id === benchmarkId ? updatedBenchmark : bench)
      );
      
      if (currentStep >= totalSteps) {
        clearInterval(benchmarkInterval);
        setIsBenchmarkRunning(false);
        setCurrentBenchmark(null);
        
        notification.success({
          message: 'Performance Benchmark Completed',
          description: `Overall Score: ${overallScore.toFixed(0)}/100`,
          placement: 'topRight'
        });
      }
    }, 10000);
    
    return () => {
      clearInterval(benchmarkInterval);
      setIsBenchmarkRunning(false);
    };
  }, [benchmarkConfig, isBenchmarkRunning]);

  // Get performance color
  const getPerformanceColor = (value: number, type: 'score' | 'latency' | 'error'): string => {
    if (type === 'score') {
      if (value >= 90) return '#52c41a';
      if (value >= 75) return '#faad14';
      if (value >= 60) return '#fa8c16';
      return '#ff4d4f';
    } else if (type === 'latency') {
      if (value <= slaThresholds.maxLatency * 0.5) return '#52c41a';
      if (value <= slaThresholds.maxLatency) return '#faad14';
      if (value <= slaThresholds.maxLatency * 2) return '#fa8c16';
      return '#ff4d4f';
    } else { // error
      if (value <= slaThresholds.maxErrorRate * 0.5) return '#52c41a';
      if (value <= slaThresholds.maxErrorRate) return '#faad14';
      if (value <= slaThresholds.maxErrorRate * 2) return '#fa8c16';
      return '#ff4d4f';
    }
  };

  // Benchmark results table columns
  const benchmarkColumns = [
    {
      title: 'Benchmark',
      dataIndex: 'name',
      key: 'name',
      render: (name: string, record: PerformanceBenchmark) => (
        <Space direction="vertical" size="small">
          <Text strong>{name}</Text>
          <Text type="secondary" style={{ fontSize: '11px' }}>
            {new Date(record.timestamp).toLocaleString()}
          </Text>
        </Space>
      )
    },
    {
      title: 'Score',
      dataIndex: 'overallScore',
      key: 'score',
      render: (score: number) => (
        <Space>
          <Text style={{ color: getPerformanceColor(score, 'score') }}>
            {score.toFixed(0)}/100
          </Text>
          <Progress
            percent={score}
            size="small"
            strokeColor={getPerformanceColor(score, 'score')}
            showInfo={false}
          />
        </Space>
      ),
      sorter: (a: PerformanceBenchmark, b: PerformanceBenchmark) => a.overallScore - b.overallScore
    },
    {
      title: 'Throughput',
      dataIndex: 'throughput',
      key: 'throughput',
      render: (throughput: number) => `${throughput.toFixed(0)} msg/s`,
      sorter: (a: PerformanceBenchmark, b: PerformanceBenchmark) => a.throughput - b.throughput
    },
    {
      title: 'Latency',
      key: 'latency',
      render: (record: PerformanceBenchmark) => (
        <Space direction="vertical" size="small">
          <Text>Avg: {record.averageLatency.toFixed(0)}ms</Text>
          <Text type="secondary" style={{ fontSize: '11px' }}>
            P95: {record.p95Latency.toFixed(0)}ms
          </Text>
        </Space>
      )
    },
    {
      title: 'Error Rate',
      dataIndex: 'errorRate',
      key: 'errorRate',
      render: (rate: number) => (
        <Text style={{ color: getPerformanceColor(rate, 'error') }}>
          {rate.toFixed(2)}%
        </Text>
      )
    },
    {
      title: 'Duration',
      dataIndex: 'duration',
      key: 'duration',
      render: (duration: number) => `${Math.floor(duration / 60)}:${(duration % 60).toString().padStart(2, '0')}`
    },
    {
      title: 'Status',
      dataIndex: 'status',
      key: 'status',
      render: (status: PerformanceBenchmark['status']) => {
        const colors = {
          running: 'processing',
          completed: 'success',
          failed: 'error'
        };
        return <Badge status={colors[status] as any} text={status.charAt(0).toUpperCase() + status.slice(1)} />;
      }
    }
  ];

  // Chart data
  const chartData = performanceHistory.slice(-50);

  return (
    <div className={className}>
      <Space direction="vertical" size="large" style={{ width: '100%' }}>
        {/* Performance Overview */}
        <Row gutter={[16, 16]}>
          <Col span={6}>
            <Card size="small">
              <Statistic
                title="Performance Index"
                value={currentPerformance.performanceIndex}
                precision={0}
                suffix="/100"
                valueStyle={{ color: getPerformanceColor(currentPerformance.performanceIndex, 'score') }}
                prefix={<TrophyOutlined />}
              />
            </Card>
          </Col>
          
          <Col span={6}>
            <Card size="small">
              <Statistic
                title="Current Throughput"
                value={currentPerformance.currentThroughput}
                suffix="msg/s"
                valueStyle={{ color: '#1890ff' }}
                prefix={<RocketOutlined />}
              />
            </Card>
          </Col>
          
          <Col span={6}>
            <Card size="small">
              <Statistic
                title="P95 Latency"
                value={currentPerformance.latencyPercentiles.p95}
                precision={0}
                suffix="ms"
                valueStyle={{ color: getPerformanceColor(currentPerformance.latencyPercentiles.p95, 'latency') }}
                prefix={<ThunderboltOutlined />}
              />
            </Card>
          </Col>
          
          <Col span={6}>
            <Card size="small">
              <Statistic
                title="Connection Stability"
                value={currentPerformance.connectionStability}
                precision={1}
                suffix="%"
                valueStyle={{ 
                  color: currentPerformance.connectionStability > 99 ? '#52c41a' : '#ff4d4f' 
                }}
                prefix={<CheckCircleOutlined />}
              />
            </Card>
          </Col>
        </Row>

        {/* SLA Compliance */}
        <Card title="SLA Compliance" size="small">
          <Row gutter={[16, 16]}>
            <Col span={6}>
              <Space direction="vertical" style={{ width: '100%' }}>
                <Text>Latency SLA</Text>
                <Progress
                  percent={Math.min(100, (slaThresholds.maxLatency / Math.max(currentPerformance.averageLatency, 1)) * 100)}
                  strokeColor={currentPerformance.averageLatency <= slaThresholds.maxLatency ? '#52c41a' : '#ff4d4f'}
                  format={() => currentPerformance.averageLatency <= slaThresholds.maxLatency ? 'Compliant' : 'Violation'}
                />
              </Space>
            </Col>
            
            <Col span={6}>
              <Space direction="vertical" style={{ width: '100%' }}>
                <Text>Throughput SLA</Text>
                <Progress
                  percent={Math.min(100, (currentPerformance.currentThroughput / slaThresholds.minThroughput) * 100)}
                  strokeColor={currentPerformance.currentThroughput >= slaThresholds.minThroughput ? '#52c41a' : '#ff4d4f'}
                  format={() => currentPerformance.currentThroughput >= slaThresholds.minThroughput ? 'Compliant' : 'Violation'}
                />
              </Space>
            </Col>
            
            <Col span={6}>
              <Space direction="vertical" style={{ width: '100%' }}>
                <Text>Error Rate SLA</Text>
                <Progress
                  percent={Math.min(100, Math.max(0, 100 - (currentPerformance.errorRate / slaThresholds.maxErrorRate) * 100))}
                  strokeColor={currentPerformance.errorRate <= slaThresholds.maxErrorRate ? '#52c41a' : '#ff4d4f'}
                  format={() => currentPerformance.errorRate <= slaThresholds.maxErrorRate ? 'Compliant' : 'Violation'}
                />
              </Space>
            </Col>
            
            <Col span={6}>
              <Space direction="vertical" style={{ width: '100%' }}>
                <Text>Uptime SLA</Text>
                <Progress
                  percent={currentPerformance.connectionStability}
                  strokeColor={currentPerformance.connectionStability >= slaThresholds.minUptime ? '#52c41a' : '#ff4d4f'}
                  format={() => currentPerformance.connectionStability >= slaThresholds.minUptime ? 'Compliant' : 'Violation'}
                />
              </Space>
            </Col>
          </Row>
        </Card>

        {/* Performance Alerts */}
        {performanceAlerts.filter(alert => !alert.resolved).length > 0 && (
          <Alert
            message={`${performanceAlerts.filter(alert => !alert.resolved).length} Active Performance Alert${performanceAlerts.filter(alert => !alert.resolved).length > 1 ? 's' : ''}`}
            description={
              <List
                size="small"
                dataSource={performanceAlerts.filter(alert => !alert.resolved).slice(-3)}
                renderItem={(alert, index) => (
                  <List.Item key={index}>
                    <Space>
                      <Tag color={alert.severity === 'critical' ? 'red' : alert.severity === 'high' ? 'orange' : 'yellow'}>
                        {alert.severity.toUpperCase()}
                      </Tag>
                      <Text>{alert.message}</Text>
                      <Text type="secondary">
                        {new Date(alert.timestamp).toLocaleTimeString()}
                      </Text>
                    </Space>
                  </List.Item>
                )}
              />
            }
            type="warning"
            showIcon
          />
        )}

        {/* Main Performance Tabs */}
        <Card
          title={
            <Space>
              <DashboardOutlined />
              <Title level={4} style={{ margin: 0 }}>Performance Tracking</Title>
            </Space>
          }
          extra={
            <Space>
              <Switch
                checkedChildren="Auto-Optimize"
                unCheckedChildren="Manual"
                checked={autoOptimization}
                onChange={setAutoOptimization}
                size="small"
              />
              <Select
                value={selectedTimeRange}
                onChange={setSelectedTimeRange}
                size="small"
                style={{ width: 80 }}
              >
                <Option value="1h">1h</Option>
                <Option value="6h">6h</Option>
                <Option value="24h">24h</Option>
                <Option value="7d">7d</Option>
              </Select>
              {enableBenchmarking && (
                <Button
                  type="primary"
                  icon={isBenchmarkRunning ? <StopOutlined /> : <ExperimentOutlined />}
                  onClick={() => setShowBenchmarkModal(true)}
                  disabled={isBenchmarkRunning}
                >
                  {isBenchmarkRunning ? 'Running...' : 'Benchmark'}
                </Button>
              )}
            </Space>
          }
        >
          <Tabs activeKey={activeTab} onChange={setActiveTab}>
            <TabPane tab="Real-time Performance" key="realtime">
              <ResponsiveContainer width="100%" height={400}>
                <ComposedChart data={chartData}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="time" />
                  <YAxis yAxisId="left" />
                  <YAxis yAxisId="right" orientation="right" />
                  <RechartsTooltip />
                  <Legend />
                  
                  <Area
                    yAxisId="left"
                    type="monotone"
                    dataKey="performanceIndex"
                    stroke="#52c41a"
                    fill="#52c41a"
                    fillOpacity={0.3}
                    name="Performance Index"
                  />
                  
                  <Line
                    yAxisId="right"
                    type="monotone"
                    dataKey="latency"
                    stroke="#ff4d4f"
                    name="Avg Latency (ms)"
                    strokeWidth={2}
                  />
                  
                  <Line
                    yAxisId="right"
                    type="monotone"
                    dataKey="p95Latency"
                    stroke="#faad14"
                    name="P95 Latency (ms)"
                    strokeWidth={2}
                    strokeDasharray="5 5"
                  />
                  
                  <Bar
                    yAxisId="left"
                    dataKey="throughput"
                    fill="#1890ff"
                    name="Throughput (msg/s)"
                    opacity={0.7}
                  />
                  
                  <ReferenceLine 
                    yAxisId="right"
                    y={slaThresholds.maxLatency} 
                    stroke="#ff4d4f" 
                    strokeDasharray="3 3" 
                    label="Latency SLA"
                  />
                </ComposedChart>
              </ResponsiveContainer>
            </TabPane>

            <TabPane tab="Latency Distribution" key="latency">
              <Row gutter={[16, 16]}>
                <Col span={12}>
                  <Card title="Latency Percentiles" size="small">
                    <Space direction="vertical" style={{ width: '100%' }}>
                      <Row justify="space-between">
                        <Text>P50:</Text>
                        <Text strong>{currentPerformance.latencyPercentiles.p50.toFixed(0)}ms</Text>
                      </Row>
                      <Progress
                        percent={(currentPerformance.latencyPercentiles.p50 / slaThresholds.maxLatency) * 100}
                        strokeColor={getPerformanceColor(currentPerformance.latencyPercentiles.p50, 'latency')}
                        showInfo={false}
                        size="small"
                      />
                      
                      <Row justify="space-between">
                        <Text>P90:</Text>
                        <Text strong>{currentPerformance.latencyPercentiles.p90.toFixed(0)}ms</Text>
                      </Row>
                      <Progress
                        percent={(currentPerformance.latencyPercentiles.p90 / slaThresholds.maxLatency) * 100}
                        strokeColor={getPerformanceColor(currentPerformance.latencyPercentiles.p90, 'latency')}
                        showInfo={false}
                        size="small"
                      />
                      
                      <Row justify="space-between">
                        <Text>P95:</Text>
                        <Text strong>{currentPerformance.latencyPercentiles.p95.toFixed(0)}ms</Text>
                      </Row>
                      <Progress
                        percent={(currentPerformance.latencyPercentiles.p95 / slaThresholds.maxLatency) * 100}
                        strokeColor={getPerformanceColor(currentPerformance.latencyPercentiles.p95, 'latency')}
                        showInfo={false}
                        size="small"
                      />
                      
                      <Row justify="space-between">
                        <Text>P99:</Text>
                        <Text strong>{currentPerformance.latencyPercentiles.p99.toFixed(0)}ms</Text>
                      </Row>
                      <Progress
                        percent={(currentPerformance.latencyPercentiles.p99 / slaThresholds.maxLatency) * 100}
                        strokeColor={getPerformanceColor(currentPerformance.latencyPercentiles.p99, 'latency')}
                        showInfo={false}
                        size="small"
                      />
                    </Space>
                  </Card>
                </Col>
                
                <Col span={12}>
                  <Card title="Performance Radar" size="small">
                    <ResponsiveContainer width="100%" height={300}>
                      <RadialBarChart 
                        cx="50%" 
                        cy="50%" 
                        innerRadius="20%" 
                        outerRadius="80%" 
                        data={[
                          {
                            name: 'Throughput',
                            value: (currentPerformance.currentThroughput / slaThresholds.minThroughput) * 100,
                            fill: '#1890ff'
                          },
                          {
                            name: 'Latency',
                            value: Math.max(0, 100 - (currentPerformance.averageLatency / slaThresholds.maxLatency) * 100),
                            fill: '#ff4d4f'
                          },
                          {
                            name: 'Stability',
                            value: currentPerformance.connectionStability,
                            fill: '#52c41a'
                          },
                          {
                            name: 'Efficiency',
                            value: currentPerformance.resourceEfficiency,
                            fill: '#faad14'
                          }
                        ]}
                      >
                        <RadialBar dataKey="value" cornerRadius={10} fill="#8884d8" />
                        <RechartsTooltip />
                      </RadialBarChart>
                    </ResponsiveContainer>
                  </Card>
                </Col>
              </Row>
            </TabPane>

            <TabPane tab="Benchmark Results" key="benchmarks">
              {benchmarkResults.length > 0 ? (
                <Table
                  dataSource={benchmarkResults.slice().reverse()}
                  columns={benchmarkColumns}
                  rowKey="id"
                  size="small"
                  pagination={{ pageSize: 10 }}
                />
              ) : (
                <Card>
                  <Text type="secondary">No benchmark results available. Run a performance benchmark to see results.</Text>
                </Card>
              )}
            </TabPane>
          </Tabs>
        </Card>

        {/* Current Benchmark Status */}
        {currentBenchmark && (
          <Card title="Current Benchmark" size="small">
            <Row gutter={[16, 16]}>
              <Col span={6}>
                <Statistic
                  title="Progress"
                  value={(currentBenchmark.duration / benchmarkConfig.duration) * 100}
                  precision={1}
                  suffix="%"
                />
              </Col>
              
              <Col span={6}>
                <Statistic
                  title="Throughput"
                  value={currentBenchmark.throughput}
                  suffix="msg/s"
                />
              </Col>
              
              <Col span={6}>
                <Statistic
                  title="Avg Latency"
                  value={currentBenchmark.averageLatency}
                  precision={0}
                  suffix="ms"
                />
              </Col>
              
              <Col span={6}>
                <Statistic
                  title="Score"
                  value={currentBenchmark.overallScore}
                  precision={0}
                  suffix="/100"
                  valueStyle={{ color: getPerformanceColor(currentBenchmark.overallScore, 'score') }}
                />
              </Col>
            </Row>
            
            <Progress
              percent={(currentBenchmark.duration / benchmarkConfig.duration) * 100}
              status="active"
              style={{ marginTop: 16 }}
            />
          </Card>
        )}

        {/* Benchmark Configuration Modal */}
        <Modal
          title="Performance Benchmark Configuration"
          open={showBenchmarkModal}
          onCancel={() => setShowBenchmarkModal(false)}
          onOk={() => {
            setShowBenchmarkModal(false);
            runBenchmark();
          }}
          okText="Start Benchmark"
          okButtonProps={{ disabled: isBenchmarkRunning }}
        >
          <Space direction="vertical" style={{ width: '100%' }}>
            <div>
              <Text>Test Duration (seconds):</Text>
              <Slider
                min={60}
                max={1800}
                value={benchmarkConfig.duration}
                onChange={(value) => setBenchmarkConfig(prev => ({ ...prev, duration: value }))}
                marks={{ 60: '1m', 300: '5m', 900: '15m', 1800: '30m' }}
              />
              <Text type="secondary">{Math.floor(benchmarkConfig.duration / 60)} minutes</Text>
            </div>
            
            <div>
              <Text>Target Throughput (msg/s):</Text>
              <InputNumber
                min={10}
                max={10000}
                value={benchmarkConfig.targetThroughput}
                onChange={(value) => setBenchmarkConfig(prev => ({ ...prev, targetThroughput: value || 1000 }))}
                style={{ width: '100%' }}
              />
            </div>
            
            <div>
              <Text>Test Type:</Text>
              <Select
                value={benchmarkConfig.testType}
                onChange={(value) => setBenchmarkConfig(prev => ({ ...prev, testType: value }))}
                style={{ width: '100%' }}
              >
                <Option value="sustained_load">Sustained Load</Option>
                <Option value="burst_test">Burst Test</Option>
                <Option value="stress_test">Stress Test</Option>
                <Option value="endurance_test">Endurance Test</Option>
              </Select>
            </div>
          </Space>
        </Modal>
      </Space>
    </div>
  );
};

export default StreamingPerformanceTracker;