/**
 * M4 Max Hardware Monitoring Dashboard
 * Monitors Neural Engine, Metal GPU, CPU cores, and Unified Memory
 * Based on FRONTEND_ENDPOINT_INTEGRATION_GUIDE.md
 */

import React, { useState, useEffect } from 'react';
import {
  Card,
  Row,
  Col,
  Statistic,
  Progress,
  Alert,
  Button,
  Select,
  Typography,
  Space,
  Tag,
  Tooltip,
  Spin,
  Tabs
} from 'antd';
import {
  ThunderboltOutlined,
  ControlOutlined,
  DatabaseOutlined,
  DashboardOutlined,
  MonitorOutlined,
  RocketOutlined,
  SyncOutlined,
  FireOutlined,
  CloudOutlined
} from '@ant-design/icons';
import { 
  LineChart, 
  Line, 
  XAxis, 
  YAxis, 
  CartesianGrid, 
  Tooltip as RechartsTooltip, 
  Legend, 
  ResponsiveContainer, 
  AreaChart, 
  Area, 
  BarChart, 
  Bar, 
  RadialBarChart, 
  RadialBar,
  PieChart,
  Pie,
  Cell
} from 'recharts';
import apiClient from '../../services/apiClient';

const { Title, Text } = Typography;
const { Option } = Select;
const { TabPane } = Tabs;

// M4 Max Hardware interfaces
interface M4MaxHardwareMetrics {
  cpu: {
    performance_cores: { count: number; utilization: number };
    efficiency_cores: { count: number; utilization: number };
  };
  gpu: {
    cores: number;
    utilization: number;
    memory_bandwidth_gbps: number;
    thermal_state: string;
  };
  neural_engine: {
    cores: number;
    tops_performance: number;
    utilization: number;
    active_models: string[];
  };
  unified_memory: {
    total_gb: number;
    used_gb: number;
    bandwidth_gbps: number;
  };
}

interface CPUOptimizationHealth {
  optimization_active: boolean;
  core_utilization: Record<string, number>;
  workload_classification: string;
  performance_mode: string;
}

interface ContainerMetrics {
  containers: Array<{
    name: string;
    cpu_percent: number;
    memory_usage_mb: number;
    status: string;
  }>;
  total_containers: number;
  average_cpu_usage: number;
  total_memory_usage_gb: number;
}

interface TradingMetrics {
  order_execution_latency_ms: number;
  throughput_orders_per_second: number;
  hardware_acceleration_impact: {
    speedup_factor: number;
    gpu_operations: number;
  };
}

const COLORS = ['#0088FE', '#00C49F', '#FFBB28', '#FF8042', '#8884d8', '#82ca9d'];
const THERMAL_COLORS = {
  normal: '#52c41a',
  warm: '#faad14',
  hot: '#f5222d'
};

const M4MaxMonitoringDashboard: React.FC = () => {
  // State management
  const [loading, setLoading] = useState(false);
  const [hardwareMetrics, setHardwareMetrics] = useState<M4MaxHardwareMetrics | null>(null);
  const [cpuHealth, setCpuHealth] = useState<CPUOptimizationHealth | null>(null);
  const [containerMetrics, setContainerMetrics] = useState<ContainerMetrics | null>(null);
  const [tradingMetrics, setTradingMetrics] = useState<TradingMetrics | null>(null);
  const [metricsHistory, setMetricsHistory] = useState<M4MaxHardwareMetrics[]>([]);
  const [refreshInterval, setRefreshInterval] = useState<NodeJS.Timeout | null>(null);
  const [autoRefresh, setAutoRefresh] = useState(true);
  const [selectedTimeframe, setSelectedTimeframe] = useState(24);
  const [error, setError] = useState<string | null>(null);

  // Load initial data and start auto-refresh
  useEffect(() => {
    loadHardwareData();
    
    if (autoRefresh) {
      const interval = setInterval(loadHardwareData, 5000); // 5-second updates
      setRefreshInterval(interval);
    }
    
    return () => {
      if (refreshInterval) clearInterval(refreshInterval);
    };
  }, [autoRefresh]);

  const loadHardwareData = async () => {
    if (!loading) setLoading(true);
    setError(null);
    
    try {
      const [hardware, cpu, containers, trading] = await Promise.all([
        apiClient.getM4MaxHardwareMetrics(),
        apiClient.getCPUOptimizationHealth(),
        apiClient.getContainerMetrics(),
        apiClient.getTradingMetrics()
      ]);

      setHardwareMetrics(hardware);
      setCpuHealth(cpu);
      setContainerMetrics(containers);
      setTradingMetrics(trading);

      // Add to history for trending
      setMetricsHistory(prev => [
        ...prev.slice(-119), // Keep last 120 points (10 minutes at 5-second intervals)
        { ...hardware, timestamp: Date.now() }
      ] as M4MaxHardwareMetrics[]);

    } catch (err) {
      setError(`Failed to load hardware metrics: ${(err as Error).message}`);
    } finally {
      setLoading(false);
    }
  };

  const loadHistoricalData = async () => {
    setLoading(true);
    try {
      const history = await apiClient.getM4MaxHardwareHistory(selectedTimeframe);
      setMetricsHistory(history);
    } catch (err) {
      setError(`Failed to load historical data: ${(err as Error).message}`);
    } finally {
      setLoading(false);
    }
  };

  const toggleAutoRefresh = () => {
    setAutoRefresh(!autoRefresh);
    if (refreshInterval) {
      clearInterval(refreshInterval);
      setRefreshInterval(null);
    }
  };

  const getThermalStateColor = (state: string) => {
    switch (state.toLowerCase()) {
      case 'normal': return THERMAL_COLORS.normal;
      case 'warm': return THERMAL_COLORS.warm;
      case 'hot': return THERMAL_COLORS.hot;
      default: return '#d9d9d9';
    }
  };

  const getUtilizationColor = (utilization: number) => {
    if (utilization >= 90) return '#f5222d';
    if (utilization >= 70) return '#faad14';
    if (utilization >= 50) return '#52c41a';
    return '#1890ff';
  };

  const formatTimestamp = (timestamp: number) => {
    return new Date(timestamp).toLocaleTimeString();
  };

  if (loading && !hardwareMetrics) {
    return (
      <div style={{ display: 'flex', justifyContent: 'center', alignItems: 'center', height: '400px' }}>
        <Spin size="large" tip="Loading M4 Max Hardware Metrics..." />
      </div>
    );
  }

  return (
    <div style={{ padding: '24px' }}>
      <Title level={2}>
        <RocketOutlined style={{ marginRight: '8px', color: '#1890ff' }} />
        M4 Max Hardware Acceleration Monitor
      </Title>

      {error && (
        <Alert
          message="Error"
          description={error}
          type="error"
          closable
          onClose={() => setError(null)}
          style={{ marginBottom: '16px' }}
        />
      )}

      {/* Control Panel */}
      <Row gutter={[16, 16]} style={{ marginBottom: '24px' }}>
        <Col span={24}>
          <Card>
            <Space>
              <Button 
                icon={<SyncOutlined />} 
                onClick={loadHardwareData} 
                loading={loading}
              >
                Refresh Now
              </Button>
              <Button 
                type={autoRefresh ? 'primary' : 'default'}
                onClick={toggleAutoRefresh}
              >
                Auto Refresh {autoRefresh ? 'ON' : 'OFF'}
              </Button>
              <Select value={selectedTimeframe} onChange={setSelectedTimeframe} style={{ width: 150 }}>
                <Option value={1}>Last Hour</Option>
                <Option value={6}>Last 6 Hours</Option>
                <Option value={24}>Last 24 Hours</Option>
                <Option value={168}>Last Week</Option>
              </Select>
              <Button onClick={loadHistoricalData}>Load Historical</Button>
              <Text type="secondary">Last Updated: {new Date().toLocaleTimeString()}</Text>
            </Space>
          </Card>
        </Col>
      </Row>

      {/* Hardware Overview Cards */}
      <Row gutter={[16, 16]} style={{ marginBottom: '24px' }}>
        <Col xs={24} sm={12} md={6}>
          <Card>
            <Statistic
              title="Neural Engine"
              value={`${hardwareMetrics?.neural_engine?.utilization || 0}%`}
              prefix={<ThunderboltOutlined />}
              valueStyle={{ color: getUtilizationColor(hardwareMetrics?.neural_engine?.utilization || 0) }}
            />
            <Progress 
              percent={hardwareMetrics?.neural_engine?.utilization || 0}
              strokeColor={getUtilizationColor(hardwareMetrics?.neural_engine?.utilization || 0)}
              size="small"
            />
            <div style={{ marginTop: '8px' }}>
              <Tag color="blue">{hardwareMetrics?.neural_engine?.cores || 16} cores</Tag>
              <Tag color="green">{hardwareMetrics?.neural_engine?.tops_performance || 38} TOPS</Tag>
            </div>
          </Card>
        </Col>
        
        <Col xs={24} sm={12} md={6}>
          <Card>
            <Statistic
              title="Metal GPU"
              value={`${hardwareMetrics?.gpu?.utilization || 0}%`}
              prefix={<ControlOutlined />}
              valueStyle={{ color: getUtilizationColor(hardwareMetrics?.gpu?.utilization || 0) }}
            />
            <Progress 
              percent={hardwareMetrics?.gpu?.utilization || 0}
              strokeColor={getUtilizationColor(hardwareMetrics?.gpu?.utilization || 0)}
              size="small"
            />
            <div style={{ marginTop: '8px' }}>
              <Tag color="purple">{hardwareMetrics?.gpu?.cores || 40} cores</Tag>
              <Tag color={getThermalStateColor(hardwareMetrics?.gpu?.thermal_state || 'normal')}>
                {hardwareMetrics?.gpu?.thermal_state || 'normal'}
              </Tag>
            </div>
          </Card>
        </Col>
        
        <Col xs={24} sm={12} md={6}>
          <Card>
            <Statistic
              title="CPU (P-cores)"
              value={`${hardwareMetrics?.cpu?.performance_cores?.utilization || 0}%`}
              prefix={<MonitorOutlined />}
              valueStyle={{ color: getUtilizationColor(hardwareMetrics?.cpu?.performance_cores?.utilization || 0) }}
            />
            <Progress 
              percent={hardwareMetrics?.cpu?.performance_cores?.utilization || 0}
              strokeColor={getUtilizationColor(hardwareMetrics?.cpu?.performance_cores?.utilization || 0)}
              size="small"
            />
            <div style={{ marginTop: '8px' }}>
              <Tag color="cyan">12 P-cores</Tag>
              <Tag color="geekblue">4 E-cores</Tag>
            </div>
          </Card>
        </Col>
        
        <Col xs={24} sm={12} md={6}>
          <Card>
            <Statistic
              title="Unified Memory"
              value={`${((hardwareMetrics?.unified_memory?.used_gb || 0) / (hardwareMetrics?.unified_memory?.total_gb || 1) * 100).toFixed(1)}%`}
              prefix={<DatabaseOutlined />}
              valueStyle={{ 
                color: getUtilizationColor(
                  (hardwareMetrics?.unified_memory?.used_gb || 0) / (hardwareMetrics?.unified_memory?.total_gb || 1) * 100
                ) 
              }}
            />
            <div style={{ marginTop: '8px' }}>
              <Text type="secondary">
                {hardwareMetrics?.unified_memory?.used_gb?.toFixed(1) || 0}GB / {hardwareMetrics?.unified_memory?.total_gb || 0}GB
              </Text>
              <br />
              <Tag color="gold">{hardwareMetrics?.unified_memory?.bandwidth_gbps || 0}GB/s</Tag>
            </div>
          </Card>
        </Col>
      </Row>

      {/* Main Content Tabs */}
      <Tabs defaultActiveKey="1">
        <TabPane tab={<span><DashboardOutlined />Real-time Metrics</span>} key="1">
          <Row gutter={[16, 16]}>
            <Col xs={24} lg={16}>
              <Card title="Hardware Utilization Trends">
                <ResponsiveContainer width="100%" height={350}>
                  <LineChart data={metricsHistory.map((metric, index) => ({
                    ...metric,
                    timestamp: formatTimestamp(metric.timestamp || Date.now() - (119 - index) * 5000),
                    neuralEngine: metric.neural_engine?.utilization || 0,
                    metalGpu: metric.gpu?.utilization || 0,
                    pCores: metric.cpu?.performance_cores?.utilization || 0,
                    eCores: metric.cpu?.efficiency_cores?.utilization || 0,
                    memory: (metric.unified_memory?.used_gb || 0) / (metric.unified_memory?.total_gb || 1) * 100
                  }))}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="timestamp" />
                    <YAxis domain={[0, 100]} />
                    <RechartsTooltip />
                    <Legend />
                    <Line type="monotone" dataKey="neuralEngine" stroke="#8884d8" strokeWidth={2} name="Neural Engine" />
                    <Line type="monotone" dataKey="metalGpu" stroke="#82ca9d" strokeWidth={2} name="Metal GPU" />
                    <Line type="monotone" dataKey="pCores" stroke="#ffc658" strokeWidth={2} name="P-Cores" />
                    <Line type="monotone" dataKey="memory" stroke="#ff7c7c" strokeWidth={2} name="Memory" />
                  </LineChart>
                </ResponsiveContainer>
              </Card>
            </Col>
            
            <Col xs={24} lg={8}>
              <Card title="Neural Engine Active Models" style={{ marginBottom: '16px' }}>
                {hardwareMetrics?.neural_engine?.active_models?.length ? (
                  <div>
                    {hardwareMetrics.neural_engine.active_models.map((model, index) => (
                      <Tag key={index} color="processing" style={{ marginBottom: '4px' }}>
                        {model}
                      </Tag>
                    ))}
                  </div>
                ) : (
                  <Text type="secondary">No active models</Text>
                )}
              </Card>

              <Card title="Hardware Capabilities">
                <div style={{ textAlign: 'center' }}>
                  <ResponsiveContainer width="100%" height={200}>
                    <PieChart>
                      <Pie
                        data={[
                          { name: 'Neural Engine', value: hardwareMetrics?.neural_engine?.utilization || 0 },
                          { name: 'Metal GPU', value: hardwareMetrics?.gpu?.utilization || 0 },
                          { name: 'CPU', value: hardwareMetrics?.cpu?.performance_cores?.utilization || 0 }
                        ]}
                        cx="50%"
                        cy="50%"
                        innerRadius={40}
                        outerRadius={80}
                        paddingAngle={5}
                        dataKey="value"
                      >
                        {COLORS.map((color, index) => (
                          <Cell key={`cell-${index}`} fill={color} />
                        ))}
                      </Pie>
                      <RechartsTooltip />
                      <Legend />
                    </PieChart>
                  </ResponsiveContainer>
                </div>
              </Card>
            </Col>
          </Row>
        </TabPane>

        <TabPane tab={<span><CloudOutlined />Container Performance</span>} key="2">
          <Row gutter={[16, 16]}>
            <Col xs={24} lg={12}>
              <Card title="Container Resource Usage">
                <ResponsiveContainer width="100%" height={300}>
                  <BarChart data={containerMetrics?.containers || []}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="name" />
                    <YAxis />
                    <RechartsTooltip />
                    <Legend />
                    <Bar dataKey="cpu_percent" fill="#8884d8" name="CPU %" />
                    <Bar dataKey="memory_usage_mb" fill="#82ca9d" name="Memory MB" />
                  </BarChart>
                </ResponsiveContainer>
              </Card>
            </Col>
            
            <Col xs={24} lg={12}>
              <Card title="Container Overview">
                <Row gutter={[16, 16]}>
                  <Col xs={12}>
                    <Statistic
                      title="Total Containers"
                      value={containerMetrics?.total_containers || 0}
                      prefix={<CloudOutlined />}
                    />
                  </Col>
                  <Col xs={12}>
                    <Statistic
                      title="Avg CPU Usage"
                      value={containerMetrics?.average_cpu_usage || 0}
                      suffix="%"
                      precision={1}
                    />
                  </Col>
                  <Col xs={12}>
                    <Statistic
                      title="Total Memory"
                      value={containerMetrics?.total_memory_usage_gb || 0}
                      suffix="GB"
                      precision={1}
                    />
                  </Col>
                  <Col xs={12}>
                    <Statistic
                      title="Active Containers"
                      value={containerMetrics?.containers?.filter(c => c.status === 'running').length || 0}
                      valueStyle={{ color: '#52c41a' }}
                    />
                  </Col>
                </Row>
              </Card>
            </Col>
          </Row>
        </TabPane>

        <TabPane tab={<span><FireOutlined />Trading Performance</span>} key="3">
          <Row gutter={[16, 16]}>
            <Col xs={24} lg={8}>
              <Card title="Order Execution Latency">
                <div style={{ textAlign: 'center' }}>
                  <ResponsiveContainer width="100%" height={180}>
                    <RadialBarChart cx="50%" cy="50%" innerRadius="40%" outerRadius="90%" data={[
                      { name: 'Latency', value: Math.min((tradingMetrics?.order_execution_latency_ms || 0), 10) * 10, fill: '#8884d8' }
                    ]}>
                      <RadialBar minAngle={15} clockWise dataKey="value" />
                      <RechartsTooltip />
                    </RadialBarChart>
                  </ResponsiveContainer>
                  <Statistic
                    value={tradingMetrics?.order_execution_latency_ms || 0}
                    suffix="ms"
                    precision={2}
                  />
                </div>
              </Card>
            </Col>
            
            <Col xs={24} lg={8}>
              <Card title="Throughput">
                <div style={{ textAlign: 'center' }}>
                  <ResponsiveContainer width="100%" height={180}>
                    <RadialBarChart cx="50%" cy="50%" innerRadius="40%" outerRadius="90%" data={[
                      { name: 'Throughput', value: Math.min((tradingMetrics?.throughput_orders_per_second || 0), 100), fill: '#82ca9d' }
                    ]}>
                      <RadialBar minAngle={15} clockWise dataKey="value" />
                      <RechartsTooltip />
                    </RadialBarChart>
                  </ResponsiveContainer>
                  <Statistic
                    value={tradingMetrics?.throughput_orders_per_second || 0}
                    suffix="ops/sec"
                    precision={1}
                  />
                </div>
              </Card>
            </Col>
            
            <Col xs={24} lg={8}>
              <Card title="Hardware Acceleration Impact">
                <Statistic
                  title="Speedup Factor"
                  value={`${tradingMetrics?.hardware_acceleration_impact?.speedup_factor || 0}x`}
                  prefix={<RocketOutlined />}
                  valueStyle={{ color: '#52c41a' }}
                />
                <Statistic
                  title="GPU Operations"
                  value={tradingMetrics?.hardware_acceleration_impact?.gpu_operations || 0}
                  style={{ marginTop: '16px' }}
                />
              </Card>
            </Col>
          </Row>
        </TabPane>

        <TabPane tab={<span><ControlOutlined />CPU Optimization</span>} key="4">
          <Row gutter={[16, 16]}>
            <Col xs={24} lg={12}>
              <Card title="CPU Core Utilization">
                <ResponsiveContainer width="100%" height={300}>
                  <BarChart data={Object.entries(cpuHealth?.core_utilization || {}).map(([core, utilization]) => ({
                    core,
                    utilization: utilization as number,
                    type: core.includes('P') ? 'Performance' : 'Efficiency'
                  }))}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="core" />
                    <YAxis domain={[0, 100]} />
                    <RechartsTooltip />
                    <Legend />
                    <Bar dataKey="utilization" fill={(entry: any) => entry.type === 'Performance' ? '#1890ff' : '#52c41a'} />
                  </BarChart>
                </ResponsiveContainer>
              </Card>
            </Col>
            
            <Col xs={24} lg={12}>
              <Card title="CPU Optimization Status">
                <Row gutter={[16, 16]}>
                  <Col xs={24}>
                    <Alert
                      message="CPU Optimization Status"
                      description={`Optimization ${cpuHealth?.optimization_active ? 'Active' : 'Inactive'}`}
                      type={cpuHealth?.optimization_active ? 'success' : 'warning'}
                      showIcon
                      style={{ marginBottom: '16px' }}
                    />
                  </Col>
                  <Col xs={12}>
                    <Statistic
                      title="Performance Mode"
                      value={cpuHealth?.performance_mode || 'Unknown'}
                    />
                  </Col>
                  <Col xs={12}>
                    <Statistic
                      title="Workload Classification"
                      value={cpuHealth?.workload_classification || 'Unknown'}
                    />
                  </Col>
                </Row>
              </Card>
            </Col>
          </Row>
        </TabPane>
      </Tabs>
    </div>
  );
};

export default M4MaxMonitoringDashboard;