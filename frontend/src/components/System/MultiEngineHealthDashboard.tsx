/**
 * Multi-Engine Health Dashboard
 * Monitors all 9 containerized engines + main backend
 * Based on FRONTEND_ENDPOINT_INTEGRATION_GUIDE.md
 */

import React, { useState, useEffect } from 'react';
import {
  Card,
  Row,
  Col,
  Statistic,
  Button,
  Alert,
  Typography,
  Space,
  Tag,
  Tooltip,
  Spin,
  Progress,
  Table,
  Badge,
  Modal
} from 'antd';
import {
  HeartOutlined,
  ExclamationCircleOutlined,
  CheckCircleOutlined,
  CloseCircleOutlined,
  WarningOutlined,
  SyncOutlined,
  ApiOutlined,
  DashboardOutlined,
  MonitorOutlined
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
  BarChart, 
  Bar,
  PieChart,
  Pie,
  Cell
} from 'recharts';
import apiClient, { HealthResponse, API_CONFIG } from '../../services/apiClient';
import { systemHealthWS, ConnectionStatus } from '../../services/websocketClient';

const { Title, Text } = Typography;

// Engine configuration
const ENGINE_CONFIG = [
  { name: 'Analytics', key: 'ANALYTICS', port: 8100, description: 'Real-time analytics processing' },
  { name: 'Risk', key: 'RISK', port: 8200, description: 'Enhanced risk management' },
  { name: 'Factor', key: 'FACTOR', port: 8300, description: 'Factor synthesis (485 factors)' },
  { name: 'ML', key: 'ML', port: 8400, description: 'Machine learning inference' },
  { name: 'Features', key: 'FEATURES', port: 8500, description: 'Feature engineering' },
  { name: 'WebSocket', key: 'WEBSOCKET', port: 8600, description: 'WebSocket streaming' },
  { name: 'Strategy', key: 'STRATEGY', port: 8700, description: 'Strategy execution' },
  { name: 'MarketData', key: 'MARKETDATA', port: 8800, description: 'Market data processing' },
  { name: 'Portfolio', key: 'PORTFOLIO', port: 8900, description: 'Portfolio management' }
];

interface EngineHealth {
  name: string;
  port: number;
  status: 'healthy' | 'degraded' | 'unhealthy' | 'error';
  response_time_ms?: number;
  uptime_seconds?: number;
  requests_processed?: number;
  error?: string;
  lastCheck: number;
}

interface SystemOverview {
  totalEngines: number;
  healthyEngines: number;
  degradedEngines: number;
  unhealthyEngines: number;
  averageResponseTime: number;
  totalRequests: number;
  systemUptime: number;
}

const COLORS = ['#52c41a', '#faad14', '#f5222d', '#d9d9d9'];
const STATUS_COLORS = {
  healthy: '#52c41a',
  degraded: '#faad14', 
  unhealthy: '#f5222d',
  error: '#d9d9d9'
};

const MultiEngineHealthDashboard: React.FC = () => {
  // State management
  const [loading, setLoading] = useState(false);
  const [enginesHealth, setEnginesHealth] = useState<EngineHealth[]>([]);
  const [systemOverview, setSystemOverview] = useState<SystemOverview | null>(null);
  const [healthHistory, setHealthHistory] = useState<Array<{ timestamp: number; [key: string]: any }>>([]);
  const [autoRefresh, setAutoRefresh] = useState(true);
  const [refreshInterval, setRefreshInterval] = useState<NodeJS.Timeout | null>(null);
  const [wsStatus, setWsStatus] = useState<ConnectionStatus>(ConnectionStatus.DISCONNECTED);
  const [error, setError] = useState<string | null>(null);
  const [selectedEngine, setSelectedEngine] = useState<EngineHealth | null>(null);
  const [detailsVisible, setDetailsVisible] = useState(false);

  // Load initial data and setup real-time updates
  useEffect(() => {
    loadEngineHealth();
    connectToRealtimeUpdates();
    
    if (autoRefresh) {
      const interval = setInterval(loadEngineHealth, 30000); // 30-second updates
      setRefreshInterval(interval);
    }
    
    return () => {
      if (refreshInterval) clearInterval(refreshInterval);
      systemHealthWS.disconnect();
    };
  }, [autoRefresh]);

  const loadEngineHealth = async () => {
    if (!loading) setLoading(true);
    setError(null);
    
    try {
      // Check main backend health
      const mainBackendHealth = await apiClient.getSystemHealth();
      
      // Check all engine health in parallel
      const engineHealthPromises = ENGINE_CONFIG.map(async (engine) => {
        try {
          const startTime = Date.now();
          const health = await apiClient.getEngineHealth(engine.key as keyof typeof API_CONFIG.ENGINES);
          const responseTime = Date.now() - startTime;
          
          return {
            name: engine.name,
            port: engine.port,
            status: health.status as 'healthy' | 'degraded' | 'unhealthy',
            response_time_ms: responseTime,
            uptime_seconds: health.uptime_seconds,
            requests_processed: health.requests_processed,
            lastCheck: Date.now()
          };
        } catch (err) {
          return {
            name: engine.name,
            port: engine.port,
            status: 'error' as const,
            error: (err as Error).message,
            lastCheck: Date.now()
          };
        }
      });

      const engineResults = await Promise.all(engineHealthPromises);
      setEnginesHealth(engineResults);

      // Calculate system overview
      const overview: SystemOverview = {
        totalEngines: engineResults.length,
        healthyEngines: engineResults.filter(e => e.status === 'healthy').length,
        degradedEngines: engineResults.filter(e => e.status === 'degraded').length,
        unhealthyEngines: engineResults.filter(e => e.status === 'unhealthy' || e.status === 'error').length,
        averageResponseTime: engineResults
          .filter(e => e.response_time_ms)
          .reduce((sum, e) => sum + (e.response_time_ms || 0), 0) / 
          engineResults.filter(e => e.response_time_ms).length || 0,
        totalRequests: engineResults
          .reduce((sum, e) => sum + (e.requests_processed || 0), 0),
        systemUptime: Math.max(...engineResults.map(e => e.uptime_seconds || 0))
      };
      setSystemOverview(overview);

      // Add to health history
      const historyEntry = {
        timestamp: Date.now(),
        ...engineResults.reduce((acc, engine) => ({
          ...acc,
          [`${engine.name}_status`]: engine.status === 'healthy' ? 100 : engine.status === 'degraded' ? 50 : 0,
          [`${engine.name}_response`]: engine.response_time_ms || 0
        }), {})
      };
      
      setHealthHistory(prev => [
        ...prev.slice(-119), // Keep last 120 points
        historyEntry
      ]);

    } catch (err) {
      setError(`Failed to load engine health: ${(err as Error).message}`);
    } finally {
      setLoading(false);
    }
  };

  const connectToRealtimeUpdates = async () => {
    try {
      await systemHealthWS.connectToSystemHealth(
        (update) => {
          // Process real-time health updates
          console.log('Real-time health update:', update);
          // Update engine statuses based on real-time data
        },
        (status) => {
          setWsStatus(status);
        }
      );
    } catch (err) {
      console.error('Failed to connect to real-time health updates:', err);
    }
  };

  const toggleAutoRefresh = () => {
    setAutoRefresh(!autoRefresh);
    if (refreshInterval) {
      clearInterval(refreshInterval);
      setRefreshInterval(null);
    }
  };

  const showEngineDetails = (engine: EngineHealth) => {
    setSelectedEngine(engine);
    setDetailsVisible(true);
  };

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'healthy': return <CheckCircleOutlined style={{ color: STATUS_COLORS.healthy }} />;
      case 'degraded': return <WarningOutlined style={{ color: STATUS_COLORS.degraded }} />;
      case 'unhealthy': return <ExclamationCircleOutlined style={{ color: STATUS_COLORS.unhealthy }} />;
      case 'error': return <CloseCircleOutlined style={{ color: STATUS_COLORS.error }} />;
      default: return <MonitorOutlined />;
    }
  };

  const getStatusBadge = (status: string) => {
    const statusMap = {
      healthy: 'success',
      degraded: 'warning',
      unhealthy: 'error',
      error: 'default'
    };
    return statusMap[status as keyof typeof statusMap] || 'default';
  };

  const formatUptime = (seconds: number) => {
    const hours = Math.floor(seconds / 3600);
    const minutes = Math.floor((seconds % 3600) / 60);
    return `${hours}h ${minutes}m`;
  };

  // Table columns for detailed view
  const columns = [
    {
      title: 'Engine',
      dataIndex: 'name',
      key: 'name',
      render: (name: string, record: EngineHealth) => (
        <Space>
          {getStatusIcon(record.status)}
          <Text strong>{name}</Text>
          <Text type="secondary">:{record.port}</Text>
        </Space>
      )
    },
    {
      title: 'Status',
      dataIndex: 'status',
      key: 'status',
      render: (status: string) => (
        <Badge status={getStatusBadge(status) as any} text={status.toUpperCase()} />
      )
    },
    {
      title: 'Response Time',
      dataIndex: 'response_time_ms',
      key: 'response_time_ms',
      render: (time: number) => time ? `${time}ms` : 'N/A'
    },
    {
      title: 'Uptime',
      dataIndex: 'uptime_seconds',
      key: 'uptime_seconds',
      render: (uptime: number) => uptime ? formatUptime(uptime) : 'N/A'
    },
    {
      title: 'Requests',
      dataIndex: 'requests_processed',
      key: 'requests_processed',
      render: (requests: number) => requests ? requests.toLocaleString() : 'N/A'
    },
    {
      title: 'Actions',
      key: 'actions',
      render: (_, record: EngineHealth) => (
        <Button size="small" onClick={() => showEngineDetails(record)}>
          Details
        </Button>
      )
    }
  ];

  if (loading && enginesHealth.length === 0) {
    return (
      <div style={{ display: 'flex', justifyContent: 'center', alignItems: 'center', height: '400px' }}>
        <Spin size="large" tip="Loading Engine Health..." />
      </div>
    );
  }

  return (
    <div style={{ padding: '24px' }}>
      <Title level={2}>
        <HeartOutlined style={{ marginRight: '8px', color: '#52c41a' }} />
        Multi-Engine Health Dashboard
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
                onClick={loadEngineHealth} 
                loading={loading}
              >
                Refresh All Engines
              </Button>
              <Button 
                type={autoRefresh ? 'primary' : 'default'}
                onClick={toggleAutoRefresh}
              >
                Auto Refresh {autoRefresh ? 'ON' : 'OFF'}
              </Button>
              <Text type="secondary">
                WebSocket Status: 
                <Tag color={wsStatus === ConnectionStatus.CONNECTED ? 'green' : 'orange'}>
                  {wsStatus}
                </Tag>
              </Text>
              <Text type="secondary">Last Updated: {new Date().toLocaleTimeString()}</Text>
            </Space>
          </Card>
        </Col>
      </Row>

      {/* System Overview */}
      {systemOverview && (
        <Row gutter={[16, 16]} style={{ marginBottom: '24px' }}>
          <Col xs={24} sm={12} md={6}>
            <Card>
              <Statistic
                title="Total Engines"
                value={systemOverview.totalEngines}
                prefix={<DashboardOutlined />}
              />
            </Card>
          </Col>
          <Col xs={24} sm={12} md={6}>
            <Card>
              <Statistic
                title="Healthy Engines"
                value={systemOverview.healthyEngines}
                prefix={<CheckCircleOutlined />}
                valueStyle={{ color: '#52c41a' }}
              />
              <Progress 
                percent={(systemOverview.healthyEngines / systemOverview.totalEngines) * 100}
                strokeColor="#52c41a"
                size="small"
                showInfo={false}
              />
            </Card>
          </Col>
          <Col xs={24} sm={12} md={6}>
            <Card>
              <Statistic
                title="Avg Response Time"
                value={systemOverview.averageResponseTime}
                precision={1}
                suffix="ms"
                prefix={<ApiOutlined />}
                valueStyle={{ 
                  color: systemOverview.averageResponseTime < 100 ? '#52c41a' : 
                         systemOverview.averageResponseTime < 500 ? '#faad14' : '#f5222d'
                }}
              />
            </Card>
          </Col>
          <Col xs={24} sm={12} md={6}>
            <Card>
              <Statistic
                title="System Uptime"
                value={formatUptime(systemOverview.systemUptime)}
                prefix={<MonitorOutlined />}
              />
            </Card>
          </Col>
        </Row>
      )}

      {/* Engine Status Grid */}
      <Row gutter={[16, 16]} style={{ marginBottom: '24px' }}>
        {enginesHealth.map((engine) => (
          <Col xs={24} sm={12} md={8} lg={6} xl={4} key={engine.name}>
            <Card 
              size="small"
              hoverable
              onClick={() => showEngineDetails(engine)}
              style={{ 
                borderColor: STATUS_COLORS[engine.status],
                borderWidth: 2
              }}
            >
              <div style={{ textAlign: 'center' }}>
                <div style={{ fontSize: '24px', marginBottom: '8px' }}>
                  {getStatusIcon(engine.status)}
                </div>
                <Text strong>{engine.name}</Text>
                <br />
                <Text type="secondary">:{engine.port}</Text>
                <br />
                {engine.response_time_ms && (
                  <Tag color="blue" size="small">{engine.response_time_ms}ms</Tag>
                )}
                <br />
                <Badge status={getStatusBadge(engine.status) as any} text={engine.status.toUpperCase()} />
              </div>
            </Card>
          </Col>
        ))}
      </Row>

      {/* Health Trends Chart */}
      <Row gutter={[16, 16]} style={{ marginBottom: '24px' }}>
        <Col xs={24} lg={16}>
          <Card title="Engine Health Trends">
            <ResponsiveContainer width="100%" height={350}>
              <LineChart data={healthHistory.map((entry, index) => ({
                ...entry,
                timestamp: new Date(entry.timestamp).toLocaleTimeString()
              }))}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="timestamp" />
                <YAxis domain={[0, 100]} />
                <RechartsTooltip />
                <Legend />
                {ENGINE_CONFIG.slice(0, 4).map((engine, index) => (
                  <Line
                    key={engine.name}
                    type="monotone"
                    dataKey={`${engine.name}_status`}
                    stroke={COLORS[index]}
                    strokeWidth={2}
                    name={engine.name}
                  />
                ))}
              </LineChart>
            </ResponsiveContainer>
          </Card>
        </Col>
        
        <Col xs={24} lg={8}>
          <Card title="Status Distribution">
            <ResponsiveContainer width="100%" height={300}>
              <PieChart>
                <Pie
                  data={[
                    { name: 'Healthy', value: systemOverview?.healthyEngines || 0 },
                    { name: 'Degraded', value: systemOverview?.degradedEngines || 0 },
                    { name: 'Unhealthy', value: systemOverview?.unhealthyEngines || 0 }
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
          </Card>
        </Col>
      </Row>

      {/* Detailed Engine Table */}
      <Row gutter={[16, 16]}>
        <Col span={24}>
          <Card title="Detailed Engine Status">
            <Table
              dataSource={enginesHealth}
              columns={columns}
              rowKey="name"
              pagination={false}
              size="middle"
              loading={loading}
            />
          </Card>
        </Col>
      </Row>

      {/* Engine Details Modal */}
      <Modal
        title={`${selectedEngine?.name} Engine Details`}
        visible={detailsVisible}
        onCancel={() => setDetailsVisible(false)}
        footer={null}
        width={600}
      >
        {selectedEngine && (
          <div>
            <Row gutter={[16, 16]}>
              <Col span={12}>
                <Statistic title="Status" value={selectedEngine.status.toUpperCase()} />
              </Col>
              <Col span={12}>
                <Statistic title="Port" value={selectedEngine.port} />
              </Col>
              <Col span={12}>
                <Statistic 
                  title="Response Time" 
                  value={selectedEngine.response_time_ms || 'N/A'} 
                  suffix={selectedEngine.response_time_ms ? 'ms' : ''}
                />
              </Col>
              <Col span={12}>
                <Statistic 
                  title="Uptime" 
                  value={selectedEngine.uptime_seconds ? formatUptime(selectedEngine.uptime_seconds) : 'N/A'} 
                />
              </Col>
              <Col span={12}>
                <Statistic 
                  title="Requests Processed" 
                  value={selectedEngine.requests_processed?.toLocaleString() || 'N/A'} 
                />
              </Col>
              <Col span={12}>
                <Statistic 
                  title="Last Check" 
                  value={new Date(selectedEngine.lastCheck).toLocaleTimeString()} 
                />
              </Col>
            </Row>
            {selectedEngine.error && (
              <Alert
                message="Error Details"
                description={selectedEngine.error}
                type="error"
                style={{ marginTop: '16px' }}
              />
            )}
          </div>
        )}
      </Modal>
    </div>
  );
};

export default MultiEngineHealthDashboard;