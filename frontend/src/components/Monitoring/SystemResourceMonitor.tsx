/**
 * Sprint 3: Enhanced System Resource Monitor
 * Advanced system resource monitoring with real-time charts and forecasting
 * CPU, Memory, Disk, Network monitoring with container-level granularity
 */

import React, { useState, useEffect, useRef } from 'react';
import {
  Card,
  Row,
  Col,
  Typography,
  Space,
  Button,
  Select,
  Progress,
  Alert,
  Spin,
  Tag,
  Tooltip,
  Table,
  Tabs,
  Statistic,
  Timeline,
  notification,
  Switch,
  Divider,
  List
} from 'antd';
import {
  DatabaseOutlined,
  ThunderboltOutlined,
  HddOutlined,
  ApiOutlined,
  ReloadOutlined,
  WarningOutlined,
  CheckCircleOutlined,
  ExclamationCircleOutlined,
  CloseCircleOutlined,
  MonitorOutlined,
  ContainerOutlined,
  CloudServerOutlined,
  LineChartOutlined,
  BarChartOutlined
} from '@ant-design/icons';
import type { ColumnsType } from 'antd/es/table';

const { Title, Text } = Typography;
const { Option } = Select;
const { TabPane } = Tabs;

interface SystemResource {
  timestamp: string;
  cpu: {
    usage_percent: number;
    per_core_usage: number[];
    load_average_1m: number;
    load_average_5m: number;
    load_average_15m: number;
    context_switches_per_sec: number;
    interrupts_per_sec: number;
    processes_running: number;
    processes_blocked: number;
  };
  memory: {
    total_gb: number;
    used_gb: number;
    available_gb: number;
    usage_percent: number;
    swap_total_gb: number;
    swap_used_gb: number;
    swap_percent: number;
    cached_gb: number;
    buffers_gb: number;
  };
  disk: {
    total_space_gb: number;
    used_space_gb: number;
    available_space_gb: number;
    usage_percent: number;
    read_iops: number;
    write_iops: number;
    read_throughput_mbps: number;
    write_throughput_mbps: number;
    avg_queue_size: number;
    avg_response_time_ms: number;
  };
  network: {
    bytes_sent_per_sec: number;
    bytes_received_per_sec: number;
    packets_sent_per_sec: number;
    packets_received_per_sec: number;
    errors_per_sec: number;
    dropped_packets_per_sec: number;
    active_connections: number;
    bandwidth_utilization_percent: number;
  };
}

interface ContainerResource {
  container_id: string;
  container_name: string;
  image: string;
  status: 'running' | 'stopped' | 'paused' | 'restarting';
  cpu_usage_percent: number;
  memory_usage_mb: number;
  memory_limit_mb: number;
  memory_usage_percent: number;
  network_io_mb: number;
  block_io_mb: number;
  pid_count: number;
  restart_count: number;
  uptime_seconds: number;
}

interface ResourceAlert {
  id: string;
  resource_type: 'cpu' | 'memory' | 'disk' | 'network';
  severity: 'critical' | 'high' | 'medium' | 'low';
  threshold_percent: number;
  current_percent: number;
  message: string;
  triggered_at: string;
  predicted_exhaustion?: string;
}

interface SystemResourceMonitorProps {
  className?: string;
}

export const SystemResourceMonitor: React.FC<SystemResourceMonitorProps> = ({
  className
}) => {
  const [activeTab, setActiveTab] = useState('overview');
  const [systemResource, setSystemResource] = useState<SystemResource | null>(null);
  const [containers, setContainers] = useState<ContainerResource[]>([]);
  const [resourceAlerts, setResourceAlerts] = useState<ResourceAlert[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [autoRefresh, setAutoRefresh] = useState(true);
  const [refreshInterval, setRefreshInterval] = useState(5);
  const [historicalData, setHistoricalData] = useState<SystemResource[]>([]);
  const [selectedTimeRange, setSelectedTimeRange] = useState('1h');
  const [lastUpdate, setLastUpdate] = useState<Date | null>(null);

  const fetchResourceData = async () => {
    try {
      const [resourceResponse, containersResponse, alertsResponse] = await Promise.all([
        fetch(`${import.meta.env.VITE_API_BASE_URL}/api/v1/system/resources/current`),
        fetch(`${import.meta.env.VITE_API_BASE_URL}/api/v1/system/containers/resources`),
        fetch(`${import.meta.env.VITE_API_BASE_URL}/api/v1/system/resources/alerts`)
      ]);

      if (!resourceResponse.ok || !containersResponse.ok || !alertsResponse.ok) {
        throw new Error('Failed to fetch resource data');
      }

      const [resourceData, containersData, alertsData] = await Promise.all([
        resourceResponse.json(),
        containersResponse.json(),
        alertsResponse.json()
      ]);

      setSystemResource(resourceData);
      setContainers(containersData);
      setResourceAlerts(alertsData);
      
      // Update historical data for charting
      setHistoricalData(prev => {
        const newData = [...prev, resourceData].slice(-60); // Keep last 60 readings
        return newData;
      });

      setError(null);
      setLastUpdate(new Date());
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to fetch resource data');
      console.error('Resource data fetch error:', err);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchResourceData();
  }, []);

  useEffect(() => {
    if (autoRefresh && refreshInterval > 0) {
      const interval = setInterval(fetchResourceData, refreshInterval * 1000);
      return () => clearInterval(interval);
    }
  }, [autoRefresh, refreshInterval]);

  const handleRefresh = async () => {
    setLoading(true);
    await fetchResourceData();
    notification.success({
      message: 'Resource Data Updated',
      description: 'All system resource metrics have been refreshed',
      duration: 2
    });
  };

  const getResourceStatus = (percent: number, type: 'cpu' | 'memory' | 'disk' | 'network') => {
    if (type === 'disk') {
      if (percent > 90) return { status: 'error', color: '#ff4d4f' };
      if (percent > 80) return { status: 'active', color: '#faad14' };
      return { status: 'success', color: '#52c41a' };
    } else {
      if (percent > 85) return { status: 'exception', color: '#ff4d4f' };
      if (percent > 70) return { status: 'active', color: '#faad14' };
      return { status: 'success', color: '#52c41a' };
    }
  };

  const formatBytes = (bytes: number, decimals = 2) => {
    if (bytes === 0) return '0 B';
    const k = 1024;
    const dm = decimals < 0 ? 0 : decimals;
    const sizes = ['B', 'KB', 'MB', 'GB', 'TB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(dm)) + ' ' + sizes[i];
  };

  const formatUptime = (seconds: number) => {
    const days = Math.floor(seconds / 86400);
    const hours = Math.floor((seconds % 86400) / 3600);
    const minutes = Math.floor((seconds % 3600) / 60);
    
    if (days > 0) return `${days}d ${hours}h`;
    if (hours > 0) return `${hours}h ${minutes}m`;
    return `${minutes}m`;
  };

  const containerColumns: ColumnsType<ContainerResource> = [
    {
      title: 'Container',
      dataIndex: 'container_name',
      key: 'name',
      render: (name, record) => (
        <Space>
          <ContainerOutlined style={{ 
            color: record.status === 'running' ? '#52c41a' : record.status === 'stopped' ? '#ff4d4f' : '#faad14' 
          }} />
          <div>
            <div>{name}</div>
            <Text type="secondary" style={{ fontSize: '11px' }}>
              {record.image.length > 30 ? record.image.substring(0, 30) + '...' : record.image}
            </Text>
          </div>
        </Space>
      )
    },
    {
      title: 'Status',
      dataIndex: 'status',
      key: 'status',
      render: (status) => (
        <Tag color={
          status === 'running' ? 'green' :
          status === 'stopped' ? 'red' :
          status === 'paused' ? 'orange' : 'blue'
        }>
          {status.toUpperCase()}
        </Tag>
      )
    },
    {
      title: 'CPU',
      dataIndex: 'cpu_usage_percent',
      key: 'cpu',
      render: (cpu) => {
        const status = getResourceStatus(cpu, 'cpu');
        return (
          <div style={{ width: 60 }}>
            <Progress
              percent={cpu}
              size="small"
              status={status.status as any}
              format={() => `${cpu.toFixed(1)}%`}
            />
          </div>
        );
      },
      sorter: (a, b) => a.cpu_usage_percent - b.cpu_usage_percent
    },
    {
      title: 'Memory',
      key: 'memory',
      render: (_, record) => {
        const percent = record.memory_usage_percent;
        const status = getResourceStatus(percent, 'memory');
        return (
          <div>
            <div style={{ width: 60 }}>
              <Progress
                percent={percent}
                size="small"
                status={status.status as any}
                format={() => `${percent.toFixed(0)}%`}
              />
            </div>
            <Text type="secondary" style={{ fontSize: '10px' }}>
              {formatBytes(record.memory_usage_mb * 1024 * 1024)} / {formatBytes(record.memory_limit_mb * 1024 * 1024)}
            </Text>
          </div>
        );
      },
      sorter: (a, b) => a.memory_usage_percent - b.memory_usage_percent
    },
    {
      title: 'Network I/O',
      dataIndex: 'network_io_mb',
      key: 'network',
      render: (io) => formatBytes(io * 1024 * 1024),
      sorter: (a, b) => a.network_io_mb - b.network_io_mb
    },
    {
      title: 'Block I/O',
      dataIndex: 'block_io_mb',
      key: 'block',
      render: (io) => formatBytes(io * 1024 * 1024),
      sorter: (a, b) => a.block_io_mb - b.block_io_mb
    },
    {
      title: 'Uptime',
      dataIndex: 'uptime_seconds',
      key: 'uptime',
      render: (uptime) => formatUptime(uptime)
    },
    {
      title: 'Restarts',
      dataIndex: 'restart_count',
      key: 'restarts',
      render: (count) => (
        <Badge 
          count={count} 
          style={{ backgroundColor: count > 0 ? '#fa541c' : '#52c41a' }}
          showZero
        />
      ),
      sorter: (a, b) => a.restart_count - b.restart_count
    }
  ];

  const renderOverviewTab = () => {
    if (!systemResource) return null;

    const cpuStatus = getResourceStatus(systemResource.cpu.usage_percent, 'cpu');
    const memoryStatus = getResourceStatus(systemResource.memory.usage_percent, 'memory');
    const diskStatus = getResourceStatus(systemResource.disk.usage_percent, 'disk');

    return (
      <div>
        {/* Critical Resource Alerts */}
        {resourceAlerts.filter(alert => alert.severity === 'critical').length > 0 && (
          <Alert
            message="Critical Resource Alerts"
            description={`${resourceAlerts.filter(alert => alert.severity === 'critical').length} resources require immediate attention`}
            type="error"
            style={{ marginBottom: 16 }}
            showIcon
            action={
              <Button size="small" danger>
                View Details
              </Button>
            }
          />
        )}

        {/* Resource Overview Cards */}
        <Row gutter={[16, 16]} style={{ marginBottom: 24 }}>
          <Col xs={24} sm={12} lg={6}>
            <Card>
              <Statistic
                title="CPU Usage"
                value={systemResource.cpu.usage_percent}
                suffix="%"
                precision={1}
                valueStyle={{ color: cpuStatus.color }}
                prefix={<ThunderboltOutlined />}
              />
              <Progress
                percent={systemResource.cpu.usage_percent}
                size="small"
                status={cpuStatus.status as any}
                showInfo={false}
              />
              <div style={{ marginTop: 4, fontSize: '11px', color: '#666' }}>
                Load: {systemResource.cpu.load_average_1m.toFixed(2)} / {systemResource.cpu.load_average_5m.toFixed(2)} / {systemResource.cpu.load_average_15m.toFixed(2)}
              </div>
            </Card>
          </Col>

          <Col xs={24} sm={12} lg={6}>
            <Card>
              <Statistic
                title="Memory Usage"
                value={systemResource.memory.usage_percent}
                suffix="%"
                precision={1}
                valueStyle={{ color: memoryStatus.color }}
                prefix={<DatabaseOutlined />}
              />
              <Progress
                percent={systemResource.memory.usage_percent}
                size="small"
                status={memoryStatus.status as any}
                showInfo={false}
              />
              <div style={{ marginTop: 4, fontSize: '11px', color: '#666' }}>
                {systemResource.memory.used_gb.toFixed(1)}GB / {systemResource.memory.total_gb.toFixed(1)}GB
              </div>
            </Card>
          </Col>

          <Col xs={24} sm={12} lg={6}>
            <Card>
              <Statistic
                title="Disk Usage"
                value={systemResource.disk.usage_percent}
                suffix="%"
                precision={1}
                valueStyle={{ color: diskStatus.color }}
                prefix={<HddOutlined />}
              />
              <Progress
                percent={systemResource.disk.usage_percent}
                size="small"
                status={diskStatus.status as any}
                showInfo={false}
              />
              <div style={{ marginTop: 4, fontSize: '11px', color: '#666' }}>
                {systemResource.disk.used_space_gb.toFixed(0)}GB / {systemResource.disk.total_space_gb.toFixed(0)}GB
              </div>
            </Card>
          </Col>

          <Col xs={24} sm={12} lg={6}>
            <Card>
              <Statistic
                title="Network I/O"
                value={systemResource.network.bytes_sent_per_sec + systemResource.network.bytes_received_per_sec}
                formatter={(value) => formatBytes(value as number, 0) + '/s'}
                valueStyle={{ color: '#1890ff' }}
                prefix={<ApiOutlined />}
              />
              <div style={{ marginTop: 8, fontSize: '11px', color: '#666' }}>
                ↑ {formatBytes(systemResource.network.bytes_sent_per_sec)}/s
                <br />
                ↓ {formatBytes(systemResource.network.bytes_received_per_sec)}/s
              </div>
            </Card>
          </Col>
        </Row>

        {/* Detailed Metrics */}
        <Row gutter={[16, 16]} style={{ marginBottom: 24 }}>
          <Col xs={24} lg={12}>
            <Card title="CPU Details" size="small">
              <Row gutter={16}>
                <Col span={12}>
                  <Statistic
                    title="Running Processes"
                    value={systemResource.cpu.processes_running}
                    valueStyle={{ color: '#52c41a' }}
                  />
                </Col>
                <Col span={12}>
                  <Statistic
                    title="Blocked Processes"
                    value={systemResource.cpu.processes_blocked}
                    valueStyle={{ color: systemResource.cpu.processes_blocked > 0 ? '#fa541c' : '#52c41a' }}
                  />
                </Col>
              </Row>
              <Divider style={{ margin: '12px 0' }} />
              <div>
                <Text type="secondary">Context Switches: </Text>
                <Text>{systemResource.cpu.context_switches_per_sec.toLocaleString()}/s</Text>
              </div>
              <div>
                <Text type="secondary">Interrupts: </Text>
                <Text>{systemResource.cpu.interrupts_per_sec.toLocaleString()}/s</Text>
              </div>
            </Card>
          </Col>

          <Col xs={24} lg={12}>
            <Card title="Memory Details" size="small">
              <Row gutter={16}>
                <Col span={12}>
                  <Statistic
                    title="Available"
                    value={systemResource.memory.available_gb}
                    suffix="GB"
                    precision={1}
                    valueStyle={{ color: '#52c41a' }}
                  />
                </Col>
                <Col span={12}>
                  <Statistic
                    title="Swap Used"
                    value={systemResource.memory.swap_percent}
                    suffix="%"
                    precision={1}
                    valueStyle={{ color: systemResource.memory.swap_percent > 50 ? '#fa541c' : '#52c41a' }}
                  />
                </Col>
              </Row>
              <Divider style={{ margin: '12px 0' }} />
              <div>
                <Text type="secondary">Cached: </Text>
                <Text>{systemResource.memory.cached_gb.toFixed(1)}GB</Text>
              </div>
              <div>
                <Text type="secondary">Buffers: </Text>
                <Text>{systemResource.memory.buffers_gb.toFixed(1)}GB</Text>
              </div>
            </Card>
          </Col>
        </Row>

        {/* Disk I/O Performance */}
        <Row gutter={[16, 16]} style={{ marginBottom: 24 }}>
          <Col xs={24} lg={12}>
            <Card title="Disk I/O Performance" size="small">
              <Row gutter={16}>
                <Col span={12}>
                  <Statistic
                    title="Read IOPS"
                    value={systemResource.disk.read_iops}
                    valueStyle={{ color: '#1890ff' }}
                  />
                </Col>
                <Col span={12}>
                  <Statistic
                    title="Write IOPS"
                    value={systemResource.disk.write_iops}
                    valueStyle={{ color: '#722ed1' }}
                  />
                </Col>
              </Row>
              <Divider style={{ margin: '12px 0' }} />
              <div>
                <Text type="secondary">Read Throughput: </Text>
                <Text>{systemResource.disk.read_throughput_mbps.toFixed(1)} MB/s</Text>
              </div>
              <div>
                <Text type="secondary">Write Throughput: </Text>
                <Text>{systemResource.disk.write_throughput_mbps.toFixed(1)} MB/s</Text>
              </div>
              <div style={{ marginTop: 8 }}>
                <Text type="secondary">Avg Response Time: </Text>
                <Text style={{ color: systemResource.disk.avg_response_time_ms > 10 ? '#fa541c' : '#52c41a' }}>
                  {systemResource.disk.avg_response_time_ms.toFixed(1)}ms
                </Text>
              </div>
            </Card>
          </Col>

          <Col xs={24} lg={12}>
            <Card title="Network Statistics" size="small">
              <Row gutter={16}>
                <Col span={12}>
                  <Statistic
                    title="Active Connections"
                    value={systemResource.network.active_connections}
                    valueStyle={{ color: '#13c2c2' }}
                  />
                </Col>
                <Col span={12}>
                  <Statistic
                    title="Packet Errors"
                    value={systemResource.network.errors_per_sec}
                    suffix="/s"
                    valueStyle={{ color: systemResource.network.errors_per_sec > 0 ? '#ff4d4f' : '#52c41a' }}
                  />
                </Col>
              </Row>
              <Divider style={{ margin: '12px 0' }} />
              <div>
                <Text type="secondary">Packets Sent: </Text>
                <Text>{systemResource.network.packets_sent_per_sec.toLocaleString()}/s</Text>
              </div>
              <div>
                <Text type="secondary">Packets Received: </Text>
                <Text>{systemResource.network.packets_received_per_sec.toLocaleString()}/s</Text>
              </div>
              <div style={{ marginTop: 8 }}>
                <Text type="secondary">Dropped Packets: </Text>
                <Text style={{ color: systemResource.network.dropped_packets_per_sec > 0 ? '#fa541c' : '#52c41a' }}>
                  {systemResource.network.dropped_packets_per_sec}/s
                </Text>
              </div>
            </Card>
          </Col>
        </Row>

        {/* Resource Alerts */}
        {resourceAlerts.length > 0 && (
          <Card title="Resource Alerts" size="small">
            <List
              dataSource={resourceAlerts}
              renderItem={(alert) => (
                <List.Item
                  key={alert.id}
                  actions={[
                    <Tag color={
                      alert.severity === 'critical' ? 'red' :
                      alert.severity === 'high' ? 'orange' :
                      alert.severity === 'medium' ? 'yellow' : 'green'
                    }>
                      {alert.severity.toUpperCase()}
                    </Tag>
                  ]}
                >
                  <List.Item.Meta
                    avatar={
                      <div style={{ fontSize: '20px', color: alert.severity === 'critical' ? '#ff4d4f' : '#faad14' }}>
                        {alert.resource_type === 'cpu' && <ThunderboltOutlined />}
                        {alert.resource_type === 'memory' && <DatabaseOutlined />}
                        {alert.resource_type === 'disk' && <HddOutlined />}
                        {alert.resource_type === 'network' && <ApiOutlined />}
                      </div>
                    }
                    title={`${alert.resource_type.toUpperCase()} Alert`}
                    description={
                      <div>
                        <div>{alert.message}</div>
                        <Text type="secondary" style={{ fontSize: '11px' }}>
                          Current: {alert.current_percent.toFixed(1)}% | Threshold: {alert.threshold_percent}% | 
                          Triggered: {new Date(alert.triggered_at).toLocaleString()}
                        </Text>
                        {alert.predicted_exhaustion && (
                          <div>
                            <Text type="danger" style={{ fontSize: '11px' }}>
                              Predicted exhaustion: {alert.predicted_exhaustion}
                            </Text>
                          </div>
                        )}
                      </div>
                    }
                  />
                </List.Item>
              )}
            />
          </Card>
        )}
      </div>
    );
  };

  const renderContainersTab = () => (
    <div>
      <Row gutter={[16, 16]} style={{ marginBottom: 16 }}>
        <Col xs={24} sm={6}>
          <Card size="small">
            <div style={{ textAlign: 'center' }}>
              <div style={{ fontSize: '24px', color: '#52c41a' }}>
                {containers.filter(c => c.status === 'running').length}
              </div>
              <Text type="secondary">Running</Text>
            </div>
          </Card>
        </Col>
        <Col xs={24} sm={6}>
          <Card size="small">
            <div style={{ textAlign: 'center' }}>
              <div style={{ fontSize: '24px', color: '#ff4d4f' }}>
                {containers.filter(c => c.status === 'stopped').length}
              </div>
              <Text type="secondary">Stopped</Text>
            </div>
          </Card>
        </Col>
        <Col xs={24} sm={6}>
          <Card size="small">
            <div style={{ textAlign: 'center' }}>
              <div style={{ fontSize: '24px', color: '#1890ff' }}>
                {containers.reduce((sum, c) => sum + c.cpu_usage_percent, 0).toFixed(0)}%
              </div>
              <Text type="secondary">Total CPU</Text>
            </div>
          </Card>
        </Col>
        <Col xs={24} sm={6}>
          <Card size="small">
            <div style={{ textAlign: 'center' }}>
              <div style={{ fontSize: '24px', color: '#722ed1' }}>
                {formatBytes(containers.reduce((sum, c) => sum + c.memory_usage_mb, 0) * 1024 * 1024, 1)}
              </div>
              <Text type="secondary">Total Memory</Text>
            </div>
          </Card>
        </Col>
      </Row>

      <Table
        columns={containerColumns}
        dataSource={containers}
        rowKey="container_id"
        size="small"
        loading={loading}
        pagination={{ pageSize: 10 }}
        scroll={{ x: 1000 }}
      />
    </div>
  );

  return (
    <div className={`system-resource-monitor ${className || ''}`}>
      {/* Header */}
      <div style={{ marginBottom: 24 }}>
        <Row justify="space-between" align="middle">
          <Col>
            <Title level={3} style={{ margin: 0 }}>
              <MonitorOutlined style={{ marginRight: 8, color: '#1890ff' }} />
              System Resource Monitor
            </Title>
            <Text type="secondary">
              Enhanced resource monitoring with container-level granularity
            </Text>
          </Col>
          <Col>
            <Space>
              <Select
                value={refreshInterval}
                onChange={setRefreshInterval}
                style={{ width: 100 }}
                size="small"
              >
                <Option value={1}>1s</Option>
                <Option value={5}>5s</Option>
                <Option value={10}>10s</Option>
                <Option value={30}>30s</Option>
              </Select>

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
              Auto-refresh: {autoRefresh ? `${refreshInterval}s` : 'Off'}
            </Text>
          </div>
        )}
      </div>

      {/* Error Display */}
      {error && (
        <Alert
          message="Resource Monitoring Error"
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
              <MonitorOutlined />
              System Overview
            </span>
          } 
          key="overview"
        >
          {loading && !systemResource ? (
            <div style={{ textAlign: 'center', padding: 60 }}>
              <Spin size="large" />
              <div style={{ marginTop: 16 }}>
                <Text type="secondary">Loading system resources...</Text>
              </div>
            </div>
          ) : (
            renderOverviewTab()
          )}
        </TabPane>

        <TabPane 
          tab={
            <span>
              <ContainerOutlined />
              Container Resources
              {containers.length > 0 && (
                <Badge 
                  count={containers.filter(c => c.status === 'running').length} 
                  offset={[8, -4]} 
                  style={{ backgroundColor: '#52c41a' }}
                />
              )}
            </span>
          } 
          key="containers"
        >
          {renderContainersTab()}
        </TabPane>

        <TabPane 
          tab={
            <span>
              <LineChartOutlined />
              Historical Trends
            </span>
          } 
          key="trends"
        >
          <div style={{ textAlign: 'center', padding: 60 }}>
            <Text type="secondary">
              Resource trend charts and capacity planning will be implemented here
            </Text>
          </div>
        </TabPane>

        <TabPane 
          tab={
            <span>
              <WarningOutlined />
              Resource Alerts
              {resourceAlerts.length > 0 && (
                <Badge 
                  count={resourceAlerts.length} 
                  offset={[8, -4]} 
                />
              )}
            </span>
          } 
          key="alerts"
        >
          <div style={{ textAlign: 'center', padding: 60 }}>
            <Text type="secondary">
              Resource alert configuration and history will be implemented here
            </Text>
          </div>
        </TabPane>
      </Tabs>
    </div>
  );
};

export default SystemResourceMonitor;