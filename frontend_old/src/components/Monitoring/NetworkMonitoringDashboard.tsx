/**
 * Network Monitoring Dashboard
 * Real-time display of network performance metrics, bandwidth usage, and connection quality
 */

import React, { useEffect, useState, useMemo } from 'react';
import { Card, Row, Col, Statistic, Alert, Table, Tag, Progress, Tooltip } from 'antd';
import { 
  SignalIcon, 
  WifiIcon, 
  ExclamationTriangleIcon,
  ArrowUpIcon,
  ArrowDownIcon,
  GlobeAltIcon
} from '@heroicons/react/24/outline';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip as RechartsTooltip, ResponsiveContainer, BarChart, Bar } from 'recharts';
import { SystemResourceMonitor, systemResourceMonitor } from '../../services/monitoring/SystemResourceMonitor';
import { SystemMetrics } from '../../types/monitoring';

interface NetworkMonitoringDashboardProps {
  refreshInterval?: number;
  showConnectionDetails?: boolean;
  compactMode?: boolean;
}

interface NetworkDataPoint {
  timestamp: string;
  time: string;
  bytes_sent_per_sec: number;
  bytes_received_per_sec: number;
  packets_sent_per_sec: number;
  packets_received_per_sec: number;
  errors_per_sec: number;
  total_throughput: number;
}

interface ConnectionInfo {
  key: string;
  endpoint: string;
  status: 'connected' | 'disconnected' | 'degraded';
  latency: number;
  throughput: number;
  error_rate: number;
  last_seen: Date;
}

export const NetworkMonitoringDashboard: React.FC<NetworkMonitoringDashboardProps> = ({
  refreshInterval = 5000,
  showConnectionDetails = true,
  compactMode = false
}) => {
  const [systemMetrics, setSystemMetrics] = useState<SystemMetrics | null>(null);
  const [networkHistory, setNetworkHistory] = useState<NetworkDataPoint[]>([]);
  const [connections, setConnections] = useState<ConnectionInfo[]>([]);
  const [networkAlerts, setNetworkAlerts] = useState<string[]>([]);

  // Generate mock connection data
  const generateConnections = (): ConnectionInfo[] => {
    const endpoints = [
      'IB Gateway (127.0.0.1:7497)',
      'Market Data Feed',
      'WebSocket API',
      'Database Pool',
      'Redis Cache',
      'External API'
    ];

    return endpoints.map((endpoint, index) => ({
      key: index.toString(),
      endpoint,
      status: Math.random() > 0.1 ? 'connected' : (Math.random() > 0.5 ? 'degraded' : 'disconnected'),
      latency: Math.round(Math.random() * 100 + 10), // 10-110ms
      throughput: Math.round(Math.random() * 1024 * 1024), // 0-1MB/s in bytes
      error_rate: Math.round(Math.random() * 5 * 100) / 100, // 0-5%
      last_seen: new Date(Date.now() - Math.random() * 60000) // Within last minute
    }));
  };

  // Convert bytes to readable format
  const formatBytes = (bytes: number, decimals = 2): string => {
    if (bytes === 0) return '0 B';
    const k = 1024;
    const dm = decimals < 0 ? 0 : decimals;
    const sizes = ['B', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(dm)) + ' ' + sizes[i];
  };

  // Format rate (bytes per second)
  const formatRate = (bytesPerSec: number): string => {
    return formatBytes(bytesPerSec) + '/s';
  };

  // Calculate network statistics
  const networkStats = useMemo(() => {
    if (networkHistory.length === 0) {
      return {
        avgThroughput: 0,
        peakThroughput: 0,
        totalErrors: 0,
        avgLatency: 0,
        connectionCount: 0
      };
    }

    const totalThroughputs = networkHistory.map(d => d.total_throughput);
    const totalErrors = networkHistory.reduce((sum, d) => sum + d.errors_per_sec, 0);
    const activeConnections = connections.filter(c => c.status === 'connected');

    return {
      avgThroughput: totalThroughputs.reduce((sum, val) => sum + val, 0) / totalThroughputs.length,
      peakThroughput: Math.max(...totalThroughputs),
      totalErrors: Math.round(totalErrors),
      avgLatency: activeConnections.reduce((sum, conn) => sum + conn.latency, 0) / Math.max(activeConnections.length, 1),
      connectionCount: activeConnections.length
    };
  }, [networkHistory, connections]);

  useEffect(() => {
    const interval = setInterval(() => {
      const metrics = systemResourceMonitor.getSystemMetrics();
      setSystemMetrics(metrics);

      // Create network data point
      const dataPoint: NetworkDataPoint = {
        timestamp: new Date().toISOString(),
        time: new Date().toLocaleTimeString(),
        bytes_sent_per_sec: metrics.network_metrics.bytes_sent_per_sec,
        bytes_received_per_sec: metrics.network_metrics.bytes_received_per_sec,
        packets_sent_per_sec: metrics.network_metrics.packets_sent_per_sec,
        packets_received_per_sec: metrics.network_metrics.packets_received_per_sec,
        errors_per_sec: metrics.network_metrics.errors_per_sec,
        total_throughput: metrics.network_metrics.bytes_sent_per_sec + metrics.network_metrics.bytes_received_per_sec
      };

      setNetworkHistory(prev => [...prev.slice(-49), dataPoint]); // Keep last 50 points
      
      // Update connections
      setConnections(generateConnections());

      // Check for network alerts
      const alerts: string[] = [];
      if (metrics.network_metrics.errors_per_sec > 10) {
        alerts.push(`High error rate: ${metrics.network_metrics.errors_per_sec} errors/sec`);
      }
      if (metrics.network_metrics.bandwidth_utilization_percent > 80) {
        alerts.push(`High bandwidth utilization: ${metrics.network_metrics.bandwidth_utilization_percent}%`);
      }
      if (dataPoint.total_throughput > 50 * 1024 * 1024) { // > 50 MB/s
        alerts.push(`High throughput: ${formatRate(dataPoint.total_throughput)}`);
      }
      setNetworkAlerts(alerts);
    }, refreshInterval);

    return () => clearInterval(interval);
  }, [refreshInterval]);

  const getStatusColor = (status: string): string => {
    switch (status) {
      case 'connected': return 'success';
      case 'degraded': return 'warning';
      case 'disconnected': return 'error';
      default: return 'default';
    }
  };

  const getLatencyColor = (latency: number): string => {
    if (latency > 100) return '#ff4d4f';
    if (latency > 50) return '#fa8c16';
    if (latency > 25) return '#fadb14';
    return '#52c41a';
  };

  const connectionColumns = [
    {
      title: 'Endpoint',
      dataIndex: 'endpoint',
      key: 'endpoint',
      render: (text: string) => <span style={{ fontWeight: 500 }}>{text}</span>
    },
    {
      title: 'Status',
      dataIndex: 'status',
      key: 'status',
      render: (status: string) => (
        <Tag color={getStatusColor(status)}>
          {status.toUpperCase()}
        </Tag>
      )
    },
    {
      title: 'Latency',
      dataIndex: 'latency',
      key: 'latency',
      render: (latency: number) => (
        <span style={{ color: getLatencyColor(latency) }}>
          {latency}ms
        </span>
      ),
      sorter: (a: ConnectionInfo, b: ConnectionInfo) => a.latency - b.latency
    },
    {
      title: 'Throughput',
      dataIndex: 'throughput',
      key: 'throughput',
      render: (throughput: number) => formatRate(throughput),
      sorter: (a: ConnectionInfo, b: ConnectionInfo) => a.throughput - b.throughput
    },
    {
      title: 'Error Rate',
      dataIndex: 'error_rate',
      key: 'error_rate',
      render: (rate: number) => (
        <span style={{ color: rate > 1 ? '#ff4d4f' : '#52c41a' }}>
          {rate.toFixed(2)}%
        </span>
      ),
      sorter: (a: ConnectionInfo, b: ConnectionInfo) => a.error_rate - b.error_rate
    }
  ];

  if (!systemMetrics) {
    return (
      <Card title="Network Monitoring Dashboard" loading>
        <div style={{ textAlign: 'center', padding: '40px 0' }}>
          Loading network metrics...
        </div>
      </Card>
    );
  }

  return (
    <div className="network-monitoring-dashboard">
      {/* Network Alerts */}
      {networkAlerts.length > 0 && (
        <Alert
          type="warning"
          showIcon
          icon={<ExclamationTriangleIcon className="w-4 h-4" />}
          message="Network Performance Alert"
          description={
            <ul style={{ margin: 0, paddingLeft: 20 }}>
              {networkAlerts.map((alert, index) => (
                <li key={index}>{alert}</li>
              ))}
            </ul>
          }
          style={{ marginBottom: 16 }}
          closable
        />
      )}

      <Row gutter={[16, 16]}>
        {/* Network Overview */}
        <Col xs={24} sm={12} lg={6}>
          <Card 
            title={
              <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                <WifiIcon className="w-5 h-5" />
                Network Overview
              </div>
            }
            size={compactMode ? 'small' : 'default'}
          >
            <Statistic
              title="Bandwidth Utilization"
              value={systemMetrics.network_metrics.bandwidth_utilization_percent}
              suffix="%"
              valueStyle={{ 
                color: systemMetrics.network_metrics.bandwidth_utilization_percent > 80 ? '#cf1322' : '#3f8600' 
              }}
            />
            <div style={{ marginTop: 16 }}>
              <Progress 
                percent={systemMetrics.network_metrics.bandwidth_utilization_percent} 
                strokeColor={
                  systemMetrics.network_metrics.bandwidth_utilization_percent > 80 ? '#ff4d4f' :
                  systemMetrics.network_metrics.bandwidth_utilization_percent > 60 ? '#fa8c16' : '#52c41a'
                }
                showInfo={false}
              />
            </div>
            
            <Row gutter={8} style={{ marginTop: 16 }}>
              <Col span={12}>
                <Statistic
                  title="Connections"
                  value={systemMetrics.network_metrics.active_connections}
                  prefix={<SignalIcon className="w-4 h-4" />}
                />
              </Col>
              <Col span={12}>
                <Statistic
                  title="Errors/sec"
                  value={systemMetrics.network_metrics.errors_per_sec}
                  valueStyle={{ 
                    color: systemMetrics.network_metrics.errors_per_sec > 5 ? '#cf1322' : '#3f8600' 
                  }}
                />
              </Col>
            </Row>
          </Card>
        </Col>

        {/* Throughput Stats */}
        <Col xs={24} sm={12} lg={6}>
          <Card 
            title="Throughput"
            size={compactMode ? 'small' : 'default'}
          >
            <div style={{ marginBottom: 16 }}>
              <Tooltip title="Data sent per second">
                <Statistic
                  title={
                    <div style={{ display: 'flex', alignItems: 'center', gap: 4 }}>
                      <ArrowUpIcon className="w-4 h-4" />
                      Upload
                    </div>
                  }
                  value={formatRate(systemMetrics.network_metrics.bytes_sent_per_sec)}
                  valueStyle={{ color: '#1890ff' }}
                />
              </Tooltip>
            </div>
            
            <div style={{ marginBottom: 16 }}>
              <Tooltip title="Data received per second">
                <Statistic
                  title={
                    <div style={{ display: 'flex', alignItems: 'center', gap: 4 }}>
                      <ArrowDownIcon className="w-4 h-4" />
                      Download
                    </div>
                  }
                  value={formatRate(systemMetrics.network_metrics.bytes_received_per_sec)}
                  valueStyle={{ color: '#52c41a' }}
                />
              </Tooltip>
            </div>

            <Statistic
              title="Total"
              value={formatRate(systemMetrics.network_metrics.bytes_sent_per_sec + systemMetrics.network_metrics.bytes_received_per_sec)}
              valueStyle={{ color: '#722ed1' }}
            />
          </Card>
        </Col>

        {/* Packet Stats */}
        <Col xs={24} sm={12} lg={6}>
          <Card 
            title="Packet Statistics"
            size={compactMode ? 'small' : 'default'}
          >
            <Row gutter={8}>
              <Col span={12}>
                <Statistic
                  title="Sent/sec"
                  value={systemMetrics.network_metrics.packets_sent_per_sec}
                  valueStyle={{ color: '#1890ff' }}
                />
              </Col>
              <Col span={12}>
                <Statistic
                  title="Received/sec"
                  value={systemMetrics.network_metrics.packets_received_per_sec}
                  valueStyle={{ color: '#52c41a' }}
                />
              </Col>
            </Row>

            <div style={{ marginTop: 16 }}>
              <Statistic
                title="Total Packets/sec"
                value={systemMetrics.network_metrics.packets_sent_per_sec + systemMetrics.network_metrics.packets_received_per_sec}
              />
            </div>

            <div style={{ marginTop: 16 }}>
              <div>Packet Loss Rate</div>
              <Progress 
                percent={Math.min(100, (systemMetrics.network_metrics.errors_per_sec / Math.max(1, systemMetrics.network_metrics.packets_received_per_sec)) * 100)}
                strokeColor="#ff4d4f"
                showInfo={true}
                format={(percent) => `${percent?.toFixed(2)}%`}
              />
            </div>
          </Card>
        </Col>

        {/* Performance Summary */}
        <Col xs={24} sm={12} lg={6}>
          <Card 
            title="Performance Summary"
            size={compactMode ? 'small' : 'default'}
          >
            <Statistic
              title="Avg Throughput"
              value={formatRate(networkStats.avgThroughput)}
              prefix={<GlobeAltIcon className="w-4 h-4" />}
              valueStyle={{ marginBottom: 16 }}
            />
            
            <Statistic
              title="Peak Throughput"
              value={formatRate(networkStats.peakThroughput)}
              valueStyle={{ marginBottom: 16 }}
            />
            
            <Statistic
              title="Active Connections"
              value={networkStats.connectionCount}
              suffix={`/ ${connections.length}`}
              valueStyle={{ 
                color: networkStats.connectionCount < connections.length ? '#fa8c16' : '#52c41a' 
              }}
            />
          </Card>
        </Col>

        {/* Network Throughput Chart */}
        <Col xs={24} lg={12}>
          <Card title="Network Throughput" size={compactMode ? 'small' : 'default'}>
            <ResponsiveContainer width="100%" height={compactMode ? 200 : 300}>
              <LineChart data={networkHistory}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis 
                  dataKey="time"
                  tick={{ fontSize: 12 }}
                />
                <YAxis 
                  tick={{ fontSize: 12 }}
                  tickFormatter={(value) => formatBytes(value)}
                />
                <RechartsTooltip 
                  formatter={(value: number, name: string) => [
                    formatRate(value),
                    name === 'bytes_sent_per_sec' ? 'Upload' : 'Download'
                  ]}
                  labelFormatter={(label) => `Time: ${label}`}
                />
                <Line
                  type="monotone"
                  dataKey="bytes_sent_per_sec"
                  stroke="#1890ff"
                  strokeWidth={2}
                  dot={false}
                  name="Upload"
                />
                <Line
                  type="monotone"
                  dataKey="bytes_received_per_sec"
                  stroke="#52c41a"
                  strokeWidth={2}
                  dot={false}
                  name="Download"
                />
              </LineChart>
            </ResponsiveContainer>
          </Card>
        </Col>

        {/* Error Rate Chart */}
        <Col xs={24} lg={12}>
          <Card title="Error Rate & Packets" size={compactMode ? 'small' : 'default'}>
            <ResponsiveContainer width="100%" height={compactMode ? 200 : 300}>
              <BarChart data={networkHistory.slice(-20)}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis 
                  dataKey="time"
                  tick={{ fontSize: 12 }}
                />
                <YAxis tick={{ fontSize: 12 }} />
                <RechartsTooltip 
                  formatter={(value: number, name: string) => [
                    value,
                    name === 'errors_per_sec' ? 'Errors/sec' : 'Packets/sec'
                  ]}
                />
                <Bar
                  dataKey="errors_per_sec"
                  fill="#ff4d4f"
                  name="Errors/sec"
                />
                <Bar
                  dataKey="packets_received_per_sec"
                  fill="#52c41a"
                  name="Packets/sec"
                />
              </BarChart>
            </ResponsiveContainer>
          </Card>
        </Col>

        {/* Connection Details Table */}
        {showConnectionDetails && (
          <Col xs={24}>
            <Card 
              title="Connection Details"
              size={compactMode ? 'small' : 'default'}
            >
              <Table
                columns={connectionColumns}
                dataSource={connections}
                size={compactMode ? 'small' : 'default'}
                pagination={false}
                scroll={{ x: 800 }}
              />
            </Card>
          </Col>
        )}
      </Row>
    </div>
  );
};

export default NetworkMonitoringDashboard;