/**
 * CPU Memory Dashboard
 * Real-time display of CPU and memory usage metrics with charts and alerts
 */

import React, { useEffect, useState, useMemo } from 'react';
import { Card, Row, Col, Progress, Statistic, Alert, Button, Switch } from 'antd';
import { 
  ProcessorIcon, 
  ChipIcon, 
  ExclamationTriangleIcon,
  ArrowTrendingUpIcon,
  ArrowTrendingDownIcon 
} from '@heroicons/react/24/outline';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, AreaChart, Area } from 'recharts';
import { SystemResourceMonitor, SystemResourceSnapshot, systemResourceMonitor } from '../../services/monitoring/SystemResourceMonitor';
import { SystemMetrics } from '../../types/monitoring';

interface CPUMemoryDashboardProps {
  refreshInterval?: number;
  showAlerts?: boolean;
  compactMode?: boolean;
}

interface ChartDataPoint {
  timestamp: string;
  time: string;
  cpu_usage: number;
  memory_usage: number;
  memory_percent: number;
  heap_used: number;
  available_memory: number;
}

export const CPUMemoryDashboard: React.FC<CPUMemoryDashboardProps> = ({
  refreshInterval = 5000,
  showAlerts = true,
  compactMode = false
}) => {
  const [systemMetrics, setSystemMetrics] = useState<SystemMetrics | null>(null);
  const [resourceSnapshots, setResourceSnapshots] = useState<SystemResourceSnapshot[]>([]);
  const [isMonitoring, setIsMonitoring] = useState(false);
  const [anomalies, setAnomalies] = useState<{
    cpu_anomaly: boolean;
    memory_anomaly: boolean;
    network_anomaly: boolean;
    details: string[];
  }>({ cpu_anomaly: false, memory_anomaly: false, network_anomaly: false, details: [] });

  // Chart data preparation
  const chartData = useMemo<ChartDataPoint[]>(() => {
    return resourceSnapshots.slice(-50).map(snapshot => ({
      timestamp: snapshot.timestamp.toISOString(),
      time: snapshot.timestamp.toLocaleTimeString(),
      cpu_usage: Math.round(snapshot.cpu_usage_percent * 100) / 100,
      memory_usage: Math.round(snapshot.memory_usage_mb * 100) / 100,
      memory_percent: Math.round(((snapshot.memory_usage_mb / (snapshot.memory_usage_mb + snapshot.available_memory_mb)) * 100) * 100) / 100,
      heap_used: Math.round(snapshot.heap_used_mb * 100) / 100,
      available_memory: Math.round(snapshot.available_memory_mb * 100) / 100
    }));
  }, [resourceSnapshots]);

  // Statistics calculations
  const resourceStats = useMemo(() => {
    if (resourceSnapshots.length === 0) {
      return { cpu: { min: 0, max: 0, avg: 0 }, memory: { min: 0, max: 0, avg: 0 }, samples: 0 };
    }
    
    return systemResourceMonitor.getResourceStatistics(5 * 60 * 1000); // Last 5 minutes
  }, [resourceSnapshots]);

  // Health score
  const healthScore = useMemo(() => {
    return systemResourceMonitor.getHealthScore();
  }, [resourceSnapshots]);

  useEffect(() => {
    const handleSnapshot = (snapshot: SystemResourceSnapshot) => {
      setResourceSnapshots(prev => [...prev.slice(-99), snapshot]); // Keep last 100 snapshots
      setSystemMetrics(systemResourceMonitor.getSystemMetrics());
      
      // Check for anomalies
      if (showAlerts) {
        const anomalyCheck = systemResourceMonitor.detectResourceAnomalies();
        setAnomalies(anomalyCheck);
      }
    };

    // Add callback
    systemResourceMonitor.onSnapshot(handleSnapshot);

    // Start monitoring if not already started
    if (isMonitoring) {
      systemResourceMonitor.start(refreshInterval);
    }

    return () => {
      systemResourceMonitor.removeCallback(handleSnapshot);
    };
  }, [refreshInterval, showAlerts, isMonitoring]);

  const toggleMonitoring = () => {
    if (isMonitoring) {
      systemResourceMonitor.stop();
      setIsMonitoring(false);
    } else {
      systemResourceMonitor.start(refreshInterval);
      setIsMonitoring(true);
    }
  };

  const getCPUStatusColor = (usage: number): string => {
    if (usage > 80) return '#ff4d4f';
    if (usage > 60) return '#fa8c16';
    if (usage > 40) return '#fadb14';
    return '#52c41a';
  };

  const getMemoryStatusColor = (usage: number): string => {
    if (usage > 90) return '#ff4d4f';
    if (usage > 75) return '#fa8c16';
    if (usage > 50) return '#fadb14';
    return '#52c41a';
  };

  const formatBytes = (bytes: number): string => {
    if (bytes === 0) return '0 B';
    const k = 1024;
    const sizes = ['B', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes * 1024 * 1024) / Math.log(k));
    return parseFloat(((bytes * 1024 * 1024) / Math.pow(k, i)).toFixed(1)) + ' ' + sizes[i];
  };

  if (!systemMetrics) {
    return (
      <Card title="CPU & Memory Dashboard" loading>
        <div style={{ textAlign: 'center', padding: '40px 0' }}>
          Loading system metrics...
        </div>
      </Card>
    );
  }

  return (
    <div className="cpu-memory-dashboard">
      {/* Control Panel */}
      <Card size="small" style={{ marginBottom: 16 }}>
        <Row align="middle" justify="space-between">
          <Col>
            <Switch 
              checked={isMonitoring}
              onChange={toggleMonitoring}
              checkedChildren="Monitoring ON"
              unCheckedChildren="Monitoring OFF"
            />
          </Col>
          <Col>
            <Statistic
              title="Health Score"
              value={healthScore}
              suffix="%"
              valueStyle={{ 
                color: healthScore > 80 ? '#3f8600' : healthScore > 60 ? '#fa8c16' : '#cf1322' 
              }}
            />
          </Col>
        </Row>
      </Card>

      {/* Alerts */}
      {showAlerts && (anomalies.cpu_anomaly || anomalies.memory_anomaly) && (
        <Alert
          type="warning"
          showIcon
          icon={<ExclamationTriangleIcon className="w-4 h-4" />}
          message="Resource Usage Anomaly Detected"
          description={
            <ul>
              {anomalies.details.map((detail, index) => (
                <li key={index}>{detail}</li>
              ))}
            </ul>
          }
          style={{ marginBottom: 16 }}
        />
      )}

      <Row gutter={[16, 16]}>
        {/* CPU Metrics */}
        <Col xs={24} sm={12} lg={8}>
          <Card 
            title={
              <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                <ProcessorIcon className="w-5 h-5" />
                CPU Usage
              </div>
            }
            size={compactMode ? 'small' : 'default'}
          >
            <div style={{ textAlign: 'center', marginBottom: 16 }}>
              <Progress
                type="circle"
                percent={systemMetrics.cpu_metrics.usage_percent}
                strokeColor={getCPUStatusColor(systemMetrics.cpu_metrics.usage_percent)}
                width={compactMode ? 80 : 120}
                format={(percent) => `${percent?.toFixed(1)}%`}
              />
            </div>
            
            <Row gutter={16}>
              <Col span={12}>
                <Statistic
                  title="Cores"
                  value={systemMetrics.cpu_metrics.core_count}
                  prefix={<ChipIcon className="w-4 h-4" />}
                />
              </Col>
              <Col span={12}>
                <Statistic
                  title="Load Avg"
                  value={systemMetrics.cpu_metrics.load_average_1m}
                  precision={2}
                />
              </Col>
            </Row>

            {!compactMode && (
              <div style={{ marginTop: 16 }}>
                <div style={{ marginBottom: 8 }}>Per-Core Usage:</div>
                {systemMetrics.cpu_metrics.per_core_usage.map((usage, index) => (
                  <div key={index} style={{ marginBottom: 4 }}>
                    <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                      <span>Core {index + 1}</span>
                      <span>{usage.toFixed(1)}%</span>
                    </div>
                    <Progress 
                      percent={usage} 
                      strokeColor={getCPUStatusColor(usage)}
                      showInfo={false}
                      size="small"
                    />
                  </div>
                ))}
              </div>
            )}
          </Card>
        </Col>

        {/* Memory Metrics */}
        <Col xs={24} sm={12} lg={8}>
          <Card 
            title={
              <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                <ChipIcon className="w-5 h-5" />
                Memory Usage
              </div>
            }
            size={compactMode ? 'small' : 'default'}
          >
            <div style={{ textAlign: 'center', marginBottom: 16 }}>
              <Progress
                type="circle"
                percent={systemMetrics.memory_metrics.usage_percent}
                strokeColor={getMemoryStatusColor(systemMetrics.memory_metrics.usage_percent)}
                width={compactMode ? 80 : 120}
                format={(percent) => `${percent?.toFixed(1)}%`}
              />
            </div>
            
            <Row gutter={16}>
              <Col span={12}>
                <Statistic
                  title="Used"
                  value={systemMetrics.memory_metrics.used_gb}
                  suffix="GB"
                  precision={2}
                />
              </Col>
              <Col span={12}>
                <Statistic
                  title="Total"
                  value={systemMetrics.memory_metrics.total_gb}
                  suffix="GB"
                  precision={2}
                />
              </Col>
            </Row>

            {!compactMode && (
              <div style={{ marginTop: 16 }}>
                <div style={{ marginBottom: 8 }}>Memory Breakdown:</div>
                <div style={{ marginBottom: 4 }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                    <span>Available</span>
                    <span>{systemMetrics.memory_metrics.available_gb.toFixed(2)} GB</span>
                  </div>
                  <Progress 
                    percent={(systemMetrics.memory_metrics.available_gb / systemMetrics.memory_metrics.total_gb) * 100}
                    strokeColor="#52c41a"
                    showInfo={false}
                    size="small"
                  />
                </div>
                <div style={{ marginBottom: 4 }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                    <span>Buffer/Cache</span>
                    <span>{systemMetrics.memory_metrics.buffer_cache_gb.toFixed(2)} GB</span>
                  </div>
                  <Progress 
                    percent={(systemMetrics.memory_metrics.buffer_cache_gb / systemMetrics.memory_metrics.total_gb) * 100}
                    strokeColor="#1890ff"
                    showInfo={false}
                    size="small"
                  />
                </div>
              </div>
            )}
          </Card>
        </Col>

        {/* Statistics */}
        <Col xs={24} sm={24} lg={8}>
          <Card 
            title="Performance Statistics"
            size={compactMode ? 'small' : 'default'}
          >
            <Row gutter={16}>
              <Col span={12}>
                <Statistic
                  title="CPU Avg"
                  value={resourceStats.cpu.avg}
                  precision={1}
                  suffix="%"
                  prefix={
                    resourceStats.cpu.avg > (resourceStats.cpu.avg + resourceStats.memory.avg) / 2 ? 
                      <ArrowTrendingUpIcon className="w-4 h-4 text-red-500" /> :
                      <ArrowTrendingDownIcon className="w-4 h-4 text-green-500" />
                  }
                />
              </Col>
              <Col span={12}>
                <Statistic
                  title="Memory Avg"
                  value={resourceStats.memory.avg}
                  precision={0}
                  suffix="MB"
                />
              </Col>
            </Row>

            <Row gutter={16} style={{ marginTop: 16 }}>
              <Col span={12}>
                <Statistic
                  title="CPU Peak"
                  value={resourceStats.cpu.max}
                  precision={1}
                  suffix="%"
                  valueStyle={{ color: getCPUStatusColor(resourceStats.cpu.max) }}
                />
              </Col>
              <Col span={12}>
                <Statistic
                  title="Memory Peak"
                  value={resourceStats.memory.max}
                  precision={0}
                  suffix="MB"
                  valueStyle={{ color: getMemoryStatusColor((resourceStats.memory.max / 1024) * 100) }}
                />
              </Col>
            </Row>

            <div style={{ marginTop: 16 }}>
              <Statistic
                title="Samples"
                value={resourceStats.samples}
                suffix="measurements"
              />
            </div>
          </Card>
        </Col>

        {/* CPU Usage Chart */}
        <Col xs={24} lg={12}>
          <Card title="CPU Usage Trend" size={compactMode ? 'small' : 'default'}>
            <ResponsiveContainer width="100%" height={compactMode ? 200 : 300}>
              <LineChart data={chartData}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis 
                  dataKey="time"
                  tick={{ fontSize: 12 }}
                />
                <YAxis 
                  domain={[0, 100]}
                  tick={{ fontSize: 12 }}
                />
                <Tooltip 
                  formatter={(value: number) => [`${value.toFixed(2)}%`, 'CPU Usage']}
                  labelFormatter={(label) => `Time: ${label}`}
                />
                <Line
                  type="monotone"
                  dataKey="cpu_usage"
                  stroke="#1890ff"
                  strokeWidth={2}
                  dot={false}
                  activeDot={{ r: 4 }}
                />
              </LineChart>
            </ResponsiveContainer>
          </Card>
        </Col>

        {/* Memory Usage Chart */}
        <Col xs={24} lg={12}>
          <Card title="Memory Usage Trend" size={compactMode ? 'small' : 'default'}>
            <ResponsiveContainer width="100%" height={compactMode ? 200 : 300}>
              <AreaChart data={chartData}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis 
                  dataKey="time"
                  tick={{ fontSize: 12 }}
                />
                <YAxis 
                  tick={{ fontSize: 12 }}
                />
                <Tooltip 
                  formatter={(value: number, name: string) => [
                    name === 'memory_usage' ? `${value.toFixed(0)} MB` : `${value.toFixed(0)} MB`,
                    name === 'memory_usage' ? 'Used Memory' : 'Available Memory'
                  ]}
                  labelFormatter={(label) => `Time: ${label}`}
                />
                <Area
                  type="monotone"
                  dataKey="memory_usage"
                  stackId="1"
                  stroke="#ff7300"
                  fill="#ff7300"
                  fillOpacity={0.6}
                />
                <Area
                  type="monotone"
                  dataKey="available_memory"
                  stackId="1"
                  stroke="#52c41a"
                  fill="#52c41a"
                  fillOpacity={0.4}
                />
              </AreaChart>
            </ResponsiveContainer>
          </Card>
        </Col>
      </Row>
    </div>
  );
};

export default CPUMemoryDashboard;