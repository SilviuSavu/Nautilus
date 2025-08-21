/**
 * ResourceMonitor - System resource usage monitoring for NautilusTrader engine
 * 
 * Displays real-time CPU, memory, network, and container health metrics
 * with visual indicators and alerts for resource thresholds.
 */

import React, { useState, useEffect } from 'react';
import { Card, Row, Col, Progress, Statistic, Alert, Badge, Space, Typography, Tooltip } from 'antd';
import {
  DashboardOutlined,
  MemoryOutlined,
  GlobalOutlined,
  HddOutlined,
  CheckCircleOutlined,
  ExclamationCircleOutlined,
  CloseCircleOutlined,
  InfoCircleOutlined
} from '@ant-design/icons';

const { Text } = Typography;

interface ResourceUsage {
  cpu_percent?: string;
  memory_usage?: string;
  memory_percent?: string;
  network_io?: string;
  block_io?: string;
  error?: string;
}

interface ContainerInfo {
  status?: string;
  running?: boolean;
  started_at?: string;
  image?: string;
  platform?: string;
  error?: string;
}

interface HealthCheck {
  status?: 'healthy' | 'unhealthy' | 'error';
  last_check?: string;
  details?: string;
}

interface ResourceMonitorProps {
  resourceUsage?: ResourceUsage;
  containerInfo?: ContainerInfo;
  healthCheck?: HealthCheck;
  refreshInterval?: number;
}

interface ParsedMetrics {
  cpu: number;
  memoryUsed: string;
  memoryTotal: string;
  memoryPercent: number;
  networkRx: string;
  networkTx: string;
  blockRead: string;
  blockWrite: string;
}

export const ResourceMonitor: React.FC<ResourceMonitorProps> = ({
  resourceUsage,
  containerInfo,
  healthCheck,
  refreshInterval = 5000
}) => {
  const [lastUpdate, setLastUpdate] = useState<Date>(new Date());
  const [alerts, setAlerts] = useState<string[]>([]);

  useEffect(() => {
    setLastUpdate(new Date());
    checkResourceAlerts();
  }, [resourceUsage]);

  const parseResourceMetrics = (): ParsedMetrics | null => {
    if (!resourceUsage || resourceUsage.error) {
      return null;
    }

    try {
      const cpu = parseFloat(resourceUsage.cpu_percent?.replace('%', '') || '0');
      const memoryPercent = parseFloat(resourceUsage.memory_percent?.replace('%', '') || '0');
      
      // Parse memory usage (format: "used / total")
      const memoryParts = resourceUsage.memory_usage?.split(' / ') || ['0B', '0B'];
      const memoryUsed = memoryParts[0] || '0B';
      const memoryTotal = memoryParts[1] || '0B';
      
      // Parse network I/O (format: "rx / tx")
      const networkParts = resourceUsage.network_io?.split(' / ') || ['0B', '0B'];
      const networkRx = networkParts[0] || '0B';
      const networkTx = networkParts[1] || '0B';
      
      // Parse block I/O (format: "read / write")
      const blockParts = resourceUsage.block_io?.split(' / ') || ['0B', '0B'];
      const blockRead = blockParts[0] || '0B';
      const blockWrite = blockParts[1] || '0B';

      return {
        cpu,
        memoryUsed,
        memoryTotal,
        memoryPercent,
        networkRx,
        networkTx,
        blockRead,
        blockWrite
      };
    } catch (error) {
      console.error('Error parsing resource metrics:', error);
      return null;
    }
  };

  const checkResourceAlerts = () => {
    const metrics = parseResourceMetrics();
    if (!metrics) return;

    const newAlerts: string[] = [];

    // CPU usage alerts
    if (metrics.cpu > 90) {
      newAlerts.push('High CPU usage detected (>90%)');
    } else if (metrics.cpu > 75) {
      newAlerts.push('Elevated CPU usage (>75%)');
    }

    // Memory usage alerts
    if (metrics.memoryPercent > 90) {
      newAlerts.push('High memory usage detected (>90%)');
    } else if (metrics.memoryPercent > 75) {
      newAlerts.push('Elevated memory usage (>75%)');
    }

    setAlerts(newAlerts);
  };

  const getProgressColor = (value: number): string => {
    if (value >= 90) return '#ff4d4f';
    if (value >= 75) return '#faad14';
    if (value >= 50) return '#1890ff';
    return '#52c41a';
  };

  const getHealthStatusBadge = () => {
    if (!healthCheck || !healthCheck.status) {
      return <Badge status="default" text="Unknown" />;
    }

    switch (healthCheck.status) {
      case 'healthy':
        return <Badge status="success" text="Healthy" />;
      case 'unhealthy':
        return <Badge status="warning" text="Unhealthy" />;
      case 'error':
        return <Badge status="error" text="Error" />;
      default:
        return <Badge status="default" text="Unknown" />;
    }
  };

  const getContainerStatusIcon = () => {
    if (!containerInfo) {
      return <InfoCircleOutlined style={{ color: '#d9d9d9' }} />;
    }

    if (containerInfo.running) {
      return <CheckCircleOutlined style={{ color: '#52c41a' }} />;
    } else {
      return <CloseCircleOutlined style={{ color: '#ff4d4f' }} />;
    }
  };

  const formatLastUpdate = (): string => {
    return lastUpdate.toLocaleTimeString();
  };

  const metrics = parseResourceMetrics();

  return (
    <div className="resource-monitor">
      <Space direction="vertical" style={{ width: '100%' }}>
        {/* Alerts */}
        {alerts.length > 0 && (
          <Alert
            type="warning"
            message="Resource Usage Alert"
            description={
              <ul style={{ margin: 0, paddingLeft: 20 }}>
                {alerts.map((alert, index) => (
                  <li key={index}>{alert}</li>
                ))}
              </ul>
            }
            showIcon
            closable
          />
        )}

        {/* Error State */}
        {resourceUsage?.error && (
          <Alert
            type="error"
            message="Resource Monitoring Error"
            description={resourceUsage.error}
            showIcon
          />
        )}

        {/* Resource Metrics */}
        {metrics && (
          <Row gutter={[12, 12]}>
            {/* CPU Usage */}
            <Col xs={24} sm={12} lg={6}>
              <Card size="small" bodyStyle={{ padding: '12px' }}>
                <Statistic
                  title={
                    <Space>
                      <DashboardOutlined />
                      <span>CPU Usage</span>
                    </Space>
                  }
                  value={metrics.cpu}
                  suffix="%"
                  precision={1}
                  valueStyle={{ color: getProgressColor(metrics.cpu) }}
                />
                <Progress
                  percent={metrics.cpu}
                  strokeColor={getProgressColor(metrics.cpu)}
                  showInfo={false}
                  size="small"
                />
              </Card>
            </Col>

            {/* Memory Usage */}
            <Col xs={24} sm={12} lg={6}>
              <Card size="small" bodyStyle={{ padding: '12px' }}>
                <Statistic
                  title={
                    <Space>
                      <MemoryOutlined />
                      <span>Memory</span>
                    </Space>
                  }
                  value={metrics.memoryPercent}
                  suffix="%"
                  precision={1}
                  valueStyle={{ color: getProgressColor(metrics.memoryPercent) }}
                />
                <Progress
                  percent={metrics.memoryPercent}
                  strokeColor={getProgressColor(metrics.memoryPercent)}
                  showInfo={false}
                  size="small"
                />
                <Text type="secondary" style={{ fontSize: '11px' }}>
                  {metrics.memoryUsed} / {metrics.memoryTotal}
                </Text>
              </Card>
            </Col>

            {/* Network I/O */}
            <Col xs={24} sm={12} lg={6}>
              <Card size="small" bodyStyle={{ padding: '12px' }}>
                <div style={{ textAlign: 'center' }}>
                  <Space direction="vertical" size="small">
                    <Space>
                      <GlobalOutlined />
                      <Text strong style={{ fontSize: '12px' }}>Network I/O</Text>
                    </Space>
                    <div>
                      <Text type="secondary" style={{ fontSize: '11px' }}>
                        ↓ {metrics.networkRx}
                      </Text>
                      <br />
                      <Text type="secondary" style={{ fontSize: '11px' }}>
                        ↑ {metrics.networkTx}
                      </Text>
                    </div>
                  </Space>
                </div>
              </Card>
            </Col>

            {/* Disk I/O */}
            <Col xs={24} sm={12} lg={6}>
              <Card size="small" bodyStyle={{ padding: '12px' }}>
                <div style={{ textAlign: 'center' }}>
                  <Space direction="vertical" size="small">
                    <Space>
                      <HddOutlined />
                      <Text strong style={{ fontSize: '12px' }}>Disk I/O</Text>
                    </Space>
                    <div>
                      <Text type="secondary" style={{ fontSize: '11px' }}>
                        Read: {metrics.blockRead}
                      </Text>
                      <br />
                      <Text type="secondary" style={{ fontSize: '11px' }}>
                        Write: {metrics.blockWrite}
                      </Text>
                    </div>
                  </Space>
                </div>
              </Card>
            </Col>
          </Row>
        )}

        {/* Container & Health Status */}
        <Row gutter={[12, 12]}>
          <Col xs={24} lg={12}>
            <Card 
              title="Container Status" 
              size="small"
              bodyStyle={{ padding: '12px' }}
            >
              <Space direction="vertical" style={{ width: '100%' }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                  <Space>
                    {getContainerStatusIcon()}
                    <Text strong>
                      {containerInfo?.running ? 'Running' : 'Stopped'}
                    </Text>
                  </Space>
                  <Text type="secondary" style={{ fontSize: '11px' }}>
                    Status: {containerInfo?.status || 'Unknown'}
                  </Text>
                </div>
                
                {containerInfo?.started_at && (
                  <Text type="secondary" style={{ fontSize: '11px' }}>
                    Started: {new Date(containerInfo.started_at).toLocaleString()}
                  </Text>
                )}
                
                {containerInfo?.image && (
                  <Text type="secondary" style={{ fontSize: '11px' }}>
                    Image: {containerInfo.image}
                  </Text>
                )}
              </Space>
            </Card>
          </Col>

          <Col xs={24} lg={12}>
            <Card 
              title="Health Check" 
              size="small"
              bodyStyle={{ padding: '12px' }}
            >
              <Space direction="vertical" style={{ width: '100%' }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                  {getHealthStatusBadge()}
                  <Text type="secondary" style={{ fontSize: '11px' }}>
                    Last Check: {healthCheck?.last_check ? 
                      new Date(healthCheck.last_check).toLocaleTimeString() : 
                      'N/A'
                    }
                  </Text>
                </div>
                
                {healthCheck?.details && (
                  <Text type="secondary" style={{ fontSize: '11px' }}>
                    {healthCheck.details}
                  </Text>
                )}
              </Space>
            </Card>
          </Col>
        </Row>

        {/* Update Info */}
        <div style={{ textAlign: 'center' }}>
          <Text type="secondary" style={{ fontSize: '11px' }}>
            Last updated: {formatLastUpdate()} • Refresh interval: {refreshInterval / 1000}s
          </Text>
        </div>
      </Space>

      <style jsx>{`
        .resource-monitor {
          width: 100%;
        }

        .ant-card-head-title {
          font-size: 14px;
          font-weight: 600;
        }

        .ant-statistic-title {
          font-size: 12px;
          margin-bottom: 4px;
        }

        .ant-statistic-content {
          font-size: 16px;
        }

        .ant-progress-line {
          margin-top: 4px;
        }

        .resource-alert {
          border-left: 4px solid #faad14;
          background: #fffbe6;
        }
      `}</style>
    </div>
  );
};

export default ResourceMonitor;