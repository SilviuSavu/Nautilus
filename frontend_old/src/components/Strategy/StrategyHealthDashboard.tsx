import React, { useState, useEffect, useRef } from 'react';
import {
  Card,
  Row,
  Col,
  Alert,
  Badge,
  Table,
  Typography,
  Space,
  Button,
  Tooltip,
  Progress,
  Statistic,
  Timeline,
  Tag,
  Select,
  DatePicker,
  Modal,
  Descriptions,
  notification
} from 'antd';
import {
  HeartOutlined,
  WarningOutlined,
  CheckCircleOutlined,
  ExclamationCircleOutlined,
  StopOutlined,
  ReloadOutlined,
  BellOutlined,
  LineChartOutlined,
  MonitorOutlined,
  AlertOutlined,
  ClockCircleOutlined,
  DatabaseOutlined,
  LinkOutlined,
  DashboardOutlined
} from '@ant-design/icons';
import type {
  LiveStrategy,
  StrategyAlert,
  HealthStatus,
  AlertType
} from '../../types/deployment';

const { Text, Title } = Typography;
const { RangePicker } = DatePicker;

interface HealthDashboardProps {
  strategies?: LiveStrategy[];
  refreshInterval?: number;
  showDetails?: boolean;
}

interface SystemHealth {
  overall: 'healthy' | 'warning' | 'critical';
  activeStrategies: number;
  totalAlerts: number;
  criticalAlerts: number;
  systemUptime: number;
  dataConnections: {
    name: string;
    status: 'connected' | 'degraded' | 'disconnected';
    latency?: number;
  }[];
  lastUpdate: Date;
}

const StrategyHealthDashboard: React.FC<HealthDashboardProps> = ({
  strategies = [],
  refreshInterval = 10000,
  showDetails = true
}) => {
  const [systemHealth, setSystemHealth] = useState<SystemHealth | null>(null);
  const [allAlerts, setAllAlerts] = useState<StrategyAlert[]>([]);
  const [loading, setLoading] = useState(true);
  const [selectedStrategy, setSelectedStrategy] = useState<string>('all');
  const [alertFilter, setAlertFilter] = useState<'all' | 'critical' | 'unacknowledged'>('unacknowledged');
  const [alertDetailsVisible, setAlertDetailsVisible] = useState(false);
  const [selectedAlert, setSelectedAlert] = useState<StrategyAlert | null>(null);
  const intervalRef = useRef<NodeJS.Timeout>();

  useEffect(() => {
    loadHealthData();
    startRealTimeUpdates();

    return () => {
      if (intervalRef.current) {
        clearInterval(intervalRef.current);
      }
    };
  }, [refreshInterval]);

  const loadHealthData = async () => {
    try {
      setLoading(true);
      
      // Load system health
      const healthResponse = await fetch('/api/v1/nautilus/monitoring/system-health');
      const healthData = await healthResponse.json();
      setSystemHealth(healthData);

      // Load all alerts
      const alertsResponse = await fetch('/api/v1/nautilus/monitoring/alerts');
      const alertsData = await alertsResponse.json();
      setAllAlerts(alertsData);

    } catch (error) {
      console.error('Error loading health data:', error);
      notification.error({
        message: 'Health Data Error',
        description: 'Failed to load system health data'
      });
    } finally {
      setLoading(false);
    }
  };

  const startRealTimeUpdates = () => {
    if (intervalRef.current) {
      clearInterval(intervalRef.current);
    }

    intervalRef.current = setInterval(() => {
      loadHealthData();
    }, refreshInterval);
  };

  const acknowledgeAlert = async (alertId: string) => {
    try {
      await fetch(`/api/v1/nautilus/monitoring/alerts/${alertId}/acknowledge`, {
        method: 'POST'
      });
      
      setAllAlerts(prev => prev.map(alert => 
        alert.alertId === alertId 
          ? { ...alert, acknowledged: true, acknowledgedBy: 'current_user' }
          : alert
      ));
      
      message.success('Alert acknowledged');
    } catch (error) {
      console.error('Error acknowledging alert:', error);
      message.error('Failed to acknowledge alert');
    }
  };

  const getHealthColor = (status: string): string => {
    switch (status) {
      case 'healthy': return 'green';
      case 'warning': return 'orange';
      case 'critical': return 'red';
      case 'connected': return 'green';
      case 'degraded': return 'orange';
      case 'disconnected': return 'red';
      default: return 'gray';
    }
  };

  const getSeverityColor = (severity: string): string => {
    switch (severity) {
      case 'critical': return 'red';
      case 'error': return 'orange';
      case 'warning': return 'yellow';
      case 'info': return 'blue';
      default: return 'gray';
    }
  };

  const getFilteredAlerts = () => {
    let filtered = allAlerts;

    if (selectedStrategy !== 'all') {
      filtered = filtered.filter(alert => alert.strategyInstanceId === selectedStrategy);
    }

    switch (alertFilter) {
      case 'critical':
        filtered = filtered.filter(alert => alert.severity === 'critical');
        break;
      case 'unacknowledged':
        filtered = filtered.filter(alert => !alert.acknowledged);
        break;
    }

    return filtered.sort((a, b) => new Date(b.timestamp).getTime() - new Date(a.timestamp).getTime());
  };

  const renderSystemOverview = () => (
    <Row gutter={16} className="mb-6">
      <Col span={6}>
        <Card size="small" className="text-center">
          <Badge status={getHealthColor(systemHealth?.overall || 'unknown') as any} />
          <div className="mt-2">
            <Text strong>System Health</Text>
            <div className="text-lg">
              {systemHealth?.overall?.toUpperCase() || 'UNKNOWN'}
            </div>
          </div>
        </Card>
      </Col>
      <Col span={6}>
        <Card size="small" className="text-center">
          <Statistic
            title="Active Strategies"
            value={systemHealth?.activeStrategies || 0}
            prefix={<MonitorOutlined />}
            valueStyle={{ color: '#1890ff' }}
          />
        </Card>
      </Col>
      <Col span={6}>
        <Card size="small" className="text-center">
          <Statistic
            title="Total Alerts"
            value={systemHealth?.totalAlerts || 0}
            prefix={<BellOutlined />}
            valueStyle={{ color: '#faad14' }}
          />
        </Card>
      </Col>
      <Col span={6}>
        <Card size="small" className="text-center">
          <Statistic
            title="Critical Alerts"
            value={systemHealth?.criticalAlerts || 0}
            prefix={<ExclamationCircleOutlined />}
            valueStyle={{ color: '#cf1322' }}
          />
        </Card>
      </Col>
    </Row>
  );

  const renderDataConnections = () => (
    <Card title="Data Connections" size="small" className="mb-4">
      <Row gutter={16}>
        {systemHealth?.dataConnections.map((connection, index) => (
          <Col span={8} key={index}>
            <div className="text-center p-3 border rounded">
              <Badge status={getHealthColor(connection.status) as any} />
              <div className="mt-2">
                <Text strong>{connection.name}</Text>
                <div>{connection.status.toUpperCase()}</div>
                {connection.latency && (
                  <Text type="secondary" style={{ fontSize: '12px' }}>
                    {connection.latency}ms
                  </Text>
                )}
              </div>
            </div>
          </Col>
        ))}
      </Row>
    </Card>
  );

  const renderStrategyHealth = () => (
    <Card title="Strategy Health Status" size="small" className="mb-4">
      <Table
        dataSource={strategies}
        pagination={false}
        size="small"
        rowKey="strategyInstanceId"
        columns={[
          {
            title: 'Strategy',
            dataIndex: 'strategyId',
            render: (id: string) => (
              <Text strong>{id}</Text>
            )
          },
          {
            title: 'State',
            dataIndex: 'state',
            render: (state: string) => (
              <Tag color={state === 'running' ? 'green' : state === 'paused' ? 'orange' : 'red'}>
                {state.toUpperCase()}
              </Tag>
            )
          },
          {
            title: 'Health',
            dataIndex: 'healthStatus',
            render: (health: HealthStatus) => (
              <Badge
                status={getHealthColor(health.overall) as any}
                text={health.overall.toUpperCase()}
              />
            )
          },
          {
            title: 'Heartbeat',
            dataIndex: ['healthStatus', 'heartbeat'],
            render: (heartbeat: string) => (
              <div className="flex items-center space-x-1">
                <HeartOutlined 
                  style={{ color: heartbeat === 'active' ? '#52c41a' : '#f5222d' }}
                />
                <span>{heartbeat}</span>
              </div>
            )
          },
          {
            title: 'Data Feed',
            dataIndex: ['healthStatus', 'dataFeed'],
            render: (dataFeed: string) => (
              <Badge
                status={getHealthColor(dataFeed) as any}
                text={dataFeed}
              />
            )
          },
          {
            title: 'Last Check',
            dataIndex: ['healthStatus', 'lastHealthCheck'],
            render: (date: Date) => date ? new Date(date).toLocaleTimeString() : 'Never'
          },
          {
            title: 'Alerts',
            key: 'alerts',
            render: (_, strategy: LiveStrategy) => {
              const strategyAlerts = allAlerts.filter(
                alert => alert.strategyInstanceId === strategy.strategyInstanceId && !alert.acknowledged
              );
              return (
                <Tag color={strategyAlerts.length > 0 ? 'red' : 'green'}>
                  {strategyAlerts.length}
                </Tag>
              );
            }
          }
        ]}
      />
    </Card>
  );

  const alertColumns = [
    {
      title: 'Time',
      dataIndex: 'timestamp',
      render: (time: Date) => new Date(time).toLocaleString(),
      width: 150
    },
    {
      title: 'Strategy',
      dataIndex: 'strategyInstanceId',
      render: (id: string) => {
        const strategy = strategies.find(s => s.strategyInstanceId === id);
        return strategy?.strategyId || id.slice(0, 8);
      },
      width: 120
    },
    {
      title: 'Severity',
      dataIndex: 'severity',
      render: (severity: string) => (
        <Tag color={getSeverityColor(severity)}>
          {severity.toUpperCase()}
        </Tag>
      ),
      width: 100
    },
    {
      title: 'Type',
      dataIndex: 'type',
      render: (type: AlertType) => type.replace('_', ' ').toUpperCase(),
      width: 150
    },
    {
      title: 'Message',
      dataIndex: 'message',
      ellipsis: true
    },
    {
      title: 'Status',
      key: 'status',
      render: (_, alert: StrategyAlert) => (
        <Tag color={alert.acknowledged ? 'green' : 'orange'}>
          {alert.acknowledged ? 'ACKNOWLEDGED' : 'ACTIVE'}
        </Tag>
      ),
      width: 120
    },
    {
      title: 'Actions',
      key: 'actions',
      render: (_, alert: StrategyAlert) => (
        <Space size="small">
          <Tooltip title="View Details">
            <Button
              size="small"
              icon={<EyeOutlined />}
              onClick={() => {
                setSelectedAlert(alert);
                setAlertDetailsVisible(true);
              }}
            />
          </Tooltip>
          {!alert.acknowledged && (
            <Tooltip title="Acknowledge">
              <Button
                size="small"
                icon={<CheckCircleOutlined />}
                onClick={() => acknowledgeAlert(alert.alertId)}
              />
            </Tooltip>
          )}
        </Space>
      ),
      width: 100
    }
  ];

  const renderAlerts = () => (
    <Card
      title="System Alerts"
      size="small"
      extra={
        <Space>
          <Select
            value={selectedStrategy}
            onChange={setSelectedStrategy}
            style={{ width: 150 }}
            size="small"
          >
            <Select.Option value="all">All Strategies</Select.Option>
            {strategies.map(strategy => (
              <Select.Option key={strategy.strategyInstanceId} value={strategy.strategyInstanceId}>
                {strategy.strategyId}
              </Select.Option>
            ))}
          </Select>
          
          <Select
            value={alertFilter}
            onChange={setAlertFilter}
            style={{ width: 120 }}
            size="small"
          >
            <Select.Option value="all">All Alerts</Select.Option>
            <Select.Option value="critical">Critical</Select.Option>
            <Select.Option value="unacknowledged">Unacknowledged</Select.Option>
          </Select>
          
          <Button
            size="small"
            icon={<ReloadOutlined />}
            onClick={loadHealthData}
            loading={loading}
          >
            Refresh
          </Button>
        </Space>
      }
    >
      <Table
        dataSource={getFilteredAlerts()}
        columns={alertColumns}
        pagination={{ pageSize: 10 }}
        size="small"
        rowKey="alertId"
        scroll={{ y: 400 }}
      />
    </Card>
  );

  if (!systemHealth) {
    return (
      <Card loading={loading}>
        <div className="text-center py-8">
          Loading system health data...
        </div>
      </Card>
    );
  }

  return (
    <div className="strategy-health-dashboard">
      <Card
        title={
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-2">
              <DashboardOutlined className="text-blue-600" />
              <span>Strategy Health Dashboard</span>
            </div>
            <div className="flex items-center space-x-2">
              <Badge
                status={getHealthColor(systemHealth.overall) as any}
                text={`System: ${systemHealth.overall.toUpperCase()}`}
              />
              <Text type="secondary">
                Last Update: {systemHealth.lastUpdate?.toLocaleTimeString() || 'Never'}
              </Text>
            </div>
          </div>
        }
        extra={
          <Button
            icon={<ReloadOutlined />}
            onClick={loadHealthData}
            loading={loading}
          >
            Refresh All
          </Button>
        }
      >
        {renderSystemOverview()}
        {renderDataConnections()}
        {showDetails && renderStrategyHealth()}
        {renderAlerts()}
      </Card>

      {/* Alert Details Modal */}
      <Modal
        title="Alert Details"
        visible={alertDetailsVisible}
        onCancel={() => setAlertDetailsVisible(false)}
        footer={[
          <Button key="close" onClick={() => setAlertDetailsVisible(false)}>
            Close
          </Button>,
          selectedAlert && !selectedAlert.acknowledged && (
            <Button
              key="acknowledge"
              type="primary"
              onClick={() => {
                acknowledgeAlert(selectedAlert.alertId);
                setAlertDetailsVisible(false);
              }}
            >
              Acknowledge
            </Button>
          )
        ]}
      >
        {selectedAlert && (
          <Descriptions bordered>
            <Descriptions.Item label="Alert ID" span={3}>
              {selectedAlert.alertId}
            </Descriptions.Item>
            <Descriptions.Item label="Strategy" span={3}>
              {strategies.find(s => s.strategyInstanceId === selectedAlert.strategyInstanceId)?.strategyId || 'Unknown'}
            </Descriptions.Item>
            <Descriptions.Item label="Type" span={1}>
              {selectedAlert.type.replace('_', ' ').toUpperCase()}
            </Descriptions.Item>
            <Descriptions.Item label="Severity" span={2}>
              <Tag color={getSeverityColor(selectedAlert.severity)}>
                {selectedAlert.severity.toUpperCase()}
              </Tag>
            </Descriptions.Item>
            <Descriptions.Item label="Message" span={3}>
              {selectedAlert.message}
            </Descriptions.Item>
            <Descriptions.Item label="Timestamp" span={3}>
              {selectedAlert.timestamp.toLocaleString()}
            </Descriptions.Item>
            <Descriptions.Item label="Status" span={3}>
              <Tag color={selectedAlert.acknowledged ? 'green' : 'orange'}>
                {selectedAlert.acknowledged ? 'ACKNOWLEDGED' : 'ACTIVE'}
              </Tag>
              {selectedAlert.acknowledgedBy && (
                <Text type="secondary" className="ml-2">
                  by {selectedAlert.acknowledgedBy}
                </Text>
              )}
            </Descriptions.Item>
          </Descriptions>
        )}
      </Modal>
    </div>
  );
};

export default StrategyHealthDashboard;