/**
 * Pipeline Monitor Component
 * Real-time deployment monitoring and alerting
 */

import React, { useState, useEffect } from 'react';
import {
  Card,
  Row,
  Col,
  Statistic,
  Alert,
  Table,
  Tag,
  Button,
  Space,
  Typography,
  Timeline,
  Progress,
  Badge,
  Tooltip,
  Modal,
  Form,
  Select,
  Input,
  Switch,
  List,
  Avatar,
  Tabs,
  Divider
} from 'antd';
import {
  MonitorOutlined,
  AlertOutlined,
  CheckCircleOutlined,
  ExclamationCircleOutlined,
  CloseCircleOutlined,
  PlayCircleOutlined,
  PauseCircleOutlined,
  ReloadOutlined,
  SettingOutlined,
  BellOutlined,
  EyeOutlined,
  ThunderboltOutlined,
  RiseOutlined,
  FallOutlined,
  MinusOutlined
} from '@ant-design/icons';
import { Line, Column } from '@ant-design/plots';
import { usePipelineMonitoring } from '../../hooks/strategy/usePipelineMonitoring';

const { Title, Text } = Typography;
const { TabPane } = Tabs;
const { Option } = Select;

interface PipelineMonitorProps {
  pipelineId?: string;
  autoRefresh?: boolean;
  refreshInterval?: number;
}

export const PipelineMonitor: React.FC<PipelineMonitorProps> = ({
  pipelineId,
  autoRefresh = true,
  refreshInterval = 10000
}) => {
  const [selectedAlert, setSelectedAlert] = useState<string | null>(null);
  const [alertModalVisible, setAlertModalVisible] = useState(false);
  const [settingsModalVisible, setSettingsModalVisible] = useState(false);
  const [selectedTimeRange, setSelectedTimeRange] = useState<'1h' | '6h' | '24h' | '7d'>('1h');

  const [settingsForm] = Form.useForm();

  const {
    pipelineExecutions,
    pipelineAlerts,
    pipelineMetrics,
    activeExecutions,
    realtimeUpdates,
    connectionStatus,
    loading,
    acknowledgeAlert,
    getUnacknowledgedAlertsCount,
    getActiveExecutionsCount,
    subscribeToPipeline,
    unsubscribeFromPipeline,
    fetchPipelineExecutions,
    fetchPipelineAlerts,
    fetchPipelineMetrics
  } = usePipelineMonitoring();

  // Subscribe to pipeline updates
  useEffect(() => {
    if (pipelineId) {
      subscribeToPipeline(pipelineId);
      return () => unsubscribeFromPipeline(pipelineId);
    }
  }, [pipelineId, subscribeToPipeline, unsubscribeFromPipeline]);

  // Auto-refresh data
  useEffect(() => {
    if (!autoRefresh) return;

    const interval = setInterval(() => {
      fetchPipelineExecutions({ limit: 50 });
      fetchPipelineAlerts({ limit: 50 });
      fetchPipelineMetrics(pipelineId);
    }, refreshInterval);

    return () => clearInterval(interval);
  }, [autoRefresh, refreshInterval, pipelineId, fetchPipelineExecutions, fetchPipelineAlerts, fetchPipelineMetrics]);

  // Get alert severity color
  const getAlertSeverityColor = (severity: string) => {
    switch (severity) {
      case 'critical':
        return '#f5222d';
      case 'high':
        return '#fa8c16';
      case 'medium':
        return '#fadb14';
      case 'low':
        return '#52c41a';
      default:
        return '#d9d9d9';
    }
  };

  // Get alert type icon
  const getAlertTypeIcon = (type: string) => {
    switch (type) {
      case 'error':
        return <CloseCircleOutlined style={{ color: '#f5222d' }} />;
      case 'warning':
        return <ExclamationCircleOutlined style={{ color: '#fa8c16' }} />;
      case 'info':
        return <CheckCircleOutlined style={{ color: '#1890ff' }} />;
      case 'success':
        return <CheckCircleOutlined style={{ color: '#52c41a' }} />;
      default:
        return <ExclamationCircleOutlined />;
    }
  };

  // Create performance trend chart
  const createPerformanceTrendChart = () => {
    const metrics = pipelineId ? pipelineMetrics[pipelineId] : null;
    if (!metrics?.performanceTrend) return null;

    const data = metrics.performanceTrend.map(point => ({
      timestamp: new Date(point.timestamp).toLocaleString(),
      duration: Math.round(point.duration / 1000),
      success: point.success ? 1 : 0
    }));

    return (
      <Line
        data={data}
        xField="timestamp"
        yField="duration"
        height={200}
        smooth
        point={{
          size: 3,
          style: {
            lineWidth: 1,
            fillOpacity: 0.8,
          },
        }}
        color="#1890ff"
        xAxis={{
          tickCount: 5
        }}
        yAxis={{
          title: {
            text: 'Duration (seconds)'
          }
        }}
        tooltip={{
          showMarkers: true
        }}
      />
    );
  };

  // Create execution status distribution chart
  const createExecutionStatusChart = () => {
    const statusCounts = pipelineExecutions.reduce((acc, exec) => {
      acc[exec.status] = (acc[exec.status] || 0) + 1;
      return acc;
    }, {} as Record<string, number>);

    const data = Object.entries(statusCounts).map(([status, count]) => ({
      status: status.charAt(0).toUpperCase() + status.slice(1),
      count
    }));

    return (
      <Column
        data={data}
        xField="status"
        yField="count"
        height={200}
        color={({ status }) => {
          switch (status.toLowerCase()) {
            case 'completed':
              return '#52c41a';
            case 'failed':
              return '#f5222d';
            case 'running':
              return '#1890ff';
            case 'pending':
              return '#faad14';
            default:
              return '#d9d9d9';
          }
        }}
      />
    );
  };

  // Alert columns
  const alertColumns = [
    {
      title: 'Severity',
      dataIndex: 'severity',
      key: 'severity',
      render: (severity: string) => (
        <Badge
          color={getAlertSeverityColor(severity)}
          text={severity.toUpperCase()}
        />
      )
    },
    {
      title: 'Type',
      dataIndex: 'type',
      key: 'type',
      render: (type: string) => (
        <Space>
          {getAlertTypeIcon(type)}
          <Text>{type}</Text>
        </Space>
      )
    },
    {
      title: 'Title',
      dataIndex: 'title',
      key: 'title',
      render: (title: string, record: any) => (
        <Button
          type="link"
          onClick={() => {
            setSelectedAlert(record.alertId);
            setAlertModalVisible(true);
          }}
          style={{ padding: 0 }}
        >
          {title}
        </Button>
      )
    },
    {
      title: 'Pipeline',
      dataIndex: 'pipelineId',
      key: 'pipelineId',
      render: (id: string) => (
        <Text code>{id.slice(0, 8)}...</Text>
      )
    },
    {
      title: 'Timestamp',
      dataIndex: 'timestamp',
      key: 'timestamp',
      render: (timestamp: Date) => new Date(timestamp).toLocaleString()
    },
    {
      title: 'Status',
      dataIndex: 'acknowledged',
      key: 'acknowledged',
      render: (acknowledged: boolean) => (
        <Tag color={acknowledged ? 'green' : 'red'}>
          {acknowledged ? 'Acknowledged' : 'Active'}
        </Tag>
      )
    },
    {
      title: 'Actions',
      key: 'actions',
      render: (_: any, record: any) => (
        <Space>
          <Button
            size="small"
            icon={<EyeOutlined />}
            onClick={() => {
              setSelectedAlert(record.alertId);
              setAlertModalVisible(true);
            }}
          >
            View
          </Button>
          {!record.acknowledged && (
            <Button
              size="small"
              type="primary"
              onClick={() => acknowledgeAlert(record.alertId, 'user')}
            >
              Acknowledge
            </Button>
          )}
        </Space>
      )
    }
  ];

  // Execution columns
  const executionColumns = [
    {
      title: 'Execution ID',
      dataIndex: 'executionId',
      key: 'executionId',
      render: (id: string) => (
        <Text code>{id.slice(0, 8)}...</Text>
      )
    },
    {
      title: 'Pipeline',
      dataIndex: 'pipelineId',
      key: 'pipelineId',
      render: (id: string) => (
        <Text code>{id.slice(0, 8)}...</Text>
      )
    },
    {
      title: 'Status',
      dataIndex: 'status',
      key: 'status',
      render: (status: string) => {
        const color = status === 'completed' ? 'green' :
                     status === 'failed' ? 'red' :
                     status === 'running' ? 'blue' : 'orange';
        return <Tag color={color}>{status}</Tag>;
      }
    },
    {
      title: 'Progress',
      key: 'progress',
      render: (_: any, record: any) => {
        if (!record.stages || record.stages.length === 0) return <Progress percent={0} size="small" />;
        
        const completedStages = record.stages.filter((stage: any) => stage.status === 'completed').length;
        const failedStages = record.stages.filter((stage: any) => stage.status === 'failed').length;
        const percent = Math.round((completedStages / record.stages.length) * 100);
        
        return (
          <Progress
            percent={percent}
            size="small"
            status={failedStages > 0 ? 'exception' : record.status === 'completed' ? 'success' : 'active'}
            format={() => `${completedStages}/${record.stages.length}`}
          />
        );
      }
    },
    {
      title: 'Duration',
      dataIndex: 'totalDuration',
      key: 'totalDuration',
      render: (duration?: number) => duration ? `${Math.round(duration / 1000)}s` : '-'
    },
    {
      title: 'Started At',
      dataIndex: 'startedAt',
      key: 'startedAt',
      render: (date: Date) => new Date(date).toLocaleString()
    }
  ];

  const selectedAlertData = pipelineAlerts.find(alert => alert.alertId === selectedAlert);

  return (
    <div className="pipeline-monitor">
      <div style={{ marginBottom: 24 }}>
        <Row gutter={[16, 16]} align="middle">
          <Col>
            <Title level={3}>
              <MonitorOutlined /> Pipeline Monitor
            </Title>
          </Col>
          <Col>
            <Space>
              <Badge
                status={connectionStatus === 'connected' ? 'success' : 'error'}
                text={connectionStatus === 'connected' ? 'Real-time Connected' : 'Disconnected'}
              />
              <Select
                value={selectedTimeRange}
                onChange={setSelectedTimeRange}
                style={{ width: 120 }}
              >
                <Option value="1h">Last Hour</Option>
                <Option value="6h">Last 6 Hours</Option>
                <Option value="24h">Last 24 Hours</Option>
                <Option value="7d">Last 7 Days</Option>
              </Select>
              <Button
                icon={<ReloadOutlined />}
                onClick={() => {
                  fetchPipelineExecutions();
                  fetchPipelineAlerts();
                  fetchPipelineMetrics(pipelineId);
                }}
                loading={loading}
              >
                Refresh
              </Button>
              <Button
                icon={<SettingOutlined />}
                onClick={() => setSettingsModalVisible(true)}
              >
                Settings
              </Button>
            </Space>
          </Col>
        </Row>
      </div>

      {/* Connection Status Alert */}
      {connectionStatus !== 'connected' && (
        <Alert
          message={`Pipeline monitoring ${connectionStatus}`}
          description="Some real-time features may not be available"
          type="warning"
          showIcon
          style={{ marginBottom: 16 }}
        />
      )}

      {/* Overview Statistics */}
      <Row gutter={16} style={{ marginBottom: 24 }}>
        <Col span={6}>
          <Card size="small">
            <Statistic
              title="Active Executions"
              value={getActiveExecutionsCount()}
              prefix={<PlayCircleOutlined />}
              valueStyle={{ color: '#1890ff' }}
            />
          </Card>
        </Col>
        <Col span={6}>
          <Card size="small">
            <Statistic
              title="Active Alerts"
              value={getUnacknowledgedAlertsCount()}
              prefix={<AlertOutlined />}
              valueStyle={{ color: '#f5222d' }}
            />
          </Card>
        </Col>
        <Col span={6}>
          <Card size="small">
            <Statistic
              title="Success Rate"
              value={pipelineId && pipelineMetrics[pipelineId] ? pipelineMetrics[pipelineId].successRate : 0}
              suffix="%"
              precision={1}
              prefix={<RiseOutlined />}
              valueStyle={{ 
                color: pipelineId && pipelineMetrics[pipelineId] && pipelineMetrics[pipelineId].successRate >= 90 
                  ? '#52c41a' : '#faad14' 
              }}
            />
          </Card>
        </Col>
        <Col span={6}>
          <Card size="small">
            <Statistic
              title="Avg Duration"
              value={pipelineId && pipelineMetrics[pipelineId] ? Math.round(pipelineMetrics[pipelineId].averageDuration / 1000) : 0}
              suffix="s"
              prefix={<ThunderboltOutlined />}
            />
          </Card>
        </Col>
      </Row>

      {/* Performance Charts */}
      <Row gutter={16} style={{ marginBottom: 24 }}>
        <Col span={12}>
          <Card title="Performance Trend" size="small">
            {createPerformanceTrendChart()}
          </Card>
        </Col>
        <Col span={12}>
          <Card title="Execution Status Distribution" size="small">
            {createExecutionStatusChart()}
          </Card>
        </Col>
      </Row>

      {/* Real-time Updates */}
      {realtimeUpdates.length > 0 && (
        <Card title="Real-time Updates" size="small" style={{ marginBottom: 16 }}>
          <Timeline mode="left" style={{ maxHeight: 200, overflowY: 'auto' }}>
            {realtimeUpdates.slice(0, 10).map((update, index) => (
              <Timeline.Item
                key={index}
                color={update.type === 'alert' ? 'red' : 'blue'}
                label={new Date(update.timestamp).toLocaleTimeString()}
              >
                <Space>
                  <Text strong>{update.type}</Text>
                  <Text code>{update.pipelineId?.slice(0, 8)}...</Text>
                  {update.executionId && <Text code>{update.executionId.slice(0, 8)}...</Text>}
                </Space>
              </Timeline.Item>
            ))}
          </Timeline>
        </Card>
      )}

      {/* Main Content Tabs */}
      <Tabs defaultActiveKey="alerts">
        <TabPane tab={`Alerts (${pipelineAlerts.length})`} key="alerts">
          <Card title="Pipeline Alerts">
            <Table
              dataSource={pipelineAlerts}
              columns={alertColumns}
              loading={loading}
              rowKey="alertId"
              pagination={{ pageSize: 20 }}
              rowClassName={(record) => 
                record.severity === 'critical' ? 'critical-alert' :
                record.severity === 'high' ? 'high-alert' : ''
              }
            />
          </Card>
        </TabPane>

        <TabPane tab={`Executions (${pipelineExecutions.length})`} key="executions">
          <Card title="Pipeline Executions">
            <Table
              dataSource={pipelineExecutions}
              columns={executionColumns}
              loading={loading}
              rowKey="executionId"
              pagination={{ pageSize: 20 }}
            />
          </Card>
        </TabPane>

        <TabPane tab="Metrics" key="metrics">
          <Row gutter={16}>
            {Object.entries(pipelineMetrics).map(([id, metrics]) => (
              <Col span={12} key={id}>
                <Card title={`Pipeline ${id.slice(0, 8)}...`} size="small" style={{ marginBottom: 16 }}>
                  <Row gutter={8}>
                    <Col span={12}>
                      <Statistic
                        title="Total Executions"
                        value={metrics.totalExecutions}
                        valueStyle={{ fontSize: 14 }}
                      />
                    </Col>
                    <Col span={12}>
                      <Statistic
                        title="Failed Executions"
                        value={metrics.failedExecutions}
                        valueStyle={{ fontSize: 14, color: '#f5222d' }}
                      />
                    </Col>
                  </Row>
                  <Divider />
                  <Row gutter={8}>
                    <Col span={12}>
                      <Statistic
                        title="Success Rate"
                        value={metrics.successRate}
                        suffix="%"
                        precision={1}
                        valueStyle={{ fontSize: 14, color: metrics.successRate >= 90 ? '#52c41a' : '#faad14' }}
                      />
                    </Col>
                    <Col span={12}>
                      <Statistic
                        title="Avg Duration"
                        value={Math.round(metrics.averageDuration / 1000)}
                        suffix="s"
                        valueStyle={{ fontSize: 14 }}
                      />
                    </Col>
                  </Row>
                </Card>
              </Col>
            ))}
          </Row>
        </TabPane>
      </Tabs>

      {/* Alert Details Modal */}
      <Modal
        title="Alert Details"
        open={alertModalVisible}
        onCancel={() => setAlertModalVisible(false)}
        footer={[
          <Button key="close" onClick={() => setAlertModalVisible(false)}>
            Close
          </Button>,
          selectedAlertData && !selectedAlertData.acknowledged && (
            <Button
              key="acknowledge"
              type="primary"
              onClick={() => {
                acknowledgeAlert(selectedAlertData.alertId, 'user');
                setAlertModalVisible(false);
              }}
            >
              Acknowledge
            </Button>
          )
        ].filter(Boolean)}
        width={600}
      >
        {selectedAlertData && (
          <div>
            <Space direction="vertical" style={{ width: '100%' }}>
              <Alert
                message={selectedAlertData.title}
                description={selectedAlertData.message}
                type={selectedAlertData.type as any}
                showIcon
              />
              
              <Row gutter={16}>
                <Col span={8}>
                  <Text strong>Severity:</Text><br />
                  <Badge
                    color={getAlertSeverityColor(selectedAlertData.severity)}
                    text={selectedAlertData.severity.toUpperCase()}
                  />
                </Col>
                <Col span={8}>
                  <Text strong>Pipeline:</Text><br />
                  <Text code>{selectedAlertData.pipelineId}</Text>
                </Col>
                <Col span={8}>
                  <Text strong>Timestamp:</Text><br />
                  <Text>{new Date(selectedAlertData.timestamp).toLocaleString()}</Text>
                </Col>
              </Row>

              {selectedAlertData.executionId && (
                <div>
                  <Text strong>Execution ID:</Text><br />
                  <Text code>{selectedAlertData.executionId}</Text>
                </div>
              )}

              {selectedAlertData.actions.length > 0 && (
                <div>
                  <Text strong>Available Actions:</Text>
                  <List
                    size="small"
                    dataSource={selectedAlertData.actions}
                    renderItem={(action) => (
                      <List.Item>
                        <Button
                          type="link"
                          onClick={() => {
                            if (action.url) {
                              window.open(action.url, '_blank');
                            }
                          }}
                        >
                          {action.label}
                        </Button>
                      </List.Item>
                    )}
                  />
                </div>
              )}

              {selectedAlertData.acknowledged && (
                <div>
                  <Text strong>Acknowledged:</Text><br />
                  <Space>
                    <Avatar size="small" icon={<CheckCircleOutlined />} style={{ backgroundColor: '#52c41a' }} />
                    <Text>
                      By {selectedAlertData.acknowledgedBy} on{' '}
                      {selectedAlertData.acknowledgedAt && new Date(selectedAlertData.acknowledgedAt).toLocaleString()}
                    </Text>
                  </Space>
                </div>
              )}
            </Space>
          </div>
        )}
      </Modal>

      {/* Settings Modal */}
      <Modal
        title="Monitor Settings"
        open={settingsModalVisible}
        onCancel={() => setSettingsModalVisible(false)}
        footer={null}
        width={500}
      >
        <Form
          form={settingsForm}
          layout="vertical"
          initialValues={{
            autoRefresh,
            refreshInterval: refreshInterval / 1000,
            alertSound: true,
            emailNotifications: false
          }}
        >
          <Form.Item
            name="autoRefresh"
            label="Auto Refresh"
            valuePropName="checked"
          >
            <Switch />
          </Form.Item>

          <Form.Item
            name="refreshInterval"
            label="Refresh Interval (seconds)"
          >
            <Select>
              <Option value={5}>5 seconds</Option>
              <Option value={10}>10 seconds</Option>
              <Option value={30}>30 seconds</Option>
              <Option value={60}>1 minute</Option>
            </Select>
          </Form.Item>

          <Form.Item
            name="alertSound"
            label="Alert Sound"
            valuePropName="checked"
          >
            <Switch />
          </Form.Item>

          <Form.Item
            name="emailNotifications"
            label="Email Notifications"
            valuePropName="checked"
          >
            <Switch />
          </Form.Item>

          <Form.Item>
            <Space>
              <Button
                type="primary"
                onClick={() => {
                  settingsForm.validateFields().then(values => {
                    console.log('Settings updated:', values);
                    setSettingsModalVisible(false);
                  });
                }}
              >
                Save Settings
              </Button>
              <Button onClick={() => setSettingsModalVisible(false)}>
                Cancel
              </Button>
            </Space>
          </Form.Item>
        </Form>
      </Modal>

      <style jsx global>{`
        .critical-alert {
          background-color: #fff2f0;
          border-left: 4px solid #f5222d;
        }
        .high-alert {
          background-color: #fff7e6;
          border-left: 4px solid #fa8c16;
        }
      `}</style>
    </div>
  );
};