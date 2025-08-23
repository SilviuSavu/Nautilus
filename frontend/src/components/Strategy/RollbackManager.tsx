import React, { useState, useEffect } from 'react';
import {
  Modal,
  Steps,
  Button,
  Alert,
  Typography,
  Space,
  Card,
  Descriptions,
  List,
  Checkbox,
  Radio,
  Input,
  Form,
  Progress,
  Result,
  Spin,
  Divider,
  Tag,
  Row,
  Col,
  Statistic,
  Timeline,
  Table,
  Tabs,
  Badge,
  Tooltip,
  notification
} from 'antd';
import {
  RollbackOutlined,
  WarningOutlined,
  CheckCircleOutlined,
  ExclamationCircleOutlined,
  InfoCircleOutlined,
  LoadingOutlined,
  SafetyCertificateOutlined,
  DatabaseOutlined,
  ApiOutlined,
  SettingOutlined,
  ThunderboltOutlined,
  MonitorOutlined,
  BellOutlined,
  EyeOutlined,
  PlayCircleOutlined,
  PauseCircleOutlined
} from '@ant-design/icons';
import { useRollbackService } from '../../hooks/strategy/useRollbackService';

const { Title, Text, Paragraph } = Typography;
const { Step } = Steps;
const { TabPane } = Tabs;
const { TextArea } = Input;

interface RollbackManagerProps {
  strategyId: string;
  environment: string;
  visible: boolean;
  onClose: () => void;
  onRollbackComplete?: (success: boolean) => void;
}

export const RollbackManager: React.FC<RollbackManagerProps> = ({
  strategyId,
  environment,
  visible,
  onClose,
  onRollbackComplete
}) => {
  const [activeTab, setActiveTab] = useState('dashboard');
  const [selectedExecution, setSelectedExecution] = useState<string | null>(null);
  const [emergencyRollbackModalVisible, setEmergencyRollbackModalVisible] = useState(false);
  const [triggerModalVisible, setTriggerModalVisible] = useState(false);
  const [monitoringConfig, setMonitoringConfig] = useState({
    window: 300,
    triggers: [] as string[],
    alertThresholds: {} as Record<string, number>
  });
  const [form] = Form.useForm();
  const [emergencyForm] = Form.useForm();
  const [triggerForm] = Form.useForm();

  const {
    rollbackPlans,
    rollbackExecutions,
    performanceMonitors,
    rollbackTriggers,
    rollbackHistory,
    activeMonitoring,
    loading,
    error,
    createRollbackPlan,
    executeRollback,
    emergencyRollback,
    createRollbackTrigger,
    updateRollbackTrigger,
    startPerformanceMonitoring,
    stopPerformanceMonitoring,
    cancelRollbackExecution,
    getRollbackPlan,
    getRollbackExecution,
    getPerformanceMonitor,
    isMonitoringActive,
    fetchRollbackPlans,
    fetchRollbackExecutions,
    fetchRollbackHistory
  } = useRollbackService();

  // Initialize data
  useEffect(() => {
    if (visible) {
      fetchRollbackPlans(strategyId, environment);
      fetchRollbackExecutions(strategyId);
      fetchRollbackHistory(strategyId);
    }
  }, [visible, strategyId, environment, fetchRollbackPlans, fetchRollbackExecutions, fetchRollbackHistory]);

  // Handle emergency rollback
  const handleEmergencyRollback = async (values: any) => {
    try {
      const result = await emergencyRollback(strategyId, environment, values.reason, 'user');
      if (result) {
        notification.success({
          message: 'Emergency Rollback Initiated',
          description: 'Emergency rollback has been started'
        });
        setEmergencyRollbackModalVisible(false);
        emergencyForm.resetFields();
        onRollbackComplete?.(true);
      }
    } catch (error) {
      notification.error({
        message: 'Emergency Rollback Failed',
        description: `Failed to initiate emergency rollback: ${error}`
      });
    }
  };

  // Handle rollback trigger creation
  const handleCreateTrigger = async (values: any) => {
    try {
      const triggerConfig = {
        name: values.name,
        type: values.type,
        enabled: true,
        conditions: values.conditions || [],
        actions: {
          rollback: values.rollback || false,
          notify: values.notify || false,
          pauseTrading: values.pauseTrading || false,
          emergencyStop: values.emergencyStop || false
        },
        metadata: values.metadata || {}
      };

      await createRollbackTrigger(strategyId, triggerConfig);
      setTriggerModalVisible(false);
      triggerForm.resetFields();
      notification.success({
        message: 'Rollback Trigger Created',
        description: 'Automated rollback trigger has been created'
      });
    } catch (error) {
      notification.error({
        message: 'Failed to Create Trigger',
        description: `Failed to create rollback trigger: ${error}`
      });
    }
  };

  // Handle performance monitoring start
  const handleStartMonitoring = async () => {
    try {
      await startPerformanceMonitoring(strategyId, environment, monitoringConfig);
      notification.success({
        message: 'Performance Monitoring Started',
        description: 'Automated rollback monitoring is now active'
      });
    } catch (error) {
      notification.error({
        message: 'Failed to Start Monitoring',
        description: `Failed to start performance monitoring: ${error}`
      });
    }
  };

  // Get status color
  const getStatusColor = (status: string) => {
    switch (status) {
      case 'completed':
        return 'green';
      case 'failed':
        return 'red';
      case 'running':
        return 'blue';
      case 'pending':
        return 'orange';
      case 'cancelled':
        return 'gray';
      default:
        return 'default';
    }
  };

  // Rollback execution columns
  const executionColumns = [
    {
      title: 'Execution ID',
      dataIndex: 'executionId',
      key: 'executionId',
      render: (id: string) => (
        <Button
          type="link"
          onClick={() => setSelectedExecution(id)}
        >
          {id.slice(0, 8)}...
        </Button>
      )
    },
    {
      title: 'Strategy',
      dataIndex: 'strategyId',
      key: 'strategyId',
      render: (id: string) => <Text code>{id}</Text>
    },
    {
      title: 'Environment',
      dataIndex: 'environment',
      key: 'environment',
      render: (env: string) => <Tag color="blue">{env}</Tag>
    },
    {
      title: 'Status',
      dataIndex: 'status',
      key: 'status',
      render: (status: string) => (
        <Tag color={getStatusColor(status)}>
          {status}
        </Tag>
      )
    },
    {
      title: 'Reason',
      dataIndex: 'triggerReason',
      key: 'triggerReason',
      render: (reason: string) => (
        <Tooltip title={reason}>
          <Text ellipsis style={{ maxWidth: 150 }}>
            {reason}
          </Text>
        </Tooltip>
      )
    },
    {
      title: 'Started At',
      dataIndex: 'startedAt',
      key: 'startedAt',
      render: (date: Date) => new Date(date).toLocaleString()
    },
    {
      title: 'Duration',
      dataIndex: 'duration',
      key: 'duration',
      render: (duration?: number) => duration ? `${Math.round(duration / 1000)}s` : '-'
    },
    {
      title: 'Actions',
      key: 'actions',
      render: (_: any, record: any) => (
        <Space>
          <Button
            size="small"
            icon={<EyeOutlined />}
            onClick={() => setSelectedExecution(record.executionId)}
          >
            View
          </Button>
          {record.status === 'running' && (
            <Button
              size="small"
              danger
              onClick={() => cancelRollbackExecution(record.executionId)}
            >
              Cancel
            </Button>
          )}
        </Space>
      )
    }
  ];

  // Rollback trigger columns
  const triggerColumns = [
    {
      title: 'Name',
      dataIndex: 'name',
      key: 'name'
    },
    {
      title: 'Type',
      dataIndex: 'type',
      key: 'type',
      render: (type: string) => <Tag>{type}</Tag>
    },
    {
      title: 'Status',
      dataIndex: 'enabled',
      key: 'enabled',
      render: (enabled: boolean) => (
        <Badge
          status={enabled ? 'success' : 'default'}
          text={enabled ? 'Enabled' : 'Disabled'}
        />
      )
    },
    {
      title: 'Conditions',
      dataIndex: 'conditions',
      key: 'conditions',
      render: (conditions: any[]) => `${conditions.length} conditions`
    },
    {
      title: 'Actions',
      key: 'actions',
      render: (_: any, record: any) => (
        <Space>
          <Button
            size="small"
            type={record.enabled ? 'default' : 'primary'}
            onClick={() => updateRollbackTrigger(record.triggerId, { enabled: !record.enabled })}
          >
            {record.enabled ? 'Disable' : 'Enable'}
          </Button>
          <Button
            size="small"
            icon={<SettingOutlined />}
          >
            Configure
          </Button>
        </Space>
      )
    }
  ];

  const selectedExecutionData = selectedExecution ? getRollbackExecution(selectedExecution) : null;

  return (
    <Modal
      title={
        <Space>
          <RollbackOutlined />
          Rollback Manager
        </Space>
      }
      open={visible}
      onCancel={onClose}
      width={1200}
      footer={[
        <Button key="close" onClick={onClose}>
          Close
        </Button>
      ]}
    >
      {error && (
        <Alert
          message="Error"
          description={error}
          type="error"
          showIcon
          closable
          style={{ marginBottom: 16 }}
        />
      )}

      <Tabs activeKey={activeTab} onChange={setActiveTab}>
        <TabPane tab="Dashboard" key="dashboard">
          <Space direction="vertical" style={{ width: '100%' }}>
            {/* Statistics */}
            <Row gutter={16}>
              <Col span={8}>
                <Card size="small">
                  <Statistic
                    title="Total Rollbacks"
                    value={rollbackHistory?.totalRollbacks || 0}
                    prefix={<RollbackOutlined />}
                  />
                </Card>
              </Col>
              <Col span={8}>
                <Card size="small">
                  <Statistic
                    title="Success Rate"
                    value={rollbackHistory?.successRate || 0}
                    suffix="%"
                    precision={1}
                    valueStyle={{ color: '#52c41a' }}
                    prefix={<CheckCircleOutlined />}
                  />
                </Card>
              </Col>
              <Col span={8}>
                <Card size="small">
                  <Statistic
                    title="Avg Duration"
                    value={rollbackHistory?.averageDuration || 0}
                    suffix="s"
                    precision={1}
                    prefix={<ThunderboltOutlined />}
                  />
                </Card>
              </Col>
            </Row>

            {/* Active Monitoring */}
            {performanceMonitors.length > 0 && (
              <Card title="Active Monitoring" size="small">
                <List
                  dataSource={performanceMonitors}
                  renderItem={(monitor: any) => (
                    <List.Item
                      actions={[
                        <Button
                          key="stop"
                          size="small"
                          danger
                          disabled={!isMonitoringActive(monitor.monitorId)}
                          onClick={() => stopPerformanceMonitoring(monitor.monitorId)}
                        >
                          Stop
                        </Button>
                      ]}
                    >
                      <List.Item.Meta
                        avatar={<MonitorOutlined style={{ fontSize: 24, color: '#1890ff' }} />}
                        title={
                          <Space>
                            <Text strong>Strategy {monitor.strategyId}</Text>
                            <Badge
                              status={isMonitoringActive(monitor.monitorId) ? 'success' : 'default'}
                              text={isMonitoringActive(monitor.monitorId) ? 'Active' : 'Inactive'}
                            />
                          </Space>
                        }
                        description={
                          <div>
                            <Text type="secondary">
                              Environment: {monitor.environment} | 
                              Window: {monitor.monitoringWindow}s | 
                              Triggers: {monitor.triggers.length}
                            </Text>
                          </div>
                        }
                      />
                    </List.Item>
                  )}
                />
              </Card>
            )}

            {/* Quick Actions */}
            <Card title="Quick Actions" size="small">
              <Space>
                <Button
                  type="primary"
                  danger
                  icon={<ThunderboltOutlined />}
                  onClick={() => setEmergencyRollbackModalVisible(true)}
                >
                  Emergency Rollback
                </Button>
                <Button
                  icon={<MonitorOutlined />}
                  onClick={handleStartMonitoring}
                >
                  Start Monitoring
                </Button>
                <Button
                  icon={<BellOutlined />}
                  onClick={() => setTriggerModalVisible(true)}
                >
                  Create Trigger
                </Button>
              </Space>
            </Card>
          </Space>
        </TabPane>

        <TabPane tab="Rollback History" key="history">
          <Card title="Rollback Executions">
            <Table
              dataSource={rollbackExecutions}
              columns={executionColumns}
              loading={loading}
              rowKey="executionId"
              pagination={{ pageSize: 10 }}
            />
          </Card>
        </TabPane>

        <TabPane tab="Automated Triggers" key="triggers">
          <Card 
            title="Rollback Triggers"
            extra={
              <Button
                type="primary"
                icon={<BellOutlined />}
                onClick={() => setTriggerModalVisible(true)}
              >
                Create Trigger
              </Button>
            }
          >
            <Table
              dataSource={rollbackTriggers}
              columns={triggerColumns}
              loading={loading}
              rowKey="triggerId"
              pagination={{ pageSize: 10 }}
            />
          </Card>
        </TabPane>

        <TabPane tab="Performance Monitoring" key="monitoring">
          <Card title="Performance Monitors">
            <List
              dataSource={performanceMonitors}
              renderItem={(monitor: any) => (
                <List.Item
                  actions={[
                    <Button
                      key="view"
                      size="small"
                      icon={<EyeOutlined />}
                    >
                      View Details
                    </Button>,
                    <Button
                      key="toggle"
                      size="small"
                      type={isMonitoringActive(monitor.monitorId) ? 'default' : 'primary'}
                      onClick={() => 
                        isMonitoringActive(monitor.monitorId)
                          ? stopPerformanceMonitoring(monitor.monitorId)
                          : handleStartMonitoring()
                      }
                    >
                      {isMonitoringActive(monitor.monitorId) ? 'Stop' : 'Start'}
                    </Button>
                  ]}
                >
                  <List.Item.Meta
                    avatar={
                      <Badge
                        status={isMonitoringActive(monitor.monitorId) ? 'success' : 'default'}
                      >
                        <MonitorOutlined style={{ fontSize: 24 }} />
                      </Badge>
                    }
                    title={`Monitor ${monitor.monitorId.slice(0, 8)}...`}
                    description={
                      <Space direction="vertical">
                        <Text type="secondary">
                          Strategy: {monitor.strategyId} | Environment: {monitor.environment}
                        </Text>
                        <Text type="secondary">
                          Window: {monitor.monitoringWindow}s | Triggers: {monitor.triggers.length}
                        </Text>
                        <Text type="secondary">
                          Last Check: {new Date(monitor.lastCheck).toLocaleString()}
                        </Text>
                      </Space>
                    }
                  />
                </List.Item>
              )}
            />
          </Card>
        </TabPane>
      </Tabs>

      {/* Emergency Rollback Modal */}
      <Modal
        title="Emergency Rollback"
        open={emergencyRollbackModalVisible}
        onCancel={() => setEmergencyRollbackModalVisible(false)}
        footer={null}
        width={600}
      >
        <Alert
          message="Emergency Rollback"
          description="This will immediately rollback the strategy to the last known stable version. Use with caution."
          type="warning"
          showIcon
          style={{ marginBottom: 16 }}
        />

        <Form
          form={emergencyForm}
          layout="vertical"
          onFinish={handleEmergencyRollback}
        >
          <Form.Item
            name="reason"
            label="Rollback Reason"
            rules={[{ required: true, message: 'Please provide a reason for emergency rollback' }]}
          >
            <TextArea
              rows={3}
              placeholder="Describe the emergency situation requiring rollback..."
            />
          </Form.Item>

          <Form.Item>
            <Space>
              <Button
                type="primary"
                danger
                htmlType="submit"
                loading={loading}
                icon={<ThunderboltOutlined />}
              >
                Execute Emergency Rollback
              </Button>
              <Button onClick={() => setEmergencyRollbackModalVisible(false)}>
                Cancel
              </Button>
            </Space>
          </Form.Item>
        </Form>
      </Modal>

      {/* Create Trigger Modal */}
      <Modal
        title="Create Rollback Trigger"
        open={triggerModalVisible}
        onCancel={() => setTriggerModalVisible(false)}
        footer={null}
        width={600}
      >
        <Form
          form={triggerForm}
          layout="vertical"
          onFinish={handleCreateTrigger}
          initialValues={{
            type: 'performance',
            rollback: true,
            notify: true
          }}
        >
          <Form.Item
            name="name"
            label="Trigger Name"
            rules={[{ required: true, message: 'Please enter trigger name' }]}
          >
            <Input placeholder="High Drawdown Trigger" />
          </Form.Item>

          <Form.Item
            name="type"
            label="Trigger Type"
            rules={[{ required: true, message: 'Please select trigger type' }]}
          >
            <Radio.Group>
              <Radio value="performance">Performance</Radio>
              <Radio value="error_rate">Error Rate</Radio>
              <Radio value="drawdown">Drawdown</Radio>
              <Radio value="custom">Custom</Radio>
            </Radio.Group>
          </Form.Item>

          <Form.Item label="Actions">
            <Space direction="vertical">
              <Form.Item name="rollback" valuePropName="checked" style={{ margin: 0 }}>
                <Checkbox>Trigger automatic rollback</Checkbox>
              </Form.Item>
              <Form.Item name="notify" valuePropName="checked" style={{ margin: 0 }}>
                <Checkbox>Send notifications</Checkbox>
              </Form.Item>
              <Form.Item name="pauseTrading" valuePropName="checked" style={{ margin: 0 }}>
                <Checkbox>Pause trading</Checkbox>
              </Form.Item>
              <Form.Item name="emergencyStop" valuePropName="checked" style={{ margin: 0 }}>
                <Checkbox>Emergency stop</Checkbox>
              </Form.Item>
            </Space>
          </Form.Item>

          <Form.Item>
            <Space>
              <Button
                type="primary"
                htmlType="submit"
                loading={loading}
              >
                Create Trigger
              </Button>
              <Button onClick={() => setTriggerModalVisible(false)}>
                Cancel
              </Button>
            </Space>
          </Form.Item>
        </Form>
      </Modal>
    </Modal>
  );
};

export default RollbackManager;