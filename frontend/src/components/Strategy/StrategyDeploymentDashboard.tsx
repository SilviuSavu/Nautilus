/**
 * Strategy Deployment Dashboard
 * Complete deployment pipeline management interface
 */

import React, { useState, useEffect } from 'react';
import {
  Card,
  Row,
  Col,
  Tabs,
  Statistic,
  Table,
  Tag,
  Button,
  Space,
  Progress,
  Timeline,
  Modal,
  Form,
  Select,
  Input,
  Switch,
  notification,
  Drawer,
  Typography,
  Alert,
  Tooltip
} from 'antd';
import {
  PlayCircleOutlined,
  PauseCircleOutlined,
  StopOutlined,
  ReloadOutlined,
  ExclamationCircleOutlined,
  CheckCircleOutlined,
  ClockCircleOutlined,
  RocketOutlined,
  GitlabOutlined,
  MonitorOutlined,
  AlertOutlined,
  SettingOutlined
} from '@ant-design/icons';
import { useStrategyDeployment } from '../../hooks/strategy/useStrategyDeployment';
import { usePipelineMonitoring } from '../../hooks/strategy/usePipelineMonitoring';
import { useVersionControl } from '../../hooks/strategy/useVersionControl';

const { Title, Text } = Typography;
const { TabPane } = Tabs;
const { Option } = Select;

interface StrategyDeploymentDashboardProps {
  strategyId?: string;
  defaultTab?: string;
}

export const StrategyDeploymentDashboard: React.FC<StrategyDeploymentDashboardProps> = ({
  strategyId,
  defaultTab = 'overview'
}) => {
  const [selectedStrategy, setSelectedStrategy] = useState<string>(strategyId || '');
  const [deployModalVisible, setDeployModalVisible] = useState(false);
  const [pipelineModalVisible, setPipelineModalVisible] = useState(false);
  const [selectedDeployment, setSelectedDeployment] = useState<string | null>(null);
  const [selectedExecution, setSelectedExecution] = useState<string | null>(null);
  
  const [form] = Form.useForm();
  const [pipelineForm] = Form.useForm();

  const {
    deployments,
    deploymentRequests,
    deploymentPipelines,
    environments,
    deploymentStats,
    loading: deploymentLoading,
    error: deploymentError,
    createDeploymentRequest,
    approveDeploymentRequest,
    deployStrategy,
    createDeploymentPipeline,
    stopDeployment,
    fetchDeployments,
    fetchDeploymentStats
  } = useStrategyDeployment();

  const {
    pipelineExecutions,
    pipelineAlerts,
    pipelineMetrics,
    activeExecutions,
    connectionStatus,
    loading: monitoringLoading,
    pauseExecution,
    resumeExecution,
    cancelExecution,
    retryStage,
    getUnacknowledgedAlertsCount,
    getActiveExecutionsCount
  } = usePipelineMonitoring();

  const {
    versions,
    branches,
    getLatestVersion,
    fetchVersions,
    fetchBranches
  } = useVersionControl();

  // Initialize data
  useEffect(() => {
    if (selectedStrategy) {
      fetchVersions(selectedStrategy);
      fetchBranches(selectedStrategy);
    }
  }, [selectedStrategy, fetchVersions, fetchBranches]);

  // Refresh data periodically
  useEffect(() => {
    const interval = setInterval(() => {
      if (selectedStrategy) {
        fetchDeployments({ strategyId: selectedStrategy });
        fetchDeploymentStats();
      }
    }, 30000); // Every 30 seconds

    return () => clearInterval(interval);
  }, [selectedStrategy, fetchDeployments, fetchDeploymentStats]);

  // Handle deployment form submission
  const handleDeploy = async (values: any) => {
    try {
      const deploymentConfig = {
        strategy: values.strategy || 'direct',
        autoRollback: values.autoRollback !== false,
        rollbackThreshold: values.rollbackThreshold || 0.05,
        canaryPercentage: values.canaryPercentage,
        approvalRequired: values.approvalRequired || false,
        notificationChannels: values.notificationChannels || [],
        resourceLimits: values.resourceLimits || {},
        healthChecks: values.healthChecks || {}
      };

      await deployStrategy(
        selectedStrategy,
        values.version,
        values.environment,
        deploymentConfig,
        'user'
      );

      setDeployModalVisible(false);
      form.resetFields();
      notification.success({
        message: 'Deployment Initiated',
        description: 'Strategy deployment has been started successfully'
      });
    } catch (error) {
      notification.error({
        message: 'Deployment Failed',
        description: `Failed to initiate deployment: ${error}`
      });
    }
  };

  // Handle pipeline creation
  const handleCreatePipeline = async (values: any) => {
    try {
      await createDeploymentPipeline(
        selectedStrategy,
        values.version,
        values.environments
      );

      setPipelineModalVisible(false);
      pipelineForm.resetFields();
      notification.success({
        message: 'Pipeline Created',
        description: 'Deployment pipeline has been created successfully'
      });
    } catch (error) {
      notification.error({
        message: 'Pipeline Creation Failed',
        description: `Failed to create pipeline: ${error}`
      });
    }
  };

  // Get status color
  const getStatusColor = (status: string) => {
    switch (status) {
      case 'deployed':
      case 'completed':
        return 'green';
      case 'deploying':
      case 'running':
        return 'blue';
      case 'failed':
        return 'red';
      case 'pending':
        return 'orange';
      case 'cancelled':
        return 'gray';
      default:
        return 'default';
    }
  };

  // Get progress percentage
  const getProgressPercentage = (execution: any) => {
    if (!execution.stages || execution.stages.length === 0) return 0;
    const completedStages = execution.stages.filter((stage: any) => 
      stage.status === 'completed'
    ).length;
    return Math.round((completedStages / execution.stages.length) * 100);
  };

  // Deployment columns
  const deploymentColumns = [
    {
      title: 'Deployment ID',
      dataIndex: 'deploymentId',
      key: 'deploymentId',
      render: (id: string) => (
        <Button 
          type="link" 
          onClick={() => setSelectedDeployment(id)}
        >
          {id.slice(0, 8)}...
        </Button>
      )
    },
    {
      title: 'Version',
      dataIndex: 'version',
      key: 'version',
      render: (version: string) => <Tag color="blue">{version}</Tag>
    },
    {
      title: 'Environment',
      dataIndex: 'environment',
      key: 'environment',
      render: (env: string) => {
        const color = env === 'production' ? 'red' : 
                     env === 'staging' ? 'orange' : 'green';
        return <Tag color={color}>{env}</Tag>;
      }
    },
    {
      title: 'Status',
      dataIndex: 'status',
      key: 'status',
      render: (status: string) => (
        <Tag color={getStatusColor(status)} icon={
          status === 'deployed' ? <CheckCircleOutlined /> :
          status === 'deploying' ? <ClockCircleOutlined /> :
          status === 'failed' ? <ExclamationCircleOutlined /> :
          <ClockCircleOutlined />
        }>
          {status}
        </Tag>
      )
    },
    {
      title: 'Deployed At',
      dataIndex: 'deployedAt',
      key: 'deployedAt',
      render: (date: Date) => new Date(date).toLocaleString()
    },
    {
      title: 'Actions',
      key: 'actions',
      render: (_: any, record: any) => (
        <Space>
          <Tooltip title="View Details">
            <Button
              size="small"
              icon={<MonitorOutlined />}
              onClick={() => setSelectedDeployment(record.deploymentId)}
            />
          </Tooltip>
          {record.status === 'deploying' && (
            <Tooltip title="Stop Deployment">
              <Button
                size="small"
                danger
                icon={<StopOutlined />}
                onClick={() => stopDeployment(record.deploymentId)}
              />
            </Tooltip>
          )}
        </Space>
      )
    }
  ];

  // Pipeline execution columns
  const pipelineColumns = [
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
      title: 'Version',
      dataIndex: 'version',
      key: 'version',
      render: (version: string) => <Tag color="blue">{version}</Tag>
    },
    {
      title: 'Progress',
      key: 'progress',
      render: (_: any, record: any) => {
        const percentage = getProgressPercentage(record);
        return (
          <Progress 
            percent={percentage} 
            size="small" 
            status={record.status === 'failed' ? 'exception' : 'active'}
          />
        );
      }
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
      title: 'Started At',
      dataIndex: 'startedAt',
      key: 'startedAt',
      render: (date: Date) => new Date(date).toLocaleString()
    },
    {
      title: 'Actions',
      key: 'actions',
      render: (_: any, record: any) => (
        <Space>
          {record.status === 'running' && (
            <Button
              size="small"
              icon={<PauseCircleOutlined />}
              onClick={() => pauseExecution(record.executionId)}
            />
          )}
          {record.status === 'paused' && (
            <Button
              size="small"
              icon={<PlayCircleOutlined />}
              onClick={() => resumeExecution(record.executionId)}
            />
          )}
          {['running', 'paused'].includes(record.status) && (
            <Button
              size="small"
              danger
              icon={<StopOutlined />}
              onClick={() => cancelExecution(record.executionId)}
            />
          )}
        </Space>
      )
    }
  ];

  const latestVersion = selectedStrategy ? getLatestVersion(selectedStrategy) : null;

  return (
    <div className="strategy-deployment-dashboard">
      <div style={{ marginBottom: 24 }}>
        <Row gutter={[16, 16]} align="middle">
          <Col>
            <Title level={3}>Strategy Deployment Dashboard</Title>
          </Col>
          <Col>
            <Select
              style={{ width: 300 }}
              placeholder="Select Strategy"
              value={selectedStrategy}
              onChange={setSelectedStrategy}
            >
              {/* This would be populated from strategy list */}
              <Option value="strategy-1">EMA Cross Strategy</Option>
              <Option value="strategy-2">Momentum Strategy</Option>
              <Option value="strategy-3">Mean Reversion Strategy</Option>
            </Select>
          </Col>
          <Col>
            <Space>
              <Button
                type="primary"
                icon={<RocketOutlined />}
                onClick={() => setDeployModalVisible(true)}
                disabled={!selectedStrategy}
              >
                Deploy Strategy
              </Button>
              <Button
                icon={<GitlabOutlined />}
                onClick={() => setPipelineModalVisible(true)}
                disabled={!selectedStrategy}
              >
                Create Pipeline
              </Button>
            </Space>
          </Col>
        </Row>
      </div>

      {/* Connection Status Alert */}
      {connectionStatus !== 'connected' && (
        <Alert
          message={`Pipeline Monitoring ${connectionStatus}`}
          type="warning"
          showIcon
          style={{ marginBottom: 16 }}
        />
      )}

      {/* Stats Overview */}
      {deploymentStats && (
        <Row gutter={16} style={{ marginBottom: 24 }}>
          <Col span={4}>
            <Card>
              <Statistic
                title="Total Deployments"
                value={deploymentStats.totalDeployments}
                prefix={<RocketOutlined />}
              />
            </Card>
          </Col>
          <Col span={4}>
            <Card>
              <Statistic
                title="Success Rate"
                value={deploymentStats.successRate}
                suffix="%"
                precision={1}
                valueStyle={{ color: '#3f8600' }}
                prefix={<CheckCircleOutlined />}
              />
            </Card>
          </Col>
          <Col span={4}>
            <Card>
              <Statistic
                title="Active Pipelines"
                value={deploymentStats.activePipelines}
                valueStyle={{ color: '#1890ff' }}
                prefix={<MonitorOutlined />}
              />
            </Card>
          </Col>
          <Col span={4}>
            <Card>
              <Statistic
                title="Pending Requests"
                value={deploymentStats.pendingRequests}
                valueStyle={{ color: '#faad14' }}
                prefix={<ClockCircleOutlined />}
              />
            </Card>
          </Col>
          <Col span={4}>
            <Card>
              <Statistic
                title="Active Executions"
                value={getActiveExecutionsCount()}
                valueStyle={{ color: '#722ed1' }}
                prefix={<PlayCircleOutlined />}
              />
            </Card>
          </Col>
          <Col span={4}>
            <Card>
              <Statistic
                title="Unack. Alerts"
                value={getUnacknowledgedAlertsCount()}
                valueStyle={{ color: '#f5222d' }}
                prefix={<AlertOutlined />}
              />
            </Card>
          </Col>
        </Row>
      )}

      {/* Main Content Tabs */}
      <Tabs defaultActiveKey={defaultTab}>
        <TabPane tab="Overview" key="overview">
          <Row gutter={16}>
            <Col span={12}>
              <Card title="Recent Deployments" size="small">
                <Table
                  dataSource={deployments.slice(0, 10)}
                  columns={deploymentColumns}
                  size="small"
                  pagination={false}
                  loading={deploymentLoading}
                  rowKey="deploymentId"
                />
              </Card>
            </Col>
            <Col span={12}>
              <Card title="Pipeline Executions" size="small">
                <Table
                  dataSource={pipelineExecutions.slice(0, 10)}
                  columns={pipelineColumns}
                  size="small"
                  pagination={false}
                  loading={monitoringLoading}
                  rowKey="executionId"
                />
              </Card>
            </Col>
          </Row>
        </TabPane>

        <TabPane tab="Deployments" key="deployments">
          <Card title="All Deployments" extra={
            <Button 
              icon={<ReloadOutlined />} 
              onClick={() => fetchDeployments({ strategyId: selectedStrategy })}
              loading={deploymentLoading}
            >
              Refresh
            </Button>
          }>
            <Table
              dataSource={deployments}
              columns={deploymentColumns}
              loading={deploymentLoading}
              rowKey="deploymentId"
              pagination={{ pageSize: 20 }}
            />
          </Card>
        </TabPane>

        <TabPane tab="Pipelines" key="pipelines">
          <Card title="Pipeline Executions">
            <Table
              dataSource={pipelineExecutions}
              columns={pipelineColumns}
              loading={monitoringLoading}
              rowKey="executionId"
              pagination={{ pageSize: 20 }}
            />
          </Card>
        </TabPane>

        <TabPane tab="Requests" key="requests">
          <Card title="Deployment Requests">
            <Table
              dataSource={deploymentRequests}
              columns={[
                {
                  title: 'Request ID',
                  dataIndex: 'requestId',
                  key: 'requestId',
                  render: (id: string) => <Text code>{id.slice(0, 8)}...</Text>
                },
                {
                  title: 'Version',
                  dataIndex: 'version',
                  key: 'version',
                  render: (version: string) => <Tag color="blue">{version}</Tag>
                },
                {
                  title: 'Target Environment',
                  dataIndex: 'targetEnvironment',
                  key: 'targetEnvironment',
                  render: (env: string) => <Tag>{env}</Tag>
                },
                {
                  title: 'Status',
                  dataIndex: 'approvalStatus',
                  key: 'approvalStatus',
                  render: (status: string) => (
                    <Tag color={status === 'approved' ? 'green' : 'orange'}>
                      {status}
                    </Tag>
                  )
                },
                {
                  title: 'Requested At',
                  dataIndex: 'requestedAt',
                  key: 'requestedAt',
                  render: (date: Date) => new Date(date).toLocaleString()
                },
                {
                  title: 'Actions',
                  key: 'actions',
                  render: (_: any, record: any) => (
                    <Space>
                      {record.approvalStatus === 'pending' && (
                        <>
                          <Button
                            size="small"
                            type="primary"
                            onClick={() => approveDeploymentRequest(record.requestId, 'user', true)}
                          >
                            Approve
                          </Button>
                          <Button
                            size="small"
                            danger
                            onClick={() => approveDeploymentRequest(record.requestId, 'user', false)}
                          >
                            Reject
                          </Button>
                        </>
                      )}
                    </Space>
                  )
                }
              ]}
              rowKey="requestId"
              pagination={{ pageSize: 20 }}
            />
          </Card>
        </TabPane>
      </Tabs>

      {/* Deploy Strategy Modal */}
      <Modal
        title="Deploy Strategy"
        open={deployModalVisible}
        onCancel={() => setDeployModalVisible(false)}
        footer={null}
        width={600}
      >
        <Form
          form={form}
          layout="vertical"
          onFinish={handleDeploy}
          initialValues={{
            version: latestVersion?.version,
            strategy: 'direct',
            autoRollback: true,
            rollbackThreshold: 0.05
          }}
        >
          <Form.Item
            name="version"
            label="Version"
            rules={[{ required: true, message: 'Please select a version' }]}
          >
            <Select placeholder="Select version">
              {versions.map(version => (
                <Option key={version.versionId} value={version.version}>
                  {version.version} ({version.branch})
                </Option>
              ))}
            </Select>
          </Form.Item>

          <Form.Item
            name="environment"
            label="Target Environment"
            rules={[{ required: true, message: 'Please select an environment' }]}
          >
            <Select placeholder="Select environment">
              {environments.map(env => (
                <Option key={env.id} value={env.id}>
                  {env.name} ({env.type})
                </Option>
              ))}
            </Select>
          </Form.Item>

          <Form.Item
            name="strategy"
            label="Deployment Strategy"
          >
            <Select>
              <Option value="direct">Direct</Option>
              <Option value="blue_green">Blue-Green</Option>
              <Option value="canary">Canary</Option>
              <Option value="rolling">Rolling</Option>
            </Select>
          </Form.Item>

          <Form.Item
            name="autoRollback"
            label="Auto Rollback"
            valuePropName="checked"
          >
            <Switch />
          </Form.Item>

          <Form.Item
            name="rollbackThreshold"
            label="Rollback Threshold (%)"
          >
            <Input type="number" min={0} max={100} />
          </Form.Item>

          <Form.Item>
            <Space>
              <Button type="primary" htmlType="submit" loading={deploymentLoading}>
                Deploy
              </Button>
              <Button onClick={() => setDeployModalVisible(false)}>
                Cancel
              </Button>
            </Space>
          </Form.Item>
        </Form>
      </Modal>

      {/* Create Pipeline Modal */}
      <Modal
        title="Create Deployment Pipeline"
        open={pipelineModalVisible}
        onCancel={() => setPipelineModalVisible(false)}
        footer={null}
        width={600}
      >
        <Form
          form={pipelineForm}
          layout="vertical"
          onFinish={handleCreatePipeline}
          initialValues={{
            version: latestVersion?.version,
            environments: ['development', 'testing', 'staging']
          }}
        >
          <Form.Item
            name="version"
            label="Version"
            rules={[{ required: true, message: 'Please select a version' }]}
          >
            <Select placeholder="Select version">
              {versions.map(version => (
                <Option key={version.versionId} value={version.version}>
                  {version.version} ({version.branch})
                </Option>
              ))}
            </Select>
          </Form.Item>

          <Form.Item
            name="environments"
            label="Target Environments (in order)"
            rules={[{ required: true, message: 'Please select environments' }]}
          >
            <Select mode="multiple" placeholder="Select environments in deployment order">
              {environments.map(env => (
                <Option key={env.id} value={env.id}>
                  {env.name} ({env.type})
                </Option>
              ))}
            </Select>
          </Form.Item>

          <Form.Item>
            <Space>
              <Button type="primary" htmlType="submit" loading={deploymentLoading}>
                Create Pipeline
              </Button>
              <Button onClick={() => setPipelineModalVisible(false)}>
                Cancel
              </Button>
            </Space>
          </Form.Item>
        </Form>
      </Modal>
    </div>
  );
};