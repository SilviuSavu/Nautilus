/**
 * Deployment Pipeline Component
 * Visual CI/CD pipeline with approval workflows
 */

import React, { useState, useEffect } from 'react';
import {
  Card,
  Row,
  Col,
  Steps,
  Button,
  Tag,
  Space,
  Modal,
  Form,
  Select,
  Input,
  Typography,
  Progress,
  Timeline,
  Alert,
  Tooltip,
  Drawer,
  Tabs,
  List,
  Avatar,
  Comment,
  Rate,
  Divider,
  Statistic,
  Table,
  Badge
} from 'antd';
import {
  PlayCircleOutlined,
  PauseCircleOutlined,
  StopOutlined,
  CheckCircleOutlined,
  CloseCircleOutlined,
  ClockCircleOutlined,
  WarningOutlined,
  RocketOutlined,
  BugOutlined,
  SettingOutlined,
  UserOutlined,
  EyeOutlined,
  ThunderboltOutlined,
  SafetyOutlined,
  DatabaseOutlined,
  CloudServerOutlined,
  MonitorOutlined
} from '@ant-design/icons';
import { usePipelineMonitoring } from '../../hooks/strategy/usePipelineMonitoring';
import { useStrategyDeployment } from '../../hooks/strategy/useStrategyDeployment';

const { Title, Text, Paragraph } = Typography;
const { Step } = Steps;
const { TabPane } = Tabs;
const { Option } = Select;
const { TextArea } = Input;

interface DeploymentPipelineProps {
  pipelineId?: string;
  strategyId?: string;
  autoRefresh?: boolean;
  onStageClick?: (stageId: string) => void;
}

export const DeploymentPipeline: React.FC<DeploymentPipelineProps> = ({
  pipelineId,
  strategyId,
  autoRefresh = true,
  onStageClick
}) => {
  const [selectedExecution, setSelectedExecution] = useState<string | null>(null);
  const [selectedStage, setSelectedStage] = useState<string | null>(null);
  const [stageDetailsVisible, setStageDetailsVisible] = useState(false);
  const [executionDetailsVisible, setExecutionDetailsVisible] = useState(false);
  const [approvalModalVisible, setApprovalModalVisible] = useState(false);
  const [currentPipelineId, setCurrentPipelineId] = useState<string>(pipelineId || '');

  const [approvalForm] = Form.useForm();

  const {
    pipelineExecutions,
    pipelineAlerts,
    pipelineMetrics,
    activeExecutions,
    connectionStatus,
    loading,
    pauseExecution,
    resumeExecution,
    cancelExecution,
    retryStage,
    acknowledgeAlert,
    getExecution,
    getMetrics,
    fetchPipelineExecutions
  } = usePipelineMonitoring();

  const {
    environments,
    deploymentStats,
    fetchDeployments
  } = useStrategyDeployment();

  // Initialize data
  useEffect(() => {
    if (currentPipelineId) {
      fetchPipelineExecutions({ pipelineId: currentPipelineId });
    } else if (strategyId) {
      fetchPipelineExecutions({ strategyId });
    }
  }, [currentPipelineId, strategyId, fetchPipelineExecutions]);

  // Auto-refresh data
  useEffect(() => {
    if (!autoRefresh) return;

    const interval = setInterval(() => {
      if (currentPipelineId) {
        fetchPipelineExecutions({ pipelineId: currentPipelineId });
      } else if (strategyId) {
        fetchPipelineExecutions({ strategyId });
      }
    }, 10000); // Every 10 seconds

    return () => clearInterval(interval);
  }, [autoRefresh, currentPipelineId, strategyId, fetchPipelineExecutions]);

  // Get stage status icon
  const getStageIcon = (status: string) => {
    switch (status) {
      case 'completed':
        return <CheckCircleOutlined style={{ color: '#52c41a' }} />;
      case 'failed':
        return <CloseCircleOutlined style={{ color: '#f5222d' }} />;
      case 'running':
        return <PlayCircleOutlined style={{ color: '#1890ff' }} />;
      case 'pending':
        return <ClockCircleOutlined style={{ color: '#faad14' }} />;
      case 'skipped':
        return <WarningOutlined style={{ color: '#d9d9d9' }} />;
      default:
        return <ClockCircleOutlined style={{ color: '#d9d9d9' }} />;
    }
  };

  // Get stage type icon
  const getStageTypeIcon = (type: string) => {
    switch (type) {
      case 'test':
        return <BugOutlined />;
      case 'deploy':
        return <RocketOutlined />;
      case 'verify':
        return <SafetyOutlined />;
      case 'approve':
        return <UserOutlined />;
      case 'rollback':
        return <ThunderboltOutlined />;
      default:
        return <SettingOutlined />;
    }
  };

  // Get execution status color
  const getExecutionStatusColor = (status: string) => {
    switch (status) {
      case 'completed':
        return 'success';
      case 'failed':
        return 'error';
      case 'running':
        return 'active';
      case 'paused':
        return 'normal';
      default:
        return 'normal';
    }
  };

  // Calculate execution progress
  const calculateProgress = (execution: any) => {
    if (!execution.stages || execution.stages.length === 0) return 0;
    
    const completedStages = execution.stages.filter((stage: any) => 
      stage.status === 'completed'
    ).length;
    
    const failedStages = execution.stages.filter((stage: any) => 
      stage.status === 'failed'
    ).length;

    if (failedStages > 0) {
      return {
        percent: Math.round((completedStages / execution.stages.length) * 100),
        status: 'exception' as const
      };
    }

    return {
      percent: Math.round((completedStages / execution.stages.length) * 100),
      status: execution.status === 'completed' ? 'success' as const : 'active' as const
    };
  };

  // Handle stage click
  const handleStageClick = (stageId: string, executionId: string) => {
    setSelectedStage(stageId);
    setSelectedExecution(executionId);
    setStageDetailsVisible(true);
    onStageClick?.(stageId);
  };

  // Handle execution click
  const handleExecutionClick = (executionId: string) => {
    setSelectedExecution(executionId);
    setExecutionDetailsVisible(true);
  };

  // Render pipeline execution
  const renderPipelineExecution = (execution: any) => {
    const progress = calculateProgress(execution);
    const currentStageIndex = execution.stages.findIndex((stage: any) => 
      stage.status === 'running' || stage.status === 'pending'
    );

    return (
      <Card
        key={execution.executionId}
        style={{ marginBottom: 16 }}
        title={
          <Space>
            <Text strong>Execution {execution.executionId.slice(0, 8)}...</Text>
            <Tag color="blue">{execution.version}</Tag>
            <Tag color="green">{execution.environment}</Tag>
            <Badge
              status={getExecutionStatusColor(execution.status)}
              text={execution.status}
            />
          </Space>
        }
        extra={
          <Space>
            {execution.status === 'running' && (
              <>
                <Button
                  size="small"
                  icon={<PauseCircleOutlined />}
                  onClick={() => pauseExecution(execution.executionId)}
                >
                  Pause
                </Button>
                <Button
                  size="small"
                  danger
                  icon={<StopOutlined />}
                  onClick={() => cancelExecution(execution.executionId)}
                >
                  Cancel
                </Button>
              </>
            )}
            {execution.status === 'paused' && (
              <Button
                size="small"
                type="primary"
                icon={<PlayCircleOutlined />}
                onClick={() => resumeExecution(execution.executionId)}
              >
                Resume
              </Button>
            )}
            <Button
              size="small"
              icon={<EyeOutlined />}
              onClick={() => handleExecutionClick(execution.executionId)}
            >
              Details
            </Button>
          </Space>
        }
      >
        <Row gutter={16} style={{ marginBottom: 16 }}>
          <Col span={12}>
            <Progress {...progress} />
          </Col>
          <Col span={6}>
            <Statistic
              title="Duration"
              value={execution.totalDuration ? Math.round(execution.totalDuration / 1000) : 0}
              suffix="s"
              valueStyle={{ fontSize: 14 }}
            />
          </Col>
          <Col span={6}>
            <Statistic
              title="Triggered By"
              value={execution.triggeredBy}
              valueStyle={{ fontSize: 12 }}
            />
          </Col>
        </Row>

        <Steps
          current={currentStageIndex >= 0 ? currentStageIndex : execution.stages.length}
          status={execution.status === 'failed' ? 'error' : 'process'}
          size="small"
        >
          {execution.stages.map((stage: any, index: number) => (
            <Step
              key={stage.stageId}
              title={
                <Button
                  type="link"
                  size="small"
                  onClick={() => handleStageClick(stage.stageId, execution.executionId)}
                  style={{ padding: 0, height: 'auto' }}
                >
                  {stage.name}
                </Button>
              }
              description={
                <div>
                  <Space>
                    {getStageTypeIcon(stage.type)}
                    <Tag color={stage.status === 'completed' ? 'green' : 
                               stage.status === 'failed' ? 'red' : 
                               stage.status === 'running' ? 'blue' : 'default'}>
                      {stage.status}
                    </Tag>
                    {stage.duration && (
                      <Text type="secondary">{Math.round(stage.duration / 1000)}s</Text>
                    )}
                  </Space>
                  {stage.retryCount > 0 && (
                    <div>
                      <Text type="secondary">Retries: {stage.retryCount}/{stage.maxRetries}</Text>
                    </div>
                  )}
                </div>
              }
              icon={getStageIcon(stage.status)}
            />
          ))}
        </Steps>

        {execution.metadata && Object.keys(execution.metadata).length > 0 && (
          <div style={{ marginTop: 16 }}>
            <Text type="secondary">
              Started: {new Date(execution.startedAt).toLocaleString()}
              {execution.completedAt && (
                <> • Completed: {new Date(execution.completedAt).toLocaleString()}</>
              )}
            </Text>
          </div>
        )}
      </Card>
    );
  };

  // Render pipeline metrics
  const renderPipelineMetrics = () => {
    const metrics = currentPipelineId ? getMetrics(currentPipelineId) : null;
    if (!metrics) return null;

    return (
      <Row gutter={16} style={{ marginBottom: 16 }}>
        <Col span={6}>
          <Card size="small">
            <Statistic
              title="Success Rate"
              value={metrics.successRate}
              suffix="%"
              precision={1}
              valueStyle={{ color: metrics.successRate >= 90 ? '#3f8600' : '#faad14' }}
            />
          </Card>
        </Col>
        <Col span={6}>
          <Card size="small">
            <Statistic
              title="Avg Duration"
              value={Math.round(metrics.averageDuration / 1000)}
              suffix="s"
            />
          </Card>
        </Col>
        <Col span={6}>
          <Card size="small">
            <Statistic
              title="Total Executions"
              value={metrics.totalExecutions}
            />
          </Card>
        </Col>
        <Col span={6}>
          <Card size="small">
            <Statistic
              title="Failed Executions"
              value={metrics.failedExecutions}
              valueStyle={{ color: '#f5222d' }}
            />
          </Card>
        </Col>
      </Row>
    );
  };

  // Get selected execution
  const selectedExecutionData = selectedExecution ? getExecution(selectedExecution) : null;
  const selectedStageData = selectedExecutionData?.stages.find((stage: any) => 
    stage.stageId === selectedStage
  );

  return (
    <div className="deployment-pipeline">
      <div style={{ marginBottom: 24 }}>
        <Row gutter={[16, 16]} align="middle">
          <Col>
            <Title level={3}>
              <DatabaseOutlined /> Deployment Pipeline
            </Title>
          </Col>
          <Col>
            <Space>
              <Select
                style={{ width: 200 }}
                placeholder="Select Pipeline"
                value={currentPipelineId}
                onChange={setCurrentPipelineId}
                allowClear
              >
                {/* This would be populated from available pipelines */}
                <Option value="pipeline-1">Main Pipeline</Option>
                <Option value="pipeline-2">Feature Pipeline</Option>
                <Option value="pipeline-3">Hotfix Pipeline</Option>
              </Select>
              <Badge status={connectionStatus === 'connected' ? 'success' : 'error'} />
              <Text type="secondary">
                {connectionStatus === 'connected' ? 'Real-time' : 'Disconnected'}
              </Text>
            </Space>
          </Col>
        </Row>
      </div>

      {/* Connection Status Alert */}
      {connectionStatus !== 'connected' && (
        <Alert
          message={`Pipeline monitoring ${connectionStatus}`}
          type="warning"
          showIcon
          style={{ marginBottom: 16 }}
        />
      )}

      {/* Pipeline Metrics */}
      {renderPipelineMetrics()}

      {/* Active Alerts */}
      {pipelineAlerts.filter(alert => !alert.acknowledged).length > 0 && (
        <Card 
          title="Active Alerts" 
          size="small" 
          style={{ marginBottom: 16 }}
          type="inner"
        >
          <List
            size="small"
            dataSource={pipelineAlerts.filter(alert => !alert.acknowledged)}
            renderItem={alert => (
              <List.Item
                actions={[
                  <Button
                    key="ack"
                    size="small"
                    onClick={() => acknowledgeAlert(alert.alertId, 'user')}
                  >
                    Acknowledge
                  </Button>
                ]}
              >
                <List.Item.Meta
                  avatar={
                    <Avatar
                      style={{
                        backgroundColor: alert.severity === 'critical' ? '#f5222d' :
                                        alert.severity === 'high' ? '#fa8c16' :
                                        alert.severity === 'medium' ? '#fadb14' : '#52c41a'
                      }}
                      size="small"
                    >
                      {alert.severity === 'critical' ? '!' : 
                       alert.severity === 'high' ? 'H' :
                       alert.severity === 'medium' ? 'M' : 'L'}
                    </Avatar>
                  }
                  title={alert.title}
                  description={
                    <div>
                      <Paragraph ellipsis={{ rows: 1 }}>{alert.message}</Paragraph>
                      <Text type="secondary">{new Date(alert.timestamp).toLocaleString()}</Text>
                    </div>
                  }
                />
              </List.Item>
            )}
          />
        </Card>
      )}

      {/* Pipeline Executions */}
      <Card title="Pipeline Executions" loading={loading}>
        {pipelineExecutions.length > 0 ? (
          <div>
            {pipelineExecutions.map(renderPipelineExecution)}
          </div>
        ) : (
          <div style={{ textAlign: 'center', padding: '40px 0' }}>
            <CloudServerOutlined style={{ fontSize: 48, color: '#d9d9d9' }} />
            <Title level={4} type="secondary">No Pipeline Executions</Title>
            <Paragraph type="secondary">
              Pipeline executions will appear here when strategies are deployed
            </Paragraph>
          </div>
        )}
      </Card>

      {/* Stage Details Drawer */}
      <Drawer
        title="Stage Details"
        placement="right"
        width={600}
        open={stageDetailsVisible}
        onClose={() => setStageDetailsVisible(false)}
      >
        {selectedStageData && (
          <div>
            <Space direction="vertical" style={{ width: '100%' }}>
              <Card size="small" title="Stage Information">
                <Row gutter={16}>
                  <Col span={8}>
                    <Text strong>Name:</Text><br />
                    <Text>{selectedStageData.name}</Text>
                  </Col>
                  <Col span={8}>
                    <Text strong>Type:</Text><br />
                    <Space>
                      {getStageTypeIcon(selectedStageData.type)}
                      <Text>{selectedStageData.type}</Text>
                    </Space>
                  </Col>
                  <Col span={8}>
                    <Text strong>Status:</Text><br />
                    <Space>
                      {getStageIcon(selectedStageData.status)}
                      <Tag color={selectedStageData.status === 'completed' ? 'green' : 
                                 selectedStageData.status === 'failed' ? 'red' : 'blue'}>
                        {selectedStageData.status}
                      </Tag>
                    </Space>
                  </Col>
                </Row>
                
                {selectedStageData.startedAt && (
                  <div style={{ marginTop: 16 }}>
                    <Text type="secondary">
                      Started: {new Date(selectedStageData.startedAt).toLocaleString()}
                      {selectedStageData.completedAt && (
                        <> • Duration: {Math.round(selectedStageData.duration / 1000)}s</>
                      )}
                    </Text>
                  </div>
                )}

                {selectedStageData.retryCount > 0 && (
                  <div style={{ marginTop: 8 }}>
                    <Text type="secondary">
                      Retries: {selectedStageData.retryCount}/{selectedStageData.maxRetries}
                    </Text>
                  </div>
                )}
              </Card>

              {selectedStageData.logs && selectedStageData.logs.length > 0 && (
                <Card size="small" title="Stage Logs">
                  <pre style={{ 
                    backgroundColor: '#f6f8fa', 
                    padding: 16, 
                    borderRadius: 6,
                    maxHeight: 300,
                    overflow: 'auto',
                    fontSize: 12,
                    fontFamily: 'monospace'
                  }}>
                    {selectedStageData.logs.join('\n')}
                  </pre>
                </Card>
              )}

              {selectedStageData.metrics && Object.keys(selectedStageData.metrics).length > 0 && (
                <Card size="small" title="Stage Metrics">
                  <Row gutter={8}>
                    {Object.entries(selectedStageData.metrics).map(([key, value]) => (
                      <Col span={12} key={key}>
                        <Statistic
                          title={key}
                          value={value as number}
                          precision={2}
                          valueStyle={{ fontSize: 14 }}
                        />
                      </Col>
                    ))}
                  </Row>
                </Card>
              )}

              {selectedStageData.status === 'failed' && (
                <Card size="small" title="Actions">
                  <Space>
                    <Button
                      type="primary"
                      icon={<PlayCircleOutlined />}
                      onClick={() => retryStage(selectedExecution!, selectedStageData.stageId)}
                    >
                      Retry Stage
                    </Button>
                  </Space>
                </Card>
              )}
            </Space>
          </div>
        )}
      </Drawer>

      {/* Execution Details Drawer */}
      <Drawer
        title="Execution Details"
        placement="right"
        width={800}
        open={executionDetailsVisible}
        onClose={() => setExecutionDetailsVisible(false)}
      >
        {selectedExecutionData && (
          <div>
            <Tabs defaultActiveKey="overview">
              <TabPane tab="Overview" key="overview">
                <Space direction="vertical" style={{ width: '100%' }}>
                  <Card size="small" title="Execution Information">
                    <Row gutter={16}>
                      <Col span={6}>
                        <Text strong>ID:</Text><br />
                        <Text code>{selectedExecutionData.executionId}</Text>
                      </Col>
                      <Col span={6}>
                        <Text strong>Version:</Text><br />
                        <Tag color="blue">{selectedExecutionData.version}</Tag>
                      </Col>
                      <Col span={6}>
                        <Text strong>Environment:</Text><br />
                        <Tag color="green">{selectedExecutionData.environment}</Tag>
                      </Col>
                      <Col span={6}>
                        <Text strong>Status:</Text><br />
                        <Badge
                          status={getExecutionStatusColor(selectedExecutionData.status)}
                          text={selectedExecutionData.status}
                        />
                      </Col>
                    </Row>
                    
                    <Divider />
                    
                    <Row gutter={16}>
                      <Col span={12}>
                        <Text strong>Started:</Text><br />
                        <Text>{new Date(selectedExecutionData.startedAt).toLocaleString()}</Text>
                      </Col>
                      <Col span={12}>
                        <Text strong>Triggered By:</Text><br />
                        <Space>
                          <Avatar size="small" icon={<UserOutlined />} />
                          <Text>{selectedExecutionData.triggeredBy}</Text>
                        </Space>
                      </Col>
                    </Row>

                    {selectedExecutionData.totalDuration && (
                      <div style={{ marginTop: 16 }}>
                        <Text strong>Duration:</Text><br />
                        <Text>{Math.round(selectedExecutionData.totalDuration / 1000)}s</Text>
                      </div>
                    )}
                  </Card>

                  <Card size="small" title="Stage Progress">
                    <Timeline>
                      {selectedExecutionData.stages.map((stage: any) => (
                        <Timeline.Item
                          key={stage.stageId}
                          dot={getStageIcon(stage.status)}
                          color={stage.status === 'completed' ? 'green' : 
                                 stage.status === 'failed' ? 'red' : 
                                 stage.status === 'running' ? 'blue' : 'gray'}
                        >
                          <Space direction="vertical">
                            <Space>
                              <Text strong>{stage.name}</Text>
                              {getStageTypeIcon(stage.type)}
                              <Tag color={stage.status === 'completed' ? 'green' : 
                                         stage.status === 'failed' ? 'red' : 'blue'}>
                                {stage.status}
                              </Tag>
                            </Space>
                            {stage.startedAt && (
                              <Text type="secondary">
                                {new Date(stage.startedAt).toLocaleString()}
                                {stage.duration && ` • ${Math.round(stage.duration / 1000)}s`}
                              </Text>
                            )}
                            {stage.retryCount > 0 && (
                              <Text type="secondary">
                                Retried {stage.retryCount}/{stage.maxRetries} times
                              </Text>
                            )}
                          </Space>
                        </Timeline.Item>
                      ))}
                    </Timeline>
                  </Card>
                </Space>
              </TabPane>

              <TabPane tab="Stages" key="stages">
                <Table
                  dataSource={selectedExecutionData.stages}
                  size="small"
                  pagination={false}
                  columns={[
                    {
                      title: 'Name',
                      dataIndex: 'name',
                      key: 'name',
                      render: (name: string, record: any) => (
                        <Space>
                          {getStageTypeIcon(record.type)}
                          <Text>{name}</Text>
                        </Space>
                      )
                    },
                    {
                      title: 'Status',
                      dataIndex: 'status',
                      key: 'status',
                      render: (status: string) => (
                        <Space>
                          {getStageIcon(status)}
                          <Tag color={status === 'completed' ? 'green' : 
                                     status === 'failed' ? 'red' : 'blue'}>
                            {status}
                          </Tag>
                        </Space>
                      )
                    },
                    {
                      title: 'Duration',
                      dataIndex: 'duration',
                      key: 'duration',
                      render: (duration?: number) => duration ? `${Math.round(duration / 1000)}s` : '-'
                    },
                    {
                      title: 'Retries',
                      key: 'retries',
                      render: (_: any, record: any) => `${record.retryCount}/${record.maxRetries}`
                    },
                    {
                      title: 'Actions',
                      key: 'actions',
                      render: (_: any, record: any) => (
                        <Space>
                          <Button
                            size="small"
                            onClick={() => handleStageClick(record.stageId, selectedExecutionData.executionId)}
                          >
                            View
                          </Button>
                          {record.status === 'failed' && (
                            <Button
                              size="small"
                              type="primary"
                              onClick={() => retryStage(selectedExecutionData.executionId, record.stageId)}
                            >
                              Retry
                            </Button>
                          )}
                        </Space>
                      )
                    }
                  ]}
                  rowKey="stageId"
                />
              </TabPane>

              <TabPane tab="Metadata" key="metadata">
                <Card size="small">
                  <pre style={{ 
                    backgroundColor: '#f6f8fa', 
                    padding: 16, 
                    borderRadius: 6
                  }}>
                    {JSON.stringify(selectedExecutionData.metadata, null, 2)}
                  </pre>
                </Card>
              </TabPane>
            </Tabs>
          </div>
        )}
      </Drawer>
    </div>
  );
};