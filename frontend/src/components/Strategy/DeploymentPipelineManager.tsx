import React, { useState, useEffect } from 'react';
import {
  Card,
  Row,
  Col,
  Steps,
  Timeline,
  Table,
  Tag,
  Space,
  Typography,
  Button,
  Modal,
  Form,
  Input,
  Select,
  Switch,
  Alert,
  Badge,
  Tooltip,
  Progress,
  List,
  Divider,
  Statistic,
  Tabs,
  Dropdown,
  message
} from 'antd';
import {
  RocketOutlined,
  PlayCircleOutlined,
  PauseCircleOutlined,
  StopOutlined,
  CheckCircleOutlined,
  ExclamationCircleOutlined,
  CloseCircleOutlined,
  ClockCircleOutlined,
  SettingOutlined,
  PlusOutlined,
  EditOutlined,
  DeleteOutlined,
  EyeOutlined,
  DownloadOutlined,
  RollbackOutlined,
  GitlabOutlined,
  DeploymentUnitOutlined,
  FileTextOutlined,
  BugOutlined,
  SafetyCertificateOutlined,
  UserOutlined,
  HistoryOutlined,
  ReloadOutlined
} from '@ant-design/icons';
import type {
  StrategyDeploymentPipeline,
  DeploymentStage,
  DeploymentRequirement,
  DeploymentLog
} from '../../types/sprint3';

const { Text, Title } = Typography;
const { TextArea } = Input;
const { Option } = Select;
const { Step } = Steps;
const { TabPane } = Tabs;

interface DeploymentPipelineManagerProps {
  compactMode?: boolean;
  showHistory?: boolean;
}

const DeploymentPipelineManager: React.FC<DeploymentPipelineManagerProps> = ({
  compactMode = false,
  showHistory = true
}) => {
  const [pipelines, setPipelines] = useState<StrategyDeploymentPipeline[]>([]);
  const [selectedPipeline, setSelectedPipeline] = useState<string | null>(null);
  const [modalVisible, setModalVisible] = useState(false);
  const [deployModalVisible, setDeployModalVisible] = useState(false);
  const [form] = Form.useForm();
  const [deployForm] = Form.useForm();

  // Initialize mock data
  useEffect(() => {
    const mockPipelines: StrategyDeploymentPipeline[] = [
      {
        id: 'pipeline-1',
        name: 'Momentum Strategy v2.1',
        version: '2.1.0',
        status: 'deploying',
        currentStage: 2,
        createdBy: 'john.doe@nautilus.com',
        createdAt: new Date(Date.now() - 7200000).toISOString(), // 2 hours ago
        lastModified: new Date(Date.now() - 1800000).toISOString(), // 30 min ago
        stages: [
          {
            id: 'validation',
            name: 'Validation',
            type: 'validation',
            status: 'completed',
            startTime: new Date(Date.now() - 7200000).toISOString(),
            endTime: new Date(Date.now() - 6600000).toISOString(),
            duration: 600,
            logs: [
              { id: 'log-1', timestamp: new Date().toISOString(), level: 'info', message: 'Configuration validation passed' }
            ],
            requirements: [
              { id: 'req-1', type: 'test', description: 'Unit tests pass', status: 'met' }
            ],
            artifacts: []
          },
          {
            id: 'testing',
            name: 'Testing',
            type: 'testing',
            status: 'completed',
            startTime: new Date(Date.now() - 6600000).toISOString(),
            endTime: new Date(Date.now() - 5400000).toISOString(),
            duration: 1200,
            logs: [
              { id: 'log-2', timestamp: new Date().toISOString(), level: 'info', message: 'Backtesting completed successfully' }
            ],
            requirements: [
              { id: 'req-2', type: 'test', description: 'Backtest performance acceptable', status: 'met' }
            ],
            artifacts: []
          },
          {
            id: 'approval',
            name: 'Approval',
            type: 'approval',
            status: 'running',
            startTime: new Date(Date.now() - 5400000).toISOString(),
            logs: [
              { id: 'log-3', timestamp: new Date().toISOString(), level: 'info', message: 'Waiting for approval from risk team' }
            ],
            requirements: [
              { id: 'req-3', type: 'approval', description: 'Risk team approval', status: 'pending' },
              { id: 'req-4', type: 'approval', description: 'Portfolio manager approval', status: 'met' }
            ],
            artifacts: []
          },
          {
            id: 'deployment',
            name: 'Deployment',
            type: 'deployment',
            status: 'pending',
            logs: [],
            requirements: [
              { id: 'req-5', type: 'resource', description: 'Deployment resources available', status: 'pending' }
            ],
            artifacts: []
          },
          {
            id: 'monitoring',
            name: 'Monitoring',
            type: 'monitoring',
            status: 'pending',
            logs: [],
            requirements: [
              { id: 'req-6', type: 'metric', description: 'Performance monitoring setup', status: 'pending' }
            ],
            artifacts: []
          }
        ]
      },
      {
        id: 'pipeline-2',
        name: 'Mean Reversion Strategy v1.3',
        version: '1.3.2',
        status: 'active',
        currentStage: 4,
        createdBy: 'jane.smith@nautilus.com',
        createdAt: new Date(Date.now() - 86400000).toISOString(), // 1 day ago
        deployedAt: new Date(Date.now() - 3600000).toISOString(), // 1 hour ago
        lastModified: new Date(Date.now() - 3600000).toISOString(),
        stages: [
          {
            id: 'validation',
            name: 'Validation',
            type: 'validation',
            status: 'completed',
            startTime: new Date(Date.now() - 86400000).toISOString(),
            endTime: new Date(Date.now() - 86000000).toISOString(),
            duration: 400,
            logs: [],
            requirements: [
              { id: 'req-1', type: 'test', description: 'Configuration validation passed', status: 'met' }
            ],
            artifacts: []
          },
          {
            id: 'testing',
            name: 'Testing',
            type: 'testing',
            status: 'completed',
            startTime: new Date(Date.now() - 86000000).toISOString(),
            endTime: new Date(Date.now() - 82800000).toISOString(),
            duration: 3200,
            logs: [],
            requirements: [
              { id: 'req-2', type: 'test', description: 'Backtest performance acceptable', status: 'met' }
            ],
            artifacts: []
          },
          {
            id: 'approval',
            name: 'Approval',
            type: 'approval',
            status: 'completed',
            startTime: new Date(Date.now() - 82800000).toISOString(),
            endTime: new Date(Date.now() - 7200000).toISOString(),
            duration: 75600,
            logs: [],
            requirements: [
              { id: 'req-3', type: 'approval', description: 'Risk team approval', status: 'met' },
              { id: 'req-4', type: 'approval', description: 'Portfolio manager approval', status: 'met' }
            ],
            artifacts: []
          },
          {
            id: 'deployment',
            name: 'Deployment',
            type: 'deployment',
            status: 'completed',
            startTime: new Date(Date.now() - 7200000).toISOString(),
            endTime: new Date(Date.now() - 3600000).toISOString(),
            duration: 3600,
            logs: [],
            requirements: [
              { id: 'req-5', type: 'resource', description: 'Deployment resources available', status: 'met' }
            ],
            artifacts: []
          },
          {
            id: 'monitoring',
            name: 'Monitoring',
            type: 'monitoring',
            status: 'running',
            startTime: new Date(Date.now() - 3600000).toISOString(),
            logs: [],
            requirements: [
              { id: 'req-6', type: 'metric', description: 'Performance monitoring setup', status: 'met' }
            ],
            artifacts: []
          }
        ]
      },
      {
        id: 'pipeline-3',
        name: 'Arbitrage Strategy v3.0',
        version: '3.0.0-beta',
        status: 'failed',
        currentStage: 1,
        createdBy: 'bob.wilson@nautilus.com',
        createdAt: new Date(Date.now() - 10800000).toISOString(), // 3 hours ago
        lastModified: new Date(Date.now() - 7200000).toISOString(), // 2 hours ago
        stages: [
          {
            id: 'validation',
            name: 'Validation',
            type: 'validation',
            status: 'completed',
            startTime: new Date(Date.now() - 10800000).toISOString(),
            endTime: new Date(Date.now() - 10200000).toISOString(),
            duration: 600,
            logs: [],
            requirements: [
              { id: 'req-1', type: 'test', description: 'Configuration validation passed', status: 'met' }
            ],
            artifacts: []
          },
          {
            id: 'testing',
            name: 'Testing',
            type: 'testing',
            status: 'failed',
            startTime: new Date(Date.now() - 10200000).toISOString(),
            endTime: new Date(Date.now() - 7200000).toISOString(),
            duration: 3000,
            logs: [
              { id: 'log-1', timestamp: new Date().toISOString(), level: 'error', message: 'Backtest failed: Insufficient historical data' }
            ],
            requirements: [
              { id: 'req-2', type: 'test', description: 'Backtest performance acceptable', status: 'failed' }
            ],
            artifacts: []
          }
        ]
      }
    ];

    setPipelines(mockPipelines);
    setSelectedPipeline(mockPipelines[0].id);
  }, []);

  // Get status display
  const getStatusDisplay = (status: StrategyDeploymentPipeline['status']) => {
    switch (status) {
      case 'draft': return { color: 'default', icon: <EditOutlined />, text: 'Draft' };
      case 'testing': return { color: 'processing', icon: <BugOutlined />, text: 'Testing' };
      case 'approval': return { color: 'warning', icon: <ClockCircleOutlined />, text: 'Approval' };
      case 'deploying': return { color: 'processing', icon: <RocketOutlined />, text: 'Deploying' };
      case 'active': return { color: 'success', icon: <CheckCircleOutlined />, text: 'Active' };
      case 'paused': return { color: 'default', icon: <PauseCircleOutlined />, text: 'Paused' };
      case 'retired': return { color: 'default', icon: <StopOutlined />, text: 'Retired' };
      case 'failed': return { color: 'error', icon: <CloseCircleOutlined />, text: 'Failed' };
      default: return { color: 'default', icon: <ClockCircleOutlined />, text: 'Unknown' };
    }
  };

  // Get stage status display
  const getStageStatusDisplay = (status: DeploymentStage['status']) => {
    switch (status) {
      case 'pending': return { color: 'default', icon: <ClockCircleOutlined /> };
      case 'running': return { color: 'processing', icon: <PlayCircleOutlined /> };
      case 'completed': return { color: 'success', icon: <CheckCircleOutlined /> };
      case 'failed': return { color: 'error', icon: <CloseCircleOutlined /> };
      case 'skipped': return { color: 'default', icon: <StopOutlined /> };
      default: return { color: 'default', icon: <ClockCircleOutlined /> };
    }
  };

  // Handle pipeline actions
  const handleStartDeployment = (pipelineId: string) => {
    const pipeline = pipelines.find(p => p.id === pipelineId);
    if (pipeline) {
      deployForm.setFieldsValue({
        pipelineId,
        name: pipeline.name,
        version: pipeline.version
      });
      setDeployModalVisible(true);
    }
  };

  const handlePausePipeline = (pipelineId: string) => {
    setPipelines(prev => prev.map(p => 
      p.id === pipelineId ? { ...p, status: 'paused' } : p
    ));
    message.success('Pipeline paused');
  };

  const handleResumePipeline = (pipelineId: string) => {
    setPipelines(prev => prev.map(p => 
      p.id === pipelineId ? { ...p, status: 'deploying' } : p
    ));
    message.success('Pipeline resumed');
  };

  const handleRollbackPipeline = (pipelineId: string) => {
    Modal.confirm({
      title: 'Rollback Strategy',
      content: 'This will rollback the strategy to the previous version. Are you sure?',
      onOk: () => {
        setPipelines(prev => prev.map(p => 
          p.id === pipelineId ? { ...p, status: 'retired' } : p
        ));
        message.success('Strategy rolled back');
      }
    });
  };

  const selectedPipelineData = pipelines.find(p => p.id === selectedPipeline);

  // Pipeline table columns
  const pipelineColumns = [
    {
      title: 'Strategy',
      key: 'strategy',
      render: (record: StrategyDeploymentPipeline) => (
        <Space>
          <div style={{ color: getStatusDisplay(record.status).color }}>
            {getStatusDisplay(record.status).icon}
          </div>
          <div>
            <Text strong>{record.name}</Text>
            <br />
            <Text type="secondary" style={{ fontSize: '12px' }}>
              v{record.version} • {record.createdBy.split('@')[0]}
            </Text>
          </div>
        </Space>
      )
    },
    {
      title: 'Status',
      dataIndex: 'status',
      key: 'status',
      render: (status: StrategyDeploymentPipeline['status']) => {
        const display = getStatusDisplay(status);
        return (
          <Badge
            status={display.color}
            text={display.text}
          />
        );
      }
    },
    {
      title: 'Progress',
      key: 'progress',
      render: (record: StrategyDeploymentPipeline) => {
        const totalStages = record.stages.length;
        const completedStages = record.stages.filter(s => s.status === 'completed').length;
        const progressPercent = (completedStages / totalStages) * 100;
        
        return (
          <div style={{ width: '120px' }}>
            <Progress
              percent={progressPercent}
              size="small"
              status={record.status === 'failed' ? 'exception' : 'normal'}
              format={() => `${completedStages}/${totalStages}`}
            />
            <Text type="secondary" style={{ fontSize: '11px' }}>
              Stage {record.currentStage + 1}: {record.stages[record.currentStage]?.name}
            </Text>
          </div>
        );
      }
    },
    {
      title: 'Created',
      dataIndex: 'createdAt',
      key: 'createdAt',
      render: (timestamp: string) => (
        <Text type="secondary" style={{ fontSize: '12px' }}>
          {new Date(timestamp).toLocaleString()}
        </Text>
      )
    },
    {
      title: 'Actions',
      key: 'actions',
      render: (record: StrategyDeploymentPipeline) => {
        const dropdownItems = {
          items: [
            {
              key: 'view',
              label: 'View Details',
              icon: <EyeOutlined />
            },
            {
              key: 'logs',
              label: 'View Logs',
              icon: <FileTextOutlined />
            },
            { type: 'divider' },
            ...(record.status === 'deploying' ? [{
              key: 'pause',
              label: 'Pause',
              icon: <PauseCircleOutlined />
            }] : []),
            ...(record.status === 'paused' ? [{
              key: 'resume',
              label: 'Resume',
              icon: <PlayCircleOutlined />
            }] : []),
            ...(record.status === 'active' ? [{
              key: 'rollback',
              label: 'Rollback',
              icon: <RollbackOutlined />,
              danger: true
            }] : []),
            ...(record.status === 'draft' ? [{
              key: 'deploy',
              label: 'Start Deployment',
              icon: <RocketOutlined />
            }] : [])
          ],
          onClick: ({ key }: { key: string }) => {
            switch (key) {
              case 'view':
                setSelectedPipeline(record.id);
                break;
              case 'pause':
                handlePausePipeline(record.id);
                break;
              case 'resume':
                handleResumePipeline(record.id);
                break;
              case 'rollback':
                handleRollbackPipeline(record.id);
                break;
              case 'deploy':
                handleStartDeployment(record.id);
                break;
            }
          }
        };

        return (
          <Space>
            <Button
              size="small"
              type="primary"
              icon={<RocketOutlined />}
              onClick={() => handleStartDeployment(record.id)}
              disabled={record.status === 'active' || record.status === 'deploying'}
            >
              Deploy
            </Button>
            <Dropdown menu={dropdownItems} placement="bottomRight">
              <Button size="small" icon={<SettingOutlined />} />
            </Dropdown>
          </Space>
        );
      }
    }
  ];

  if (compactMode) {
    const activePipelines = pipelines.filter(p => p.status === 'active' || p.status === 'deploying');
    return (
      <Card
        title={
          <Space>
            <DeploymentUnitOutlined />
            Strategy Deployment
            {activePipelines.length > 0 && <Badge count={activePipelines.length} />}
          </Space>
        }
        size="small"
      >
        <Row gutter={[8, 8]}>
          <Col span={6}>
            <Statistic
              title="Active"
              value={pipelines.filter(p => p.status === 'active').length}
              prefix={<CheckCircleOutlined />}
              valueStyle={{ fontSize: '16px', color: '#52c41a' }}
            />
          </Col>
          <Col span={6}>
            <Statistic
              title="Deploying"
              value={pipelines.filter(p => p.status === 'deploying').length}
              prefix={<RocketOutlined />}
              valueStyle={{ fontSize: '16px', color: '#1890ff' }}
            />
          </Col>
          <Col span={6}>
            <Statistic
              title="Pending"
              value={pipelines.filter(p => p.status === 'approval').length}
              prefix={<ClockCircleOutlined />}
              valueStyle={{ fontSize: '16px', color: '#faad14' }}
            />
          </Col>
          <Col span={6}>
            <Statistic
              title="Failed"
              value={pipelines.filter(p => p.status === 'failed').length}
              prefix={<CloseCircleOutlined />}
              valueStyle={{ fontSize: '16px', color: '#ff4d4f' }}
            />
          </Col>
        </Row>
      </Card>
    );
  }

  return (
    <div style={{ width: '100%' }}>
      {/* Control Panel */}
      <Card size="small" style={{ marginBottom: 16 }}>
        <Row gutter={[16, 8]} align="middle">
          <Col>
            <Space>
              <DeploymentUnitOutlined style={{ color: '#1890ff' }} />
              <Text strong>Strategy Deployment Pipeline</Text>
            </Space>
          </Col>
          <Col flex="auto" />
          <Col>
            <Space>
              <Button
                size="small"
                type="primary"
                icon={<PlusOutlined />}
                onClick={() => setModalVisible(true)}
              >
                New Pipeline
              </Button>
              <Button size="small" icon={<ReloadOutlined />}>
                Refresh
              </Button>
              <Button size="small" icon={<SettingOutlined />}>
                Settings
              </Button>
            </Space>
          </Col>
        </Row>
      </Card>

      <Row gutter={[16, 16]}>
        {/* Pipelines List */}
        <Col xs={24} lg={14}>
          <Card
            title={
              <Space>
                <GitlabOutlined />
                Deployment Pipelines
                <Badge count={pipelines.length} color="blue" />
              </Space>
            }
            size="small"
          >
            <Table
              columns={pipelineColumns}
              dataSource={pipelines}
              rowKey="id"
              size="small"
              pagination={{ pageSize: 10 }}
              scroll={{ x: 'max-content' }}
              onRow={(record) => ({
                onClick: () => setSelectedPipeline(record.id),
                style: {
                  cursor: 'pointer',
                  backgroundColor: selectedPipeline === record.id ? '#f0f0f0' : undefined
                }
              })}
            />
          </Card>
        </Col>

        {/* Pipeline Details */}
        <Col xs={24} lg={10}>
          {selectedPipelineData && (
            <Card
              title={
                <Space>
                  <RocketOutlined />
                  Pipeline Details
                  <Tag color={getStatusDisplay(selectedPipelineData.status).color}>
                    {getStatusDisplay(selectedPipelineData.status).text}
                  </Tag>
                </Space>
              }
              size="small"
            >
              <Tabs size="small">
                <TabPane tab="Stages" key="stages">
                  <Steps
                    direction="vertical"
                    size="small"
                    current={selectedPipelineData.currentStage}
                    items={selectedPipelineData.stages.map((stage, index) => {
                      const statusDisplay = getStageStatusDisplay(stage.status);
                      return {
                        title: stage.name,
                        description: (
                          <div>
                            <Text type="secondary" style={{ fontSize: '12px' }}>
                              {stage.status === 'completed' && stage.duration && `Completed in ${Math.floor(stage.duration / 60)}m ${stage.duration % 60}s`}
                              {stage.status === 'running' && 'In progress...'}
                              {stage.status === 'failed' && 'Failed - see logs'}
                              {stage.status === 'pending' && 'Waiting to start'}
                            </Text>
                            {stage.requirements.length > 0 && (
                              <div style={{ marginTop: '4px' }}>
                                {stage.requirements.map(req => (
                                  <Tag
                                    key={req.id}
                                    size="small"
                                    color={
                                      req.status === 'met' ? 'success' :
                                      req.status === 'failed' ? 'error' : 'processing'
                                    }
                                  >
                                    {req.description}
                                  </Tag>
                                ))}
                              </div>
                            )}
                          </div>
                        ),
                        status: stage.status === 'completed' ? 'finish' :
                               stage.status === 'running' ? 'process' :
                               stage.status === 'failed' ? 'error' : 'wait',
                        icon: statusDisplay.icon
                      };
                    })}
                  />
                </TabPane>

                <TabPane tab="Info" key="info">
                  <List
                    size="small"
                    dataSource={[
                      { label: 'Name', value: selectedPipelineData.name },
                      { label: 'Version', value: selectedPipelineData.version },
                      { label: 'Created By', value: selectedPipelineData.createdBy },
                      { label: 'Created At', value: new Date(selectedPipelineData.createdAt).toLocaleString() },
                      { label: 'Last Modified', value: new Date(selectedPipelineData.lastModified).toLocaleString() },
                      ...(selectedPipelineData.deployedAt ? [{ label: 'Deployed At', value: new Date(selectedPipelineData.deployedAt).toLocaleString() }] : [])
                    ]}
                    renderItem={(item) => (
                      <List.Item style={{ padding: '4px 0' }}>
                        <Text type="secondary" style={{ fontSize: '12px' }}>{item.label}:</Text>
                        <Text strong style={{ fontSize: '12px', marginLeft: '8px' }}>{item.value}</Text>
                      </List.Item>
                    )}
                  />
                </TabPane>

                <TabPane tab="Logs" key="logs">
                  <div style={{ maxHeight: '300px', overflowY: 'auto' }}>
                    {selectedPipelineData.stages
                      .flatMap(stage => stage.logs)
                      .sort((a, b) => new Date(b.timestamp).getTime() - new Date(a.timestamp).getTime())
                      .map((log, index) => (
                        <div key={index} style={{ marginBottom: '8px', padding: '4px 8px', backgroundColor: '#fafafa', borderRadius: '4px' }}>
                          <Space>
                            <Tag
                              size="small"
                              color={
                                log.level === 'error' ? 'error' :
                                log.level === 'warning' ? 'warning' : 'default'
                              }
                            >
                              {log.level.toUpperCase()}
                            </Tag>
                            <Text type="secondary" style={{ fontSize: '11px' }}>
                              {new Date(log.timestamp).toLocaleTimeString()}
                            </Text>
                          </Space>
                          <div>
                            <Text style={{ fontSize: '12px' }}>{log.message}</Text>
                          </div>
                        </div>
                      ))}
                  </div>
                </TabPane>
              </Tabs>
            </Card>
          )}
        </Col>
      </Row>

      {/* New Pipeline Modal */}
      <Modal
        title="Create New Deployment Pipeline"
        open={modalVisible}
        onOk={() => {
          form.validateFields().then(values => {
            const newPipeline: StrategyDeploymentPipeline = {
              id: `pipeline-${Date.now()}`,
              ...values,
              status: 'draft',
              currentStage: 0,
              createdAt: new Date().toISOString(),
              lastModified: new Date().toISOString(),
              stages: [
                {
                  id: 'validation',
                  name: 'Validation',
                  type: 'validation',
                  status: 'pending',
                  logs: [],
                  requirements: [],
                  artifacts: []
                }
              ]
            };
            setPipelines(prev => [...prev, newPipeline]);
            setModalVisible(false);
            form.resetFields();
          });
        }}
        onCancel={() => setModalVisible(false)}
        width={600}
      >
        <Form form={form} layout="vertical">
          <Form.Item name="name" label="Strategy Name" rules={[{ required: true }]}>
            <Input placeholder="Enter strategy name" />
          </Form.Item>
          <Form.Item name="version" label="Version" rules={[{ required: true }]}>
            <Input placeholder="e.g., 1.0.0" />
          </Form.Item>
          <Form.Item name="createdBy" label="Created By" rules={[{ required: true }]}>
            <Input placeholder="user@nautilus.com" />
          </Form.Item>
        </Form>
      </Modal>

      {/* Deploy Modal */}
      <Modal
        title="Start Strategy Deployment"
        open={deployModalVisible}
        onOk={() => {
          deployForm.validateFields().then(values => {
            message.success('Deployment started successfully');
            setDeployModalVisible(false);
            deployForm.resetFields();
          });
        }}
        onCancel={() => setDeployModalVisible(false)}
      >
        <Form form={deployForm} layout="vertical">
          <Alert
            message="Ready to Deploy"
            description="This will start the automated deployment pipeline. All tests and approvals must pass before the strategy goes live."
            type="info"
            showIcon
            style={{ marginBottom: 16 }}
          />
          <Form.Item name="environment" label="Target Environment" rules={[{ required: true }]}>
            <Select placeholder="Select environment">
              <Option value="staging">Staging</Option>
              <Option value="production">Production</Option>
            </Select>
          </Form.Item>
          <Form.Item name="notes" label="Deployment Notes">
            <TextArea rows={3} placeholder="Optional deployment notes..." />
          </Form.Item>
        </Form>
      </Modal>
    </div>
  );
};

export default DeploymentPipelineManager;