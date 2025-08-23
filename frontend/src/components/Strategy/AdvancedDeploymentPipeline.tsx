import React, { useState, useEffect, useCallback } from 'react';
import {
  Card,
  Steps,
  Button,
  Form,
  Input,
  Select,
  InputNumber,
  Switch,
  Alert,
  Progress,
  Space,
  Divider,
  Tag,
  Typography,
  Row,
  Col,
  Tabs,
  Timeline,
  Spin,
  notification,
  Modal,
  Tooltip,
  Badge,
  Statistic,
  List
} from 'antd';
import {
  PlayCircleOutlined,
  StopOutlined,
  CheckCircleOutlined,
  ExclamationCircleOutlined,
  ClockCircleOutlined,
  SettingOutlined,
  SecurityScanOutlined,
  DeploymentUnitOutlined,
  MonitorOutlined,
  RobotOutlined,
  CloudUploadOutlined,
  BugOutlined,
  LineChartOutlined,
  SyncOutlined,
  PauseCircleOutlined,
  FastForwardOutlined
} from '@ant-design/icons';
import type {
  AdvancedDeploymentPipelineProps,
  AdvancedDeploymentPipeline,
  DeploymentPipelineStage,
  PipelineConfiguration,
  CreatePipelineRequest,
  PipelineStatusResponse
} from './types/deploymentTypes';

const { Step } = Steps;
const { Option } = Select;
const { Text, Title, Paragraph } = Typography;
const { TabPane } = Tabs;

const API_BASE = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8001';

const AdvancedDeploymentPipeline: React.FC<AdvancedDeploymentPipelineProps> = ({
  strategyId,
  initialConfiguration,
  onPipelineCreated,
  onPipelineCompleted,
  onClose
}) => {
  const [form] = Form.useForm();
  const [loading, setLoading] = useState(false);
  const [pipeline, setPipeline] = useState<AdvancedDeploymentPipeline | null>(null);
  const [currentView, setCurrentView] = useState<'configuration' | 'execution'>('configuration');
  const [pipelineStatus, setPipelineStatus] = useState<'idle' | 'running' | 'completed' | 'failed'>('idle');
  const [stageLogs, setStageLogs] = useState<Record<string, string[]>>({});
  const [autoRefresh, setAutoRefresh] = useState(true);

  // Default pipeline configuration
  const defaultConfiguration: PipelineConfiguration = {
    deployment_strategy: 'blue_green',
    auto_advance: false,
    rollback_triggers: [
      {
        id: 'performance_rollback',
        type: 'performance_based',
        enabled: true,
        conditions: [
          { metric: 'drawdown_percent', operator: 'gt', value: 5, window_minutes: 30 },
          { metric: 'pnl_daily', operator: 'lt', value: -1000, window_minutes: 60 }
        ],
        action: 'rollback'
      },
      {
        id: 'ml_prediction_rollback',
        type: 'ml_prediction',
        enabled: true,
        conditions: [
          { metric: 'prediction_confidence', operator: 'gt', value: 0.8, window_minutes: 15 }
        ],
        action: 'pause'
      }
    ],
    notification_settings: {
      channels: ['email', 'webhook'],
      recipients: ['trading-team@example.com'],
      events: ['stage_complete', 'stage_failed', 'approval_required', 'rollback_triggered']
    },
    resource_limits: {
      max_cpu_cores: 4,
      max_memory_mb: 8192,
      max_disk_mb: 10240
    },
    timeout_settings: {
      validation_timeout_minutes: 30,
      backtesting_timeout_minutes: 120,
      paper_trading_timeout_minutes: 480,
      staging_timeout_minutes: 240,
      production_timeout_minutes: 60
    }
  };

  // Default pipeline stages
  const defaultStages: DeploymentPipelineStage[] = [
    {
      id: 'validation',
      name: 'Code Validation',
      type: 'validation',
      status: 'pending',
      auto_advance: true,
      success_criteria: {
        validation_checks: [
          { name: 'syntax_check', description: 'Python syntax validation', required: true },
          { name: 'dependency_check', description: 'Package dependency validation', required: true },
          { name: 'config_validation', description: 'Strategy configuration validation', required: true },
          { name: 'risk_parameter_check', description: 'Risk parameter validation', required: true }
        ]
      },
      stage_config: {
        run_linting: true,
        check_security: true,
        validate_nautilus_compatibility: true
      }
    },
    {
      id: 'backtesting',
      name: 'Automated Backtesting',
      type: 'backtesting',
      status: 'pending',
      auto_advance: false,
      required_approvals: ['senior_trader'],
      success_criteria: {
        min_trades: 10,
        max_drawdown: 0.15,
        min_sharpe_ratio: 0.5,
        win_rate_threshold: 0.4
      },
      stage_config: {
        backtest_period: '6M',
        benchmark: 'SPY',
        slippage_model: 'linear',
        commission_model: 'per_share'
      }
    },
    {
      id: 'paper_trading',
      name: 'Paper Trading Validation',
      type: 'paper_trading',
      status: 'pending',
      auto_advance: false,
      required_approvals: ['risk_manager'],
      success_criteria: {
        min_trades: 5,
        max_drawdown: 0.05,
        max_var_95: 0.03
      },
      stage_config: {
        duration_hours: 8,
        position_size_percent: 25,
        max_positions: 3
      }
    },
    {
      id: 'staging',
      name: 'Staging Environment',
      type: 'staging',
      status: 'pending',
      auto_advance: false,
      required_approvals: ['head_of_trading'],
      success_criteria: {
        min_trades: 3,
        max_drawdown: 0.03,
        profit_factor_min: 1.1
      },
      stage_config: {
        duration_hours: 4,
        position_size_percent: 50,
        max_positions: 5
      }
    },
    {
      id: 'production',
      name: 'Production Deployment',
      type: 'production',
      status: 'pending',
      auto_advance: false,
      required_approvals: ['head_of_trading', 'cto'],
      stage_config: {
        rollout_strategy: 'gradual',
        initial_position_size_percent: 25,
        full_rollout_after_hours: 24,
        monitoring_intensive_hours: 48
      }
    }
  ];

  useEffect(() => {
    form.setFieldsValue({
      deployment_strategy: defaultConfiguration.deployment_strategy,
      auto_advance: defaultConfiguration.auto_advance,
      max_cpu_cores: defaultConfiguration.resource_limits.max_cpu_cores,
      max_memory_mb: defaultConfiguration.resource_limits.max_memory_mb,
      validation_timeout: defaultConfiguration.timeout_settings.validation_timeout_minutes,
      backtesting_timeout: defaultConfiguration.timeout_settings.backtesting_timeout_minutes,
      paper_trading_timeout: defaultConfiguration.timeout_settings.paper_trading_timeout_minutes,
      notification_channels: defaultConfiguration.notification_settings.channels,
      enable_ml_rollback: true,
      enable_performance_rollback: true
    });
  }, [form]);

  useEffect(() => {
    let intervalId: NodeJS.Timeout;
    
    if (pipeline && autoRefresh && (pipelineStatus === 'running')) {
      intervalId = setInterval(() => {
        fetchPipelineStatus(pipeline.pipeline_id);
      }, 5000);
    }
    
    return () => {
      if (intervalId) clearInterval(intervalId);
    };
  }, [pipeline, autoRefresh, pipelineStatus]);

  const fetchPipelineStatus = useCallback(async (pipelineId: string) => {
    try {
      const response = await fetch(`${API_BASE}/api/v1/strategies/pipeline/${pipelineId}/status`);
      const data: PipelineStatusResponse = await response.json();
      
      setPipeline(data.pipeline);
      setPipelineStatus(data.pipeline.status as any);
      
      if (data.logs) {
        const logs: Record<string, string[]> = {};
        data.pipeline.stages.forEach(stage => {
          logs[stage.id] = data.logs?.filter(log => log.includes(`[${stage.id}]`)) || [];
        });
        setStageLogs(logs);
      }
      
      if (data.pipeline.status === 'completed') {
        notification.success({
          message: 'Pipeline Completed',
          description: 'Deployment pipeline completed successfully!'
        });
        onPipelineCompleted?.(pipelineId, true);
      } else if (data.pipeline.status === 'failed') {
        notification.error({
          message: 'Pipeline Failed',
          description: 'Deployment pipeline failed. Check logs for details.'
        });
        onPipelineCompleted?.(pipelineId, false);
      }
    } catch (error) {
      console.error('Error fetching pipeline status:', error);
    }
  }, [onPipelineCompleted]);

  const createPipeline = async (formValues: any) => {
    setLoading(true);
    try {
      const configuration: PipelineConfiguration = {
        deployment_strategy: formValues.deployment_strategy,
        auto_advance: formValues.auto_advance,
        rollback_triggers: [
          ...(formValues.enable_performance_rollback ? [defaultConfiguration.rollback_triggers[0]] : []),
          ...(formValues.enable_ml_rollback ? [defaultConfiguration.rollback_triggers[1]] : [])
        ],
        notification_settings: {
          channels: formValues.notification_channels,
          recipients: defaultConfiguration.notification_settings.recipients,
          events: defaultConfiguration.notification_settings.events
        },
        resource_limits: {
          max_cpu_cores: formValues.max_cpu_cores,
          max_memory_mb: formValues.max_memory_mb,
          max_disk_mb: defaultConfiguration.resource_limits.max_disk_mb
        },
        timeout_settings: {
          validation_timeout_minutes: formValues.validation_timeout,
          backtesting_timeout_minutes: formValues.backtesting_timeout,
          paper_trading_timeout_minutes: formValues.paper_trading_timeout,
          staging_timeout_minutes: defaultConfiguration.timeout_settings.staging_timeout_minutes,
          production_timeout_minutes: defaultConfiguration.timeout_settings.production_timeout_minutes
        }
      };

      const request: CreatePipelineRequest = {
        strategy_id: strategyId,
        configuration,
        stages: defaultStages,
        auto_start: formValues.auto_start || false
      };

      const response = await fetch(`${API_BASE}/api/v1/strategies/deploy`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(request)
      });

      if (!response.ok) {
        throw new Error(`Failed to create pipeline: ${response.statusText}`);
      }

      const result = await response.json();
      
      const newPipeline: AdvancedDeploymentPipeline = {
        pipeline_id: result.pipeline_id || `pipeline_${Date.now()}`,
        strategy_id: strategyId,
        version: result.version || '1.0.0',
        created_by: 'current_user',
        created_at: new Date(),
        status: formValues.auto_start ? 'running' : 'draft',
        stages: defaultStages,
        current_stage: defaultStages[0].id,
        configuration,
        progress: {
          overall_progress: 0,
          current_stage_progress: 0,
          stages_completed: 0,
          stages_total: defaultStages.length,
          elapsed_minutes: 0
        }
      };

      setPipeline(newPipeline);
      setPipelineStatus(newPipeline.status as any);
      setCurrentView('execution');
      
      onPipelineCreated?.(newPipeline.pipeline_id);
      
      notification.success({
        message: 'Pipeline Created',
        description: `Deployment pipeline ${newPipeline.pipeline_id} created successfully`
      });
      
    } catch (error) {
      console.error('Error creating pipeline:', error);
      notification.error({
        message: 'Pipeline Creation Failed',
        description: error instanceof Error ? error.message : 'Unknown error occurred'
      });
    } finally {
      setLoading(false);
    }
  };

  const startPipeline = async () => {
    if (!pipeline) return;
    
    setLoading(true);
    try {
      const response = await fetch(`${API_BASE}/api/v1/strategies/pipeline/${pipeline.pipeline_id}/start`, {
        method: 'POST'
      });
      
      if (!response.ok) {
        throw new Error('Failed to start pipeline');
      }
      
      setPipelineStatus('running');
      notification.success({
        message: 'Pipeline Started',
        description: 'Deployment pipeline execution started'
      });
      
    } catch (error) {
      console.error('Error starting pipeline:', error);
      notification.error({
        message: 'Failed to Start Pipeline',
        description: error instanceof Error ? error.message : 'Unknown error'
      });
    } finally {
      setLoading(false);
    }
  };

  const pausePipeline = async () => {
    if (!pipeline) return;
    
    setLoading(true);
    try {
      await fetch(`${API_BASE}/api/v1/strategies/pipeline/${pipeline.pipeline_id}/pause`, {
        method: 'POST'
      });
      
      setPipelineStatus('idle');
      notification.info({
        message: 'Pipeline Paused',
        description: 'Deployment pipeline execution paused'
      });
      
    } catch (error) {
      console.error('Error pausing pipeline:', error);
    } finally {
      setLoading(false);
    }
  };

  const getStageIcon = (stage: DeploymentPipelineStage) => {
    const iconProps = { style: { fontSize: '16px' } };
    
    switch (stage.type) {
      case 'validation': return <BugOutlined {...iconProps} />;
      case 'backtesting': return <LineChartOutlined {...iconProps} />;
      case 'paper_trading': return <RobotOutlined {...iconProps} />;
      case 'staging': return <CloudUploadOutlined {...iconProps} />;
      case 'production': return <DeploymentUnitOutlined {...iconProps} />;
      default: return <SettingOutlined {...iconProps} />;
    }
  };

  const getStageStatus = (stage: DeploymentPipelineStage) => {
    switch (stage.status) {
      case 'completed': return 'finish';
      case 'running': return 'process';
      case 'failed': return 'error';
      case 'pending': return 'wait';
      default: return 'wait';
    }
  };

  const renderConfigurationView = () => (
    <Card title="Pipeline Configuration" className="mb-4">
      <Form
        form={form}
        layout="vertical"
        onFinish={createPipeline}
      >
        <Row gutter={24}>
          <Col span={12}>
            <Form.Item 
              label="Deployment Strategy" 
              name="deployment_strategy"
              tooltip="Choose how the strategy will be deployed to production"
            >
              <Select>
                <Option value="direct">Direct Deployment</Option>
                <Option value="blue_green">Blue-Green Deployment</Option>
                <Option value="canary">Canary Deployment</Option>
                <Option value="rolling">Rolling Deployment</Option>
              </Select>
            </Form.Item>
          </Col>
          <Col span={12}>
            <Form.Item 
              label="Auto Advance Stages" 
              name="auto_advance"
              valuePropName="checked"
              tooltip="Automatically advance through pipeline stages on success"
            >
              <Switch />
            </Form.Item>
          </Col>
        </Row>

        <Divider>Resource Limits</Divider>
        
        <Row gutter={16}>
          <Col span={8}>
            <Form.Item label="Max CPU Cores" name="max_cpu_cores">
              <InputNumber min={1} max={16} />
            </Form.Item>
          </Col>
          <Col span={8}>
            <Form.Item label="Max Memory (MB)" name="max_memory_mb">
              <InputNumber min={1024} max={32768} step={1024} />
            </Form.Item>
          </Col>
          <Col span={8}>
            <Form.Item label="Auto Start Pipeline" name="auto_start" valuePropName="checked">
              <Switch />
            </Form.Item>
          </Col>
        </Row>

        <Divider>Timeout Settings (Minutes)</Divider>
        
        <Row gutter={16}>
          <Col span={8}>
            <Form.Item label="Validation Timeout" name="validation_timeout">
              <InputNumber min={5} max={120} />
            </Form.Item>
          </Col>
          <Col span={8}>
            <Form.Item label="Backtesting Timeout" name="backtesting_timeout">
              <InputNumber min={30} max={480} />
            </Form.Item>
          </Col>
          <Col span={8}>
            <Form.Item label="Paper Trading Timeout" name="paper_trading_timeout">
              <InputNumber min={60} max={1440} />
            </Form.Item>
          </Col>
        </Row>

        <Divider>Rollback Configuration</Divider>
        
        <Row gutter={16}>
          <Col span={12}>
            <Form.Item 
              label="Enable ML-Based Rollback" 
              name="enable_ml_rollback"
              valuePropName="checked"
              tooltip="Use machine learning to predict and trigger rollbacks"
            >
              <Switch />
            </Form.Item>
          </Col>
          <Col span={12}>
            <Form.Item 
              label="Enable Performance Rollback" 
              name="enable_performance_rollback"
              valuePropName="checked"
              tooltip="Automatically rollback based on performance thresholds"
            >
              <Switch />
            </Form.Item>
          </Col>
        </Row>

        <Form.Item 
          label="Notification Channels" 
          name="notification_channels"
          tooltip="Select channels for pipeline notifications"
        >
          <Select mode="multiple">
            <Option value="email">Email</Option>
            <Option value="slack">Slack</Option>
            <Option value="webhook">Webhook</Option>
          </Select>
        </Form.Item>

        <Form.Item>
          <Space>
            <Button type="primary" htmlType="submit" loading={loading} size="large">
              Create Pipeline
            </Button>
            {onClose && (
              <Button onClick={onClose} size="large">
                Cancel
              </Button>
            )}
          </Space>
        </Form.Item>
      </Form>
    </Card>
  );

  const renderExecutionView = () => {
    if (!pipeline) return null;

    const currentStageIndex = pipeline.stages.findIndex(s => s.id === pipeline.current_stage);
    
    return (
      <div>
        <Card 
          title={
            <div className="flex items-center justify-between">
              <div className="flex items-center space-x-2">
                <DeploymentUnitOutlined />
                <span>Pipeline Execution: {pipeline.pipeline_id}</span>
                <Tag color={
                  pipelineStatus === 'running' ? 'processing' :
                  pipelineStatus === 'completed' ? 'success' :
                  pipelineStatus === 'failed' ? 'error' : 'default'
                }>
                  {pipelineStatus.toUpperCase()}
                </Tag>
              </div>
              <Space>
                <Switch 
                  checked={autoRefresh} 
                  onChange={setAutoRefresh}
                  checkedChildren="Auto Refresh"
                  unCheckedChildren="Manual"
                />
                {pipelineStatus === 'idle' && (
                  <Button 
                    type="primary" 
                    icon={<PlayCircleOutlined />}
                    onClick={startPipeline}
                    loading={loading}
                  >
                    Start
                  </Button>
                )}
                {pipelineStatus === 'running' && (
                  <Button 
                    icon={<PauseCircleOutlined />}
                    onClick={pausePipeline}
                    loading={loading}
                  >
                    Pause
                  </Button>
                )}
              </Space>
            </div>
          }
          className="mb-4"
        >
          <Row gutter={16} className="mb-4">
            <Col span={6}>
              <Statistic 
                title="Overall Progress" 
                value={pipeline.progress.overall_progress} 
                suffix="%" 
                prefix={<SyncOutlined spin={pipelineStatus === 'running'} />}
              />
            </Col>
            <Col span={6}>
              <Statistic 
                title="Stages Completed" 
                value={pipeline.progress.stages_completed}
                suffix={`/ ${pipeline.progress.stages_total}`}
              />
            </Col>
            <Col span={6}>
              <Statistic 
                title="Elapsed Time" 
                value={pipeline.progress.elapsed_minutes} 
                suffix="min" 
              />
            </Col>
            <Col span={6}>
              <Statistic 
                title="Current Stage" 
                value={pipeline.current_stage || 'None'}
              />
            </Col>
          </Row>

          <Progress 
            percent={pipeline.progress.overall_progress} 
            status={pipelineStatus === 'running' ? 'active' : 
                   pipelineStatus === 'failed' ? 'exception' : 'normal'}
            className="mb-4"
          />
        </Card>

        <Card title="Pipeline Stages">
          <Steps 
            current={currentStageIndex} 
            direction="horizontal" 
            size="small"
            className="mb-6"
          >
            {pipeline.stages.map((stage, index) => (
              <Step
                key={stage.id}
                title={stage.name}
                icon={getStageIcon(stage)}
                status={getStageStatus(stage)}
                description={
                  <div>
                    <div>{stage.type.replace('_', ' ')}</div>
                    {stage.duration_ms && (
                      <div className="text-xs text-gray-500">
                        {Math.round(stage.duration_ms / 1000 / 60)}min
                      </div>
                    )}
                  </div>
                }
              />
            ))}
          </Steps>

          <Tabs defaultActiveKey="timeline">
            <TabPane tab="Timeline" key="timeline">
              <Timeline mode="left">
                {pipeline.stages.map(stage => (
                  <Timeline.Item
                    key={stage.id}
                    dot={getStageIcon(stage)}
                    color={
                      stage.status === 'completed' ? 'green' :
                      stage.status === 'running' ? 'blue' :
                      stage.status === 'failed' ? 'red' : 'gray'
                    }
                  >
                    <div>
                      <Title level={5}>{stage.name}</Title>
                      <Paragraph>
                        Status: <Tag color={
                          stage.status === 'completed' ? 'success' :
                          stage.status === 'running' ? 'processing' :
                          stage.status === 'failed' ? 'error' : 'default'
                        }>
                          {stage.status.toUpperCase()}
                        </Tag>
                      </Paragraph>
                      {stage.started_at && (
                        <Paragraph type="secondary">
                          Started: {new Date(stage.started_at).toLocaleString()}
                        </Paragraph>
                      )}
                      {stage.completed_at && (
                        <Paragraph type="secondary">
                          Completed: {new Date(stage.completed_at).toLocaleString()}
                        </Paragraph>
                      )}
                      {stage.error_message && (
                        <Alert 
                          message={stage.error_message} 
                          type="error" 
                          size="small" 
                          className="mt-2"
                        />
                      )}
                      {stage.required_approvals && stage.required_approvals.length > 0 && (
                        <div className="mt-2">
                          <Text strong>Required Approvals: </Text>
                          {stage.required_approvals.map(approval => (
                            <Tag key={approval}>{approval}</Tag>
                          ))}
                        </div>
                      )}
                    </div>
                  </Timeline.Item>
                ))}
              </Timeline>
            </TabPane>

            <TabPane tab="Logs" key="logs">
              {Object.entries(stageLogs).map(([stageId, logs]) => (
                <Card key={stageId} size="small" title={stageId} className="mb-2">
                  <List
                    size="small"
                    dataSource={logs}
                    renderItem={log => (
                      <List.Item>
                        <Text code>{log}</Text>
                      </List.Item>
                    )}
                  />
                </Card>
              ))}
            </TabPane>

            <TabPane tab="Configuration" key="config">
              <Tabs size="small" tabPosition="left">
                <TabPane tab="Strategy" key="strategy">
                  <pre>{JSON.stringify(pipeline.configuration.deployment_strategy, null, 2)}</pre>
                </TabPane>
                <TabPane tab="Rollback Triggers" key="rollback">
                  <List
                    dataSource={pipeline.configuration.rollback_triggers}
                    renderItem={trigger => (
                      <List.Item>
                        <Badge 
                          status={trigger.enabled ? 'success' : 'default'} 
                          text={`${trigger.type} - ${trigger.action}`} 
                        />
                      </List.Item>
                    )}
                  />
                </TabPane>
                <TabPane tab="Resources" key="resources">
                  <pre>{JSON.stringify(pipeline.configuration.resource_limits, null, 2)}</pre>
                </TabPane>
                <TabPane tab="Timeouts" key="timeouts">
                  <pre>{JSON.stringify(pipeline.configuration.timeout_settings, null, 2)}</pre>
                </TabPane>
              </Tabs>
            </TabPane>
          </Tabs>
        </Card>
      </div>
    );
  };

  return (
    <div className="advanced-deployment-pipeline">
      {currentView === 'configuration' ? renderConfigurationView() : renderExecutionView()}
    </div>
  );
};

export default AdvancedDeploymentPipeline;