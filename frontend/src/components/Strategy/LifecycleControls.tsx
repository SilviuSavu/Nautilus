import React, { useState, useEffect, useCallback } from 'react';
import {
  Card,
  Button,
  Typography,
  Space,
  Alert,
  Modal,
  Form,
  Select,
  InputNumber,
  Switch,
  Progress,
  Tag,
  Tooltip,
  Timeline,
  Badge,
  Dropdown,
  Divider,
  notification,
  Row,
  Col,
  Statistic
} from 'antd';
import {
  PlayCircleOutlined,
  PauseCircleOutlined,
  StopOutlined,
  ReloadOutlined,
  DeleteOutlined,
  SettingOutlined,
  MonitorOutlined,
  CloudUploadOutlined,
  ExclamationCircleOutlined,
  CheckCircleOutlined,
  ClockCircleOutlined,
  WarningOutlined,
  SyncOutlined,
  ThunderboltOutlined,
  LineChartOutlined
} from '@ant-design/icons';
import { StrategyConfig, StrategyInstance, StrategyState } from './types/strategyTypes';
import strategyService from './services/strategyService';

const { Title, Text } = Typography;
const { Option } = Select;

interface DeploymentProgress {
  stage: string;
  status: 'pending' | 'in-progress' | 'completed' | 'failed';
  message: string;
  progress: number;
  timestamp: Date;
}

interface LifecycleControlsProps {
  strategy: StrategyConfig;
  instance?: StrategyInstance;
  onInstanceUpdate?: (instance: StrategyInstance) => void;
  onStrategyUpdate?: (strategy: StrategyConfig) => void;
  className?: string;
}

export const LifecycleControls: React.FC<LifecycleControlsProps> = ({
  strategy,
  instance,
  onInstanceUpdate,
  onStrategyUpdate,
  className
}) => {
  const [deploymentVisible, setDeploymentVisible] = useState(false);
  const [deploymentProgress, setDeploymentProgress] = useState<DeploymentProgress[]>([]);
  const [isDeploying, setIsDeploying] = useState(false);
  const [controlLoading, setControlLoading] = useState<string | null>(null);
  const [deploymentForm] = Form.useForm();

  // Real-time status polling
  useEffect(() => {
    if (instance && ['running', 'paused', 'stopping'].includes(instance.state)) {
      const intervalId = setInterval(async () => {
        try {
          const status = await strategyService.getStrategyStatus(instance.id);
          if (onInstanceUpdate) {
            onInstanceUpdate({
              ...instance,
              state: status.state as StrategyState,
              performance_metrics: status.performance_metrics,
              runtime_info: status.runtime_info
            });
          }
        } catch (error) {
          console.error('Failed to poll strategy status:', error);
        }
      }, 5000); // Poll every 5 seconds

      return () => clearInterval(intervalId);
    }
  }, [instance, onInstanceUpdate]);

  const getStateColor = (state: StrategyState): string => {
    switch (state) {
      case 'running': return '#52c41a';
      case 'paused': return '#fa8c16';
      case 'stopped': return '#d9d9d9';
      case 'error': return '#ff4d4f';
      case 'initializing': return '#1890ff';
      case 'stopping': return '#fa8c16';
      case 'completed': return '#722ed1';
      default: return '#d9d9d9';
    }
  };

  const getStateIcon = (state: StrategyState) => {
    switch (state) {
      case 'running': return <PlayCircleOutlined style={{ color: '#52c41a' }} />;
      case 'paused': return <PauseCircleOutlined style={{ color: '#fa8c16' }} />;
      case 'stopped': return <StopOutlined style={{ color: '#d9d9d9' }} />;
      case 'error': return <ExclamationCircleOutlined style={{ color: '#ff4d4f' }} />;
      case 'initializing': return <SyncOutlined spin style={{ color: '#1890ff' }} />;
      case 'stopping': return <ClockCircleOutlined style={{ color: '#fa8c16' }} />;
      case 'completed': return <CheckCircleOutlined style={{ color: '#722ed1' }} />;
      default: return <MonitorOutlined />;
    }
  };

  const handleDeploy = async (deploymentSettings: any) => {
    try {
      setIsDeploying(true);
      setDeploymentProgress([]);

      // Simulate deployment progress
      const stages = [
        { stage: 'Validation', message: 'Validating strategy configuration' },
        { stage: 'Compilation', message: 'Compiling strategy to NautilusTrader format' },
        { stage: 'Resource Check', message: 'Checking system resources and permissions' },
        { stage: 'Environment Setup', message: 'Setting up execution environment' },
        { stage: 'Deployment', message: 'Deploying strategy to execution engine' },
        { stage: 'Initialization', message: 'Initializing strategy components' },
        { stage: 'Activation', message: 'Activating strategy and starting execution' }
      ];

      for (let i = 0; i < stages.length; i++) {
        const stage = stages[i];
        
        // Add stage as in-progress
        setDeploymentProgress(prev => [...prev, {
          ...stage,
          status: 'in-progress',
          progress: (i / stages.length) * 100,
          timestamp: new Date()
        }]);

        // Simulate processing time
        await new Promise(resolve => setTimeout(resolve, 1000 + Math.random() * 2000));

        // Update stage as completed
        setDeploymentProgress(prev => 
          prev.map((p, index) => 
            index === prev.length - 1 
              ? { ...p, status: 'completed', progress: ((i + 1) / stages.length) * 100 }
              : p
          )
        );
      }

      // Call actual deployment API
      const deploymentResult = await strategyService.deployStrategy({
        strategy_id: strategy.id,
        deployment_mode: deploymentSettings.mode
      });

      notification.success({
        message: 'Strategy Deployed Successfully',
        description: `Strategy "${strategy.name}" has been deployed and is starting up.`,
        duration: 4
      });

      setDeploymentVisible(false);
      
      // Update strategy status
      if (onStrategyUpdate) {
        onStrategyUpdate({
          ...strategy,
          status: 'deployed'
        });
      }

    } catch (error: any) {
      console.error('Deployment failed:', error);
      
      setDeploymentProgress(prev => 
        prev.length > 0 
          ? [...prev.slice(0, -1), { ...prev[prev.length - 1], status: 'failed' }]
          : [{
              stage: 'Deployment',
              status: 'failed',
              message: error.message || 'Deployment failed',
              progress: 0,
              timestamp: new Date()
            }]
      );

      notification.error({
        message: 'Deployment Failed',
        description: error.message || 'Failed to deploy strategy',
        duration: 6
      });
    } finally {
      setIsDeploying(false);
    }
  };

  const handleControl = async (action: 'start' | 'stop' | 'pause' | 'resume', force = false) => {
    if (!instance) return;

    try {
      setControlLoading(action);

      const result = await strategyService.controlStrategy(instance.id, {
        action,
        force
      });

      notification.success({
        message: `Strategy ${action.charAt(0).toUpperCase() + action.slice(1)}`,
        description: result.message || `Strategy ${action} completed successfully`,
        duration: 3
      });

      // Update instance state
      if (onInstanceUpdate) {
        onInstanceUpdate({
          ...instance,
          state: result.new_state as StrategyState
        });
      }

    } catch (error: any) {
      console.error(`Failed to ${action} strategy:`, error);
      notification.error({
        message: `${action.charAt(0).toUpperCase() + action.slice(1)} Failed`,
        description: error.message || `Failed to ${action} strategy`,
        duration: 6
      });
    } finally {
      setControlLoading(null);
    }
  };

  const handleForceStop = () => {
    Modal.confirm({
      title: 'Force Stop Strategy',
      content: 'Are you sure you want to force stop this strategy? This may result in incomplete trades or data loss.',
      icon: <ExclamationCircleOutlined />,
      okText: 'Force Stop',
      okType: 'danger',
      cancelText: 'Cancel',
      onOk: () => handleControl('stop', true)
    });
  };

  const handleDelete = () => {
    Modal.confirm({
      title: 'Delete Strategy',
      content: 'Are you sure you want to delete this strategy? This action cannot be undone.',
      icon: <ExclamationCircleOutlined />,
      okText: 'Delete',
      okType: 'danger',
      cancelText: 'Cancel',
      onOk: async () => {
        try {
          await strategyService.deleteConfiguration(strategy.id);
          notification.success({
            message: 'Strategy Deleted',
            description: 'Strategy has been successfully deleted.',
            duration: 3
          });
        } catch (error: any) {
          notification.error({
            message: 'Delete Failed',
            description: error.message || 'Failed to delete strategy',
            duration: 6
          });
        }
      }
    });
  };

  const renderDeploymentModal = () => (
    <Modal
      title="Deploy Strategy"
      open={deploymentVisible}
      onCancel={() => !isDeploying && setDeploymentVisible(false)}
      footer={null}
      width={600}
      closable={!isDeploying}
    >
      {!isDeploying ? (
        <Form
          form={deploymentForm}
          layout="vertical"
          onFinish={handleDeploy}
          initialValues={{
            mode: 'paper',
            auto_start: true,
            risk_check: true
          }}
        >
          <Form.Item
            label="Deployment Mode"
            name="mode"
            required
            tooltip="Live trading uses real money, paper trading is simulated"
          >
            <Select>
              <Option value="paper">Paper Trading (Simulated)</Option>
              <Option value="live">Live Trading (Real Money)</Option>
              <Option value="backtest">Backtest Mode</Option>
            </Select>
          </Form.Item>

          <Form.Item
            label="Auto Start"
            name="auto_start"
            valuePropName="checked"
            tooltip="Automatically start the strategy after deployment"
          >
            <Switch />
          </Form.Item>

          <Form.Item
            label="Risk Check"
            name="risk_check"
            valuePropName="checked"
            tooltip="Perform additional risk validation before deployment"
          >
            <Switch />
          </Form.Item>

          <Alert
            type="info"
            message="Deployment Information"
            description="The strategy will be compiled and deployed to the NautilusTrader execution engine. This process may take a few minutes."
            style={{ marginBottom: 16 }}
          />

          <div style={{ display: 'flex', justifyContent: 'flex-end', gap: 8 }}>
            <Button onClick={() => setDeploymentVisible(false)}>
              Cancel
            </Button>
            <Button type="primary" htmlType="submit" icon={<CloudUploadOutlined />}>
              Deploy Strategy
            </Button>
          </div>
        </Form>
      ) : (
        <div>
          <div style={{ marginBottom: 24 }}>
            <Text strong>Deploying Strategy: {strategy.name}</Text>
            <Progress 
              percent={deploymentProgress.length > 0 ? deploymentProgress[deploymentProgress.length - 1].progress : 0}
              status={deploymentProgress.some(p => p.status === 'failed') ? 'exception' : 'active'}
              strokeColor="#1890ff"
            />
          </div>

          <Timeline>
            {deploymentProgress.map((stage, index) => (
              <Timeline.Item
                key={index}
                color={
                  stage.status === 'completed' ? 'green' :
                  stage.status === 'failed' ? 'red' :
                  stage.status === 'in-progress' ? 'blue' : 'gray'
                }
                dot={
                  stage.status === 'completed' ? <CheckCircleOutlined /> :
                  stage.status === 'failed' ? <ExclamationCircleOutlined /> :
                  stage.status === 'in-progress' ? <SyncOutlined spin /> :
                  <ClockCircleOutlined />
                }
              >
                <div>
                  <Text strong>{stage.stage}</Text>
                  <br />
                  <Text type="secondary">{stage.message}</Text>
                  <br />
                  <Text type="secondary" style={{ fontSize: 12 }}>
                    {stage.timestamp.toLocaleTimeString()}
                  </Text>
                </div>
              </Timeline.Item>
            ))}
          </Timeline>
        </div>
      )}
    </Modal>
  );

  const renderControlButtons = () => {
    if (!instance) {
      return (
        <Button
          type="primary"
          icon={<CloudUploadOutlined />}
          onClick={() => setDeploymentVisible(true)}
          size="large"
        >
          Deploy Strategy
        </Button>
      );
    }

    const { state } = instance;
    const canStart = ['stopped', 'paused', 'error'].includes(state);
    const canPause = state === 'running';
    const canStop = ['running', 'paused'].includes(state);

    return (
      <Space size="small">
        {canStart && (
          <Button
            type="primary"
            icon={<PlayCircleOutlined />}
            loading={controlLoading === 'start'}
            onClick={() => handleControl('start')}
          >
            Start
          </Button>
        )}

        {canPause && (
          <Button
            icon={<PauseCircleOutlined />}
            loading={controlLoading === 'pause'}
            onClick={() => handleControl('pause')}
          >
            Pause
          </Button>
        )}

        {state === 'paused' && (
          <Button
            type="primary"
            icon={<PlayCircleOutlined />}
            loading={controlLoading === 'resume'}
            onClick={() => handleControl('resume')}
          >
            Resume
          </Button>
        )}

        {canStop && (
          <Dropdown
            menu={{
              items: [
                {
                  key: 'stop',
                  label: 'Stop Gracefully',
                  icon: <StopOutlined />,
                  onClick: () => handleControl('stop')
                },
                {
                  key: 'force-stop',
                  label: 'Force Stop',
                  icon: <ExclamationCircleOutlined />,
                  danger: true,
                  onClick: handleForceStop
                }
              ]
            }}
          >
            <Button
              danger
              icon={<StopOutlined />}
              loading={controlLoading === 'stop'}
            >
              Stop
            </Button>
          </Dropdown>
        )}

        <Button
          icon={<ReloadOutlined />}
          loading={controlLoading === 'restart'}
          onClick={() => handleControl('stop')}
          disabled={!canStop}
        >
          Restart
        </Button>
      </Space>
    );
  };

  const renderStatusCard = () => {
    if (!instance) {
      return (
        <Card size="small" title="Strategy Status">
          <div style={{ textAlign: 'center', padding: '20px 0' }}>
            <Text type="secondary">Strategy not deployed</Text>
          </div>
        </Card>
      );
    }

    const { state, performance_metrics, runtime_info } = instance;

    return (
      <Card size="small" title="Strategy Status">
        <Row gutter={[16, 16]}>
          <Col span={6}>
            <div style={{ textAlign: 'center' }}>
              <div style={{ fontSize: 24, marginBottom: 8 }}>
                {getStateIcon(state)}
              </div>
              <Tag color={getStateColor(state)} style={{ margin: 0 }}>
                {state.toUpperCase()}
              </Tag>
            </div>
          </Col>

          <Col span={6}>
            <Statistic
              title="Total P&L"
              value={Number(performance_metrics.total_pnl)}
              precision={2}
              prefix="$"
              valueStyle={{ 
                color: Number(performance_metrics.total_pnl) >= 0 ? '#3f8600' : '#cf1322' 
              }}
            />
          </Col>

          <Col span={6}>
            <Statistic
              title="Win Rate"
              value={performance_metrics.win_rate * 100}
              precision={1}
              suffix="%"
              valueStyle={{ color: '#1890ff' }}
            />
          </Col>

          <Col span={6}>
            <Statistic
              title="Uptime"
              value={Math.floor(runtime_info.uptime_seconds / 3600)}
              suffix="hrs"
              valueStyle={{ color: '#722ed1' }}
            />
          </Col>
        </Row>

        <Divider style={{ margin: '16px 0' }} />

        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <Space>
            <Text type="secondary">Trades: {performance_metrics.total_trades}</Text>
            <Text type="secondary">Orders: {runtime_info.orders_placed}</Text>
            <Text type="secondary">Positions: {runtime_info.positions_opened}</Text>
          </Space>
          <Text type="secondary" style={{ fontSize: 12 }}>
            Updated: {new Date(performance_metrics.last_updated).toLocaleTimeString()}
          </Text>
        </div>
      </Card>
    );
  };

  return (
    <div className={`lifecycle-controls ${className || ''}`}>
      <Card>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 16 }}>
          <div>
            <Title level={4} style={{ margin: 0 }}>
              Strategy Lifecycle
            </Title>
            <Text type="secondary">
              Deploy and manage strategy execution
            </Text>
          </div>

          <Space>
            {renderControlButtons()}
            
            <Dropdown
              menu={{
                items: [
                  {
                    key: 'settings',
                    label: 'Strategy Settings',
                    icon: <SettingOutlined />
                  },
                  {
                    key: 'monitor',
                    label: 'Performance Monitor',
                    icon: <LineChartOutlined />
                  },
                  {
                    type: 'divider'
                  },
                  {
                    key: 'delete',
                    label: 'Delete Strategy',
                    icon: <DeleteOutlined />,
                    danger: true,
                    onClick: handleDelete
                  }
                ]
              }}
            >
              <Button icon={<SettingOutlined />} />
            </Dropdown>
          </Space>
        </div>

        {renderStatusCard()}

        {instance && instance.error_log.length > 0 && (
          <Alert
            type="warning"
            message="Recent Errors"
            description={
              <div>
                {instance.error_log.slice(-3).map((error, index) => (
                  <div key={index} style={{ marginBottom: 4 }}>
                    <Text type="secondary">{new Date(error.timestamp).toLocaleString()}</Text>
                    <br />
                    <Text>{error.message}</Text>
                  </div>
                ))}
              </div>
            }
            style={{ marginTop: 16 }}
            showIcon
          />
        )}
      </Card>

      {renderDeploymentModal()}
    </div>
  );
};