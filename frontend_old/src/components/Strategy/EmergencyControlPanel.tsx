import React, { useState, useEffect } from 'react';
import {
  Card,
  Button,
  Modal,
  Form,
  Input,
  Alert,
  Space,
  Typography,
  Row,
  Col,
  Statistic,
  Badge,
  Tag,
  Table,
  Tooltip,
  Progress,
  Checkbox,
  Select,
  notification,
  Popconfirm
} from 'antd';
import {
  AlertOutlined,
  StopOutlined,
  PauseCircleOutlined,
  PlayCircleOutlined,
  WarningOutlined,
  ExclamationCircleOutlined,
  CheckCircleOutlined,
  CloseCircleOutlined,
  ReloadOutlined,
  SettingOutlined,
  DashboardOutlined
} from '@ant-design/icons';
import type {
  LiveStrategy,
  EmergencyAction,
  EmergencyControlsProps,
  ControlStrategyRequest,
  EmergencyStopConfig
} from '../../types/deployment';

const { Text, Title } = Typography;
const { Option } = Select;

interface EmergencyControlPanelState {
  emergencyModalVisible: boolean;
  bulkActionModalVisible: boolean;
  configModalVisible: boolean;
  selectedStrategy: LiveStrategy | null;
  selectedStrategies: string[];
  emergencyAction: 'emergency_stop' | 'pause' | 'reduce_size' | 'close_positions' | null;
  confirmationStep: number;
  emergencyConfig: EmergencyStopConfig;
}

const EmergencyControlPanel: React.FC<EmergencyControlsProps> = ({
  strategies,
  onEmergencyAction,
  requireConfirmation = true
}) => {
  const [state, setState] = useState<EmergencyControlPanelState>({
    emergencyModalVisible: false,
    bulkActionModalVisible: false,
    configModalVisible: false,
    selectedStrategy: null,
    selectedStrategies: [],
    emergencyAction: null,
    confirmationStep: 0,
    emergencyConfig: {
      enabled: true,
      triggers: {
        maxDailyLoss: 5000,
        maxDrawdown: 0.1,
        consecutiveLosses: 10,
        manualTrigger: true
      },
      actions: {
        closePositions: true,
        cancelOrders: true,
        notifyRiskTeam: true,
        createIncidentReport: true
      }
    }
  });
  
  const [emergencyForm] = Form.useForm();
  const [bulkForm] = Form.useForm();
  const [configForm] = Form.useForm();
  const [loading, setLoading] = useState(false);
  const [systemStatus, setSystemStatus] = useState<{
    totalStrategies: number;
    runningStrategies: number;
    pausedStrategies: number;
    errorStrategies: number;
    totalExposure: number;
    dailyPnL: number;
  }>({
    totalStrategies: 0,
    runningStrategies: 0,
    pausedStrategies: 0,
    errorStrategies: 0,
    totalExposure: 0,
    dailyPnL: 0
  });

  useEffect(() => {
    updateSystemStatus();
    loadEmergencyConfig();
  }, [strategies]);

  const updateSystemStatus = () => {
    const status = {
      totalStrategies: strategies.length,
      runningStrategies: strategies.filter(s => s.state === 'running').length,
      pausedStrategies: strategies.filter(s => s.state === 'paused').length,
      errorStrategies: strategies.filter(s => s.state === 'error').length,
      totalExposure: strategies.reduce((sum, s) => sum + Math.abs(s.currentPosition.marketValue), 0),
      dailyPnL: strategies.reduce((sum, s) => sum + s.performanceMetrics.dailyPnL, 0)
    };
    setSystemStatus(status);
  };

  const loadEmergencyConfig = async () => {
    try {
      const response = await fetch('/api/v1/nautilus/emergency/config');
      if (response.ok) {
        const config = await response.json();
        setState(prev => ({ ...prev, emergencyConfig: config }));
        configForm.setFieldsValue(config);
      }
    } catch (error) {
      console.error('Error loading emergency config:', error);
    }
  };

  const executeEmergencyAction = async (
    strategyId: string,
    action: string,
    reason: string,
    force = false
  ) => {
    setLoading(true);
    try {
      const emergencyAction: EmergencyAction = {
        actionId: `emergency_${Date.now()}`,
        strategyInstanceId: strategyId,
        actionType: action as any,
        reason,
        initiatedBy: 'current_user',
        initiatedAt: new Date(),
        status: 'pending',
        confirmationRequired: requireConfirmation,
        secondConfirmationRequired: action === 'emergency_stop'
      };

      await onEmergencyAction?.(emergencyAction);
      
      notification.success({
        message: 'Emergency Action Executed',
        description: `${action.replace('_', ' ').toUpperCase()} initiated for strategy`,
        duration: 5
      });

      setState(prev => ({
        ...prev,
        emergencyModalVisible: false,
        selectedStrategy: null,
        emergencyAction: null,
        confirmationStep: 0
      }));
      
    } catch (error) {
      console.error('Emergency action failed:', error);
      notification.error({
        message: 'Emergency Action Failed',
        description: 'Failed to execute emergency action'
      });
    } finally {
      setLoading(false);
    }
  };

  const executeBulkAction = async (values: any) => {
    setLoading(true);
    try {
      const promises = state.selectedStrategies.map(strategyId =>
        executeEmergencyAction(strategyId, values.action, values.reason, values.force)
      );
      
      await Promise.all(promises);
      
      notification.success({
        message: 'Bulk Action Completed',
        description: `${values.action} executed on ${state.selectedStrategies.length} strategies`
      });

      setState(prev => ({
        ...prev,
        bulkActionModalVisible: false,
        selectedStrategies: []
      }));
      
    } catch (error) {
      console.error('Bulk action failed:', error);
      notification.error({
        message: 'Bulk Action Failed',
        description: 'Some actions may have failed'
      });
    } finally {
      setLoading(false);
    }
  };

  const handleEmergencyStop = (strategy: LiveStrategy) => {
    Modal.confirm({
      title: '🚨 EMERGENCY STOP CONFIRMATION',
      icon: <ExclamationCircleOutlined style={{ color: '#ff4d4f' }} />,
      content: (
        <div>
          <Alert
            message="CRITICAL ACTION"
            description="This will immediately stop the strategy and close all positions. This action cannot be undone."
            type="error"
            className="mb-4"
          />
          <div>
            <Text strong>Strategy: </Text>{strategy.strategyId}<br />
            <Text strong>Current P&L: </Text>
            <Text style={{ color: strategy.realizedPnL >= 0 ? '#52c41a' : '#f5222d' }}>
              ${(strategy.realizedPnL + strategy.unrealizedPnL).toFixed(2)}
            </Text><br />
            <Text strong>Position Value: </Text>${Math.abs(strategy.currentPosition.marketValue).toFixed(2)}
          </div>
          <div className="mt-4">
            <Text strong>Type "EMERGENCY STOP" to confirm:</Text>
            <Input 
              id="emergency-confirmation"
              placeholder="Type here..."
              className="mt-2"
            />
          </div>
        </div>
      ),
      okText: 'EXECUTE EMERGENCY STOP',
      okType: 'danger',
      cancelText: 'Cancel',
      width: 500,
      onOk: () => {
        const input = document.getElementById('emergency-confirmation') as HTMLInputElement;
        if (input?.value === 'EMERGENCY STOP') {
          executeEmergencyAction(strategy.strategyInstanceId, 'emergency_stop', 'Manual emergency stop', true);
        } else {
          notification.error({
            message: 'Confirmation Failed',
            description: 'Please type "EMERGENCY STOP" exactly to confirm'
          });
          return Promise.reject();
        }
      }
    });
  };

  const getStrategyStatusColor = (state: string): string => {
    switch (state) {
      case 'running': return 'success';
      case 'paused': return 'warning';
      case 'stopped': return 'default';
      case 'error': return 'error';
      case 'emergency_stopped': return 'error';
      default: return 'default';
    }
  };

  const strategyColumns = [
    {
      title: 'Selection',
      key: 'selection',
      render: (_, strategy: LiveStrategy) => (
        <Checkbox
          checked={state.selectedStrategies.includes(strategy.strategyInstanceId)}
          onChange={(e) => {
            const strategyId = strategy.strategyInstanceId;
            setState(prev => ({
              ...prev,
              selectedStrategies: e.target.checked
                ? [...prev.selectedStrategies, strategyId]
                : prev.selectedStrategies.filter(id => id !== strategyId)
            }));
          }}
        />
      ),
      width: 80
    },
    {
      title: 'Strategy',
      dataIndex: 'strategyId',
      render: (id: string) => <Text strong>{id}</Text>
    },
    {
      title: 'State',
      dataIndex: 'state',
      render: (state: string) => (
        <Badge
          status={getStrategyStatusColor(state) as any}
          text={state.replace('_', ' ').toUpperCase()}
        />
      )
    },
    {
      title: 'P&L',
      key: 'pnl',
      render: (_, strategy: LiveStrategy) => {
        const totalPnL = strategy.realizedPnL + strategy.unrealizedPnL;
        return (
          <Text style={{ color: totalPnL >= 0 ? '#52c41a' : '#f5222d' }}>
            ${totalPnL.toFixed(2)}
          </Text>
        );
      }
    },
    {
      title: 'Position',
      key: 'position',
      render: (_, strategy: LiveStrategy) => (
        <div>
          <Text>{strategy.currentPosition.side}</Text>
          <br />
          <Text type="secondary" style={{ fontSize: '12px' }}>
            ${Math.abs(strategy.currentPosition.marketValue).toFixed(0)}
          </Text>
        </div>
      )
    },
    {
      title: 'Risk',
      key: 'risk',
      render: (_, strategy: LiveStrategy) => (
        <div>
          <Progress
            percent={Math.min(strategy.riskMetrics.currentDrawdown * 10, 100)}
            size="small"
            status={strategy.riskMetrics.currentDrawdown > 5 ? 'exception' : 'normal'}
            format={() => `${strategy.riskMetrics.currentDrawdown.toFixed(1)}%`}
          />
        </div>
      )
    },
    {
      title: 'Quick Actions',
      key: 'actions',
      render: (_, strategy: LiveStrategy) => (
        <Space size="small">
          {strategy.state === 'running' && (
            <Tooltip title="Pause Strategy">
              <Button
                size="small"
                icon={<PauseCircleOutlined />}
                onClick={() => executeEmergencyAction(strategy.strategyInstanceId, 'pause', 'Manual pause')}
              />
            </Tooltip>
          )}
          {strategy.state === 'paused' && (
            <Tooltip title="Resume Strategy">
              <Button
                size="small"
                icon={<PlayCircleOutlined />}
                onClick={() => executeEmergencyAction(strategy.strategyInstanceId, 'resume', 'Manual resume')}
              />
            </Tooltip>
          )}
          <Tooltip title="Emergency Stop">
            <Button
              size="small"
              danger
              icon={<StopOutlined />}
              onClick={() => handleEmergencyStop(strategy)}
            />
          </Tooltip>
        </Space>
      )
    }
  ];

  const renderSystemOverview = () => (
    <Row gutter={16} className="mb-6">
      <Col span={4}>
        <Card size="small" className="text-center">
          <Statistic
            title="Total"
            value={systemStatus.totalStrategies}
            prefix={<DashboardOutlined />}
          />
        </Card>
      </Col>
      <Col span={4}>
        <Card size="small" className="text-center">
          <Statistic
            title="Running"
            value={systemStatus.runningStrategies}
            valueStyle={{ color: '#52c41a' }}
          />
        </Card>
      </Col>
      <Col span={4}>
        <Card size="small" className="text-center">
          <Statistic
            title="Paused"
            value={systemStatus.pausedStrategies}
            valueStyle={{ color: '#faad14' }}
          />
        </Card>
      </Col>
      <Col span={4}>
        <Card size="small" className="text-center">
          <Statistic
            title="Errors"
            value={systemStatus.errorStrategies}
            valueStyle={{ color: '#f5222d' }}
          />
        </Card>
      </Col>
      <Col span={4}>
        <Card size="small" className="text-center">
          <Statistic
            title="Exposure"
            value={systemStatus.totalExposure}
            precision={0}
            prefix="$"
            valueStyle={{ fontSize: '14px' }}
          />
        </Card>
      </Col>
      <Col span={4}>
        <Card size="small" className="text-center">
          <Statistic
            title="Daily P&L"
            value={systemStatus.dailyPnL}
            precision={2}
            prefix="$"
            valueStyle={{ 
              color: systemStatus.dailyPnL >= 0 ? '#52c41a' : '#f5222d',
              fontSize: '14px'
            }}
          />
        </Card>
      </Col>
    </Row>
  );

  return (
    <div className="emergency-control-panel">
      <Card
        title={
          <div className="flex items-center space-x-2">
            <AlertOutlined className="text-red-600" />
            <span>Emergency Control Panel</span>
            <Tag color="red">CRITICAL OPERATIONS</Tag>
          </div>
        }
        extra={
          <Space>
            <Button
              icon={<SettingOutlined />}
              onClick={() => setState(prev => ({ ...prev, configModalVisible: true }))}
            >
              Config
            </Button>
            <Button
              danger
              disabled={state.selectedStrategies.length === 0}
              onClick={() => setState(prev => ({ ...prev, bulkActionModalVisible: true }))}
            >
              Bulk Actions ({state.selectedStrategies.length})
            </Button>
            <Popconfirm
              title="🚨 EMERGENCY STOP ALL"
              description="This will stop ALL running strategies immediately!"
              onConfirm={() => {
                const runningStrategies = strategies.filter(s => s.state === 'running');
                runningStrategies.forEach(strategy => 
                  executeEmergencyAction(strategy.strategyInstanceId, 'emergency_stop', 'System-wide emergency stop', true)
                );
              }}
              okText="STOP ALL"
              okType="danger"
              cancelText="Cancel"
            >
              <Button
                type="primary"
                danger
                icon={<ExclamationCircleOutlined />}
                disabled={systemStatus.runningStrategies === 0}
              >
                EMERGENCY STOP ALL
              </Button>
            </Popconfirm>
          </Space>
        }
      >
        {/* System Overview */}
        {renderSystemOverview()}

        {/* Critical Alerts */}
        {systemStatus.errorStrategies > 0 && (
          <Alert
            message="Strategy Errors Detected"
            description={`${systemStatus.errorStrategies} strategies are in error state and may require immediate attention.`}
            type="error"
            showIcon
            className="mb-4"
          />
        )}

        {systemStatus.dailyPnL < -1000 && (
          <Alert
            message="Significant Daily Loss"
            description={`Daily P&L is ${systemStatus.dailyPnL.toFixed(2)}. Consider risk management actions.`}
            type="warning"
            showIcon
            className="mb-4"
          />
        )}

        {/* Strategy Table */}
        <Table
          dataSource={strategies}
          columns={strategyColumns}
          rowKey="strategyInstanceId"
          pagination={false}
          size="small"
          scroll={{ y: 400 }}
          rowSelection={{
            selectedRowKeys: state.selectedStrategies,
            onChange: (selectedRowKeys) => {
              setState(prev => ({
                ...prev,
                selectedStrategies: selectedRowKeys as string[]
              }));
            }
          }}
        />
      </Card>

      {/* Bulk Action Modal */}
      <Modal
        title="Bulk Emergency Actions"
        visible={state.bulkActionModalVisible}
        onCancel={() => setState(prev => ({ ...prev, bulkActionModalVisible: false }))}
        footer={null}
      >
        <Alert
          message="Bulk Action Warning"
          description={`This action will affect ${state.selectedStrategies.length} selected strategies.`}
          type="warning"
          className="mb-4"
        />

        <Form form={bulkForm} layout="vertical" onFinish={executeBulkAction}>
          <Form.Item label="Action" name="action" required>
            <Select placeholder="Select action">
              <Option value="pause">Pause All Selected</Option>
              <Option value="emergency_stop">Emergency Stop All Selected</Option>
              <Option value="reduce_size">Reduce Position Size</Option>
              <Option value="close_positions">Close All Positions</Option>
            </Select>
          </Form.Item>

          <Form.Item label="Reason" name="reason" required>
            <Input.TextArea rows={3} placeholder="Reason for bulk action..." />
          </Form.Item>

          <Form.Item name="force" valuePropName="checked">
            <Checkbox>Force action (skip confirmations)</Checkbox>
          </Form.Item>

          <div className="flex justify-end space-x-2">
            <Button onClick={() => setState(prev => ({ ...prev, bulkActionModalVisible: false }))}>
              Cancel
            </Button>
            <Button type="primary" danger htmlType="submit" loading={loading}>
              Execute Bulk Action
            </Button>
          </div>
        </Form>
      </Modal>

      {/* Emergency Configuration Modal */}
      <Modal
        title="Emergency Stop Configuration"
        visible={state.configModalVisible}
        onCancel={() => setState(prev => ({ ...prev, configModalVisible: false }))}
        onOk={() => {
          configForm.validateFields().then(values => {
            setState(prev => ({ ...prev, emergencyConfig: values, configModalVisible: false }));
            // Save config to backend
            fetch('/api/v1/nautilus/emergency/config', {
              method: 'PUT',
              headers: { 'Content-Type': 'application/json' },
              body: JSON.stringify(values)
            });
            notification.success({ message: 'Emergency configuration updated' });
          });
        }}
      >
        <Form form={configForm} layout="vertical" initialValues={state.emergencyConfig}>
          <Form.Item name={['triggers', 'maxDailyLoss']} label="Max Daily Loss ($)">
            <Input type="number" />
          </Form.Item>
          
          <Form.Item name={['triggers', 'maxDrawdown']} label="Max Drawdown (%)">
            <Input type="number" step="0.01" />
          </Form.Item>
          
          <Form.Item name={['triggers', 'consecutiveLosses']} label="Consecutive Losses Limit">
            <Input type="number" />
          </Form.Item>

          <Title level={5}>Automatic Actions</Title>
          
          <Form.Item name={['actions', 'closePositions']} valuePropName="checked">
            <Checkbox>Close all positions</Checkbox>
          </Form.Item>
          
          <Form.Item name={['actions', 'cancelOrders']} valuePropName="checked">
            <Checkbox>Cancel pending orders</Checkbox>
          </Form.Item>
          
          <Form.Item name={['actions', 'notifyRiskTeam']} valuePropName="checked">
            <Checkbox>Notify risk management team</Checkbox>
          </Form.Item>
          
          <Form.Item name={['actions', 'createIncidentReport']} valuePropName="checked">
            <Checkbox>Create incident report</Checkbox>
          </Form.Item>
        </Form>
      </Modal>
    </div>
  );
};

export default EmergencyControlPanel;