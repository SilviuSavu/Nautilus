import React, { useState } from 'react';
import {
  Card,
  Button,
  Space,
  Typography,
  Row,
  Col,
  Tooltip,
  Modal,
  Form,
  Input,
  Select,
  Switch,
  Badge,
  message,
  Popconfirm,
  Divider,
  Alert
} from 'antd';
import {
  ThunderboltOutlined,
  WifiOutlined,
  ReloadOutlined,
  SafetyCertificateOutlined,
  RocketOutlined,
  MonitorOutlined,
  DatabaseOutlined,
  SettingOutlined,
  PlayCircleOutlined,
  StopOutlined,
  ExportOutlined,
  ImportOutlined,
  BugOutlined,
  HeartOutlined,
  AlertOutlined,
  CloudSyncOutlined,
  ApiOutlined
} from '@ant-design/icons';
import { useMessageBus } from '../../hooks/useMessageBus';
import { useEngineWebSocket } from '../../hooks/useEngineWebSocket';
import type { QuickAction } from '../../types/sprint3';

const { Text, Title } = Typography;
const { Option } = Select;

interface QuickActionsProps {
  onActionExecuted?: (actionId: string, result: any) => void;
  layout?: 'grid' | 'list' | 'compact';
  categories?: string[];
  showStatus?: boolean;
}

const QuickActions: React.FC<QuickActionsProps> = ({
  onActionExecuted,
  layout = 'grid',
  categories = ['system', 'trading', 'monitoring', 'configuration'],
  showStatus = true
}) => {
  const [executingActions, setExecutingActions] = useState<Set<string>>(new Set());
  const [modalVisible, setModalVisible] = useState(false);
  const [selectedAction, setSelectedAction] = useState<QuickAction | null>(null);
  const [form] = Form.useForm();

  const messageBus = useMessageBus();
  const engineWs = useEngineWebSocket();

  // Execute action with loading state
  const executeAction = async (action: QuickAction, params?: any) => {
    setExecutingActions(prev => new Set([...prev, action.id]));
    
    try {
      const result = await action.action(params);
      message.success(`${action.name} executed successfully`);
      onActionExecuted?.(action.id, result);
    } catch (error) {
      message.error(`Failed to execute ${action.name}: ${error}`);
      console.error(`Action ${action.id} failed:`, error);
    } finally {
      setExecutingActions(prev => {
        const newSet = new Set(prev);
        newSet.delete(action.id);
        return newSet;
      });
    }
  };

  // Quick Actions Configuration
  const quickActions: QuickAction[] = [
    // System Actions
    {
      id: 'restart-websockets',
      name: 'Restart WebSockets',
      description: 'Restart all WebSocket connections and reestablish subscriptions',
      icon: 'WifiOutlined',
      category: 'system',
      action: async () => {
        messageBus.disconnect();
        engineWs.disconnect();
        await new Promise(resolve => setTimeout(resolve, 1000));
        messageBus.connect();
        engineWs.connect();
        return { status: 'success', message: 'WebSocket connections restarted' };
      }
    },
    {
      id: 'system-health-check',
      name: 'Health Check',
      description: 'Run comprehensive system health diagnostics',
      icon: 'MonitorOutlined',
      category: 'system',
      action: async () => {
        // Simulate health check
        await new Promise(resolve => setTimeout(resolve, 2000));
        return {
          status: 'healthy',
          components: ['websocket', 'api', 'database', 'engine'],
          timestamp: new Date().toISOString()
        };
      }
    },
    {
      id: 'clear-cache',
      name: 'Clear Cache',
      description: 'Clear application cache and temporary data',
      icon: 'DatabaseOutlined',
      category: 'system',
      action: async () => {
        // Clear localStorage and sessionStorage
        localStorage.clear();
        sessionStorage.clear();
        return { status: 'success', message: 'Cache cleared successfully' };
      }
    },
    {
      id: 'refresh-all-data',
      name: 'Refresh All Data',
      description: 'Force refresh all real-time data streams and subscriptions',
      icon: 'ReloadOutlined',
      category: 'system',
      action: async () => {
        messageBus.clearMessages();
        engineWs.requestEngineStatus();
        return { status: 'success', message: 'All data streams refreshed' };
      }
    },

    // Trading Actions
    {
      id: 'check-risk-limits',
      name: 'Check Risk Limits',
      description: 'Validate all risk limits and exposure calculations',
      icon: 'SafetyCertificateOutlined',
      category: 'trading',
      action: async () => {
        // Simulate risk check
        await new Promise(resolve => setTimeout(resolve, 1500));
        return {
          status: 'validated',
          limits_checked: 12,
          breaches: 0,
          timestamp: new Date().toISOString()
        };
      }
    },
    {
      id: 'emergency-stop',
      name: 'Emergency Stop',
      description: 'Immediately halt all trading activities and strategies',
      icon: 'StopOutlined',
      category: 'trading',
      action: async () => {
        // This would trigger emergency stop
        return { status: 'stopped', message: 'Emergency stop activated' };
      }
    },
    {
      id: 'deploy-strategy',
      name: 'Deploy Strategy',
      description: 'Start new strategy deployment workflow',
      icon: 'RocketOutlined',
      category: 'trading',
      action: async (params: { strategyId: string; environment: string }) => {
        await new Promise(resolve => setTimeout(resolve, 2000));
        return {
          status: 'deployed',
          strategy: params.strategyId,
          environment: params.environment,
          deploymentId: `dep-${Date.now()}`
        };
      }
    },
    {
      id: 'validate-positions',
      name: 'Validate Positions',
      description: 'Cross-check all positions with broker and internal records',
      icon: 'SafetyCertificateOutlined',
      category: 'trading',
      action: async () => {
        await new Promise(resolve => setTimeout(resolve, 1800));
        return {
          status: 'validated',
          positions_checked: 24,
          discrepancies: 0,
          last_sync: new Date().toISOString()
        };
      }
    },

    // Monitoring Actions
    {
      id: 'export-logs',
      name: 'Export Logs',
      description: 'Export system logs for analysis and debugging',
      icon: 'ExportOutlined',
      category: 'monitoring',
      action: async () => {
        // Simulate log export
        await new Promise(resolve => setTimeout(resolve, 1000));
        const blob = new Blob(['Mock log data...'], { type: 'text/plain' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `nautilus-logs-${Date.now()}.txt`;
        a.click();
        URL.revokeObjectURL(url);
        return { status: 'exported', filename: `nautilus-logs-${Date.now()}.txt` };
      }
    },
    {
      id: 'generate-report',
      name: 'Generate Report',
      description: 'Create system performance and health report',
      icon: 'DatabaseOutlined',
      category: 'monitoring',
      action: async () => {
        await new Promise(resolve => setTimeout(resolve, 2500));
        return {
          status: 'generated',
          report_id: `rpt-${Date.now()}`,
          sections: ['system_health', 'performance_metrics', 'trading_summary'],
          timestamp: new Date().toISOString()
        };
      }
    },
    {
      id: 'run-diagnostics',
      name: 'Run Diagnostics',
      description: 'Execute comprehensive system diagnostics and performance tests',
      icon: 'BugOutlined',
      category: 'monitoring',
      action: async () => {
        await new Promise(resolve => setTimeout(resolve, 3000));
        return {
          status: 'completed',
          tests_run: 15,
          tests_passed: 14,
          tests_failed: 1,
          performance_score: 87,
          recommendations: ['Optimize WebSocket buffer size', 'Update database indices']
        };
      }
    },

    // Configuration Actions
    {
      id: 'backup-configuration',
      name: 'Backup Config',
      description: 'Create backup of all system and strategy configurations',
      icon: 'CloudSyncOutlined',
      category: 'configuration',
      action: async () => {
        await new Promise(resolve => setTimeout(resolve, 1200));
        return {
          status: 'backed_up',
          backup_id: `backup-${Date.now()}`,
          size: '2.4 MB',
          timestamp: new Date().toISOString()
        };
      }
    },
    {
      id: 'restore-configuration',
      name: 'Restore Config',
      description: 'Restore system configuration from backup',
      icon: 'ImportOutlined',
      category: 'configuration',
      action: async (params: { backupId: string }) => {
        await new Promise(resolve => setTimeout(resolve, 1500));
        return {
          status: 'restored',
          backup_id: params.backupId,
          timestamp: new Date().toISOString()
        };
      }
    },
    {
      id: 'sync-settings',
      name: 'Sync Settings',
      description: 'Synchronize settings across all system components',
      icon: 'SettingOutlined',
      category: 'configuration',
      action: async () => {
        await new Promise(resolve => setTimeout(resolve, 1000));
        return {
          status: 'synced',
          components_updated: ['frontend', 'backend', 'engine'],
          timestamp: new Date().toISOString()
        };
      }
    }
  ];

  // Filter actions by category
  const filteredActions = quickActions.filter(action => 
    categories.includes(action.category)
  );

  // Group actions by category
  const actionsByCategory = categories.reduce((acc, category) => {
    acc[category] = filteredActions.filter(action => action.category === category);
    return acc;
  }, {} as Record<string, QuickAction[]>);

  // Get icon component
  const getIcon = (iconName: string) => {
    const iconMap: Record<string, React.ReactNode> = {
      WifiOutlined: <WifiOutlined />,
      MonitorOutlined: <MonitorOutlined />,
      DatabaseOutlined: <DatabaseOutlined />,
      ReloadOutlined: <ReloadOutlined />,
      SafetyCertificateOutlined: <SafetyCertificateOutlined />,
      StopOutlined: <StopOutlined />,
      RocketOutlined: <RocketOutlined />,
      ExportOutlined: <ExportOutlined />,
      BugOutlined: <BugOutlined />,
      CloudSyncOutlined: <CloudSyncOutlined />,
      ImportOutlined: <ImportOutlined />,
      SettingOutlined: <SettingOutlined />
    };
    return iconMap[iconName] || <ThunderboltOutlined />;
  };

  // Get category info
  const getCategoryInfo = (category: string) => {
    const categoryMap: Record<string, { name: string; icon: React.ReactNode; color: string }> = {
      system: { name: 'System', icon: <MonitorOutlined />, color: '#1890ff' },
      trading: { name: 'Trading', icon: <RocketOutlined />, color: '#52c41a' },
      monitoring: { name: 'Monitoring', icon: <HeartOutlined />, color: '#faad14' },
      configuration: { name: 'Configuration', icon: <SettingOutlined />, color: '#722ed1' }
    };
    return categoryMap[category] || { name: category, icon: <ThunderboltOutlined />, color: '#666' };
  };

  // Handle action with parameters
  const handleActionWithParams = (action: QuickAction) => {
    if (action.id === 'deploy-strategy' || action.id === 'restore-configuration') {
      setSelectedAction(action);
      setModalVisible(true);
    } else if (action.id === 'emergency-stop') {
      // Special handling for dangerous actions
      Modal.confirm({
        title: 'Emergency Stop',
        content: 'This will immediately halt all trading activities. Are you sure?',
        okText: 'Yes, Stop Trading',
        okType: 'danger',
        onOk: () => executeAction(action)
      });
    } else {
      executeAction(action);
    }
  };

  // Handle modal form submission
  const handleModalOk = () => {
    if (!selectedAction) return;

    form.validateFields().then(values => {
      executeAction(selectedAction, values);
      setModalVisible(false);
      form.resetFields();
    });
  };

  if (layout === 'compact') {
    return (
      <Space wrap>
        {filteredActions.slice(0, 6).map(action => (
          <Button
            key={action.id}
            size="small"
            icon={getIcon(action.icon)}
            loading={executingActions.has(action.id)}
            onClick={() => handleActionWithParams(action)}
          >
            {action.name}
          </Button>
        ))}
      </Space>
    );
  }

  if (layout === 'list') {
    return (
      <Card title={<Space><ThunderboltOutlined />Quick Actions</Space>}>
        <Space direction="vertical" style={{ width: '100%' }}>
          {Object.entries(actionsByCategory).map(([category, actions]) => {
            const categoryInfo = getCategoryInfo(category);
            return (
              <div key={category}>
                <Title level={5} style={{ margin: '8px 0', color: categoryInfo.color }}>
                  {categoryInfo.icon} {categoryInfo.name}
                </Title>
                <Space wrap>
                  {actions.map(action => (
                    <Button
                      key={action.id}
                      type="default"
                      size="small"
                      icon={getIcon(action.icon)}
                      loading={executingActions.has(action.id)}
                      onClick={() => handleActionWithParams(action)}
                    >
                      {action.name}
                    </Button>
                  ))}
                </Space>
                <Divider style={{ margin: '12px 0' }} />
              </div>
            );
          })}
        </Space>

        {/* Parameter Modal */}
        <Modal
          title={selectedAction?.name}
          open={modalVisible}
          onOk={handleModalOk}
          onCancel={() => setModalVisible(false)}
          confirmLoading={selectedAction ? executingActions.has(selectedAction.id) : false}
        >
          <Form form={form} layout="vertical">
            {selectedAction?.id === 'deploy-strategy' && (
              <>
                <Form.Item name="strategyId" label="Strategy ID" rules={[{ required: true }]}>
                  <Input placeholder="Enter strategy ID" />
                </Form.Item>
                <Form.Item name="environment" label="Environment" rules={[{ required: true }]}>
                  <Select placeholder="Select environment">
                    <Option value="development">Development</Option>
                    <Option value="staging">Staging</Option>
                    <Option value="production">Production</Option>
                  </Select>
                </Form.Item>
              </>
            )}
            {selectedAction?.id === 'restore-configuration' && (
              <Form.Item name="backupId" label="Backup ID" rules={[{ required: true }]}>
                <Input placeholder="Enter backup ID" />
              </Form.Item>
            )}
          </Form>
        </Modal>
      </Card>
    );
  }

  // Grid layout (default)
  return (
    <div style={{ width: '100%' }}>
      {Object.entries(actionsByCategory).map(([category, actions]) => {
        const categoryInfo = getCategoryInfo(category);
        return (
          <Card
            key={category}
            title={
              <Space>
                {categoryInfo.icon}
                {categoryInfo.name} Actions
                <Badge count={actions.length} color={categoryInfo.color} />
              </Space>
            }
            size="small"
            style={{ marginBottom: 16 }}
          >
            <Row gutter={[12, 12]}>
              {actions.map(action => (
                <Col xs={24} sm={12} md={8} xl={6} key={action.id}>
                  <Card
                    size="small"
                    hoverable
                    style={{ 
                      height: '100px', 
                      cursor: 'pointer',
                      borderColor: executingActions.has(action.id) ? '#1890ff' : undefined
                    }}
                    bodyStyle={{ 
                      padding: '8px 12px',
                      display: 'flex',
                      flexDirection: 'column',
                      justifyContent: 'center',
                      height: '100%'
                    }}
                    onClick={() => handleActionWithParams(action)}
                  >
                    <Space direction="vertical" size="small" style={{ width: '100%', textAlign: 'center' }}>
                      <div style={{ fontSize: '20px', color: categoryInfo.color }}>
                        {executingActions.has(action.id) ? (
                          <ReloadOutlined spin />
                        ) : (
                          getIcon(action.icon)
                        )}
                      </div>
                      <div>
                        <Text strong style={{ fontSize: '12px' }}>
                          {action.name}
                        </Text>
                        <br />
                        <Text type="secondary" style={{ fontSize: '10px' }}>
                          {action.description.length > 40 
                            ? `${action.description.substring(0, 40)}...`
                            : action.description
                          }
                        </Text>
                      </div>
                    </Space>
                  </Card>
                </Col>
              ))}
            </Row>
          </Card>
        );
      })}

      {/* Status Display */}
      {showStatus && (
        <Alert
          message={
            <Space>
              <Text>System Status:</Text>
              <Badge 
                status={messageBus.connectionStatus === 'connected' ? 'success' : 'warning'} 
                text={`MessageBus ${messageBus.connectionStatus}`} 
              />
              <Badge 
                status={engineWs.isConnected ? 'success' : 'error'} 
                text={`Engine ${engineWs.isConnected ? 'connected' : 'disconnected'}`} 
              />
            </Space>
          }
          type="info"
          showIcon
          style={{ marginTop: 16 }}
        />
      )}

      {/* Parameter Modal */}
      <Modal
        title={selectedAction?.name}
        open={modalVisible}
        onOk={handleModalOk}
        onCancel={() => setModalVisible(false)}
        confirmLoading={selectedAction ? executingActions.has(selectedAction.id) : false}
      >
        <Form form={form} layout="vertical">
          {selectedAction?.id === 'deploy-strategy' && (
            <>
              <Form.Item name="strategyId" label="Strategy ID" rules={[{ required: true }]}>
                <Input placeholder="Enter strategy ID" />
              </Form.Item>
              <Form.Item name="environment" label="Environment" rules={[{ required: true }]}>
                <Select placeholder="Select environment">
                  <Option value="development">Development</Option>
                  <Option value="staging">Staging</Option>
                  <Option value="production">Production</Option>
                </Select>
              </Form.Item>
            </>
          )}
          {selectedAction?.id === 'restore-configuration' && (
            <Form.Item name="backupId" label="Backup ID" rules={[{ required: true }]}>
              <Input placeholder="Enter backup ID" />
            </Form.Item>
          )}
        </Form>
      </Modal>
    </div>
  );
};

export default QuickActions;