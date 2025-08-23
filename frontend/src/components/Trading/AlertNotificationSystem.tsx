import React, { useState, useEffect, useRef, useMemo, useCallback, memo } from 'react';
import {
  Card,
  Space,
  Typography,
  Button,
  Modal,
  List,
  Tag,
  Badge,
  Switch,
  Select,
  Input,
  Form,
  InputNumber,
  DatePicker,
  TimePicker,
  Tooltip,
  Popover,
  Dropdown,
  Menu,
  Tabs,
  Alert,
  Divider,
  Row,
  Col,
  Progress,
  Avatar,
  Timeline,
  notification,
  message
} from 'antd';
import {
  BellOutlined,
  WarningOutlined,
  InfoCircleOutlined,
  CheckCircleOutlined,
  CloseCircleOutlined,
  SettingOutlined,
  PlusOutlined,
  DeleteOutlined,
  EditOutlined,
  PlayCircleOutlined,
  PauseCircleOutlined,
  SoundOutlined,
  NotificationOutlined,
  MailOutlined,
  MessageOutlined,
  PhoneOutlined,
  FilterOutlined,
  HistoryOutlined,
  BarChartOutlined,
  ThunderboltOutlined
} from '@ant-design/icons';
import moment from 'moment';
import { useWebSocketStream } from '../../hooks/useWebSocketStream';

const { Text, Title } = Typography;
const { TextArea } = Input;
const { RangePicker } = DatePicker;
const { TabPane } = Tabs;

interface AlertCondition {
  id: string;
  field: 'price' | 'volume' | 'pnl' | 'risk' | 'custom';
  operator: '>' | '<' | '=' | '>=' | '<=' | '!=' | 'between' | 'contains';
  value: any;
  value2?: any; // For 'between' operator
  logicalOperator?: 'AND' | 'OR'; // For combining conditions
}

interface AlertRule {
  id: string;
  name: string;
  description: string;
  enabled: boolean;
  priority: 'low' | 'medium' | 'high' | 'critical';
  category: 'price' | 'risk' | 'performance' | 'system' | 'custom';
  conditions: AlertCondition[];
  actions: AlertAction[];
  schedule: {
    enabled: boolean;
    startTime?: string;
    endTime?: string;
    days?: string[];
    timezone?: string;
  };
  cooldown: number; // Minutes between alerts
  maxAlerts: number; // Maximum alerts per day
  createdAt: number;
  updatedAt: number;
  lastTriggered?: number;
  triggerCount: number;
  metadata: Record<string, any>;
}

interface AlertAction {
  id: string;
  type: 'notification' | 'email' | 'sms' | 'webhook' | 'sound' | 'popup';
  enabled: boolean;
  config: Record<string, any>;
  priority: 'low' | 'medium' | 'high';
}

interface AlertInstance {
  id: string;
  ruleId: string;
  ruleName: string;
  title: string;
  message: string;
  priority: 'low' | 'medium' | 'high' | 'critical';
  category: string;
  status: 'active' | 'acknowledged' | 'resolved' | 'dismissed';
  timestamp: number;
  acknowledgedAt?: number;
  acknowledgedBy?: string;
  resolvedAt?: number;
  resolvedBy?: string;
  data: Record<string, any>;
  actions: AlertAction[];
  escalated: boolean;
  escalatedAt?: number;
  snoozedUntil?: number;
}

interface AlertNotificationSystemProps {
  portfolioId?: string;
  symbols?: string[];
  strategies?: string[];
  autoConnect?: boolean;
  soundEnabled?: boolean;
  popupEnabled?: boolean;
  maxDisplayedAlerts?: number;
  compactMode?: boolean;
  height?: number;
  onAlertCreate?: (rule: AlertRule) => void;
  onAlertTriggered?: (alert: AlertInstance) => void;
  onAlertAcknowledged?: (alertId: string) => void;
  onAlertResolved?: (alertId: string) => void;
}

const AlertNotificationSystem: React.FC<AlertNotificationSystemProps> = memo(({
  portfolioId = 'default',
  symbols = [],
  strategies = [],
  autoConnect = true,
  soundEnabled = true,
  popupEnabled = true,
  maxDisplayedAlerts = 50,
  compactMode = false,
  height = 600,
  onAlertCreate,
  onAlertTriggered,
  onAlertAcknowledged,
  onAlertResolved
}) => {
  const [alertRules, setAlertRules] = useState<AlertRule[]>([]);
  const [activeAlerts, setActiveAlerts] = useState<AlertInstance[]>([]);
  const [alertHistory, setAlertHistory] = useState<AlertInstance[]>([]);
  
  const [ruleModalVisible, setRuleModalVisible] = useState(false);
  const [selectedRule, setSelectedRule] = useState<AlertRule | null>(null);
  const [activeTab, setActiveTab] = useState('active');
  const [filterSeverity, setFilterSeverity] = useState<string>('all');
  const [filterCategory, setFilterCategory] = useState<string>('all');
  const [unacknowledgedCount, setUnacknowledgedCount] = useState(0);

  const audioRef = useRef<HTMLAudioElement>(null);
  const notificationApiRef = useRef<any>(null);

  // WebSocket connection for real-time alerts
  const {
    isConnected,
    lastMessage,
    error: wsError
  } = useWebSocketStream({
    url: `ws://localhost:8001/ws/alerts`,
    autoConnect,
    protocols: ['alert-protocol']
  });

  // Process incoming WebSocket messages
  useEffect(() => {
    if (!lastMessage) return;

    try {
      const data = JSON.parse(lastMessage.data);
      
      switch (data.type) {
        case 'alert_triggered':
          handleNewAlert(data.alert);
          break;
        case 'alert_updated':
          updateAlert(data.alert);
          break;
        case 'rule_created':
          addAlertRule(data.rule);
          break;
        case 'rule_updated':
          updateAlertRule(data.rule);
          break;
        case 'rule_deleted':
          removeAlertRule(data.ruleId);
          break;
      }
    } catch (error) {
      console.error('Error processing alert message:', error);
    }
  }, [lastMessage]);

  // Handle new alert
  const handleNewAlert = useCallback((alert: AlertInstance) => {
    setActiveAlerts(prev => [alert, ...prev].slice(0, maxDisplayedAlerts));
    setAlertHistory(prev => [alert, ...prev]);
    setUnacknowledgedCount(prev => prev + 1);

    // Execute alert actions
    alert.actions.forEach(action => {
      if (!action.enabled) return;

      switch (action.type) {
        case 'sound':
          if (soundEnabled && audioRef.current) {
            audioRef.current.play().catch(console.error);
          }
          break;

        case 'notification':
          if ('Notification' in window && Notification.permission === 'granted') {
            new Notification(alert.title, {
              body: alert.message,
              icon: '/favicon.ico',
              tag: alert.id
            });
          }
          break;

        case 'popup':
          if (popupEnabled) {
            const notifyType = alert.priority === 'critical' ? 'error' : 
                             alert.priority === 'high' ? 'warning' : 
                             'info';
            
            notification[notifyType]({
              message: alert.title,
              description: alert.message,
              duration: alert.priority === 'critical' ? 0 : 
                       alert.priority === 'high' ? 10 : 5,
              key: alert.id,
              onClick: () => acknowledgeAlert(alert.id)
            });
          }
          break;
      }
    });

    onAlertTriggered?.(alert);
  }, [soundEnabled, popupEnabled, maxDisplayedAlerts, onAlertTriggered]);

  // Update existing alert
  const updateAlert = useCallback((updatedAlert: AlertInstance) => {
    setActiveAlerts(prev => prev.map(alert => 
      alert.id === updatedAlert.id ? updatedAlert : alert
    ));
    setAlertHistory(prev => prev.map(alert => 
      alert.id === updatedAlert.id ? updatedAlert : alert
    ));
  }, []);

  // Acknowledge alert
  const acknowledgeAlert = useCallback((alertId: string) => {
    const alert = activeAlerts.find(a => a.id === alertId);
    if (!alert) return;

    const updatedAlert: AlertInstance = {
      ...alert,
      status: 'acknowledged',
      acknowledgedAt: Date.now(),
      acknowledgedBy: 'current_user'
    };

    updateAlert(updatedAlert);
    setUnacknowledgedCount(prev => Math.max(0, prev - 1));
    onAlertAcknowledged?.(alertId);

    // Close notification
    notification.close(alertId);
  }, [activeAlerts, updateAlert, onAlertAcknowledged]);

  // Resolve alert
  const resolveAlert = useCallback((alertId: string) => {
    const alert = activeAlerts.find(a => a.id === alertId);
    if (!alert) return;

    const updatedAlert: AlertInstance = {
      ...alert,
      status: 'resolved',
      resolvedAt: Date.now(),
      resolvedBy: 'current_user'
    };

    updateAlert(updatedAlert);
    setActiveAlerts(prev => prev.filter(a => a.id !== alertId));
    onAlertResolved?.(alertId);
  }, [activeAlerts, updateAlert, onAlertResolved]);

  // Snooze alert
  const snoozeAlert = useCallback((alertId: string, minutes: number) => {
    const alert = activeAlerts.find(a => a.id === alertId);
    if (!alert) return;

    const updatedAlert: AlertInstance = {
      ...alert,
      snoozedUntil: Date.now() + (minutes * 60 * 1000)
    };

    updateAlert(updatedAlert);
    message.success(`Alert snoozed for ${minutes} minutes`);
  }, [activeAlerts, updateAlert]);

  // Alert rule management
  const addAlertRule = useCallback((rule: AlertRule) => {
    setAlertRules(prev => [...prev, rule]);
  }, []);

  const updateAlertRule = useCallback((updatedRule: AlertRule) => {
    setAlertRules(prev => prev.map(rule => 
      rule.id === updatedRule.id ? updatedRule : rule
    ));
  }, []);

  const removeAlertRule = useCallback((ruleId: string) => {
    setAlertRules(prev => prev.filter(rule => rule.id !== ruleId));
  }, []);

  // Create new alert rule
  const createAlertRule = useCallback((ruleData: Partial<AlertRule>) => {
    const newRule: AlertRule = {
      id: `rule_${Date.now()}`,
      name: ruleData.name || 'New Alert Rule',
      description: ruleData.description || '',
      enabled: true,
      priority: ruleData.priority || 'medium',
      category: ruleData.category || 'custom',
      conditions: ruleData.conditions || [],
      actions: ruleData.actions || [
        {
          id: `action_${Date.now()}`,
          type: 'popup',
          enabled: true,
          config: {},
          priority: 'medium'
        }
      ],
      schedule: {
        enabled: false
      },
      cooldown: 5,
      maxAlerts: 100,
      createdAt: Date.now(),
      updatedAt: Date.now(),
      triggerCount: 0,
      metadata: {}
    };

    addAlertRule(newRule);
    onAlertCreate?.(newRule);
    return newRule;
  }, [addAlertRule, onAlertCreate]);

  // Filter alerts based on severity and category
  const filteredActiveAlerts = useMemo(() => {
    return activeAlerts.filter(alert => {
      const severityMatch = filterSeverity === 'all' || alert.priority === filterSeverity;
      const categoryMatch = filterCategory === 'all' || alert.category === filterCategory;
      const notSnoozed = !alert.snoozedUntil || alert.snoozedUntil < Date.now();
      
      return severityMatch && categoryMatch && notSnoozed;
    });
  }, [activeAlerts, filterSeverity, filterCategory]);

  // Get alert icon and color
  const getAlertIcon = (priority: string, status: string) => {
    if (status === 'resolved') return <CheckCircleOutlined style={{ color: '#52c41a' }} />;
    if (status === 'acknowledged') return <InfoCircleOutlined style={{ color: '#1890ff' }} />;
    
    switch (priority) {
      case 'critical': return <CloseCircleOutlined style={{ color: '#ff4d4f' }} />;
      case 'high': return <WarningOutlined style={{ color: '#fa8c16' }} />;
      case 'medium': return <InfoCircleOutlined style={{ color: '#1890ff' }} />;
      case 'low': return <CheckCircleOutlined style={{ color: '#52c41a' }} />;
      default: return <BellOutlined />;
    }
  };

  // Alert rule form
  const AlertRuleForm = () => {
    const [form] = Form.useForm();

    const handleSubmit = async () => {
      try {
        const values = await form.validateFields();
        
        const rule = selectedRule ? {
          ...selectedRule,
          ...values,
          updatedAt: Date.now()
        } : createAlertRule(values);

        if (selectedRule) {
          updateAlertRule(rule);
        }

        setRuleModalVisible(false);
        setSelectedRule(null);
        form.resetFields();
        message.success(`Alert rule ${selectedRule ? 'updated' : 'created'} successfully`);
      } catch (error) {
        console.error('Form validation failed:', error);
      }
    };

    return (
      <Modal
        title={`${selectedRule ? 'Edit' : 'Create'} Alert Rule`}
        open={ruleModalVisible}
        onCancel={() => {
          setRuleModalVisible(false);
          setSelectedRule(null);
          form.resetFields();
        }}
        onOk={handleSubmit}
        width={800}
      >
        <Form
          form={form}
          layout="vertical"
          initialValues={selectedRule}
        >
          <Row gutter={16}>
            <Col span={12}>
              <Form.Item
                name="name"
                label="Rule Name"
                rules={[{ required: true, message: 'Please enter a rule name' }]}
              >
                <Input placeholder="Enter rule name" />
              </Form.Item>
            </Col>
            <Col span={12}>
              <Form.Item name="priority" label="Priority">
                <Select>
                  <Select.Option value="low">Low</Select.Option>
                  <Select.Option value="medium">Medium</Select.Option>
                  <Select.Option value="high">High</Select.Option>
                  <Select.Option value="critical">Critical</Select.Option>
                </Select>
              </Form.Item>
            </Col>
          </Row>

          <Form.Item name="description" label="Description">
            <TextArea rows={2} placeholder="Describe when this alert should trigger" />
          </Form.Item>

          <Row gutter={16}>
            <Col span={8}>
              <Form.Item name="category" label="Category">
                <Select>
                  <Select.Option value="price">Price</Select.Option>
                  <Select.Option value="risk">Risk</Select.Option>
                  <Select.Option value="performance">Performance</Select.Option>
                  <Select.Option value="system">System</Select.Option>
                  <Select.Option value="custom">Custom</Select.Option>
                </Select>
              </Form.Item>
            </Col>
            <Col span={8}>
              <Form.Item name="cooldown" label="Cooldown (minutes)">
                <InputNumber min={1} max={1440} />
              </Form.Item>
            </Col>
            <Col span={8}>
              <Form.Item name="maxAlerts" label="Max Daily Alerts">
                <InputNumber min={1} max={1000} />
              </Form.Item>
            </Col>
          </Row>

          <Divider>Alert Conditions</Divider>
          
          {/* Simplified conditions form - in real implementation, this would be more dynamic */}
          <Row gutter={16}>
            <Col span={8}>
              <Form.Item label="Field">
                <Select defaultValue="price">
                  <Select.Option value="price">Price</Select.Option>
                  <Select.Option value="volume">Volume</Select.Option>
                  <Select.Option value="pnl">P&L</Select.Option>
                  <Select.Option value="risk">Risk</Select.Option>
                </Select>
              </Form.Item>
            </Col>
            <Col span={8}>
              <Form.Item label="Operator">
                <Select defaultValue=">">
                  <Select.Option value=">">Greater than</Select.Option>
                  <Select.Option value="<">Less than</Select.Option>
                  <Select.Option value="=">Equal to</Select.Option>
                  <Select.Option value=">=">Greater or equal</Select.Option>
                  <Select.Option value="<=">Less or equal</Select.Option>
                </Select>
              </Form.Item>
            </Col>
            <Col span={8}>
              <Form.Item label="Value">
                <InputNumber placeholder="Enter value" style={{ width: '100%' }} />
              </Form.Item>
            </Col>
          </Row>

          <Divider>Alert Actions</Divider>
          
          <Row gutter={16}>
            <Col span={6}>
              <Form.Item label="Sound">
                <Switch defaultChecked />
              </Form.Item>
            </Col>
            <Col span={6}>
              <Form.Item label="Popup">
                <Switch defaultChecked />
              </Form.Item>
            </Col>
            <Col span={6}>
              <Form.Item label="Email">
                <Switch />
              </Form.Item>
            </Col>
            <Col span={6}>
              <Form.Item label="SMS">
                <Switch />
              </Form.Item>
            </Col>
          </Row>
        </Form>
      </Modal>
    );
  };

  // Request notification permission on mount
  useEffect(() => {
    if ('Notification' in window && Notification.permission === 'default') {
      Notification.requestPermission();
    }
  }, []);

  return (
    <Card
      title={
        <Space>
          <BellOutlined />
          <Text strong>Alert Notifications</Text>
          <Badge count={unacknowledgedCount} style={{ backgroundColor: '#f5222d' }} />
          <Badge status={isConnected ? 'success' : 'error'} />
        </Space>
      }
      extra={
        <Space>
          {/* Filters */}
          <Select
            value={filterSeverity}
            onChange={setFilterSeverity}
            size="small"
            style={{ width: 100 }}
          >
            <Select.Option value="all">All</Select.Option>
            <Select.Option value="critical">Critical</Select.Option>
            <Select.Option value="high">High</Select.Option>
            <Select.Option value="medium">Medium</Select.Option>
            <Select.Option value="low">Low</Select.Option>
          </Select>

          <Select
            value={filterCategory}
            onChange={setFilterCategory}
            size="small"
            style={{ width: 100 }}
          >
            <Select.Option value="all">All</Select.Option>
            <Select.Option value="price">Price</Select.Option>
            <Select.Option value="risk">Risk</Select.Option>
            <Select.Option value="performance">Performance</Select.Option>
            <Select.Option value="system">System</Select.Option>
          </Select>

          {/* Create Rule */}
          <Button
            size="small"
            icon={<PlusOutlined />}
            onClick={() => setRuleModalVisible(true)}
          >
            New Rule
          </Button>

          {/* Settings */}
          <Button size="small" icon={<SettingOutlined />} />
        </Space>
      }
      size={compactMode ? 'small' : 'default'}
      style={{ height }}
    >
      <Tabs activeKey={activeTab} onChange={setActiveTab} size="small">
        {/* Active Alerts */}
        <TabPane
          tab={
            <Space>
              <ThunderboltOutlined />
              Active
              <Badge count={filteredActiveAlerts.length} size="small" />
            </Space>
          }
          key="active"
        >
          <List
            dataSource={filteredActiveAlerts}
            renderItem={alert => (
              <List.Item
                actions={[
                  <Dropdown
                    key="snooze"
                    overlay={
                      <Menu
                        onClick={({ key }) => snoozeAlert(alert.id, parseInt(key))}
                        items={[
                          { key: '5', label: '5 minutes' },
                          { key: '15', label: '15 minutes' },
                          { key: '30', label: '30 minutes' },
                          { key: '60', label: '1 hour' },
                          { key: '240', label: '4 hours' }
                        ]}
                      />
                    }
                    trigger={['click']}
                  >
                    <Button size="small">Snooze</Button>
                  </Dropdown>,
                  <Button
                    key="ack"
                    size="small"
                    onClick={() => acknowledgeAlert(alert.id)}
                    disabled={alert.status === 'acknowledged'}
                  >
                    Acknowledge
                  </Button>,
                  <Button
                    key="resolve"
                    size="small"
                    type="primary"
                    onClick={() => resolveAlert(alert.id)}
                  >
                    Resolve
                  </Button>
                ]}
                style={{
                  backgroundColor: alert.status === 'acknowledged' ? '#f6ffed' : 
                                  alert.priority === 'critical' ? '#fff2f0' :
                                  alert.priority === 'high' ? '#fff7e6' : 'transparent'
                }}
              >
                <List.Item.Meta
                  avatar={
                    <Avatar
                      icon={getAlertIcon(alert.priority, alert.status)}
                      size="small"
                    />
                  }
                  title={
                    <Space>
                      <Text strong>{alert.title}</Text>
                      <Tag color={
                        alert.priority === 'critical' ? 'red' :
                        alert.priority === 'high' ? 'orange' :
                        alert.priority === 'medium' ? 'blue' : 'green'
                      }>
                        {alert.priority.toUpperCase()}
                      </Tag>
                      <Text type="secondary" style={{ fontSize: '11px' }}>
                        {moment(alert.timestamp).fromNow()}
                      </Text>
                    </Space>
                  }
                  description={
                    <Space direction="vertical" size="small">
                      <Text>{alert.message}</Text>
                      {alert.status === 'acknowledged' && (
                        <Text type="secondary" style={{ fontSize: '11px' }}>
                          Acknowledged {moment(alert.acknowledgedAt).fromNow()}
                        </Text>
                      )}
                    </Space>
                  }
                />
              </List.Item>
            )}
            style={{ height: height - 150, overflow: 'auto' }}
          />
        </TabPane>

        {/* Alert Rules */}
        <TabPane
          tab={
            <Space>
              <SettingOutlined />
              Rules
              <Badge count={alertRules.filter(r => r.enabled).length} size="small" />
            </Space>
          }
          key="rules"
        >
          <List
            dataSource={alertRules}
            renderItem={rule => (
              <List.Item
                actions={[
                  <Switch
                    key="toggle"
                    checked={rule.enabled}
                    onChange={(checked) => {
                      updateAlertRule({ ...rule, enabled: checked });
                    }}
                    size="small"
                  />,
                  <Button
                    key="edit"
                    size="small"
                    icon={<EditOutlined />}
                    onClick={() => {
                      setSelectedRule(rule);
                      setRuleModalVisible(true);
                    }}
                  />,
                  <Button
                    key="delete"
                    size="small"
                    icon={<DeleteOutlined />}
                    danger
                    onClick={() => removeAlertRule(rule.id)}
                  />
                ]}
              >
                <List.Item.Meta
                  avatar={<Avatar icon={<BellOutlined />} size="small" />}
                  title={
                    <Space>
                      <Text strong>{rule.name}</Text>
                      <Tag color={rule.enabled ? 'green' : 'default'}>
                        {rule.enabled ? 'Enabled' : 'Disabled'}
                      </Tag>
                      <Tag color={
                        rule.priority === 'critical' ? 'red' :
                        rule.priority === 'high' ? 'orange' :
                        rule.priority === 'medium' ? 'blue' : 'green'
                      }>
                        {rule.priority}
                      </Tag>
                    </Space>
                  }
                  description={
                    <Space direction="vertical" size="small">
                      <Text>{rule.description}</Text>
                      <Text type="secondary" style={{ fontSize: '11px' }}>
                        Triggered {rule.triggerCount} times • 
                        Cooldown: {rule.cooldown}m • 
                        Max daily: {rule.maxAlerts}
                      </Text>
                    </Space>
                  }
                />
              </List.Item>
            )}
            style={{ height: height - 150, overflow: 'auto' }}
          />
        </TabPane>

        {/* Alert History */}
        <TabPane
          tab={
            <Space>
              <HistoryOutlined />
              History
              <Badge count={alertHistory.length} size="small" />
            </Space>
          }
          key="history"
        >
          <Timeline
            style={{ height: height - 150, overflow: 'auto', padding: '16px 0' }}
          >
            {alertHistory.slice(0, 50).map(alert => (
              <Timeline.Item
                key={alert.id}
                dot={getAlertIcon(alert.priority, alert.status)}
                color={
                  alert.status === 'resolved' ? 'green' :
                  alert.status === 'acknowledged' ? 'blue' :
                  alert.priority === 'critical' ? 'red' :
                  alert.priority === 'high' ? 'orange' : 'blue'
                }
              >
                <div>
                  <Text strong>{alert.title}</Text>
                  <br />
                  <Text type="secondary">{alert.message}</Text>
                  <br />
                  <Text type="secondary" style={{ fontSize: '11px' }}>
                    {moment(alert.timestamp).format('MMM DD, HH:mm:ss')} • 
                    Status: {alert.status} • 
                    Priority: {alert.priority}
                  </Text>
                </div>
              </Timeline.Item>
            ))}
          </Timeline>
        </TabPane>

        {/* Analytics */}
        <TabPane
          tab={
            <Space>
              <BarChartOutlined />
              Analytics
            </Space>
          }
          key="analytics"
        >
          <Row gutter={[16, 16]}>
            <Col span={6}>
              <Card size="small">
                <Text type="secondary">Total Alerts</Text>
                <div style={{ fontSize: '24px', fontWeight: 'bold' }}>
                  {alertHistory.length}
                </div>
              </Card>
            </Col>
            <Col span={6}>
              <Card size="small">
                <Text type="secondary">Active Rules</Text>
                <div style={{ fontSize: '24px', fontWeight: 'bold' }}>
                  {alertRules.filter(r => r.enabled).length}
                </div>
              </Card>
            </Col>
            <Col span={6}>
              <Card size="small">
                <Text type="secondary">Critical Alerts</Text>
                <div style={{ fontSize: '24px', fontWeight: 'bold', color: '#f5222d' }}>
                  {alertHistory.filter(a => a.priority === 'critical').length}
                </div>
              </Card>
            </Col>
            <Col span={6}>
              <Card size="small">
                <Text type="secondary">Avg Response Time</Text>
                <div style={{ fontSize: '24px', fontWeight: 'bold' }}>
                  {Math.round(
                    alertHistory
                      .filter(a => a.acknowledgedAt)
                      .reduce((sum, a) => sum + ((a.acknowledgedAt! - a.timestamp) / 60000), 0) /
                    Math.max(1, alertHistory.filter(a => a.acknowledgedAt).length)
                  )}m
                </div>
              </Card>
            </Col>
          </Row>

          <Divider>Alert Distribution</Divider>
          
          <Row gutter={[16, 16]}>
            <Col span={12}>
              <Card size="small" title="By Priority">
                {['critical', 'high', 'medium', 'low'].map(priority => {
                  const count = alertHistory.filter(a => a.priority === priority).length;
                  const percentage = alertHistory.length > 0 ? (count / alertHistory.length) * 100 : 0;
                  
                  return (
                    <div key={priority} style={{ marginBottom: 8 }}>
                      <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                        <Text style={{ textTransform: 'capitalize' }}>{priority}</Text>
                        <Text>{count}</Text>
                      </div>
                      <Progress
                        percent={percentage}
                        size="small"
                        strokeColor={
                          priority === 'critical' ? '#f5222d' :
                          priority === 'high' ? '#fa8c16' :
                          priority === 'medium' ? '#1890ff' : '#52c41a'
                        }
                        showInfo={false}
                      />
                    </div>
                  );
                })}
              </Card>
            </Col>
            <Col span={12}>
              <Card size="small" title="By Category">
                {['price', 'risk', 'performance', 'system'].map(category => {
                  const count = alertHistory.filter(a => a.category === category).length;
                  const percentage = alertHistory.length > 0 ? (count / alertHistory.length) * 100 : 0;
                  
                  return (
                    <div key={category} style={{ marginBottom: 8 }}>
                      <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                        <Text style={{ textTransform: 'capitalize' }}>{category}</Text>
                        <Text>{count}</Text>
                      </div>
                      <Progress
                        percent={percentage}
                        size="small"
                        showInfo={false}
                      />
                    </div>
                  );
                })}
              </Card>
            </Col>
          </Row>
        </TabPane>
      </Tabs>

      {/* Alert Rule Modal */}
      <AlertRuleForm />

      {/* Hidden audio element for alert sounds */}
      <audio
        ref={audioRef}
        preload="auto"
        src="data:audio/wav;base64,UklGRuICAABXQVZFZm10IBAAAAABAAEAgD4AAIB+AQACABAAZGXFigAAAAAAAAAAAFAAAAHAAAAAgD4AAIB+AQACABAAZGEEBEJMTElTVwAAAGluZm8AAAAPAAAAUQAAUQAGBggRERMnJygpKTY2OFA="
      />

      {/* Connection Status */}
      {!compactMode && (
        <div style={{
          position: 'absolute',
          bottom: 8,
          right: 16,
          fontSize: '11px',
          color: '#999'
        }}>
          <Space>
            <Text type="secondary">
              WebSocket: {isConnected ? 'Connected' : 'Disconnected'}
            </Text>
            {wsError && (
              <Tooltip title={wsError}>
                <WarningOutlined style={{ color: '#f5222d' }} />
              </Tooltip>
            )}
          </Space>
        </div>
      )}
    </Card>
  );
});

AlertNotificationSystem.displayName = 'AlertNotificationSystem';

export default AlertNotificationSystem;