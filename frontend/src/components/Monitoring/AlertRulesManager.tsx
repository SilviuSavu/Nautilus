/**
 * Sprint 3: Alert Rules Manager
 * Advanced alert rule management with 30+ alerting rules across 6 categories
 * ML-based breach prediction and automated response workflows
 */

import React, { useState, useEffect } from 'react';
import {
  Card,
  Row,
  Col,
  Typography,
  Space,
  Button,
  Table,
  Alert,
  Spin,
  Tag,
  Switch,
  Modal,
  Form,
  Input,
  Select,
  InputNumber,
  Tooltip,
  Badge,
  List,
  Divider,
  notification,
  Popconfirm,
  Tabs,
  Progress,
  Timeline,
  Drawer
} from 'antd';
import {
  AlertOutlined,
  SettingOutlined,
  PlusOutlined,
  EditOutlined,
  DeleteOutlined,
  PlayCircleOutlined,
  PauseCircleOutlined,
  BellOutlined,
  ThunderboltOutlined,
  WarningOutlined,
  CheckCircleOutlined,
  CloseCircleOutlined,
  ExclamationCircleOutlined,
  RobotOutlined,
  MailOutlined,
  PhoneOutlined,
  SlackOutlined
} from '@ant-design/icons';
import type { ColumnsType } from 'antd/es/table';

const { Title, Text } = Typography;
const { Option } = Select;
const { TabPane } = Tabs;
const { TextArea } = Input;

interface AlertRule {
  id: string;
  name: string;
  category: 'system' | 'trading' | 'risk' | 'websocket' | 'database' | 'strategy';
  metric_name: string;
  condition: 'greater_than' | 'less_than' | 'equals' | 'not_equals' | 'rate_increase' | 'rate_decrease';
  threshold_value: number;
  severity: 'critical' | 'high' | 'medium' | 'low';
  enabled: boolean;
  notification_channels: {
    email: string[];
    slack_webhook?: string;
    teams_webhook?: string;
    sms_numbers?: string[];
  };
  escalation_rules: {
    escalate_after_minutes: number;
    escalation_contacts: string[];
  };
  auto_resolution: {
    enabled: boolean;
    resolution_threshold?: number;
    max_attempts?: number;
  };
  ml_prediction: {
    enabled: boolean;
    prediction_horizon_minutes: number;
    confidence_threshold: number;
  };
  created_at: string;
  last_modified: string;
  last_triggered?: string;
  trigger_count_24h: number;
  avg_resolution_time_minutes: number;
}

interface ActiveAlert {
  id: string;
  rule_id: string;
  rule_name: string;
  category: string;
  severity: 'critical' | 'high' | 'medium' | 'low';
  metric_name: string;
  current_value: number;
  threshold_value: number;
  triggered_at: string;
  acknowledged: boolean;
  acknowledged_by?: string;
  acknowledged_at?: string;
  auto_resolve_attempts: number;
  escalation_level: number;
  notifications_sent: number;
  predicted_breach?: {
    confidence: number;
    time_to_breach_minutes: number;
  };
}

interface AlertRulesManagerProps {
  className?: string;
}

export const AlertRulesManager: React.FC<AlertRulesManagerProps> = ({
  className
}) => {
  const [activeTab, setActiveTab] = useState('rules');
  const [alertRules, setAlertRules] = useState<AlertRule[]>([]);
  const [activeAlerts, setActiveAlerts] = useState<ActiveAlert[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [createModalVisible, setCreateModalVisible] = useState(false);
  const [editingRule, setEditingRule] = useState<AlertRule | null>(null);
  const [selectedCategory, setSelectedCategory] = useState<string>('all');
  const [alertDetailDrawer, setAlertDetailDrawer] = useState<string | null>(null);
  const [form] = Form.useForm();

  // Pre-defined alert rule templates for Sprint 3 components
  const ruleTemplates = [
    {
      category: 'websocket',
      name: 'WebSocket Connection Overload',
      metric_name: 'websocket_connections_active',
      condition: 'greater_than',
      threshold_value: 1000,
      severity: 'high',
      description: 'WebSocket connections exceeding capacity'
    },
    {
      category: 'websocket',
      name: 'Message Throughput Critical',
      metric_name: 'websocket_messages_per_second',
      condition: 'greater_than',
      threshold_value: 50000,
      severity: 'critical',
      description: 'Message throughput approaching system limits'
    },
    {
      category: 'trading',
      name: 'Trading Latency Spike',
      metric_name: 'trading_latency_ms',
      condition: 'greater_than',
      threshold_value: 100,
      severity: 'high',
      description: 'Trading execution latency above acceptable threshold'
    },
    {
      category: 'risk',
      name: 'Risk Violation Rate High',
      metric_name: 'risk_violations_per_minute',
      condition: 'greater_than',
      threshold_value: 5,
      severity: 'critical',
      description: 'High rate of risk rule violations detected'
    },
    {
      category: 'system',
      name: 'CPU Usage Critical',
      metric_name: 'system_cpu_usage_percent',
      condition: 'greater_than',
      threshold_value: 85,
      severity: 'critical',
      description: 'System CPU usage approaching critical levels'
    },
    {
      category: 'database',
      name: 'Database Query Latency',
      metric_name: 'database_query_duration_ms',
      condition: 'greater_than',
      threshold_value: 1000,
      severity: 'medium',
      description: 'Database query latency degradation detected'
    }
  ];

  const fetchAlertRules = async () => {
    try {
      const response = await fetch(`${import.meta.env.VITE_API_BASE_URL}/api/v1/system/alerts/rules`);
      if (!response.ok) {
        throw new Error(`Failed to fetch alert rules: ${response.statusText}`);
      }
      const data: AlertRule[] = await response.json();
      setAlertRules(data);
      setError(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to fetch alert rules');
      console.error('Alert rules fetch error:', err);
    }
  };

  const fetchActiveAlerts = async () => {
    try {
      const response = await fetch(`${import.meta.env.VITE_API_BASE_URL}/api/v1/system/alerts/active`);
      if (!response.ok) {
        throw new Error(`Failed to fetch active alerts: ${response.statusText}`);
      }
      const data: ActiveAlert[] = await response.json();
      setActiveAlerts(data);
      setError(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to fetch active alerts');
      console.error('Active alerts fetch error:', err);
    }
  };

  const fetchData = async () => {
    setLoading(true);
    try {
      await Promise.all([fetchAlertRules(), fetchActiveAlerts()]);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchData();
    const interval = setInterval(fetchActiveAlerts, 10000); // Refresh alerts every 10 seconds
    return () => clearInterval(interval);
  }, []);

  const handleCreateRule = async (values: any) => {
    try {
      const response = await fetch(`${import.meta.env.VITE_API_BASE_URL}/api/v1/system/alerts/rules`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          ...values,
          notification_channels: {
            email: values.email_recipients?.split(',').map((email: string) => email.trim()) || [],
            slack_webhook: values.slack_webhook || undefined,
            sms_numbers: values.sms_numbers?.split(',').map((num: string) => num.trim()) || []
          },
          escalation_rules: {
            escalate_after_minutes: values.escalate_after_minutes || 30,
            escalation_contacts: values.escalation_contacts?.split(',').map((contact: string) => contact.trim()) || []
          },
          auto_resolution: {
            enabled: values.auto_resolution_enabled || false,
            resolution_threshold: values.resolution_threshold,
            max_attempts: values.max_auto_attempts || 3
          },
          ml_prediction: {
            enabled: values.ml_prediction_enabled || false,
            prediction_horizon_minutes: values.prediction_horizon_minutes || 15,
            confidence_threshold: values.ml_confidence_threshold || 0.8
          }
        })
      });

      if (response.ok) {
        notification.success({
          message: 'Alert Rule Created',
          description: 'New alert rule has been created successfully',
          duration: 3
        });
        setCreateModalVisible(false);
        form.resetFields();
        await fetchAlertRules();
      } else {
        throw new Error('Failed to create alert rule');
      }
    } catch (err) {
      notification.error({
        message: 'Failed to Create Rule',
        description: 'Could not create the alert rule',
        duration: 4
      });
    }
  };

  const handleToggleRule = async (ruleId: string, enabled: boolean) => {
    try {
      const response = await fetch(`${import.meta.env.VITE_API_BASE_URL}/api/v1/system/alerts/rules/${ruleId}`, {
        method: 'PATCH',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ enabled })
      });

      if (response.ok) {
        notification.success({
          message: `Alert Rule ${enabled ? 'Enabled' : 'Disabled'}`,
          description: `Rule has been ${enabled ? 'enabled' : 'disabled'}`,
          duration: 2
        });
        await fetchAlertRules();
      }
    } catch (err) {
      notification.error({
        message: 'Failed to Update Rule',
        description: 'Could not update the alert rule status',
        duration: 4
      });
    }
  };

  const handleDeleteRule = async (ruleId: string) => {
    try {
      const response = await fetch(`${import.meta.env.VITE_API_BASE_URL}/api/v1/system/alerts/rules/${ruleId}`, {
        method: 'DELETE'
      });

      if (response.ok) {
        notification.success({
          message: 'Alert Rule Deleted',
          description: 'Rule has been deleted successfully',
          duration: 2
        });
        await fetchAlertRules();
      }
    } catch (err) {
      notification.error({
        message: 'Failed to Delete Rule',
        description: 'Could not delete the alert rule',
        duration: 4
      });
    }
  };

  const handleAcknowledgeAlert = async (alertId: string) => {
    try {
      const response = await fetch(`${import.meta.env.VITE_API_BASE_URL}/api/v1/system/alerts/${alertId}/acknowledge`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ acknowledged_by: 'user' })
      });

      if (response.ok) {
        notification.success({
          message: 'Alert Acknowledged',
          description: 'Alert has been acknowledged',
          duration: 2
        });
        await fetchActiveAlerts();
      }
    } catch (err) {
      notification.error({
        message: 'Failed to Acknowledge Alert',
        description: 'Could not acknowledge the alert',
        duration: 4
      });
    }
  };

  const getSeverityColor = (severity: string) => {
    switch (severity) {
      case 'critical': return '#ff4d4f';
      case 'high': return '#fa8c16';
      case 'medium': return '#faad14';
      case 'low': return '#52c41a';
      default: return '#d9d9d9';
    }
  };

  const getCategoryIcon = (category: string) => {
    switch (category) {
      case 'websocket': return <ThunderboltOutlined />;
      case 'trading': return <BellOutlined />;
      case 'risk': return <WarningOutlined />;
      case 'system': return <SettingOutlined />;
      case 'database': return <CheckCircleOutlined />;
      case 'strategy': return <RobotOutlined />;
      default: return <AlertOutlined />;
    }
  };

  const ruleColumns: ColumnsType<AlertRule> = [
    {
      title: 'Rule Name',
      dataIndex: 'name',
      key: 'name',
      render: (text, record) => (
        <Space>
          {getCategoryIcon(record.category)}
          <span>{text}</span>
          {record.ml_prediction.enabled && (
            <Tooltip title="ML Prediction Enabled">
              <RobotOutlined style={{ color: '#722ed1' }} />
            </Tooltip>
          )}
        </Space>
      )
    },
    {
      title: 'Category',
      dataIndex: 'category',
      key: 'category',
      render: (category) => (
        <Tag color="blue">{category.toUpperCase()}</Tag>
      ),
      filters: [
        { text: 'WebSocket', value: 'websocket' },
        { text: 'Trading', value: 'trading' },
        { text: 'Risk', value: 'risk' },
        { text: 'System', value: 'system' },
        { text: 'Database', value: 'database' },
        { text: 'Strategy', value: 'strategy' }
      ],
      onFilter: (value, record) => record.category === value
    },
    {
      title: 'Severity',
      dataIndex: 'severity',
      key: 'severity',
      render: (severity) => (
        <Tag color={getSeverityColor(severity)}>{severity.toUpperCase()}</Tag>
      )
    },
    {
      title: 'Threshold',
      dataIndex: 'threshold_value',
      key: 'threshold',
      render: (value, record) => (
        <span>{record.metric_name} {record.condition.replace('_', ' ')} {value}</span>
      )
    },
    {
      title: 'Status',
      dataIndex: 'enabled',
      key: 'enabled',
      render: (enabled, record) => (
        <Switch
          checked={enabled}
          onChange={(checked) => handleToggleRule(record.id, checked)}
          size="small"
        />
      )
    },
    {
      title: 'Triggers (24h)',
      dataIndex: 'trigger_count_24h',
      key: 'triggers',
      render: (count) => (
        <Badge count={count} overflowCount={99} style={{ backgroundColor: count > 0 ? '#fa541c' : '#52c41a' }} />
      )
    },
    {
      title: 'Avg Resolution',
      dataIndex: 'avg_resolution_time_minutes',
      key: 'resolution_time',
      render: (minutes) => `${minutes}m`
    },
    {
      title: 'Actions',
      key: 'actions',
      render: (_, record) => (
        <Space size="small">
          <Tooltip title="Edit Rule">
            <Button
              size="small"
              icon={<EditOutlined />}
              onClick={() => setEditingRule(record)}
            />
          </Tooltip>
          <Popconfirm
            title="Delete this rule?"
            onConfirm={() => handleDeleteRule(record.id)}
          >
            <Button size="small" danger icon={<DeleteOutlined />} />
          </Popconfirm>
        </Space>
      )
    }
  ];

  const renderRulesTab = () => (
    <div>
      <Row justify="space-between" style={{ marginBottom: 16 }}>
        <Col>
          <Space>
            <Select
              value={selectedCategory}
              onChange={setSelectedCategory}
              style={{ width: 150 }}
              size="small"
            >
              <Option value="all">All Categories</Option>
              <Option value="websocket">WebSocket</Option>
              <Option value="trading">Trading</Option>
              <Option value="risk">Risk</Option>
              <Option value="system">System</Option>
              <Option value="database">Database</Option>
              <Option value="strategy">Strategy</Option>
            </Select>
          </Space>
        </Col>
        <Col>
          <Button
            type="primary"
            icon={<PlusOutlined />}
            onClick={() => setCreateModalVisible(true)}
          >
            Create Rule
          </Button>
        </Col>
      </Row>

      <Table
        columns={ruleColumns}
        dataSource={selectedCategory === 'all' ? alertRules : alertRules.filter(rule => rule.category === selectedCategory)}
        rowKey="id"
        size="small"
        loading={loading}
        pagination={{ pageSize: 10 }}
      />
    </div>
  );

  const renderActiveAlertsTab = () => {
    const criticalAlerts = activeAlerts.filter(alert => alert.severity === 'critical');
    const highAlerts = activeAlerts.filter(alert => alert.severity === 'high');

    return (
      <div>
        {criticalAlerts.length > 0 && (
          <Alert
            message={`${criticalAlerts.length} Critical Alerts Require Immediate Attention`}
            type="error"
            style={{ marginBottom: 16 }}
            action={
              <Button size="small" danger>
                Acknowledge All Critical
              </Button>
            }
          />
        )}

        <Row gutter={[16, 16]} style={{ marginBottom: 16 }}>
          <Col xs={24} sm={6}>
            <Card size="small">
              <div style={{ textAlign: 'center' }}>
                <div style={{ fontSize: '24px', color: '#ff4d4f' }}>{criticalAlerts.length}</div>
                <Text type="secondary">Critical</Text>
              </div>
            </Card>
          </Col>
          <Col xs={24} sm={6}>
            <Card size="small">
              <div style={{ textAlign: 'center' }}>
                <div style={{ fontSize: '24px', color: '#fa8c16' }}>{highAlerts.length}</div>
                <Text type="secondary">High</Text>
              </div>
            </Card>
          </Col>
          <Col xs={24} sm={6}>
            <Card size="small">
              <div style={{ textAlign: 'center' }}>
                <div style={{ fontSize: '24px', color: '#1890ff' }}>
                  {activeAlerts.filter(a => a.predicted_breach).length}
                </div>
                <Text type="secondary">ML Predicted</Text>
              </div>
            </Card>
          </Col>
          <Col xs={24} sm={6}>
            <Card size="small">
              <div style={{ textAlign: 'center' }}>
                <div style={{ fontSize: '24px', color: '#52c41a' }}>
                  {activeAlerts.filter(a => a.acknowledged).length}
                </div>
                <Text type="secondary">Acknowledged</Text>
              </div>
            </Card>
          </Col>
        </Row>

        <List
          dataSource={activeAlerts}
          renderItem={(alert) => (
            <List.Item
              key={alert.id}
              actions={[
                !alert.acknowledged && (
                  <Button
                    size="small"
                    type="primary"
                    onClick={() => handleAcknowledgeAlert(alert.id)}
                  >
                    Acknowledge
                  </Button>
                ),
                <Button
                  size="small"
                  onClick={() => setAlertDetailDrawer(alert.id)}
                >
                  Details
                </Button>
              ].filter(Boolean)}
            >
              <List.Item.Meta
                avatar={
                  <div style={{ fontSize: '20px', color: getSeverityColor(alert.severity) }}>
                    {getCategoryIcon(alert.category)}
                  </div>
                }
                title={
                  <Space>
                    <span>{alert.rule_name}</span>
                    <Tag color={getSeverityColor(alert.severity)}>{alert.severity.toUpperCase()}</Tag>
                    {alert.predicted_breach && (
                      <Tooltip title={`ML Prediction: ${(alert.predicted_breach.confidence * 100).toFixed(0)}% confidence`}>
                        <Tag color="purple">
                          <RobotOutlined /> PREDICTED
                        </Tag>
                      </Tooltip>
                    )}
                    {alert.acknowledged && <Tag color="green">ACKNOWLEDGED</Tag>}
                  </Space>
                }
                description={
                  <div>
                    <Text>{alert.metric_name}: {alert.current_value} (threshold: {alert.threshold_value})</Text>
                    <br />
                    <Text type="secondary">
                      Triggered: {new Date(alert.triggered_at).toLocaleString()} | 
                      Escalation Level: {alert.escalation_level} | 
                      Notifications: {alert.notifications_sent}
                    </Text>
                  </div>
                }
              />
            </List.Item>
          )}
          pagination={{ pageSize: 5 }}
        />
      </div>
    );
  };

  const renderCreateRuleModal = () => (
    <Modal
      title="Create Alert Rule"
      open={createModalVisible}
      onCancel={() => setCreateModalVisible(false)}
      footer={null}
      width={800}
    >
      <Form
        form={form}
        layout="vertical"
        onFinish={handleCreateRule}
      >
        <Row gutter={16}>
          <Col span={12}>
            <Form.Item name="name" label="Rule Name" rules={[{ required: true }]}>
              <Input placeholder="Enter rule name" />
            </Form.Item>
          </Col>
          <Col span={12}>
            <Form.Item name="category" label="Category" rules={[{ required: true }]}>
              <Select placeholder="Select category">
                <Option value="websocket">WebSocket</Option>
                <Option value="trading">Trading</Option>
                <Option value="risk">Risk</Option>
                <Option value="system">System</Option>
                <Option value="database">Database</Option>
                <Option value="strategy">Strategy</Option>
              </Select>
            </Form.Item>
          </Col>
        </Row>

        <Row gutter={16}>
          <Col span={8}>
            <Form.Item name="metric_name" label="Metric Name" rules={[{ required: true }]}>
              <Input placeholder="e.g., cpu_usage_percent" />
            </Form.Item>
          </Col>
          <Col span={8}>
            <Form.Item name="condition" label="Condition" rules={[{ required: true }]}>
              <Select placeholder="Select condition">
                <Option value="greater_than">Greater Than</Option>
                <Option value="less_than">Less Than</Option>
                <Option value="equals">Equals</Option>
                <Option value="not_equals">Not Equals</Option>
                <Option value="rate_increase">Rate Increase</Option>
                <Option value="rate_decrease">Rate Decrease</Option>
              </Select>
            </Form.Item>
          </Col>
          <Col span={8}>
            <Form.Item name="threshold_value" label="Threshold Value" rules={[{ required: true }]}>
              <InputNumber style={{ width: '100%' }} placeholder="Enter threshold" />
            </Form.Item>
          </Col>
        </Row>

        <Row gutter={16}>
          <Col span={12}>
            <Form.Item name="severity" label="Severity" rules={[{ required: true }]}>
              <Select placeholder="Select severity">
                <Option value="critical">Critical</Option>
                <Option value="high">High</Option>
                <Option value="medium">Medium</Option>
                <Option value="low">Low</Option>
              </Select>
            </Form.Item>
          </Col>
          <Col span={12}>
            <Form.Item name="enabled" label="Enabled" valuePropName="checked">
              <Switch defaultChecked />
            </Form.Item>
          </Col>
        </Row>

        <Divider>Notifications</Divider>

        <Form.Item name="email_recipients" label="Email Recipients">
          <Input placeholder="user1@example.com, user2@example.com" />
        </Form.Item>

        <Row gutter={16}>
          <Col span={12}>
            <Form.Item name="slack_webhook" label="Slack Webhook URL">
              <Input placeholder="https://hooks.slack.com/..." />
            </Form.Item>
          </Col>
          <Col span={12}>
            <Form.Item name="sms_numbers" label="SMS Numbers">
              <Input placeholder="+1234567890, +0987654321" />
            </Form.Item>
          </Col>
        </Row>

        <Divider>Advanced Settings</Divider>

        <Row gutter={16}>
          <Col span={8}>
            <Form.Item name="escalate_after_minutes" label="Escalate After (minutes)">
              <InputNumber style={{ width: '100%' }} defaultValue={30} min={1} />
            </Form.Item>
          </Col>
          <Col span={8}>
            <Form.Item name="auto_resolution_enabled" label="Auto Resolution" valuePropName="checked">
              <Switch />
            </Form.Item>
          </Col>
          <Col span={8}>
            <Form.Item name="ml_prediction_enabled" label="ML Prediction" valuePropName="checked">
              <Switch />
            </Form.Item>
          </Col>
        </Row>

        <Form.Item>
          <Space>
            <Button type="primary" htmlType="submit">
              Create Rule
            </Button>
            <Button onClick={() => setCreateModalVisible(false)}>
              Cancel
            </Button>
          </Space>
        </Form.Item>
      </Form>
    </Modal>
  );

  return (
    <div className={`alert-rules-manager ${className || ''}`}>
      {/* Header */}
      <div style={{ marginBottom: 24 }}>
        <Title level={3} style={{ margin: 0 }}>
          <AlertOutlined style={{ marginRight: 8, color: '#fa541c' }} />
          Alert Rules Manager
        </Title>
        <Text type="secondary">
          Configure and manage 30+ alerting rules across 6 categories
        </Text>
      </div>

      {/* Error Display */}
      {error && (
        <Alert
          message="Alert Rules Error"
          description={error}
          type="error"
          style={{ marginBottom: 16 }}
          action={
            <Button size="small" onClick={fetchData}>
              Retry
            </Button>
          }
        />
      )}

      {/* Main Content */}
      <Tabs activeKey={activeTab} onChange={setActiveTab}>
        <TabPane 
          tab={
            <span>
              <SettingOutlined />
              Alert Rules
              {alertRules.length > 0 && (
                <Badge 
                  count={alertRules.filter(rule => rule.enabled).length} 
                  offset={[8, -4]} 
                  style={{ backgroundColor: '#52c41a' }}
                />
              )}
            </span>
          } 
          key="rules"
        >
          {renderRulesTab()}
        </TabPane>

        <TabPane 
          tab={
            <span>
              <BellOutlined />
              Active Alerts
              {activeAlerts.length > 0 && (
                <Badge 
                  count={activeAlerts.filter(alert => !alert.acknowledged).length} 
                  offset={[8, -4]} 
                />
              )}
            </span>
          } 
          key="active"
        >
          {renderActiveAlertsTab()}
        </TabPane>

        <TabPane 
          tab={
            <span>
              <RobotOutlined />
              ML Predictions
            </span>
          } 
          key="predictions"
        >
          <div style={{ textAlign: 'center', padding: 60 }}>
            <Text type="secondary">
              ML-based breach predictions and intelligent alerting will be displayed here
            </Text>
          </div>
        </TabPane>

        <TabPane 
          tab={
            <span>
              <MailOutlined />
              Notification History
            </span>
          } 
          key="notifications"
        >
          <div style={{ textAlign: 'center', padding: 60 }}>
            <Text type="secondary">
              Alert notification history and delivery status will be displayed here
            </Text>
          </div>
        </TabPane>
      </Tabs>

      {/* Create Rule Modal */}
      {renderCreateRuleModal()}

      {/* Loading State */}
      {loading && (
        <div style={{ textAlign: 'center', padding: 40 }}>
          <Spin size="large" />
        </div>
      )}
    </div>
  );
};

export default AlertRulesManager;