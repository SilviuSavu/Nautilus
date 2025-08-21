import React, { useState, useEffect } from 'react';
import {
  Card,
  Row,
  Col,
  Typography,
  Form,
  Input,
  InputNumber,
  Select,
  Button,
  Switch,
  Table,
  Tag,
  Space,
  Modal,
  notification,
  Alert,
  Divider,
  Tooltip
} from 'antd';
import {
  BellOutlined,
  PlusOutlined,
  EditOutlined,
  DeleteOutlined,
  WarningOutlined,
  CheckCircleOutlined,
  ExclamationCircleOutlined,
  MailOutlined,
  MessageOutlined,
  AlertOutlined
} from '@ant-design/icons';
import type { ColumnsType } from 'antd/es/table';

import { StrategyInstance } from '../Strategy/types/strategyTypes';

const { Title, Text } = Typography;
const { Option } = Select;

interface AlertSystemProps {
  strategies: StrategyInstance[];
  performanceData: any;
  className?: string;
}

interface PerformanceAlert {
  id: string;
  name: string;
  strategy_id: string;
  alert_type: 'pnl_threshold' | 'drawdown_limit' | 'win_rate_drop' | 'trade_count' | 'volatility' | 'sharpe_ratio';
  condition: 'above' | 'below' | 'equals';
  threshold_value: number;
  current_value?: number;
  is_active: boolean;
  notification_methods: ('email' | 'sms' | 'dashboard')[];
  email_addresses?: string[];
  phone_numbers?: string[];
  last_triggered?: Date;
  trigger_count: number;
  created_at: Date;
  updated_at: Date;
}

interface AlertTrigger {
  id: string;
  alert_id: string;
  strategy_id: string;
  triggered_at: Date;
  value_at_trigger: number;
  threshold_value: number;
  alert_type: string;
  resolved: boolean;
  resolved_at?: Date;
  notification_sent: boolean;
  acknowledgement_required: boolean;
  acknowledged_by?: string;
  acknowledged_at?: Date;
}

export const AlertSystem: React.FC<AlertSystemProps> = ({
  strategies,
  performanceData,
  className
}) => {
  const [alerts, setAlerts] = useState<PerformanceAlert[]>([]);
  const [triggers, setTriggers] = useState<AlertTrigger[]>([]);
  const [loading, setLoading] = useState(false);
  const [modalVisible, setModalVisible] = useState(false);
  const [editingAlert, setEditingAlert] = useState<PerformanceAlert | null>(null);
  const [form] = Form.useForm();

  useEffect(() => {
    loadAlerts();
    loadTriggers();
    
    // Poll for active alerts every 30 seconds
    const interval = setInterval(() => {
      checkActiveAlerts();
    }, 30000);
    
    return () => clearInterval(interval);
  }, [strategies]);

  const loadAlerts = async () => {
    try {
      setLoading(true);
      const response = await fetch('/api/v1/performance/alerts');
      if (response.ok) {
        const data = await response.json();
        setAlerts(data.alerts || []);
      }
    } catch (error: any) {
      console.error('Failed to load alerts:', error);
    } finally {
      setLoading(false);
    }
  };

  const loadTriggers = async () => {
    try {
      const response = await fetch('/api/v1/performance/alerts/triggers?limit=50');
      if (response.ok) {
        const data = await response.json();
        setTriggers(data.triggers || []);
      }
    } catch (error: any) {
      console.error('Failed to load alert triggers:', error);
    }
  };

  const checkActiveAlerts = async () => {
    try {
      const response = await fetch('/api/v1/performance/alerts/check');
      if (response.ok) {
        const data = await response.json();
        if (data.triggered_alerts && data.triggered_alerts.length > 0) {
          // Show notifications for newly triggered alerts
          data.triggered_alerts.forEach((trigger: AlertTrigger) => {
            notification.warning({
              message: 'Performance Alert Triggered',
              description: `${trigger.alert_type} threshold exceeded for strategy ${trigger.strategy_id}`,
              duration: 10,
              icon: <WarningOutlined style={{ color: '#fa8c16' }} />
            });
          });
          loadTriggers(); // Refresh triggers list
        }
      }
    } catch (error: any) {
      console.error('Failed to check alerts:', error);
    }
  };

  const handleCreateAlert = async (values: any) => {
    try {
      const alertData = {
        name: values.name,
        strategy_id: values.strategy_id,
        alert_type: values.alert_type,
        condition: values.condition,
        threshold_value: values.threshold_value,
        is_active: values.is_active,
        notification_methods: values.notification_methods || ['dashboard'],
        email_addresses: values.email_addresses?.split(',').map((e: string) => e.trim()) || [],
        phone_numbers: values.phone_numbers?.split(',').map((p: string) => p.trim()) || []
      };

      const response = await fetch('/api/v1/performance/alerts', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(alertData)
      });

      if (response.ok) {
        notification.success({
          message: 'Alert Created',
          description: 'Performance alert has been created successfully',
          duration: 3
        });
        loadAlerts();
        setModalVisible(false);
        form.resetFields();
      } else {
        throw new Error('Failed to create alert');
      }
    } catch (error: any) {
      notification.error({
        message: 'Create Alert Failed',
        description: error.message || 'Failed to create performance alert',
        duration: 4
      });
    }
  };

  const handleUpdateAlert = async (alertId: string, values: any) => {
    try {
      const response = await fetch(`/api/v1/performance/alerts/${alertId}`, {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(values)
      });

      if (response.ok) {
        notification.success({
          message: 'Alert Updated',
          description: 'Performance alert has been updated successfully',
          duration: 3
        });
        loadAlerts();
        setModalVisible(false);
        setEditingAlert(null);
        form.resetFields();
      } else {
        throw new Error('Failed to update alert');
      }
    } catch (error: any) {
      notification.error({
        message: 'Update Alert Failed',
        description: error.message || 'Failed to update performance alert',
        duration: 4
      });
    }
  };

  const handleDeleteAlert = (alertId: string) => {
    Modal.confirm({
      title: 'Delete Alert',
      content: 'Are you sure you want to delete this alert? This action cannot be undone.',
      icon: <ExclamationCircleOutlined />,
      okText: 'Delete',
      okType: 'danger',
      onOk: async () => {
        try {
          const response = await fetch(`/api/v1/performance/alerts/${alertId}`, {
            method: 'DELETE'
          });

          if (response.ok) {
            notification.success({
              message: 'Alert Deleted',
              description: 'Performance alert has been deleted successfully',
              duration: 3
            });
            loadAlerts();
          } else {
            throw new Error('Failed to delete alert');
          }
        } catch (error: any) {
          notification.error({
            message: 'Delete Alert Failed',
            description: error.message || 'Failed to delete performance alert',
            duration: 4
          });
        }
      }
    });
  };

  const handleToggleAlert = async (alertId: string, isActive: boolean) => {
    try {
      const response = await fetch(`/api/v1/performance/alerts/${alertId}/toggle`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ is_active: isActive })
      });

      if (response.ok) {
        loadAlerts();
      }
    } catch (error: any) {
      notification.error({
        message: 'Toggle Alert Failed',
        description: 'Failed to update alert status',
        duration: 4
      });
    }
  };

  const acknowledgeAlert = async (triggerId: string) => {
    try {
      const response = await fetch(`/api/v1/performance/alerts/triggers/${triggerId}/acknowledge`, {
        method: 'POST'
      });

      if (response.ok) {
        loadTriggers();
        notification.success({
          message: 'Alert Acknowledged',
          description: 'Alert has been acknowledged successfully',
          duration: 3
        });
      }
    } catch (error: any) {
      notification.error({
        message: 'Acknowledge Failed',
        description: 'Failed to acknowledge alert',
        duration: 4
      });
    }
  };

  const getAlertTypeColor = (type: string): string => {
    switch (type) {
      case 'pnl_threshold': return 'blue';
      case 'drawdown_limit': return 'red';
      case 'win_rate_drop': return 'orange';
      case 'trade_count': return 'green';
      case 'volatility': return 'purple';
      case 'sharpe_ratio': return 'cyan';
      default: return 'default';
    }
  };

  const getAlertTypeIcon = (type: string) => {
    switch (type) {
      case 'pnl_threshold': return '💰';
      case 'drawdown_limit': return '📉';
      case 'win_rate_drop': return '🎯';
      case 'trade_count': return '📊';
      case 'volatility': return '📊';
      case 'sharpe_ratio': return '📈';
      default: return '⚠️';
    }
  };

  const alertColumns: ColumnsType<PerformanceAlert> = [
    {
      title: 'Alert Name',
      dataIndex: 'name',
      key: 'name',
      render: (name: string, record: PerformanceAlert) => (
        <div>
          <Text strong>{name}</Text>
          <br />
          <Text type="secondary" style={{ fontSize: 12 }}>
            {record.strategy_id === 'all' ? 'All Strategies' : record.strategy_id}
          </Text>
        </div>
      )
    },
    {
      title: 'Type',
      dataIndex: 'alert_type',
      key: 'alert_type',
      render: (type: string) => (
        <Tag color={getAlertTypeColor(type)}>
          {getAlertTypeIcon(type)} {type.replace('_', ' ').toUpperCase()}
        </Tag>
      )
    },
    {
      title: 'Condition',
      key: 'condition',
      render: (_, record: PerformanceAlert) => (
        <Text>
          {record.condition} {record.threshold_value}
          {record.alert_type.includes('rate') || record.alert_type.includes('drawdown') ? '%' : ''}
        </Text>
      )
    },
    {
      title: 'Current Value',
      key: 'current_value',
      render: (_, record: PerformanceAlert) => {
        const currentValue = record.current_value || 0;
        const isTriggered = record.condition === 'above' ? 
          currentValue > record.threshold_value : 
          currentValue < record.threshold_value;
        
        return (
          <Text style={{ color: isTriggered ? '#cf1322' : '#3f8600' }}>
            {currentValue.toFixed(2)}
            {record.alert_type.includes('rate') || record.alert_type.includes('drawdown') ? '%' : ''}
          </Text>
        );
      }
    },
    {
      title: 'Status',
      dataIndex: 'is_active',
      key: 'is_active',
      render: (isActive: boolean, record: PerformanceAlert) => (
        <div>
          <Switch
            checked={isActive}
            onChange={(checked) => handleToggleAlert(record.id, checked)}
            size="small"
          />
          <br />
          <Text type="secondary" style={{ fontSize: 12 }}>
            Triggers: {record.trigger_count}
          </Text>
        </div>
      )
    },
    {
      title: 'Notifications',
      dataIndex: 'notification_methods',
      key: 'notification_methods',
      render: (methods: string[]) => (
        <Space>
          {methods.includes('email') && <MailOutlined style={{ color: '#1890ff' }} />}
          {methods.includes('sms') && <MessageOutlined style={{ color: '#52c41a' }} />}
          {methods.includes('dashboard') && <BellOutlined style={{ color: '#fa8c16' }} />}
        </Space>
      )
    },
    {
      title: 'Actions',
      key: 'actions',
      render: (_, record: PerformanceAlert) => (
        <Space>
          <Tooltip title="Edit Alert">
            <Button
              type="text"
              icon={<EditOutlined />}
              onClick={() => {
                setEditingAlert(record);
                form.setFieldsValue(record);
                setModalVisible(true);
              }}
            />
          </Tooltip>
          <Tooltip title="Delete Alert">
            <Button
              type="text"
              danger
              icon={<DeleteOutlined />}
              onClick={() => handleDeleteAlert(record.id)}
            />
          </Tooltip>
        </Space>
      )
    }
  ];

  const triggerColumns: ColumnsType<AlertTrigger> = [
    {
      title: 'Triggered At',
      dataIndex: 'triggered_at',
      key: 'triggered_at',
      render: (date: Date) => (
        <Text>{new Date(date).toLocaleString()}</Text>
      )
    },
    {
      title: 'Alert Type',
      dataIndex: 'alert_type',
      key: 'alert_type',
      render: (type: string) => (
        <Tag color={getAlertTypeColor(type)}>
          {getAlertTypeIcon(type)} {type.replace('_', ' ').toUpperCase()}
        </Tag>
      )
    },
    {
      title: 'Strategy',
      dataIndex: 'strategy_id',
      key: 'strategy_id'
    },
    {
      title: 'Value',
      key: 'value',
      render: (_, record: AlertTrigger) => (
        <div>
          <Text strong style={{ color: '#cf1322' }}>
            {record.value_at_trigger.toFixed(2)}
          </Text>
          <br />
          <Text type="secondary" style={{ fontSize: 12 }}>
            Threshold: {record.threshold_value.toFixed(2)}
          </Text>
        </div>
      )
    },
    {
      title: 'Status',
      key: 'status',
      render: (_, record: AlertTrigger) => (
        <div>
          {record.resolved ? (
            <Tag color="green">
              <CheckCircleOutlined /> Resolved
            </Tag>
          ) : (
            <Tag color="red">
              <WarningOutlined /> Active
            </Tag>
          )}
          {record.acknowledgement_required && !record.acknowledged_at && (
            <div style={{ marginTop: 4 }}>
              <Button
                size="small"
                type="primary"
                onClick={() => acknowledgeAlert(record.id)}
              >
                Acknowledge
              </Button>
            </div>
          )}
        </div>
      )
    }
  ];

  return (
    <div className={`alert-system ${className || ''}`}>
      <Row gutter={[16, 16]}>
        {/* Header */}
        <Col span={24}>
          <Card>
            <Row justify="space-between" align="middle">
              <Col>
                <Title level={4} style={{ margin: 0 }}>
                  <BellOutlined style={{ marginRight: 8 }} />
                  Performance Alert System
                </Title>
                <Text type="secondary">
                  Configure alerts for performance thresholds and risk limits
                </Text>
              </Col>
              <Col>
                <Button
                  type="primary"
                  icon={<PlusOutlined />}
                  onClick={() => {
                    setEditingAlert(null);
                    form.resetFields();
                    setModalVisible(true);
                  }}
                >
                  Create Alert
                </Button>
              </Col>
            </Row>
          </Card>
        </Col>

        {/* Active Alerts Summary */}
        <Col span={24}>
          <Alert
            message={`${alerts.filter(a => a.is_active).length} active alerts monitoring ${strategies.length} strategies`}
            type="info"
            showIcon
            icon={<AlertOutlined />}
          />
        </Col>

        {/* Alerts Table */}
        <Col span={24}>
          <Card title="Alert Configuration">
            <Table
              columns={alertColumns}
              dataSource={alerts}
              rowKey="id"
              loading={loading}
              pagination={{ pageSize: 10 }}
            />
          </Card>
        </Col>

        {/* Recent Triggers */}
        <Col span={24}>
          <Card title="Recent Alert Triggers">
            <Table
              columns={triggerColumns}
              dataSource={triggers}
              rowKey="id"
              pagination={{ pageSize: 10 }}
            />
          </Card>
        </Col>
      </Row>

      {/* Create/Edit Alert Modal */}
      <Modal
        title={editingAlert ? 'Edit Alert' : 'Create Performance Alert'}
        open={modalVisible}
        onCancel={() => {
          setModalVisible(false);
          setEditingAlert(null);
          form.resetFields();
        }}
        footer={null}
        width={600}
      >
        <Form
          form={form}
          layout="vertical"
          onFinish={(values) => {
            if (editingAlert) {
              handleUpdateAlert(editingAlert.id, values);
            } else {
              handleCreateAlert(values);
            }
          }}
        >
          <Form.Item
            name="name"
            label="Alert Name"
            rules={[{ required: true, message: 'Please enter alert name' }]}
          >
            <Input placeholder="e.g., High Drawdown Alert" />
          </Form.Item>

          <Row gutter={16}>
            <Col span={12}>
              <Form.Item
                name="strategy_id"
                label="Strategy"
                rules={[{ required: true, message: 'Please select strategy' }]}
              >
                <Select placeholder="Select strategy">
                  <Option value="all">All Strategies</Option>
                  {strategies.map(strategy => (
                    <Option key={strategy.id} value={strategy.id}>
                      {strategy.id}
                    </Option>
                  ))}
                </Select>
              </Form.Item>
            </Col>
            
            <Col span={12}>
              <Form.Item
                name="alert_type"
                label="Alert Type"
                rules={[{ required: true, message: 'Please select alert type' }]}
              >
                <Select placeholder="Select alert type">
                  <Option value="pnl_threshold">P&L Threshold</Option>
                  <Option value="drawdown_limit">Drawdown Limit</Option>
                  <Option value="win_rate_drop">Win Rate Drop</Option>
                  <Option value="trade_count">Trade Count</Option>
                  <Option value="volatility">Volatility</Option>
                  <Option value="sharpe_ratio">Sharpe Ratio</Option>
                </Select>
              </Form.Item>
            </Col>
          </Row>

          <Row gutter={16}>
            <Col span={12}>
              <Form.Item
                name="condition"
                label="Condition"
                rules={[{ required: true, message: 'Please select condition' }]}
              >
                <Select placeholder="Select condition">
                  <Option value="above">Above</Option>
                  <Option value="below">Below</Option>
                  <Option value="equals">Equals</Option>
                </Select>
              </Form.Item>
            </Col>
            
            <Col span={12}>
              <Form.Item
                name="threshold_value"
                label="Threshold Value"
                rules={[{ required: true, message: 'Please enter threshold value' }]}
              >
                <InputNumber 
                  placeholder="Enter threshold"
                  style={{ width: '100%' }}
                  precision={2}
                />
              </Form.Item>
            </Col>
          </Row>

          <Form.Item
            name="notification_methods"
            label="Notification Methods"
          >
            <Select mode="multiple" placeholder="Select notification methods">
              <Option value="dashboard">Dashboard</Option>
              <Option value="email">Email</Option>
              <Option value="sms">SMS</Option>
            </Select>
          </Form.Item>

          <Form.Item
            name="email_addresses"
            label="Email Addresses"
            help="Comma-separated email addresses"
          >
            <Input placeholder="email1@example.com, email2@example.com" />
          </Form.Item>

          <Form.Item
            name="phone_numbers"
            label="Phone Numbers"
            help="Comma-separated phone numbers"
          >
            <Input placeholder="+1234567890, +0987654321" />
          </Form.Item>

          <Form.Item name="is_active" valuePropName="checked" initialValue={true}>
            <Switch /> <Text>Alert Active</Text>
          </Form.Item>

          <Divider />

          <Form.Item style={{ marginBottom: 0 }}>
            <Space>
              <Button
                type="primary"
                htmlType="submit"
                loading={loading}
              >
                {editingAlert ? 'Update Alert' : 'Create Alert'}
              </Button>
              <Button
                onClick={() => {
                  setModalVisible(false);
                  setEditingAlert(null);
                  form.resetFields();
                }}
              >
                Cancel
              </Button>
            </Space>
          </Form.Item>
        </Form>
      </Modal>
    </div>
  );
};