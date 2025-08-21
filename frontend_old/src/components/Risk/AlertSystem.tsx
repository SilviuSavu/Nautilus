import React, { useState, useEffect } from 'react';
import { 
  Card, 
  Alert, 
  Table, 
  Button, 
  Modal, 
  Form, 
  Input, 
  Select, 
  InputNumber, 
  Switch, 
  Tag, 
  Space, 
  Tooltip,
  notification,
  Popconfirm
} from 'antd';
import { 
  BellOutlined, 
  ExclamationCircleOutlined, 
  WarningOutlined, 
  InfoCircleOutlined,
  PlusOutlined,
  EditOutlined,
  DeleteOutlined
} from '@ant-design/icons';
import { RiskAlert, RiskLimit } from './types/riskTypes';
import { riskService } from './services/riskService';

const { Option } = Select;

interface AlertSystemProps {
  portfolioId: string;
  refreshInterval?: number;
}

const AlertSystem: React.FC<AlertSystemProps> = ({ 
  portfolioId, 
  refreshInterval = 10000 
}) => {
  const [alerts, setAlerts] = useState<RiskAlert[]>([]);
  const [limits, setLimits] = useState<RiskLimit[]>([]);
  const [loading, setLoading] = useState(true);
  const [modalVisible, setModalVisible] = useState(false);
  const [editingLimit, setEditingLimit] = useState<RiskLimit | null>(null);
  const [form] = Form.useForm();

  const fetchData = async () => {
    try {
      const [alertsData, limitsData] = await Promise.all([
        riskService.getRiskAlerts(portfolioId),
        riskService.getRiskLimits(portfolioId)
      ]);
      
      setAlerts(alertsData.alerts || []);
      setLimits(limitsData.limits || []);
    } catch (error) {
      notification.error({
        message: 'Error Loading Data',
        description: 'Failed to fetch alerts and limits'
      });
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchData();
    
    if (refreshInterval > 0) {
      const interval = setInterval(fetchData, refreshInterval);
      return () => clearInterval(interval);
    }
  }, [portfolioId, refreshInterval]);

  const handleAcknowledgeAlert = async (alertId: string) => {
    try {
      await riskService.acknowledgeAlert(alertId, 'User');
      notification.success({
        message: 'Alert Acknowledged',
        description: 'Alert has been marked as acknowledged'
      });
      fetchData();
    } catch (error) {
      notification.error({
        message: 'Error',
        description: 'Failed to acknowledge alert'
      });
    }
  };

  const handleCreateLimit = () => {
    setEditingLimit(null);
    setModalVisible(true);
    form.resetFields();
  };

  const handleEditLimit = (limit: RiskLimit) => {
    setEditingLimit(limit);
    setModalVisible(true);
    form.setFieldsValue({
      name: limit.name,
      limit_type: limit.limit_type,
      threshold_value: parseFloat(limit.threshold_value),
      warning_threshold: parseFloat(limit.warning_threshold),
      action: limit.action,
      active: limit.active
    });
  };

  const handleDeleteLimit = async (limitId: string) => {
    try {
      await riskService.deleteRiskLimit(limitId);
      notification.success({
        message: 'Limit Deleted',
        description: 'Risk limit has been removed'
      });
      fetchData();
    } catch (error) {
      notification.error({
        message: 'Error',
        description: 'Failed to delete risk limit'
      });
    }
  };

  const handleSubmitLimit = async (values: any) => {
    try {
      const limitData = {
        ...values,
        portfolio_id: portfolioId,
        threshold_value: values.threshold_value.toString(),
        warning_threshold: values.warning_threshold.toString()
      };

      if (editingLimit) {
        await riskService.updateRiskLimit(editingLimit.id, limitData);
        notification.success({
          message: 'Limit Updated',
          description: 'Risk limit has been updated successfully'
        });
      } else {
        await riskService.createRiskLimit(limitData);
        notification.success({
          message: 'Limit Created',
          description: 'New risk limit has been created'
        });
      }

      setModalVisible(false);
      fetchData();
    } catch (error) {
      notification.error({
        message: 'Error',
        description: 'Failed to save risk limit'
      });
    }
  };

  const getSeverityIcon = (severity: string) => {
    switch (severity) {
      case 'critical':
        return <ExclamationCircleOutlined style={{ color: '#ff4d4f' }} />;
      case 'warning':
        return <WarningOutlined style={{ color: '#faad14' }} />;
      default:
        return <InfoCircleOutlined style={{ color: '#1890ff' }} />;
    }
  };

  const getSeverityColor = (severity: string) => {
    switch (severity) {
      case 'critical':
        return '#ff4d4f';
      case 'warning':
        return '#faad14';
      default:
        return '#1890ff';
    }
  };

  const alertColumns = [
    {
      title: 'Severity',
      dataIndex: 'severity',
      key: 'severity',
      width: 100,
      render: (severity: string) => (
        <Tag color={getSeverityColor(severity)} icon={getSeverityIcon(severity)}>
          {severity.toUpperCase()}
        </Tag>
      )
    },
    {
      title: 'Type',
      dataIndex: 'alert_type',
      key: 'alert_type',
      width: 150,
      render: (type: string) => type.replace('_', ' ').toUpperCase()
    },
    {
      title: 'Message',
      dataIndex: 'message',
      key: 'message'
    },
    {
      title: 'Triggered',
      dataIndex: 'triggered_at',
      key: 'triggered_at',
      width: 150,
      render: (date: string) => new Date(date).toLocaleString()
    },
    {
      title: 'Status',
      dataIndex: 'acknowledged',
      key: 'acknowledged',
      width: 100,
      render: (acknowledged: boolean) => (
        <Tag color={acknowledged ? 'green' : 'orange'}>
          {acknowledged ? 'Acknowledged' : 'Active'}
        </Tag>
      )
    },
    {
      title: 'Actions',
      key: 'actions',
      width: 120,
      render: (record: RiskAlert) => (
        <Space>
          {!record.acknowledged && (
            <Button
              type="link"
              size="small"
              onClick={() => handleAcknowledgeAlert(record.id)}
            >
              Acknowledge
            </Button>
          )}
        </Space>
      )
    }
  ];

  const limitColumns = [
    {
      title: 'Name',
      dataIndex: 'name',
      key: 'name'
    },
    {
      title: 'Type',
      dataIndex: 'limit_type',
      key: 'limit_type',
      render: (type: string) => type.replace('_', ' ').toUpperCase()
    },
    {
      title: 'Warning Threshold',
      dataIndex: 'warning_threshold',
      key: 'warning_threshold',
      render: (value: string) => parseFloat(value).toLocaleString()
    },
    {
      title: 'Critical Threshold',
      dataIndex: 'threshold_value',
      key: 'threshold_value',
      render: (value: string) => parseFloat(value).toLocaleString()
    },
    {
      title: 'Action',
      dataIndex: 'action',
      key: 'action',
      render: (action: string) => action.toUpperCase()
    },
    {
      title: 'Status',
      dataIndex: 'active',
      key: 'active',
      render: (active: boolean) => (
        <Tag color={active ? 'green' : 'red'}>
          {active ? 'Active' : 'Disabled'}
        </Tag>
      )
    },
    {
      title: 'Breaches',
      dataIndex: 'breach_count',
      key: 'breach_count'
    },
    {
      title: 'Actions',
      key: 'actions',
      render: (record: RiskLimit) => (
        <Space>
          <Tooltip title="Edit Limit">
            <Button
              type="link"
              icon={<EditOutlined />}
              onClick={() => handleEditLimit(record)}
            />
          </Tooltip>
          <Popconfirm
            title="Are you sure you want to delete this limit?"
            onConfirm={() => handleDeleteLimit(record.id)}
            okText="Yes"
            cancelText="No"
          >
            <Tooltip title="Delete Limit">
              <Button
                type="link"
                danger
                icon={<DeleteOutlined />}
              />
            </Tooltip>
          </Popconfirm>
        </Space>
      )
    }
  ];

  const activeAlertsCount = alerts.filter(alert => !alert.acknowledged).length;
  const criticalAlertsCount = alerts.filter(alert => 
    alert.severity === 'critical' && !alert.acknowledged
  ).length;

  return (
    <div>
      {/* Alert Summary */}
      {activeAlertsCount > 0 && (
        <Alert
          message={`${activeAlertsCount} Active Alert${activeAlertsCount > 1 ? 's' : ''}`}
          description={
            criticalAlertsCount > 0 
              ? `${criticalAlertsCount} critical alert${criticalAlertsCount > 1 ? 's' : ''} requiring immediate attention`
              : 'Review and acknowledge alerts below'
          }
          type={criticalAlertsCount > 0 ? 'error' : 'warning'}
          showIcon
          style={{ marginBottom: 16 }}
        />
      )}

      {/* Active Alerts */}
      <Card 
        title={
          <span>
            <BellOutlined /> Active Alerts
            {activeAlertsCount > 0 && (
              <Tag color="red" style={{ marginLeft: 8 }}>
                {activeAlertsCount}
              </Tag>
            )}
          </span>
        }
        style={{ marginBottom: 16 }}
      >
        <Table
          columns={alertColumns}
          dataSource={alerts.map(alert => ({ ...alert, key: alert.id }))}
          loading={loading}
          pagination={false}
          size="small"
          locale={{ emptyText: 'No active alerts' }}
        />
      </Card>

      {/* Risk Limits */}
      <Card 
        title="Risk Limits"
        extra={
          <Button
            type="primary"
            icon={<PlusOutlined />}
            onClick={handleCreateLimit}
          >
            Add Limit
          </Button>
        }
      >
        <Table
          columns={limitColumns}
          dataSource={limits.map(limit => ({ ...limit, key: limit.id }))}
          loading={loading}
          pagination={{ pageSize: 10 }}
          size="small"
        />
      </Card>

      {/* Limit Configuration Modal */}
      <Modal
        title={editingLimit ? 'Edit Risk Limit' : 'Create Risk Limit'}
        open={modalVisible}
        onCancel={() => setModalVisible(false)}
        footer={null}
        width={600}
      >
        <Form
          form={form}
          layout="vertical"
          onFinish={handleSubmitLimit}
        >
          <Form.Item
            name="name"
            label="Limit Name"
            rules={[{ required: true, message: 'Please enter a limit name' }]}
          >
            <Input placeholder="e.g., Daily VaR Limit" />
          </Form.Item>

          <Form.Item
            name="limit_type"
            label="Limit Type"
            rules={[{ required: true, message: 'Please select a limit type' }]}
          >
            <Select placeholder="Select limit type">
              <Option value="var">Value at Risk</Option>
              <Option value="concentration">Position Concentration</Option>
              <Option value="position_size">Position Size</Option>
              <Option value="leverage">Portfolio Leverage</Option>
              <Option value="correlation">Asset Correlation</Option>
            </Select>
          </Form.Item>

          <Form.Item
            name="warning_threshold"
            label="Warning Threshold"
            rules={[{ required: true, message: 'Please enter warning threshold' }]}
          >
            <InputNumber
              style={{ width: '100%' }}
              placeholder="Warning level"
              min={0}
            />
          </Form.Item>

          <Form.Item
            name="threshold_value"
            label="Critical Threshold"
            rules={[{ required: true, message: 'Please enter critical threshold' }]}
          >
            <InputNumber
              style={{ width: '100%' }}
              placeholder="Critical level"
              min={0}
            />
          </Form.Item>

          <Form.Item
            name="action"
            label="Action"
            rules={[{ required: true, message: 'Please select an action' }]}
          >
            <Select placeholder="Select action on breach">
              <Option value="warn">Send Warning</Option>
              <Option value="block">Block New Trades</Option>
              <Option value="reduce">Auto-Reduce Position</Option>
              <Option value="notify">Notify Manager</Option>
            </Select>
          </Form.Item>

          <Form.Item
            name="active"
            label="Active"
            valuePropName="checked"
            initialValue={true}
          >
            <Switch />
          </Form.Item>

          <Form.Item style={{ marginBottom: 0, textAlign: 'right' }}>
            <Space>
              <Button onClick={() => setModalVisible(false)}>
                Cancel
              </Button>
              <Button type="primary" htmlType="submit">
                {editingLimit ? 'Update' : 'Create'}
              </Button>
            </Space>
          </Form.Item>
        </Form>
      </Modal>
    </div>
  );
};

export default AlertSystem;