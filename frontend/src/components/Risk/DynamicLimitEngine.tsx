import React, { useState, useCallback } from 'react';
import {
  Card,
  Table,
  Button,
  Switch,
  Progress,
  Tag,
  Space,
  Tooltip,
  Modal,
  Form,
  Input,
  Select,
  InputNumber,
  Alert,
  Popconfirm,
  Row,
  Col,
  Statistic,
  Drawer,
  Typography,
  Divider,
  Badge,
  Timeline
} from 'antd';
import {
  PlusOutlined,
  EditOutlined,
  DeleteOutlined,
  SettingOutlined,
  WarningOutlined,
  RiseOutlined,
  FallOutlined,
  StopOutlined,
  ThunderboltOutlined,
  BellOutlined,
  InfoCircleOutlined
} from '@ant-design/icons';
import { useDynamicLimits, CreateLimitParams, UpdateLimitParams } from '../../hooks/risk/useDynamicLimits';
import { RiskLimit, EscalationRule } from './types/riskTypes';

const { Title, Text } = Typography;
const { Option } = Select;

interface DynamicLimitEngineProps {
  portfolioId: string;
  className?: string;
}

const DynamicLimitEngine: React.FC<DynamicLimitEngineProps> = ({
  portfolioId,
  className
}) => {
  const {
    limits,
    configuration,
    loading,
    error,
    activeLimits,
    breachedLimits,
    warningLimits,
    limitsByType,
    riskScore,
    createLimit,
    updateLimit,
    deleteLimit,
    toggleLimit,
    configureNotifications,
    clearError,
    refresh,
    isDeleting
  } = useDynamicLimits({ portfolioId });

  const [showCreateModal, setShowCreateModal] = useState(false);
  const [showEditModal, setShowEditModal] = useState(false);
  const [showConfigDrawer, setShowConfigDrawer] = useState(false);
  const [editingLimit, setEditingLimit] = useState<RiskLimit | null>(null);
  const [createForm] = Form.useForm();
  const [editForm] = Form.useForm();
  const [configForm] = Form.useForm();

  const handleCreateLimit = useCallback(async (values: any) => {
    try {
      const params: CreateLimitParams = {
        name: values.name,
        limit_type: values.limit_type,
        threshold_value: values.threshold_value.toString(),
        warning_threshold: values.warning_threshold.toString(),
        action: values.action,
        active: values.active ?? true
      };

      await createLimit(params);
      setShowCreateModal(false);
      createForm.resetFields();
    } catch (error) {
      console.error('Error creating limit:', error);
    }
  }, [createLimit, createForm]);

  const handleEditLimit = useCallback(async (values: any) => {
    if (!editingLimit) return;

    try {
      const params: UpdateLimitParams = {
        threshold_value: values.threshold_value?.toString(),
        warning_threshold: values.warning_threshold?.toString(),
        action: values.action,
        active: values.active
      };

      await updateLimit(editingLimit.id, params);
      setShowEditModal(false);
      setEditingLimit(null);
      editForm.resetFields();
    } catch (error) {
      console.error('Error updating limit:', error);
    }
  }, [updateLimit, editingLimit, editForm]);

  const handleDeleteLimit = useCallback(async (limitId: string) => {
    try {
      await deleteLimit(limitId);
    } catch (error) {
      console.error('Error deleting limit:', error);
    }
  }, [deleteLimit]);

  const handleToggleLimit = useCallback(async (limitId: string, active: boolean) => {
    try {
      await toggleLimit(limitId, active);
    } catch (error) {
      console.error('Error toggling limit:', error);
    }
  }, [toggleLimit]);

  const openEditModal = useCallback((limit: RiskLimit) => {
    setEditingLimit(limit);
    editForm.setFieldsValue({
      threshold_value: parseFloat(limit.threshold_value),
      warning_threshold: parseFloat(limit.warning_threshold),
      action: limit.action,
      active: limit.active
    });
    setShowEditModal(true);
  }, [editForm]);

  const getLimitStatusColor = (limit: RiskLimit) => {
    if (!limit.active) return 'default';
    if (limit.breach_count > 0) return 'error';
    return 'success';
  };

  const getLimitStatusText = (limit: RiskLimit) => {
    if (!limit.active) return 'Inactive';
    if (limit.breach_count > 0) return 'Breached';
    return 'Normal';
  };

  const getUtilizationColor = (percentage: number) => {
    if (percentage >= 100) return '#ff4d4f';
    if (percentage >= 80) return '#faad14';
    if (percentage >= 60) return '#1890ff';
    return '#52c41a';
  };

  const columns = [
    {
      title: 'Limit Name',
      dataIndex: 'name',
      key: 'name',
      width: 200,
      render: (name: string, record: RiskLimit) => (
        <Space direction="vertical" size={0}>
          <Text strong>{name}</Text>
          <Text type="secondary" style={{ fontSize: '12px' }}>
            {record.limit_type.toUpperCase()}
          </Text>
        </Space>
      )
    },
    {
      title: 'Current / Threshold',
      key: 'utilization',
      width: 200,
      render: (record: RiskLimit) => {
        // Mock current value calculation
        const currentValue = parseFloat(record.threshold_value) * (0.3 + Math.random() * 0.6);
        const thresholdValue = parseFloat(record.threshold_value);
        const percentage = (currentValue / thresholdValue) * 100;

        return (
          <Space direction="vertical" size={4} style={{ width: '100%' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between' }}>
              <Text style={{ fontSize: '12px' }}>
                {currentValue.toLocaleString()} / {thresholdValue.toLocaleString()}
              </Text>
              <Text style={{ fontSize: '12px', color: getUtilizationColor(percentage) }}>
                {percentage.toFixed(1)}%
              </Text>
            </div>
            <Progress
              percent={percentage}
              size="small"
              strokeColor={getUtilizationColor(percentage)}
              showInfo={false}
            />
          </Space>
        );
      }
    },
    {
      title: 'Warning Level',
      dataIndex: 'warning_threshold',
      key: 'warning_threshold',
      width: 120,
      render: (value: string) => (
        <Text>{parseFloat(value).toLocaleString()}</Text>
      )
    },
    {
      title: 'Action',
      dataIndex: 'action',
      key: 'action',
      width: 100,
      render: (action: string) => {
        const actionConfig = {
          warn: { color: 'orange', icon: <WarningOutlined /> },
          block: { color: 'red', icon: <StopOutlined /> },
          reduce: { color: 'blue', icon: <FallOutlined /> },
          notify: { color: 'green', icon: <BellOutlined /> }
        };
        const config = actionConfig[action as keyof typeof actionConfig];
        
        return (
          <Tag color={config.color} icon={config.icon}>
            {action.toUpperCase()}
          </Tag>
        );
      }
    },
    {
      title: 'Status',
      key: 'status',
      width: 100,
      render: (record: RiskLimit) => (
        <Tag color={getLimitStatusColor(record)}>
          {getLimitStatusText(record)}
        </Tag>
      )
    },
    {
      title: 'Breach Count',
      dataIndex: 'breach_count',
      key: 'breach_count',
      width: 100,
      render: (count: number) => (
        <Badge count={count} style={{ backgroundColor: count > 0 ? '#ff4d4f' : '#52c41a' }} />
      )
    },
    {
      title: 'Active',
      key: 'active',
      width: 80,
      render: (record: RiskLimit) => (
        <Switch
          checked={record.active}
          onChange={(checked) => handleToggleLimit(record.id, checked)}
          loading={loading.updating}
        />
      )
    },
    {
      title: 'Actions',
      key: 'actions',
      width: 120,
      render: (record: RiskLimit) => (
        <Space>
          <Tooltip title="Edit Limit">
            <Button
              type="text"
              icon={<EditOutlined />}
              size="small"
              onClick={() => openEditModal(record)}
              loading={loading.updating}
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
                type="text"
                icon={<DeleteOutlined />}
                size="small"
                danger
                loading={isDeleting(record.id)}
              />
            </Tooltip>
          </Popconfirm>
        </Space>
      )
    }
  ];

  return (
    <div className={className}>
      {/* Header Statistics */}
      <Row gutter={16} style={{ marginBottom: 16 }}>
        <Col span={6}>
          <Card size="small">
            <Statistic
              title="Active Limits"
              value={activeLimits.length}
              prefix={<ThunderboltOutlined style={{ color: '#1890ff' }} />}
            />
          </Card>
        </Col>
        <Col span={6}>
          <Card size="small">
            <Statistic
              title="Breached Limits"
              value={breachedLimits.length}
              prefix={<WarningOutlined style={{ color: '#ff4d4f' }} />}
              valueStyle={{ color: breachedLimits.length > 0 ? '#ff4d4f' : undefined }}
            />
          </Card>
        </Col>
        <Col span={6}>
          <Card size="small">
            <Statistic
              title="Warning Limits"
              value={warningLimits.length}
              prefix={<InfoCircleOutlined style={{ color: '#faad14' }} />}
              valueStyle={{ color: warningLimits.length > 0 ? '#faad14' : undefined }}
            />
          </Card>
        </Col>
        <Col span={6}>
          <Card size="small">
            <Statistic
              title="Risk Score"
              value={riskScore}
              precision={1}
              suffix="%"
              prefix={<RiseOutlined style={{ color: riskScore > 50 ? '#ff4d4f' : '#52c41a' }} />}
              valueStyle={{ color: riskScore > 50 ? '#ff4d4f' : '#52c41a' }}
            />
          </Card>
        </Col>
      </Row>

      {/* Error Alert */}
      {error && (
        <Alert
          message="Risk Limit Engine Error"
          description={error}
          type="error"
          showIcon
          closable
          onClose={clearError}
          style={{ marginBottom: 16 }}
          action={
            <Button size="small" onClick={refresh}>
              Retry
            </Button>
          }
        />
      )}

      {/* Main Table */}
      <Card
        title={
          <Space>
            <Title level={4} style={{ margin: 0 }}>Dynamic Risk Limits</Title>
            <Badge count={breachedLimits.length} style={{ backgroundColor: '#ff4d4f' }} />
          </Space>
        }
        extra={
          <Space>
            <Button
              type="primary"
              icon={<PlusOutlined />}
              onClick={() => setShowCreateModal(true)}
            >
              Add Limit
            </Button>
            <Button
              icon={<SettingOutlined />}
              onClick={() => setShowConfigDrawer(true)}
            >
              Configure
            </Button>
          </Space>
        }
      >
        <Table
          dataSource={limits}
          columns={columns}
          rowKey="id"
          loading={loading.limits}
          pagination={{
            pageSize: 10,
            showSizeChanger: true,
            showQuickJumper: true,
            showTotal: (total) => `Total ${total} limits`
          }}
          rowClassName={(record) => {
            if (record.breach_count > 0) return 'limit-breached';
            if (!record.active) return 'limit-inactive';
            return '';
          }}
        />
      </Card>

      {/* Create Limit Modal */}
      <Modal
        title="Create New Risk Limit"
        open={showCreateModal}
        onCancel={() => {
          setShowCreateModal(false);
          createForm.resetFields();
        }}
        footer={null}
        width={600}
      >
        <Form
          form={createForm}
          layout="vertical"
          onFinish={handleCreateLimit}
          initialValues={{ active: true, action: 'warn' }}
        >
          <Form.Item
            name="name"
            label="Limit Name"
            rules={[{ required: true, message: 'Please enter limit name' }]}
          >
            <Input placeholder="e.g., Daily VaR Limit" />
          </Form.Item>

          <Form.Item
            name="limit_type"
            label="Limit Type"
            rules={[{ required: true, message: 'Please select limit type' }]}
          >
            <Select placeholder="Select limit type">
              <Option value="var">Value at Risk (VaR)</Option>
              <Option value="concentration">Position Concentration</Option>
              <Option value="position_size">Position Size</Option>
              <Option value="leverage">Portfolio Leverage</Option>
              <Option value="correlation">Portfolio Correlation</Option>
            </Select>
          </Form.Item>

          <Row gutter={16}>
            <Col span={12}>
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
            </Col>
            <Col span={12}>
              <Form.Item
                name="threshold_value"
                label="Breach Threshold"
                rules={[{ required: true, message: 'Please enter breach threshold' }]}
              >
                <InputNumber
                  style={{ width: '100%' }}
                  placeholder="Maximum allowed value"
                  min={0}
                />
              </Form.Item>
            </Col>
          </Row>

          <Form.Item
            name="action"
            label="Action on Breach"
            rules={[{ required: true, message: 'Please select action' }]}
          >
            <Select>
              <Option value="warn">Warning Only</Option>
              <Option value="block">Block New Positions</Option>
              <Option value="reduce">Auto-Reduce Positions</Option>
              <Option value="notify">Notify Risk Team</Option>
            </Select>
          </Form.Item>

          <Form.Item name="active" label="Active" valuePropName="checked">
            <Switch />
          </Form.Item>

          <Form.Item style={{ marginBottom: 0 }}>
            <Space style={{ width: '100%', justifyContent: 'flex-end' }}>
              <Button onClick={() => {
                setShowCreateModal(false);
                createForm.resetFields();
              }}>
                Cancel
              </Button>
              <Button type="primary" htmlType="submit" loading={loading.updating}>
                Create Limit
              </Button>
            </Space>
          </Form.Item>
        </Form>
      </Modal>

      {/* Edit Limit Modal */}
      <Modal
        title="Edit Risk Limit"
        open={showEditModal}
        onCancel={() => {
          setShowEditModal(false);
          setEditingLimit(null);
          editForm.resetFields();
        }}
        footer={null}
        width={600}
      >
        <Form
          form={editForm}
          layout="vertical"
          onFinish={handleEditLimit}
        >
          <Row gutter={16}>
            <Col span={12}>
              <Form.Item
                name="warning_threshold"
                label="Warning Threshold"
                rules={[{ required: true, message: 'Please enter warning threshold' }]}
              >
                <InputNumber
                  style={{ width: '100%' }}
                  min={0}
                />
              </Form.Item>
            </Col>
            <Col span={12}>
              <Form.Item
                name="threshold_value"
                label="Breach Threshold"
                rules={[{ required: true, message: 'Please enter breach threshold' }]}
              >
                <InputNumber
                  style={{ width: '100%' }}
                  min={0}
                />
              </Form.Item>
            </Col>
          </Row>

          <Form.Item
            name="action"
            label="Action on Breach"
            rules={[{ required: true, message: 'Please select action' }]}
          >
            <Select>
              <Option value="warn">Warning Only</Option>
              <Option value="block">Block New Positions</Option>
              <Option value="reduce">Auto-Reduce Positions</Option>
              <Option value="notify">Notify Risk Team</Option>
            </Select>
          </Form.Item>

          <Form.Item name="active" label="Active" valuePropName="checked">
            <Switch />
          </Form.Item>

          <Form.Item style={{ marginBottom: 0 }}>
            <Space style={{ width: '100%', justifyContent: 'flex-end' }}>
              <Button onClick={() => {
                setShowEditModal(false);
                setEditingLimit(null);
                editForm.resetFields();
              }}>
                Cancel
              </Button>
              <Button type="primary" htmlType="submit" loading={loading.updating}>
                Update Limit
              </Button>
            </Space>
          </Form.Item>
        </Form>
      </Modal>

      {/* Configuration Drawer */}
      <Drawer
        title="Risk Limit Configuration"
        placement="right"
        width={600}
        onClose={() => setShowConfigDrawer(false)}
        open={showConfigDrawer}
      >
        <Form
          form={configForm}
          layout="vertical"
          initialValues={configuration}
        >
          <Title level={5}>Notification Settings</Title>
          <Form.Item name={['notification_settings', 'email_alerts']} label="Email Alerts" valuePropName="checked">
            <Switch />
          </Form.Item>
          <Form.Item name={['notification_settings', 'dashboard_alerts']} label="Dashboard Alerts" valuePropName="checked">
            <Switch />
          </Form.Item>
          <Form.Item name={['notification_settings', 'webhook_url']} label="Webhook URL">
            <Input placeholder="https://your-webhook-url.com" />
          </Form.Item>

          <Divider />

          <Title level={5}>Escalation Rules</Title>
          <Alert
            message="Escalation Configuration"
            description="Configure automatic escalation rules when multiple breaches occur or critical thresholds are exceeded."
            type="info"
            style={{ marginBottom: 16 }}
          />

          {/* Simplified escalation configuration - would be more complex in real implementation */}
          <Form.Item label="Auto Escalation Time">
            <InputNumber
              style={{ width: '100%' }}
              placeholder="Minutes before escalation"
              min={1}
              max={1440}
            />
          </Form.Item>

          <div style={{ marginTop: 24 }}>
            <Button type="primary" size="large" block>
              Save Configuration
            </Button>
          </div>
        </Form>
      </Drawer>

      <style jsx>{`
        .limit-breached {
          background-color: #fff2f0 !important;
        }
        .limit-inactive {
          background-color: #f5f5f5 !important;
          opacity: 0.7;
        }
      `}</style>
    </div>
  );
};

export default DynamicLimitEngine;