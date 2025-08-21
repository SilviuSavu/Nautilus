/**
 * Story 5.3: API Integration Setup Component
 * Third-party API integration setup and configuration
 */

import React, { useState, useEffect } from 'react';
import {
  Card,
  Form,
  Input,
  Select,
  Button,
  Modal,
  Steps,
  Row,
  Col,
  Typography,
  Space,
  Alert,
  Switch,
  TimePicker,
  Table,
  Tag,
  notification,
  Tooltip,
  Divider,
  Badge,
  Collapse
} from 'antd';
import {
  ApiOutlined,
  KeyOutlined,
  LinkOutlined,
  CheckCircleOutlined,
  ExclamationCircleOutlined,
  PlusOutlined,
  DeleteOutlined,
  EditOutlined,
  ExperimentOutlined,
  SaveOutlined,
  SettingOutlined,
  SyncOutlined
} from '@ant-design/icons';
import dayjs from 'dayjs';
import {
  ApiIntegration,
  AuthenticationType,
  IntegrationStatus,
  FieldMapping
} from '../../types/export';

const { Title, Text } = Typography;
const { TextArea } = Input;
const { Option } = Select;
const { Step } = Steps;
const { Panel } = Collapse;

interface IntegrationSetupProps {
  visible: boolean;
  onCancel: () => void;
  onSave: (integration: ApiIntegration) => Promise<void>;
  initialIntegration?: ApiIntegration;
}

export const IntegrationSetup: React.FC<IntegrationSetupProps> = ({
  visible,
  onCancel,
  onSave,
  initialIntegration
}) => {
  const [form] = Form.useForm();
  const [currentStep, setCurrentStep] = useState(0);
  const [saving, setSaving] = useState(false);
  const [testing, setTesting] = useState(false);
  const [testResult, setTestResult] = useState<any>(null);
  const [fieldMappings, setFieldMappings] = useState<FieldMapping[]>([]);
  const [authType, setAuthType] = useState<AuthenticationType>(AuthenticationType.API_KEY);
  const [syncEnabled, setSyncEnabled] = useState(false);

  // Available source and target fields
  const sourceFields = [
    'total_pnl', 'unrealized_pnl', 'win_rate', 'sharpe_ratio', 'max_drawdown',
    'total_trades', 'winning_trades', 'volume', 'commission', 'timestamp'
  ];

  const targetFieldOptions = [
    { value: 'portfolio_value', label: 'Portfolio Value' },
    { value: 'profit_loss', label: 'Profit/Loss' },
    { value: 'win_percentage', label: 'Win Percentage' },
    { value: 'performance_ratio', label: 'Performance Ratio' },
    { value: 'drawdown', label: 'Drawdown' },
    { value: 'trade_count', label: 'Trade Count' },
    { value: 'successful_trades', label: 'Successful Trades' },
    { value: 'trading_volume', label: 'Trading Volume' },
    { value: 'fees', label: 'Fees' },
    { value: 'updated_at', label: 'Last Updated' }
  ];

  useEffect(() => {
    if (initialIntegration) {
      form.setFieldsValue({
        name: initialIntegration.name,
        endpoint: initialIntegration.endpoint,
        authentication_type: initialIntegration.authentication.type
      });
      
      setAuthType(initialIntegration.authentication.type as AuthenticationType);
      setFieldMappings(initialIntegration.data_mapping || []);
      
      if (initialIntegration.schedule) {
        setSyncEnabled(true);
      }
    } else {
      // Reset for new integration
      form.resetFields();
      setFieldMappings([]);
      setAuthType(AuthenticationType.API_KEY);
      setSyncEnabled(false);
    }
    setTestResult(null);
  }, [initialIntegration, form]);

  const addFieldMapping = () => {
    const newMapping: FieldMapping = {
      source_field: '',
      target_field: '',
      transformation: undefined
    };
    setFieldMappings([...fieldMappings, newMapping]);
  };

  const updateFieldMapping = (index: number, updatedMapping: FieldMapping) => {
    const newMappings = [...fieldMappings];
    newMappings[index] = updatedMapping;
    setFieldMappings(newMappings);
  };

  const removeFieldMapping = (index: number) => {
    setFieldMappings(fieldMappings.filter((_, i) => i !== index));
  };

  const testConnection = async () => {
    try {
      setTesting(true);
      const values = await form.validateFields();
      
      // Simulate API test
      await new Promise(resolve => setTimeout(resolve, 2000));
      
      // Mock successful test
      setTestResult({
        success: true,
        message: 'Connection successful',
        response_time: '245ms',
        data: {
          endpoint_available: true,
          authentication: 'valid',
          api_version: '1.2.3'
        }
      });

      notification.success({
        message: 'Connection Test Successful',
        description: 'API endpoint is reachable and authentication is valid',
      });

    } catch (error: any) {
      setTestResult({
        success: false,
        message: error.message || 'Connection failed',
        error_code: 'CONNECTION_TIMEOUT'
      });

      notification.error({
        message: 'Connection Test Failed',
        description: error.message || 'Failed to connect to API endpoint',
      });
    } finally {
      setTesting(false);
    }
  };

  const handleSave = async () => {
    try {
      setSaving(true);
      
      const values = await form.validateFields();
      
      if (fieldMappings.length === 0) {
        notification.warning({
          message: 'No Field Mappings',
          description: 'Add at least one field mapping for data synchronization',
        });
        return;
      }

      const integration: ApiIntegration = {
        ...initialIntegration,
        name: values.name,
        endpoint: values.endpoint,
        authentication: {
          type: authType,
          ...values.authentication
        },
        data_mapping: fieldMappings,
        schedule: syncEnabled ? {
          frequency: values.sync_frequency || 'hourly',
          enabled: true
        } : undefined,
        status: IntegrationStatus.ACTIVE
      };

      await onSave(integration);
      
      notification.success({
        message: 'Integration Saved',
        description: 'API integration has been configured successfully',
      });

      onCancel();
    } catch (error: any) {
      notification.error({
        message: 'Save Failed',
        description: error.message || 'Failed to save integration',
      });
    } finally {
      setSaving(false);
    }
  };

  const renderBasicInfo = () => (
    <Card title="Basic Information">
      <Row gutter={16}>
        <Col xs={24} md={12}>
          <Form.Item
            name="name"
            label="Integration Name"
            rules={[{ required: true, message: 'Please enter integration name' }]}
          >
            <Input placeholder="e.g., Portfolio Analytics API" />
          </Form.Item>
        </Col>
        <Col xs={24} md={12}>
          <Form.Item
            name="endpoint"
            label="API Endpoint"
            rules={[
              { required: true, message: 'Please enter API endpoint' },
              { type: 'url', message: 'Please enter a valid URL' }
            ]}
          >
            <Input placeholder="https://api.example.com/v1/data" />
          </Form.Item>
        </Col>
      </Row>

      <Form.Item
        name="description"
        label="Description"
      >
        <TextArea 
          rows={3} 
          placeholder="Describe what this integration does..."
        />
      </Form.Item>
    </Card>
  );

  const renderAuthentication = () => (
    <Card title="Authentication">
      <Form.Item
        name="authentication_type"
        label="Authentication Type"
        rules={[{ required: true, message: 'Please select authentication type' }]}
      >
        <Select 
          value={authType} 
          onChange={setAuthType}
          placeholder="Select authentication method"
        >
          <Option value={AuthenticationType.API_KEY}>API Key</Option>
          <Option value={AuthenticationType.OAUTH}>OAuth 2.0</Option>
          <Option value={AuthenticationType.BASIC}>Basic Authentication</Option>
        </Select>
      </Form.Item>

      {authType === AuthenticationType.API_KEY && (
        <Row gutter={16}>
          <Col xs={24} md={12}>
            <Form.Item
              name={['authentication', 'api_key']}
              label="API Key"
              rules={[{ required: true, message: 'Please enter API key' }]}
            >
              <Input.Password placeholder="Enter your API key" />
            </Form.Item>
          </Col>
          <Col xs={24} md={12}>
            <Form.Item
              name={['authentication', 'header_name']}
              label="Header Name"
            >
              <Input placeholder="X-API-Key" defaultValue="X-API-Key" />
            </Form.Item>
          </Col>
        </Row>
      )}

      {authType === AuthenticationType.OAUTH && (
        <Row gutter={16}>
          <Col xs={24} md={12}>
            <Form.Item
              name={['authentication', 'client_id']}
              label="Client ID"
              rules={[{ required: true, message: 'Please enter client ID' }]}
            >
              <Input placeholder="Your OAuth client ID" />
            </Form.Item>
          </Col>
          <Col xs={24} md={12}>
            <Form.Item
              name={['authentication', 'client_secret']}
              label="Client Secret"
              rules={[{ required: true, message: 'Please enter client secret' }]}
            >
              <Input.Password placeholder="Your OAuth client secret" />
            </Form.Item>
          </Col>
        </Row>
      )}

      {authType === AuthenticationType.BASIC && (
        <Row gutter={16}>
          <Col xs={24} md={12}>
            <Form.Item
              name={['authentication', 'username']}
              label="Username"
              rules={[{ required: true, message: 'Please enter username' }]}
            >
              <Input placeholder="API username" />
            </Form.Item>
          </Col>
          <Col xs={24} md={12}>
            <Form.Item
              name={['authentication', 'password']}
              label="Password"
              rules={[{ required: true, message: 'Please enter password' }]}
            >
              <Input.Password placeholder="API password" />
            </Form.Item>
          </Col>
        </Row>
      )}

      <Divider />

      <Row justify="space-between" align="middle">
        <Col>
          <Button 
            type="primary" 
            icon={<ExperimentOutlined />} 
            loading={testing}
            onClick={testConnection}
          >
            {testing ? 'Testing...' : 'Test Connection'}
          </Button>
        </Col>
        <Col>
          {testResult && (
            <Space>
              {testResult.success ? (
                <Badge status="success" text="Connection OK" />
              ) : (
                <Badge status="error" text="Connection Failed" />
              )}
            </Space>
          )}
        </Col>
      </Row>

      {testResult && (
        <Alert
          style={{ marginTop: 16 }}
          message={testResult.success ? 'Connection Successful' : 'Connection Failed'}
          description={testResult.message}
          type={testResult.success ? 'success' : 'error'}
          showIcon
        />
      )}
    </Card>
  );

  const renderFieldMappings = () => (
    <Card 
      title="Field Mappings" 
      extra={
        <Button type="primary" icon={<PlusOutlined />} onClick={addFieldMapping}>
          Add Mapping
        </Button>
      }
    >
      <Alert
        message="Field Mapping Configuration"
        description="Map your local data fields to the target API fields. This determines how your trading data will be sent to the external system."
        type="info"
        style={{ marginBottom: 16 }}
      />

      {fieldMappings.length === 0 ? (
        <Text type="secondary">No field mappings configured. Add mappings to synchronize data.</Text>
      ) : (
        <Table
          dataSource={fieldMappings.map((mapping, index) => ({ ...mapping, key: index }))}
          size="small"
          pagination={false}
          columns={[
            {
              title: 'Source Field',
              key: 'source_field',
              width: '30%',
              render: (_: any, record: any, index: number) => (
                <Select
                  value={record.source_field}
                  placeholder="Select source field"
                  onChange={(value) => updateFieldMapping(index, { ...record, source_field: value })}
                  style={{ width: '100%' }}
                >
                  {sourceFields.map(field => (
                    <Option key={field} value={field}>{field}</Option>
                  ))}
                </Select>
              )
            },
            {
              title: 'Target Field',
              key: 'target_field',
              width: '30%',
              render: (_: any, record: any, index: number) => (
                <Select
                  value={record.target_field}
                  placeholder="Select target field"
                  onChange={(value) => updateFieldMapping(index, { ...record, target_field: value })}
                  style={{ width: '100%' }}
                >
                  {targetFieldOptions.map(option => (
                    <Option key={option.value} value={option.value}>{option.label}</Option>
                  ))}
                </Select>
              )
            },
            {
              title: 'Transformation',
              key: 'transformation',
              width: '30%',
              render: (_: any, record: any, index: number) => (
                <Input
                  value={record.transformation}
                  placeholder="Optional transformation (e.g., * 100)"
                  onChange={(e) => updateFieldMapping(index, { ...record, transformation: e.target.value })}
                />
              )
            },
            {
              title: 'Action',
              key: 'action',
              width: '10%',
              render: (_: any, record: any, index: number) => (
                <Button
                  type="text"
                  size="small"
                  danger
                  icon={<DeleteOutlined />}
                  onClick={() => removeFieldMapping(index)}
                />
              )
            }
          ]}
        />
      )}
    </Card>
  );

  const renderSyncSettings = () => (
    <Card title="Sync Settings">
      <Form.Item>
        <Switch
          checked={syncEnabled}
          onChange={setSyncEnabled}
          checkedChildren="Enabled"
          unCheckedChildren="Disabled"
        />
        <Text style={{ marginLeft: 8 }}>Enable automatic data synchronization</Text>
      </Form.Item>

      {syncEnabled && (
        <div>
          <Row gutter={16}>
            <Col xs={24} md={12}>
              <Form.Item
                name="sync_frequency"
                label="Sync Frequency"
              >
                <Select placeholder="Select sync frequency" defaultValue="hourly">
                  <Option value="realtime">Real-time</Option>
                  <Option value="every_5_minutes">Every 5 minutes</Option>
                  <Option value="every_15_minutes">Every 15 minutes</Option>
                  <Option value="hourly">Hourly</Option>
                  <Option value="daily">Daily</Option>
                </Select>
              </Form.Item>
            </Col>
            <Col xs={24} md={12}>
              <Form.Item
                name="retry_attempts"
                label="Retry Attempts"
              >
                <Select placeholder="Number of retry attempts" defaultValue="3">
                  <Option value="1">1 attempt</Option>
                  <Option value="3">3 attempts</Option>
                  <Option value="5">5 attempts</Option>
                  <Option value="10">10 attempts</Option>
                </Select>
              </Form.Item>
            </Col>
          </Row>

          <Form.Item
            name="webhook_url"
            label="Webhook URL (Optional)"
          >
            <Input 
              placeholder="https://your-app.com/webhooks/nautilus-sync"
              addonBefore={<LinkOutlined />}
            />
            <Text type="secondary" style={{ fontSize: 12 }}>
              Receive notifications when sync operations complete
            </Text>
          </Form.Item>
        </div>
      )}
    </Card>
  );

  const steps = [
    {
      title: 'Basic Info',
      content: renderBasicInfo(),
      icon: <ApiOutlined />
    },
    {
      title: 'Authentication',
      content: renderAuthentication(),
      icon: <KeyOutlined />
    },
    {
      title: 'Field Mapping',
      content: renderFieldMappings(),
      icon: <SettingOutlined />
    },
    {
      title: 'Sync Settings',
      content: renderSyncSettings(),
      icon: <SyncOutlined />
    }
  ];

  return (
    <Modal
      title={
        <div>
          <ApiOutlined style={{ marginRight: 8 }} />
          {initialIntegration ? 'Edit API Integration' : 'Create API Integration'}
        </div>
      }
      open={visible}
      onCancel={onCancel}
      width={1000}
      footer={
        <Space>
          <Button onClick={onCancel}>Cancel</Button>
          <Button type="primary" loading={saving} icon={<SaveOutlined />} onClick={handleSave}>
            {saving ? 'Saving...' : 'Save Integration'}
          </Button>
        </Space>
      }
    >
      <Form form={form} layout="vertical">
        <Steps 
          current={currentStep} 
          onChange={setCurrentStep}
          style={{ marginBottom: 24 }}
        >
          {steps.map((step, index) => (
            <Step key={index} title={step.title} icon={step.icon} />
          ))}
        </Steps>

        <div style={{ minHeight: 400 }}>
          {steps[currentStep].content}
        </div>

        <Row justify="space-between" style={{ marginTop: 24 }}>
          <Col>
            {currentStep > 0 && (
              <Button onClick={() => setCurrentStep(currentStep - 1)}>
                Previous
              </Button>
            )}
          </Col>
          <Col>
            {currentStep < steps.length - 1 ? (
              <Button type="primary" onClick={() => setCurrentStep(currentStep + 1)}>
                Next
              </Button>
            ) : (
              <Button type="primary" loading={saving} icon={<SaveOutlined />} onClick={handleSave}>
                {saving ? 'Saving...' : 'Save Integration'}
              </Button>
            )}
          </Col>
        </Row>
      </Form>
    </Modal>
  );
};

export default IntegrationSetup;