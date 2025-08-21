/**
 * EngineConfigPanel - Configuration management for NautilusTrader engine
 * 
 * Provides interface for modifying engine settings including resource limits,
 * risk parameters, and trading mode configuration.
 */

import React, { useState, useEffect } from 'react';
import { 
  Form, 
  Row, 
  Col, 
  Select, 
  Switch, 
  Slider, 
  Input, 
  Button, 
  Card, 
  Space, 
  Alert, 
  Divider, 
  Typography,
  Tooltip,
  Tag
} from 'antd';
import {
  SettingOutlined,
  SaveOutlined,
  ReloadOutlined,
  InfoCircleOutlined,
  WarningOutlined,
  ExclamationCircleOutlined
} from '@ant-design/icons';

const { Text, Title } = Typography;
const { Option } = Select;

interface EngineConfig {
  engine_type: string;
  log_level: string;
  instance_id: string;
  trading_mode: 'paper' | 'live';
  max_memory: string;
  max_cpu: string;
  data_catalog_path: string;
  cache_database_path: string;
  risk_engine_enabled: boolean;
  max_position_size?: number;
  max_order_rate?: number;
}

interface EngineConfigPanelProps {
  config: EngineConfig;
  engineState: string;
  onConfigChange: (config: EngineConfig) => void;
  showAdvanced?: boolean;
}

const DEFAULT_CONFIG: EngineConfig = {
  engine_type: 'live',
  log_level: 'INFO',
  instance_id: 'nautilus-001',
  trading_mode: 'paper',
  max_memory: '2g',
  max_cpu: '2.0',
  data_catalog_path: '/app/data',
  cache_database_path: '/app/cache',
  risk_engine_enabled: true,
  max_position_size: 100000,
  max_order_rate: 100
};

export const EngineConfigPanel: React.FC<EngineConfigPanelProps> = ({
  config,
  engineState,
  onConfigChange,
  showAdvanced = false
}) => {
  const [form] = Form.useForm();
  const [hasChanges, setHasChanges] = useState(false);
  const [expandedAdvanced, setExpandedAdvanced] = useState(showAdvanced);
  const [validationErrors, setValidationErrors] = useState<string[]>([]);

  const isRunning = engineState === 'running';
  const isTransitioning = engineState === 'starting' || engineState === 'stopping';

  useEffect(() => {
    form.setFieldsValue(config);
    setHasChanges(false);
  }, [config, form]);

  const handleFormChange = () => {
    setHasChanges(true);
    validateConfiguration();
  };

  const validateConfiguration = () => {
    const errors: string[] = [];
    const formValues = form.getFieldsValue();

    // Validate memory configuration
    const memoryValue = parseInt(formValues.max_memory?.replace('g', '') || '0');
    if (memoryValue < 1) {
      errors.push('Memory allocation must be at least 1GB');
    }

    // Validate CPU configuration
    const cpuValue = parseFloat(formValues.max_cpu || '0');
    if (cpuValue < 0.5) {
      errors.push('CPU allocation must be at least 0.5 cores');
    }

    // Validate risk settings
    if (formValues.risk_engine_enabled) {
      if (formValues.max_position_size && formValues.max_position_size < 1000) {
        errors.push('Max position size should be at least $1,000');
      }
      if (formValues.max_order_rate && formValues.max_order_rate < 1) {
        errors.push('Max order rate should be at least 1 order per minute');
      }
    }

    // Live trading warnings
    if (formValues.trading_mode === 'live' && !formValues.risk_engine_enabled) {
      errors.push('Risk engine is highly recommended for live trading');
    }

    setValidationErrors(errors);
  };

  const handleSave = async () => {
    try {
      const values = await form.validateFields();
      onConfigChange(values as EngineConfig);
      setHasChanges(false);
    } catch (error) {
      console.error('Form validation failed:', error);
    }
  };

  const handleReset = () => {
    form.setFieldsValue(config);
    setHasChanges(false);
    setValidationErrors([]);
  };

  const handleDefaults = () => {
    form.setFieldsValue(DEFAULT_CONFIG);
    setHasChanges(true);
    validateConfiguration();
  };

  const getMemoryOptions = () => [
    { label: '512MB', value: '512m' },
    { label: '1GB', value: '1g' },
    { label: '2GB', value: '2g' },
    { label: '4GB', value: '4g' },
    { label: '8GB', value: '8g' },
    { label: '16GB', value: '16g' }
  ];

  const getCpuOptions = () => [
    { label: '0.5 Cores', value: '0.5' },
    { label: '1 Core', value: '1.0' },
    { label: '2 Cores', value: '2.0' },
    { label: '4 Cores', value: '4.0' },
    { label: '8 Cores', value: '8.0' }
  ];

  const getInstanceOptions = () => [
    { label: 'nautilus-001', value: 'nautilus-001' },
    { label: 'nautilus-002', value: 'nautilus-002' },
    { label: 'nautilus-003', value: 'nautilus-003' },
    { label: 'nautilus-dev', value: 'nautilus-dev' },
    { label: 'nautilus-test', value: 'nautilus-test' }
  ];

  return (
    <div className="engine-config-panel">
      <div style={{ marginBottom: 16 }}>
        <Space>
          <SettingOutlined />
          <Title level={4} style={{ margin: 0 }}>Engine Configuration</Title>
          {isRunning && (
            <Tag color="orange">Restart required for changes</Tag>
          )}
        </Space>
      </div>

      {/* Configuration Alerts */}
      {validationErrors.length > 0 && (
        <Alert
          type="warning"
          message="Configuration Issues"
          description={
            <ul style={{ margin: 0, paddingLeft: 20 }}>
              {validationErrors.map((error, index) => (
                <li key={index}>{error}</li>
              ))}
            </ul>
          }
          style={{ marginBottom: 16 }}
          showIcon
        />
      )}

      {isRunning && (
        <Alert
          type="info"
          message="Engine Running"
          description="The engine is currently running. Configuration changes will require a restart to take effect."
          style={{ marginBottom: 16 }}
          showIcon
        />
      )}

      <Form
        form={form}
        layout="vertical"
        onFieldsChange={handleFormChange}
        disabled={isTransitioning}
      >
        {/* Basic Configuration */}
        <Card title="Basic Settings" size="small" style={{ marginBottom: 16 }}>
          <Row gutter={16}>
            <Col xs={24} sm={12} lg={8}>
              <Form.Item
                label={
                  <Space>
                    <span>Engine Type</span>
                    <Tooltip title="Type of trading engine">
                      <InfoCircleOutlined />
                    </Tooltip>
                  </Space>
                }
                name="engine_type"
                rules={[{ required: true }]}
              >
                <Select>
                  <Option value="live">Live Trading</Option>
                  <Option value="backtest">Backtest</Option>
                  <Option value="sandbox">Sandbox</Option>
                </Select>
              </Form.Item>
            </Col>

            <Col xs={24} sm={12} lg={8}>
              <Form.Item
                label={
                  <Space>
                    <span>Trading Mode</span>
                    <Tooltip title="Paper trading uses simulated money, live trading uses real money">
                      <InfoCircleOutlined />
                    </Tooltip>
                  </Space>
                }
                name="trading_mode"
                rules={[{ required: true }]}
              >
                <Select>
                  <Option value="paper">
                    <Space>
                      📊 Paper Trading
                      <Text type="secondary">(Simulated)</Text>
                    </Space>
                  </Option>
                  <Option value="live">
                    <Space>
                      🚨 Live Trading
                      <Text type="danger">(Real Money)</Text>
                    </Space>
                  </Option>
                </Select>
              </Form.Item>
            </Col>

            <Col xs={24} sm={12} lg={8}>
              <Form.Item
                label="Instance ID"
                name="instance_id"
                rules={[{ required: true }]}
              >
                <Select>
                  {getInstanceOptions().map(option => (
                    <Option key={option.value} value={option.value}>
                      {option.label}
                    </Option>
                  ))}
                </Select>
              </Form.Item>
            </Col>
          </Row>

          <Row gutter={16}>
            <Col xs={24} sm={12}>
              <Form.Item
                label="Log Level"
                name="log_level"
                rules={[{ required: true }]}
              >
                <Select>
                  <Option value="DEBUG">Debug (Verbose)</Option>
                  <Option value="INFO">Info (Normal)</Option>
                  <Option value="WARNING">Warning (Reduced)</Option>
                  <Option value="ERROR">Error (Minimal)</Option>
                </Select>
              </Form.Item>
            </Col>
          </Row>
        </Card>

        {/* Resource Configuration */}
        <Card title="Resource Limits" size="small" style={{ marginBottom: 16 }}>
          <Row gutter={16}>
            <Col xs={24} sm={12}>
              <Form.Item
                label={
                  <Space>
                    <span>Memory Limit</span>
                    <Tooltip title="Maximum memory the engine can use">
                      <InfoCircleOutlined />
                    </Tooltip>
                  </Space>
                }
                name="max_memory"
                rules={[{ required: true }]}
              >
                <Select>
                  {getMemoryOptions().map(option => (
                    <Option key={option.value} value={option.value}>
                      {option.label}
                    </Option>
                  ))}
                </Select>
              </Form.Item>
            </Col>

            <Col xs={24} sm={12}>
              <Form.Item
                label={
                  <Space>
                    <span>CPU Limit</span>
                    <Tooltip title="Maximum CPU cores the engine can use">
                      <InfoCircleOutlined />
                    </Tooltip>
                  </Space>
                }
                name="max_cpu"
                rules={[{ required: true }]}
              >
                <Select>
                  {getCpuOptions().map(option => (
                    <Option key={option.value} value={option.value}>
                      {option.label}
                    </Option>
                  ))}
                </Select>
              </Form.Item>
            </Col>
          </Row>
        </Card>

        {/* Risk Management */}
        <Card title="Risk Management" size="small" style={{ marginBottom: 16 }}>
          <Form.Item
            label={
              <Space>
                <span>Risk Engine</span>
                <Tooltip title="Enable risk engine for position and order validation">
                  <InfoCircleOutlined />
                </Tooltip>
              </Space>
            }
            name="risk_engine_enabled"
            valuePropName="checked"
          >
            <Switch 
              checkedChildren="Enabled" 
              unCheckedChildren="Disabled"
              style={{ background: form.getFieldValue('risk_engine_enabled') ? '#52c41a' : '#d9d9d9' }}
            />
          </Form.Item>

          <Form.Item
            noStyle
            shouldUpdate={(prevValues, currentValues) => 
              prevValues.risk_engine_enabled !== currentValues.risk_engine_enabled
            }
          >
            {({ getFieldValue }) =>
              getFieldValue('risk_engine_enabled') ? (
                <Row gutter={16}>
                  <Col xs={24} lg={12}>
                    <Form.Item
                      label={
                        <Space>
                          <span>Max Position Size</span>
                          <Tooltip title="Maximum position size per instrument">
                            <InfoCircleOutlined />
                          </Tooltip>
                        </Space>
                      }
                      name="max_position_size"
                    >
                      <Slider
                        min={1000}
                        max={10000000}
                        step={1000}
                        marks={{
                          1000: '$1K',
                          100000: '$100K',
                          1000000: '$1M',
                          10000000: '$10M'
                        }}
                        tooltip={{
                          formatter: (value) => `$${(value || 0).toLocaleString()}`
                        }}
                      />
                    </Form.Item>
                  </Col>
                  
                  <Col xs={24} lg={12}>
                    <Form.Item
                      label={
                        <Space>
                          <span>Max Order Rate</span>
                          <Tooltip title="Maximum orders per minute">
                            <InfoCircleOutlined />
                          </Tooltip>
                        </Space>
                      }
                      name="max_order_rate"
                    >
                      <Slider
                        min={1}
                        max={1000}
                        step={1}
                        marks={{
                          1: '1',
                          100: '100',
                          500: '500',
                          1000: '1000'
                        }}
                        tooltip={{
                          formatter: (value) => `${value} orders/min`
                        }}
                      />
                    </Form.Item>
                  </Col>
                </Row>
              ) : (
                <Alert
                  type="warning"
                  message="Risk Engine Disabled"
                  description="Without the risk engine, the system will not validate position sizes or order rates. This is not recommended for live trading."
                  showIcon
                  style={{ marginTop: 8 }}
                />
              )
            }
          </Form.Item>
        </Card>

        {/* Advanced Configuration */}
        <Card 
          title={
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
              <span>Advanced Settings</span>
              <Button 
                type="link" 
                size="small"
                onClick={() => setExpandedAdvanced(!expandedAdvanced)}
              >
                {expandedAdvanced ? 'Hide' : 'Show'} Advanced
              </Button>
            </div>
          }
          size="small" 
          style={{ marginBottom: 16 }}
        >
          {expandedAdvanced && (
            <Row gutter={16}>
              <Col xs={24} lg={12}>
                <Form.Item
                  label={
                    <Space>
                      <span>Data Catalog Path</span>
                      <Tooltip title="Path to market data catalog">
                        <InfoCircleOutlined />
                      </Tooltip>
                    </Space>
                  }
                  name="data_catalog_path"
                >
                  <Input placeholder="/app/data" />
                </Form.Item>
              </Col>
              
              <Col xs={24} lg={12}>
                <Form.Item
                  label={
                    <Space>
                      <span>Cache Database Path</span>
                      <Tooltip title="Path to cache database">
                        <InfoCircleOutlined />
                      </Tooltip>
                    </Space>
                  }
                  name="cache_database_path"
                >
                  <Input placeholder="/app/cache" />
                </Form.Item>
              </Col>
            </Row>
          )}
        </Card>

        {/* Action Buttons */}
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <Space>
            <Button onClick={handleDefaults}>
              <ReloadOutlined />
              Reset to Defaults
            </Button>
          </Space>
          
          <Space>
            <Button onClick={handleReset} disabled={!hasChanges}>
              Cancel Changes
            </Button>
            <Button
              type="primary"
              icon={<SaveOutlined />}
              onClick={handleSave}
              disabled={!hasChanges || validationErrors.length > 0}
            >
              Save Configuration
            </Button>
          </Space>
        </div>

        {/* Live Trading Warning */}
        <Form.Item
          noStyle
          shouldUpdate={(prevValues, currentValues) => 
            prevValues.trading_mode !== currentValues.trading_mode
          }
        >
          {({ getFieldValue }) =>
            getFieldValue('trading_mode') === 'live' ? (
              <Alert
                type="error"
                message="Live Trading Mode Selected"
                description="You have selected live trading mode. This will use real money for all trades. Ensure you understand the risks before starting the engine."
                showIcon
                icon={<ExclamationCircleOutlined />}
                style={{ marginTop: 16 }}
              />
            ) : null
          }
        </Form.Item>
      </Form>

      <style jsx>{`
        .engine-config-panel {
          width: 100%;
        }

        .ant-slider {
          margin: 8px 0;
        }

        .ant-form-item-label {
          font-weight: 500;
        }

        .ant-card-head-title {
          font-size: 14px;
          font-weight: 600;
        }

        .config-warning {
          border-left: 4px solid #faad14;
        }

        .config-error {
          border-left: 4px solid #ff4d4f;
        }
      `}</style>
    </div>
  );
};

export default EngineConfigPanel;