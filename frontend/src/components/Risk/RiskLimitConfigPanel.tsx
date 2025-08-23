import React, { useState, useEffect } from 'react';
import {
  Card,
  Form,
  Input,
  InputNumber,
  Select,
  Switch,
  Button,
  Space,
  Alert,
  Typography,
  Row,
  Col,
  Table,
  Modal,
  Tabs,
  Slider,
  Progress,
  Tag,
  Tooltip,
  Divider,
  List,
  Avatar,
  Badge,
  Popconfirm,
  notification,
  Collapse,
  Timeline,
  Statistic
} from 'antd';
import {
  SettingOutlined,
  PlusOutlined,
  EditOutlined,
  DeleteOutlined,
  ThunderboltOutlined,
  RobotOutlined,
  WarningOutlined,
  CheckCircleOutlined,
  CloseCircleOutlined,
  SyncOutlined,
  BulbOutlined,
  LineChartOutlined,
  ShieldOutlined,
  ExperimentOutlined,
  BarChartOutlined,
  DashboardOutlined,
  MonitorOutlined,
  AlertOutlined,
  SaveOutlined,
  ReloadOutlined,
  InfoCircleOutlined
} from '@ant-design/icons';

import { 
  DynamicRiskLimit, 
  RiskLimitType, 
  RiskConfigurationSprint3,
  ActiveRiskModel,
  NotificationPreference,
  CustomThreshold
} from './types/riskTypes';
import { riskService } from './services/riskService';
import { useDynamicLimits } from '../../hooks/risk/useDynamicLimits';

const { Title, Text, Paragraph } = Typography;
const { Option } = Select;
const { Panel } = Collapse;

interface RiskLimitConfigPanelProps {
  portfolioId: string;
  className?: string;
}

const RiskLimitConfigPanel: React.FC<RiskLimitConfigPanelProps> = ({
  portfolioId,
  className
}) => {
  console.log('🎯 RiskLimitConfigPanel rendering for portfolio:', portfolioId);

  const [form] = Form.useForm();
  const [limits, setLimits] = useState<DynamicRiskLimit[]>([]);
  const [limitTypes, setLimitTypes] = useState<RiskLimitType[]>([]);
  const [configuration, setConfiguration] = useState<RiskConfigurationSprint3 | null>(null);
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [editModalVisible, setEditModalVisible] = useState(false);
  const [selectedLimit, setSelectedLimit] = useState<DynamicRiskLimit | null>(null);
  const [testModalVisible, setTestModalVisible] = useState(false);
  const [activeTab, setActiveTab] = useState('limits');

  // Use dynamic limits hook
  const {
    limits: hookLimits,
    breachedLimits,
    activeLimits,
    riskScore
  } = useDynamicLimits({ portfolioId });

  const fetchData = async () => {
    try {
      setError(null);
      const [limitsData, typesData, configData] = await Promise.all([
        riskService.getDynamicLimits(portfolioId),
        riskService.getAvailableRiskLimitTypes(),
        riskService.getRiskConfiguration(portfolioId)
      ]);
      setLimits(limitsData);
      setLimitTypes(typesData);
      setConfiguration(configData);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to fetch risk configuration');
      console.error('Risk config fetch error:', err);
    } finally {
      setLoading(false);
    }
  };

  const createLimit = async (values: any) => {
    try {
      setSaving(true);
      const limitType = limitTypes.find(t => t.id === values.limitTypeId);
      
      const newLimit: Omit<DynamicRiskLimit, 'id' | 'created_at' | 'updated_at'> = {
        name: values.name,
        portfolio_id: portfolioId,
        limit_type: values.limitType,
        threshold_value: values.thresholdValue.toString(),
        warning_threshold: values.warningThreshold.toString(),
        action: values.action,
        active: values.active ?? true,
        breach_count: 0,
        auto_adjustment_enabled: values.autoAdjustmentEnabled ?? false,
        ml_prediction_enabled: values.mlPredictionEnabled ?? false,
        adjustment_frequency_minutes: values.adjustmentFrequency ?? 60,
        sensitivity_factor: values.sensitivityFactor ?? 1.0,
        adjustment_history: []
      };

      const created = await riskService.createDynamicLimit(newLimit);
      setLimits(prev => [created, ...prev]);
      form.resetFields();
      
      notification.success({
        message: 'Risk Limit Created',
        description: `${values.name} has been created successfully`,
        duration: 3
      });
      
    } catch (err) {
      notification.error({
        message: 'Failed to Create Limit',
        description: err instanceof Error ? err.message : 'Unknown error occurred',
        duration: 4
      });
    } finally {
      setSaving(false);
    }
  };

  const updateLimit = async (limitId: string, updates: Partial<DynamicRiskLimit>) => {
    try {
      const updated = await riskService.updateDynamicLimit(limitId, updates);
      setLimits(prev => prev.map(limit => limit.id === limitId ? updated : limit));
      
      notification.success({
        message: 'Risk Limit Updated',
        description: 'Limit configuration has been saved',
        duration: 3
      });
    } catch (error) {
      notification.error({
        message: 'Update Failed',
        description: 'Unable to update the risk limit',
        duration: 4
      });
    }
  };

  const deleteLimit = async (limitId: string) => {
    try {
      await riskService.deleteRiskLimit(limitId);
      setLimits(prev => prev.filter(limit => limit.id !== limitId));
      
      notification.success({
        message: 'Risk Limit Deleted',
        description: 'Limit has been removed from the portfolio',
        duration: 3
      });
    } catch (error) {
      notification.error({
        message: 'Delete Failed',
        description: 'Unable to delete the risk limit',
        duration: 4
      });
    }
  };

  const adjustLimit = async (limitId: string, reason: string) => {
    try {
      const adjusted = await riskService.adjustLimitAutomatically(limitId, reason);
      setLimits(prev => prev.map(limit => limit.id === limitId ? adjusted : limit));
      
      notification.success({
        message: 'Limit Adjusted',
        description: 'Risk limit has been automatically adjusted',
        duration: 3
      });
    } catch (error) {
      notification.error({
        message: 'Adjustment Failed',
        description: 'Unable to adjust the risk limit',
        duration: 4
      });
    }
  };

  const testLimitImpact = async (limit: DynamicRiskLimit) => {
    try {
      const testScenarios = ['market_stress', 'volatility_spike', 'correlation_increase'];
      const results = await riskService.testRiskLimitImpact(portfolioId, limit.id, testScenarios);
      
      Modal.info({
        title: 'Limit Impact Test Results',
        width: 600,
        content: (
          <div>
            {results.test_results.map((result, index) => (
              <Card key={index} size="small" style={{ marginBottom: 8 }}>
                <Row justify="space-between" align="middle">
                  <Col>
                    <Text strong>{result.scenario.replace('_', ' ').toUpperCase()}</Text>
                  </Col>
                  <Col>
                    <Tag color={result.would_trigger ? 'red' : 'green'}>
                      {result.would_trigger ? 'WOULD TRIGGER' : 'SAFE'}
                    </Tag>
                  </Col>
                </Row>
                <Paragraph style={{ fontSize: '12px', margin: '8px 0 0 0' }}>
                  Impact: {result.estimated_impact}
                </Paragraph>
                <Paragraph style={{ fontSize: '12px', margin: '4px 0 0 0' }}>
                  Recommended: {result.recommended_action}
                </Paragraph>
              </Card>
            ))}
          </div>
        )
      });
    } catch (error) {
      notification.error({
        message: 'Test Failed',
        description: 'Unable to test limit impact',
        duration: 4
      });
    }
  };

  const saveConfiguration = async (config: Partial<RiskConfigurationSprint3>) => {
    try {
      setSaving(true);
      const updated = await riskService.updateRiskConfiguration(portfolioId, config);
      setConfiguration(updated);
      
      notification.success({
        message: 'Configuration Saved',
        description: 'Risk management configuration has been updated',
        duration: 3
      });
    } catch (error) {
      notification.error({
        message: 'Save Failed',
        description: 'Unable to save configuration changes',
        duration: 4
      });
    } finally {
      setSaving(false);
    }
  };

  const getLimitStatusColor = (limit: DynamicRiskLimit) => {
    if (!limit.active) return '#d9d9d9';
    if (breachedLimits.some(bl => bl.id === limit.id)) return '#ff4d4f';
    const utilization = parseFloat(limit.threshold_value) / parseFloat(limit.warning_threshold);
    if (utilization > 0.9) return '#fa8c16';
    return '#52c41a';
  };

  const getLimitTypeIcon = (type: string) => {
    const icons: Record<string, React.ReactNode> = {
      'var': <LineChartOutlined />,
      'concentration': <BarChartOutlined />,
      'position_size': <DashboardOutlined />,
      'leverage': <ThunderboltOutlined />,
      'correlation': <SyncOutlined />,
      'exposure': <MonitorOutlined />,
      'drawdown': <WarningOutlined />,
      'volatility': <AlertOutlined />
    };
    return icons[type] || <SettingOutlined />;
  };

  const limitsColumns = [
    {
      title: 'Limit',
      key: 'limit',
      render: (record: DynamicRiskLimit) => (
        <Space>
          <Avatar 
            size="small" 
            icon={getLimitTypeIcon(record.limit_type)}
            style={{ backgroundColor: getLimitStatusColor(record) }}
          />
          <div>
            <Text strong>{record.name}</Text>
            <br />
            <Text type="secondary" style={{ fontSize: '12px' }}>
              {record.limit_type.replace('_', ' ').toUpperCase()}
            </Text>
          </div>
        </Space>
      )
    },
    {
      title: 'Status',
      key: 'status',
      render: (record: DynamicRiskLimit) => (
        <Space direction="vertical" size={0}>
          <Badge 
            status={record.active ? 'processing' : 'default'}
            text={record.active ? 'Active' : 'Inactive'}
          />
          {record.ml_prediction_enabled && (
            <Tag size="small" color="blue">
              <RobotOutlined /> ML
            </Tag>
          )}
          {record.auto_adjustment_enabled && (
            <Tag size="small" color="green">
              <ThunderboltOutlined /> Auto
            </Tag>
          )}
        </Space>
      )
    },
    {
      title: 'Current / Limit',
      key: 'values',
      render: (record: DynamicRiskLimit) => {
        const currentValue = Math.random() * parseFloat(record.threshold_value); // Mock current value
        const utilization = (currentValue / parseFloat(record.threshold_value)) * 100;
        
        return (
          <div style={{ width: 120 }}>
            <div style={{ fontSize: '12px', marginBottom: 4 }}>
              <Text>{currentValue.toFixed(2)} / {record.threshold_value}</Text>
            </div>
            <Progress
              percent={Math.min(utilization, 100)}
              size="small"
              strokeColor={utilization > 90 ? '#ff4d4f' : utilization > 70 ? '#fa8c16' : '#52c41a'}
              showInfo={false}
            />
            <div style={{ fontSize: '11px', color: '#999' }}>
              {utilization.toFixed(1)}% utilized
            </div>
          </div>
        );
      }
    },
    {
      title: 'Breaches',
      dataIndex: 'breach_count',
      key: 'breach_count',
      render: (count: number, record: DynamicRiskLimit) => (
        <Space>
          <Badge count={count} style={{ backgroundColor: count > 0 ? '#ff4d4f' : '#52c41a' }} />
          {record.last_breach && (
            <Tooltip title={`Last breach: ${new Date(record.last_breach).toLocaleString()}`}>
              <WarningOutlined style={{ color: '#fa8c16', fontSize: '12px' }} />
            </Tooltip>
          )}
        </Space>
      )
    },
    {
      title: 'ML Confidence',
      key: 'ml_confidence',
      render: (record: DynamicRiskLimit) => (
        <div>
          {record.ml_confidence_score ? (
            <Progress
              type="circle"
              percent={Math.round(record.ml_confidence_score * 100)}
              size={40}
              strokeColor={record.ml_confidence_score > 0.8 ? '#52c41a' : '#faad14'}
            />
          ) : (
            <Text type="secondary">N/A</Text>
          )}
        </div>
      )
    },
    {
      title: 'Actions',
      key: 'actions',
      render: (record: DynamicRiskLimit) => (
        <Space>
          <Tooltip title="Edit limit">
            <Button
              size="small"
              icon={<EditOutlined />}
              onClick={() => {
                setSelectedLimit(record);
                setEditModalVisible(true);
              }}
            />
          </Tooltip>
          
          <Tooltip title="Test impact">
            <Button
              size="small"
              icon={<ExperimentOutlined />}
              onClick={() => testLimitImpact(record)}
            />
          </Tooltip>
          
          {record.auto_adjustment_enabled && (
            <Tooltip title="Manual adjustment">
              <Button
                size="small"
                icon={<SyncOutlined />}
                onClick={() => adjustLimit(record.id, 'manual_trigger')}
              />
            </Tooltip>
          )}
          
          <Popconfirm
            title="Delete this limit?"
            onConfirm={() => deleteLimit(record.id)}
          >
            <Tooltip title="Delete limit">
              <Button
                size="small"
                icon={<DeleteOutlined />}
                danger
              />
            </Tooltip>
          </Popconfirm>
        </Space>
      )
    }
  ];

  // Initial data fetch
  useEffect(() => {
    fetchData();
  }, [portfolioId]);

  // Use limits from hook if available
  const currentLimits = hookLimits.length > 0 ? hookLimits : limits;
  const activeCount = currentLimits.filter(l => l.active).length;
  const breachedCount = breachedLimits.length;
  const mlEnabledCount = currentLimits.filter(l => l.ml_prediction_enabled).length;

  const tabItems = [
    {
      key: 'limits',
      label: (
        <Space>
          <ShieldOutlined />
          Risk Limits
          <Badge count={activeCount} />
        </Space>
      ),
      children: (
        <div>
          {/* Limits Summary */}
          <Row gutter={16} style={{ marginBottom: 16 }}>
            <Col xs={24} sm={6}>
              <Card size="small">
                <Statistic
                  title="Active Limits"
                  value={activeCount}
                  prefix={<CheckCircleOutlined />}
                  valueStyle={{ color: '#52c41a' }}
                />
              </Card>
            </Col>
            <Col xs={24} sm={6}>
              <Card size="small">
                <Statistic
                  title="Breached Limits"
                  value={breachedCount}
                  prefix={<WarningOutlined />}
                  valueStyle={{ color: breachedCount > 0 ? '#ff4d4f' : '#52c41a' }}
                />
              </Card>
            </Col>
            <Col xs={24} sm={6}>
              <Card size="small">
                <Statistic
                  title="ML Enabled"
                  value={mlEnabledCount}
                  prefix={<RobotOutlined />}
                  valueStyle={{ color: '#1890ff' }}
                />
              </Card>
            </Col>
            <Col xs={24} sm={6}>
              <Card size="small">
                <Statistic
                  title="Risk Score"
                  value={riskScore || 0}
                  precision={1}
                  suffix="%"
                  prefix={<DashboardOutlined />}
                  valueStyle={{ 
                    color: (riskScore || 0) > 70 ? '#ff4d4f' : 
                           (riskScore || 0) > 40 ? '#faad14' : '#52c41a'
                  }}
                />
              </Card>
            </Col>
          </Row>

          {/* Create New Limit Form */}
          <Card 
            title={
              <Space>
                <PlusOutlined />
                Create New Risk Limit
              </Space>
            }
            size="small"
            style={{ marginBottom: 16 }}
          >
            <Form
              form={form}
              layout="vertical"
              onFinish={createLimit}
            >
              <Row gutter={16}>
                <Col xs={24} sm={12} md={6}>
                  <Form.Item
                    name="name"
                    label="Limit Name"
                    rules={[{ required: true, message: 'Please enter a name' }]}
                  >
                    <Input placeholder="e.g., Portfolio VaR Limit" />
                  </Form.Item>
                </Col>
                
                <Col xs={24} sm={12} md={6}>
                  <Form.Item
                    name="limitType"
                    label="Limit Type"
                    rules={[{ required: true, message: 'Please select a type' }]}
                  >
                    <Select placeholder="Select type">
                      {limitTypes.map(type => (
                        <Option key={type.id} value={type.category}>
                          <Space>
                            {getLimitTypeIcon(type.category)}
                            {type.name}
                          </Space>
                        </Option>
                      ))}
                    </Select>
                  </Form.Item>
                </Col>

                <Col xs={24} sm={12} md={4}>
                  <Form.Item
                    name="thresholdValue"
                    label="Threshold"
                    rules={[{ required: true, message: 'Please enter threshold' }]}
                  >
                    <InputNumber
                      placeholder="1000000"
                      style={{ width: '100%' }}
                      formatter={(value) => `${value}`.replace(/\B(?=(\d{3})+(?!\d))/g, ',')}
                    />
                  </Form.Item>
                </Col>

                <Col xs={24} sm={12} md={4}>
                  <Form.Item
                    name="warningThreshold"
                    label="Warning"
                    rules={[{ required: true, message: 'Please enter warning threshold' }]}
                  >
                    <InputNumber
                      placeholder="800000"
                      style={{ width: '100%' }}
                      formatter={(value) => `${value}`.replace(/\B(?=(\d{3})+(?!\d))/g, ',')}
                    />
                  </Form.Item>
                </Col>

                <Col xs={24} sm={12} md={4}>
                  <Form.Item
                    name="action"
                    label="Action"
                    rules={[{ required: true, message: 'Please select action' }]}
                  >
                    <Select>
                      <Option value="warn">Warn</Option>
                      <Option value="block">Block</Option>
                      <Option value="reduce">Reduce</Option>
                      <Option value="notify">Notify</Option>
                    </Select>
                  </Form.Item>
                </Col>
              </Row>

              <Row gutter={16}>
                <Col xs={24} sm={8}>
                  <Form.Item name="autoAdjustmentEnabled" valuePropName="checked">
                    <Switch /> Auto Adjustment
                  </Form.Item>
                </Col>
                <Col xs={24} sm={8}>
                  <Form.Item name="mlPredictionEnabled" valuePropName="checked">
                    <Switch /> ML Prediction
                  </Form.Item>
                </Col>
                <Col xs={24} sm={8}>
                  <Form.Item name="active" valuePropName="checked" initialValue={true}>
                    <Switch /> Active
                  </Form.Item>
                </Col>
              </Row>

              <Form.Item>
                <Button 
                  type="primary" 
                  htmlType="submit" 
                  loading={saving}
                  icon={<SaveOutlined />}
                >
                  Create Limit
                </Button>
              </Form.Item>
            </Form>
          </Card>

          {/* Limits Table */}
          <Card
            title={
              <Space>
                <SettingOutlined />
                Risk Limits Configuration
              </Space>
            }
            extra={
              <Button 
                icon={<ReloadOutlined />}
                onClick={fetchData}
                loading={loading}
                size="small"
              >
                Refresh
              </Button>
            }
          >
            <Table<DynamicRiskLimit>
              columns={limitsColumns}
              dataSource={currentLimits}
              rowKey="id"
              loading={loading}
              size="small"
              pagination={{
                pageSize: 10,
                showSizeChanger: true,
                showTotal: (total, range) => `${range[0]}-${range[1]} of ${total} limits`
              }}
            />
          </Card>
        </div>
      )
    },
    {
      key: 'configuration',
      label: (
        <Space>
          <SettingOutlined />
          Configuration
        </Space>
      ),
      children: (
        <div>
          {configuration && (
            <Collapse defaultActiveKey={['monitoring', 'models', 'notifications']}>
              <Panel
                header={
                  <Space>
                    <MonitorOutlined />
                    Monitoring Settings
                  </Space>
                }
                key="monitoring"
              >
                <Row gutter={16}>
                  <Col xs={24} sm={12}>
                    <Card size="small">
                      <Space direction="vertical" style={{ width: '100%' }}>
                        <div>
                          <Text strong>Real-time Monitoring</Text>
                          <Switch
                            checked={configuration.monitoring_enabled}
                            onChange={(enabled) => 
                              saveConfiguration({ monitoring_enabled: enabled })
                            }
                            style={{ float: 'right' }}
                          />
                        </div>
                        <div>
                          <Text>Update Frequency: {configuration.update_frequency_seconds}s</Text>
                          <Slider
                            min={1}
                            max={60}
                            value={configuration.update_frequency_seconds}
                            onChange={(value) => 
                              saveConfiguration({ update_frequency_seconds: value })
                            }
                            marks={{ 1: '1s', 5: '5s', 10: '10s', 30: '30s', 60: '60s' }}
                          />
                        </div>
                      </Space>
                    </Card>
                  </Col>
                  <Col xs={24} sm={12}>
                    <Card size="small">
                      <Space direction="vertical" style={{ width: '100%' }}>
                        <div>
                          <Text strong>ML Predictions</Text>
                          <Switch
                            checked={configuration.ml_predictions_enabled}
                            onChange={(enabled) => 
                              saveConfiguration({ ml_predictions_enabled: enabled })
                            }
                            style={{ float: 'right' }}
                          />
                        </div>
                        <div>
                          <Text strong>Auto Limit Adjustment</Text>
                          <Switch
                            checked={configuration.auto_limit_adjustment}
                            onChange={(enabled) => 
                              saveConfiguration({ auto_limit_adjustment: enabled })
                            }
                            style={{ float: 'right' }}
                          />
                        </div>
                      </Space>
                    </Card>
                  </Col>
                </Row>
              </Panel>

              <Panel
                header={
                  <Space>
                    <RobotOutlined />
                    Risk Models
                    <Badge count={configuration.risk_models.filter(m => m.enabled).length} />
                  </Space>
                }
                key="models"
              >
                <List
                  dataSource={configuration.risk_models}
                  renderItem={(model: ActiveRiskModel) => (
                    <List.Item
                      actions={[
                        <Switch
                          key="toggle"
                          checked={model.enabled}
                          onChange={(enabled) => {
                            const updatedModels = configuration.risk_models.map(m =>
                              m.model_id === model.model_id ? { ...m, enabled } : m
                            );
                            saveConfiguration({ risk_models: updatedModels });
                          }}
                        />
                      ]}
                    >
                      <List.Item.Meta
                        avatar={
                          <Avatar 
                            style={{ backgroundColor: model.enabled ? '#52c41a' : '#d9d9d9' }}
                          >
                            {model.model_type[0].toUpperCase()}
                          </Avatar>
                        }
                        title={model.model_name}
                        description={
                          <Space>
                            <Tag>{model.model_type.toUpperCase()}</Tag>
                            <Text type="secondary">
                              Confidence: {(model.confidence_threshold * 100).toFixed(0)}%
                            </Text>
                            <Text type="secondary">
                              Update: {model.update_frequency_minutes}min
                            </Text>
                          </Space>
                        }
                      />
                    </List.Item>
                  )}
                />
              </Panel>

              <Panel
                header={
                  <Space>
                    <BellOutlined />
                    Notification Preferences
                  </Space>
                }
                key="notifications"
              >
                <List
                  dataSource={configuration.notification_preferences}
                  renderItem={(pref: NotificationPreference) => (
                    <List.Item>
                      <Card size="small" style={{ width: '100%' }}>
                        <Row align="middle" justify="space-between">
                          <Col>
                            <Text strong>{pref.event_type.replace('_', ' ').toUpperCase()}</Text>
                          </Col>
                          <Col>
                            <Space>
                              {pref.channels.map(channel => (
                                <Tag key={channel} color="blue">{channel}</Tag>
                              ))}
                              <Tag color={pref.severity_threshold === 'critical' ? 'red' : 'orange'}>
                                {pref.severity_threshold.toUpperCase()}
                              </Tag>
                            </Space>
                          </Col>
                        </Row>
                      </Card>
                    </List.Item>
                  )}
                />
              </Panel>
            </Collapse>
          )}
        </div>
      )
    }
  ];

  return (
    <div className={className}>
      {/* Error Alert */}
      {error && (
        <Alert
          message="Configuration Error"
          description={error}
          type="error"
          showIcon
          closable
          onClose={() => setError(null)}
          style={{ marginBottom: 16 }}
        />
      )}

      {/* Main Configuration Tabs */}
      <Card>
        <Tabs
          activeKey={activeTab}
          onChange={setActiveTab}
          items={tabItems}
        />
      </Card>

      {/* Edit Limit Modal */}
      <Modal
        title={
          <Space>
            <EditOutlined />
            Edit Risk Limit
          </Space>
        }
        open={editModalVisible}
        onCancel={() => setEditModalVisible(false)}
        footer={null}
        width={800}
      >
        {selectedLimit && (
          <div>
            {/* Limit details and editing form would go here */}
            <Paragraph>
              Editing limit: <Text strong>{selectedLimit.name}</Text>
            </Paragraph>
            
            {selectedLimit.adjustment_history.length > 0 && (
              <div>
                <Title level={5}>Adjustment History</Title>
                <Timeline
                  items={selectedLimit.adjustment_history.slice(-5).map(adjustment => ({
                    children: (
                      <div>
                        <Text strong>{adjustment.adjustment_reason.replace('_', ' ').toUpperCase()}</Text>
                        <br />
                        <Text type="secondary">
                          {adjustment.old_value} → {adjustment.new_value}
                        </Text>
                        <br />
                        <Text type="secondary" style={{ fontSize: '12px' }}>
                          {new Date(adjustment.timestamp).toLocaleString()}
                          {adjustment.triggered_by && ` by ${adjustment.triggered_by}`}
                        </Text>
                      </div>
                    ),
                    color: adjustment.adjustment_reason === 'breach_recovery' ? 'red' : 'blue'
                  }))}
                />
              </div>
            )}
          </div>
        )}
      </Modal>
    </div>
  );
};

export default RiskLimitConfigPanel;