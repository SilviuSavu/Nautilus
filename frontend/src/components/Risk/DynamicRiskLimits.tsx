import React, { useState, useEffect } from 'react';
import {
  Card,
  Row,
  Col,
  Table,
  Tag,
  Space,
  Typography,
  Progress,
  Button,
  Modal,
  Form,
  Input,
  InputNumber,
  Select,
  Switch,
  Alert,
  Badge,
  Tooltip,
  List,
  Statistic,
  Divider,
  Timeline,
  Popconfirm
} from 'antd';
import {
  SafetyCertificateOutlined,
  WarningOutlined,
  AlertOutlined,
  SettingOutlined,
  PlusOutlined,
  EditOutlined,
  DeleteOutlined,
  PauseCircleOutlined,
  PlayCircleOutlined,
  CheckCircleOutlined,
  CloseCircleOutlined,
  ReloadOutlined,
  BellOutlined,
  FireOutlined,
  ThunderboltOutlined,
  RobotOutlined,
  SyncOutlined,
  LineChartOutlined,
  DashboardOutlined,
  MonitorOutlined,
  ExperimentOutlined
} from '@ant-design/icons';
import { useMessageBus } from '../../hooks/useMessageBus';
import { useEngineWebSocket } from '../../hooks/useEngineWebSocket';
import type {
  DynamicRiskLimit,
  RiskBreach,
  RiskBreachAction,
  RealTimeRiskMetrics
} from '../../types/sprint3';

const { Text, Title } = Typography;
const { Option } = Select;

interface DynamicRiskLimitsProps {
  portfolioId: string;
  showBreachHistory?: boolean;
  autoUpdate?: boolean;
  compactMode?: boolean;
}

const DynamicRiskLimits: React.FC<DynamicRiskLimitsProps> = ({
  portfolioId,
  showBreachHistory = true,
  autoUpdate = true,
  compactMode = false
}) => {
  const [limits, setLimits] = useState<DynamicRiskLimit[]>([]);
  const [breaches, setBreaches] = useState<RiskBreach[]>([]);
  const [riskMetrics, setRiskMetrics] = useState<RealTimeRiskMetrics | null>(null);
  const [modalVisible, setModalVisible] = useState(false);
  const [editingLimit, setEditingLimit] = useState<DynamicRiskLimit | null>(null);
  const [form] = Form.useForm();

  const messageBus = useMessageBus();
  const engineWs = useEngineWebSocket();

  // Initialize mock data
  useEffect(() => {
    const mockLimits: DynamicRiskLimit[] = [
      {
        id: 'pos-limit-1',
        name: 'Maximum Position Size',
        type: 'position_size',
        value: 250000,
        threshold: 500000,
        warningThreshold: 400000,
        enabled: true,
        autoAdjust: true,
        lastUpdated: new Date().toISOString(),
        breachCount: 0
      },
      {
        id: 'exposure-limit-1',
        name: 'Total Portfolio Exposure',
        type: 'exposure',
        value: 1200000,
        threshold: 2000000,
        warningThreshold: 1500000,
        enabled: true,
        autoAdjust: true,
        lastUpdated: new Date().toISOString(),
        breachCount: 1
      },
      {
        id: 'drawdown-limit-1',
        name: 'Maximum Drawdown',
        type: 'drawdown',
        value: 5.2,
        threshold: 10.0,
        warningThreshold: 7.5,
        enabled: true,
        autoAdjust: false,
        lastUpdated: new Date().toISOString(),
        breachCount: 0
      },
      {
        id: 'var-limit-1',
        name: 'Value at Risk (95%)',
        type: 'var',
        value: 45000,
        threshold: 75000,
        warningThreshold: 60000,
        enabled: true,
        autoAdjust: true,
        lastUpdated: new Date().toISOString(),
        breachCount: 2
      },
      {
        id: 'concentration-limit-1',
        name: 'Single Asset Concentration',
        type: 'custom',
        value: 15.5,
        threshold: 25.0,
        warningThreshold: 20.0,
        enabled: true,
        autoAdjust: false,
        lastUpdated: new Date().toISOString(),
        breachCount: 0
      },
      {
        id: 'leverage-limit-1',
        name: 'Portfolio Leverage',
        type: 'custom',
        value: 2.1,
        threshold: 3.0,
        warningThreshold: 2.5,
        enabled: false,
        autoAdjust: true,
        lastUpdated: new Date(Date.now() - 3600000).toISOString(),
        breachCount: 0
      }
    ];

    const mockBreaches: RiskBreach[] = [
      {
        id: 'breach-1',
        limitId: 'exposure-limit-1',
        limitName: 'Total Portfolio Exposure',
        severity: 'warning',
        actualValue: 1520000,
        thresholdValue: 1500000,
        timestamp: new Date(Date.now() - 1800000).toISOString(), // 30 min ago
        resolved: true,
        actions: [
          {
            id: 'action-1',
            type: 'alert',
            status: 'completed',
            details: 'Portfolio manager notified',
            timestamp: new Date(Date.now() - 1800000).toISOString()
          }
        ]
      },
      {
        id: 'breach-2',
        limitId: 'var-limit-1',
        limitName: 'Value at Risk (95%)',
        severity: 'critical',
        actualValue: 78000,
        thresholdValue: 75000,
        timestamp: new Date(Date.now() - 600000).toISOString(), // 10 min ago
        resolved: false,
        actions: [
          {
            id: 'action-2',
            type: 'alert',
            status: 'completed',
            details: 'Risk team alerted',
            timestamp: new Date(Date.now() - 600000).toISOString()
          },
          {
            id: 'action-3',
            type: 'position_reduce',
            status: 'executing',
            details: 'Reducing high-risk positions by 10%',
            timestamp: new Date(Date.now() - 300000).toISOString()
          }
        ]
      }
    ];

    const mockMetrics: RealTimeRiskMetrics = {
      var: 45000 + (Math.random() - 0.5) * 10000,
      drawdown: 5.2 + (Math.random() - 0.5) * 2,
      exposure: 1200000 + (Math.random() - 0.5) * 200000,
      concentration: 15.5 + (Math.random() - 0.5) * 5,
      correlation: 0.65 + (Math.random() - 0.5) * 0.3,
      leverage: 2.1 + (Math.random() - 0.5) * 0.5,
      liquidity: 0.85 + (Math.random() - 0.5) * 0.2,
      timestamp: new Date().toISOString()
    };

    setLimits(mockLimits);
    setBreaches(mockBreaches);
    setRiskMetrics(mockMetrics);
  }, []);

  // Auto-update risk metrics
  useEffect(() => {
    if (!autoUpdate) return;

    const interval = setInterval(() => {
      setRiskMetrics(prev => prev ? {
        ...prev,
        var: Math.max(0, prev.var + (Math.random() - 0.5) * 2000),
        drawdown: Math.max(0, prev.drawdown + (Math.random() - 0.5) * 0.5),
        exposure: Math.max(0, prev.exposure + (Math.random() - 0.5) * 50000),
        concentration: Math.max(0, Math.min(100, prev.concentration + (Math.random() - 0.5) * 2)),
        correlation: Math.max(-1, Math.min(1, prev.correlation + (Math.random() - 0.5) * 0.1)),
        leverage: Math.max(0, prev.leverage + (Math.random() - 0.5) * 0.1),
        liquidity: Math.max(0, Math.min(1, prev.liquidity + (Math.random() - 0.5) * 0.05)),
        timestamp: new Date().toISOString()
      } : null);

      // Update limit values based on current metrics
      setLimits(prev => prev.map(limit => {
        const newValue = riskMetrics ? (
          limit.type === 'var' ? riskMetrics.var :
          limit.type === 'drawdown' ? riskMetrics.drawdown :
          limit.type === 'exposure' ? riskMetrics.exposure :
          limit.value
        ) : limit.value;

        return {
          ...limit,
          value: newValue,
          lastUpdated: new Date().toISOString()
        };
      }));
    }, 5000);

    return () => clearInterval(interval);
  }, [autoUpdate, riskMetrics]);

  // Get limit status
  const getLimitStatus = (limit: DynamicRiskLimit) => {
    const utilizationPercent = (limit.value / limit.threshold) * 100;
    const warningPercent = limit.warningThreshold ? (limit.warningThreshold / limit.threshold) * 100 : 75;
    
    if (!limit.enabled) return { status: 'disabled', color: '#999', percent: 0 };
    if (utilizationPercent >= 100) return { status: 'breach', color: '#ff4d4f', percent: utilizationPercent };
    if (utilizationPercent >= warningPercent) return { status: 'warning', color: '#faad14', percent: utilizationPercent };
    return { status: 'normal', color: '#52c41a', percent: utilizationPercent };
  };

  // Handle limit operations
  const handleCreateLimit = () => {
    setEditingLimit(null);
    setModalVisible(true);
    form.resetFields();
  };

  const handleEditLimit = (limit: DynamicRiskLimit) => {
    setEditingLimit(limit);
    setModalVisible(true);
    form.setFieldsValue(limit);
  };

  const handleToggleLimit = (limitId: string) => {
    setLimits(prev => prev.map(limit => 
      limit.id === limitId ? { ...limit, enabled: !limit.enabled } : limit
    ));
  };

  const handleDeleteLimit = (limitId: string) => {
    setLimits(prev => prev.filter(limit => limit.id !== limitId));
  };

  const handleModalOk = () => {
    form.validateFields().then(values => {
      if (editingLimit) {
        // Update existing limit
        setLimits(prev => prev.map(limit => 
          limit.id === editingLimit.id 
            ? { ...limit, ...values, lastUpdated: new Date().toISOString() }
            : limit
        ));
      } else {
        // Create new limit
        const newLimit: DynamicRiskLimit = {
          id: `limit-${Date.now()}`,
          ...values,
          lastUpdated: new Date().toISOString(),
          breachCount: 0
        };
        setLimits(prev => [...prev, newLimit]);
      }
      setModalVisible(false);
    });
  };

  // Table columns for limits
  const limitColumns = [
    {
      title: 'Limit',
      key: 'limit',
      render: (record: DynamicRiskLimit) => (
        <Space>
          <div style={{ color: getLimitStatus(record).color }}>
            {record.enabled ? <SafetyCertificateOutlined /> : <PauseCircleOutlined />}
          </div>
          <div>
            <Text strong>{record.name}</Text>
            <br />
            <Tag size="small" color={
              record.type === 'position_size' ? 'blue' :
              record.type === 'exposure' ? 'green' :
              record.type === 'drawdown' ? 'orange' :
              record.type === 'var' ? 'red' : 'purple'
            }>
              {record.type.replace('_', ' ').toUpperCase()}
            </Tag>
          </div>
        </Space>
      )
    },
    {
      title: 'Current / Threshold',
      key: 'values',
      render: (record: DynamicRiskLimit) => (
        <div>
          <Text strong style={{ fontSize: '14px' }}>
            {record.type === 'drawdown' || record.type === 'custom' && record.name.includes('Concentration')
              ? `${record.value.toFixed(1)}%`
              : record.type === 'custom' && record.name.includes('Leverage')
              ? `${record.value.toFixed(1)}x`
              : `$${record.value.toLocaleString()}`
            }
          </Text>
          <Text type="secondary"> / </Text>
          <Text>
            {record.type === 'drawdown' || record.type === 'custom' && record.name.includes('Concentration')
              ? `${record.threshold.toFixed(1)}%`
              : record.type === 'custom' && record.name.includes('Leverage')
              ? `${record.threshold.toFixed(1)}x`
              : `$${record.threshold.toLocaleString()}`
            }
          </Text>
          {record.warningThreshold && (
            <>
              <br />
              <Text type="warning" style={{ fontSize: '12px' }}>
                Warning: {record.type === 'drawdown' || record.type === 'custom' && record.name.includes('Concentration')
                  ? `${record.warningThreshold.toFixed(1)}%`
                  : record.type === 'custom' && record.name.includes('Leverage')
                  ? `${record.warningThreshold.toFixed(1)}x`
                  : `$${record.warningThreshold.toLocaleString()}`
                }
              </Text>
            </>
          )}
        </div>
      )
    },
    {
      title: 'Utilization',
      key: 'utilization',
      render: (record: DynamicRiskLimit) => {
        const status = getLimitStatus(record);
        return (
          <div style={{ width: '120px' }}>
            <Progress
              percent={Math.min(100, status.percent)}
              status={
                status.status === 'breach' ? 'exception' :
                status.status === 'warning' ? 'active' :
                status.status === 'disabled' ? 'normal' : 'success'
              }
              size="small"
              format={(percent) => `${Math.round(percent || 0)}%`}
            />
            <Text type="secondary" style={{ fontSize: '11px' }}>
              {record.breachCount > 0 && `${record.breachCount} breaches`}
            </Text>
          </div>
        );
      }
    },
    {
      title: 'Status',
      key: 'status',
      render: (record: DynamicRiskLimit) => {
        const status = getLimitStatus(record);
        return (
          <Space direction="vertical" size="small">
            <Badge
              status={
                status.status === 'breach' ? 'error' :
                status.status === 'warning' ? 'warning' :
                status.status === 'disabled' ? 'default' : 'success'
              }
              text={
                status.status === 'breach' ? 'BREACH' :
                status.status === 'warning' ? 'WARNING' :
                status.status === 'disabled' ? 'DISABLED' : 'NORMAL'
              }
            />
            <div>
              {record.autoAdjust && (
                <Tag size="small" color="cyan">Auto-adjust</Tag>
              )}
            </div>
          </Space>
        );
      }
    },
    {
      title: 'Actions',
      key: 'actions',
      render: (record: DynamicRiskLimit) => (
        <Space>
          <Tooltip title={record.enabled ? 'Disable limit' : 'Enable limit'}>
            <Button
              size="small"
              icon={record.enabled ? <PauseCircleOutlined /> : <PlayCircleOutlined />}
              onClick={() => handleToggleLimit(record.id)}
            />
          </Tooltip>
          <Button
            size="small"
            icon={<EditOutlined />}
            onClick={() => handleEditLimit(record)}
          />
          <Popconfirm
            title="Delete this limit?"
            onConfirm={() => handleDeleteLimit(record.id)}
          >
            <Button
              size="small"
              danger
              icon={<DeleteOutlined />}
            />
          </Popconfirm>
        </Space>
      )
    }
  ];

  // Active breaches for alert
  const activeBreaches = breaches.filter(b => !b.resolved);
  const criticalBreaches = activeBreaches.filter(b => b.severity === 'critical');

  if (compactMode) {
    const activeLimits = limits.filter(l => l.enabled);
    const breachedLimits = activeLimits.filter(l => getLimitStatus(l).status === 'breach');
    const warningLimits = activeLimits.filter(l => getLimitStatus(l).status === 'warning');

    return (
      <Card
        title={
          <Space>
            <SafetyCertificateOutlined />
            Risk Limits
            {breachedLimits.length > 0 && <Badge count={breachedLimits.length} />}
          </Space>
        }
        size="small"
      >
        <Row gutter={[8, 8]}>
          <Col span={6}>
            <Statistic
              title="Active"
              value={activeLimits.length}
              prefix={<SafetyCertificateOutlined />}
              valueStyle={{ fontSize: '16px', color: '#52c41a' }}
            />
          </Col>
          <Col span={6}>
            <Statistic
              title="Breached"
              value={breachedLimits.length}
              prefix={<WarningOutlined />}
              valueStyle={{ fontSize: '16px', color: '#ff4d4f' }}
            />
          </Col>
          <Col span={6}>
            <Statistic
              title="Warning"
              value={warningLimits.length}
              prefix={<WarningOutlined />}
              valueStyle={{ fontSize: '16px', color: '#faad14' }}
            />
          </Col>
          <Col span={6}>
            <Statistic
              title="Auto-adjust"
              value={limits.filter(l => l.autoAdjust).length}
              prefix={<SettingOutlined />}
              valueStyle={{ fontSize: '16px' }}
            />
          </Col>
        </Row>
      </Card>
    );
  }

  return (
    <div style={{ width: '100%' }}>
      {/* Critical Breach Alert */}
      {criticalBreaches.length > 0 && (
        <Alert
          message={`${criticalBreaches.length} Critical Risk Breach${criticalBreaches.length > 1 ? 'es' : ''} Active`}
          description={
            <div>
              {criticalBreaches.map(breach => (
                <div key={breach.id} style={{ marginBottom: '8px' }}>
                  <Text strong>{breach.limitName}: </Text>
                  <Text>
                    {breach.actualValue.toLocaleString()} exceeds limit of {breach.thresholdValue.toLocaleString()}
                  </Text>
                  <br />
                  <Text type="secondary" style={{ fontSize: '12px' }}>
                    Actions: {breach.actions.filter(a => a.status !== 'completed').length} in progress
                  </Text>
                </div>
              ))}
            </div>
          }
          type="error"
          showIcon
          style={{ marginBottom: 16 }}
          action={
            <Button size="small" type="primary" danger>
              Emergency Actions
            </Button>
          }
        />
      )}

      {/* Control Panel */}
      <Card size="small" style={{ marginBottom: 16 }}>
        <Row gutter={[16, 8]} align="middle">
          <Col>
            <Space>
              <SafetyCertificateOutlined style={{ color: '#52c41a' }} />
              <Text strong>Dynamic Risk Limits</Text>
              {autoUpdate && <Badge status="processing" text="Live" />}
            </Space>
          </Col>
          <Col>
            <Space>
              <Text type="secondary">Portfolio:</Text>
              <Text>{portfolioId}</Text>
            </Space>
          </Col>
          <Col flex="auto" />
          <Col>
            <Space>
              <Button
                size="small"
                type="primary"
                icon={<PlusOutlined />}
                onClick={handleCreateLimit}
              >
                Add Limit
              </Button>
              <Button size="small" icon={<ReloadOutlined />}>
                Refresh
              </Button>
              <Button size="small" icon={<SettingOutlined />}>
                Settings
              </Button>
            </Space>
          </Col>
        </Row>
      </Card>

      <Row gutter={[16, 16]}>
        {/* Limits Table */}
        <Col xs={24} lg={16}>
          <Card
            title={
              <Space>
                <SafetyCertificateOutlined />
                Risk Limits
                <Badge count={limits.filter(l => l.enabled).length} color="blue" />
              </Space>
            }
            size="small"
          >
            <Table
              columns={limitColumns}
              dataSource={limits}
              rowKey="id"
              size="small"
              pagination={false}
              scroll={{ x: 'max-content' }}
            />
          </Card>
        </Col>

        {/* Real-time Metrics */}
        <Col xs={24} lg={8}>
          <Card
            title={
              <Space>
                <FireOutlined />
                Real-time Risk Metrics
              </Space>
            }
            size="small"
          >
            {riskMetrics && (
              <List
                size="small"
                dataSource={[
                  { key: 'var', label: 'Value at Risk', value: `$${riskMetrics.var.toFixed(0)}`, color: riskMetrics.var > 60000 ? '#ff4d4f' : '#52c41a' },
                  { key: 'drawdown', label: 'Max Drawdown', value: `${riskMetrics.drawdown.toFixed(1)}%`, color: riskMetrics.drawdown > 7 ? '#ff4d4f' : '#52c41a' },
                  { key: 'exposure', label: 'Total Exposure', value: `$${riskMetrics.exposure.toFixed(0)}`, color: riskMetrics.exposure > 1500000 ? '#faad14' : '#52c41a' },
                  { key: 'concentration', label: 'Concentration', value: `${riskMetrics.concentration.toFixed(1)}%`, color: riskMetrics.concentration > 20 ? '#faad14' : '#52c41a' },
                  { key: 'correlation', label: 'Avg Correlation', value: riskMetrics.correlation.toFixed(2), color: Math.abs(riskMetrics.correlation) > 0.8 ? '#faad14' : '#52c41a' },
                  { key: 'leverage', label: 'Leverage', value: `${riskMetrics.leverage.toFixed(1)}x`, color: riskMetrics.leverage > 2.5 ? '#faad14' : '#52c41a' }
                ]}
                renderItem={(item) => (
                  <List.Item style={{ padding: '8px 12px' }}>
                    <List.Item.Meta
                      title={
                        <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                          <Text style={{ fontSize: '12px' }}>{item.label}</Text>
                          <Text strong style={{ fontSize: '14px', color: item.color }}>
                            {item.value}
                          </Text>
                        </div>
                      }
                    />
                  </List.Item>
                )}
              />
            )}
          </Card>
        </Col>
      </Row>

      {/* Recent Breaches */}
      {showBreachHistory && (
        <Card
          title={
            <Space>
              <BellOutlined />
              Recent Breaches
              {activeBreaches.length > 0 && <Badge count={activeBreaches.length} />}
            </Space>
          }
          size="small"
          style={{ marginTop: 16 }}
        >
          <Timeline
            size="small"
            items={breaches.slice(0, 5).map(breach => ({
              color: breach.severity === 'critical' ? 'red' : 
                    breach.severity === 'warning' ? 'orange' : 'blue',
              children: (
                <div key={breach.id}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                    <Text strong>{breach.limitName}</Text>
                    <Space>
                      <Tag color={breach.severity === 'critical' ? 'error' : 'warning'}>
                        {breach.severity}
                      </Tag>
                      {breach.resolved ? (
                        <Badge status="success" text="Resolved" />
                      ) : (
                        <Badge status="error" text="Active" />
                      )}
                    </Space>
                  </div>
                  <div>
                    <Text>
                      Value: {breach.actualValue.toLocaleString()} (Threshold: {breach.thresholdValue.toLocaleString()})
                    </Text>
                  </div>
                  <div>
                    <Text type="secondary" style={{ fontSize: '12px' }}>
                      {new Date(breach.timestamp).toLocaleString()} • 
                      {breach.actions.length} action{breach.actions.length !== 1 ? 's' : ''} taken
                    </Text>
                  </div>
                </div>
              )
            }))}
          />
        </Card>
      )}

      {/* Limit Configuration Modal */}
      <Modal
        title={editingLimit ? 'Edit Risk Limit' : 'Create Risk Limit'}
        open={modalVisible}
        onOk={handleModalOk}
        onCancel={() => setModalVisible(false)}
        width={600}
      >
        <Form
          form={form}
          layout="vertical"
          initialValues={{
            type: 'position_size',
            enabled: true,
            autoAdjust: false
          }}
        >
          <Row gutter={16}>
            <Col span={12}>
              <Form.Item name="name" label="Limit Name" rules={[{ required: true }]}>
                <Input placeholder="Enter limit name" />
              </Form.Item>
            </Col>
            <Col span={12}>
              <Form.Item name="type" label="Limit Type" rules={[{ required: true }]}>
                <Select>
                  <Option value="position_size">Position Size</Option>
                  <Option value="exposure">Total Exposure</Option>
                  <Option value="drawdown">Drawdown</Option>
                  <Option value="var">Value at Risk</Option>
                  <Option value="custom">Custom</Option>
                </Select>
              </Form.Item>
            </Col>
          </Row>
          
          <Row gutter={16}>
            <Col span={8}>
              <Form.Item name="threshold" label="Threshold Value" rules={[{ required: true }]}>
                <InputNumber
                  style={{ width: '100%' }}
                  min={0}
                  placeholder="Threshold"
                />
              </Form.Item>
            </Col>
            <Col span={8}>
              <Form.Item name="warningThreshold" label="Warning Threshold">
                <InputNumber
                  style={{ width: '100%' }}
                  min={0}
                  placeholder="Warning level"
                />
              </Form.Item>
            </Col>
            <Col span={8}>
              <Form.Item name="value" label="Current Value">
                <InputNumber
                  style={{ width: '100%' }}
                  min={0}
                  placeholder="Current value"
                />
              </Form.Item>
            </Col>
          </Row>
          
          <Row gutter={16}>
            <Col span={12}>
              <Form.Item name="enabled" valuePropName="checked">
                <Switch checkedChildren="Enabled" unCheckedChildren="Disabled" />
              </Form.Item>
            </Col>
            <Col span={12}>
              <Form.Item name="autoAdjust" valuePropName="checked">
                <Switch checkedChildren="Auto-adjust" unCheckedChildren="Manual" />
              </Form.Item>
            </Col>
          </Row>
        </Form>
      </Modal>
    </div>
  );
};

export default DynamicRiskLimits;