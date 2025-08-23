/**
 * Subscription Manager Component
 * Sprint 3: Advanced WebSocket Infrastructure
 * 
 * Manages WebSocket subscriptions with advanced filtering, rate limiting,
 * and subscription analytics. Integrates with Redis pub/sub backend.
 */

import React, { useState, useEffect, useCallback } from 'react';
import { 
  Card, 
  Table, 
  Button, 
  Modal, 
  Form, 
  Input, 
  Select, 
  Tag, 
  Space, 
  Typography, 
  Row, 
  Col, 
  Badge, 
  Tooltip,
  Switch,
  InputNumber,
  Alert,
  Statistic,
  Progress
} from 'antd';
import { 
  PlusOutlined, 
  DeleteOutlined, 
  EditOutlined,
  FilterOutlined,
  StopOutlined,
  PlayCircleOutlined,
  SettingOutlined,
  BarChartOutlined,
  WarningOutlined
} from '@ant-design/icons';
import { useSubscriptionManager } from '../../hooks/useSubscriptionManager';

const { Title, Text } = Typography;
const { Option } = Select;

interface Subscription {
  id: string;
  type: string;
  filters: Record<string, any>;
  isActive: boolean;
  messageCount: number;
  errorCount: number;
  createdAt: string;
  lastActivity: string;
  rateLimit?: number;
  queueSize?: number;
}

interface SubscriptionManagerProps {
  className?: string;
  showAdvancedFilters?: boolean;
  showRateControls?: boolean;
  showAnalytics?: boolean;
  maxSubscriptions?: number;
}

export const SubscriptionManager: React.FC<SubscriptionManagerProps> = ({
  className,
  showAdvancedFilters = true,
  showRateControls = true,
  showAnalytics = true,
  maxSubscriptions = 50
}) => {
  const {
    subscriptions,
    subscriptionStats,
    subscribe,
    unsubscribe,
    updateSubscription,
    pauseSubscription,
    resumeSubscription,
    getSubscriptionAnalytics,
    clearSubscriptionHistory
  } = useSubscriptionManager();

  const [isModalOpen, setIsModalOpen] = useState(false);
  const [editingSubscription, setEditingSubscription] = useState<Subscription | null>(null);
  const [form] = Form.useForm();
  const [selectedSubscriptions, setSelectedSubscriptions] = useState<string[]>([]);
  const [filterType, setFilterType] = useState<string>('all');

  // Subscription types available
  const subscriptionTypes = [
    { value: 'market_data', label: 'Market Data', color: 'blue' },
    { value: 'trade_updates', label: 'Trade Updates', color: 'green' },
    { value: 'risk_alerts', label: 'Risk Alerts', color: 'red' },
    { value: 'engine_status', label: 'Engine Status', color: 'purple' },
    { value: 'system_health', label: 'System Health', color: 'orange' },
    { value: 'performance_updates', label: 'Performance Updates', color: 'cyan' },
    { value: 'order_updates', label: 'Order Updates', color: 'magenta' },
    { value: 'position_updates', label: 'Position Updates', color: 'gold' },
  ];

  // Get subscription type color
  const getTypeColor = (type: string) => {
    return subscriptionTypes.find(t => t.value === type)?.color || 'default';
  };

  // Get subscription type label
  const getTypeLabel = (type: string) => {
    return subscriptionTypes.find(t => t.value === type)?.label || type;
  };

  // Filter subscriptions based on type filter
  const filteredSubscriptions = subscriptions.filter(sub => {
    if (filterType === 'all') return true;
    if (filterType === 'active') return sub.isActive;
    if (filterType === 'inactive') return !sub.isActive;
    return sub.type === filterType;
  });

  // Handle create/edit subscription
  const handleSubmit = async (values: any) => {
    try {
      const subscriptionData = {
        type: values.type,
        filters: {
          symbols: values.symbols ? values.symbols.split(',').map((s: string) => s.trim()) : [],
          portfolio_ids: values.portfolioIds ? values.portfolioIds.split(',').map((s: string) => s.trim()) : [],
          strategy_ids: values.strategyIds ? values.strategyIds.split(',').map((s: string) => s.trim()) : [],
          user_id: values.userId,
          severity: values.severity,
          min_price: values.minPrice,
          max_price: values.maxPrice,
        },
        rateLimit: values.rateLimit,
        queueSize: values.queueSize || 100,
      };

      if (editingSubscription) {
        await updateSubscription(editingSubscription.id, subscriptionData);
      } else {
        await subscribe(subscriptionData.type, subscriptionData.filters);
      }

      setIsModalOpen(false);
      setEditingSubscription(null);
      form.resetFields();
    } catch (error) {
      console.error('Failed to save subscription:', error);
    }
  };

  // Handle subscription deletion
  const handleDelete = async (subscriptionId: string) => {
    try {
      await unsubscribe(subscriptionId);
    } catch (error) {
      console.error('Failed to delete subscription:', error);
    }
  };

  // Handle bulk operations
  const handleBulkPause = async () => {
    for (const id of selectedSubscriptions) {
      await pauseSubscription(id);
    }
    setSelectedSubscriptions([]);
  };

  const handleBulkResume = async () => {
    for (const id of selectedSubscriptions) {
      await resumeSubscription(id);
    }
    setSelectedSubscriptions([]);
  };

  const handleBulkDelete = async () => {
    for (const id of selectedSubscriptions) {
      await handleDelete(id);
    }
    setSelectedSubscriptions([]);
  };

  // Open edit modal
  const handleEdit = (subscription: Subscription) => {
    setEditingSubscription(subscription);
    form.setFieldsValue({
      type: subscription.type,
      symbols: subscription.filters.symbols?.join(', ') || '',
      portfolioIds: subscription.filters.portfolio_ids?.join(', ') || '',
      strategyIds: subscription.filters.strategy_ids?.join(', ') || '',
      userId: subscription.filters.user_id || '',
      severity: subscription.filters.severity || '',
      minPrice: subscription.filters.min_price,
      maxPrice: subscription.filters.max_price,
      rateLimit: subscription.rateLimit,
      queueSize: subscription.queueSize,
    });
    setIsModalOpen(true);
  };

  // Table columns
  const columns = [
    {
      title: 'Type',
      dataIndex: 'type',
      key: 'type',
      render: (type: string) => (
        <Tag color={getTypeColor(type)}>
          {getTypeLabel(type)}
        </Tag>
      ),
    },
    {
      title: 'Status',
      dataIndex: 'isActive',
      key: 'status',
      render: (isActive: boolean) => (
        <Badge 
          status={isActive ? 'success' : 'default'} 
          text={isActive ? 'Active' : 'Paused'} 
        />
      ),
    },
    {
      title: 'Filters',
      dataIndex: 'filters',
      key: 'filters',
      render: (filters: Record<string, any>) => (
        <Space wrap size="small">
          {Object.entries(filters).map(([key, value]) => {
            if (Array.isArray(value) && value.length > 0) {
              return (
                <Tooltip key={key} title={`${key}: ${value.join(', ')}`}>
                  <Tag size="small">{key}: {value.length} items</Tag>
                </Tooltip>
              );
            } else if (value && !Array.isArray(value)) {
              return (
                <Tooltip key={key} title={`${key}: ${value}`}>
                  <Tag size="small">{key}</Tag>
                </Tooltip>
              );
            }
            return null;
          })}
        </Space>
      ),
    },
    {
      title: 'Messages',
      dataIndex: 'messageCount',
      key: 'messageCount',
      render: (count: number, record: Subscription) => (
        <Space direction="vertical" size="small">
          <Text>{count.toLocaleString()}</Text>
          {record.errorCount > 0 && (
            <Text type="danger">
              <WarningOutlined /> {record.errorCount} errors
            </Text>
          )}
        </Space>
      ),
    },
    {
      title: 'Rate Limit',
      dataIndex: 'rateLimit',
      key: 'rateLimit',
      render: (limit: number) => limit ? `${limit}/sec` : 'None',
    },
    {
      title: 'Last Activity',
      dataIndex: 'lastActivity',
      key: 'lastActivity',
      render: (date: string) => new Date(date).toLocaleTimeString(),
    },
    {
      title: 'Actions',
      key: 'actions',
      render: (_, record: Subscription) => (
        <Space size="small">
          <Tooltip title={record.isActive ? 'Pause' : 'Resume'}>
            <Button
              size="small"
              type="text"
              icon={record.isActive ? <StopOutlined /> : <PlayCircleOutlined />}
              onClick={() => record.isActive ? pauseSubscription(record.id) : resumeSubscription(record.id)}
            />
          </Tooltip>
          
          <Tooltip title="Edit">
            <Button
              size="small"
              type="text"
              icon={<EditOutlined />}
              onClick={() => handleEdit(record)}
            />
          </Tooltip>
          
          <Tooltip title="Delete">
            <Button
              size="small"
              type="text"
              danger
              icon={<DeleteOutlined />}
              onClick={() => handleDelete(record.id)}
            />
          </Tooltip>
        </Space>
      ),
    },
  ];

  // Row selection configuration
  const rowSelection = {
    selectedRowKeys: selectedSubscriptions,
    onChange: (keys: React.Key[]) => setSelectedSubscriptions(keys as string[]),
    getCheckboxProps: (record: Subscription) => ({
      disabled: false,
      name: record.id,
    }),
  };

  return (
    <div className={className}>
      <Card
        title={
          <Space>
            <SettingOutlined />
            <span>Subscription Manager</span>
            <Badge count={subscriptions.length} showZero color="blue" />
          </Space>
        }
        extra={
          <Space>
            <Button
              type="primary"
              icon={<PlusOutlined />}
              onClick={() => {
                setEditingSubscription(null);
                form.resetFields();
                setIsModalOpen(true);
              }}
              disabled={subscriptions.length >= maxSubscriptions}
            >
              New Subscription
            </Button>
          </Space>
        }
      >
        {/* Analytics Overview */}
        {showAnalytics && subscriptionStats && (
          <Row gutter={[16, 16]} style={{ marginBottom: 16 }}>
            <Col span={6}>
              <Statistic 
                title="Active Subscriptions" 
                value={subscriptionStats.activeCount} 
                valueStyle={{ color: '#3f8600' }}
              />
            </Col>
            <Col span={6}>
              <Statistic 
                title="Total Messages" 
                value={subscriptionStats.totalMessages} 
                formatter={(value) => `${Number(value).toLocaleString()}`}
              />
            </Col>
            <Col span={6}>
              <Statistic 
                title="Avg Latency" 
                value={subscriptionStats.averageLatency} 
                precision={0}
                suffix="ms"
              />
            </Col>
            <Col span={6}>
              <Statistic 
                title="Error Rate" 
                value={subscriptionStats.errorRate} 
                precision={2}
                suffix="%"
                valueStyle={{ color: subscriptionStats.errorRate > 5 ? '#cf1322' : '#3f8600' }}
              />
            </Col>
          </Row>
        )}

        {/* Subscription Limit Warning */}
        {subscriptions.length >= maxSubscriptions * 0.8 && (
          <Alert
            message="Subscription Limit Warning"
            description={`You have ${subscriptions.length} of ${maxSubscriptions} allowed subscriptions. Consider removing unused subscriptions.`}
            type="warning"
            showIcon
            style={{ marginBottom: 16 }}
          />
        )}

        {/* Filters and Bulk Actions */}
        <Row gutter={[16, 16]} style={{ marginBottom: 16 }}>
          <Col span={12}>
            <Space>
              <Text>Filter:</Text>
              <Select 
                value={filterType} 
                onChange={setFilterType}
                style={{ width: 150 }}
                size="small"
              >
                <Option value="all">All Subscriptions</Option>
                <Option value="active">Active Only</Option>
                <Option value="inactive">Inactive Only</Option>
                {subscriptionTypes.map(type => (
                  <Option key={type.value} value={type.value}>
                    {type.label}
                  </Option>
                ))}
              </Select>
            </Space>
          </Col>
          
          <Col span={12} style={{ textAlign: 'right' }}>
            <Space>
              {selectedSubscriptions.length > 0 && (
                <>
                  <Text>{selectedSubscriptions.length} selected</Text>
                  <Button 
                    size="small" 
                    icon={<StopOutlined />}
                    onClick={handleBulkPause}
                  >
                    Pause
                  </Button>
                  <Button 
                    size="small" 
                    icon={<PlayCircleOutlined />}
                    onClick={handleBulkResume}
                  >
                    Resume
                  </Button>
                  <Button 
                    size="small" 
                    danger
                    icon={<DeleteOutlined />}
                    onClick={handleBulkDelete}
                  >
                    Delete
                  </Button>
                </>
              )}
            </Space>
          </Col>
        </Row>

        {/* Subscriptions Table */}
        <Table
          dataSource={filteredSubscriptions}
          columns={columns}
          rowKey="id"
          size="small"
          pagination={{
            pageSize: 10,
            showSizeChanger: true,
            showQuickJumper: true,
            showTotal: (total, range) => 
              `${range[0]}-${range[1]} of ${total} subscriptions`,
          }}
          rowSelection={rowSelection}
        />

        {/* Create/Edit Subscription Modal */}
        <Modal
          title={editingSubscription ? 'Edit Subscription' : 'Create Subscription'}
          open={isModalOpen}
          onCancel={() => {
            setIsModalOpen(false);
            setEditingSubscription(null);
          }}
          footer={null}
          width={600}
        >
          <Form
            form={form}
            layout="vertical"
            onFinish={handleSubmit}
          >
            <Form.Item
              name="type"
              label="Subscription Type"
              rules={[{ required: true, message: 'Please select a subscription type' }]}
            >
              <Select placeholder="Select subscription type">
                {subscriptionTypes.map(type => (
                  <Option key={type.value} value={type.value}>
                    <Tag color={type.color} style={{ marginRight: 8 }}>
                      {type.label}
                    </Tag>
                  </Option>
                ))}
              </Select>
            </Form.Item>

            <Title level={5}>Filters</Title>
            
            <Row gutter={16}>
              <Col span={12}>
                <Form.Item
                  name="symbols"
                  label="Symbols"
                  tooltip="Comma-separated list of trading symbols (e.g., AAPL, GOOGL, TSLA)"
                >
                  <Input placeholder="AAPL, GOOGL, TSLA" />
                </Form.Item>
              </Col>
              
              <Col span={12}>
                <Form.Item
                  name="portfolioIds"
                  label="Portfolio IDs"
                  tooltip="Comma-separated list of portfolio identifiers"
                >
                  <Input placeholder="portfolio1, portfolio2" />
                </Form.Item>
              </Col>
            </Row>

            <Row gutter={16}>
              <Col span={12}>
                <Form.Item
                  name="strategyIds"
                  label="Strategy IDs"
                  tooltip="Comma-separated list of strategy identifiers"
                >
                  <Input placeholder="strategy1, strategy2" />
                </Form.Item>
              </Col>
              
              <Col span={12}>
                <Form.Item
                  name="userId"
                  label="User ID"
                  tooltip="Filter by specific user ID"
                >
                  <Input placeholder="user123" />
                </Form.Item>
              </Col>
            </Row>

            {showAdvancedFilters && (
              <>
                <Title level={5}>Advanced Filters</Title>
                
                <Row gutter={16}>
                  <Col span={8}>
                    <Form.Item
                      name="severity"
                      label="Alert Severity"
                      tooltip="For risk alerts and system alerts"
                    >
                      <Select placeholder="Select severity">
                        <Option value="low">Low</Option>
                        <Option value="medium">Medium</Option>
                        <Option value="high">High</Option>
                        <Option value="critical">Critical</Option>
                      </Select>
                    </Form.Item>
                  </Col>
                  
                  <Col span={8}>
                    <Form.Item
                      name="minPrice"
                      label="Min Price"
                      tooltip="Minimum price filter for market data"
                    >
                      <InputNumber 
                        placeholder="0.01" 
                        min={0} 
                        step={0.01}
                        style={{ width: '100%' }}
                      />
                    </Form.Item>
                  </Col>
                  
                  <Col span={8}>
                    <Form.Item
                      name="maxPrice"
                      label="Max Price"
                      tooltip="Maximum price filter for market data"
                    >
                      <InputNumber 
                        placeholder="1000" 
                        min={0} 
                        step={0.01}
                        style={{ width: '100%' }}
                      />
                    </Form.Item>
                  </Col>
                </Row>
              </>
            )}

            {showRateControls && (
              <>
                <Title level={5}>Rate Controls</Title>
                
                <Row gutter={16}>
                  <Col span={12}>
                    <Form.Item
                      name="rateLimit"
                      label="Rate Limit (msgs/sec)"
                      tooltip="Maximum messages per second (0 = no limit)"
                    >
                      <InputNumber 
                        placeholder="10" 
                        min={0} 
                        max={1000}
                        style={{ width: '100%' }}
                      />
                    </Form.Item>
                  </Col>
                  
                  <Col span={12}>
                    <Form.Item
                      name="queueSize"
                      label="Queue Size"
                      tooltip="Maximum queued messages"
                      initialValue={100}
                    >
                      <InputNumber 
                        placeholder="100" 
                        min={10} 
                        max={10000}
                        style={{ width: '100%' }}
                      />
                    </Form.Item>
                  </Col>
                </Row>
              </>
            )}

            <Form.Item style={{ marginBottom: 0, textAlign: 'right' }}>
              <Space>
                <Button onClick={() => setIsModalOpen(false)}>
                  Cancel
                </Button>
                <Button type="primary" htmlType="submit">
                  {editingSubscription ? 'Update' : 'Create'} Subscription
                </Button>
              </Space>
            </Form.Item>
          </Form>
        </Modal>
      </Card>
    </div>
  );
};

export default SubscriptionManager;