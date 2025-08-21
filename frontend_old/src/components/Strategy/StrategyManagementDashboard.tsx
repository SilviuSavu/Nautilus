import React, { useState, useEffect } from 'react';
import {
  Card,
  Row,
  Col,
  Typography,
  Space,
  Button,
  Table,
  Tag,
  Dropdown,
  Modal,
  Input,
  Select,
  Badge,
  Tooltip,
  notification,
  Tabs,
  Empty,
  Divider
} from 'antd';
import {
  PlusOutlined,
  SettingOutlined,
  SearchOutlined,
  FilterOutlined,
  ReloadOutlined,
  EyeOutlined,
  EditOutlined,
  PlayCircleOutlined,
  PauseCircleOutlined,
  StopOutlined,
  DeleteOutlined,
  BarChartOutlined,
  TrophyOutlined,
  ExclamationCircleOutlined
} from '@ant-design/icons';

import { EnhancedStrategyBuilder } from './EnhancedStrategyBuilder';
import { LifecycleControls } from './LifecycleControls';
import { MultiStrategyCoordinator } from './MultiStrategyCoordinator';
import AdvancedStrategyConfiguration from './AdvancedStrategyConfiguration';
import LiveStrategyMonitoring from './LiveStrategyMonitoring';
import StrategyPerformanceAnalysis from './StrategyPerformanceAnalysis';
import { StrategyConfig, StrategyInstance, StrategyState } from './types/strategyTypes';
import strategyService from './services/strategyService';

const { Title, Text } = Typography;
const { TabPane } = Tabs;
const { Search } = Input;
const { Option } = Select;

interface StrategyManagementDashboardProps {
  className?: string;
}

export const StrategyManagementDashboard: React.FC<StrategyManagementDashboardProps> = ({
  className
}) => {
  const [strategies, setStrategies] = useState<StrategyConfig[]>([]);
  const [instances, setInstances] = useState<Record<string, StrategyInstance>>({});
  const [loading, setLoading] = useState(false);
  const [activeTab, setActiveTab] = useState('overview');
  const [builderVisible, setBuilderVisible] = useState(false);
  const [selectedStrategy, setSelectedStrategy] = useState<StrategyConfig | null>(null);
  const [searchTerm, setSearchTerm] = useState('');
  const [statusFilter, setStatusFilter] = useState<string>('all');
  const [refreshInterval, setRefreshInterval] = useState<NodeJS.Timeout | null>(null);

  // Load strategies on mount
  useEffect(() => {
    loadStrategies();
    startAutoRefresh();
    return () => stopAutoRefresh();
  }, []);

  const startAutoRefresh = () => {
    const interval = setInterval(() => {
      loadStrategies();
      loadInstances();
    }, 10000); // Refresh every 10 seconds
    setRefreshInterval(interval);
  };

  const stopAutoRefresh = () => {
    if (refreshInterval) {
      clearInterval(refreshInterval);
      setRefreshInterval(null);
    }
  };

  const loadStrategies = async () => {
    try {
      setLoading(true);
      const response = await strategyService.listConfigurations();
      setStrategies(response || []);
    } catch (error: any) {
      console.error('Failed to load strategies:', error);
      notification.error({
        message: 'Failed to Load Strategies',
        description: error.message || 'Could not load strategy configurations',
        duration: 4
      });
    } finally {
      setLoading(false);
    }
  };

  const loadInstances = async () => {
    try {
      const response = await strategyService.getActiveInstances();
      const instanceMap: Record<string, StrategyInstance> = {};
      response.instances?.forEach((instance: StrategyInstance) => {
        instanceMap[instance.config_id] = instance;
      });
      setInstances(instanceMap);
    } catch (error: any) {
      console.error('Failed to load instances:', error);
    }
  };

  const handleStrategyCreated = (strategy: StrategyConfig) => {
    setStrategies(prev => [...prev, strategy]);
    setBuilderVisible(false);
    notification.success({
      message: 'Strategy Created',
      description: `Strategy "${strategy.name}" has been created successfully.`,
      duration: 3
    });
  };

  const handleInstanceUpdate = (strategyId: string) => (instance: StrategyInstance) => {
    setInstances(prev => ({
      ...prev,
      [strategyId]: instance
    }));
  };

  const getStateColor = (state: StrategyState): string => {
    switch (state) {
      case 'running': return 'success';
      case 'paused': return 'warning';
      case 'stopped': return 'default';
      case 'error': return 'error';
      case 'initializing': return 'processing';
      case 'stopping': return 'warning';
      case 'completed': return 'purple';
      default: return 'default';
    }
  };

  const getStatusBadge = (strategy: StrategyConfig) => {
    const instance = instances[strategy.id];
    if (!instance) {
      return <Tag color="default">Not Deployed</Tag>;
    }
    return <Tag color={getStateColor(instance.state)}>{instance.state.toUpperCase()}</Tag>;
  };

  const getPerformanceIndicator = (strategy: StrategyConfig) => {
    const instance = instances[strategy.id];
    if (!instance || !instance.performance_metrics) {
      return <Text type="secondary">-</Text>;
    }

    const pnl = Number(instance.performance_metrics.total_pnl);
    const color = pnl >= 0 ? '#3f8600' : '#cf1322';
    return (
      <Text style={{ color }}>
        ${pnl.toFixed(2)}
      </Text>
    );
  };

  const filteredStrategies = strategies.filter(strategy => {
    const matchesSearch = strategy.name.toLowerCase().includes(searchTerm.toLowerCase()) ||
                         strategy.template_id.toLowerCase().includes(searchTerm.toLowerCase());
    
    if (!matchesSearch) return false;

    if (statusFilter === 'all') return true;
    
    const instance = instances[strategy.id];
    if (statusFilter === 'deployed') return !!instance;
    if (statusFilter === 'running') return instance?.state === 'running';
    if (statusFilter === 'stopped') return !instance || instance.state === 'stopped';
    
    return true;
  });

  const columns = [
    {
      title: 'Strategy Name',
      dataIndex: 'name',
      key: 'name',
      render: (name: string, strategy: StrategyConfig) => (
        <div>
          <Text strong>{name}</Text>
          <br />
          <Text type="secondary" style={{ fontSize: 12 }}>
            {strategy.template_id}
          </Text>
        </div>
      )
    },
    {
      title: 'Status',
      key: 'status',
      render: (_, strategy: StrategyConfig) => getStatusBadge(strategy)
    },
    {
      title: 'Performance',
      key: 'performance',
      render: (_, strategy: StrategyConfig) => getPerformanceIndicator(strategy)
    },
    {
      title: 'Created',
      dataIndex: 'created_at',
      key: 'created_at',
      render: (date: string) => new Date(date).toLocaleDateString()
    },
    {
      title: 'Actions',
      key: 'actions',
      render: (_, strategy: StrategyConfig) => {
        const instance = instances[strategy.id];
        return (
          <Space>
            <Tooltip title="View Details">
              <Button
                type="text"
                icon={<EyeOutlined />}
                onClick={() => {
                  setSelectedStrategy(strategy);
                  setActiveTab('details');
                }}
              />
            </Tooltip>
            
            {instance ? (
              <Tooltip title="Manage Lifecycle">
                <Button
                  type="text"
                  icon={<SettingOutlined />}
                  onClick={() => {
                    setSelectedStrategy(strategy);
                    setActiveTab('lifecycle');
                  }}
                />
              </Tooltip>
            ) : (
              <Tooltip title="Deploy Strategy">
                <Button
                  type="text"
                  icon={<PlayCircleOutlined />}
                  onClick={() => {
                    setSelectedStrategy(strategy);
                    setActiveTab('lifecycle');
                  }}
                />
              </Tooltip>
            )}

            <Dropdown
              menu={{
                items: [
                  {
                    key: 'edit',
                    label: 'Edit Configuration',
                    icon: <EditOutlined />
                  },
                  {
                    key: 'clone',
                    label: 'Clone Strategy',
                    icon: <PlusOutlined />
                  },
                  {
                    type: 'divider'
                  },
                  {
                    key: 'delete',
                    label: 'Delete Strategy',
                    icon: <DeleteOutlined />,
                    danger: true,
                    onClick: () => handleDeleteStrategy(strategy)
                  }
                ]
              }}
            >
              <Button type="text" icon={<SettingOutlined />} />
            </Dropdown>
          </Space>
        );
      }
    }
  ];

  const handleDeleteStrategy = (strategy: StrategyConfig) => {
    const instance = instances[strategy.id];
    
    if (instance && ['running', 'paused'].includes(instance.state)) {
      notification.error({
        message: 'Cannot Delete Running Strategy',
        description: 'Please stop the strategy before deleting it.',
        duration: 4
      });
      return;
    }

    Modal.confirm({
      title: 'Delete Strategy',
      content: `Are you sure you want to delete "${strategy.name}"? This action cannot be undone.`,
      icon: <ExclamationCircleOutlined />,
      okText: 'Delete',
      okType: 'danger',
      cancelText: 'Cancel',
      onOk: async () => {
        try {
          await strategyService.deleteConfiguration(strategy.id);
          setStrategies(prev => prev.filter(s => s.id !== strategy.id));
          notification.success({
            message: 'Strategy Deleted',
            description: `"${strategy.name}" has been deleted successfully.`,
            duration: 3
          });
        } catch (error: any) {
          notification.error({
            message: 'Delete Failed',
            description: error.message || 'Failed to delete strategy',
            duration: 4
          });
        }
      }
    });
  };

  const renderOverview = () => (
    <div>
      <Row gutter={[16, 16]} style={{ marginBottom: 24 }}>
        <Col xs={24} sm={12} lg={6}>
          <Card>
            <div style={{ textAlign: 'center' }}>
              <Title level={2} style={{ margin: '0 0 8px 0' }}>
                {strategies.length}
              </Title>
              <Text type="secondary">Total Strategies</Text>
            </div>
          </Card>
        </Col>
        
        <Col xs={24} sm={12} lg={6}>
          <Card>
            <div style={{ textAlign: 'center' }}>
              <Title level={2} style={{ margin: '0 0 8px 0', color: '#52c41a' }}>
                {Object.values(instances).filter(i => i.state === 'running').length}
              </Title>
              <Text type="secondary">Running</Text>
            </div>
          </Card>
        </Col>
        
        <Col xs={24} sm={12} lg={6}>
          <Card>
            <div style={{ textAlign: 'center' }}>
              <Title level={2} style={{ margin: '0 0 8px 0', color: '#fa8c16' }}>
                {Object.values(instances).filter(i => i.state === 'paused').length}
              </Title>
              <Text type="secondary">Paused</Text>
            </div>
          </Card>
        </Col>
        
        <Col xs={24} sm={12} lg={6}>
          <Card>
            <div style={{ textAlign: 'center' }}>
              <Title level={2} style={{ margin: '0 0 8px 0', color: '#ff4d4f' }}>
                {Object.values(instances).filter(i => i.state === 'error').length}
              </Title>
              <Text type="secondary">Errors</Text>
            </div>
          </Card>
        </Col>
      </Row>

      <Card
        title="Strategy List"
        extra={
          <Space>
            <Search
              placeholder="Search strategies..."
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
              style={{ width: 200 }}
            />
            
            <Select
              value={statusFilter}
              onChange={setStatusFilter}
              style={{ width: 120 }}
            >
              <Option value="all">All Status</Option>
              <Option value="deployed">Deployed</Option>
              <Option value="running">Running</Option>
              <Option value="stopped">Stopped</Option>
            </Select>
            
            <Button
              icon={<ReloadOutlined />}
              onClick={loadStrategies}
              loading={loading}
            >
              Refresh
            </Button>
            
            <Button
              type="primary"
              icon={<PlusOutlined />}
              onClick={() => setBuilderVisible(true)}
            >
              New Strategy
            </Button>
          </Space>
        }
      >
        <Table
          columns={columns}
          dataSource={filteredStrategies}
          rowKey="id"
          loading={loading}
          locale={{
            emptyText: (
              <Empty
                description="No strategies found"
                image={Empty.PRESENTED_IMAGE_SIMPLE}
              >
                <Button
                  type="primary"
                  icon={<PlusOutlined />}
                  onClick={() => setBuilderVisible(true)}
                >
                  Create Your First Strategy
                </Button>
              </Empty>
            )
          }}
        />
      </Card>
    </div>
  );

  const renderDetails = () => {
    if (!selectedStrategy) {
      return (
        <Empty
          description="Select a strategy to view details"
          image={Empty.PRESENTED_IMAGE_SIMPLE}
        />
      );
    }

    return (
      <Row gutter={[16, 16]}>
        <Col xs={24} lg={16}>
          <Card title="Strategy Configuration">
            <Space direction="vertical" size="middle" style={{ width: '100%' }}>
              <div>
                <Text strong>Name:</Text> {selectedStrategy.name}
              </div>
              <div>
                <Text strong>Template:</Text> {selectedStrategy.template_id}
              </div>
              <div>
                <Text strong>Status:</Text> {getStatusBadge(selectedStrategy)}
              </div>
              <div>
                <Text strong>Created:</Text> {new Date(selectedStrategy.created_at).toLocaleString()}
              </div>
              <div>
                <Text strong>Updated:</Text> {new Date(selectedStrategy.updated_at).toLocaleString()}
              </div>
              
              <Divider />
              
              <div>
                <Text strong>Parameters:</Text>
                <div style={{ marginTop: 8, marginLeft: 16 }}>
                  {Object.entries(selectedStrategy.parameters).map(([key, value]) => (
                    <div key={key} style={{ marginBottom: 4 }}>
                      <Text code>{key}:</Text> <Text>{String(value)}</Text>
                    </div>
                  ))}
                </div>
              </div>
            </Space>
          </Card>
        </Col>
        
        <Col xs={24} lg={8}>
          {instances[selectedStrategy.id] && (
            <Card title="Performance Metrics">
              <Space direction="vertical" size="middle" style={{ width: '100%' }}>
                {Object.entries(instances[selectedStrategy.id].performance_metrics).map(([key, value]) => (
                  <div key={key}>
                    <Text type="secondary">{key.replace(/_/g, ' ')}:</Text>
                    <br />
                    <Text strong>{String(value)}</Text>
                  </div>
                ))}
              </Space>
            </Card>
          )}
        </Col>
      </Row>
    );
  };

  const renderLifecycle = () => {
    if (!selectedStrategy) {
      return (
        <Empty
          description="Select a strategy to manage lifecycle"
          image={Empty.PRESENTED_IMAGE_SIMPLE}
        />
      );
    }

    return (
      <LifecycleControls
        strategy={selectedStrategy}
        instance={instances[selectedStrategy.id]}
        onInstanceUpdate={handleInstanceUpdate(selectedStrategy.id)}
        onStrategyUpdate={(updated) => {
          setStrategies(prev => prev.map(s => s.id === updated.id ? updated : s));
        }}
      />
    );
  };

  const renderCoordination = () => (
    <MultiStrategyCoordinator
      strategies={strategies}
      instances={instances}
      onInstanceUpdate={(strategyId, instance) => {
        setInstances(prev => ({
          ...prev,
          [strategyId]: instance
        }));
      }}
    />
  );

  return (
    <div className={`strategy-management-dashboard ${className || ''}`}>
      <Card>
        <div style={{ marginBottom: 24 }}>
          <Title level={2}>Strategy Management Dashboard</Title>
          <Text type="secondary">
            Create, deploy, and manage your trading strategies
          </Text>
        </div>

        <Tabs activeKey={activeTab} onChange={setActiveTab}>
          <TabPane tab="Overview" key="overview">
            {renderOverview()}
          </TabPane>
          
          <TabPane tab="Strategy Details" key="details">
            {renderDetails()}
          </TabPane>
          
          <TabPane tab="Lifecycle Management" key="lifecycle">
            {renderLifecycle()}
          </TabPane>
          
          <TabPane tab="Multi-Strategy Coordination" key="coordination">
            {renderCoordination()}
          </TabPane>

          <TabPane tab="Advanced Configuration" key="advanced-config">
            <AdvancedStrategyConfiguration />
          </TabPane>

          <TabPane tab="Live Monitoring" key="live-monitoring">
            <LiveStrategyMonitoring />
          </TabPane>

          <TabPane tab="Performance Analysis" key="performance-analysis">
            <StrategyPerformanceAnalysis />
          </TabPane>
        </Tabs>
      </Card>

      {/* Strategy Builder Modal */}
      <Modal
        title="Create New Strategy"
        open={builderVisible}
        onCancel={() => setBuilderVisible(false)}
        width={1200}
        footer={null}
        destroyOnClose
      >
        <EnhancedStrategyBuilder
          onStrategyCreated={handleStrategyCreated}
        />
      </Modal>
    </div>
  );
};