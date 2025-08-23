import React, { useState, useEffect, useCallback } from 'react';
import {
  Card,
  Row,
  Col,
  Statistic,
  Table,
  Button,
  Space,
  Tag,
  Progress,
  Alert,
  Typography,
  Select,
  Input,
  Badge,
  Tooltip,
  Modal,
  Tabs,
  List,
  Avatar,
  Spin,
  notification,
  Popconfirm
} from 'antd';
import {
  DeploymentUnitOutlined,
  PlayCircleOutlined,
  PauseCircleOutlined,
  StopOutlined,
  ReloadOutlined,
  EyeOutlined,
  DeleteOutlined,
  ExclamationCircleOutlined,
  CheckCircleOutlined,
  ClockCircleOutlined,
  ThunderboltOutlined,
  BarChartOutlined,
  ServerOutlined,
  MonitorOutlined,
  ControlOutlined
} from '@ant-design/icons';
import type { ColumnType } from 'antd/es/table';
import type {
  DeploymentOrchestratorProps,
  AdvancedDeploymentPipeline,
  DeploymentOrchestrator,
  ResourceUsageStats,
  OrchestratorMetrics
} from './types/deploymentTypes';
import AdvancedDeploymentPipeline from './AdvancedDeploymentPipeline';
import PipelineStatusMonitor from './PipelineStatusMonitor';

const { Title, Text, Paragraph } = Typography;
const { Option } = Select;
const { Search } = Input;
const { TabPane } = Tabs;

const API_BASE = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8001';

const DeploymentOrchestrator: React.FC<DeploymentOrchestratorProps> = ({
  maxConcurrentPipelines = 10,
  autoRefreshInterval = 30000,
  showMetrics = true,
  onPipelineSelect
}) => {
  const [orchestratorData, setOrchestratorData] = useState<DeploymentOrchestrator | null>(null);
  const [loading, setLoading] = useState(false);
  const [selectedPipeline, setSelectedPipeline] = useState<AdvancedDeploymentPipeline | null>(null);
  const [showCreateModal, setShowCreateModal] = useState(false);
  const [showDetailModal, setShowDetailModal] = useState(false);
  const [newStrategyId, setNewStrategyId] = useState('');
  const [filterStatus, setFilterStatus] = useState<string>('all');
  const [searchTerm, setSearchTerm] = useState('');
  const [autoRefresh, setAutoRefresh] = useState(true);
  const [viewMode, setViewMode] = useState<'grid' | 'table'>('table');

  const fetchOrchestratorData = useCallback(async () => {
    try {
      const response = await fetch(`${API_BASE}/api/v1/strategies/orchestrator/status`);
      if (!response.ok) throw new Error('Failed to fetch orchestrator data');
      
      const data: DeploymentOrchestrator = await response.json();
      setOrchestratorData(data);
    } catch (error) {
      console.error('Error fetching orchestrator data:', error);
      notification.error({
        message: 'Data Fetch Failed',
        description: 'Failed to load deployment orchestrator data'
      });
    }
  }, []);

  useEffect(() => {
    fetchOrchestratorData();
    
    let intervalId: NodeJS.Timeout;
    if (autoRefresh) {
      intervalId = setInterval(fetchOrchestratorData, autoRefreshInterval);
    }
    
    return () => {
      if (intervalId) clearInterval(intervalId);
    };
  }, [fetchOrchestratorData, autoRefresh, autoRefreshInterval]);

  const handlePipelineAction = async (pipelineId: string, action: 'start' | 'pause' | 'stop' | 'cancel') => {
    setLoading(true);
    try {
      const response = await fetch(`${API_BASE}/api/v1/strategies/pipeline/${pipelineId}/${action}`, {
        method: 'POST'
      });
      
      if (!response.ok) throw new Error(`Failed to ${action} pipeline`);
      
      notification.success({
        message: 'Action Completed',
        description: `Pipeline ${action} completed successfully`
      });
      
      await fetchOrchestratorData();
    } catch (error) {
      console.error(`Error ${action} pipeline:`, error);
      notification.error({
        message: 'Action Failed',
        description: `Failed to ${action} pipeline`
      });
    } finally {
      setLoading(false);
    }
  };

  const handleDeletePipeline = async (pipelineId: string) => {
    setLoading(true);
    try {
      const response = await fetch(`${API_BASE}/api/v1/strategies/pipeline/${pipelineId}`, {
        method: 'DELETE'
      });
      
      if (!response.ok) throw new Error('Failed to delete pipeline');
      
      notification.success({
        message: 'Pipeline Deleted',
        description: 'Pipeline deleted successfully'
      });
      
      await fetchOrchestratorData();
    } catch (error) {
      console.error('Error deleting pipeline:', error);
      notification.error({
        message: 'Delete Failed',
        description: 'Failed to delete pipeline'
      });
    } finally {
      setLoading(false);
    }
  };

  const getStatusColor = (status: string): string => {
    switch (status) {
      case 'running': return 'processing';
      case 'completed': return 'success';
      case 'failed': return 'error';
      case 'paused': return 'warning';
      case 'cancelled': return 'default';
      default: return 'default';
    }
  };

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'running': return <PlayCircleOutlined style={{ color: '#1890ff' }} />;
      case 'completed': return <CheckCircleOutlined style={{ color: '#52c41a' }} />;
      case 'failed': return <ExclamationCircleOutlined style={{ color: '#ff4d4f' }} />;
      case 'paused': return <PauseCircleOutlined style={{ color: '#faad14' }} />;
      case 'cancelled': return <StopOutlined style={{ color: '#8c8c8c' }} />;
      default: return <ClockCircleOutlined style={{ color: '#8c8c8c' }} />;
    }
  };

  const renderActionButtons = (pipeline: AdvancedDeploymentPipeline) => (
    <Space size="small">
      <Tooltip title="View Details">
        <Button
          size="small"
          icon={<EyeOutlined />}
          onClick={() => {
            setSelectedPipeline(pipeline);
            setShowDetailModal(true);
            onPipelineSelect?.(pipeline.pipeline_id);
          }}
        />
      </Tooltip>
      
      {pipeline.status === 'draft' && (
        <Tooltip title="Start Pipeline">
          <Button
            size="small"
            type="primary"
            icon={<PlayCircleOutlined />}
            onClick={() => handlePipelineAction(pipeline.pipeline_id, 'start')}
          />
        </Tooltip>
      )}
      
      {pipeline.status === 'running' && (
        <Tooltip title="Pause Pipeline">
          <Button
            size="small"
            icon={<PauseCircleOutlined />}
            onClick={() => handlePipelineAction(pipeline.pipeline_id, 'pause')}
          />
        </Tooltip>
      )}
      
      {pipeline.status === 'paused' && (
        <Tooltip title="Resume Pipeline">
          <Button
            size="small"
            type="primary"
            icon={<PlayCircleOutlined />}
            onClick={() => handlePipelineAction(pipeline.pipeline_id, 'start')}
          />
        </Tooltip>
      )}
      
      {(['running', 'paused'].includes(pipeline.status)) && (
        <Popconfirm
          title="Cancel Pipeline"
          description="Are you sure you want to cancel this pipeline?"
          onConfirm={() => handlePipelineAction(pipeline.pipeline_id, 'cancel')}
        >
          <Tooltip title="Cancel Pipeline">
            <Button
              size="small"
              danger
              icon={<StopOutlined />}
            />
          </Tooltip>
        </Popconfirm>
      )}
      
      {(['completed', 'failed', 'cancelled'].includes(pipeline.status)) && (
        <Popconfirm
          title="Delete Pipeline"
          description="Are you sure you want to delete this pipeline?"
          onConfirm={() => handleDeletePipeline(pipeline.pipeline_id)}
        >
          <Tooltip title="Delete Pipeline">
            <Button
              size="small"
              danger
              icon={<DeleteOutlined />}
            />
          </Tooltip>
        </Popconfirm>
      )}
    </Space>
  );

  const pipelineColumns: ColumnType<AdvancedDeploymentPipeline>[] = [
    {
      title: 'Pipeline ID',
      dataIndex: 'pipeline_id',
      key: 'pipeline_id',
      width: 150,
      ellipsis: true,
      render: (id: string) => (
        <Tooltip title={id}>
          <Text code>{id.substring(0, 8)}...</Text>
        </Tooltip>
      )
    },
    {
      title: 'Strategy',
      dataIndex: 'strategy_id',
      key: 'strategy_id',
      width: 120,
      ellipsis: true
    },
    {
      title: 'Version',
      dataIndex: 'version',
      key: 'version',
      width: 80,
      render: (version: string) => <Tag>{version}</Tag>
    },
    {
      title: 'Status',
      dataIndex: 'status',
      key: 'status',
      width: 120,
      render: (status: string) => (
        <Space>
          {getStatusIcon(status)}
          <Tag color={getStatusColor(status)}>{status.toUpperCase()}</Tag>
        </Space>
      ),
      filters: [
        { text: 'Running', value: 'running' },
        { text: 'Completed', value: 'completed' },
        { text: 'Failed', value: 'failed' },
        { text: 'Paused', value: 'paused' },
        { text: 'Draft', value: 'draft' }
      ],
      onFilter: (value, record) => record.status === value
    },
    {
      title: 'Progress',
      key: 'progress',
      width: 150,
      render: (_, pipeline: AdvancedDeploymentPipeline) => (
        <div>
          <Progress
            percent={pipeline.progress.overall_progress}
            size="small"
            status={pipeline.status === 'failed' ? 'exception' : 'normal'}
          />
          <Text type="secondary" style={{ fontSize: '12px' }}>
            {pipeline.progress.stages_completed}/{pipeline.progress.stages_total} stages
          </Text>
        </div>
      )
    },
    {
      title: 'Created',
      dataIndex: 'created_at',
      key: 'created_at',
      width: 120,
      render: (date: Date) => new Date(date).toLocaleDateString(),
      sorter: (a, b) => new Date(a.created_at).getTime() - new Date(b.created_at).getTime()
    },
    {
      title: 'Duration',
      key: 'duration',
      width: 100,
      render: (_, pipeline: AdvancedDeploymentPipeline) => (
        <Text>{pipeline.progress.elapsed_minutes}min</Text>
      )
    },
    {
      title: 'Actions',
      key: 'actions',
      width: 140,
      fixed: 'right',
      render: (_, pipeline: AdvancedDeploymentPipeline) => renderActionButtons(pipeline)
    }
  ];

  const getAllPipelines = (): AdvancedDeploymentPipeline[] => {
    if (!orchestratorData) return [];
    return [
      ...orchestratorData.active_pipelines,
      ...orchestratorData.completed_pipelines,
      ...orchestratorData.failed_pipelines
    ];
  };

  const getFilteredPipelines = (): AdvancedDeploymentPipeline[] => {
    const allPipelines = getAllPipelines();
    
    let filtered = allPipelines;
    
    if (filterStatus !== 'all') {
      filtered = filtered.filter(pipeline => pipeline.status === filterStatus);
    }
    
    if (searchTerm) {
      filtered = filtered.filter(pipeline =>
        pipeline.pipeline_id.toLowerCase().includes(searchTerm.toLowerCase()) ||
        pipeline.strategy_id.toLowerCase().includes(searchTerm.toLowerCase())
      );
    }
    
    return filtered.sort((a, b) => new Date(b.created_at).getTime() - new Date(a.created_at).getTime());
  };

  const renderMetricsCards = () => {
    if (!orchestratorData || !showMetrics) return null;

    const { resource_usage, performance_metrics } = orchestratorData;

    return (
      <Row gutter={16} className="mb-4">
        <Col span={4}>
          <Card size="small">
            <Statistic
              title="Active Pipelines"
              value={orchestratorData.active_pipelines.length}
              suffix={`/ ${maxConcurrentPipelines}`}
              prefix={<ThunderboltOutlined />}
              valueStyle={{ color: resource_usage.concurrent_pipelines >= maxConcurrentPipelines * 0.8 ? '#cf1322' : '#3f8600' }}
            />
          </Card>
        </Col>
        
        <Col span={4}>
          <Card size="small">
            <Statistic
              title="Success Rate"
              value={performance_metrics.success_rate * 100}
              precision={1}
              suffix="%"
              prefix={<BarChartOutlined />}
              valueStyle={{ color: performance_metrics.success_rate >= 0.9 ? '#3f8600' : '#cf1322' }}
            />
          </Card>
        </Col>
        
        <Col span={4}>
          <Card size="small">
            <Statistic
              title="Avg Duration"
              value={performance_metrics.average_deployment_time_minutes}
              precision={0}
              suffix="min"
              prefix={<ClockCircleOutlined />}
            />
          </Card>
        </Col>
        
        <Col span={4}>
          <Card size="small">
            <Statistic
              title="CPU Usage"
              value={resource_usage.cpu_usage_percent}
              precision={1}
              suffix="%"
              prefix={<ServerOutlined />}
              valueStyle={{ color: resource_usage.cpu_usage_percent >= 80 ? '#cf1322' : '#3f8600' }}
            />
          </Card>
        </Col>
        
        <Col span={4}>
          <Card size="small">
            <Statistic
              title="Memory Usage"
              value={resource_usage.memory_usage_percent}
              precision={1}
              suffix="%"
              prefix={<MonitorOutlined />}
              valueStyle={{ color: resource_usage.memory_usage_percent >= 80 ? '#cf1322' : '#3f8600' }}
            />
          </Card>
        </Col>
        
        <Col span={4}>
          <Card size="small">
            <Statistic
              title="System Uptime"
              value={performance_metrics.uptime_percent}
              precision={2}
              suffix="%"
              prefix={<CheckCircleOutlined />}
              valueStyle={{ color: performance_metrics.uptime_percent >= 99.9 ? '#3f8600' : '#cf1322' }}
            />
          </Card>
        </Col>
      </Row>
    );
  };

  const renderResourceUsage = () => {
    if (!orchestratorData?.resource_usage) return null;

    const { resource_usage } = orchestratorData;

    return (
      <Card title="Resource Usage" size="small" className="mb-4">
        <Row gutter={16}>
          <Col span={6}>
            <Progress
              type="circle"
              percent={resource_usage.cpu_usage_percent}
              format={() => `${resource_usage.cpu_usage_percent.toFixed(1)}%`}
              width={80}
              status={resource_usage.cpu_usage_percent >= 80 ? 'exception' : 'normal'}
            />
            <div className="text-center mt-2">CPU</div>
          </Col>
          <Col span={6}>
            <Progress
              type="circle"
              percent={resource_usage.memory_usage_percent}
              format={() => `${resource_usage.memory_usage_percent.toFixed(1)}%`}
              width={80}
              status={resource_usage.memory_usage_percent >= 80 ? 'exception' : 'normal'}
            />
            <div className="text-center mt-2">Memory</div>
          </Col>
          <Col span={6}>
            <Progress
              type="circle"
              percent={resource_usage.disk_usage_percent}
              format={() => `${resource_usage.disk_usage_percent.toFixed(1)}%`}
              width={80}
              status={resource_usage.disk_usage_percent >= 90 ? 'exception' : 'normal'}
            />
            <div className="text-center mt-2">Disk</div>
          </Col>
          <Col span={6}>
            <Progress
              type="circle"
              percent={(resource_usage.concurrent_pipelines / resource_usage.max_concurrent_pipelines) * 100}
              format={() => `${resource_usage.concurrent_pipelines}/${resource_usage.max_concurrent_pipelines}`}
              width={80}
              status={resource_usage.concurrent_pipelines >= resource_usage.max_concurrent_pipelines * 0.8 ? 'exception' : 'normal'}
            />
            <div className="text-center mt-2">Pipelines</div>
          </Col>
        </Row>
      </Card>
    );
  };

  if (!orchestratorData) {
    return (
      <div className="flex justify-center items-center h-64">
        <Spin size="large" />
      </div>
    );
  }

  return (
    <div className="deployment-orchestrator">
      <Card
        title={
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-2">
              <ControlOutlined />
              <span>Deployment Orchestrator</span>
              <Badge
                count={orchestratorData.active_pipelines.length}
                showZero
                style={{ backgroundColor: '#52c41a' }}
              />
            </div>
            <Space>
              <Button
                icon={<ReloadOutlined />}
                onClick={fetchOrchestratorData}
                loading={loading}
              >
                Refresh
              </Button>
              <Button
                type="primary"
                icon={<DeploymentUnitOutlined />}
                onClick={() => setShowCreateModal(true)}
              >
                Create Pipeline
              </Button>
            </Space>
          </div>
        }
      >
        {renderMetricsCards()}
        {renderResourceUsage()}

        <div className="mb-4">
          <Space>
            <Search
              placeholder="Search pipelines..."
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
              style={{ width: 250 }}
            />
            <Select
              value={filterStatus}
              onChange={setFilterStatus}
              style={{ width: 150 }}
            >
              <Option value="all">All Status</Option>
              <Option value="running">Running</Option>
              <Option value="completed">Completed</Option>
              <Option value="failed">Failed</Option>
              <Option value="paused">Paused</Option>
              <Option value="draft">Draft</Option>
            </Select>
          </Space>
        </div>

        <Table
          columns={pipelineColumns}
          dataSource={getFilteredPipelines()}
          rowKey="pipeline_id"
          size="small"
          scroll={{ x: 1200 }}
          pagination={{
            pageSize: 20,
            showSizeChanger: true,
            showQuickJumper: true,
            showTotal: (total, range) => `${range[0]}-${range[1]} of ${total} pipelines`
          }}
        />
      </Card>

      <Modal
        title="Create New Pipeline"
        open={showCreateModal}
        onCancel={() => setShowCreateModal(false)}
        footer={null}
        width={800}
        destroyOnClose
      >
        <div className="mb-4">
          <Input
            placeholder="Enter Strategy ID"
            value={newStrategyId}
            onChange={(e) => setNewStrategyId(e.target.value)}
            addonBefore="Strategy ID"
          />
        </div>
        {newStrategyId && (
          <AdvancedDeploymentPipeline
            strategyId={newStrategyId}
            onPipelineCreated={(pipelineId) => {
              setShowCreateModal(false);
              fetchOrchestratorData();
              notification.success({
                message: 'Pipeline Created',
                description: `Pipeline ${pipelineId} created successfully`
              });
            }}
            onClose={() => setShowCreateModal(false)}
          />
        )}
      </Modal>

      <Modal
        title={`Pipeline Details: ${selectedPipeline?.pipeline_id}`}
        open={showDetailModal}
        onCancel={() => setShowDetailModal(false)}
        footer={null}
        width={1200}
        destroyOnClose
      >
        {selectedPipeline && (
          <PipelineStatusMonitor
            pipelineId={selectedPipeline.pipeline_id}
            autoRefresh={true}
            showLogs={true}
          />
        )}
      </Modal>
    </div>
  );
};

export default DeploymentOrchestrator;