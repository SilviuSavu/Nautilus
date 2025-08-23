/**
 * AnalyticsAggregator Component - Sprint 3 Integration
 * Data aggregation and compression interface for analytics data management
 */

import React, { useState, useEffect, useMemo, useCallback } from 'react';
import { 
  Card, Row, Col, Table, Button, Space, Typography, Select, DatePicker, 
  Progress, Alert, Modal, Form, Input, InputNumber, Checkbox, Tag, Tooltip
} from 'antd';
import { 
  DatabaseOutlined,
  PlayCircleOutlined,
  StopOutlined,
  DownloadOutlined,
  UploadOutlined,
  CompressOutlined,
  SettingOutlined,
  ClockCircleOutlined,
  CheckCircleOutlined,
  WarningOutlined,
  InfoCircleOutlined
} from '@ant-design/icons';
import dayjs from 'dayjs';

const { Title, Text } = Typography;
const { Option } = Select;
const { RangePicker } = DatePicker;

interface AnalyticsAggregatorProps {
  className?: string;
  height?: number;
  autoRefresh?: boolean;
}

interface AggregationJob {
  job_id: string;
  job_name: string;
  data_type: 'performance' | 'risk' | 'execution' | 'strategy' | 'all';
  interval: 'minute' | 'hour' | 'day' | 'week' | 'month';
  source_table: string;
  target_table: string;
  start_date: string;
  end_date: string;
  status: 'pending' | 'running' | 'completed' | 'failed' | 'cancelled';
  progress: number;
  records_processed: number;
  total_records: number;
  compression_ratio?: number;
  created_at: string;
  started_at?: string;
  completed_at?: string;
  error_message?: string;
  config: AggregationConfig;
}

interface AggregationConfig {
  aggregate_functions: string[];
  group_by_fields: string[];
  filter_conditions?: Record<string, any>;
  compression_level: 'low' | 'medium' | 'high';
  retain_raw_data: boolean;
  enable_indexing: boolean;
  partition_strategy?: 'daily' | 'weekly' | 'monthly';
}

interface AggregatedData {
  data_type: string;
  interval: string;
  date_range: [string, string];
  record_count: number;
  file_size_mb: number;
  compression_ratio: number;
  last_updated: string;
  available_metrics: string[];
  download_url?: string;
}

const AnalyticsAggregator: React.FC<AnalyticsAggregatorProps> = ({
  className,
  height = 800,
  autoRefresh = true,
}) => {
  // State
  const [activeTab, setActiveTab] = useState<'jobs' | 'data' | 'create'>('jobs');
  const [jobs, setJobs] = useState<AggregationJob[]>([]);
  const [aggregatedData, setAggregatedData] = useState<AggregatedData[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [showCreateModal, setShowCreateModal] = useState(false);
  const [runningJobs, setRunningJobs] = useState<Set<string>>(new Set());
  const [selectedJob, setSelectedJob] = useState<AggregationJob | null>(null);
  const [autoRefreshInterval, setAutoRefreshInterval] = useState<NodeJS.Timeout | null>(null);
  
  // Form for creating new aggregation jobs
  const [form] = Form.useForm();
  
  // API base URL
  const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8001';
  
  // Fetch aggregation jobs
  const fetchJobs = useCallback(async () => {
    try {
      setIsLoading(true);
      const response = await fetch(`${API_BASE_URL}/api/v1/sprint3/aggregation/jobs`);
      
      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
      }
      
      const jobsData = await response.json();
      setJobs(jobsData.jobs || []);
      setError(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to fetch jobs');
      console.error('Failed to fetch aggregation jobs:', err);
    } finally {
      setIsLoading(false);
    }
  }, [API_BASE_URL]);
  
  // Fetch aggregated data
  const fetchAggregatedData = useCallback(async () => {
    try {
      const response = await fetch(`${API_BASE_URL}/api/v1/sprint3/aggregation/data-summary`);
      
      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
      }
      
      const data = await response.json();
      setAggregatedData(data.aggregated_data || []);
    } catch (err) {
      console.error('Failed to fetch aggregated data:', err);
    }
  }, [API_BASE_URL]);
  
  // Create new aggregation job
  const createAggregationJob = useCallback(async (values: any) => {
    try {
      setIsLoading(true);
      const response = await fetch(`${API_BASE_URL}/api/v1/sprint3/aggregation/create-job`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          job_name: values.job_name,
          data_type: values.data_type,
          interval: values.interval,
          start_date: values.date_range[0].format('YYYY-MM-DD'),
          end_date: values.date_range[1].format('YYYY-MM-DD'),
          config: {
            aggregate_functions: values.aggregate_functions,
            group_by_fields: values.group_by_fields,
            compression_level: values.compression_level,
            retain_raw_data: values.retain_raw_data,
            enable_indexing: values.enable_indexing,
            partition_strategy: values.partition_strategy,
          },
        }),
      });
      
      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
      }
      
      const result = await response.json();
      
      // Refresh jobs list
      await fetchJobs();
      
      setShowCreateModal(false);
      form.resetFields();
      
      return result;
    } catch (err) {
      throw err;
    } finally {
      setIsLoading(false);
    }
  }, [API_BASE_URL, fetchJobs, form]);
  
  // Run aggregation job
  const runJob = useCallback(async (jobId: string) => {
    try {
      setRunningJobs(prev => new Set([...prev, jobId]));
      
      const response = await fetch(`${API_BASE_URL}/api/v1/sprint3/aggregation/run/${jobId}`, {
        method: 'POST',
      });
      
      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
      }
      
      // Refresh jobs to update status
      await fetchJobs();
    } catch (err) {
      console.error('Failed to run job:', err);
    } finally {
      setRunningJobs(prev => {
        const newSet = new Set(prev);
        newSet.delete(jobId);
        return newSet;
      });
    }
  }, [API_BASE_URL, fetchJobs]);
  
  // Cancel aggregation job
  const cancelJob = useCallback(async (jobId: string) => {
    try {
      const response = await fetch(`${API_BASE_URL}/api/v1/sprint3/aggregation/cancel/${jobId}`, {
        method: 'POST',
      });
      
      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
      }
      
      await fetchJobs();
    } catch (err) {
      console.error('Failed to cancel job:', err);
    }
  }, [API_BASE_URL, fetchJobs]);
  
  // Download aggregated data
  const downloadData = useCallback(async (dataType: string, interval: string) => {
    try {
      const response = await fetch(
        `${API_BASE_URL}/api/v1/sprint3/aggregation/data/${dataType}/${interval}`,
        {
          method: 'GET',
        }
      );
      
      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
      }
      
      // Create download
      const blob = await response.blob();
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `${dataType}_${interval}_${dayjs().format('YYYYMMDD')}.json`;
      a.click();
      URL.revokeObjectURL(url);
    } catch (err) {
      console.error('Failed to download data:', err);
    }
  }, [API_BASE_URL]);
  
  // Auto-refresh setup
  useEffect(() => {
    if (autoRefresh) {
      const interval = setInterval(() => {
        fetchJobs();
        fetchAggregatedData();
      }, 10000); // Refresh every 10 seconds
      
      setAutoRefreshInterval(interval);
      
      return () => {
        if (interval) {
          clearInterval(interval);
        }
      };
    }
  }, [autoRefresh, fetchJobs, fetchAggregatedData]);
  
  // Initial data fetch
  useEffect(() => {
    fetchJobs();
    fetchAggregatedData();
  }, [fetchJobs, fetchAggregatedData]);
  
  // Job status statistics
  const jobStats = useMemo(() => {
    const stats = {
      total: jobs.length,
      pending: 0,
      running: 0,
      completed: 0,
      failed: 0,
    };
    
    jobs.forEach(job => {
      stats[job.status]++;
    });
    
    return stats;
  }, [jobs]);
  
  // Data summary statistics
  const dataStats = useMemo(() => {
    return {
      total_datasets: aggregatedData.length,
      total_records: aggregatedData.reduce((sum, data) => sum + data.record_count, 0),
      total_size_mb: aggregatedData.reduce((sum, data) => sum + data.file_size_mb, 0),
      avg_compression: aggregatedData.length > 0 
        ? aggregatedData.reduce((sum, data) => sum + data.compression_ratio, 0) / aggregatedData.length 
        : 0,
    };
  }, [aggregatedData]);
  
  // Table columns for jobs
  const jobColumns = [
    {
      title: 'Job Name',
      dataIndex: 'job_name',
      key: 'job_name',
      width: 200,
    },
    {
      title: 'Data Type',
      dataIndex: 'data_type',
      key: 'data_type',
      render: (type: string) => (
        <Tag color="blue">{type.toUpperCase()}</Tag>
      ),
      width: 100,
    },
    {
      title: 'Interval',
      dataIndex: 'interval',
      key: 'interval',
      render: (interval: string) => (
        <Tag>{interval.toUpperCase()}</Tag>
      ),
      width: 80,
    },
    {
      title: 'Status',
      dataIndex: 'status',
      key: 'status',
      render: (status: string) => {
        const colors = {
          pending: 'default',
          running: 'processing',
          completed: 'success',
          failed: 'error',
          cancelled: 'warning',
        };
        
        const icons = {
          pending: <ClockCircleOutlined />,
          running: <PlayCircleOutlined />,
          completed: <CheckCircleOutlined />,
          failed: <WarningOutlined />,
          cancelled: <StopOutlined />,
        };
        
        return (
          <Tag color={colors[status as keyof typeof colors]} icon={icons[status as keyof typeof icons]}>
            {status.toUpperCase()}
          </Tag>
        );
      },
      width: 120,
    },
    {
      title: 'Progress',
      dataIndex: 'progress',
      key: 'progress',
      render: (progress: number, record: AggregationJob) => (
        <div style={{ width: 100 }}>
          <Progress 
            percent={progress} 
            size="small" 
            status={record.status === 'failed' ? 'exception' : undefined}
            format={percent => `${percent}%`}
          />
          {record.records_processed > 0 && (
            <Text type="secondary" style={{ fontSize: '11px' }}>
              {record.records_processed.toLocaleString()} / {record.total_records.toLocaleString()}
            </Text>
          )}
        </div>
      ),
      width: 120,
    },
    {
      title: 'Date Range',
      key: 'date_range',
      render: (record: AggregationJob) => (
        <div>
          <Text style={{ fontSize: '12px' }}>
            {dayjs(record.start_date).format('MMM DD')} - {dayjs(record.end_date).format('MMM DD, YYYY')}
          </Text>
        </div>
      ),
      width: 120,
    },
    {
      title: 'Compression',
      dataIndex: 'compression_ratio',
      key: 'compression_ratio',
      render: (ratio: number) => ratio ? `${ratio.toFixed(1)}x` : '-',
      width: 100,
    },
    {
      title: 'Created',
      dataIndex: 'created_at',
      key: 'created_at',
      render: (date: string) => dayjs(date).format('MMM DD, HH:mm'),
      width: 100,
    },
    {
      title: 'Actions',
      key: 'actions',
      render: (record: AggregationJob) => (
        <Space size="small">
          {record.status === 'pending' && (
            <Button
              size="small"
              type="primary"
              icon={<PlayCircleOutlined />}
              onClick={() => runJob(record.job_id)}
              loading={runningJobs.has(record.job_id)}
            >
              Run
            </Button>
          )}
          {record.status === 'running' && (
            <Button
              size="small"
              danger
              icon={<StopOutlined />}
              onClick={() => cancelJob(record.job_id)}
            >
              Cancel
            </Button>
          )}
          <Button
            size="small"
            icon={<InfoCircleOutlined />}
            onClick={() => setSelectedJob(record)}
          >
            Details
          </Button>
        </Space>
      ),
      width: 150,
      fixed: 'right' as const,
    },
  ];
  
  // Table columns for aggregated data
  const dataColumns = [
    {
      title: 'Data Type',
      dataIndex: 'data_type',
      key: 'data_type',
      render: (type: string) => <Tag color="blue">{type.toUpperCase()}</Tag>,
    },
    {
      title: 'Interval',
      dataIndex: 'interval',
      key: 'interval',
      render: (interval: string) => <Tag>{interval.toUpperCase()}</Tag>,
    },
    {
      title: 'Date Range',
      key: 'date_range',
      render: (record: AggregatedData) => (
        <Text style={{ fontSize: '12px' }}>
          {dayjs(record.date_range[0]).format('MMM DD')} - {dayjs(record.date_range[1]).format('MMM DD, YYYY')}
        </Text>
      ),
    },
    {
      title: 'Records',
      dataIndex: 'record_count',
      key: 'record_count',
      render: (count: number) => count.toLocaleString(),
    },
    {
      title: 'Size',
      dataIndex: 'file_size_mb',
      key: 'file_size_mb',
      render: (size: number) => `${size.toFixed(1)} MB`,
    },
    {
      title: 'Compression',
      dataIndex: 'compression_ratio',
      key: 'compression_ratio',
      render: (ratio: number) => `${ratio.toFixed(1)}x`,
    },
    {
      title: 'Last Updated',
      dataIndex: 'last_updated',
      key: 'last_updated',
      render: (date: string) => dayjs(date).format('MMM DD, HH:mm'),
    },
    {
      title: 'Actions',
      key: 'actions',
      render: (record: AggregatedData) => (
        <Button
          size="small"
          icon={<DownloadOutlined />}
          onClick={() => downloadData(record.data_type, record.interval)}
        >
          Download
        </Button>
      ),
    },
  ];
  
  return (
    <div className={className} style={{ height }}>
      <Row gutter={[16, 16]}>
        {/* Header */}
        <Col span={24}>
          <Card size="small">
            <Row justify="space-between" align="middle">
              <Col>
                <Space>
                  <DatabaseOutlined style={{ color: '#1890ff', fontSize: '18px' }} />
                  <Title level={4} style={{ margin: 0 }}>
                    Analytics Aggregator
                  </Title>
                  <Tag color={autoRefresh ? 'green' : 'default'}>
                    Auto-refresh {autoRefresh ? 'ON' : 'OFF'}
                  </Tag>
                </Space>
              </Col>
              <Col>
                <Space>
                  <Button
                    type="primary"
                    icon={<CompressOutlined />}
                    onClick={() => setShowCreateModal(true)}
                  >
                    Create Job
                  </Button>
                  <Button 
                    icon={<SettingOutlined />}
                    onClick={() => {/* Settings modal */}}
                  >
                    Settings
                  </Button>
                </Space>
              </Col>
            </Row>
          </Card>
        </Col>
        
        {/* Error Alert */}
        {error && (
          <Col span={24}>
            <Alert
              message="Error"
              description={error}
              type="error"
              showIcon
              closable
              onClose={() => setError(null)}
            />
          </Col>
        )}
        
        {/* Statistics Cards */}
        <Col span={24}>
          <Row gutter={[16, 16]}>
            <Col xs={12} sm={6}>
              <Card size="small" style={{ textAlign: 'center' }}>
                <div>
                  <Text type="secondary">Total Jobs</Text>
                  <br />
                  <Title level={3} style={{ margin: 0, color: '#1890ff' }}>
                    {jobStats.total}
                  </Title>
                </div>
              </Card>
            </Col>
            <Col xs={12} sm={6}>
              <Card size="small" style={{ textAlign: 'center' }}>
                <div>
                  <Text type="secondary">Running</Text>
                  <br />
                  <Title level={3} style={{ margin: 0, color: '#52c41a' }}>
                    {jobStats.running}
                  </Title>
                </div>
              </Card>
            </Col>
            <Col xs={12} sm={6}>
              <Card size="small" style={{ textAlign: 'center' }}>
                <div>
                  <Text type="secondary">Datasets</Text>
                  <br />
                  <Title level={3} style={{ margin: 0, color: '#faad14' }}>
                    {dataStats.total_datasets}
                  </Title>
                </div>
              </Card>
            </Col>
            <Col xs={12} sm={6}>
              <Card size="small" style={{ textAlign: 'center' }}>
                <div>
                  <Text type="secondary">Total Size</Text>
                  <br />
                  <Title level={3} style={{ margin: 0, color: '#722ed1' }}>
                    {dataStats.total_size_mb.toFixed(1)} MB
                  </Title>
                </div>
              </Card>
            </Col>
          </Row>
        </Col>
        
        {/* Jobs Table */}
        <Col span={24}>
          <Card 
            title="Aggregation Jobs" 
            extra={
              <Space>
                <Text type="secondary">
                  Completed: {jobStats.completed} | Failed: {jobStats.failed}
                </Text>
                <Button 
                  size="small" 
                  onClick={fetchJobs}
                  loading={isLoading}
                >
                  Refresh
                </Button>
              </Space>
            }
          >
            <Table
              dataSource={jobs}
              columns={jobColumns}
              rowKey="job_id"
              size="small"
              scroll={{ x: 1200 }}
              pagination={{
                pageSize: 10,
                showSizeChanger: true,
              }}
              loading={isLoading}
            />
          </Card>
        </Col>
        
        {/* Aggregated Data Table */}
        <Col span={24}>
          <Card 
            title="Aggregated Data" 
            extra={
              <Text type="secondary">
                {dataStats.total_records.toLocaleString()} total records | 
                Avg compression: {dataStats.avg_compression.toFixed(1)}x
              </Text>
            }
          >
            <Table
              dataSource={aggregatedData}
              columns={dataColumns}
              rowKey={(record) => `${record.data_type}_${record.interval}`}
              size="small"
              pagination={false}
            />
          </Card>
        </Col>
      </Row>
      
      {/* Create Job Modal */}
      <Modal
        title="Create Aggregation Job"
        open={showCreateModal}
        onCancel={() => {
          setShowCreateModal(false);
          form.resetFields();
        }}
        footer={null}
        width={600}
      >
        <Form
          form={form}
          layout="vertical"
          onFinish={createAggregationJob}
          initialValues={{
            compression_level: 'medium',
            retain_raw_data: true,
            enable_indexing: true,
            aggregate_functions: ['avg', 'sum', 'count'],
            group_by_fields: ['date', 'symbol'],
          }}
        >
          <Form.Item
            name="job_name"
            label="Job Name"
            rules={[{ required: true, message: 'Please enter job name' }]}
          >
            <Input placeholder="e.g. Daily Performance Aggregation" />
          </Form.Item>
          
          <Row gutter={16}>
            <Col span={12}>
              <Form.Item
                name="data_type"
                label="Data Type"
                rules={[{ required: true }]}
              >
                <Select placeholder="Select data type">
                  <Option value="performance">Performance</Option>
                  <Option value="risk">Risk</Option>
                  <Option value="execution">Execution</Option>
                  <Option value="strategy">Strategy</Option>
                  <Option value="all">All Types</Option>
                </Select>
              </Form.Item>
            </Col>
            <Col span={12}>
              <Form.Item
                name="interval"
                label="Aggregation Interval"
                rules={[{ required: true }]}
              >
                <Select placeholder="Select interval">
                  <Option value="minute">Minute</Option>
                  <Option value="hour">Hour</Option>
                  <Option value="day">Day</Option>
                  <Option value="week">Week</Option>
                  <Option value="month">Month</Option>
                </Select>
              </Form.Item>
            </Col>
          </Row>
          
          <Form.Item
            name="date_range"
            label="Date Range"
            rules={[{ required: true, message: 'Please select date range' }]}
          >
            <RangePicker style={{ width: '100%' }} />
          </Form.Item>
          
          <Form.Item
            name="aggregate_functions"
            label="Aggregate Functions"
          >
            <Checkbox.Group
              options={[
                { label: 'Average', value: 'avg' },
                { label: 'Sum', value: 'sum' },
                { label: 'Count', value: 'count' },
                { label: 'Min', value: 'min' },
                { label: 'Max', value: 'max' },
                { label: 'Std Dev', value: 'stddev' },
              ]}
            />
          </Form.Item>
          
          <Form.Item
            name="compression_level"
            label="Compression Level"
          >
            <Select>
              <Option value="low">Low (Fast, larger files)</Option>
              <Option value="medium">Medium (Balanced)</Option>
              <Option value="high">High (Slower, smaller files)</Option>
            </Select>
          </Form.Item>
          
          <Row gutter={16}>
            <Col span={12}>
              <Form.Item name="retain_raw_data" valuePropName="checked">
                <Checkbox>Retain Raw Data</Checkbox>
              </Form.Item>
            </Col>
            <Col span={12}>
              <Form.Item name="enable_indexing" valuePropName="checked">
                <Checkbox>Enable Indexing</Checkbox>
              </Form.Item>
            </Col>
          </Row>
          
          <Row justify="end" gutter={8}>
            <Col>
              <Button onClick={() => setShowCreateModal(false)}>
                Cancel
              </Button>
            </Col>
            <Col>
              <Button type="primary" htmlType="submit" loading={isLoading}>
                Create Job
              </Button>
            </Col>
          </Row>
        </Form>
      </Modal>
      
      {/* Job Details Modal */}
      <Modal
        title={selectedJob?.job_name || 'Job Details'}
        open={!!selectedJob}
        onCancel={() => setSelectedJob(null)}
        footer={[
          <Button key="close" onClick={() => setSelectedJob(null)}>
            Close
          </Button>
        ]}
        width={700}
      >
        {selectedJob && (
          <div>
            <Row gutter={[16, 16]}>
              <Col span={12}>
                <Text strong>Job ID:</Text><br />
                <Text code>{selectedJob.job_id}</Text>
              </Col>
              <Col span={12}>
                <Text strong>Status:</Text><br />
                <Tag color={selectedJob.status === 'completed' ? 'green' : selectedJob.status === 'failed' ? 'red' : 'blue'}>
                  {selectedJob.status.toUpperCase()}
                </Tag>
              </Col>
              <Col span={12}>
                <Text strong>Data Type:</Text><br />
                <Text>{selectedJob.data_type}</Text>
              </Col>
              <Col span={12}>
                <Text strong>Interval:</Text><br />
                <Text>{selectedJob.interval}</Text>
              </Col>
              <Col span={12}>
                <Text strong>Progress:</Text><br />
                <Progress percent={selectedJob.progress} size="small" />
              </Col>
              <Col span={12}>
                <Text strong>Records:</Text><br />
                <Text>{selectedJob.records_processed.toLocaleString()} / {selectedJob.total_records.toLocaleString()}</Text>
              </Col>
            </Row>
            
            {selectedJob.error_message && (
              <div style={{ marginTop: 16 }}>
                <Text strong>Error:</Text><br />
                <Alert
                  message={selectedJob.error_message}
                  type="error"
                  size="small"
                />
              </div>
            )}
            
            <div style={{ marginTop: 16 }}>
              <Text strong>Configuration:</Text>
              <pre style={{ 
                background: '#f5f5f5', 
                padding: 12, 
                borderRadius: 4, 
                fontSize: '12px',
                marginTop: 8
              }}>
                {JSON.stringify(selectedJob.config, null, 2)}
              </pre>
            </div>
          </div>
        )}
      </Modal>
    </div>
  );
};

export default AnalyticsAggregator;