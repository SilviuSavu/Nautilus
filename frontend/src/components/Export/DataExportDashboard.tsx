/**
 * Story 5.3: Data Export Dashboard
 * Main dashboard component for data export and reporting functionality
 */

import React, { useState, useEffect } from 'react';
import {
  Card,
  Row,
  Col,
  Typography,
  Space,
  Button,
  Tabs,
  Alert,
  Select,
  DatePicker,
  Form,
  Input,
  Modal,
  Progress,
  Tag,
  Table,
  Tooltip,
  notification,
  Statistic
} from 'antd';
import {
  ExportOutlined,
  FileTextOutlined,
  ApiOutlined,
  DownloadOutlined,
  DeleteOutlined,
  ReloadOutlined,
  PlusOutlined,
  SettingOutlined,
  CheckCircleOutlined,
  CloseCircleOutlined,
  ClockCircleOutlined,
  LoadingOutlined
} from '@ant-design/icons';
import dayjs from 'dayjs';
import relativeTime from 'dayjs/plugin/relativeTime';

dayjs.extend(relativeTime);
import { useExportManager } from '../../hooks/export/useExportManager';
import TemplateManager from './TemplateManager';
import ScheduledReports from './ScheduledReports';
import IntegrationSetup from './IntegrationSetup';
import { 
  ExportRequest, 
  ExportType, 
  DataSource, 
  ExportStatus, 
  ExportHistory
} from '../../types/export';

const { Title, Text } = Typography;
const { TabPane } = Tabs;
const { RangePicker } = DatePicker;
const { Option } = Select;

interface DataExportDashboardProps {
  className?: string;
}

export const DataExportDashboard: React.FC<DataExportDashboardProps> = ({
  className
}) => {
  const [activeTab, setActiveTab] = useState('exports');
  const [exportModalVisible, setExportModalVisible] = useState(false);
  const [form] = Form.useForm();

  const {
    exports,
    loading,
    error,
    createExport,
    downloadExport,
    deleteExport,
    refreshExports,
    getAvailableFields
  } = useExportManager();

  const [availableFields, setAvailableFields] = useState<string[]>([]);
  const [selectedDataSource, setSelectedDataSource] = useState<DataSource>(DataSource.TRADES);

  // Load available fields when data source changes
  useEffect(() => {
    const loadFields = async () => {
      try {
        const response = await getAvailableFields(selectedDataSource);
        setAvailableFields(response.available_fields);
      } catch (error) {
        console.error('Failed to load available fields:', error);
      }
    };

    loadFields();
  }, [selectedDataSource, getAvailableFields]);

  const handleCreateExport = async (values: any) => {
    try {
      const exportRequest: ExportRequest = {
        type: values.type,
        data_source: values.data_source,
        filters: {
          date_range: {
            start_date: values.date_range[0].toDate(),
            end_date: values.date_range[1].toDate()
          },
          symbols: values.symbols?.split(',').map((s: string) => s.trim()).filter(Boolean),
          accounts: values.accounts?.split(',').map((a: string) => a.trim()).filter(Boolean),
          strategies: values.strategies?.split(',').map((s: string) => s.trim()).filter(Boolean),
          venues: values.venues
        },
        fields: values.fields || availableFields,
        options: {
          include_headers: values.include_headers ?? true,
          compression: values.compression ?? false,
          precision: values.precision ?? 4,
          timezone: values.timezone || 'UTC',
          currency: values.currency || 'USD'
        },
        status: ExportStatus.PENDING,
        progress: 0
      };

      const response = await createExport(exportRequest);
      
      notification.success({
        message: 'Export Created',
        description: `Export request ${response.export_id} has been created successfully`,
        duration: 4
      });

      setExportModalVisible(false);
      form.resetFields();
    } catch (error: any) {
      notification.error({
        message: 'Export Failed',
        description: error.message || 'Failed to create export request',
        duration: 6
      });
    }
  };

  const handleDownloadExport = async (exportId: string) => {
    try {
      await downloadExport(exportId);
      notification.success({
        message: 'Download Started',
        description: 'Export file download has been initiated',
        duration: 3
      });
    } catch (error: any) {
      notification.error({
        message: 'Download Failed',
        description: error.message || 'Failed to download export file',
        duration: 4
      });
    }
  };

  const handleDeleteExport = (exportId: string) => {
    Modal.confirm({
      title: 'Delete Export',
      content: 'Are you sure you want to delete this export? This action cannot be undone.',
      okText: 'Delete',
      okType: 'danger',
      onOk: async () => {
        try {
          await deleteExport(exportId);
          notification.success({
            message: 'Export Deleted',
            description: 'Export has been deleted successfully',
            duration: 3
          });
        } catch (error: any) {
          notification.error({
            message: 'Delete Failed',
            description: error.message || 'Failed to delete export',
            duration: 4
          });
        }
      }
    });
  };

  const getStatusIcon = (status: ExportStatus) => {
    switch (status) {
      case ExportStatus.COMPLETED:
        return <CheckCircleOutlined style={{ color: '#52c41a' }} />;
      case ExportStatus.FAILED:
        return <CloseCircleOutlined style={{ color: '#ff4d4f' }} />;
      case ExportStatus.PROCESSING:
        return <LoadingOutlined style={{ color: '#1890ff' }} />;
      default:
        return <ClockCircleOutlined style={{ color: '#faad14' }} />;
    }
  };

  const getStatusColor = (status: ExportStatus): string => {
    switch (status) {
      case ExportStatus.COMPLETED:
        return 'success';
      case ExportStatus.FAILED:
        return 'error';
      case ExportStatus.PROCESSING:
        return 'processing';
      default:
        return 'warning';
    }
  };

  const exportColumns = [
    {
      title: 'Export ID',
      dataIndex: 'id',
      key: 'id',
      render: (id: string) => (
        <Text code style={{ fontSize: 12 }}>
          {id.slice(-8)}
        </Text>
      )
    },
    {
      title: 'Type',
      dataIndex: 'type',
      key: 'type',
      render: (type: ExportType) => (
        <Tag color="blue">{type.toUpperCase()}</Tag>
      )
    },
    {
      title: 'Data Source',
      dataIndex: 'data_source',
      key: 'data_source',
      render: (source: DataSource) => (
        <Tag color="green">{source.replace('_', ' ').toUpperCase()}</Tag>
      )
    },
    {
      title: 'Status',
      dataIndex: 'status',
      key: 'status',
      render: (status: ExportStatus, record: ExportHistory) => (
        <Space>
          {getStatusIcon(status)}
          <Tag color={getStatusColor(status)}>{status.toUpperCase()}</Tag>
          {(status === ExportStatus.PROCESSING || status === ExportStatus.PENDING) && (
            <Progress
              percent={record.progress}
              size="small"
              style={{ width: 60 }}
              showInfo={false}
            />
          )}
        </Space>
      )
    },
    {
      title: 'Created',
      dataIndex: 'created_at',
      key: 'created_at',
      render: (date: Date) => (
        <Text style={{ fontSize: 12 }}>
          {dayjs(date).format('MMM DD, HH:mm')}
        </Text>
      )
    },
    {
      title: 'Actions',
      key: 'actions',
      render: (_: any, record: ExportHistory) => (
        <Space>
          {record.status === ExportStatus.COMPLETED && record.download_url && (
            <Tooltip title="Download">
              <Button
                type="text"
                size="small"
                icon={<DownloadOutlined />}
                onClick={() => handleDownloadExport(record.id)}
              />
            </Tooltip>
          )}
          <Tooltip title="Delete">
            <Button
              type="text"
              size="small"
              danger
              icon={<DeleteOutlined />}
              onClick={() => handleDeleteExport(record.id)}
            />
          </Tooltip>
        </Space>
      )
    }
  ];

  const renderExportsTab = () => (
    <div>
      {/* Error Alert */}
      {error && (
        <Alert
          message="Export Error"
          description={error}
          type="error"
          style={{ marginBottom: 16 }}
          action={
            <Button size="small" onClick={refreshExports}>
              Retry
            </Button>
          }
        />
      )}

      {/* Export Statistics */}
      <Row gutter={[16, 16]} style={{ marginBottom: 24 }}>
        <Col xs={24} sm={6}>
          <Card>
            <Statistic
              title="Total Exports"
              value={exports.length}
              prefix={<ExportOutlined />}
            />
          </Card>
        </Col>
        <Col xs={24} sm={6}>
          <Card>
            <Statistic
              title="Completed"
              value={exports.filter(e => e.status === ExportStatus.COMPLETED).length}
              valueStyle={{ color: '#3f8600' }}
              prefix={<CheckCircleOutlined />}
            />
          </Card>
        </Col>
        <Col xs={24} sm={6}>
          <Card>
            <Statistic
              title="Processing"
              value={exports.filter(e => e.status === ExportStatus.PROCESSING).length}
              valueStyle={{ color: '#1890ff' }}
              prefix={<LoadingOutlined />}
            />
          </Card>
        </Col>
        <Col xs={24} sm={6}>
          <Card>
            <Statistic
              title="Failed"
              value={exports.filter(e => e.status === ExportStatus.FAILED).length}
              valueStyle={{ color: '#cf1322' }}
              prefix={<CloseCircleOutlined />}
            />
          </Card>
        </Col>
      </Row>

      {/* Export Table */}
      <Card
        title="Export History"
        extra={
          <Space>
            <Button
              type="primary"
              icon={<PlusOutlined />}
              onClick={() => setExportModalVisible(true)}
            >
              New Export
            </Button>
            <Button
              icon={<ReloadOutlined />}
              onClick={refreshExports}
              loading={loading}
            >
              Refresh
            </Button>
          </Space>
        }
      >
        <Table
          columns={exportColumns}
          dataSource={exports}
          rowKey="id"
          loading={loading}
          pagination={{
            pageSize: 10,
            showSizeChanger: true,
            showQuickJumper: true,
            showTotal: (total, range) => 
              `${range[0]}-${range[1]} of ${total} exports`
          }}
          scroll={{ x: 800 }}
        />
      </Card>
    </div>
  );

  const renderReportsTab = () => (
    <Tabs defaultActiveKey="templates" type="card">
      <TabPane tab="Templates" key="templates" icon={<FileTextOutlined />}>
        <TemplateManager onGenerateReport={undefined} />
      </TabPane>
      
      <TabPane tab="Scheduled Reports" key="scheduled" icon={<ClockCircleOutlined />}>
        <ScheduledReports />
      </TabPane>
    </Tabs>
  );

  const renderIntegrationsTab = () => {
    const [integrationSetupVisible, setIntegrationSetupVisible] = useState(false);
    const [integrations, setIntegrations] = useState<any[]>([]);
    
    // Mock integrations data
    const mockIntegrations = [
      {
        id: 'integration-001',
        name: 'Portfolio Analytics API',
        endpoint: 'https://api.example.com/portfolio',
        status: 'active',
        last_sync: new Date(Date.now() - 3600000),
        data_mapping: [
          { source_field: 'total_pnl', target_field: 'portfolio_value' },
          { source_field: 'win_rate', target_field: 'success_rate' }
        ]
      }
    ];

    React.useEffect(() => {
      setIntegrations(mockIntegrations);
    }, []);

    return (
      <div>
        <Row gutter={16} style={{ marginBottom: 24 }}>
          <Col xs={24} sm={8}>
            <Card size="small">
              <div style={{ textAlign: 'center' }}>
                <div style={{ fontSize: 24, fontWeight: 'bold', color: '#1890ff' }}>
                  {integrations.length}
                </div>
                <div style={{ fontSize: 12, color: '#666' }}>Active Integrations</div>
              </div>
            </Card>
          </Col>
          <Col xs={24} sm={8}>
            <Card size="small">
              <div style={{ textAlign: 'center' }}>
                <div style={{ fontSize: 24, fontWeight: 'bold', color: '#52c41a' }}>
                  {integrations.filter(i => i.status === 'active').length}
                </div>
                <div style={{ fontSize: 12, color: '#666' }}>Healthy</div>
              </div>
            </Card>
          </Col>
          <Col xs={24} sm={8}>
            <Card size="small">
              <div style={{ textAlign: 'center' }}>
                <div style={{ fontSize: 24, fontWeight: 'bold', color: '#fa8c16' }}>
                  {integrations.filter(i => i.last_sync && dayjs(i.last_sync).isAfter(dayjs().subtract(1, 'hour'))).length}
                </div>
                <div style={{ fontSize: 12, color: '#666' }}>Recently Synced</div>
              </div>
            </Card>
          </Col>
        </Row>

        <Card 
          title="API Integrations" 
          extra={
            <Button 
              type="primary" 
              icon={<PlusOutlined />}
              onClick={() => setIntegrationSetupVisible(true)}
            >
              New Integration
            </Button>
          }
        >
          {integrations.length === 0 ? (
            <div style={{ textAlign: 'center', padding: 60 }}>
              <ApiOutlined style={{ fontSize: 48, color: '#d9d9d9', marginBottom: 16 }} />
              <Title level={4} type="secondary">No Integrations</Title>
              <Text type="secondary">
                Create your first API integration to sync data with external systems
              </Text>
            </div>
          ) : (
            <Table
              dataSource={integrations}
              rowKey="id"
              size="small"
              columns={[
                {
                  title: 'Integration Name',
                  dataIndex: 'name',
                  key: 'name'
                },
                {
                  title: 'Endpoint',
                  dataIndex: 'endpoint',
                  key: 'endpoint',
                  render: (endpoint: string) => (
                    <Text code style={{ fontSize: 11 }}>{endpoint}</Text>
                  )
                },
                {
                  title: 'Status',
                  dataIndex: 'status',
                  key: 'status',
                  render: (status: string) => (
                    <Tag color={status === 'active' ? 'success' : 'error'}>
                      {status.toUpperCase()}
                    </Tag>
                  )
                },
                {
                  title: 'Last Sync',
                  dataIndex: 'last_sync',
                  key: 'last_sync',
                  render: (date: Date) => (
                    <Text style={{ fontSize: 12 }}>
                      {dayjs(date).fromNow()}
                    </Text>
                  )
                },
                {
                  title: 'Mappings',
                  dataIndex: 'data_mapping',
                  key: 'mappings',
                  render: (mappings: any[]) => (
                    <Text style={{ fontSize: 12 }}>
                      {mappings?.length || 0} fields
                    </Text>
                  )
                }
              ]}
            />
          )}
        </Card>

        <IntegrationSetup
          visible={integrationSetupVisible}
          onCancel={() => setIntegrationSetupVisible(false)}
          onSave={async (integration) => {
            // Handle save
            setIntegrations(prev => [...prev, { ...integration, id: Date.now().toString() }]);
          }}
        />
      </div>
    );
  };

  return (
    <div className={`data-export-dashboard ${className || ''}`}>
      <Card>
        <div style={{ marginBottom: 24 }}>
          <Row justify="space-between" align="middle">
            <Col>
              <Title level={2} style={{ margin: 0 }}>
                <ExportOutlined style={{ marginRight: 8 }} />
                Data Export & Reporting
              </Title>
              <Text type="secondary">
                Export trading data, generate reports, and manage API integrations
              </Text>
            </Col>
            <Col>
              <Space>
                <Button
                  type="primary"
                  icon={<PlusOutlined />}
                  onClick={() => setExportModalVisible(true)}
                >
                  New Export
                </Button>
                <Button
                  icon={<SettingOutlined />}
                  disabled
                >
                  Settings
                </Button>
              </Space>
            </Col>
          </Row>
        </div>

        <Tabs activeKey={activeTab} onChange={setActiveTab}>
          <TabPane tab="Data Exports" key="exports" icon={<ExportOutlined />}>
            {renderExportsTab()}
          </TabPane>
          
          <TabPane tab="Report Templates" key="reports" icon={<FileTextOutlined />}>
            {renderReportsTab()}
          </TabPane>
          
          <TabPane tab="API Integrations" key="integrations" icon={<ApiOutlined />}>
            {renderIntegrationsTab()}
          </TabPane>
        </Tabs>
      </Card>

      {/* Export Creation Modal */}
      <Modal
        title="Create Data Export"
        open={exportModalVisible}
        onCancel={() => {
          setExportModalVisible(false);
          form.resetFields();
        }}
        footer={null}
        width={800}
      >
        <Form
          form={form}
          layout="vertical"
          onFinish={handleCreateExport}
          initialValues={{
            type: ExportType.CSV,
            data_source: DataSource.TRADES,
            include_headers: true,
            compression: false,
            precision: 4,
            timezone: 'UTC',
            currency: 'USD',
            date_range: [dayjs().subtract(7, 'days'), dayjs()]
          }}
        >
          <Row gutter={16}>
            <Col xs={24} md={12}>
              <Form.Item
                name="type"
                label="Export Format"
                rules={[{ required: true, message: 'Please select export format' }]}
              >
                <Select>
                  <Option value={ExportType.CSV}>CSV (Comma-Separated Values)</Option>
                  <Option value={ExportType.JSON}>JSON (JavaScript Object Notation)</Option>
                  <Option value={ExportType.EXCEL}>Excel Spreadsheet</Option>
                  <Option value={ExportType.PDF}>PDF Document</Option>
                </Select>
              </Form.Item>
            </Col>
            <Col xs={24} md={12}>
              <Form.Item
                name="data_source"
                label="Data Source"
                rules={[{ required: true, message: 'Please select data source' }]}
              >
                <Select onChange={setSelectedDataSource}>
                  <Option value={DataSource.TRADES}>Trading Data</Option>
                  <Option value={DataSource.POSITIONS}>Positions</Option>
                  <Option value={DataSource.PERFORMANCE}>Performance Metrics</Option>
                  <Option value={DataSource.ORDERS}>Order History</Option>
                  <Option value={DataSource.SYSTEM_METRICS}>System Metrics</Option>
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

          <Row gutter={16}>
            <Col xs={24} md={8}>
              <Form.Item name="symbols" label="Symbols (comma-separated)">
                <Input placeholder="AAPL, MSFT, GOOGL" />
              </Form.Item>
            </Col>
            <Col xs={24} md={8}>
              <Form.Item name="accounts" label="Accounts (comma-separated)">
                <Input placeholder="ACC001, ACC002" />
              </Form.Item>
            </Col>
            <Col xs={24} md={8}>
              <Form.Item name="strategies" label="Strategies (comma-separated)">
                <Input placeholder="momentum_1, mean_reversion" />
              </Form.Item>
            </Col>
          </Row>

          <Form.Item name="fields" label="Fields to Export">
            <Select
              mode="multiple"
              placeholder="Select fields to export (default: all fields)"
              options={availableFields.map(field => ({ label: field, value: field }))}
            />
          </Form.Item>

          <Row gutter={16}>
            <Col xs={24} md={8}>
              <Form.Item name="precision" label="Decimal Precision">
                <Select>
                  <Option value={2}>2 decimal places</Option>
                  <Option value={4}>4 decimal places</Option>
                  <Option value={6}>6 decimal places</Option>
                  <Option value={8}>8 decimal places</Option>
                </Select>
              </Form.Item>
            </Col>
            <Col xs={24} md={8}>
              <Form.Item name="timezone" label="Timezone">
                <Select>
                  <Option value="UTC">UTC</Option>
                  <Option value="America/New_York">Eastern Time</Option>
                  <Option value="America/Chicago">Central Time</Option>
                  <Option value="America/Los_Angeles">Pacific Time</Option>
                </Select>
              </Form.Item>
            </Col>
            <Col xs={24} md={8}>
              <Form.Item name="currency" label="Currency">
                <Select>
                  <Option value="USD">USD</Option>
                  <Option value="EUR">EUR</Option>
                  <Option value="GBP">GBP</Option>
                  <Option value="JPY">JPY</Option>
                </Select>
              </Form.Item>
            </Col>
          </Row>

          <Form.Item>
            <Space>
              <Button type="primary" htmlType="submit" loading={loading}>
                Create Export
              </Button>
              <Button onClick={() => {
                setExportModalVisible(false);
                form.resetFields();
              }}>
                Cancel
              </Button>
            </Space>
          </Form.Item>
        </Form>
      </Modal>
    </div>
  );
};

export default DataExportDashboard;