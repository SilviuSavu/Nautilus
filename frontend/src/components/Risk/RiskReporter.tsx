import React, { useState, useCallback } from 'react';
import {
  Card,
  Row,
  Col,
  Button,
  Table,
  Progress,
  Tag,
  Space,
  Modal,
  Form,
  Input,
  Select,
  DatePicker,
  Switch,
  Checkbox,
  Alert,
  Typography,
  Tabs,
  List,
  Statistic,
  Tooltip,
  Badge,
  Divider,
  Upload,
  Timeline,
  Descriptions,
  Empty
} from 'antd';
import {
  FileTextOutlined,
  DownloadOutlined,
  ScheduleOutlined,
  SettingOutlined,
  PlusOutlined,
  EyeOutlined,
  DeleteOutlined,
  FilePdfOutlined,
  FileExcelOutlined,
  SendOutlined,
  ClockCircleOutlined,
  CheckCircleOutlined,
  ExclamationCircleOutlined,
  UploadOutlined,
  MailOutlined,
  CalendarOutlined,
  LineChartOutlined
} from '@ant-design/icons';
import dayjs from 'dayjs';
import { useRiskReporting, GenerateReportParams, ReportSchedule } from '../../hooks/risk/useRiskReporting';

const { Title, Text, Paragraph } = Typography;
const { RangePicker } = DatePicker;
const { Option } = Select;
const { TextArea } = Input;

interface RiskReporterProps {
  portfolioId: string;
  className?: string;
}

const RiskReporter: React.FC<RiskReporterProps> = ({
  portfolioId,
  className
}) => {
  const {
    reports,
    templates,
    schedules,
    loading,
    error,
    recentReports,
    completedReports,
    failedReports,
    generatingReports,
    activeSchedules,
    reportsByType,
    generateReport,
    downloadReport,
    deleteReport,
    createSchedule,
    clearError,
    isGenerating,
    getTemplate
  } = useRiskReporting({ portfolioId });

  const [showGenerateModal, setShowGenerateModal] = useState(false);
  const [showScheduleModal, setShowScheduleModal] = useState(false);
  const [showPreviewModal, setShowPreviewModal] = useState(false);
  const [selectedReport, setSelectedReport] = useState<any>(null);
  const [generateForm] = Form.useForm();
  const [scheduleForm] = Form.useForm();

  const handleGenerateReport = useCallback(async (values: any) => {
    try {
      const params: GenerateReportParams = {
        report_type: values.report_type,
        format: values.format,
        period_start: values.date_range[0].toDate(),
        period_end: values.date_range[1].toDate(),
        template_id: values.template_id,
        include_stress_tests: values.include_stress_tests,
        include_scenarios: values.include_scenarios,
        custom_parameters: values.custom_parameters ? JSON.parse(values.custom_parameters) : undefined
      };

      await generateReport(params);
      setShowGenerateModal(false);
      generateForm.resetFields();
    } catch (error) {
      console.error('Error generating report:', error);
    }
  }, [generateReport, generateForm]);

  const handleCreateSchedule = useCallback(async (values: any) => {
    try {
      const scheduleData: Omit<ReportSchedule, 'id' | 'next_run'> = {
        template_id: values.template_id,
        portfolio_id: portfolioId,
        frequency: values.frequency,
        time_of_day: values.time_of_day.format('HH:mm'),
        day_of_week: values.frequency === 'weekly' ? values.day_of_week : undefined,
        day_of_month: values.frequency === 'monthly' ? values.day_of_month : undefined,
        enabled: values.enabled ?? true,
        recipients: values.recipients || [],
        delivery_method: values.delivery_method,
        last_run: undefined
      };

      await createSchedule(scheduleData);
      setShowScheduleModal(false);
      scheduleForm.resetFields();
    } catch (error) {
      console.error('Error creating schedule:', error);
    }
  }, [createSchedule, portfolioId, scheduleForm]);

  const handleDownloadReport = useCallback(async (reportId: string) => {
    try {
      await downloadReport(reportId);
    } catch (error) {
      console.error('Error downloading report:', error);
    }
  }, [downloadReport]);

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'completed': return <CheckCircleOutlined style={{ color: '#52c41a' }} />;
      case 'generating': return <ClockCircleOutlined style={{ color: '#1890ff' }} />;
      case 'failed': return <ExclamationCircleOutlined style={{ color: '#ff4d4f' }} />;
      case 'scheduled': return <ScheduleOutlined style={{ color: '#faad14' }} />;
      default: return <FileTextOutlined />;
    }
  };

  const getFormatIcon = (format: string) => {
    switch (format) {
      case 'pdf': return <FilePdfOutlined style={{ color: '#ff4d4f' }} />;
      case 'excel': return <FileExcelOutlined style={{ color: '#52c41a' }} />;
      case 'html': return <FileTextOutlined style={{ color: '#1890ff' }} />;
      default: return <FileTextOutlined />;
    }
  };

  const reportColumns = [
    {
      title: 'Report',
      key: 'report',
      width: 250,
      render: (record: any) => (
        <Space direction="vertical" size={0}>
          <Space>
            {getFormatIcon(record.format)}
            <Text strong>{record.report_type.charAt(0).toUpperCase() + record.report_type.slice(1)} Report</Text>
          </Space>
          <Text type="secondary" style={{ fontSize: '12px' }}>
            {record.period_start && record.period_end && 
              `${dayjs(record.period_start).format('MMM D')} - ${dayjs(record.period_end).format('MMM D, YYYY')}`}
          </Text>
        </Space>
      )
    },
    {
      title: 'Status',
      dataIndex: 'status',
      key: 'status',
      width: 120,
      render: (status: string) => (
        <Tag icon={getStatusIcon(status)} color={
          status === 'completed' ? 'success' :
          status === 'generating' ? 'processing' :
          status === 'failed' ? 'error' : 'default'
        }>
          {status.toUpperCase()}
        </Tag>
      )
    },
    {
      title: 'Generated',
      dataIndex: 'generated_at',
      key: 'generated_at',
      width: 150,
      render: (date: Date) => (
        <Space direction="vertical" size={0}>
          <Text>{dayjs(date).format('MMM D, YYYY')}</Text>
          <Text type="secondary" style={{ fontSize: '12px' }}>
            {dayjs(date).format('h:mm A')}
          </Text>
        </Space>
      )
    },
    {
      title: 'Size',
      dataIndex: 'file_size_bytes',
      key: 'file_size',
      width: 100,
      render: (bytes?: number) => {
        if (!bytes) return <Text type="secondary">-</Text>;
        const mb = bytes / (1024 * 1024);
        return <Text>{mb.toFixed(1)} MB</Text>;
      }
    },
    {
      title: 'Generation Time',
      key: 'generation_time',
      width: 120,
      render: (record: any) => {
        const time = record.metadata?.generation_time_seconds;
        if (!time) return <Text type="secondary">-</Text>;
        return <Text>{time}s</Text>;
      }
    },
    {
      title: 'Actions',
      key: 'actions',
      width: 150,
      render: (record: any) => (
        <Space>
          {record.status === 'completed' && (
            <Tooltip title="Download Report">
              <Button
                type="text"
                icon={<DownloadOutlined />}
                size="small"
                onClick={() => handleDownloadReport(record.id)}
              />
            </Tooltip>
          )}
          <Tooltip title="Preview Report">
            <Button
              type="text"
              icon={<EyeOutlined />}
              size="small"
              onClick={() => {
                setSelectedReport(record);
                setShowPreviewModal(true);
              }}
              disabled={record.status !== 'completed'}
            />
          </Tooltip>
          <Tooltip title="Delete Report">
            <Button
              type="text"
              icon={<DeleteOutlined />}
              size="small"
              danger
              onClick={() => deleteReport(record.id)}
            />
          </Tooltip>
        </Space>
      )
    }
  ];

  const scheduleColumns = [
    {
      title: 'Template',
      key: 'template',
      width: 200,
      render: (record: ReportSchedule) => {
        const template = getTemplate(record.template_id);
        return (
          <Space direction="vertical" size={0}>
            <Text strong>{template?.name || 'Unknown Template'}</Text>
            <Tag color="blue" size="small">{record.frequency.toUpperCase()}</Tag>
          </Space>
        );
      }
    },
    {
      title: 'Schedule',
      key: 'schedule',
      width: 150,
      render: (record: ReportSchedule) => (
        <Space direction="vertical" size={0}>
          <Text>{record.time_of_day}</Text>
          {record.frequency === 'weekly' && record.day_of_week !== undefined && (
            <Text type="secondary" style={{ fontSize: '12px' }}>
              {['Sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat'][record.day_of_week]}
            </Text>
          )}
          {record.frequency === 'monthly' && record.day_of_month && (
            <Text type="secondary" style={{ fontSize: '12px' }}>
              Day {record.day_of_month}
            </Text>
          )}
        </Space>
      )
    },
    {
      title: 'Next Run',
      dataIndex: 'next_run',
      key: 'next_run',
      width: 150,
      render: (date: Date) => (
        <Space direction="vertical" size={0}>
          <Text>{dayjs(date).format('MMM D, YYYY')}</Text>
          <Text type="secondary" style={{ fontSize: '12px' }}>
            {dayjs(date).fromNow()}
          </Text>
        </Space>
      )
    },
    {
      title: 'Recipients',
      dataIndex: 'recipients',
      key: 'recipients',
      width: 200,
      render: (recipients: string[]) => (
        <div>
          {recipients.slice(0, 2).map((email, index) => (
            <Tag key={index} size="small">{email}</Tag>
          ))}
          {recipients.length > 2 && (
            <Tag size="small">+{recipients.length - 2} more</Tag>
          )}
        </div>
      )
    },
    {
      title: 'Status',
      dataIndex: 'enabled',
      key: 'enabled',
      width: 100,
      render: (enabled: boolean) => (
        <Tag color={enabled ? 'success' : 'default'}>
          {enabled ? 'Active' : 'Disabled'}
        </Tag>
      )
    }
  ];

  const availableSections = [
    { id: 'executive_summary', name: 'Executive Summary', description: 'High-level risk overview' },
    { id: 'risk_metrics', name: 'Risk Metrics', description: 'VaR, ES, volatility metrics' },
    { id: 'exposure_analysis', name: 'Exposure Analysis', description: 'Portfolio exposure breakdown' },
    { id: 'correlation_matrix', name: 'Correlation Matrix', description: 'Asset correlation analysis' },
    { id: 'stress_tests', name: 'Stress Tests', description: 'Scenario stress test results' },
    { id: 'limit_monitoring', name: 'Limit Monitoring', description: 'Risk limit status and breaches' },
    { id: 'performance_attribution', name: 'Performance Attribution', description: 'Risk-adjusted returns' },
    { id: 'compliance_summary', name: 'Compliance Summary', description: 'Regulatory compliance status' }
  ];

  return (
    <div className={className}>
      {/* Header Statistics */}
      <Row gutter={16} style={{ marginBottom: 16 }}>
        <Col span={6}>
          <Card size="small">
            <Statistic
              title="Total Reports"
              value={reports.length}
              prefix={<FileTextOutlined style={{ color: '#1890ff' }} />}
            />
          </Card>
        </Col>
        <Col span={6}>
          <Card size="small">
            <Statistic
              title="Generating"
              value={generatingReports.length}
              prefix={<ClockCircleOutlined style={{ color: '#faad14' }} />}
              valueStyle={{ color: generatingReports.length > 0 ? '#faad14' : undefined }}
            />
          </Card>
        </Col>
        <Col span={6}>
          <Card size="small">
            <Statistic
              title="Active Schedules"
              value={activeSchedules.length}
              prefix={<ScheduleOutlined style={{ color: '#52c41a' }} />}
            />
          </Card>
        </Col>
        <Col span={6}>
          <Card size="small">
            <Statistic
              title="Failed Reports"
              value={failedReports.length}
              prefix={<ExclamationCircleOutlined style={{ color: '#ff4d4f' }} />}
              valueStyle={{ color: failedReports.length > 0 ? '#ff4d4f' : undefined }}
            />
          </Card>
        </Col>
      </Row>

      {/* Error Alert */}
      {error && (
        <Alert
          message="Report Generation Error"
          description={error}
          type="error"
          showIcon
          closable
          onClose={clearError}
          style={{ marginBottom: 16 }}
        />
      )}

      <Row gutter={16}>
        {/* Main Content */}
        <Col span={18}>
          <Tabs
            defaultActiveKey="reports"
            items={[
              {
                key: 'reports',
                label: (
                  <Space>
                    <FileTextOutlined />
                    Reports
                    <Badge count={reports.length} size="small" />
                  </Space>
                ),
                children: (
                  <Card
                    title="Risk Reports"
                    extra={
                      <Space>
                        <Button
                          type="primary"
                          icon={<PlusOutlined />}
                          onClick={() => setShowGenerateModal(true)}
                          loading={isGenerating()}
                        >
                          Generate Report
                        </Button>
                        <Button icon={<SettingOutlined />}>
                          Templates
                        </Button>
                      </Space>
                    }
                  >
                    <Table
                      dataSource={reports}
                      columns={reportColumns}
                      rowKey="id"
                      loading={loading.reports}
                      pagination={{
                        pageSize: 10,
                        showSizeChanger: true,
                        showQuickJumper: true,
                        showTotal: (total) => `Total ${total} reports`
                      }}
                    />
                  </Card>
                )
              },
              {
                key: 'schedules',
                label: (
                  <Space>
                    <CalendarOutlined />
                    Schedules
                    <Badge count={schedules.length} size="small" />
                  </Space>
                ),
                children: (
                  <Card
                    title="Report Schedules"
                    extra={
                      <Button
                        type="primary"
                        icon={<PlusOutlined />}
                        onClick={() => setShowScheduleModal(true)}
                      >
                        Add Schedule
                      </Button>
                    }
                  >
                    <Table
                      dataSource={schedules}
                      columns={scheduleColumns}
                      rowKey="id"
                      loading={loading.schedules}
                      pagination={{
                        pageSize: 10,
                        showSizeChanger: true,
                        showTotal: (total) => `Total ${total} schedules`
                      }}
                    />
                  </Card>
                )
              }
            ]}
          />
        </Col>

        {/* Sidebar */}
        <Col span={6}>
          {/* Quick Actions */}
          <Card title="Quick Actions" size="small" style={{ marginBottom: 16 }}>
            <Space direction="vertical" style={{ width: '100%' }}>
              <Button
                block
                icon={<FileTextOutlined />}
                onClick={() => {
                  generateForm.setFieldsValue({
                    report_type: 'daily',
                    format: 'pdf',
                    date_range: [dayjs().subtract(1, 'day'), dayjs()],
                    include_stress_tests: true
                  });
                  setShowGenerateModal(true);
                }}
              >
                Daily Risk Report
              </Button>
              <Button
                block
                icon={<LineChartOutlined />}
                onClick={() => {
                  generateForm.setFieldsValue({
                    report_type: 'weekly',
                    format: 'pdf',
                    date_range: [dayjs().subtract(1, 'week'), dayjs()],
                    include_scenarios: true
                  });
                  setShowGenerateModal(true);
                }}
              >
                Weekly Analysis
              </Button>
              <Button
                block
                icon={<SendOutlined />}
                onClick={() => {
                  generateForm.setFieldsValue({
                    report_type: 'compliance',
                    format: 'excel',
                    date_range: [dayjs().subtract(1, 'month'), dayjs()]
                  });
                  setShowGenerateModal(true);
                }}
              >
                Compliance Report
              </Button>
            </Space>
          </Card>

          {/* Recent Activity */}
          <Card title="Recent Activity" size="small" style={{ marginBottom: 16 }}>
            <Timeline size="small">
              {recentReports.slice(0, 5).map((report) => (
                <Timeline.Item
                  key={report.id}
                  color={report.status === 'completed' ? 'green' : report.status === 'failed' ? 'red' : 'blue'}
                >
                  <div style={{ fontSize: '12px' }}>
                    <div style={{ fontWeight: 'bold' }}>
                      {report.report_type.charAt(0).toUpperCase() + report.report_type.slice(1)} Report
                    </div>
                    <div style={{ color: '#666' }}>
                      {dayjs(report.generated_at).fromNow()}
                    </div>
                    <div style={{ color: report.status === 'completed' ? '#52c41a' : report.status === 'failed' ? '#ff4d4f' : '#1890ff' }}>
                      {report.status}
                    </div>
                  </div>
                </Timeline.Item>
              ))}
            </Timeline>
            {recentReports.length === 0 && (
              <Empty description="No recent reports" image={Empty.PRESENTED_IMAGE_SIMPLE} />
            )}
          </Card>

          {/* Templates */}
          <Card title="Available Templates" size="small">
            <List
              size="small"
              dataSource={templates}
              renderItem={(template) => (
                <List.Item
                  actions={[
                    <Button
                      size="small"
                      type="link"
                      onClick={() => {
                        generateForm.setFieldsValue({
                          template_id: template.id,
                          report_type: template.report_type,
                          format: template.default_format
                        });
                        setShowGenerateModal(true);
                      }}
                    >
                      Use
                    </Button>
                  ]}
                >
                  <List.Item.Meta
                    title={<Text style={{ fontSize: '13px' }}>{template.name}</Text>}
                    description={
                      <Text type="secondary" style={{ fontSize: '11px' }}>
                        {template.description}
                      </Text>
                    }
                  />
                </List.Item>
              )}
            />
          </Card>
        </Col>
      </Row>

      {/* Generate Report Modal */}
      <Modal
        title="Generate Risk Report"
        open={showGenerateModal}
        onCancel={() => {
          setShowGenerateModal(false);
          generateForm.resetFields();
        }}
        footer={null}
        width={700}
      >
        <Form
          form={generateForm}
          layout="vertical"
          onFinish={handleGenerateReport}
          initialValues={{
            format: 'pdf',
            include_stress_tests: false,
            include_scenarios: false
          }}
        >
          <Row gutter={16}>
            <Col span={12}>
              <Form.Item
                name="report_type"
                label="Report Type"
                rules={[{ required: true, message: 'Please select report type' }]}
              >
                <Select placeholder="Select report type">
                  <Option value="daily">Daily Risk Report</Option>
                  <Option value="weekly">Weekly Risk Report</Option>
                  <Option value="monthly">Monthly Risk Report</Option>
                  <Option value="custom">Custom Report</Option>
                  <Option value="compliance">Compliance Report</Option>
                </Select>
              </Form.Item>
            </Col>
            <Col span={12}>
              <Form.Item
                name="format"
                label="Format"
                rules={[{ required: true, message: 'Please select format' }]}
              >
                <Select>
                  <Option value="pdf">PDF</Option>
                  <Option value="excel">Excel</Option>
                  <Option value="html">HTML</Option>
                  <Option value="json">JSON</Option>
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

          <Form.Item name="template_id" label="Template (Optional)">
            <Select placeholder="Select template" allowClear>
              {templates.map((template) => (
                <Option key={template.id} value={template.id}>
                  {template.name}
                </Option>
              ))}
            </Select>
          </Form.Item>

          <Form.Item label="Report Sections">
            <Checkbox.Group style={{ width: '100%' }}>
              <Row>
                {availableSections.map((section) => (
                  <Col span={24} key={section.id} style={{ marginBottom: 8 }}>
                    <Checkbox value={section.id}>
                      <div>
                        <div style={{ fontWeight: 'bold' }}>{section.name}</div>
                        <div style={{ fontSize: '12px', color: '#666' }}>
                          {section.description}
                        </div>
                      </div>
                    </Checkbox>
                  </Col>
                ))}
              </Row>
            </Checkbox.Group>
          </Form.Item>

          <Row gutter={16}>
            <Col span={12}>
              <Form.Item name="include_stress_tests" label="Include Stress Tests" valuePropName="checked">
                <Switch />
              </Form.Item>
            </Col>
            <Col span={12}>
              <Form.Item name="include_scenarios" label="Include Scenarios" valuePropName="checked">
                <Switch />
              </Form.Item>
            </Col>
          </Row>

          <Form.Item name="custom_parameters" label="Custom Parameters (JSON)">
            <TextArea
              rows={3}
              placeholder='{"confidence_levels": [95, 99], "stress_scenarios": ["market_crash"]}'
            />
          </Form.Item>

          <Form.Item style={{ marginBottom: 0 }}>
            <Space style={{ width: '100%', justifyContent: 'flex-end' }}>
              <Button onClick={() => {
                setShowGenerateModal(false);
                generateForm.resetFields();
              }}>
                Cancel
              </Button>
              <Button type="primary" htmlType="submit" loading={loading.generating}>
                Generate Report
              </Button>
            </Space>
          </Form.Item>
        </Form>
      </Modal>

      {/* Schedule Modal */}
      <Modal
        title="Create Report Schedule"
        open={showScheduleModal}
        onCancel={() => {
          setShowScheduleModal(false);
          scheduleForm.resetFields();
        }}
        footer={null}
        width={600}
      >
        <Form
          form={scheduleForm}
          layout="vertical"
          onFinish={handleCreateSchedule}
          initialValues={{ enabled: true, delivery_method: 'email' }}
        >
          {/* Schedule form implementation would continue here... */}
          <Paragraph type="secondary">
            Schedule configuration form would be implemented here with template selection,
            frequency settings, recipient management, and delivery options.
          </Paragraph>

          <Form.Item style={{ marginBottom: 0 }}>
            <Space style={{ width: '100%', justifyContent: 'flex-end' }}>
              <Button onClick={() => {
                setShowScheduleModal(false);
                scheduleForm.resetFields();
              }}>
                Cancel
              </Button>
              <Button type="primary" htmlType="submit">
                Create Schedule
              </Button>
            </Space>
          </Form.Item>
        </Form>
      </Modal>

      {/* Preview Modal */}
      <Modal
        title="Report Preview"
        open={showPreviewModal}
        onCancel={() => {
          setShowPreviewModal(false);
          setSelectedReport(null);
        }}
        width={800}
        footer={[
          <Button key="download" icon={<DownloadOutlined />} onClick={() => {
            if (selectedReport) handleDownloadReport(selectedReport.id);
          }}>
            Download
          </Button>,
          <Button key="close" onClick={() => {
            setShowPreviewModal(false);
            setSelectedReport(null);
          }}>
            Close
          </Button>
        ]}
      >
        {selectedReport && (
          <div>
            <Descriptions bordered size="small">
              <Descriptions.Item label="Report Type" span={2}>
                {selectedReport.report_type}
              </Descriptions.Item>
              <Descriptions.Item label="Format">
                {selectedReport.format.toUpperCase()}
              </Descriptions.Item>
              <Descriptions.Item label="Period" span={3}>
                {dayjs(selectedReport.period_start).format('MMM D, YYYY')} - {dayjs(selectedReport.period_end).format('MMM D, YYYY')}
              </Descriptions.Item>
              <Descriptions.Item label="Generated" span={2}>
                {dayjs(selectedReport.generated_at).format('MMM D, YYYY h:mm A')}
              </Descriptions.Item>
              <Descriptions.Item label="File Size">
                {selectedReport.file_size_bytes ? `${(selectedReport.file_size_bytes / (1024 * 1024)).toFixed(1)} MB` : 'N/A'}
              </Descriptions.Item>
            </Descriptions>

            <Divider>Report Sections</Divider>
            <List
              size="small"
              dataSource={selectedReport.sections_included || []}
              renderItem={(section: any) => (
                <List.Item>
                  <CheckCircleOutlined style={{ color: '#52c41a', marginRight: 8 }} />
                  {section.name}
                </List.Item>
              )}
            />
          </div>
        )}
      </Modal>
    </div>
  );
};

export default RiskReporter;