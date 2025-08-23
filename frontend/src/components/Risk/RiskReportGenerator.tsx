import React, { useState, useEffect } from 'react';
import {
  Card,
  Form,
  Select,
  DatePicker,
  Switch,
  Button,
  Space,
  Alert,
  Typography,
  Row,
  Col,
  Table,
  Progress,
  Badge,
  Tooltip,
  Modal,
  Checkbox,
  Divider,
  Tag,
  List,
  Avatar,
  notification,
  Popconfirm
} from 'antd';
import {
  FileTextOutlined,
  DownloadOutlined,
  FilePdfOutlined,
  FileExcelOutlined,
  FileOutlined,
  CodeOutlined,
  GlobalOutlined,
  CalendarOutlined,
  SettingOutlined,
  DeleteOutlined,
  EyeOutlined,
  ClockCircleOutlined,
  CheckCircleOutlined,
  ExclamationCircleOutlined,
  LoadingOutlined,
  ReloadOutlined,
  PlusOutlined
} from '@ant-design/icons';
import dayjs from 'dayjs';

import { RiskReportRequest, RiskReport, ReportSection } from './types/riskTypes';
import { riskService } from './services/riskService';
import { useRiskReporting } from '../../hooks/risk/useRiskReporting';

const { Title, Text, Paragraph } = Typography;
const { Option } = Select;
const { RangePicker } = DatePicker;
const { CheckboxGroup } = Checkbox;

interface RiskReportGeneratorProps {
  portfolioId: string;
  className?: string;
}

const RiskReportGenerator: React.FC<RiskReportGeneratorProps> = ({
  portfolioId,
  className
}) => {
  console.log('🎯 RiskReportGenerator rendering for portfolio:', portfolioId);

  const [form] = Form.useForm();
  const [reports, setReports] = useState<RiskReport[]>([]);
  const [loading, setLoading] = useState(false);
  const [fetchingReports, setFetchingReports] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [previewVisible, setPreviewVisible] = useState(false);
  const [previewReport, setPreviewReport] = useState<RiskReport | null>(null);

  // Use risk reporting hook
  const {
    reports: hookReports,
    generatingReports,
    completedReports,
    failedReports
  } = useRiskReporting({ portfolioId });

  const reportTypes = [
    { value: 'daily', label: 'Daily Risk Report', description: 'Daily risk metrics and alerts' },
    { value: 'weekly', label: 'Weekly Risk Summary', description: 'Weekly risk analysis and trends' },
    { value: 'monthly', label: 'Monthly Risk Review', description: 'Comprehensive monthly risk assessment' },
    { value: 'custom', label: 'Custom Report', description: 'Customizable risk report with date range' },
    { value: 'regulatory', label: 'Regulatory Report', description: 'Compliance and regulatory reporting' }
  ];

  const formats = [
    { value: 'pdf', label: 'PDF', icon: <FilePdfOutlined />, color: '#ff4d4f' },
    { value: 'excel', label: 'Excel', icon: <FileExcelOutlined />, color: '#52c41a' },
    { value: 'csv', label: 'CSV', icon: <FileOutlined />, color: '#1890ff' },
    { value: 'json', label: 'JSON', icon: <CodeOutlined />, color: '#722ed1' },
    { value: 'html', label: 'HTML', icon: <GlobalOutlined />, color: '#fa8c16' }
  ];

  const regulatoryFrameworks = [
    { value: 'basel_iii', label: 'Basel III' },
    { value: 'mifid_ii', label: 'MiFID II' },
    { value: 'dodd_frank', label: 'Dodd-Frank' },
    { value: 'custom', label: 'Custom Framework' }
  ];

  const availableSections: ReportSection[] = [
    {
      section_type: 'executive_summary',
      enabled: true,
      detail_level: 'summary'
    },
    {
      section_type: 'var_analysis',
      enabled: true,
      detail_level: 'detailed'
    },
    {
      section_type: 'concentration',
      enabled: true,
      detail_level: 'detailed'
    },
    {
      section_type: 'stress_tests',
      enabled: false,
      detail_level: 'summary'
    },
    {
      section_type: 'limit_breaches',
      enabled: true,
      detail_level: 'full'
    },
    {
      section_type: 'recommendations',
      enabled: true,
      detail_level: 'detailed'
    }
  ];

  const fetchReports = async () => {
    try {
      setError(null);
      const data = await riskService.getRiskReports(portfolioId);
      setReports(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to fetch reports');
      console.error('Reports fetch error:', err);
    } finally {
      setFetchingReports(false);
    }
  };

  const generateReport = async (values: any) => {
    try {
      setLoading(true);
      
      const request: RiskReportRequest = {
        portfolio_id: portfolioId,
        report_type: values.reportType,
        format: values.format,
        sections: values.sections || availableSections,
        date_range: values.dateRange ? {
          start: values.dateRange[0].toDate(),
          end: values.dateRange[1].toDate()
        } : undefined,
        include_charts: values.includeCharts,
        include_recommendations: values.includeRecommendations,
        regulatory_framework: values.regulatoryFramework
      };

      const report = await riskService.requestRiskReport(request);
      
      notification.success({
        message: 'Report Generation Started',
        description: `${values.reportType} report is being generated`,
        duration: 4
      });

      setReports(prev => [report, ...prev]);
      form.resetFields();
      
    } catch (err) {
      notification.error({
        message: 'Report Generation Failed',
        description: err instanceof Error ? err.message : 'Failed to generate report',
        duration: 5
      });
    } finally {
      setLoading(false);
    }
  };

  const downloadReport = async (reportId: string, fileName: string) => {
    try {
      const blob = await riskService.downloadRiskReport(reportId);
      const url = window.URL.createObjectURL(blob);
      const link = document.createElement('a');
      link.href = url;
      link.download = fileName;
      document.body.appendChild(link);
      link.click();
      window.URL.revokeObjectURL(url);
      document.body.removeChild(link);
      
      notification.success({
        message: 'Download Started',
        description: `${fileName} is being downloaded`,
        duration: 3
      });
    } catch (error) {
      notification.error({
        message: 'Download Failed',
        description: 'Unable to download the report',
        duration: 4
      });
    }
  };

  const cancelReport = async (reportId: string) => {
    try {
      await riskService.cancelRiskReport(reportId);
      setReports(prev => prev.filter(r => r.id !== reportId));
      notification.info({
        message: 'Report Cancelled',
        description: 'Report generation has been cancelled',
        duration: 3
      });
    } catch (error) {
      notification.error({
        message: 'Cancel Failed',
        description: 'Unable to cancel the report generation',
        duration: 4
      });
    }
  };

  const previewReportData = async (report: RiskReport) => {
    try {
      const fullReport = await riskService.getRiskReport(report.id);
      setPreviewReport(fullReport);
      setPreviewVisible(true);
    } catch (error) {
      notification.error({
        message: 'Preview Failed',
        description: 'Unable to load report preview',
        duration: 4
      });
    }
  };

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'generating': return <LoadingOutlined spin />;
      case 'completed': return <CheckCircleOutlined style={{ color: '#52c41a' }} />;
      case 'failed': return <ExclamationCircleOutlined style={{ color: '#ff4d4f' }} />;
      case 'cancelled': return <ExclamationCircleOutlined style={{ color: '#faad14' }} />;
      default: return <ClockCircleOutlined />;
    }
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'generating': return 'processing';
      case 'completed': return 'success';
      case 'failed': return 'error';
      case 'cancelled': return 'warning';
      default: return 'default';
    }
  };

  const formatFileSize = (bytes: number): string => {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
  };

  const getFormatIcon = (format: string) => {
    const formatConfig = formats.find(f => f.value === format);
    return formatConfig ? formatConfig.icon : <FileOutlined />;
  };

  const getFormatColor = (format: string) => {
    const formatConfig = formats.find(f => f.value === format);
    return formatConfig ? formatConfig.color : '#1890ff';
  };

  const columns = [
    {
      title: 'Report',
      key: 'report',
      render: (record: RiskReport) => (
        <Space>
          {getFormatIcon(record.request.format)}
          <div>
            <Text strong>{record.request.report_type.replace('_', ' ').toUpperCase()}</Text>
            <br />
            <Text type="secondary" style={{ fontSize: '12px' }}>
              {record.request.format.toUpperCase()}
            </Text>
          </div>
        </Space>
      )
    },
    {
      title: 'Status',
      dataIndex: 'status',
      key: 'status',
      render: (status: string, record: RiskReport) => (
        <Space>
          {getStatusIcon(status)}
          <Badge status={getStatusColor(status)} text={status.toUpperCase()} />
          {status === 'generating' && (
            <Progress
              percent={75} // Mock progress
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
        <Space direction="vertical" size={0}>
          <Text style={{ fontSize: '12px' }}>
            {dayjs(date).format('MMM DD, YYYY')}
          </Text>
          <Text type="secondary" style={{ fontSize: '11px' }}>
            {dayjs(date).format('HH:mm:ss')}
          </Text>
        </Space>
      ),
      sorter: (a: RiskReport, b: RiskReport) => 
        new Date(a.created_at).getTime() - new Date(b.created_at).getTime()
    },
    {
      title: 'Size',
      dataIndex: 'file_size_bytes',
      key: 'file_size_bytes',
      render: (size: number | undefined) => (
        <Text style={{ fontSize: '12px' }}>
          {size ? formatFileSize(size) : '-'}
        </Text>
      )
    },
    {
      title: 'Actions',
      key: 'actions',
      render: (record: RiskReport) => (
        <Space>
          {record.status === 'completed' && (
            <Tooltip title="Download report">
              <Button
                size="small"
                icon={<DownloadOutlined />}
                onClick={() => downloadReport(
                  record.id, 
                  `risk_report_${record.request.report_type}_${dayjs(record.created_at).format('YYYYMMDD')}.${record.request.format}`
                )}
              />
            </Tooltip>
          )}
          
          {record.preview_data && (
            <Tooltip title="Preview report">
              <Button
                size="small"
                icon={<EyeOutlined />}
                onClick={() => previewReportData(record)}
              />
            </Tooltip>
          )}
          
          {record.status === 'generating' && (
            <Popconfirm
              title="Cancel report generation?"
              onConfirm={() => cancelReport(record.id)}
            >
              <Tooltip title="Cancel generation">
                <Button
                  size="small"
                  icon={<DeleteOutlined />}
                  danger
                />
              </Tooltip>
            </Popconfirm>
          )}
        </Space>
      )
    }
  ];

  // Initial data fetch
  useEffect(() => {
    fetchReports();
  }, [portfolioId]);

  // Use reports from hook if available
  const currentReports = hookReports.length > 0 ? hookReports : reports;

  return (
    <div className={className}>
      {/* Report Generation Form */}
      <Card
        title={
          <Space>
            <FileTextOutlined />
            Generate Risk Report
          </Space>
        }
        style={{ marginBottom: 16 }}
      >
        <Form
          form={form}
          layout="vertical"
          onFinish={generateReport}
          initialValues={{
            reportType: 'daily',
            format: 'pdf',
            includeCharts: true,
            includeRecommendations: true,
            sections: availableSections
          }}
        >
          <Row gutter={16}>
            <Col xs={24} sm={12} md={8}>
              <Form.Item
                name="reportType"
                label="Report Type"
                rules={[{ required: true, message: 'Please select a report type' }]}
              >
                <Select>
                  {reportTypes.map(type => (
                    <Option key={type.value} value={type.value}>
                      <div>
                        <Text strong>{type.label}</Text>
                        <br />
                        <Text type="secondary" style={{ fontSize: '12px' }}>
                          {type.description}
                        </Text>
                      </div>
                    </Option>
                  ))}
                </Select>
              </Form.Item>
            </Col>
            
            <Col xs={24} sm={12} md={8}>
              <Form.Item
                name="format"
                label="Format"
                rules={[{ required: true, message: 'Please select a format' }]}
              >
                <Select>
                  {formats.map(format => (
                    <Option key={format.value} value={format.value}>
                      <Space>
                        <span style={{ color: format.color }}>{format.icon}</span>
                        {format.label}
                      </Space>
                    </Option>
                  ))}
                </Select>
              </Form.Item>
            </Col>
            
            <Col xs={24} sm={12} md={8}>
              <Form.Item
                name="dateRange"
                label="Date Range"
                tooltip="Leave empty for default range based on report type"
              >
                <RangePicker style={{ width: '100%' }} />
              </Form.Item>
            </Col>
          </Row>

          <Row gutter={16}>
            <Col xs={24} sm={12}>
              <Form.Item name="includeCharts" valuePropName="checked">
                <Switch /> Include Charts and Visualizations
              </Form.Item>
            </Col>
            <Col xs={24} sm={12}>
              <Form.Item name="includeRecommendations" valuePropName="checked">
                <Switch /> Include Risk Recommendations
              </Form.Item>
            </Col>
          </Row>

          <Form.Item
            noStyle
            shouldUpdate={(prevValues, currentValues) => 
              prevValues.reportType !== currentValues.reportType
            }
          >
            {({ getFieldValue }) =>
              getFieldValue('reportType') === 'regulatory' && (
                <Form.Item
                  name="regulatoryFramework"
                  label="Regulatory Framework"
                >
                  <Select placeholder="Select regulatory framework">
                    {regulatoryFrameworks.map(framework => (
                      <Option key={framework.value} value={framework.value}>
                        {framework.label}
                      </Option>
                    ))}
                  </Select>
                </Form.Item>
              )
            }
          </Form.Item>

          <Form.Item>
            <Space>
              <Button 
                type="primary" 
                htmlType="submit" 
                loading={loading}
                icon={<PlusOutlined />}
              >
                Generate Report
              </Button>
              <Button onClick={() => form.resetFields()}>
                Reset
              </Button>
            </Space>
          </Form.Item>
        </Form>
      </Card>

      {/* Error Alert */}
      {error && (
        <Alert
          message="Report Generation Error"
          description={error}
          type="error"
          showIcon
          closable
          onClose={() => setError(null)}
          style={{ marginBottom: 16 }}
        />
      )}

      {/* Reports Summary */}
      <Row gutter={16} style={{ marginBottom: 16 }}>
        <Col xs={24} sm={6}>
          <Card size="small">
            <Statistic
              title="Total Reports"
              value={currentReports.length}
              prefix={<FileTextOutlined />}
              loading={fetchingReports}
            />
          </Card>
        </Col>
        <Col xs={24} sm={6}>
          <Card size="small">
            <Statistic
              title="Generating"
              value={generatingReports.length}
              prefix={<LoadingOutlined />}
              valueStyle={{ color: '#1890ff' }}
              loading={fetchingReports}
            />
          </Card>
        </Col>
        <Col xs={24} sm={6}>
          <Card size="small">
            <Statistic
              title="Completed"
              value={completedReports.length}
              prefix={<CheckCircleOutlined />}
              valueStyle={{ color: '#52c41a' }}
              loading={fetchingReports}
            />
          </Card>
        </Col>
        <Col xs={24} sm={6}>
          <Card size="small">
            <Statistic
              title="Failed"
              value={failedReports.length}
              prefix={<ExclamationCircleOutlined />}
              valueStyle={{ color: '#ff4d4f' }}
              loading={fetchingReports}
            />
          </Card>
        </Col>
      </Row>

      {/* Reports Table */}
      <Card
        title={
          <Space>
            <FileTextOutlined />
            Generated Reports
          </Space>
        }
        extra={
          <Button 
            icon={<ReloadOutlined />}
            onClick={fetchReports}
            loading={fetchingReports}
            size="small"
          >
            Refresh
          </Button>
        }
      >
        <Table<RiskReport>
          columns={columns}
          dataSource={currentReports}
          rowKey="id"
          loading={fetchingReports}
          size="small"
          pagination={{
            pageSize: 10,
            showSizeChanger: true,
            showQuickJumper: true,
            showTotal: (total, range) => `${range[0]}-${range[1]} of ${total} reports`
          }}
        />
      </Card>

      {/* Preview Modal */}
      <Modal
        title={
          <Space>
            <EyeOutlined />
            Report Preview
          </Space>
        }
        open={previewVisible}
        onCancel={() => setPreviewVisible(false)}
        footer={[
          <Button 
            key="download" 
            type="primary" 
            icon={<DownloadOutlined />}
            onClick={() => {
              if (previewReport) {
                downloadReport(
                  previewReport.id,
                  `risk_report_${previewReport.request.report_type}_${dayjs(previewReport.created_at).format('YYYYMMDD')}.${previewReport.request.format}`
                );
              }
            }}
          >
            Download
          </Button>,
          <Button key="close" onClick={() => setPreviewVisible(false)}>
            Close
          </Button>
        ]}
        width={800}
      >
        {previewReport && (
          <div>
            <Paragraph>
              <Text strong>Report Type:</Text> {previewReport.request.report_type.replace('_', ' ').toUpperCase()}
            </Paragraph>
            <Paragraph>
              <Text strong>Format:</Text> {previewReport.request.format.toUpperCase()}
            </Paragraph>
            <Paragraph>
              <Text strong>Generated:</Text> {dayjs(previewReport.created_at).format('MMMM DD, YYYY HH:mm:ss')}
            </Paragraph>
            {previewReport.completed_at && (
              <Paragraph>
                <Text strong>Completed:</Text> {dayjs(previewReport.completed_at).format('MMMM DD, YYYY HH:mm:ss')}
              </Paragraph>
            )}
            
            <Divider />
            
            <Title level={5}>Report Sections</Title>
            <List
              size="small"
              dataSource={previewReport.request.sections}
              renderItem={(section) => (
                <List.Item>
                  <Space>
                    <Avatar size="small" style={{ backgroundColor: '#1890ff' }}>
                      {section.section_type[0].toUpperCase()}
                    </Avatar>
                    <Text>{section.section_type.replace('_', ' ').toUpperCase()}</Text>
                    <Tag>{section.detail_level}</Tag>
                    {section.enabled ? (
                      <CheckCircleOutlined style={{ color: '#52c41a' }} />
                    ) : (
                      <ExclamationCircleOutlined style={{ color: '#faad14' }} />
                    )}
                  </Space>
                </List.Item>
              )}
            />
            
            {previewReport.preview_data && (
              <div style={{ marginTop: 16 }}>
                <Title level={5}>Preview Data</Title>
                <pre style={{ 
                  background: '#f5f5f5', 
                  padding: '12px', 
                  borderRadius: '4px',
                  fontSize: '12px',
                  overflow: 'auto',
                  maxHeight: '200px'
                }}>
                  {JSON.stringify(previewReport.preview_data, null, 2)}
                </pre>
              </div>
            )}
          </div>
        )}
      </Modal>
    </div>
  );
};

export default RiskReportGenerator;