import React, { useState, useEffect } from 'react'
import {
  Card,
  Row,
  Col,
  Button,
  Form,
  Select,
  DatePicker,
  Checkbox,
  Table,
  Progress,
  Tag,
  Space,
  Typography,
  Alert,
  Modal,
  message,
  Upload,
  Divider,
  Steps,
  Result,
  List,
  Tooltip,
  Switch,
  InputNumber,
  Input
} from 'antd'
import {
  DownloadOutlined,
  CloudDownloadOutlined,
  FileExcelOutlined,
  FilePdfOutlined,
  FileTextOutlined,
  UploadOutlined,
  ReloadOutlined,
  DeleteOutlined,
  EyeOutlined,
  SettingOutlined,
  ScheduleOutlined,
  ShareAltOutlined,
  HistoryOutlined
} from '@ant-design/icons'
import dayjs from 'dayjs'

const { Title, Text, Paragraph } = Typography
const { RangePicker } = DatePicker
const { Step } = Steps
const { TextArea } = Input

interface ExportJob {
  jobId: string
  name: string
  format: 'excel' | 'pdf' | 'csv' | 'json' | 'parquet'
  status: 'queued' | 'processing' | 'completed' | 'failed' | 'cancelled'
  progress: number
  createdAt: string
  completedAt?: string
  fileSize?: number
  downloadUrl?: string
  errorMessage?: string
  parameters: ExportParameters
}

interface ExportParameters {
  dateRange: {
    start: string
    end: string
  }
  dataTypes: string[]
  strategies: string[]
  metrics: string[]
  format: string
  includeCharts: boolean
  includeRawData: boolean
  compression: boolean
  customFilters: Record<string, any>
}

interface ReportTemplate {
  templateId: string
  name: string
  description: string
  category: 'performance' | 'risk' | 'compliance' | 'custom'
  parameters: ExportParameters
  isDefault: boolean
  lastUsed?: string
  usageCount: number
}

interface ScheduledReport {
  scheduleId: string
  name: string
  template: string
  frequency: 'daily' | 'weekly' | 'monthly' | 'quarterly'
  nextRun: string
  lastRun?: string
  status: 'active' | 'paused' | 'failed'
  recipients: string[]
  parameters: ExportParameters
}

const DataExportReporting: React.FC = () => {
  const [exportJobs, setExportJobs] = useState<ExportJob[]>([])
  const [reportTemplates, setReportTemplates] = useState<ReportTemplate[]>([])
  const [scheduledReports, setScheduledReports] = useState<ScheduledReport[]>([])
  const [exportModalVisible, setExportModalVisible] = useState(false)
  const [templateModalVisible, setTemplateModalVisible] = useState(false)
  const [scheduleModalVisible, setScheduleModalVisible] = useState(false)
  const [currentStep, setCurrentStep] = useState(0)
  const [loading, setLoading] = useState(false)
  const [selectedTemplate, setSelectedTemplate] = useState<ReportTemplate | null>(null)
  const [form] = Form.useForm()
  const [templateForm] = Form.useForm()
  const [scheduleForm] = Form.useForm()

  const apiUrl = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8001'

  useEffect(() => {
    loadExportJobs()
    loadReportTemplates()
    loadScheduledReports()

    // Polling for active export jobs
    const interval = setInterval(() => {
      const activeJobs = exportJobs.filter(job => 
        job.status === 'queued' || job.status === 'processing'
      )
      if (activeJobs.length > 0) {
        loadExportJobs()
      }
    }, 5000)

    return () => clearInterval(interval)
  }, [])

  const loadExportJobs = async () => {
    try {
      const response = await fetch(`${apiUrl}/api/v1/performance/export/jobs`)
      if (response.ok) {
        const data = await response.json()
        setExportJobs(data.jobs || [])
      }
    } catch (error) {
      console.error('Failed to load export jobs:', error)
    }
  }

  const loadReportTemplates = async () => {
    try {
      const response = await fetch(`${apiUrl}/api/v1/performance/templates`)
      if (response.ok) {
        const data = await response.json()
        setReportTemplates(data.templates || [])
      }
    } catch (error) {
      console.error('Failed to load report templates:', error)
    }
  }

  const loadScheduledReports = async () => {
    try {
      const response = await fetch(`${apiUrl}/api/v1/performance/scheduled-reports`)
      if (response.ok) {
        const data = await response.json()
        setScheduledReports(data.reports || [])
      }
    } catch (error) {
      console.error('Failed to load scheduled reports:', error)
    }
  }

  const startExport = async (parameters: ExportParameters) => {
    setLoading(true)
    try {
      const response = await fetch(`${apiUrl}/api/v1/performance/export/start`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(parameters)
      })

      if (response.ok) {
        const result = await response.json()
        message.success(`Export job started: ${result.jobId}`)
        setExportModalVisible(false)
        setCurrentStep(0)
        form.resetFields()
        loadExportJobs()
      } else {
        const error = await response.json()
        message.error(`Failed to start export: ${error.detail}`)
      }
    } catch (error) {
      message.error('Failed to start export')
      console.error('Export start error:', error)
    } finally {
      setLoading(false)
    }
  }

  const cancelExport = async (jobId: string) => {
    try {
      const response = await fetch(`${apiUrl}/api/v1/performance/export/cancel/${jobId}`, {
        method: 'POST'
      })

      if (response.ok) {
        message.success('Export cancelled')
        loadExportJobs()
      } else {
        message.error('Failed to cancel export')
      }
    } catch (error) {
      message.error('Failed to cancel export')
      console.error('Cancel export error:', error)
    }
  }

  const downloadExport = async (job: ExportJob) => {
    try {
      if (job.downloadUrl) {
        window.open(job.downloadUrl, '_blank')
        message.success('Download started')
      } else {
        const response = await fetch(`${apiUrl}/api/v1/performance/export/download/${job.jobId}`)
        if (response.ok) {
          const blob = await response.blob()
          const url = window.URL.createObjectURL(blob)
          const link = document.createElement('a')
          link.href = url
          link.download = `${job.name}.${job.format}`
          link.click()
          window.URL.revokeObjectURL(url)
          message.success('Download completed')
        } else {
          message.error('Failed to download file')
        }
      }
    } catch (error) {
      message.error('Download failed')
      console.error('Download error:', error)
    }
  }

  const saveTemplate = async (templateData: any) => {
    try {
      const response = await fetch(`${apiUrl}/api/v1/performance/templates`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(templateData)
      })

      if (response.ok) {
        message.success('Template saved successfully')
        setTemplateModalVisible(false)
        templateForm.resetFields()
        loadReportTemplates()
      } else {
        message.error('Failed to save template')
      }
    } catch (error) {
      message.error('Failed to save template')
      console.error('Save template error:', error)
    }
  }

  const scheduleReport = async (scheduleData: any) => {
    try {
      const response = await fetch(`${apiUrl}/api/v1/performance/schedule-report`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(scheduleData)
      })

      if (response.ok) {
        message.success('Report scheduled successfully')
        setScheduleModalVisible(false)
        scheduleForm.resetFields()
        loadScheduledReports()
      } else {
        message.error('Failed to schedule report')
      }
    } catch (error) {
      message.error('Failed to schedule report')
      console.error('Schedule report error:', error)
    }
  }

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'completed': return 'success'
      case 'processing': return 'processing'
      case 'failed': return 'error'
      case 'cancelled': return 'default'
      default: return 'default'
    }
  }

  const getFormatIcon = (format: string) => {
    switch (format) {
      case 'excel': return <FileExcelOutlined style={{ color: '#52c41a' }} />
      case 'pdf': return <FilePdfOutlined style={{ color: '#ff4d4f' }} />
      case 'csv': return <FileTextOutlined style={{ color: '#1890ff' }} />
      case 'json': return <FileTextOutlined style={{ color: '#faad14' }} />
      case 'parquet': return <FileTextOutlined style={{ color: '#722ed1' }} />
      default: return <FileTextOutlined />
    }
  }

  const exportJobColumns = [
    {
      title: 'Name',
      dataIndex: 'name',
      key: 'name',
      render: (text: string, record: ExportJob) => (
        <Space>
          {getFormatIcon(record.format)}
          <Text strong>{text}</Text>
          <Tag>{record.format.toUpperCase()}</Tag>
        </Space>
      )
    },
    {
      title: 'Status',
      dataIndex: 'status',
      key: 'status',
      render: (status: string) => (
        <Tag color={getStatusColor(status)}>{status.toUpperCase()}</Tag>
      )
    },
    {
      title: 'Progress',
      dataIndex: 'progress',
      key: 'progress',
      render: (progress: number, record: ExportJob) => (
        record.status === 'processing' ? (
          <Progress percent={progress} size="small" />
        ) : record.status === 'completed' ? (
          <Progress percent={100} size="small" status="success" />
        ) : record.status === 'failed' ? (
          <Progress percent={0} size="small" status="exception" />
        ) : null
      )
    },
    {
      title: 'Created',
      dataIndex: 'createdAt',
      key: 'createdAt',
      render: (date: string) => dayjs(date).format('MM/DD/YY HH:mm')
    },
    {
      title: 'Size',
      dataIndex: 'fileSize',
      key: 'fileSize',
      render: (size: number) => size ? `${(size / 1024 / 1024).toFixed(2)} MB` : '-'
    },
    {
      title: 'Actions',
      key: 'actions',
      render: (_, record: ExportJob) => (
        <Space size="small">
          {record.status === 'completed' && (
            <Tooltip title="Download">
              <Button 
                size="small" 
                icon={<DownloadOutlined />}
                onClick={() => downloadExport(record)}
              />
            </Tooltip>
          )}
          {(record.status === 'queued' || record.status === 'processing') && (
            <Tooltip title="Cancel">
              <Button 
                size="small" 
                danger
                icon={<DeleteOutlined />}
                onClick={() => cancelExport(record.jobId)}
              />
            </Tooltip>
          )}
          <Tooltip title="View Details">
            <Button 
              size="small" 
              icon={<EyeOutlined />}
              onClick={() => {
                Modal.info({
                  title: 'Export Job Details',
                  content: (
                    <div>
                      <p><strong>Job ID:</strong> {record.jobId}</p>
                      <p><strong>Format:</strong> {record.format}</p>
                      <p><strong>Date Range:</strong> {record.parameters.dateRange.start} to {record.parameters.dateRange.end}</p>
                      <p><strong>Data Types:</strong> {record.parameters.dataTypes.join(', ')}</p>
                      {record.errorMessage && (
                        <Alert message="Error" description={record.errorMessage} type="error" />
                      )}
                    </div>
                  )
                })
              }}
            />
          </Tooltip>
        </Space>
      )
    }
  ]

  const templateColumns = [
    {
      title: 'Name',
      dataIndex: 'name',
      key: 'name',
      render: (text: string, record: ReportTemplate) => (
        <Space>
          <Text strong>{text}</Text>
          {record.isDefault && <Tag color="blue">Default</Tag>}
        </Space>
      )
    },
    {
      title: 'Category',
      dataIndex: 'category',
      key: 'category',
      render: (category: string) => (
        <Tag color={
          category === 'performance' ? 'green' :
          category === 'risk' ? 'orange' :
          category === 'compliance' ? 'red' : 'default'
        }>
          {category.toUpperCase()}
        </Tag>
      )
    },
    {
      title: 'Usage Count',
      dataIndex: 'usageCount',
      key: 'usageCount'
    },
    {
      title: 'Last Used',
      dataIndex: 'lastUsed',
      key: 'lastUsed',
      render: (date: string) => date ? dayjs(date).format('MM/DD/YY') : 'Never'
    },
    {
      title: 'Actions',
      key: 'actions',
      render: (_, record: ReportTemplate) => (
        <Space size="small">
          <Button 
            size="small" 
            onClick={() => {
              form.setFieldsValue(record.parameters)
              setSelectedTemplate(record)
              setExportModalVisible(true)
            }}
          >
            Use Template
          </Button>
          <Button 
            size="small" 
            icon={<EyeOutlined />}
            onClick={() => {
              Modal.info({
                title: record.name,
                content: (
                  <div>
                    <p>{record.description}</p>
                    <Divider />
                    <p><strong>Data Types:</strong> {record.parameters.dataTypes.join(', ')}</p>
                    <p><strong>Metrics:</strong> {record.parameters.metrics.join(', ')}</p>
                  </div>
                )
              })
            }}
          />
        </Space>
      )
    }
  ]

  const exportSteps = [
    {
      title: 'Data Selection',
      content: (
        <Row gutter={16}>
          <Col span={12}>
            <Form.Item name="dateRange" label="Date Range" rules={[{ required: true }]}>
              <RangePicker 
                style={{ width: '100%' }}
                presets={[
                  { label: 'Last 1M', value: [dayjs().subtract(1, 'month'), dayjs()] },
                  { label: 'Last 3M', value: [dayjs().subtract(3, 'month'), dayjs()] },
                  { label: 'Last 6M', value: [dayjs().subtract(6, 'month'), dayjs()] },
                  { label: 'Last 1Y', value: [dayjs().subtract(1, 'year'), dayjs()] }
                ]}
              />
            </Form.Item>
          </Col>
          <Col span={12}>
            <Form.Item name="dataTypes" label="Data Types" rules={[{ required: true }]}>
              <Select mode="multiple" placeholder="Select data types">
                <Select.Option value="performance">Performance Metrics</Select.Option>
                <Select.Option value="trades">Trade History</Select.Option>
                <Select.Option value="positions">Position Data</Select.Option>
                <Select.Option value="risk">Risk Metrics</Select.Option>
                <Select.Option value="market_data">Market Data</Select.Option>
              </Select>
            </Form.Item>
          </Col>
          <Col span={12}>
            <Form.Item name="strategies" label="Strategies">
              <Select mode="multiple" placeholder="All strategies" allowClear>
                <Select.Option value="strategy1">Strategy Alpha</Select.Option>
                <Select.Option value="strategy2">Strategy Beta</Select.Option>
                <Select.Option value="strategy3">Strategy Gamma</Select.Option>
              </Select>
            </Form.Item>
          </Col>
          <Col span={12}>
            <Form.Item name="metrics" label="Metrics">
              <Select mode="multiple" placeholder="Select metrics">
                <Select.Option value="returns">Returns</Select.Option>
                <Select.Option value="sharpe">Sharpe Ratio</Select.Option>
                <Select.Option value="drawdown">Drawdown</Select.Option>
                <Select.Option value="volatility">Volatility</Select.Option>
                <Select.Option value="var">Value at Risk</Select.Option>
              </Select>
            </Form.Item>
          </Col>
        </Row>
      )
    },
    {
      title: 'Format & Options',
      content: (
        <Row gutter={16}>
          <Col span={12}>
            <Form.Item name="format" label="Export Format" rules={[{ required: true }]}>
              <Select placeholder="Select format">
                <Select.Option value="excel">
                  <Space><FileExcelOutlined /> Excel (.xlsx)</Space>
                </Select.Option>
                <Select.Option value="pdf">
                  <Space><FilePdfOutlined /> PDF Report</Space>
                </Select.Option>
                <Select.Option value="csv">
                  <Space><FileTextOutlined /> CSV Data</Space>
                </Select.Option>
                <Select.Option value="json">
                  <Space><FileTextOutlined /> JSON Data</Space>
                </Select.Option>
                <Select.Option value="parquet">
                  <Space><FileTextOutlined /> Parquet (NautilusTrader)</Space>
                </Select.Option>
              </Select>
            </Form.Item>
          </Col>
          <Col span={12}>
            <Form.Item name="name" label="Export Name" rules={[{ required: true }]}>
              <Input placeholder="Performance Report" />
            </Form.Item>
          </Col>
          <Col span={24}>
            <Form.Item name="includeCharts" valuePropName="checked">
              <Checkbox>Include Charts and Visualizations</Checkbox>
            </Form.Item>
            <Form.Item name="includeRawData" valuePropName="checked">
              <Checkbox>Include Raw Trade Data</Checkbox>
            </Form.Item>
            <Form.Item name="compression" valuePropName="checked">
              <Checkbox>Enable Compression</Checkbox>
            </Form.Item>
          </Col>
        </Row>
      )
    },
    {
      title: 'Review & Export',
      content: (
        <Result
          icon={<CloudDownloadOutlined style={{ color: '#1890ff' }} />}
          title="Ready to Export"
          subTitle="Review your export configuration and click 'Start Export' to begin processing."
          extra={
            <div style={{ textAlign: 'left', maxWidth: 400, margin: '0 auto' }}>
              <Title level={5}>Export Summary:</Title>
              <Text>• Date Range: {form.getFieldValue('dateRange')?.[0]?.format('MM/DD/YYYY')} - {form.getFieldValue('dateRange')?.[1]?.format('MM/DD/YYYY')}</Text><br />
              <Text>• Format: {form.getFieldValue('format')?.toUpperCase()}</Text><br />
              <Text>• Data Types: {form.getFieldValue('dataTypes')?.join(', ')}</Text><br />
              <Text>• Include Charts: {form.getFieldValue('includeCharts') ? 'Yes' : 'No'}</Text>
            </div>
          }
        />
      )
    }
  ]

  return (
    <div style={{ padding: 24 }}>
      <Row gutter={[16, 16]}>
        {/* Header */}
        <Col xs={24}>
          <Card>
            <Row justify="space-between" align="middle">
              <Col>
                <Title level={3} style={{ margin: 0 }}>
                  <DownloadOutlined style={{ marginRight: 8 }} />
                  Data Export & Reporting
                </Title>
                <Text type="secondary">
                  Export performance data and generate custom reports
                </Text>
              </Col>
              <Col>
                <Space>
                  <Button 
                    icon={<ReloadOutlined />}
                    onClick={() => {
                      loadExportJobs()
                      loadReportTemplates()
                      loadScheduledReports()
                    }}
                  >
                    Refresh
                  </Button>
                  <Button 
                    icon={<SettingOutlined />}
                    onClick={() => setTemplateModalVisible(true)}
                  >
                    Templates
                  </Button>
                  <Button 
                    icon={<ScheduleOutlined />}
                    onClick={() => setScheduleModalVisible(true)}
                  >
                    Schedule Report
                  </Button>
                  <Button 
                    type="primary"
                    icon={<DownloadOutlined />}
                    onClick={() => setExportModalVisible(true)}
                  >
                    New Export
                  </Button>
                </Space>
              </Col>
            </Row>
          </Card>
        </Col>

        {/* Quick Export Templates */}
        <Col xs={24}>
          <Card title="Quick Export Templates" size="small">
            <Row gutter={[16, 16]}>
              {reportTemplates.filter(t => t.isDefault).map(template => (
                <Col xs={24} sm={12} md={8} lg={6} key={template.templateId}>
                  <Card 
                    size="small" 
                    hoverable
                    onClick={() => {
                      form.setFieldsValue(template.parameters)
                      setSelectedTemplate(template)
                      setExportModalVisible(true)
                    }}
                  >
                    <Space direction="vertical" size="small" style={{ width: '100%' }}>
                      <Text strong>{template.name}</Text>
                      <Text type="secondary" style={{ fontSize: 12 }}>
                        {template.description}
                      </Text>
                      <Tag color={
                        template.category === 'performance' ? 'green' :
                        template.category === 'risk' ? 'orange' : 'blue'
                      }>
                        {template.category}
                      </Tag>
                    </Space>
                  </Card>
                </Col>
              ))}
            </Row>
          </Card>
        </Col>

        {/* Export Jobs */}
        <Col xs={24}>
          <Card title="Export Jobs" extra={<Text type="secondary">{exportJobs.length} jobs</Text>}>
            <Table
              dataSource={exportJobs}
              columns={exportJobColumns}
              rowKey="jobId"
              pagination={{ pageSize: 10 }}
              size="middle"
            />
          </Card>
        </Col>

        {/* Report Templates Management */}
        <Col xs={24}>
          <Card title="Report Templates">
            <Table
              dataSource={reportTemplates}
              columns={templateColumns}
              rowKey="templateId"
              pagination={{ pageSize: 5 }}
              size="small"
            />
          </Card>
        </Col>

        {/* Scheduled Reports */}
        <Col xs={24}>
          <Card title="Scheduled Reports">
            <List
              dataSource={scheduledReports}
              renderItem={(report) => (
                <List.Item
                  actions={[
                    <Switch 
                      key="toggle"
                      checked={report.status === 'active'}
                      checkedChildren="Active"
                      unCheckedChildren="Paused"
                    />,
                    <Button key="edit" size="small" icon={<SettingOutlined />} />,
                    <Button key="delete" size="small" danger icon={<DeleteOutlined />} />
                  ]}
                >
                  <List.Item.Meta
                    title={report.name}
                    description={
                      <Space>
                        <Tag>{report.frequency}</Tag>
                        <Text type="secondary">Next: {dayjs(report.nextRun).format('MM/DD/YY HH:mm')}</Text>
                        <Text type="secondary">Recipients: {report.recipients.length}</Text>
                      </Space>
                    }
                  />
                </List.Item>
              )}
            />
          </Card>
        </Col>
      </Row>

      {/* Export Modal */}
      <Modal
        title="Create Data Export"
        open={exportModalVisible}
        onCancel={() => {
          setExportModalVisible(false)
          setCurrentStep(0)
          form.resetFields()
          setSelectedTemplate(null)
        }}
        footer={[
          <Button key="cancel" onClick={() => setExportModalVisible(false)}>
            Cancel
          </Button>,
          currentStep > 0 && (
            <Button key="back" onClick={() => setCurrentStep(currentStep - 1)}>
              Back
            </Button>
          ),
          currentStep < exportSteps.length - 1 ? (
            <Button key="next" type="primary" onClick={() => setCurrentStep(currentStep + 1)}>
              Next
            </Button>
          ) : (
            <Button 
              key="export" 
              type="primary" 
              loading={loading}
              onClick={() => {
                form.validateFields().then(values => {
                  const [startDate, endDate] = values.dateRange
                  const parameters: ExportParameters = {
                    ...values,
                    dateRange: {
                      start: startDate.format('YYYY-MM-DD'),
                      end: endDate.format('YYYY-MM-DD')
                    },
                    customFilters: {}
                  }
                  startExport(parameters)
                })
              }}
            >
              Start Export
            </Button>
          )
        ]}
        width={800}
      >
        <Steps current={currentStep} style={{ marginBottom: 24 }}>
          {exportSteps.map((step, index) => (
            <Step key={index} title={step.title} />
          ))}
        </Steps>
        
        <Form form={form} layout="vertical">
          {exportSteps[currentStep].content}
        </Form>
      </Modal>

      {/* Template Management Modal */}
      <Modal
        title="Manage Report Templates"
        open={templateModalVisible}
        onCancel={() => setTemplateModalVisible(false)}
        footer={[
          <Button key="cancel" onClick={() => setTemplateModalVisible(false)}>
            Close
          </Button>,
          <Button key="save" type="primary" onClick={() => templateForm.submit()}>
            Save Template
          </Button>
        ]}
        width={600}
      >
        <Form 
          form={templateForm} 
          layout="vertical"
          onFinish={saveTemplate}
        >
          <Form.Item name="name" label="Template Name" rules={[{ required: true }]}>
            <Input placeholder="My Custom Report Template" />
          </Form.Item>
          <Form.Item name="description" label="Description">
            <TextArea rows={3} placeholder="Describe what this template is used for..." />
          </Form.Item>
          <Form.Item name="category" label="Category" rules={[{ required: true }]}>
            <Select>
              <Select.Option value="performance">Performance</Select.Option>
              <Select.Option value="risk">Risk</Select.Option>
              <Select.Option value="compliance">Compliance</Select.Option>
              <Select.Option value="custom">Custom</Select.Option>
            </Select>
          </Form.Item>
          <Form.Item name="isDefault" valuePropName="checked">
            <Checkbox>Show in Quick Templates</Checkbox>
          </Form.Item>
        </Form>
      </Modal>

      {/* Schedule Report Modal */}
      <Modal
        title="Schedule Automated Report"
        open={scheduleModalVisible}
        onCancel={() => setScheduleModalVisible(false)}
        footer={[
          <Button key="cancel" onClick={() => setScheduleModalVisible(false)}>
            Cancel
          </Button>,
          <Button key="schedule" type="primary" onClick={() => scheduleForm.submit()}>
            Schedule Report
          </Button>
        ]}
        width={600}
      >
        <Form 
          form={scheduleForm} 
          layout="vertical"
          onFinish={scheduleReport}
        >
          <Form.Item name="name" label="Report Name" rules={[{ required: true }]}>
            <Input placeholder="Weekly Performance Report" />
          </Form.Item>
          <Form.Item name="template" label="Template" rules={[{ required: true }]}>
            <Select placeholder="Select a template">
              {reportTemplates.map(template => (
                <Select.Option key={template.templateId} value={template.templateId}>
                  {template.name}
                </Select.Option>
              ))}
            </Select>
          </Form.Item>
          <Form.Item name="frequency" label="Frequency" rules={[{ required: true }]}>
            <Select>
              <Select.Option value="daily">Daily</Select.Option>
              <Select.Option value="weekly">Weekly</Select.Option>
              <Select.Option value="monthly">Monthly</Select.Option>
              <Select.Option value="quarterly">Quarterly</Select.Option>
            </Select>
          </Form.Item>
          <Form.Item name="recipients" label="Email Recipients" rules={[{ required: true }]}>
            <Select mode="tags" placeholder="Enter email addresses">
              <Select.Option value="admin@company.com">admin@company.com</Select.Option>
              <Select.Option value="trader@company.com">trader@company.com</Select.Option>
            </Select>
          </Form.Item>
        </Form>
      </Modal>
    </div>
  )
}

export default DataExportReporting