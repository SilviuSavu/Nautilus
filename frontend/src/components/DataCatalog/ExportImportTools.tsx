import React, { useState, useEffect } from 'react'
import {
  Card,
  Form,
  Select,
  DatePicker,
  Button,
  Space,
  Row,
  Col,
  Table,
  Progress,
  Tag,
  Alert,
  Upload,
  Modal,
  Checkbox,
  Input,
  InputNumber,
  Switch,
  Tooltip,
  notification,
  Tabs,
  List,
  Statistic
} from 'antd'
import {
  DownloadOutlined,
  UploadOutlined,
  FileTextOutlined,
  CheckCircleOutlined,
  CloseCircleOutlined,
  SyncOutlined,
  InboxOutlined,
  DeleteOutlined,
  EyeOutlined,
  WarningOutlined
} from '@ant-design/icons'
import type { ColumnsType } from 'antd/es/table'
import type { UploadProps } from 'antd'
import dayjs, { Dayjs } from 'dayjs'
import { dataCatalogService } from '../../services/dataCatalogService'
import {
  DataExportRequest,
  ExportResult,
  ImportRequest,
  ImportResult,
  InstrumentMetadata
} from '../../types/dataCatalog'

const { Option } = Select
const { RangePicker } = DatePicker
const { Dragger } = Upload
const { TabPane } = Tabs

interface ExportJob {
  id: string
  request: DataExportRequest
  result?: ExportResult
  status: 'pending' | 'processing' | 'completed' | 'failed'
  progress: number
  createdAt: Date
  estimatedCompletion?: Date
}

interface ImportJob {
  id: string
  request: ImportRequest
  result?: ImportResult
  status: 'pending' | 'processing' | 'completed' | 'failed'
  progress: number
  createdAt: Date
  warnings?: string[]
}

export const ExportImportTools: React.FC = () => {
  const [exportForm] = Form.useForm()
  const [importForm] = Form.useForm()
  const [exportJobs, setExportJobs] = useState<ExportJob[]>([])
  const [importJobs, setImportJobs] = useState<ImportJob[]>([])
  const [availableInstruments, setAvailableInstruments] = useState<InstrumentMetadata[]>([])
  const [loading, setLoading] = useState(false)
  const [previewModalVisible, setPreviewModalVisible] = useState(false)
  const [selectedExport, setSelectedExport] = useState<ExportJob | null>(null)
  const [uploadFiles, setUploadFiles] = useState<any[]>([])

  useEffect(() => {
    loadAvailableInstruments()
    loadExportHistory()
    loadImportHistory()
  }, [])

  const loadAvailableInstruments = async () => {
    try {
      const catalog = await dataCatalogService.getCatalog()
      setAvailableInstruments(catalog.instruments)
    } catch (error) {
      console.error('Failed to load instruments:', error)
    }
  }

  const loadExportHistory = async () => {
    // In a real implementation, this would fetch export history from the API
    // For now, we'll use mock data
    const mockExports: ExportJob[] = [
      {
        id: 'exp_001',
        request: {
          instrumentIds: ['EURUSD.SIM'],
          format: 'parquet',
          dateRange: { start: '2024-01-01', end: '2024-01-31' },
          compression: true
        },
        result: {
          exportId: 'exp_001',
          success: true,
          filePath: '/exports/eurusd_jan2024.parquet',
          downloadUrl: '/api/downloads/exp_001',
          recordCount: 1250000,
          fileSize: 45678912,
          format: 'parquet',
          createdAt: new Date('2024-01-31T10:30:00'),
          completedAt: new Date('2024-01-31T10:45:00')
        },
        status: 'completed',
        progress: 100,
        createdAt: new Date('2024-01-31T10:30:00')
      },
      {
        id: 'exp_002',
        request: {
          instrumentIds: ['GBPUSD.SIM', 'USDJPY.SIM'],
          format: 'csv',
          dateRange: { start: '2024-02-01', end: '2024-02-15' },
          compression: false
        },
        status: 'processing',
        progress: 65,
        createdAt: new Date(),
        estimatedCompletion: new Date(Date.now() + 300000) // 5 minutes from now
      }
    ]
    
    setExportJobs(mockExports)
  }

  const loadImportHistory = async () => {
    // Mock import history
    const mockImports: ImportJob[] = [
      {
        id: 'imp_001',
        request: {
          filePath: '/uploads/custom_data.csv',
          format: 'csv',
          instrumentId: 'CUSTOM.SIM',
          venue: 'SIM',
          dataType: 'bar',
          validateData: true
        },
        result: {
          importId: 'imp_001',
          success: true,
          recordCount: 50000,
          instrumentId: 'CUSTOM.SIM',
          processedAt: new Date('2024-01-30T14:20:00')
        },
        status: 'completed',
        progress: 100,
        createdAt: new Date('2024-01-30T14:15:00')
      }
    ]
    
    setImportJobs(mockImports)
  }

  const handleExport = async (values: any) => {
    try {
      setLoading(true)
      
      const exportRequest: DataExportRequest = {
        instrumentIds: values.instrumentIds,
        format: values.format,
        dateRange: {
          start: values.dateRange[0].format('YYYY-MM-DD'),
          end: values.dateRange[1].format('YYYY-MM-DD')
        },
        timeframes: values.timeframes,
        compression: values.compression,
        includeMetadata: values.includeMetadata,
        maxRecords: values.maxRecords
      }

      const result = await dataCatalogService.exportData(exportRequest)
      
      const newJob: ExportJob = {
        id: result.exportId,
        request: exportRequest,
        result,
        status: result.success ? 'completed' : 'failed',
        progress: result.success ? 100 : 0,
        createdAt: new Date()
      }
      
      setExportJobs([newJob, ...exportJobs])
      exportForm.resetFields()
      
      notification.success({
        message: 'Export started',
        description: `Export job ${result.exportId} has been queued`
      })
    } catch (error) {
      notification.error({
        message: 'Export failed',
        description: 'Failed to start export job'
      })
    } finally {
      setLoading(false)
    }
  }

  const handleImport = async (values: any) => {
    try {
      setLoading(true)
      
      if (uploadFiles.length === 0) {
        notification.error({
          message: 'No file selected',
          description: 'Please select a file to import'
        })
        return
      }

      const importRequest: ImportRequest = {
        filePath: uploadFiles[0].response?.filePath || uploadFiles[0].name,
        format: values.format,
        instrumentId: values.instrumentId,
        venue: values.venue,
        dataType: values.dataType,
        validateData: values.validateData,
        overwrite: values.overwrite
      }

      const result = await dataCatalogService.importData(importRequest)
      
      const newJob: ImportJob = {
        id: result.importId,
        request: importRequest,
        result,
        status: result.success ? 'completed' : 'failed',
        progress: result.success ? 100 : 0,
        createdAt: new Date(),
        warnings: result.warnings
      }
      
      setImportJobs([newJob, ...importJobs])
      importForm.resetFields()
      setUploadFiles([])
      
      notification.success({
        message: 'Import completed',
        description: `Successfully imported ${result.recordCount?.toLocaleString()} records`
      })
    } catch (error) {
      notification.error({
        message: 'Import failed',
        description: 'Failed to import data'
      })
    } finally {
      setLoading(false)
    }
  }

  const downloadExport = (job: ExportJob) => {
    if (job.result?.downloadUrl) {
      window.open(job.result.downloadUrl, '_blank')
    }
  }

  const previewExport = (job: ExportJob) => {
    setSelectedExport(job)
    setPreviewModalVisible(true)
  }

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'completed': return 'success'
      case 'failed': return 'error'
      case 'processing': return 'processing'
      case 'pending': return 'default'
      default: return 'default'
    }
  }

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'completed': return <CheckCircleOutlined />
      case 'failed': return <CloseCircleOutlined />
      case 'processing': return <SyncOutlined spin />
      default: return null
    }
  }

  const formatFileSize = (bytes: number) => {
    if (bytes === 0) return '0 Bytes'
    const k = 1024
    const sizes = ['Bytes', 'KB', 'MB', 'GB']
    const i = Math.floor(Math.log(bytes) / Math.log(k))
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i]
  }

  const exportColumns: ColumnsType<ExportJob> = [
    {
      title: 'Job ID',
      dataIndex: 'id',
      key: 'id',
      width: 120,
      render: (id) => <code>{id}</code>
    },
    {
      title: 'Instruments',
      key: 'instruments',
      render: (_, record) => (
        <div>
          {record.request.instrumentIds.slice(0, 2).map(id => (
            <Tag key={id}>{id}</Tag>
          ))}
          {record.request.instrumentIds.length > 2 && (
            <Tag>+{record.request.instrumentIds.length - 2} more</Tag>
          )}
        </div>
      )
    },
    {
      title: 'Format',
      dataIndex: ['request', 'format'],
      key: 'format',
      render: (format) => <Tag color="blue">{format.toUpperCase()}</Tag>
    },
    {
      title: 'Date Range',
      key: 'dateRange',
      render: (_, record) => (
        <div style={{ fontSize: '12px' }}>
          <div>{record.request.dateRange.start}</div>
          <div>to {record.request.dateRange.end}</div>
        </div>
      )
    },
    {
      title: 'Status',
      dataIndex: 'status',
      key: 'status',
      render: (status, record) => (
        <Space direction="vertical">
          <Tag color={getStatusColor(status)} icon={getStatusIcon(status)}>
            {status.toUpperCase()}
          </Tag>
          {status === 'processing' && (
            <Progress percent={record.progress} style={{ width: 80 }} />
          )}
        </Space>
      )
    },
    {
      title: 'Records',
      key: 'records',
      render: (_, record) => record.result?.recordCount?.toLocaleString() || '-'
    },
    {
      title: 'File Size',
      key: 'fileSize',
      render: (_, record) => record.result?.fileSize ? formatFileSize(record.result.fileSize) : '-'
    },
    {
      title: 'Created',
      dataIndex: 'createdAt',
      key: 'createdAt',
      render: (date) => dayjs(date).format('MM/DD HH:mm')
    },
    {
      title: 'Actions',
      key: 'actions',
      render: (_, record) => (
        <Space>
          {record.status === 'completed' && (
            <>
              <Tooltip title="Download">
                <Button
                 
                  icon={<DownloadOutlined />}
                  onClick={() => downloadExport(record)}
                />
              </Tooltip>
              <Tooltip title="Preview">
                <Button
                 
                  icon={<EyeOutlined />}
                  onClick={() => previewExport(record)}
                />
              </Tooltip>
            </>
          )}
          <Tooltip title="Delete">
            <Button
             
              icon={<DeleteOutlined />}
              danger
            />
          </Tooltip>
        </Space>
      )
    }
  ]

  const importColumns: ColumnsType<ImportJob> = [
    {
      title: 'Job ID',
      dataIndex: 'id',
      key: 'id',
      width: 120,
      render: (id) => <code>{id}</code>
    },
    {
      title: 'File',
      dataIndex: ['request', 'filePath'],
      key: 'filePath',
      render: (path) => path.split('/').pop()
    },
    {
      title: 'Format',
      dataIndex: ['request', 'format'],
      key: 'format',
      render: (format) => <Tag color="green">{format.toUpperCase()}</Tag>
    },
    {
      title: 'Instrument',
      dataIndex: ['request', 'instrumentId'],
      key: 'instrumentId'
    },
    {
      title: 'Status',
      dataIndex: 'status',
      key: 'status',
      render: (status, record) => (
        <Space direction="vertical">
          <Tag color={getStatusColor(status)} icon={getStatusIcon(status)}>
            {status.toUpperCase()}
          </Tag>
          {record.warnings && record.warnings.length > 0 && (
            <Tag color="orange" icon={<WarningOutlined />}>
              {record.warnings.length} warnings
            </Tag>
          )}
        </Space>
      )
    },
    {
      title: 'Records',
      key: 'records',
      render: (_, record) => record.result?.recordCount?.toLocaleString() || '-'
    },
    {
      title: 'Created',
      dataIndex: 'createdAt',
      key: 'createdAt',
      render: (date) => dayjs(date).format('MM/DD HH:mm')
    }
  ]

  const uploadProps: UploadProps = {
    name: 'file',
    multiple: false,
    accept: '.csv,.parquet,.json',
    fileList: uploadFiles,
    onChange: (info) => {
      setUploadFiles(info.fileList)
    },
    onDrop: (e) => {
      console.log('Dropped files', e.dataTransfer.files)
    }
  }

  return (
    <div>
      <Tabs defaultActiveKey="export">
        <TabPane tab="Export Data" key="export">
          <Row gutter={[24, 24]}>
            <Col span={8}>
              <Card title="Export Configuration" style={{ height: '100%' }}>
                <Form
                  form={exportForm}
                  layout="vertical"
                  onFinish={handleExport}
                  initialValues={{
                    format: 'parquet',
                    compression: true,
                    includeMetadata: true
                  }}
                >
                  <Form.Item
                    name="instrumentIds"
                    label="Instruments"
                    rules={[{ required: true, message: 'Select at least one instrument' }]}
                  >
                    <Select
                      mode="multiple"
                      placeholder="Select instruments"
                      showSearch
                      optionFilterProp="children"
                    >
                      {availableInstruments.map(inst => (
                        <Option key={inst.instrumentId} value={inst.instrumentId}>
                          {inst.symbol} ({inst.venue})
                        </Option>
                      ))}
                    </Select>
                  </Form.Item>

                  <Form.Item
                    name="dateRange"
                    label="Date Range"
                    rules={[{ required: true, message: 'Select date range' }]}
                  >
                    <RangePicker style={{ width: '100%' }} />
                  </Form.Item>

                  <Form.Item
                    name="format"
                    label="Format"
                    rules={[{ required: true }]}
                  >
                    <Select>
                      <Option value="parquet">Parquet (Recommended)</Option>
                      <Option value="csv">CSV</Option>
                      <Option value="json">JSON</Option>
                      <Option value="nautilus">NautilusTrader</Option>
                    </Select>
                  </Form.Item>

                  <Form.Item name="timeframes" label="Timeframes">
                    <Checkbox.Group>
                      <Checkbox value="1-MINUTE">1 Minute</Checkbox>
                      <Checkbox value="5-MINUTE">5 Minutes</Checkbox>
                      <Checkbox value="1-HOUR">1 Hour</Checkbox>
                      <Checkbox value="1-DAY">Daily</Checkbox>
                    </Checkbox.Group>
                  </Form.Item>

                  <Form.Item name="maxRecords" label="Max Records">
                    <InputNumber
                      placeholder="Leave empty for all"
                      style={{ width: '100%' }}
                      formatter={value => `${value}`.replace(/\B(?=(\d{3})+(?!\d))/g, ',')}
                    />
                  </Form.Item>

                  <Form.Item>
                    <Space direction="vertical" style={{ width: '100%' }}>
                      <Form.Item name="compression" valuePropName="checked" noStyle>
                        <Checkbox>Enable Compression</Checkbox>
                      </Form.Item>
                      <Form.Item name="includeMetadata" valuePropName="checked" noStyle>
                        <Checkbox>Include Metadata</Checkbox>
                      </Form.Item>
                    </Space>
                  </Form.Item>

                  <Form.Item>
                    <Button
                      type="primary"
                      htmlType="submit"
                      icon={<DownloadOutlined />}
                      loading={loading}
                      block
                    >
                      Start Export
                    </Button>
                  </Form.Item>
                </Form>
              </Card>
            </Col>

            <Col span={16}>
              <Card title="Export History">
                <Table
                  columns={exportColumns}
                  dataSource={exportJobs}
                  rowKey="id"
                  pagination={{ pageSize: 10 }}
                 
                />
              </Card>
            </Col>
          </Row>
        </TabPane>

        <TabPane tab="Import Data" key="import">
          <Row gutter={[24, 24]}>
            <Col span={8}>
              <Card title="Import Configuration" style={{ height: '100%' }}>
                <Form
                  form={importForm}
                  layout="vertical"
                  onFinish={handleImport}
                  initialValues={{
                    format: 'csv',
                    validateData: true,
                    overwrite: false
                  }}
                >
                  <Form.Item label="Upload File">
                    <Dragger {...uploadProps}>
                      <p className="ant-upload-drag-icon">
                        <InboxOutlined />
                      </p>
                      <p className="ant-upload-text">
                        Click or drag file to this area to upload
                      </p>
                      <p className="ant-upload-hint">
                        Support for .csv, .parquet, and .json files
                      </p>
                    </Dragger>
                  </Form.Item>

                  <Form.Item
                    name="format"
                    label="File Format"
                    rules={[{ required: true }]}
                  >
                    <Select>
                      <Option value="csv">CSV</Option>
                      <Option value="parquet">Parquet</Option>
                      <Option value="json">JSON</Option>
                    </Select>
                  </Form.Item>

                  <Form.Item
                    name="instrumentId"
                    label="Instrument ID"
                    rules={[{ required: true }]}
                  >
                    <Input placeholder="e.g., CUSTOM.SIM" />
                  </Form.Item>

                  <Form.Item
                    name="venue"
                    label="Venue"
                    rules={[{ required: true }]}
                  >
                    <Input placeholder="e.g., SIM" />
                  </Form.Item>

                  <Form.Item
                    name="dataType"
                    label="Data Type"
                    rules={[{ required: true }]}
                  >
                    <Select>
                      <Option value="tick">Tick Data</Option>
                      <Option value="quote">Quote Data</Option>
                      <Option value="bar">Bar Data</Option>
                    </Select>
                  </Form.Item>

                  <Form.Item>
                    <Space direction="vertical" style={{ width: '100%' }}>
                      <Form.Item name="validateData" valuePropName="checked" noStyle>
                        <Checkbox>Validate Data Quality</Checkbox>
                      </Form.Item>
                      <Form.Item name="overwrite" valuePropName="checked" noStyle>
                        <Checkbox>Overwrite Existing Data</Checkbox>
                      </Form.Item>
                    </Space>
                  </Form.Item>

                  <Form.Item>
                    <Button
                      type="primary"
                      htmlType="submit"
                      icon={<UploadOutlined />}
                      loading={loading}
                      disabled={uploadFiles.length === 0}
                      block
                    >
                      Start Import
                    </Button>
                  </Form.Item>
                </Form>
              </Card>
            </Col>

            <Col span={16}>
              <Card title="Import History">
                <Table
                  columns={importColumns}
                  dataSource={importJobs}
                  rowKey="id"
                  pagination={{ pageSize: 10 }}
                 
                />
              </Card>
            </Col>
          </Row>
        </TabPane>
      </Tabs>

      {/* Export Preview Modal */}
      <Modal
        title={`Export Details: ${selectedExport?.id}`}
        open={previewModalVisible}
        onCancel={() => setPreviewModalVisible(false)}
        footer={[
          <Button key="close" onClick={() => setPreviewModalVisible(false)}>
            Close
          </Button>,
          <Button
            key="download"
            type="primary"
            icon={<DownloadOutlined />}
            onClick={() => selectedExport && downloadExport(selectedExport)}
          >
            Download
          </Button>
        ]}
        width={800}
      >
        {selectedExport && (
          <div>
            <Row gutter={[16, 16]}>
              <Col span={12}>
                <Statistic
                  title="Records Exported"
                  value={selectedExport.result?.recordCount || 0}
                  formatter={value => value?.toLocaleString()}
                />
              </Col>
              <Col span={12}>
                <Statistic
                  title="File Size"
                  value={selectedExport.result?.fileSize ? formatFileSize(selectedExport.result.fileSize) : 'N/A'}
                />
              </Col>
            </Row>
            
            <div style={{ marginTop: 16 }}>
              <h4>Export Configuration:</h4>
              <List>
                <List.Item>
                  <strong>Instruments:</strong> {selectedExport.request.instrumentIds.join(', ')}
                </List.Item>
                <List.Item>
                  <strong>Format:</strong> {selectedExport.request.format}
                </List.Item>
                <List.Item>
                  <strong>Date Range:</strong> {selectedExport.request.dateRange.start} to {selectedExport.request.dateRange.end}
                </List.Item>
                <List.Item>
                  <strong>Compression:</strong> {selectedExport.request.compression ? 'Enabled' : 'Disabled'}
                </List.Item>
                <List.Item>
                  <strong>Metadata:</strong> {selectedExport.request.includeMetadata ? 'Included' : 'Excluded'}
                </List.Item>
              </List>
            </div>
          </div>
        )}
      </Modal>
    </div>
  )
}

export default ExportImportTools