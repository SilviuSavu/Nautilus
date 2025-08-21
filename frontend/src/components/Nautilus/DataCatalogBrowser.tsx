import React, { useState, useEffect, useMemo, useCallback } from 'react'
import {
  Card,
  Row,
  Col,
  Button,
  Input,
  Select,
  Table,
  Tree,
  Progress,
  Statistic,
  Tag,
  Space,
  Modal,
  Typography,
  Alert,
  Tabs,
  message,
  Spin,
  Badge,
  Tooltip,
  DatePicker,
  Empty,
  Collapse,
  Switch
} from 'antd'
import {
  DatabaseOutlined,
  SearchOutlined,
  ExportOutlined,
  ImportOutlined,
  ReloadOutlined,
  DownloadOutlined,
  WarningOutlined,
  CheckCircleOutlined,
  CloseCircleOutlined,
  InfoCircleOutlined,
  FolderOutlined,
  FileOutlined,
  LineChartOutlined,
  TableOutlined,
  FilterOutlined
} from '@ant-design/icons'
import dayjs from 'dayjs'
import { FixedSizeList as VirtualList } from 'react-window'

const { Title, Text } = Typography
const { TabPane } = Tabs
const { RangePicker } = DatePicker
const { Panel } = Collapse
const { Search } = Input

interface DataCatalog {
  instruments: InstrumentMetadata[]
  venues: VenueMetadata[]
  dataSources: DataSourceInfo[]
  qualityMetrics: QualityMetrics
  lastUpdated: Date
  totalSize: number
  totalRecords: number
}

interface InstrumentMetadata {
  instrumentId: string
  venue: string
  dataType: 'tick' | 'quote' | 'bar'
  timeframes: string[]
  dateRange: {
    start: Date
    end: Date
  }
  recordCount: number
  qualityScore: number
  gaps: DataGap[]
  sizeBytes: number
  lastUpdated: Date
}

interface VenueMetadata {
  venue: string
  instrumentCount: number
  dataTypes: string[]
  timeframes: string[]
  qualityScore: number
  status: 'active' | 'inactive' | 'degraded'
}

interface DataSourceInfo {
  sourceId: string
  name: string
  type: 'live' | 'historical' | 'reference'
  status: 'connected' | 'disconnected' | 'error'
  latency: number
  throughput: number
  errorRate: number
  lastUpdate: Date
}

interface QualityMetrics {
  overallScore: number
  completeness: number
  accuracy: number
  timeliness: number
  consistency: number
  totalIssues: number
  criticalIssues: number
}

interface DataGap {
  gapId: string
  instrumentId: string
  timeframe: string
  gapStart: Date
  gapEnd: Date
  severity: 'low' | 'medium' | 'high' | 'critical'
  durationMinutes: number
  impact: string
}

interface DataFeedStatus {
  feedId: string
  source: string
  status: 'connected' | 'disconnected' | 'degraded'
  latency: number
  throughput: number
  lastUpdate: Date
  errorCount: number
  qualityScore: number
}

interface DataExportRequest {
  instrumentIds: string[]
  venues: string[]
  timeframes: string[]
  dateRange: {
    start: Date
    end: Date
  }
  format: 'parquet' | 'csv' | 'json' | 'nautilus'
  compression: boolean
  includeMetadata: boolean
}

const DataCatalogBrowser: React.FC = () => {
  const [catalog, setCatalog] = useState<DataCatalog | null>(null)
  const [filteredInstruments, setFilteredInstruments] = useState<InstrumentMetadata[]>([])
  const [feedStatuses, setFeedStatuses] = useState<DataFeedStatus[]>([])
  const [selectedInstruments, setSelectedInstruments] = useState<string[]>([])
  const [searchTerm, setSearchTerm] = useState('')
  const [venueFilter, setVenueFilter] = useState<string[]>([])
  const [timeframeFilter, setTimeframeFilter] = useState<string[]>([])
  const [qualityFilter, setQualityFilter] = useState<[number, number]>([0, 100])
  const [exportModalVisible, setExportModalVisible] = useState(false)
  const [qualityModalVisible, setQualityModalVisible] = useState(false)
  const [selectedInstrument, setSelectedInstrument] = useState<InstrumentMetadata | null>(null)
  const [loading, setLoading] = useState(false)
  const [realTimeMode, setRealTimeMode] = useState(true)
  const [autoRefresh, setAutoRefresh] = useState(true)

  const apiUrl = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8002'

  // Virtualization settings for large datasets
  const ITEM_HEIGHT = 48
  const MAX_VISIBLE_ITEMS = 100

  useEffect(() => {
    loadCatalog()
    loadFeedStatuses()

    if (autoRefresh) {
      const interval = setInterval(() => {
        if (realTimeMode) {
          loadFeedStatuses()
          // Partial catalog refresh every 30 seconds
          refreshQualityMetrics()
        }
      }, 30000)

      return () => clearInterval(interval)
    }
  }, [autoRefresh, realTimeMode])

  // Optimized filtering with useMemo to prevent unnecessary re-renders
  const filteredData = useMemo(() => {
    if (!catalog) return []

    return catalog.instruments.filter(instrument => {
      const matchesSearch = !searchTerm || 
        instrument.instrumentId.toLowerCase().includes(searchTerm.toLowerCase()) ||
        instrument.venue.toLowerCase().includes(searchTerm.toLowerCase())

      const matchesVenue = venueFilter.length === 0 || venueFilter.includes(instrument.venue)
      
      const matchesTimeframe = timeframeFilter.length === 0 || 
        instrument.timeframes.some(tf => timeframeFilter.includes(tf))

      const matchesQuality = instrument.qualityScore >= qualityFilter[0] / 100 && 
        instrument.qualityScore <= qualityFilter[1] / 100

      return matchesSearch && matchesVenue && matchesTimeframe && matchesQuality
    })
  }, [catalog, searchTerm, venueFilter, timeframeFilter, qualityFilter])

  const loadCatalog = async () => {
    setLoading(true)
    try {
      const response = await fetch(`${apiUrl}/api/v1/nautilus/data/catalog`)
      if (response.ok) {
        const data = await response.json()
        setCatalog(data)
        setFilteredInstruments(data.instruments)
      } else {
        message.error('Failed to load data catalog')
      }
    } catch (error) {
      message.error('Failed to load data catalog')
      console.error('Load catalog error:', error)
    } finally {
      setLoading(false)
    }
  }

  const loadFeedStatuses = async () => {
    try {
      const response = await fetch(`${apiUrl}/api/v1/nautilus/data/feeds/status`)
      if (response.ok) {
        const data = await response.json()
        setFeedStatuses(data.feeds || [])
      }
    } catch (error) {
      console.error('Failed to load feed statuses:', error)
    }
  }

  const refreshQualityMetrics = async () => {
    try {
      const response = await fetch(`${apiUrl}/api/v1/nautilus/data/quality/refresh`, {
        method: 'POST'
      })
      if (response.ok) {
        loadCatalog()
      }
    } catch (error) {
      console.error('Failed to refresh quality metrics:', error)
    }
  }

  const analyzeDataGaps = async (instrumentId: string) => {
    try {
      const response = await fetch(`${apiUrl}/api/v1/nautilus/data/gaps/${instrumentId}`)
      if (response.ok) {
        const gaps = await response.json()
        const instrument = catalog?.instruments.find(i => i.instrumentId === instrumentId)
        if (instrument) {
          setSelectedInstrument({ ...instrument, gaps: gaps.gaps })
          setQualityModalVisible(true)
        }
      }
    } catch (error) {
      message.error('Failed to analyze data gaps')
      console.error('Analyze gaps error:', error)
    }
  }

  const exportData = async (request: DataExportRequest) => {
    setLoading(true)
    try {
      const response = await fetch(`${apiUrl}/api/v1/nautilus/data/export`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(request)
      })

      if (response.ok) {
        const result = await response.json()
        message.success(`Export started: ${result.exportId}`)
        setExportModalVisible(false)
        // Start polling for export completion
        pollExportStatus(result.exportId)
      } else {
        message.error('Failed to start data export')
      }
    } catch (error) {
      message.error('Failed to export data')
      console.error('Export error:', error)
    } finally {
      setLoading(false)
    }
  }

  const pollExportStatus = async (exportId: string) => {
    const maxAttempts = 60 // 5 minutes with 5s intervals
    let attempts = 0

    const checkStatus = async () => {
      try {
        const response = await fetch(`${apiUrl}/api/v1/nautilus/data/export/${exportId}/status`)
        if (response.ok) {
          const status = await response.json()
          
          if (status.status === 'completed') {
            message.success(`Export completed: ${status.filePath}`)
            return
          } else if (status.status === 'failed') {
            message.error(`Export failed: ${status.error}`)
            return
          }

          attempts++
          if (attempts < maxAttempts) {
            setTimeout(checkStatus, 5000)
          } else {
            message.warning('Export status check timed out')
          }
        }
      } catch (error) {
        console.error('Export status check error:', error)
      }
    }

    setTimeout(checkStatus, 5000)
  }

  const getQualityColor = (score: number) => {
    if (score >= 0.9) return 'success'
    if (score >= 0.7) return 'warning'
    return 'error'
  }

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'connected': case 'active': return 'success'
      case 'degraded': return 'warning'
      case 'disconnected': case 'inactive': case 'error': return 'error'
      default: return 'default'
    }
  }

  // Optimized virtual list renderer for large instrument lists
  const InstrumentRow = ({ index, style }: { index: number; style: React.CSSProperties }) => {
    const instrument = filteredData[index]
    if (!instrument) return null

    return (
      <div style={style}>
        <Card 
          size="small" 
          style={{ margin: '4px 8px', cursor: 'pointer' }}
          onClick={() => analyzeDataGaps(instrument.instrumentId)}
          hoverable
        >
          <Row justify="space-between" align="middle">
            <Col flex="auto">
              <Space>
                <Text strong>{instrument.instrumentId}</Text>
                <Tag color="blue">{instrument.venue}</Tag>
                <Badge 
                  status={getQualityColor(instrument.qualityScore)} 
                  text={`${(instrument.qualityScore * 100).toFixed(1)}%`} 
                />
              </Space>
            </Col>
            <Col>
              <Space>
                <Text type="secondary">{instrument.recordCount.toLocaleString()} records</Text>
                <Text type="secondary">{(instrument.sizeBytes / 1024 / 1024).toFixed(1)} MB</Text>
              </Space>
            </Col>
          </Row>
        </Card>
      </div>
    )
  }

  const instrumentColumns = [
    {
      title: 'Instrument',
      dataIndex: 'instrumentId',
      key: 'instrumentId',
      width: 150,
      fixed: 'left' as const,
      sorter: (a: InstrumentMetadata, b: InstrumentMetadata) => a.instrumentId.localeCompare(b.instrumentId)
    },
    {
      title: 'Venue',
      dataIndex: 'venue',
      key: 'venue',
      width: 100,
      filters: catalog?.venues.map(v => ({ text: v.venue, value: v.venue })) || [],
      onFilter: (value: any, record: InstrumentMetadata) => record.venue === value
    },
    {
      title: 'Quality Score',
      dataIndex: 'qualityScore',
      key: 'qualityScore',
      width: 120,
      render: (score: number) => (
        <Progress
          percent={score * 100}
          size="small"
          status={score >= 0.9 ? 'success' : score >= 0.7 ? 'active' : 'exception'}
          format={() => `${(score * 100).toFixed(1)}%`}
        />
      ),
      sorter: (a: InstrumentMetadata, b: InstrumentMetadata) => a.qualityScore - b.qualityScore
    },
    {
      title: 'Records',
      dataIndex: 'recordCount',
      key: 'recordCount',
      width: 100,
      render: (count: number) => count.toLocaleString(),
      sorter: (a: InstrumentMetadata, b: InstrumentMetadata) => a.recordCount - b.recordCount
    },
    {
      title: 'Size',
      dataIndex: 'sizeBytes',
      key: 'sizeBytes',
      width: 80,
      render: (bytes: number) => `${(bytes / 1024 / 1024).toFixed(1)} MB`,
      sorter: (a: InstrumentMetadata, b: InstrumentMetadata) => a.sizeBytes - b.sizeBytes
    },
    {
      title: 'Timeframes',
      dataIndex: 'timeframes',
      key: 'timeframes',
      width: 150,
      render: (timeframes: string[]) => (
        <Space wrap>
          {timeframes.slice(0, 3).map(tf => <Tag key={tf} size="small">{tf}</Tag>)}
          {timeframes.length > 3 && <Tag size="small">+{timeframes.length - 3}</Tag>}
        </Space>
      )
    },
    {
      title: 'Date Range',
      key: 'dateRange',
      width: 200,
      render: (_, record: InstrumentMetadata) => (
        <Text type="secondary">
          {dayjs(record.dateRange.start).format('MM/DD/YY')} - {dayjs(record.dateRange.end).format('MM/DD/YY')}
        </Text>
      )
    },
    {
      title: 'Actions',
      key: 'actions',
      width: 120,
      fixed: 'right' as const,
      render: (_, record: InstrumentMetadata) => (
        <Space size="small">
          <Button
            size="small"
            icon={<InfoCircleOutlined />}
            onClick={(e) => {
              e.stopPropagation()
              analyzeDataGaps(record.instrumentId)
            }}
          />
          <Button
            size="small"
            icon={<ExportOutlined />}
            onClick={(e) => {
              e.stopPropagation()
              setSelectedInstruments([record.instrumentId])
              setExportModalVisible(true)
            }}
          />
        </Space>
      )
    }
  ]

  return (
    <div style={{ padding: 24 }}>
      <Row gutter={[16, 16]}>
        {/* Header with Real-time Controls */}
        <Col xs={24}>
          <Card>
            <Row justify="space-between" align="middle">
              <Col>
                <Title level={3} style={{ margin: 0 }}>
                  <DatabaseOutlined style={{ marginRight: 8 }} />
                  Data Catalog & Pipeline Monitor
                </Title>
                <Text type="secondary">
                  {catalog && `${catalog.totalRecords.toLocaleString()} records across ${catalog.instruments.length} instruments`}
                </Text>
              </Col>
              <Col>
                <Space>
                  <Switch 
                    checked={realTimeMode}
                    onChange={setRealTimeMode}
                    checkedChildren="Real-time"
                    unCheckedChildren="Static"
                  />
                  <Switch 
                    checked={autoRefresh}
                    onChange={setAutoRefresh}
                    checkedChildren="Auto-refresh"
                    unCheckedChildren="Manual"
                  />
                  <Button 
                    icon={<ReloadOutlined />} 
                    onClick={loadCatalog}
                    loading={loading}
                  >
                    Refresh
                  </Button>
                  <Button 
                    type="primary" 
                    icon={<ExportOutlined />}
                    onClick={() => setExportModalVisible(true)}
                    disabled={selectedInstruments.length === 0}
                  >
                    Export ({selectedInstruments.length})
                  </Button>
                </Space>
              </Col>
            </Row>
          </Card>
        </Col>

        {/* Quality Overview Cards */}
        {catalog && (
          <Col xs={24}>
            <Row gutter={[16, 16]}>
              <Col xs={12} sm={6} md={4}>
                <Card size="small">
                  <Statistic 
                    title="Overall Quality" 
                    value={catalog.qualityMetrics.overallScore * 100} 
                    suffix="%" 
                    precision={1}
                    valueStyle={{ color: getQualityColor(catalog.qualityMetrics.overallScore) === 'success' ? '#3f8600' : '#cf1322' }}
                  />
                </Card>
              </Col>
              <Col xs={12} sm={6} md={4}>
                <Card size="small">
                  <Statistic 
                    title="Total Size" 
                    value={catalog.totalSize / 1024 / 1024 / 1024} 
                    suffix="GB" 
                    precision={2}
                  />
                </Card>
              </Col>
              <Col xs={12} sm={6} md={4}>
                <Card size="small">
                  <Statistic 
                    title="Active Feeds" 
                    value={feedStatuses.filter(f => f.status === 'connected').length}
                    suffix={`/ ${feedStatuses.length}`}
                  />
                </Card>
              </Col>
              <Col xs={12} sm={6} md={4}>
                <Card size="small">
                  <Statistic 
                    title="Data Issues" 
                    value={catalog.qualityMetrics.totalIssues}
                    valueStyle={{ color: catalog.qualityMetrics.criticalIssues > 0 ? '#cf1322' : '#3f8600' }}
                  />
                </Card>
              </Col>
              <Col xs={12} sm={6} md={4}>
                <Card size="small">
                  <Statistic 
                    title="Avg Latency" 
                    value={feedStatuses.reduce((acc, f) => acc + f.latency, 0) / Math.max(feedStatuses.length, 1)}
                    suffix="ms" 
                    precision={0}
                  />
                </Card>
              </Col>
              <Col xs={12} sm={6} md={4}>
                <Card size="small">
                  <Statistic 
                    title="Last Updated" 
                    value={dayjs(catalog.lastUpdated).fromNow()}
                  />
                </Card>
              </Col>
            </Row>
          </Col>
        )}

        {/* Filters and Search */}
        <Col xs={24}>
          <Card title="Filters & Search" size="small">
            <Row gutter={[16, 8]}>
              <Col xs={24} sm={8} md={6}>
                <Search
                  placeholder="Search instruments..."
                  value={searchTerm}
                  onChange={(e) => setSearchTerm(e.target.value)}
                  allowClear
                />
              </Col>
              <Col xs={24} sm={8} md={6}>
                <Select
                  mode="multiple"
                  placeholder="Filter by venue"
                  value={venueFilter}
                  onChange={setVenueFilter}
                  style={{ width: '100%' }}
                  maxTagCount="responsive"
                >
                  {catalog?.venues.map(venue => (
                    <Select.Option key={venue.venue} value={venue.venue}>
                      {venue.venue} ({venue.instrumentCount})
                    </Select.Option>
                  ))}
                </Select>
              </Col>
              <Col xs={24} sm={8} md={6}>
                <Select
                  mode="multiple"
                  placeholder="Filter by timeframe"
                  value={timeframeFilter}
                  onChange={setTimeframeFilter}
                  style={{ width: '100%' }}
                  maxTagCount="responsive"
                >
                  {Array.from(new Set(catalog?.instruments.flatMap(i => i.timeframes) || [])).map(tf => (
                    <Select.Option key={tf} value={tf}>{tf}</Select.Option>
                  ))}
                </Select>
              </Col>
              <Col xs={24} sm={12} md={6}>
                <Text type="secondary">Quality Score: {qualityFilter[0]}% - {qualityFilter[1]}%</Text>
              </Col>
            </Row>
          </Card>
        </Col>

        {/* Main Data Table with Virtualization */}
        <Col xs={24}>
          <Card 
            title={`Instruments (${filteredData.length})`}
            extra={
              <Space>
                <Text type="secondary">
                  Showing {Math.min(filteredData.length, MAX_VISIBLE_ITEMS)} of {filteredData.length}
                </Text>
                <Button 
                  size="small" 
                  icon={<FilterOutlined />}
                  type={filteredData.length !== catalog?.instruments.length ? 'primary' : 'default'}
                >
                  Filtered
                </Button>
              </Space>
            }
          >
            {loading ? (
              <div style={{ textAlign: 'center', padding: 50 }}>
                <Spin size="large" />
                <div style={{ marginTop: 16 }}>Loading catalog...</div>
              </div>
            ) : filteredData.length === 0 ? (
              <Empty description="No instruments match your filters" />
            ) : (
              <Table
                dataSource={filteredData.slice(0, MAX_VISIBLE_ITEMS)}
                columns={instrumentColumns}
                rowKey="instrumentId"
                rowSelection={{
                  selectedRowKeys: selectedInstruments,
                  onChange: setSelectedInstruments,
                  getCheckboxProps: (record) => ({
                    name: record.instrumentId
                  })
                }}
                pagination={{
                  pageSize: 50,
                  showSizeChanger: true,
                  showQuickJumper: true,
                  showTotal: (total, range) => `${range[0]}-${range[1]} of ${total} items`
                }}
                scroll={{ x: 1200, y: 600 }}
                size="small"
              />
            )}
          </Card>
        </Col>

        {/* Real-time Feed Status */}
        <Col xs={24}>
          <Card title="Real-time Data Feeds" size="small">
            <Row gutter={[16, 8]}>
              {feedStatuses.map(feed => (
                <Col xs={24} sm={12} md={8} lg={6} key={feed.feedId}>
                  <Card size="small" style={{ backgroundColor: feed.status === 'connected' ? '#f6ffed' : '#fff2f0' }}>
                    <Space direction="vertical" size="small" style={{ width: '100%' }}>
                      <Row justify="space-between">
                        <Text strong>{feed.source}</Text>
                        <Badge status={getStatusColor(feed.status)} text={feed.status} />
                      </Row>
                      <Row justify="space-between">
                        <Text type="secondary">Latency:</Text>
                        <Text>{feed.latency}ms</Text>
                      </Row>
                      <Row justify="space-between">
                        <Text type="secondary">Quality:</Text>
                        <Text>{(feed.qualityScore * 100).toFixed(1)}%</Text>
                      </Row>
                    </Space>
                  </Card>
                </Col>
              ))}
            </Row>
          </Card>
        </Col>
      </Row>

      {/* Export Modal */}
      <Modal
        title="Export Data"
        open={exportModalVisible}
        onCancel={() => setExportModalVisible(false)}
        footer={[
          <Button key="cancel" onClick={() => setExportModalVisible(false)}>
            Cancel
          </Button>,
          <Button key="export" type="primary" loading={loading}>
            Start Export
          </Button>
        ]}
        width={800}
      >
        <Alert
          message="Data Export Configuration"
          description="Configure your data export parameters. Large exports may take several minutes to complete."
          type="info"
          showIcon
          style={{ marginBottom: 16 }}
        />
        
        <Space direction="vertical" style={{ width: '100%' }}>
          <div>
            <Text strong>Selected Instruments ({selectedInstruments.length}):</Text>
            <div style={{ marginTop: 8, maxHeight: 100, overflow: 'auto' }}>
              {selectedInstruments.map(id => (
                <Tag key={id} style={{ margin: 2 }}>{id}</Tag>
              ))}
            </div>
          </div>
          
          <Row gutter={16}>
            <Col span={12}>
              <Text strong>Export Format:</Text>
              <Select defaultValue="parquet" style={{ width: '100%', marginTop: 4 }}>
                <Select.Option value="parquet">Parquet (Recommended)</Select.Option>
                <Select.Option value="csv">CSV</Select.Option>
                <Select.Option value="json">JSON</Select.Option>
                <Select.Option value="nautilus">NautilusTrader Format</Select.Option>
              </Select>
            </Col>
            <Col span={12}>
              <Text strong>Date Range:</Text>
              <RangePicker style={{ width: '100%', marginTop: 4 }} />
            </Col>
          </Row>
        </Space>
      </Modal>

      {/* Quality Analysis Modal */}
      <Modal
        title={`Data Quality Analysis: ${selectedInstrument?.instrumentId}`}
        open={qualityModalVisible}
        onCancel={() => setQualityModalVisible(false)}
        footer={[
          <Button key="close" onClick={() => setQualityModalVisible(false)}>
            Close
          </Button>
        ]}
        width={900}
      >
        {selectedInstrument && (
          <Tabs defaultActiveKey="overview">
            <TabPane tab="Overview" key="overview">
              <Row gutter={16}>
                <Col span={8}>
                  <Statistic title="Quality Score" value={selectedInstrument.qualityScore * 100} suffix="%" precision={1} />
                </Col>
                <Col span={8}>
                  <Statistic title="Total Records" value={selectedInstrument.recordCount.toLocaleString()} />
                </Col>
                <Col span={8}>
                  <Statistic title="Data Gaps" value={selectedInstrument.gaps?.length || 0} />
                </Col>
              </Row>
            </TabPane>
            <TabPane tab="Data Gaps" key="gaps">
              <Table
                dataSource={selectedInstrument.gaps || []}
                columns={[
                  { title: 'Severity', dataIndex: 'severity', key: 'severity', render: (severity: string) => (
                    <Tag color={severity === 'critical' ? 'red' : severity === 'high' ? 'orange' : 'yellow'}>
                      {severity.toUpperCase()}
                    </Tag>
                  )},
                  { title: 'Start', dataIndex: 'gapStart', key: 'gapStart', render: (date: Date) => dayjs(date).format('MM/DD/YY HH:mm') },
                  { title: 'End', dataIndex: 'gapEnd', key: 'gapEnd', render: (date: Date) => dayjs(date).format('MM/DD/YY HH:mm') },
                  { title: 'Duration', dataIndex: 'durationMinutes', key: 'duration', render: (mins: number) => `${mins}m` },
                  { title: 'Impact', dataIndex: 'impact', key: 'impact' }
                ]}
                size="small"
                pagination={false}
              />
            </TabPane>
          </Tabs>
        )}
      </Modal>
    </div>
  )
}

export default DataCatalogBrowser