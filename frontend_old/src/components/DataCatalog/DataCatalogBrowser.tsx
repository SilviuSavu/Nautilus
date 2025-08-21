import React, { useState, useEffect, useMemo } from 'react'
import {
  Card,
  Table,
  Tree,
  Input,
  Select,
  Row,
  Col,
  Statistic,
  Tag,
  Progress,
  Space,
  Button,
  Modal,
  Tooltip,
  Alert,
  DatePicker,
  InputNumber,
  Switch
} from 'antd'
import {
  SearchOutlined,
  ReloadOutlined,
  ExportOutlined,
  WarningOutlined,
  CheckCircleOutlined,
  ClockCircleOutlined,
  DatabaseOutlined,
  FolderOutlined,
  FileTextOutlined
} from '@ant-design/icons'
import type { ColumnsType } from 'antd/es/table'
import type { DataNode } from 'antd/es/tree'
import dayjs from 'dayjs'
import { dataCatalogService } from '../../services/dataCatalogService'
import {
  DataCatalog,
  InstrumentMetadata,
  CatalogSearchFilters,
  CatalogSearchResult,
  DataGap
} from '../../types/dataCatalog'

const { Search } = Input
const { Option } = Select
const { RangePicker } = DatePicker

interface DataCatalogBrowserProps {
  onInstrumentSelect?: (instrument: InstrumentMetadata) => void
  onExportRequest?: (instrumentIds: string[]) => void
}

export const DataCatalogBrowser: React.FC<DataCatalogBrowserProps> = ({
  onInstrumentSelect,
  onExportRequest
}) => {
  const [catalog, setCatalog] = useState<DataCatalog | null>(null)
  const [loading, setLoading] = useState(true)
  const [searchResult, setSearchResult] = useState<CatalogSearchResult | null>(null)
  const [selectedInstruments, setSelectedInstruments] = useState<string[]>([])
  const [expandedKeys, setExpandedKeys] = useState<React.Key[]>([])
  const [selectedKeys, setSelectedKeys] = useState<React.Key[]>([])
  const [viewMode, setViewMode] = useState<'tree' | 'table'>('tree')
  const [detailModalVisible, setDetailModalVisible] = useState(false)
  const [selectedInstrument, setSelectedInstrument] = useState<InstrumentMetadata | null>(null)
  const [filters, setFilters] = useState<CatalogSearchFilters>({})

  useEffect(() => {
    loadCatalog()
  }, [])

  const loadCatalog = async () => {
    try {
      setLoading(true)
      const catalogData = await dataCatalogService.getCatalog()
      setCatalog(catalogData)
    } catch (error) {
      console.error('Failed to load catalog:', error)
    } finally {
      setLoading(false)
    }
  }

  const searchInstruments = async () => {
    try {
      setLoading(true)
      const result = await dataCatalogService.searchInstruments(filters)
      setSearchResult(result)
      setViewMode('table')
    } catch (error) {
      console.error('Search failed:', error)
    } finally {
      setLoading(false)
    }
  }

  const clearSearch = () => {
    setSearchResult(null)
    setFilters({})
    setViewMode('tree')
  }

  const treeData = useMemo(() => {
    if (!catalog) return []

    const venueGroups = catalog.venues.reduce((acc, venue) => {
      const instruments = catalog.instruments.filter(inst => inst.venue === venue.id)
      
      if (instruments.length === 0) return acc

      const assetClassGroups = instruments.reduce((assetAcc, instrument) => {
        const assetClass = instrument.assetClass || 'Unknown'
        if (!assetAcc[assetClass]) {
          assetAcc[assetClass] = []
        }
        assetAcc[assetClass].push(instrument)
        return assetAcc
      }, {} as Record<string, InstrumentMetadata[]>)

      const assetClassNodes: DataNode[] = Object.entries(assetClassGroups).map(([assetClass, classInstruments]) => ({
        title: (
          <Space>
            <DatabaseOutlined />
            <span>{assetClass}</span>
            <Tag color="blue">{classInstruments.length}</Tag>
          </Space>
        ),
        key: `${venue.id}-${assetClass}`,
        children: classInstruments.map(instrument => ({
          title: (
            <Space>
              <FileTextOutlined />
              <span>{instrument.symbol}</span>
              {instrument.qualityScore < 0.8 && <WarningOutlined style={{ color: '#ff4d4f' }} />}
              {instrument.gaps.length > 0 && <ClockCircleOutlined style={{ color: '#faad14' }} />}
              <Tag color={getQualityColor(instrument.qualityScore)}>
                {Math.round(instrument.qualityScore * 100)}%
              </Tag>
            </Space>
          ),
          key: instrument.instrumentId,
          isLeaf: true,
          data: instrument
        }))
      }))

      acc.push({
        title: (
          <Space>
            <FolderOutlined />
            <span>{venue.name}</span>
            <Tag color="green">{instruments.length}</Tag>
          </Space>
        ),
        key: venue.id,
        children: assetClassNodes
      })

      return acc
    }, [] as DataNode[])

    return venueGroups
  }, [catalog])

  const tableColumns: ColumnsType<InstrumentMetadata> = [
    {
      title: 'Symbol',
      dataIndex: 'symbol',
      key: 'symbol',
      sorter: (a, b) => a.symbol.localeCompare(b.symbol),
      render: (text, record) => (
        <Space>
          <Button
            type="link"
            onClick={() => showInstrumentDetail(record)}
            style={{ padding: 0 }}
          >
            {text}
          </Button>
          {record.gaps.length > 0 && (
            <Tooltip title={`${record.gaps.length} data gaps detected`}>
              <WarningOutlined style={{ color: '#faad14' }} />
            </Tooltip>
          )}
        </Space>
      )
    },
    {
      title: 'Venue',
      dataIndex: 'venue',
      key: 'venue',
      filters: catalog?.venues.map(v => ({ text: v.name, value: v.id })),
      onFilter: (value, record) => record.venue === value
    },
    {
      title: 'Asset Class',
      dataIndex: 'assetClass',
      key: 'assetClass',
      filters: [...new Set(catalog?.instruments.map(i => i.assetClass))].map(ac => ({ text: ac, value: ac })),
      onFilter: (value, record) => record.assetClass === value
    },
    {
      title: 'Data Type',
      dataIndex: 'dataType',
      key: 'dataType',
      filters: [
        { text: 'Tick', value: 'tick' },
        { text: 'Quote', value: 'quote' },
        { text: 'Bar', value: 'bar' }
      ],
      onFilter: (value, record) => record.dataType === value,
      render: (dataType) => <Tag color={getDataTypeColor(dataType)}>{dataType.toUpperCase()}</Tag>
    },
    {
      title: 'Quality Score',
      dataIndex: 'qualityScore',
      key: 'qualityScore',
      sorter: (a, b) => a.qualityScore - b.qualityScore,
      render: (score) => (
        <Space>
          <Progress
            percent={Math.round(score * 100)}
           
            strokeColor={getQualityColor(score)}
            style={{ width: 80 }}
          />
          <span>{Math.round(score * 100)}%</span>
        </Space>
      )
    },
    {
      title: 'Records',
      dataIndex: 'recordCount',
      key: 'recordCount',
      sorter: (a, b) => a.recordCount - b.recordCount,
      render: (count) => count.toLocaleString()
    },
    {
      title: 'Date Range',
      key: 'dateRange',
      render: (_, record) => (
        <div>
          <div>{dayjs(record.dateRange.start).format('YYYY-MM-DD')}</div>
          <div style={{ fontSize: '12px', color: '#666' }}>
            to {dayjs(record.dateRange.end).format('YYYY-MM-DD')}
          </div>
        </div>
      )
    },
    {
      title: 'Last Updated',
      dataIndex: 'lastUpdated',
      key: 'lastUpdated',
      sorter: (a, b) => new Date(a.lastUpdated).getTime() - new Date(b.lastUpdated).getTime(),
      render: (date) => dayjs(date).format('YYYY-MM-DD HH:mm')
    }
  ]

  const showInstrumentDetail = (instrument: InstrumentMetadata) => {
    setSelectedInstrument(instrument)
    setDetailModalVisible(true)
    onInstrumentSelect?.(instrument)
  }

  const handleTreeSelect = (selectedKeys: React.Key[], info: any) => {
    setSelectedKeys(selectedKeys)
    const selectedNode = info.node
    if (selectedNode?.isLeaf && selectedNode.data) {
      showInstrumentDetail(selectedNode.data)
    }
  }

  const handleExport = () => {
    if (selectedInstruments.length > 0) {
      onExportRequest?.(selectedInstruments)
    }
  }

  const getQualityColor = (score: number) => {
    if (score >= 0.9) return '#52c41a'
    if (score >= 0.7) return '#faad14'
    return '#ff4d4f'
  }

  const getDataTypeColor = (dataType: string) => {
    switch (dataType) {
      case 'tick': return 'green'
      case 'quote': return 'blue'
      case 'bar': return 'orange'
      default: return 'default'
    }
  }

  const renderGapSummary = (gaps: DataGap[]) => {
    if (gaps.length === 0) {
      return <Tag color="green" icon={<CheckCircleOutlined />}>No Gaps</Tag>
    }

    const severityCounts = gaps.reduce((acc, gap) => {
      acc[gap.severity] = (acc[gap.severity] || 0) + 1
      return acc
    }, {} as Record<string, number>)

    return (
      <Space>
        {Object.entries(severityCounts).map(([severity, count]) => (
          <Tag
            key={severity}
            color={severity === 'high' ? 'red' : severity === 'medium' ? 'orange' : 'yellow'}
            icon={<WarningOutlined />}
          >
            {count} {severity}
          </Tag>
        ))}
      </Space>
    )
  }

  if (loading && !catalog) {
    return (
      <Card>
        <div style={{ textAlign: 'center', padding: '40px 0' }}>
          Loading data catalog...
        </div>
      </Card>
    )
  }

  return (
    <div>
      {/* Summary Statistics */}
      <Row gutter={[16, 16]} style={{ marginBottom: 16 }}>
        <Col span={6}>
          <Card>
            <Statistic
              title="Total Instruments"
              value={catalog?.totalInstruments || 0}
              prefix={<DatabaseOutlined />}
            />
          </Card>
        </Col>
        <Col span={6}>
          <Card>
            <Statistic
              title="Total Records"
              value={catalog?.totalRecords || 0}
              formatter={(value) => value?.toLocaleString()}
            />
          </Card>
        </Col>
        <Col span={6}>
          <Card>
            <Statistic
              title="Storage Size"
              value={catalog?.storageSize || 0}
              suffix="MB"
              formatter={(value) => (Number(value) / 1024 / 1024).toFixed(1)}
            />
          </Card>
        </Col>
        <Col span={6}>
          <Card>
            <Statistic
              title="Average Quality"
              value={catalog?.qualityMetrics.overall || 0}
              suffix="%"
              formatter={(value) => Math.round(Number(value) * 100)}
              valueStyle={{ color: getQualityColor(catalog?.qualityMetrics.overall || 0) }}
            />
          </Card>
        </Col>
      </Row>

      {/* Search and Filter Controls */}
      <Card style={{ marginBottom: 16 }}>
        <Row gutter={[16, 16]} align="middle">
          <Col span={8}>
            <Search
              placeholder="Search instruments..."
              value={filters.venue}
              onChange={(e) => setFilters({ ...filters, venue: e.target.value })}
              allowClear
            />
          </Col>
          <Col span={4}>
            <Select
              placeholder="Asset Class"
              value={filters.assetClass}
              onChange={(value) => setFilters({ ...filters, assetClass: value })}
              allowClear
              style={{ width: '100%' }}
            >
              {[...new Set(catalog?.instruments.map(i => i.assetClass))].map(ac => (
                <Option key={ac} value={ac}>{ac}</Option>
              ))}
            </Select>
          </Col>
          <Col span={4}>
            <Select
              placeholder="Data Type"
              value={filters.dataType}
              onChange={(value) => setFilters({ ...filters, dataType: value })}
              allowClear
              style={{ width: '100%' }}
            >
              <Option value="tick">Tick</Option>
              <Option value="quote">Quote</Option>
              <Option value="bar">Bar</Option>
            </Select>
          </Col>
          <Col span={4}>
            <InputNumber
              placeholder="Min Quality %"
              value={filters.qualityThreshold ? filters.qualityThreshold * 100 : undefined}
              onChange={(value) => setFilters({ ...filters, qualityThreshold: value ? value / 100 : undefined })}
              min={0}
              max={100}
              style={{ width: '100%' }}
            />
          </Col>
          <Col span={4}>
            <Space>
              <Button type="primary" icon={<SearchOutlined />} onClick={searchInstruments}>
                Search
              </Button>
              <Button icon={<ReloadOutlined />} onClick={clearSearch}>
                Clear
              </Button>
            </Space>
          </Col>
        </Row>
        <Row style={{ marginTop: 16 }}>
          <Col span={8}>
            <RangePicker
              placeholder={['Start Date', 'End Date']}
              value={filters.dateRange ? [dayjs(filters.dateRange.start), dayjs(filters.dateRange.end)] : undefined}
              onChange={(dates) => {
                if (dates && dates[0] && dates[1]) {
                  setFilters({
                    ...filters,
                    dateRange: {
                      start: dates[0].format('YYYY-MM-DD'),
                      end: dates[1].format('YYYY-MM-DD')
                    }
                  })
                } else {
                  setFilters({ ...filters, dateRange: undefined })
                }
              }}
              style={{ width: '100%' }}
            />
          </Col>
          <Col span={4} offset={1}>
            <Space>
              <span>Has Gaps:</span>
              <Switch
                checked={filters.hasGaps}
                onChange={(checked) => setFilters({ ...filters, hasGaps: checked })}
              />
            </Space>
          </Col>
          <Col span={4} offset={1}>
            <Space>
              <span>View:</span>
              <Select
                value={viewMode}
                onChange={setViewMode}
                style={{ width: 100 }}
              >
                <Option value="tree">Tree</Option>
                <Option value="table">Table</Option>
              </Select>
            </Space>
          </Col>
        </Row>
      </Card>

      {/* Data Display */}
      <Card
        title="Data Catalog"
        extra={
          <Space>
            {selectedInstruments.length > 0 && (
              <Button
                type="primary"
                icon={<ExportOutlined />}
                onClick={handleExport}
              >
                Export Selected ({selectedInstruments.length})
              </Button>
            )}
            <Button icon={<ReloadOutlined />} onClick={loadCatalog} loading={loading}>
              Refresh
            </Button>
          </Space>
        }
      >
        {searchResult && (
          <Alert
            message={`Found ${searchResult.totalCount} instruments matching your criteria`}
            type="info"
            showIcon
            closable
            onClose={clearSearch}
            style={{ marginBottom: 16 }}
          />
        )}

        {viewMode === 'tree' && !searchResult ? (
          <Tree
            treeData={treeData}
            expandedKeys={expandedKeys}
            selectedKeys={selectedKeys}
            onExpand={setExpandedKeys}
            onSelect={handleTreeSelect}
            style={{ maxHeight: 600, overflow: 'auto' }}
          />
        ) : (
          <Table
            columns={tableColumns}
            dataSource={searchResult?.instruments || catalog?.instruments}
            rowKey="instrumentId"
            rowSelection={{
              selectedRowKeys: selectedInstruments,
              onChange: (selectedRowKeys) => setSelectedInstruments(selectedRowKeys as string[]),
              preserveSelectedRowKeys: true
            }}
            pagination={{
              pageSize: 50,
              showSizeChanger: true,
              showQuickJumper: true,
              showTotal: (total, range) => `${range[0]}-${range[1]} of ${total} instruments`
            }}
            loading={loading}
            scroll={{ y: 500 }}
          />
        )}
      </Card>

      {/* Instrument Detail Modal */}
      <Modal
        title={`Instrument Details: ${selectedInstrument?.symbol}`}
        open={detailModalVisible}
        onCancel={() => setDetailModalVisible(false)}
        footer={null}
        width={800}
      >
        {selectedInstrument && (
          <div>
            <Row gutter={[16, 16]}>
              <Col span={12}>
                <Card title="Basic Information">
                  <p><strong>Symbol:</strong> {selectedInstrument.symbol}</p>
                  <p><strong>Venue:</strong> {selectedInstrument.venue}</p>
                  <p><strong>Asset Class:</strong> {selectedInstrument.assetClass}</p>
                  <p><strong>Currency:</strong> {selectedInstrument.currency}</p>
                  <p><strong>Data Type:</strong> {selectedInstrument.dataType}</p>
                </Card>
              </Col>
              <Col span={12}>
                <Card title="Data Metrics">
                  <p><strong>Records:</strong> {selectedInstrument.recordCount.toLocaleString()}</p>
                  <p><strong>Quality Score:</strong> {Math.round(selectedInstrument.qualityScore * 100)}%</p>
                  <p><strong>File Size:</strong> {selectedInstrument.fileSize ? (selectedInstrument.fileSize / 1024 / 1024).toFixed(2) + ' MB' : 'N/A'}</p>
                  <p><strong>Last Updated:</strong> {dayjs(selectedInstrument.lastUpdated).format('YYYY-MM-DD HH:mm')}</p>
                </Card>
              </Col>
            </Row>
            <Row style={{ marginTop: 16 }}>
              <Col span={24}>
                <Card title="Date Range">
                  <p>
                    <strong>From:</strong> {dayjs(selectedInstrument.dateRange.start).format('YYYY-MM-DD')} 
                    {' '}to{' '}
                    <strong>{dayjs(selectedInstrument.dateRange.end).format('YYYY-MM-DD')}</strong>
                  </p>
                  <p><strong>Timeframes:</strong> {selectedInstrument.timeframes.join(', ')}</p>
                </Card>
              </Col>
            </Row>
            <Row style={{ marginTop: 16 }}>
              <Col span={24}>
                <Card title="Data Quality">
                  {renderGapSummary(selectedInstrument.gaps)}
                  {selectedInstrument.gaps.length > 0 && (
                    <div style={{ marginTop: 8 }}>
                      <p><strong>Gap Details:</strong></p>
                      {selectedInstrument.gaps.slice(0, 5).map((gap, index) => (
                        <div key={index} style={{ fontSize: '12px', marginLeft: 16 }}>
                          {dayjs(gap.start).format('YYYY-MM-DD HH:mm')} - {dayjs(gap.end).format('YYYY-MM-DD HH:mm')} 
                          <Tag color={gap.severity === 'high' ? 'red' : gap.severity === 'medium' ? 'orange' : 'yellow'}>
                            {gap.severity}
                          </Tag>
                        </div>
                      ))}
                      {selectedInstrument.gaps.length > 5 && (
                        <div style={{ fontSize: '12px', marginLeft: 16, color: '#666' }}>
                          ... and {selectedInstrument.gaps.length - 5} more gaps
                        </div>
                      )}
                    </div>
                  )}
                </Card>
              </Col>
            </Row>
          </div>
        )}
      </Modal>
    </div>
  )
}

export default DataCatalogBrowser