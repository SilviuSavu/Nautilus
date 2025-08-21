import React, { useState, useEffect, useMemo } from 'react'
import {
  Card,
  Row,
  Col,
  Table,
  Select,
  DatePicker,
  Button,
  Space,
  Tag,
  Tooltip,
  Progress,
  Alert,
  Statistic,
  Timeline,
  Empty,
  Spin,
  Switch,
  InputNumber,
  notification
} from 'antd'
import {
  WarningOutlined,
  ClockCircleOutlined,
  CheckCircleOutlined,
  ReloadOutlined,
  DownloadOutlined,
  BugOutlined,
  ExclamationCircleOutlined,
  FileSearchOutlined,
  CalendarOutlined
} from '@ant-design/icons'
import type { ColumnsType } from 'antd/es/table'
import dayjs from 'dayjs'
import { dataCatalogService } from '../../services/dataCatalogService'
import {
  DataGap,
  InstrumentMetadata,
  CatalogSearchFilters
} from '../../types/dataCatalog'

const { Option } = Select
const { RangePicker } = DatePicker

interface GapAnalysisData {
  instrument: InstrumentMetadata
  gaps: DataGap[]
  totalMissingTime: number // in minutes
  completenessRatio: number
  lastAnalyzed: Date
}

interface GapSummaryStats {
  totalInstruments: number
  instrumentsWithGaps: number
  totalGaps: number
  highSeverityGaps: number
  averageCompleteness: number
  totalMissingTime: number
}

interface TimelineGap {
  date: string
  gaps: number
  severity: 'low' | 'medium' | 'high'
  totalMissingMinutes: number
}

export const GapAnalysisView: React.FC = () => {
  const [gapData, setGapData] = useState<GapAnalysisData[]>([])
  const [summaryStats, setSummaryStats] = useState<GapSummaryStats | null>(null)
  const [timelineData, setTimelineData] = useState<TimelineGap[]>([])
  const [loading, setLoading] = useState(true)
  const [selectedInstrument, setSelectedInstrument] = useState<string>('all')
  const [selectedTimeframe, setSelectedTimeframe] = useState<string>('7d')
  const [severityFilter, setSeverityFilter] = useState<string>('all')
  const [dateRange, setDateRange] = useState<[dayjs.Dayjs, dayjs.Dayjs] | null>(null)
  const [showOnlyWithGaps, setShowOnlyWithGaps] = useState(false)
  const [minGapDuration, setMinGapDuration] = useState<number>(1) // minutes
  const [availableInstruments, setAvailableInstruments] = useState<InstrumentMetadata[]>([])

  useEffect(() => {
    loadData()
    loadAvailableInstruments()
  }, [selectedTimeframe, severityFilter, dateRange])

  const loadAvailableInstruments = async () => {
    try {
      const catalog = await dataCatalogService.getCatalog()
      setAvailableInstruments(catalog.instruments)
    } catch (error) {
      console.error('Failed to load instruments:', error)
    }
  }

  const loadData = async () => {
    try {
      setLoading(true)
      
      // In a real implementation, this would call the API with filters
      // For now, we'll generate mock data
      const mockGapData: GapAnalysisData[] = [
        {
          instrument: {
            instrumentId: 'EURUSD.SIM',
            venue: 'SIM',
            symbol: 'EUR/USD',
            assetClass: 'Currency',
            currency: 'USD',
            dataType: 'tick',
            timeframes: ['1-MINUTE', '5-MINUTE'],
            dateRange: {
              start: new Date('2024-01-01'),
              end: new Date('2024-01-31')
            },
            recordCount: 1250000,
            qualityScore: 0.92,
            gaps: [],
            lastUpdated: new Date()
          },
          gaps: [
            {
              id: 'gap_001',
              start: new Date('2024-01-15T09:30:00'),
              end: new Date('2024-01-15T09:45:00'),
              severity: 'medium',
              reason: 'Market data feed interruption',
              detectedAt: new Date('2024-01-15T10:00:00')
            },
            {
              id: 'gap_002',
              start: new Date('2024-01-20T14:15:00'),
              end: new Date('2024-01-20T14:20:00'),
              severity: 'low',
              reason: 'Brief connection timeout',
              detectedAt: new Date('2024-01-20T14:25:00')
            }
          ],
          totalMissingTime: 20, // 15 + 5 minutes
          completenessRatio: 0.998,
          lastAnalyzed: new Date()
        },
        {
          instrument: {
            instrumentId: 'GBPUSD.SIM',
            venue: 'SIM',
            symbol: 'GBP/USD',
            assetClass: 'Currency',
            currency: 'USD',
            dataType: 'tick',
            timeframes: ['1-MINUTE'],
            dateRange: {
              start: new Date('2024-01-01'),
              end: new Date('2024-01-31')
            },
            recordCount: 980000,
            qualityScore: 0.85,
            gaps: [],
            lastUpdated: new Date()
          },
          gaps: [
            {
              id: 'gap_003',
              start: new Date('2024-01-10T16:00:00'),
              end: new Date('2024-01-10T17:30:00'),
              severity: 'high',
              reason: 'System maintenance window',
              detectedAt: new Date('2024-01-10T17:35:00')
            }
          ],
          totalMissingTime: 90, // 1.5 hours
          completenessRatio: 0.94,
          lastAnalyzed: new Date()
        }
      ]

      // Generate timeline data
      const mockTimelineData: TimelineGap[] = Array.from({ length: 7 }, (_, i) => ({
        date: dayjs().subtract(6 - i, 'day').format('YYYY-MM-DD'),
        gaps: Math.floor(Math.random() * 10),
        severity: Math.random() > 0.7 ? 'high' : Math.random() > 0.4 ? 'medium' : 'low',
        totalMissingMinutes: Math.floor(Math.random() * 120)
      }))

      // Calculate summary stats
      const stats: GapSummaryStats = {
        totalInstruments: mockGapData.length,
        instrumentsWithGaps: mockGapData.filter(d => d.gaps.length > 0).length,
        totalGaps: mockGapData.reduce((sum, d) => sum + d.gaps.length, 0),
        highSeverityGaps: mockGapData.reduce((sum, d) => sum + d.gaps.filter(g => g.severity === 'high').length, 0),
        averageCompleteness: mockGapData.reduce((sum, d) => sum + d.completenessRatio, 0) / mockGapData.length,
        totalMissingTime: mockGapData.reduce((sum, d) => sum + d.totalMissingTime, 0)
      }

      setGapData(mockGapData)
      setTimelineData(mockTimelineData)
      setSummaryStats(stats)
    } catch (error) {
      console.error('Failed to load gap analysis data:', error)
      notification.error({
        message: 'Failed to load gap analysis',
        description: 'Unable to retrieve gap analysis data'
      })
    } finally {
      setLoading(false)
    }
  }

  const analyzeGaps = async (instrumentId?: string) => {
    try {
      setLoading(true)
      
      if (instrumentId) {
        await dataCatalogService.analyzeDataGaps(instrumentId)
        notification.success({
          message: 'Gap analysis completed',
          description: `Gap analysis completed for ${instrumentId}`
        })
      } else {
        // Analyze all instruments
        notification.success({
          message: 'Gap analysis started',
          description: 'Analyzing gaps for all instruments...'
        })
      }
      
      await loadData()
    } catch (error) {
      notification.error({
        message: 'Gap analysis failed',
        description: 'Failed to perform gap analysis'
      })
    } finally {
      setLoading(false)
    }
  }

  const filteredGapData = useMemo(() => {
    let filtered = gapData

    if (selectedInstrument !== 'all') {
      filtered = filtered.filter(d => d.instrument.instrumentId === selectedInstrument)
    }

    if (showOnlyWithGaps) {
      filtered = filtered.filter(d => d.gaps.length > 0)
    }

    if (severityFilter !== 'all') {
      filtered = filtered.filter(d => 
        d.gaps.some(gap => gap.severity === severityFilter)
      )
    }

    if (minGapDuration > 1) {
      filtered = filtered.map(d => ({
        ...d,
        gaps: d.gaps.filter(gap => {
          const durationMinutes = (gap.end.getTime() - gap.start.getTime()) / (1000 * 60)
          return durationMinutes >= minGapDuration
        })
      })).filter(d => d.gaps.length > 0)
    }

    return filtered
  }, [gapData, selectedInstrument, showOnlyWithGaps, severityFilter, minGapDuration])

  const getSeverityColor = (severity: 'low' | 'medium' | 'high') => {
    switch (severity) {
      case 'high': return 'red'
      case 'medium': return 'orange'
      case 'low': return 'yellow'
      default: return 'default'
    }
  }

  const getSeverityIcon = (severity: 'low' | 'medium' | 'high') => {
    switch (severity) {
      case 'high': return <ExclamationCircleOutlined />
      case 'medium': return <WarningOutlined />
      case 'low': return <BugOutlined />
      default: return <ClockCircleOutlined />
    }
  }

  const formatDuration = (minutes: number) => {
    if (minutes < 60) return `${minutes}m`
    const hours = Math.floor(minutes / 60)
    const remainingMinutes = minutes % 60
    return remainingMinutes > 0 ? `${hours}h ${remainingMinutes}m` : `${hours}h`
  }

  const instrumentColumns: ColumnsType<GapAnalysisData> = [
    {
      title: 'Instrument',
      key: 'instrument',
      render: (_, record) => (
        <Space>
          <span style={{ fontWeight: 500 }}>{record.instrument.symbol}</span>
          <Tag>{record.instrument.venue}</Tag>
        </Space>
      )
    },
    {
      title: 'Completeness',
      key: 'completeness',
      render: (_, record) => (
        <Space>
          <Progress
            percent={Math.round(record.completenessRatio * 100)}
           
            strokeColor={record.completenessRatio >= 0.95 ? '#52c41a' : record.completenessRatio >= 0.85 ? '#faad14' : '#ff4d4f'}
            style={{ width: 100 }}
          />
          <span>{Math.round(record.completenessRatio * 100)}%</span>
        </Space>
      ),
      sorter: (a, b) => a.completenessRatio - b.completenessRatio
    },
    {
      title: 'Total Gaps',
      key: 'totalGaps',
      render: (_, record) => (
        <Space>
          <span>{record.gaps.length}</span>
          {record.gaps.length > 0 && (
            <div>
              {record.gaps.filter(g => g.severity === 'high').length > 0 && (
                <Tag color="red">{record.gaps.filter(g => g.severity === 'high').length} high</Tag>
              )}
              {record.gaps.filter(g => g.severity === 'medium').length > 0 && (
                <Tag color="orange">{record.gaps.filter(g => g.severity === 'medium').length} medium</Tag>
              )}
              {record.gaps.filter(g => g.severity === 'low').length > 0 && (
                <Tag color="yellow">{record.gaps.filter(g => g.severity === 'low').length} low</Tag>
              )}
            </div>
          )}
        </Space>
      ),
      sorter: (a, b) => a.gaps.length - b.gaps.length
    },
    {
      title: 'Missing Time',
      key: 'missingTime',
      render: (_, record) => (
        <span style={{ color: record.totalMissingTime > 60 ? '#ff4d4f' : '#666' }}>
          {formatDuration(record.totalMissingTime)}
        </span>
      ),
      sorter: (a, b) => a.totalMissingTime - b.totalMissingTime
    },
    {
      title: 'Last Analyzed',
      key: 'lastAnalyzed',
      render: (_, record) => (
        <Tooltip title={dayjs(record.lastAnalyzed).format('YYYY-MM-DD HH:mm:ss')}>
          {dayjs(record.lastAnalyzed).fromNow()}
        </Tooltip>
      )
    },
    {
      title: 'Actions',
      key: 'actions',
      render: (_, record) => (
        <Space>
          <Button
           
            icon={<FileSearchOutlined />}
            onClick={() => analyzeGaps(record.instrument.instrumentId)}
          >
            Analyze
          </Button>
          <Button
           
            type="link"
            onClick={() => console.log('View details:', record.instrument.instrumentId)}
          >
            Details
          </Button>
        </Space>
      )
    }
  ]

  const gapDetailColumns: ColumnsType<DataGap> = [
    {
      title: 'Start Time',
      dataIndex: 'start',
      key: 'start',
      render: (date) => dayjs(date).format('YYYY-MM-DD HH:mm:ss')
    },
    {
      title: 'End Time',
      dataIndex: 'end',
      key: 'end',
      render: (date) => dayjs(date).format('YYYY-MM-DD HH:mm:ss')
    },
    {
      title: 'Duration',
      key: 'duration',
      render: (_, record) => {
        const durationMinutes = (record.end.getTime() - record.start.getTime()) / (1000 * 60)
        return formatDuration(Math.round(durationMinutes))
      }
    },
    {
      title: 'Severity',
      dataIndex: 'severity',
      key: 'severity',
      render: (severity) => (
        <Tag color={getSeverityColor(severity)} icon={getSeverityIcon(severity)}>
          {severity.toUpperCase()}
        </Tag>
      )
    },
    {
      title: 'Reason',
      dataIndex: 'reason',
      key: 'reason',
      ellipsis: true
    },
    {
      title: 'Detected',
      dataIndex: 'detectedAt',
      key: 'detectedAt',
      render: (date) => dayjs(date).format('MM/DD HH:mm')
    }
  ]

  const allGaps = filteredGapData.flatMap(d => 
    d.gaps.map(gap => ({ ...gap, instrumentId: d.instrument.instrumentId, symbol: d.instrument.symbol }))
  )

  if (loading && !summaryStats) {
    return (
      <div style={{ textAlign: 'center', padding: '60px 0' }}>
        <Spin size="large" />
        <div style={{ marginTop: 16 }}>Loading gap analysis...</div>
      </div>
    )
  }

  return (
    <div>
      {/* Controls */}
      <Row gutter={[16, 16]} style={{ marginBottom: 16 }}>
        <Col span={6}>
          <Select
            placeholder="Select instrument"
            value={selectedInstrument}
            onChange={setSelectedInstrument}
            style={{ width: '100%' }}
          >
            <Option value="all">All Instruments</Option>
            {availableInstruments.map(inst => (
              <Option key={inst.instrumentId} value={inst.instrumentId}>
                {inst.symbol} ({inst.venue})
              </Option>
            ))}
          </Select>
        </Col>
        <Col span={4}>
          <Select
            value={selectedTimeframe}
            onChange={setSelectedTimeframe}
            style={{ width: '100%' }}
          >
            <Option value="24h">Last 24 Hours</Option>
            <Option value="7d">Last 7 Days</Option>
            <Option value="30d">Last 30 Days</Option>
            <Option value="90d">Last 90 Days</Option>
          </Select>
        </Col>
        <Col span={4}>
          <Select
            placeholder="Severity"
            value={severityFilter}
            onChange={setSeverityFilter}
            style={{ width: '100%' }}
          >
            <Option value="all">All Severities</Option>
            <Option value="high">High</Option>
            <Option value="medium">Medium</Option>
            <Option value="low">Low</Option>
          </Select>
        </Col>
        <Col span={4}>
          <InputNumber
            placeholder="Min duration (min)"
            value={minGapDuration}
            onChange={(value) => setMinGapDuration(value as number)}
            min={1}
            style={{ width: '100%' }}
          />
        </Col>
        <Col span={3}>
          <Space>
            <span>Gaps only:</span>
            <Switch checked={showOnlyWithGaps} onChange={setShowOnlyWithGaps} />
          </Space>
        </Col>
        <Col span={3}>
          <Button
            icon={<ReloadOutlined />}
            onClick={() => loadData()}
            loading={loading}
            block
          >
            Refresh
          </Button>
        </Col>
      </Row>

      {/* Summary Statistics */}
      {summaryStats && (
        <Row gutter={[16, 16]} style={{ marginBottom: 16 }}>
          <Col span={6}>
            <Card>
              <Statistic
                title="Instruments with Gaps"
                value={`${summaryStats.instrumentsWithGaps}/${summaryStats.totalInstruments}`}
                suffix={`(${Math.round((summaryStats.instrumentsWithGaps / summaryStats.totalInstruments) * 100)}%)`}
                prefix={<WarningOutlined />}
                valueStyle={{ color: summaryStats.instrumentsWithGaps > 0 ? '#faad14' : '#52c41a' }}
              />
            </Card>
          </Col>
          <Col span={6}>
            <Card>
              <Statistic
                title="Total Gaps"
                value={summaryStats.totalGaps}
                prefix={<ExclamationCircleOutlined />}
                valueStyle={{ color: summaryStats.totalGaps > 0 ? '#ff4d4f' : '#52c41a' }}
              />
            </Card>
          </Col>
          <Col span={6}>
            <Card>
              <Statistic
                title="Average Completeness"
                value={Math.round(summaryStats.averageCompleteness * 100)}
                suffix="%"
                prefix={<CheckCircleOutlined />}
                valueStyle={{ color: summaryStats.averageCompleteness >= 0.95 ? '#52c41a' : '#faad14' }}
              />
            </Card>
          </Col>
          <Col span={6}>
            <Card>
              <Statistic
                title="Total Missing Time"
                value={formatDuration(summaryStats.totalMissingTime)}
                prefix={<ClockCircleOutlined />}
                valueStyle={{ color: summaryStats.totalMissingTime > 60 ? '#ff4d4f' : '#666' }}
              />
            </Card>
          </Col>
        </Row>
      )}

      {/* Gap Analysis Results */}
      <Row gutter={[16, 16]}>
        <Col span={16}>
          <Card
            title="Gap Analysis by Instrument"
            extra={
              <Space>
                <Button
                  icon={<FileSearchOutlined />}
                  onClick={() => analyzeGaps()}
                  loading={loading}
                >
                  Analyze All
                </Button>
                <Button icon={<DownloadOutlined />}>
                  Export
                </Button>
              </Space>
            }
          >
            <Table
              columns={instrumentColumns}
              dataSource={filteredGapData}
              rowKey="instrument.instrumentId"
              pagination={{ pageSize: 10 }}
              loading={loading}
              expandable={{
                expandedRowRender: (record) => (
                  <div style={{ margin: 0 }}>
                    <h4>Gap Details for {record.instrument.symbol}</h4>
                    {record.gaps.length > 0 ? (
                      <Table
                        columns={gapDetailColumns}
                        dataSource={record.gaps}
                        rowKey="id"
                        pagination={false}
                       
                      />
                    ) : (
                      <Empty
                        image={Empty.PRESENTED_IMAGE_SIMPLE}
                        description="No gaps detected"
                      />
                    )}
                  </div>
                )
              }}
            />
          </Card>
        </Col>

        <Col span={8}>
          <Card title="Gap Timeline" style={{ marginBottom: 16 }}>
            <Timeline
              items={timelineData.slice(0, 5).map(item => ({
                color: getSeverityColor(item.severity),
                children: (
                  <div>
                    <div style={{ fontWeight: 500 }}>{dayjs(item.date).format('MMM DD')}</div>
                    <div style={{ fontSize: '12px', color: '#666' }}>
                      {item.gaps} gaps, {formatDuration(item.totalMissingMinutes)} missing
                    </div>
                  </div>
                )
              }))}
            />
          </Card>

          <Card title="Recent Critical Gaps">
            <div style={{ maxHeight: 300, overflow: 'auto' }}>
              {allGaps.filter(gap => gap.severity === 'high').slice(0, 5).map(gap => (
                <Alert
                  key={gap.id}
                  message={`${gap.symbol} - ${gap.severity.toUpperCase()}`}
                  description={
                    <div>
                      <div>{dayjs(gap.start).format('MM/DD HH:mm')} - {dayjs(gap.end).format('HH:mm')}</div>
                      <div style={{ fontSize: '12px' }}>{gap.reason}</div>
                    </div>
                  }
                  type="error"
                 
                  style={{ marginBottom: 8 }}
                />
              ))}
              {allGaps.filter(gap => gap.severity === 'high').length === 0 && (
                <Empty
                  image={Empty.PRESENTED_IMAGE_SIMPLE}
                  description="No critical gaps"
                />
              )}
            </div>
          </Card>
        </Col>
      </Row>
    </div>
  )
}

export default GapAnalysisView