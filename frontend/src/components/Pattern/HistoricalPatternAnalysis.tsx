/**
 * Historical Pattern Analysis Component
 * Analyzes historical pattern performance and provides insights
 */

import React, { useState, useEffect, useMemo } from 'react'
import { 
  Card, 
  Table, 
  Select, 
  DatePicker, 
  Button,
  Space,
  Statistic,
  Row,
  Col,
  Progress,
  Tag,
  Tooltip,
  Modal,
  Tabs,
  Timeline,
  Empty
} from 'antd'
import { 
  BarChartOutlined, 
  ArrowUpOutlined,
  ArrowDownOutlined,
  InfoCircleOutlined,
  DownloadOutlined,
  FilterOutlined
} from '@ant-design/icons'
import { Column } from '@ant-design/plots'
import dayjs from 'dayjs'
import { ChartPattern, PatternDefinition } from '../../types/charting'
import { patternRecognition } from '../../services/patternRecognition'

const { RangePicker } = DatePicker
const { Option } = Select

interface HistoricalPattern extends ChartPattern {
  outcome: 'success' | 'failure' | 'partial' | 'pending'
  targetHit: boolean
  stopLossHit: boolean
  maxGain: number
  maxLoss: number
  daysToCompletion: number
  volumeAtDetection: number
  marketCondition: 'bull' | 'bear' | 'sideways'
}

interface PatternPerformance {
  patternId: string
  patternName: string
  totalOccurrences: number
  successfulOccurrences: number
  avgSuccessTime: number
  avgGain: number
  avgLoss: number
  successRate: number
  profitFactor: number
  bestPerformance: number
  worstPerformance: number
  reliability: 'high' | 'medium' | 'low'
}

interface AnalysisFilters {
  dateRange: [dayjs.Dayjs, dayjs.Dayjs] | null
  instruments: string[]
  timeframes: string[]
  patterns: string[]
  marketConditions: string[]
  minConfidence: number
}

export const HistoricalPatternAnalysis: React.FC = () => {
  const [historicalData, setHistoricalData] = useState<HistoricalPattern[]>([])
  const [loading, setLoading] = useState(false)
  const [filters, setFilters] = useState<AnalysisFilters>({
    dateRange: [dayjs().subtract(1, 'year'), dayjs()],
    instruments: [],
    timeframes: [],
    patterns: [],
    marketConditions: [],
    minConfidence: 0.5
  })
  const [selectedPattern, setSelectedPattern] = useState<PatternPerformance | null>(null)
  const [isDetailModalVisible, setIsDetailModalVisible] = useState(false)

  useEffect(() => {
    loadHistoricalData()
  }, [filters])

  const loadHistoricalData = async () => {
    setLoading(true)
    try {
      // In a real implementation, this would fetch from an API
      // For now, we'll generate mock historical data
      const mockData = generateMockHistoricalData()
      setHistoricalData(mockData)
    } catch (error) {
      console.error('Failed to load historical data:', error)
    } finally {
      setLoading(false)
    }
  }

  const generateMockHistoricalData = (): HistoricalPattern[] => {
    const patterns = patternRecognition.getPatternDefinitions()
    const instruments = ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'AMZN', 'META', 'NFLX']
    const timeframes = ['1h', '4h', '1d']
    const mockData: HistoricalPattern[] = []

    const startDate = filters.dateRange?.[0] || dayjs().subtract(1, 'year')
    const endDate = filters.dateRange?.[1] || dayjs()

    for (let i = 0; i < 500; i++) {
      const randomDate = dayjs(startDate).add(Math.random() * startDate.diff(endDate, 'day'), 'day')
      const randomPattern = patterns[Math.floor(Math.random() * patterns.length)]
      const randomInstrument = instruments[Math.floor(Math.random() * instruments.length)]
      const randomTimeframe = timeframes[Math.floor(Math.random() * timeframes.length)]
      
      const confidence = Math.random() * 0.4 + 0.6
      const outcome = Math.random() > 0.4 ? 'success' : Math.random() > 0.7 ? 'failure' : 'partial'
      const daysToCompletion = Math.floor(Math.random() * 30) + 1
      const maxGain = outcome === 'success' ? Math.random() * 15 + 2 : Math.random() * 5
      const maxLoss = outcome === 'failure' ? -(Math.random() * 10 + 1) : -(Math.random() * 3)

      if (filters.instruments.length === 0 || filters.instruments.includes(randomInstrument)) {
        if (filters.timeframes.length === 0 || filters.timeframes.includes(randomTimeframe)) {
          if (filters.patterns.length === 0 || filters.patterns.includes(randomPattern.id)) {
            if (confidence >= filters.minConfidence) {
              mockData.push({
                id: `hist_${i}`,
                name: randomPattern.name,
                type: randomPattern.type,
                confidence,
                coordinates: { points: [], boundingBox: { left: 0, top: 0, right: 0, bottom: 0 } },
                timeframe: randomTimeframe,
                status: 'completed',
                detectedAt: randomDate.toISOString(),
                instrument: randomInstrument,
                outcome,
                targetHit: outcome === 'success',
                stopLossHit: outcome === 'failure',
                maxGain,
                maxLoss,
                daysToCompletion,
                volumeAtDetection: Math.floor(Math.random() * 1000000) + 100000,
                marketCondition: Math.random() > 0.6 ? 'bull' : Math.random() > 0.3 ? 'bear' : 'sideways'
              })
            }
          }
        }
      }
    }

    return mockData.sort((a, b) => new Date(b.detectedAt).getTime() - new Date(a.detectedAt).getTime())
  }

  const patternPerformanceData = useMemo((): PatternPerformance[] => {
    const performanceMap = new Map<string, HistoricalPattern[]>()

    // Group patterns by type
    historicalData.forEach(pattern => {
      const key = `${pattern.type}_${pattern.name}`
      if (!performanceMap.has(key)) {
        performanceMap.set(key, [])
      }
      performanceMap.get(key)!.push(pattern)
    })

    // Calculate performance metrics
    return Array.from(performanceMap.entries()).map(([key, patterns]) => {
      const [patternId, patternName] = key.split('_', 2)
      const totalOccurrences = patterns.length
      const successfulOccurrences = patterns.filter(p => p.outcome === 'success').length
      const avgSuccessTime = patterns
        .filter(p => p.outcome === 'success')
        .reduce((sum, p) => sum + p.daysToCompletion, 0) / successfulOccurrences || 0
      
      const gains = patterns.filter(p => p.maxGain > 0).map(p => p.maxGain)
      const losses = patterns.filter(p => p.maxLoss < 0).map(p => Math.abs(p.maxLoss))
      
      const avgGain = gains.length > 0 ? gains.reduce((sum, g) => sum + g, 0) / gains.length : 0
      const avgLoss = losses.length > 0 ? losses.reduce((sum, l) => sum + l, 0) / losses.length : 0
      const successRate = successfulOccurrences / totalOccurrences
      const profitFactor = avgLoss > 0 ? avgGain / avgLoss : 0
      
      const bestPerformance = Math.max(...patterns.map(p => p.maxGain))
      const worstPerformance = Math.min(...patterns.map(p => p.maxLoss))
      
      let reliability: 'high' | 'medium' | 'low' = 'low'
      if (successRate > 0.7 && totalOccurrences > 20) reliability = 'high'
      else if (successRate > 0.5 && totalOccurrences > 10) reliability = 'medium'

      return {
        patternId,
        patternName,
        totalOccurrences,
        successfulOccurrences,
        avgSuccessTime,
        avgGain,
        avgLoss,
        successRate,
        profitFactor,
        bestPerformance,
        worstPerformance,
        reliability
      }
    }).sort((a, b) => b.successRate - a.successRate)
  }, [historicalData])

  const overallStats = useMemo(() => {
    const total = historicalData.length
    const successful = historicalData.filter(p => p.outcome === 'success').length
    const avgConfidence = historicalData.reduce((sum, p) => sum + p.confidence, 0) / total || 0
    const avgGain = historicalData
      .filter(p => p.outcome === 'success')
      .reduce((sum, p) => sum + p.maxGain, 0) / successful || 0

    return {
      totalPatterns: total,
      successRate: successful / total || 0,
      avgConfidence,
      avgGain,
      avgCompletionTime: historicalData.reduce((sum, p) => sum + p.daysToCompletion, 0) / total || 0
    }
  }, [historicalData])

  const performanceColumns = [
    {
      title: 'Pattern',
      dataIndex: 'patternName',
      key: 'patternName',
      render: (text: string, record: PatternPerformance) => (
        <div>
          <div style={{ fontWeight: 500 }}>{text}</div>
          <Tag color={record.reliability === 'high' ? 'green' : record.reliability === 'medium' ? 'orange' : 'red'}>
            {record.reliability} reliability
          </Tag>
        </div>
      )
    },
    {
      title: 'Occurrences',
      dataIndex: 'totalOccurrences',
      key: 'totalOccurrences',
      align: 'center' as const,
      sorter: (a: PatternPerformance, b: PatternPerformance) => a.totalOccurrences - b.totalOccurrences
    },
    {
      title: 'Success Rate',
      key: 'successRate',
      align: 'center' as const,
      render: (record: PatternPerformance) => (
        <div>
          <Progress 
            percent={record.successRate * 100}
            size="small"
            strokeColor={record.successRate > 0.7 ? '#52c41a' : record.successRate > 0.5 ? '#faad14' : '#ff4d4f'}
          />
          <div style={{ fontSize: '12px', marginTop: 4 }}>
            {record.successfulOccurrences}/{record.totalOccurrences}
          </div>
        </div>
      ),
      sorter: (a: PatternPerformance, b: PatternPerformance) => a.successRate - b.successRate
    },
    {
      title: 'Avg Gain',
      key: 'avgGain',
      align: 'right' as const,
      render: (record: PatternPerformance) => (
        <span style={{ color: '#52c41a' }}>
          +{record.avgGain.toFixed(2)}%
        </span>
      ),
      sorter: (a: PatternPerformance, b: PatternPerformance) => a.avgGain - b.avgGain
    },
    {
      title: 'Avg Loss',
      key: 'avgLoss',
      align: 'right' as const,
      render: (record: PatternPerformance) => (
        <span style={{ color: '#ff4d4f' }}>
          -{record.avgLoss.toFixed(2)}%
        </span>
      ),
      sorter: (a: PatternPerformance, b: PatternPerformance) => a.avgLoss - b.avgLoss
    },
    {
      title: 'Profit Factor',
      key: 'profitFactor',
      align: 'right' as const,
      render: (record: PatternPerformance) => (
        <span style={{ color: record.profitFactor > 1 ? '#52c41a' : '#ff4d4f' }}>
          {record.profitFactor.toFixed(2)}
        </span>
      ),
      sorter: (a: PatternPerformance, b: PatternPerformance) => a.profitFactor - b.profitFactor
    },
    {
      title: 'Actions',
      key: 'actions',
      align: 'center' as const,
      render: (record: PatternPerformance) => (
        <Button
          type="link"
          size="small"
          icon={<InfoCircleOutlined />}
          onClick={() => {
            setSelectedPattern(record)
            setIsDetailModalVisible(true)
          }}
        />
      )
    }
  ]

  const chartData = patternPerformanceData.slice(0, 10).map(pattern => ({
    pattern: pattern.patternName,
    successRate: pattern.successRate * 100,
    occurrences: pattern.totalOccurrences
  }))

  return (
    <Card title="Historical Pattern Analysis" size="small">
      {/* Filters */}
      <div style={{ marginBottom: 16 }}>
        <Space wrap>
          <RangePicker
            value={filters.dateRange}
            onChange={(dates) => setFilters({ ...filters, dateRange: dates })}
            size="small"
          />
          
          <Select
            mode="multiple"
            placeholder="Instruments"
            style={{ minWidth: 120 }}
            value={filters.instruments}
            onChange={(instruments) => setFilters({ ...filters, instruments })}
            size="small"
          >
            <Option value="AAPL">AAPL</Option>
            <Option value="MSFT">MSFT</Option>
            <Option value="GOOGL">GOOGL</Option>
            <Option value="TSLA">TSLA</Option>
          </Select>

          <Select
            mode="multiple"
            placeholder="Timeframes"
            style={{ minWidth: 120 }}
            value={filters.timeframes}
            onChange={(timeframes) => setFilters({ ...filters, timeframes })}
            size="small"
          >
            <Option value="1h">1H</Option>
            <Option value="4h">4H</Option>
            <Option value="1d">1D</Option>
          </Select>

          <Button size="small" icon={<FilterOutlined />}>
            More Filters
          </Button>

          <Button size="small" icon={<DownloadOutlined />}>
            Export Report
          </Button>
        </Space>
      </div>

      <Tabs
        items={[
          {
            key: 'overview',
            label: 'Overview',
            children: (
              <div>
                {/* Overall Statistics */}
                <Row gutter={16} style={{ marginBottom: 24 }}>
                  <Col span={6}>
                    <Card size="small">
                      <Statistic
                        title="Total Patterns"
                        value={overallStats.totalPatterns}
                        prefix={<BarChartOutlined />}
                      />
                    </Card>
                  </Col>
                  <Col span={6}>
                    <Card size="small">
                      <Statistic
                        title="Success Rate"
                        value={overallStats.successRate * 100}
                        precision={1}
                        suffix="%"
                        valueStyle={{ color: overallStats.successRate > 0.6 ? '#3f8600' : '#cf1322' }}
                        prefix={overallStats.successRate > 0.6 ? <ArrowUpOutlined /> : <ArrowDownOutlined />}
                      />
                    </Card>
                  </Col>
                  <Col span={6}>
                    <Card size="small">
                      <Statistic
                        title="Avg Confidence"
                        value={overallStats.avgConfidence * 100}
                        precision={1}
                        suffix="%"
                      />
                    </Card>
                  </Col>
                  <Col span={6}>
                    <Card size="small">
                      <Statistic
                        title="Avg Gain"
                        value={overallStats.avgGain}
                        precision={2}
                        suffix="%"
                        valueStyle={{ color: '#3f8600' }}
                        prefix={<ArrowUpOutlined />}
                      />
                    </Card>
                  </Col>
                </Row>

                {/* Performance Chart */}
                <Card title="Top Performing Patterns" size="small" style={{ marginBottom: 16 }}>
                  {chartData.length > 0 ? (
                    <Column
                      data={chartData}
                      xField="pattern"
                      yField="successRate"
                      height={300}
                      columnWidthRatio={0.6}
                      label={{
                        position: 'middle',
                        style: {
                          fill: '#FFFFFF',
                          opacity: 0.8,
                        },
                      }}
                      meta={{
                        pattern: { alias: 'Pattern' },
                        successRate: { alias: 'Success Rate (%)' }
                      }}
                    />
                  ) : (
                    <Empty description="No data available" />
                  )}
                </Card>
              </div>
            )
          },
          {
            key: 'performance',
            label: 'Performance Table',
            children: (
              <Table
                columns={performanceColumns}
                dataSource={patternPerformanceData}
                rowKey="patternId"
                size="small"
                loading={loading}
                pagination={{
                  pageSize: 10,
                  showSizeChanger: true,
                  showTotal: (total) => `${total} patterns analyzed`
                }}
              />
            )
          },
          {
            key: 'insights',
            label: 'Insights',
            children: (
              <div style={{ display: 'grid', gap: 16 }}>
                <Card title="Key Insights" size="small">
                  <Timeline
                    items={[
                      {
                        children: `Best performing pattern: ${patternPerformanceData[0]?.patternName || 'N/A'} with ${(patternPerformanceData[0]?.successRate * 100 || 0).toFixed(1)}% success rate`
                      },
                      {
                        children: `Most reliable pattern: ${patternPerformanceData.find(p => p.reliability === 'high')?.patternName || 'None'}`
                      },
                      {
                        children: `Average completion time: ${overallStats.avgCompletionTime.toFixed(1)} days`
                      },
                      {
                        children: `Patterns work best in ${Math.random() > 0.5 ? 'bull' : 'bear'} market conditions`
                      }
                    ]}
                  />
                </Card>

                <Card title="Recommendations" size="small">
                  <ul>
                    <li>Focus on patterns with high reliability scores for better consistency</li>
                    <li>Combine pattern recognition with volume confirmation for improved accuracy</li>
                    <li>Consider market conditions when evaluating pattern signals</li>
                    <li>Use profit factors to optimize risk/reward ratios</li>
                  </ul>
                </Card>
              </div>
            )
          }
        ]}
      />

      {/* Pattern Detail Modal */}
      <Modal
        title={selectedPattern ? `${selectedPattern.patternName} Analysis` : 'Pattern Analysis'}
        open={isDetailModalVisible}
        onCancel={() => setIsDetailModalVisible(false)}
        footer={null}
        width={800}
      >
        {selectedPattern && (
          <div style={{ display: 'grid', gap: 16 }}>
            <Row gutter={16}>
              <Col span={8}>
                <Statistic
                  title="Total Occurrences"
                  value={selectedPattern.totalOccurrences}
                />
              </Col>
              <Col span={8}>
                <Statistic
                  title="Success Rate"
                  value={selectedPattern.successRate * 100}
                  precision={1}
                  suffix="%"
                />
              </Col>
              <Col span={8}>
                <Statistic
                  title="Profit Factor"
                  value={selectedPattern.profitFactor}
                  precision={2}
                />
              </Col>
            </Row>

            <Row gutter={16}>
              <Col span={12}>
                <Card size="small" title="Performance Metrics">
                  <div style={{ display: 'grid', gap: 8 }}>
                    <div>Average Gain: <span style={{ color: '#52c41a' }}>+{selectedPattern.avgGain.toFixed(2)}%</span></div>
                    <div>Average Loss: <span style={{ color: '#ff4d4f' }}>-{selectedPattern.avgLoss.toFixed(2)}%</span></div>
                    <div>Best Performance: <span style={{ color: '#52c41a' }}>+{selectedPattern.bestPerformance.toFixed(2)}%</span></div>
                    <div>Worst Performance: <span style={{ color: '#ff4d4f' }}>{selectedPattern.worstPerformance.toFixed(2)}%</span></div>
                    <div>Avg Success Time: {selectedPattern.avgSuccessTime.toFixed(1)} days</div>
                  </div>
                </Card>
              </Col>
              <Col span={12}>
                <Card size="small" title="Pattern Insights">
                  <div style={{ display: 'grid', gap: 8 }}>
                    <div>Reliability: <Tag color={selectedPattern.reliability === 'high' ? 'green' : 'orange'}>{selectedPattern.reliability}</Tag></div>
                    <div>Successful Trades: {selectedPattern.successfulOccurrences}</div>
                    <div>Failed Trades: {selectedPattern.totalOccurrences - selectedPattern.successfulOccurrences}</div>
                  </div>
                </Card>
              </Col>
            </Row>
          </div>
        )}
      </Modal>
    </Card>
  )
}

export default HistoricalPatternAnalysis