import React, { useState, useEffect, useMemo } from 'react'
import {
  Card,
  Row,
  Col,
  Button,
  Select,
  DatePicker,
  Table,
  Statistic,
  Tag,
  Space,
  Typography,
  Alert,
  Tabs,
  Tooltip,
  Progress,
  Collapse,
  Switch,
  Slider,
  InputNumber,
  Form,
  Modal,
  message,
  Segmented
} from 'antd'
import {
  LineChartOutlined,
  BarChartOutlined,
  PieChartOutlined,
  TableOutlined,
  DownloadOutlined,
  SettingOutlined,
  ReloadOutlined,
  FilterOutlined,
  RiseOutlined,
  FallOutlined,
  AlertOutlined,
  StarOutlined,
  ThunderboltOutlined
} from '@ant-design/icons'
import dayjs from 'dayjs'
import { ResponsiveContainer, LineChart, Line, BarChart, Bar, PieChart, Pie, Cell, XAxis, YAxis, CartesianGrid, Tooltip as RechartsTooltip, Legend, ComposedChart, Area, AreaChart, ScatterChart, Scatter, Heatmap } from 'recharts'

const { Title, Text, Paragraph } = Typography
const { TabPane } = Tabs
const { RangePicker } = DatePicker
const { Panel } = Collapse

interface AdvancedMetrics {
  // Risk-Adjusted Returns
  sharpeRatio: number
  sortinoRatio: number
  calmarRatio: number
  omegaRatio: number
  informationRatio: number
  
  // Risk Metrics
  var95: number
  var99: number
  cvar95: number
  maxDrawdown: number
  averageDrawdown: number
  drawdownDuration: number
  
  // Performance Attribution
  alpha: number
  beta: number
  correlation: number
  trackingError: number
  activeReturn: number
  
  // Trade Analysis
  winRate: number
  profitFactor: number
  expectancy: number
  averageWin: number
  averageLoss: number
  largestWin: number
  largestLoss: number
  consecutiveWins: number
  consecutiveLosses: number
  
  // Portfolio Statistics
  totalReturn: number
  annualizedReturn: number
  volatility: number
  skewness: number
  kurtosis: number
  
  // Time-Based Analytics
  bestMonth: { period: string; return: number }
  worstMonth: { period: string; return: number }
  monthlyWinRate: number
  quarterlyReturns: Array<{ quarter: string; return: number }>
}

interface PerformanceBreakdown {
  strategy: string
  allocation: number
  return: number
  contribution: number
  sharpe: number
  maxDrawdown: number
  status: 'active' | 'paused' | 'stopped'
}

interface RiskAttribution {
  factor: string
  contribution: number
  percentage: number
  trend: 'up' | 'down' | 'stable'
}

interface CorrelationMatrix {
  [strategy: string]: { [otherStrategy: string]: number }
}

interface TimeSeriesData {
  date: string
  portfolioValue: number
  benchmark: number
  drawdown: number
  dailyReturn: number
  volatility: number
  rollingSharp: number
}

interface TradeDistribution {
  returnBucket: string
  count: number
  percentage: number
  pnl: number
}

const AdvancedAnalyticsDashboard: React.FC = () => {
  const [metrics, setMetrics] = useState<AdvancedMetrics | null>(null)
  const [performanceBreakdown, setPerformanceBreakdown] = useState<PerformanceBreakdown[]>([])
  const [riskAttribution, setRiskAttribution] = useState<RiskAttribution[]>([])
  const [correlationMatrix, setCorrelationMatrix] = useState<CorrelationMatrix>({})
  const [timeSeriesData, setTimeSeriesData] = useState<TimeSeriesData[]>([])
  const [tradeDistribution, setTradeDistribution] = useState<TradeDistribution[]>([])
  const [dateRange, setDateRange] = useState<[dayjs.Dayjs, dayjs.Dayjs]>([
    dayjs().subtract(3, 'month'),
    dayjs()
  ])
  const [selectedMetrics, setSelectedMetrics] = useState<string[]>(['sharpeRatio', 'maxDrawdown', 'winRate'])
  const [benchmarkEnabled, setBenchmarkEnabled] = useState(true)
  const [analysisType, setAnalysisType] = useState<'portfolio' | 'strategy' | 'risk'>('portfolio')
  const [loading, setLoading] = useState(false)
  const [configModalVisible, setConfigModalVisible] = useState(false)

  const apiUrl = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8001'

  useEffect(() => {
    loadAdvancedAnalytics()
  }, [dateRange, analysisType])

  const loadAdvancedAnalytics = async () => {
    setLoading(true)
    try {
      const [startDate, endDate] = dateRange
      
      // Load all analytics data in parallel
      const [
        metricsResponse,
        breakdownResponse,
        riskResponse,
        correlationResponse,
        timeSeriesResponse,
        distributionResponse
      ] = await Promise.all([
        fetch(`${apiUrl}/api/v1/performance/advanced-metrics?start=${startDate.format('YYYY-MM-DD')}&end=${endDate.format('YYYY-MM-DD')}&type=${analysisType}`),
        fetch(`${apiUrl}/api/v1/performance/breakdown?start=${startDate.format('YYYY-MM-DD')}&end=${endDate.format('YYYY-MM-DD')}`),
        fetch(`${apiUrl}/api/v1/performance/risk-attribution?start=${startDate.format('YYYY-MM-DD')}&end=${endDate.format('YYYY-MM-DD')}`),
        fetch(`${apiUrl}/api/v1/performance/correlation-matrix?start=${startDate.format('YYYY-MM-DD')}&end=${endDate.format('YYYY-MM-DD')}`),
        fetch(`${apiUrl}/api/v1/performance/time-series?start=${startDate.format('YYYY-MM-DD')}&end=${endDate.format('YYYY-MM-DD')}&include_benchmark=${benchmarkEnabled}`),
        fetch(`${apiUrl}/api/v1/performance/trade-distribution?start=${startDate.format('YYYY-MM-DD')}&end=${endDate.format('YYYY-MM-DD')}`)
      ])

      if (metricsResponse.ok) {
        const data = await metricsResponse.json()
        setMetrics(data.metrics)
      }

      if (breakdownResponse.ok) {
        const data = await breakdownResponse.json()
        setPerformanceBreakdown(data.breakdown)
      }

      if (riskResponse.ok) {
        const data = await riskResponse.json()
        setRiskAttribution(data.attribution)
      }

      if (correlationResponse.ok) {
        const data = await correlationResponse.json()
        setCorrelationMatrix(data.matrix)
      }

      if (timeSeriesResponse.ok) {
        const data = await timeSeriesResponse.json()
        setTimeSeriesData(data.timeSeries)
      }

      if (distributionResponse.ok) {
        const data = await distributionResponse.json()
        setTradeDistribution(data.distribution)
      }

    } catch (error) {
      message.error('Failed to load advanced analytics')
      console.error('Analytics load error:', error)
    } finally {
      setLoading(false)
    }
  }

  const exportAnalytics = async (format: 'pdf' | 'excel' | 'csv') => {
    try {
      const [startDate, endDate] = dateRange
      const response = await fetch(`${apiUrl}/api/v1/performance/export`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          startDate: startDate.format('YYYY-MM-DD'),
          endDate: endDate.format('YYYY-MM-DD'),
          format,
          includeCharts: true,
          selectedMetrics
        })
      })

      if (response.ok) {
        const blob = await response.blob()
        const url = window.URL.createObjectURL(blob)
        const link = document.createElement('a')
        link.href = url
        link.download = `analytics-report-${startDate.format('YYYY-MM-DD')}-${endDate.format('YYYY-MM-DD')}.${format}`
        link.click()
        window.URL.revokeObjectURL(url)
        message.success(`Report exported as ${format.toUpperCase()}`)
      } else {
        message.error('Failed to export report')
      }
    } catch (error) {
      message.error('Export failed')
      console.error('Export error:', error)
    }
  }

  const formatNumber = (value: number, precision: number = 2, suffix: string = '') => {
    if (value === null || value === undefined) return 'N/A'
    return `${value.toFixed(precision)}${suffix}`
  }

  const getMetricColor = (value: number, type: 'return' | 'ratio' | 'risk') => {
    switch (type) {
      case 'return':
        return value >= 0 ? '#3f8600' : '#cf1322'
      case 'ratio':
        return value >= 1 ? '#3f8600' : value >= 0.5 ? '#faad14' : '#cf1322'
      case 'risk':
        return value <= 0.05 ? '#3f8600' : value <= 0.1 ? '#faad14' : '#cf1322'
      default:
        return '#1890ff'
    }
  }

  const renderRiskAdjustedMetrics = () => (
    <Row gutter={[16, 16]}>
      {metrics && (
        <>
          <Col xs={12} sm={8} md={6} lg={4}>
            <Card size="small">
              <Statistic
                title="Sharpe Ratio"
                value={metrics.sharpeRatio}
                precision={3}
                valueStyle={{ color: getMetricColor(metrics.sharpeRatio, 'ratio') }}
                prefix={metrics.sharpeRatio > 1 ? <RiseOutlined /> : <FallOutlined />}
              />
            </Card>
          </Col>
          <Col xs={12} sm={8} md={6} lg={4}>
            <Card size="small">
              <Statistic
                title="Sortino Ratio"
                value={metrics.sortinoRatio}
                precision={3}
                valueStyle={{ color: getMetricColor(metrics.sortinoRatio, 'ratio') }}
              />
            </Card>
          </Col>
          <Col xs={12} sm={8} md={6} lg={4}>
            <Card size="small">
              <Statistic
                title="Calmar Ratio"
                value={metrics.calmarRatio}
                precision={3}
                valueStyle={{ color: getMetricColor(metrics.calmarRatio, 'ratio') }}
              />
            </Card>
          </Col>
          <Col xs={12} sm={8} md={6} lg={4}>
            <Card size="small">
              <Statistic
                title="Information Ratio"
                value={metrics.informationRatio}
                precision={3}
                valueStyle={{ color: getMetricColor(metrics.informationRatio, 'ratio') }}
              />
            </Card>
          </Col>
          <Col xs={12} sm={8} md={6} lg={4}>
            <Card size="small">
              <Statistic
                title="Alpha"
                value={metrics.alpha}
                precision={3}
                suffix="%"
                valueStyle={{ color: getMetricColor(metrics.alpha, 'return') }}
              />
            </Card>
          </Col>
          <Col xs={12} sm={8} md={6} lg={4}>
            <Card size="small">
              <Statistic
                title="Beta"
                value={metrics.beta}
                precision={3}
                valueStyle={{ color: metrics.beta <= 1 ? '#3f8600' : '#faad14' }}
              />
            </Card>
          </Col>
        </>
      )}
    </Row>
  )

  const renderPerformanceChart = () => (
    <ResponsiveContainer width="100%" height={400}>
      <ComposedChart data={timeSeriesData}>
        <CartesianGrid strokeDasharray="3 3" />
        <XAxis dataKey="date" tick={{ fontSize: 12 }} />
        <YAxis yAxisId="left" orientation="left" tick={{ fontSize: 12 }} />
        <YAxis yAxisId="right" orientation="right" tick={{ fontSize: 12 }} />
        <RechartsTooltip 
          formatter={(value: any, name: string) => [
            typeof value === 'number' ? formatNumber(value, 2, name.includes('Return') ? '%' : '') : value,
            name
          ]}
          labelFormatter={(label) => `Date: ${label}`}
        />
        <Legend />
        <Area
          yAxisId="left"
          type="monotone"
          dataKey="portfolioValue"
          fill="#1890ff"
          fillOpacity={0.1}
          stroke="#1890ff"
          strokeWidth={2}
          name="Portfolio Value"
        />
        {benchmarkEnabled && (
          <Line
            yAxisId="left"
            type="monotone"
            dataKey="benchmark"
            stroke="#52c41a"
            strokeWidth={2}
            strokeDasharray="5 5"
            name="Benchmark"
            dot={false}
          />
        )}
        <Bar
          yAxisId="right"
          dataKey="drawdown"
          fill="#ff4d4f"
          fillOpacity={0.6}
          name="Drawdown %"
        />
      </ComposedChart>
    </ResponsiveContainer>
  )

  const renderRiskMetrics = () => (
    <Row gutter={[16, 16]}>
      {metrics && (
        <>
          <Col xs={12} sm={8} md={6}>
            <Card size="small">
              <Statistic
                title="VaR (95%)"
                value={metrics.var95}
                precision={2}
                suffix="%"
                valueStyle={{ color: getMetricColor(metrics.var95, 'risk') }}
                prefix={<AlertOutlined />}
              />
            </Card>
          </Col>
          <Col xs={12} sm={8} md={6}>
            <Card size="small">
              <Statistic
                title="CVaR (95%)"
                value={metrics.cvar95}
                precision={2}
                suffix="%"
                valueStyle={{ color: getMetricColor(metrics.cvar95, 'risk') }}
              />
            </Card>
          </Col>
          <Col xs={12} sm={8} md={6}>
            <Card size="small">
              <Statistic
                title="Max Drawdown"
                value={metrics.maxDrawdown}
                precision={2}
                suffix="%"
                valueStyle={{ color: '#cf1322' }}
              />
            </Card>
          </Col>
          <Col xs={12} sm={8} md={6}>
            <Card size="small">
              <Statistic
                title="Volatility"
                value={metrics.volatility}
                precision={2}
                suffix="%"
                valueStyle={{ color: getMetricColor(metrics.volatility, 'risk') }}
              />
            </Card>
          </Col>
        </>
      )}
    </Row>
  )

  const renderCorrelationHeatmap = () => {
    const strategies = Object.keys(correlationMatrix)
    const heatmapData = strategies.map(strategy1 => 
      strategies.map(strategy2 => ({
        x: strategy1,
        y: strategy2,
        value: correlationMatrix[strategy1]?.[strategy2] || 0
      }))
    ).flat()

    return (
      <ResponsiveContainer width="100%" height={300}>
        <ScatterChart>
          <XAxis type="category" dataKey="x" tick={{ fontSize: 10 }} />
          <YAxis type="category" dataKey="y" tick={{ fontSize: 10 }} />
          <RechartsTooltip 
            formatter={(value: any) => [formatNumber(value as number, 3), 'Correlation']}
          />
          <Scatter 
            data={heatmapData} 
            fill={(entry: any) => {
              const value = entry.value
              if (value > 0.7) return '#ff4d4f'
              if (value > 0.3) return '#faad14'
              if (value > -0.3) return '#52c41a'
              return '#1890ff'
            }}
          />
        </ScatterChart>
      </ResponsiveContainer>
    )
  }

  const renderTradeDistribution = () => (
    <ResponsiveContainer width="100%" height={300}>
      <BarChart data={tradeDistribution}>
        <CartesianGrid strokeDasharray="3 3" />
        <XAxis dataKey="returnBucket" tick={{ fontSize: 10 }} />
        <YAxis tick={{ fontSize: 12 }} />
        <RechartsTooltip 
          formatter={(value: any, name: string) => [
            name === 'count' ? value : formatNumber(value as number, 0),
            name === 'count' ? 'Trades' : 'P&L'
          ]}
        />
        <Bar dataKey="count" fill="#1890ff" name="Trade Count" />
        <Bar dataKey="pnl" fill="#52c41a" name="Total P&L" />
      </BarChart>
    </ResponsiveContainer>
  )

  const performanceColumns = [
    {
      title: 'Strategy',
      dataIndex: 'strategy',
      key: 'strategy',
      render: (text: string, record: PerformanceBreakdown) => (
        <Space>
          <Text strong>{text}</Text>
          <Tag color={record.status === 'active' ? 'green' : record.status === 'paused' ? 'orange' : 'red'}>
            {record.status}
          </Tag>
        </Space>
      )
    },
    {
      title: 'Allocation',
      dataIndex: 'allocation',
      key: 'allocation',
      render: (value: number) => (
        <Progress 
          percent={value * 100} 
          size="small" 
          format={(percent) => `${percent?.toFixed(1)}%`}
        />
      )
    },
    {
      title: 'Return',
      dataIndex: 'return',
      key: 'return',
      render: (value: number) => (
        <Text type={value >= 0 ? 'success' : 'danger'}>
          {formatNumber(value, 2, '%')}
        </Text>
      ),
      sorter: (a: PerformanceBreakdown, b: PerformanceBreakdown) => a.return - b.return
    },
    {
      title: 'Contribution',
      dataIndex: 'contribution',
      key: 'contribution',
      render: (value: number) => formatNumber(value, 2, '%'),
      sorter: (a: PerformanceBreakdown, b: PerformanceBreakdown) => a.contribution - b.contribution
    },
    {
      title: 'Sharpe',
      dataIndex: 'sharpe',
      key: 'sharpe',
      render: (value: number) => (
        <Text style={{ color: getMetricColor(value, 'ratio') }}>
          {formatNumber(value, 3)}
        </Text>
      ),
      sorter: (a: PerformanceBreakdown, b: PerformanceBreakdown) => a.sharpe - b.sharpe
    },
    {
      title: 'Max DD',
      dataIndex: 'maxDrawdown',
      key: 'maxDrawdown',
      render: (value: number) => (
        <Text type="danger">{formatNumber(value, 2, '%')}</Text>
      ),
      sorter: (a: PerformanceBreakdown, b: PerformanceBreakdown) => a.maxDrawdown - b.maxDrawdown
    }
  ]

  return (
    <div style={{ padding: 24 }}>
      <Row gutter={[16, 16]}>
        {/* Header Controls */}
        <Col xs={24}>
          <Card>
            <Row justify="space-between" align="middle">
              <Col>
                <Title level={3} style={{ margin: 0 }}>
                  <LineChartOutlined style={{ marginRight: 8 }} />
                  Advanced Analytics Dashboard
                </Title>
                <Text type="secondary">
                  Comprehensive performance and risk analysis
                </Text>
              </Col>
              <Col>
                <Space>
                  <Segmented
                    value={analysisType}
                    onChange={setAnalysisType}
                    options={[
                      { label: 'Portfolio', value: 'portfolio' },
                      { label: 'Strategy', value: 'strategy' },
                      { label: 'Risk', value: 'risk' }
                    ]}
                  />
                  <RangePicker
                    value={dateRange}
                    onChange={(dates) => dates && setDateRange(dates as [dayjs.Dayjs, dayjs.Dayjs])}
                    presets={[
                      { label: 'Last 1M', value: [dayjs().subtract(1, 'month'), dayjs()] },
                      { label: 'Last 3M', value: [dayjs().subtract(3, 'month'), dayjs()] },
                      { label: 'Last 6M', value: [dayjs().subtract(6, 'month'), dayjs()] },
                      { label: 'Last 1Y', value: [dayjs().subtract(1, 'year'), dayjs()] }
                    ]}
                  />
                  <Button 
                    icon={<ReloadOutlined />} 
                    onClick={loadAdvancedAnalytics}
                    loading={loading}
                  >
                    Refresh
                  </Button>
                  <Button 
                    icon={<SettingOutlined />}
                    onClick={() => setConfigModalVisible(true)}
                  >
                    Configure
                  </Button>
                  <Button 
                    type="primary"
                    icon={<DownloadOutlined />}
                    onClick={() => exportAnalytics('pdf')}
                  >
                    Export
                  </Button>
                </Space>
              </Col>
            </Row>
          </Card>
        </Col>

        {/* Main Analytics Content */}
        <Col xs={24}>
          <Tabs defaultActiveKey="overview">
            <TabPane tab="Overview" key="overview">
              <Row gutter={[16, 16]}>
                <Col xs={24}>
                  <Card title="Risk-Adjusted Performance Metrics">
                    {renderRiskAdjustedMetrics()}
                  </Card>
                </Col>
                <Col xs={24}>
                  <Card 
                    title="Portfolio Performance vs Benchmark"
                    extra={
                      <Switch
                        checked={benchmarkEnabled}
                        onChange={setBenchmarkEnabled}
                        checkedChildren="Benchmark"
                        unCheckedChildren="Portfolio Only"
                      />
                    }
                  >
                    {renderPerformanceChart()}
                  </Card>
                </Col>
              </Row>
            </TabPane>

            <TabPane tab="Risk Analysis" key="risk">
              <Row gutter={[16, 16]}>
                <Col xs={24} lg={12}>
                  <Card title="Risk Metrics">
                    {renderRiskMetrics()}
                  </Card>
                </Col>
                <Col xs={24} lg={12}>
                  <Card title="Risk Attribution">
                    <Table
                      dataSource={riskAttribution}
                      columns={[
                        { title: 'Factor', dataIndex: 'factor', key: 'factor' },
                        { 
                          title: 'Contribution', 
                          dataIndex: 'contribution', 
                          key: 'contribution',
                          render: (value: number) => formatNumber(value, 2, '%')
                        },
                        { 
                          title: 'Weight', 
                          dataIndex: 'percentage', 
                          key: 'percentage',
                          render: (value: number) => (
                            <Progress percent={value} size="small" />
                          )
                        },
                        {
                          title: 'Trend',
                          dataIndex: 'trend',
                          key: 'trend',
                          render: (trend: string) => (
                            <Tag color={trend === 'up' ? 'green' : trend === 'down' ? 'red' : 'blue'}>
                              {trend === 'up' ? '↑' : trend === 'down' ? '↓' : '→'} {trend}
                            </Tag>
                          )
                        }
                      ]}
                      size="small"
                      pagination={false}
                    />
                  </Card>
                </Col>
                <Col xs={24}>
                  <Card title="Strategy Correlation Matrix">
                    {renderCorrelationHeatmap()}
                  </Card>
                </Col>
              </Row>
            </TabPane>

            <TabPane tab="Performance Attribution" key="attribution">
              <Row gutter={[16, 16]}>
                <Col xs={24}>
                  <Card title="Strategy Performance Breakdown">
                    <Table
                      dataSource={performanceBreakdown}
                      columns={performanceColumns}
                      rowKey="strategy"
                      size="middle"
                      pagination={{ pageSize: 10 }}
                    />
                  </Card>
                </Col>
                <Col xs={24} lg={12}>
                  <Card title="Trade Return Distribution">
                    {renderTradeDistribution()}
                  </Card>
                </Col>
                <Col xs={24} lg={12}>
                  <Card title="Monthly Performance">
                    {metrics && (
                      <div>
                        <Row gutter={16}>
                          <Col span={12}>
                            <Statistic
                              title="Best Month"
                              value={metrics.bestMonth.return}
                              precision={2}
                              suffix="%"
                              valueStyle={{ color: '#3f8600' }}
                              prefix={<StarOutlined />}
                            />
                            <Text type="secondary">{metrics.bestMonth.period}</Text>
                          </Col>
                          <Col span={12}>
                            <Statistic
                              title="Worst Month"
                              value={metrics.worstMonth.return}
                              precision={2}
                              suffix="%"
                              valueStyle={{ color: '#cf1322' }}
                              prefix={<AlertOutlined />}
                            />
                            <Text type="secondary">{metrics.worstMonth.period}</Text>
                          </Col>
                        </Row>
                        <div style={{ marginTop: 16 }}>
                          <Text>Monthly Win Rate: </Text>
                          <Progress 
                            percent={metrics.monthlyWinRate * 100} 
                            size="small"
                            format={(percent) => `${percent?.toFixed(1)}%`}
                          />
                        </div>
                      </div>
                    )}
                  </Card>
                </Col>
              </Row>
            </TabPane>

            <TabPane tab="Trade Analysis" key="trades">
              <Row gutter={[16, 16]}>
                {metrics && (
                  <>
                    <Col xs={12} sm={8} md={6} lg={4}>
                      <Card size="small">
                        <Statistic
                          title="Win Rate"
                          value={metrics.winRate}
                          precision={1}
                          suffix="%"
                          valueStyle={{ color: getMetricColor(metrics.winRate, 'ratio') }}
                        />
                      </Card>
                    </Col>
                    <Col xs={12} sm={8} md={6} lg={4}>
                      <Card size="small">
                        <Statistic
                          title="Profit Factor"
                          value={metrics.profitFactor}
                          precision={2}
                          valueStyle={{ color: getMetricColor(metrics.profitFactor, 'ratio') }}
                        />
                      </Card>
                    </Col>
                    <Col xs={12} sm={8} md={6} lg={4}>
                      <Card size="small">
                        <Statistic
                          title="Expectancy"
                          value={metrics.expectancy}
                          precision={2}
                          valueStyle={{ color: getMetricColor(metrics.expectancy, 'return') }}
                        />
                      </Card>
                    </Col>
                    <Col xs={12} sm={8} md={6} lg={4}>
                      <Card size="small">
                        <Statistic
                          title="Avg Win"
                          value={metrics.averageWin}
                          precision={2}
                          prefix="$"
                          valueStyle={{ color: '#3f8600' }}
                        />
                      </Card>
                    </Col>
                    <Col xs={12} sm={8} md={6} lg={4}>
                      <Card size="small">
                        <Statistic
                          title="Avg Loss"
                          value={Math.abs(metrics.averageLoss)}
                          precision={2}
                          prefix="$"
                          valueStyle={{ color: '#cf1322' }}
                        />
                      </Card>
                    </Col>
                    <Col xs={12} sm={8} md={6} lg={4}>
                      <Card size="small">
                        <Statistic
                          title="Largest Win"
                          value={metrics.largestWin}
                          precision={2}
                          prefix="$"
                          valueStyle={{ color: '#3f8600' }}
                        />
                      </Card>
                    </Col>
                  </>
                )}
              </Row>
            </TabPane>
          </Tabs>
        </Col>
      </Row>

      {/* Configuration Modal */}
      <Modal
        title="Analytics Configuration"
        open={configModalVisible}
        onCancel={() => setConfigModalVisible(false)}
        footer={[
          <Button key="cancel" onClick={() => setConfigModalVisible(false)}>
            Cancel
          </Button>,
          <Button key="save" type="primary">
            Save Configuration
          </Button>
        ]}
      >
        <Form layout="vertical">
          <Form.Item label="Display Metrics">
            <Select
              mode="multiple"
              value={selectedMetrics}
              onChange={setSelectedMetrics}
              style={{ width: '100%' }}
            >
              <Select.Option value="sharpeRatio">Sharpe Ratio</Select.Option>
              <Select.Option value="sortinoRatio">Sortino Ratio</Select.Option>
              <Select.Option value="maxDrawdown">Max Drawdown</Select.Option>
              <Select.Option value="winRate">Win Rate</Select.Option>
              <Select.Option value="profitFactor">Profit Factor</Select.Option>
              <Select.Option value="alpha">Alpha</Select.Option>
              <Select.Option value="beta">Beta</Select.Option>
            </Select>
          </Form.Item>
          
          <Form.Item label="Chart Refresh Interval (seconds)">
            <Slider min={10} max={300} defaultValue={60} marks={{ 10: '10s', 60: '1m', 300: '5m' }} />
          </Form.Item>
          
          <Form.Item label="Risk Threshold Alert">
            <InputNumber
              min={0}
              max={100}
              defaultValue={5}
              formatter={value => `${value}%`}
              parser={value => value!.replace('%', '')}
              style={{ width: '100%' }}
            />
          </Form.Item>
        </Form>
      </Modal>
    </div>
  )
}

export default AdvancedAnalyticsDashboard