import React, { useState, useEffect } from 'react'
import {
  Card,
  Row,
  Col,
  Table,
  Select,
  DatePicker,
  Button,
  Space,
  Typography,
  Statistic,
  Progress,
  Tag,
  Tooltip,
  Modal,
  Tabs,
  Alert
} from 'antd'
import {
  BarChartOutlined,
  LineChartOutlined,
  PieChartOutlined,
  DownloadOutlined,
  CompareOutlined,
  TrophyOutlined,
  RiseOutlined,
  FallOutlined
} from '@ant-design/icons'
import dayjs from 'dayjs'
import { ResponsiveContainer, LineChart, Line, BarChart, Bar, PieChart, Pie, Cell, XAxis, YAxis, CartesianGrid, Tooltip as RechartsTooltip, Legend, ComposedChart, Area, AreaChart } from 'recharts'

const { Title, Text } = Typography
const { RangePicker } = DatePicker
const { TabPane } = Tabs

interface StrategyPerformance {
  strategyId: string
  name: string
  totalReturn: number
  sharpeRatio: number
  maxDrawdown: number
  winRate: number
  profitFactor: number
  volatility: number
  calmarRatio: number
  sortinoRatio: number
  alpha: number
  beta: number
  tradesCount: number
  avgWin: number
  avgLoss: number
  maxWin: number
  maxLoss: number
}

interface ComparisonData {
  date: string
  [key: string]: number | string
}

const StrategyPerformanceAnalysis: React.FC = () => {
  const [strategies, setStrategies] = useState<StrategyPerformance[]>([])
  const [selectedStrategies, setSelectedStrategies] = useState<string[]>([])
  const [comparisonData, setComparisonData] = useState<ComparisonData[]>([])
  const [dateRange, setDateRange] = useState<[dayjs.Dayjs, dayjs.Dayjs]>([
    dayjs().subtract(3, 'month'),
    dayjs()
  ])
  const [loading, setLoading] = useState(false)

  const apiUrl = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8002'

  useEffect(() => {
    loadStrategyPerformance()
  }, [dateRange])

  useEffect(() => {
    if (selectedStrategies.length > 0) {
      loadComparisonData()
    }
  }, [selectedStrategies, dateRange])

  const loadStrategyPerformance = async () => {
    setLoading(true)
    try {
      const [startDate, endDate] = dateRange
      const response = await fetch(
        `${apiUrl}/api/v1/strategy/performance?start=${startDate.format('YYYY-MM-DD')}&end=${endDate.format('YYYY-MM-DD')}`
      )
      if (response.ok) {
        const data = await response.json()
        setStrategies(data.strategies || [])
      }
    } catch (error) {
      console.error('Failed to load strategy performance:', error)
    } finally {
      setLoading(false)
    }
  }

  const loadComparisonData = async () => {
    try {
      const [startDate, endDate] = dateRange
      const response = await fetch(`${apiUrl}/api/v1/strategy/comparison`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          strategies: selectedStrategies,
          startDate: startDate.format('YYYY-MM-DD'),
          endDate: endDate.format('YYYY-MM-DD')
        })
      })
      if (response.ok) {
        const data = await response.json()
        setComparisonData(data.comparison || [])
      }
    } catch (error) {
      console.error('Failed to load comparison data:', error)
    }
  }

  const performanceColumns = [
    {
      title: 'Strategy',
      dataIndex: 'name',
      key: 'name',
      fixed: 'left' as const,
      width: 150,
      render: (text: string, record: StrategyPerformance) => (
        <Space direction="vertical" size={2}>
          <Text strong>{text}</Text>
          <Text type="secondary" style={{ fontSize: 12 }}>{record.strategyId}</Text>
        </Space>
      )
    },
    {
      title: 'Total Return',
      dataIndex: 'totalReturn',
      key: 'totalReturn',
      width: 120,
      render: (value: number) => (
        <Text type={value >= 0 ? 'success' : 'danger'}>
          {value >= 0 ? '+' : ''}{value.toFixed(2)}%
        </Text>
      ),
      sorter: (a: StrategyPerformance, b: StrategyPerformance) => a.totalReturn - b.totalReturn
    },
    {
      title: 'Sharpe Ratio',
      dataIndex: 'sharpeRatio',
      key: 'sharpeRatio',
      width: 110,
      render: (value: number) => (
        <Tooltip title={`${value > 1 ? 'Excellent' : value > 0.5 ? 'Good' : 'Poor'}`}>
          <Progress
            percent={Math.min(value * 50, 100)}
            size="small"
            status={value > 1 ? 'success' : value > 0.5 ? 'active' : 'exception'}
            format={() => value.toFixed(2)}
          />
        </Tooltip>
      ),
      sorter: (a: StrategyPerformance, b: StrategyPerformance) => a.sharpeRatio - b.sharpeRatio
    },
    {
      title: 'Max Drawdown',
      dataIndex: 'maxDrawdown',
      key: 'maxDrawdown',
      width: 120,
      render: (value: number) => (
        <Text type="danger">{value.toFixed(2)}%</Text>
      ),
      sorter: (a: StrategyPerformance, b: StrategyPerformance) => a.maxDrawdown - b.maxDrawdown
    },
    {
      title: 'Win Rate',
      dataIndex: 'winRate',
      key: 'winRate',
      width: 100,
      render: (value: number) => (
        <Progress
          percent={value}
          size="small"
          status={value > 60 ? 'success' : value > 40 ? 'active' : 'exception'}
          format={() => `${value.toFixed(0)}%`}
        />
      ),
      sorter: (a: StrategyPerformance, b: StrategyPerformance) => a.winRate - b.winRate
    },
    {
      title: 'Profit Factor',
      dataIndex: 'profitFactor',
      key: 'profitFactor',
      width: 110,
      render: (value: number) => (
        <Text type={value > 1 ? 'success' : 'danger'}>
          {value.toFixed(2)}
        </Text>
      ),
      sorter: (a: StrategyPerformance, b: StrategyPerformance) => a.profitFactor - b.profitFactor
    },
    {
      title: 'Volatility',
      dataIndex: 'volatility',
      key: 'volatility',
      width: 100,
      render: (value: number) => `${value.toFixed(1)}%`,
      sorter: (a: StrategyPerformance, b: StrategyPerformance) => a.volatility - b.volatility
    },
    {
      title: 'Alpha',
      dataIndex: 'alpha',
      key: 'alpha',
      width: 80,
      render: (value: number) => (
        <Text type={value > 0 ? 'success' : 'danger'}>
          {value.toFixed(3)}
        </Text>
      ),
      sorter: (a: StrategyPerformance, b: StrategyPerformance) => a.alpha - b.alpha
    },
    {
      title: 'Beta',
      dataIndex: 'beta',
      key: 'beta',
      width: 80,
      render: (value: number) => value.toFixed(2),
      sorter: (a: StrategyPerformance, b: StrategyPerformance) => a.beta - b.beta
    },
    {
      title: 'Trades',
      dataIndex: 'tradesCount',
      key: 'tradesCount',
      width: 80,
      render: (value: number) => value.toLocaleString(),
      sorter: (a: StrategyPerformance, b: StrategyPerformance) => a.tradesCount - b.tradesCount
    }
  ]

  const renderComparisonChart = () => (
    <ResponsiveContainer width="100%" height={400}>
      <LineChart data={comparisonData}>
        <CartesianGrid strokeDasharray="3 3" />
        <XAxis dataKey="date" tick={{ fontSize: 12 }} />
        <YAxis tick={{ fontSize: 12 }} />
        <RechartsTooltip 
          formatter={(value: any, name: string) => [`${value}%`, name]}
          labelFormatter={(label) => `Date: ${label}`}
        />
        <Legend />
        {selectedStrategies.map((strategyId, index) => {
          const colors = ['#1890ff', '#52c41a', '#fa541c', '#722ed1', '#eb2f96', '#faad14']
          return (
            <Line
              key={strategyId}
              type="monotone"
              dataKey={strategyId}
              stroke={colors[index % colors.length]}
              strokeWidth={2}
              dot={false}
              name={strategies.find(s => s.strategyId === strategyId)?.name || strategyId}
            />
          )
        })}
      </LineChart>
    </ResponsiveContainer>
  )

  const renderPerformanceRadar = () => {
    if (selectedStrategies.length === 0) return null

    const selectedData = strategies.filter(s => selectedStrategies.includes(s.strategyId))
    const pieData = selectedData.map((strategy, index) => ({
      name: strategy.name,
      value: strategy.totalReturn,
      color: ['#1890ff', '#52c41a', '#fa541c', '#722ed1', '#eb2f96'][index % 5]
    }))

    return (
      <ResponsiveContainer width="100%" height={300}>
        <PieChart>
          <Pie
            data={pieData}
            cx="50%"
            cy="50%"
            outerRadius={100}
            dataKey="value"
            nameKey="name"
            label={({ name, value }) => `${name}: ${value.toFixed(1)}%`}
          >
            {pieData.map((entry, index) => (
              <Cell key={`cell-${index}`} fill={entry.color} />
            ))}
          </Pie>
          <RechartsTooltip formatter={(value: any) => [`${value}%`, 'Return']} />
        </PieChart>
      </ResponsiveContainer>
    )
  }

  const topPerformers = strategies
    .sort((a, b) => b.totalReturn - a.totalReturn)
    .slice(0, 5)

  const worstPerformers = strategies
    .sort((a, b) => a.totalReturn - b.totalReturn)
    .slice(0, 5)

  return (
    <div style={{ padding: 24 }}>
      <Row gutter={[16, 16]}>
        {/* Header */}
        <Col xs={24}>
          <Card>
            <Row justify="space-between" align="middle">
              <Col>
                <Title level={3} style={{ margin: 0 }}>
                  <BarChartOutlined style={{ marginRight: 8 }} />
                  Strategy Performance Analysis
                </Title>
                <Text type="secondary">
                  Comprehensive analysis of {strategies.length} strategies
                </Text>
              </Col>
              <Col>
                <Space>
                  <RangePicker
                    value={dateRange}
                    onChange={(dates) => dates && setDateRange(dates as [dayjs.Dayjs, dayjs.Dayjs])}
                    presets={[
                      { label: 'Last 1M', value: [dayjs().subtract(1, 'month'), dayjs()] },
                      { label: 'Last 3M', value: [dayjs().subtract(3, 'month'), dayjs()] },
                      { label: 'Last 6M', value: [dayjs().subtract(6, 'month'), dayjs()] },
                      { label: 'YTD', value: [dayjs().startOf('year'), dayjs()] }
                    ]}
                  />
                  <Button icon={<DownloadOutlined />}>
                    Export
                  </Button>
                </Space>
              </Col>
            </Row>
          </Card>
        </Col>

        {/* Performance Summary */}
        <Col xs={24}>
          <Row gutter={[16, 16]}>
            <Col xs={12} sm={6} md={4}>
              <Card size="small">
                <Statistic
                  title="Avg Return"
                  value={strategies.reduce((sum, s) => sum + s.totalReturn, 0) / strategies.length || 0}
                  precision={2}
                  suffix="%"
                  valueStyle={{ 
                    color: (strategies.reduce((sum, s) => sum + s.totalReturn, 0) / strategies.length || 0) >= 0 ? '#3f8600' : '#cf1322' 
                  }}
                />
              </Card>
            </Col>
            <Col xs={12} sm={6} md={4}>
              <Card size="small">
                <Statistic
                  title="Avg Sharpe"
                  value={strategies.reduce((sum, s) => sum + s.sharpeRatio, 0) / strategies.length || 0}
                  precision={2}
                />
              </Card>
            </Col>
            <Col xs={12} sm={6} md={4}>
              <Card size="small">
                <Statistic
                  title="Best Return"
                  value={Math.max(...strategies.map(s => s.totalReturn))}
                  precision={2}
                  suffix="%"
                  prefix={<RiseOutlined />}
                  valueStyle={{ color: '#3f8600' }}
                />
              </Card>
            </Col>
            <Col xs={12} sm={6} md={4}>
              <Card size="small">
                <Statistic
                  title="Worst Return"
                  value={Math.min(...strategies.map(s => s.totalReturn))}
                  precision={2}
                  suffix="%"
                  prefix={<FallOutlined />}
                  valueStyle={{ color: '#cf1322' }}
                />
              </Card>
            </Col>
            <Col xs={12} sm={6} md={4}>
              <Card size="small">
                <Statistic
                  title="Profitable"
                  value={strategies.filter(s => s.totalReturn > 0).length}
                  suffix={`/ ${strategies.length}`}
                  valueStyle={{ color: '#3f8600' }}
                />
              </Card>
            </Col>
            <Col xs={12} sm={6} md={4}>
              <Card size="small">
                <Statistic
                  title="Total Trades"
                  value={strategies.reduce((sum, s) => sum + s.tradesCount, 0)}
                  formatter={value => value.toLocaleString()}
                />
              </Card>
            </Col>
          </Row>
        </Col>

        {/* Strategy Comparison */}
        <Col xs={24}>
          <Card 
            title="Strategy Comparison"
            extra={
              <Select
                mode="multiple"
                placeholder="Select strategies to compare"
                value={selectedStrategies}
                onChange={setSelectedStrategies}
                style={{ minWidth: 300 }}
                maxTagCount="responsive"
              >
                {strategies.map(strategy => (
                  <Select.Option key={strategy.strategyId} value={strategy.strategyId}>
                    {strategy.name}
                  </Select.Option>
                ))}
              </Select>
            }
          >
            <Tabs defaultActiveKey="chart">
              <TabPane tab="Performance Chart" key="chart">
                {selectedStrategies.length > 0 ? (
                  renderComparisonChart()
                ) : (
                  <div style={{ textAlign: 'center', padding: 50 }}>
                    <Text type="secondary">Select strategies to compare their performance</Text>
                  </div>
                )}
              </TabPane>
              <TabPane tab="Return Distribution" key="pie">
                {selectedStrategies.length > 0 ? (
                  renderPerformanceRadar()
                ) : (
                  <div style={{ textAlign: 'center', padding: 50 }}>
                    <Text type="secondary">Select strategies to view return distribution</Text>
                  </div>
                )}
              </TabPane>
            </Tabs>
          </Card>
        </Col>

        {/* Top and Worst Performers */}
        <Col xs={24} lg={12}>
          <Card title={<Space><TrophyOutlined />Top Performers</Space>}>
            <Space direction="vertical" style={{ width: '100%' }}>
              {topPerformers.map((strategy, index) => (
                <Card key={strategy.strategyId} size="small" style={{ backgroundColor: index === 0 ? '#f6ffed' : undefined }}>
                  <Row justify="space-between" align="middle">
                    <Col>
                      <Space>
                        <Tag color="gold">#{index + 1}</Tag>
                        <Text strong>{strategy.name}</Text>
                      </Space>
                    </Col>
                    <Col>
                      <Space>
                        <Text type="success">+{strategy.totalReturn.toFixed(2)}%</Text>
                        <Text type="secondary">Sharpe: {strategy.sharpeRatio.toFixed(2)}</Text>
                      </Space>
                    </Col>
                  </Row>
                </Card>
              ))}
            </Space>
          </Card>
        </Col>

        <Col xs={24} lg={12}>
          <Card title="Worst Performers">
            <Space direction="vertical" style={{ width: '100%' }}>
              {worstPerformers.map((strategy, index) => (
                <Card key={strategy.strategyId} size="small" style={{ backgroundColor: index === 0 ? '#fff2f0' : undefined }}>
                  <Row justify="space-between" align="middle">
                    <Col>
                      <Space>
                        <Tag color="red">#{index + 1}</Tag>
                        <Text strong>{strategy.name}</Text>
                      </Space>
                    </Col>
                    <Col>
                      <Space>
                        <Text type="danger">{strategy.totalReturn.toFixed(2)}%</Text>
                        <Text type="secondary">DD: {strategy.maxDrawdown.toFixed(2)}%</Text>
                      </Space>
                    </Col>
                  </Row>
                </Card>
              ))}
            </Space>
          </Card>
        </Col>

        {/* Detailed Performance Table */}
        <Col xs={24}>
          <Card title="Detailed Performance Metrics">
            <Table
              dataSource={strategies}
              columns={performanceColumns}
              rowKey="strategyId"
              pagination={{ pageSize: 20, showSizeChanger: true }}
              scroll={{ x: 1200 }}
              size="middle"
              loading={loading}
              rowSelection={{
                selectedRowKeys: selectedStrategies,
                onChange: setSelectedStrategies,
                getCheckboxProps: (record) => ({
                  name: record.name
                })
              }}
            />
          </Card>
        </Col>
      </Row>
    </div>
  )
}

export default StrategyPerformanceAnalysis