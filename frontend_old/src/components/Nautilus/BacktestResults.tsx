import React, { useState, useMemo } from 'react'
import {
  Card,
  Tabs,
  Row,
  Col,
  Statistic,
  Table,
  Tag,
  Space,
  Button,
  Typography,
  Alert,
  Tooltip,
  Progress,
  Select,
  Input,
  DatePicker,
  Drawer
} from 'antd'
import {
  DownloadOutlined,
  LineChartOutlined,
  TableOutlined,
  BarChartOutlined,
  ShareAltOutlined,
  FilterOutlined,
  InfoCircleOutlined,
  TrophyOutlined,
  WarningOutlined
} from '@ant-design/icons'
import dayjs from 'dayjs'
import { BacktestResult, TradeResult, PerformanceMetrics } from '../../services/backtestService'
import EquityCurveChart from './EquityCurveChart'

const { Title, Text } = Typography
const { TabPane } = Tabs
const { RangePicker } = DatePicker

interface BacktestResultsProps {
  backtest: BacktestResult
  onExport?: (format: 'pdf' | 'excel' | 'csv') => void
  onCompare?: (backtestIds: string[]) => void
  showComparison?: boolean
}

interface TradeFilters {
  instrument?: string
  side?: 'buy' | 'sell'
  profitability?: 'profitable' | 'losing'
  dateRange?: [dayjs.Dayjs, dayjs.Dayjs]
  minPnl?: number
  maxPnl?: number
}

const BacktestResults: React.FC<BacktestResultsProps> = ({
  backtest,
  onExport,
  onCompare,
  showComparison = false
}) => {
  const [activeTab, setActiveTab] = useState<'overview' | 'trades' | 'charts' | 'analysis'>('overview')
  const [tradeFilters, setTradeFilters] = useState<TradeFilters>({})
  const [filterDrawerVisible, setFilterDrawerVisible] = useState(false)
  const [selectedTrades, setSelectedTrades] = useState<string[]>([])

  // Process and filter trades
  const filteredTrades = useMemo(() => {
    if (!backtest.trades) return []
    
    return backtest.trades.filter(trade => {
      if (tradeFilters.instrument && trade.instrumentId !== tradeFilters.instrument) return false
      if (tradeFilters.side && trade.side !== tradeFilters.side) return false
      if (tradeFilters.profitability === 'profitable' && trade.pnl <= 0) return false
      if (tradeFilters.profitability === 'losing' && trade.pnl >= 0) return false
      if (tradeFilters.minPnl && trade.pnl < tradeFilters.minPnl) return false
      if (tradeFilters.maxPnl && trade.pnl > tradeFilters.maxPnl) return false
      
      if (tradeFilters.dateRange) {
        const tradeDate = dayjs(trade.entryTime)
        if (tradeDate.isBefore(tradeFilters.dateRange[0]) || tradeDate.isAfter(tradeFilters.dateRange[1])) {
          return false
        }
      }
      
      return true
    })
  }, [backtest.trades, tradeFilters])

  // Calculate filtered metrics
  const filteredMetrics = useMemo(() => {
    if (filteredTrades.length === 0) return null
    
    const totalPnl = filteredTrades.reduce((sum, trade) => sum + trade.pnl, 0)
    const totalCommission = filteredTrades.reduce((sum, trade) => sum + trade.commission, 0)
    const netPnl = totalPnl - totalCommission
    
    const winningTrades = filteredTrades.filter(t => t.pnl > 0)
    const losingTrades = filteredTrades.filter(t => t.pnl < 0)
    
    const avgWin = winningTrades.length > 0 ? 
      winningTrades.reduce((sum, t) => sum + t.pnl, 0) / winningTrades.length : 0
    const avgLoss = losingTrades.length > 0 ? 
      losingTrades.reduce((sum, t) => sum + t.pnl, 0) / losingTrades.length : 0
    
    const profitFactor = Math.abs(avgLoss) > 0 ? Math.abs(avgWin / avgLoss) : 0
    
    return {
      totalTrades: filteredTrades.length,
      totalPnl,
      netPnl,
      totalCommission,
      winningTrades: winningTrades.length,
      losingTrades: losingTrades.length,
      winRate: (winningTrades.length / filteredTrades.length) * 100,
      avgWin,
      avgLoss,
      profitFactor,
      largestWin: Math.max(...filteredTrades.map(t => t.pnl)),
      largestLoss: Math.min(...filteredTrades.map(t => t.pnl))
    }
  }, [filteredTrades])

  // Get available instruments for filtering
  const availableInstruments = useMemo(() => {
    if (!backtest.trades) return []
    const instruments = new Set(backtest.trades.map(t => t.instrumentId))
    return Array.from(instruments).sort()
  }, [backtest.trades])

  const handleExport = (format: 'pdf' | 'excel' | 'csv') => {
    if (onExport) {
      onExport(format)
    }
  }

  const renderMetricsCards = (metrics: PerformanceMetrics) => (
    <Row gutter={[16, 16]}>
      <Col xs={12} sm={8} md={6} lg={4}>
        <Card size="small">
          <Statistic 
            title="Total Return" 
            value={metrics.totalReturn} 
            suffix="%" 
            precision={2}
            valueStyle={{ color: metrics.totalReturn >= 0 ? '#3f8600' : '#cf1322' }}
          />
        </Card>
      </Col>
      <Col xs={12} sm={8} md={6} lg={4}>
        <Card size="small">
          <Statistic 
            title="Annualized Return" 
            value={metrics.annualizedReturn} 
            suffix="%" 
            precision={2}
            valueStyle={{ color: metrics.annualizedReturn >= 0 ? '#3f8600' : '#cf1322' }}
          />
        </Card>
      </Col>
      <Col xs={12} sm={8} md={6} lg={4}>
        <Card size="small">
          <Statistic 
            title="Sharpe Ratio" 
            value={metrics.sharpeRatio} 
            precision={2}
            valueStyle={{ color: metrics.sharpeRatio >= 1 ? '#3f8600' : '#cf1322' }}
          />
        </Card>
      </Col>
      <Col xs={12} sm={8} md={6} lg={4}>
        <Card size="small">
          <Statistic 
            title="Sortino Ratio" 
            value={metrics.sortinoRatio} 
            precision={2}
            valueStyle={{ color: metrics.sortinoRatio >= 1 ? '#3f8600' : '#cf1322' }}
          />
        </Card>
      </Col>
      <Col xs={12} sm={8} md={6} lg={4}>
        <Card size="small">
          <Statistic 
            title="Max Drawdown" 
            value={metrics.maxDrawdown} 
            suffix="%" 
            precision={2}
            valueStyle={{ color: '#cf1322' }}
          />
        </Card>
      </Col>
      <Col xs={12} sm={8} md={6} lg={4}>
        <Card size="small">
          <Statistic 
            title="Win Rate" 
            value={metrics.winRate} 
            suffix="%" 
            precision={1}
            valueStyle={{ color: metrics.winRate >= 50 ? '#3f8600' : '#cf1322' }}
          />
        </Card>
      </Col>
      <Col xs={12} sm={8} md={6} lg={4}>
        <Card size="small">
          <Statistic 
            title="Profit Factor" 
            value={metrics.profitFactor} 
            precision={2}
            valueStyle={{ color: metrics.profitFactor >= 1 ? '#3f8600' : '#cf1322' }}
          />
        </Card>
      </Col>
      <Col xs={12} sm={8} md={6} lg={4}>
        <Card size="small">
          <Statistic 
            title="Total Trades" 
            value={metrics.totalTrades}
          />
        </Card>
      </Col>
      <Col xs={12} sm={8} md={6} lg={4}>
        <Card size="small">
          <Statistic 
            title="Volatility" 
            value={metrics.volatility} 
            suffix="%" 
            precision={2}
          />
        </Card>
      </Col>
      <Col xs={12} sm={8} md={6} lg={4}>
        <Card size="small">
          <Statistic 
            title="Calmar Ratio" 
            value={metrics.calmarRatio || 0} 
            precision={2}
            valueStyle={{ color: (metrics.calmarRatio || 0) >= 1 ? '#3f8600' : '#cf1322' }}
          />
        </Card>
      </Col>
      <Col xs={12} sm={8} md={6} lg={4}>
        <Card size="small">
          <Statistic 
            title="Alpha" 
            value={metrics.alpha || 0} 
            suffix="%" 
            precision={2}
            valueStyle={{ color: (metrics.alpha || 0) >= 0 ? '#3f8600' : '#cf1322' }}
          />
        </Card>
      </Col>
      <Col xs={12} sm={8} md={6} lg={4}>
        <Card size="small">
          <Statistic 
            title="Beta" 
            value={metrics.beta || 0} 
            precision={2}
          />
        </Card>
      </Col>
    </Row>
  )

  const tradeColumns = [
    {
      title: 'Trade ID',
      dataIndex: 'tradeId',
      key: 'tradeId',
      width: 120,
      render: (text: string) => (
        <Text code style={{ fontSize: '11px' }}>
          {text.substring(0, 8)}...
        </Text>
      )
    },
    {
      title: 'Instrument',
      dataIndex: 'instrumentId',
      key: 'instrumentId',
      width: 100,
      sorter: (a: TradeResult, b: TradeResult) => a.instrumentId.localeCompare(b.instrumentId)
    },
    {
      title: 'Side',
      dataIndex: 'side',
      key: 'side',
      width: 60,
      render: (side: string) => (
        <Tag color={side === 'buy' ? 'green' : 'red'}>{side.toUpperCase()}</Tag>
      ),
      filters: [
        { text: 'Buy', value: 'buy' },
        { text: 'Sell', value: 'sell' }
      ],
      onFilter: (value: any, record: TradeResult) => record.side === value
    },
    {
      title: 'Quantity',
      dataIndex: 'quantity',
      key: 'quantity',
      width: 80,
      render: (qty: number) => qty.toLocaleString(),
      sorter: (a: TradeResult, b: TradeResult) => a.quantity - b.quantity
    },
    {
      title: 'Entry Price',
      dataIndex: 'entryPrice',
      key: 'entryPrice',
      width: 100,
      render: (price: number) => `$${price.toFixed(4)}`,
      sorter: (a: TradeResult, b: TradeResult) => a.entryPrice - b.entryPrice
    },
    {
      title: 'Exit Price',
      dataIndex: 'exitPrice',
      key: 'exitPrice',
      width: 100,
      render: (price: number) => `$${price.toFixed(4)}`,
      sorter: (a: TradeResult, b: TradeResult) => a.exitPrice - b.exitPrice
    },
    {
      title: 'P&L',
      dataIndex: 'pnl',
      key: 'pnl',
      width: 100,
      render: (pnl: number) => (
        <Text type={pnl >= 0 ? 'success' : 'danger'} strong>
          ${pnl.toFixed(2)}
        </Text>
      ),
      sorter: (a: TradeResult, b: TradeResult) => a.pnl - b.pnl
    },
    {
      title: 'Commission',
      dataIndex: 'commission',
      key: 'commission',
      width: 80,
      render: (commission: number) => `$${commission.toFixed(2)}`,
      sorter: (a: TradeResult, b: TradeResult) => a.commission - b.commission
    },
    {
      title: 'Duration',
      key: 'duration',
      width: 80,
      render: (_: any, record: TradeResult) => {
        const duration = dayjs(record.exitTime).diff(dayjs(record.entryTime), 'minute')
        if (duration < 60) return `${duration}m`
        if (duration < 1440) return `${Math.floor(duration / 60)}h`
        return `${Math.floor(duration / 1440)}d`
      },
      sorter: (a: TradeResult, b: TradeResult) => {
        const aDuration = dayjs(a.exitTime).diff(dayjs(a.entryTime), 'minute')
        const bDuration = dayjs(b.exitTime).diff(dayjs(b.entryTime), 'minute')
        return aDuration - bDuration
      }
    },
    {
      title: 'Entry Time',
      dataIndex: 'entryTime',
      key: 'entryTime',
      width: 130,
      render: (time: string) => dayjs(time).format('MM/DD HH:mm'),
      sorter: (a: TradeResult, b: TradeResult) => 
        dayjs(a.entryTime).valueOf() - dayjs(b.entryTime).valueOf()
    }
  ]

  const renderFilterDrawer = () => (
    <Drawer
      title="Trade Filters"
      placement="right"
      onClose={() => setFilterDrawerVisible(false)}
      open={filterDrawerVisible}
      width={300}
    >
      <Space direction="vertical" style={{ width: '100%' }}>
        <div>
          <Text strong>Instrument</Text>
          <Select
            style={{ width: '100%', marginTop: 4 }}
            placeholder="All instruments"
            allowClear
            value={tradeFilters.instrument}
            onChange={(value) => setTradeFilters({ ...tradeFilters, instrument: value })}
          >
            {availableInstruments.map(instrument => (
              <Select.Option key={instrument} value={instrument}>
                {instrument}
              </Select.Option>
            ))}
          </Select>
        </div>

        <div>
          <Text strong>Side</Text>
          <Select
            style={{ width: '100%', marginTop: 4 }}
            placeholder="All sides"
            allowClear
            value={tradeFilters.side}
            onChange={(value) => setTradeFilters({ ...tradeFilters, side: value })}
          >
            <Select.Option value="buy">Buy</Select.Option>
            <Select.Option value="sell">Sell</Select.Option>
          </Select>
        </div>

        <div>
          <Text strong>Profitability</Text>
          <Select
            style={{ width: '100%', marginTop: 4 }}
            placeholder="All trades"
            allowClear
            value={tradeFilters.profitability}
            onChange={(value) => setTradeFilters({ ...tradeFilters, profitability: value })}
          >
            <Select.Option value="profitable">Profitable</Select.Option>
            <Select.Option value="losing">Losing</Select.Option>
          </Select>
        </div>

        <div>
          <Text strong>Date Range</Text>
          <RangePicker
            style={{ width: '100%', marginTop: 4 }}
            value={tradeFilters.dateRange}
            onChange={(dates) => setTradeFilters({ 
              ...tradeFilters, 
              dateRange: dates as [dayjs.Dayjs, dayjs.Dayjs] 
            })}
          />
        </div>

        <div>
          <Text strong>P&L Range</Text>
          <Input.Group compact style={{ marginTop: 4 }}>
            <Input
              style={{ width: '50%' }}
              placeholder="Min P&L"
              value={tradeFilters.minPnl}
              onChange={(e) => setTradeFilters({ 
                ...tradeFilters, 
                minPnl: e.target.value ? parseFloat(e.target.value) : undefined 
              })}
            />
            <Input
              style={{ width: '50%' }}
              placeholder="Max P&L"
              value={tradeFilters.maxPnl}
              onChange={(e) => setTradeFilters({ 
                ...tradeFilters, 
                maxPnl: e.target.value ? parseFloat(e.target.value) : undefined 
              })}
            />
          </Input.Group>
        </div>

        <Button 
          block 
          onClick={() => setTradeFilters({})}
        >
          Clear Filters
        </Button>
      </Space>
    </Drawer>
  )

  if (!backtest || backtest.status !== 'completed') {
    return (
      <Alert
        type="info"
        message="Backtest results not available"
        description="Backtest is still running or has not completed successfully."
        showIcon
      />
    )
  }

  return (
    <div>
      {/* Header */}
      <Card style={{ marginBottom: 16 }}>
        <Row justify="space-between" align="middle">
          <Col>
            <Title level={3} style={{ margin: 0 }}>
              <TrophyOutlined style={{ marginRight: 8 }} />
              Backtest Results
            </Title>
            <Space>
              <Text type="secondary">ID: {backtest.backtestId}</Text>
              <Text type="secondary">•</Text>
              <Text type="secondary">
                {dayjs(backtest.startTime).format('MMM DD, YYYY')} - {dayjs(backtest.endTime).format('MMM DD, YYYY')}
              </Text>
            </Space>
          </Col>
          <Col>
            <Space>
              <Button 
                icon={<DownloadOutlined />}
                onClick={() => handleExport('pdf')}
              >
                Export PDF
              </Button>
              <Button 
                icon={<DownloadOutlined />}
                onClick={() => handleExport('excel')}
              >
                Export Excel
              </Button>
              {showComparison && (
                <Button 
                  icon={<ShareAltOutlined />}
                  onClick={() => onCompare && onCompare([backtest.backtestId])}
                >
                  Compare
                </Button>
              )}
            </Space>
          </Col>
        </Row>
      </Card>

      {/* Main Content */}
      <Tabs activeKey={activeTab} onChange={setActiveTab}>
        <TabPane tab="Overview" key="overview" icon={<BarChartOutlined />}>
          {backtest.metrics && renderMetricsCards(backtest.metrics)}
        </TabPane>

        <TabPane tab="Trades" key="trades" icon={<TableOutlined />}>
          <Card 
            title={
              <Space>
                Trade Analysis
                {filteredMetrics && (
                  <Tag color="blue">
                    {filteredMetrics.totalTrades} trades filtered
                  </Tag>
                )}
              </Space>
            }
            extra={
              <Space>
                <Button
                  icon={<FilterOutlined />}
                  onClick={() => setFilterDrawerVisible(true)}
                  type={Object.keys(tradeFilters).length > 0 ? 'primary' : 'default'}
                >
                  Filters
                </Button>
              </Space>
            }
          >
            {filteredMetrics && (
              <Row gutter={[16, 16]} style={{ marginBottom: 16 }}>
                <Col xs={8} sm={6} md={4}>
                  <Statistic 
                    title="Win Rate" 
                    value={filteredMetrics.winRate} 
                    suffix="%" 
                    precision={1}
                    valueStyle={{ 
                      color: filteredMetrics.winRate >= 50 ? '#3f8600' : '#cf1322',
                      fontSize: '14px' 
                    }}
                  />
                </Col>
                <Col xs={8} sm={6} md={4}>
                  <Statistic 
                    title="Net P&L" 
                    value={filteredMetrics.netPnl} 
                    precision={2}
                    prefix="$"
                    valueStyle={{ 
                      color: filteredMetrics.netPnl >= 0 ? '#3f8600' : '#cf1322',
                      fontSize: '14px' 
                    }}
                  />
                </Col>
                <Col xs={8} sm={6} md={4}>
                  <Statistic 
                    title="Profit Factor" 
                    value={filteredMetrics.profitFactor} 
                    precision={2}
                    valueStyle={{ 
                      color: filteredMetrics.profitFactor >= 1 ? '#3f8600' : '#cf1322',
                      fontSize: '14px' 
                    }}
                  />
                </Col>
                <Col xs={8} sm={6} md={4}>
                  <Statistic 
                    title="Avg Win" 
                    value={filteredMetrics.avgWin} 
                    precision={2}
                    prefix="$"
                    valueStyle={{ fontSize: '14px' }}
                  />
                </Col>
                <Col xs={8} sm={6} md={4}>
                  <Statistic 
                    title="Avg Loss" 
                    value={Math.abs(filteredMetrics.avgLoss)} 
                    precision={2}
                    prefix="-$"
                    valueStyle={{ fontSize: '14px' }}
                  />
                </Col>
                <Col xs={8} sm={6} md={4}>
                  <Statistic 
                    title="Commission" 
                    value={filteredMetrics.totalCommission} 
                    precision={2}
                    prefix="$"
                    valueStyle={{ fontSize: '14px' }}
                  />
                </Col>
              </Row>
            )}
            
            <Table
              dataSource={filteredTrades}
              columns={tradeColumns}
              rowKey="tradeId"
              pagination={{ 
                pageSize: 50,
                showSizeChanger: true,
                showTotal: (total, range) => `${range[0]}-${range[1]} of ${total} trades`
              }}
              size="small"
              scroll={{ x: 1000 }}
              rowSelection={{
                selectedRowKeys: selectedTrades,
                onChange: setSelectedTrades,
                type: 'checkbox'
              }}
            />
          </Card>
        </TabPane>

        <TabPane tab="Charts" key="charts" icon={<LineChartOutlined />}>
          {backtest.equityCurve && backtest.metrics && (
            <EquityCurveChart 
              data={backtest.equityCurve}
              metrics={backtest.metrics}
              height={500}
              showDrawdown={true}
            />
          )}
        </TabPane>

        <TabPane tab="Analysis" key="analysis" icon={<InfoCircleOutlined />}>
          <Row gutter={[16, 16]}>
            <Col xs={24} lg={12}>
              <Card title="Risk Analysis" size="small">
                {backtest.metrics && (
                  <Space direction="vertical" style={{ width: '100%' }}>
                    <div>
                      <Text strong>Risk-Adjusted Returns</Text>
                      <Row gutter={16} style={{ marginTop: 8 }}>
                        <Col span={12}>
                          <Statistic 
                            title="Sharpe Ratio" 
                            value={backtest.metrics.sharpeRatio}
                            precision={2}
                          />
                        </Col>
                        <Col span={12}>
                          <Statistic 
                            title="Sortino Ratio" 
                            value={backtest.metrics.sortinoRatio}
                            precision={2}
                          />
                        </Col>
                      </Row>
                    </div>
                    
                    <div>
                      <Text strong>Drawdown Analysis</Text>
                      <Row gutter={16} style={{ marginTop: 8 }}>
                        <Col span={12}>
                          <Statistic 
                            title="Max Drawdown" 
                            value={backtest.metrics.maxDrawdown}
                            suffix="%"
                            precision={2}
                          />
                        </Col>
                        <Col span={12}>
                          <Statistic 
                            title="Calmar Ratio" 
                            value={backtest.metrics.calmarRatio || 0}
                            precision={2}
                          />
                        </Col>
                      </Row>
                    </div>
                  </Space>
                )}
              </Card>
            </Col>
            
            <Col xs={24} lg={12}>
              <Card title="Trade Statistics" size="small">
                {backtest.metrics && (
                  <Space direction="vertical" style={{ width: '100%' }}>
                    <div>
                      <Text strong>Win/Loss Distribution</Text>
                      <Row gutter={16} style={{ marginTop: 8 }}>
                        <Col span={12}>
                          <Statistic 
                            title="Winning Trades" 
                            value={backtest.metrics.winningTrades}
                          />
                        </Col>
                        <Col span={12}>
                          <Statistic 
                            title="Losing Trades" 
                            value={backtest.metrics.losingTrades}
                          />
                        </Col>
                      </Row>
                    </div>
                    
                    <div>
                      <Text strong>P&L Extremes</Text>
                      <Row gutter={16} style={{ marginTop: 8 }}>
                        <Col span={12}>
                          <Statistic 
                            title="Largest Win" 
                            value={backtest.metrics.largestWin}
                            prefix="$"
                            precision={2}
                          />
                        </Col>
                        <Col span={12}>
                          <Statistic 
                            title="Largest Loss" 
                            value={Math.abs(backtest.metrics.largestLoss)}
                            prefix="-$"
                            precision={2}
                          />
                        </Col>
                      </Row>
                    </div>
                  </Space>
                )}
              </Card>
            </Col>
          </Row>
        </TabPane>
      </Tabs>

      {renderFilterDrawer()}
    </div>
  )
}

export default BacktestResults