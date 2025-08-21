import React, { useState, useEffect } from 'react'
import {
  Card,
  Row,
  Col,
  Table,
  Tag,
  Space,
  Button,
  Progress,
  Statistic,
  Alert,
  Typography,
  Switch,
  Select,
  Modal,
  message,
  Tooltip,
  Badge,
  List,
  Timeline,
  Drawer,
  Descriptions,
  Tabs,
  Input
} from 'antd'
import {
  PlayCircleOutlined,
  PauseOutlined,
  StopOutlined,
  ReloadOutlined,
  EyeOutlined,
  SettingOutlined,
  WarningOutlined,
  CheckCircleOutlined,
  ExclamationCircleOutlined,
  DollarOutlined,
  RiseOutlined,
  FallOutlined,
  ClockCircleOutlined,
  ThunderboltOutlined
} from '@ant-design/icons'
import dayjs from 'dayjs'
import { ResponsiveContainer, LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip as RechartsTooltip, AreaChart, Area } from 'recharts'

const { Title, Text } = Typography
const { TabPane } = Tabs
const { Search } = Input

interface LiveStrategy {
  strategyId: string
  name: string
  version: string
  status: 'running' | 'paused' | 'stopped' | 'error' | 'starting'
  health: 'healthy' | 'warning' | 'critical'
  uptime: number
  lastHeartbeat: string
  performance: {
    totalPnL: number
    todayPnL: number
    totalReturn: number
    sharpeRatio: number
    maxDrawdown: number
    winRate: number
    tradesCount: number
  }
  positions: {
    openPositions: number
    totalExposure: number
    maxPosition: number
    averageHoldTime: number
  }
  risk: {
    currentRisk: number
    var95: number
    correlation: number
    leverage: number
  }
  execution: {
    ordersPerMinute: number
    fillRate: number
    averageSlippage: number
    latency: number
  }
  alerts: Array<{
    id: string
    type: 'info' | 'warning' | 'error'
    message: string
    timestamp: string
    acknowledged: boolean
  }>
  config: {
    maxPositionSize: number
    riskLimit: number
    tradingEnabled: boolean
  }
}

interface StrategyMetrics {
  timestamp: string
  pnl: number
  drawdown: number
  positions: number
  volume: number
  exposure: number
}

const LiveStrategyMonitoring: React.FC = () => {
  const [strategies, setStrategies] = useState<LiveStrategy[]>([])
  const [selectedStrategy, setSelectedStrategy] = useState<LiveStrategy | null>(null)
  const [metricsData, setMetricsData] = useState<StrategyMetrics[]>([])
  const [detailsDrawerVisible, setDetailsDrawerVisible] = useState(false)
  const [alertModalVisible, setAlertModalVisible] = useState(false)
  const [loading, setLoading] = useState(false)
  const [autoRefresh, setAutoRefresh] = useState(true)
  const [filterStatus, setFilterStatus] = useState<string[]>([])
  const [searchTerm, setSearchTerm] = useState('')

  const apiUrl = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8002'

  useEffect(() => {
    loadLiveStrategies()

    if (autoRefresh) {
      const interval = setInterval(() => {
        loadLiveStrategies()
        if (selectedStrategy) {
          loadStrategyMetrics(selectedStrategy.strategyId)
        }
      }, 5000) // Refresh every 5 seconds

      return () => clearInterval(interval)
    }
  }, [autoRefresh, selectedStrategy])

  const loadLiveStrategies = async () => {
    try {
      const response = await fetch(`${apiUrl}/api/v1/strategy/live`)
      if (response.ok) {
        const data = await response.json()
        setStrategies(data.strategies || [])
      }
    } catch (error) {
      console.error('Failed to load live strategies:', error)
    }
  }

  const loadStrategyMetrics = async (strategyId: string) => {
    try {
      const response = await fetch(`${apiUrl}/api/v1/strategy/metrics/${strategyId}?timeframe=1h`)
      if (response.ok) {
        const data = await response.json()
        setMetricsData(data.metrics || [])
      }
    } catch (error) {
      console.error('Failed to load strategy metrics:', error)
    }
  }

  const controlStrategy = async (strategyId: string, action: 'start' | 'pause' | 'stop') => {
    setLoading(true)
    try {
      const response = await fetch(`${apiUrl}/api/v1/strategy/${action}/${strategyId}`, {
        method: 'POST'
      })

      if (response.ok) {
        message.success(`Strategy ${action}ed successfully`)
        loadLiveStrategies()
      } else {
        const error = await response.json()
        message.error(`Failed to ${action} strategy: ${error.detail}`)
      }
    } catch (error) {
      message.error(`Failed to ${action} strategy`)
      console.error(`${action} strategy error:`, error)
    } finally {
      setLoading(false)
    }
  }

  const acknowledgeAlert = async (strategyId: string, alertId: string) => {
    try {
      const response = await fetch(`${apiUrl}/api/v1/strategy/alerts/acknowledge`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ strategyId, alertId })
      })

      if (response.ok) {
        message.success('Alert acknowledged')
        loadLiveStrategies()
      } else {
        message.error('Failed to acknowledge alert')
      }
    } catch (error) {
      message.error('Failed to acknowledge alert')
    }
  }

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'running': return 'success'
      case 'paused': return 'warning'
      case 'stopped': return 'default'
      case 'error': return 'error'
      case 'starting': return 'processing'
      default: return 'default'
    }
  }

  const getHealthColor = (health: string) => {
    switch (health) {
      case 'healthy': return 'success'
      case 'warning': return 'warning'
      case 'critical': return 'error'
      default: return 'default'
    }
  }

  const filteredStrategies = strategies.filter(strategy => {
    const matchesSearch = !searchTerm || 
      strategy.name.toLowerCase().includes(searchTerm.toLowerCase()) ||
      strategy.strategyId.toLowerCase().includes(searchTerm.toLowerCase())

    const matchesStatus = filterStatus.length === 0 || filterStatus.includes(strategy.status)

    return matchesSearch && matchesStatus
  })

  const strategyColumns = [
    {
      title: 'Strategy',
      key: 'strategy',
      width: 200,
      render: (_, record: LiveStrategy) => (
        <Space direction="vertical" size={2}>
          <Text strong>{record.name}</Text>
          <Text type="secondary" style={{ fontSize: 12 }}>{record.strategyId}</Text>
          <Tag size="small">{record.version}</Tag>
        </Space>
      )
    },
    {
      title: 'Status',
      key: 'status',
      width: 120,
      render: (_, record: LiveStrategy) => (
        <Space direction="vertical" size={2}>
          <Badge status={getStatusColor(record.status)} text={record.status} />
          <Badge status={getHealthColor(record.health)} text={record.health} />
        </Space>
      )
    },
    {
      title: 'Performance',
      key: 'performance',
      width: 150,
      render: (_, record: LiveStrategy) => (
        <Space direction="vertical" size={2}>
          <Text type={record.performance.totalPnL >= 0 ? 'success' : 'danger'}>
            ${record.performance.totalPnL.toFixed(2)}
          </Text>
          <Text type="secondary" style={{ fontSize: 11 }}>
            Today: ${record.performance.todayPnL.toFixed(2)}
          </Text>
          <Text type="secondary" style={{ fontSize: 11 }}>
            Return: {record.performance.totalReturn.toFixed(1)}%
          </Text>
        </Space>
      )
    },
    {
      title: 'Positions',
      key: 'positions',
      width: 120,
      render: (_, record: LiveStrategy) => (
        <Space direction="vertical" size={2}>
          <Text>{record.positions.openPositions} open</Text>
          <Text type="secondary" style={{ fontSize: 11 }}>
            Exposure: ${record.positions.totalExposure.toLocaleString()}
          </Text>
        </Space>
      )
    },
    {
      title: 'Risk',
      key: 'risk',
      width: 100,
      render: (_, record: LiveStrategy) => (
        <Progress
          type="circle"
          size={50}
          percent={Math.min(record.risk.currentRisk * 100, 100)}
          status={record.risk.currentRisk > 0.8 ? 'exception' : 'normal'}
          format={() => `${(record.risk.currentRisk * 100).toFixed(0)}%`}
        />
      )
    },
    {
      title: 'Alerts',
      key: 'alerts',
      width: 80,
      render: (_, record: LiveStrategy) => {
        const unacknowledged = record.alerts.filter(a => !a.acknowledged).length
        return unacknowledged > 0 ? (
          <Badge count={unacknowledged} style={{ backgroundColor: '#ff4d4f' }}>
            <Button 
              size="small" 
              icon={<WarningOutlined />}
              onClick={() => {
                setSelectedStrategy(record)
                setAlertModalVisible(true)
              }}
            />
          </Badge>
        ) : (
          <CheckCircleOutlined style={{ color: '#52c41a' }} />
        )
      }
    },
    {
      title: 'Uptime',
      key: 'uptime',
      width: 100,
      render: (_, record: LiveStrategy) => (
        <Space direction="vertical" size={2}>
          <Text style={{ fontSize: 11 }}>
            {Math.floor(record.uptime / 3600)}h {Math.floor((record.uptime % 3600) / 60)}m
          </Text>
          <Text type="secondary" style={{ fontSize: 10 }}>
            {dayjs(record.lastHeartbeat).fromNow()}
          </Text>
        </Space>
      )
    },
    {
      title: 'Actions',
      key: 'actions',
      width: 150,
      render: (_, record: LiveStrategy) => (
        <Space>
          {record.status === 'running' && (
            <Tooltip title="Pause Strategy">
              <Button 
                size="small"
                icon={<PauseOutlined />}
                onClick={() => controlStrategy(record.strategyId, 'pause')}
                loading={loading}
              />
            </Tooltip>
          )}
          {record.status === 'paused' && (
            <Tooltip title="Resume Strategy">
              <Button 
                size="small"
                type="primary"
                icon={<PlayCircleOutlined />}
                onClick={() => controlStrategy(record.strategyId, 'start')}
                loading={loading}
              />
            </Tooltip>
          )}
          {record.status !== 'stopped' && (
            <Tooltip title="Stop Strategy">
              <Button 
                size="small"
                danger
                icon={<StopOutlined />}
                onClick={() => {
                  Modal.confirm({
                    title: 'Stop Strategy',
                    content: 'Are you sure you want to stop this strategy? All positions will be closed.',
                    onOk: () => controlStrategy(record.strategyId, 'stop')
                  })
                }}
                loading={loading}
              />
            </Tooltip>
          )}
          <Tooltip title="View Details">
            <Button 
              size="small"
              icon={<EyeOutlined />}
              onClick={() => {
                setSelectedStrategy(record)
                loadStrategyMetrics(record.strategyId)
                setDetailsDrawerVisible(true)
              }}
            />
          </Tooltip>
        </Space>
      )
    }
  ]

  const renderMetricsChart = () => (
    <ResponsiveContainer width="100%" height={200}>
      <AreaChart data={metricsData}>
        <CartesianGrid strokeDasharray="3 3" />
        <XAxis dataKey="timestamp" tick={{ fontSize: 12 }} />
        <YAxis tick={{ fontSize: 12 }} />
        <RechartsTooltip />
        <Area 
          type="monotone" 
          dataKey="pnl" 
          stroke="#1890ff" 
          fill="#1890ff" 
          fillOpacity={0.1}
          name="P&L" 
        />
      </AreaChart>
    </ResponsiveContainer>
  )

  const alertItems = selectedStrategy?.alerts.map(alert => ({
    key: alert.id,
    label: (
      <Space>
        <Badge status={alert.type === 'error' ? 'error' : alert.type === 'warning' ? 'warning' : 'processing'} />
        <Text strong>{alert.message}</Text>
        <Text type="secondary">{dayjs(alert.timestamp).format('HH:mm:ss')}</Text>
        {!alert.acknowledged && (
          <Button 
            size="small" 
            onClick={() => acknowledgeAlert(selectedStrategy.strategyId, alert.id)}
          >
            Acknowledge
          </Button>
        )}
      </Space>
    )
  })) || []

  return (
    <div style={{ padding: 24 }}>
      <Row gutter={[16, 16]}>
        {/* Header */}
        <Col xs={24}>
          <Card>
            <Row justify="space-between" align="middle">
              <Col>
                <Title level={3} style={{ margin: 0 }}>
                  <ThunderboltOutlined style={{ marginRight: 8 }} />
                  Live Strategy Monitoring
                </Title>
                <Text type="secondary">
                  Real-time monitoring of {strategies.length} active strategies
                </Text>
              </Col>
              <Col>
                <Space>
                  <Switch
                    checked={autoRefresh}
                    onChange={setAutoRefresh}
                    checkedChildren="Auto"
                    unCheckedChildren="Manual"
                  />
                  <Button 
                    icon={<ReloadOutlined />}
                    onClick={loadLiveStrategies}
                    loading={loading}
                  >
                    Refresh
                  </Button>
                </Space>
              </Col>
            </Row>
          </Card>
        </Col>

        {/* Filters */}
        <Col xs={24}>
          <Card size="small">
            <Row gutter={16} align="middle">
              <Col xs={24} sm={8} md={6}>
                <Search
                  placeholder="Search strategies..."
                  value={searchTerm}
                  onChange={(e) => setSearchTerm(e.target.value)}
                  allowClear
                />
              </Col>
              <Col xs={24} sm={8} md={6}>
                <Select
                  mode="multiple"
                  placeholder="Filter by status"
                  value={filterStatus}
                  onChange={setFilterStatus}
                  style={{ width: '100%' }}
                >
                  <Select.Option value="running">Running</Select.Option>
                  <Select.Option value="paused">Paused</Select.Option>
                  <Select.Option value="stopped">Stopped</Select.Option>
                  <Select.Option value="error">Error</Select.Option>
                </Select>
              </Col>
              <Col>
                <Space>
                  <Text type="secondary">
                    {filteredStrategies.length} of {strategies.length} strategies
                  </Text>
                  <Badge 
                    count={strategies.filter(s => s.status === 'running').length} 
                    style={{ backgroundColor: '#52c41a' }}
                  />
                </Space>
              </Col>
            </Row>
          </Card>
        </Col>

        {/* Summary Cards */}
        <Col xs={24}>
          <Row gutter={[16, 16]}>
            <Col xs={12} sm={6} md={3}>
              <Card size="small">
                <Statistic
                  title="Total P&L"
                  value={strategies.reduce((sum, s) => sum + s.performance.totalPnL, 0)}
                  precision={2}
                  prefix="$"
                  valueStyle={{ 
                    color: strategies.reduce((sum, s) => sum + s.performance.totalPnL, 0) >= 0 ? '#3f8600' : '#cf1322' 
                  }}
                />
              </Card>
            </Col>
            <Col xs={12} sm={6} md={3}>
              <Card size="small">
                <Statistic
                  title="Today's P&L"
                  value={strategies.reduce((sum, s) => sum + s.performance.todayPnL, 0)}
                  precision={2}
                  prefix="$"
                  valueStyle={{ 
                    color: strategies.reduce((sum, s) => sum + s.performance.todayPnL, 0) >= 0 ? '#3f8600' : '#cf1322' 
                  }}
                />
              </Card>
            </Col>
            <Col xs={12} sm={6} md={3}>
              <Card size="small">
                <Statistic
                  title="Active Strategies"
                  value={strategies.filter(s => s.status === 'running').length}
                  suffix={`/ ${strategies.length}`}
                />
              </Card>
            </Col>
            <Col xs={12} sm={6} md={3}>
              <Card size="small">
                <Statistic
                  title="Open Positions"
                  value={strategies.reduce((sum, s) => sum + s.positions.openPositions, 0)}
                />
              </Card>
            </Col>
            <Col xs={12} sm={6} md={3}>
              <Card size="small">
                <Statistic
                  title="Total Exposure"
                  value={strategies.reduce((sum, s) => sum + s.positions.totalExposure, 0)}
                  formatter={value => `$${value.toLocaleString()}`}
                />
              </Card>
            </Col>
            <Col xs={12} sm={6} md={3}>
              <Card size="small">
                <Statistic
                  title="Avg Win Rate"
                  value={strategies.reduce((sum, s) => sum + s.performance.winRate, 0) / strategies.length || 0}
                  precision={1}
                  suffix="%"
                />
              </Card>
            </Col>
          </Row>
        </Col>

        {/* Main Strategy Table */}
        <Col xs={24}>
          <Card>
            <Table
              dataSource={filteredStrategies}
              columns={strategyColumns}
              rowKey="strategyId"
              pagination={{ pageSize: 10, showSizeChanger: true }}
              size="middle"
              scroll={{ x: 1000 }}
              rowClassName={(record) => {
                if (record.health === 'critical') return 'error-row'
                if (record.health === 'warning') return 'warning-row'
                return ''
              }}
            />
          </Card>
        </Col>
      </Row>

      {/* Strategy Details Drawer */}
      <Drawer
        title={`Strategy Details: ${selectedStrategy?.name}`}
        placement="right"
        width={720}
        open={detailsDrawerVisible}
        onClose={() => setDetailsDrawerVisible(false)}
      >
        {selectedStrategy && (
          <Tabs defaultActiveKey="overview">
            <TabPane tab="Overview" key="overview">
              <Space direction="vertical" style={{ width: '100%' }}>
                <Descriptions bordered column={2} size="small">
                  <Descriptions.Item label="Strategy ID">{selectedStrategy.strategyId}</Descriptions.Item>
                  <Descriptions.Item label="Version">{selectedStrategy.version}</Descriptions.Item>
                  <Descriptions.Item label="Status">
                    <Badge status={getStatusColor(selectedStrategy.status)} text={selectedStrategy.status} />
                  </Descriptions.Item>
                  <Descriptions.Item label="Health">
                    <Badge status={getHealthColor(selectedStrategy.health)} text={selectedStrategy.health} />
                  </Descriptions.Item>
                  <Descriptions.Item label="Uptime">
                    {Math.floor(selectedStrategy.uptime / 3600)}h {Math.floor((selectedStrategy.uptime % 3600) / 60)}m
                  </Descriptions.Item>
                  <Descriptions.Item label="Last Heartbeat">
                    {dayjs(selectedStrategy.lastHeartbeat).format('YYYY-MM-DD HH:mm:ss')}
                  </Descriptions.Item>
                </Descriptions>

                <Card title="Performance Metrics" size="small">
                  <Row gutter={16}>
                    <Col span={8}>
                      <Statistic
                        title="Total P&L"
                        value={selectedStrategy.performance.totalPnL}
                        precision={2}
                        prefix="$"
                        valueStyle={{ 
                          color: selectedStrategy.performance.totalPnL >= 0 ? '#3f8600' : '#cf1322' 
                        }}
                      />
                    </Col>
                    <Col span={8}>
                      <Statistic
                        title="Sharpe Ratio"
                        value={selectedStrategy.performance.sharpeRatio}
                        precision={2}
                      />
                    </Col>
                    <Col span={8}>
                      <Statistic
                        title="Win Rate"
                        value={selectedStrategy.performance.winRate}
                        precision={1}
                        suffix="%"
                      />
                    </Col>
                  </Row>
                </Card>

                <Card title="P&L Chart" size="small">
                  {renderMetricsChart()}
                </Card>
              </Space>
            </TabPane>

            <TabPane tab="Risk Metrics" key="risk">
              <Row gutter={[16, 16]}>
                <Col span={12}>
                  <Card size="small">
                    <Statistic
                      title="Current Risk"
                      value={selectedStrategy.risk.currentRisk * 100}
                      precision={1}
                      suffix="%"
                      valueStyle={{ 
                        color: selectedStrategy.risk.currentRisk > 0.8 ? '#cf1322' : '#3f8600' 
                      }}
                    />
                  </Card>
                </Col>
                <Col span={12}>
                  <Card size="small">
                    <Statistic
                      title="VaR (95%)"
                      value={selectedStrategy.risk.var95}
                      precision={2}
                      suffix="%"
                    />
                  </Card>
                </Col>
                <Col span={12}>
                  <Card size="small">
                    <Statistic
                      title="Leverage"
                      value={selectedStrategy.risk.leverage}
                      precision={2}
                      suffix="x"
                    />
                  </Card>
                </Col>
                <Col span={12}>
                  <Card size="small">
                    <Statistic
                      title="Correlation"
                      value={selectedStrategy.risk.correlation}
                      precision={3}
                    />
                  </Card>
                </Col>
              </Row>
            </TabPane>

            <TabPane tab="Execution" key="execution">
              <Row gutter={[16, 16]}>
                <Col span={12}>
                  <Card size="small">
                    <Statistic
                      title="Orders/Min"
                      value={selectedStrategy.execution.ordersPerMinute}
                      precision={1}
                    />
                  </Card>
                </Col>
                <Col span={12}>
                  <Card size="small">
                    <Statistic
                      title="Fill Rate"
                      value={selectedStrategy.execution.fillRate * 100}
                      precision={1}
                      suffix="%"
                    />
                  </Card>
                </Col>
                <Col span={12}>
                  <Card size="small">
                    <Statistic
                      title="Avg Slippage"
                      value={selectedStrategy.execution.averageSlippage * 10000}
                      precision={1}
                      suffix="bps"
                    />
                  </Card>
                </Col>
                <Col span={12}>
                  <Card size="small">
                    <Statistic
                      title="Latency"
                      value={selectedStrategy.execution.latency}
                      precision={1}
                      suffix="ms"
                    />
                  </Card>
                </Col>
              </Row>
            </TabPane>
          </Tabs>
        )}
      </Drawer>

      {/* Alerts Modal */}
      <Modal
        title={`Alerts: ${selectedStrategy?.name}`}
        open={alertModalVisible}
        onCancel={() => setAlertModalVisible(false)}
        footer={null}
        width={600}
      >
        <List
          dataSource={selectedStrategy?.alerts || []}
          renderItem={(alert) => (
            <List.Item
              actions={[
                !alert.acknowledged && (
                  <Button 
                    size="small" 
                    type="primary"
                    onClick={() => acknowledgeAlert(selectedStrategy!.strategyId, alert.id)}
                  >
                    Acknowledge
                  </Button>
                )
              ]}
            >
              <List.Item.Meta
                avatar={
                  <Badge 
                    status={alert.type === 'error' ? 'error' : alert.type === 'warning' ? 'warning' : 'processing'} 
                  />
                }
                title={alert.message}
                description={dayjs(alert.timestamp).format('YYYY-MM-DD HH:mm:ss')}
              />
            </List.Item>
          )}
        />
      </Modal>
    </div>
  )
}

export default LiveStrategyMonitoring