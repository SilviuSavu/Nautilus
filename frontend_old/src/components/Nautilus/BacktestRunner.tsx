import React, { useState, useEffect } from 'react'
import {
  Card,
  Row,
  Col,
  Button,
  Table,
  Progress,
  Statistic,
  Tag,
  Space,
  Modal,
  Typography,
  Alert,
  message,
  Spin,
  Badge,
  Tooltip,
  Drawer,
  notification
} from 'antd'
import {
  PlayCircleOutlined,
  PauseOutlined,
  StopOutlined,
  DownloadOutlined,
  LineChartOutlined,
  TableOutlined,
  BarChartOutlined,
  DeleteOutlined,
  EyeOutlined,
  ReloadOutlined,
  SettingOutlined,
  ClockCircleOutlined
} from '@ant-design/icons'
import dayjs from 'dayjs'
import { 
  BacktestConfig, 
  BacktestResult, 
  BacktestProgressMonitor,
  backtestService 
} from '../../services/backtestService'
import BacktestConfiguration from './BacktestConfiguration'
import BacktestResults from './BacktestResults'

const { Title, Text } = Typography

const BacktestRunner: React.FC = () => {
  const [backtests, setBacktests] = useState<BacktestResult[]>([])
  const [selectedBacktest, setSelectedBacktest] = useState<BacktestResult | null>(null)
  const [configDrawerVisible, setConfigDrawerVisible] = useState(false)
  const [resultsDrawerVisible, setResultsDrawerVisible] = useState(false)
  const [loading, setLoading] = useState(false)
  const [currentConfig, setCurrentConfig] = useState<BacktestConfig | null>(null)
  const [configValid, setConfigValid] = useState(false)
  const [progressMonitor] = useState(() => new BacktestProgressMonitor())
  const [runningBacktests, setRunningBacktests] = useState<Set<string>>(new Set())

  useEffect(() => {
    loadBacktests()
    
    // Set up polling for running backtests
    const interval = setInterval(() => {
      const activeBacktests = backtests.filter(bt => bt.status === 'running' || bt.status === 'queued')
      if (activeBacktests.length > 0) {
        loadBacktests()
      }
    }, 5000)
    
    return () => {
      clearInterval(interval)
      progressMonitor.destroy()
    }
  }, [])

  const loadBacktests = async () => {
    try {
      const backtestList = await backtestService.listBacktests()
      setBacktests(backtestList)
    } catch (error) {
      console.error('Failed to load backtests:', error)
      message.error('Failed to load backtests')
    }
  }

  const handleConfigChange = (config: BacktestConfig, isValid: boolean) => {
    setCurrentConfig(config)
    setConfigValid(isValid)
  }

  const startBacktest = async () => {
    if (!currentConfig || !configValid) {
      message.error('Please configure a valid backtest first')
      return
    }

    setLoading(true)
    try {
      const result = await backtestService.startBacktest(currentConfig)
      
      // Set up progress monitoring
      progressMonitor.subscribeToBacktest(result.backtestId, (data) => {
        if ('percentage' in data) {
          // Progress update
          setBacktests(prev => prev.map(bt => 
            bt.backtestId === result.backtestId 
              ? { ...bt, progress: data } 
              : bt
          ))
        } else if ('errorType' in data) {
          // Error update
          setBacktests(prev => prev.map(bt => 
            bt.backtestId === result.backtestId 
              ? { ...bt, status: 'failed', error: data } 
              : bt
          ))
          setRunningBacktests(prev => {
            const updated = new Set(prev)
            updated.delete(result.backtestId)
            return updated
          })
          notification.error({
            message: 'Backtest Failed',
            description: data.errorMessage,
            duration: 10
          })
        } else if ('status' in data) {
          // Completion update
          setBacktests(prev => prev.map(bt => 
            bt.backtestId === result.backtestId 
              ? { ...bt, status: data.status, completion: data } 
              : bt
          ))
          setRunningBacktests(prev => {
            const updated = new Set(prev)
            updated.delete(result.backtestId)
            return updated
          })
          
          if (data.status === 'completed') {
            notification.success({
              message: 'Backtest Completed',
              description: `Final return: ${data.summary.totalReturn.toFixed(2)}%`,
              duration: 8,
              btn: (
                <Button type="primary" onClick={() => viewResults(result.backtestId)}>
                  View Results
                </Button>
              )
            })
          }
        }
      })

      setRunningBacktests(prev => new Set(prev).add(result.backtestId))
      setConfigDrawerVisible(false)
      message.success(`Backtest started: ${result.backtestId}`)
      loadBacktests()
    } catch (error) {
      message.error(`Failed to start backtest: ${error}`)
      console.error('Backtest start error:', error)
    } finally {
      setLoading(false)
    }
  }

  const cancelBacktest = async (backtestId: string) => {
    try {
      await backtestService.cancelBacktest(backtestId)
      setRunningBacktests(prev => {
        const updated = new Set(prev)
        updated.delete(backtestId)
        return updated
      })
      message.success('Backtest cancelled')
      loadBacktests()
    } catch (error) {
      message.error('Failed to cancel backtest')
      console.error('Cancel backtest error:', error)
    }
  }

  const deleteBacktest = async (backtestId: string) => {
    try {
      await backtestService.deleteBacktest(backtestId)
      message.success('Backtest deleted')
      loadBacktests()
    } catch (error) {
      message.error('Failed to delete backtest')
      console.error('Delete backtest error:', error)
    }
  }

  const viewResults = async (backtestId: string) => {
    try {
      const results = await backtestService.getBacktestResults(backtestId)
      setSelectedBacktest(results)
      setResultsDrawerVisible(true)
    } catch (error) {
      message.error('Failed to load backtest results')
      console.error('Load results error:', error)
    }
  }

  const handleExport = (format: 'pdf' | 'excel' | 'csv') => {
    // Export functionality would be implemented here
    message.info(`Exporting backtest results as ${format.toUpperCase()}...`)
  }

  const handleCompare = (backtestIds: string[]) => {
    // Comparison functionality would be implemented here
    message.info(`Comparing ${backtestIds.length} backtests...`)
  }

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'completed': return 'success'
      case 'running': return 'processing'
      case 'failed': return 'error'
      case 'cancelled': return 'default'
      default: return 'default'
    }
  }

  const getProgressDisplay = (backtest: BacktestResult) => {
    if (backtest.status === 'running' && backtest.progress) {
      return (
        <div>
          <Progress 
            percent={backtest.progress.percentage} 
            size="small" 
            format={(percent) => `${percent}%`}
          />
          <div style={{ fontSize: '11px', color: '#666', marginTop: 2 }}>
            <ClockCircleOutlined style={{ marginRight: 4 }} />
            ETA: {Math.round(backtest.progress.estimatedTimeRemaining / 60)}m
          </div>
        </div>
      )
    } else if (backtest.status === 'completed') {
      return <Progress percent={100} size="small" status="success" />
    } else if (backtest.status === 'failed') {
      return <Progress percent={0} size="small" status="exception" />
    } else {
      return <Progress percent={0} size="small" />
    }
  }

  const backtestColumns = [
    {
      title: 'Backtest ID',
      dataIndex: 'backtestId',
      key: 'backtestId',
      width: 200,
      render: (text: string) => (
        <Text code style={{ fontSize: '12px' }}>
          {text.substring(0, 8)}...
        </Text>
      )
    },
    {
      title: 'Status',
      dataIndex: 'status',
      key: 'status',
      width: 120,
      render: (status: string, record: BacktestResult) => (
        <Space>
          <Badge status={getStatusColor(status)} text={status} />
          {runningBacktests.has(record.backtestId) && (
            <Spin size="small" />
          )}
        </Space>
      )
    },
    {
      title: 'Progress',
      key: 'progress',
      width: 180,
      render: (_, record: BacktestResult) => getProgressDisplay(record)
    },
    {
      title: 'Strategy',
      key: 'strategy',
      width: 150,
      render: (_, record: BacktestResult) => (
        record.config ? (
          <Tooltip title={record.config.strategyClass}>
            <Text ellipsis style={{ maxWidth: 120 }}>
              {record.config.strategyClass.split('.').pop() || record.config.strategyClass}
            </Text>
          </Tooltip>
        ) : '-'
      )
    },
    {
      title: 'Start Time',
      dataIndex: 'startTime',
      key: 'startTime',
      width: 150,
      render: (time: string) => time ? dayjs(time).format('MM/DD HH:mm') : '-'
    },
    {
      title: 'Duration',
      key: 'duration',
      width: 100,
      render: (_, record: BacktestResult) => {
        if (record.startTime && record.endTime) {
          const duration = dayjs(record.endTime).diff(dayjs(record.startTime), 'minute')
          return `${duration}m`
        }
        return record.status === 'running' && record.startTime ? (
          `${dayjs().diff(dayjs(record.startTime), 'minute')}m`
        ) : '-'
      }
    },
    {
      title: 'Return',
      key: 'return',
      width: 100,
      render: (_, record: BacktestResult) => (
        record.metrics ? (
          <Text type={record.metrics.totalReturn >= 0 ? 'success' : 'danger'}>
            {record.metrics.totalReturn.toFixed(2)}%
          </Text>
        ) : record.completion?.summary ? (
          <Text type={record.completion.summary.totalReturn >= 0 ? 'success' : 'danger'}>
            {record.completion.summary.totalReturn.toFixed(2)}%
          </Text>
        ) : '-'
      )
    },
    {
      title: 'Actions',
      key: 'actions',
      width: 150,
      render: (_, record: BacktestResult) => (
        <Space size="small">
          {record.status === 'running' && (
            <Tooltip title="Cancel Backtest">
              <Button 
                size="small" 
                icon={<StopOutlined />} 
                onClick={() => cancelBacktest(record.backtestId)}
                danger
              />
            </Tooltip>
          )}
          {record.status === 'completed' && (
            <Tooltip title="View Results">
              <Button 
                size="small" 
                icon={<EyeOutlined />} 
                onClick={() => viewResults(record.backtestId)}
                type="primary"
              />
            </Tooltip>
          )}
          <Tooltip title="Delete Backtest">
            <Button 
              size="small" 
              icon={<DeleteOutlined />} 
              onClick={() => deleteBacktest(record.backtestId)}
              danger
            />
          </Tooltip>
        </Space>
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
                  <BarChartOutlined style={{ marginRight: 8 }} />
                  Backtesting Engine
                </Title>
                <Text type="secondary">
                  Run and analyze historical strategy backtests with real-time monitoring
                </Text>
              </Col>
              <Col>
                <Space>
                  <Statistic
                    title="Active Backtests"
                    value={runningBacktests.size}
                    valueStyle={{ fontSize: '16px' }}
                  />
                  <Button 
                    icon={<ReloadOutlined />} 
                    onClick={loadBacktests}
                  >
                    Refresh
                  </Button>
                  <Button 
                    type="primary" 
                    icon={<SettingOutlined />}
                    onClick={() => setConfigDrawerVisible(true)}
                  >
                    Configure Backtest
                  </Button>
                </Space>
              </Col>
            </Row>
          </Card>
        </Col>

        {/* Summary Stats */}
        <Col xs={24}>
          <Row gutter={16}>
            <Col xs={6}>
              <Card size="small">
                <Statistic
                  title="Total Backtests"
                  value={backtests.length}
                  valueStyle={{ fontSize: '18px' }}
                />
              </Card>
            </Col>
            <Col xs={6}>
              <Card size="small">
                <Statistic
                  title="Completed"
                  value={backtests.filter(bt => bt.status === 'completed').length}
                  valueStyle={{ color: '#3f8600', fontSize: '18px' }}
                />
              </Card>
            </Col>
            <Col xs={6}>
              <Card size="small">
                <Statistic
                  title="Running"
                  value={backtests.filter(bt => bt.status === 'running').length}
                  valueStyle={{ color: '#1890ff', fontSize: '18px' }}
                />
              </Card>
            </Col>
            <Col xs={6}>
              <Card size="small">
                <Statistic
                  title="Failed"
                  value={backtests.filter(bt => bt.status === 'failed').length}
                  valueStyle={{ color: '#cf1322', fontSize: '18px' }}
                />
              </Card>
            </Col>
          </Row>
        </Col>

        {/* Backtests List */}
        <Col xs={24}>
          <Card title="Backtest History">
            <Table
              dataSource={backtests}
              columns={backtestColumns}
              rowKey="backtestId"
              pagination={{ 
                pageSize: 10,
                showSizeChanger: true,
                showTotal: (total, range) => `${range[0]}-${range[1]} of ${total} backtests`
              }}
              size="middle"
              scroll={{ x: 1000 }}
              loading={loading}
            />
          </Card>
        </Col>
      </Row>

      {/* Configuration Drawer */}
      <Drawer
        title="Backtest Configuration"
        placement="right"
        onClose={() => setConfigDrawerVisible(false)}
        open={configDrawerVisible}
        width={800}
        extra={
          <Button 
            type="primary" 
            icon={<PlayCircleOutlined />}
            loading={loading}
            disabled={!configValid}
            onClick={startBacktest}
          >
            Start Backtest
          </Button>
        }
      >
        <BacktestConfiguration
          onConfigChange={handleConfigChange}
          initialConfig={currentConfig || undefined}
        />
      </Drawer>

      {/* Results Drawer */}
      <Drawer
        title="Backtest Results"
        placement="right"
        onClose={() => setResultsDrawerVisible(false)}
        open={resultsDrawerVisible}
        width="90%"
        style={{ maxWidth: 1400 }}
      >
        {selectedBacktest && (
          <BacktestResults
            backtest={selectedBacktest}
            onExport={handleExport}
            onCompare={handleCompare}
            showComparison={true}
          />
        )}
      </Drawer>
    </div>
  )
}

export default BacktestRunner