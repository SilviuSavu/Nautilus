import React, { useEffect, useState } from 'react'
import { Card, Row, Col, Typography, Button, Space, Alert, Badge, Statistic, Table, Tabs, FloatButton, Progress, Tag, Switch, Tooltip } from 'antd'
import { ApiOutlined, DatabaseOutlined, WifiOutlined, MessageOutlined, PlayCircleOutlined, StopOutlined, TrophyOutlined, ShoppingCartOutlined, LineChartOutlined, HistoryOutlined, SearchOutlined, FolderOutlined, RocketOutlined, DashboardOutlined, ControlOutlined, AlertOutlined, BarChartOutlined, DeploymentUnitOutlined, SwapOutlined } from '@ant-design/icons'
import { useMessageBus } from '../hooks/useMessageBus'
import MessageBusViewer from '../components/MessageBusViewer'
import IBDashboard from '../components/IBDashboard'
import IBOrderPlacement from '../components/IBOrderPlacement'
import { TimeframeSelector, InstrumentSelector, ChartComponent, IndicatorPanel } from '../components/Chart'
import { InstrumentSearch, WatchlistManager } from '../components/Instruments'
import { StrategyManagementDashboard } from '../components/Strategy'
import { PerformanceDashboard } from '../components/Performance'
import { RiskDashboard } from '../components/Risk'
import { PortfolioVisualization } from '../components/Portfolio'
import NautilusEngineManager from '../components/Nautilus/NautilusEngineManager'
import BacktestRunner from '../components/Nautilus/BacktestRunner'
import StrategyDeploymentPipeline from '../components/Nautilus/StrategyDeploymentPipeline'
import DataCatalogBrowser from '../components/Nautilus/DataCatalogBrowser'
import ErrorBoundary from '../components/ErrorBoundary'

const { Title, Text } = Typography

interface BackfillStatus {
  is_running: boolean
  queue_size: number
  active_requests: number
  completed_requests: number
  failed_requests: number
  database_size_gb: number
  total_bars: number
  unique_instruments: number
  unique_timeframes: number
  earliest_data_ns?: number
  latest_data_ns?: number
  progress_details: Array<{
    request_id: string
    symbol: string
    timeframe: string
    status: string
    success_count: number
    error_count: number
    last_error?: string
  }>
}

interface YFinanceStatus {
  service: string
  status: string
  initialized: boolean
  connected: boolean
  instruments_loaded: number
}

interface BackfillMode {
  current_mode: 'ibkr' | 'yfinance'
  available_modes: string[]
  is_running: boolean
}

interface UnifiedBackfillStatus {
  controller: BackfillMode
  service_status: any
  timestamp: string
  last_request: string | null
  rate_limit_delay: number
  cache_expiry_seconds: number
  error_message: string | null
  adapter_version: string
}

const Dashboard: React.FC = () => {
  const [backendStatus, setBackendStatus] = useState<string>('checking')
  const [apiUrl] = useState(import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000')
  const [orderModalVisible, setOrderModalVisible] = useState(false)
  const [backfillStatus, setBackfillStatus] = useState<BackfillStatus | null>(null)
  const [yfinanceStatus, setYfinanceStatus] = useState<YFinanceStatus | null>(null)
  const [unifiedBackfillStatus, setUnifiedBackfillStatus] = useState<UnifiedBackfillStatus | null>(null)
  const [activeTab, setActiveTab] = useState('system')
  
  const {
    connectionStatus,
    messages,
    latestMessage,
    connectionInfo,
    messagesReceived,
    connect,
    disconnect,
    clearMessages,
    getStats
  } = useMessageBus()

  useEffect(() => {
    checkBackendHealth()
    checkUnifiedBackfillStatus()
    checkYfinanceStatus()
    // Set up intervals for regular status checks
    const healthInterval = setInterval(checkBackendHealth, 30000) // Every 30 seconds
    const unifiedBackfillInterval = setInterval(checkUnifiedBackfillStatus, 5000) // Every 5 seconds
    const yfinanceInterval = setInterval(checkYfinanceStatus, 10000) // Every 10 seconds
    
    return () => {
      clearInterval(healthInterval)
      clearInterval(unifiedBackfillInterval)
      clearInterval(yfinanceInterval)
    }
  }, [])

  const checkBackendHealth = async () => {
    try {
      const response = await fetch(`${apiUrl}/health`)
      if (response.ok) {
        setBackendStatus('connected')
      } else {
        setBackendStatus('error')
      }
    } catch (error) {
      setBackendStatus('error')
    }
  }

  const checkBackfillStatus = async () => {
    try {
      const response = await fetch(`${apiUrl}/api/v1/historical/backfill/status`)
      if (response.ok) {
        const data = await response.json()
        setBackfillStatus(data.backfill_status)
      }
    } catch (error) {
      console.error('Failed to fetch backfill status:', error)
    }
  }

  const checkYfinanceStatus = async () => {
    try {
      const response = await fetch(`${apiUrl}/api/v1/yfinance/status`)
      if (response.ok) {
        const data = await response.json()
        setYfinanceStatus(data)
      }
    } catch (error) {
      console.error('Failed to fetch YFinance status:', error)
    }
  }

  const checkUnifiedBackfillStatus = async () => {
    try {
      const response = await fetch(`${apiUrl}/api/v1/historical/backfill/status`)
      if (response.ok) {
        const data = await response.json()
        setUnifiedBackfillStatus(data)
        // Also update legacy backfill status for existing components
        if (data.service_status) {
          setBackfillStatus(data.service_status)
        }
      }
    } catch (error) {
      console.error('Failed to fetch unified backfill status:', error)
    }
  }

  const setBackfillMode = async (mode: 'ibkr' | 'yfinance') => {
    try {
      const response = await fetch(`${apiUrl}/api/v1/historical/backfill/set-mode`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ mode }),
      })
      if (response.ok) {
        checkUnifiedBackfillStatus()
      }
    } catch (error) {
      console.error('Failed to set backfill mode:', error)
    }
  }

  const startUnifiedBackfill = async () => {
    try {
      const response = await fetch(`${apiUrl}/api/v1/historical/backfill/start`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({}),
      })
      if (response.ok) {
        checkUnifiedBackfillStatus()
      }
    } catch (error) {
      console.error('Failed to start unified backfill:', error)
    }
  }

  const stopUnifiedBackfill = async () => {
    try {
      const response = await fetch(`${apiUrl}/api/v1/historical/backfill/stop`, {
        method: 'POST',
      })
      if (response.ok) {
        checkUnifiedBackfillStatus()
      }
    } catch (error) {
      console.error('Failed to stop unified backfill:', error)
    }
  }

  const getStatusColor = () => {
    switch (backendStatus) {
      case 'connected': return 'success'
      case 'error': return 'error'
      default: return 'info'
    }
  }

  const getStatusText = () => {
    switch (backendStatus) {
      case 'connected': return 'Backend Connected'
      case 'error': return 'Backend Disconnected'
      default: return 'Checking Backend...'
    }
  }

  const getMessageBusStatusColor = () => {
    switch (connectionStatus) {
      case 'connected': return 'success'
      case 'connecting': return 'processing'
      case 'error': return 'error'
      default: return 'default'
    }
  }

  const getMessageBusStatusText = () => {
    switch (connectionStatus) {
      case 'connected': return 'MessageBus Connected'
      case 'connecting': return 'Connecting to MessageBus...'
      case 'disconnected': return 'MessageBus Disconnected'
      case 'error': return 'MessageBus Connection Error'
      default: return 'Unknown'
    }
  }

  const stats = getStats()
  const recentMessages = messages.slice(-5).reverse() // Show last 5 messages

  const systemOverviewTab = (
    <>
      {/* Backend Status Alert */}
      <Alert
        message={getStatusText()}
        type={getStatusColor()}
        showIcon
        style={{ marginBottom: 16 }}
        action={
          <Button size="small" onClick={checkBackendHealth}>
            Refresh
          </Button>
        }
      />

      {/* MessageBus Status Alert */}
      <Alert
        message={
          <Space>
            <Badge status={getMessageBusStatusColor()} />
            {getMessageBusStatusText()}
            {connectionStatus === 'connected' && messagesReceived > 0 && (
              <Text type="secondary">({messagesReceived} messages received)</Text>
            )}
          </Space>
        }
        type={connectionStatus === 'connected' ? 'success' : connectionStatus === 'connecting' ? 'info' : 'warning'}
        showIcon
        style={{ marginBottom: 24 }}
        action={
          <Space>
            {connectionStatus === 'connected' ? (
              <Button size="small" icon={<StopOutlined />} onClick={disconnect}>
                Disconnect
              </Button>
            ) : (
              <Button size="small" icon={<PlayCircleOutlined />} onClick={connect}>
                Connect
              </Button>
            )}
            <Button size="small" onClick={clearMessages}>
              Clear
            </Button>
          </Space>
        }
      />

      <Row gutter={[16, 16]}>
        {/* API Status Card */}
        <Col xs={24} sm={12} md={8}>
          <Card
            title={
              <Space>
                <ApiOutlined />
                API Status
              </Space>
            }
          >
            <Text>Backend URL: {apiUrl}</Text>
            <br />
            <Text type={backendStatus === 'connected' ? 'success' : 'danger'}>
              Status: {getStatusText()}
            </Text>
          </Card>
        </Col>

        {/* Environment Card */}
        <Col xs={24} sm={12} md={8}>
          <Card
            title={
              <Space>
                <DatabaseOutlined />
                Environment
              </Space>
            }
          >
            <Text>Mode: {import.meta.env.MODE}</Text>
            <br />
            <Text>Debug: {import.meta.env.VITE_DEBUG || 'false'}</Text>
          </Card>
        </Col>

        {/* MessageBus Connection Info */}
        <Col xs={24} sm={12} md={8}>
          <Card
            title={
              <Space>
                <WifiOutlined />
                MessageBus Connection
              </Space>
            }
          >
            <Space direction="vertical" style={{ width: '100%' }}>
              <Text>Status: <Badge status={getMessageBusStatusColor()} text={connectionStatus} /></Text>
              {connectionInfo && (
                <>
                  <Text type="secondary">Backend State: {connectionInfo.connection_state}</Text>
                  <Text type="secondary">Reconnect Attempts: {connectionInfo.reconnect_attempts}</Text>
                  {connectionInfo.error_message && (
                    <Text type="danger">Error: {connectionInfo.error_message}</Text>
                  )}
                </>
              )}
            </Space>
          </Card>
        </Col>

        {/* Message Statistics */}
        <Col xs={24} sm={12} md={8}>
          <Card
            title={
              <Space>
                <MessageOutlined />
                Message Statistics
              </Space>
            }
          >
            <Row gutter={16}>
              <Col span={12}>
                <Statistic title="Total Messages" value={stats.totalMessages} />
              </Col>
              <Col span={12}>
                <Statistic title="Unique Topics" value={stats.uniqueTopics} />
              </Col>
            </Row>
            {latestMessage && (
              <div style={{ marginTop: 16 }}>
                <Text type="secondary">Latest Topic: {latestMessage.topic}</Text>
              </div>
            )}
          </Card>
        </Col>

        {/* Unified Data Backfill System */}
        <Col xs={24}>
          <Card
            title={
              <Space>
                <HistoryOutlined />
                Data Backfill System
                {unifiedBackfillStatus?.controller.is_running && (
                  <Badge 
                    status="processing" 
                    text={`${unifiedBackfillStatus.controller.current_mode.toUpperCase()} Running`} 
                  />
                )}
                {!unifiedBackfillStatus?.controller.is_running && (
                  <Badge status="default" text="Ready" />
                )}
              </Space>
            }
          >
            {/* Mode Toggle Section */}
            <Card 
              size="small" 
              title={
                <Space>
                  <SwapOutlined />
                  Backfill Mode
                  <Tag color={unifiedBackfillStatus?.controller.current_mode === 'ibkr' ? 'blue' : 'orange'}>
                    {unifiedBackfillStatus?.controller.current_mode.toUpperCase() || 'IBKR'}
                  </Tag>
                </Space>
              }
              style={{ marginBottom: 16 }}
            >
              <Row gutter={[16, 16]} align="middle">
                <Col xs={24} sm={12}>
                  <Space direction="vertical" size="small" style={{ width: '100%' }}>
                    <div>
                      <Text strong>Data Source Mode:</Text>
                    </div>
                    <Space>
                      <Tooltip title="Professional-grade real-time and historical data from Interactive Brokers">
                        <Tag color="blue" style={{ margin: 0 }}>
                          <DatabaseOutlined /> IBKR Gateway
                        </Tag>
                      </Tooltip>
                      <Switch
                        checked={unifiedBackfillStatus?.controller.current_mode === 'yfinance'}
                        disabled={unifiedBackfillStatus?.controller.is_running}
                        onChange={(checked) => setBackfillMode(checked ? 'yfinance' : 'ibkr')}
                        checkedChildren="YFinance"
                        unCheckedChildren="IBKR"
                      />
                      <Tooltip title="Historical data supplement for extended history and additional symbols">
                        <Tag color="orange" style={{ margin: 0 }}>
                          <ApiOutlined /> YFinance
                        </Tag>
                      </Tooltip>
                    </Space>
                    {unifiedBackfillStatus?.controller.is_running && (
                      <Alert
                        message="Cannot change mode while backfill is running"
                        type="info"
                        size="small"
                        showIcon
                      />
                    )}
                  </Space>
                </Col>
                <Col xs={24} sm={12}>
                  <Space direction="vertical" size="small" style={{ width: '100%' }}>
                    <Button
                      type="primary"
                      icon={<PlayCircleOutlined />}
                      block
                      disabled={backendStatus !== 'connected' || unifiedBackfillStatus?.controller.is_running}
                      onClick={startUnifiedBackfill}
                    >
                      Start {unifiedBackfillStatus?.controller.current_mode.toUpperCase() || 'IBKR'} Backfill
                    </Button>
                    <Button
                      icon={<StopOutlined />}
                      block
                      disabled={!unifiedBackfillStatus?.controller.is_running}
                      onClick={stopUnifiedBackfill}
                    >
                      Stop Backfill
                    </Button>
                    <Button 
                      size="small" 
                      block 
                      onClick={checkUnifiedBackfillStatus}
                    >
                      Refresh Status
                    </Button>
                  </Space>
                </Col>
              </Row>
            </Card>

            {/* Service Status Display */}
            {unifiedBackfillStatus && (
              <Card
                size="small"
                title={
                  <Space>
                    <DatabaseOutlined />
                    {unifiedBackfillStatus.controller.current_mode === 'ibkr' ? 'IBKR Gateway Status' : 'YFinance Service Status'}
                  </Space>
                }
              >
                {unifiedBackfillStatus.controller.current_mode === 'ibkr' && unifiedBackfillStatus.service_status && (
                  <Row gutter={[12, 12]}>
                    <Col xs={12} sm={6} md={4}>
                      <Card size="small" style={{ textAlign: 'center' }}>
                        <Statistic title="Queue Size" value={unifiedBackfillStatus.service_status.queue_size || 0} />
                      </Card>
                    </Col>
                    <Col xs={12} sm={6} md={4}>
                      <Card size="small" style={{ textAlign: 'center' }}>
                        <Statistic title="Active" value={unifiedBackfillStatus.service_status.active_requests || 0} />
                      </Card>
                    </Col>
                    <Col xs={12} sm={6} md={4}>
                      <Card size="small" style={{ textAlign: 'center' }}>
                        <Statistic title="Completed" value={unifiedBackfillStatus.service_status.completed_requests || 0} />
                      </Card>
                    </Col>
                    <Col xs={12} sm={6} md={4}>
                      <Card size="small" style={{ textAlign: 'center' }}>
                        <Statistic title="Total Bars" value={unifiedBackfillStatus.service_status.total_bars || 0} />
                      </Card>
                    </Col>
                    <Col xs={12} sm={6} md={4}>
                      <Card size="small" style={{ textAlign: 'center' }}>
                        <Statistic title="Instruments" value={unifiedBackfillStatus.service_status.unique_instruments || 0} />
                      </Card>
                    </Col>
                    <Col xs={12} sm={6} md={4}>
                      <Card size="small" style={{ textAlign: 'center' }}>
                        <Statistic 
                          title="Database Size" 
                          value={unifiedBackfillStatus.service_status.database_size_gb || 0} 
                          suffix="GB" 
                          precision={3}
                        />
                      </Card>
                    </Col>
                  </Row>
                )}
                
                {unifiedBackfillStatus.controller.current_mode === 'yfinance' && yfinanceStatus && (
                  <Space direction="vertical" style={{ width: '100%' }}>
                    <Row gutter={[16, 16]}>
                      <Col xs={24} sm={12} md={8}>
                        <Statistic 
                          title="Service Status" 
                          value={yfinanceStatus.status} 
                          valueStyle={{ color: yfinanceStatus.status === 'operational' ? '#3f8600' : '#cf1322' }}
                        />
                      </Col>
                      <Col xs={24} sm={12} md={8}>
                        <Statistic 
                          title="Instruments Loaded" 
                          value={yfinanceStatus.instruments_loaded} 
                        />
                      </Col>
                      <Col xs={24} sm={12} md={8}>
                        <Statistic 
                          title="Connection Status" 
                          value={yfinanceStatus.connected ? 'Connected' : 'Disconnected'}
                          valueStyle={{ color: yfinanceStatus.connected ? '#3f8600' : '#cf1322' }}
                        />
                      </Col>
                    </Row>
                    <Alert
                      message="YFinance Service Configuration"
                      description="Used as historical data supplement when IBKR data unavailable. Rate limited to prevent API blocks."
                      type="info"
                      showIcon
                    />
                  </Space>
                )}
              </Card>
            )}
          </Card>
        </Col>

        {/* Message Bus Latest Message */}
        {latestMessage && (
          <Col xs={24}>
            <Card title="Latest Message">
              <Space direction="vertical">
                <Text><strong>Type:</strong> {latestMessage.type}</Text>
                <Text><strong>Timestamp:</strong> {new Date(latestMessage.timestamp / 1000000).toLocaleTimeString()}</Text>
                <div style={{ maxHeight: 100, overflow: 'auto', background: '#f5f5f5', padding: 8, borderRadius: 4 }}>
                  <Text code style={{ fontSize: '12px' }}>
                    {JSON.stringify(latestMessage.payload, null, 2)}
                  </Text>
                </div>
              </Space>
            </Card>
          </Col>
        )}

      </Row>
    </>
  )

  const chartTab = (
    <Row gutter={[16, 16]}>
      <Col xs={24} lg={8}>
        <Card 
          title="Instrument Selection" 
          size="small"
          bodyStyle={{ padding: '12px 16px' }}
        >
          <InstrumentSelector />
        </Card>
      </Col>
      <Col xs={24} lg={8}>
        <Card 
          title="Timeframe Selection" 
          size="small"
          bodyStyle={{ padding: '12px 16px' }}
        >
          <TimeframeSelector />
        </Card>
      </Col>
      <Col xs={24} lg={8}>
        <Card 
          title="Technical Indicators" 
          size="small"
          bodyStyle={{ padding: '12px 16px' }}
        >
          <IndicatorPanel />
        </Card>
      </Col>
      <Col xs={24}>
        <ErrorBoundary
          fallbackTitle="Chart Component Error"
          fallbackMessage="The financial chart component encountered an error. This may be due to data loading issues or chart rendering problems."
        >
          <ChartComponent height={600} />
        </ErrorBoundary>
      </Col>
    </Row>
  )

  const instrumentSearchTab = (
    <Row gutter={[16, 16]}>
      <Col xs={24} lg={16}>
        <Card 
          title="Universal Instrument Search" 
          size="default"
          bodyStyle={{ padding: '24px' }}
        >
          <ErrorBoundary
            fallbackTitle="Instrument Search Error"
            fallbackMessage="The instrument search component encountered an error. This may be due to data loading issues or search API problems."
          >
            <InstrumentSearch
              onInstrumentSelect={(instrument) => {
                console.log('Selected instrument:', instrument)
                // TODO: Integrate with chart selection
              }}
              showFavorites={true}
              showRecentSelections={true}
              maxResults={100}
            />
          </ErrorBoundary>
        </Card>
      </Col>
      <Col xs={24} lg={8}>
        <Card 
          title="Search Features" 
          size="small"
          bodyStyle={{ padding: '16px' }}
        >
          <div style={{ fontSize: '14px', color: '#666' }}>
            <div style={{ marginBottom: '12px' }}>
              <strong>🔍 Search Capabilities:</strong>
            </div>
            <ul style={{ paddingLeft: '20px', margin: 0 }}>
              <li>Fuzzy symbol matching</li>
              <li>Company name search</li>
              <li>Venue filtering</li>
              <li>Asset class categorization</li>
              <li>Real-time venue status</li>
            </ul>
            <div style={{ marginTop: '16px', marginBottom: '8px' }}>
              <strong>⭐ Features:</strong>
            </div>
            <ul style={{ paddingLeft: '20px', margin: 0 }}>
              <li>Favorites management</li>
              <li>Recent selections</li>
              <li>Watchlist support</li>
              <li>Multi-venue search</li>
            </ul>
          </div>
        </Card>
        
        <Card 
          title="Supported Asset Classes" 
          size="small"
          style={{ marginTop: '16px' }}
          bodyStyle={{ padding: '16px' }}
        >
          <Space direction="vertical" size="small" style={{ width: '100%' }}>
            <Tag color="blue">STK - Stocks</Tag>
            <Tag color="green">CASH - Forex</Tag>
            <Tag color="orange">FUT - Futures</Tag>
            <Tag color="purple">IND - Indices</Tag>
            <Tag color="red">OPT - Options</Tag>
            <Tag color="cyan">BOND - Bonds</Tag>
            <Tag color="gold">CRYPTO - Crypto ETFs</Tag>
          </Space>
        </Card>
      </Col>
    </Row>
  )

  const watchlistTab = (
    <Row gutter={[16, 16]}>
      <Col xs={24} lg={16}>
        <ErrorBoundary
          fallbackTitle="Watchlist Manager Error"
          fallbackMessage="The watchlist manager component encountered an error. This may be due to storage issues or watchlist API problems."
        >
          <WatchlistManager
            onInstrumentSelect={(instrument) => {
              console.log('Selected instrument from watchlist:', instrument)
              // TODO: Integrate with chart selection
            }}
            showCreateButton={true}
            compactMode={false}
          />
        </ErrorBoundary>
      </Col>
      <Col xs={24} lg={8}>
        <Card 
          title="Watchlist Features" 
          size="small"
          bodyStyle={{ padding: '16px' }}
        >
          <div style={{ fontSize: '14px', color: '#666' }}>
            <div style={{ marginBottom: '12px' }}>
              <strong>📁 Watchlist Management:</strong>
            </div>
            <ul style={{ paddingLeft: '20px', margin: 0 }}>
              <li>Create multiple watchlists</li>
              <li>Drag & drop organization</li>
              <li>Add notes to instruments</li>
              <li>Real-time venue status</li>
              <li>Import/Export functionality</li>
            </ul>
            <div style={{ marginTop: '16px', marginBottom: '8px' }}>
              <strong>⚡ Quick Actions:</strong>
            </div>
            <ul style={{ paddingLeft: '20px', margin: 0 }}>
              <li>Add to favorites</li>
              <li>Remove from watchlist</li>
              <li>Export as CSV/JSON</li>
              <li>Persistent storage</li>
            </ul>
          </div>
        </Card>
        
        <Card 
          title="Data Formats" 
          size="small"
          style={{ marginTop: '16px' }}
          bodyStyle={{ padding: '16px' }}
        >
          <div style={{ fontSize: '14px', color: '#666' }}>
            <div style={{ marginBottom: '8px' }}>
              <strong>Export Formats:</strong>
            </div>
            <Space direction="vertical" size="small">
              <Tag color="blue">JSON - Full data preservation</Tag>
              <Tag color="green">CSV - Spreadsheet compatible</Tag>
            </Space>
          </div>
        </Card>
      </Col>
    </Row>
  )

  const tabItems = [
    {
      key: 'system',
      label: (
        <Space size={4}>
          <ApiOutlined />
          <span style={{ fontSize: '12px' }}>System</span>
        </Space>
      ),
      children: systemOverviewTab,
    },
    {
      key: 'nautilus-engine',
      label: (
        <Space size={4}>
          <ControlOutlined />
          <span style={{ fontSize: '12px' }}>Engine</span>
        </Space>
      ),
      children: (
        <ErrorBoundary
          fallbackTitle="NautilusTrader Engine Error"
          fallbackMessage="The NautilusTrader Engine component encountered an error. This may be due to Docker connectivity issues or engine communication problems."
        >
          <NautilusEngineManager />
        </ErrorBoundary>
      ),
    },
    {
      key: 'backtesting',
      label: (
        <Space size={4}>
          <BarChartOutlined />
          <span style={{ fontSize: '12px' }}>Backtest</span>
        </Space>
      ),
      children: (
        <ErrorBoundary
          fallbackTitle="Backtesting Engine Error"
          fallbackMessage="The Backtesting Engine component encountered an error. This may be due to NautilusTrader engine connectivity or backtest configuration issues."
        >
          <BacktestRunner />
        </ErrorBoundary>
      ),
    },
    {
      key: 'deployment',
      label: (
        <Space size={4}>
          <DeploymentUnitOutlined />
          <span style={{ fontSize: '12px' }}>Deploy</span>
        </Space>
      ),
      children: (
        <ErrorBoundary
          fallbackTitle="Strategy Deployment Error"
          fallbackMessage="The Strategy Deployment component encountered an error. This may be due to deployment pipeline or approval workflow issues."
        >
          <StrategyDeploymentPipeline />
        </ErrorBoundary>
      ),
    },
    {
      key: 'data-catalog',
      label: (
        <Space size={4}>
          <DatabaseOutlined />
          <span style={{ fontSize: '12px' }}>Data</span>
        </Space>
      ),
      children: (
        <ErrorBoundary
          fallbackTitle="Data Catalog Error"
          fallbackMessage="The Data Catalog component encountered an error. This may be due to data pipeline connectivity or catalog service issues."
        >
          <DataCatalogBrowser />
        </ErrorBoundary>
      ),
    },
    {
      key: 'instruments',
      label: (
        <Space size={4}>
          <SearchOutlined />
          <span style={{ fontSize: '12px' }}>Search</span>
        </Space>
      ),
      children: instrumentSearchTab,
    },
    {
      key: 'watchlists',
      label: (
        <Space size={4}>
          <FolderOutlined />
          <span style={{ fontSize: '12px' }}>Watchlist</span>
        </Space>
      ),
      children: watchlistTab,
    },
    {
      key: 'chart',
      label: (
        <Space size={4}>
          <LineChartOutlined />
          <span style={{ fontSize: '12px' }}>Chart</span>
        </Space>
      ),
      children: chartTab,
    },
    {
      key: 'strategy',
      label: (
        <Space size={4}>
          <RocketOutlined />
          <span style={{ fontSize: '12px' }}>Strategy</span>
        </Space>
      ),
      children: (
        <ErrorBoundary
          fallbackTitle="Strategy Management Error"
          fallbackMessage="The Strategy Management component encountered an error. This may be due to strategy service issues or configuration problems."
        >
          <StrategyManagementDashboard />
        </ErrorBoundary>
      ),
    },
    {
      key: 'performance',
      label: (
        <Space size={4}>
          <DashboardOutlined />
          <span style={{ fontSize: '12px' }}>Perform</span>
        </Space>
      ),
      children: (
        <ErrorBoundary
          fallbackTitle="Performance Monitoring Error"
          fallbackMessage="The Performance Monitoring component encountered an error. This may be due to API connectivity issues or data loading problems."
        >
          <PerformanceDashboard />
        </ErrorBoundary>
      ),
    },
    {
      key: 'portfolio',
      label: (
        <Space size={4}>
          <TrophyOutlined />
          <span style={{ fontSize: '12px' }}>Portfolio</span>
        </Space>
      ),
      children: (
        <ErrorBoundary
          fallbackTitle="Portfolio Visualization Error"
          fallbackMessage="The Portfolio Visualization component encountered an error. This may be due to portfolio API connectivity issues or data aggregation problems."
        >
          <PortfolioVisualization />
        </ErrorBoundary>
      ),
    },
    {
      key: 'risk',
      label: (
        <Space size={4}>
          <AlertOutlined />
          <span style={{ fontSize: '12px' }}>Risk</span>
        </Space>
      ),
      children: (
        <ErrorBoundary
          fallbackTitle="Risk Management Dashboard Error"
          fallbackMessage="The Risk Management component encountered an error. This may be due to risk API connectivity issues or calculation problems."
        >
          <RiskDashboard portfolioId="default" />
        </ErrorBoundary>
      ),
    },
    {
      key: 'ib',
      label: (
        <Space size={4}>
          <TrophyOutlined />
          <span style={{ fontSize: '12px' }}>IB</span>
        </Space>
      ),
      children: (
        <ErrorBoundary
          fallbackTitle="Interactive Brokers Dashboard Error"
          fallbackMessage="The IB Dashboard component encountered an error. This may be due to IB Gateway connection issues or API endpoint problems."
        >
          <IBDashboard />
        </ErrorBoundary>
      ),
    },
  ]

  return (
    <div data-testid="dashboard" style={{ width: '100%', maxWidth: '100vw', overflow: 'hidden' }}>
      <Title level={2}>NautilusTrader Dashboard</Title>
      
      <div style={{ 
        width: '100%', 
        overflowX: 'auto',
        overflowY: 'hidden',
        marginBottom: '16px'
      }}>
        <Tabs 
          activeKey={activeTab} 
          onChange={(key) => {
            console.log('Tab changed to:', key);
            setActiveTab(key);
          }} 
          items={tabItems}
          size="small"
          tabBarStyle={{ 
            margin: 0,
            minWidth: 'max-content',
            paddingBottom: '4px',
            fontSize: '12px'
          }}
          moreIcon={<>•••</>}
          tabBarGutter={4}
          style={{
            minWidth: '600px' // Reduced from 800px for smaller screens
          }}
        />
      </div>

      {/* Floating Action Button for Order Placement */}
      <FloatButton
        icon={<ShoppingCartOutlined />}
        tooltip="Place IB Order"
        onClick={() => setOrderModalVisible(true)}
        style={{ right: 24, bottom: 24 }}
      />

      {/* Order Placement Modal */}
      <IBOrderPlacement
        visible={orderModalVisible}
        onClose={() => setOrderModalVisible(false)}
        onOrderPlaced={(orderData) => {
          console.log('Order placed:', orderData);
          // Could trigger refresh of orders or show notification
        }}
      />
    </div>
  )
}

export default Dashboard