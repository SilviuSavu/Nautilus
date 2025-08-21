import React, { useEffect, useState } from 'react'
import { Card, Row, Col, Typography, Button, Space, Alert, Badge, Statistic, Table, Tabs, FloatButton, Progress, Tag } from 'antd'
import { ApiOutlined, DatabaseOutlined, WifiOutlined, MessageOutlined, PlayCircleOutlined, StopOutlined, TrophyOutlined, ShoppingCartOutlined, LineChartOutlined, HistoryOutlined, SearchOutlined, FolderOutlined, RocketOutlined } from '@ant-design/icons'
import { useMessageBus } from '../hooks/useMessageBus'
import MessageBusViewer from '../components/MessageBusViewer'
import IBDashboard from '../components/IBDashboard'
import IBOrderPlacement from '../components/IBOrderPlacement'
import { TimeframeSelector, InstrumentSelector, ChartComponent, IndicatorPanel } from '../components/Chart'
import { InstrumentSearch, WatchlistManager } from '../components/Instruments'
import { StrategyManagementDashboard } from '../components/Strategy'
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
  last_request: string | null
  rate_limit_delay: number
  cache_expiry_seconds: number
  error_message: string | null
  adapter_version: string
}

const Dashboard: React.FC = () => {
  const [backendStatus, setBackendStatus] = useState<string>('checking')
  const [apiUrl] = useState(import.meta.env.VITE_API_BASE_URL || 'http://localhost:8002')
  const [orderModalVisible, setOrderModalVisible] = useState(false)
  const [backfillStatus, setBackfillStatus] = useState<BackfillStatus | null>(null)
  const [yfinanceStatus, setYfinanceStatus] = useState<YFinanceStatus | null>(null)
  
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
    checkBackfillStatus()
    checkYfinanceStatus()
    // Set up intervals for regular status checks
    const healthInterval = setInterval(checkBackendHealth, 30000) // Every 30 seconds
    const backfillInterval = setInterval(checkBackfillStatus, 5000) // Every 5 seconds
    const yfinanceInterval = setInterval(checkYfinanceStatus, 10000) // Every 10 seconds
    
    return () => {
      clearInterval(healthInterval)
      clearInterval(backfillInterval)
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

        {/* Historical Data Backfill Status */}
        <Col xs={24}>
          <Card
            title={
              <Space>
                <HistoryOutlined />
                Historical Data Backfill Status
                {backfillStatus?.is_running && <Badge status="processing" text="IB Gateway Running" />}
                {yfinanceStatus?.status === 'operational' && <Badge status="success" text="YFinance Available" />}
              </Space>
            }
          >
            {/* YFinance Data Source Section */}
            <Card 
              size="small" 
              title={
                <Space>
                  <ApiOutlined />
                  YFinance Data Source
                  {yfinanceStatus && (
                    <Badge 
                      status={yfinanceStatus.status === 'operational' ? 'success' : 'error'} 
                      text={yfinanceStatus.status === 'operational' ? 'Available' : 'Unavailable'} 
                    />
                  )}
                </Space>
              }
              style={{ marginBottom: 16 }}
            >
              <Row gutter={[16, 16]}>
                <Col xs={24} sm={8}>
                  <Space direction="vertical" size="small">
                    <Text type="secondary">Service Status:</Text>
                    <Text strong>{yfinanceStatus?.status || 'Loading...'}</Text>
                  </Space>
                </Col>
                <Col xs={24} sm={8}>
                  <Space direction="vertical" size="small">
                    <Text type="secondary">Initialized:</Text>
                    <Text strong>{yfinanceStatus?.initialized ? 'Yes' : 'No'}</Text>
                  </Space>
                </Col>
                <Col xs={24} sm={8}>
                  <Space direction="vertical" size="small">
                    <Text type="secondary">Rate Limit:</Text>
                    <Text strong>{yfinanceStatus?.rate_limit_delay || 0}s delay</Text>
                  </Space>
                </Col>
                <Col xs={24}>
                  <Space>
                    <Button 
                      type="primary"
                      size="small"
                      disabled={backendStatus !== 'connected' || yfinanceStatus?.status !== 'operational'}
                      onClick={async () => {
                        try {
                          const response = await fetch(`${apiUrl}/api/v1/yfinance/backfill/start`, {
                            method: 'POST',
                            headers: {
                              'X-API-Key': 'nautilus-dev-key-123',
                              'Content-Type': 'application/json'
                            },
                            body: JSON.stringify({
                              symbols: ['AAPL', 'MSFT', 'TSLA', 'GOOGL', 'AMZN', 'NVDA', 'META', 'SPY', 'QQQ'],
                              timeframes: ['1d', '1h', '30m', '15m'],
                              store_to_database: true
                            })
                          });
                          if (response.ok) {
                            const data = await response.json();
                            console.log('YFinance backfill started:', data);
                          }
                        } catch (error) {
                          console.error('Failed to start YFinance backfill:', error);
                        }
                      }}
                    >
                      Start YFinance Backfill
                    </Button>
                    <Button size="small" onClick={checkYfinanceStatus}>
                      Refresh YFinance Status
                    </Button>
                  </Space>
                </Col>
              </Row>
            </Card>

            {/* IB Gateway Backfill Section */}
            <Card 
              size="small" 
              title={
                <Space>
                  <DatabaseOutlined />
                  IB Gateway Backfill 
                  {backfillStatus?.is_running && <Badge status="processing" text="Running" />}
                </Space>
              }
            >
            {backfillStatus ? (
              <>
                {/* Progress Overview Cards */}
                <Row gutter={[12, 12]} style={{ marginBottom: 16 }}>
                  <Col xs={12} sm={6} md={3}>
                    <Card size="small" style={{ textAlign: 'center' }}>
                      <Statistic title="Queue Size" value={backfillStatus.queue_size} />
                    </Card>
                  </Col>
                  <Col xs={12} sm={6} md={3}>
                    <Card size="small" style={{ textAlign: 'center' }}>
                      <Statistic title="Active" value={backfillStatus.active_requests} />
                    </Card>
                  </Col>
                  <Col xs={12} sm={6} md={3}>
                    <Card size="small" style={{ textAlign: 'center' }}>
                      <Statistic title="Completed" value={backfillStatus.completed_requests} />
                    </Card>
                  </Col>
                  <Col xs={12} sm={6} md={3}>
                    <Card size="small" style={{ textAlign: 'center' }}>
                      <Statistic 
                        title="Failed" 
                        value={backfillStatus.failed_requests}
                        valueStyle={backfillStatus.failed_requests > 0 ? { color: '#cf1322' } : {}}
                      />
                    </Card>
                  </Col>
                </Row>

                {/* Database Stats Cards */}
                <Row gutter={[12, 12]} style={{ marginBottom: 16 }}>
                  <Col xs={12} sm={6} md={3}>
                    <Card size="small" style={{ textAlign: 'center' }}>
                      <Statistic 
                        title="Database Size" 
                        value={backfillStatus.database_size_gb} 
                        suffix="GB" 
                        precision={3}
                        valueStyle={{ color: '#1890ff' }}
                      />
                    </Card>
                  </Col>
                  <Col xs={12} sm={6} md={3}>
                    <Card size="small" style={{ textAlign: 'center' }}>
                      <Statistic 
                        title="Total Bars" 
                        value={backfillStatus.total_bars}
                        formatter={(value) => value ? value.toLocaleString() : '0'}
                      />
                    </Card>
                  </Col>
                  <Col xs={12} sm={6} md={3}>
                    <Card size="small" style={{ textAlign: 'center' }}>
                      <Statistic title="Instruments" value={backfillStatus.unique_instruments} />
                    </Card>
                  </Col>
                  <Col xs={12} sm={6} md={3}>
                    <Card size="small" style={{ textAlign: 'center' }}>
                      <Statistic title="Timeframes" value={backfillStatus.unique_timeframes} />
                    </Card>
                  </Col>
                </Row>
                
                {/* Progress Circle and Actions */}
                <Row gutter={[16, 16]} style={{ marginBottom: 16 }}>
                  <Col xs={24} md={12}>
                    <div style={{ textAlign: 'center' }}>
                      <Text type="secondary" style={{ fontSize: '14px', marginBottom: 8, display: 'block' }}>Current Batch Progress</Text>
                      <Progress
                        type="circle"
                        size={100}
                        percent={
                          backfillStatus.completed_requests + backfillStatus.active_requests > 0
                            ? Math.round(
                                (backfillStatus.completed_requests / 
                                 (backfillStatus.completed_requests + backfillStatus.active_requests)) * 100
                              )
                            : 0
                        }
                        status={backfillStatus.failed_requests > 0 ? 'exception' : 'active'}
                      />
                      <div style={{ marginTop: 12, fontSize: '12px', color: '#666' }}>
                        <div>{backfillStatus.completed_requests} of {backfillStatus.completed_requests + backfillStatus.active_requests} active requests done</div>
                        <div>{backfillStatus.queue_size} instruments queued</div>
                      </div>
                    </div>
                  </Col>
                  <Col xs={24} md={12}>
                    <div style={{ display: 'flex', flexDirection: 'column', justifyContent: 'center', height: '100%' }}>
                      <Space direction="vertical" style={{ width: '100%' }}>
                        <Button 
                          type="primary"
                          size="large"
                          block
                          disabled={backendStatus !== 'connected' || backfillStatus.is_running}
                          onClick={async () => {
                            try {
                              const response = await fetch(`${apiUrl}/api/v1/historical/backfill/start`, {
                                method: 'POST'
                              });
                              if (response.ok) {
                                checkBackfillStatus();
                              }
                            } catch (error) {
                              console.error('Failed to start backfill:', error);
                            }
                          }}
                        >
                          Start IB Gateway Backfill
                        </Button>
                        <Button 
                          block
                          disabled={backendStatus !== 'connected' || !backfillStatus.is_running}
                          onClick={async () => {
                            try {
                              const response = await fetch(`${apiUrl}/api/v1/historical/backfill/stop`, {
                                method: 'POST'
                              });
                              if (response.ok) {
                                checkBackfillStatus();
                              }
                            } catch (error) {
                              console.error('Failed to stop backfill:', error);
                            }
                          }}
                        >
                          Stop Backfill
                        </Button>
                        <Button size="small" block onClick={checkBackfillStatus}>
                          Refresh IB Gateway Status
                        </Button>
                      </Space>
                    </div>
                  </Col>
                </Row>

                {/* Active Request Details */}
                {backfillStatus.progress_details && backfillStatus.progress_details.length > 0 && (
                  <Col xs={24}>
                    <div style={{ marginTop: 16 }}>
                      <Text strong>Active Requests:</Text>
                      <Table
                        dataSource={backfillStatus.progress_details.map((detail, index) => ({
                          key: index,
                          ...detail
                        }))}
                        columns={[
                          { 
                            title: 'Symbol', 
                            dataIndex: 'symbol', 
                            key: 'symbol', 
                            width: 80 
                          },
                          { 
                            title: 'Timeframe', 
                            dataIndex: 'timeframe', 
                            key: 'timeframe', 
                            width: 80 
                          },
                          { 
                            title: 'Status', 
                            dataIndex: 'status', 
                            key: 'status', 
                            width: 100,
                            render: (status: string) => (
                              <Badge 
                                status={
                                  status === 'completed' ? 'success' : 
                                  status === 'failed' ? 'error' : 
                                  'processing'
                                } 
                                text={status}
                              />
                            )
                          },
                          { 
                            title: 'Success Count', 
                            dataIndex: 'success_count', 
                            key: 'success_count', 
                            width: 100 
                          },
                          { 
                            title: 'Errors', 
                            dataIndex: 'error_count', 
                            key: 'error_count', 
                            width: 80,
                            render: (errors: number) => (
                              <Text type={errors > 0 ? 'danger' : 'secondary'}>{errors}</Text>
                            )
                          },
                          { 
                            title: 'Progress', 
                            key: 'progress', 
                            width: 120,
                            render: (_, record) => (
                              <Progress 
                                percent={
                                  record.status === 'completed' ? 100 :
                                  record.status === 'failed' ? 0 :
                                  Math.min(Math.round((record.success_count / 1000) * 100), 99)
                                }
                                size="small"
                                status={
                                  record.status === 'completed' ? 'success' :
                                  record.status === 'failed' ? 'exception' :
                                  'active'
                                }
                              />
                            )
                          }
                        ]}
                        pagination={false}
                        size="small"
                        scroll={{ x: 600 }}
                      />
                    </div>
                  </Col>
                )}

              </>
            ) : (
              <div style={{ textAlign: 'center', padding: '20px 0' }}>
                <Text type="secondary">Loading IB Gateway backfill status...</Text>
              </div>
            )}
            </Card>
          </Card>
        </Col>


        {/* Latest Message */}
        {latestMessage && (
          <Col xs={24} sm={12} md={8}>
            <Card title="Latest Message">
              <Space direction="vertical" style={{ width: '100%' }}>
                <Text><strong>Topic:</strong> {latestMessage.topic}</Text>
                <Text><strong>Type:</strong> {latestMessage.message_type}</Text>
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

        {/* Recent Messages */}
        {recentMessages.length > 0 && (
          <Col xs={24}>
            <Card title="Recent Messages">
              <Table
                dataSource={recentMessages.map((msg, index) => ({
                  key: index,
                  topic: msg.topic,
                  message_type: msg.message_type,
                  timestamp: new Date(msg.timestamp / 1000000).toLocaleString(),
                  payload: JSON.stringify(msg.payload)
                }))}
                columns={[
                  { title: 'Topic', dataIndex: 'topic', key: 'topic', width: 200 },
                  { title: 'Type', dataIndex: 'message_type', key: 'message_type', width: 150 },
                  { title: 'Timestamp', dataIndex: 'timestamp', key: 'timestamp', width: 200 },
                  { 
                    title: 'Payload', 
                    dataIndex: 'payload', 
                    key: 'payload',
                    render: (text: string) => (
                      <Text code style={{ fontSize: '12px' }}>
                        {text.length > 100 ? text.substring(0, 100) + '...' : text}
                      </Text>
                    )
                  }
                ]}
                pagination={false}
                size="small"
                scroll={{ x: 800 }}
              />
            </Card>
          </Col>
        )}

        {/* Real-time MessageBus Data Viewer */}
        <Col xs={24}>
          <ErrorBoundary
            fallbackTitle="MessageBus Viewer Error"
            fallbackMessage="The MessageBus viewer component encountered an error. This may be due to WebSocket connection issues or message parsing problems."
          >
            <MessageBusViewer maxDisplayMessages={100} showFilters={true} />
          </ErrorBoundary>
        </Col>
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
        <Space>
          <ApiOutlined />
          System Overview
        </Space>
      ),
      children: systemOverviewTab,
    },
    {
      key: 'instruments',
      label: (
        <Space>
          <SearchOutlined />
          Instrument Search
        </Space>
      ),
      children: instrumentSearchTab,
    },
    {
      key: 'watchlists',
      label: (
        <Space>
          <FolderOutlined />
          Watchlists
        </Space>
      ),
      children: watchlistTab,
    },
    {
      key: 'chart',
      label: (
        <Space>
          <LineChartOutlined />
          Financial Chart
        </Space>
      ),
      children: chartTab,
    },
    {
      key: 'strategy',
      label: (
        <Space>
          <RocketOutlined />
          Strategy Management
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
      key: 'ib',
      label: (
        <Space>
          <TrophyOutlined />
          Interactive Brokers
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
    <div data-testid="dashboard">
      <Title level={2}>NautilusTrader Dashboard</Title>
      
      <Tabs defaultActiveKey="system" items={tabItems} />

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