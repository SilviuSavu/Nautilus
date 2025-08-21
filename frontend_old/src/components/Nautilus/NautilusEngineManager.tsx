import React, { useState, useEffect } from 'react'
import { Card, Row, Col, Button, Alert, Badge, Statistic, Modal, Form, Select, Slider, Switch, Typography, Space, Spin, Progress, Tag } from 'antd'
import { PlayCircleOutlined, StopOutlined, ReloadOutlined, SettingOutlined, ExclamationCircleOutlined, CheckCircleOutlined, CloseCircleOutlined } from '@ant-design/icons'

const { Title, Text } = Typography
const { Option } = Select
const { confirm } = Modal

interface EngineConfig {
  engine_type: 'live' | 'backtest' | 'sandbox'
  log_level: 'DEBUG' | 'INFO' | 'WARNING' | 'ERROR'
  instance_id: string
  trading_mode: 'paper' | 'live'
  max_memory: string
  max_cpu: string
  risk_engine_enabled: boolean
  max_position_size?: number
  max_order_rate?: number
}

interface EngineStatus {
  state: 'stopped' | 'starting' | 'running' | 'stopping' | 'error'
  config?: EngineConfig
  started_at?: string
  last_error?: string
  resource_usage: {
    cpu_percent: string
    memory_usage: string
    memory_percent: string
    network_io: string
    block_io: string
  }
  container_info: {
    status: string
    running: boolean
    started_at: string
    image: string
  }
  active_backtests: number
  health_check: {
    status: 'healthy' | 'unhealthy' | 'error'
    last_check: string
    details: string
  }
}

const NautilusEngineManager: React.FC = () => {
  const [engineStatus, setEngineStatus] = useState<EngineStatus | null>(null)
  const [loading, setLoading] = useState(true)
  const [actionLoading, setActionLoading] = useState<string | null>(null)
  const [configModalVisible, setConfigModalVisible] = useState(false)
  const [form] = Form.useForm()

  const defaultConfig: EngineConfig = {
    engine_type: 'live',
    log_level: 'INFO',
    instance_id: 'nautilus-001',
    trading_mode: 'paper',
    max_memory: '2g',
    max_cpu: '2.0',
    risk_engine_enabled: true,
    max_position_size: 1000000,
    max_order_rate: 100
  }

  useEffect(() => {
    fetchEngineStatus()
    const interval = setInterval(fetchEngineStatus, 5000) // Update every 5 seconds
    return () => clearInterval(interval)
  }, [])

  const fetchEngineStatus = async () => {
    try {
      const response = await fetch('/api/v1/nautilus/engine/status', {
        headers: {
          'Authorization': `Bearer ${localStorage.getItem('token')}`
        }
      })
      if (response.ok) {
        const data = await response.json()
        // CRITICAL FIX: Handle correct response format from backend
        setEngineStatus(data.status || data)
      } else {
        console.error('Failed to fetch engine status', response.status)
      }
    } catch (error) {
      console.error('Error fetching engine status:', error)
    } finally {
      setLoading(false)
    }
  }

  const handleStartEngine = async (config?: EngineConfig) => {
    const finalConfig = config || defaultConfig
    
    if (finalConfig.trading_mode === 'live') {
      confirm({
        title: '🚨 START LIVE TRADING ENGINE',
        icon: <ExclamationCircleOutlined />,
        content: (
          <div>
            <Alert
              message="WARNING: Live Trading Mode"
              description="This will enable live trading with real money. All strategy executions will use real funds and place actual trades."
              type="warning"
              style={{ marginBottom: 16 }}
            />
            <p><strong>Risks:</strong></p>
            <ul>
              <li>Real money trading</li>
              <li>Automatic order execution</li>
              <li>Market risk exposure</li>
              <li>Potential financial losses</li>
            </ul>
            <p><strong>Are you absolutely sure you want to proceed?</strong></p>
          </div>
        ),
        okText: 'START LIVE TRADING',
        okType: 'danger',
        cancelText: 'Cancel',
        onOk: () => startEngineWithConfig(finalConfig),
      })
    } else {
      startEngineWithConfig(finalConfig)
    }
  }

  const startEngineWithConfig = async (config: EngineConfig) => {
    setActionLoading('starting')
    try {
      // CRITICAL FIX: Send correct request format matching backend API
      const requestBody = {
        config: config,
        confirm_live_trading: config.trading_mode === 'live'
      }
      
      const response = await fetch('/api/v1/nautilus/engine/start', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${localStorage.getItem('token')}` // Add auth header
        },
        body: JSON.stringify(requestBody),
      })

      const result = await response.json()
      
      if (result.success) {
        await fetchEngineStatus()
      } else {
        Modal.error({
          title: 'Failed to Start Engine',
          content: result.message,
        })
      }
    } catch (error) {
      Modal.error({
        title: 'Error Starting Engine',
        content: `An error occurred while starting the engine: ${error}`,
      })
    } finally {
      setActionLoading(null)
    }
  }

  const handleStopEngine = (force = false) => {
    confirm({
      title: force ? '🚨 FORCE STOP ENGINE' : 'Stop NautilusTrader Engine',
      icon: <ExclamationCircleOutlined />,
      content: force 
        ? 'This will immediately terminate the engine and all running strategies. Any open positions will remain in the market.'
        : 'This will gracefully stop the engine and all running strategies. Open positions will be handled according to strategy settings.',
      okText: force ? 'FORCE STOP' : 'Stop Engine',
      okType: 'danger',
      cancelText: 'Cancel',
      onOk: () => stopEngine(force),
    })
  }

  const stopEngine = async (force = false) => {
    setActionLoading('stopping')
    try {
      const response = await fetch('/api/v1/nautilus/engine/stop', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${localStorage.getItem('token')}` // Add auth header
        },
        body: JSON.stringify({ force }),
      })

      const result = await response.json()
      
      if (result.success) {
        await fetchEngineStatus()
      } else {
        Modal.error({
          title: 'Failed to Stop Engine',
          content: result.message,
        })
      }
    } catch (error) {
      Modal.error({
        title: 'Error Stopping Engine',
        content: `An error occurred while stopping the engine: ${error}`,
      })
    } finally {
      setActionLoading(null)
    }
  }

  const handleRestartEngine = () => {
    confirm({
      title: 'Restart NautilusTrader Engine',
      icon: <ExclamationCircleOutlined />,
      content: 'This will stop and restart the engine with the current configuration. Any running strategies will be restarted.',
      okText: 'Restart Engine',
      okType: 'primary',
      cancelText: 'Cancel',
      onOk: restartEngine,
    })
  }

  const restartEngine = async () => {
    setActionLoading('restarting')
    try {
      const response = await fetch('/api/v1/nautilus/engine/restart', {
        method: 'POST',
      })

      const result = await response.json()
      
      if (result.success) {
        await fetchEngineStatus()
      } else {
        Modal.error({
          title: 'Failed to Restart Engine',
          content: result.message,
        })
      }
    } catch (error) {
      Modal.error({
        title: 'Error Restarting Engine',
        content: `An error occurred while restarting the engine: ${error}`,
      })
    } finally {
      setActionLoading(null)
    }
  }

  const showConfigModal = () => {
    form.setFieldsValue(engineStatus?.config || defaultConfig)
    setConfigModalVisible(true)
  }

  const handleConfigSubmit = async () => {
    try {
      const config = await form.validateFields()
      setConfigModalVisible(false)
      await handleStartEngine(config)
    } catch (error) {
      console.error('Form validation failed:', error)
    }
  }

  const getStatusColor = (state: string) => {
    switch (state) {
      case 'running': return 'success'
      case 'starting': case 'stopping': return 'processing'
      case 'error': return 'error'
      case 'stopped': return 'default'
      default: return 'default'
    }
  }

  const getStatusIcon = (state: string) => {
    switch (state) {
      case 'running': return <CheckCircleOutlined />
      case 'starting': case 'stopping': return <Spin />
      case 'error': return <CloseCircleOutlined />
      case 'stopped': return <StopOutlined />
      default: return <StopOutlined />
    }
  }

  const renderResourceUsage = () => {
    if (!engineStatus?.resource_usage) return null

    const { cpu_percent, memory_usage, memory_percent } = engineStatus.resource_usage
    
    return (
      <Card title="Resource Usage" size="small">
        <Row gutter={16}>
          <Col span={8}>
            <Statistic 
              title="CPU Usage"
              value={parseFloat(cpu_percent.replace('%', ''))}
              suffix="%"
              precision={1}
            />
          </Col>
          <Col span={8}>
            <Statistic 
              title="Memory"
              value={memory_usage}
              suffix={`(${memory_percent})`}
            />
          </Col>
          <Col span={8}>
            <div>
              <Text type="secondary">Health Status</Text>
              <br />
              <Badge 
                status={engineStatus.health_check.status === 'healthy' ? 'success' : 'error'}
                text={engineStatus.health_check.status}
              />
            </div>
          </Col>
        </Row>
      </Card>
    )
  }

  if (loading) {
    return (
      <Card title="NautilusTrader Engine Manager">
        <div style={{ textAlign: 'center', padding: '50px 0' }}>
          <Spin size="large" />
          <p style={{ marginTop: 16 }}>Loading engine status...</p>
        </div>
      </Card>
    )
  }

  if (!engineStatus) {
    return (
      <Card title="NautilusTrader Engine Manager">
        <Alert 
          message="Unable to load engine status"
          description="Please check the backend connection and try refreshing."
          type="error"
          showIcon
        />
      </Card>
    )
  }

  return (
    <div className="nautilus-engine-manager">
      <Card 
        title={
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
            <span>NautilusTrader Engine Manager</span>
            <Space>
              <Badge 
                status={getStatusColor(engineStatus.state)}
                text={engineStatus.state.toUpperCase()}
              />
              {engineStatus.config?.trading_mode === 'live' && (
                <Tag color="red">LIVE TRADING</Tag>
              )}
            </Space>
          </div>
        }
        extra={
          <Space>
            <Button 
              icon={<ReloadOutlined />}
              onClick={fetchEngineStatus}
              loading={loading}
            >
              Refresh
            </Button>
            <Button 
              icon={<SettingOutlined />}
              onClick={showConfigModal}
              disabled={engineStatus.state === 'running'}
            >
              Configure
            </Button>
          </Space>
        }
      >
        {/* Engine Status Overview */}
        <Row gutter={[16, 16]} style={{ marginBottom: 24 }}>
          <Col xs={24} sm={12} lg={6}>
            <Card size="small" style={{ textAlign: 'center' }}>
              <div style={{ fontSize: '24px', marginBottom: 8 }}>
                {getStatusIcon(engineStatus.state)}
              </div>
              <Text strong>{engineStatus.state.toUpperCase()}</Text>
              {engineStatus.started_at && (
                <div>
                  <Text type="secondary" style={{ fontSize: '12px' }}>
                    Started: {new Date(engineStatus.started_at).toLocaleString()}
                  </Text>
                </div>
              )}
            </Card>
          </Col>
          <Col xs={24} sm={12} lg={6}>
            <Card size="small" style={{ textAlign: 'center' }}>
              <Statistic 
                title="Active Backtests"
                value={engineStatus.active_backtests}
                prefix={<PlayCircleOutlined />}
              />
            </Card>
          </Col>
          <Col xs={24} sm={12} lg={6}>
            <Card size="small" style={{ textAlign: 'center' }}>
              <Statistic 
                title="Trading Mode"
                value={engineStatus.config?.trading_mode?.toUpperCase() || 'N/A'}
                valueStyle={{ 
                  color: engineStatus.config?.trading_mode === 'live' ? '#ff4d4f' : '#52c41a' 
                }}
              />
            </Card>
          </Col>
          <Col xs={24} sm={12} lg={6}>
            <Card size="small" style={{ textAlign: 'center' }}>
              <Statistic 
                title="Instance ID"
                value={engineStatus.config?.instance_id || 'N/A'}
              />
            </Card>
          </Col>
        </Row>

        {/* Error Display */}
        {engineStatus.last_error && (
          <Alert
            message="Engine Error"
            description={engineStatus.last_error}
            type="error"
            style={{ marginBottom: 16 }}
            showIcon
          />
        )}

        {/* Control Buttons */}
        <Row gutter={16} style={{ marginBottom: 24 }}>
          <Col>
            <Button
              type="primary"
              icon={<PlayCircleOutlined />}
              size="large"
              onClick={() => handleStartEngine()}
              loading={actionLoading === 'starting'}
              disabled={engineStatus.state === 'running' || actionLoading !== null}
            >
              Start Engine
            </Button>
          </Col>
          <Col>
            <Button
              danger
              icon={<StopOutlined />}
              size="large"
              onClick={() => handleStopEngine(false)}
              loading={actionLoading === 'stopping'}
              disabled={engineStatus.state === 'stopped' || actionLoading !== null}
            >
              Stop Engine
            </Button>
          </Col>
          <Col>
            <Button
              icon={<ReloadOutlined />}
              size="large"
              onClick={handleRestartEngine}
              loading={actionLoading === 'restarting'}
              disabled={engineStatus.state === 'stopped' || actionLoading !== null}
            >
              Restart Engine
            </Button>
          </Col>
          <Col>
            <Button
              danger
              ghost
              onClick={() => handleStopEngine(true)}
              loading={actionLoading === 'stopping'}
              disabled={engineStatus.state === 'stopped' || actionLoading !== null}
            >
              Force Stop
            </Button>
          </Col>
        </Row>

        {/* Resource Usage */}
        <Row gutter={16}>
          <Col span={24}>
            {renderResourceUsage()}
          </Col>
        </Row>
      </Card>

      {/* Configuration Modal */}
      <Modal
        title="Engine Configuration"
        open={configModalVisible}
        onOk={handleConfigSubmit}
        onCancel={() => setConfigModalVisible(false)}
        okText="Start with Configuration"
        cancelText="Cancel"
        width={600}
      >
        <Form
          form={form}
          layout="vertical"
          initialValues={defaultConfig}
        >
          <Row gutter={16}>
            <Col span={12}>
              <Form.Item
                label="Engine Type"
                name="engine_type"
                rules={[{ required: true }]}
              >
                <Select>
                  <Option value="live">Live Trading</Option>
                  <Option value="backtest">Backtest</Option>
                  <Option value="sandbox">Sandbox</Option>
                </Select>
              </Form.Item>
            </Col>
            <Col span={12}>
              <Form.Item
                label="Trading Mode"
                name="trading_mode"
                rules={[{ required: true }]}
              >
                <Select>
                  <Option value="paper">Paper Trading</Option>
                  <Option value="live">Live Trading</Option>
                </Select>
              </Form.Item>
            </Col>
          </Row>

          <Row gutter={16}>
            <Col span={12}>
              <Form.Item
                label="Log Level"
                name="log_level"
                rules={[{ required: true }]}
              >
                <Select>
                  <Option value="DEBUG">Debug</Option>
                  <Option value="INFO">Info</Option>
                  <Option value="WARNING">Warning</Option>
                  <Option value="ERROR">Error</Option>
                </Select>
              </Form.Item>
            </Col>
            <Col span={12}>
              <Form.Item
                label="Instance ID"
                name="instance_id"
                rules={[{ required: true }]}
              >
                <Select>
                  <Option value="nautilus-001">nautilus-001</Option>
                  <Option value="nautilus-002">nautilus-002</Option>
                  <Option value="nautilus-003">nautilus-003</Option>
                </Select>
              </Form.Item>
            </Col>
          </Row>

          <Row gutter={16}>
            <Col span={12}>
              <Form.Item
                label="Max Memory"
                name="max_memory"
              >
                <Select>
                  <Option value="1g">1 GB</Option>
                  <Option value="2g">2 GB</Option>
                  <Option value="4g">4 GB</Option>
                  <Option value="8g">8 GB</Option>
                </Select>
              </Form.Item>
            </Col>
            <Col span={12}>
              <Form.Item
                label="Max CPU"
                name="max_cpu"
              >
                <Select>
                  <Option value="1.0">1 Core</Option>
                  <Option value="2.0">2 Cores</Option>
                  <Option value="4.0">4 Cores</Option>
                </Select>
              </Form.Item>
            </Col>
          </Row>

          <Form.Item
            label="Risk Engine"
            name="risk_engine_enabled"
            valuePropName="checked"
          >
            <Switch checkedChildren="Enabled" unCheckedChildren="Disabled" />
          </Form.Item>

          <Form.Item
            noStyle
            shouldUpdate={(prevValues, currentValues) => 
              prevValues.risk_engine_enabled !== currentValues.risk_engine_enabled
            }
          >
            {({ getFieldValue }) =>
              getFieldValue('risk_engine_enabled') ? (
                <Row gutter={16}>
                  <Col span={12}>
                    <Form.Item
                      label="Max Position Size"
                      name="max_position_size"
                    >
                      <Slider
                        min={10000}
                        max={10000000}
                        step={10000}
                        marks={{
                          10000: '$10K',
                          1000000: '$1M',
                          10000000: '$10M'
                        }}
                        tooltip={{
                          formatter: (value) => `$${(value || 0).toLocaleString()}`
                        }}
                      />
                    </Form.Item>
                  </Col>
                  <Col span={12}>
                    <Form.Item
                      label="Max Order Rate (per minute)"
                      name="max_order_rate"
                    >
                      <Slider
                        min={10}
                        max={1000}
                        step={10}
                        marks={{
                          10: '10',
                          100: '100',
                          1000: '1000'
                        }}
                      />
                    </Form.Item>
                  </Col>
                </Row>
              ) : null
            }
          </Form.Item>
        </Form>
      </Modal>
    </div>
  )
}

export default NautilusEngineManager