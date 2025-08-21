import React, { useState, useEffect, useCallback } from 'react'
import {
  Card,
  Row,
  Col,
  Form,
  Input,
  InputNumber,
  Select,
  Switch,
  Slider,
  Button,
  Space,
  Typography,
  Alert,
  Tabs,
  Collapse,
  Divider,
  Tag,
  Table,
  Modal,
  message,
  Tooltip,
  Progress,
  Tree,
  Radio,
  Checkbox,
  TimePicker,
  Upload
} from 'antd'
import {
  SettingOutlined,
  SaveOutlined,
  ExperimentOutlined,
  LineChartOutlined,
  SafetyOutlined,
  ThunderboltOutlined,
  ClockCircleOutlined,
  DollarOutlined,
  WarningOutlined,
  InfoCircleOutlined,
  PlusOutlined,
  DeleteOutlined,
  CopyOutlined,
  ImportOutlined,
  ExportOutlined,
  PlayCircleOutlined
} from '@ant-design/icons'
import dayjs from 'dayjs'

const { Title, Text, Paragraph } = Typography
const { TabPane } = Tabs
const { Panel } = Collapse
const { TextArea } = Input
const { Dragger } = Upload

interface StrategyParameter {
  name: string
  type: 'number' | 'string' | 'boolean' | 'select' | 'range' | 'time'
  value: any
  defaultValue: any
  min?: number
  max?: number
  step?: number
  options?: Array<{ label: string; value: any }>
  description: string
  category: 'entry' | 'exit' | 'risk' | 'timing' | 'other'
  validation?: {
    required?: boolean
    pattern?: string
    customValidator?: (value: any) => boolean
  }
  dependencies?: Array<{
    parameter: string
    condition: any
    action: 'show' | 'hide' | 'enable' | 'disable'
  }>
}

interface RiskManagement {
  maxPositionSize: number
  maxDailyLoss: number
  maxDrawdown: number
  stopLossType: 'fixed' | 'trailing' | 'atr' | 'percentage'
  stopLossValue: number
  takeProfitType: 'fixed' | 'ratio' | 'atr' | 'percentage'
  takeProfitValue: number
  riskPerTrade: number
  maxConcurrentTrades: number
  correlationLimit: number
  exposureLimit: number
}

interface TradingSchedule {
  enabled: boolean
  timezone: string
  sessions: Array<{
    name: string
    startTime: string
    endTime: string
    daysOfWeek: number[]
    enabled: boolean
  }>
  holidays: string[]
  preMarketTrading: boolean
  afterHoursTrading: boolean
}

interface BacktestConfiguration {
  startDate: string
  endDate: string
  initialCapital: number
  commission: number
  slippage: number
  benchmarks: string[]
  dataQuality: 'tick' | 'minute' | 'hour' | 'daily'
  includeWeekends: boolean
  dividendAdjustment: boolean
}

interface StrategyTemplate {
  templateId: string
  name: string
  description: string
  category: string
  parameters: StrategyParameter[]
  riskManagement: RiskManagement
  schedule: TradingSchedule
  tags: string[]
  isPublic: boolean
  createdBy: string
  createdAt: string
  usageCount: number
  rating: number
}

const AdvancedStrategyConfiguration: React.FC = () => {
  const [form] = Form.useForm()
  const [parameters, setParameters] = useState<StrategyParameter[]>([])
  const [riskConfig, setRiskConfig] = useState<RiskManagement>({
    maxPositionSize: 100000,
    maxDailyLoss: 5000,
    maxDrawdown: 10,
    stopLossType: 'percentage',
    stopLossValue: 2,
    takeProfitType: 'ratio',
    takeProfitValue: 3,
    riskPerTrade: 1,
    maxConcurrentTrades: 5,
    correlationLimit: 0.7,
    exposureLimit: 25
  })
  const [schedule, setSchedule] = useState<TradingSchedule>({
    enabled: true,
    timezone: 'America/New_York',
    sessions: [
      {
        name: 'Regular Trading',
        startTime: '09:30',
        endTime: '16:00',
        daysOfWeek: [1, 2, 3, 4, 5],
        enabled: true
      }
    ],
    holidays: [],
    preMarketTrading: false,
    afterHoursTrading: false
  })
  const [backtestConfig, setBacktestConfig] = useState<BacktestConfiguration>({
    startDate: dayjs().subtract(1, 'year').format('YYYY-MM-DD'),
    endDate: dayjs().format('YYYY-MM-DD'),
    initialCapital: 100000,
    commission: 1,
    slippage: 0.01,
    benchmarks: ['SPY'],
    dataQuality: 'minute',
    includeWeekends: false,
    dividendAdjustment: true
  })
  const [templates, setTemplates] = useState<StrategyTemplate[]>([])
  const [selectedTemplate, setSelectedTemplate] = useState<StrategyTemplate | null>(null)
  const [templateModalVisible, setTemplateModalVisible] = useState(false)
  const [validationErrors, setValidationErrors] = useState<Record<string, string>>({})
  const [loading, setLoading] = useState(false)
  const [previewMode, setPreviewMode] = useState(false)

  const apiUrl = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8002'

  useEffect(() => {
    loadStrategyTemplates()
    loadDefaultParameters()
  }, [])

  const loadStrategyTemplates = async () => {
    try {
      const response = await fetch(`${apiUrl}/api/v1/strategy/templates`)
      if (response.ok) {
        const data = await response.json()
        setTemplates(data.templates || [])
      }
    } catch (error) {
      console.error('Failed to load strategy templates:', error)
    }
  }

  const loadDefaultParameters = () => {
    const defaultParameters: StrategyParameter[] = [
      {
        name: 'fast_ema_period',
        type: 'number',
        value: 12,
        defaultValue: 12,
        min: 5,
        max: 50,
        step: 1,
        description: 'Period for fast EMA calculation',
        category: 'entry',
        validation: { required: true }
      },
      {
        name: 'slow_ema_period',
        type: 'number',
        value: 26,
        defaultValue: 26,
        min: 20,
        max: 100,
        step: 1,
        description: 'Period for slow EMA calculation',
        category: 'entry',
        validation: { required: true }
      },
      {
        name: 'signal_period',
        type: 'number',
        value: 9,
        defaultValue: 9,
        min: 5,
        max: 20,
        step: 1,
        description: 'Period for signal line calculation',
        category: 'entry',
        validation: { required: true }
      },
      {
        name: 'entry_threshold',
        type: 'number',
        value: 0.01,
        defaultValue: 0.01,
        min: 0.001,
        max: 0.1,
        step: 0.001,
        description: 'Minimum signal strength for entry',
        category: 'entry',
        validation: { required: true }
      },
      {
        name: 'volatility_filter',
        type: 'boolean',
        value: true,
        defaultValue: true,
        description: 'Enable volatility-based filtering',
        category: 'risk'
      },
      {
        name: 'time_filter',
        type: 'select',
        value: 'market_hours',
        defaultValue: 'market_hours',
        options: [
          { label: 'Market Hours Only', value: 'market_hours' },
          { label: 'Extended Hours', value: 'extended' },
          { label: 'All Hours', value: 'all' }
        ],
        description: 'Time-based filtering for trade execution',
        category: 'timing'
      },
      {
        name: 'position_sizing',
        type: 'select',
        value: 'fixed',
        defaultValue: 'fixed',
        options: [
          { label: 'Fixed Size', value: 'fixed' },
          { label: 'Percentage Risk', value: 'risk_pct' },
          { label: 'Kelly Criterion', value: 'kelly' },
          { label: 'Volatility Adjusted', value: 'vol_adj' }
        ],
        description: 'Method for determining position size',
        category: 'risk'
      }
    ]
    setParameters(defaultParameters)
  }

  const validateConfiguration = useCallback(() => {
    const errors: Record<string, string> = {}

    // Validate parameters
    parameters.forEach(param => {
      if (param.validation?.required && !param.value) {
        errors[param.name] = `${param.name} is required`
      }
      
      if (param.type === 'number') {
        if (param.min !== undefined && param.value < param.min) {
          errors[param.name] = `Value must be >= ${param.min}`
        }
        if (param.max !== undefined && param.value > param.max) {
          errors[param.name] = `Value must be <= ${param.max}`
        }
      }
    })

    // Validate risk management
    if (riskConfig.stopLossValue <= 0) {
      errors.stopLoss = 'Stop loss must be positive'
    }
    if (riskConfig.takeProfitValue <= 0) {
      errors.takeProfit = 'Take profit must be positive'
    }
    if (riskConfig.riskPerTrade > 10) {
      errors.riskPerTrade = 'Risk per trade should not exceed 10%'
    }

    // Cross-parameter validation
    const fastEma = parameters.find(p => p.name === 'fast_ema_period')?.value
    const slowEma = parameters.find(p => p.name === 'slow_ema_period')?.value
    if (fastEma && slowEma && fastEma >= slowEma) {
      errors.ema_periods = 'Fast EMA period must be less than slow EMA period'
    }

    setValidationErrors(errors)
    return Object.keys(errors).length === 0
  }, [parameters, riskConfig])

  const handleParameterChange = (paramName: string, value: any) => {
    setParameters(prev => prev.map(param => 
      param.name === paramName ? { ...param, value } : param
    ))
  }

  const saveConfiguration = async () => {
    if (!validateConfiguration()) {
      message.error('Please fix validation errors before saving')
      return
    }

    setLoading(true)
    try {
      const configuration = {
        parameters: parameters.reduce((acc, param) => {
          acc[param.name] = param.value
          return acc
        }, {} as Record<string, any>),
        riskManagement: riskConfig,
        schedule,
        backtestConfig
      }

      const response = await fetch(`${apiUrl}/api/v1/strategy/configuration`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(configuration)
      })

      if (response.ok) {
        message.success('Configuration saved successfully')
      } else {
        message.error('Failed to save configuration')
      }
    } catch (error) {
      message.error('Failed to save configuration')
      console.error('Save configuration error:', error)
    } finally {
      setLoading(false)
    }
  }

  const runBacktest = async () => {
    if (!validateConfiguration()) {
      message.error('Please fix validation errors before running backtest')
      return
    }

    setLoading(true)
    try {
      const response = await fetch(`${apiUrl}/api/v1/strategy/backtest/start`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          parameters: parameters.reduce((acc, param) => {
            acc[param.name] = param.value
            return acc
          }, {} as Record<string, any>),
          riskManagement: riskConfig,
          backtestConfig
        })
      })

      if (response.ok) {
        const result = await response.json()
        message.success(`Backtest started: ${result.backtestId}`)
        // Could navigate to backtest results page
      } else {
        message.error('Failed to start backtest')
      }
    } catch (error) {
      message.error('Failed to start backtest')
      console.error('Backtest error:', error)
    } finally {
      setLoading(false)
    }
  }

  const loadTemplate = (template: StrategyTemplate) => {
    setParameters(template.parameters)
    setRiskConfig(template.riskManagement)
    setSchedule(template.schedule)
    message.success(`Loaded template: ${template.name}`)
    setTemplateModalVisible(false)
  }

  const exportConfiguration = () => {
    const config = {
      parameters,
      riskManagement: riskConfig,
      schedule,
      backtestConfig,
      exportedAt: new Date().toISOString()
    }

    const blob = new Blob([JSON.stringify(config, null, 2)], { type: 'application/json' })
    const url = URL.createObjectURL(blob)
    const link = document.createElement('a')
    link.href = url
    link.download = `strategy-config-${dayjs().format('YYYY-MM-DD-HH-mm')}.json`
    link.click()
    URL.revokeObjectURL(url)
    message.success('Configuration exported')
  }

  const renderParameterInput = (param: StrategyParameter) => {
    const hasError = validationErrors[param.name]
    const errorProps = hasError ? { status: 'error' as const, help: validationErrors[param.name] } : {}

    switch (param.type) {
      case 'number':
        return (
          <Form.Item 
            label={param.name.replace(/_/g, ' ').toUpperCase()} 
            {...errorProps}
            extra={param.description}
          >
            <InputNumber
              value={param.value}
              onChange={(value) => handleParameterChange(param.name, value)}
              min={param.min}
              max={param.max}
              step={param.step}
              style={{ width: '100%' }}
            />
          </Form.Item>
        )

      case 'boolean':
        return (
          <Form.Item 
            label={param.name.replace(/_/g, ' ').toUpperCase()}
            extra={param.description}
          >
            <Switch
              checked={param.value}
              onChange={(checked) => handleParameterChange(param.name, checked)}
            />
          </Form.Item>
        )

      case 'select':
        return (
          <Form.Item 
            label={param.name.replace(/_/g, ' ').toUpperCase()} 
            {...errorProps}
            extra={param.description}
          >
            <Select
              value={param.value}
              onChange={(value) => handleParameterChange(param.name, value)}
              style={{ width: '100%' }}
            >
              {param.options?.map(option => (
                <Select.Option key={option.value} value={option.value}>
                  {option.label}
                </Select.Option>
              ))}
            </Select>
          </Form.Item>
        )

      case 'range':
        return (
          <Form.Item 
            label={param.name.replace(/_/g, ' ').toUpperCase()}
            extra={param.description}
          >
            <Slider
              value={param.value}
              onChange={(value) => handleParameterChange(param.name, value)}
              min={param.min}
              max={param.max}
              step={param.step}
              marks={param.min && param.max ? {
                [param.min]: param.min,
                [param.max]: param.max
              } : undefined}
            />
          </Form.Item>
        )

      default:
        return (
          <Form.Item 
            label={param.name.replace(/_/g, ' ').toUpperCase()} 
            {...errorProps}
            extra={param.description}
          >
            <Input
              value={param.value}
              onChange={(e) => handleParameterChange(param.name, e.target.value)}
            />
          </Form.Item>
        )
    }
  }

  const parametersByCategory = parameters.reduce((acc, param) => {
    if (!acc[param.category]) acc[param.category] = []
    acc[param.category].push(param)
    return acc
  }, {} as Record<string, StrategyParameter[]>)

  const templateColumns = [
    {
      title: 'Name',
      dataIndex: 'name',
      key: 'name',
      render: (text: string, record: StrategyTemplate) => (
        <Space>
          <Text strong>{text}</Text>
          {record.isPublic && <Tag color="green">Public</Tag>}
        </Space>
      )
    },
    {
      title: 'Category',
      dataIndex: 'category',
      key: 'category'
    },
    {
      title: 'Rating',
      dataIndex: 'rating',
      key: 'rating',
      render: (rating: number) => (
        <Progress
          percent={rating * 20}
          size="small"
          format={() => `${rating}/5`}
        />
      )
    },
    {
      title: 'Usage',
      dataIndex: 'usageCount',
      key: 'usageCount'
    },
    {
      title: 'Actions',
      key: 'actions',
      render: (_, record: StrategyTemplate) => (
        <Space>
          <Button size="small" onClick={() => loadTemplate(record)}>
            Load
          </Button>
          <Button size="small" icon={<CopyOutlined />}>
            Clone
          </Button>
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
                  <SettingOutlined style={{ marginRight: 8 }} />
                  Advanced Strategy Configuration
                </Title>
                <Text type="secondary">
                  Configure strategy parameters, risk management, and trading schedule
                </Text>
              </Col>
              <Col>
                <Space>
                  <Switch
                    checked={previewMode}
                    onChange={setPreviewMode}
                    checkedChildren="Preview"
                    unCheckedChildren="Edit"
                  />
                  <Button 
                    icon={<ImportOutlined />}
                    onClick={() => setTemplateModalVisible(true)}
                  >
                    Templates
                  </Button>
                  <Button 
                    icon={<ExportOutlined />}
                    onClick={exportConfiguration}
                  >
                    Export
                  </Button>
                  <Button 
                    icon={<ExperimentOutlined />}
                    onClick={runBacktest}
                    loading={loading}
                  >
                    Backtest
                  </Button>
                  <Button 
                    type="primary"
                    icon={<SaveOutlined />}
                    onClick={saveConfiguration}
                    loading={loading}
                  >
                    Save Configuration
                  </Button>
                </Space>
              </Col>
            </Row>

            {/* Validation Errors Alert */}
            {Object.keys(validationErrors).length > 0 && (
              <Alert
                style={{ marginTop: 16 }}
                type="error"
                message="Configuration Errors"
                description={
                  <ul>
                    {Object.entries(validationErrors).map(([key, error]) => (
                      <li key={key}>{error}</li>
                    ))}
                  </ul>
                }
                showIcon
              />
            )}
          </Card>
        </Col>

        {/* Main Configuration Tabs */}
        <Col xs={24}>
          <Card>
            <Tabs defaultActiveKey="parameters">
              <TabPane tab="Strategy Parameters" key="parameters">
                <Form form={form} layout="vertical">
                  <Collapse defaultActiveKey={['entry', 'risk']}>
                    {Object.entries(parametersByCategory).map(([category, params]) => (
                      <Panel 
                        key={category}
                        header={
                          <Space>
                            <Text strong>{category.toUpperCase()} PARAMETERS</Text>
                            <Tag>{params.length} parameters</Tag>
                          </Space>
                        }
                      >
                        <Row gutter={[16, 8]}>
                          {params.map(param => (
                            <Col xs={24} sm={12} md={8} lg={6} key={param.name}>
                              {renderParameterInput(param)}
                            </Col>
                          ))}
                        </Row>
                      </Panel>
                    ))}
                  </Collapse>
                </Form>
              </TabPane>

              <TabPane tab="Risk Management" key="risk">
                <Row gutter={[24, 16]}>
                  <Col xs={24} lg={12}>
                    <Card title="Position Sizing & Limits" size="small">
                      <Row gutter={[16, 8]}>
                        <Col span={12}>
                          <Text>Max Position Size</Text>
                          <InputNumber
                            value={riskConfig.maxPositionSize}
                            onChange={(value) => setRiskConfig(prev => ({ ...prev, maxPositionSize: value || 0 }))}
                            formatter={value => `$ ${value}`.replace(/\B(?=(\d{3})+(?!\d))/g, ',')}
                            parser={value => value!.replace(/\$\s?|(,*)/g, '')}
                            style={{ width: '100%', marginTop: 4 }}
                          />
                        </Col>
                        <Col span={12}>
                          <Text>Max Daily Loss</Text>
                          <InputNumber
                            value={riskConfig.maxDailyLoss}
                            onChange={(value) => setRiskConfig(prev => ({ ...prev, maxDailyLoss: value || 0 }))}
                            formatter={value => `$ ${value}`.replace(/\B(?=(\d{3})+(?!\d))/g, ',')}
                            parser={value => value!.replace(/\$\s?|(,*)/g, '')}
                            style={{ width: '100%', marginTop: 4 }}
                          />
                        </Col>
                        <Col span={12}>
                          <Text>Risk Per Trade (%)</Text>
                          <Slider
                            value={riskConfig.riskPerTrade}
                            onChange={(value) => setRiskConfig(prev => ({ ...prev, riskPerTrade: value }))}
                            min={0.1}
                            max={5}
                            step={0.1}
                            marks={{ 0.1: '0.1%', 2.5: '2.5%', 5: '5%' }}
                            style={{ marginTop: 8 }}
                          />
                        </Col>
                        <Col span={12}>
                          <Text>Max Concurrent Trades</Text>
                          <InputNumber
                            value={riskConfig.maxConcurrentTrades}
                            onChange={(value) => setRiskConfig(prev => ({ ...prev, maxConcurrentTrades: value || 1 }))}
                            min={1}
                            max={20}
                            style={{ width: '100%', marginTop: 4 }}
                          />
                        </Col>
                      </Row>
                    </Card>
                  </Col>
                  
                  <Col xs={24} lg={12}>
                    <Card title="Stop Loss & Take Profit" size="small">
                      <Row gutter={[16, 8]}>
                        <Col span={12}>
                          <Text>Stop Loss Type</Text>
                          <Select
                            value={riskConfig.stopLossType}
                            onChange={(value) => setRiskConfig(prev => ({ ...prev, stopLossType: value }))}
                            style={{ width: '100%', marginTop: 4 }}
                          >
                            <Select.Option value="fixed">Fixed Price</Select.Option>
                            <Select.Option value="percentage">Percentage</Select.Option>
                            <Select.Option value="trailing">Trailing</Select.Option>
                            <Select.Option value="atr">ATR Based</Select.Option>
                          </Select>
                        </Col>
                        <Col span={12}>
                          <Text>Stop Loss Value</Text>
                          <InputNumber
                            value={riskConfig.stopLossValue}
                            onChange={(value) => setRiskConfig(prev => ({ ...prev, stopLossValue: value || 0 }))}
                            min={0.1}
                            max={20}
                            step={0.1}
                            suffix={riskConfig.stopLossType === 'percentage' ? '%' : ''}
                            style={{ width: '100%', marginTop: 4 }}
                          />
                        </Col>
                        <Col span={12}>
                          <Text>Take Profit Type</Text>
                          <Select
                            value={riskConfig.takeProfitType}
                            onChange={(value) => setRiskConfig(prev => ({ ...prev, takeProfitType: value }))}
                            style={{ width: '100%', marginTop: 4 }}
                          >
                            <Select.Option value="fixed">Fixed Price</Select.Option>
                            <Select.Option value="percentage">Percentage</Select.Option>
                            <Select.Option value="ratio">Risk Ratio</Select.Option>
                            <Select.Option value="atr">ATR Based</Select.Option>
                          </Select>
                        </Col>
                        <Col span={12}>
                          <Text>Take Profit Value</Text>
                          <InputNumber
                            value={riskConfig.takeProfitValue}
                            onChange={(value) => setRiskConfig(prev => ({ ...prev, takeProfitValue: value || 0 }))}
                            min={0.1}
                            max={50}
                            step={0.1}
                            suffix={riskConfig.takeProfitType === 'ratio' ? ':1' : riskConfig.takeProfitType === 'percentage' ? '%' : ''}
                            style={{ width: '100%', marginTop: 4 }}
                          />
                        </Col>
                      </Row>
                    </Card>
                  </Col>
                </Row>
              </TabPane>

              <TabPane tab="Trading Schedule" key="schedule">
                <Row gutter={[24, 16]}>
                  <Col xs={24} lg={16}>
                    <Card title="Trading Sessions" size="small">
                      {schedule.sessions.map((session, index) => (
                        <Row key={index} gutter={[16, 8]} style={{ marginBottom: 16, padding: 16, border: '1px solid #f0f0f0', borderRadius: 6 }}>
                          <Col span={6}>
                            <Text strong>{session.name}</Text>
                            <br />
                            <Switch 
                              checked={session.enabled}
                              size="small"
                              style={{ marginTop: 4 }}
                            />
                          </Col>
                          <Col span={6}>
                            <Text>Start Time</Text>
                            <TimePicker
                              value={dayjs(session.startTime, 'HH:mm')}
                              format="HH:mm"
                              style={{ width: '100%', marginTop: 4 }}
                            />
                          </Col>
                          <Col span={6}>
                            <Text>End Time</Text>
                            <TimePicker
                              value={dayjs(session.endTime, 'HH:mm')}
                              format="HH:mm"
                              style={{ width: '100%', marginTop: 4 }}
                            />
                          </Col>
                          <Col span={6}>
                            <Text>Days</Text>
                            <Checkbox.Group
                              value={session.daysOfWeek}
                              style={{ marginTop: 4 }}
                            >
                              <Row>
                                {['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'].map((day, dayIndex) => (
                                  <Col key={dayIndex} span={8}>
                                    <Checkbox value={dayIndex + 1} style={{ fontSize: 12 }}>
                                      {day}
                                    </Checkbox>
                                  </Col>
                                ))}
                              </Row>
                            </Checkbox.Group>
                          </Col>
                        </Row>
                      ))}
                    </Card>
                  </Col>

                  <Col xs={24} lg={8}>
                    <Card title="Additional Settings" size="small">
                      <Space direction="vertical" style={{ width: '100%' }}>
                        <div>
                          <Text>Timezone</Text>
                          <Select
                            value={schedule.timezone}
                            onChange={(value) => setSchedule(prev => ({ ...prev, timezone: value }))}
                            style={{ width: '100%', marginTop: 4 }}
                          >
                            <Select.Option value="America/New_York">Eastern Time</Select.Option>
                            <Select.Option value="America/Chicago">Central Time</Select.Option>
                            <Select.Option value="America/Los_Angeles">Pacific Time</Select.Option>
                            <Select.Option value="Europe/London">London Time</Select.Option>
                            <Select.Option value="Asia/Tokyo">Tokyo Time</Select.Option>
                          </Select>
                        </div>
                        
                        <div>
                          <Checkbox 
                            checked={schedule.preMarketTrading}
                            onChange={(e) => setSchedule(prev => ({ ...prev, preMarketTrading: e.target.checked }))}
                          >
                            Pre-Market Trading
                          </Checkbox>
                        </div>
                        
                        <div>
                          <Checkbox 
                            checked={schedule.afterHoursTrading}
                            onChange={(e) => setSchedule(prev => ({ ...prev, afterHoursTrading: e.target.checked }))}
                          >
                            After-Hours Trading
                          </Checkbox>
                        </div>
                      </Space>
                    </Card>
                  </Col>
                </Row>
              </TabPane>

              <TabPane tab="Backtest Settings" key="backtest">
                <Row gutter={[24, 16]}>
                  <Col xs={24} lg={12}>
                    <Card title="Data Configuration" size="small">
                      <Row gutter={[16, 8]}>
                        <Col span={12}>
                          <Text>Initial Capital</Text>
                          <InputNumber
                            value={backtestConfig.initialCapital}
                            onChange={(value) => setBacktestConfig(prev => ({ ...prev, initialCapital: value || 0 }))}
                            formatter={value => `$ ${value}`.replace(/\B(?=(\d{3})+(?!\d))/g, ',')}
                            parser={value => value!.replace(/\$\s?|(,*)/g, '')}
                            style={{ width: '100%', marginTop: 4 }}
                          />
                        </Col>
                        <Col span={12}>
                          <Text>Commission per Trade</Text>
                          <InputNumber
                            value={backtestConfig.commission}
                            onChange={(value) => setBacktestConfig(prev => ({ ...prev, commission: value || 0 }))}
                            min={0}
                            step={0.1}
                            prefix="$"
                            style={{ width: '100%', marginTop: 4 }}
                          />
                        </Col>
                        <Col span={12}>
                          <Text>Slippage (%)</Text>
                          <InputNumber
                            value={backtestConfig.slippage}
                            onChange={(value) => setBacktestConfig(prev => ({ ...prev, slippage: value || 0 }))}
                            min={0}
                            max={1}
                            step={0.001}
                            formatter={value => `${(Number(value) * 100).toFixed(3)}%`}
                            parser={value => (parseFloat(value!.replace('%', '')) / 100)}
                            style={{ width: '100%', marginTop: 4 }}
                          />
                        </Col>
                        <Col span={12}>
                          <Text>Data Quality</Text>
                          <Select
                            value={backtestConfig.dataQuality}
                            onChange={(value) => setBacktestConfig(prev => ({ ...prev, dataQuality: value }))}
                            style={{ width: '100%', marginTop: 4 }}
                          >
                            <Select.Option value="tick">Tick Data</Select.Option>
                            <Select.Option value="minute">Minute Data</Select.Option>
                            <Select.Option value="hour">Hourly Data</Select.Option>
                            <Select.Option value="daily">Daily Data</Select.Option>
                          </Select>
                        </Col>
                      </Row>
                    </Card>
                  </Col>

                  <Col xs={24} lg={12}>
                    <Card title="Benchmarks & Adjustments" size="small">
                      <Row gutter={[16, 8]}>
                        <Col span={24}>
                          <Text>Benchmark Instruments</Text>
                          <Select
                            mode="multiple"
                            value={backtestConfig.benchmarks}
                            onChange={(value) => setBacktestConfig(prev => ({ ...prev, benchmarks: value }))}
                            placeholder="Select benchmarks"
                            style={{ width: '100%', marginTop: 4 }}
                          >
                            <Select.Option value="SPY">S&P 500 (SPY)</Select.Option>
                            <Select.Option value="QQQ">Nasdaq (QQQ)</Select.Option>
                            <Select.Option value="IWM">Russell 2000 (IWM)</Select.Option>
                            <Select.Option value="GLD">Gold (GLD)</Select.Option>
                            <Select.Option value="TLT">Bonds (TLT)</Select.Option>
                          </Select>
                        </Col>
                        <Col span={24} style={{ marginTop: 16 }}>
                          <Checkbox 
                            checked={backtestConfig.includeWeekends}
                            onChange={(e) => setBacktestConfig(prev => ({ ...prev, includeWeekends: e.target.checked }))}
                          >
                            Include Weekend Data
                          </Checkbox>
                        </Col>
                        <Col span={24}>
                          <Checkbox 
                            checked={backtestConfig.dividendAdjustment}
                            onChange={(e) => setBacktestConfig(prev => ({ ...prev, dividendAdjustment: e.target.checked }))}
                          >
                            Dividend Adjustment
                          </Checkbox>
                        </Col>
                      </Row>
                    </Card>
                  </Col>
                </Row>
              </TabPane>
            </Tabs>
          </Card>
        </Col>
      </Row>

      {/* Template Selection Modal */}
      <Modal
        title="Strategy Templates"
        open={templateModalVisible}
        onCancel={() => setTemplateModalVisible(false)}
        footer={null}
        width={800}
      >
        <Table
          dataSource={templates}
          columns={templateColumns}
          rowKey="templateId"
          pagination={{ pageSize: 5 }}
          size="small"
        />
      </Modal>
    </div>
  )
}

export default AdvancedStrategyConfiguration