import React, { useState, useEffect } from 'react'
import {
  Form,
  Select,
  DatePicker,
  InputNumber,
  Row,
  Col,
  Card,
  Switch,
  Input,
  Divider,
  Alert,
  Tooltip,
  Space,
  Typography,
  Tag
} from 'antd'
import {
  InfoCircleOutlined,
  WarningOutlined,
  CheckCircleOutlined
} from '@ant-design/icons'
import dayjs from 'dayjs'
import { BacktestConfig, BacktestValidationResult, backtestValidator } from '../../services/backtestService'

const { Title, Text } = Typography
const { RangePicker } = DatePicker
const { TextArea } = Input

interface BacktestConfigurationProps {
  onConfigChange: (config: BacktestConfig, isValid: boolean) => void
  initialConfig?: Partial<BacktestConfig>
}

interface StrategyTemplate {
  name: string
  class: string
  description: string
  parameters: Record<string, any>
  category: string
  complexity: 'beginner' | 'intermediate' | 'advanced'
}

const STRATEGY_TEMPLATES: StrategyTemplate[] = [
  {
    name: 'Moving Average Cross',
    class: 'MovingAverageCrossStrategy',
    description: 'Simple moving average crossover strategy',
    parameters: {
      fast_period: 10,
      slow_period: 20,
      stop_loss: 0.02,
      take_profit: 0.04
    },
    category: 'Trend Following',
    complexity: 'beginner'
  },
  {
    name: 'Mean Reversion',
    class: 'MeanReversionStrategy',
    description: 'Bollinger Bands mean reversion strategy',
    parameters: {
      period: 20,
      std_dev: 2.0,
      rsi_period: 14,
      rsi_oversold: 30,
      rsi_overbought: 70
    },
    category: 'Mean Reversion',
    complexity: 'intermediate'
  },
  {
    name: 'Breakout Strategy',
    class: 'BreakoutStrategy',
    description: 'Price breakout with volume confirmation',
    parameters: {
      lookback_period: 20,
      volume_factor: 1.5,
      breakout_threshold: 0.01,
      max_hold_days: 5
    },
    category: 'Breakout',
    complexity: 'intermediate'
  },
  {
    name: 'Pairs Trading',
    class: 'PairsTradingStrategy',
    description: 'Statistical arbitrage between correlated pairs',
    parameters: {
      correlation_window: 60,
      zscore_entry: 2.0,
      zscore_exit: 0.5,
      hedge_ratio_window: 30
    },
    category: 'Statistical Arbitrage',
    complexity: 'advanced'
  }
]

const COMMON_INSTRUMENTS = [
  { value: 'AAPL', label: 'Apple Inc. (AAPL)', sector: 'Technology' },
  { value: 'MSFT', label: 'Microsoft Corp. (MSFT)', sector: 'Technology' },
  { value: 'GOOGL', label: 'Alphabet Inc. (GOOGL)', sector: 'Technology' },
  { value: 'TSLA', label: 'Tesla Inc. (TSLA)', sector: 'Automotive' },
  { value: 'AMZN', label: 'Amazon.com Inc. (AMZN)', sector: 'Consumer Discretionary' },
  { value: 'META', label: 'Meta Platforms Inc. (META)', sector: 'Technology' },
  { value: 'NVDA', label: 'NVIDIA Corp. (NVDA)', sector: 'Technology' },
  { value: 'JPM', label: 'JPMorgan Chase & Co. (JPM)', sector: 'Financial Services' },
  { value: 'JNJ', label: 'Johnson & Johnson (JNJ)', sector: 'Healthcare' },
  { value: 'V', label: 'Visa Inc. (V)', sector: 'Financial Services' }
]

const VENUE_OPTIONS = [
  { value: 'NASDAQ', label: 'NASDAQ', description: 'Technology-focused exchange' },
  { value: 'NYSE', label: 'New York Stock Exchange', description: 'Major US exchange' },
  { value: 'SMART', label: 'IB Smart Routing', description: 'Intelligent order routing' },
  { value: 'SIM', label: 'Simulated Exchange', description: 'Simulation venue for backtesting' }
]

const BacktestConfiguration: React.FC<BacktestConfigurationProps> = ({
  onConfigChange,
  initialConfig
}) => {
  const [form] = Form.useForm()
  const [selectedStrategy, setSelectedStrategy] = useState<StrategyTemplate | null>(null)
  const [validation, setValidation] = useState<BacktestValidationResult>({ isValid: false, errors: [] })
  const [advancedMode, setAdvancedMode] = useState(false)
  const [currentConfig, setCurrentConfig] = useState<BacktestConfig | null>(null)

  useEffect(() => {
    if (initialConfig) {
      form.setFieldsValue(initialConfig)
      validateCurrentConfig()
    }
  }, [initialConfig])

  const validateCurrentConfig = () => {
    const values = form.getFieldsValue()
    if (values.dateRange && values.dateRange.length === 2) {
      const config: BacktestConfig = {
        strategyClass: values.strategyClass || '',
        strategyConfig: values.strategyConfig || {},
        startDate: values.dateRange[0].format('YYYY-MM-DD'),
        endDate: values.dateRange[1].format('YYYY-MM-DD'),
        instruments: values.instruments || [],
        venues: values.venues || ['SIM'],
        initialBalance: values.initialBalance || 100000,
        baseCurrency: values.baseCurrency || 'USD',
        dataConfiguration: {
          dataType: values.dataType || 'bar',
          barType: values.barType,
          resolution: values.resolution,
          dataQuality: values.dataQuality || 'cleaned'
        },
        executionSettings: advancedMode ? {
          commissionModel: values.commissionModel,
          slippageModel: values.slippageModel,
          fillModel: values.fillModel
        } : undefined,
        riskSettings: advancedMode ? {
          positionSizing: values.positionSizing,
          leverageLimit: values.leverageLimit,
          maxPortfolioRisk: values.maxPortfolioRisk
        } : undefined
      }

      const validationResult = backtestValidator.validateComplete(config)
      setValidation(validationResult)
      setCurrentConfig(config)
      onConfigChange(config, validationResult.isValid)
    }
  }

  const handleStrategyChange = (strategyClass: string) => {
    const template = STRATEGY_TEMPLATES.find(t => t.class === strategyClass)
    setSelectedStrategy(template || null)
    
    if (template) {
      form.setFieldsValue({
        strategyConfig: template.parameters
      })
    }
    
    validateCurrentConfig()
  }

  const handleFormChange = () => {
    setTimeout(validateCurrentConfig, 100) // Debounce validation
  }

  const getComplexityColor = (complexity: string) => {
    switch (complexity) {
      case 'beginner': return 'green'
      case 'intermediate': return 'orange'
      case 'advanced': return 'red'
      default: return 'default'
    }
  }

  return (
    <Card title="Backtest Configuration" size="small">
      <Form
        form={form}
        layout="vertical"
        onValuesChange={handleFormChange}
        initialValues={{
          baseCurrency: 'USD',
          initialBalance: 100000,
          venues: ['SIM'],
          dataType: 'bar',
          dataQuality: 'cleaned'
        }}
      >
        {/* Strategy Configuration */}
        <Card size="small" title="Strategy Selection" style={{ marginBottom: 16 }}>
          <Row gutter={16}>
            <Col xs={24} lg={12}>
              <Form.Item
                name="strategyClass"
                label="Strategy Template"
                rules={[{ required: true, message: 'Please select a strategy' }]}
              >
                <Select 
                  placeholder="Select strategy template"
                  onChange={handleStrategyChange}
                  optionLabelProp="label"
                >
                  {STRATEGY_TEMPLATES.map(template => (
                    <Select.Option 
                      key={template.class} 
                      value={template.class}
                      label={template.name}
                    >
                      <div>
                        <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                          <Text strong>{template.name}</Text>
                          <Tag color={getComplexityColor(template.complexity)}>
                            {template.complexity}
                          </Tag>
                        </div>
                        <Text type="secondary" style={{ fontSize: '12px' }}>
                          {template.description}
                        </Text>
                      </div>
                    </Select.Option>
                  ))}
                </Select>
              </Form.Item>
            </Col>
            <Col xs={24} lg={12}>
              {selectedStrategy && (
                <div>
                  <Text strong>Category: </Text>
                  <Tag>{selectedStrategy.category}</Tag>
                  <br />
                  <Text type="secondary" style={{ fontSize: '12px' }}>
                    {selectedStrategy.description}
                  </Text>
                </div>
              )}
            </Col>
          </Row>

          {selectedStrategy && (
            <Form.Item
              name="strategyConfig"
              label={
                <Space>
                  Strategy Parameters
                  <Tooltip title="Adjust strategy-specific parameters">
                    <InfoCircleOutlined />
                  </Tooltip>
                </Space>
              }
            >
              <TextArea
                rows={4}
                placeholder="Strategy parameters in JSON format"
                defaultValue={JSON.stringify(selectedStrategy.parameters, null, 2)}
              />
            </Form.Item>
          )}
        </Card>

        {/* Market Data Configuration */}
        <Card size="small" title="Market Data & Time Range" style={{ marginBottom: 16 }}>
          <Row gutter={16}>
            <Col xs={24} lg={12}>
              <Form.Item
                name="dateRange"
                label="Backtest Period"
                rules={[{ required: true, message: 'Please select date range' }]}
              >
                <RangePicker 
                  style={{ width: '100%' }}
                  presets={[
                    { label: 'Last 1 Month', value: [dayjs().subtract(1, 'month'), dayjs()] },
                    { label: 'Last 3 Months', value: [dayjs().subtract(3, 'month'), dayjs()] },
                    { label: 'Last 6 Months', value: [dayjs().subtract(6, 'month'), dayjs()] },
                    { label: 'Last 1 Year', value: [dayjs().subtract(1, 'year'), dayjs()] },
                    { label: 'Last 2 Years', value: [dayjs().subtract(2, 'year'), dayjs()] },
                  ]}
                />
              </Form.Item>
            </Col>
            <Col xs={24} lg={12}>
              <Form.Item
                name="dataType"
                label="Data Type"
                rules={[{ required: true, message: 'Please select data type' }]}
              >
                <Select>
                  <Select.Option value="bar">Bar Data (OHLCV)</Select.Option>
                  <Select.Option value="tick">Tick Data (L1)</Select.Option>
                </Select>
              </Form.Item>
            </Col>
          </Row>

          <Row gutter={16}>
            <Col xs={24} lg={8}>
              <Form.Item
                name="instruments"
                label="Instruments"
                rules={[{ required: true, message: 'Please select instruments' }]}
              >
                <Select 
                  mode="multiple" 
                  placeholder="Select instruments"
                  optionLabelProp="label"
                  showSearch
                  filterOption={(input, option) =>
                    (option?.label ?? '').toLowerCase().includes(input.toLowerCase())
                  }
                >
                  {COMMON_INSTRUMENTS.map(instrument => (
                    <Select.Option 
                      key={instrument.value} 
                      value={instrument.value}
                      label={instrument.value}
                    >
                      <div>
                        <Text>{instrument.label}</Text>
                        <br />
                        <Text type="secondary" style={{ fontSize: '11px' }}>
                          {instrument.sector}
                        </Text>
                      </div>
                    </Select.Option>
                  ))}
                </Select>
              </Form.Item>
            </Col>
            <Col xs={24} lg={8}>
              <Form.Item
                name="venues"
                label="Trading Venues"
                rules={[{ required: true, message: 'Please select venues' }]}
              >
                <Select mode="multiple" placeholder="Select venues">
                  {VENUE_OPTIONS.map(venue => (
                    <Select.Option key={venue.value} value={venue.value}>
                      <Tooltip title={venue.description}>
                        {venue.label}
                      </Tooltip>
                    </Select.Option>
                  ))}
                </Select>
              </Form.Item>
            </Col>
            <Col xs={24} lg={8}>
              <Form.Item
                name="dataQuality"
                label="Data Quality"
              >
                <Select>
                  <Select.Option value="raw">Raw Data</Select.Option>
                  <Select.Option value="cleaned">Cleaned Data</Select.Option>
                  <Select.Option value="validated">Validated Data</Select.Option>
                </Select>
              </Form.Item>
            </Col>
          </Row>
        </Card>

        {/* Capital Configuration */}
        <Card size="small" title="Capital & Currency" style={{ marginBottom: 16 }}>
          <Row gutter={16}>
            <Col xs={24} sm={12} lg={8}>
              <Form.Item
                name="initialBalance"
                label="Initial Balance"
                rules={[
                  { required: true, message: 'Please enter initial balance' },
                  { type: 'number', min: 1000, message: 'Minimum balance is $1,000' },
                  { type: 'number', max: 100000000, message: 'Maximum balance is $100,000,000' }
                ]}
              >
                <InputNumber
                  style={{ width: '100%' }}
                  placeholder="100000"
                  min={1000}
                  max={100000000}
                  formatter={value => `$ ${value}`.replace(/\B(?=(\d{3})+(?!\d))/g, ',')}
                  parser={value => value!.replace(/\$\s?|(,*)/g, '')}
                />
              </Form.Item>
            </Col>
            <Col xs={24} sm={12} lg={8}>
              <Form.Item
                name="baseCurrency"
                label="Base Currency"
                rules={[{ required: true, message: 'Please select base currency' }]}
              >
                <Select>
                  <Select.Option value="USD">USD - US Dollar</Select.Option>
                  <Select.Option value="EUR">EUR - Euro</Select.Option>
                  <Select.Option value="GBP">GBP - British Pound</Select.Option>
                  <Select.Option value="JPY">JPY - Japanese Yen</Select.Option>
                  <Select.Option value="CHF">CHF - Swiss Franc</Select.Option>
                  <Select.Option value="CAD">CAD - Canadian Dollar</Select.Option>
                  <Select.Option value="AUD">AUD - Australian Dollar</Select.Option>
                </Select>
              </Form.Item>
            </Col>
            <Col xs={24} lg={8}>
              <Form.Item label="Advanced Settings">
                <Switch 
                  checked={advancedMode}
                  onChange={setAdvancedMode}
                  checkedChildren="Advanced"
                  unCheckedChildren="Basic"
                />
              </Form.Item>
            </Col>
          </Row>
        </Card>

        {/* Advanced Settings */}
        {advancedMode && (
          <>
            <Card size="small" title="Execution Settings" style={{ marginBottom: 16 }}>
              <Row gutter={16}>
                <Col xs={24} lg={8}>
                  <Form.Item name="commissionModel" label="Commission Model">
                    <Select placeholder="Select commission model">
                      <Select.Option value="fixed">Fixed per Trade</Select.Option>
                      <Select.Option value="percentage">Percentage</Select.Option>
                      <Select.Option value="tiered">Tiered</Select.Option>
                      <Select.Option value="interactive_brokers">Interactive Brokers</Select.Option>
                    </Select>
                  </Form.Item>
                </Col>
                <Col xs={24} lg={8}>
                  <Form.Item name="slippageModel" label="Slippage Model">
                    <Select placeholder="Select slippage model">
                      <Select.Option value="fixed">Fixed</Select.Option>
                      <Select.Option value="volume_based">Volume Based</Select.Option>
                      <Select.Option value="market_impact">Market Impact</Select.Option>
                    </Select>
                  </Form.Item>
                </Col>
                <Col xs={24} lg={8}>
                  <Form.Item name="fillModel" label="Fill Model">
                    <Select placeholder="Select fill model">
                      <Select.Option value="market">Market Fill</Select.Option>
                      <Select.Option value="limit">Limit Fill</Select.Option>
                      <Select.Option value="probabilistic">Probabilistic</Select.Option>
                    </Select>
                  </Form.Item>
                </Col>
              </Row>
            </Card>

            <Card size="small" title="Risk Management" style={{ marginBottom: 16 }}>
              <Row gutter={16}>
                <Col xs={24} lg={8}>
                  <Form.Item name="positionSizing" label="Position Sizing">
                    <Select placeholder="Position sizing method">
                      <Select.Option value="fixed">Fixed Amount</Select.Option>
                      <Select.Option value="percent_equity">Percent of Equity</Select.Option>
                      <Select.Option value="volatility_target">Volatility Target</Select.Option>
                      <Select.Option value="kelly">Kelly Criterion</Select.Option>
                    </Select>
                  </Form.Item>
                </Col>
                <Col xs={24} lg={8}>
                  <Form.Item name="leverageLimit" label="Max Leverage">
                    <InputNumber
                      style={{ width: '100%' }}
                      min={1}
                      max={10}
                      step={0.1}
                      placeholder="1.0"
                      formatter={value => `${value}x`}
                      parser={value => value!.replace('x', '')}
                    />
                  </Form.Item>
                </Col>
                <Col xs={24} lg={8}>
                  <Form.Item name="maxPortfolioRisk" label="Max Portfolio Risk (%)">
                    <InputNumber
                      style={{ width: '100%' }}
                      min={0.1}
                      max={100}
                      step={0.1}
                      placeholder="2.0"
                      formatter={value => `${value}%`}
                      parser={value => value!.replace('%', '')}
                    />
                  </Form.Item>
                </Col>
              </Row>
            </Card>
          </>
        )}

        {/* Validation Results */}
        {validation.errors.length > 0 && (
          <Alert
            type="error"
            icon={<WarningOutlined />}
            message="Configuration Validation Failed"
            description={
              <ul style={{ margin: 0, paddingLeft: 20 }}>
                {validation.errors.map((error, index) => (
                  <li key={index}>{error}</li>
                ))}
              </ul>
            }
            style={{ marginBottom: 16 }}
          />
        )}

        {validation.isValid && currentConfig && (
          <Alert
            type="success"
            icon={<CheckCircleOutlined />}
            message="Configuration Valid"
            description="Backtest configuration is valid and ready to run."
            style={{ marginBottom: 16 }}
          />
        )}
      </Form>
    </Card>
  )
}

export default BacktestConfiguration