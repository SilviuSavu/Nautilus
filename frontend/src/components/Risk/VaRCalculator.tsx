import React, { useState, useCallback, useEffect } from 'react';
import {
  Card,
  Row,
  Col,
  Form,
  InputNumber,
  Select,
  Button,
  Table,
  Typography,
  Space,
  Alert,
  Tabs,
  Statistic,
  Progress,
  Tag,
  Tooltip,
  Radio,
  DatePicker,
  Switch,
  Divider,
  Descriptions,
  Badge
} from 'antd';
import {
  CalculatorOutlined,
  LineChartOutlined,
  BarChartOutlined,
  RiseOutlined,
  InfoCircleOutlined,
  ExperimentOutlined,
  ThunderboltOutlined,
  BulbOutlined,
  WarningOutlined
} from '@ant-design/icons';
import { Line, Column } from '@ant-design/plots';
import dayjs from 'dayjs';

const { Title, Text, Paragraph } = Typography;
const { Option } = Select;
const { RangePicker } = DatePicker;

interface VaRCalculatorProps {
  portfolioId: string;
  className?: string;
}

interface VaRResult {
  method: string;
  confidence_level: number;
  time_horizon: number;
  var_amount: number;
  var_percentage: number;
  expected_shortfall: number;
  back_test_score?: number;
  model_accuracy?: number;
  calculation_time_ms: number;
}

interface VaRBacktest {
  date: string;
  predicted_var: number;
  actual_pnl: number;
  breach: boolean;
  confidence_level: number;
}

interface ModelComparison {
  method: string;
  var_95: number;
  var_99: number;
  expected_shortfall: number;
  accuracy_score: number;
  computation_time: number;
  pros: string[];
  cons: string[];
}

const VaRCalculator: React.FC<VaRCalculatorProps> = ({
  portfolioId,
  className
}) => {
  const [form] = Form.useForm();
  const [calculating, setCalculating] = useState(false);
  const [results, setResults] = useState<VaRResult[]>([]);
  const [selectedMethod, setSelectedMethod] = useState<string>('historical');
  const [backtestData, setBacktestData] = useState<VaRBacktest[]>([]);
  const [modelComparison, setModelComparison] = useState<ModelComparison[]>([]);
  const [advancedMode, setAdvancedMode] = useState(false);

  // VaR calculation methods
  const varMethods = [
    {
      id: 'historical',
      name: 'Historical Simulation',
      description: 'Uses historical returns to simulate potential losses',
      accuracy: 85,
      speed: 95,
      interpretability: 100
    },
    {
      id: 'parametric',
      name: 'Parametric (Normal)',
      description: 'Assumes normal distribution of returns',
      accuracy: 75,
      speed: 100,
      interpretability: 95
    },
    {
      id: 'monte_carlo',
      name: 'Monte Carlo Simulation',
      description: 'Generates random scenarios based on statistical models',
      accuracy: 90,
      speed: 70,
      interpretability: 80
    },
    {
      id: 'garch',
      name: 'GARCH Model',
      description: 'Considers volatility clustering and time-varying volatility',
      accuracy: 88,
      speed: 75,
      interpretability: 70
    },
    {
      id: 'extreme_value',
      name: 'Extreme Value Theory',
      description: 'Focuses on tail risks and extreme losses',
      accuracy: 92,
      speed: 65,
      interpretability: 65
    },
    {
      id: 'filtered_hs',
      name: 'Filtered Historical Simulation',
      description: 'Combines historical simulation with volatility modeling',
      accuracy: 89,
      speed: 80,
      interpretability: 75
    }
  ];

  // Generate mock VaR calculation
  const calculateVaR = useCallback(async (values: any) => {
    setCalculating(true);

    try {
      // Simulate API delay
      await new Promise(resolve => setTimeout(resolve, 2000));

      const portfolioValue = 10000000; // Mock portfolio value
      const methods = values.methods || [selectedMethod];
      const confidenceLevels = values.confidence_levels || [95, 99];
      const timeHorizon = values.time_horizon || 1;

      const newResults: VaRResult[] = [];

      methods.forEach((method: string) => {
        confidenceLevels.forEach((confidence: number) => {
          // Mock VaR calculation based on method and confidence level
          const baseVaR = portfolioValue * 0.02; // 2% base VaR
          const methodMultiplier = {
            historical: 1.0,
            parametric: 0.85,
            monte_carlo: 1.1,
            garch: 1.05,
            extreme_value: 1.25,
            filtered_hs: 1.02
          }[method] || 1.0;

          const confidenceMultiplier = confidence === 99 ? 1.5 : 1.0;
          const timeMultiplier = Math.sqrt(timeHorizon);

          const varAmount = baseVaR * methodMultiplier * confidenceMultiplier * timeMultiplier;
          const varPercentage = (varAmount / portfolioValue) * 100;
          const expectedShortfall = varAmount * 1.3; // ES typically 30% higher than VaR

          newResults.push({
            method,
            confidence_level: confidence,
            time_horizon: timeHorizon,
            var_amount: varAmount,
            var_percentage: varPercentage,
            expected_shortfall: expectedShortfall,
            back_test_score: Math.random() * 20 + 80, // 80-100%
            model_accuracy: Math.random() * 10 + 85, // 85-95%
            calculation_time_ms: Math.random() * 500 + 100
          });
        });
      });

      setResults(newResults);

      // Generate backtest data
      generateBacktestData(newResults[0]);

      // Generate model comparison
      generateModelComparison();

    } catch (error) {
      console.error('VaR calculation error:', error);
    } finally {
      setCalculating(false);
    }
  }, [selectedMethod]);

  const generateBacktestData = (result: VaRResult) => {
    const backtests: VaRBacktest[] = [];
    const days = 252; // One trading year

    for (let i = 0; i < days; i++) {
      const date = dayjs().subtract(days - i, 'day').format('YYYY-MM-DD');
      const predictedVar = result.var_amount * (0.9 + Math.random() * 0.2); // ±10% variation
      const actualPnL = (Math.random() - 0.5) * result.var_amount * 3; // Random P&L
      const breach = actualPnL < -predictedVar;

      backtests.push({
        date,
        predicted_var: predictedVar,
        actual_pnl: actualPnL,
        breach,
        confidence_level: result.confidence_level
      });
    }

    setBacktestData(backtests);
  };

  const generateModelComparison = () => {
    const comparison: ModelComparison[] = varMethods.map(method => ({
      method: method.name,
      var_95: Math.random() * 100000 + 150000,
      var_99: Math.random() * 150000 + 200000,
      expected_shortfall: Math.random() * 200000 + 250000,
      accuracy_score: method.accuracy + Math.random() * 10 - 5,
      computation_time: (100 - method.speed) * 10 + Math.random() * 50,
      pros: getMethodPros(method.id),
      cons: getMethodCons(method.id)
    }));

    setModelComparison(comparison);
  };

  const getMethodPros = (methodId: string): string[] => {
    const pros: Record<string, string[]> = {
      historical: ['No distributional assumptions', 'Easy to understand', 'Captures fat tails'],
      parametric: ['Fast computation', 'Smooth estimates', 'Good for normal markets'],
      monte_carlo: ['Flexible modeling', 'Captures complex relationships', 'Handles non-linear portfolios'],
      garch: ['Models volatility clustering', 'Adapts to market conditions', 'Good for time series'],
      extreme_value: ['Excellent for tail risks', 'Robust to outliers', 'Regulatory preferred'],
      filtered_hs: ['Combines best of both worlds', 'Adaptive volatility', 'Empirically robust']
    };
    return pros[methodId] || [];
  };

  const getMethodCons = (methodId: string): string[] => {
    const cons: Record<string, string[]> = {
      historical: ['Limited by historical data', 'Ghost features', 'Slow adaptation'],
      parametric: ['Strong assumptions', 'Poor tail modeling', 'Underestimates extreme risks'],
      monte_carlo: ['Computationally intensive', 'Model risk', 'Complex calibration'],
      garch: ['Parameter uncertainty', 'Complex interpretation', 'Requires long series'],
      extreme_value: ['Limited data for estimation', 'Complex methodology', 'Threshold selection'],
      filtered_hs: ['Complex implementation', 'Model selection risk', 'Calibration challenges']
    };
    return cons[methodId] || [];
  };

  const getMethodColor = (accuracy: number) => {
    if (accuracy >= 90) return '#52c41a';
    if (accuracy >= 80) return '#faad14';
    return '#ff4d4f';
  };

  const resultColumns = [
    {
      title: 'Method',
      dataIndex: 'method',
      key: 'method',
      width: 150,
      render: (method: string) => {
        const methodInfo = varMethods.find(m => m.id === method);
        return (
          <Space direction="vertical" size={0}>
            <Text strong>{methodInfo?.name || method}</Text>
            <Text type="secondary" style={{ fontSize: '11px' }}>
              {methodInfo?.description}
            </Text>
          </Space>
        );
      }
    },
    {
      title: 'Confidence',
      dataIndex: 'confidence_level',
      key: 'confidence_level',
      width: 80,
      render: (level: number) => <Text>{level}%</Text>
    },
    {
      title: 'Horizon',
      dataIndex: 'time_horizon',
      key: 'time_horizon',
      width: 80,
      render: (days: number) => <Text>{days}d</Text>
    },
    {
      title: 'VaR Amount',
      dataIndex: 'var_amount',
      key: 'var_amount',
      width: 120,
      render: (amount: number) => (
        <Text strong style={{ color: '#ff4d4f' }}>
          ${amount.toLocaleString(undefined, { maximumFractionDigits: 0 })}
        </Text>
      )
    },
    {
      title: 'VaR %',
      dataIndex: 'var_percentage',
      key: 'var_percentage',
      width: 80,
      render: (percentage: number) => (
        <Text>{percentage.toFixed(2)}%</Text>
      )
    },
    {
      title: 'Expected Shortfall',
      dataIndex: 'expected_shortfall',
      key: 'expected_shortfall',
      width: 140,
      render: (es: number) => (
        <Text style={{ color: '#fa8c16' }}>
          ${es.toLocaleString(undefined, { maximumFractionDigits: 0 })}
        </Text>
      )
    },
    {
      title: 'Accuracy',
      dataIndex: 'model_accuracy',
      key: 'model_accuracy',
      width: 100,
      render: (accuracy?: number) => {
        if (!accuracy) return <Text type="secondary">N/A</Text>;
        return (
          <Space>
            <Progress
              type="circle"
              percent={accuracy}
              width={30}
              strokeColor={getMethodColor(accuracy)}
              format={() => ''}
            />
            <Text style={{ color: getMethodColor(accuracy) }}>
              {accuracy.toFixed(1)}%
            </Text>
          </Space>
        );
      }
    },
    {
      title: 'Time',
      dataIndex: 'calculation_time_ms',
      key: 'calculation_time_ms',
      width: 80,
      render: (ms: number) => (
        <Text style={{ fontSize: '12px' }}>{ms.toFixed(0)}ms</Text>
      )
    }
  ];

  const comparisonColumns = [
    {
      title: 'Method',
      dataIndex: 'method',
      key: 'method',
      width: 150,
      render: (method: string) => <Text strong>{method}</Text>
    },
    {
      title: 'VaR 95%',
      dataIndex: 'var_95',
      key: 'var_95',
      width: 120,
      render: (value: number) => (
        <Text>${value.toLocaleString(undefined, { maximumFractionDigits: 0 })}</Text>
      )
    },
    {
      title: 'VaR 99%',
      dataIndex: 'var_99',
      key: 'var_99',
      width: 120,
      render: (value: number) => (
        <Text>${value.toLocaleString(undefined, { maximumFractionDigits: 0 })}</Text>
      )
    },
    {
      title: 'Accuracy Score',
      dataIndex: 'accuracy_score',
      key: 'accuracy_score',
      width: 120,
      render: (score: number) => (
        <Badge
          color={getMethodColor(score)}
          text={`${score.toFixed(1)}%`}
        />
      )
    },
    {
      title: 'Computation Time',
      dataIndex: 'computation_time',
      key: 'computation_time',
      width: 120,
      render: (time: number) => (
        <Text>{time.toFixed(0)}ms</Text>
      )
    }
  ];

  const backtestChartData = backtestData.map(item => ({
    date: item.date,
    'Predicted VaR': -item.predicted_var / 1000, // Convert to thousands and make negative
    'Actual P&L': item.actual_pnl / 1000,
    breach: item.breach
  }));

  const backtestChartConfig = {
    data: backtestChartData,
    xField: 'date',
    yField: 'value',
    seriesField: 'type',
    smooth: true,
    animation: {
      appear: {
        animation: 'path-in',
        duration: 1000,
      },
    },
    color: ['#ff4d4f', '#1890ff'],
    point: {
      size: 2,
      shape: 'circle',
    }
  };

  // Calculate breach statistics
  const breachStats = backtestData.reduce((acc, item) => {
    acc.total++;
    if (item.breach) acc.breaches++;
    return acc;
  }, { total: 0, breaches: 0 });

  const breachRate = breachStats.total > 0 ? (breachStats.breaches / breachStats.total) * 100 : 0;
  const expectedBreachRate = results.length > 0 ? 100 - results[0].confidence_level : 5;

  return (
    <div className={className}>
      {/* Header */}
      <Row gutter={16} style={{ marginBottom: 16 }}>
        <Col span={18}>
          <Card>
            <Title level={3}>
              <CalculatorOutlined style={{ marginRight: 8 }} />
              Value at Risk (VaR) Calculator
            </Title>
            <Paragraph type="secondary">
              Calculate portfolio Value at Risk using multiple methodologies and compare results.
              Advanced risk metrics include Expected Shortfall and backtesting analysis.
            </Paragraph>
          </Card>
        </Col>
        <Col span={6}>
          <Card size="small">
            <Space direction="vertical" style={{ width: '100%' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                <Text>Advanced Mode</Text>
                <Switch
                  checked={advancedMode}
                  onChange={setAdvancedMode}
                  size="small"
                />
              </div>
              <Divider style={{ margin: '8px 0' }} />
              <Text type="secondary" style={{ fontSize: '11px' }}>
                Enable advanced parameters and model comparisons
              </Text>
            </Space>
          </Card>
        </Col>
      </Row>

      <Row gutter={16}>
        <Col span={8}>
          {/* Calculation Parameters */}
          <Card
            title={
              <Space>
                <SettingOutlined />
                Calculation Parameters
              </Space>
            }
            size="small"
          >
            <Form
              form={form}
              layout="vertical"
              onFinish={calculateVaR}
              initialValues={{
                methods: ['historical'],
                confidence_levels: [95, 99],
                time_horizon: 1,
                lookback_period: 252,
                portfolio_value: 10000000
              }}
            >
              <Form.Item
                name="methods"
                label="VaR Methods"
                tooltip="Select one or more VaR calculation methods"
              >
                <Select
                  mode="multiple"
                  placeholder="Select methods"
                  style={{ width: '100%' }}
                >
                  {varMethods.map(method => (
                    <Option key={method.id} value={method.id}>
                      <Space>
                        {method.name}
                        <Badge
                          color={getMethodColor(method.accuracy)}
                          text={`${method.accuracy}%`}
                        />
                      </Space>
                    </Option>
                  ))}
                </Select>
              </Form.Item>

              <Row gutter={8}>
                <Col span={12}>
                  <Form.Item
                    name="confidence_levels"
                    label="Confidence Levels"
                    tooltip="Statistical confidence levels for VaR calculation"
                  >
                    <Select mode="multiple" placeholder="Select levels">
                      <Option value={90}>90%</Option>
                      <Option value={95}>95%</Option>
                      <Option value={99}>99%</Option>
                      <Option value={99.9}>99.9%</Option>
                    </Select>
                  </Form.Item>
                </Col>
                <Col span={12}>
                  <Form.Item
                    name="time_horizon"
                    label="Time Horizon"
                    tooltip="Number of days for VaR calculation"
                  >
                    <Select>
                      <Option value={1}>1 Day</Option>
                      <Option value={10}>10 Days</Option>
                      <Option value={21}>1 Month</Option>
                      <Option value={63}>3 Months</Option>
                      <Option value={252}>1 Year</Option>
                    </Select>
                  </Form.Item>
                </Col>
              </Row>

              {advancedMode && (
                <>
                  <Form.Item
                    name="lookback_period"
                    label="Lookback Period (Days)"
                    tooltip="Historical data period for calculation"
                  >
                    <InputNumber
                      style={{ width: '100%' }}
                      min={30}
                      max={2520}
                      placeholder="252"
                    />
                  </Form.Item>

                  <Form.Item
                    name="decay_factor"
                    label="Decay Factor"
                    tooltip="Exponential decay factor for weighted historical simulation"
                  >
                    <InputNumber
                      style={{ width: '100%' }}
                      min={0.9}
                      max={1.0}
                      step={0.01}
                      placeholder="0.94"
                    />
                  </Form.Item>

                  <Form.Item name="include_correlations" valuePropName="checked">
                    <Switch /> Include Asset Correlations
                  </Form.Item>

                  <Form.Item name="include_volatility_scaling" valuePropName="checked">
                    <Switch /> Apply Volatility Scaling
                  </Form.Item>
                </>
              )}

              <Form.Item>
                <Button
                  type="primary"
                  htmlType="submit"
                  loading={calculating}
                  icon={<CalculatorOutlined />}
                  block
                  size="large"
                >
                  Calculate VaR
                </Button>
              </Form.Item>
            </Form>
          </Card>

          {/* Method Information */}
          {!calculating && (
            <Card
              title="Method Characteristics"
              size="small"
              style={{ marginTop: 16 }}
            >
              {varMethods.map(method => (
                <div key={method.id} style={{ marginBottom: 12 }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                    <Text strong style={{ fontSize: '12px' }}>{method.name}</Text>
                  </div>
                  <div style={{ marginTop: 4 }}>
                    <Row gutter={4}>
                      <Col span={8}>
                        <Text style={{ fontSize: '10px', color: '#666' }}>Accuracy</Text>
                        <Progress
                          percent={method.accuracy}
                          size="small"
                          showInfo={false}
                          strokeColor={getMethodColor(method.accuracy)}
                        />
                      </Col>
                      <Col span={8}>
                        <Text style={{ fontSize: '10px', color: '#666' }}>Speed</Text>
                        <Progress
                          percent={method.speed}
                          size="small"
                          showInfo={false}
                          strokeColor="#1890ff"
                        />
                      </Col>
                      <Col span={8}>
                        <Text style={{ fontSize: '10px', color: '#666' }}>Clarity</Text>
                        <Progress
                          percent={method.interpretability}
                          size="small"
                          showInfo={false}
                          strokeColor="#52c41a"
                        />
                      </Col>
                    </Row>
                  </div>
                </div>
              ))}
            </Card>
          )}
        </Col>

        <Col span={16}>
          {/* Results */}
          {calculating && (
            <Card>
              <div style={{ textAlign: 'center', padding: '40px 0' }}>
                <Space direction="vertical">
                  <ThunderboltOutlined style={{ fontSize: '48px', color: '#1890ff' }} />
                  <Title level={4}>Calculating VaR...</Title>
                  <Text type="secondary">Running statistical models and risk calculations</Text>
                  <Progress percent={30} status="active" />
                </Space>
              </div>
            </Card>
          )}

          {!calculating && results.length > 0 && (
            <Tabs
              defaultActiveKey="results"
              items={[
                {
                  key: 'results',
                  label: (
                    <Space>
                      <CalculatorOutlined />
                      Results
                      <Badge count={results.length} size="small" />
                    </Space>
                  ),
                  children: (
                    <Card title="VaR Calculation Results">
                      <Table
                        dataSource={results}
                        columns={resultColumns}
                        rowKey={(record) => `${record.method}_${record.confidence_level}`}
                        pagination={false}
                        size="small"
                      />
                    </Card>
                  )
                },
                {
                  key: 'backtest',
                  label: (
                    <Space>
                      <LineChartOutlined />
                      Backtesting
                      <Badge
                        color={breachRate <= expectedBreachRate * 1.5 ? 'green' : 'red'}
                        text={`${breachRate.toFixed(1)}%`}
                      />
                    </Space>
                  ),
                  children: (
                    <Card title="VaR Model Backtesting">
                      <Row gutter={16} style={{ marginBottom: 16 }}>
                        <Col span={8}>
                          <Statistic
                            title="Breach Rate"
                            value={breachRate}
                            precision={1}
                            suffix="%"
                            valueStyle={{ 
                              color: breachRate <= expectedBreachRate * 1.5 ? '#52c41a' : '#ff4d4f' 
                            }}
                          />
                        </Col>
                        <Col span={8}>
                          <Statistic
                            title="Expected Breach Rate"
                            value={expectedBreachRate}
                            precision={1}
                            suffix="%"
                          />
                        </Col>
                        <Col span={8}>
                          <Statistic
                            title="Total Breaches"
                            value={breachStats.breaches}
                            suffix={`/ ${breachStats.total}`}
                          />
                        </Col>
                      </Row>

                      {breachRate > expectedBreachRate * 2 && (
                        <Alert
                          message="Model Performance Warning"
                          description={`Breach rate (${breachRate.toFixed(1)}%) significantly exceeds expected rate (${expectedBreachRate}%). Consider recalibrating the model.`}
                          type="warning"
                          showIcon
                          style={{ marginBottom: 16 }}
                        />
                      )}

                      <div style={{ height: 300 }}>
                        <Line
                          data={backtestChartData.flatMap(item => [
                            { date: item.date, value: item['Predicted VaR'], type: 'Predicted VaR' },
                            { date: item.date, value: item['Actual P&L'], type: 'Actual P&L' }
                          ])}
                          xField="date"
                          yField="value"
                          seriesField="type"
                          smooth={false}
                          color={['#ff4d4f', '#1890ff']}
                          point={{ size: 1 }}
                          animation={{
                            appear: {
                              animation: 'path-in',
                              duration: 1000,
                            },
                          }}
                        />
                      </div>
                    </Card>
                  )
                },
                {
                  key: 'comparison',
                  label: (
                    <Space>
                      <BarChartOutlined />
                      Model Comparison
                    </Space>
                  ),
                  children: (
                    <Card title="VaR Method Comparison">
                      <Table
                        dataSource={modelComparison}
                        columns={comparisonColumns}
                        rowKey="method"
                        pagination={false}
                        size="small"
                        expandable={{
                          expandedRowRender: (record) => (
                            <div style={{ padding: 16 }}>
                              <Row gutter={16}>
                                <Col span={12}>
                                  <Title level={5} style={{ color: '#52c41a' }}>
                                    <BulbOutlined /> Advantages
                                  </Title>
                                  <ul>
                                    {record.pros.map((pro, index) => (
                                      <li key={index} style={{ fontSize: '13px' }}>{pro}</li>
                                    ))}
                                  </ul>
                                </Col>
                                <Col span={12}>
                                  <Title level={5} style={{ color: '#ff4d4f' }}>
                                    <WarningOutlined /> Limitations
                                  </Title>
                                  <ul>
                                    {record.cons.map((con, index) => (
                                      <li key={index} style={{ fontSize: '13px' }}>{con}</li>
                                    ))}
                                  </ul>
                                </Col>
                              </Row>
                            </div>
                          ),
                          rowExpandable: () => true,
                        }}
                      />
                    </Card>
                  )
                }
              ]}
            />
          )}

          {!calculating && results.length === 0 && (
            <Card>
              <div style={{ textAlign: 'center', padding: '40px 0' }}>
                <Space direction="vertical">
                  <ExperimentOutlined style={{ fontSize: '48px', color: '#d9d9d9' }} />
                  <Title level={4} type="secondary">No Results Yet</Title>
                  <Text type="secondary">
                    Configure your parameters and click "Calculate VaR" to run the analysis
                  </Text>
                </Space>
              </div>
            </Card>
          )}
        </Col>
      </Row>
    </div>
  );
};

export default VaRCalculator;