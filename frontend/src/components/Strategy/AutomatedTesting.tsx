/**
 * Automated Testing Interface
 * Strategy testing interface with backtesting and validation
 */

import React, { useState, useEffect } from 'react';
import {
  Card,
  Row,
  Col,
  Button,
  Form,
  Select,
  Input,
  InputNumber,
  Switch,
  Space,
  Table,
  Tag,
  Progress,
  Modal,
  Tabs,
  Alert,
  Statistic,
  Timeline,
  Tooltip,
  Typography,
  Divider,
  List,
  Drawer,
  Result
} from 'antd';
import {
  PlayCircleOutlined,
  PauseCircleOutlined,
  StopOutlined,
  ReloadOutlined,
  ExperimentOutlined,
  CheckCircleOutlined,
  CloseCircleOutlined,
  ExclamationCircleOutlined,
  LineChartOutlined,
  FileTextOutlined,
  SettingOutlined,
  HistoryOutlined,
  TrophyOutlined,
  WarningOutlined
} from '@ant-design/icons';
import { Line } from '@ant-design/plots';
import { useStrategyTesting } from '../../hooks/strategy/useStrategyTesting';

const { Title, Text, Paragraph } = Typography;
const { TabPane } = Tabs;
const { Option } = Select;

interface AutomatedTestingProps {
  strategyId: string;
  strategyCode?: string;
  strategyConfig?: Record<string, any>;
}

export const AutomatedTesting: React.FC<AutomatedTestingProps> = ({
  strategyId,
  strategyCode = '',
  strategyConfig = {}
}) => {
  const [testConfigForm] = Form.useForm();
  const [quickTestForm] = Form.useForm();
  
  const [activeTab, setActiveTab] = useState('configure');
  const [selectedTestSuite, setSelectedTestSuite] = useState<string | null>(null);
  const [selectedBacktest, setSelectedBacktest] = useState<string | null>(null);
  const [testModalVisible, setTestModalVisible] = useState(false);
  const [resultsDrawerVisible, setResultsDrawerVisible] = useState(false);
  const [validationDrawerVisible, setValidationDrawerVisible] = useState(false);

  const {
    testSuites,
    backtestResults,
    validationResults,
    activeTests,
    loading,
    error,
    startTestSuite,
    startBacktest,
    startPaperTradingTest,
    startValidationTest,
    stopTest,
    runQuickValidation,
    getTestSuite,
    getBacktestResult,
    getValidationResult,
    isTestActive,
    fetchTestSuites,
    fetchBacktestResult
  } = useStrategyTesting();

  // Initialize data
  useEffect(() => {
    fetchTestSuites(strategyId);
  }, [strategyId, fetchTestSuites]);

  // Handle full test suite
  const handleFullTestSuite = async (values: any) => {
    try {
      const testConfig = {
        backtest: {
          durationDays: values.backtestDays,
          initialCapital: values.initialCapital,
          instruments: values.instruments
        },
        paperTrading: {
          durationMinutes: values.paperTradingMinutes,
          maxPositions: values.maxPositions,
          riskLimits: values.riskLimits || {}
        },
        stressTesting: {
          scenarios: values.stressScenarios || ['high_volatility', 'market_crash'],
          volatilityMultiplier: values.volatilityMultiplier || 2.0
        },
        performanceTargets: {
          minSharpeRatio: values.minSharpeRatio || 1.0,
          maxDrawdown: values.maxDrawdown || 0.15,
          minWinRate: values.minWinRate || 0.5
        }
      };

      await startTestSuite(strategyId, 'latest', testConfig);
      setTestModalVisible(false);
    } catch (error) {
      console.error('Failed to start test suite:', error);
    }
  };

  // Handle quick backtest
  const handleQuickBacktest = async (values: any) => {
    try {
      const backtestConfig = {
        durationDays: values.quickDays || 30,
        initialCapital: values.quickCapital || 100000,
        instruments: values.quickInstruments || ['AAPL', 'MSFT']
      };

      await startBacktest(strategyCode, strategyConfig, backtestConfig);
    } catch (error) {
      console.error('Failed to start quick backtest:', error);
    }
  };

  // Handle validation test
  const handleValidation = async () => {
    try {
      await startValidationTest(strategyCode, strategyConfig);
      setValidationDrawerVisible(true);
    } catch (error) {
      console.error('Failed to start validation:', error);
    }
  };

  // Handle quick validation
  const handleQuickValidation = async () => {
    try {
      const result = await runQuickValidation(strategyCode);
      Modal.info({
        title: 'Quick Validation Result',
        content: (
          <div>
            <p><strong>Valid:</strong> {result.isValid ? 'Yes' : 'No'}</p>
            {result.errors.length > 0 && (
              <div>
                <p><strong>Errors:</strong></p>
                <ul>
                  {result.errors.map((error, index) => (
                    <li key={index} style={{ color: 'red' }}>{error}</li>
                  ))}
                </ul>
              </div>
            )}
            {result.warnings.length > 0 && (
              <div>
                <p><strong>Warnings:</strong></p>
                <ul>
                  {result.warnings.map((warning, index) => (
                    <li key={index} style={{ color: 'orange' }}>{warning}</li>
                  ))}
                </ul>
              </div>
            )}
          </div>
        )
      });
    } catch (error) {
      console.error('Failed to run quick validation:', error);
    }
  };

  // Get status color
  const getStatusColor = (status: string) => {
    switch (status) {
      case 'completed':
      case 'passed':
        return 'green';
      case 'running':
        return 'blue';
      case 'failed':
        return 'red';
      case 'pending':
        return 'orange';
      case 'cancelled':
        return 'gray';
      default:
        return 'default';
    }
  };

  // Test suite columns
  const testSuiteColumns = [
    {
      title: 'Suite ID',
      dataIndex: 'suiteId',
      key: 'suiteId',
      render: (id: string) => (
        <Button 
          type="link" 
          onClick={() => setSelectedTestSuite(id)}
        >
          {id.slice(0, 8)}...
        </Button>
      )
    },
    {
      title: 'Version',
      dataIndex: 'version',
      key: 'version',
      render: (version: string) => <Tag color="blue">{version}</Tag>
    },
    {
      title: 'Status',
      dataIndex: 'status',
      key: 'status',
      render: (status: string) => (
        <Tag color={getStatusColor(status)} icon={
          status === 'passed' ? <CheckCircleOutlined /> :
          status === 'failed' ? <CloseCircleOutlined /> :
          status === 'running' ? <PlayCircleOutlined /> :
          <ExclamationCircleOutlined />
        }>
          {status}
        </Tag>
      )
    },
    {
      title: 'Overall Score',
      dataIndex: 'overallScore',
      key: 'overallScore',
      render: (score: number) => (
        <Progress
          percent={score}
          size="small"
          strokeColor={score >= 80 ? '#52c41a' : score >= 60 ? '#faad14' : '#ff4d4f'}
        />
      )
    },
    {
      title: 'Tests',
      dataIndex: 'tests',
      key: 'tests',
      render: (tests: any[]) => `${tests.length} tests`
    },
    {
      title: 'Duration',
      dataIndex: 'totalDuration',
      key: 'totalDuration',
      render: (duration?: number) => duration ? `${Math.round(duration / 1000)}s` : '-'
    },
    {
      title: 'Started At',
      dataIndex: 'startedAt',
      key: 'startedAt',
      render: (date: Date) => new Date(date).toLocaleString()
    },
    {
      title: 'Actions',
      key: 'actions',
      render: (_: any, record: any) => (
        <Space>
          <Tooltip title="View Details">
            <Button
              size="small"
              icon={<FileTextOutlined />}
              onClick={() => setSelectedTestSuite(record.suiteId)}
            />
          </Tooltip>
          {isTestActive(record.suiteId) && (
            <Tooltip title="Stop Test">
              <Button
                size="small"
                danger
                icon={<StopOutlined />}
                onClick={() => stopTest(record.suiteId)}
              />
            </Tooltip>
          )}
        </Space>
      )
    }
  ];

  // Create equity curve chart data
  const createEquityChart = (backtestResult: any) => {
    if (!backtestResult?.timeSeries) return null;

    const data = backtestResult.timeSeries.map((point: any) => ({
      date: new Date(point.timestamp).toLocaleDateString(),
      equity: point.equity,
      drawdown: point.drawdown * 100
    }));

    return (
      <Line
        data={data}
        xField="date"
        yField="equity"
        height={300}
        smooth
        point={{
          size: 2,
          style: {
            lineWidth: 1,
            fillOpacity: 0.6,
          },
        }}
        color="#1890ff"
        xAxis={{
          type: 'timeCat',
          tickCount: 5
        }}
        yAxis={{
          title: {
            text: 'Portfolio Equity ($)'
          }
        }}
        tooltip={{
          showMarkers: true
        }}
        interactions={[{ type: 'marker-active' }]}
      />
    );
  };

  return (
    <div className="automated-testing">
      <div style={{ marginBottom: 24 }}>
        <Row gutter={[16, 16]} align="middle">
          <Col>
            <Title level={3}>
              <ExperimentOutlined /> Automated Testing
            </Title>
          </Col>
          <Col>
            <Space>
              <Button
                type="primary"
                icon={<PlayCircleOutlined />}
                onClick={() => setTestModalVisible(true)}
              >
                Run Full Test Suite
              </Button>
              <Button
                icon={<LineChartOutlined />}
                onClick={() => handleQuickValidation()}
              >
                Quick Validation
              </Button>
              <Button
                icon={<ReloadOutlined />}
                onClick={() => fetchTestSuites(strategyId)}
                loading={loading}
              >
                Refresh
              </Button>
            </Space>
          </Col>
        </Row>
      </div>

      {error && (
        <Alert
          message="Error"
          description={error}
          type="error"
          showIcon
          closable
          style={{ marginBottom: 16 }}
        />
      )}

      <Tabs activeKey={activeTab} onChange={setActiveTab}>
        <TabPane tab="Configure Tests" key="configure">
          <Row gutter={16}>
            <Col span={12}>
              <Card title="Quick Backtest" size="small">
                <Form
                  form={quickTestForm}
                  layout="vertical"
                  onFinish={handleQuickBacktest}
                  initialValues={{
                    quickDays: 30,
                    quickCapital: 100000,
                    quickInstruments: ['AAPL']
                  }}
                >
                  <Row gutter={16}>
                    <Col span={12}>
                      <Form.Item
                        name="quickDays"
                        label="Duration (Days)"
                      >
                        <InputNumber min={1} max={365} />
                      </Form.Item>
                    </Col>
                    <Col span={12}>
                      <Form.Item
                        name="quickCapital"
                        label="Initial Capital"
                      >
                        <InputNumber min={1000} formatter={value => `$ ${value}`.replace(/\B(?=(\d{3})+(?!\d))/g, ',')} />
                      </Form.Item>
                    </Col>
                  </Row>
                  
                  <Form.Item
                    name="quickInstruments"
                    label="Instruments"
                  >
                    <Select mode="tags" placeholder="Enter symbols">
                      <Option value="AAPL">AAPL</Option>
                      <Option value="MSFT">MSFT</Option>
                      <Option value="GOOGL">GOOGL</Option>
                      <Option value="TSLA">TSLA</Option>
                    </Select>
                  </Form.Item>

                  <Form.Item>
                    <Button type="primary" htmlType="submit" loading={loading}>
                      Run Quick Backtest
                    </Button>
                  </Form.Item>
                </Form>
              </Card>
            </Col>

            <Col span={12}>
              <Card title="Validation Tests" size="small">
                <Space direction="vertical" style={{ width: '100%' }}>
                  <Paragraph>
                    Run comprehensive validation tests to ensure your strategy meets quality standards:
                  </Paragraph>
                  
                  <List size="small">
                    <List.Item>
                      <CheckCircleOutlined style={{ color: 'green' }} /> Code Quality Analysis
                    </List.Item>
                    <List.Item>
                      <CheckCircleOutlined style={{ color: 'green' }} /> Risk Compliance Check
                    </List.Item>
                    <List.Item>
                      <CheckCircleOutlined style={{ color: 'green' }} /> Performance Constraints
                    </List.Item>
                    <List.Item>
                      <CheckCircleOutlined style={{ color: 'green' }} /> Dependency Validation
                    </List.Item>
                  </List>

                  <Space>
                    <Button
                      type="primary"
                      icon={<CheckCircleOutlined />}
                      onClick={handleValidation}
                      loading={loading}
                    >
                      Run Validation
                    </Button>
                    <Button
                      icon={<ExperimentOutlined />}
                      onClick={handleQuickValidation}
                    >
                      Quick Check
                    </Button>
                  </Space>
                </Space>
              </Card>
            </Col>
          </Row>
        </TabPane>

        <TabPane tab="Test History" key="history">
          <Card title="Test Suite History">
            <Table
              dataSource={testSuites}
              columns={testSuiteColumns}
              loading={loading}
              rowKey="suiteId"
              pagination={{ pageSize: 20 }}
            />
          </Card>
        </TabPane>

        <TabPane tab="Results Analysis" key="results">
          {Object.keys(backtestResults).length > 0 ? (
            <Row gutter={16}>
              <Col span={8}>
                <Card title="Backtest Results" size="small">
                  <List
                    size="small"
                    dataSource={Object.values(backtestResults)}
                    renderItem={(result: any) => (
                      <List.Item>
                        <Button
                          type="link"
                          onClick={() => {
                            setSelectedBacktest(result.testId);
                            setResultsDrawerVisible(true);
                          }}
                        >
                          {result.testId.slice(0, 8)}... - {result.summary?.totalReturn || 0}% Return
                        </Button>
                      </List.Item>
                    )}
                  />
                </Card>
              </Col>
              <Col span={16}>
                {selectedBacktest && backtestResults[selectedBacktest] && (
                  <Card title="Performance Overview">
                    {createEquityChart(backtestResults[selectedBacktest])}
                  </Card>
                )}
              </Col>
            </Row>
          ) : (
            <Result
              icon={<LineChartOutlined />}
              title="No Test Results Yet"
              subTitle="Run a backtest to see performance analysis here"
              extra={
                <Button type="primary" onClick={() => setTestModalVisible(true)}>
                  Run Test Suite
                </Button>
              }
            />
          )}
        </TabPane>

        <TabPane tab="Performance Metrics" key="metrics">
          <Row gutter={16}>
            {Object.values(backtestResults).map((result: any) => (
              <Col span={8} key={result.testId}>
                <Card 
                  title={`Test ${result.testId.slice(0, 8)}...`} 
                  size="small"
                  style={{ marginBottom: 16 }}
                >
                  <Row gutter={8}>
                    <Col span={12}>
                      <Statistic
                        title="Total Return"
                        value={result.summary?.totalReturn || 0}
                        suffix="%"
                        precision={2}
                        valueStyle={{ 
                          color: (result.summary?.totalReturn || 0) >= 0 ? '#3f8600' : '#cf1322' 
                        }}
                      />
                    </Col>
                    <Col span={12}>
                      <Statistic
                        title="Sharpe Ratio"
                        value={result.summary?.sharpeRatio || 0}
                        precision={2}
                        valueStyle={{ 
                          color: (result.summary?.sharpeRatio || 0) >= 1 ? '#3f8600' : '#faad14' 
                        }}
                      />
                    </Col>
                  </Row>
                  <Divider />
                  <Row gutter={8}>
                    <Col span={12}>
                      <Statistic
                        title="Max Drawdown"
                        value={Math.abs(result.summary?.maxDrawdown || 0)}
                        suffix="%"
                        precision={2}
                        valueStyle={{ color: '#cf1322' }}
                      />
                    </Col>
                    <Col span={12}>
                      <Statistic
                        title="Win Rate"
                        value={(result.summary?.winRate || 0) * 100}
                        suffix="%"
                        precision={1}
                        valueStyle={{ 
                          color: (result.summary?.winRate || 0) >= 0.5 ? '#3f8600' : '#faad14' 
                        }}
                      />
                    </Col>
                  </Row>
                </Card>
              </Col>
            ))}
          </Row>
        </TabPane>
      </Tabs>

      {/* Full Test Suite Modal */}
      <Modal
        title="Configure Full Test Suite"
        open={testModalVisible}
        onCancel={() => setTestModalVisible(false)}
        footer={null}
        width={700}
      >
        <Form
          form={testConfigForm}
          layout="vertical"
          onFinish={handleFullTestSuite}
          initialValues={{
            backtestDays: 90,
            initialCapital: 100000,
            instruments: ['AAPL', 'MSFT'],
            paperTradingMinutes: 30,
            maxPositions: 5,
            volatilityMultiplier: 2.0,
            minSharpeRatio: 1.0,
            maxDrawdown: 0.15,
            minWinRate: 0.5
          }}
        >
          <Title level={4}>Backtest Configuration</Title>
          <Row gutter={16}>
            <Col span={8}>
              <Form.Item name="backtestDays" label="Duration (Days)">
                <InputNumber min={30} max={365} />
              </Form.Item>
            </Col>
            <Col span={8}>
              <Form.Item name="initialCapital" label="Initial Capital">
                <InputNumber 
                  min={10000} 
                  formatter={value => `$ ${value}`.replace(/\B(?=(\d{3})+(?!\d))/g, ',')}
                />
              </Form.Item>
            </Col>
            <Col span={8}>
              <Form.Item name="instruments" label="Instruments">
                <Select mode="multiple" placeholder="Select instruments">
                  <Option value="AAPL">AAPL</Option>
                  <Option value="MSFT">MSFT</Option>
                  <Option value="GOOGL">GOOGL</Option>
                  <Option value="TSLA">TSLA</Option>
                </Select>
              </Form.Item>
            </Col>
          </Row>

          <Divider />
          <Title level={4}>Paper Trading Test</Title>
          <Row gutter={16}>
            <Col span={12}>
              <Form.Item name="paperTradingMinutes" label="Duration (Minutes)">
                <InputNumber min={10} max={120} />
              </Form.Item>
            </Col>
            <Col span={12}>
              <Form.Item name="maxPositions" label="Max Positions">
                <InputNumber min={1} max={20} />
              </Form.Item>
            </Col>
          </Row>

          <Divider />
          <Title level={4}>Performance Targets</Title>
          <Row gutter={16}>
            <Col span={8}>
              <Form.Item name="minSharpeRatio" label="Min Sharpe Ratio">
                <InputNumber min={0} max={5} step={0.1} />
              </Form.Item>
            </Col>
            <Col span={8}>
              <Form.Item name="maxDrawdown" label="Max Drawdown">
                <InputNumber min={0} max={1} step={0.01} />
              </Form.Item>
            </Col>
            <Col span={8}>
              <Form.Item name="minWinRate" label="Min Win Rate">
                <InputNumber min={0} max={1} step={0.01} />
              </Form.Item>
            </Col>
          </Row>

          <Form.Item>
            <Space>
              <Button type="primary" htmlType="submit" loading={loading}>
                Start Test Suite
              </Button>
              <Button onClick={() => setTestModalVisible(false)}>
                Cancel
              </Button>
            </Space>
          </Form.Item>
        </Form>
      </Modal>

      {/* Results Drawer */}
      <Drawer
        title="Backtest Results"
        placement="right"
        width={800}
        open={resultsDrawerVisible}
        onClose={() => setResultsDrawerVisible(false)}
      >
        {selectedBacktest && backtestResults[selectedBacktest] && (
          <div>
            <Row gutter={16} style={{ marginBottom: 24 }}>
              <Col span={6}>
                <Statistic
                  title="Total Return"
                  value={backtestResults[selectedBacktest].summary?.totalReturn || 0}
                  suffix="%"
                  precision={2}
                  valueStyle={{ 
                    color: (backtestResults[selectedBacktest].summary?.totalReturn || 0) >= 0 ? '#3f8600' : '#cf1322' 
                  }}
                />
              </Col>
              <Col span={6}>
                <Statistic
                  title="Sharpe Ratio"
                  value={backtestResults[selectedBacktest].summary?.sharpeRatio || 0}
                  precision={2}
                />
              </Col>
              <Col span={6}>
                <Statistic
                  title="Max Drawdown"
                  value={Math.abs(backtestResults[selectedBacktest].summary?.maxDrawdown || 0)}
                  suffix="%"
                  precision={2}
                  valueStyle={{ color: '#cf1322' }}
                />
              </Col>
              <Col span={6}>
                <Statistic
                  title="Total Trades"
                  value={backtestResults[selectedBacktest].summary?.totalTrades || 0}
                />
              </Col>
            </Row>

            <Card title="Equity Curve" style={{ marginBottom: 24 }}>
              {createEquityChart(backtestResults[selectedBacktest])}
            </Card>

            <Card title="Risk Metrics">
              <Row gutter={16}>
                <Col span={8}>
                  <Statistic
                    title="VaR (95%)"
                    value={backtestResults[selectedBacktest].riskMetrics?.var95 || 0}
                    suffix="%"
                    precision={2}
                  />
                </Col>
                <Col span={8}>
                  <Statistic
                    title="Expected Shortfall"
                    value={backtestResults[selectedBacktest].riskMetrics?.expectedShortfall || 0}
                    suffix="%"
                    precision={2}
                  />
                </Col>
                <Col span={8}>
                  <Statistic
                    title="Beta"
                    value={backtestResults[selectedBacktest].riskMetrics?.beta || 0}
                    precision={3}
                  />
                </Col>
              </Row>
            </Card>
          </div>
        )}
      </Drawer>
    </div>
  );
};