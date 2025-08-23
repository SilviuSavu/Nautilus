import React, { useState, useEffect, useCallback } from 'react';
import {
  Card,
  Button,
  Table,
  Progress,
  Alert,
  Tag,
  Space,
  Typography,
  Row,
  Col,
  Statistic,
  Timeline,
  Tabs,
  List,
  Switch,
  Select,
  InputNumber,
  Form,
  Spin,
  Badge,
  Tooltip,
  Modal,
  notification
} from 'antd';
import {
  PlayCircleOutlined,
  StopOutlined,
  CheckCircleOutlined,
  ExclamationCircleOutlined,
  BugOutlined,
  LineChartOutlined,
  SecurityScanOutlined,
  ThunderboltOutlined,
  ReloadOutlined,
  SettingOutlined,
  FileTextOutlined,
  DownloadOutlined,
  EyeOutlined
} from '@ant-design/icons';
import type { ColumnType } from 'antd/es/table';
import type {
  AutomatedTestingSuiteProps,
  EnhancedTestSuite,
  TestCase,
  TestSuiteResults,
  TestResult,
  ExecuteTestSuiteRequest
} from './types/deploymentTypes';

const { Title, Text, Paragraph } = Typography;
const { Option } = Select;
const { TabPane } = Tabs;

const API_BASE = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8001';

const AutomatedTestingSuite: React.FC<AutomatedTestingSuiteProps> = ({
  strategyId,
  testSuite,
  environment = 'development',
  onTestComplete
}) => {
  const [form] = Form.useForm();
  const [loading, setLoading] = useState(false);
  const [executing, setExecuting] = useState(false);
  const [currentTestSuite, setCurrentTestSuite] = useState<EnhancedTestSuite | null>(testSuite || null);
  const [testResults, setTestResults] = useState<TestSuiteResults | null>(null);
  const [executionProgress, setExecutionProgress] = useState<number>(0);
  const [currentTest, setCurrentTest] = useState<string>('');
  const [showConfigModal, setShowConfigModal] = useState(false);
  const [selectedTest, setSelectedTest] = useState<TestResult | null>(null);
  const [showDetailModal, setShowDetailModal] = useState(false);

  // Default test suite configuration
  const defaultTestSuite: EnhancedTestSuite = {
    suite_id: `suite_${strategyId}_${Date.now()}`,
    name: `${strategyId} Automated Test Suite`,
    version: '1.0.0',
    execution_settings: {
      parallel_execution: false,
      max_parallel_tests: 3,
      retry_failed_tests: true,
      max_retries: 2,
      fail_fast: false,
      generate_reports: true
    },
    tests: [
      {
        id: 'syntax_validation',
        name: 'Syntax Validation',
        type: 'syntax',
        description: 'Validate Python syntax and imports',
        enabled: true,
        timeout_seconds: 60,
        test_config: {
          check_imports: true,
          check_syntax: true,
          check_nautilus_compatibility: true
        }
      },
      {
        id: 'unit_tests',
        name: 'Unit Tests',
        type: 'unit',
        description: 'Run strategy unit tests',
        enabled: true,
        timeout_seconds: 300,
        test_config: {
          test_patterns: ['test_*.py', '*_test.py'],
          coverage_threshold: 0.8,
          include_performance_tests: true
        }
      },
      {
        id: 'integration_tests',
        name: 'Integration Tests',
        type: 'integration',
        description: 'Test integration with NautilusTrader',
        enabled: true,
        timeout_seconds: 600,
        test_config: {
          mock_data: true,
          test_venues: ['SIM'],
          test_instruments: ['EUR/USD', 'BTC/USD']
        }
      },
      {
        id: 'performance_tests',
        name: 'Performance Tests',
        type: 'performance',
        description: 'Validate performance characteristics',
        enabled: true,
        timeout_seconds: 900,
        test_config: {
          max_memory_mb: 512,
          max_cpu_percent: 80,
          max_latency_ms: 100,
          duration_minutes: 10
        },
        success_criteria: {
          max_memory_usage: 512,
          max_cpu_usage: 80,
          max_avg_latency: 100
        }
      },
      {
        id: 'risk_validation',
        name: 'Risk Validation',
        type: 'risk',
        description: 'Validate risk management controls',
        enabled: true,
        timeout_seconds: 300,
        test_config: {
          test_position_limits: true,
          test_stop_loss: true,
          test_max_drawdown: true,
          simulate_adverse_conditions: true
        },
        success_criteria: {
          max_position_breach: 0,
          stop_loss_effectiveness: 0.95,
          max_drawdown_compliance: 1.0
        }
      },
      {
        id: 'regression_tests',
        name: 'Regression Tests',
        type: 'regression',
        description: 'Compare against historical performance',
        enabled: false,
        timeout_seconds: 1800,
        test_config: {
          baseline_version: 'latest_stable',
          comparison_period: '3M',
          tolerance_percent: 5,
          key_metrics: ['pnl', 'sharpe_ratio', 'max_drawdown', 'win_rate']
        }
      }
    ]
  };

  useEffect(() => {
    if (!currentTestSuite) {
      setCurrentTestSuite(defaultTestSuite);
    }
    
    // Initialize form with default values
    form.setFieldsValue({
      parallel_execution: defaultTestSuite.execution_settings.parallel_execution,
      max_parallel_tests: defaultTestSuite.execution_settings.max_parallel_tests,
      retry_failed_tests: defaultTestSuite.execution_settings.retry_failed_tests,
      max_retries: defaultTestSuite.execution_settings.max_retries,
      fail_fast: defaultTestSuite.execution_settings.fail_fast,
      generate_reports: defaultTestSuite.execution_settings.generate_reports
    });
  }, [currentTestSuite, form]);

  const executeTestSuite = useCallback(async () => {
    if (!currentTestSuite) return;
    
    setExecuting(true);
    setExecutionProgress(0);
    setCurrentTest('');
    setTestResults(null);
    
    try {
      const request: ExecuteTestSuiteRequest = {
        strategy_id: strategyId,
        test_suite: currentTestSuite,
        environment
      };
      
      const response = await fetch(`${API_BASE}/api/v1/strategies/test/${strategyId}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(request)
      });
      
      if (!response.ok) {
        throw new Error(`Failed to start test execution: ${response.statusText}`);
      }
      
      const { execution_id } = await response.json();
      
      // Poll for test results
      const pollInterval = setInterval(async () => {
        try {
          const statusResponse = await fetch(`${API_BASE}/api/v1/strategies/test/${execution_id}/status`);
          const statusData = await statusResponse.json();
          
          setExecutionProgress(statusData.progress || 0);
          setCurrentTest(statusData.current_test || '');
          
          if (statusData.completed) {
            clearInterval(pollInterval);
            const resultsResponse = await fetch(`${API_BASE}/api/v1/strategies/test/${execution_id}/results`);
            const results: TestSuiteResults = await resultsResponse.json();
            
            setTestResults(results);
            setExecuting(false);
            setExecutionProgress(100);
            setCurrentTest('');
            
            onTestComplete?.(results);
            
            notification.success({
              message: 'Test Suite Completed',
              description: `${results.passed_tests}/${results.total_tests} tests passed`
            });
          } else if (statusData.failed) {
            clearInterval(pollInterval);
            setExecuting(false);
            notification.error({
              message: 'Test Suite Failed',
              description: statusData.error || 'Unknown error occurred'
            });
          }
        } catch (error) {
          console.error('Error polling test status:', error);
        }
      }, 2000);
      
    } catch (error) {
      console.error('Error executing test suite:', error);
      setExecuting(false);
      notification.error({
        message: 'Test Execution Failed',
        description: error instanceof Error ? error.message : 'Unknown error'
      });
    }
  }, [currentTestSuite, strategyId, environment, onTestComplete]);

  const stopExecution = async () => {
    setExecuting(false);
    setExecutionProgress(0);
    setCurrentTest('');
    notification.info({
      message: 'Test Execution Stopped',
      description: 'Test execution has been stopped'
    });
  };

  const updateTestSuiteSettings = (values: any) => {
    if (!currentTestSuite) return;
    
    const updatedSuite: EnhancedTestSuite = {
      ...currentTestSuite,
      execution_settings: {
        ...currentTestSuite.execution_settings,
        ...values
      }
    };
    
    setCurrentTestSuite(updatedSuite);
    setShowConfigModal(false);
    
    notification.success({
      message: 'Settings Updated',
      description: 'Test suite settings have been updated'
    });
  };

  const toggleTestEnabled = (testId: string, enabled: boolean) => {
    if (!currentTestSuite) return;
    
    const updatedTests = currentTestSuite.tests.map(test =>
      test.id === testId ? { ...test, enabled } : test
    );
    
    setCurrentTestSuite({
      ...currentTestSuite,
      tests: updatedTests
    });
  };

  const getTestStatusIcon = (result: TestResult) => {
    switch (result.status) {
      case 'passed': return <CheckCircleOutlined style={{ color: '#52c41a' }} />;
      case 'failed': return <ExclamationCircleOutlined style={{ color: '#ff4d4f' }} />;
      case 'skipped': return <StopOutlined style={{ color: '#faad14' }} />;
      case 'error': return <BugOutlined style={{ color: '#ff4d4f' }} />;
      default: return <CheckCircleOutlined style={{ color: '#d9d9d9' }} />;
    }
  };

  const getTestTypeIcon = (type: string) => {
    const iconProps = { style: { fontSize: '16px' } };
    switch (type) {
      case 'syntax': return <FileTextOutlined {...iconProps} />;
      case 'unit': return <BugOutlined {...iconProps} />;
      case 'integration': return <LineChartOutlined {...iconProps} />;
      case 'performance': return <ThunderboltOutlined {...iconProps} />;
      case 'risk': return <SecurityScanOutlined {...iconProps} />;
      case 'regression': return <ReloadOutlined {...iconProps} />;
      default: return <CheckCircleOutlined {...iconProps} />;
    }
  };

  const testColumns: ColumnType<TestCase>[] = [
    {
      title: 'Test',
      key: 'test',
      render: (_, test) => (
        <Space>
          {getTestTypeIcon(test.type)}
          <div>
            <Text strong>{test.name}</Text>
            <br />
            <Text type="secondary" style={{ fontSize: '12px' }}>
              {test.description}
            </Text>
          </div>
        </Space>
      )
    },
    {
      title: 'Type',
      dataIndex: 'type',
      key: 'type',
      width: 100,
      render: (type: string) => (
        <Tag color={
          type === 'syntax' ? 'blue' :
          type === 'unit' ? 'green' :
          type === 'integration' ? 'orange' :
          type === 'performance' ? 'red' :
          type === 'risk' ? 'purple' : 'default'
        }>
          {type.toUpperCase()}
        </Tag>
      )
    },
    {
      title: 'Status',
      key: 'status',
      width: 100,
      render: (_, test) => {
        if (!testResults) return <Tag>Not Run</Tag>;
        
        const result = testResults.test_results.find(r => r.test_id === test.id);
        if (!result) return <Tag>Not Run</Tag>;
        
        return (
          <Space>
            {getTestStatusIcon(result)}
            <Tag color={
              result.status === 'passed' ? 'success' :
              result.status === 'failed' ? 'error' :
              result.status === 'skipped' ? 'warning' : 'default'
            }>
              {result.status.toUpperCase()}
            </Tag>
          </Space>
        );
      }
    },
    {
      title: 'Duration',
      key: 'duration',
      width: 100,
      render: (_, test) => {
        if (!testResults) return '-';
        
        const result = testResults.test_results.find(r => r.test_id === test.id);
        if (!result) return '-';
        
        return `${result.execution_time_seconds.toFixed(1)}s`;
      }
    },
    {
      title: 'Enabled',
      key: 'enabled',
      width: 80,
      render: (_, test) => (
        <Switch
          checked={test.enabled}
          onChange={(checked) => toggleTestEnabled(test.id, checked)}
          disabled={executing}
        />
      )
    },
    {
      title: 'Actions',
      key: 'actions',
      width: 80,
      render: (_, test) => {
        const result = testResults?.test_results.find(r => r.test_id === test.id);
        return (
          <Space size="small">
            {result && (
              <Tooltip title="View Details">
                <Button
                  size="small"
                  icon={<EyeOutlined />}
                  onClick={() => {
                    setSelectedTest(result);
                    setShowDetailModal(true);
                  }}
                />
              </Tooltip>
            )}
          </Space>
        );
      }
    }
  ];

  const renderExecutionProgress = () => {
    if (!executing && !testResults) return null;
    
    return (
      <Card title="Execution Progress" className="mb-4">
        <Row gutter={16}>
          <Col span={12}>
            <Progress
              type="circle"
              percent={executionProgress}
              format={() => `${Math.round(executionProgress)}%`}
              status={executing ? 'active' : 'normal'}
              width={100}
            />
          </Col>
          <Col span={12}>
            <Space direction="vertical" style={{ width: '100%' }}>
              <Text strong>Current Test:</Text>
              <Text>{currentTest || 'Preparing...'}</Text>
              
              {testResults && (
                <>
                  <Statistic
                    title="Tests Passed"
                    value={testResults.passed_tests}
                    suffix={`/ ${testResults.total_tests}`}
                    valueStyle={{ color: '#3f8600' }}
                  />
                  <Statistic
                    title="Execution Time"
                    value={testResults.execution_time_seconds}
                    suffix="seconds"
                  />
                </>
              )}
            </Space>
          </Col>
        </Row>
        
        {executing && (
          <div className="mt-4">
            <Progress
              percent={executionProgress}
              status="active"
              showInfo={false}
            />
          </div>
        )}
      </Card>
    );
  };

  const renderTestResults = () => {
    if (!testResults) return null;
    
    return (
      <Card title="Test Results Summary" className="mb-4">
        <Row gutter={16}>
          <Col span={6}>
            <Statistic
              title="Total Tests"
              value={testResults.total_tests}
              prefix={<FileTextOutlined />}
            />
          </Col>
          <Col span={6}>
            <Statistic
              title="Passed"
              value={testResults.passed_tests}
              valueStyle={{ color: '#3f8600' }}
              prefix={<CheckCircleOutlined />}
            />
          </Col>
          <Col span={6}>
            <Statistic
              title="Failed"
              value={testResults.failed_tests}
              valueStyle={{ color: '#cf1322' }}
              prefix={<ExclamationCircleOutlined />}
            />
          </Col>
          <Col span={6}>
            <Statistic
              title="Skipped"
              value={testResults.skipped_tests}
              valueStyle={{ color: '#faad14' }}
              prefix={<StopOutlined />}
            />
          </Col>
        </Row>
        
        <div className="mt-4">
          <Alert
            message={`Test Suite ${testResults.summary.overall_status.toUpperCase()}`}
            description={
              testResults.summary.overall_status === 'passed'
                ? 'All critical tests passed successfully'
                : `${testResults.summary.critical_failures} critical failures, ${testResults.summary.warnings} warnings`
            }
            type={
              testResults.summary.overall_status === 'passed' ? 'success' :
              testResults.summary.overall_status === 'failed' ? 'error' : 'warning'
            }
            showIcon
          />
        </div>
      </Card>
    );
  };

  return (
    <div className="automated-testing-suite">
      <Card
        title={
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-2">
              <BugOutlined />
              <span>Automated Testing Suite</span>
              <Tag>{environment.toUpperCase()}</Tag>
            </div>
            <Space>
              <Button
                icon={<SettingOutlined />}
                onClick={() => setShowConfigModal(true)}
                disabled={executing}
              >
                Configure
              </Button>
              {!executing ? (
                <Button
                  type="primary"
                  icon={<PlayCircleOutlined />}
                  onClick={executeTestSuite}
                  disabled={!currentTestSuite || currentTestSuite.tests.filter(t => t.enabled).length === 0}
                >
                  Run Tests
                </Button>
              ) : (
                <Button
                  danger
                  icon={<StopOutlined />}
                  onClick={stopExecution}
                >
                  Stop
                </Button>
              )}
            </Space>
          </div>
        }
      >
        {renderExecutionProgress()}
        {renderTestResults()}
        
        <Table
          columns={testColumns}
          dataSource={currentTestSuite?.tests || []}
          rowKey="id"
          size="small"
          pagination={false}
        />
      </Card>

      {/* Configuration Modal */}
      <Modal
        title="Test Suite Configuration"
        open={showConfigModal}
        onCancel={() => setShowConfigModal(false)}
        onOk={() => form.submit()}
        width={600}
      >
        <Form
          form={form}
          layout="vertical"
          onFinish={updateTestSuiteSettings}
        >
          <Row gutter={16}>
            <Col span={12}>
              <Form.Item
                label="Parallel Execution"
                name="parallel_execution"
                valuePropName="checked"
              >
                <Switch />
              </Form.Item>
            </Col>
            <Col span={12}>
              <Form.Item
                label="Max Parallel Tests"
                name="max_parallel_tests"
              >
                <InputNumber min={1} max={10} />
              </Form.Item>
            </Col>
          </Row>
          
          <Row gutter={16}>
            <Col span={12}>
              <Form.Item
                label="Retry Failed Tests"
                name="retry_failed_tests"
                valuePropName="checked"
              >
                <Switch />
              </Form.Item>
            </Col>
            <Col span={12}>
              <Form.Item
                label="Max Retries"
                name="max_retries"
              >
                <InputNumber min={1} max={5} />
              </Form.Item>
            </Col>
          </Row>
          
          <Row gutter={16}>
            <Col span={12}>
              <Form.Item
                label="Fail Fast"
                name="fail_fast"
                valuePropName="checked"
                tooltip="Stop execution on first failure"
              >
                <Switch />
              </Form.Item>
            </Col>
            <Col span={12}>
              <Form.Item
                label="Generate Reports"
                name="generate_reports"
                valuePropName="checked"
              >
                <Switch />
              </Form.Item>
            </Col>
          </Row>
        </Form>
      </Modal>

      {/* Test Detail Modal */}
      <Modal
        title={`Test Details: ${selectedTest?.test_id}`}
        open={showDetailModal}
        onCancel={() => setShowDetailModal(false)}
        footer={null}
        width={800}
      >
        {selectedTest && (
          <div>
            <Row gutter={16} className="mb-4">
              <Col span={8}>
                <Statistic
                  title="Status"
                  value={selectedTest.status.toUpperCase()}
                  valueStyle={{
                    color: selectedTest.status === 'passed' ? '#3f8600' : '#cf1322'
                  }}
                />
              </Col>
              <Col span={8}>
                <Statistic
                  title="Duration"
                  value={selectedTest.execution_time_seconds}
                  suffix="seconds"
                />
              </Col>
              <Col span={8}>
                {selectedTest.artifacts && selectedTest.artifacts.length > 0 && (
                  <Button icon={<DownloadOutlined />}>
                    Download Artifacts ({selectedTest.artifacts.length})
                  </Button>
                )}
              </Col>
            </Row>
            
            {selectedTest.error_message && (
              <Alert
                message="Error"
                description={selectedTest.error_message}
                type="error"
                className="mb-4"
              />
            )}
            
            {selectedTest.details && (
              <Card title="Details" size="small">
                <pre style={{ whiteSpace: 'pre-wrap', fontSize: '12px' }}>
                  {JSON.stringify(selectedTest.details, null, 2)}
                </pre>
              </Card>
            )}
          </div>
        )}
      </Modal>
    </div>
  );
};

export default AutomatedTestingSuite;