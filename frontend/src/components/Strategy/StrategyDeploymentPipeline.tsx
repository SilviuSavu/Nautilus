import React, { useState, useEffect } from 'react';
import {
  Card,
  Steps,
  Button,
  Form,
  Input,
  Select,
  InputNumber,
  Switch,
  Descriptions,
  Alert,
  Progress,
  Modal,
  Space,
  Divider,
  Tag,
  Typography,
  Row,
  Col,
  Tabs,
  Table,
  Spin,
  Popconfirm,
  message
} from 'antd';
import {
  PlayCircleOutlined,
  StopOutlined,
  CheckCircleOutlined,
  ExclamationCircleOutlined,
  ClockCircleOutlined,
  SettingOutlined,
  SecurityScanOutlined,
  DeploymentUnitOutlined,
  MonitorOutlined
} from '@ant-design/icons';
import type {
  DeploymentRequest,
  CreateDeploymentRequest,
  StrategyDeployment,
  RolloutPlan,
  RolloutPhase,
  DeploymentStatus,
  DeploymentPipelineProps,
  BacktestResults,
  RiskAssessment
} from '../../types/deployment';
import type { StrategyConfig } from './types/strategyTypes';

const { Step } = Steps;
const { Option } = Select;
const { Text, Title } = Typography;
const { TabPane } = Tabs;

interface DeploymentFormData {
  strategyId: string;
  version: string;
  environment: 'production' | 'staging' | 'development';
  approvalRequired: boolean;
  requiredApprovals: string[];
  initialPositionSize: number;
  maxDailyLoss: number;
  maxPositions: number;
  enableGradualRollout: boolean;
  rolloutPhases: RolloutPhase[];
}

const StrategyDeploymentPipeline: React.FC<DeploymentPipelineProps> = ({
  strategyId,
  onDeploymentCreated,
  onClose
}) => {
  const [currentStep, setCurrentStep] = useState(0);
  const [deploymentForm] = Form.useForm();
  const [loading, setLoading] = useState(false);
  const [deployment, setDeployment] = useState<StrategyDeployment | null>(null);
  const [strategy, setStrategy] = useState<StrategyConfig | null>(null);
  const [backtestResults, setBacktestResults] = useState<BacktestResults | null>(null);
  const [riskAssessment, setRiskAssessment] = useState<RiskAssessment | null>(null);
  const [validationErrors, setValidationErrors] = useState<string[]>([]);

  useEffect(() => {
    if (strategyId) {
      loadStrategyData(strategyId);
    }
  }, [strategyId]);

  const loadStrategyData = async (strategyId: string) => {
    setLoading(true);
    try {
      // Load strategy configuration
      const strategyResponse = await fetch(`/api/v1/strategies/${strategyId}`);
      const strategyData = await strategyResponse.json();
      setStrategy(strategyData);

      // Load latest backtest results if available
      const backtestResponse = await fetch(`/api/v1/strategies/${strategyId}/backtests/latest`);
      if (backtestResponse.ok) {
        const backtestData = await backtestResponse.json();
        setBacktestResults(backtestData);
      }

      // Pre-populate form with strategy data
      deploymentForm.setFieldsValue({
        strategyId: strategyData.id,
        version: `${strategyData.version}.0`,
        environment: 'production',
        approvalRequired: true,
        requiredApprovals: ['senior_trader', 'risk_manager'],
        initialPositionSize: 25,
        maxDailyLoss: 1000,
        maxPositions: 5,
        enableGradualRollout: true,
        rolloutPhases: [
          {
            name: 'validation',
            positionSizePercent: 25,
            duration: 7200,
            successCriteria: { minTrades: 5, maxDrawdown: 0.03, pnlThreshold: -0.01 }
          },
          {
            name: 'scaling',
            positionSizePercent: 50,
            duration: 14400,
            successCriteria: { minTrades: 10, maxDrawdown: 0.05, pnlThreshold: -0.02 }
          },
          {
            name: 'full_deployment',
            positionSizePercent: 100,
            duration: -1,
            successCriteria: { ongoing: true }
          }
        ]
      });
    } catch (error) {
      console.error('Error loading strategy data:', error);
      message.error('Failed to load strategy data');
    } finally {
      setLoading(false);
    }
  };

  const performRiskAssessment = async (formData: DeploymentFormData) => {
    setLoading(true);
    try {
      const response = await fetch('/api/v1/nautilus/deployment/risk-assessment', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          strategyId: formData.strategyId,
          proposedConfig: formData,
          backtestResults
        })
      });

      const riskData = await response.json();
      setRiskAssessment(riskData);

      if (riskData.blockers?.length > 0) {
        setValidationErrors(riskData.blockers);
        return false;
      }

      setCurrentStep(2);
      return true;
    } catch (error) {
      console.error('Risk assessment failed:', error);
      message.error('Risk assessment failed');
      return false;
    } finally {
      setLoading(false);
    }
  };

  const submitDeploymentRequest = async () => {
    setLoading(true);
    try {
      const formData = deploymentForm.getFieldsValue() as DeploymentFormData;
      
      const deploymentRequest: CreateDeploymentRequest = {
        strategyId: formData.strategyId,
        version: formData.version,
        proposedConfig: {
          ...strategy!,
          riskEngine: {
            enabled: true,
            maxOrderSize: 100000,
            maxNotionalPerOrder: 50000,
            maxDailyLoss: formData.maxDailyLoss,
            positionLimits: {
              maxPositions: formData.maxPositions,
              maxPositionSize: 25000
            }
          },
          venues: [{
            name: 'INTERACTIVE_BROKERS',
            venueType: 'ECN',
            accountId: 'DU123456',
            routing: 'SMART',
            clientId: '${IB_CLIENT_ID}',
            gatewayHost: 'localhost',
            gatewayPort: 7497
          }],
          dataEngine: {
            timeBarsTimestampOnClose: true,
            validateDataSequence: true,
            bufferDeltas: true
          },
          execEngine: {
            reconciliation: true,
            inflightCheckIntervalMs: 5000,
            snapshotOrders: true,
            snapshotPositions: true
          },
          environment: {
            containerName: 'nautilus-backend',
            databaseUrl: '${DATABASE_URL}',
            redisUrl: '${REDIS_URL}',
            deploymentId: 'deployment_uuid_here',
            monitoringEnabled: true,
            loggingLevel: 'INFO'
          }
        },
        rolloutPlan: {
          phases: formData.rolloutPhases,
          currentPhase: 0,
          escalationCriteria: {
            maxLossPercentage: 0.05,
            consecutiveLosses: 5,
            correlationThreshold: 0.8
          }
        },
        riskAssessment
      };

      const response = await fetch('/api/v1/nautilus/deployment/create', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(deploymentRequest)
      });

      const result = await response.json();
      
      if (result.deploymentId) {
        setDeployment({
          deploymentId: result.deploymentId,
          strategyId: formData.strategyId,
          version: formData.version,
          deploymentConfig: deploymentRequest.proposedConfig,
          rolloutPlan: deploymentRequest.rolloutPlan!,
          status: 'pending_approval',
          createdBy: 'current_user',
          createdAt: new Date(),
          approvalChain: []
        });
        setCurrentStep(3);
        onDeploymentCreated?.(result.deploymentId);
        message.success('Deployment request created successfully');
      }
    } catch (error) {
      console.error('Failed to create deployment request:', error);
      message.error('Failed to create deployment request');
    } finally {
      setLoading(false);
    }
  };

  const renderConfigurationStep = () => (
    <Card title="Deployment Configuration" className="mb-4">
      <Form form={deploymentForm} layout="vertical">
        <Row gutter={16}>
          <Col span={12}>
            <Form.Item label="Strategy Version" name="version" required>
              <Input placeholder="e.g., 2.1.0" />
            </Form.Item>
          </Col>
          <Col span={12}>
            <Form.Item label="Environment" name="environment" required>
              <Select>
                <Option value="production">Production</Option>
                <Option value="staging">Staging</Option>
                <Option value="development">Development</Option>
              </Select>
            </Form.Item>
          </Col>
        </Row>

        <Row gutter={16}>
          <Col span={12}>
            <Form.Item label="Initial Position Size %" name="initialPositionSize">
              <InputNumber min={1} max={100} />
            </Form.Item>
          </Col>
          <Col span={12}>
            <Form.Item label="Max Daily Loss ($)" name="maxDailyLoss">
              <InputNumber min={100} max={10000} />
            </Form.Item>
          </Col>
        </Row>

        <Row gutter={16}>
          <Col span={12}>
            <Form.Item label="Max Positions" name="maxPositions">
              <InputNumber min={1} max={20} />
            </Form.Item>
          </Col>
          <Col span={12}>
            <Form.Item label="Enable Gradual Rollout" name="enableGradualRollout" valuePropName="checked">
              <Switch />
            </Form.Item>
          </Col>
        </Row>

        <Divider>Approval Settings</Divider>
        
        <Form.Item label="Approval Required" name="approvalRequired" valuePropName="checked">
          <Switch />
        </Form.Item>

        <Form.Item label="Required Approvals" name="requiredApprovals">
          <Select mode="multiple" placeholder="Select required approvers">
            <Option value="senior_trader">Senior Trader</Option>
            <Option value="risk_manager">Risk Manager</Option>
            <Option value="compliance">Compliance</Option>
            <Option value="head_of_trading">Head of Trading</Option>
          </Select>
        </Form.Item>
      </Form>
    </Card>
  );

  const renderRiskAssessmentStep = () => (
    <Card title="Risk Assessment" className="mb-4">
      {loading ? (
        <div className="text-center py-8">
          <Spin size="large" />
          <p>Performing risk assessment...</p>
        </div>
      ) : riskAssessment ? (
        <>
          <Alert
            message="Risk Assessment Complete"
            type={riskAssessment.risk_level === 'low' ? 'success' : 
                  riskAssessment.risk_level === 'medium' ? 'warning' : 'error'}
            className="mb-4"
          />
          
          <Descriptions bordered>
            <Descriptions.Item label="Risk Level" span={3}>
              <Tag color={riskAssessment.risk_level === 'low' ? 'green' : 
                         riskAssessment.risk_level === 'medium' ? 'orange' : 'red'}>
                {riskAssessment.risk_level.toUpperCase()}
              </Tag>
            </Descriptions.Item>
            <Descriptions.Item label="Portfolio Impact" span={3}>
              {riskAssessment.portfolioImpact}
            </Descriptions.Item>
            <Descriptions.Item label="Correlation Risk" span={3}>
              {riskAssessment.correlationRisk}
            </Descriptions.Item>
            <Descriptions.Item label="Max Drawdown Estimate" span={3}>
              {(riskAssessment.maxDrawdownEstimate * 100).toFixed(2)}%
            </Descriptions.Item>
            <Descriptions.Item label="VaR Estimate" span={3}>
              {(riskAssessment.varEstimate * 100).toFixed(2)}%
            </Descriptions.Item>
            <Descriptions.Item label="Liquidity Risk" span={3}>
              {riskAssessment.liquidityRisk}
            </Descriptions.Item>
          </Descriptions>

          {riskAssessment.warnings.length > 0 && (
            <Alert
              message="Risk Warnings"
              description={
                <ul>
                  {riskAssessment.warnings.map((warning, index) => (
                    <li key={index}>{warning}</li>
                  ))}
                </ul>
              }
              type="warning"
              className="mt-4"
            />
          )}

          {riskAssessment.recommendations.length > 0 && (
            <Alert
              message="Recommendations"
              description={
                <ul>
                  {riskAssessment.recommendations.map((rec, index) => (
                    <li key={index}>{rec}</li>
                  ))}
                </ul>
              }
              type="info"
              className="mt-4"
            />
          )}
        </>
      ) : (
        <div className="text-center py-8">
          <Button 
            type="primary" 
            onClick={() => performRiskAssessment(deploymentForm.getFieldsValue())}
            loading={loading}
          >
            Perform Risk Assessment
          </Button>
        </div>
      )}

      {validationErrors.length > 0 && (
        <Alert
          message="Validation Errors"
          description={
            <ul>
              {validationErrors.map((error, index) => (
                <li key={index}>{error}</li>
              ))}
            </ul>
          }
          type="error"
          className="mt-4"
        />
      )}
    </Card>
  );

  const renderReviewStep = () => (
    <Card title="Deployment Review" className="mb-4">
      <Tabs defaultActiveKey="1">
        <TabPane tab="Configuration" key="1">
          <Descriptions bordered>
            <Descriptions.Item label="Strategy" span={3}>
              {strategy?.name}
            </Descriptions.Item>
            <Descriptions.Item label="Version" span={3}>
              {deploymentForm.getFieldValue('version')}
            </Descriptions.Item>
            <Descriptions.Item label="Environment" span={3}>
              {deploymentForm.getFieldValue('environment')}
            </Descriptions.Item>
            <Descriptions.Item label="Max Daily Loss" span={3}>
              ${deploymentForm.getFieldValue('maxDailyLoss')}
            </Descriptions.Item>
            <Descriptions.Item label="Max Positions" span={3}>
              {deploymentForm.getFieldValue('maxPositions')}
            </Descriptions.Item>
          </Descriptions>
        </TabPane>

        <TabPane tab="Backtest Results" key="2">
          {backtestResults ? (
            <Descriptions bordered>
              <Descriptions.Item label="Total Return" span={3}>
                {(backtestResults.totalReturn * 100).toFixed(2)}%
              </Descriptions.Item>
              <Descriptions.Item label="Sharpe Ratio" span={3}>
                {backtestResults.sharpeRatio.toFixed(2)}
              </Descriptions.Item>
              <Descriptions.Item label="Max Drawdown" span={3}>
                {(backtestResults.maxDrawdown * 100).toFixed(2)}%
              </Descriptions.Item>
              <Descriptions.Item label="Win Rate" span={3}>
                {(backtestResults.winRate * 100).toFixed(2)}%
              </Descriptions.Item>
              <Descriptions.Item label="Total Trades" span={3}>
                {backtestResults.totalTrades}
              </Descriptions.Item>
            </Descriptions>
          ) : (
            <Alert message="No backtest results available" type="warning" />
          )}
        </TabPane>

        <TabPane tab="Rollout Plan" key="3">
          <Table
            dataSource={deploymentForm.getFieldValue('rolloutPhases')}
            pagination={false}
            rowKey="name"
            columns={[
              { title: 'Phase', dataIndex: 'name' },
              { title: 'Position Size %', dataIndex: 'positionSizePercent' },
              { title: 'Duration (min)', dataIndex: 'duration', render: (val) => val === -1 ? 'Indefinite' : Math.round(val / 60) },
              { title: 'Min Trades', dataIndex: ['successCriteria', 'minTrades'] },
              { title: 'Max Drawdown %', dataIndex: ['successCriteria', 'maxDrawdown'], render: (val) => val ? (val * 100).toFixed(1) : '-' }
            ]}
          />
        </TabPane>
      </Tabs>
    </Card>
  );

  const renderApprovalStep = () => (
    <Card title="Approval Status" className="mb-4">
      {deployment ? (
        <>
          <Alert
            message={`Deployment request ${deployment.deploymentId} submitted`}
            description="Waiting for approval from required reviewers"
            type="info"
            className="mb-4"
          />
          
          <Descriptions bordered>
            <Descriptions.Item label="Status" span={3}>
              <Tag color="blue">{deployment.status.replace('_', ' ').toUpperCase()}</Tag>
            </Descriptions.Item>
            <Descriptions.Item label="Created" span={3}>
              {deployment.createdAt.toLocaleString()}
            </Descriptions.Item>
            <Descriptions.Item label="Required Approvals" span={3}>
              {deploymentForm.getFieldValue('requiredApprovals').join(', ')}
            </Descriptions.Item>
          </Descriptions>

          <div className="mt-4">
            <Title level={5}>Approval Progress</Title>
            <Progress
              percent={0}
              format={() => '0/2 approvals'}
              status="active"
            />
          </div>
        </>
      ) : (
        <div className="text-center py-8">
          <Button
            type="primary"
            size="large"
            onClick={submitDeploymentRequest}
            loading={loading}
          >
            Submit for Approval
          </Button>
        </div>
      )}
    </Card>
  );

  const steps = [
    {
      title: 'Configuration',
      icon: <SettingOutlined />,
      content: renderConfigurationStep(),
    },
    {
      title: 'Risk Assessment',
      icon: <SecurityScanOutlined />,
      content: renderRiskAssessmentStep(),
    },
    {
      title: 'Review',
      icon: <CheckCircleOutlined />,
      content: renderReviewStep(),
    },
    {
      title: 'Approval',
      icon: <ClockCircleOutlined />,
      content: renderApprovalStep(),
    },
  ];

  const nextStep = async () => {
    if (currentStep === 0) {
      try {
        await deploymentForm.validateFields();
        setCurrentStep(1);
      } catch (error) {
        message.error('Please fill in all required fields');
      }
    } else if (currentStep === 1) {
      const success = await performRiskAssessment(deploymentForm.getFieldsValue());
      if (!success && validationErrors.length > 0) {
        return;
      }
    } else if (currentStep === 2) {
      setCurrentStep(3);
    }
  };

  const prevStep = () => {
    setCurrentStep(currentStep - 1);
  };

  return (
    <div className="strategy-deployment-pipeline">
      <Card
        title={
          <div className="flex items-center space-x-2">
            <DeploymentUnitOutlined className="text-blue-600" />
            <span>Strategy Deployment Pipeline</span>
          </div>
        }
        extra={
          onClose && (
            <Button onClick={onClose}>Close</Button>
          )
        }
      >
        <Steps current={currentStep} className="mb-6">
          {steps.map((step, index) => (
            <Step key={index} title={step.title} icon={step.icon} />
          ))}
        </Steps>

        <div className="step-content">
          {steps[currentStep].content}
        </div>

        <div className="flex justify-between mt-6">
          <Button
            onClick={prevStep}
            disabled={currentStep === 0}
          >
            Previous
          </Button>
          
          <Space>
            {currentStep < steps.length - 1 && (
              <Button
                type="primary"
                onClick={nextStep}
                loading={loading}
                disabled={currentStep === 1 && (!riskAssessment || validationErrors.length > 0)}
              >
                Next
              </Button>
            )}
          </Space>
        </div>
      </Card>
    </div>
  );
};

export default StrategyDeploymentPipeline;