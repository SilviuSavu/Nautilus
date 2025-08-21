import React, { useState, useEffect } from 'react';
import {
  Modal,
  Steps,
  Button,
  Alert,
  Typography,
  Space,
  Card,
  Descriptions,
  List,
  Checkbox,
  Radio,
  Input,
  Form,
  Progress,
  Result,
  Spin,
  Divider,
  Tag,
  Row,
  Col,
  Statistic,
  Timeline
} from 'antd';
import {
  RollbackOutlined,
  WarningOutlined,
  CheckCircleOutlined,
  ExclamationCircleOutlined,
  InfoCircleOutlined,
  LoadingOutlined,
  SafetyCertificateOutlined,
  DatabaseOutlined,
  ApiOutlined,
  SettingOutlined
} from '@ant-design/icons';
import {
  StrategyVersion,
  RollbackPlan,
  RollbackValidation,
  RollbackProgress,
  StrategyConfig,
  BackupSnapshot
} from './types/strategyTypes';
import { strategyService } from './services/strategyService';

const { Title, Text, Paragraph } = Typography;
const { Step } = Steps;
const { TextArea } = Input;

interface RollbackManagerProps {
  strategyId: string;
  currentVersion: number;
  targetVersion: StrategyVersion;
  visible: boolean;
  onClose: () => void;
  onRollbackComplete?: (success: boolean, newVersion?: number) => void;
}

export const RollbackManager: React.FC<RollbackManagerProps> = ({
  strategyId,
  currentVersion,
  targetVersion,
  visible,
  onClose,
  onRollbackComplete
}) => {
  const [currentStep, setCurrentStep] = useState(0);
  const [rollbackPlan, setRollbackPlan] = useState<RollbackPlan | null>(null);
  const [validation, setValidation] = useState<RollbackValidation | null>(null);
  const [progress, setProgress] = useState<RollbackProgress | null>(null);
  const [loading, setLoading] = useState(false);
  const [rollbackSettings, setRollbackSettings] = useState({
    createBackup: true,
    stopStrategy: true,
    validateBeforeRollback: true,
    forceRollback: false,
    reason: ''
  });
  const [form] = Form.useForm();

  useEffect(() => {
    if (visible) {
      generateRollbackPlan();
    }
  }, [visible, targetVersion]);

  const generateRollbackPlan = async () => {
    setLoading(true);
    try {
      const plan = await strategyService.generateRollbackPlan(
        strategyId,
        currentVersion,
        targetVersion.version_number
      );
      setRollbackPlan(plan);
      
      // Auto-validate if safe rollback
      if (plan.risk_assessment.risk_level === 'low') {
        await performValidation(plan);
      }
    } catch (error) {
      console.error('Failed to generate rollback plan:', error);
    } finally {
      setLoading(false);
    }
  };

  const performValidation = async (plan?: RollbackPlan) => {
    setLoading(true);
    try {
      const validationResult = await strategyService.validateRollback(
        strategyId,
        targetVersion.version_number,
        plan || rollbackPlan!
      );
      setValidation(validationResult);
    } catch (error) {
      console.error('Rollback validation failed:', error);
    } finally {
      setLoading(false);
    }
  };

  const executeRollback = async () => {
    if (!rollbackPlan || !validation) return;

    setLoading(true);
    setCurrentStep(3);

    try {
      const progressCallback = (progress: RollbackProgress) => {
        setProgress(progress);
      };

      const result = await strategyService.executeRollback(
        strategyId,
        targetVersion.version_number,
        {
          ...rollbackSettings,
          rollback_plan: rollbackPlan,
          validation_result: validation
        },
        progressCallback
      );

      if (result.success) {
        setCurrentStep(4);
        onRollbackComplete?.(true, targetVersion.version_number);
      } else {
        setCurrentStep(5);
        onRollbackComplete?.(false);
      }
    } catch (error) {
      console.error('Rollback execution failed:', error);
      setCurrentStep(5);
      onRollbackComplete?.(false);
    } finally {
      setLoading(false);
    }
  };

  const getRiskColor = (level: string) => {
    switch (level) {
      case 'low': return 'green';
      case 'medium': return 'orange';
      case 'high': return 'red';
      default: return 'default';
    }
  };

  const renderRollbackPlan = () => {
    if (!rollbackPlan) return <Spin />;

    return (
      <Space direction="vertical" style={{ width: '100%' }} size="large">
        <Alert
          message={`Rollback Plan: Version ${currentVersion} → Version ${targetVersion.version_number}`}
          description={`This will revert ${rollbackPlan.changes_to_revert.length} configuration changes.`}
          type="info"
          showIcon
        />

        <Card title="Risk Assessment" size="small">
          <Space direction="vertical" style={{ width: '100%' }}>
            <Row gutter={16}>
              <Col span={8}>
                <Statistic
                  title="Risk Level"
                  value={rollbackPlan.risk_assessment.risk_level.toUpperCase()}
                  valueStyle={{ 
                    color: rollbackPlan.risk_assessment.risk_level === 'low' ? '#3f8600' :
                           rollbackPlan.risk_assessment.risk_level === 'medium' ? '#cf1322' : '#cf1322'
                  }}
                />
              </Col>
              <Col span={8}>
                <Statistic
                  title="Changes to Revert"
                  value={rollbackPlan.changes_to_revert.length}
                />
              </Col>
              <Col span={8}>
                <Statistic
                  title="Estimated Time"
                  value={rollbackPlan.estimated_duration_seconds}
                  suffix="sec"
                />
              </Col>
            </Row>

            {rollbackPlan.risk_assessment.warnings.length > 0 && (
              <>
                <Divider />
                <Alert
                  message="Warnings"
                  description={
                    <ul>
                      {rollbackPlan.risk_assessment.warnings.map((warning, idx) => (
                        <li key={idx}>{warning}</li>
                      ))}
                    </ul>
                  }
                  type="warning"
                  showIcon
                />
              </>
            )}
          </Space>
        </Card>

        <Card title="Changes to be Reverted" size="small">
          <List
            size="small"
            dataSource={rollbackPlan.changes_to_revert}
            renderItem={change => (
              <List.Item>
                <Space direction="vertical" style={{ width: '100%' }}>
                  <Space>
                    <Tag color="blue">{change.change_type}</Tag>
                    <Text>{change.description}</Text>
                    <Text type="secondary">
                      {new Date(change.timestamp).toLocaleDateString()}
                    </Text>
                  </Space>
                  {change.parameters_affected && change.parameters_affected.length > 0 && (
                    <Text type="secondary">
                      Affects: {change.parameters_affected.join(', ')}
                    </Text>
                  )}
                </Space>
              </List.Item>
            )}
          />
        </Card>

        <Card title="Rollback Steps" size="small">
          <Timeline
            size="small"
            items={rollbackPlan.execution_steps.map((step, idx) => ({
              dot: step.critical ? <WarningOutlined style={{ color: '#ff4d4f' }} /> : undefined,
              children: (
                <Space direction="vertical">
                  <Text strong>{step.description}</Text>
                  {step.estimated_duration && (
                    <Text type="secondary">~{step.estimated_duration}s</Text>
                  )}
                  {step.critical && (
                    <Tag color="red" size="small">Critical</Tag>
                  )}
                </Space>
              )
            }))}
          />
        </Card>
      </Space>
    );
  };

  const renderValidation = () => {
    if (!validation) return <Spin />;

    const hasErrors = validation.validation_errors.length > 0;
    const hasWarnings = validation.warnings.length > 0;

    return (
      <Space direction="vertical" style={{ width: '100%' }} size="large">
        <Result
          status={hasErrors ? 'error' : hasWarnings ? 'warning' : 'success'}
          title={
            hasErrors ? 'Validation Failed' :
            hasWarnings ? 'Validation Passed with Warnings' :
            'Validation Successful'
          }
          subTitle={
            hasErrors ? 'Please review and fix the issues below before proceeding.' :
            hasWarnings ? 'Review warnings before proceeding with rollback.' :
            'Rollback is safe to proceed.'
          }
        />

        {hasErrors && (
          <Alert
            message="Validation Errors"
            description={
              <ul>
                {validation.validation_errors.map((error, idx) => (
                  <li key={idx}>{error}</li>
                ))}
              </ul>
            }
            type="error"
            showIcon
          />
        )}

        {hasWarnings && (
          <Alert
            message="Validation Warnings"
            description={
              <ul>
                {validation.warnings.map((warning, idx) => (
                  <li key={idx}>{warning}</li>
                ))}
              </ul>
            }
            type="warning"
            showIcon
          />
        )}

        <Card title="Pre-rollback Checks" size="small">
          <List
            size="small"
            dataSource={validation.pre_rollback_checks}
            renderItem={check => (
              <List.Item>
                <Space>
                  {check.passed ? 
                    <CheckCircleOutlined style={{ color: '#52c41a' }} /> :
                    <ExclamationCircleOutlined style={{ color: '#ff4d4f' }} />
                  }
                  <Text>{check.description}</Text>
                  {check.details && (
                    <Text type="secondary">({check.details})</Text>
                  )}
                </Space>
              </List.Item>
            )}
          />
        </Card>

        {validation.backup_verification && (
          <Card title="Backup Status" size="small">
            <Descriptions size="small" bordered>
              <Descriptions.Item label="Backup Created">
                {validation.backup_verification.backup_created ? 'Yes' : 'No'}
              </Descriptions.Item>
              <Descriptions.Item label="Backup Size">
                {validation.backup_verification.backup_size_mb} MB
              </Descriptions.Item>
              <Descriptions.Item label="Backup Location">
                {validation.backup_verification.backup_path}
              </Descriptions.Item>
            </Descriptions>
          </Card>
        )}
      </Space>
    );
  };

  const renderProgress = () => {
    if (!progress) return <Spin size="large" />;

    return (
      <Space direction="vertical" style={{ width: '100%' }} size="large">
        <Card>
          <Space direction="vertical" style={{ width: '100%' }} size="middle">
            <Title level={4}>Rollback in Progress</Title>
            
            <Progress
              percent={progress.overall_progress}
              status={progress.status === 'failed' ? 'exception' : 'active'}
              strokeColor={progress.status === 'failed' ? '#ff4d4f' : undefined}
            />

            <Descriptions size="small" bordered column={2}>
              <Descriptions.Item label="Current Step">
                {progress.current_step}
              </Descriptions.Item>
              <Descriptions.Item label="Steps Completed">
                {progress.completed_steps} / {progress.total_steps}
              </Descriptions.Item>
              <Descriptions.Item label="Elapsed Time">
                {progress.elapsed_seconds}s
              </Descriptions.Item>
              <Descriptions.Item label="Estimated Remaining">
                {progress.estimated_remaining_seconds}s
              </Descriptions.Item>
            </Descriptions>

            {progress.current_operation && (
              <Alert
                message={progress.current_operation}
                type="info"
                showIcon
                icon={<LoadingOutlined />}
              />
            )}

            {progress.errors.length > 0 && (
              <Alert
                message="Errors Occurred"
                description={
                  <ul>
                    {progress.errors.map((error, idx) => (
                      <li key={idx}>{error}</li>
                    ))}
                  </ul>
                }
                type="error"
                showIcon
              />
            )}
          </Space>
        </Card>
      </Space>
    );
  };

  const renderSettings = () => (
    <Form form={form} layout="vertical">
      <Space direction="vertical" style={{ width: '100%' }} size="middle">
        <Alert
          message="Rollback Configuration"
          description="Configure how the rollback should be performed."
          type="info"
          showIcon
        />

        <Card title="Safety Options" size="small">
          <Space direction="vertical" style={{ width: '100%' }}>
            <Checkbox
              checked={rollbackSettings.createBackup}
              onChange={(e) => setRollbackSettings(prev => ({ ...prev, createBackup: e.target.checked }))}
            >
              Create backup before rollback (recommended)
            </Checkbox>
            
            <Checkbox
              checked={rollbackSettings.stopStrategy}
              onChange={(e) => setRollbackSettings(prev => ({ ...prev, stopStrategy: e.target.checked }))}
            >
              Stop strategy execution during rollback
            </Checkbox>
            
            <Checkbox
              checked={rollbackSettings.validateBeforeRollback}
              onChange={(e) => setRollbackSettings(prev => ({ ...prev, validateBeforeRollback: e.target.checked }))}
            >
              Perform validation checks before rollback
            </Checkbox>
          </Space>
        </Card>

        <Card title="Advanced Options" size="small">
          <Space direction="vertical" style={{ width: '100%' }}>
            <Checkbox
              checked={rollbackSettings.forceRollback}
              onChange={(e) => setRollbackSettings(prev => ({ ...prev, forceRollback: e.target.checked }))}
            >
              Force rollback (ignore non-critical warnings)
            </Checkbox>

            <Form.Item label="Rollback Reason" style={{ margin: 0 }}>
              <TextArea
                value={rollbackSettings.reason}
                onChange={(e) => setRollbackSettings(prev => ({ ...prev, reason: e.target.value }))}
                placeholder="Describe why you're performing this rollback..."
                rows={3}
              />
            </Form.Item>
          </Space>
        </Card>
      </Space>
    </Form>
  );

  const steps = [
    {
      title: 'Plan',
      content: renderRollbackPlan(),
      icon: <SettingOutlined />
    },
    {
      title: 'Configure',
      content: renderSettings(),
      icon: <SafetyCertificateOutlined />
    },
    {
      title: 'Validate',
      content: renderValidation(),
      icon: <DatabaseOutlined />
    },
    {
      title: 'Execute',
      content: renderProgress(),
      icon: <ApiOutlined />
    },
    {
      title: 'Complete',
      content: (
        <Result
          status="success"
          title="Rollback Completed Successfully"
          subTitle={`Strategy has been rolled back to version ${targetVersion.version_number}`}
        />
      ),
      icon: <CheckCircleOutlined />
    },
    {
      title: 'Failed',
      content: (
        <Result
          status="error"
          title="Rollback Failed"
          subTitle="The rollback operation could not be completed. Please check the logs for details."
        />
      ),
      icon: <ExclamationCircleOutlined />
    }
  ];

  const canProceed = () => {
    switch (currentStep) {
      case 0: return !!rollbackPlan;
      case 1: return true;
      case 2: return !!validation && validation.validation_errors.length === 0;
      default: return false;
    }
  };

  const getNextStepTitle = () => {
    switch (currentStep) {
      case 0: return 'Configure';
      case 1: return 'Validate';
      case 2: return 'Execute Rollback';
      default: return 'Next';
    }
  };

  return (
    <Modal
      title={
        <Space>
          <RollbackOutlined />
          Rollback to Version {targetVersion.version_number}
        </Space>
      }
      open={visible}
      onCancel={onClose}
      width={1000}
      footer={
        currentStep < 3 ? [
          <Button key="cancel" onClick={onClose}>
            Cancel
          </Button>,
          currentStep > 0 && (
            <Button
              key="back"
              onClick={() => setCurrentStep(prev => prev - 1)}
              disabled={loading}
            >
              Back
            </Button>
          ),
          <Button
            key="next"
            type="primary"
            onClick={() => {
              if (currentStep === 2) {
                executeRollback();
              } else if (currentStep === 1) {
                performValidation();
                setCurrentStep(2);
              } else {
                setCurrentStep(prev => prev + 1);
              }
            }}
            disabled={!canProceed() || loading}
            loading={loading}
          >
            {currentStep === 2 ? 'Execute Rollback' : getNextStepTitle()}
          </Button>
        ] : [
          <Button key="close" type="primary" onClick={onClose}>
            Close
          </Button>
        ]
      }
    >
      <Space direction="vertical" style={{ width: '100%' }} size="large">
        <Steps
          current={Math.min(currentStep, 4)}
          items={steps.slice(0, 5).map((step, idx) => ({
            title: step.title,
            icon: step.icon,
            status: currentStep === 5 && idx === 4 ? 'error' : 
                   currentStep > idx ? 'finish' : 
                   currentStep === idx ? 'process' : 'wait'
          }))}
        />

        <div style={{ minHeight: 400 }}>
          {steps[Math.min(currentStep, 5)].content}
        </div>
      </Space>
    </Modal>
  );
};

export default RollbackManager;