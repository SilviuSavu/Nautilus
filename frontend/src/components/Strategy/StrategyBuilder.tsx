import React, { useState, useEffect } from 'react';
import {
  Card,
  Row,
  Col,
  Steps,
  Button,
  Typography,
  Space,
  Alert,
  Spin,
  notification,
  Drawer,
  Divider
} from 'antd';
import {
  AppstoreOutlined,
  SettingOutlined,
  RocketOutlined,
  CheckCircleOutlined,
  EyeOutlined
} from '@ant-design/icons';

import { TemplateLibrary } from './TemplateLibrary';
import { ParameterConfig } from './ParameterConfig';
import { TemplatePreview } from './TemplatePreview';
import { StrategyTemplate, StrategyConfig } from './types/strategyTypes';
import strategyService from './services/strategyService';

const { Title, Text } = Typography;
const { Step } = Steps;

interface StrategyBuilderProps {
  onStrategyCreated?: (strategy: StrategyConfig) => void;
  className?: string;
}

export const StrategyBuilder: React.FC<StrategyBuilderProps> = ({
  onStrategyCreated,
  className
}) => {
  const [currentStep, setCurrentStep] = useState(0);
  const [selectedTemplate, setSelectedTemplate] = useState<StrategyTemplate | null>(null);
  const [strategyName, setStrategyName] = useState('');
  const [parameters, setParameters] = useState<Record<string, any>>({});
  const [isParametersValid, setIsParametersValid] = useState(false);
  const [parameterErrors, setParameterErrors] = useState<string[]>([]);
  const [loading, setLoading] = useState(false);
  const [previewVisible, setPreviewVisible] = useState(false);
  const [createdStrategy, setCreatedStrategy] = useState<StrategyConfig | null>(null);

  const steps = [
    {
      title: 'Select Template',
      description: 'Choose a strategy template',
      icon: <AppstoreOutlined />
    },
    {
      title: 'Configure Parameters',
      description: 'Set strategy parameters',
      icon: <SettingOutlined />
    },
    {
      title: 'Review & Create',
      description: 'Review and create strategy',
      icon: <RocketOutlined />
    }
  ];

  useEffect(() => {
    if (selectedTemplate) {
      // Generate a default strategy name
      const timestamp = new Date().toLocaleString();
      setStrategyName(`${selectedTemplate.name} - ${timestamp}`);
    }
  }, [selectedTemplate]);

  const handleTemplateSelect = (template: StrategyTemplate) => {
    setSelectedTemplate(template);
    setParameters({});
    setCurrentStep(1);
  };

  const handleParametersChange = (values: Record<string, any>) => {
    setParameters(values);
  };

  const handleParametersValidation = (isValid: boolean, errors: string[]) => {
    setIsParametersValid(isValid);
    setParameterErrors(errors);
  };

  const handleCreateStrategy = async () => {
    if (!selectedTemplate || !isParametersValid) return;

    try {
      setLoading(true);

      const response = await strategyService.createConfiguration({
        template_id: selectedTemplate.id,
        name: strategyName.trim() || `${selectedTemplate.name} Strategy`,
        parameters,
        risk_settings: extractRiskSettings()
      });

      const strategy = response.config;
      setCreatedStrategy(strategy);
      
      notification.success({
        message: 'Strategy Created Successfully',
        description: `Strategy "${strategy.name}" has been created and is ready for deployment.`,
        duration: 4
      });

      setCurrentStep(2);
      onStrategyCreated?.(strategy);

    } catch (error: any) {
      console.error('Failed to create strategy:', error);
      notification.error({
        message: 'Strategy Creation Failed',
        description: error.message || 'Failed to create strategy configuration',
        duration: 6
      });
    } finally {
      setLoading(false);
    }
  };

  const extractRiskSettings = () => {
    const riskSettings: Record<string, any> = {};
    
    if (!selectedTemplate) return riskSettings;

    selectedTemplate.risk_parameters.forEach(param => {
      if (parameters[param.name] !== undefined) {
        riskSettings[param.name] = parameters[param.name];
      }
    });

    return riskSettings;
  };

  const handleReset = () => {
    setCurrentStep(0);
    setSelectedTemplate(null);
    setParameters({});
    setStrategyName('');
    setIsParametersValid(false);
    setParameterErrors([]);
    setCreatedStrategy(null);
  };

  const handleBack = () => {
    if (currentStep > 0) {
      setCurrentStep(currentStep - 1);
    }
  };

  const canProceedToNext = () => {
    switch (currentStep) {
      case 0:
        return selectedTemplate !== null;
      case 1:
        return isParametersValid && strategyName.trim().length > 0;
      default:
        return false;
    }
  };

  const renderStepContent = () => {
    switch (currentStep) {
      case 0:
        return (
          <Row gutter={[16, 16]}>
            <Col xs={24} lg={selectedTemplate ? 16 : 24}>
              <TemplateLibrary
                onTemplateSelect={handleTemplateSelect}
                selectedTemplateId={selectedTemplate?.id}
              />
            </Col>
            {selectedTemplate && (
              <Col xs={24} lg={8}>
                <Card
                  title="Selected Template"
                  extra={
                    <Button
                      type="text"
                      icon={<EyeOutlined />}
                      onClick={() => setPreviewVisible(true)}
                    >
                      Preview
                    </Button>
                  }
                  size="small"
                >
                  <Space direction="vertical" size="small" style={{ width: '100%' }}>
                    <div>
                      <Text strong>{selectedTemplate.name}</Text>
                    </div>
                    <div>
                      <Text type="secondary">{selectedTemplate.description}</Text>
                    </div>
                    <div>
                      <Text type="secondary">
                        {selectedTemplate.parameters.length} parameters, 
                        {selectedTemplate.risk_parameters.length} risk controls
                      </Text>
                    </div>
                  </Space>
                </Card>
              </Col>
            )}
          </Row>
        );

      case 1:
        return selectedTemplate ? (
          <Row gutter={[16, 16]}>
            <Col xs={24} lg={16}>
              <Space direction="vertical" size="large" style={{ width: '100%' }}>
                <Card title="Strategy Name" size="small">
                  <input
                    type="text"
                    value={strategyName}
                    onChange={(e) => setStrategyName(e.target.value)}
                    placeholder="Enter strategy name..."
                    style={{
                      width: '100%',
                      padding: '8px 12px',
                      border: '1px solid #d9d9d9',
                      borderRadius: '6px',
                      fontSize: '14px'
                    }}
                  />
                </Card>

                <ParameterConfig
                  template={selectedTemplate}
                  initialValues={parameters}
                  onChange={handleParametersChange}
                  onValidation={handleParametersValidation}
                />
              </Space>
            </Col>
            
            <Col xs={24} lg={8}>
              <Card title="Validation Status" size="small">
                <Space direction="vertical" size="middle" style={{ width: '100%' }}>
                  {parameterErrors.length > 0 && (
                    <Alert
                      type="error"
                      message="Parameter Errors"
                      description={
                        <ul style={{ margin: 0, paddingLeft: 16 }}>
                          {parameterErrors.map((error, index) => (
                            <li key={index}>{error}</li>
                          ))}
                        </ul>
                      }
                      size="small"
                    />
                  )}
                  
                  {isParametersValid && (
                    <Alert
                      type="success"
                      message="Parameters Valid"
                      description="All parameters have been validated successfully"
                      size="small"
                      showIcon
                    />
                  )}

                  <div>
                    <Text type="secondary">
                      Strategy Name: {strategyName.trim() ? '✓' : '✗ Required'}
                    </Text>
                  </div>
                </Space>
              </Card>
            </Col>
          </Row>
        ) : null;

      case 2:
        return createdStrategy ? (
          <Row gutter={[16, 16]}>
            <Col span={24}>
              <Alert
                type="success"
                message="Strategy Created Successfully"
                description={`Strategy "${createdStrategy.name}" has been configured and is ready for deployment.`}
                showIcon
              />
            </Col>
            
            <Col span={24}>
              <Card title="Strategy Summary">
                <Row gutter={[16, 16]}>
                  <Col xs={24} md={12}>
                    <div><Text strong>Name:</Text> {createdStrategy.name}</div>
                    <div><Text strong>Template:</Text> {selectedTemplate?.name}</div>
                    <div><Text strong>Status:</Text> {createdStrategy.status}</div>
                    <div><Text strong>Version:</Text> {createdStrategy.version}</div>
                  </Col>
                  <Col xs={24} md={12}>
                    <div><Text strong>Parameters:</Text> {Object.keys(parameters).length}</div>
                    <div><Text strong>Created:</Text> {new Date(createdStrategy.created_at).toLocaleString()}</div>
                    <div><Text strong>ID:</Text> <Text code>{createdStrategy.id}</Text></div>
                  </Col>
                </Row>
              </Card>
            </Col>
          </Row>
        ) : null;

      default:
        return null;
    }
  };

  return (
    <div className={`strategy-builder ${className || ''}`}>
      <Card>
        <div style={{ marginBottom: 24 }}>
          <Title level={3}>Strategy Builder</Title>
          <Text type="secondary">
            Create and configure trading strategies using pre-built templates
          </Text>
        </div>

        <Steps current={currentStep} items={steps} style={{ marginBottom: 32 }} />

        <Spin spinning={loading}>
          <div style={{ minHeight: 400 }}>
            {renderStepContent()}
          </div>
        </Spin>

        <Divider />

        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <div>
            {currentStep > 0 && (
              <Button onClick={handleBack}>
                Back
              </Button>
            )}
          </div>

          <Space>
            <Button onClick={handleReset}>
              Reset
            </Button>
            
            {currentStep < 2 && (
              <Button
                type="primary"
                disabled={!canProceedToNext()}
                onClick={() => {
                  if (currentStep === 1) {
                    handleCreateStrategy();
                  } else {
                    setCurrentStep(currentStep + 1);
                  }
                }}
                icon={currentStep === 1 ? <CheckCircleOutlined /> : undefined}
              >
                {currentStep === 1 ? 'Create Strategy' : 'Next'}
              </Button>
            )}

            {currentStep === 2 && (
              <Button type="primary" onClick={handleReset}>
                Create Another Strategy
              </Button>
            )}
          </Space>
        </div>
      </Card>

      {/* Template Preview Drawer */}
      <Drawer
        title="Template Preview"
        placement="right"
        size="large"
        open={previewVisible}
        onClose={() => setPreviewVisible(false)}
      >
        {selectedTemplate && (
          <TemplatePreview template={selectedTemplate} />
        )}
      </Drawer>
    </div>
  );
};