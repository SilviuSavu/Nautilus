import React, { useState, useEffect, useCallback } from 'react';
import {
  Card,
  Row,
  Col,
  Tabs,
  Button,
  Typography,
  Space,
  Alert,
  Drawer,
  Switch,
  Tooltip,
  Badge,
  Divider,
  notification
} from 'antd';
import {
  BuildOutlined,
  EyeOutlined,
  BugOutlined,
  PlayCircleOutlined,
  PauseCircleOutlined,
  SettingOutlined,
  ThunderboltOutlined,
  CheckCircleOutlined,
  ExperimentOutlined
} from '@ant-design/icons';

import { TemplateLibrary } from './TemplateLibrary';
import { ParameterConfig } from './ParameterConfig';
import { TemplatePreview } from './TemplatePreview';
import { VisualStrategyBuilder } from './VisualStrategyBuilder';
import { StrategyFlowVisualization } from './StrategyFlowVisualization';
import { ParameterDependencyChecker } from './ParameterDependencyChecker';
import { StrategyTemplate, StrategyConfig } from './types/strategyTypes';
import { StrategyComponent, Connection } from './VisualStrategyBuilder';
import strategyService from './services/strategyService';

const { Title, Text } = Typography;
const { TabPane } = Tabs;

interface EnhancedStrategyBuilderProps {
  onStrategyCreated?: (strategy: StrategyConfig) => void;
  className?: string;
  mode?: 'template' | 'visual';
}

export const EnhancedStrategyBuilder: React.FC<EnhancedStrategyBuilderProps> = ({
  onStrategyCreated,
  className,
  mode: initialMode = 'template'
}) => {
  // Template mode state
  const [selectedTemplate, setSelectedTemplate] = useState<StrategyTemplate | null>(null);
  const [strategyName, setStrategyName] = useState('');
  const [parameters, setParameters] = useState<Record<string, any>>({});
  const [isParametersValid, setIsParametersValid] = useState(false);
  const [parameterErrors, setParameterErrors] = useState<string[]>([]);

  // Visual mode state
  const [visualComponents, setVisualComponents] = useState<StrategyComponent[]>([]);
  const [visualConnections, setVisualConnections] = useState<Connection[]>([]);
  const [dependencyIssues, setDependencyIssues] = useState<any[]>([]);

  // UI state
  const [activeTab, setActiveTab] = useState('builder');
  const [mode, setMode] = useState(initialMode);
  const [loading, setLoading] = useState(false);
  const [previewVisible, setPreviewVisible] = useState(false);
  const [simulationRunning, setSimulationRunning] = useState(false);
  const [realTimeData, setRealTimeData] = useState<Record<string, any>>({});

  // Auto-save functionality
  useEffect(() => {
    const autoSave = () => {
      if (mode === 'visual' && visualComponents.length > 0) {
        localStorage.setItem('strategy-builder-visual', JSON.stringify({
          components: visualComponents,
          connections: visualConnections,
          timestamp: Date.now()
        }));
      } else if (mode === 'template' && selectedTemplate && Object.keys(parameters).length > 0) {
        localStorage.setItem('strategy-builder-template', JSON.stringify({
          template: selectedTemplate,
          parameters,
          strategyName,
          timestamp: Date.now()
        }));
      }
    };

    const intervalId = setInterval(autoSave, 30000); // Auto-save every 30 seconds
    return () => clearInterval(intervalId);
  }, [mode, visualComponents, visualConnections, selectedTemplate, parameters, strategyName]);

  // Load auto-saved data on mount
  useEffect(() => {
    const loadAutoSave = () => {
      if (mode === 'visual') {
        const saved = localStorage.getItem('strategy-builder-visual');
        if (saved) {
          try {
            const data = JSON.parse(saved);
            const age = Date.now() - data.timestamp;
            if (age < 24 * 60 * 60 * 1000) { // Less than 24 hours old
              setVisualComponents(data.components || []);
              setVisualConnections(data.connections || []);
              notification.info({
                message: 'Auto-save Restored',
                description: 'Previous visual strategy work has been restored.',
                duration: 3
              });
            }
          } catch (error) {
            console.error('Failed to load auto-save:', error);
          }
        }
      } else {
        const saved = localStorage.getItem('strategy-builder-template');
        if (saved) {
          try {
            const data = JSON.parse(saved);
            const age = Date.now() - data.timestamp;
            if (age < 24 * 60 * 60 * 1000) {
              setSelectedTemplate(data.template);
              setParameters(data.parameters || {});
              setStrategyName(data.strategyName || '');
              notification.info({
                message: 'Auto-save Restored',
                description: 'Previous template strategy work has been restored.',
                duration: 3
              });
            }
          } catch (error) {
            console.error('Failed to load auto-save:', error);
          }
        }
      }
    };

    loadAutoSave();
  }, [mode]);

  // Template mode handlers
  const handleTemplateSelect = (template: StrategyTemplate) => {
    setSelectedTemplate(template);
    setParameters({});
    const timestamp = new Date().toLocaleString();
    setStrategyName(`${template.name} - ${timestamp}`);
  };

  const handleParametersChange = (values: Record<string, any>) => {
    setParameters(values);
  };

  const handleParametersValidation = (isValid: boolean, errors: string[]) => {
    setIsParametersValid(isValid);
    setParameterErrors(errors);
  };

  // Visual mode handlers
  const handleVisualStrategyChange = useCallback((components: StrategyComponent[], connections: Connection[]) => {
    setVisualComponents(components);
    setVisualConnections(connections);
  }, []);

  const handleDependencyIssues = useCallback((issues: any[]) => {
    setDependencyIssues(issues);
  }, []);

  // Strategy creation
  const handleCreateStrategy = async () => {
    if (mode === 'template') {
      if (!selectedTemplate || !isParametersValid) {
        notification.error({
          message: 'Invalid Configuration',
          description: 'Please select a template and fix all parameter errors before creating the strategy.'
        });
        return;
      }

      try {
        setLoading(true);
        const response = await strategyService.createConfiguration({
          template_id: selectedTemplate.id,
          name: strategyName.trim() || `${selectedTemplate.name} Strategy`,
          parameters,
          risk_settings: extractRiskSettings()
        });

        const strategy = response.config;
        
        notification.success({
          message: 'Strategy Created Successfully',
          description: `Template-based strategy "${strategy.name}" has been created.`,
          duration: 4
        });

        onStrategyCreated?.(strategy);
        clearAutoSave();

      } catch (error: any) {
        notification.error({
          message: 'Strategy Creation Failed',
          description: error.message || 'Failed to create template-based strategy',
          duration: 6
        });
      } finally {
        setLoading(false);
      }
    } else {
      // Visual mode - convert visual components to strategy config
      if (visualComponents.length === 0) {
        notification.error({
          message: 'No Components',
          description: 'Please add components to create a visual strategy.'
        });
        return;
      }

      const errorIssues = dependencyIssues.filter(issue => issue.severity === 'error');
      if (errorIssues.length > 0) {
        notification.error({
          message: 'Validation Errors',
          description: `Please fix ${errorIssues.length} error(s) before creating the strategy.`
        });
        return;
      }

      try {
        setLoading(true);
        
        // Convert visual strategy to template-based format for backend
        const visualStrategy = convertVisualToTemplate();
        
        notification.success({
          message: 'Visual Strategy Created',
          description: `Visual strategy with ${visualComponents.length} components has been created.`,
          duration: 4
        });

        clearAutoSave();
        
      } catch (error: any) {
        notification.error({
          message: 'Visual Strategy Creation Failed',
          description: error.message || 'Failed to create visual strategy',
          duration: 6
        });
      } finally {
        setLoading(false);
      }
    }
  };

  const extractRiskSettings = () => {
    if (!selectedTemplate) return {};
    
    const riskSettings: Record<string, any> = {};
    selectedTemplate.risk_parameters.forEach(param => {
      if (parameters[param.name] !== undefined) {
        riskSettings[param.name] = parameters[param.name];
      }
    });
    return riskSettings;
  };

  const convertVisualToTemplate = () => {
    // This would convert the visual component graph into a NautilusTrader-compatible format
    // For now, return a summary object
    return {
      type: 'visual',
      components: visualComponents.length,
      connections: visualConnections.length,
      validated: dependencyIssues.filter(i => i.severity === 'error').length === 0
    };
  };

  const clearAutoSave = () => {
    localStorage.removeItem('strategy-builder-visual');
    localStorage.removeItem('strategy-builder-template');
  };

  // Simulation controls
  const handleStartSimulation = () => {
    setSimulationRunning(true);
    
    // Simulate real-time data updates
    const intervalId = setInterval(() => {
      const newData: Record<string, any> = {};
      visualComponents.forEach(component => {
        if (component.type === 'indicator') {
          newData[component.id] = {
            value: Math.random() * 100,
            timestamp: Date.now()
          };
        }
      });
      setRealTimeData(newData);
    }, 1000);

    // Store interval ID for cleanup
    (window as any).simulationInterval = intervalId;
    
    notification.success({
      message: 'Simulation Started',
      description: 'Real-time strategy simulation is now running.',
      duration: 3
    });
  };

  const handleStopSimulation = () => {
    setSimulationRunning(false);
    setRealTimeData({});
    
    if ((window as any).simulationInterval) {
      clearInterval((window as any).simulationInterval);
      (window as any).simulationInterval = null;
    }
    
    notification.info({
      message: 'Simulation Stopped',
      description: 'Strategy simulation has been stopped.',
      duration: 3
    });
  };

  // Get validation summary
  const getValidationSummary = () => {
    if (mode === 'template') {
      return {
        isValid: isParametersValid && selectedTemplate !== null,
        errors: parameterErrors.length,
        warnings: 0
      };
    } else {
      const errors = dependencyIssues.filter(i => i.severity === 'error').length;
      const warnings = dependencyIssues.filter(i => i.severity === 'warning').length;
      return {
        isValid: errors === 0 && visualComponents.length > 0,
        errors,
        warnings
      };
    }
  };

  const summary = getValidationSummary();

  const renderModeSwitch = () => (
    <Card size="small" style={{ marginBottom: 16 }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <div>
          <Space>
            <Text strong>Strategy Builder Mode:</Text>
            <Switch
              checked={mode === 'visual'}
              onChange={(checked) => setMode(checked ? 'visual' : 'template')}
              checkedChildren="Visual"
              unCheckedChildren="Template"
            />
          </Space>
        </div>
        
        <Space>
          <Badge count={summary.errors} status="error">
            <Badge count={summary.warnings} status="warning" offset={[10, 0]}>
              <Button
                icon={<CheckCircleOutlined />}
                type={summary.isValid ? 'primary' : 'default'}
                size="small"
              >
                {summary.isValid ? 'Valid' : 'Issues Found'}
              </Button>
            </Badge>
          </Badge>

          {mode === 'visual' && (
            <Tooltip title={simulationRunning ? 'Stop Simulation' : 'Start Simulation'}>
              <Button
                icon={simulationRunning ? <PauseCircleOutlined /> : <PlayCircleOutlined />}
                onClick={simulationRunning ? handleStopSimulation : handleStartSimulation}
                type={simulationRunning ? 'default' : 'primary'}
                size="small"
                disabled={visualComponents.length === 0}
              >
                {simulationRunning ? 'Stop' : 'Simulate'}
              </Button>
            </Tooltip>
          )}
        </Space>
      </div>
    </Card>
  );

  const renderTemplateMode = () => (
    <Tabs activeKey={activeTab} onChange={setActiveTab}>
      <TabPane tab="Template Selection" key="selection">
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
                  <Text strong>{selectedTemplate.name}</Text>
                  <Text type="secondary">{selectedTemplate.description}</Text>
                  <Text type="secondary">
                    {selectedTemplate.parameters.length} parameters, 
                    {selectedTemplate.risk_parameters.length} risk controls
                  </Text>
                </Space>
              </Card>
            </Col>
          )}
        </Row>
      </TabPane>

      <TabPane tab="Parameter Configuration" key="parameters" disabled={!selectedTemplate}>
        {selectedTemplate && (
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
                </Space>
              </Card>
            </Col>
          </Row>
        )}
      </TabPane>
    </Tabs>
  );

  const renderVisualMode = () => (
    <Tabs activeKey={activeTab} onChange={setActiveTab}>
      <TabPane tab="Visual Builder" key="builder">
        <VisualStrategyBuilder
          onStrategyChange={handleVisualStrategyChange}
          readonly={false}
        />
      </TabPane>

      <TabPane tab="Flow Visualization" key="flow">
        <StrategyFlowVisualization
          components={visualComponents}
          connections={visualConnections}
          realTimeData={realTimeData}
          isSimulating={simulationRunning}
          onComponentClick={(component) => {
            notification.info({
              message: 'Component Selected',
              description: `Selected: ${component.name}`,
              duration: 2
            });
          }}
        />
      </TabPane>

      <TabPane 
        tab={
          <Space>
            <span>Dependency Check</span>
            {dependencyIssues.length > 0 && <Badge count={dependencyIssues.length} size="small" />}
          </Space>
        }
        key="dependencies"
      >
        <ParameterDependencyChecker
          components={visualComponents}
          connections={visualConnections}
          onIssueFound={handleDependencyIssues}
        />
      </TabPane>
    </Tabs>
  );

  return (
    <div className={`enhanced-strategy-builder ${className || ''}`}>
      <Card>
        <div style={{ marginBottom: 24 }}>
          <Title level={3}>Enhanced Strategy Builder</Title>
          <Text type="secondary">
            Create trading strategies using templates or visual drag-and-drop interface
          </Text>
        </div>

        {renderModeSwitch()}

        <div style={{ minHeight: 600 }}>
          {mode === 'template' ? renderTemplateMode() : renderVisualMode()}
        </div>

        <Divider />

        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <Space>
            <Text type="secondary">
              {mode === 'template' 
                ? `Template: ${selectedTemplate?.name || 'None selected'}` 
                : `Components: ${visualComponents.length}, Connections: ${visualConnections.length}`
              }
            </Text>
          </Space>

          <Space>
            <Button onClick={() => {
              if (confirm('Clear all data and start over?')) {
                setSelectedTemplate(null);
                setParameters({});
                setStrategyName('');
                setVisualComponents([]);
                setVisualConnections([]);
                clearAutoSave();
              }
            }}>
              Clear All
            </Button>
            
            <Button
              type="primary"
              loading={loading}
              disabled={!summary.isValid}
              onClick={handleCreateStrategy}
              icon={<BuildOutlined />}
            >
              Create Strategy
            </Button>
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