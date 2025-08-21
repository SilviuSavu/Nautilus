import React, { useMemo, useEffect, useState } from 'react';
import { Card, Alert, Tree, Typography, Space, Badge, Tooltip, Button, Collapse } from 'antd';
import {
  WarningOutlined,
  CheckCircleOutlined,
  ExclamationCircleOutlined,
  InfoCircleOutlined,
  BugOutlined,
  LinkOutlined,
  DisconnectOutlined
} from '@ant-design/icons';
import { StrategyComponent, Connection } from './VisualStrategyBuilder';
import { ParameterDefinition } from './types/strategyTypes';

const { Title, Text } = Typography;
const { Panel } = Collapse;

interface DependencyIssue {
  type: 'circular' | 'missing' | 'type_mismatch' | 'constraint_violation' | 'performance_warning';
  severity: 'error' | 'warning' | 'info';
  componentId: string;
  parameterId?: string;
  message: string;
  suggestion?: string;
  affectedComponents: string[];
}

interface ParameterDependency {
  sourceComponent: string;
  sourceParameter: string;
  targetComponent: string;
  targetParameter: string;
  relationship: 'equals' | 'greater_than' | 'less_than' | 'range' | 'multiplier' | 'inverse';
  constraint?: any;
}

interface ParameterDependencyCheckerProps {
  components: StrategyComponent[];
  connections: Connection[];
  onIssueFound?: (issues: DependencyIssue[]) => void;
  className?: string;
}

export const ParameterDependencyChecker: React.FC<ParameterDependencyCheckerProps> = ({
  components,
  connections,
  onIssueFound,
  className
}) => {
  const [issues, setIssues] = useState<DependencyIssue[]>([]);
  const [dependencies, setDependencies] = useState<ParameterDependency[]>([]);
  const [autoFixSuggestions, setAutoFixSuggestions] = useState<string[]>([]);

  // Common parameter dependencies for strategy components
  const commonDependencies: ParameterDependency[] = [
    // Moving Average dependencies
    {
      sourceComponent: 'indicator',
      sourceParameter: 'fast_period',
      targetComponent: 'indicator',
      targetParameter: 'slow_period',
      relationship: 'less_than',
      constraint: { message: 'Fast period must be less than slow period' }
    },
    // RSI threshold dependencies
    {
      sourceComponent: 'signal',
      sourceParameter: 'oversold_threshold',
      targetComponent: 'signal',
      targetParameter: 'overbought_threshold',
      relationship: 'less_than',
      constraint: { message: 'Oversold threshold must be less than overbought threshold' }
    },
    // Risk management dependencies
    {
      sourceComponent: 'risk_control',
      sourceParameter: 'stop_loss_pct',
      targetComponent: 'risk_control',
      targetParameter: 'take_profit_pct',
      relationship: 'less_than',
      constraint: { message: 'Stop loss should typically be smaller than take profit' }
    },
    // Position sizing dependencies
    {
      sourceComponent: 'action',
      sourceParameter: 'quantity',
      targetComponent: 'risk_control',
      targetParameter: 'max_position_size',
      relationship: 'less_than',
      constraint: { message: 'Order quantity must not exceed maximum position size' }
    }
  ];

  // Comprehensive dependency analysis
  const analyzeParameterDependencies = useMemo(() => {
    const issues: DependencyIssue[] = [];
    const detectedDependencies: ParameterDependency[] = [];

    // 1. Check for circular dependencies in connections
    const visited = new Set<string>();
    const recursionStack = new Set<string>();

    const hasCycle = (nodeId: string, path: string[] = []): boolean => {
      if (recursionStack.has(nodeId)) {
        issues.push({
          type: 'circular',
          severity: 'error',
          componentId: nodeId,
          message: `Circular dependency detected in component chain: ${[...path, nodeId].join(' → ')}`,
          suggestion: 'Remove one of the connections to break the cycle',
          affectedComponents: [...path, nodeId]
        });
        return true;
      }

      if (visited.has(nodeId)) return false;

      visited.add(nodeId);
      recursionStack.add(nodeId);

      const outgoingConnections = connections.filter(conn => conn.sourceId === nodeId);
      for (const conn of outgoingConnections) {
        if (hasCycle(conn.targetId, [...path, nodeId])) {
          return true;
        }
      }

      recursionStack.delete(nodeId);
      return false;
    };

    components.forEach(component => {
      if (!visited.has(component.id)) {
        hasCycle(component.id);
      }
    });

    // 2. Check parameter constraints within components
    components.forEach(component => {
      const params = component.parameters;

      // Check Moving Average constraints
      if (component.name.toLowerCase().includes('moving average') || component.type === 'indicator') {
        if (params.fast_period && params.slow_period) {
          if (Number(params.fast_period) >= Number(params.slow_period)) {
            issues.push({
              type: 'constraint_violation',
              severity: 'error',
              componentId: component.id,
              parameterId: 'fast_period',
              message: 'Fast period must be less than slow period for moving average crossover',
              suggestion: `Set fast period to a value less than ${params.slow_period}`,
              affectedComponents: [component.id]
            });
          }

          detectedDependencies.push({
            sourceComponent: component.id,
            sourceParameter: 'fast_period',
            targetComponent: component.id,
            targetParameter: 'slow_period',
            relationship: 'less_than'
          });
        }
      }

      // Check RSI constraints
      if (component.name.toLowerCase().includes('rsi') || 
          (params.oversold_threshold && params.overbought_threshold)) {
        const oversold = Number(params.oversold_threshold);
        const overbought = Number(params.overbought_threshold);
        
        if (oversold >= overbought) {
          issues.push({
            type: 'constraint_violation',
            severity: 'error',
            componentId: component.id,
            parameterId: 'oversold_threshold',
            message: 'Oversold threshold must be less than overbought threshold',
            suggestion: `Set oversold threshold below ${overbought} or increase overbought threshold above ${oversold}`,
            affectedComponents: [component.id]
          });
        }

        if (oversold < 0 || oversold > 100 || overbought < 0 || overbought > 100) {
          issues.push({
            type: 'constraint_violation',
            severity: 'error',
            componentId: component.id,
            message: 'RSI thresholds must be between 0 and 100',
            suggestion: 'Adjust thresholds to valid RSI range (typically 20-80)',
            affectedComponents: [component.id]
          });
        }
      }

      // Check Bollinger Bands constraints
      if (component.name.toLowerCase().includes('bollinger') && params.deviation) {
        const deviation = Number(params.deviation);
        if (deviation <= 0) {
          issues.push({
            type: 'constraint_violation',
            severity: 'error',
            componentId: component.id,
            parameterId: 'deviation',
            message: 'Bollinger Bands deviation must be positive',
            suggestion: 'Set deviation to a value greater than 0 (typically 2.0)',
            affectedComponents: [component.id]
          });
        }
      }

      // Check risk management constraints
      if (component.type === 'risk_control') {
        if (params.stop_loss_pct && params.take_profit_pct) {
          const stopLoss = Number(params.stop_loss_pct);
          const takeProfit = Number(params.take_profit_pct);
          
          if (stopLoss >= takeProfit) {
            issues.push({
              type: 'constraint_violation',
              severity: 'warning',
              componentId: component.id,
              message: 'Stop loss percentage is not less than take profit percentage',
              suggestion: 'Consider setting stop loss lower than take profit for better risk/reward ratio',
              affectedComponents: [component.id]
            });
          }

          if (stopLoss > 10) {
            issues.push({
              type: 'performance_warning',
              severity: 'warning',
              componentId: component.id,
              parameterId: 'stop_loss_pct',
              message: 'Stop loss percentage is quite high (>10%)',
              suggestion: 'Consider reducing stop loss to limit potential losses',
              affectedComponents: [component.id]
            });
          }
        }
      }

      // Check position sizing constraints
      if (component.type === 'action' && params.quantity) {
        const quantity = Number(params.quantity);
        if (quantity <= 0) {
          issues.push({
            type: 'constraint_violation',
            severity: 'error',
            componentId: component.id,
            parameterId: 'quantity',
            message: 'Order quantity must be positive',
            suggestion: 'Set quantity to a positive value',
            affectedComponents: [component.id]
          });
        }
      }
    });

    // 3. Check cross-component dependencies
    const riskComponents = components.filter(c => c.type === 'risk_control');
    const actionComponents = components.filter(c => c.type === 'action');

    actionComponents.forEach(actionComp => {
      const quantity = Number(actionComp.parameters.quantity || 0);
      
      riskComponents.forEach(riskComp => {
        const maxPosition = Number(riskComp.parameters.max_position_size || 0);
        
        if (quantity > maxPosition && maxPosition > 0) {
          issues.push({
            type: 'constraint_violation',
            severity: 'error',
            componentId: actionComp.id,
            message: `Order quantity (${quantity}) exceeds maximum position size (${maxPosition})`,
            suggestion: `Reduce order quantity to ${maxPosition} or increase maximum position size`,
            affectedComponents: [actionComp.id, riskComp.id]
          });

          detectedDependencies.push({
            sourceComponent: actionComp.id,
            sourceParameter: 'quantity',
            targetComponent: riskComp.id,
            targetParameter: 'max_position_size',
            relationship: 'less_than'
          });
        }
      });
    });

    // 4. Check for missing required connections
    components.forEach(component => {
      component.inputs.forEach(input => {
        if (input.required && !input.connected) {
          const connectedInput = connections.find(conn => 
            conn.targetId === component.id && conn.targetInput === input.id
          );
          
          if (!connectedInput) {
            issues.push({
              type: 'missing',
              severity: 'error',
              componentId: component.id,
              message: `Required input '${input.name}' is not connected`,
              suggestion: 'Connect a compatible output to this input',
              affectedComponents: [component.id]
            });
          }
        }
      });
    });

    // 5. Check type compatibility in connections
    connections.forEach(connection => {
      const sourceComponent = components.find(c => c.id === connection.sourceId);
      const targetComponent = components.find(c => c.id === connection.targetId);
      
      if (sourceComponent && targetComponent) {
        const sourceOutput = sourceComponent.outputs.find(o => o.id === connection.sourceOutput);
        const targetInput = targetComponent.inputs.find(i => i.id === connection.targetInput);
        
        if (sourceOutput && targetInput && sourceOutput.type !== targetInput.type) {
          issues.push({
            type: 'type_mismatch',
            severity: 'warning',
            componentId: targetComponent.id,
            message: `Type mismatch: connecting ${sourceOutput.type} output to ${targetInput.type} input`,
            suggestion: 'Verify that the data types are compatible or add a converter component',
            affectedComponents: [sourceComponent.id, targetComponent.id]
          });
        }
      }
    });

    setDependencies(detectedDependencies);
    return issues;
  }, [components, connections]);

  // Auto-fix suggestions
  const generateAutoFixSuggestions = useMemo(() => {
    const suggestions: string[] = [];

    issues.forEach(issue => {
      switch (issue.type) {
        case 'constraint_violation':
          if (issue.parameterId) {
            suggestions.push(`Auto-fix: Adjust ${issue.parameterId} in ${issue.componentId}`);
          }
          break;
        case 'missing':
          suggestions.push(`Auto-fix: Add default connection for ${issue.componentId}`);
          break;
        case 'circular':
          suggestions.push(`Auto-fix: Remove circular dependency in ${issue.affectedComponents.join(' → ')}`);
          break;
      }
    });

    return suggestions;
  }, [issues]);

  // Update issues when analysis changes
  useEffect(() => {
    setIssues(analyzeParameterDependencies);
    setAutoFixSuggestions(generateAutoFixSuggestions);
    onIssueFound?.(analyzeParameterDependencies);
  }, [analyzeParameterDependencies, generateAutoFixSuggestions, onIssueFound]);

  // Group issues by severity
  const groupedIssues = useMemo(() => {
    return {
      error: issues.filter(issue => issue.severity === 'error'),
      warning: issues.filter(issue => issue.severity === 'warning'),
      info: issues.filter(issue => issue.severity === 'info')
    };
  }, [issues]);

  const getIssueIcon = (type: string, severity: string) => {
    if (severity === 'error') return <ExclamationCircleOutlined style={{ color: '#ff4d4f' }} />;
    if (severity === 'warning') return <WarningOutlined style={{ color: '#fa8c16' }} />;
    return <InfoCircleOutlined style={{ color: '#1890ff' }} />;
  };

  const getComponentName = (componentId: string) => {
    const component = components.find(c => c.id === componentId);
    return component?.name || componentId;
  };

  const renderIssueList = (issueList: DependencyIssue[], title: string, type: 'error' | 'warning' | 'info') => {
    if (issueList.length === 0) return null;

    const alertType = type === 'error' ? 'error' : type === 'warning' ? 'warning' : 'info';
    
    return (
      <Panel 
        key={type}
        header={
          <Space>
            <Badge count={issueList.length} status={type === 'error' ? 'error' : type === 'warning' ? 'warning' : 'processing'} />
            <Text strong>{title}</Text>
          </Space>
        }
      >
        {issueList.map((issue, index) => (
          <Alert
            key={index}
            type={alertType}
            message={
              <Space>
                {getIssueIcon(issue.type, issue.severity)}
                <Text strong>{getComponentName(issue.componentId)}</Text>
                {issue.parameterId && <Text code>{issue.parameterId}</Text>}
              </Space>
            }
            description={
              <div>
                <div>{issue.message}</div>
                {issue.suggestion && (
                  <div style={{ marginTop: 8, fontStyle: 'italic', color: '#666' }}>
                    💡 {issue.suggestion}
                  </div>
                )}
                {issue.affectedComponents.length > 1 && (
                  <div style={{ marginTop: 8 }}>
                    <Text type="secondary">Affects: </Text>
                    {issue.affectedComponents.map(id => getComponentName(id)).join(', ')}
                  </div>
                )}
              </div>
            }
            style={{ marginBottom: 8 }}
            showIcon={false}
          />
        ))}
      </Panel>
    );
  };

  const renderDependencyTree = () => {
    const treeData = dependencies.map((dep, index) => ({
      title: (
        <Space>
          <Text>{getComponentName(dep.sourceComponent)}</Text>
          <Text code>{dep.sourceParameter}</Text>
          <Text type="secondary">{dep.relationship.replace('_', ' ')}</Text>
          <Text>{getComponentName(dep.targetComponent)}</Text>
          <Text code>{dep.targetParameter}</Text>
        </Space>
      ),
      key: `dep-${index}`,
      icon: <LinkOutlined />
    }));

    return treeData.length > 0 ? (
      <Tree
        treeData={treeData}
        defaultExpandAll
        showIcon
      />
    ) : (
      <Text type="secondary">No parameter dependencies detected</Text>
    );
  };

  const totalIssues = issues.length;
  const hasErrors = groupedIssues.error.length > 0;
  const hasWarnings = groupedIssues.warning.length > 0;

  return (
    <div className={`parameter-dependency-checker ${className || ''}`}>
      <Card>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 16 }}>
          <div>
            <Title level={4} style={{ margin: 0 }}>
              Parameter Dependency Analysis
            </Title>
            <Text type="secondary">
              Real-time validation of parameter constraints and dependencies
            </Text>
          </div>

          <Space>
            {totalIssues === 0 ? (
              <Badge status="success" text="All Valid" />
            ) : (
              <>
                {hasErrors && <Badge count={groupedIssues.error.length} status="error" />}
                {hasWarnings && <Badge count={groupedIssues.warning.length} status="warning" />}
                {groupedIssues.info.length > 0 && <Badge count={groupedIssues.info.length} status="processing" />}
              </>
            )}
          </Space>
        </div>

        {totalIssues === 0 ? (
          <Alert
            type="success"
            message="No Dependency Issues Found"
            description="All parameter constraints and dependencies are satisfied."
            showIcon
            icon={<CheckCircleOutlined />}
          />
        ) : (
          <Collapse>
            {renderIssueList(groupedIssues.error, 'Critical Issues', 'error')}
            {renderIssueList(groupedIssues.warning, 'Warnings', 'warning')}
            {renderIssueList(groupedIssues.info, 'Information', 'info')}
            
            <Panel header="Parameter Dependencies" key="dependencies">
              {renderDependencyTree()}
            </Panel>

            {autoFixSuggestions.length > 0 && (
              <Panel header="Auto-Fix Suggestions" key="suggestions">
                {autoFixSuggestions.map((suggestion, index) => (
                  <div key={index} style={{ marginBottom: 8 }}>
                    <Button size="small" type="link" icon={<BugOutlined />}>
                      {suggestion}
                    </Button>
                  </div>
                ))}
              </Panel>
            )}
          </Collapse>
        )}
      </Card>
    </div>
  );
};