import React, { useState, useCallback, useRef, useEffect } from 'react';
import {
  Card,
  Button,
  Typography,
  Space,
  Alert,
  Drawer,
  Tree,
  Tooltip,
  Badge,
  Divider,
  Input,
  Select,
  Collapse
} from 'antd';
import {
  DndContext,
  DragEndEvent,
  DragOverlay,
  DragStartEvent,
  PointerSensor,
  useSensor,
  useSensors,
  closestCenter,
  defaultDropAnimationSideEffects
} from '@dnd-kit/core';
import {
  SortableContext,
  arrayMove,
  verticalListSortingStrategy,
  useSortable
} from '@dnd-kit/sortable';
import {
  CSS
} from '@dnd-kit/utilities';
import {
  BuildOutlined,
  PlayCircleOutlined,
  StopOutlined,
  SettingOutlined,
  InfoCircleOutlined,
  ExclamationCircleOutlined,
  CheckCircleOutlined,
  BugOutlined,
  CodeOutlined,
  NodeIndexOutlined,
  FunctionOutlined,
  ClockCircleOutlined
} from '@ant-design/icons';

const { Title, Text } = Typography;
const { Panel } = Collapse;

// Strategy Component Types
export interface StrategyComponent {
  id: string;
  type: 'indicator' | 'signal' | 'condition' | 'action' | 'risk_control';
  name: string;
  description: string;
  parameters: Record<string, any>;
  inputs: ComponentInput[];
  outputs: ComponentOutput[];
  position: { x: number; y: number };
  isValid: boolean;
  errors: string[];
  dependencies: string[];
}

export interface ComponentInput {
  id: string;
  name: string;
  type: 'number' | 'boolean' | 'signal' | 'price_data';
  required: boolean;
  connected: boolean;
  sourceId?: string;
  sourceOutput?: string;
}

export interface ComponentOutput {
  id: string;
  name: string;
  type: 'number' | 'boolean' | 'signal' | 'price_data';
  value?: any;
}

export interface Connection {
  id: string;
  sourceId: string;
  sourceOutput: string;
  targetId: string;
  targetInput: string;
}

interface VisualStrategyBuilderProps {
  onStrategyChange?: (components: StrategyComponent[], connections: Connection[]) => void;
  className?: string;
  readonly?: boolean;
}

// Draggable Component Item
const SortableComponent: React.FC<{
  component: StrategyComponent;
  onUpdate: (id: string, updates: Partial<StrategyComponent>) => void;
  onRemove: (id: string) => void;
  connections: Connection[];
}> = ({ component, onUpdate, onRemove, connections }) => {
  const {
    attributes,
    listeners,
    setNodeRef,
    transform,
    transition,
    isDragging
  } = useSortable({ id: component.id });

  const style = {
    transform: CSS.Transform.toString(transform),
    transition,
    opacity: isDragging ? 0.5 : 1
  };

  const getTypeIcon = (type: string) => {
    switch (type) {
      case 'indicator': return <NodeIndexOutlined />;
      case 'signal': return <PlayCircleOutlined />;
      case 'condition': return <BugOutlined />;
      case 'action': return <FunctionOutlined />;
      case 'risk_control': return <StopOutlined />;
      default: return <SettingOutlined />;
    }
  };

  const getTypeColor = (type: string) => {
    switch (type) {
      case 'indicator': return '#1890ff';
      case 'signal': return '#52c41a';
      case 'condition': return '#fa8c16';
      case 'action': return '#722ed1';
      case 'risk_control': return '#f5222d';
      default: return '#666666';
    }
  };

  const getConnectedInputs = () => {
    return connections.filter(conn => conn.targetId === component.id).length;
  };

  const getConnectedOutputs = () => {
    return connections.filter(conn => conn.sourceId === component.id).length;
  };

  return (
    <div ref={setNodeRef} style={style} {...attributes} {...listeners}>
      <Card
        size="small"
        style={{
          margin: '8px 0',
          border: `2px solid ${component.isValid ? getTypeColor(component.type) : '#ff4d4f'}`,
          borderRadius: '8px',
          backgroundColor: isDragging ? '#f0f0f0' : 'white'
        }}
        styles={{ body: { padding: '12px' } }}
      >
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 8, flex: 1 }}>
            <div style={{ color: getTypeColor(component.type), fontSize: 16 }}>
              {getTypeIcon(component.type)}
            </div>
            
            <div style={{ flex: 1 }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                <Text strong style={{ fontSize: 14 }}>{component.name}</Text>
                {!component.isValid && (
                  <Tooltip title={component.errors.join(', ')}>
                    <ExclamationCircleOutlined style={{ color: '#ff4d4f' }} />
                  </Tooltip>
                )}
              </div>
              <Text type="secondary" style={{ fontSize: 12 }}>
                {component.description}
              </Text>
            </div>
          </div>

          <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
            <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 2 }}>
              <Badge 
                count={getConnectedInputs()} 
                size="small" 
                style={{ backgroundColor: '#1890ff' }}
              >
                <div style={{ fontSize: 10, color: '#666' }}>IN</div>
              </Badge>
              <Badge 
                count={getConnectedOutputs()} 
                size="small" 
                style={{ backgroundColor: '#52c41a' }}
              >
                <div style={{ fontSize: 10, color: '#666' }}>OUT</div>
              </Badge>
            </div>
            
            <Button
              type="text"
              size="small"
              icon={<SettingOutlined />}
              onClick={(e) => {
                e.stopPropagation();
                // Open component configuration modal
              }}
            />
            
            <Button
              type="text"
              size="small"
              danger
              icon={<StopOutlined />}
              onClick={(e) => {
                e.stopPropagation();
                onRemove(component.id);
              }}
            />
          </div>
        </div>

        {/* Parameter Summary */}
        {Object.keys(component.parameters).length > 0 && (
          <div style={{ marginTop: 8, fontSize: 11, color: '#666' }}>
            <Text>
              {Object.entries(component.parameters).slice(0, 2).map(([key, value], index) => (
                <span key={key}>
                  {index > 0 && ', '}
                  {key}: {String(value)}
                </span>
              ))}
              {Object.keys(component.parameters).length > 2 && '...'}
            </Text>
          </div>
        )}
      </Card>
    </div>
  );
};

// Component Library Panel
const ComponentLibrary: React.FC<{
  onAddComponent: (type: string, template: any) => void;
}> = ({ onAddComponent }) => {
  const componentTemplates = {
    indicator: [
      {
        name: 'Moving Average',
        description: 'Simple or Exponential Moving Average',
        parameters: { period: 20, type: 'SMA' },
        inputs: [{ id: 'price', name: 'Price', type: 'price_data', required: true }],
        outputs: [{ id: 'ma_value', name: 'MA Value', type: 'number' }]
      },
      {
        name: 'RSI',
        description: 'Relative Strength Index',
        parameters: { period: 14 },
        inputs: [{ id: 'price', name: 'Price', type: 'price_data', required: true }],
        outputs: [{ id: 'rsi_value', name: 'RSI Value', type: 'number' }]
      },
      {
        name: 'Bollinger Bands',
        description: 'Bollinger Bands with configurable deviation',
        parameters: { period: 20, deviation: 2 },
        inputs: [{ id: 'price', name: 'Price', type: 'price_data', required: true }],
        outputs: [
          { id: 'upper_band', name: 'Upper Band', type: 'number' },
          { id: 'lower_band', name: 'Lower Band', type: 'number' },
          { id: 'middle_band', name: 'Middle Band', type: 'number' }
        ]
      }
    ],
    signal: [
      {
        name: 'Cross Above',
        description: 'Signal when value crosses above threshold',
        parameters: { threshold: 0 },
        inputs: [
          { id: 'value', name: 'Value', type: 'number', required: true },
          { id: 'threshold', name: 'Threshold', type: 'number', required: true }
        ],
        outputs: [{ id: 'signal', name: 'Signal', type: 'boolean' }]
      },
      {
        name: 'Cross Below',
        description: 'Signal when value crosses below threshold',
        parameters: { threshold: 0 },
        inputs: [
          { id: 'value', name: 'Value', type: 'number', required: true },
          { id: 'threshold', name: 'Threshold', type: 'number', required: true }
        ],
        outputs: [{ id: 'signal', name: 'Signal', type: 'boolean' }]
      }
    ],
    condition: [
      {
        name: 'AND Gate',
        description: 'Logical AND of multiple signals',
        parameters: {},
        inputs: [
          { id: 'signal1', name: 'Signal 1', type: 'boolean', required: true },
          { id: 'signal2', name: 'Signal 2', type: 'boolean', required: true }
        ],
        outputs: [{ id: 'result', name: 'Result', type: 'boolean' }]
      },
      {
        name: 'OR Gate',
        description: 'Logical OR of multiple signals',
        parameters: {},
        inputs: [
          { id: 'signal1', name: 'Signal 1', type: 'boolean', required: true },
          { id: 'signal2', name: 'Signal 2', type: 'boolean', required: true }
        ],
        outputs: [{ id: 'result', name: 'Result', type: 'boolean' }]
      }
    ],
    action: [
      {
        name: 'Buy Order',
        description: 'Execute buy order when triggered',
        parameters: { quantity: 100, order_type: 'MARKET' },
        inputs: [{ id: 'trigger', name: 'Trigger', type: 'boolean', required: true }],
        outputs: [{ id: 'order_id', name: 'Order ID', type: 'signal' }]
      },
      {
        name: 'Sell Order',
        description: 'Execute sell order when triggered',
        parameters: { quantity: 100, order_type: 'MARKET' },
        inputs: [{ id: 'trigger', name: 'Trigger', type: 'boolean', required: true }],
        outputs: [{ id: 'order_id', name: 'Order ID', type: 'signal' }]
      }
    ],
    risk_control: [
      {
        name: 'Stop Loss',
        description: 'Automatic stop loss protection',
        parameters: { stop_pct: 2.0 },
        inputs: [{ id: 'position', name: 'Position', type: 'signal', required: true }],
        outputs: [{ id: 'stop_signal', name: 'Stop Signal', type: 'boolean' }]
      },
      {
        name: 'Position Size Limiter',
        description: 'Limit maximum position size',
        parameters: { max_position: 1000 },
        inputs: [{ id: 'order_signal', name: 'Order Signal', type: 'signal', required: true }],
        outputs: [{ id: 'adjusted_signal', name: 'Adjusted Signal', type: 'signal' }]
      }
    ]
  };

  const renderComponentList = (category: string, templates: any[]) => (
    <div key={category}>
      <Title level={5} style={{ margin: '12px 0 8px 0', textTransform: 'capitalize' }}>
        {category.replace('_', ' ')}
      </Title>
      {templates.map((template, index) => (
        <Button
          key={`${category}-${index}`}
          type="ghost"
          block
          style={{ 
            marginBottom: 4, 
            textAlign: 'left', 
            height: 'auto', 
            padding: '8px 12px',
            border: '1px dashed #d9d9d9'
          }}
          onClick={() => onAddComponent(category, template)}
        >
          <div>
            <div style={{ fontWeight: 500 }}>{template.name}</div>
            <div style={{ fontSize: 12, color: '#666', marginTop: 2 }}>
              {template.description}
            </div>
          </div>
        </Button>
      ))}
    </div>
  );

  return (
    <div style={{ padding: '16px', maxHeight: '70vh', overflowY: 'auto' }}>
      <Title level={4}>Component Library</Title>
      <Text type="secondary" style={{ display: 'block', marginBottom: 16 }}>
        Drag components to build your strategy
      </Text>
      
      {Object.entries(componentTemplates).map(([category, templates]) =>
        renderComponentList(category, templates)
      )}
    </div>
  );
};

export const VisualStrategyBuilder: React.FC<VisualStrategyBuilderProps> = ({
  onStrategyChange,
  className,
  readonly = false
}) => {
  const [components, setComponents] = useState<StrategyComponent[]>([]);
  const [connections, setConnections] = useState<Connection[]>([]);
  const [activeId, setActiveId] = useState<string | null>(null);
  const [libraryVisible, setLibraryVisible] = useState(false);
  const [validationResults, setValidationResults] = useState<Record<string, string[]>>({});

  const sensors = useSensors(
    useSensor(PointerSensor, {
      activationConstraint: {
        distance: 8,
      },
    })
  );

  // Real-time validation
  const validateComponent = useCallback((component: StrategyComponent): { isValid: boolean; errors: string[] } => {
    const errors: string[] = [];

    // Check required inputs
    component.inputs.forEach(input => {
      if (input.required && !input.connected) {
        errors.push(`Required input '${input.name}' is not connected`);
      }
    });

    // Check parameter validation
    Object.entries(component.parameters).forEach(([key, value]) => {
      if (value === undefined || value === null || value === '') {
        errors.push(`Parameter '${key}' is required`);
      }
    });

    // Check dependencies
    component.dependencies.forEach(depId => {
      const depExists = components.some(c => c.id === depId);
      if (!depExists) {
        errors.push(`Dependency component '${depId}' not found`);
      }
    });

    return {
      isValid: errors.length === 0,
      errors
    };
  }, [components]);

  // Validate all components when components or connections change
  useEffect(() => {
    const updatedComponents = components.map(component => {
      const validation = validateComponent(component);
      return {
        ...component,
        isValid: validation.isValid,
        errors: validation.errors
      };
    });

    if (JSON.stringify(updatedComponents) !== JSON.stringify(components)) {
      setComponents(updatedComponents);
    }

    // Update validation results
    const newValidationResults: Record<string, string[]> = {};
    updatedComponents.forEach(component => {
      if (!component.isValid) {
        newValidationResults[component.id] = component.errors;
      }
    });
    setValidationResults(newValidationResults);

    // Notify parent of changes
    onStrategyChange?.(updatedComponents, connections);
  }, [components.length, connections.length, validateComponent, onStrategyChange]);

  const handleDragStart = (event: DragStartEvent) => {
    setActiveId(String(event.active.id));
  };

  const handleDragEnd = (event: DragEndEvent) => {
    const { active, over } = event;

    if (active.id !== over?.id) {
      setComponents((items) => {
        const oldIndex = items.findIndex(item => item.id === active.id);
        const newIndex = items.findIndex(item => item.id === over?.id);

        return arrayMove(items, oldIndex, newIndex);
      });
    }

    setActiveId(null);
  };

  const addComponent = (type: string, template: any) => {
    const newComponent: StrategyComponent = {
      id: `${type}_${Date.now()}`,
      type: type as any,
      name: template.name,
      description: template.description,
      parameters: { ...template.parameters },
      inputs: template.inputs.map((input: any) => ({ ...input, connected: false })),
      outputs: template.outputs,
      position: { x: 100, y: 100 },
      isValid: false,
      errors: [],
      dependencies: []
    };

    setComponents(prev => [...prev, newComponent]);
    setLibraryVisible(false);
  };

  const updateComponent = (id: string, updates: Partial<StrategyComponent>) => {
    setComponents(prev => 
      prev.map(comp => comp.id === id ? { ...comp, ...updates } : comp)
    );
  };

  const removeComponent = (id: string) => {
    setComponents(prev => prev.filter(comp => comp.id !== id));
    setConnections(prev => prev.filter(conn => 
      conn.sourceId !== id && conn.targetId !== id
    ));
  };

  const getValidationSummary = () => {
    const totalComponents = components.length;
    const validComponents = components.filter(c => c.isValid).length;
    const totalErrors = Object.values(validationResults).flat().length;

    return { totalComponents, validComponents, totalErrors };
  };

  const summary = getValidationSummary();

  return (
    <div className={`visual-strategy-builder ${className || ''}`}>
      <Card>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 16 }}>
          <div>
            <Title level={4} style={{ margin: 0 }}>
              Visual Strategy Builder
            </Title>
            <Text type="secondary">
              Build strategies using drag-and-drop components
            </Text>
          </div>
          
          <Space>
            <Badge count={summary.totalErrors} status="error">
              <Button
                icon={<InfoCircleOutlined />}
                size="small"
              >
                {summary.validComponents}/{summary.totalComponents} Valid
              </Button>
            </Badge>
            
            {!readonly && (
              <Button
                type="primary"
                icon={<BuildOutlined />}
                onClick={() => setLibraryVisible(true)}
              >
                Add Component
              </Button>
            )}
          </Space>
        </div>

        {components.length === 0 ? (
          <div style={{ 
            textAlign: 'center', 
            padding: '60px 20px',
            border: '2px dashed #d9d9d9',
            borderRadius: '8px',
            backgroundColor: '#fafafa'
          }}>
            <CodeOutlined style={{ fontSize: 48, color: '#d9d9d9', marginBottom: 16 }} />
            <div>
              <Title level={4} type="secondary">No Components Added</Title>
              <Text type="secondary">
                Start building your strategy by adding components from the library
              </Text>
            </div>
            {!readonly && (
              <Button
                type="primary"
                icon={<BuildOutlined />}
                onClick={() => setLibraryVisible(true)}
                style={{ marginTop: 16 }}
              >
                Add Your First Component
              </Button>
            )}
          </div>
        ) : (
          <DndContext
            sensors={sensors}
            collisionDetection={closestCenter}
            onDragStart={handleDragStart}
            onDragEnd={handleDragEnd}
          >
            <SortableContext items={components.map(c => c.id)} strategy={verticalListSortingStrategy}>
              <div style={{ 
                minHeight: '400px',
                padding: '16px',
                border: '1px solid #f0f0f0',
                borderRadius: '8px',
                backgroundColor: '#fafafa'
              }}>
                {components.map(component => (
                  <SortableComponent
                    key={component.id}
                    component={component}
                    onUpdate={updateComponent}
                    onRemove={removeComponent}
                    connections={connections}
                  />
                ))}
              </div>
            </SortableContext>

            <DragOverlay>
              {activeId ? (
                <div style={{ opacity: 0.8 }}>
                  <Card size="small" style={{ border: '2px solid #1890ff' }}>
                    <Text strong>
                      {components.find(c => c.id === activeId)?.name}
                    </Text>
                  </Card>
                </div>
              ) : null}
            </DragOverlay>
          </DndContext>
        )}

        {/* Validation Summary */}
        {Object.keys(validationResults).length > 0 && (
          <Alert
            type="warning"
            message="Strategy Validation Issues"
            description={
              <Collapse ghost>
                <Panel header={`${summary.totalErrors} issues found`} key="1">
                  {Object.entries(validationResults).map(([componentId, errors]) => {
                    const component = components.find(c => c.id === componentId);
                    return (
                      <div key={componentId} style={{ marginBottom: 8 }}>
                        <Text strong>{component?.name}:</Text>
                        <ul style={{ margin: '4px 0 0 16px' }}>
                          {errors.map((error, index) => (
                            <li key={index}>{error}</li>
                          ))}
                        </ul>
                      </div>
                    );
                  })}
                </Panel>
              </Collapse>
            }
            style={{ marginTop: 16 }}
          />
        )}
      </Card>

      {/* Component Library Drawer */}
      <Drawer
        title="Component Library"
        placement="right"
        size="default"
        open={libraryVisible}
        onClose={() => setLibraryVisible(false)}
      >
        <ComponentLibrary onAddComponent={addComponent} />
      </Drawer>
    </div>
  );
};