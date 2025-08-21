import React, { useRef, useEffect, useState, useMemo } from 'react';
import { Card, Button, Typography, Space, Select, Tooltip, Alert, Switch } from 'antd';
import {
  ZoomInOutlined,
  ZoomOutOutlined,
  ReloadOutlined,
  FullscreenOutlined,
  DownloadOutlined,
  PlayCircleOutlined,
  PauseCircleOutlined
} from '@ant-design/icons';
import { StrategyComponent, Connection, ComponentInput, ComponentOutput } from './VisualStrategyBuilder';

const { Title, Text } = Typography;
const { Option } = Select;

interface Node {
  id: string;
  x: number;
  y: number;
  width: number;
  height: number;
  component: StrategyComponent;
}

interface Edge {
  id: string;
  source: Node;
  target: Node;
  sourceOutput: ComponentOutput;
  targetInput: ComponentInput;
  connection: Connection;
}

interface FlowVisualizationProps {
  components: StrategyComponent[];
  connections: Connection[];
  className?: string;
  width?: number;
  height?: number;
  onComponentClick?: (component: StrategyComponent) => void;
  onConnectionClick?: (connection: Connection) => void;
  realTimeData?: Record<string, any>;
  isSimulating?: boolean;
}

export const StrategyFlowVisualization: React.FC<FlowVisualizationProps> = ({
  components,
  connections,
  className,
  width = 800,
  height = 600,
  onComponentClick,
  onConnectionClick,
  realTimeData = {},
  isSimulating = false
}) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const [zoom, setZoom] = useState(1);
  const [pan, setPan] = useState({ x: 0, y: 0 });
  const [selectedComponent, setSelectedComponent] = useState<string | null>(null);
  const [layoutAlgorithm, setLayoutAlgorithm] = useState<'hierarchical' | 'force' | 'circular'>('hierarchical');
  const [showDataFlow, setShowDataFlow] = useState(false);
  const [animationSpeed, setAnimationSpeed] = useState(1);

  // Calculate node positions using different layout algorithms
  const { nodes, edges } = useMemo(() => {
    const nodeMap = new Map<string, Node>();
    const edgeList: Edge[] = [];

    // Create nodes
    components.forEach((component, index) => {
      let position = calculateNodePosition(component, index, components.length, layoutAlgorithm);
      
      const node: Node = {
        id: component.id,
        x: position.x,
        y: position.y,
        width: 180,
        height: 100,
        component
      };
      nodeMap.set(component.id, node);
    });

    // Create edges
    connections.forEach(connection => {
      const sourceNode = nodeMap.get(connection.sourceId);
      const targetNode = nodeMap.get(connection.targetId);
      
      if (sourceNode && targetNode) {
        const sourceOutput = sourceNode.component.outputs.find(o => o.id === connection.sourceOutput);
        const targetInput = targetNode.component.inputs.find(i => i.id === connection.targetInput);
        
        if (sourceOutput && targetInput) {
          edgeList.push({
            id: connection.id,
            source: sourceNode,
            target: targetNode,
            sourceOutput,
            targetInput,
            connection
          });
        }
      }
    });

    return { nodes: Array.from(nodeMap.values()), edges: edgeList };
  }, [components, connections, layoutAlgorithm]);

  // Calculate node position based on layout algorithm
  function calculateNodePosition(
    component: StrategyComponent, 
    index: number, 
    total: number, 
    algorithm: string
  ): { x: number; y: number } {
    const centerX = width / 2;
    const centerY = height / 2;
    const padding = 200;

    switch (algorithm) {
      case 'hierarchical':
        // Arrange by component type in layers
        const layerMap: Record<string, number> = {
          'indicator': 0,
          'signal': 1,
          'condition': 2,
          'action': 3,
          'risk_control': 4
        };
        const layer = layerMap[component.type] || 0;
        const itemsInLayer = components.filter(c => c.type === component.type).length;
        const layerIndex = components.filter(c => c.type === component.type).indexOf(component);
        
        return {
          x: padding + (layerIndex * (width - 2 * padding)) / Math.max(1, itemsInLayer - 1),
          y: padding + (layer * (height - 2 * padding)) / 4
        };

      case 'force':
        // Simple force-directed positioning (simplified)
        const angle = (index / total) * 2 * Math.PI;
        const radius = Math.min(width, height) / 4;
        return {
          x: centerX + Math.cos(angle) * radius,
          y: centerY + Math.sin(angle) * radius
        };

      case 'circular':
        // Circular arrangement
        const circleAngle = (index / total) * 2 * Math.PI;
        const circleRadius = Math.min(width, height) / 3;
        return {
          x: centerX + Math.cos(circleAngle) * circleRadius,
          y: centerY + Math.sin(circleAngle) * circleRadius
        };

      default:
        return { x: 100 + (index % 4) * 200, y: 100 + Math.floor(index / 4) * 150 };
    }
  }

  // Drawing functions
  const drawNode = (ctx: CanvasRenderingContext2D, node: Node) => {
    const { x, y, width: w, height: h, component } = node;
    const isSelected = selectedComponent === component.id;
    const hasData = realTimeData[component.id];

    // Node background
    ctx.fillStyle = isSelected ? '#e6f7ff' : component.isValid ? '#f6ffed' : '#fff2f0';
    ctx.strokeStyle = isSelected ? '#1890ff' : component.isValid ? '#52c41a' : '#ff4d4f';
    ctx.lineWidth = isSelected ? 3 : 2;
    
    // Rounded rectangle
    roundRect(ctx, x, y, w, h, 8);
    ctx.fill();
    ctx.stroke();

    // Component type indicator
    const typeColors: Record<string, string> = {
      'indicator': '#1890ff',
      'signal': '#52c41a',
      'condition': '#fa8c16',
      'action': '#722ed1',
      'risk_control': '#f5222d'
    };
    
    ctx.fillStyle = typeColors[component.type] || '#666666';
    ctx.fillRect(x + 2, y + 2, 6, h - 4);

    // Component name
    ctx.fillStyle = '#000000';
    ctx.font = 'bold 14px Arial';
    ctx.textAlign = 'left';
    ctx.fillText(component.name, x + 15, y + 20);

    // Component description
    ctx.fillStyle = '#666666';
    ctx.font = '12px Arial';
    const descWords = component.description.split(' ');
    let line1 = '', line2 = '';
    
    for (let i = 0; i < descWords.length; i++) {
      const testLine = line1 + descWords[i] + ' ';
      if (ctx.measureText(testLine).width > w - 20 && line1 !== '') {
        line2 = descWords.slice(i).join(' ');
        break;
      }
      line1 = testLine;
    }
    
    ctx.fillText(line1.trim(), x + 15, y + 38);
    if (line2) {
      ctx.fillText(line2.length > 20 ? line2.substring(0, 17) + '...' : line2, x + 15, y + 52);
    }

    // Real-time data indicator
    if (hasData && showDataFlow) {
      ctx.fillStyle = '#52c41a';
      ctx.beginPath();
      ctx.arc(x + w - 15, y + 15, 5, 0, 2 * Math.PI);
      ctx.fill();
      
      // Pulse animation
      const pulseRadius = 5 + Math.sin(Date.now() / 200) * 2;
      ctx.strokeStyle = '#52c41a';
      ctx.lineWidth = 2;
      ctx.globalAlpha = 0.6;
      ctx.beginPath();
      ctx.arc(x + w - 15, y + 15, pulseRadius, 0, 2 * Math.PI);
      ctx.stroke();
      ctx.globalAlpha = 1;
    }

    // Input/Output ports
    component.inputs.forEach((input, index) => {
      const portY = y + 25 + index * 15;
      ctx.fillStyle = input.connected ? '#52c41a' : '#d9d9d9';
      ctx.beginPath();
      ctx.arc(x - 5, portY, 4, 0, 2 * Math.PI);
      ctx.fill();
    });

    component.outputs.forEach((output, index) => {
      const portY = y + 25 + index * 15;
      ctx.fillStyle = '#1890ff';
      ctx.beginPath();
      ctx.arc(x + w + 5, portY, 4, 0, 2 * Math.PI);
      ctx.fill();
    });
  };

  const drawEdge = (ctx: CanvasRenderingContext2D, edge: Edge) => {
    const { source, target, sourceOutput, targetInput } = edge;
    
    // Find port positions
    const sourcePortIndex = source.component.outputs.findIndex(o => o.id === sourceOutput.id);
    const targetPortIndex = target.component.inputs.findIndex(i => i.id === targetInput.id);
    
    const startX = source.x + source.width + 5;
    const startY = source.y + 25 + sourcePortIndex * 15;
    const endX = target.x - 5;
    const endY = target.y + 25 + targetPortIndex * 15;

    // Connection line with bezier curve
    const controlPoint1X = startX + (endX - startX) * 0.6;
    const controlPoint1Y = startY;
    const controlPoint2X = endX - (endX - startX) * 0.6;
    const controlPoint2Y = endY;

    // Line style based on data type
    const typeColors: Record<string, string> = {
      'number': '#1890ff',
      'boolean': '#52c41a',
      'signal': '#722ed1',
      'price_data': '#fa8c16'
    };

    ctx.strokeStyle = typeColors[sourceOutput.type] || '#666666';
    ctx.lineWidth = 2;
    
    // Animate data flow
    if (showDataFlow && isSimulating) {
      const dashOffset = (Date.now() * animationSpeed / 10) % 20;
      ctx.setLineDash([10, 10]);
      ctx.lineDashOffset = -dashOffset;
    } else {
      ctx.setLineDash([]);
    }

    ctx.beginPath();
    ctx.moveTo(startX, startY);
    ctx.bezierCurveTo(controlPoint1X, controlPoint1Y, controlPoint2X, controlPoint2Y, endX, endY);
    ctx.stroke();

    // Arrow head
    const angle = Math.atan2(endY - controlPoint2Y, endX - controlPoint2X);
    const arrowLength = 8;
    
    ctx.setLineDash([]);
    ctx.fillStyle = ctx.strokeStyle;
    ctx.beginPath();
    ctx.moveTo(endX, endY);
    ctx.lineTo(
      endX - arrowLength * Math.cos(angle - Math.PI / 6),
      endY - arrowLength * Math.sin(angle - Math.PI / 6)
    );
    ctx.lineTo(
      endX - arrowLength * Math.cos(angle + Math.PI / 6),
      endY - arrowLength * Math.sin(angle + Math.PI / 6)
    );
    ctx.closePath();
    ctx.fill();
  };

  const roundRect = (
    ctx: CanvasRenderingContext2D,
    x: number,
    y: number,
    width: number,
    height: number,
    radius: number
  ) => {
    ctx.beginPath();
    ctx.moveTo(x + radius, y);
    ctx.lineTo(x + width - radius, y);
    ctx.quadraticCurveTo(x + width, y, x + width, y + radius);
    ctx.lineTo(x + width, y + height - radius);
    ctx.quadraticCurveTo(x + width, y + height, x + width - radius, y + height);
    ctx.lineTo(x + radius, y + height);
    ctx.quadraticCurveTo(x, y + height, x, y + height - radius);
    ctx.lineTo(x, y + radius);
    ctx.quadraticCurveTo(x, y, x + radius, y);
    ctx.closePath();
  };

  // Render canvas
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    // Clear canvas
    ctx.clearRect(0, 0, width, height);

    // Apply zoom and pan
    ctx.save();
    ctx.scale(zoom, zoom);
    ctx.translate(pan.x, pan.y);

    // Draw grid
    drawGrid(ctx);

    // Draw edges first (behind nodes)
    edges.forEach(edge => drawEdge(ctx, edge));

    // Draw nodes
    nodes.forEach(node => drawNode(ctx, node));

    ctx.restore();
  }, [nodes, edges, zoom, pan, selectedComponent, showDataFlow, isSimulating, animationSpeed, realTimeData]);

  const drawGrid = (ctx: CanvasRenderingContext2D) => {
    const gridSize = 50;
    ctx.strokeStyle = '#f0f0f0';
    ctx.lineWidth = 1;

    for (let x = 0; x <= width; x += gridSize) {
      ctx.beginPath();
      ctx.moveTo(x, 0);
      ctx.lineTo(x, height);
      ctx.stroke();
    }

    for (let y = 0; y <= height; y += gridSize) {
      ctx.beginPath();
      ctx.moveTo(0, y);
      ctx.lineTo(width, y);
      ctx.stroke();
    }
  };

  // Event handlers
  const handleCanvasClick = (event: React.MouseEvent) => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const rect = canvas.getBoundingClientRect();
    const x = (event.clientX - rect.left) / zoom - pan.x;
    const y = (event.clientY - rect.top) / zoom - pan.y;

    // Find clicked node
    const clickedNode = nodes.find(node => 
      x >= node.x && x <= node.x + node.width &&
      y >= node.y && y <= node.y + node.height
    );

    if (clickedNode) {
      setSelectedComponent(clickedNode.id);
      onComponentClick?.(clickedNode.component);
    } else {
      setSelectedComponent(null);
    }
  };

  const handleZoomIn = () => setZoom(prev => Math.min(prev * 1.2, 3));
  const handleZoomOut = () => setZoom(prev => Math.max(prev / 1.2, 0.2));
  const handleReset = () => {
    setZoom(1);
    setPan({ x: 0, y: 0 });
    setSelectedComponent(null);
  };

  const exportCanvas = () => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const link = document.createElement('a');
    link.download = 'strategy-flow.png';
    link.href = canvas.toDataURL();
    link.click();
  };

  return (
    <div className={`strategy-flow-visualization ${className || ''}`} ref={containerRef}>
      <Card>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 16 }}>
          <div>
            <Title level={4} style={{ margin: 0 }}>
              Strategy Flow Visualization
            </Title>
            <Text type="secondary">
              Interactive visualization of strategy components and data flow
            </Text>
          </div>

          <Space>
            <Select
              value={layoutAlgorithm}
              onChange={setLayoutAlgorithm}
              style={{ width: 120 }}
              size="small"
            >
              <Option value="hierarchical">Hierarchical</Option>
              <Option value="force">Force</Option>
              <Option value="circular">Circular</Option>
            </Select>

            <Tooltip title="Show Data Flow">
              <Switch
                checked={showDataFlow}
                onChange={setShowDataFlow}
                checkedChildren="Flow"
                unCheckedChildren="Static"
                size="small"
              />
            </Tooltip>

            <Space.Compact>
              <Button icon={<ZoomOutOutlined />} onClick={handleZoomOut} size="small" />
              <Button icon={<ZoomInOutlined />} onClick={handleZoomIn} size="small" />
              <Button icon={<ReloadOutlined />} onClick={handleReset} size="small" />
            </Space.Compact>

            <Button icon={<DownloadOutlined />} onClick={exportCanvas} size="small">
              Export
            </Button>
          </Space>
        </div>

        {components.length === 0 ? (
          <div style={{ 
            textAlign: 'center', 
            padding: '60px 20px',
            border: '2px dashed #d9d9d9',
            borderRadius: '8px'
          }}>
            <Text type="secondary">
              Add components to see the strategy flow visualization
            </Text>
          </div>
        ) : (
          <div style={{ position: 'relative', border: '1px solid #d9d9d9', borderRadius: '8px' }}>
            <canvas
              ref={canvasRef}
              width={width}
              height={height}
              onClick={handleCanvasClick}
              style={{ cursor: 'pointer', display: 'block' }}
            />
            
            {selectedComponent && (
              <div style={{
                position: 'absolute',
                top: 10,
                right: 10,
                background: 'rgba(255, 255, 255, 0.95)',
                padding: '8px 12px',
                borderRadius: '6px',
                border: '1px solid #d9d9d9',
                maxWidth: 200
              }}>
                <Text strong>
                  {nodes.find(n => n.id === selectedComponent)?.component.name}
                </Text>
                <br />
                <Text type="secondary" style={{ fontSize: 12 }}>
                  Click outside to deselect
                </Text>
              </div>
            )}
          </div>
        )}

        {showDataFlow && isSimulating && (
          <Alert
            type="info"
            message="Real-time Simulation Active"
            description="Green indicators show components receiving real-time data. Animated connections show data flow direction."
            showIcon
            style={{ marginTop: 16 }}
          />
        )}
      </Card>
    </div>
  );
};