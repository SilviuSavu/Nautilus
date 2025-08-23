import React, { useState, useEffect, useRef, useMemo, useCallback, memo } from 'react';
import {
  Card,
  Space,
  Typography,
  Button,
  Modal,
  Row,
  Col,
  Select,
  Input,
  Switch,
  Divider,
  List,
  Tag,
  Alert,
  Tooltip,
  Popover,
  Badge,
  Menu,
  Dropdown,
  Tabs,
  Form,
  InputNumber,
  ColorPicker,
  Slider
} from 'antd';
import {
  AppstoreOutlined,
  DragOutlined,
  PlusOutlined,
  SettingOutlined,
  SaveOutlined,
  LoadingOutlined,
  DeleteOutlined,
  CopyOutlined,
  FullscreenOutlined,
  ExportOutlined,
  LayoutOutlined,
  BgColorsOutlined,
  BorderOutlined
} from '@ant-design/icons';
import { DragDropContext, Droppable, Draggable, DropResult } from '@hello-pangea/dnd';
import { Responsive, WidthProvider } from 'react-grid-layout';
import 'react-grid-layout/css/styles.css';
import 'react-resizable/css/styles.css';

// Import all custom trading widgets
import AdvancedOrderBookWidget from './AdvancedOrderBookWidget';
import PnLWaterfallWidget from './PnLWaterfallWidget';
import RiskHeatmapWidget from './RiskHeatmapWidget';
import StrategyPerformanceWidget from './StrategyPerformanceWidget';
import AdvancedTradingChart from './AdvancedTradingChart';

const { Text, Title } = Typography;
const { TabPane } = Tabs;
const ResponsiveGridLayout = WidthProvider(Responsive);

interface DashboardWidget {
  id: string;
  type: 'orderbook' | 'pnl-waterfall' | 'risk-heatmap' | 'strategy-performance' | 'trading-chart' | 'custom';
  title: string;
  component: string;
  config: Record<string, any>;
  layout: {
    x: number;
    y: number;
    w: number;
    h: number;
    minW?: number;
    minH?: number;
    maxW?: number;
    maxH?: number;
  };
  style: {
    backgroundColor?: string;
    borderColor?: string;
    borderWidth?: number;
    borderRadius?: number;
    opacity?: number;
  };
  enabled: boolean;
  locked: boolean;
}

interface DashboardTemplate {
  id: string;
  name: string;
  description: string;
  category: 'trading' | 'risk' | 'performance' | 'analytics' | 'custom';
  widgets: DashboardWidget[];
  layout: 'grid' | 'masonry' | 'tabs' | 'sidebar';
  theme: 'light' | 'dark';
  responsiveBreakpoints: Record<string, number>;
}

interface CustomDashboardBuilderProps {
  initialTemplate?: DashboardTemplate;
  availableSymbols?: string[];
  availablePortfolios?: string[];
  availableStrategies?: string[];
  onSave?: (template: DashboardTemplate) => void;
  onLoad?: (templateId: string) => void;
  onExport?: (template: DashboardTemplate) => void;
  templates?: DashboardTemplate[];
  readOnly?: boolean;
  height?: number;
  theme?: 'light' | 'dark';
}

const WIDGET_TYPES = [
  {
    type: 'orderbook',
    name: 'Advanced Order Book',
    description: 'Professional order book with heatmap and volume profile',
    icon: <AppstoreOutlined />,
    category: 'trading',
    defaultConfig: {
      symbol: 'AAPL',
      depth: 20,
      showHeatmap: true,
      showVolumeProfile: true
    },
    defaultLayout: { w: 6, h: 8, minW: 4, minH: 6 }
  },
  {
    type: 'pnl-waterfall',
    name: 'P&L Waterfall Chart',
    description: 'Real-time P&L breakdown with animation',
    icon: <AppstoreOutlined />,
    category: 'performance',
    defaultConfig: {
      portfolioId: 'default',
      showBreakdown: true,
      showAnimation: true
    },
    defaultLayout: { w: 8, h: 6, minW: 6, minH: 4 }
  },
  {
    type: 'risk-heatmap',
    name: 'Risk Heatmap',
    description: 'Multi-dimensional risk visualization with drill-down',
    icon: <AppstoreOutlined />,
    category: 'risk',
    defaultConfig: {
      portfolioId: 'default',
      viewDimensions: 'symbol-risk',
      colorScheme: 'red-green'
    },
    defaultLayout: { w: 8, h: 8, minW: 6, minH: 6 }
  },
  {
    type: 'strategy-performance',
    name: 'Strategy Performance',
    description: 'Comprehensive strategy analysis and comparison',
    icon: <AppstoreOutlined />,
    category: 'performance',
    defaultConfig: {
      strategyIds: ['strategy1'],
      showComparison: true,
      showAttribution: true
    },
    defaultLayout: { w: 12, h: 10, minW: 8, minH: 8 }
  },
  {
    type: 'trading-chart',
    name: 'Advanced Trading Chart',
    description: 'Professional TradingView-style charting',
    icon: <AppstoreOutlined />,
    category: 'trading',
    defaultConfig: {
      symbol: 'AAPL',
      timeframe: '1h',
      chartType: 'candlestick',
      showVolume: true
    },
    defaultLayout: { w: 12, h: 12, minW: 8, minH: 8 }
  }
];

const DASHBOARD_TEMPLATES: DashboardTemplate[] = [
  {
    id: 'trader-pro',
    name: 'Professional Trader',
    description: 'Complete trading setup with charts, order book, and P&L',
    category: 'trading',
    widgets: [],
    layout: 'grid',
    theme: 'dark',
    responsiveBreakpoints: { lg: 1200, md: 996, sm: 768, xs: 480 }
  },
  {
    id: 'risk-manager',
    name: 'Risk Manager',
    description: 'Risk monitoring and analysis dashboard',
    category: 'risk',
    widgets: [],
    layout: 'grid',
    theme: 'light',
    responsiveBreakpoints: { lg: 1200, md: 996, sm: 768, xs: 480 }
  },
  {
    id: 'portfolio-analyst',
    name: 'Portfolio Analyst',
    description: 'Performance analysis and strategy comparison',
    category: 'performance',
    widgets: [],
    layout: 'grid',
    theme: 'light',
    responsiveBreakpoints: { lg: 1200, md: 996, sm: 768, xs: 480 }
  }
];

const CustomDashboardBuilder: React.FC<CustomDashboardBuilderProps> = memo(({
  initialTemplate,
  availableSymbols = ['AAPL', 'GOOGL', 'MSFT', 'TSLA'],
  availablePortfolios = ['default', 'aggressive', 'conservative'],
  availableStrategies = ['strategy1', 'strategy2', 'strategy3'],
  onSave,
  onLoad,
  onExport,
  templates = DASHBOARD_TEMPLATES,
  readOnly = false,
  height = 800,
  theme = 'light'
}) => {
  const [dashboard, setDashboard] = useState<DashboardTemplate>(
    initialTemplate || {
      id: 'custom',
      name: 'Custom Dashboard',
      description: 'My custom trading dashboard',
      category: 'custom',
      widgets: [],
      layout: 'grid',
      theme: 'light',
      responsiveBreakpoints: { lg: 1200, md: 996, sm: 768, xs: 480 }
    }
  );

  const [editMode, setEditMode] = useState(!readOnly);
  const [selectedWidget, setSelectedWidget] = useState<string | null>(null);
  const [widgetConfigVisible, setWidgetConfigVisible] = useState(false);
  const [templateModalVisible, setTemplateModalVisible] = useState(false);
  const [saveModalVisible, setSaveModalVisible] = useState(false);
  const [draggedWidget, setDraggedWidget] = useState<string | null>(null);
  const [layouts, setLayouts] = useState<any>({});
  const [breakpoint, setBreakpoint] = useState('lg');

  const containerRef = useRef<HTMLDivElement>(null);

  // Convert widgets to grid layout format
  const layoutItems = useMemo(() => {
    return dashboard.widgets.map(widget => ({
      i: widget.id,
      x: widget.layout.x,
      y: widget.layout.y,
      w: widget.layout.w,
      h: widget.layout.h,
      minW: widget.layout.minW,
      minH: widget.layout.minH,
      maxW: widget.layout.maxW,
      maxH: widget.layout.maxH,
      static: widget.locked || !editMode
    }));
  }, [dashboard.widgets, editMode]);

  // Add widget to dashboard
  const addWidget = useCallback((widgetType: string) => {
    const widgetTemplate = WIDGET_TYPES.find(w => w.type === widgetType);
    if (!widgetTemplate) return;

    const newWidget: DashboardWidget = {
      id: `${widgetType}-${Date.now()}`,
      type: widgetType as any,
      title: widgetTemplate.name,
      component: widgetType,
      config: { ...widgetTemplate.defaultConfig },
      layout: {
        x: 0,
        y: 0,
        ...widgetTemplate.defaultLayout
      },
      style: {
        backgroundColor: '#ffffff',
        borderColor: '#d9d9d9',
        borderWidth: 1,
        borderRadius: 6,
        opacity: 1
      },
      enabled: true,
      locked: false
    };

    // Find available position
    const existingPositions = dashboard.widgets.map(w => ({
      x: w.layout.x,
      y: w.layout.y,
      w: w.layout.w,
      h: w.layout.h
    }));

    let x = 0, y = 0;
    let positionFound = false;

    for (let row = 0; row < 20 && !positionFound; row++) {
      for (let col = 0; col <= 12 - newWidget.layout.w && !positionFound; col++) {
        const conflicts = existingPositions.some(pos => 
          !(col >= pos.x + pos.w || col + newWidget.layout.w <= pos.x ||
            row >= pos.y + pos.h || row + newWidget.layout.h <= pos.y)
        );
        
        if (!conflicts) {
          x = col;
          y = row;
          positionFound = true;
        }
      }
    }

    newWidget.layout.x = x;
    newWidget.layout.y = y;

    setDashboard(prev => ({
      ...prev,
      widgets: [...prev.widgets, newWidget]
    }));
  }, [dashboard.widgets]);

  // Remove widget
  const removeWidget = useCallback((widgetId: string) => {
    setDashboard(prev => ({
      ...prev,
      widgets: prev.widgets.filter(w => w.id !== widgetId)
    }));
  }, []);

  // Update widget config
  const updateWidgetConfig = useCallback((widgetId: string, config: Record<string, any>) => {
    setDashboard(prev => ({
      ...prev,
      widgets: prev.widgets.map(w => 
        w.id === widgetId ? { ...w, config: { ...w.config, ...config } } : w
      )
    }));
  }, []);

  // Handle layout changes
  const handleLayoutChange = useCallback((newLayout: any[]) => {
    if (!editMode) return;

    setDashboard(prev => ({
      ...prev,
      widgets: prev.widgets.map(widget => {
        const layoutItem = newLayout.find(item => item.i === widget.id);
        if (layoutItem) {
          return {
            ...widget,
            layout: {
              ...widget.layout,
              x: layoutItem.x,
              y: layoutItem.y,
              w: layoutItem.w,
              h: layoutItem.h
            }
          };
        }
        return widget;
      })
    }));
  }, [editMode]);

  // Handle responsive breakpoint change
  const handleBreakpointChange = useCallback((newBreakpoint: string, newCols: number) => {
    setBreakpoint(newBreakpoint);
  }, []);

  // Render widget component
  const renderWidget = useCallback((widget: DashboardWidget) => {
    const commonProps = {
      key: widget.id,
      compactMode: false,
      theme: dashboard.theme,
      exportEnabled: true
    };

    switch (widget.type) {
      case 'orderbook':
        return (
          <AdvancedOrderBookWidget
            {...commonProps}
            symbol={widget.config.symbol || 'AAPL'}
            depth={widget.config.depth || 20}
            showHeatmap={widget.config.showHeatmap !== false}
            showVolumeProfile={widget.config.showVolumeProfile !== false}
            height={300}
          />
        );

      case 'pnl-waterfall':
        return (
          <PnLWaterfallWidget
            {...commonProps}
            portfolioId={widget.config.portfolioId || 'default'}
            showBreakdown={widget.config.showBreakdown !== false}
            showAnimation={widget.config.showAnimation !== false}
            height={300}
          />
        );

      case 'risk-heatmap':
        return (
          <RiskHeatmapWidget
            {...commonProps}
            portfolioId={widget.config.portfolioId || 'default'}
            viewDimensions={widget.config.viewDimensions || 'symbol-risk'}
            colorScheme={widget.config.colorScheme || 'red-green'}
            height={400}
          />
        );

      case 'strategy-performance':
        return (
          <StrategyPerformanceWidget
            {...commonProps}
            strategyIds={widget.config.strategyIds || ['strategy1']}
            showComparison={widget.config.showComparison !== false}
            showAttribution={widget.config.showAttribution !== false}
            height={500}
          />
        );

      case 'trading-chart':
        return (
          <AdvancedTradingChart
            {...commonProps}
            symbol={widget.config.symbol || 'AAPL'}
            timeframe={widget.config.timeframe || '1h'}
            chartType={widget.config.chartType || 'candlestick'}
            showVolume={widget.config.showVolume !== false}
            height={600}
          />
        );

      default:
        return (
          <Card title={widget.title} style={{ height: '100%' }}>
            <Text>Unknown widget type: {widget.type}</Text>
          </Card>
        );
    }
  }, [dashboard.theme]);

  // Widget configuration form
  const renderWidgetConfig = () => {
    if (!selectedWidget) return null;
    
    const widget = dashboard.widgets.find(w => w.id === selectedWidget);
    if (!widget) return null;

    return (
      <Modal
        title={`Configure ${widget.title}`}
        open={widgetConfigVisible}
        onCancel={() => setWidgetConfigVisible(false)}
        onOk={() => setWidgetConfigVisible(false)}
        width={600}
      >
        <Tabs defaultActiveKey="config">
          <TabPane tab="Configuration" key="config">
            <Form layout="vertical">
              {widget.type === 'orderbook' && (
                <>
                  <Form.Item label="Symbol">
                    <Select
                      value={widget.config.symbol}
                      onChange={(value) => updateWidgetConfig(widget.id, { symbol: value })}
                      options={availableSymbols.map(s => ({ label: s, value: s }))}
                    />
                  </Form.Item>
                  <Form.Item label="Depth">
                    <InputNumber
                      value={widget.config.depth}
                      onChange={(value) => updateWidgetConfig(widget.id, { depth: value })}
                      min={5}
                      max={50}
                    />
                  </Form.Item>
                  <Form.Item label="Show Heatmap">
                    <Switch
                      checked={widget.config.showHeatmap}
                      onChange={(checked) => updateWidgetConfig(widget.id, { showHeatmap: checked })}
                    />
                  </Form.Item>
                </>
              )}

              {widget.type === 'pnl-waterfall' && (
                <>
                  <Form.Item label="Portfolio">
                    <Select
                      value={widget.config.portfolioId}
                      onChange={(value) => updateWidgetConfig(widget.id, { portfolioId: value })}
                      options={availablePortfolios.map(p => ({ label: p, value: p }))}
                    />
                  </Form.Item>
                  <Form.Item label="Show Animation">
                    <Switch
                      checked={widget.config.showAnimation}
                      onChange={(checked) => updateWidgetConfig(widget.id, { showAnimation: checked })}
                    />
                  </Form.Item>
                </>
              )}

              {widget.type === 'trading-chart' && (
                <>
                  <Form.Item label="Symbol">
                    <Select
                      value={widget.config.symbol}
                      onChange={(value) => updateWidgetConfig(widget.id, { symbol: value })}
                      options={availableSymbols.map(s => ({ label: s, value: s }))}
                    />
                  </Form.Item>
                  <Form.Item label="Timeframe">
                    <Select
                      value={widget.config.timeframe}
                      onChange={(value) => updateWidgetConfig(widget.id, { timeframe: value })}
                      options={[
                        { label: '1m', value: '1m' },
                        { label: '5m', value: '5m' },
                        { label: '15m', value: '15m' },
                        { label: '1h', value: '1h' },
                        { label: '4h', value: '4h' },
                        { label: '1d', value: '1d' }
                      ]}
                    />
                  </Form.Item>
                  <Form.Item label="Chart Type">
                    <Select
                      value={widget.config.chartType}
                      onChange={(value) => updateWidgetConfig(widget.id, { chartType: value })}
                      options={[
                        { label: 'Candlestick', value: 'candlestick' },
                        { label: 'Line', value: 'line' },
                        { label: 'Area', value: 'area' }
                      ]}
                    />
                  </Form.Item>
                </>
              )}
            </Form>
          </TabPane>

          <TabPane tab="Style" key="style">
            <Form layout="vertical">
              <Form.Item label="Background Color">
                <ColorPicker
                  value={widget.style.backgroundColor}
                  onChange={(color) => updateWidgetConfig(widget.id, { 
                    style: { ...widget.style, backgroundColor: color.toHexString() }
                  })}
                />
              </Form.Item>
              <Form.Item label="Border Width">
                <Slider
                  min={0}
                  max={5}
                  value={widget.style.borderWidth}
                  onChange={(value) => updateWidgetConfig(widget.id, {
                    style: { ...widget.style, borderWidth: value }
                  })}
                />
              </Form.Item>
              <Form.Item label="Border Radius">
                <Slider
                  min={0}
                  max={20}
                  value={widget.style.borderRadius}
                  onChange={(value) => updateWidgetConfig(widget.id, {
                    style: { ...widget.style, borderRadius: value }
                  })}
                />
              </Form.Item>
              <Form.Item label="Opacity">
                <Slider
                  min={0.1}
                  max={1}
                  step={0.1}
                  value={widget.style.opacity}
                  onChange={(value) => updateWidgetConfig(widget.id, {
                    style: { ...widget.style, opacity: value }
                  })}
                />
              </Form.Item>
            </Form>
          </TabPane>
        </Tabs>
      </Modal>
    );
  };

  // Widget toolbar
  const renderWidgetToolbar = (widget: DashboardWidget) => {
    if (!editMode) return null;

    return (
      <div style={{
        position: 'absolute',
        top: 5,
        right: 5,
        zIndex: 1000,
        display: 'flex',
        gap: 4
      }}>
        <Tooltip title="Configure">
          <Button
            size="small"
            icon={<SettingOutlined />}
            onClick={() => {
              setSelectedWidget(widget.id);
              setWidgetConfigVisible(true);
            }}
            style={{ opacity: 0.8 }}
          />
        </Tooltip>
        <Tooltip title="Delete">
          <Button
            size="small"
            icon={<DeleteOutlined />}
            danger
            onClick={() => removeWidget(widget.id)}
            style={{ opacity: 0.8 }}
          />
        </Tooltip>
      </div>
    );
  };

  // Save dashboard
  const saveDashboard = useCallback(() => {
    onSave?.(dashboard);
    setSaveModalVisible(false);
  }, [dashboard, onSave]);

  // Export dashboard
  const exportDashboard = useCallback(() => {
    const dataStr = JSON.stringify(dashboard, null, 2);
    const dataBlob = new Blob([dataStr], { type: 'application/json' });
    const url = URL.createObjectURL(dataBlob);
    const link = document.createElement('a');
    link.href = url;
    link.download = `dashboard-${dashboard.name}-${Date.now()}.json`;
    link.click();
    URL.revokeObjectURL(url);
  }, [dashboard]);

  return (
    <div ref={containerRef} style={{ height, overflow: 'auto' }}>
      <Card
        title={
          <Space>
            <LayoutOutlined />
            <Text strong>{dashboard.name}</Text>
            <Badge count={dashboard.widgets.length} style={{ backgroundColor: '#52c41a' }} />
            {editMode && <Tag color="orange">Edit Mode</Tag>}
          </Space>
        }
        extra={
          <Space>
            {/* Template Selector */}
            <Button size="small" onClick={() => setTemplateModalVisible(true)}>
              Templates
            </Button>

            {/* Edit Mode Toggle */}
            <Switch
              checked={editMode}
              onChange={setEditMode}
              checkedChildren="Edit"
              unCheckedChildren="View"
              disabled={readOnly}
            />

            {/* Save */}
            <Button
              size="small"
              icon={<SaveOutlined />}
              onClick={() => setSaveModalVisible(true)}
              disabled={readOnly}
            >
              Save
            </Button>

            {/* Export */}
            <Button
              size="small"
              icon={<ExportOutlined />}
              onClick={exportDashboard}
            >
              Export
            </Button>
          </Space>
        }
        style={{ margin: 0 }}
      >
        {/* Widget Palette */}
        {editMode && (
          <Card size="small" style={{ marginBottom: 16 }}>
            <Title level={5}>Available Widgets</Title>
            <Row gutter={[8, 8]}>
              {WIDGET_TYPES.map(widgetType => (
                <Col key={widgetType.type}>
                  <Button
                    icon={widgetType.icon}
                    onClick={() => addWidget(widgetType.type)}
                    size="small"
                    style={{ width: 160 }}
                  >
                    {widgetType.name}
                  </Button>
                </Col>
              ))}
            </Row>
          </Card>
        )}

        {/* Dashboard Grid */}
        <ResponsiveGridLayout
          className="layout"
          layouts={layouts}
          breakpoints={dashboard.responsiveBreakpoints}
          cols={{ lg: 12, md: 10, sm: 6, xs: 4 }}
          rowHeight={60}
          width={containerRef.current?.clientWidth || 1200}
          onLayoutChange={handleLayoutChange}
          onBreakpointChange={handleBreakpointChange}
          isDraggable={editMode}
          isResizable={editMode}
          margin={[16, 16]}
          useCSSTransforms={true}
        >
          {dashboard.widgets.map(widget => (
            <div
              key={widget.id}
              style={{
                backgroundColor: widget.style.backgroundColor,
                borderColor: widget.style.borderColor,
                borderWidth: widget.style.borderWidth,
                borderRadius: widget.style.borderRadius,
                opacity: widget.style.opacity,
                borderStyle: 'solid',
                position: 'relative',
                overflow: 'hidden'
              }}
            >
              {renderWidgetToolbar(widget)}
              {renderWidget(widget)}
            </div>
          ))}
        </ResponsiveGridLayout>

        {/* Empty State */}
        {dashboard.widgets.length === 0 && (
          <div style={{
            textAlign: 'center',
            padding: '60px 20px',
            background: '#fafafa',
            borderRadius: 8,
            border: '2px dashed #d9d9d9'
          }}>
            <AppstoreOutlined style={{ fontSize: 48, color: '#d9d9d9', marginBottom: 16 }} />
            <Title level={4} type="secondary">No widgets added</Title>
            <Text type="secondary">
              {editMode 
                ? 'Click the widget buttons above to add components to your dashboard'
                : 'Switch to edit mode to customize this dashboard'
              }
            </Text>
          </div>
        )}
      </Card>

      {/* Widget Configuration Modal */}
      {renderWidgetConfig()}

      {/* Template Selection Modal */}
      <Modal
        title="Dashboard Templates"
        open={templateModalVisible}
        onCancel={() => setTemplateModalVisible(false)}
        footer={null}
        width={800}
      >
        <Row gutter={[16, 16]}>
          {templates.map(template => (
            <Col span={8} key={template.id}>
              <Card
                size="small"
                hoverable
                onClick={() => {
                  setDashboard(template);
                  setTemplateModalVisible(false);
                }}
                cover={
                  <div style={{
                    height: 120,
                    background: `linear-gradient(135deg, ${template.category === 'trading' ? '#1890ff' : 
                                                      template.category === 'risk' ? '#f5222d' : 
                                                      template.category === 'performance' ? '#52c41a' : '#722ed1'} 0%, rgba(255,255,255,0.1) 100%)`,
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center'
                  }}>
                    <LayoutOutlined style={{ fontSize: 32, color: 'white' }} />
                  </div>
                }
              >
                <Card.Meta
                  title={template.name}
                  description={template.description}
                />
                <div style={{ marginTop: 8 }}>
                  <Tag color={template.category === 'trading' ? 'blue' : 
                            template.category === 'risk' ? 'red' : 
                            template.category === 'performance' ? 'green' : 'purple'}>
                    {template.category}
                  </Tag>
                  <Text type="secondary" style={{ fontSize: '11px' }}>
                    {template.widgets.length} widgets
                  </Text>
                </div>
              </Card>
            </Col>
          ))}
        </Row>
      </Modal>

      {/* Save Modal */}
      <Modal
        title="Save Dashboard"
        open={saveModalVisible}
        onCancel={() => setSaveModalVisible(false)}
        onOk={saveDashboard}
      >
        <Form layout="vertical">
          <Form.Item label="Dashboard Name">
            <Input
              value={dashboard.name}
              onChange={(e) => setDashboard(prev => ({ ...prev, name: e.target.value }))}
              placeholder="Enter dashboard name"
            />
          </Form.Item>
          <Form.Item label="Description">
            <Input.TextArea
              value={dashboard.description}
              onChange={(e) => setDashboard(prev => ({ ...prev, description: e.target.value }))}
              placeholder="Enter dashboard description"
              rows={3}
            />
          </Form.Item>
          <Form.Item label="Category">
            <Select
              value={dashboard.category}
              onChange={(value) => setDashboard(prev => ({ ...prev, category: value }))}
              options={[
                { label: 'Trading', value: 'trading' },
                { label: 'Risk Management', value: 'risk' },
                { label: 'Performance', value: 'performance' },
                { label: 'Analytics', value: 'analytics' },
                { label: 'Custom', value: 'custom' }
              ]}
            />
          </Form.Item>
        </Form>
      </Modal>
    </div>
  );
});

CustomDashboardBuilder.displayName = 'CustomDashboardBuilder';

export default CustomDashboardBuilder;