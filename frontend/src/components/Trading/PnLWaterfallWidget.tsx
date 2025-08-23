import React, { useState, useEffect, useRef, useMemo, useCallback, memo } from 'react';
import {
  Card,
  Space,
  Typography,
  Select,
  Button,
  Tooltip,
  Badge,
  Row,
  Col,
  Statistic,
  Switch,
  DatePicker,
  Slider,
  Dropdown,
  Menu,
  Progress,
  Alert
} from 'antd';
import {
  BarChartOutlined,
  LineChartOutlined,
  SettingOutlined,
  ExportOutlined,
  FullscreenOutlined,
  PlayCircleOutlined,
  PauseCircleOutlined,
  ReloadOutlined,
  FilterOutlined
} from '@ant-design/icons';
import * as d3 from 'd3';
import moment from 'moment';
import { useRealTimeData } from '../../hooks/useRealTimeData';

const { Text, Title } = Typography;
const { RangePicker } = DatePicker;

interface PnLComponent {
  id: string;
  name: string;
  value: number;
  previousValue: number;
  change: number;
  changePercent: number;
  category: 'trading' | 'fees' | 'funding' | 'other';
  subComponents?: PnLComponent[];
  timestamp: number;
  metadata?: Record<string, any>;
}

interface PnLWaterfallData {
  portfolioId: string;
  timestamp: number;
  totalPnL: number;
  previousTotalPnL: number;
  components: PnLComponent[];
  cumulativeChanges: number[];
  breakdown: {
    gross: number;
    net: number;
    fees: number;
    slippage: number;
    funding: number;
  };
}

interface PnLWaterfallWidgetProps {
  portfolioId: string;
  timeRange?: [moment.Moment, moment.Moment];
  granularity?: 'minute' | 'hour' | 'day';
  showBreakdown?: boolean;
  showAnimation?: boolean;
  compactMode?: boolean;
  height?: number;
  autoRefresh?: boolean;
  refreshInterval?: number;
  theme?: 'light' | 'dark';
  onComponentClick?: (component: PnLComponent) => void;
  onDrillDown?: (component: PnLComponent) => void;
  exportEnabled?: boolean;
  alertThresholds?: {
    loss: number;
    gain: number;
  };
}

interface WaterfallBar {
  id: string;
  name: string;
  value: number;
  cumulative: number;
  start: number;
  end: number;
  type: 'positive' | 'negative' | 'total' | 'starting';
  category: string;
  isAnimating: boolean;
}

const PnLWaterfallWidget: React.FC<PnLWaterfallWidgetProps> = memo(({
  portfolioId,
  timeRange,
  granularity = 'hour',
  showBreakdown = true,
  showAnimation = true,
  compactMode = false,
  height = 500,
  autoRefresh = true,
  refreshInterval = 5000,
  theme = 'light',
  onComponentClick,
  onDrillDown,
  exportEnabled = true,
  alertThresholds = { loss: -1000, gain: 1000 }
}) => {
  const [viewMode, setViewMode] = useState<'waterfall' | 'breakdown' | 'timeline'>('waterfall');
  const [animationSpeed, setAnimationSpeed] = useState(1000);
  const [isPlaying, setIsPlaying] = useState(showAnimation);
  const [selectedComponent, setSelectedComponent] = useState<PnLComponent | null>(null);
  const [filterCategory, setFilterCategory] = useState<string>('all');
  const [fullscreen, setFullscreen] = useState(false);
  const [showAlerts, setShowAlerts] = useState(true);

  const svgRef = useRef<SVGSVGElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const animationRef = useRef<number>();
  const waterfallData = useRef<WaterfallBar[]>([]);

  // Use real-time data hook for P&L updates
  const {
    data: rawPnLData,
    isConnected,
    lastUpdate,
    error
  } = useRealTimeData({
    type: 'pnl',
    portfolioId,
    granularity,
    autoRefresh
  });

  // Process P&L data into waterfall format
  const processedData = useMemo(() => {
    if (!rawPnLData) return null;

    const components = rawPnLData.components || [];
    
    // Filter components by category if specified
    const filteredComponents = filterCategory === 'all' 
      ? components 
      : components.filter((comp: PnLComponent) => comp.category === filterCategory);

    // Build waterfall structure
    let cumulative = 0;
    const waterfallBars: WaterfallBar[] = [];

    // Starting value
    waterfallBars.push({
      id: 'starting',
      name: 'Starting P&L',
      value: rawPnLData.previousTotalPnL,
      cumulative: rawPnLData.previousTotalPnL,
      start: 0,
      end: rawPnLData.previousTotalPnL,
      type: 'starting',
      category: 'base',
      isAnimating: false
    });

    cumulative = rawPnLData.previousTotalPnL;

    // Add each component
    filteredComponents.forEach((component: PnLComponent) => {
      const start = cumulative;
      const end = cumulative + component.change;
      
      waterfallBars.push({
        id: component.id,
        name: component.name,
        value: component.change,
        cumulative: end,
        start: Math.min(start, end),
        end: Math.max(start, end),
        type: component.change >= 0 ? 'positive' : 'negative',
        category: component.category,
        isAnimating: isPlaying
      });

      cumulative = end;
    });

    // Final total
    waterfallBars.push({
      id: 'ending',
      name: 'Ending P&L',
      value: rawPnLData.totalPnL,
      cumulative: rawPnLData.totalPnL,
      start: 0,
      end: rawPnLData.totalPnL,
      type: 'total',
      category: 'total',
      isAnimating: false
    });

    waterfallData.current = waterfallBars;

    return {
      ...rawPnLData,
      waterfallBars,
      filteredComponents
    };
  }, [rawPnLData, filterCategory, isPlaying]);

  // Render waterfall chart
  const renderWaterfallChart = useCallback(() => {
    if (!processedData || !svgRef.current) return;

    const svg = d3.select(svgRef.current);
    svg.selectAll('*').remove();

    const margin = { top: 30, right: 30, bottom: 60, left: 80 };
    const width = (containerRef.current?.clientWidth || 800) - margin.left - margin.right;
    const chartHeight = height - margin.top - margin.bottom;

    const g = svg.append('g')
      .attr('transform', `translate(${margin.left},${margin.top})`);

    const data = processedData.waterfallBars;
    
    // Scales
    const xScale = d3.scaleBand()
      .domain(data.map(d => d.id))
      .range([0, width])
      .padding(0.1);

    const yExtent = d3.extent(data.flatMap(d => [d.start, d.end])) as [number, number];
    const yScale = d3.scaleLinear()
      .domain([
        Math.min(yExtent[0], 0) * 1.1,
        Math.max(yExtent[1], 0) * 1.1
      ])
      .range([chartHeight, 0]);

    // Color scale
    const colorScale = (type: string, category: string) => {
      switch (type) {
        case 'positive': return '#4CAF50';
        case 'negative': return '#F44336';
        case 'total': return '#2196F3';
        case 'starting': return '#9E9E9E';
        default: return '#FF9800';
      }
    };

    // Draw waterfall bars
    const bars = g.selectAll('.waterfall-bar')
      .data(data)
      .enter()
      .append('g')
      .attr('class', 'waterfall-bar')
      .attr('transform', d => `translate(${xScale(d.id) || 0}, 0)`);

    bars.append('rect')
      .attr('x', 0)
      .attr('y', d => yScale(d.end))
      .attr('width', xScale.bandwidth())
      .attr('height', d => Math.abs(yScale(d.end) - yScale(d.start)))
      .attr('fill', d => colorScale(d.type, d.category))
      .attr('stroke', '#fff')
      .attr('stroke-width', 1)
      .attr('opacity', 0.8)
      .style('cursor', 'pointer')
      .on('click', function(event, d) {
        const component = processedData.filteredComponents.find((c: PnLComponent) => c.id === d.id);
        if (component) {
          setSelectedComponent(component);
          onComponentClick?.(component);
        }
      })
      .on('mouseover', function(event, d) {
        d3.select(this).attr('opacity', 1);
        
        // Tooltip
        const tooltip = g.append('g')
          .attr('class', 'tooltip')
          .attr('transform', `translate(${(xScale(d.id) || 0) + xScale.bandwidth() / 2}, ${yScale(d.end) - 10})`);

        const tooltipRect = tooltip.append('rect')
          .attr('x', -50)
          .attr('y', -25)
          .attr('width', 100)
          .attr('height', 20)
          .attr('fill', '#333')
          .attr('opacity', 0.9)
          .attr('rx', 3);

        tooltip.append('text')
          .attr('text-anchor', 'middle')
          .attr('fill', '#fff')
          .attr('font-size', '11px')
          .text(`${d.name}: $${d.value.toLocaleString()}`);
      })
      .on('mouseout', function(event, d) {
        d3.select(this).attr('opacity', 0.8);
        g.selectAll('.tooltip').remove();
      });

    // Add connecting lines for waterfall effect
    data.forEach((d, i) => {
      if (i === 0 || d.type === 'total') return;
      
      const prevBar = data[i - 1];
      const startX = (xScale(prevBar.id) || 0) + xScale.bandwidth();
      const endX = xScale(d.id) || 0;
      const y = yScale(d.type === 'positive' ? d.start : d.end);

      g.append('line')
        .attr('x1', startX)
        .attr('x2', endX)
        .attr('y1', y)
        .attr('y2', y)
        .attr('stroke', '#999')
        .attr('stroke-width', 1)
        .attr('stroke-dasharray', '3,3')
        .attr('opacity', 0.5);
    });

    // Add value labels
    bars.append('text')
      .attr('x', xScale.bandwidth() / 2)
      .attr('y', d => yScale(d.end) + (d.value >= 0 ? -5 : 15))
      .attr('text-anchor', 'middle')
      .attr('font-size', '11px')
      .attr('font-weight', 'bold')
      .attr('fill', d => d.value >= 0 ? '#4CAF50' : '#F44336')
      .text(d => `$${d.value.toLocaleString()}`);

    // Add zero line
    g.append('line')
      .attr('x1', 0)
      .attr('x2', width)
      .attr('y1', yScale(0))
      .attr('y2', yScale(0))
      .attr('stroke', '#333')
      .attr('stroke-width', 2);

    // Add axes
    const xAxis = d3.axisBottom(xScale)
      .tickFormat(d => {
        const bar = data.find(b => b.id === d);
        return bar ? bar.name.substring(0, 10) : '';
      });

    g.append('g')
      .attr('transform', `translate(0, ${chartHeight})`)
      .call(xAxis)
      .selectAll('text')
      .style('text-anchor', 'end')
      .attr('transform', 'rotate(-45)')
      .attr('dx', '-0.5em')
      .attr('dy', '0.15em');

    const yAxis = d3.axisLeft(yScale)
      .tickFormat(d => `$${d3.format('.2s')(d)}`);

    g.append('g').call(yAxis);

    // Add title
    g.append('text')
      .attr('x', width / 2)
      .attr('y', -10)
      .attr('text-anchor', 'middle')
      .attr('font-size', '14px')
      .attr('font-weight', 'bold')
      .text(`P&L Waterfall - ${processedData.portfolioId}`);

  }, [processedData, height, onComponentClick]);

  // Animation effect
  useEffect(() => {
    if (isPlaying && showAnimation) {
      const animate = () => {
        renderWaterfallChart();
        animationRef.current = setTimeout(animate, animationSpeed);
      };
      animate();
    } else {
      renderWaterfallChart();
    }

    return () => {
      if (animationRef.current) {
        clearTimeout(animationRef.current);
      }
    };
  }, [isPlaying, showAnimation, animationSpeed, renderWaterfallChart]);

  // Handle export
  const handleExport = useCallback(() => {
    if (!svgRef.current || !exportEnabled) return;

    const svgData = new XMLSerializer().serializeToString(svgRef.current);
    const canvas = document.createElement('canvas');
    const ctx = canvas.getContext('2d');
    const img = new Image();

    img.onload = () => {
      canvas.width = img.width;
      canvas.height = img.height;
      ctx?.drawImage(img, 0, 0);
      
      const link = document.createElement('a');
      link.download = `pnl-waterfall-${portfolioId}-${Date.now()}.png`;
      link.href = canvas.toDataURL();
      link.click();
    };

    img.src = `data:image/svg+xml;base64,${btoa(svgData)}`;
  }, [portfolioId, exportEnabled]);

  // Settings menu
  const settingsMenu = (
    <Menu>
      <Menu.SubMenu key="animation" title="Animation">
        <Menu.Item key="speed">
          <Space direction="vertical" style={{ width: 150 }}>
            <Text style={{ fontSize: '11px' }}>Speed (ms)</Text>
            <Slider
              min={500}
              max={5000}
              step={100}
              value={animationSpeed}
              onChange={setAnimationSpeed}
            />
          </Space>
        </Menu.Item>
        <Menu.Item key="toggle-animation">
          <Switch
            checked={isPlaying}
            onChange={setIsPlaying}
            checkedChildren="Playing"
            unCheckedChildren="Paused"
          />
        </Menu.Item>
      </Menu.SubMenu>
      <Menu.SubMenu key="filters" title="Filters">
        <Menu.Item
          key="all"
          onClick={() => setFilterCategory('all')}
          style={{ backgroundColor: filterCategory === 'all' ? '#e6f7ff' : 'transparent' }}
        >
          All Categories
        </Menu.Item>
        <Menu.Item
          key="trading"
          onClick={() => setFilterCategory('trading')}
          style={{ backgroundColor: filterCategory === 'trading' ? '#e6f7ff' : 'transparent' }}
        >
          Trading
        </Menu.Item>
        <Menu.Item
          key="fees"
          onClick={() => setFilterCategory('fees')}
          style={{ backgroundColor: filterCategory === 'fees' ? '#e6f7ff' : 'transparent' }}
        >
          Fees
        </Menu.Item>
        <Menu.Item
          key="funding"
          onClick={() => setFilterCategory('funding')}
          style={{ backgroundColor: filterCategory === 'funding' ? '#e6f7ff' : 'transparent' }}
        >
          Funding
        </Menu.Item>
      </Menu.SubMenu>
    </Menu>
  );

  // Check for alerts
  const hasAlerts = processedData && (
    processedData.totalPnL <= alertThresholds.loss ||
    processedData.totalPnL >= alertThresholds.gain
  );

  if (error) {
    return (
      <Card title="P&L Waterfall - Error" style={{ height }}>
        <Text type="danger">{error}</Text>
      </Card>
    );
  }

  return (
    <div ref={containerRef} style={{ position: 'relative' }}>
      <Card
        title={
          <Space>
            <BarChartOutlined />
            <Text strong>P&L Waterfall - {portfolioId}</Text>
            <Badge 
              status={isConnected ? 'success' : 'error'} 
              text={isConnected ? 'Live' : 'Disconnected'} 
            />
            {hasAlerts && (
              <Badge status="warning" text="Alert" />
            )}
          </Space>
        }
        extra={
          <Space>
            {/* View Mode */}
            <Select
              value={viewMode}
              onChange={setViewMode}
              size="small"
              style={{ width: 120 }}
            >
              <Select.Option value="waterfall">Waterfall</Select.Option>
              <Select.Option value="breakdown">Breakdown</Select.Option>
              <Select.Option value="timeline">Timeline</Select.Option>
            </Select>

            {/* Category Filter */}
            <Select
              value={filterCategory}
              onChange={setFilterCategory}
              size="small"
              style={{ width: 100 }}
            >
              <Select.Option value="all">All</Select.Option>
              <Select.Option value="trading">Trading</Select.Option>
              <Select.Option value="fees">Fees</Select.Option>
              <Select.Option value="funding">Funding</Select.Option>
              <Select.Option value="other">Other</Select.Option>
            </Select>

            {/* Animation Controls */}
            <Button
              size="small"
              icon={isPlaying ? <PauseCircleOutlined /> : <PlayCircleOutlined />}
              onClick={() => setIsPlaying(!isPlaying)}
            />

            <Button size="small" icon={<ReloadOutlined />} onClick={renderWaterfallChart} />

            <Dropdown overlay={settingsMenu} trigger={['click']}>
              <Button size="small" icon={<SettingOutlined />} />
            </Dropdown>

            {exportEnabled && (
              <Button size="small" icon={<ExportOutlined />} onClick={handleExport} />
            )}

            <Button
              size="small"
              icon={<FullscreenOutlined />}
              onClick={() => setFullscreen(!fullscreen)}
            />
          </Space>
        }
        size={compactMode ? 'small' : 'default'}
        style={{ 
          height: fullscreen ? '100vh' : height,
          position: fullscreen ? 'fixed' : 'relative',
          top: fullscreen ? 0 : 'auto',
          left: fullscreen ? 0 : 'auto',
          width: fullscreen ? '100vw' : '100%',
          zIndex: fullscreen ? 1000 : 'auto'
        }}
      >
        {/* Alert Banner */}
        {hasAlerts && showAlerts && processedData && (
          <Alert
            message={
              processedData.totalPnL <= alertThresholds.loss
                ? `Loss Alert: P&L below threshold ($${alertThresholds.loss.toLocaleString()})`
                : `Gain Alert: P&L above threshold ($${alertThresholds.gain.toLocaleString()})`
            }
            type={processedData.totalPnL <= alertThresholds.loss ? 'error' : 'success'}
            showIcon
            closable
            onClose={() => setShowAlerts(false)}
            style={{ marginBottom: 16 }}
          />
        )}

        {/* Statistics Row */}
        {!compactMode && processedData && (
          <Row gutter={16} style={{ marginBottom: 16 }}>
            <Col span={6}>
              <Statistic
                title="Total P&L"
                value={processedData.totalPnL}
                precision={2}
                prefix="$"
                valueStyle={{ 
                  color: processedData.totalPnL >= 0 ? '#4CAF50' : '#F44336',
                  fontSize: '18px'
                }}
              />
            </Col>
            <Col span={6}>
              <Statistic
                title="Today's Change"
                value={processedData.totalPnL - processedData.previousTotalPnL}
                precision={2}
                prefix="$"
                valueStyle={{ 
                  color: (processedData.totalPnL - processedData.previousTotalPnL) >= 0 ? '#4CAF50' : '#F44336',
                  fontSize: '16px'
                }}
              />
            </Col>
            <Col span={6}>
              <Statistic
                title="Components"
                value={processedData.filteredComponents.length}
                valueStyle={{ fontSize: '16px' }}
              />
            </Col>
            <Col span={6}>
              <Statistic
                title="% Change"
                value={((processedData.totalPnL - processedData.previousTotalPnL) / 
                       Math.abs(processedData.previousTotalPnL) * 100)}
                precision={2}
                suffix="%"
                valueStyle={{ 
                  color: (processedData.totalPnL - processedData.previousTotalPnL) >= 0 ? '#4CAF50' : '#F44336',
                  fontSize: '16px'
                }}
              />
            </Col>
          </Row>
        )}

        {/* Visualization */}
        <div style={{ 
          width: '100%', 
          height: compactMode ? height - 100 : height - 200,
          overflow: 'hidden'
        }}>
          <svg
            ref={svgRef}
            width="100%"
            height="100%"
            style={{ background: theme === 'dark' ? '#1f1f1f' : '#ffffff' }}
          />
        </div>

        {/* Selected Component Details */}
        {selectedComponent && !compactMode && (
          <Card size="small" style={{ marginTop: 12 }}>
            <Row gutter={16}>
              <Col span={8}>
                <Text strong>{selectedComponent.name}</Text>
                <br />
                <Text type="secondary">{selectedComponent.category}</Text>
              </Col>
              <Col span={4}>
                <Statistic
                  title="Current"
                  value={selectedComponent.value}
                  precision={2}
                  prefix="$"
                  size="small"
                />
              </Col>
              <Col span={4}>
                <Statistic
                  title="Previous"
                  value={selectedComponent.previousValue}
                  precision={2}
                  prefix="$"
                  size="small"
                />
              </Col>
              <Col span={4}>
                <Statistic
                  title="Change"
                  value={selectedComponent.change}
                  precision={2}
                  prefix="$"
                  size="small"
                  valueStyle={{ 
                    color: selectedComponent.change >= 0 ? '#4CAF50' : '#F44336'
                  }}
                />
              </Col>
              <Col span={4}>
                <Button
                  size="small"
                  type="primary"
                  onClick={() => onDrillDown?.(selectedComponent)}
                  disabled={!selectedComponent.subComponents?.length}
                >
                  Drill Down
                </Button>
              </Col>
            </Row>
          </Card>
        )}

        {/* Footer */}
        {!compactMode && (
          <div style={{ 
            marginTop: 12, 
            padding: '8px 0',
            borderTop: '1px solid #f0f0f0',
            display: 'flex',
            justifyContent: 'space-between',
            alignItems: 'center'
          }}>
            <Space>
              <Text type="secondary" style={{ fontSize: '11px' }}>
                Last Update: {lastUpdate ? new Date(lastUpdate).toLocaleTimeString() : 'Never'}
              </Text>
              <Text type="secondary" style={{ fontSize: '11px' }}>
                Granularity: {granularity}
              </Text>
              <Text type="secondary" style={{ fontSize: '11px' }}>
                Filter: {filterCategory}
              </Text>
            </Space>
            
            <Space>
              {isPlaying && (
                <Badge status="processing" text="Animating" />
              )}
              <Text type="secondary" style={{ fontSize: '11px' }}>
                Speed: {animationSpeed}ms
              </Text>
            </Space>
          </div>
        )}
      </Card>
    </div>
  );
});

PnLWaterfallWidget.displayName = 'PnLWaterfallWidget';

export default PnLWaterfallWidget;