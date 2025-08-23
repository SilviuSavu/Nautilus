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
  Slider,
  Dropdown,
  Menu,
  Progress,
  Alert,
  Modal,
  Table,
  Tag
} from 'antd';
import {
  HeatMapOutlined,
  WarningOutlined,
  SettingOutlined,
  ExportOutlined,
  FullscreenOutlined,
  DrillDownOutlined,
  FilterOutlined,
  ReloadOutlined,
  ZoomInOutlined,
  ZoomOutOutlined
} from '@ant-design/icons';
import * as d3 from 'd3';
import { useRiskMonitoring } from '../../hooks/risk/useRiskMonitoring';

const { Text, Title } = Typography;

interface RiskCell {
  id: string;
  symbol: string;
  riskType: 'var' | 'exposure' | 'concentration' | 'leverage' | 'correlation' | 'liquidity';
  value: number;
  normalizedValue: number; // 0-1 scale for color mapping
  limit: number;
  utilization: number; // value / limit
  severity: 'low' | 'medium' | 'high' | 'critical';
  trend: 'increasing' | 'decreasing' | 'stable';
  lastUpdate: number;
  metadata: {
    position: number;
    marketValue: number;
    volatility: number;
    beta: number;
    sector: string;
    geography: string;
  };
  drillDownData?: RiskCell[];
}

interface RiskHeatmapData {
  portfolioId: string;
  timestamp: number;
  cells: RiskCell[];
  aggregatedRisk: {
    totalVaR: number;
    totalExposure: number;
    riskUtilization: number;
    breachCount: number;
  };
  dimensions: {
    symbols: string[];
    riskTypes: string[];
    sectors: string[];
    geographies: string[];
  };
}

interface RiskHeatmapWidgetProps {
  portfolioId: string;
  viewDimensions?: 'symbol-risk' | 'sector-risk' | 'geo-risk' | 'correlation';
  riskTypes?: string[];
  showSeverityOnly?: boolean;
  heatmapSize?: 'small' | 'medium' | 'large';
  colorScheme?: 'red-green' | 'traffic-light' | 'viridis';
  compactMode?: boolean;
  height?: number;
  autoRefresh?: boolean;
  refreshInterval?: number;
  theme?: 'light' | 'dark';
  onCellClick?: (cell: RiskCell) => void;
  onDrillDown?: (cell: RiskCell) => void;
  exportEnabled?: boolean;
  alertEnabled?: boolean;
}

interface HeatmapCell {
  x: number;
  y: number;
  width: number;
  height: number;
  cell: RiskCell;
  color: string;
  isSelected: boolean;
  isHighlighted: boolean;
}

const RiskHeatmapWidget: React.FC<RiskHeatmapWidgetProps> = memo(({
  portfolioId,
  viewDimensions = 'symbol-risk',
  riskTypes = ['var', 'exposure', 'concentration', 'leverage'],
  showSeverityOnly = false,
  heatmapSize = 'medium',
  colorScheme = 'red-green',
  compactMode = false,
  height = 600,
  autoRefresh = true,
  refreshInterval = 5000,
  theme = 'light',
  onCellClick,
  onDrillDown,
  exportEnabled = true,
  alertEnabled = true
}) => {
  const [selectedCell, setSelectedCell] = useState<RiskCell | null>(null);
  const [drillDownVisible, setDrillDownVisible] = useState(false);
  const [filterSeverity, setFilterSeverity] = useState<string>('all');
  const [zoom, setZoom] = useState(1);
  const [highlightedSector, setHighlightedSector] = useState<string | null>(null);
  const [fullscreen, setFullscreen] = useState(false);
  const [showTooltip, setShowTooltip] = useState(true);

  const svgRef = useRef<SVGSVGElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const tooltipRef = useRef<HTMLDivElement>(null);
  const heatmapCells = useRef<HeatmapCell[]>([]);

  // Use risk monitoring hook
  const {
    riskData,
    breaches,
    isConnected,
    lastUpdate,
    error
  } = useRiskMonitoring({
    portfolioId,
    autoRefresh,
    refreshInterval
  });

  // Process risk data into heatmap format
  const processedData = useMemo(() => {
    if (!riskData) return null;

    const cells: RiskCell[] = [];
    
    // Build cells based on view dimensions
    if (viewDimensions === 'symbol-risk') {
      riskData.positions?.forEach((position: any) => {
        riskTypes.forEach(riskType => {
          const riskValue = position.risk?.[riskType] || 0;
          const limit = position.limits?.[riskType] || 1;
          const utilization = riskValue / limit;
          
          cells.push({
            id: `${position.symbol}-${riskType}`,
            symbol: position.symbol,
            riskType: riskType as any,
            value: riskValue,
            normalizedValue: Math.min(utilization, 1),
            limit,
            utilization,
            severity: getSeverity(utilization),
            trend: getTrend(position.risk?.history?.[riskType] || []),
            lastUpdate: Date.now(),
            metadata: {
              position: position.quantity || 0,
              marketValue: position.marketValue || 0,
              volatility: position.volatility || 0,
              beta: position.beta || 1,
              sector: position.sector || 'Unknown',
              geography: position.geography || 'Unknown'
            }
          });
        });
      });
    } else if (viewDimensions === 'sector-risk') {
      const sectorData = groupBy(riskData.positions || [], 'sector');
      Object.entries(sectorData).forEach(([sector, positions]: [string, any[]]) => {
        riskTypes.forEach(riskType => {
          const aggregatedRisk = positions.reduce((sum, pos) => 
            sum + (pos.risk?.[riskType] || 0), 0);
          const aggregatedLimit = positions.reduce((sum, pos) => 
            sum + (pos.limits?.[riskType] || 0), 0);
          const utilization = aggregatedRisk / aggregatedLimit;

          cells.push({
            id: `${sector}-${riskType}`,
            symbol: sector,
            riskType: riskType as any,
            value: aggregatedRisk,
            normalizedValue: Math.min(utilization, 1),
            limit: aggregatedLimit,
            utilization,
            severity: getSeverity(utilization),
            trend: 'stable',
            lastUpdate: Date.now(),
            metadata: {
              position: positions.reduce((sum, pos) => sum + (pos.quantity || 0), 0),
              marketValue: positions.reduce((sum, pos) => sum + (pos.marketValue || 0), 0),
              volatility: positions.reduce((sum, pos) => sum + (pos.volatility || 0), 0) / positions.length,
              beta: positions.reduce((sum, pos) => sum + (pos.beta || 1), 0) / positions.length,
              sector,
              geography: 'Mixed'
            },
            drillDownData: positions.map(pos => ({
              id: `${pos.symbol}-${riskType}`,
              symbol: pos.symbol,
              riskType: riskType as any,
              value: pos.risk?.[riskType] || 0,
              normalizedValue: Math.min((pos.risk?.[riskType] || 0) / (pos.limits?.[riskType] || 1), 1),
              limit: pos.limits?.[riskType] || 1,
              utilization: (pos.risk?.[riskType] || 0) / (pos.limits?.[riskType] || 1),
              severity: getSeverity((pos.risk?.[riskType] || 0) / (pos.limits?.[riskType] || 1)),
              trend: 'stable',
              lastUpdate: Date.now(),
              metadata: {
                position: pos.quantity || 0,
                marketValue: pos.marketValue || 0,
                volatility: pos.volatility || 0,
                beta: pos.beta || 1,
                sector: pos.sector || 'Unknown',
                geography: pos.geography || 'Unknown'
              }
            }))
          });
        });
      });
    }

    // Filter by severity if specified
    const filteredCells = filterSeverity === 'all' 
      ? cells 
      : cells.filter(cell => cell.severity === filterSeverity);

    return {
      portfolioId,
      timestamp: Date.now(),
      cells: filteredCells,
      aggregatedRisk: {
        totalVaR: cells.filter(c => c.riskType === 'var').reduce((sum, c) => sum + c.value, 0),
        totalExposure: cells.filter(c => c.riskType === 'exposure').reduce((sum, c) => sum + c.value, 0),
        riskUtilization: cells.reduce((sum, c) => sum + c.utilization, 0) / cells.length,
        breachCount: cells.filter(c => c.utilization > 1).length
      },
      dimensions: {
        symbols: [...new Set(cells.map(c => c.symbol))],
        riskTypes: [...new Set(cells.map(c => c.riskType))],
        sectors: [...new Set(cells.map(c => c.metadata.sector))],
        geographies: [...new Set(cells.map(c => c.metadata.geography))]
      }
    };
  }, [riskData, viewDimensions, riskTypes, filterSeverity]);

  // Helper functions
  const getSeverity = (utilization: number): 'low' | 'medium' | 'high' | 'critical' => {
    if (utilization <= 0.5) return 'low';
    if (utilization <= 0.75) return 'medium';
    if (utilization <= 0.95) return 'high';
    return 'critical';
  };

  const getTrend = (history: number[]): 'increasing' | 'decreasing' | 'stable' => {
    if (history.length < 2) return 'stable';
    const recent = history.slice(-3);
    const slope = (recent[recent.length - 1] - recent[0]) / recent.length;
    if (slope > 0.01) return 'increasing';
    if (slope < -0.01) return 'decreasing';
    return 'stable';
  };

  const groupBy = (array: any[], key: string) => {
    return array.reduce((result, item) => {
      const group = item[key];
      if (!result[group]) result[group] = [];
      result[group].push(item);
      return result;
    }, {});
  };

  // Color scale based on scheme
  const getColorScale = useCallback(() => {
    switch (colorScheme) {
      case 'red-green':
        return d3.scaleSequential(d3.interpolateRdYlGn).domain([1, 0]);
      case 'traffic-light':
        return d3.scaleOrdinal()
          .domain(['low', 'medium', 'high', 'critical'])
          .range(['#4CAF50', '#FFC107', '#FF9800', '#F44336']);
      case 'viridis':
        return d3.scaleSequential(d3.interpolateViridis).domain([0, 1]);
      default:
        return d3.scaleSequential(d3.interpolateRdYlGn).domain([1, 0]);
    }
  }, [colorScheme]);

  // Render heatmap
  const renderHeatmap = useCallback(() => {
    if (!processedData || !svgRef.current) return;

    const svg = d3.select(svgRef.current);
    svg.selectAll('*').remove();

    const margin = { top: 40, right: 40, bottom: 60, left: 100 };
    const width = (containerRef.current?.clientWidth || 800) - margin.left - margin.right;
    const chartHeight = height - margin.top - margin.bottom;

    const g = svg.append('g')
      .attr('transform', `translate(${margin.left},${margin.top})`);

    const data = processedData.cells;
    const colorScale = getColorScale();

    // Scales for positioning
    const symbols = [...new Set(data.map(d => d.symbol))];
    const riskTypesList = [...new Set(data.map(d => d.riskType))];

    const xScale = d3.scaleBand()
      .domain(symbols)
      .range([0, width])
      .padding(0.02);

    const yScale = d3.scaleBand()
      .domain(riskTypesList)
      .range([0, chartHeight])
      .padding(0.02);

    const cellWidth = xScale.bandwidth();
    const cellHeight = yScale.bandwidth();

    // Build heatmap cells
    heatmapCells.current = data.map(cell => ({
      x: xScale(cell.symbol) || 0,
      y: yScale(cell.riskType) || 0,
      width: cellWidth,
      height: cellHeight,
      cell,
      color: colorScheme === 'traffic-light' 
        ? colorScale(cell.severity) as string
        : colorScale(cell.normalizedValue) as string,
      isSelected: selectedCell?.id === cell.id,
      isHighlighted: highlightedSector ? cell.metadata.sector === highlightedSector : false
    }));

    // Render cells
    const cells = g.selectAll('.risk-cell')
      .data(heatmapCells.current)
      .enter()
      .append('g')
      .attr('class', 'risk-cell')
      .attr('transform', d => `translate(${d.x}, ${d.y})`);

    cells.append('rect')
      .attr('width', d => d.width)
      .attr('height', d => d.height)
      .attr('fill', d => d.color)
      .attr('stroke', d => d.isSelected ? '#1890ff' : '#fff')
      .attr('stroke-width', d => d.isSelected ? 3 : 1)
      .attr('opacity', d => {
        if (d.isHighlighted) return 1;
        if (highlightedSector && !d.isHighlighted) return 0.3;
        return 0.8;
      })
      .style('cursor', 'pointer')
      .on('click', function(event, d) {
        setSelectedCell(d.cell);
        onCellClick?.(d.cell);
      })
      .on('mouseover', function(event, d) {
        if (!showTooltip) return;
        
        // Show tooltip
        const tooltip = g.append('g')
          .attr('class', 'heatmap-tooltip')
          .attr('transform', `translate(${d.x + d.width / 2}, ${d.y - 10})`);

        const tooltipBg = tooltip.append('rect')
          .attr('x', -80)
          .attr('y', -40)
          .attr('width', 160)
          .attr('height', 35)
          .attr('fill', '#333')
          .attr('opacity', 0.9)
          .attr('rx', 4);

        tooltip.append('text')
          .attr('text-anchor', 'middle')
          .attr('fill', '#fff')
          .attr('font-size', '11px')
          .attr('y', -25)
          .text(`${d.cell.symbol} - ${d.cell.riskType}`);

        tooltip.append('text')
          .attr('text-anchor', 'middle')
          .attr('fill', '#fff')
          .attr('font-size', '10px')
          .attr('y', -12)
          .text(`Value: ${d.cell.value.toFixed(2)}`);

        tooltip.append('text')
          .attr('text-anchor', 'middle')
          .attr('fill', '#fff')
          .attr('font-size', '10px')
          .attr('y', -1)
          .text(`Utilization: ${(d.cell.utilization * 100).toFixed(1)}%`);
      })
      .on('mouseout', function() {
        g.selectAll('.heatmap-tooltip').remove();
      });

    // Add value labels for larger cells
    if (cellWidth > 60 && cellHeight > 30) {
      cells.append('text')
        .attr('x', d => d.width / 2)
        .attr('y', d => d.height / 2)
        .attr('text-anchor', 'middle')
        .attr('dominant-baseline', 'middle')
        .attr('font-size', '10px')
        .attr('font-weight', 'bold')
        .attr('fill', d => d.cell.utilization > 0.5 ? '#fff' : '#333')
        .text(d => `${(d.cell.utilization * 100).toFixed(0)}%`);
    }

    // Add severity indicators
    cells
      .filter(d => d.cell.severity === 'critical')
      .append('circle')
      .attr('cx', d => d.width - 8)
      .attr('cy', 8)
      .attr('r', 4)
      .attr('fill', '#F44336')
      .attr('stroke', '#fff')
      .attr('stroke-width', 1);

    // Add trend arrows
    cells
      .filter(d => d.cell.trend === 'increasing')
      .append('polygon')
      .attr('points', d => `${d.width - 15},${d.height - 5} ${d.width - 10},${d.height - 10} ${d.width - 5},${d.height - 5}`)
      .attr('fill', '#F44336')
      .attr('opacity', 0.7);

    cells
      .filter(d => d.cell.trend === 'decreasing')
      .append('polygon')
      .attr('points', d => `${d.width - 15},${d.height - 10} ${d.width - 10},${d.height - 5} ${d.width - 5},${d.height - 10}`)
      .attr('fill', '#4CAF50')
      .attr('opacity', 0.7);

    // Add axes
    const xAxis = d3.axisBottom(xScale);
    g.append('g')
      .attr('transform', `translate(0, ${chartHeight})`)
      .call(xAxis)
      .selectAll('text')
      .style('text-anchor', 'end')
      .attr('transform', 'rotate(-45)')
      .attr('dx', '-0.5em')
      .attr('dy', '0.15em');

    const yAxis = d3.axisLeft(yScale);
    g.append('g').call(yAxis);

    // Add title
    g.append('text')
      .attr('x', width / 2)
      .attr('y', -15)
      .attr('text-anchor', 'middle')
      .attr('font-size', '14px')
      .attr('font-weight', 'bold')
      .text(`Risk Heatmap - ${viewDimensions.replace('-', ' vs ')}`);

    // Add legend
    const legendWidth = 200;
    const legendHeight = 15;
    const legend = g.append('g')
      .attr('transform', `translate(${width - legendWidth - 20}, -35)`);

    const legendScale = d3.scaleLinear()
      .domain([0, 1])
      .range([0, legendWidth]);

    const legendAxis = d3.axisBottom(legendScale)
      .tickFormat(d => `${(d * 100).toFixed(0)}%`);

    // Create gradient for legend
    const defs = svg.append('defs');
    const gradient = defs.append('linearGradient')
      .attr('id', 'legend-gradient');

    if (colorScheme === 'traffic-light') {
      gradient.selectAll('stop')
        .data(['#4CAF50', '#FFC107', '#FF9800', '#F44336'])
        .enter()
        .append('stop')
        .attr('offset', (d, i) => `${i * 25}%`)
        .attr('stop-color', d => d);
    } else {
      const steps = 10;
      gradient.selectAll('stop')
        .data(d3.range(steps + 1))
        .enter()
        .append('stop')
        .attr('offset', d => `${(d / steps) * 100}%`)
        .attr('stop-color', d => colorScale(d / steps) as string);
    }

    legend.append('rect')
      .attr('width', legendWidth)
      .attr('height', legendHeight)
      .attr('fill', 'url(#legend-gradient)');

    legend.append('g')
      .attr('transform', `translate(0, ${legendHeight})`)
      .call(legendAxis);

    legend.append('text')
      .attr('x', legendWidth / 2)
      .attr('y', -5)
      .attr('text-anchor', 'middle')
      .attr('font-size', '11px')
      .text('Risk Utilization');

  }, [processedData, height, selectedCell, highlightedSector, colorScheme, getColorScale, viewDimensions, showTooltip]);

  // Update visualization
  useEffect(() => {
    renderHeatmap();
  }, [renderHeatmap]);

  // Handle drill down
  const handleDrillDown = useCallback((cell: RiskCell) => {
    setSelectedCell(cell);
    setDrillDownVisible(true);
    onDrillDown?.(cell);
  }, [onDrillDown]);

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
      link.download = `risk-heatmap-${portfolioId}-${Date.now()}.png`;
      link.href = canvas.toDataURL();
      link.click();
    };

    img.src = `data:image/svg+xml;base64,${btoa(svgData)}`;
  }, [portfolioId, exportEnabled]);

  // Settings menu
  const settingsMenu = (
    <Menu>
      <Menu.SubMenu key="view" title="View Options">
        <Menu.Item
          key="symbol-risk"
          onClick={() => {/* setViewDimensions('symbol-risk') */}}
          style={{ backgroundColor: viewDimensions === 'symbol-risk' ? '#e6f7ff' : 'transparent' }}
        >
          Symbol vs Risk
        </Menu.Item>
        <Menu.Item
          key="sector-risk"
          onClick={() => {/* setViewDimensions('sector-risk') */}}
          style={{ backgroundColor: viewDimensions === 'sector-risk' ? '#e6f7ff' : 'transparent' }}
        >
          Sector vs Risk
        </Menu.Item>
      </Menu.SubMenu>
      <Menu.SubMenu key="color" title="Color Scheme">
        <Menu.Item key="red-green">Red-Green</Menu.Item>
        <Menu.Item key="traffic-light">Traffic Light</Menu.Item>
        <Menu.Item key="viridis">Viridis</Menu.Item>
      </Menu.SubMenu>
      <Menu.SubMenu key="filters" title="Filters">
        <Menu.Item
          key="all-severity"
          onClick={() => setFilterSeverity('all')}
          style={{ backgroundColor: filterSeverity === 'all' ? '#e6f7ff' : 'transparent' }}
        >
          All Severity
        </Menu.Item>
        <Menu.Item
          key="critical-only"
          onClick={() => setFilterSeverity('critical')}
          style={{ backgroundColor: filterSeverity === 'critical' ? '#e6f7ff' : 'transparent' }}
        >
          Critical Only
        </Menu.Item>
        <Menu.Item
          key="high-critical"
          onClick={() => setFilterSeverity('high')}
          style={{ backgroundColor: filterSeverity === 'high' ? '#e6f7ff' : 'transparent' }}
        >
          High Risk+
        </Menu.Item>
      </Menu.SubMenu>
    </Menu>
  );

  // Drill down table columns
  const drillDownColumns = [
    {
      title: 'Symbol',
      dataIndex: 'symbol',
      key: 'symbol',
      render: (text: string, record: RiskCell) => (
        <Space>
          <Text strong>{text}</Text>
          <Tag color={record.severity === 'critical' ? 'red' : record.severity === 'high' ? 'orange' : 'green'}>
            {record.severity}
          </Tag>
        </Space>
      )
    },
    {
      title: 'Risk Value',
      dataIndex: 'value',
      key: 'value',
      render: (value: number) => value.toFixed(2)
    },
    {
      title: 'Limit',
      dataIndex: 'limit',
      key: 'limit',
      render: (limit: number) => limit.toFixed(2)
    },
    {
      title: 'Utilization',
      dataIndex: 'utilization',
      key: 'utilization',
      render: (utilization: number) => (
        <Space>
          <Progress
            percent={Math.min(utilization * 100, 100)}
            size="small"
            status={utilization > 0.95 ? 'exception' : utilization > 0.75 ? 'active' : 'success'}
            showInfo={false}
          />
          <Text>{(utilization * 100).toFixed(1)}%</Text>
        </Space>
      )
    },
    {
      title: 'Position',
      dataIndex: ['metadata', 'position'],
      key: 'position',
      render: (position: number) => position.toLocaleString()
    }
  ];

  if (error) {
    return (
      <Card title="Risk Heatmap - Error" style={{ height }}>
        <Text type="danger">{error}</Text>
      </Card>
    );
  }

  return (
    <div ref={containerRef} style={{ position: 'relative' }}>
      <Card
        title={
          <Space>
            <HeatMapOutlined />
            <Text strong>Risk Heatmap - {portfolioId}</Text>
            <Badge 
              status={isConnected ? 'success' : 'error'} 
              text={isConnected ? 'Live' : 'Disconnected'} 
            />
            {processedData?.aggregatedRisk.breachCount > 0 && (
              <Badge count={processedData.aggregatedRisk.breachCount} style={{ backgroundColor: '#f5222d' }} />
            )}
          </Space>
        }
        extra={
          <Space>
            {/* View Dimensions */}
            <Select
              value={viewDimensions}
              onChange={(value) => {/* setViewDimensions(value) */}}
              size="small"
              style={{ width: 140 }}
            >
              <Select.Option value="symbol-risk">Symbol vs Risk</Select.Option>
              <Select.Option value="sector-risk">Sector vs Risk</Select.Option>
              <Select.Option value="geo-risk">Geography vs Risk</Select.Option>
            </Select>

            {/* Severity Filter */}
            <Select
              value={filterSeverity}
              onChange={setFilterSeverity}
              size="small"
              style={{ width: 100 }}
            >
              <Select.Option value="all">All</Select.Option>
              <Select.Option value="critical">Critical</Select.Option>
              <Select.Option value="high">High+</Select.Option>
            </Select>

            {/* Controls */}
            <Tooltip title="Show Tooltips">
              <Switch
                checked={showTooltip}
                onChange={setShowTooltip}
                size="small"
              />
            </Tooltip>

            <Button size="small" icon={<ReloadOutlined />} onClick={renderHeatmap} />

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
        {/* Alert for critical risks */}
        {alertEnabled && processedData?.aggregatedRisk.breachCount > 0 && (
          <Alert
            message={`${processedData.aggregatedRisk.breachCount} Risk Limit Breaches Detected`}
            type="error"
            showIcon
            closable
            style={{ marginBottom: 16 }}
          />
        )}

        {/* Statistics Row */}
        {!compactMode && processedData && (
          <Row gutter={16} style={{ marginBottom: 16 }}>
            <Col span={6}>
              <Statistic
                title="Total VaR"
                value={processedData.aggregatedRisk.totalVaR}
                precision={0}
                prefix="$"
                valueStyle={{ fontSize: '16px' }}
              />
            </Col>
            <Col span={6}>
              <Statistic
                title="Risk Utilization"
                value={processedData.aggregatedRisk.riskUtilization * 100}
                precision={1}
                suffix="%"
                valueStyle={{ 
                  color: processedData.aggregatedRisk.riskUtilization > 0.8 ? '#F44336' : '#4CAF50',
                  fontSize: '16px'
                }}
              />
            </Col>
            <Col span={6}>
              <Statistic
                title="Breaches"
                value={processedData.aggregatedRisk.breachCount}
                valueStyle={{ 
                  color: processedData.aggregatedRisk.breachCount > 0 ? '#F44336' : '#4CAF50',
                  fontSize: '16px'
                }}
              />
            </Col>
            <Col span={6}>
              <Statistic
                title="Total Cells"
                value={processedData.cells.length}
                valueStyle={{ fontSize: '16px' }}
              />
            </Col>
          </Row>
        )}

        {/* Heatmap Visualization */}
        <div style={{ 
          width: '100%', 
          height: compactMode ? height - 100 : height - 200,
          overflow: 'auto'
        }}>
          <svg
            ref={svgRef}
            width="100%"
            height="100%"
            style={{ 
              background: theme === 'dark' ? '#1f1f1f' : '#ffffff',
              minHeight: 400,
              minWidth: 600
            }}
          />
        </div>

        {/* Selected Cell Details */}
        {selectedCell && !compactMode && (
          <Card size="small" style={{ marginTop: 12 }}>
            <Row gutter={16} align="middle">
              <Col span={4}>
                <Space direction="vertical" size="small">
                  <Text strong>{selectedCell.symbol}</Text>
                  <Text type="secondary">{selectedCell.riskType}</Text>
                </Space>
              </Col>
              <Col span={4}>
                <Statistic
                  title="Risk Value"
                  value={selectedCell.value}
                  precision={2}
                  size="small"
                />
              </Col>
              <Col span={4}>
                <Statistic
                  title="Limit"
                  value={selectedCell.limit}
                  precision={2}
                  size="small"
                />
              </Col>
              <Col span={4}>
                <Statistic
                  title="Utilization"
                  value={selectedCell.utilization * 100}
                  precision={1}
                  suffix="%"
                  size="small"
                  valueStyle={{ 
                    color: selectedCell.utilization > 0.8 ? '#F44336' : '#4CAF50'
                  }}
                />
              </Col>
              <Col span={4}>
                <Space direction="vertical" size="small">
                  <Tag color={
                    selectedCell.severity === 'critical' ? 'red' :
                    selectedCell.severity === 'high' ? 'orange' :
                    selectedCell.severity === 'medium' ? 'yellow' : 'green'
                  }>
                    {selectedCell.severity.toUpperCase()}
                  </Tag>
                  <Text style={{ fontSize: '11px' }}>
                    Trend: {selectedCell.trend}
                  </Text>
                </Space>
              </Col>
              <Col span={4}>
                <Button
                  size="small"
                  type="primary"
                  icon={<DrillDownOutlined />}
                  onClick={() => handleDrillDown(selectedCell)}
                  disabled={!selectedCell.drillDownData?.length}
                >
                  Drill Down
                </Button>
              </Col>
            </Row>
          </Card>
        )}

        {/* Drill Down Modal */}
        <Modal
          title={`Drill Down: ${selectedCell?.symbol} - ${selectedCell?.riskType}`}
          open={drillDownVisible}
          onCancel={() => setDrillDownVisible(false)}
          width={800}
          footer={null}
        >
          {selectedCell?.drillDownData && (
            <Table
              dataSource={selectedCell.drillDownData}
              columns={drillDownColumns}
              rowKey="id"
              size="small"
              pagination={false}
              scroll={{ y: 300 }}
            />
          )}
        </Modal>

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
                View: {viewDimensions.replace('-', ' vs ')}
              </Text>
              <Text type="secondary" style={{ fontSize: '11px' }}>
                Filter: {filterSeverity === 'all' ? 'All Severity' : `${filterSeverity}+`}
              </Text>
            </Space>
            
            <Space>
              <Text type="secondary" style={{ fontSize: '11px' }}>
                Color: {colorScheme}
              </Text>
              {highlightedSector && (
                <Tag size="small" closable onClose={() => setHighlightedSector(null)}>
                  Sector: {highlightedSector}
                </Tag>
              )}
            </Space>
          </div>
        )}
      </Card>
    </div>
  );
});

RiskHeatmapWidget.displayName = 'RiskHeatmapWidget';

export default RiskHeatmapWidget;