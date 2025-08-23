import React, { useState, useEffect, useRef, useMemo, useCallback, memo } from 'react';
import {
  Card,
  Space,
  Typography,
  Divider,
  Select,
  Button,
  Tooltip,
  Badge,
  Progress,
  Row,
  Col,
  Statistic,
  Switch,
  InputNumber,
  Dropdown,
  Menu
} from 'antd';
import {
  ExpandOutlined,
  SettingOutlined,
  ExportOutlined,
  FullscreenOutlined,
  ZoomInOutlined,
  ZoomOutOutlined,
  FilterOutlined,
  BarChartOutlined
} from '@ant-design/icons';
import * as d3 from 'd3';
import { useOrderBookData } from '../../hooks/useOrderBookData';

const { Text, Title } = Typography;

interface OrderBookLevel {
  price: number;
  size: number;
  count: number;
  cumulativeSize: number;
  percentage: number;
}

interface OrderBookData {
  symbol: string;
  bids: OrderBookLevel[];
  asks: OrderBookLevel[];
  spread: number;
  spreadBps: number;
  midPrice: number;
  totalBidVolume: number;
  totalAskVolume: number;
  timestamp: number;
  depth: number;
}

interface AdvancedOrderBookWidgetProps {
  symbol: string;
  depth?: number;
  showHeatmap?: boolean;
  showVolumeProfile?: boolean;
  showSpreadAnalysis?: boolean;
  showMarketImpact?: boolean;
  aggregationLevels?: number[];
  theme?: 'light' | 'dark';
  compactMode?: boolean;
  height?: number;
  autoRefresh?: boolean;
  refreshInterval?: number;
  onLevelClick?: (level: OrderBookLevel, side: 'bid' | 'ask') => void;
  onSpreadClick?: (spread: number) => void;
  exportEnabled?: boolean;
}

interface HeatmapCell {
  price: number;
  volume: number;
  intensity: number;
  time: number;
}

interface VolumeProfileLevel {
  price: number;
  volume: number;
  buyVolume: number;
  sellVolume: number;
  percentage: number;
}

const AdvancedOrderBookWidget: React.FC<AdvancedOrderBookWidgetProps> = memo(({
  symbol,
  depth = 20,
  showHeatmap = true,
  showVolumeProfile = true,
  showSpreadAnalysis = true,
  showMarketImpact = true,
  aggregationLevels = [0.01, 0.05, 0.1, 0.25],
  theme = 'light',
  compactMode = false,
  height = 600,
  autoRefresh = true,
  refreshInterval = 100,
  onLevelClick,
  onSpreadClick,
  exportEnabled = true
}) => {
  const [viewMode, setViewMode] = useState<'traditional' | 'heatmap' | 'volumeProfile' | 'spread'>('traditional');
  const [aggregation, setAggregation] = useState(0.01);
  const [zoom, setZoom] = useState(1);
  const [showCumulative, setShowCumulative] = useState(true);
  const [priceFilter, setPriceFilter] = useState<{ min?: number; max?: number }>({});
  const [sizeFilter, setSizeFilter] = useState<{ min?: number; max?: number }>({});
  const [fullscreen, setFullscreen] = useState(false);
  
  const svgRef = useRef<SVGSVGElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const heatmapData = useRef<HeatmapCell[]>([]);
  const volumeProfileData = useRef<VolumeProfileLevel[]>([]);

  // Use the existing order book hook
  const {
    data: rawOrderBookData,
    isConnected,
    lastUpdate,
    error
  } = useOrderBookData({
    symbol,
    depth,
    autoSubscribe: autoRefresh
  });

  // Process and aggregate order book data
  const processedData = useMemo(() => {
    if (!rawOrderBookData) return null;

    const processBids = (bids: any[]) => {
      return bids
        .filter(bid => {
          const passesPrice = (!priceFilter.min || bid.price >= priceFilter.min) && 
                            (!priceFilter.max || bid.price <= priceFilter.max);
          const passesSize = (!sizeFilter.min || bid.size >= sizeFilter.min) && 
                           (!sizeFilter.max || bid.size <= sizeFilter.max);
          return passesPrice && passesSize;
        })
        .map((bid, index) => ({
          ...bid,
          cumulativeSize: bids.slice(0, index + 1).reduce((sum, b) => sum + b.size, 0),
          percentage: (bid.size / bids.reduce((sum, b) => sum + b.size, 0)) * 100
        }));
    };

    const processAsks = (asks: any[]) => {
      return asks
        .filter(ask => {
          const passesPrice = (!priceFilter.min || ask.price >= priceFilter.min) && 
                            (!priceFilter.max || ask.price <= priceFilter.max);
          const passesSize = (!sizeFilter.min || ask.size >= sizeFilter.min) && 
                           (!sizeFilter.max || ask.size <= sizeFilter.max);
          return passesPrice && passesSize;
        })
        .map((ask, index) => ({
          ...ask,
          cumulativeSize: asks.slice(0, index + 1).reduce((sum, a) => sum + a.size, 0),
          percentage: (ask.size / asks.reduce((sum, a) => sum + a.size, 0)) * 100
        }));
    };

    const processedBids = processBids(rawOrderBookData.bids);
    const processedAsks = processAsks(rawOrderBookData.asks);

    return {
      ...rawOrderBookData,
      bids: processedBids,
      asks: processedAsks,
      totalBidVolume: processedBids.reduce((sum, bid) => sum + bid.size, 0),
      totalAskVolume: processedAsks.reduce((sum, ask) => sum + ask.size, 0)
    };
  }, [rawOrderBookData, priceFilter, sizeFilter, aggregation]);

  // Update heatmap data
  useEffect(() => {
    if (!processedData) return;

    const now = Date.now();
    const newCells: HeatmapCell[] = [
      ...processedData.bids.map(bid => ({
        price: bid.price,
        volume: bid.size,
        intensity: Math.min(bid.size / Math.max(...processedData.bids.map(b => b.size)), 1),
        time: now
      })),
      ...processedData.asks.map(ask => ({
        price: ask.price,
        volume: ask.size,
        intensity: Math.min(ask.size / Math.max(...processedData.asks.map(a => a.size)), 1),
        time: now
      }))
    ];

    // Keep only last 100 updates for performance
    heatmapData.current = [...heatmapData.current, ...newCells].slice(-1000);

    // Update volume profile
    const priceRanges = d3.range(
      Math.min(...processedData.bids.map(b => b.price), ...processedData.asks.map(a => a.price)),
      Math.max(...processedData.bids.map(b => b.price), ...processedData.asks.map(a => a.price)),
      aggregation
    );

    volumeProfileData.current = priceRanges.map(price => {
      const bidsAtLevel = processedData.bids.filter(b => 
        Math.abs(b.price - price) < aggregation / 2
      );
      const asksAtLevel = processedData.asks.filter(a => 
        Math.abs(a.price - price) < aggregation / 2
      );
      
      const buyVolume = bidsAtLevel.reduce((sum, b) => sum + b.size, 0);
      const sellVolume = asksAtLevel.reduce((sum, a) => sum + a.size, 0);
      const totalVolume = buyVolume + sellVolume;

      return {
        price,
        volume: totalVolume,
        buyVolume,
        sellVolume,
        percentage: totalVolume / (processedData.totalBidVolume + processedData.totalAskVolume) * 100
      };
    }).filter(level => level.volume > 0);

  }, [processedData, aggregation]);

  // Traditional order book visualization
  const renderTraditionalOrderBook = useCallback(() => {
    if (!processedData || !svgRef.current) return;

    const svg = d3.select(svgRef.current);
    svg.selectAll('*').remove();

    const margin = { top: 20, right: 20, bottom: 20, left: 80 };
    const width = (containerRef.current?.clientWidth || 800) - margin.left - margin.right;
    const chartHeight = height - margin.top - margin.bottom;

    const g = svg.append('g')
      .attr('transform', `translate(${margin.left},${margin.top})`);

    // Scales
    const maxPrice = Math.max(
      ...processedData.bids.map(b => b.price),
      ...processedData.asks.map(a => a.price)
    );
    const minPrice = Math.min(
      ...processedData.bids.map(b => b.price),
      ...processedData.asks.map(a => a.price)
    );

    const priceScale = d3.scaleLinear()
      .domain([minPrice, maxPrice])
      .range([chartHeight, 0]);

    const maxSize = Math.max(
      ...processedData.bids.map(b => b.size),
      ...processedData.asks.map(a => a.size)
    );

    const sizeScale = d3.scaleLinear()
      .domain([0, maxSize])
      .range([0, width / 2 - 10]);

    // Render bid levels
    g.selectAll('.bid-level')
      .data(processedData.bids.slice(0, depth))
      .enter()
      .append('g')
      .attr('class', 'bid-level')
      .each(function(d) {
        const bidGroup = d3.select(this);
        
        // Background bar
        bidGroup.append('rect')
          .attr('x', width / 2 - sizeScale(d.size))
          .attr('y', priceScale(d.price) - 8)
          .attr('width', sizeScale(d.size))
          .attr('height', 16)
          .attr('fill', `rgba(76, 175, 80, ${0.3 + d.percentage / 100 * 0.7})`)
          .attr('stroke', '#4CAF50')
          .attr('stroke-width', 0.5);

        // Price text
        bidGroup.append('text')
          .attr('x', width / 2 - 5)
          .attr('y', priceScale(d.price) + 4)
          .attr('text-anchor', 'end')
          .attr('font-size', '11px')
          .attr('font-weight', 'bold')
          .attr('fill', '#4CAF50')
          .text(d.price.toFixed(2));

        // Size text
        bidGroup.append('text')
          .attr('x', width / 2 - sizeScale(d.size) - 5)
          .attr('y', priceScale(d.price) + 4)
          .attr('text-anchor', 'end')
          .attr('font-size', '10px')
          .attr('fill', '#666')
          .text(d.size.toLocaleString());
      })
      .style('cursor', 'pointer')
      .on('click', (event, d) => onLevelClick?.(d, 'bid'));

    // Render ask levels
    g.selectAll('.ask-level')
      .data(processedData.asks.slice(0, depth))
      .enter()
      .append('g')
      .attr('class', 'ask-level')
      .each(function(d) {
        const askGroup = d3.select(this);
        
        // Background bar
        askGroup.append('rect')
          .attr('x', width / 2)
          .attr('y', priceScale(d.price) - 8)
          .attr('width', sizeScale(d.size))
          .attr('height', 16)
          .attr('fill', `rgba(244, 67, 54, ${0.3 + d.percentage / 100 * 0.7})`)
          .attr('stroke', '#F44336')
          .attr('stroke-width', 0.5);

        // Price text
        askGroup.append('text')
          .attr('x', width / 2 + 5)
          .attr('y', priceScale(d.price) + 4)
          .attr('text-anchor', 'start')
          .attr('font-size', '11px')
          .attr('font-weight', 'bold')
          .attr('fill', '#F44336')
          .text(d.price.toFixed(2));

        // Size text
        askGroup.append('text')
          .attr('x', width / 2 + sizeScale(d.size) + 5)
          .attr('y', priceScale(d.price) + 4)
          .attr('text-anchor', 'start')
          .attr('font-size', '10px')
          .attr('fill', '#666')
          .text(d.size.toLocaleString());
      })
      .style('cursor', 'pointer')
      .on('click', (event, d) => onLevelClick?.(d, 'ask'));

    // Spread indicator
    if (processedData.spread) {
      const midY = (priceScale(processedData.bids[0]?.price || 0) + 
                   priceScale(processedData.asks[0]?.price || 0)) / 2;
      
      g.append('line')
        .attr('x1', 0)
        .attr('x2', width)
        .attr('y1', midY)
        .attr('y2', midY)
        .attr('stroke', '#FFA726')
        .attr('stroke-width', 2)
        .attr('stroke-dasharray', '3,3');

      g.append('text')
        .attr('x', width / 2)
        .attr('y', midY - 5)
        .attr('text-anchor', 'middle')
        .attr('font-size', '10px')
        .attr('font-weight', 'bold')
        .attr('fill', '#FF9800')
        .text(`Spread: ${processedData.spread.toFixed(4)} (${processedData.spreadBps.toFixed(1)} bps)`);
    }

  }, [processedData, depth, height, onLevelClick]);

  // Heatmap visualization
  const renderHeatmap = useCallback(() => {
    if (!heatmapData.current.length || !svgRef.current) return;

    const svg = d3.select(svgRef.current);
    svg.selectAll('*').remove();

    const margin = { top: 20, right: 20, bottom: 20, left: 80 };
    const width = (containerRef.current?.clientWidth || 800) - margin.left - margin.right;
    const chartHeight = height - margin.top - margin.bottom;

    const g = svg.append('g')
      .attr('transform', `translate(${margin.left},${margin.top})`);

    // Group data by price levels
    const priceGroups = d3.groups(heatmapData.current, d => Math.round(d.price / aggregation) * aggregation);
    const timeExtent = d3.extent(heatmapData.current, d => d.time) as [number, number];

    const priceScale = d3.scaleLinear()
      .domain(d3.extent(priceGroups, d => d[0]) as [number, number])
      .range([chartHeight, 0]);

    const timeScale = d3.scaleLinear()
      .domain(timeExtent)
      .range([0, width]);

    const colorScale = d3.scaleSequential(d3.interpolateRdYlGn)
      .domain([0, 1]);

    // Render heatmap cells
    priceGroups.forEach(([price, cells]) => {
      const timeGroups = d3.groups(cells, d => Math.floor(d.time / 1000) * 1000);
      
      timeGroups.forEach(([time, timeCells]) => {
        const avgIntensity = d3.mean(timeCells, d => d.intensity) || 0;
        
        g.append('rect')
          .attr('x', timeScale(time))
          .attr('y', priceScale(price) - 2)
          .attr('width', Math.max(2, timeScale(timeExtent[1]) - timeScale(timeExtent[0])) / timeGroups.length)
          .attr('height', 4)
          .attr('fill', colorScale(avgIntensity))
          .attr('opacity', 0.8);
      });
    });

    // Add price axis
    const priceAxis = d3.axisLeft(priceScale).tickFormat(d => `$${d}`);
    g.append('g').call(priceAxis);

    // Add time axis
    const timeAxis = d3.axisBottom(timeScale)
      .tickFormat(d => new Date(d).toLocaleTimeString());
    g.append('g')
      .attr('transform', `translate(0,${chartHeight})`)
      .call(timeAxis);

  }, [height, aggregation]);

  // Volume profile visualization
  const renderVolumeProfile = useCallback(() => {
    if (!volumeProfileData.current.length || !svgRef.current) return;

    const svg = d3.select(svgRef.current);
    svg.selectAll('*').remove();

    const margin = { top: 20, right: 20, bottom: 20, left: 80 };
    const width = (containerRef.current?.clientWidth || 800) - margin.left - margin.right;
    const chartHeight = height - margin.top - margin.bottom;

    const g = svg.append('g')
      .attr('transform', `translate(${margin.left},${margin.top})`);

    const data = volumeProfileData.current;
    const priceScale = d3.scaleLinear()
      .domain(d3.extent(data, d => d.price) as [number, number])
      .range([chartHeight, 0]);

    const volumeScale = d3.scaleLinear()
      .domain([0, d3.max(data, d => d.volume) || 0])
      .range([0, width]);

    // Render volume bars
    g.selectAll('.volume-bar')
      .data(data)
      .enter()
      .append('g')
      .attr('class', 'volume-bar')
      .each(function(d) {
        const barGroup = d3.select(this);
        
        // Buy volume (left side)
        barGroup.append('rect')
          .attr('x', width / 2 - volumeScale(d.buyVolume))
          .attr('y', priceScale(d.price) - 2)
          .attr('width', volumeScale(d.buyVolume))
          .attr('height', 4)
          .attr('fill', '#4CAF50')
          .attr('opacity', 0.7);

        // Sell volume (right side)
        barGroup.append('rect')
          .attr('x', width / 2)
          .attr('y', priceScale(d.price) - 2)
          .attr('width', volumeScale(d.sellVolume))
          .attr('height', 4)
          .attr('fill', '#F44336')
          .attr('opacity', 0.7);

        // Price label
        barGroup.append('text')
          .attr('x', width / 2)
          .attr('y', priceScale(d.price) + 1)
          .attr('text-anchor', 'middle')
          .attr('font-size', '9px')
          .attr('fill', '#666')
          .text(`$${d.price.toFixed(2)}`);
      });

    // Add axes
    const priceAxis = d3.axisLeft(priceScale).tickFormat(d => `$${d}`);
    g.append('g').call(priceAxis);

  }, [height]);

  // Update visualization based on view mode
  useEffect(() => {
    switch (viewMode) {
      case 'traditional':
        renderTraditionalOrderBook();
        break;
      case 'heatmap':
        renderHeatmap();
        break;
      case 'volumeProfile':
        renderVolumeProfile();
        break;
      default:
        renderTraditionalOrderBook();
    }
  }, [viewMode, processedData, renderTraditionalOrderBook, renderHeatmap, renderVolumeProfile]);

  // Handle export functionality
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
      link.download = `orderbook-${symbol}-${viewMode}-${Date.now()}.png`;
      link.href = canvas.toDataURL();
      link.click();
    };

    img.src = `data:image/svg+xml;base64,${btoa(svgData)}`;
  }, [symbol, viewMode, exportEnabled]);

  // Settings menu
  const settingsMenu = (
    <Menu>
      <Menu.SubMenu key="aggregation" title="Price Aggregation">
        {aggregationLevels.map(level => (
          <Menu.Item
            key={level}
            onClick={() => setAggregation(level)}
            style={{ backgroundColor: aggregation === level ? '#e6f7ff' : 'transparent' }}
          >
            ${level.toFixed(4)}
          </Menu.Item>
        ))}
      </Menu.SubMenu>
      <Menu.SubMenu key="filters" title="Filters">
        <Menu.Item key="price-range">
          <Space direction="vertical" size="small">
            <Text style={{ fontSize: '11px' }}>Price Range</Text>
            <Space>
              <InputNumber
                size="small"
                placeholder="Min"
                value={priceFilter.min}
                onChange={(value) => setPriceFilter(prev => ({ ...prev, min: value || undefined }))}
                style={{ width: 70 }}
              />
              <InputNumber
                size="small"
                placeholder="Max"
                value={priceFilter.max}
                onChange={(value) => setPriceFilter(prev => ({ ...prev, max: value || undefined }))}
                style={{ width: 70 }}
              />
            </Space>
          </Space>
        </Menu.Item>
        <Menu.Item key="size-range">
          <Space direction="vertical" size="small">
            <Text style={{ fontSize: '11px' }}>Size Range</Text>
            <Space>
              <InputNumber
                size="small"
                placeholder="Min"
                value={sizeFilter.min}
                onChange={(value) => setSizeFilter(prev => ({ ...prev, min: value || undefined }))}
                style={{ width: 70 }}
              />
              <InputNumber
                size="small"
                placeholder="Max"
                value={sizeFilter.max}
                onChange={(value) => setSizeFilter(prev => ({ ...prev, max: value || undefined }))}
                style={{ width: 70 }}
              />
            </Space>
          </Space>
        </Menu.Item>
      </Menu.SubMenu>
      <Menu.Item key="reset-filters" onClick={() => {
        setPriceFilter({});
        setSizeFilter({});
      }}>
        Reset Filters
      </Menu.Item>
    </Menu>
  );

  if (error) {
    return (
      <Card title="Order Book - Error" style={{ height }}>
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
            <Text strong>{symbol} Order Book</Text>
            <Badge 
              status={isConnected ? 'success' : 'error'} 
              text={isConnected ? 'Live' : 'Disconnected'} 
            />
          </Space>
        }
        extra={
          <Space>
            {/* View Mode Selector */}
            <Select
              value={viewMode}
              onChange={setViewMode}
              size="small"
              style={{ width: 120 }}
            >
              <Select.Option value="traditional">Traditional</Select.Option>
              <Select.Option value="heatmap">Heatmap</Select.Option>
              <Select.Option value="volumeProfile">Volume Profile</Select.Option>
            </Select>

            {/* Zoom Controls */}
            <Button.Group size="small">
              <Button
                icon={<ZoomOutOutlined />}
                onClick={() => setZoom(prev => Math.max(0.5, prev - 0.1))}
                disabled={zoom <= 0.5}
              />
              <Button
                icon={<ZoomInOutlined />}
                onClick={() => setZoom(prev => Math.min(2, prev + 0.1))}
                disabled={zoom >= 2}
              />
            </Button.Group>

            {/* Controls */}
            <Tooltip title="Cumulative View">
              <Switch
                checked={showCumulative}
                onChange={setShowCumulative}
                size="small"
              />
            </Tooltip>

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
        {/* Statistics Row */}
        {!compactMode && processedData && (
          <Row gutter={16} style={{ marginBottom: 16 }}>
            <Col span={6}>
              <Statistic
                title="Bid Volume"
                value={processedData.totalBidVolume}
                precision={0}
                valueStyle={{ color: '#4CAF50', fontSize: '14px' }}
              />
            </Col>
            <Col span={6}>
              <Statistic
                title="Ask Volume"
                value={processedData.totalAskVolume}
                precision={0}
                valueStyle={{ color: '#F44336', fontSize: '14px' }}
              />
            </Col>
            <Col span={6}>
              <Statistic
                title="Spread"
                value={processedData.spread}
                precision={4}
                suffix="$"
                valueStyle={{ color: '#FF9800', fontSize: '14px' }}
              />
            </Col>
            <Col span={6}>
              <Statistic
                title="Mid Price"
                value={processedData.midPrice}
                precision={2}
                suffix="$"
                valueStyle={{ color: '#2196F3', fontSize: '14px' }}
              />
            </Col>
          </Row>
        )}

        {/* Visualization */}
        <div style={{ 
          width: '100%', 
          height: compactMode ? height - 100 : height - 150,
          overflow: 'hidden'
        }}>
          <svg
            ref={svgRef}
            width="100%"
            height="100%"
            style={{ background: theme === 'dark' ? '#1f1f1f' : '#ffffff' }}
          />
        </div>

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
                Depth: {depth} levels
              </Text>
              <Text type="secondary" style={{ fontSize: '11px' }}>
                Aggregation: ${aggregation.toFixed(4)}
              </Text>
            </Space>
            
            <Space>
              <Progress
                percent={Math.round((processedData?.totalBidVolume || 0) / 
                        ((processedData?.totalBidVolume || 0) + (processedData?.totalAskVolume || 0)) * 100)}
                size="small"
                strokeColor="#4CAF50"
                trailColor="#F44336"
                showInfo={false}
                style={{ width: 100 }}
              />
              <Text style={{ fontSize: '11px' }}>Bid/Ask Ratio</Text>
            </Space>
          </div>
        )}
      </Card>
    </div>
  );
});

AdvancedOrderBookWidget.displayName = 'AdvancedOrderBookWidget';

export default AdvancedOrderBookWidget;