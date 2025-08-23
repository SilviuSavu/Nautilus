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
  Switch,
  Slider,
  Dropdown,
  Menu,
  Modal,
  Input,
  Divider,
  Tag
} from 'antd';
import {
  LineChartOutlined,
  BarChartOutlined,
  SettingOutlined,
  ExportOutlined,
  FullscreenOutlined,
  ZoomInOutlined,
  ZoomOutOutlined,
  DragOutlined,
  EditOutlined,
  SaveOutlined,
  ReloadOutlined,
  PlusOutlined,
  DeleteOutlined
} from '@ant-design/icons';
import * as d3 from 'd3';
import { useMarketData } from '../../hooks/useMarketData';
import { useChartStore } from '../Chart/hooks/useChartStore';

const { Text, Title } = Typography;

interface CandlestickData {
  timestamp: number;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
  vwap?: number;
}

interface TechnicalIndicator {
  id: string;
  name: string;
  type: 'overlay' | 'oscillator' | 'volume';
  enabled: boolean;
  parameters: Record<string, any>;
  data: Array<{
    timestamp: number;
    value: number | number[];
    color?: string;
  }>;
  style: {
    lineColor: string;
    lineWidth: number;
    lineStyle: 'solid' | 'dashed' | 'dotted';
    fillColor?: string;
    opacity: number;
  };
}

interface DrawingTool {
  id: string;
  type: 'trendline' | 'horizontal' | 'vertical' | 'rectangle' | 'fibonacci' | 'text';
  points: Array<{ x: number; y: number; timestamp?: number; price?: number }>;
  style: {
    color: string;
    width: number;
    style: 'solid' | 'dashed' | 'dotted';
    fill?: string;
    opacity?: number;
  };
  text?: string;
  locked: boolean;
  visible: boolean;
}

interface AdvancedTradingChartProps {
  symbol: string;
  timeframe?: '1m' | '5m' | '15m' | '1h' | '4h' | '1d';
  chartType?: 'candlestick' | 'line' | 'area' | 'heikin-ashi' | 'renko';
  indicators?: TechnicalIndicator[];
  drawings?: DrawingTool[];
  showVolume?: boolean;
  showOrderBook?: boolean;
  showTrades?: boolean;
  enableDrawing?: boolean;
  enableCrosshair?: boolean;
  compactMode?: boolean;
  height?: number;
  theme?: 'light' | 'dark';
  onCandleClick?: (candle: CandlestickData) => void;
  onDrawingComplete?: (drawing: DrawingTool) => void;
  onIndicatorChange?: (indicators: TechnicalIndicator[]) => void;
  exportEnabled?: boolean;
}

interface ChartState {
  zoom: number;
  pan: { x: number; y: number };
  crosshair: { x: number; y: number; visible: boolean };
  selectedDrawing: string | null;
  drawingMode: string | null;
  mousePosition: { x: number; y: number };
  visibleRange: { start: number; end: number };
}

const AdvancedTradingChart: React.FC<AdvancedTradingChartProps> = memo(({
  symbol,
  timeframe = '1h',
  chartType = 'candlestick',
  indicators = [],
  drawings = [],
  showVolume = true,
  showOrderBook = false,
  showTrades = true,
  enableDrawing = true,
  enableCrosshair = true,
  compactMode = false,
  height = 600,
  theme = 'light',
  onCandleClick,
  onDrawingComplete,
  onIndicatorChange,
  exportEnabled = true
}) => {
  const [chartState, setChartState] = useState<ChartState>({
    zoom: 1,
    pan: { x: 0, y: 0 },
    crosshair: { x: 0, y: 0, visible: false },
    selectedDrawing: null,
    drawingMode: null,
    mousePosition: { x: 0, y: 0 },
    visibleRange: { start: 0, end: 100 }
  });

  const [activeIndicators, setActiveIndicators] = useState<TechnicalIndicator[]>(indicators);
  const [activeDrawings, setActiveDrawings] = useState<DrawingTool[]>(drawings);
  const [indicatorModalVisible, setIndicatorModalVisible] = useState(false);
  const [selectedIndicator, setSelectedIndicator] = useState<string | null>(null);
  const [fullscreen, setFullscreen] = useState(false);

  const svgRef = useRef<SVGSVGElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const chartStore = useChartStore();

  // Market data hook
  const {
    data: marketData,
    isConnected,
    lastUpdate,
    error
  } = useMarketData({
    symbol,
    timeframe,
    autoRefresh: true
  });

  // Process market data into candlesticks
  const candlestickData = useMemo(() => {
    if (!marketData?.bars) return [];
    
    return marketData.bars.map((bar: any) => ({
      timestamp: new Date(bar.timestamp).getTime(),
      open: bar.open,
      high: bar.high,
      low: bar.low,
      close: bar.close,
      volume: bar.volume,
      vwap: bar.vwap
    }));
  }, [marketData]);

  // Calculate technical indicators
  const calculateIndicators = useCallback((data: CandlestickData[], indicators: TechnicalIndicator[]) => {
    return indicators.map(indicator => {
      const calculatedData: Array<{ timestamp: number; value: number | number[] }> = [];
      
      switch (indicator.name) {
        case 'SMA':
          const period = indicator.parameters.period || 20;
          data.forEach((candle, index) => {
            if (index >= period - 1) {
              const sum = data.slice(index - period + 1, index + 1)
                .reduce((acc, c) => acc + c.close, 0);
              calculatedData.push({
                timestamp: candle.timestamp,
                value: sum / period
              });
            }
          });
          break;
          
        case 'EMA':
          const emaPeriod = indicator.parameters.period || 20;
          const multiplier = 2 / (emaPeriod + 1);
          let ema = data[0]?.close || 0;
          
          data.forEach((candle, index) => {
            if (index === 0) {
              ema = candle.close;
            } else {
              ema = (candle.close - ema) * multiplier + ema;
            }
            calculatedData.push({
              timestamp: candle.timestamp,
              value: ema
            });
          });
          break;
          
        case 'RSI':
          const rsiPeriod = indicator.parameters.period || 14;
          const gains: number[] = [];
          const losses: number[] = [];
          
          data.forEach((candle, index) => {
            if (index > 0) {
              const change = candle.close - data[index - 1].close;
              gains.push(Math.max(change, 0));
              losses.push(Math.max(-change, 0));
              
              if (gains.length >= rsiPeriod) {
                const avgGain = gains.slice(-rsiPeriod).reduce((a, b) => a + b, 0) / rsiPeriod;
                const avgLoss = losses.slice(-rsiPeriod).reduce((a, b) => a + b, 0) / rsiPeriod;
                const rs = avgGain / (avgLoss || 1);
                const rsi = 100 - (100 / (1 + rs));
                
                calculatedData.push({
                  timestamp: candle.timestamp,
                  value: rsi
                });
              }
            }
          });
          break;
          
        case 'MACD':
          const fastPeriod = indicator.parameters.fastPeriod || 12;
          const slowPeriod = indicator.parameters.slowPeriod || 26;
          const signalPeriod = indicator.parameters.signalPeriod || 9;
          
          // Calculate EMAs
          const fastEMA = calculateEMA(data.map(d => d.close), fastPeriod);
          const slowEMA = calculateEMA(data.map(d => d.close), slowPeriod);
          
          // Calculate MACD line
          const macdLine = fastEMA.map((fast, index) => fast - slowEMA[index]);
          
          // Calculate signal line
          const signalLine = calculateEMA(macdLine, signalPeriod);
          
          // Calculate histogram
          data.forEach((candle, index) => {
            if (index < slowPeriod - 1) return;
            
            const macd = macdLine[index];
            const signal = signalLine[index];
            const histogram = macd - signal;
            
            calculatedData.push({
              timestamp: candle.timestamp,
              value: [macd, signal, histogram]
            });
          });
          break;
          
        case 'Bollinger Bands':
          const bbPeriod = indicator.parameters.period || 20;
          const bbStdDev = indicator.parameters.stdDev || 2;
          
          data.forEach((candle, index) => {
            if (index >= bbPeriod - 1) {
              const prices = data.slice(index - bbPeriod + 1, index + 1).map(d => d.close);
              const sma = prices.reduce((a, b) => a + b, 0) / bbPeriod;
              const variance = prices.reduce((acc, price) => acc + Math.pow(price - sma, 2), 0) / bbPeriod;
              const stdDev = Math.sqrt(variance);
              
              const upper = sma + (bbStdDev * stdDev);
              const lower = sma - (bbStdDev * stdDev);
              
              calculatedData.push({
                timestamp: candle.timestamp,
                value: [upper, sma, lower]
              });
            }
          });
          break;
      }
      
      return {
        ...indicator,
        data: calculatedData
      };
    });
  }, []);

  // Helper function to calculate EMA
  const calculateEMA = (prices: number[], period: number): number[] => {
    const multiplier = 2 / (period + 1);
    const ema: number[] = [];
    
    prices.forEach((price, index) => {
      if (index === 0) {
        ema.push(price);
      } else {
        ema.push((price - ema[index - 1]) * multiplier + ema[index - 1]);
      }
    });
    
    return ema;
  };

  // Update indicators when data changes
  useEffect(() => {
    if (candlestickData.length > 0 && activeIndicators.length > 0) {
      const calculatedIndicators = calculateIndicators(candlestickData, activeIndicators);
      setActiveIndicators(calculatedIndicators);
      onIndicatorChange?.(calculatedIndicators);
    }
  }, [candlestickData, calculateIndicators, onIndicatorChange]);

  // Render main chart
  const renderChart = useCallback(() => {
    if (!candlestickData.length || !svgRef.current) return;

    const svg = d3.select(svgRef.current);
    svg.selectAll('*').remove();

    const margin = { top: 20, right: 60, bottom: showVolume ? 80 : 40, left: 60 };
    const width = (containerRef.current?.clientWidth || 800) - margin.left - margin.right;
    const mainHeight = (showVolume ? height * 0.7 : height) - margin.top - margin.bottom;
    const volumeHeight = showVolume ? height * 0.25 : 0;

    const g = svg.append('g')
      .attr('transform', `translate(${margin.left},${margin.top})`);

    // Visible data based on zoom and pan
    const visibleData = candlestickData.slice(
      Math.max(0, Math.floor(chartState.visibleRange.start)),
      Math.min(candlestickData.length, Math.ceil(chartState.visibleRange.end))
    );

    if (!visibleData.length) return;

    // Scales
    const xScale = d3.scaleTime()
      .domain(d3.extent(visibleData, d => new Date(d.timestamp)) as [Date, Date])
      .range([0, width]);

    const yScale = d3.scaleLinear()
      .domain(d3.extent(visibleData.flatMap(d => [d.high, d.low])) as [number, number])
      .range([mainHeight, 0]);

    const volumeScale = d3.scaleLinear()
      .domain([0, d3.max(visibleData, d => d.volume) || 0])
      .range([mainHeight + volumeHeight, mainHeight + 10]);

    // Render candlesticks or chart type
    if (chartType === 'candlestick') {
      const candleWidth = Math.max(1, width / visibleData.length * 0.8);
      
      visibleData.forEach(candle => {
        const x = xScale(new Date(candle.timestamp));
        const isGreen = candle.close >= candle.open;
        
        // Candle body
        g.append('rect')
          .attr('x', x - candleWidth / 2)
          .attr('y', yScale(Math.max(candle.open, candle.close)))
          .attr('width', candleWidth)
          .attr('height', Math.max(1, Math.abs(yScale(candle.open) - yScale(candle.close))))
          .attr('fill', isGreen ? '#26a69a' : '#ef5350')
          .attr('stroke', isGreen ? '#26a69a' : '#ef5350')
          .style('cursor', 'pointer')
          .on('click', () => onCandleClick?.(candle));

        // Wicks
        g.append('line')
          .attr('x1', x)
          .attr('x2', x)
          .attr('y1', yScale(candle.high))
          .attr('y2', yScale(candle.low))
          .attr('stroke', isGreen ? '#26a69a' : '#ef5350')
          .attr('stroke-width', 1);
      });
    } else if (chartType === 'line') {
      const line = d3.line<CandlestickData>()
        .x(d => xScale(new Date(d.timestamp)))
        .y(d => yScale(d.close))
        .curve(d3.curveMonotoneX);

      g.append('path')
        .datum(visibleData)
        .attr('fill', 'none')
        .attr('stroke', '#1890ff')
        .attr('stroke-width', 2)
        .attr('d', line);
    } else if (chartType === 'area') {
      const area = d3.area<CandlestickData>()
        .x(d => xScale(new Date(d.timestamp)))
        .y0(mainHeight)
        .y1(d => yScale(d.close))
        .curve(d3.curveMonotoneX);

      g.append('path')
        .datum(visibleData)
        .attr('fill', 'rgba(24, 144, 255, 0.3)')
        .attr('stroke', '#1890ff')
        .attr('stroke-width', 2)
        .attr('d', area);
    }

    // Render volume bars
    if (showVolume) {
      const volumeBarWidth = Math.max(1, width / visibleData.length * 0.6);
      
      visibleData.forEach(candle => {
        const x = xScale(new Date(candle.timestamp));
        const isGreen = candle.close >= candle.open;
        
        g.append('rect')
          .attr('x', x - volumeBarWidth / 2)
          .attr('y', volumeScale(candle.volume))
          .attr('width', volumeBarWidth)
          .attr('height', mainHeight + volumeHeight - volumeScale(candle.volume))
          .attr('fill', isGreen ? 'rgba(38, 166, 154, 0.5)' : 'rgba(239, 83, 80, 0.5)');
      });
    }

    // Render indicators
    activeIndicators.forEach(indicator => {
      if (!indicator.enabled || !indicator.data.length) return;

      const indicatorData = indicator.data.filter(d => 
        visibleData.some(vd => vd.timestamp === d.timestamp)
      );

      if (indicator.type === 'overlay') {
        // Overlay indicators (on main chart)
        if (Array.isArray(indicatorData[0]?.value)) {
          // Multiple lines (e.g., Bollinger Bands)
          (indicatorData[0].value as number[]).forEach((_, lineIndex) => {
            const line = d3.line<any>()
              .x(d => xScale(new Date(d.timestamp)))
              .y(d => yScale(d.value[lineIndex]))
              .curve(d3.curveMonotoneX);

            g.append('path')
              .datum(indicatorData.filter(d => d.value[lineIndex] !== undefined))
              .attr('fill', 'none')
              .attr('stroke', getIndicatorColor(indicator.name, lineIndex))
              .attr('stroke-width', indicator.style.lineWidth || 1)
              .attr('stroke-dasharray', getStrokeDashArray(indicator.style.lineStyle))
              .attr('opacity', indicator.style.opacity || 1)
              .attr('d', line);
          });
        } else {
          // Single line
          const line = d3.line<any>()
            .x(d => xScale(new Date(d.timestamp)))
            .y(d => yScale(d.value as number))
            .curve(d3.curveMonotoneX);

          g.append('path')
            .datum(indicatorData)
            .attr('fill', 'none')
            .attr('stroke', indicator.style.lineColor)
            .attr('stroke-width', indicator.style.lineWidth || 1)
            .attr('stroke-dasharray', getStrokeDashArray(indicator.style.lineStyle))
            .attr('opacity', indicator.style.opacity || 1)
            .attr('d', line);
        }
      } else if (indicator.type === 'oscillator') {
        // Oscillator indicators (separate panel below main chart)
        const oscillatorY = mainHeight + (showVolume ? volumeHeight + 20 : 20);
        const oscillatorHeight = 80;
        
        const oscillatorScale = indicator.name === 'RSI' 
          ? d3.scaleLinear().domain([0, 100]).range([oscillatorY + oscillatorHeight, oscillatorY])
          : d3.scaleLinear()
              .domain(d3.extent(indicatorData, d => Array.isArray(d.value) ? d.value[0] : d.value) as [number, number])
              .range([oscillatorY + oscillatorHeight, oscillatorY]);

        // Background
        g.append('rect')
          .attr('x', 0)
          .attr('y', oscillatorY)
          .attr('width', width)
          .attr('height', oscillatorHeight)
          .attr('fill', theme === 'dark' ? '#1f1f1f' : '#fafafa')
          .attr('stroke', '#d9d9d9');

        if (Array.isArray(indicatorData[0]?.value)) {
          // MACD
          indicatorData.forEach(d => {
            const x = xScale(new Date(d.timestamp));
            const [macd, signal, histogram] = d.value as number[];
            
            // Histogram
            const histogramColor = histogram >= 0 ? '#26a69a' : '#ef5350';
            g.append('rect')
              .attr('x', x - 1)
              .attr('y', Math.min(oscillatorScale(histogram), oscillatorScale(0)))
              .attr('width', 2)
              .attr('height', Math.abs(oscillatorScale(histogram) - oscillatorScale(0)))
              .attr('fill', histogramColor)
              .attr('opacity', 0.6);
          });

          // MACD and Signal lines
          const macdLine = d3.line<any>()
            .x(d => xScale(new Date(d.timestamp)))
            .y(d => oscillatorScale(d.value[0]))
            .curve(d3.curveMonotoneX);

          const signalLine = d3.line<any>()
            .x(d => xScale(new Date(d.timestamp)))
            .y(d => oscillatorScale(d.value[1]))
            .curve(d3.curveMonotoneX);

          g.append('path')
            .datum(indicatorData)
            .attr('fill', 'none')
            .attr('stroke', '#1890ff')
            .attr('stroke-width', 1)
            .attr('d', macdLine);

          g.append('path')
            .datum(indicatorData)
            .attr('fill', 'none')
            .attr('stroke', '#f5222d')
            .attr('stroke-width', 1)
            .attr('d', signalLine);
        } else {
          // Single line oscillator (RSI)
          const line = d3.line<any>()
            .x(d => xScale(new Date(d.timestamp)))
            .y(d => oscillatorScale(d.value as number))
            .curve(d3.curveMonotoneX);

          g.append('path')
            .datum(indicatorData)
            .attr('fill', 'none')
            .attr('stroke', indicator.style.lineColor)
            .attr('stroke-width', indicator.style.lineWidth || 1)
            .attr('d', line);

          // RSI reference lines
          if (indicator.name === 'RSI') {
            [70, 50, 30].forEach(level => {
              g.append('line')
                .attr('x1', 0)
                .attr('x2', width)
                .attr('y1', oscillatorScale(level))
                .attr('y2', oscillatorScale(level))
                .attr('stroke', level === 50 ? '#999' : '#ddd')
                .attr('stroke-dasharray', '3,3')
                .attr('opacity', 0.5);
            });
          }
        }
      }
    });

    // Render drawings
    activeDrawings.forEach(drawing => {
      if (!drawing.visible) return;

      switch (drawing.type) {
        case 'trendline':
          if (drawing.points.length >= 2) {
            const [p1, p2] = drawing.points;
            g.append('line')
              .attr('x1', xScale(new Date(p1.timestamp!)))
              .attr('y1', yScale(p1.price!))
              .attr('x2', xScale(new Date(p2.timestamp!)))
              .attr('y2', yScale(p2.price!))
              .attr('stroke', drawing.style.color)
              .attr('stroke-width', drawing.style.width)
              .attr('stroke-dasharray', getStrokeDashArray(drawing.style.style));
          }
          break;

        case 'horizontal':
          if (drawing.points.length >= 1) {
            const price = drawing.points[0].price!;
            g.append('line')
              .attr('x1', 0)
              .attr('x2', width)
              .attr('y1', yScale(price))
              .attr('y2', yScale(price))
              .attr('stroke', drawing.style.color)
              .attr('stroke-width', drawing.style.width)
              .attr('stroke-dasharray', getStrokeDashArray(drawing.style.style));
          }
          break;

        case 'rectangle':
          if (drawing.points.length >= 2) {
            const [p1, p2] = drawing.points;
            const x1 = xScale(new Date(p1.timestamp!));
            const y1 = yScale(p1.price!);
            const x2 = xScale(new Date(p2.timestamp!));
            const y2 = yScale(p2.price!);
            
            g.append('rect')
              .attr('x', Math.min(x1, x2))
              .attr('y', Math.min(y1, y2))
              .attr('width', Math.abs(x2 - x1))
              .attr('height', Math.abs(y2 - y1))
              .attr('fill', drawing.style.fill || 'transparent')
              .attr('stroke', drawing.style.color)
              .attr('stroke-width', drawing.style.width)
              .attr('opacity', drawing.style.opacity || 1);
          }
          break;
      }
    });

    // Crosshair
    if (enableCrosshair && chartState.crosshair.visible) {
      const crosshairGroup = g.append('g').attr('class', 'crosshair');
      
      // Vertical line
      crosshairGroup.append('line')
        .attr('x1', chartState.crosshair.x)
        .attr('x2', chartState.crosshair.x)
        .attr('y1', 0)
        .attr('y2', mainHeight)
        .attr('stroke', '#999')
        .attr('stroke-width', 1)
        .attr('stroke-dasharray', '3,3')
        .attr('opacity', 0.7);

      // Horizontal line
      crosshairGroup.append('line')
        .attr('x1', 0)
        .attr('x2', width)
        .attr('y1', chartState.crosshair.y)
        .attr('y2', chartState.crosshair.y)
        .attr('stroke', '#999')
        .attr('stroke-width', 1)
        .attr('stroke-dasharray', '3,3')
        .attr('opacity', 0.7);

      // Price label
      const price = yScale.invert(chartState.crosshair.y);
      crosshairGroup.append('rect')
        .attr('x', width + 5)
        .attr('y', chartState.crosshair.y - 10)
        .attr('width', 50)
        .attr('height', 20)
        .attr('fill', '#333')
        .attr('opacity', 0.8);

      crosshairGroup.append('text')
        .attr('x', width + 30)
        .attr('y', chartState.crosshair.y + 4)
        .attr('text-anchor', 'middle')
        .attr('fill', '#fff')
        .attr('font-size', '11px')
        .text(price.toFixed(2));
    }

    // Axes
    const xAxis = d3.axisBottom(xScale).tickFormat(d3.timeFormat('%H:%M'));
    g.append('g')
      .attr('transform', `translate(0,${mainHeight})`)
      .call(xAxis);

    const yAxis = d3.axisRight(yScale).tickFormat(d => `$${d.toFixed(2)}`);
    g.append('g')
      .attr('transform', `translate(${width},0)`)
      .call(yAxis);

    // Volume axis
    if (showVolume) {
      const volumeAxis = d3.axisRight(volumeScale).tickFormat(d3.format('.2s'));
      g.append('g')
        .attr('transform', `translate(${width},0)`)
        .call(volumeAxis);
    }

    // Mouse events for crosshair and interactions
    svg.on('mousemove', function(event) {
      const [mouseX, mouseY] = d3.pointer(event, g.node());
      
      setChartState(prev => ({
        ...prev,
        crosshair: { x: mouseX, y: mouseY, visible: enableCrosshair },
        mousePosition: { x: mouseX, y: mouseY }
      }));
    })
    .on('mouseleave', function() {
      setChartState(prev => ({
        ...prev,
        crosshair: { ...prev.crosshair, visible: false }
      }));
    });

  }, [candlestickData, chartState, activeIndicators, activeDrawings, chartType, showVolume, height, enableCrosshair, theme, onCandleClick]);

  // Helper functions
  const getIndicatorColor = (name: string, index: number = 0): string => {
    const colorMap: Record<string, string[]> = {
      'SMA': ['#1890ff'],
      'EMA': ['#f5222d'],
      'Bollinger Bands': ['#722ed1', '#1890ff', '#722ed1'],
      'RSI': ['#fa541c'],
      'MACD': ['#1890ff', '#f5222d', '#52c41a']
    };
    return colorMap[name]?.[index] || '#1890ff';
  };

  const getStrokeDashArray = (style: string): string => {
    switch (style) {
      case 'dashed': return '5,5';
      case 'dotted': return '2,2';
      default: return 'none';
    }
  };

  // Add indicator
  const addIndicator = (type: string) => {
    const newIndicator: TechnicalIndicator = {
      id: `${type}-${Date.now()}`,
      name: type,
      type: ['RSI', 'MACD'].includes(type) ? 'oscillator' : 'overlay',
      enabled: true,
      parameters: getDefaultParameters(type),
      data: [],
      style: {
        lineColor: getIndicatorColor(type),
        lineWidth: 1,
        lineStyle: 'solid',
        opacity: 1
      }
    };

    const updated = [...activeIndicators, newIndicator];
    setActiveIndicators(updated);
    onIndicatorChange?.(updated);
  };

  const getDefaultParameters = (type: string): Record<string, any> => {
    switch (type) {
      case 'SMA':
      case 'EMA':
        return { period: 20 };
      case 'RSI':
        return { period: 14 };
      case 'MACD':
        return { fastPeriod: 12, slowPeriod: 26, signalPeriod: 9 };
      case 'Bollinger Bands':
        return { period: 20, stdDev: 2 };
      default:
        return {};
    }
  };

  // Update chart when data or settings change
  useEffect(() => {
    renderChart();
  }, [renderChart]);

  // Export functionality
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
      link.download = `trading-chart-${symbol}-${timeframe}-${Date.now()}.png`;
      link.href = canvas.toDataURL();
      link.click();
    };

    img.src = `data:image/svg+xml;base64,${btoa(svgData)}`;
  }, [symbol, timeframe, exportEnabled]);

  // Settings menu
  const indicatorMenu = (
    <Menu>
      <Menu.SubMenu key="trend" title="Trend Indicators">
        <Menu.Item key="sma" onClick={() => addIndicator('SMA')}>Simple Moving Average</Menu.Item>
        <Menu.Item key="ema" onClick={() => addIndicator('EMA')}>Exponential Moving Average</Menu.Item>
        <Menu.Item key="bb" onClick={() => addIndicator('Bollinger Bands')}>Bollinger Bands</Menu.Item>
      </Menu.SubMenu>
      <Menu.SubMenu key="momentum" title="Momentum Indicators">
        <Menu.Item key="rsi" onClick={() => addIndicator('RSI')}>RSI</Menu.Item>
        <Menu.Item key="macd" onClick={() => addIndicator('MACD')}>MACD</Menu.Item>
      </Menu.SubMenu>
    </Menu>
  );

  const drawingMenu = (
    <Menu>
      <Menu.Item key="trendline" onClick={() => setChartState(prev => ({ ...prev, drawingMode: 'trendline' }))}>
        Trend Line
      </Menu.Item>
      <Menu.Item key="horizontal" onClick={() => setChartState(prev => ({ ...prev, drawingMode: 'horizontal' }))}>
        Horizontal Line
      </Menu.Item>
      <Menu.Item key="rectangle" onClick={() => setChartState(prev => ({ ...prev, drawingMode: 'rectangle' }))}>
        Rectangle
      </Menu.Item>
    </Menu>
  );

  if (error) {
    return (
      <Card title="Advanced Trading Chart - Error" style={{ height }}>
        <Text type="danger">{error}</Text>
      </Card>
    );
  }

  return (
    <div ref={containerRef} style={{ position: 'relative' }}>
      <Card
        title={
          <Space>
            <LineChartOutlined />
            <Text strong>{symbol} - {timeframe}</Text>
            <Badge status={isConnected ? 'success' : 'error'} />
            <Tag color="blue">{chartType}</Tag>
          </Space>
        }
        extra={
          <Space>
            {/* Timeframe Selector */}
            <Select value={timeframe} size="small" style={{ width: 70 }}>
              <Select.Option value="1m">1m</Select.Option>
              <Select.Option value="5m">5m</Select.Option>
              <Select.Option value="15m">15m</Select.Option>
              <Select.Option value="1h">1h</Select.Option>
              <Select.Option value="4h">4h</Select.Option>
              <Select.Option value="1d">1d</Select.Option>
            </Select>

            {/* Chart Type */}
            <Select value={chartType} size="small" style={{ width: 100 }}>
              <Select.Option value="candlestick">Candles</Select.Option>
              <Select.Option value="line">Line</Select.Option>
              <Select.Option value="area">Area</Select.Option>
            </Select>

            {/* Indicators */}
            <Dropdown overlay={indicatorMenu} trigger={['click']}>
              <Button size="small" icon={<PlusOutlined />}>
                Indicators
              </Button>
            </Dropdown>

            {/* Drawing Tools */}
            {enableDrawing && (
              <Dropdown overlay={drawingMenu} trigger={['click']}>
                <Button size="small" icon={<EditOutlined />}>
                  Draw
                </Button>
              </Dropdown>
            )}

            {/* Settings */}
            <Button size="small" icon={<SettingOutlined />} />

            {/* Export */}
            {exportEnabled && (
              <Button size="small" icon={<ExportOutlined />} onClick={handleExport} />
            )}

            {/* Fullscreen */}
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
        {/* Active Indicators */}
        {!compactMode && activeIndicators.length > 0 && (
          <div style={{ marginBottom: 8 }}>
            <Text type="secondary" style={{ fontSize: '11px', marginRight: 8 }}>Indicators:</Text>
            {activeIndicators.map(indicator => (
              <Tag
                key={indicator.id}
                closable
                onClose={() => {
                  const updated = activeIndicators.filter(i => i.id !== indicator.id);
                  setActiveIndicators(updated);
                  onIndicatorChange?.(updated);
                }}
                style={{ marginBottom: 4 }}
              >
                {indicator.name}
              </Tag>
            ))}
          </div>
        )}

        {/* Chart */}
        <div style={{ 
          width: '100%', 
          height: compactMode ? height - 100 : height - 120,
          overflow: 'hidden'
        }}>
          <svg
            ref={svgRef}
            width="100%"
            height="100%"
            style={{ background: theme === 'dark' ? '#1f1f1f' : '#ffffff' }}
          />
        </div>

        {/* Status Bar */}
        {!compactMode && (
          <div style={{ 
            marginTop: 8, 
            padding: '4px 0',
            borderTop: '1px solid #f0f0f0',
            display: 'flex',
            justifyContent: 'space-between',
            alignItems: 'center'
          }}>
            <Space size="small">
              <Text type="secondary" style={{ fontSize: '11px' }}>
                Last: {candlestickData[candlestickData.length - 1]?.close?.toFixed(2) || 'N/A'}
              </Text>
              <Text type="secondary" style={{ fontSize: '11px' }}>
                Volume: {candlestickData[candlestickData.length - 1]?.volume?.toLocaleString() || 'N/A'}
              </Text>
              <Text type="secondary" style={{ fontSize: '11px' }}>
                Bars: {candlestickData.length}
              </Text>
            </Space>
            
            <Space size="small">
              <Tooltip title="Show Volume">
                <Switch
                  checked={showVolume}
                  size="small"
                  onChange={(checked) => {/* Handle volume toggle */}}
                />
              </Tooltip>
              <Tooltip title="Show Crosshair">
                <Switch
                  checked={enableCrosshair}
                  size="small"
                  onChange={(checked) => {/* Handle crosshair toggle */}}
                />
              </Tooltip>
              <Text type="secondary" style={{ fontSize: '11px' }}>
                Updated: {lastUpdate ? new Date(lastUpdate).toLocaleTimeString() : 'Never'}
              </Text>
            </Space>
          </div>
        )}
      </Card>
    </div>
  );
});

AdvancedTradingChart.displayName = 'AdvancedTradingChart';

export default AdvancedTradingChart;