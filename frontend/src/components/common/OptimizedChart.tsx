/**
 * Optimized Chart Component
 * High-performance chart wrapper with virtualization and data optimization
 */

import React, { memo, useMemo, useCallback, useState, useEffect, useRef } from 'react';
import {
  LineChart,
  Line,
  AreaChart,
  Area,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  ReferenceLine,
  Legend
} from 'recharts';
import { Card, Select, Space, Button, Typography } from 'antd';
import {
  FullscreenOutlined,
  CompressOutlined,
  DownloadOutlined,
  ReloadOutlined
} from '@ant-design/icons';
import { UI_CONSTANTS } from '../../constants/ui';
import LoadingState from './LoadingState';

const { Text } = Typography;
const { Option } = Select;

export interface ChartDataPoint {
  [key: string]: any;
  timestamp?: number;
  time?: string;
}

export interface ChartConfig {
  dataKey: string;
  stroke?: string;
  fill?: string;
  name?: string;
  yAxisId?: string;
  strokeWidth?: number;
  dot?: boolean;
  connectNulls?: boolean;
}

export interface OptimizedChartProps {
  /** Chart data */
  data: ChartDataPoint[];
  /** Chart type */
  type: 'line' | 'area' | 'bar';
  /** Chart configurations for different data series */
  configs: ChartConfig[];
  /** Chart title */
  title?: string;
  /** Chart height */
  height?: number;
  /** Show legend */
  showLegend?: boolean;
  /** Show grid */
  showGrid?: boolean;
  /** Show tooltip */
  showTooltip?: boolean;
  /** X-axis data key */
  xAxisKey?: string;
  /** Y-axis configurations */
  yAxes?: Array<{
    yAxisId?: string;
    orientation?: 'left' | 'right';
    domain?: [number | string, number | string];
    label?: string;
  }>;
  /** Reference lines */
  referenceLines?: Array<{
    value: number;
    stroke?: string;
    strokeDasharray?: string;
    label?: string;
    yAxisId?: string;
  }>;
  /** Loading state */
  loading?: boolean;
  /** Error message */
  error?: string;
  /** Enable data sampling for performance */
  enableSampling?: boolean;
  /** Maximum data points to render */
  maxDataPoints?: number;
  /** Enable fullscreen mode */
  enableFullscreen?: boolean;
  /** Enable data export */
  enableExport?: boolean;
  /** Custom export filename */
  exportFilename?: string;
  /** Refresh callback */
  onRefresh?: () => void;
  /** Chart container class */
  className?: string;
  /** Additional container styles */
  style?: React.CSSProperties;
  /** Animation duration */
  animationDuration?: number;
  /** Responsive breakpoint */
  responsiveBreakpoint?: number;
}

const OptimizedChart: React.FC<OptimizedChartProps> = memo(({
  data,
  type,
  configs,
  title,
  height = 300,
  showLegend = true,
  showGrid = true,
  showTooltip = true,
  xAxisKey = 'time',
  yAxes = [{ orientation: 'left' }],
  referenceLines = [],
  loading = false,
  error,
  enableSampling = true,
  maxDataPoints = UI_CONSTANTS.DATA_LIMITS.MAX_CHART_POINTS,
  enableFullscreen = true,
  enableExport = true,
  exportFilename = 'chart-data',
  onRefresh,
  className,
  style,
  animationDuration = UI_CONSTANTS.ANIMATIONS.NORMAL,
  responsiveBreakpoint = UI_CONSTANTS.BREAKPOINTS.MD
}) => {
  const [isFullscreen, setIsFullscreen] = useState(false);
  const [samplingRate, setSamplingRate] = useState(1);
  const chartRef = useRef<HTMLDivElement>(null);

  // Optimize data with sampling if needed
  const optimizedData = useMemo(() => {
    if (!data || data.length === 0) return [];
    
    if (!enableSampling || data.length <= maxDataPoints) {
      return data;
    }

    const ratio = Math.ceil(data.length / maxDataPoints);
    return data.filter((_, index) => index % ratio === 0);
  }, [data, enableSampling, maxDataPoints]);

  // Calculate dynamic sampling rate
  useEffect(() => {
    if (data && data.length > 0) {
      const newSamplingRate = Math.max(1, Math.ceil(data.length / maxDataPoints));
      setSamplingRate(newSamplingRate);
    }
  }, [data, maxDataPoints]);

  // Fullscreen handling
  const toggleFullscreen = useCallback(() => {
    setIsFullscreen(prev => !prev);
  }, []);

  // Export functionality
  const handleExport = useCallback(() => {
    const dataStr = JSON.stringify(optimizedData, null, 2);
    const dataBlob = new Blob([dataStr], { type: 'application/json' });
    const url = URL.createObjectURL(dataBlob);
    
    const link = document.createElement('a');
    link.href = url;
    link.download = `${exportFilename}-${Date.now()}.json`;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(url);
  }, [optimizedData, exportFilename]);

  // Custom tooltip formatter
  const customTooltipFormatter = useCallback((value: any, name: string, props: any) => {
    const config = configs.find(c => c.dataKey === props.dataKey);
    const displayName = config?.name || name;
    
    if (typeof value === 'number') {
      return [value.toFixed(2), displayName];
    }
    return [value, displayName];
  }, [configs]);

  // Render chart based on type
  const renderChart = useCallback(() => {
    const commonProps = {
      data: optimizedData,
      margin: { top: 5, right: 30, left: 20, bottom: 5 },
      animationDuration
    };

    const xAxis = (
      <XAxis 
        dataKey={xAxisKey}
        axisLine={false}
        tickLine={false}
        tick={{ fontSize: 12 }}
      />
    );

    const yAxisElements = yAxes.map((yAxisConfig, index) => (
      <YAxis
        key={index}
        yAxisId={yAxisConfig.yAxisId || 'default'}
        orientation={yAxisConfig.orientation}
        domain={yAxisConfig.domain}
        axisLine={false}
        tickLine={false}
        tick={{ fontSize: 12 }}
        label={yAxisConfig.label ? { 
          value: yAxisConfig.label, 
          angle: -90, 
          position: 'insideLeft' 
        } : undefined}
      />
    ));

    const grid = showGrid ? <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" /> : null;
    
    const tooltip = showTooltip ? (
      <Tooltip
        formatter={customTooltipFormatter}
        labelFormatter={(value) => `Time: ${value}`}
        contentStyle={{
          backgroundColor: '#fff',
          border: '1px solid #d9d9d9',
          borderRadius: '4px',
          fontSize: '12px'
        }}
      />
    ) : null;

    const legend = showLegend ? (
      <Legend 
        wrapperStyle={{ fontSize: '12px' }}
        iconType="line"
      />
    ) : null;

    const refLines = referenceLines.map((refLine, index) => (
      <ReferenceLine
        key={index}
        y={refLine.value}
        stroke={refLine.stroke || UI_CONSTANTS.CHART_COLORS.WARNING}
        strokeDasharray={refLine.strokeDasharray || "5 5"}
        label={refLine.label}
        yAxisId={refLine.yAxisId}
      />
    ));

    switch (type) {
      case 'area':
        return (
          <AreaChart {...commonProps}>
            {grid}
            {xAxis}
            {yAxisElements}
            {tooltip}
            {legend}
            {refLines}
            {configs.map((config, index) => (
              <Area
                key={index}
                type="monotone"
                dataKey={config.dataKey}
                stroke={config.stroke || UI_CONSTANTS.CHART_COLORS.PRIMARY}
                fill={config.fill || config.stroke || UI_CONSTANTS.CHART_COLORS.PRIMARY}
                fillOpacity={0.6}
                strokeWidth={config.strokeWidth || 2}
                yAxisId={config.yAxisId || 'default'}
                connectNulls={config.connectNulls !== false}
                animationDuration={animationDuration}
              />
            ))}
          </AreaChart>
        );

      case 'bar':
        return (
          <BarChart {...commonProps}>
            {grid}
            {xAxis}
            {yAxisElements}
            {tooltip}
            {legend}
            {refLines}
            {configs.map((config, index) => (
              <Bar
                key={index}
                dataKey={config.dataKey}
                fill={config.fill || UI_CONSTANTS.CHART_COLORS.PRIMARY}
                yAxisId={config.yAxisId || 'default'}
                animationDuration={animationDuration}
              />
            ))}
          </BarChart>
        );

      default: // line
        return (
          <LineChart {...commonProps}>
            {grid}
            {xAxis}
            {yAxisElements}
            {tooltip}
            {legend}
            {refLines}
            {configs.map((config, index) => (
              <Line
                key={index}
                type="monotone"
                dataKey={config.dataKey}
                stroke={config.stroke || UI_CONSTANTS.CHART_COLORS.PRIMARY}
                strokeWidth={config.strokeWidth || 2}
                dot={config.dot !== false ? { r: 2 } : false}
                yAxisId={config.yAxisId || 'default'}
                connectNulls={config.connectNulls !== false}
                animationDuration={animationDuration}
              />
            ))}
          </LineChart>
        );
    }
  }, [
    optimizedData, 
    type, 
    configs, 
    xAxisKey, 
    yAxes, 
    showGrid, 
    showTooltip, 
    showLegend, 
    referenceLines, 
    customTooltipFormatter,
    animationDuration
  ]);

  if (loading) {
    return (
      <div style={{ height, ...style }} className={className}>
        <LoadingState message="Loading chart data..." minHeight={height} />
      </div>
    );
  }

  if (error) {
    return (
      <Card style={{ height, ...style }} className={className}>
        <div style={{ 
          display: 'flex', 
          alignItems: 'center', 
          justifyContent: 'center', 
          height: '100%',
          flexDirection: 'column'
        }}>
          <Text type="danger">{error}</Text>
          {onRefresh && (
            <Button 
              icon={<ReloadOutlined />} 
              onClick={onRefresh}
              style={{ marginTop: 16 }}
            >
              Retry
            </Button>
          )}
        </div>
      </Card>
    );
  }

  const chartHeight = isFullscreen ? '80vh' : height;
  const containerStyle: React.CSSProperties = {
    height: chartHeight,
    position: isFullscreen ? 'fixed' : 'relative',
    top: isFullscreen ? '10vh' : 'auto',
    left: isFullscreen ? '5vw' : 'auto',
    width: isFullscreen ? '90vw' : '100%',
    zIndex: isFullscreen ? 1000 : 'auto',
    backgroundColor: isFullscreen ? '#fff' : 'transparent',
    ...style
  };

  return (
    <Card
      ref={chartRef}
      style={containerStyle}
      className={className}
      title={title}
      extra={
        <Space size="small">
          {enableSampling && data.length > maxDataPoints && (
            <>
              <Text type="secondary" style={{ fontSize: 12 }}>
                Showing {optimizedData.length} of {data.length} points
              </Text>
              <Select
                size="small"
                value={samplingRate}
                onChange={setSamplingRate}
                style={{ width: 80 }}
              >
                <Option value={1}>All</Option>
                <Option value={2}>1/2</Option>
                <Option value={5}>1/5</Option>
                <Option value={10}>1/10</Option>
              </Select>
            </>
          )}
          {onRefresh && (
            <Button 
              size="small"
              icon={<ReloadOutlined />}
              onClick={onRefresh}
            />
          )}
          {enableExport && (
            <Button
              size="small"
              icon={<DownloadOutlined />}
              onClick={handleExport}
            />
          )}
          {enableFullscreen && (
            <Button
              size="small"
              icon={isFullscreen ? <CompressOutlined /> : <FullscreenOutlined />}
              onClick={toggleFullscreen}
            />
          )}
        </Space>
      }
    >
      <ResponsiveContainer width="100%" height="100%">
        {renderChart()}
      </ResponsiveContainer>
    </Card>
  );
});

OptimizedChart.displayName = 'OptimizedChart';

export default OptimizedChart;