/**
 * PerformanceHeatmap Component - Sprint 3 Integration
 * Portfolio performance heatmap visualization with advanced filtering
 */

import React, { useState, useMemo } from 'react';
import { Card, Row, Col, Select, Switch, Space, Typography, Tooltip, Button, Slider } from 'antd';
import { 
  HeatMapOutlined, 
  CalendarOutlined, 
  FilterOutlined,
  SettingOutlined,
  DownloadOutlined,
  InfoCircleOutlined
} from '@ant-design/icons';
import { Heatmap } from '@ant-design/plots';
import dayjs from 'dayjs';

const { Title, Text } = Typography;
const { Option } = Select;

interface PerformanceHeatmapProps {
  data: HeatmapData[];
  className?: string;
  height?: number;
  showControls?: boolean;
  interactive?: boolean;
}

interface HeatmapData {
  date: string;
  asset: string;
  return: number;
  volatility?: number;
  sharpe?: number;
  weight?: number;
  sector?: string;
  market_cap?: number;
}

type HeatmapMetric = 'return' | 'volatility' | 'sharpe' | 'weight';
type TimeAggregation = 'daily' | 'weekly' | 'monthly' | 'quarterly';

const PerformanceHeatmap: React.FC<PerformanceHeatmapProps> = ({
  data,
  className,
  height = 600,
  showControls = true,
  interactive = true,
}) => {
  // State
  const [selectedMetric, setSelectedMetric] = useState<HeatmapMetric>('return');
  const [timeAggregation, setTimeAggregation] = useState<TimeAggregation>('monthly');
  const [showWeekends, setShowWeekends] = useState(false);
  const [colorScale, setColorScale] = useState<'diverging' | 'sequential'>('diverging');
  const [sectorFilter, setSectorFilter] = useState<string[]>([]);
  const [valueRange, setValueRange] = useState<[number, number]>([0, 100]);
  
  // Get unique sectors for filtering
  const uniqueSectors = useMemo(() => {
    return Array.from(new Set(data.map(d => d.sector).filter(Boolean))) as string[];
  }, [data]);
  
  // Process and aggregate data based on selected parameters
  const processedData = useMemo(() => {
    let filteredData = data;
    
    // Apply sector filter
    if (sectorFilter.length > 0) {
      filteredData = filteredData.filter(d => sectorFilter.includes(d.sector || ''));
    }
    
    // Filter weekends if needed
    if (!showWeekends) {
      filteredData = filteredData.filter(d => {
        const dayOfWeek = dayjs(d.date).day();
        return dayOfWeek !== 0 && dayOfWeek !== 6; // 0 = Sunday, 6 = Saturday
      });
    }
    
    // Time aggregation
    const aggregatedData = new Map<string, Map<string, { values: number[], count: number }>>();
    
    filteredData.forEach(item => {
      let timeKey: string;
      const date = dayjs(item.date);
      
      switch (timeAggregation) {
        case 'daily':
          timeKey = date.format('YYYY-MM-DD');
          break;
        case 'weekly':
          timeKey = `${date.year()}-W${date.week()}`;
          break;
        case 'monthly':
          timeKey = date.format('YYYY-MM');
          break;
        case 'quarterly':
          timeKey = `${date.year()}-Q${date.quarter()}`;
          break;
        default:
          timeKey = date.format('YYYY-MM-DD');
      }
      
      if (!aggregatedData.has(timeKey)) {
        aggregatedData.set(timeKey, new Map());
      }
      
      const timeGroup = aggregatedData.get(timeKey)!;
      if (!timeGroup.has(item.asset)) {
        timeGroup.set(item.asset, { values: [], count: 0 });
      }
      
      const assetData = timeGroup.get(item.asset)!;
      const value = item[selectedMetric] || 0;
      assetData.values.push(value);
      assetData.count++;
    });
    
    // Calculate aggregated values
    const result: { x: string; y: string; value: number }[] = [];
    
    aggregatedData.forEach((timeGroup, timeKey) => {
      timeGroup.forEach((assetData, asset) => {
        let aggregatedValue: number;
        
        // Different aggregation methods for different metrics
        if (selectedMetric === 'return') {
          // For returns, use cumulative return
          aggregatedValue = assetData.values.reduce((cum, ret) => (1 + cum) * (1 + ret) - 1, 0);
        } else if (selectedMetric === 'volatility') {
          // For volatility, use standard deviation
          const mean = assetData.values.reduce((sum, val) => sum + val, 0) / assetData.values.length;
          const variance = assetData.values.reduce((sum, val) => sum + Math.pow(val - mean, 2), 0) / assetData.values.length;
          aggregatedValue = Math.sqrt(variance);
        } else {
          // For other metrics, use average
          aggregatedValue = assetData.values.reduce((sum, val) => sum + val, 0) / assetData.values.length;
        }
        
        result.push({
          x: timeKey,
          y: asset,
          value: aggregatedValue,
        });
      });
    });
    
    return result;
  }, [data, selectedMetric, timeAggregation, showWeekends, sectorFilter]);
  
  // Calculate value range for color scaling
  const dataRange = useMemo(() => {
    if (processedData.length === 0) return { min: 0, max: 1 };
    
    const values = processedData.map(d => d.value);
    const min = Math.min(...values);
    const max = Math.max(...values);
    
    return { min, max };
  }, [processedData]);
  
  // Color scheme configuration
  const getColorScheme = () => {
    if (colorScale === 'diverging') {
      if (selectedMetric === 'return') {
        return ['#ff4d4f', '#ffffff', '#52c41a']; // Red-White-Green for returns
      }
      return ['#1890ff', '#ffffff', '#ff4d4f']; // Blue-White-Red for other metrics
    } else {
      if (selectedMetric === 'return') {
        return ['#fff7f0', '#ff4d4f']; // Light to dark red for positive values
      } else if (selectedMetric === 'volatility') {
        return ['#f6ffed', '#ff4d4f']; // Light green to red for volatility
      }
      return ['#f0f9ff', '#1890ff']; // Light to dark blue
    }
  };
  
  // Heatmap configuration
  const heatmapConfig = {
    data: processedData,
    xField: 'x',
    yField: 'y',
    colorField: 'value',
    height: height - 100,
    color: getColorScheme(),
    meta: {
      value: {
        min: dataRange.min,
        max: dataRange.max,
      },
    },
    tooltip: {
      formatter: (data: any) => {
        let formattedValue: string;
        
        if (selectedMetric === 'return') {
          formattedValue = `${(data.value * 100).toFixed(2)}%`;
        } else if (selectedMetric === 'volatility') {
          formattedValue = `${(data.value * 100).toFixed(2)}%`;
        } else if (selectedMetric === 'sharpe') {
          formattedValue = data.value.toFixed(3);
        } else if (selectedMetric === 'weight') {
          formattedValue = `${(data.value * 100).toFixed(2)}%`;
        } else {
          formattedValue = data.value.toFixed(4);
        }
        
        return {
          name: `${data.y} - ${data.x}`,
          value: formattedValue,
        };
      },
    },
    xAxis: {
      label: {
        autoRotate: true,
        style: {
          fontSize: 10,
        },
      },
      title: {
        text: getTimeAxisLabel(),
      },
    },
    yAxis: {
      title: {
        text: 'Assets',
      },
      label: {
        style: {
          fontSize: 10,
        },
      },
    },
    legend: {
      position: 'bottom' as const,
      title: {
        text: getMetricLabel(),
      },
    },
    interactions: interactive ? [
      { type: 'element-active' },
      { type: 'brush' },
    ] : [],
  };
  
  function getMetricLabel(): string {
    switch (selectedMetric) {
      case 'return': return 'Return (%)';
      case 'volatility': return 'Volatility (%)';
      case 'sharpe': return 'Sharpe Ratio';
      case 'weight': return 'Weight (%)';
      default: return 'Value';
    }
  }
  
  function getTimeAxisLabel(): string {
    switch (timeAggregation) {
      case 'daily': return 'Date';
      case 'weekly': return 'Week';
      case 'monthly': return 'Month';
      case 'quarterly': return 'Quarter';
      default: return 'Time';
    }
  }
  
  const handleExport = () => {
    // Create export data
    const exportData = {
      metadata: {
        metric: selectedMetric,
        timeAggregation,
        showWeekends,
        colorScale,
        sectorFilter,
        exportDate: dayjs().format(),
      },
      data: processedData,
      stats: {
        totalDataPoints: processedData.length,
        uniqueAssets: new Set(processedData.map(d => d.y)).size,
        uniqueTimePoints: new Set(processedData.map(d => d.x)).size,
        valueRange: dataRange,
      },
    };
    
    const blob = new Blob([JSON.stringify(exportData, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `performance_heatmap_${selectedMetric}_${timeAggregation}_${dayjs().format('YYYYMMDD')}.json`;
    a.click();
    URL.revokeObjectURL(url);
  };
  
  if (!data || data.length === 0) {
    return (
      <Card className={className} style={{ height }}>
        <div style={{ 
          height: height - 50, 
          display: 'flex', 
          alignItems: 'center', 
          justifyContent: 'center',
          flexDirection: 'column'
        }}>
          <HeatMapOutlined style={{ fontSize: 48, color: '#d9d9d9', marginBottom: 16 }} />
          <Text type="secondary">No data available for heatmap visualization</Text>
        </div>
      </Card>
    );
  }
  
  return (
    <Card className={className} style={{ height }}>
      {/* Header with controls */}
      {showControls && (
        <Row gutter={[16, 16]} align="middle" style={{ marginBottom: 16 }}>
          <Col>
            <Space>
              <HeatMapOutlined style={{ color: '#1890ff', fontSize: '18px' }} />
              <Title level={5} style={{ margin: 0 }}>
                Performance Heatmap
              </Title>
            </Space>
          </Col>
          
          <Col flex="auto">
            <Row gutter={[8, 8]} justify="end">
              <Col>
                <Space size="small">
                  <Text type="secondary" style={{ fontSize: '12px' }}>Metric:</Text>
                  <Select
                    value={selectedMetric}
                    onChange={setSelectedMetric}
                    size="small"
                    style={{ width: 90 }}
                  >
                    <Option value="return">Return</Option>
                    <Option value="volatility">Volatility</Option>
                    <Option value="sharpe">Sharpe</Option>
                    <Option value="weight">Weight</Option>
                  </Select>
                </Space>
              </Col>
              
              <Col>
                <Space size="small">
                  <Text type="secondary" style={{ fontSize: '12px' }}>Period:</Text>
                  <Select
                    value={timeAggregation}
                    onChange={setTimeAggregation}
                    size="small"
                    style={{ width: 80 }}
                  >
                    <Option value="daily">Daily</Option>
                    <Option value="weekly">Weekly</Option>
                    <Option value="monthly">Monthly</Option>
                    <Option value="quarterly">Quarterly</Option>
                  </Select>
                </Space>
              </Col>
              
              {uniqueSectors.length > 0 && (
                <Col>
                  <Space size="small">
                    <Text type="secondary" style={{ fontSize: '12px' }}>Sector:</Text>
                    <Select
                      mode="multiple"
                      value={sectorFilter}
                      onChange={setSectorFilter}
                      size="small"
                      style={{ width: 120 }}
                      placeholder="All"
                      allowClear
                    >
                      {uniqueSectors.map(sector => (
                        <Option key={sector} value={sector}>{sector}</Option>
                      ))}
                    </Select>
                  </Space>
                </Col>
              )}
              
              <Col>
                <Space size="small">
                  <Text type="secondary" style={{ fontSize: '12px' }}>Weekends:</Text>
                  <Switch 
                    size="small"
                    checked={showWeekends}
                    onChange={setShowWeekends}
                  />
                </Space>
              </Col>
              
              <Col>
                <Space size="small">
                  <Text type="secondary" style={{ fontSize: '12px' }}>Colors:</Text>
                  <Select
                    value={colorScale}
                    onChange={setColorScale}
                    size="small"
                    style={{ width: 90 }}
                  >
                    <Option value="diverging">Diverging</Option>
                    <Option value="sequential">Sequential</Option>
                  </Select>
                </Space>
              </Col>
              
              <Col>
                <Button 
                  icon={<DownloadOutlined />} 
                  size="small"
                  onClick={handleExport}
                  type="text"
                >
                  Export
                </Button>
              </Col>
            </Row>
          </Col>
        </Row>
      )}
      
      {/* Statistics Summary */}
      <Row gutter={[16, 8]} style={{ marginBottom: 16 }}>
        <Col>
          <Text type="secondary" style={{ fontSize: '12px' }}>
            Assets: {new Set(processedData.map(d => d.y)).size} | 
            Periods: {new Set(processedData.map(d => d.x)).size} | 
            Data Points: {processedData.length}
          </Text>
        </Col>
        <Col>
          <Text type="secondary" style={{ fontSize: '12px' }}>
            Range: {dataRange.min.toFixed(4)} to {dataRange.max.toFixed(4)}
            {selectedMetric === 'return' && ` (${(dataRange.min * 100).toFixed(2)}% to ${(dataRange.max * 100).toFixed(2)}%)`}
          </Text>
        </Col>
        <Col>
          <Tooltip title="Click and drag to select a region. Double-click to reset zoom.">
            <InfoCircleOutlined style={{ color: '#8c8c8c' }} />
          </Tooltip>
        </Col>
      </Row>
      
      {/* Heatmap */}
      <div style={{ height: height - 150 }}>
        {processedData.length > 0 ? (
          <Heatmap {...heatmapConfig} />
        ) : (
          <div style={{ 
            height: '100%', 
            display: 'flex', 
            alignItems: 'center', 
            justifyContent: 'center',
            flexDirection: 'column'
          }}>
            <FilterOutlined style={{ fontSize: 32, color: '#d9d9d9', marginBottom: 8 }} />
            <Text type="secondary">No data matches current filters</Text>
          </div>
        )}
      </div>
    </Card>
  );
};

export default PerformanceHeatmap;