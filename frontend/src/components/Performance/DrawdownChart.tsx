/**
 * DrawdownChart Component - Sprint 3 Integration
 * Underwater equity curve and comprehensive drawdown analysis
 */

import React, { useState, useMemo } from 'react';
import { Card, Row, Col, Select, Switch, Space, Typography, Statistic, Tag, Button, Tooltip } from 'antd';
import { 
  FallOutlined,
  RiseOutlined,
  LineChartOutlined,
  BarChartOutlined,
  DownloadOutlined,
  InfoCircleOutlined,
  ClockCircleOutlined,
  FallOutlined
} from '@ant-design/icons';
import { Line, Column, Area } from '@ant-design/plots';
import dayjs from 'dayjs';

const { Title, Text } = Typography;
const { Option } = Select;

interface DrawdownChartProps {
  data: DrawdownDataPoint[];
  benchmarkData?: DrawdownDataPoint[];
  className?: string;
  height?: number;
  showBenchmark?: boolean;
  showRecoveryPeriods?: boolean;
  interactive?: boolean;
}

interface DrawdownDataPoint {
  date: string;
  portfolio_value: number;
  peak_value: number;
  drawdown: number; // As percentage (negative value)
  underwater_days?: number;
  recovery_date?: string;
  drawdown_duration?: number;
}

interface DrawdownPeriod {
  start_date: string;
  end_date: string;
  peak_date: string;
  trough_date: string;
  recovery_date?: string;
  peak_value: number;
  trough_value: number;
  max_drawdown: number;
  duration_days: number;
  recovery_days?: number;
  still_underwater: boolean;
}

type ChartType = 'underwater' | 'equity' | 'both' | 'histogram';
type TimeAggregation = 'daily' | 'weekly' | 'monthly';

const DrawdownChart: React.FC<DrawdownChartProps> = ({
  data,
  benchmarkData = [],
  className,
  height = 700,
  showBenchmark = true,
  showRecoveryPeriods = true,
  interactive = true,
}) => {
  // State
  const [chartType, setChartType] = useState<ChartType>('both');
  const [timeAggregation, setTimeAggregation] = useState<TimeAggregation>('daily');
  const [showMaxDrawdownLine, setShowMaxDrawdownLine] = useState(true);
  const [highlightPeriods, setHighlightPeriods] = useState(true);
  const [minDrawdownThreshold, setMinDrawdownThreshold] = useState(0.05); // 5%
  
  // Calculate drawdown statistics
  const drawdownStats = useMemo(() => {
    if (!data || data.length === 0) return null;
    
    const drawdowns = data.map(d => Math.abs(d.drawdown));
    const maxDrawdown = Math.max(...drawdowns);
    const avgDrawdown = drawdowns.reduce((sum, dd) => sum + dd, 0) / drawdowns.length;
    const currentDrawdown = Math.abs(data[data.length - 1]?.drawdown || 0);
    
    // Count underwater days
    const underwaterDays = data.filter(d => Math.abs(d.drawdown) > 0.001).length;
    const totalDays = data.length;
    const underwaterPercentage = (underwaterDays / totalDays) * 100;
    
    // Calculate recovery statistics
    let totalRecoveryTime = 0;
    let recoveryCount = 0;
    let maxRecoveryTime = 0;
    
    data.forEach(point => {
      if (point.recovery_date && point.drawdown_duration) {
        totalRecoveryTime += point.drawdown_duration;
        recoveryCount++;
        maxRecoveryTime = Math.max(maxRecoveryTime, point.drawdown_duration);
      }
    });
    
    const avgRecoveryTime = recoveryCount > 0 ? totalRecoveryTime / recoveryCount : 0;
    
    return {
      maxDrawdown,
      avgDrawdown,
      currentDrawdown,
      underwaterDays,
      underwaterPercentage,
      avgRecoveryTime,
      maxRecoveryTime,
      recoveryCount,
    };
  }, [data]);
  
  // Identify significant drawdown periods
  const drawdownPeriods = useMemo((): DrawdownPeriod[] => {
    if (!data || data.length === 0) return [];
    
    const periods: DrawdownPeriod[] = [];
    let currentPeriod: Partial<DrawdownPeriod> | null = null;
    let peakValue = data[0].portfolio_value;
    let peakDate = data[0].date;
    
    for (let i = 0; i < data.length; i++) {
      const point = data[i];
      const currentDrawdown = Math.abs(point.drawdown);
      
      // Track peak
      if (point.portfolio_value > peakValue) {
        peakValue = point.portfolio_value;
        peakDate = point.date;
      }
      
      // Start of drawdown period
      if (currentDrawdown >= minDrawdownThreshold && !currentPeriod) {
        currentPeriod = {
          start_date: point.date,
          peak_date: peakDate,
          peak_value: peakValue,
          max_drawdown: currentDrawdown,
          trough_value: point.portfolio_value,
          trough_date: point.date,
        };
      }
      
      // Update ongoing drawdown period
      if (currentPeriod && currentDrawdown >= minDrawdownThreshold) {
        if (currentDrawdown > currentPeriod.max_drawdown!) {
          currentPeriod.max_drawdown = currentDrawdown;
          currentPeriod.trough_value = point.portfolio_value;
          currentPeriod.trough_date = point.date;
        }
      }
      
      // End of drawdown period (recovery)
      if (currentPeriod && currentDrawdown < 0.001) {
        const startDate = dayjs(currentPeriod.start_date!);
        const endDate = dayjs(point.date);
        const recoveryDate = point.recovery_date ? dayjs(point.recovery_date) : endDate;
        
        periods.push({
          ...currentPeriod,
          end_date: point.date,
          recovery_date: point.recovery_date || point.date,
          duration_days: endDate.diff(startDate, 'day'),
          recovery_days: point.recovery_date ? recoveryDate.diff(endDate, 'day') : 0,
          still_underwater: false,
        } as DrawdownPeriod);
        
        currentPeriod = null;
        peakValue = point.portfolio_value;
        peakDate = point.date;
      }
    }
    
    // Handle ongoing drawdown period
    if (currentPeriod) {
      const startDate = dayjs(currentPeriod.start_date!);
      const endDate = dayjs(data[data.length - 1].date);
      
      periods.push({
        ...currentPeriod,
        end_date: data[data.length - 1].date,
        duration_days: endDate.diff(startDate, 'day'),
        still_underwater: true,
      } as DrawdownPeriod);
    }
    
    return periods.filter(p => p.max_drawdown >= minDrawdownThreshold);
  }, [data, minDrawdownThreshold]);
  
  // Process data for charting
  const processedData = useMemo(() => {
    if (!data || data.length === 0) return [];
    
    let aggregatedData = data;
    
    // Apply time aggregation if needed
    if (timeAggregation !== 'daily') {
      const grouped = new Map<string, DrawdownDataPoint[]>();
      
      data.forEach(point => {
        let key: string;
        const date = dayjs(point.date);
        
        if (timeAggregation === 'weekly') {
          key = `${date.year()}-W${date.week()}`;
        } else { // monthly
          key = date.format('YYYY-MM');
        }
        
        if (!grouped.has(key)) {
          grouped.set(key, []);
        }
        grouped.get(key)!.push(point);
      });
      
      aggregatedData = Array.from(grouped.entries()).map(([key, points]) => {
        const lastPoint = points[points.length - 1];
        const maxDrawdown = Math.max(...points.map(p => Math.abs(p.drawdown)));
        
        return {
          ...lastPoint,
          drawdown: -maxDrawdown, // Keep as negative
        };
      });
    }
    
    return aggregatedData.map(point => ({
      date: point.date,
      portfolio_value: point.portfolio_value,
      peak_value: point.peak_value,
      drawdown: point.drawdown * 100, // Convert to percentage
      drawdown_abs: Math.abs(point.drawdown) * 100,
      underwater_days: point.underwater_days || 0,
    }));
  }, [data, timeAggregation]);
  
  // Process benchmark data
  const processedBenchmarkData = useMemo(() => {
    if (!showBenchmark || !benchmarkData || benchmarkData.length === 0) return [];
    
    return benchmarkData.map(point => ({
      date: point.date,
      portfolio_value: point.portfolio_value,
      drawdown: point.drawdown * 100,
      type: 'benchmark',
    }));
  }, [benchmarkData, showBenchmark]);
  
  // Equity curve chart data
  const equityData = useMemo(() => {
    const portfolioData = processedData.map(d => ({
      date: d.date,
      value: d.portfolio_value,
      type: 'Portfolio',
    }));
    
    const benchmarkEquity = processedBenchmarkData.map(d => ({
      date: d.date,
      value: d.portfolio_value,
      type: 'Benchmark',
    }));
    
    return [...portfolioData, ...benchmarkEquity];
  }, [processedData, processedBenchmarkData]);
  
  // Underwater curve chart data
  const underwaterData = useMemo(() => {
    const portfolioData = processedData.map(d => ({
      date: d.date,
      drawdown: d.drawdown,
      type: 'Portfolio',
    }));
    
    const benchmarkUnderwater = processedBenchmarkData.map(d => ({
      date: d.date,
      drawdown: d.drawdown,
      type: 'Benchmark',
    }));
    
    return [...portfolioData, ...benchmarkUnderwater];
  }, [processedData, processedBenchmarkData]);
  
  // Drawdown histogram data
  const histogramData = useMemo(() => {
    if (!processedData || processedData.length === 0) return [];
    
    const drawdowns = processedData.map(d => Math.abs(d.drawdown)).filter(dd => dd > 0);
    
    // Create bins
    const maxDD = Math.max(...drawdowns);
    const binCount = 20;
    const binSize = maxDD / binCount;
    
    const bins = Array.from({ length: binCount }, (_, i) => ({
      range: `${(i * binSize).toFixed(1)}% - ${((i + 1) * binSize).toFixed(1)}%`,
      count: 0,
      minValue: i * binSize,
      maxValue: (i + 1) * binSize,
    }));
    
    drawdowns.forEach(dd => {
      const binIndex = Math.min(Math.floor(dd / binSize), binCount - 1);
      bins[binIndex].count++;
    });
    
    return bins.filter(bin => bin.count > 0);
  }, [processedData]);
  
  // Chart configurations
  const equityChartConfig = {
    data: equityData,
    xField: 'date',
    yField: 'value',
    seriesField: 'type',
    height: chartType === 'both' ? (height - 200) / 2 : height - 150,
    color: ['#1890ff', '#ff4d4f'],
    smooth: true,
    xAxis: {
      type: 'time' as const,
    },
    yAxis: {
      label: {
        formatter: (v: string) => `$${(Number(v) / 1000000).toFixed(1)}M`,
      },
    },
    tooltip: {
      formatter: (datum: any) => ({
        name: datum.type,
        value: `$${datum.value.toLocaleString()}`,
      }),
    },
    slider: {
      start: 0.8,
      end: 1.0,
    },
  };
  
  const underwaterChartConfig = {
    data: underwaterData,
    xField: 'date',
    yField: 'drawdown',
    seriesField: 'type',
    height: chartType === 'both' ? (height - 200) / 2 : height - 150,
    color: ['#ff4d4f', '#faad14'],
    smooth: false,
    area: {
      style: {
        fillOpacity: 0.6,
      },
    },
    xAxis: {
      type: 'time' as const,
    },
    yAxis: {
      label: {
        formatter: (v: string) => `${v}%`,
      },
      max: 0,
    },
    tooltip: {
      formatter: (datum: any) => ({
        name: `${datum.type} Drawdown`,
        value: `${Math.abs(datum.drawdown).toFixed(2)}%`,
      }),
    },
    annotations: showMaxDrawdownLine && drawdownStats ? [{
      type: 'line',
      start: ['min', -(drawdownStats.maxDrawdown * 100)],
      end: ['max', -(drawdownStats.maxDrawdown * 100)],
      style: {
        stroke: '#ff4d4f',
        lineDash: [4, 4],
        lineWidth: 2,
      },
      text: {
        content: `Max DD: ${(drawdownStats.maxDrawdown * 100).toFixed(2)}%`,
        position: 'end',
        style: {
          fill: '#ff4d4f',
          fontSize: 12,
        },
      },
    }] : [],
  };
  
  const histogramConfig = {
    data: histogramData,
    xField: 'range',
    yField: 'count',
    height: height - 150,
    color: '#ff4d4f',
    label: {
      position: 'top' as const,
    },
    xAxis: {
      label: {
        autoRotate: true,
      },
    },
    tooltip: {
      formatter: (datum: any) => ({
        name: 'Frequency',
        value: datum.count,
      }),
    },
  };
  
  const handleExport = () => {
    const exportData = {
      metadata: {
        chartType,
        timeAggregation,
        minDrawdownThreshold,
        exportDate: new Date().toISOString(),
      },
      statistics: drawdownStats,
      drawdown_periods: drawdownPeriods,
      data: processedData,
      benchmark_data: processedBenchmarkData,
    };
    
    const blob = new Blob([JSON.stringify(exportData, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `drawdown_analysis_${chartType}_${dayjs().format('YYYYMMDD')}.json`;
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
          <FallOutlined style={{ fontSize: 48, color: '#d9d9d9', marginBottom: 16 }} />
          <Text type="secondary">No data available for drawdown analysis</Text>
        </div>
      </Card>
    );
  }
  
  return (
    <Card className={className} style={{ height }}>
      {/* Header with controls */}
      <Row gutter={[16, 8]} align="middle" style={{ marginBottom: 16 }}>
        <Col>
          <Space>
            <FallOutlined style={{ color: '#ff4d4f', fontSize: '18px' }} />
            <Title level={5} style={{ margin: 0 }}>
              Drawdown Analysis
            </Title>
          </Space>
        </Col>
        
        <Col flex="auto">
          <Row gutter={[8, 8]} justify="end" align="middle">
            <Col>
              <Space size="small">
                <Text type="secondary" style={{ fontSize: '12px' }}>View:</Text>
                <Select
                  value={chartType}
                  onChange={setChartType}
                  size="small"
                  style={{ width: 100 }}
                >
                  <Option value="underwater">Underwater</Option>
                  <Option value="equity">Equity</Option>
                  <Option value="both">Both</Option>
                  <Option value="histogram">Histogram</Option>
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
                </Select>
              </Space>
            </Col>
            
            <Col>
              <Space size="small">
                <Text type="secondary" style={{ fontSize: '12px' }}>Max DD Line:</Text>
                <Switch 
                  size="small"
                  checked={showMaxDrawdownLine}
                  onChange={setShowMaxDrawdownLine}
                />
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
      
      {/* Statistics Summary */}
      {drawdownStats && (
        <Row gutter={[16, 8]} style={{ marginBottom: 16 }}>
          <Col xs={12} sm={6} md={4}>
            <Card size="small" style={{ textAlign: 'center' }}>
              <Statistic
                title="Max Drawdown"
                value={drawdownStats.maxDrawdown * 100}
                precision={2}
                suffix="%"
                valueStyle={{ color: '#ff4d4f', fontSize: '16px' }}
                prefix={<FallOutlined />}
              />
            </Card>
          </Col>
          <Col xs={12} sm={6} md={4}>
            <Card size="small" style={{ textAlign: 'center' }}>
              <Statistic
                title="Current Drawdown"
                value={drawdownStats.currentDrawdown * 100}
                precision={2}
                suffix="%"
                valueStyle={{ 
                  color: drawdownStats.currentDrawdown > 0.05 ? '#ff4d4f' : '#52c41a', 
                  fontSize: '16px' 
                }}
                prefix={drawdownStats.currentDrawdown > 0.001 ? <FallOutlined /> : <RiseOutlined />}
              />
            </Card>
          </Col>
          <Col xs={12} sm={6} md={4}>
            <Card size="small" style={{ textAlign: 'center' }}>
              <Statistic
                title="Avg Drawdown"
                value={drawdownStats.avgDrawdown * 100}
                precision={2}
                suffix="%"
                valueStyle={{ fontSize: '16px' }}
              />
            </Card>
          </Col>
          <Col xs={12} sm={6} md={4}>
            <Card size="small" style={{ textAlign: 'center' }}>
              <Statistic
                title="Underwater %"
                value={drawdownStats.underwaterPercentage}
                precision={1}
                suffix="%"
                valueStyle={{ fontSize: '16px' }}
                prefix={<ClockCircleOutlined />}
              />
            </Card>
          </Col>
          <Col xs={12} sm={6} md={4}>
            <Card size="small" style={{ textAlign: 'center' }}>
              <Statistic
                title="Avg Recovery"
                value={drawdownStats.avgRecoveryTime}
                precision={0}
                suffix=" days"
                valueStyle={{ fontSize: '16px' }}
              />
            </Card>
          </Col>
          <Col xs={12} sm={6} md={4}>
            <Card size="small" style={{ textAlign: 'center' }}>
              <Statistic
                title="DD Periods"
                value={drawdownPeriods.length}
                valueStyle={{ fontSize: '16px' }}
                prefix={<BarChartOutlined />}
              />
            </Card>
          </Col>
        </Row>
      )}
      
      {/* Significant Drawdown Periods */}
      {drawdownPeriods.length > 0 && showRecoveryPeriods && (
        <Row style={{ marginBottom: 16 }}>
          <Col span={24}>
            <Text strong>Significant Drawdown Periods:</Text>
            <div style={{ marginTop: 8 }}>
              <Space wrap>
                {drawdownPeriods.map((period, index) => (
                  <Tooltip
                    key={index}
                    title={
                      <div>
                        <div>Peak: {dayjs(period.peak_date).format('MMM DD, YYYY')}</div>
                        <div>Trough: {dayjs(period.trough_date).format('MMM DD, YYYY')}</div>
                        <div>Duration: {period.duration_days} days</div>
                        {period.recovery_date && (
                          <div>Recovery: {dayjs(period.recovery_date).format('MMM DD, YYYY')}</div>
                        )}
                        <div>Peak Value: ${period.peak_value.toLocaleString()}</div>
                        <div>Trough Value: ${period.trough_value.toLocaleString()}</div>
                      </div>
                    }
                  >
                    <Tag 
                      color={period.still_underwater ? 'red' : 'orange'}
                      style={{ cursor: 'pointer' }}
                    >
                      {(period.max_drawdown * 100).toFixed(1)}% 
                      ({period.duration_days}d)
                      {period.still_underwater && ' *'}
                    </Tag>
                  </Tooltip>
                ))}
              </Space>
              <div style={{ marginTop: 4 }}>
                <Text type="secondary" style={{ fontSize: '11px' }}>
                  * Still underwater periods
                </Text>
              </div>
            </div>
          </Col>
        </Row>
      )}
      
      {/* Charts */}
      <div style={{ height: height - (drawdownStats ? 300 : 150) }}>
        {chartType === 'equity' && <Line {...equityChartConfig} />}
        {chartType === 'underwater' && <Area {...underwaterChartConfig} />}
        {chartType === 'histogram' && <Column {...histogramConfig} />}
        {chartType === 'both' && (
          <div>
            <Line {...equityChartConfig} />
            <div style={{ marginTop: 16 }}>
              <Area {...underwaterChartConfig} />
            </div>
          </div>
        )}
      </div>
    </Card>
  );
};

export default DrawdownChart;