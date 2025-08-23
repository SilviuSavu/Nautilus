/**
 * RiskReturnScatterPlot Component - Sprint 3 Integration
 * Risk-return analysis visualization with efficient frontier and style analysis
 */

import React, { useState, useMemo, useRef, useEffect } from 'react';
import { Card, Row, Col, Select, Switch, Space, Typography, Tooltip, Button, Slider, Tag } from 'antd';
import { 
  DotChartOutlined, 
  LineChartOutlined,
  SettingOutlined,
  DownloadOutlined,
  InfoCircleOutlined,
  ZoomInOutlined,
  ZoomOutOutlined,
  AimOutlined
} from '@ant-design/icons';
import { Scatter } from '@ant-design/plots';

const { Title, Text } = Typography;
const { Option } = Select;

interface RiskReturnScatterPlotProps {
  data: ScatterDataPoint[];
  benchmarkData?: ScatterDataPoint[];
  efficientFrontier?: EfficientFrontierPoint[];
  className?: string;
  height?: number;
  interactive?: boolean;
  showBenchmark?: boolean;
  showEfficientFrontier?: boolean;
}

interface ScatterDataPoint {
  id: string;
  name: string;
  risk: number; // Volatility or other risk measure
  return: number; // Expected or realized return
  sharpe?: number;
  category?: string;
  size?: number; // For bubble chart (e.g., AUM, market cap)
  color?: string;
  alpha?: number;
  beta?: number;
  correlation?: number;
  max_drawdown?: number;
  calmar_ratio?: number;
  sortino_ratio?: number;
}

interface EfficientFrontierPoint {
  risk: number;
  return: number;
  sharpe: number;
  weights?: Record<string, number>;
}

type RiskMeasure = 'volatility' | 'var' | 'max_drawdown' | 'tracking_error';
type ReturnMeasure = 'total_return' | 'excess_return' | 'alpha' | 'annualized_return';
type SizeMetric = 'aum' | 'market_cap' | 'sharpe' | 'volume';

const RiskReturnScatterPlot: React.FC<RiskReturnScatterPlotProps> = ({
  data,
  benchmarkData = [],
  efficientFrontier = [],
  className,
  height = 600,
  interactive = true,
  showBenchmark = true,
  showEfficientFrontier = true,
}) => {
  // State
  const [riskMeasure, setRiskMeasure] = useState<RiskMeasure>('volatility');
  const [returnMeasure, setReturnMeasure] = useState<ReturnMeasure>('total_return');
  const [sizeMetric, setSizeMetric] = useState<SizeMetric>('sharpe');
  const [showBubbles, setShowBubbles] = useState(true);
  const [colorByCategory, setColorByCategory] = useState(true);
  const [showQuadrants, setShowQuadrants] = useState(true);
  const [selectedCategories, setSelectedCategories] = useState<string[]>([]);
  const [riskRange, setRiskRange] = useState<[number, number]>([0, 100]);
  const [returnRange, setReturnRange] = useState<[number, number]>([0, 100]);
  const [highlightedPoint, setHighlightedPoint] = useState<string | null>(null);
  
  const plotRef = useRef<any>(null);
  
  // Get unique categories for filtering
  const uniqueCategories = useMemo(() => {
    return Array.from(new Set(data.map(d => d.category).filter(Boolean))) as string[];
  }, [data]);
  
  // Process data based on selected measures and filters
  const processedData = useMemo(() => {
    let filteredData = data;
    
    // Apply category filter
    if (selectedCategories.length > 0) {
      filteredData = filteredData.filter(d => selectedCategories.includes(d.category || ''));
    }
    
    // Map data to plot format
    const plotData = filteredData.map(point => {
      let riskValue: number;
      let returnValue: number;
      let sizeValue: number;
      
      // Get risk value based on selected measure
      switch (riskMeasure) {
        case 'volatility':
          riskValue = point.risk * 100; // Convert to percentage
          break;
        case 'var':
          // Assume VaR is stored as risk measure or calculate proxy
          riskValue = point.risk * 100;
          break;
        case 'max_drawdown':
          riskValue = Math.abs(point.max_drawdown || point.risk) * 100;
          break;
        case 'tracking_error':
          riskValue = point.risk * 100;
          break;
        default:
          riskValue = point.risk * 100;
      }
      
      // Get return value based on selected measure
      switch (returnMeasure) {
        case 'total_return':
          returnValue = point.return * 100;
          break;
        case 'excess_return':
          returnValue = point.return * 100;
          break;
        case 'alpha':
          returnValue = (point.alpha || point.return) * 100;
          break;
        case 'annualized_return':
          returnValue = point.return * 100;
          break;
        default:
          returnValue = point.return * 100;
      }
      
      // Get size value for bubbles
      switch (sizeMetric) {
        case 'aum':
          sizeValue = point.size || 50;
          break;
        case 'market_cap':
          sizeValue = point.size || 50;
          break;
        case 'sharpe':
          sizeValue = Math.max(10, Math.min(100, (point.sharpe || 0) * 20 + 50));
          break;
        case 'volume':
          sizeValue = point.size || 50;
          break;
        default:
          sizeValue = 50;
      }
      
      return {
        x: riskValue,
        y: returnValue,
        size: showBubbles ? sizeValue : 8,
        name: point.name,
        id: point.id,
        category: point.category || 'Other',
        sharpe: point.sharpe || 0,
        alpha: point.alpha || 0,
        beta: point.beta || 1,
        correlation: point.correlation || 0,
        max_drawdown: point.max_drawdown || 0,
        calmar_ratio: point.calmar_ratio || 0,
        sortino_ratio: point.sortino_ratio || 0,
        color: colorByCategory ? getCategoryColor(point.category || 'Other') : '#1890ff',
      };
    });
    
    return plotData;
  }, [data, riskMeasure, returnMeasure, sizeMetric, showBubbles, colorByCategory, selectedCategories]);
  
  // Process benchmark data
  const processedBenchmarkData = useMemo(() => {
    if (!showBenchmark || benchmarkData.length === 0) return [];
    
    return benchmarkData.map(point => ({
      x: point.risk * 100,
      y: point.return * 100,
      size: 12,
      name: point.name,
      category: 'Benchmark',
      color: '#ff4d4f',
      shape: 'diamond',
    }));
  }, [benchmarkData, showBenchmark, riskMeasure, returnMeasure]);
  
  // Process efficient frontier
  const efficientFrontierData = useMemo(() => {
    if (!showEfficientFrontier || efficientFrontier.length === 0) return [];
    
    return efficientFrontier.map((point, index) => ({
      x: point.risk * 100,
      y: point.return * 100,
      size: 6,
      name: `EF Point ${index + 1}`,
      category: 'Efficient Frontier',
      color: '#52c41a',
      sharpe: point.sharpe,
      weights: point.weights,
    }));
  }, [efficientFrontier, showEfficientFrontier]);
  
  // Combine all data
  const combinedData = useMemo(() => {
    return [
      ...processedData,
      ...processedBenchmarkData,
      ...efficientFrontierData,
    ];
  }, [processedData, processedBenchmarkData, efficientFrontierData]);
  
  // Calculate quadrant lines (median risk and return)
  const quadrantLines = useMemo(() => {
    if (!showQuadrants || combinedData.length === 0) return null;
    
    const risks = combinedData.map(d => d.x);
    const returns = combinedData.map(d => d.y);
    
    const medianRisk = risks.sort((a, b) => a - b)[Math.floor(risks.length / 2)];
    const medianReturn = returns.sort((a, b) => a - b)[Math.floor(returns.length / 2)];
    
    return { medianRisk, medianReturn };
  }, [combinedData, showQuadrants]);
  
  // Chart configuration
  const scatterConfig = {
    data: combinedData,
    xField: 'x',
    yField: 'y',
    colorField: 'category',
    sizeField: showBubbles ? 'size' : undefined,
    size: showBubbles ? [8, 80] : 8,
    height: height - 120,
    color: colorByCategory ? getCategoryColors() : '#1890ff',
    shape: 'circle',
    tooltip: {
      formatter: (data: any) => {
        const tooltipData = [
          { name: 'Name', value: data.name },
          { name: getRiskLabel(), value: `${data.x.toFixed(2)}%` },
          { name: getReturnLabel(), value: `${data.y.toFixed(2)}%` },
        ];
        
        if (data.sharpe) {
          tooltipData.push({ name: 'Sharpe Ratio', value: data.sharpe.toFixed(3) });
        }
        if (data.alpha) {
          tooltipData.push({ name: 'Alpha', value: `${(data.alpha * 100).toFixed(2)}%` });
        }
        if (data.beta) {
          tooltipData.push({ name: 'Beta', value: data.beta.toFixed(3) });
        }
        if (data.max_drawdown) {
          tooltipData.push({ name: 'Max Drawdown', value: `${(Math.abs(data.max_drawdown) * 100).toFixed(2)}%` });
        }
        
        return tooltipData;
      },
    },
    xAxis: {
      title: {
        text: getRiskLabel(),
      },
      label: {
        formatter: (v: string) => `${v}%`,
      },
    },
    yAxis: {
      title: {
        text: getReturnLabel(),
      },
      label: {
        formatter: (v: string) => `${v}%`,
      },
    },
    legend: {
      position: 'bottom' as const,
    },
    interactions: interactive ? [
      { type: 'element-active' },
      { type: 'brush' },
      { type: 'tooltip' },
    ] : [],
    annotations: showQuadrants && quadrantLines ? [
      {
        type: 'line',
        start: [quadrantLines.medianRisk, 'min'],
        end: [quadrantLines.medianRisk, 'max'],
        style: {
          stroke: '#d9d9d9',
          lineDash: [4, 4],
        },
      },
      {
        type: 'line',
        start: ['min', quadrantLines.medianReturn],
        end: ['max', quadrantLines.medianReturn],
        style: {
          stroke: '#d9d9d9',
          lineDash: [4, 4],
        },
      },
      // Quadrant labels
      {
        type: 'text',
        position: ['95%', '95%'],
        content: 'High Risk\nHigh Return',
        style: {
          fontSize: 12,
          fill: '#8c8c8c',
          textAlign: 'end',
        },
      },
      {
        type: 'text',
        position: ['5%', '95%'],
        content: 'Low Risk\nHigh Return',
        style: {
          fontSize: 12,
          fill: '#8c8c8c',
          textAlign: 'start',
        },
      },
      {
        type: 'text',
        position: ['5%', '5%'],
        content: 'Low Risk\nLow Return',
        style: {
          fontSize: 12,
          fill: '#8c8c8c',
          textAlign: 'start',
        },
      },
      {
        type: 'text',
        position: ['95%', '5%'],
        content: 'High Risk\nLow Return',
        style: {
          fontSize: 12,
          fill: '#8c8c8c',
          textAlign: 'end',
        },
      },
    ] : [],
  };
  
  function getCategoryColor(category: string): string {
    const colorMap: Record<string, string> = {
      'Equity': '#1890ff',
      'Fixed Income': '#52c41a',
      'Alternative': '#faad14',
      'Commodity': '#722ed1',
      'Currency': '#13c2c2',
      'Strategy': '#1890ff',
      'Benchmark': '#ff4d4f',
      'Efficient Frontier': '#52c41a',
      'Other': '#8c8c8c',
    };
    
    return colorMap[category] || '#8c8c8c';
  }
  
  function getCategoryColors(): string[] {
    return uniqueCategories.map(cat => getCategoryColor(cat));
  }
  
  function getRiskLabel(): string {
    switch (riskMeasure) {
      case 'volatility': return 'Volatility (%)';
      case 'var': return 'Value at Risk (%)';
      case 'max_drawdown': return 'Maximum Drawdown (%)';
      case 'tracking_error': return 'Tracking Error (%)';
      default: return 'Risk (%)';
    }
  }
  
  function getReturnLabel(): string {
    switch (returnMeasure) {
      case 'total_return': return 'Total Return (%)';
      case 'excess_return': return 'Excess Return (%)';
      case 'alpha': return 'Alpha (%)';
      case 'annualized_return': return 'Annualized Return (%)';
      default: return 'Return (%)';
    }
  }
  
  const handleExport = () => {
    const exportData = {
      metadata: {
        riskMeasure,
        returnMeasure,
        sizeMetric,
        showBubbles,
        colorByCategory,
        showQuadrants,
        exportDate: new Date().toISOString(),
      },
      data: combinedData,
      quadrantLines,
      stats: {
        totalPoints: combinedData.length,
        categories: uniqueCategories.length,
      },
    };
    
    const blob = new Blob([JSON.stringify(exportData, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `risk_return_analysis_${riskMeasure}_${returnMeasure}.json`;
    a.click();
    URL.revokeObjectURL(url);
  };
  
  const handleZoomTo = (category: string) => {
    const categoryData = combinedData.filter(d => d.category === category);
    if (categoryData.length === 0) return;
    
    const risks = categoryData.map(d => d.x);
    const returns = categoryData.map(d => d.y);
    
    const minRisk = Math.min(...risks);
    const maxRisk = Math.max(...risks);
    const minReturn = Math.min(...returns);
    const maxReturn = Math.max(...returns);
    
    // Add some padding
    const riskPadding = (maxRisk - minRisk) * 0.1;
    const returnPadding = (maxReturn - minReturn) * 0.1;
    
    // This would require access to the chart instance to set axis ranges
    console.log('Zoom to:', {
      risk: [minRisk - riskPadding, maxRisk + riskPadding],
      return: [minReturn - returnPadding, maxReturn + returnPadding],
    });
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
          <DotChartOutlined style={{ fontSize: 48, color: '#d9d9d9', marginBottom: 16 }} />
          <Text type="secondary">No data available for risk-return analysis</Text>
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
            <DotChartOutlined style={{ color: '#1890ff', fontSize: '18px' }} />
            <Title level={5} style={{ margin: 0 }}>
              Risk-Return Analysis
            </Title>
          </Space>
        </Col>
        
        <Col flex="auto">
          <Row gutter={[8, 8]} justify="end" align="middle">
            <Col>
              <Space size="small">
                <Text type="secondary" style={{ fontSize: '12px' }}>Risk:</Text>
                <Select
                  value={riskMeasure}
                  onChange={setRiskMeasure}
                  size="small"
                  style={{ width: 100 }}
                >
                  <Option value="volatility">Volatility</Option>
                  <Option value="var">VaR</Option>
                  <Option value="max_drawdown">Max DD</Option>
                  <Option value="tracking_error">Track Err</Option>
                </Select>
              </Space>
            </Col>
            
            <Col>
              <Space size="small">
                <Text type="secondary" style={{ fontSize: '12px' }}>Return:</Text>
                <Select
                  value={returnMeasure}
                  onChange={setReturnMeasure}
                  size="small"
                  style={{ width: 100 }}
                >
                  <Option value="total_return">Total</Option>
                  <Option value="excess_return">Excess</Option>
                  <Option value="alpha">Alpha</Option>
                  <Option value="annualized_return">Annualized</Option>
                </Select>
              </Space>
            </Col>
            
            <Col>
              <Space size="small">
                <Text type="secondary" style={{ fontSize: '12px' }}>Size:</Text>
                <Select
                  value={sizeMetric}
                  onChange={setSizeMetric}
                  size="small"
                  style={{ width: 80 }}
                  disabled={!showBubbles}
                >
                  <Option value="aum">AUM</Option>
                  <Option value="market_cap">Mkt Cap</Option>
                  <Option value="sharpe">Sharpe</Option>
                  <Option value="volume">Volume</Option>
                </Select>
              </Space>
            </Col>
            
            <Col>
              <Space size="small">
                <Text type="secondary" style={{ fontSize: '12px' }}>Bubbles:</Text>
                <Switch 
                  size="small"
                  checked={showBubbles}
                  onChange={setShowBubbles}
                />
              </Space>
            </Col>
            
            <Col>
              <Space size="small">
                <Text type="secondary" style={{ fontSize: '12px' }}>Quadrants:</Text>
                <Switch 
                  size="small"
                  checked={showQuadrants}
                  onChange={setShowQuadrants}
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
      
      {/* Category filters */}
      {uniqueCategories.length > 0 && (
        <Row style={{ marginBottom: 16 }}>
          <Col span={24}>
            <Space wrap>
              <Text type="secondary" style={{ fontSize: '12px' }}>Categories:</Text>
              {uniqueCategories.map(category => {
                const categoryCount = combinedData.filter(d => d.category === category).length;
                return (
                  <Tag
                    key={category}
                    color={selectedCategories.length === 0 || selectedCategories.includes(category) ? 
                           getCategoryColor(category) : 'default'}
                    style={{ 
                      cursor: 'pointer',
                      opacity: selectedCategories.length === 0 || selectedCategories.includes(category) ? 1 : 0.5
                    }}
                    onClick={() => {
                      if (selectedCategories.includes(category)) {
                        setSelectedCategories(selectedCategories.filter(c => c !== category));
                      } else {
                        setSelectedCategories([...selectedCategories, category]);
                      }
                    }}
                  >
                    {category} ({categoryCount})
                  </Tag>
                );
              })}
              {selectedCategories.length > 0 && (
                <Button size="small" type="link" onClick={() => setSelectedCategories([])}>
                  Clear All
                </Button>
              )}
            </Space>
          </Col>
        </Row>
      )}
      
      {/* Statistics */}
      <Row style={{ marginBottom: 16 }}>
        <Col span={24}>
          <Text type="secondary" style={{ fontSize: '12px' }}>
            Points: {combinedData.length} | 
            Categories: {uniqueCategories.length} |
            Risk Range: {Math.min(...combinedData.map(d => d.x)).toFixed(1)}% - {Math.max(...combinedData.map(d => d.x)).toFixed(1)}% |
            Return Range: {Math.min(...combinedData.map(d => d.y)).toFixed(1)}% - {Math.max(...combinedData.map(d => d.y)).toFixed(1)}%
          </Text>
        </Col>
      </Row>
      
      {/* Scatter plot */}
      <div style={{ height: height - 200 }}>
        <Scatter ref={plotRef} {...scatterConfig} />
      </div>
    </Card>
  );
};

export default RiskReturnScatterPlot;