/**
 * PerformanceAttribution Component - Sprint 3 Integration
 * Advanced performance attribution analysis with factor breakdown
 */

import React, { useState, useEffect, useMemo } from 'react';
import { Card, Row, Col, Table, Tabs, Select, DatePicker, Button, Space, Typography, Tooltip, Progress, Tag } from 'antd';
import { 
  PieChartOutlined, 
  BarChartOutlined, 
  TableOutlined,
  DownloadOutlined,
  InfoCircleOutlined,
  RiseOutlined,
  FallOutlined
} from '@ant-design/icons';
import { Pie, Column, Treemap, Waterfall } from '@ant-design/plots';
import dayjs from 'dayjs';
import usePerformanceCalculator from '../../hooks/analytics/usePerformanceCalculator';

const { Title, Text } = Typography;
const { TabPane } = Tabs;
const { Option } = Select;
const { RangePicker } = DatePicker;

interface PerformanceAttributionProps {
  portfolioId: string;
  className?: string;
  defaultBenchmark?: string;
  height?: number;
  showExportOptions?: boolean;
}

type AttributionType = 'sector' | 'security' | 'factor' | 'style';
type TimeHorizon = '1M' | '3M' | '6M' | '1Y' | 'YTD' | 'Custom';

const PerformanceAttribution: React.FC<PerformanceAttributionProps> = ({
  portfolioId,
  className,
  defaultBenchmark = 'SPY',
  height = 800,
  showExportOptions = true,
}) => {
  // State
  const [attributionType, setAttributionType] = useState<AttributionType>('sector');
  const [timeHorizon, setTimeHorizon] = useState<TimeHorizon>('3M');
  const [customDateRange, setCustomDateRange] = useState<[dayjs.Dayjs, dayjs.Dayjs] | null>(null);
  const [benchmark, setBenchmark] = useState(defaultBenchmark);
  const [activeTab, setActiveTab] = useState('overview');
  
  const { 
    result, 
    isCalculating, 
    error, 
    calculate,
    getDetailedAttribution,
  } = usePerformanceCalculator({
    autoCalculate: false,
    cacheResults: true,
  });
  
  const [attributionData, setAttributionData] = useState<any>(null);
  
  // Calculate date range based on time horizon
  const dateRange = useMemo(() => {
    const endDate = dayjs();
    let startDate: dayjs.Dayjs;
    
    switch (timeHorizon) {
      case '1M':
        startDate = endDate.subtract(1, 'month');
        break;
      case '3M':
        startDate = endDate.subtract(3, 'month');
        break;
      case '6M':
        startDate = endDate.subtract(6, 'month');
        break;
      case '1Y':
        startDate = endDate.subtract(1, 'year');
        break;
      case 'YTD':
        startDate = dayjs().startOf('year');
        break;
      case 'Custom':
        if (customDateRange) {
          return {
            start_date: customDateRange[0].format('YYYY-MM-DD'),
            end_date: customDateRange[1].format('YYYY-MM-DD'),
          };
        }
        startDate = endDate.subtract(3, 'month');
        break;
      default:
        startDate = endDate.subtract(3, 'month');
    }
    
    return {
      start_date: startDate.format('YYYY-MM-DD'),
      end_date: endDate.format('YYYY-MM-DD'),
    };
  }, [timeHorizon, customDateRange]);
  
  // Fetch attribution data when parameters change
  useEffect(() => {
    const fetchAttribution = async () => {
      try {
        const attributionResult = await getDetailedAttribution({
          portfolio_id: portfolioId,
          attribution_type: attributionType,
          benchmark: benchmark,
          ...dateRange,
        });
        setAttributionData(attributionResult);
      } catch (err) {
        console.error('Failed to fetch attribution data:', err);
      }
    };
    
    fetchAttribution();
  }, [portfolioId, attributionType, benchmark, dateRange, getDetailedAttribution]);
  
  // Prepare chart data
  const attributionBreakdownData = useMemo(() => {
    if (!attributionData || !attributionData.breakdown) return [];
    
    return attributionData.breakdown.map((item: any) => ({
      name: item.name,
      allocation: item.allocation_contribution,
      selection: item.selection_contribution,
      total: item.total_contribution,
      weight_diff: item.portfolio_weight - item.benchmark_weight,
      portfolio_weight: item.portfolio_weight,
      benchmark_weight: item.benchmark_weight,
    }));
  }, [attributionData]);
  
  // Waterfall chart data for total attribution
  const waterfallData = useMemo(() => {
    if (!attributionData) return [];
    
    const data = [
      { name: 'Benchmark Return', value: 0, type: 'start' },
      { name: 'Asset Allocation', value: attributionData.allocation_effect, type: 'positive' },
      { name: 'Security Selection', value: attributionData.selection_effect, type: 'positive' },
    ];
    
    if (attributionData.interaction_effect) {
      data.push({ name: 'Interaction', value: attributionData.interaction_effect, type: 'positive' });
    }
    
    if (attributionData.currency_effect) {
      data.push({ name: 'Currency', value: attributionData.currency_effect, type: 'positive' });
    }
    
    data.push({ 
      name: 'Total Active Return', 
      value: attributionData.total_active_return, 
      type: 'end' 
    });
    
    return data;
  }, [attributionData]);
  
  // Treemap data for contribution visualization
  const treemapData = useMemo(() => {
    if (!attributionBreakdownData || attributionBreakdownData.length === 0) return { name: 'root', children: [] };
    
    return {
      name: 'Attribution',
      children: attributionBreakdownData
        .filter((item: any) => Math.abs(item.total) > 0.001) // Filter out negligible contributions
        .map((item: any) => ({
          name: item.name,
          value: Math.abs(item.total),
          contribution: item.total,
          allocation: item.allocation,
          selection: item.selection,
        })),
    };
  }, [attributionBreakdownData]);
  
  // Table columns for detailed breakdown
  const columns = [
    {
      title: 'Name',
      dataIndex: 'name',
      key: 'name',
      fixed: 'left' as const,
      width: 150,
    },
    {
      title: 'Portfolio Weight',
      dataIndex: 'portfolio_weight',
      key: 'portfolio_weight',
      render: (value: number) => `${(value * 100).toFixed(2)}%`,
      sorter: (a: any, b: any) => a.portfolio_weight - b.portfolio_weight,
    },
    {
      title: 'Benchmark Weight',
      dataIndex: 'benchmark_weight',
      key: 'benchmark_weight',
      render: (value: number) => `${(value * 100).toFixed(2)}%`,
      sorter: (a: any, b: any) => a.benchmark_weight - b.benchmark_weight,
    },
    {
      title: 'Weight Difference',
      dataIndex: 'weight_diff',
      key: 'weight_diff',
      render: (value: number) => (
        <span style={{ color: value >= 0 ? '#52c41a' : '#ff4d4f' }}>
          {value >= 0 ? '+' : ''}{(value * 100).toFixed(2)}%
        </span>
      ),
      sorter: (a: any, b: any) => a.weight_diff - b.weight_diff,
    },
    {
      title: (
        <Tooltip title="Contribution from asset allocation decisions">
          <Space>
            Allocation Effect
            <InfoCircleOutlined />
          </Space>
        </Tooltip>
      ),
      dataIndex: 'allocation',
      key: 'allocation',
      render: (value: number) => (
        <span style={{ color: value >= 0 ? '#52c41a' : '#ff4d4f' }}>
          {value >= 0 ? '+' : ''}{(value * 100).toFixed(3)}%
        </span>
      ),
      sorter: (a: any, b: any) => a.allocation - b.allocation,
    },
    {
      title: (
        <Tooltip title="Contribution from security selection decisions">
          <Space>
            Selection Effect
            <InfoCircleOutlined />
          </Space>
        </Tooltip>
      ),
      dataIndex: 'selection',
      key: 'selection',
      render: (value: number) => (
        <span style={{ color: value >= 0 ? '#52c41a' : '#ff4d4f' }}>
          {value >= 0 ? '+' : ''}{(value * 100).toFixed(3)}%
        </span>
      ),
      sorter: (a: any, b: any) => a.selection - b.selection,
    },
    {
      title: 'Total Contribution',
      dataIndex: 'total',
      key: 'total',
      render: (value: number) => (
        <span style={{ color: value >= 0 ? '#52c41a' : '#ff4d4f', fontWeight: 'bold' }}>
          {value >= 0 ? '+' : ''}{(value * 100).toFixed(3)}%
        </span>
      ),
      sorter: (a: any, b: any) => a.total - b.total,
    },
  ];
  
  // Chart configurations
  const pieChartConfig = {
    data: attributionBreakdownData.filter((item: any) => item.total > 0).slice(0, 10),
    angleField: 'total',
    colorField: 'name',
    radius: 0.8,
    height: 300,
    label: {
      type: 'outer',
      content: (data: any) => `${data.name}\n${(data.total * 100).toFixed(2)}%`,
    },
    tooltip: {
      formatter: (data: any) => ({
        name: data.name,
        value: `${(data.total * 100).toFixed(3)}%`,
      }),
    },
  };
  
  const columnChartConfig = {
    data: attributionBreakdownData.slice(0, 15),
    isStack: true,
    xField: 'name',
    yField: ['allocation', 'selection'],
    seriesField: 'type',
    height: 300,
    label: {
      position: 'middle' as const,
      formatter: (data: any) => `${(data * 100).toFixed(1)}%`,
    },
    xAxis: {
      label: {
        autoRotate: true,
      },
    },
    yAxis: {
      label: {
        formatter: (v: string) => `${(Number(v) * 100).toFixed(1)}%`,
      },
    },
    tooltip: {
      formatter: (data: any) => [
        { name: 'Allocation', value: `${(data.allocation * 100).toFixed(3)}%` },
        { name: 'Selection', value: `${(data.selection * 100).toFixed(3)}%` },
        { name: 'Total', value: `${(data.total * 100).toFixed(3)}%` },
      ],
    },
  };
  
  const waterfallConfig = {
    data: waterfallData,
    xField: 'name',
    yField: 'value',
    height: 300,
    color: ({ type }: any) => {
      if (type === 'positive') return '#52c41a';
      if (type === 'negative') return '#ff4d4f';
      return '#1890ff';
    },
    label: {
      formatter: (data: any) => `${(data.value * 100).toFixed(2)}%`,
    },
    yAxis: {
      label: {
        formatter: (v: string) => `${(Number(v) * 100).toFixed(1)}%`,
      },
    },
  };
  
  const treemapConfig = {
    data: treemapData,
    colorField: 'name',
    height: 400,
    tooltip: {
      formatter: (data: any) => [
        { name: 'Total Contribution', value: `${(data.contribution * 100).toFixed(3)}%` },
        { name: 'Allocation', value: `${(data.allocation * 100).toFixed(3)}%` },
        { name: 'Selection', value: `${(data.selection * 100).toFixed(3)}%` },
      ],
    },
  };
  
  const handleExport = () => {
    // Implement export functionality
    const exportData = {
      attribution_type: attributionType,
      time_horizon: timeHorizon,
      benchmark,
      date_range: dateRange,
      attribution_data: attributionData,
      breakdown_data: attributionBreakdownData,
    };
    
    const blob = new Blob([JSON.stringify(exportData, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `attribution_${portfolioId}_${attributionType}_${timeHorizon}.json`;
    a.click();
    URL.revokeObjectURL(url);
  };
  
  return (
    <div className={className} style={{ height }}>
      <Row gutter={[16, 16]}>
        {/* Controls */}
        <Col span={24}>
          <Card size="small">
            <Row gutter={[16, 8]} align="middle">
              <Col xs={24} sm={8} md={4}>
                <Space direction="vertical" size={2}>
                  <Text type="secondary" style={{ fontSize: '12px' }}>Attribution Type</Text>
                  <Select
                    value={attributionType}
                    onChange={setAttributionType}
                    style={{ width: '100%' }}
                    size="small"
                  >
                    <Option value="sector">Sector</Option>
                    <Option value="security">Security</Option>
                    <Option value="factor">Factor</Option>
                    <Option value="style">Style</Option>
                  </Select>
                </Space>
              </Col>
              <Col xs={24} sm={8} md={4}>
                <Space direction="vertical" size={2}>
                  <Text type="secondary" style={{ fontSize: '12px' }}>Time Horizon</Text>
                  <Select
                    value={timeHorizon}
                    onChange={setTimeHorizon}
                    style={{ width: '100%' }}
                    size="small"
                  >
                    <Option value="1M">1 Month</Option>
                    <Option value="3M">3 Months</Option>
                    <Option value="6M">6 Months</Option>
                    <Option value="1Y">1 Year</Option>
                    <Option value="YTD">YTD</Option>
                    <Option value="Custom">Custom</Option>
                  </Select>
                </Space>
              </Col>
              {timeHorizon === 'Custom' && (
                <Col xs={24} sm={8} md={6}>
                  <Space direction="vertical" size={2}>
                    <Text type="secondary" style={{ fontSize: '12px' }}>Date Range</Text>
                    <RangePicker
                      value={customDateRange}
                      onChange={setCustomDateRange}
                      size="small"
                      style={{ width: '100%' }}
                    />
                  </Space>
                </Col>
              )}
              <Col xs={24} sm={8} md={4}>
                <Space direction="vertical" size={2}>
                  <Text type="secondary" style={{ fontSize: '12px' }}>Benchmark</Text>
                  <Select
                    value={benchmark}
                    onChange={setBenchmark}
                    style={{ width: '100%' }}
                    size="small"
                  >
                    <Option value="SPY">S&P 500 (SPY)</Option>
                    <Option value="QQQ">NASDAQ (QQQ)</Option>
                    <Option value="IWM">Russell 2000 (IWM)</Option>
                    <Option value="VTI">Total Market (VTI)</Option>
                  </Select>
                </Space>
              </Col>
              {showExportOptions && (
                <Col xs={24} sm={8} md={4}>
                  <Button 
                    icon={<DownloadOutlined />} 
                    size="small"
                    onClick={handleExport}
                    style={{ marginTop: 18 }}
                  >
                    Export
                  </Button>
                </Col>
              )}
            </Row>
          </Card>
        </Col>
        
        {/* Summary Cards */}
        {attributionData && (
          <Col span={24}>
            <Row gutter={[16, 16]}>
              <Col xs={12} sm={6}>
                <Card size="small">
                  <div style={{ textAlign: 'center' }}>
                    <Title level={4} style={{ margin: 0, color: '#1890ff' }}>
                      {(attributionData.total_active_return * 100).toFixed(3)}%
                    </Title>
                    <Text type="secondary">Total Active Return</Text>
                  </div>
                </Card>
              </Col>
              <Col xs={12} sm={6}>
                <Card size="small">
                  <div style={{ textAlign: 'center' }}>
                    <Title level={4} style={{ 
                      margin: 0, 
                      color: attributionData.allocation_effect >= 0 ? '#52c41a' : '#ff4d4f' 
                    }}>
                      {attributionData.allocation_effect >= 0 ? '+' : ''}
                      {(attributionData.allocation_effect * 100).toFixed(3)}%
                    </Title>
                    <Text type="secondary">Asset Allocation</Text>
                  </div>
                </Card>
              </Col>
              <Col xs={12} sm={6}>
                <Card size="small">
                  <div style={{ textAlign: 'center' }}>
                    <Title level={4} style={{ 
                      margin: 0, 
                      color: attributionData.selection_effect >= 0 ? '#52c41a' : '#ff4d4f' 
                    }}>
                      {attributionData.selection_effect >= 0 ? '+' : ''}
                      {(attributionData.selection_effect * 100).toFixed(3)}%
                    </Title>
                    <Text type="secondary">Security Selection</Text>
                  </div>
                </Card>
              </Col>
              <Col xs={12} sm={6}>
                <Card size="small">
                  <div style={{ textAlign: 'center' }}>
                    <Title level={4} style={{ 
                      margin: 0, 
                      color: (attributionData.interaction_effect || 0) >= 0 ? '#52c41a' : '#ff4d4f' 
                    }}>
                      {(attributionData.interaction_effect || 0) >= 0 ? '+' : ''}
                      {((attributionData.interaction_effect || 0) * 100).toFixed(3)}%
                    </Title>
                    <Text type="secondary">Interaction</Text>
                  </div>
                </Card>
              </Col>
            </Row>
          </Col>
        )}
        
        {/* Main Content */}
        <Col span={24}>
          <Card>
            <Tabs activeKey={activeTab} onChange={setActiveTab}>
              <TabPane tab={<Space><PieChartOutlined />Overview</Space>} key="overview">
                <Row gutter={[16, 16]}>
                  <Col xs={24} lg={12}>
                    <Card title="Attribution Breakdown" size="small">
                      {attributionBreakdownData.length > 0 ? (
                        <Pie {...pieChartConfig} />
                      ) : (
                        <div style={{ height: 300, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                          No data available
                        </div>
                      )}
                    </Card>
                  </Col>
                  <Col xs={24} lg={12}>
                    <Card title="Attribution Waterfall" size="small">
                      {waterfallData.length > 0 ? (
                        <Waterfall {...waterfallConfig} />
                      ) : (
                        <div style={{ height: 300, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                          No data available
                        </div>
                      )}
                    </Card>
                  </Col>
                </Row>
              </TabPane>
              
              <TabPane tab={<Space><BarChartOutlined />Detailed Analysis</Space>} key="detailed">
                <Row gutter={[16, 16]}>
                  <Col span={24}>
                    <Card title="Contribution by Component" size="small">
                      {attributionBreakdownData.length > 0 ? (
                        <Column {...columnChartConfig} />
                      ) : (
                        <div style={{ height: 300, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                          No data available
                        </div>
                      )}
                    </Card>
                  </Col>
                  <Col span={24}>
                    <Card title="Contribution Treemap" size="small">
                      {treemapData.children && treemapData.children.length > 0 ? (
                        <Treemap {...treemapConfig} />
                      ) : (
                        <div style={{ height: 400, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                          No data available
                        </div>
                      )}
                    </Card>
                  </Col>
                </Row>
              </TabPane>
              
              <TabPane tab={<Space><TableOutlined />Data Table</Space>} key="table">
                <Table
                  dataSource={attributionBreakdownData}
                  columns={columns}
                  rowKey="name"
                  size="small"
                  pagination={false}
                  scroll={{ x: 1000, y: 400 }}
                  loading={isCalculating}
                />
              </TabPane>
            </Tabs>
          </Card>
        </Col>
      </Row>
    </div>
  );
};

export default PerformanceAttribution;