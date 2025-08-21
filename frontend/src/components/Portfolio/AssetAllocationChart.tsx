/**
 * Asset Allocation Chart Component with pie chart and treemap options
 */

import React, { useState, useEffect, useMemo } from 'react';
import {
  Chart as ChartJS,
  ArcElement,
  CategoryScale,
  LinearScale,
  BarElement,
  Title,
  Tooltip,
  Legend,
} from 'chart.js';
import { Pie, Doughnut, Bar } from 'react-chartjs-2';
import { Card, Select, Radio, Row, Col, Statistic, Alert, Space, Button, Table } from 'antd';
import { ColumnType } from 'antd/es/table';
import { PieChartOutlined, BarChartOutlined, TableOutlined } from '@ant-design/icons';
import { 
  portfolioAggregationService, 
  PortfolioAggregation 
} from '../../services/portfolioAggregationService';
import { Position } from '../../types/position';

ChartJS.register(ArcElement, CategoryScale, LinearScale, BarElement, Title, Tooltip, Legend);

const { Option } = Select;

interface AllocationData {
  category: string;
  value: number;
  percentage: number;
  positions_count: number;
  color: string;
  subcategories?: AllocationData[];
}

interface AssetAllocationChartProps {
  height?: number;
  groupBy?: 'asset_class' | 'sector' | 'geography' | 'strategy' | 'currency';
  chartType?: 'pie' | 'doughnut' | 'treemap' | 'bar';
}

const AssetAllocationChart: React.FC<AssetAllocationChartProps> = ({
  height = 400,
  groupBy = 'asset_class',
  chartType = 'doughnut'
}) => {
  const [portfolioData, setPortfolioData] = useState<PortfolioAggregation | null>(null);
  const [allPositions, setAllPositions] = useState<Position[]>([]);
  const [selectedGroupBy, setSelectedGroupBy] = useState(groupBy);
  const [selectedChartType, setSelectedChartType] = useState(chartType);
  const [showSubcategories, setShowSubcategories] = useState(false);
  const [loading, setLoading] = useState(true);

  // Asset class mappings
  const assetClassMappings: Record<string, string> = {
    'STK': 'Stocks',
    'OPT': 'Options',
    'FUT': 'Futures',
    'CASH': 'Cash',
    'BOND': 'Bonds',
    'CRYPTO': 'Cryptocurrency',
    'COMMODITY': 'Commodities',
    'FOREX': 'Forex'
  };

  // Sector mappings (simplified)
  const sectorMappings: Record<string, string> = {
    'AAPL': 'Technology',
    'MSFT': 'Technology',
    'GOOGL': 'Technology',
    'AMZN': 'Technology',
    'TSLA': 'Technology',
    'JPM': 'Financials',
    'BAC': 'Financials',
    'WFC': 'Financials',
    'JNJ': 'Healthcare',
    'PFE': 'Healthcare',
    'KO': 'Consumer Goods',
    'PG': 'Consumer Goods',
    'XOM': 'Energy',
    'CVX': 'Energy'
  };

  // Geography mappings (simplified)
  const geographyMappings: Record<string, string> = {
    'NASDAQ': 'North America',
    'NYSE': 'North America',
    'LSE': 'Europe',
    'EUREX': 'Europe',
    'TSE': 'Asia',
    'HKEX': 'Asia'
  };

  // Color palettes for different chart types
  const colorPalettes = {
    primary: ['#1890ff', '#52c41a', '#faad14', '#ff4d4f', '#722ed1', '#13c2c2', '#eb2f96', '#f5222d'],
    cool: ['#1890ff', '#096dd9', '#0050b3', '#003a8c', '#002766', '#13c2c2', '#08979c', '#006d75'],
    warm: ['#ff4d4f', '#ff7875', '#ffa39e', '#ffccc7', '#ffe1e1', '#faad14', '#ffc53d', '#ffd666'],
    earth: ['#52c41a', '#73d13d', '#95de64', '#b7eb8f', '#d9f7be', '#faad14', '#ffc53d', '#ffd666']
  };

  useEffect(() => {
    const handleAggregationUpdate = (aggregation: PortfolioAggregation) => {
      setPortfolioData(aggregation);
      
      // Get all positions from the aggregation service
      const positions = portfolioAggregationService.getAllPositions();
      setAllPositions(positions);
      
      setLoading(false);
    };

    portfolioAggregationService.addAggregationHandler(handleAggregationUpdate);

    const initialData = portfolioAggregationService.getPortfolioAggregation();
    if (initialData) {
      handleAggregationUpdate(initialData);
    }

    return () => {
      portfolioAggregationService.removeAggregationHandler(handleAggregationUpdate);
    };
  }, []);

  // Process allocation data based on grouping
  const allocationData = useMemo(() => {
    if (!portfolioData || allPositions.length === 0) return [];

    const totalValue = allPositions.reduce((sum, pos) => 
      sum + Math.abs(pos.currentPrice * pos.quantity), 0);

    if (totalValue === 0) return [];

    const groups: Record<string, {value: number, positions: Position[], subcategories?: Record<string, {value: number, positions: Position[]}>}> = {};

    allPositions.forEach(position => {
      const value = Math.abs(position.currentPrice * position.quantity);
      let category = 'Other';

      switch (selectedGroupBy) {
        case 'asset_class':
          // Determine asset class from symbol pattern
          if (position.symbol.includes('USD') || position.symbol.includes('EUR')) {
            category = 'Forex';
          } else if (position.symbol.length <= 5 && /^[A-Z]+$/.test(position.symbol)) {
            category = 'Stocks';
          } else {
            category = 'Other';
          }
          break;
        
        case 'sector':
          category = sectorMappings[position.symbol] || 'Other';
          break;
        
        case 'geography':
          category = geographyMappings[position.venue] || 'Other';
          break;
        
        case 'strategy':
          // Would need strategy mapping - for now use mock data
          const strategies = ['Growth Strategy', 'Value Strategy', 'Momentum Strategy', 'Defensive Strategy'];
          category = strategies[Math.floor(Math.random() * strategies.length)];
          break;
        
        case 'currency':
          category = position.currency;
          break;
      }

      if (!groups[category]) {
        groups[category] = { value: 0, positions: [], subcategories: {} };
      }
      
      groups[category].value += value;
      groups[category].positions.push(position);

      // Add subcategory logic for detailed breakdown
      if (showSubcategories && selectedGroupBy === 'sector') {
        const subcategory = position.symbol;
        if (!groups[category].subcategories![subcategory]) {
          groups[category].subcategories![subcategory] = { value: 0, positions: [] };
        }
        groups[category].subcategories![subcategory].value += value;
        groups[category].subcategories![subcategory].positions.push(position);
      }
    });

    // Convert to allocation data format
    const allocations: AllocationData[] = Object.entries(groups)
      .map(([category, data], index) => ({
        category,
        value: data.value,
        percentage: (data.value / totalValue) * 100,
        positions_count: data.positions.length,
        color: colorPalettes.primary[index % colorPalettes.primary.length],
        subcategories: showSubcategories && data.subcategories ? 
          Object.entries(data.subcategories).map(([subcat, subdata], subindex) => ({
            category: subcat,
            value: subdata.value,
            percentage: (subdata.value / totalValue) * 100,
            positions_count: subdata.positions.length,
            color: colorPalettes.cool[subindex % colorPalettes.cool.length]
          })) : undefined
      }))
      .sort((a, b) => b.value - a.value);

    return allocations;
  }, [portfolioData, allPositions, selectedGroupBy, showSubcategories]);

  // Chart data configuration
  const chartData = useMemo(() => {
    const labels = allocationData.map(item => item.category);
    const data = allocationData.map(item => item.percentage);
    const backgroundColor = allocationData.map(item => item.color);
    const borderColor = backgroundColor.map(color => color);

    return {
      labels,
      datasets: [
        {
          label: 'Allocation %',
          data,
          backgroundColor,
          borderColor,
          borderWidth: 2,
          hoverOffset: 4,
        },
      ],
    };
  }, [allocationData]);

  // Chart options
  const chartOptions = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: {
        position: 'right' as const,
        labels: {
          usePointStyle: true,
          padding: 20,
          generateLabels: (chart: any) => {
            const data = chart.data;
            return data.labels.map((label: string, index: number) => ({
              text: `${label} (${data.datasets[0].data[index].toFixed(1)}%)`,
              fillStyle: data.datasets[0].backgroundColor[index],
              strokeStyle: data.datasets[0].borderColor[index],
              lineWidth: 2,
              hidden: false,
              index
            }));
          }
        }
      },
      title: {
        display: true,
        text: `Asset Allocation by ${selectedGroupBy.replace('_', ' ').toUpperCase()}`,
      },
      tooltip: {
        callbacks: {
          label: function(context: any) {
            const allocation = allocationData[context.dataIndex];
            return [
              `${context.label}: ${context.parsed.toFixed(1)}%`,
              `Value: $${allocation.value.toLocaleString()}`,
              `Positions: ${allocation.positions_count}`
            ];
          }
        }
      }
    },
    elements: {
      arc: {
        borderWidth: 2,
      }
    }
  };

  // Bar chart options
  const barChartOptions = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: {
        display: false
      },
      title: {
        display: true,
        text: `Asset Allocation by ${selectedGroupBy.replace('_', ' ').toUpperCase()}`,
      },
      tooltip: {
        callbacks: {
          label: function(context: any) {
            const allocation = allocationData[context.dataIndex];
            return [
              `${context.label}: ${context.parsed.y.toFixed(1)}%`,
              `Value: $${allocation.value.toLocaleString()}`,
              `Positions: ${allocation.positions_count}`
            ];
          }
        }
      }
    },
    scales: {
      y: {
        beginAtZero: true,
        title: {
          display: true,
          text: 'Allocation (%)'
        },
        ticks: {
          callback: function(value: any) {
            return value + '%';
          }
        }
      },
      x: {
        title: {
          display: true,
          text: 'Category'
        }
      }
    },
  };

  // Table columns for detailed view
  const tableColumns: ColumnType<AllocationData>[] = [
    {
      title: 'Category',
      dataIndex: 'category',
      key: 'category',
      width: 150,
      render: (category: string, record: AllocationData) => (
        <div style={{ display: 'flex', alignItems: 'center' }}>
          <div 
            style={{ 
              width: 16, 
              height: 16, 
              backgroundColor: record.color,
              marginRight: 8,
              borderRadius: '2px'
            }}
          />
          {category}
        </div>
      ),
    },
    {
      title: 'Value',
      dataIndex: 'value',
      key: 'value',
      width: 120,
      render: (value: number) => `$${value.toLocaleString()}`,
      sorter: (a, b) => a.value - b.value,
    },
    {
      title: 'Percentage',
      dataIndex: 'percentage',
      key: 'percentage',
      width: 100,
      render: (percentage: number) => (
        <span style={{ fontWeight: 'bold' }}>
          {percentage.toFixed(1)}%
        </span>
      ),
      sorter: (a, b) => a.percentage - b.percentage,
    },
    {
      title: 'Positions',
      dataIndex: 'positions_count',
      key: 'positions_count',
      width: 100,
      sorter: (a, b) => a.positions_count - b.positions_count,
    },
  ];

  // Summary statistics
  const summaryStats = useMemo(() => {
    if (allocationData.length === 0) return null;

    const totalValue = allocationData.reduce((sum, item) => sum + item.value, 0);
    const largestAllocation = Math.max(...allocationData.map(item => item.percentage));
    const top3Concentration = allocationData.slice(0, 3).reduce((sum, item) => sum + item.percentage, 0);
    const diversificationScore = 100 - largestAllocation; // Higher is more diversified

    return {
      totalValue,
      largestAllocation,
      top3Concentration,
      diversificationScore,
      categoriesCount: allocationData.length
    };
  }, [allocationData]);

  const renderChart = () => {
    if (allocationData.length === 0) {
      return (
        <Alert
          message="No Allocation Data"
          description="No positions found to analyze allocation."
          type="info"
          showIcon
        />
      );
    }

    switch (selectedChartType) {
      case 'pie':
        return <Pie data={chartData} options={chartOptions} height={height} />;
      case 'doughnut':
        return <Doughnut data={chartData} options={chartOptions} height={height} />;
      case 'bar':
        return <Bar data={chartData} options={barChartOptions} height={height} />;
      case 'treemap':
        return (
          <div style={{ height, display: 'flex', flexWrap: 'wrap', alignItems: 'flex-start' }}>
            {allocationData.map((item, index) => (
              <div
                key={index}
                style={{
                  width: `${Math.sqrt(item.percentage) * 8}px`,
                  height: `${Math.sqrt(item.percentage) * 8}px`,
                  backgroundColor: item.color,
                  margin: '2px',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  color: 'white',
                  fontSize: '10px',
                  fontWeight: 'bold',
                  textAlign: 'center',
                  minWidth: '60px',
                  minHeight: '40px'
                }}
                title={`${item.category}: ${item.percentage.toFixed(1)}%`}
              >
                <div>
                  <div>{item.category}</div>
                  <div>{item.percentage.toFixed(1)}%</div>
                </div>
              </div>
            ))}
          </div>
        );
      default:
        return <Doughnut data={chartData} options={chartOptions} height={height} />;
    }
  };

  if (loading) {
    return (
      <Card title="Asset Allocation Analysis" loading={true}>
        <div style={{ height }}>Loading allocation data...</div>
      </Card>
    );
  }

  return (
    <Card
      title="Asset Allocation Analysis"
      extra={
        <Space>
          <Select 
            value={selectedGroupBy} 
            onChange={setSelectedGroupBy}
            style={{ width: 130 }}
          >
            <Option value="asset_class">Asset Class</Option>
            <Option value="sector">Sector</Option>
            <Option value="geography">Geography</Option>
            <Option value="strategy">Strategy</Option>
            <Option value="currency">Currency</Option>
          </Select>
          
          <Radio.Group 
            value={selectedChartType} 
            onChange={(e) => setSelectedChartType(e.target.value)}
            size="small"
          >
            <Radio.Button value="doughnut">
              <PieChartOutlined /> Doughnut
            </Radio.Button>
            <Radio.Button value="pie">
              <PieChartOutlined /> Pie
            </Radio.Button>
            <Radio.Button value="bar">
              <BarChartOutlined /> Bar
            </Radio.Button>
            <Radio.Button value="treemap">
              <TableOutlined /> Treemap
            </Radio.Button>
          </Radio.Group>
        </Space>
      }
    >
      {/* Summary Statistics */}
      {summaryStats && (
        <Row gutter={16} style={{ marginBottom: 16 }}>
          <Col span={6}>
            <Statistic
              title="Total Portfolio Value"
              value={summaryStats.totalValue}
              precision={0}
              formatter={(value) => `$${Number(value).toLocaleString()}`}
              valueStyle={{ color: '#1890ff' }}
            />
          </Col>
          <Col span={6}>
            <Statistic
              title="Largest Allocation"
              value={summaryStats.largestAllocation}
              precision={1}
              suffix="%"
              valueStyle={{ 
                color: summaryStats.largestAllocation > 50 ? '#ff4d4f' : 
                       summaryStats.largestAllocation > 30 ? '#faad14' : '#52c41a'
              }}
            />
          </Col>
          <Col span={6}>
            <Statistic
              title="Top 3 Concentration"
              value={summaryStats.top3Concentration}
              precision={1}
              suffix="%"
              valueStyle={{ 
                color: summaryStats.top3Concentration > 75 ? '#ff4d4f' : 
                       summaryStats.top3Concentration > 50 ? '#faad14' : '#52c41a'
              }}
            />
          </Col>
          <Col span={6}>
            <Statistic
              title="Diversification Score"
              value={summaryStats.diversificationScore}
              precision={0}
              suffix="/100"
              valueStyle={{ 
                color: summaryStats.diversificationScore > 70 ? '#52c41a' : 
                       summaryStats.diversificationScore > 50 ? '#faad14' : '#ff4d4f'
              }}
            />
          </Col>
        </Row>
      )}

      {/* Chart */}
      <div style={{ height }}>
        {renderChart()}
      </div>

      {/* Detailed Table */}
      <div style={{ marginTop: 16 }}>
        <Table
          columns={tableColumns}
          dataSource={allocationData}
          rowKey="category"
          pagination={false}
          size="small"
          bordered
          title={() => `Allocation Breakdown (${allocationData.length} categories)`}
        />
      </div>
    </Card>
  );
};

export default AssetAllocationChart;