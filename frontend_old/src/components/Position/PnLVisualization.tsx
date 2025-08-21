/**
 * P&L Visualization component for realized/unrealized P&L charts
 */

import React, { useMemo, useState } from 'react';
import { Card, Row, Col, Typography, Space, Radio, Statistic, Alert, Tabs } from 'antd';
import { 
  LineChart, 
  Line, 
  AreaChart, 
  Area, 
  XAxis, 
  YAxis, 
  CartesianGrid, 
  Tooltip, 
  ResponsiveContainer,
  BarChart,
  Bar,
  ReferenceLine
} from 'recharts';
import { 
  LineChartOutlined, 
  AreaChartOutlined, 
  BarChartOutlined, 
  RiseOutlined,
  FallOutlined
} from '@ant-design/icons';
import { usePositionData } from '../../hooks/usePositionData';
import { positionService } from '../../services/positionService';

const { Title, Text } = Typography;
const { TabPane } = Tabs;

interface PnLVisualizationProps {
  className?: string;
  height?: number;
  timeRange?: '1D' | '1W' | '1M' | '3M' | '1Y';
}

interface PnLChartData {
  date: string;
  unrealizedPnL: number;
  realizedPnL: number;
  totalPnL: number;
  cumulativePnL: number;
}

interface PositionPnLData {
  symbol: string;
  unrealizedPnL: number;
  realizedPnL: number;
  totalPnL: number;
  percentage: number;
  contribution: number;
}

export const PnLVisualization: React.FC<PnLVisualizationProps> = ({
  className,
  height = 400,
  timeRange = '1M'
}) => {
  const { positions, positionSummary, hasPositions } = usePositionData();
  const [chartType, setChartType] = useState<'line' | 'area' | 'bar'>('area');
  const [activeTab, setActiveTab] = useState<'timeline' | 'breakdown'>('timeline');

  // Historical P&L data (mock data for demonstration)
  const historicalPnLData = useMemo((): PnLChartData[] => {
    // In a real implementation, this would come from the positionService
    const days = timeRange === '1D' ? 1 : timeRange === '1W' ? 7 : timeRange === '1M' ? 30 : timeRange === '3M' ? 90 : 365;
    const data: PnLChartData[] = [];
    let cumulativePnL = 0;
    
    for (let i = days; i >= 0; i--) {
      const date = new Date();
      date.setDate(date.getDate() - i);
      
      // Generate sample data - in real app this would be actual historical data
      const dailyUnrealized = Math.random() * 2000 - 1000;
      const dailyRealized = Math.random() * 500 - 250;
      const totalDaily = dailyUnrealized + dailyRealized;
      cumulativePnL += totalDaily * 0.1; // Reduce volatility for demo
      
      data.push({
        date: date.toISOString().split('T')[0],
        unrealizedPnL: dailyUnrealized,
        realizedPnL: dailyRealized,
        totalPnL: totalDaily,
        cumulativePnL
      });
    }
    
    return data;
  }, [timeRange]);

  // Position P&L breakdown
  const positionPnLData = useMemo((): PositionPnLData[] => {
    if (!hasPositions || !positionSummary) return [];

    const totalPnL = positionSummary.pnl.totalPnl;
    
    return positions.map(position => {
      const totalPositionPnL = position.unrealizedPnl + position.realizedPnl;
      const exposure = position.currentPrice * position.quantity;
      const percentage = exposure > 0 ? (totalPositionPnL / exposure) * 100 : 0;
      const contribution = totalPnL !== 0 ? (totalPositionPnL / totalPnL) * 100 : 0;
      
      return {
        symbol: position.symbol,
        unrealizedPnL: position.unrealizedPnl,
        realizedPnL: position.realizedPnl,
        totalPnL: totalPositionPnL,
        percentage,
        contribution
      };
    }).sort((a, b) => Math.abs(b.totalPnL) - Math.abs(a.totalPnL));
  }, [positions, positionSummary, hasPositions]);

  const formatCurrency = (value: number) => {
    return new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency: 'USD',
      minimumFractionDigits: 0,
      maximumFractionDigits: 0
    }).format(value);
  };

  const CustomTooltip = ({ active, payload, label }: any) => {
    if (active && payload && payload.length) {
      return (
        <div style={{ 
          backgroundColor: 'white', 
          padding: '10px', 
          border: '1px solid #ccc',
          borderRadius: '4px',
          boxShadow: '0 2px 4px rgba(0,0,0,0.1)'
        }}>
          <Text strong>{label}</Text><br />
          {payload.map((entry: any, index: number) => (
            <div key={index}>
              <Text style={{ color: entry.color }}>
                {entry.name}: {formatCurrency(entry.value)}
              </Text><br />
            </div>
          ))}
        </div>
      );
    }
    return null;
  };

  const TimelineChart = () => {
    const commonProps = {
      data: historicalPnLData,
      margin: { top: 20, right: 30, left: 20, bottom: 5 }
    };

    switch (chartType) {
      case 'line':
        return (
          <ResponsiveContainer width="100%" height={height}>
            <LineChart {...commonProps}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="date" />
              <YAxis tickFormatter={formatCurrency} />
              <Tooltip content={<CustomTooltip />} />
              <ReferenceLine y={0} stroke="#666" strokeDasharray="2 2" />
              <Line type="monotone" dataKey="unrealizedPnL" stroke="#1890ff" name="Unrealized P&L" />
              <Line type="monotone" dataKey="realizedPnL" stroke="#52c41a" name="Realized P&L" />
              <Line type="monotone" dataKey="cumulativePnL" stroke="#722ed1" name="Cumulative P&L" strokeWidth={2} />
            </LineChart>
          </ResponsiveContainer>
        );
      
      case 'area':
        return (
          <ResponsiveContainer width="100%" height={height}>
            <AreaChart {...commonProps}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="date" />
              <YAxis tickFormatter={formatCurrency} />
              <Tooltip content={<CustomTooltip />} />
              <ReferenceLine y={0} stroke="#666" strokeDasharray="2 2" />
              <Area type="monotone" dataKey="unrealizedPnL" stackId="1" stroke="#1890ff" fill="#1890ff" fillOpacity={0.6} name="Unrealized P&L" />
              <Area type="monotone" dataKey="realizedPnL" stackId="1" stroke="#52c41a" fill="#52c41a" fillOpacity={0.6} name="Realized P&L" />
            </AreaChart>
          </ResponsiveContainer>
        );
      
      case 'bar':
        return (
          <ResponsiveContainer width="100%" height={height}>
            <BarChart {...commonProps}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="date" />
              <YAxis tickFormatter={formatCurrency} />
              <Tooltip content={<CustomTooltip />} />
              <ReferenceLine y={0} stroke="#666" strokeDasharray="2 2" />
              <Bar dataKey="unrealizedPnL" fill="#1890ff" name="Unrealized P&L" />
              <Bar dataKey="realizedPnL" fill="#52c41a" name="Realized P&L" />
            </BarChart>
          </ResponsiveContainer>
        );
      
      default:
        return null;
    }
  };

  const BreakdownChart = () => (
    <ResponsiveContainer width="100%" height={height}>
      <BarChart 
        data={positionPnLData.slice(0, 10)} 
        margin={{ top: 20, right: 30, left: 20, bottom: 80 }}
      >
        <CartesianGrid strokeDasharray="3 3" />
        <XAxis 
          dataKey="symbol" 
          angle={-45}
          textAnchor="end"
          height={80}
          interval={0}
        />
        <YAxis tickFormatter={formatCurrency} />
        <Tooltip content={<CustomTooltip />} />
        <ReferenceLine y={0} stroke="#666" strokeDasharray="2 2" />
        <Bar dataKey="unrealizedPnL" fill="#1890ff" name="Unrealized P&L" />
        <Bar dataKey="realizedPnL" fill="#52c41a" name="Realized P&L" />
      </BarChart>
    </ResponsiveContainer>
  );

  if (!hasPositions && !positionSummary) {
    return (
      <Card className={className}>
        <Alert
          message="No P&L Data"
          description="P&L visualization requires position data"
          type="info"
          showIcon
        />
      </Card>
    );
  }

  return (
    <div className={className}>
      <Card
        title={
          <Space>
            <LineChartOutlined />
            <span>P&L Analysis</span>
          </Space>
        }
        extra={
          activeTab === 'timeline' && (
            <Radio.Group
              value={chartType}
              onChange={(e) => setChartType(e.target.value)}
              size="small"
            >
              <Radio.Button value="line">
                <LineChartOutlined /> Line
              </Radio.Button>
              <Radio.Button value="area">
                <AreaChartOutlined /> Area
              </Radio.Button>
              <Radio.Button value="bar">
                <BarChartOutlined /> Bar
              </Radio.Button>
            </Radio.Group>
          )
        }
      >
        {/* P&L Summary Statistics */}
        {positionSummary && (
          <Row gutter={[16, 16]} style={{ marginBottom: 16 }}>
            <Col xs={24} sm={8}>
              <Statistic
                title="Unrealized P&L"
                value={positionSummary.pnl.unrealizedPnl}
                formatter={(value) => formatCurrency(Number(value))}
                prefix={positionSummary.pnl.unrealizedPnl >= 0 ? <RiseOutlined /> : <FallOutlined />}
                valueStyle={{ color: positionSummary.pnl.unrealizedPnl >= 0 ? '#3f8600' : '#cf1322' }}
              />
            </Col>
            <Col xs={24} sm={8}>
              <Statistic
                title="Realized P&L"
                value={positionSummary.pnl.realizedPnl}
                formatter={(value) => formatCurrency(Number(value))}
                valueStyle={{ color: positionSummary.pnl.realizedPnl >= 0 ? '#3f8600' : '#cf1322' }}
              />
            </Col>
            <Col xs={24} sm={8}>
              <Statistic
                title="Total P&L"
                value={positionSummary.pnl.totalPnl}
                formatter={(value) => formatCurrency(Number(value))}
                prefix={positionSummary.pnl.totalPnl >= 0 ? <RiseOutlined /> : <FallOutlined />}
                valueStyle={{ color: positionSummary.pnl.totalPnl >= 0 ? '#3f8600' : '#cf1322', fontSize: '20px' }}
              />
            </Col>
          </Row>
        )}

        {/* Chart Tabs */}
        <Tabs activeKey={activeTab} onChange={setActiveTab}>
          <TabPane tab="P&L Timeline" key="timeline">
            <TimelineChart />
          </TabPane>
          <TabPane tab="Position Breakdown" key="breakdown">
            <BreakdownChart />
            
            {/* Top Positions Summary */}
            <Row gutter={[16, 16]} style={{ marginTop: 16 }}>
              <Col span={24}>
                <Title level={5}>Top P&L Contributors</Title>
                <Row gutter={[8, 8]}>
                  {positionPnLData.slice(0, 5).map((item, index) => (
                    <Col xs={24} sm={12} md={8} lg={4} key={item.symbol}>
                      <Card size="small">
                        <Space direction="vertical" style={{ width: '100%' }}>
                          <Text strong>{item.symbol}</Text>
                          <Text 
                            style={{ 
                              color: item.totalPnL >= 0 ? '#3f8600' : '#cf1322',
                              fontSize: '14px'
                            }}
                          >
                            {formatCurrency(item.totalPnL)}
                          </Text>
                          <Text type="secondary" style={{ fontSize: '12px' }}>
                            {item.percentage.toFixed(1)}% return
                          </Text>
                        </Space>
                      </Card>
                    </Col>
                  ))}
                </Row>
              </Col>
            </Row>
          </TabPane>
        </Tabs>
      </Card>
    </div>
  );
};

export default PnLVisualization;