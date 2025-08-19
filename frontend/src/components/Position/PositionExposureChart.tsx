/**
 * Position Exposure Chart component for visualizing position sizes and exposure
 */

import React, { useMemo } from 'react';
import { Card, Row, Col, Typography, Space, Radio, Alert } from 'antd';
import { PieChart, Pie, Cell, BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend } from 'recharts';
import { PieChartOutlined, BarChartOutlined } from '@ant-design/icons';
import { usePositionData } from '../../hooks/usePositionData';
import { Position } from '../../types/position';

const { Title, Text } = Typography;

interface PositionExposureChartProps {
  className?: string;
  chartType?: 'pie' | 'bar';
  height?: number;
}

interface ChartData {
  name: string;
  value: number;
  pnl: number;
  percentage: number;
  side: 'LONG' | 'SHORT';
  color: string;
}

const COLORS = [
  '#1890ff', '#52c41a', '#faad14', '#f5222d', '#722ed1', 
  '#fa8c16', '#13c2c2', '#eb2f96', '#52c41a', '#1890ff'
];

export const PositionExposureChart: React.FC<PositionExposureChartProps> = ({
  className,
  chartType: initialChartType = 'pie',
  height = 400
}) => {
  const { positions, positionSummary, hasPositions } = usePositionData();
  const [chartType, setChartType] = React.useState<'pie' | 'bar'>(initialChartType);

  const chartData = useMemo((): ChartData[] => {
    if (!hasPositions) return [];

    const totalExposure = positionSummary?.totalExposure || 1;

    return positions.map((position, index) => {
      const exposure = Math.abs(position.currentPrice * position.quantity);
      const percentage = (exposure / totalExposure) * 100;
      
      return {
        name: position.symbol,
        value: exposure,
        pnl: position.unrealizedPnl,
        percentage,
        side: position.side,
        color: COLORS[index % COLORS.length]
      };
    }).sort((a, b) => b.value - a.value);
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
      const data = payload[0].payload;
      return (
        <div style={{ 
          backgroundColor: 'white', 
          padding: '10px', 
          border: '1px solid #ccc',
          borderRadius: '4px',
          boxShadow: '0 2px 4px rgba(0,0,0,0.1)'
        }}>
          <Text strong>{data.name}</Text><br />
          <Text>Exposure: {formatCurrency(data.value)}</Text><br />
          <Text>Percentage: {data.percentage.toFixed(1)}%</Text><br />
          <Text>Side: {data.side}</Text><br />
          <Text style={{ color: data.pnl >= 0 ? '#3f8600' : '#cf1322' }}>
            P&L: {formatCurrency(data.pnl)}
          </Text>
        </div>
      );
    }
    return null;
  };

  const PieChartComponent = () => (
    <ResponsiveContainer width="100%" height={height}>
      <PieChart>
        <Pie
          data={chartData}
          cx="50%"
          cy="50%"
          labelLine={false}
          label={({ name, percentage }) => `${name} ${percentage.toFixed(1)}%`}
          outerRadius={height / 3}
          fill="#8884d8"
          dataKey="value"
        >
          {chartData.map((entry, index) => (
            <Cell key={`cell-${index}`} fill={entry.color} />
          ))}
        </Pie>
        <Tooltip content={<CustomTooltip />} />
        <Legend />
      </PieChart>
    </ResponsiveContainer>
  );

  const BarChartComponent = () => (
    <ResponsiveContainer width="100%" height={height}>
      <BarChart data={chartData} margin={{ top: 20, right: 30, left: 20, bottom: 80 }}>
        <CartesianGrid strokeDasharray="3 3" />
        <XAxis 
          dataKey="name" 
          angle={-45}
          textAnchor="end"
          height={80}
          interval={0}
        />
        <YAxis 
          tickFormatter={(value) => formatCurrency(value)}
        />
        <Tooltip content={<CustomTooltip />} />
        <Bar dataKey="value" name="Exposure">
          {chartData.map((entry, index) => (
            <Cell key={`cell-${index}`} fill={entry.color} />
          ))}
        </Bar>
      </BarChart>
    </ResponsiveContainer>
  );

  if (!hasPositions) {
    return (
      <Card className={className}>
        <Alert
          message="No Position Data"
          description="Position exposure chart requires active positions"
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
            {chartType === 'pie' ? <PieChartOutlined /> : <BarChartOutlined />}
            <span>Position Exposure</span>
          </Space>
        }
        extra={
          <Radio.Group
            value={chartType}
            onChange={(e) => setChartType(e.target.value)}
            size="small"
          >
            <Radio.Button value="pie">
              <PieChartOutlined /> Pie
            </Radio.Button>
            <Radio.Button value="bar">
              <BarChartOutlined /> Bar
            </Radio.Button>
          </Radio.Group>
        }
      >
        <Row gutter={[16, 16]}>
          {/* Chart */}
          <Col xs={24} lg={16}>
            {chartType === 'pie' ? <PieChartComponent /> : <BarChartComponent />}
          </Col>

          {/* Summary Statistics */}
          <Col xs={24} lg={8}>
            <Space direction="vertical" style={{ width: '100%' }}>
              <Title level={5}>Top Positions by Exposure</Title>
              
              {chartData.slice(0, 5).map((item, index) => (
                <Card key={item.name} size="small" style={{ marginBottom: 8 }}>
                  <Row align="middle">
                    <Col span={2}>
                      <div
                        style={{
                          width: 12,
                          height: 12,
                          backgroundColor: item.color,
                          borderRadius: 2
                        }}
                      />
                    </Col>
                    <Col span={8}>
                      <Text strong>{item.name}</Text>
                    </Col>
                    <Col span={6}>
                      <Text type="secondary">{item.percentage.toFixed(1)}%</Text>
                    </Col>
                    <Col span={8}>
                      <Text style={{ fontSize: '12px' }}>
                        {formatCurrency(item.value)}
                      </Text>
                    </Col>
                  </Row>
                  <Row>
                    <Col span={2} />
                    <Col span={22}>
                      <Text 
                        style={{ 
                          fontSize: '11px',
                          color: item.pnl >= 0 ? '#3f8600' : '#cf1322' 
                        }}
                      >
                        {item.side} • P&L: {formatCurrency(item.pnl)}
                      </Text>
                    </Col>
                  </Row>
                </Card>
              ))}

              {chartData.length > 5 && (
                <Text type="secondary" style={{ fontSize: '12px' }}>
                  ...and {chartData.length - 5} more positions
                </Text>
              )}
            </Space>
          </Col>
        </Row>

        {/* Exposure Summary */}
        <Row gutter={[16, 16]} style={{ marginTop: 16, paddingTop: 16, borderTop: '1px solid #f0f0f0' }}>
          <Col xs={24} sm={8}>
            <Card size="small">
              <Text type="secondary">Total Exposure</Text><br />
              <Text strong style={{ fontSize: '16px' }}>
                {formatCurrency(positionSummary?.totalExposure || 0)}
              </Text>
            </Card>
          </Col>
          <Col xs={24} sm={8}>
            <Card size="small">
              <Text type="secondary">Net Exposure</Text><br />
              <Text 
                strong 
                style={{ 
                  fontSize: '16px',
                  color: (positionSummary?.netExposure || 0) >= 0 ? '#3f8600' : '#cf1322'
                }}
              >
                {formatCurrency(positionSummary?.netExposure || 0)}
              </Text>
            </Card>
          </Col>
          <Col xs={24} sm={8}>
            <Card size="small">
              <Text type="secondary">Gross Exposure</Text><br />
              <Text strong style={{ fontSize: '16px' }}>
                {formatCurrency(positionSummary?.grossExposure || 0)}
              </Text>
            </Card>
          </Col>
        </Row>
      </Card>
    </div>
  );
};

export default PositionExposureChart;