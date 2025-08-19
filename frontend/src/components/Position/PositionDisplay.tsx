/**
 * Position Display component for current positions
 */

import React, { useState, useMemo } from 'react';
import { 
  Table, 
  Card, 
  Tag, 
  Typography, 
  Row, 
  Col, 
  Statistic, 
  Progress, 
  Space, 
  Button,
  Tooltip,
  Switch,
  Select,
  Alert
} from 'antd';
import type { ColumnsType } from 'antd/es/table';
import { 
  ArrowUpOutlined, 
  ArrowDownOutlined,
  DollarOutlined,
  BarChartOutlined,
  ReloadOutlined,
  FilterOutlined
} from '@ant-design/icons';
import { usePositionData } from '../../hooks/usePositionData';
import { Position, PositionSummary } from '../../types/position';

const { Title, Text } = Typography;
const { Option } = Select;

interface PositionDisplayProps {
  className?: string;
  showSummary?: boolean;
  showFilters?: boolean;
  maxHeight?: number;
}

export const PositionDisplay: React.FC<PositionDisplayProps> = ({
  className,
  showSummary = true,
  showFilters = true,
  maxHeight = 600
}) => {
  const { 
    positions, 
    positionSummary, 
    isLoading, 
    error, 
    refreshData,
    hasPositions 
  } = usePositionData();

  const [groupByVenue, setGroupByVenue] = useState(false);
  const [filterSide, setFilterSide] = useState<'ALL' | 'LONG' | 'SHORT'>('ALL');
  const [sortBy, setSortBy] = useState<'symbol' | 'pnl' | 'exposure'>('symbol');

  // Filter and sort positions
  const filteredPositions = useMemo(() => {
    let filtered = [...positions];

    // Filter by side
    if (filterSide !== 'ALL') {
      filtered = filtered.filter(pos => pos.side === filterSide);
    }

    // Sort positions
    filtered.sort((a, b) => {
      switch (sortBy) {
        case 'pnl':
          return b.unrealizedPnl - a.unrealizedPnl;
        case 'exposure':
          return (b.currentPrice * b.quantity) - (a.currentPrice * a.quantity);
        case 'symbol':
        default:
          return a.symbol.localeCompare(b.symbol);
      }
    });

    return filtered;
  }, [positions, filterSide, sortBy]);

  const formatCurrency = (value: number, currency: string = 'USD') => {
    return new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency: currency,
      minimumFractionDigits: 2,
      maximumFractionDigits: 2
    }).format(value);
  };

  const formatPercentage = (value: number) => {
    return `${value >= 0 ? '+' : ''}${value.toFixed(2)}%`;
  };

  const columns: ColumnsType<Position> = [
    {
      title: 'Symbol',
      dataIndex: 'symbol',
      key: 'symbol',
      fixed: 'left',
      width: 100,
      render: (symbol: string, record: Position) => (
        <Space direction="vertical" size="small">
          <Text strong>{symbol}</Text>
          <Text type="secondary" style={{ fontSize: '12px' }}>{record.venue}</Text>
        </Space>
      ),
    },
    {
      title: 'Side',
      dataIndex: 'side',
      key: 'side',
      width: 80,
      render: (side: 'LONG' | 'SHORT') => (
        <Tag color={side === 'LONG' ? 'green' : 'red'}>
          {side}
        </Tag>
      ),
    },
    {
      title: 'Quantity',
      dataIndex: 'quantity',
      key: 'quantity',
      width: 100,
      align: 'right',
      render: (quantity: number) => quantity.toLocaleString(),
    },
    {
      title: 'Avg Price',
      dataIndex: 'averagePrice',
      key: 'averagePrice',
      width: 100,
      align: 'right',
      render: (price: number, record: Position) => formatCurrency(price, record.currency),
    },
    {
      title: 'Current Price',
      dataIndex: 'currentPrice',
      key: 'currentPrice',
      width: 100,
      align: 'right',
      render: (price: number, record: Position) => formatCurrency(price, record.currency),
    },
    {
      title: 'Market Value',
      key: 'marketValue',
      width: 120,
      align: 'right',
      render: (_, record: Position) => {
        const marketValue = record.currentPrice * record.quantity;
        return formatCurrency(marketValue, record.currency);
      },
    },
    {
      title: 'Unrealized P&L',
      dataIndex: 'unrealizedPnl',
      key: 'unrealizedPnl',
      width: 120,
      align: 'right',
      render: (pnl: number, record: Position) => {
        const marketValue = record.currentPrice * record.quantity;
        const pnlPercent = marketValue > 0 ? (pnl / marketValue) * 100 : 0;
        
        return (
          <Space direction="vertical" size="small">
            <Text style={{ color: pnl >= 0 ? '#3f8600' : '#cf1322' }}>
              {formatCurrency(pnl, record.currency)}
            </Text>
            <Text 
              type="secondary" 
              style={{ 
                fontSize: '12px',
                color: pnl >= 0 ? '#3f8600' : '#cf1322' 
              }}
            >
              {formatPercentage(pnlPercent)}
            </Text>
          </Space>
        );
      },
      sorter: (a, b) => a.unrealizedPnl - b.unrealizedPnl,
    },
    {
      title: 'Realized P&L',
      dataIndex: 'realizedPnl',
      key: 'realizedPnl',
      width: 120,
      align: 'right',
      render: (pnl: number, record: Position) => (
        <Text style={{ color: pnl >= 0 ? '#3f8600' : '#cf1322' }}>
          {formatCurrency(pnl, record.currency)}
        </Text>
      ),
    },
  ];

  if (error) {
    return (
      <Alert 
        message="Position Data Error" 
        description={error} 
        type="error" 
        showIcon 
        className={className}
      />
    );
  }

  return (
    <div className={className}>
      <Row gutter={[16, 16]}>
        {/* Position Summary */}
        {showSummary && positionSummary && (
          <Col span={24}>
            <Card>
              <Row gutter={[16, 16]}>
                <Col xs={24} sm={6}>
                  <Statistic
                    title="Total Positions"
                    value={positionSummary.totalPositions}
                    prefix={<BarChartOutlined />}
                  />
                </Col>
                <Col xs={24} sm={6}>
                  <Statistic
                    title="Long Positions"
                    value={positionSummary.longPositions}
                    valueStyle={{ color: '#3f8600' }}
                  />
                </Col>
                <Col xs={24} sm={6}>
                  <Statistic
                    title="Short Positions"
                    value={positionSummary.shortPositions}
                    valueStyle={{ color: '#cf1322' }}
                  />
                </Col>
                <Col xs={24} sm={6}>
                  <Statistic
                    title="Net Exposure"
                    value={positionSummary.netExposure}
                    formatter={(value) => formatCurrency(Number(value), positionSummary.currency)}
                    prefix={positionSummary.netExposure >= 0 ? <ArrowUpOutlined /> : <ArrowDownOutlined />}
                    valueStyle={{ color: positionSummary.netExposure >= 0 ? '#3f8600' : '#cf1322' }}
                  />
                </Col>
              </Row>
            </Card>
          </Col>
        )}

        {/* Filters and Controls */}
        {showFilters && (
          <Col span={24}>
            <Card size="small">
              <Row gutter={[16, 8]} align="middle">
                <Col>
                  <Space>
                    <FilterOutlined />
                    <Text strong>Filters:</Text>
                  </Space>
                </Col>
                <Col>
                  <Space>
                    <Text>Side:</Text>
                    <Select
                      value={filterSide}
                      onChange={setFilterSide}
                      style={{ width: 100 }}
                      size="small"
                    >
                      <Option value="ALL">All</Option>
                      <Option value="LONG">Long</Option>
                      <Option value="SHORT">Short</Option>
                    </Select>
                  </Space>
                </Col>
                <Col>
                  <Space>
                    <Text>Sort by:</Text>
                    <Select
                      value={sortBy}
                      onChange={setSortBy}
                      style={{ width: 120 }}
                      size="small"
                    >
                      <Option value="symbol">Symbol</Option>
                      <Option value="pnl">P&L</Option>
                      <Option value="exposure">Exposure</Option>
                    </Select>
                  </Space>
                </Col>
                <Col>
                  <Space>
                    <Text>Group by venue:</Text>
                    <Switch
                      checked={groupByVenue}
                      onChange={setGroupByVenue}
                      size="small"
                    />
                  </Space>
                </Col>
                <Col flex="auto" />
                <Col>
                  <Tooltip title="Refresh position data">
                    <Button
                      icon={<ReloadOutlined />}
                      onClick={refreshData}
                      loading={isLoading}
                      size="small"
                    >
                      Refresh
                    </Button>
                  </Tooltip>
                </Col>
              </Row>
            </Card>
          </Col>
        )}

        {/* Positions Table */}
        <Col span={24}>
          <Card 
            title={
              <Space>
                <BarChartOutlined />
                <span>Current Positions</span>
                {hasPositions && (
                  <Tag color="blue">{filteredPositions.length} positions</Tag>
                )}
              </Space>
            }
          >
            {hasPositions ? (
              <Table
                columns={columns}
                dataSource={filteredPositions}
                rowKey="id"
                size="small"
                loading={isLoading}
                scroll={{ 
                  x: 800,
                  y: maxHeight - 200 
                }}
                pagination={{
                  pageSize: 20,
                  showSizeChanger: true,
                  showQuickJumper: true,
                  showTotal: (total, range) => 
                    `${range[0]}-${range[1]} of ${total} positions`
                }}
                summary={(data) => {
                  if (!data.length) return null;
                  
                  const totalUnrealizedPnl = data.reduce((sum, pos) => sum + pos.unrealizedPnl, 0);
                  const totalRealizedPnl = data.reduce((sum, pos) => sum + pos.realizedPnl, 0);
                  const totalMarketValue = data.reduce((sum, pos) => sum + (pos.currentPrice * pos.quantity), 0);
                  
                  return (
                    <Table.Summary fixed>
                      <Table.Summary.Row>
                        <Table.Summary.Cell index={0} colSpan={5}>
                          <Text strong>Totals ({data.length} positions)</Text>
                        </Table.Summary.Cell>
                        <Table.Summary.Cell index={5}>
                          <Text strong>{formatCurrency(totalMarketValue)}</Text>
                        </Table.Summary.Cell>
                        <Table.Summary.Cell index={6}>
                          <Text 
                            strong 
                            style={{ color: totalUnrealizedPnl >= 0 ? '#3f8600' : '#cf1322' }}
                          >
                            {formatCurrency(totalUnrealizedPnl)}
                          </Text>
                        </Table.Summary.Cell>
                        <Table.Summary.Cell index={7}>
                          <Text 
                            strong 
                            style={{ color: totalRealizedPnl >= 0 ? '#3f8600' : '#cf1322' }}
                          >
                            {formatCurrency(totalRealizedPnl)}
                          </Text>
                        </Table.Summary.Cell>
                      </Table.Summary.Row>
                    </Table.Summary>
                  );
                }}
              />
            ) : (
              <Alert
                message="No Positions"
                description="You currently have no open positions"
                type="info"
                showIcon
                style={{ margin: '20px 0' }}
              />
            )}
          </Card>
        </Col>
      </Row>
    </div>
  );
};

export default PositionDisplay;