import React from 'react';
import { Card, Row, Col, Typography, Statistic } from 'antd';
import {
  DollarOutlined,
  TrophyOutlined,
  WarningOutlined,
  LineChartOutlined,
  RiseOutlined,
  FallOutlined,
  ArrowUpOutlined,
  ArrowDownOutlined
} from '@ant-design/icons';

const { Text } = Typography;

interface MetricsCardsProps {
  totalPnL: number;
  unrealizedPnL: number;
  winRate: number;
  sharpeRatio: number;
  maxDrawdown: number;
  totalTrades: number;
  winningTrades: number;
  dailyChange: number;
  weeklyChange: number;
  monthlyChange: number;
}

export const MetricsCards: React.FC<MetricsCardsProps> = ({
  totalPnL,
  unrealizedPnL,
  winRate,
  sharpeRatio,
  maxDrawdown,
  totalTrades,
  winningTrades,
  dailyChange,
  weeklyChange,
  monthlyChange
}) => {
  const getValueColor = (value: number): string => {
    if (value > 0) return '#3f8600';
    if (value < 0) return '#cf1322';
    return '#666666';
  };

  const getChangeIcon = (value: number) => {
    return value >= 0 ? <ArrowUpOutlined /> : <ArrowDownOutlined />;
  };

  const formatPnL = (value: number): string => {
    return `$${Math.abs(value).toFixed(2)}`;
  };

  const formatPercentage = (value: number): string => {
    return `${Math.abs(value).toFixed(2)}%`;
  };

  return (
    <Row gutter={[16, 16]}>
      {/* Total P&L */}
      <Col xs={24} sm={12} lg={6}>
        <Card>
          <Statistic
            title="Total P&L"
            value={totalPnL}
            precision={2}
            valueStyle={{ color: getValueColor(totalPnL) }}
            prefix={<DollarOutlined />}
            suffix={
              dailyChange !== 0 && (
                <span style={{ 
                  fontSize: '14px',
                  color: getValueColor(dailyChange),
                  marginLeft: '8px'
                }}>
                  {getChangeIcon(dailyChange)} {formatPercentage(dailyChange)}
                </span>
              )
            }
          />
          <div style={{ marginTop: 8, fontSize: '12px', color: '#666' }}>
            Daily change
          </div>
        </Card>
      </Col>

      {/* Unrealized P&L */}
      <Col xs={24} sm={12} lg={6}>
        <Card>
          <Statistic
            title="Unrealized P&L"
            value={unrealizedPnL}
            precision={2}
            valueStyle={{ color: getValueColor(unrealizedPnL) }}
            prefix={<LineChartOutlined />}
          />
          <div style={{ marginTop: 8, fontSize: '12px', color: '#666' }}>
            Open positions
          </div>
        </Card>
      </Col>

      {/* Win Rate */}
      <Col xs={24} sm={12} lg={6}>
        <Card>
          <Statistic
            title="Win Rate"
            value={winRate * 100}
            precision={1}
            valueStyle={{ color: winRate > 0.5 ? '#3f8600' : '#666666' }}
            prefix={<TrophyOutlined />}
            suffix="%"
          />
          <div style={{ marginTop: 8, fontSize: '12px', color: '#666' }}>
            {winningTrades} of {totalTrades} trades
          </div>
        </Card>
      </Col>

      {/* Sharpe Ratio */}
      <Col xs={24} sm={12} lg={6}>
        <Card>
          <Statistic
            title="Sharpe Ratio"
            value={sharpeRatio}
            precision={2}
            valueStyle={{ 
              color: sharpeRatio > 1 ? '#3f8600' : sharpeRatio > 0 ? '#fa8c16' : '#cf1322' 
            }}
            prefix={<RiseOutlined />}
          />
          <div style={{ marginTop: 8, fontSize: '12px', color: '#666' }}>
            Risk-adjusted return
          </div>
        </Card>
      </Col>

      {/* Max Drawdown */}
      <Col xs={24} sm={12} lg={6}>
        <Card>
          <Statistic
            title="Max Drawdown"
            value={maxDrawdown}
            precision={2}
            valueStyle={{ 
              color: maxDrawdown > 10 ? '#cf1322' : maxDrawdown > 5 ? '#fa8c16' : '#3f8600' 
            }}
            prefix={<WarningOutlined />}
            suffix="%"
          />
          <div style={{ marginTop: 8, fontSize: '12px', color: '#666' }}>
            Maximum loss from peak
          </div>
        </Card>
      </Col>

      {/* Weekly Performance */}
      <Col xs={24} sm={12} lg={6}>
        <Card>
          <Statistic
            title="Weekly Change"
            value={Math.abs(weeklyChange)}
            precision={2}
            valueStyle={{ color: getValueColor(weeklyChange) }}
            prefix={getChangeIcon(weeklyChange)}
            suffix="%"
          />
          <div style={{ marginTop: 8, fontSize: '12px', color: '#666' }}>
            7-day performance
          </div>
        </Card>
      </Col>

      {/* Monthly Performance */}
      <Col xs={24} sm={12} lg={6}>
        <Card>
          <Statistic
            title="Monthly Change"
            value={Math.abs(monthlyChange)}
            precision={2}
            valueStyle={{ color: getValueColor(monthlyChange) }}
            prefix={getChangeIcon(monthlyChange)}
            suffix="%"
          />
          <div style={{ marginTop: 8, fontSize: '12px', color: '#666' }}>
            30-day performance
          </div>
        </Card>
      </Col>

      {/* Trade Count */}
      <Col xs={24} sm={12} lg={6}>
        <Card>
          <Statistic
            title="Total Trades"
            value={totalTrades}
            valueStyle={{ color: '#1890ff' }}
            prefix={<LineChartOutlined />}
          />
          <div style={{ marginTop: 8, fontSize: '12px', color: '#666' }}>
            Executed orders
          </div>
        </Card>
      </Col>
    </Row>
  );
};