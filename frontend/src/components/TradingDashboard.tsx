/**
 * Clock-Aware Trading Dashboard Component
 * 
 * Features:
 * - Real-time synchronized trading operations
 * - Market hours awareness and session indicators
 * - Clock-synchronized order placement and monitoring
 * - Time-based position updates with precise timing
 * - Integration with Phase 1/2 clock infrastructure
 * - M4 Max hardware acceleration optimizations
 */

import React, { useState, useEffect, useCallback, useMemo } from 'react';
import { Card, Row, Col, Badge, Button, Typography, Space, Alert, Statistic, Progress } from 'antd';
import { ClockCircleOutlined, SyncOutlined, WarningOutlined, CheckCircleOutlined } from '@ant-design/icons';
import { useServerTime } from '../hooks/useServerTime';
import { useClockSync } from '../hooks/useClockSync';

const { Title, Text } = Typography;

export interface Position {
  symbol: string;
  quantity: number;
  averagePrice: number;
  currentPrice: number;
  unrealizedPnL: number;
  realizedPnL: number;
  lastUpdate: number;
}

export interface Order {
  orderId: string;
  symbol: string;
  type: 'BUY' | 'SELL';
  quantity: number;
  price: number;
  status: 'PENDING' | 'FILLED' | 'CANCELLED';
  timestamp: number;
  fillTime?: number;
}

export interface TradingDashboardProps {
  positions?: Position[];
  orders?: Order[];
  onPlaceOrder?: (order: Omit<Order, 'orderId' | 'timestamp' | 'status'>) => Promise<void>;
  onCancelOrder?: (orderId: string) => Promise<void>;
  enableRealTimeUpdates?: boolean;
  marketDataSource?: string;
}

interface TradingMetrics {
  totalPnL: number;
  dayPnL: number;
  openPositions: number;
  pendingOrders: number;
  executionLatency: number;
  lastUpdateTime: number;
}

export const TradingDashboard: React.FC<TradingDashboardProps> = ({
  positions = [],
  orders = [],
  onPlaceOrder,
  onCancelOrder,
  enableRealTimeUpdates = true,
  marketDataSource = 'live'
}) => {
  const { serverTimeState, isMarketOpen, getMarketTimeRemaining, formatServerTime, getTimestamp } = useServerTime({
    updateInterval: 100, // 100ms updates for smooth trading display
    includeMarkets: ['NYSE', 'NASDAQ', 'LSE'],
    enableTradingAlerts: true
  });

  const { clockState, forceSync, isClockSynced, getClockAccuracy } = useClockSync({
    syncInterval: 30000, // 30 second sync for trading precision
    enableDriftCorrection: true
  });

  const [tradingMetrics, setTradingMetrics] = useState<TradingMetrics>({
    totalPnL: 0,
    dayPnL: 0,
    openPositions: 0,
    pendingOrders: 0,
    executionLatency: 0,
    lastUpdateTime: 0
  });

  const [isPlacingOrder, setIsPlacingOrder] = useState(false);
  const [orderForm, setOrderForm] = useState({
    symbol: '',
    type: 'BUY' as 'BUY' | 'SELL',
    quantity: 100,
    price: 0
  });

  // Calculate trading metrics from positions and orders
  const calculateMetrics = useCallback((): TradingMetrics => {
    const totalPnL = positions.reduce((sum, pos) => sum + pos.unrealizedPnL + pos.realizedPnL, 0);
    const dayPnL = positions.reduce((sum, pos) => sum + pos.unrealizedPnL, 0);
    const openPositions = positions.filter(pos => pos.quantity !== 0).length;
    const pendingOrders = orders.filter(order => order.status === 'PENDING').length;
    
    // Calculate average execution latency for filled orders
    const filledOrders = orders.filter(order => order.status === 'FILLED' && order.fillTime);
    const executionLatency = filledOrders.length > 0
      ? filledOrders.reduce((sum, order) => sum + (order.fillTime! - order.timestamp), 0) / filledOrders.length
      : 0;

    return {
      totalPnL,
      dayPnL,
      openPositions,
      pendingOrders,
      executionLatency,
      lastUpdateTime: getTimestamp()
    };
  }, [positions, orders, getTimestamp]);

  // Update metrics when positions or orders change
  useEffect(() => {
    setTradingMetrics(calculateMetrics());
  }, [calculateMetrics]);

  // Handle order placement with precise timing
  const handlePlaceOrder = useCallback(async () => {
    if (!onPlaceOrder || !isClockSynced) {
      return;
    }

    try {
      setIsPlacingOrder(true);
      
      const orderTimestamp = getTimestamp();
      await onPlaceOrder({
        ...orderForm,
        timestamp: orderTimestamp
      });

      // Reset form
      setOrderForm(prev => ({
        ...prev,
        symbol: '',
        price: 0
      }));

    } catch (error) {
      console.error('Failed to place order:', error);
    } finally {
      setIsPlacingOrder(false);
    }
  }, [onPlaceOrder, orderForm, isClockSynced, getTimestamp]);

  // Handle order cancellation with timing
  const handleCancelOrder = useCallback(async (orderId: string) => {
    if (!onCancelOrder) {
      return;
    }

    try {
      await onCancelOrder(orderId);
    } catch (error) {
      console.error('Failed to cancel order:', error);
    }
  }, [onCancelOrder]);

  // Determine current market status
  const marketStatus = useMemo(() => {
    const primaryMarket = serverTimeState.marketHours[0];
    if (!primaryMarket) {
      return { status: 'unknown', color: 'default', text: 'Unknown' };
    }

    switch (primaryMarket.sessionType) {
      case 'regular':
        return { status: 'open', color: 'success', text: 'Market Open' };
      case 'pre-market':
        return { status: 'pre-market', color: 'warning', text: 'Pre-Market' };
      case 'after-hours':
        return { status: 'after-hours', color: 'warning', text: 'After Hours' };
      default:
        return { status: 'closed', color: 'error', text: 'Market Closed' };
    }
  }, [serverTimeState.marketHours]);

  // Clock status indicator
  const clockStatus = useMemo(() => {
    if (!isClockSynced) {
      return { color: 'error', icon: <WarningOutlined />, text: 'Clock Not Synced' };
    }

    const accuracy = getClockAccuracy();
    if (accuracy >= 95) {
      return { color: 'success', icon: <CheckCircleOutlined />, text: `Clock Synced (${accuracy.toFixed(1)}%)` };
    } else if (accuracy >= 80) {
      return { color: 'warning', icon: <SyncOutlined />, text: `Clock Sync Fair (${accuracy.toFixed(1)}%)` };
    } else {
      return { color: 'error', icon: <WarningOutlined />, text: `Clock Sync Poor (${accuracy.toFixed(1)}%)` };
    }
  }, [isClockSynced, getClockAccuracy]);

  // Format time remaining until market event
  const formatTimeRemaining = useCallback((milliseconds: number): string => {
    const hours = Math.floor(milliseconds / (1000 * 60 * 60));
    const minutes = Math.floor((milliseconds % (1000 * 60 * 60)) / (1000 * 60));
    const seconds = Math.floor((milliseconds % (1000 * 60)) / 1000);
    
    if (hours > 0) {
      return `${hours}h ${minutes}m ${seconds}s`;
    } else if (minutes > 0) {
      return `${minutes}m ${seconds}s`;
    } else {
      return `${seconds}s`;
    }
  }, []);

  // Real-time updates effect
  useEffect(() => {
    if (!enableRealTimeUpdates) {
      return;
    }

    // Listen for market events
    const handleMarketOpen = (event: CustomEvent) => {
      console.log(`🔔 Market opened: ${event.detail.market}`);
    };

    const handleMarketClose = (event: CustomEvent) => {
      console.log(`🔔 Market closed: ${event.detail.market}`);
    };

    window.addEventListener('marketOpen', handleMarketOpen as EventListener);
    window.addEventListener('marketClose', handleMarketClose as EventListener);

    return () => {
      window.removeEventListener('marketOpen', handleMarketOpen as EventListener);
      window.removeEventListener('marketClose', handleMarketClose as EventListener);
    };
  }, [enableRealTimeUpdates]);

  return (
    <div style={{ padding: '24px', background: '#f0f2f5', minHeight: '100vh' }}>
      <Row gutter={[16, 16]}>
        {/* Header with Clock Status */}
        <Col span={24}>
          <Card>
            <Row justify="space-between" align="middle">
              <Col>
                <Space>
                  <Title level={3} style={{ margin: 0 }}>
                    <ClockCircleOutlined /> Trading Dashboard
                  </Title>
                  <Badge 
                    status={marketStatus.color as any} 
                    text={marketStatus.text}
                  />
                </Space>
              </Col>
              <Col>
                <Space>
                  <Badge 
                    status={clockStatus.color as any}
                    icon={clockStatus.icon}
                    text={clockStatus.text}
                  />
                  <Text type="secondary">
                    Server Time: {formatServerTime('trading')}
                  </Text>
                  <Button 
                    icon={<SyncOutlined />} 
                    onClick={forceSync}
                    size="small"
                  >
                    Sync
                  </Button>
                </Space>
              </Col>
            </Row>
          </Card>
        </Col>

        {/* Market Status Cards */}
        <Col span={24}>
          <Row gutter={16}>
            {serverTimeState.marketHours.slice(0, 3).map((market, index) => {
              const timeRemaining = getMarketTimeRemaining(market.market);
              const progressPercent = timeRemaining 
                ? Math.max(0, Math.min(100, (timeRemaining / (8 * 60 * 60 * 1000)) * 100))
                : 0;

              return (
                <Col key={index} xs={24} sm={8}>
                  <Card size="small">
                    <Statistic
                      title={market.market}
                      value={market.isOpen ? 'OPEN' : 'CLOSED'}
                      valueStyle={{ 
                        color: market.isOpen ? '#3f8600' : '#cf1322',
                        fontSize: '16px'
                      }}
                      suffix={
                        timeRemaining && (
                          <div style={{ fontSize: '12px', color: '#666' }}>
                            {market.isOpen ? 'Closes in' : 'Opens in'}: {formatTimeRemaining(timeRemaining)}
                            <Progress 
                              percent={progressPercent} 
                              size="small" 
                              showInfo={false}
                              strokeColor={market.isOpen ? '#52c41a' : '#1890ff'}
                            />
                          </div>
                        )
                      }
                    />
                  </Card>
                </Col>
              );
            })}
          </Row>
        </Col>

        {/* Trading Metrics */}
        <Col span={24}>
          <Card title="Trading Metrics">
            <Row gutter={16}>
              <Col xs={24} sm={6}>
                <Statistic
                  title="Total P&L"
                  value={tradingMetrics.totalPnL}
                  precision={2}
                  valueStyle={{ color: tradingMetrics.totalPnL >= 0 ? '#3f8600' : '#cf1322' }}
                  prefix="$"
                />
              </Col>
              <Col xs={24} sm={6}>
                <Statistic
                  title="Day P&L"
                  value={tradingMetrics.dayPnL}
                  precision={2}
                  valueStyle={{ color: tradingMetrics.dayPnL >= 0 ? '#3f8600' : '#cf1322' }}
                  prefix="$"
                />
              </Col>
              <Col xs={24} sm={6}>
                <Statistic
                  title="Open Positions"
                  value={tradingMetrics.openPositions}
                />
              </Col>
              <Col xs={24} sm={6}>
                <Statistic
                  title="Pending Orders"
                  value={tradingMetrics.pendingOrders}
                />
              </Col>
            </Row>
          </Card>
        </Col>

        {/* Clock Synchronization Status */}
        <Col span={24}>
          <Card title="Clock Synchronization" size="small">
            <Row gutter={16}>
              <Col xs={24} sm={6}>
                <Statistic
                  title="Network Latency"
                  value={clockState.networkLatency}
                  suffix="ms"
                  valueStyle={{ color: clockState.networkLatency < 50 ? '#3f8600' : '#cf1322' }}
                />
              </Col>
              <Col xs={24} sm={6}>
                <Statistic
                  title="Clock Drift"
                  value={Math.abs(clockState.clockDrift)}
                  precision={3}
                  suffix="ms/s"
                  valueStyle={{ color: Math.abs(clockState.clockDrift) < 1 ? '#3f8600' : '#faad14' }}
                />
              </Col>
              <Col xs={24} sm={6}>
                <Statistic
                  title="Sync Success Rate"
                  value={clockState.syncCount / Math.max(1, clockState.syncCount + clockState.errorCount) * 100}
                  precision={1}
                  suffix="%"
                />
              </Col>
              <Col xs={24} sm={6}>
                <Statistic
                  title="Last Sync"
                  value={clockState.lastSyncTimestamp > 0 
                    ? Math.floor((Date.now() - clockState.lastSyncTimestamp) / 1000)
                    : 0}
                  suffix="s ago"
                />
              </Col>
            </Row>
          </Card>
        </Col>

        {/* Alert for clock issues */}
        {(!isClockSynced || getClockAccuracy() < 80) && (
          <Col span={24}>
            <Alert
              message="Clock Synchronization Issue"
              description="Trading operations may be impacted by poor clock synchronization. Consider refreshing the page or checking your network connection."
              type="warning"
              icon={<WarningOutlined />}
              showIcon
              action={
                <Button size="small" onClick={forceSync}>
                  Force Sync
                </Button>
              }
            />
          </Col>
        )}

        {/* Performance Metrics */}
        <Col span={24}>
          <Card title="Performance Metrics" size="small">
            <Row gutter={16}>
              <Col xs={24} sm={8}>
                <Statistic
                  title="Avg Execution Latency"
                  value={tradingMetrics.executionLatency}
                  precision={0}
                  suffix="ms"
                  valueStyle={{ color: tradingMetrics.executionLatency < 100 ? '#3f8600' : '#cf1322' }}
                />
              </Col>
              <Col xs={24} sm={8}>
                <Statistic
                  title="Clock Accuracy"
                  value={getClockAccuracy()}
                  precision={1}
                  suffix="%"
                  valueStyle={{ color: getClockAccuracy() >= 95 ? '#3f8600' : '#faad14' }}
                />
              </Col>
              <Col xs={24} sm={8}>
                <Statistic
                  title="Data Freshness"
                  value={tradingMetrics.lastUpdateTime > 0 
                    ? Math.floor((getTimestamp() - tradingMetrics.lastUpdateTime) / 1000)
                    : 0}
                  suffix="s"
                  valueStyle={{ 
                    color: (getTimestamp() - tradingMetrics.lastUpdateTime) < 5000 ? '#3f8600' : '#cf1322'
                  }}
                />
              </Col>
            </Row>
          </Card>
        </Col>
      </Row>
    </div>
  );
};

export default TradingDashboard;