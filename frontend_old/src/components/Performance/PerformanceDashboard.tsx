import React, { useState, useEffect } from 'react';
import {
  Card,
  Row,
  Col,
  Typography,
  Space,
  Button,
  Spin,
  Alert,
  Select,
  DatePicker,
  notification,
  Tabs
} from 'antd';
import {
  ReloadOutlined,
  BarChartOutlined,
  TrophyOutlined,
  WarningOutlined,
  LineChartOutlined,
  DollarOutlined
} from '@ant-design/icons';
import dayjs from 'dayjs';

import { MetricsCards } from './MetricsCards';
import { StrategyComparison } from './StrategyComparison';
import { AlertSystem } from './AlertSystem';
import { RealTimeMonitor } from './RealTimeMonitor';
import { ExecutionAnalytics } from './ExecutionAnalytics';
import AdvancedAnalyticsDashboard from './AdvancedAnalyticsDashboard';
import Story5AdvancedAnalyticsDashboard from './Story5AdvancedAnalyticsDashboard';
import { DataExportDashboard } from '../Export';
import { SystemPerformanceDashboard } from '../Monitoring';
import { PerformanceMetrics, StrategyInstance } from '../Strategy/types/strategyTypes';

const { Title, Text } = Typography;
const { RangePicker } = DatePicker;
const { Option } = Select;

interface PerformanceDashboardProps {
  className?: string;
}

interface ExtendedPerformanceMetrics extends PerformanceMetrics {
  daily_pnl_change?: number;
  weekly_pnl_change?: number;
  monthly_pnl_change?: number;
  volatility?: number;
  calmar_ratio?: number;
  sortino_ratio?: number;
  max_consecutive_wins?: number;
  max_consecutive_losses?: number;
  profit_factor?: number;
  recovery_factor?: number;
  daily_returns?: number[];
  trade_history?: TradeEntry[];
}

interface TradeEntry {
  id: string;
  strategy_id: string;
  timestamp: Date;
  instrument: string;
  side: 'buy' | 'sell';
  quantity: number;
  price: number;
  pnl: number;
  commission: number;
  slippage?: number;
  execution_time_ms?: number;
}

interface PerformanceSnapshot {
  timestamp: Date;
  total_pnl: number;
  unrealized_pnl: number;
  drawdown: number;
  sharpe_ratio: number;
  win_rate: number;
}

export const PerformanceDashboard: React.FC<PerformanceDashboardProps> = ({
  className
}) => {
  const [loading, setLoading] = useState(false);
  const [activeTab, setActiveTab] = useState('overview');
  const [selectedStrategy, setSelectedStrategy] = useState<string>('all');
  const [timeRange, setTimeRange] = useState<[dayjs.Dayjs, dayjs.Dayjs]>([
    dayjs().subtract(30, 'days'),
    dayjs()
  ]);
  const [refreshInterval, setRefreshInterval] = useState<NodeJS.Timeout | null>(null);
  const [strategies, setStrategies] = useState<StrategyInstance[]>([]);
  const [performanceData, setPerformanceData] = useState<ExtendedPerformanceMetrics | null>(null);
  const [performanceHistory, setPerformanceHistory] = useState<PerformanceSnapshot[]>([]);
  const [lastUpdate, setLastUpdate] = useState<Date | null>(null);

  useEffect(() => {
    loadPerformanceData();
    startRealTimeUpdates();
    return () => stopRealTimeUpdates();
  }, [selectedStrategy, timeRange]);

  const startRealTimeUpdates = () => {
    const interval = setInterval(() => {
      loadPerformanceData();
    }, 5000); // Update every 5 seconds for real-time monitoring
    setRefreshInterval(interval);
  };

  const stopRealTimeUpdates = () => {
    if (refreshInterval) {
      clearInterval(refreshInterval);
      setRefreshInterval(null);
    }
  };

  const loadPerformanceData = async () => {
    try {
      setLoading(true);
      
      // Load active strategies
      const strategiesResponse = await fetch('/api/v1/strategies/active');
      if (strategiesResponse.ok) {
        const strategiesData = await strategiesResponse.json();
        setStrategies(strategiesData.instances || []);
      }

      // Load performance metrics for selected strategy or all strategies
      const performanceEndpoint = selectedStrategy === 'all' 
        ? '/api/v1/performance/aggregate'
        : `/api/v1/strategies/${selectedStrategy}/performance`;
      
      const performanceResponse = await fetch(
        `${performanceEndpoint}?start_date=${timeRange[0].toISOString()}&end_date=${timeRange[1].toISOString()}`
      );
      
      if (performanceResponse.ok) {
        const performance = await performanceResponse.json();
        setPerformanceData(performance);
        setLastUpdate(new Date());
      } else {
        throw new Error('Failed to load performance data');
      }

      // Load performance history for charts
      const historyResponse = await fetch(
        `/api/v1/performance/history?strategy_id=${selectedStrategy}&start_date=${timeRange[0].toISOString()}&end_date=${timeRange[1].toISOString()}`
      );
      
      if (historyResponse.ok) {
        const history = await historyResponse.json();
        setPerformanceHistory(history.snapshots || []);
      }

    } catch (error: any) {
      console.error('Failed to load performance data:', error);
      notification.error({
        message: 'Performance Data Error',
        description: error.message || 'Failed to load performance metrics',
        duration: 4
      });
    } finally {
      setLoading(false);
    }
  };

  const calculateRealTimePnL = (): number => {
    if (!performanceData) return 0;
    
    const totalPnL = Number(performanceData.total_pnl) + Number(performanceData.unrealized_pnl);
    return totalPnL;
  };

  const calculateSharpeRatio = (): number => {
    if (!performanceData || !performanceData.daily_returns || performanceData.daily_returns.length === 0) {
      return performanceData?.sharpe_ratio || 0;
    }

    const returns = performanceData.daily_returns;
    const meanReturn = returns.reduce((sum, r) => sum + r, 0) / returns.length;
    const variance = returns.reduce((sum, r) => sum + Math.pow(r - meanReturn, 2), 0) / returns.length;
    const standardDeviation = Math.sqrt(variance);
    
    // Annualized Sharpe ratio (assuming 252 trading days)
    const annualizedReturn = meanReturn * 252;
    const annualizedVolatility = standardDeviation * Math.sqrt(252);
    
    return annualizedVolatility > 0 ? annualizedReturn / annualizedVolatility : 0;
  };

  const calculateMaxDrawdown = (): { maxDrawdown: number; recovery: number } => {
    if (!performanceHistory || performanceHistory.length === 0) {
      return { 
        maxDrawdown: performanceData ? Number(performanceData.max_drawdown) : 0, 
        recovery: 0 
      };
    }

    let peak = performanceHistory[0].total_pnl;
    let maxDrawdown = 0;
    let currentDrawdown = 0;
    let recoveryDays = 0;
    let inDrawdown = false;

    for (let i = 1; i < performanceHistory.length; i++) {
      const pnl = performanceHistory[i].total_pnl;
      
      if (pnl > peak) {
        peak = pnl;
        if (inDrawdown) {
          inDrawdown = false;
          recoveryDays = 0;
        }
      } else {
        currentDrawdown = (peak - pnl) / peak;
        if (currentDrawdown > maxDrawdown) {
          maxDrawdown = currentDrawdown;
        }
        if (!inDrawdown) {
          inDrawdown = true;
        }
        recoveryDays++;
      }
    }

    return { maxDrawdown: maxDrawdown * 100, recovery: recoveryDays };
  };

  const getPerformanceColor = (value: number): string => {
    if (value > 0) return '#3f8600';
    if (value < 0) return '#cf1322';
    return '#666666';
  };

  const handleRefresh = () => {
    loadPerformanceData();
  };

  const renderOverview = () => {
    if (!performanceData) {
      return (
        <div style={{ textAlign: 'center', padding: '60px 0' }}>
          <Spin size="large" />
          <div style={{ marginTop: 16 }}>
            <Text type="secondary">Loading performance data...</Text>
          </div>
        </div>
      );
    }

    const realTimePnL = calculateRealTimePnL();
    const sharpeRatio = calculateSharpeRatio();
    const { maxDrawdown, recovery } = calculateMaxDrawdown();

    return (
      <div>
        {/* Performance Alert */}
        {maxDrawdown > 10 && (
          <Alert
            message="High Drawdown Warning"
            description={`Current maximum drawdown is ${maxDrawdown.toFixed(2)}%, which exceeds the 10% threshold.`}
            type="warning"
            icon={<WarningOutlined />}
            style={{ marginBottom: 16 }}
            showIcon
          />
        )}

        {/* Key Performance Metrics Cards */}
        <MetricsCards
          totalPnL={realTimePnL}
          unrealizedPnL={Number(performanceData.unrealized_pnl)}
          winRate={performanceData.win_rate}
          sharpeRatio={sharpeRatio}
          maxDrawdown={maxDrawdown}
          totalTrades={performanceData.total_trades}
          winningTrades={performanceData.winning_trades}
          dailyChange={performanceData.daily_pnl_change || 0}
          weeklyChange={performanceData.weekly_pnl_change || 0}
          monthlyChange={performanceData.monthly_pnl_change || 0}
        />

        {/* Performance Summary */}
        <Row gutter={[16, 16]} style={{ marginTop: 24 }}>
          <Col xs={24} lg={16}>
            <Card title="Performance Summary" extra={
              <Space>
                <Text type="secondary" style={{ fontSize: 12 }}>
                  Last updated: {lastUpdate?.toLocaleTimeString()}
                </Text>
                <Button
                  type="text"
                  icon={<ReloadOutlined />}
                  onClick={handleRefresh}
                  loading={loading}
                />
              </Space>
            }>
              <Row gutter={[16, 16]}>
                <Col span={12}>
                  <div>
                    <Text type="secondary">Total P&L</Text>
                    <div>
                      <Title 
                        level={3} 
                        style={{ 
                          margin: 0, 
                          color: getPerformanceColor(realTimePnL) 
                        }}
                      >
                        ${realTimePnL.toFixed(2)}
                      </Title>
                    </div>
                  </div>
                </Col>
                <Col span={12}>
                  <div>
                    <Text type="secondary">Sharpe Ratio</Text>
                    <div>
                      <Title 
                        level={3} 
                        style={{ 
                          margin: 0, 
                          color: sharpeRatio > 1 ? '#3f8600' : '#666666' 
                        }}
                      >
                        {sharpeRatio.toFixed(2)}
                      </Title>
                    </div>
                  </div>
                </Col>
                <Col span={12}>
                  <div>
                    <Text type="secondary">Max Drawdown</Text>
                    <div>
                      <Title 
                        level={3} 
                        style={{ 
                          margin: 0, 
                          color: maxDrawdown > 5 ? '#cf1322' : '#666666' 
                        }}
                      >
                        {maxDrawdown.toFixed(2)}%
                      </Title>
                    </div>
                  </div>
                </Col>
                <Col span={12}>
                  <div>
                    <Text type="secondary">Win Rate</Text>
                    <div>
                      <Title 
                        level={3} 
                        style={{ 
                          margin: 0, 
                          color: performanceData.win_rate > 0.5 ? '#3f8600' : '#666666' 
                        }}
                      >
                        {(performanceData.win_rate * 100).toFixed(1)}%
                      </Title>
                    </div>
                  </div>
                </Col>
              </Row>
            </Card>
          </Col>

          <Col xs={24} lg={8}>
            <Card title="Trading Statistics">
              <Space direction="vertical" style={{ width: '100%' }}>
                <div>
                  <Text type="secondary">Total Trades</Text>
                  <div><Text strong>{performanceData.total_trades}</Text></div>
                </div>
                <div>
                  <Text type="secondary">Winning Trades</Text>
                  <div><Text strong>{performanceData.winning_trades}</Text></div>
                </div>
                <div>
                  <Text type="secondary">Losing Trades</Text>
                  <div><Text strong>{performanceData.total_trades - performanceData.winning_trades}</Text></div>
                </div>
                {performanceData.profit_factor && (
                  <div>
                    <Text type="secondary">Profit Factor</Text>
                    <div><Text strong>{performanceData.profit_factor.toFixed(2)}</Text></div>
                  </div>
                )}
                {recovery > 0 && (
                  <div>
                    <Text type="secondary">Recovery Period</Text>
                    <div><Text strong>{recovery} days</Text></div>
                  </div>
                )}
              </Space>
            </Card>
          </Col>
        </Row>
      </div>
    );
  };

  return (
    <div className={`performance-dashboard ${className || ''}`}>
      <Card>
        <div style={{ marginBottom: 24 }}>
          <Row justify="space-between" align="middle">
            <Col>
              <Title level={2} style={{ margin: 0 }}>
                <BarChartOutlined style={{ marginRight: 8 }} />
                Performance Dashboard
              </Title>
              <Text type="secondary">
                Real-time strategy performance monitoring and analytics
              </Text>
            </Col>
            <Col>
              <Space>
                <Select
                  value={selectedStrategy}
                  onChange={setSelectedStrategy}
                  style={{ width: 200 }}
                  placeholder="Select Strategy"
                >
                  <Option value="all">All Strategies</Option>
                  {strategies.map(strategy => (
                    <Option key={strategy.id} value={strategy.id}>
                      {strategy.id}
                    </Option>
                  ))}
                </Select>
                
                <RangePicker
                  value={timeRange}
                  onChange={(dates) => dates && setTimeRange(dates)}
                  style={{ width: 240 }}
                />
                
                <Button
                  type="primary"
                  icon={<ReloadOutlined />}
                  onClick={handleRefresh}
                  loading={loading}
                >
                  Refresh
                </Button>
              </Space>
            </Col>
          </Row>
        </div>

        <Tabs 
          activeKey={activeTab} 
          onChange={setActiveTab}
          items={[
            {
              key: 'overview',
              label: 'Overview',
              children: renderOverview()
            },
            {
              key: 'monitor',
              label: 'Real-Time Monitor',
              children: <RealTimeMonitor />
            },
            {
              key: 'comparison',
              label: 'Strategy Comparison',
              children: (
                <StrategyComparison 
                  strategies={strategies}
                  timeRange={timeRange}
                />
              )
            },
            {
              key: 'execution',
              label: 'Execution Analytics',
              children: <ExecutionAnalytics />
            },
            {
              key: 'alerts',
              label: 'Alert System',
              children: (
                <AlertSystem 
                  strategies={strategies}
                  performanceData={performanceData}
                />
              )
            },
            {
              key: 'advanced',
              label: 'Advanced Analytics',
              children: <AdvancedAnalyticsDashboard />
            },
            {
              key: 'story5-analytics',
              label: 'Story 5.1 Analytics',
              children: <Story5AdvancedAnalyticsDashboard />
            },
            {
              key: 'system-monitoring',
              label: 'System Monitoring',
              children: <SystemPerformanceDashboard />
            },
            {
              key: 'export',
              label: 'Data Export',
              children: <DataExportDashboard />
            }
          ]}
        />
      </Card>
    </div>
  );
};