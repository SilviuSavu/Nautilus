import React, { useState, useEffect } from 'react';
import {
  Card,
  Row,
  Col,
  Typography,
  Table,
  Tag,
  Progress,
  Statistic,
  Space,
  Button,
  Badge,
  Tooltip,
  Alert,
  Timeline,
  Spin
} from 'antd';
import {
  PlayCircleOutlined,
  PauseCircleOutlined,
  StopOutlined,
  ExclamationCircleOutlined,
  CheckCircleOutlined,
  ClockCircleOutlined,
  LineChartOutlined,
  FireOutlined,
  HeartOutlined,
  EyeOutlined
} from '@ant-design/icons';
import type { ColumnsType } from 'antd/es/table';

import { StrategyInstance, StrategyState } from '../Strategy/types/strategyTypes';

const { Title, Text } = Typography;

interface RealTimeMonitorProps {
  className?: string;
}

interface StrategyMonitorData extends StrategyInstance {
  health_score: number;
  connection_status: 'connected' | 'disconnected' | 'reconnecting';
  last_signal: string;
  last_signal_time: Date;
  active_positions: number;
  pending_orders: number;
  recent_trades: number;
  latency_ms: number;
  cpu_usage: number;
  memory_usage: number;
  error_rate: number;
  uptime_hours: number;
}

interface SignalData {
  id: string;
  strategy_id: string;
  signal_type: 'buy' | 'sell' | 'hold' | 'close';
  instrument: string;
  confidence: number;
  generated_at: Date;
  executed: boolean;
  execution_time?: Date;
  execution_price?: number;
  reasoning?: string;
}

interface PositionData {
  id: string;
  strategy_id: string;
  instrument: string;
  side: 'long' | 'short';
  quantity: number;
  entry_price: number;
  current_price: number;
  unrealized_pnl: number;
  duration_hours: number;
  risk_percentage: number;
}

export const RealTimeMonitor: React.FC<RealTimeMonitorProps> = ({ className }) => {
  const [loading, setLoading] = useState(false);
  const [strategies, setStrategies] = useState<StrategyMonitorData[]>([]);
  const [signals, setSignals] = useState<SignalData[]>([]);
  const [positions, setPositions] = useState<PositionData[]>([]);
  const [selectedStrategy, setSelectedStrategy] = useState<string | null>(null);

  useEffect(() => {
    loadMonitoringData();
    
    // Real-time updates every 2 seconds
    const interval = setInterval(() => {
      loadMonitoringData();
    }, 2000);
    
    return () => clearInterval(interval);
  }, []);

  const loadMonitoringData = async () => {
    try {
      setLoading(true);
      
      // Load strategy monitoring data
      const strategiesResponse = await fetch('/api/v1/strategies/monitoring');
      if (strategiesResponse.ok) {
        const data = await strategiesResponse.json();
        setStrategies(data.strategies || []);
      }

      // Load recent signals
      const signalsResponse = await fetch('/api/v1/strategies/signals/recent?limit=50');
      if (signalsResponse.ok) {
        const data = await signalsResponse.json();
        setSignals(data.signals || []);
      }

      // Load active positions
      const positionsResponse = await fetch('/api/v1/strategies/positions/active');
      if (positionsResponse.ok) {
        const data = await positionsResponse.json();
        setPositions(data.positions || []);
      }
    } catch (error: any) {
      console.error('Failed to load monitoring data:', error);
    } finally {
      setLoading(false);
    }
  };

  const getStateColor = (state: StrategyState): string => {
    switch (state) {
      case 'running': return 'success';
      case 'paused': return 'warning';
      case 'stopped': return 'default';
      case 'error': return 'error';
      case 'initializing': return 'processing';
      case 'stopping': return 'warning';
      case 'completed': return 'purple';
      default: return 'default';
    }
  };

  const getStateIcon = (state: StrategyState) => {
    switch (state) {
      case 'running': return <PlayCircleOutlined style={{ color: '#52c41a' }} />;
      case 'paused': return <PauseCircleOutlined style={{ color: '#fa8c16' }} />;
      case 'stopped': return <StopOutlined style={{ color: '#8c8c8c' }} />;
      case 'error': return <ExclamationCircleOutlined style={{ color: '#ff4d4f' }} />;
      case 'initializing': return <ClockCircleOutlined style={{ color: '#1890ff' }} />;
      default: return <ClockCircleOutlined />;
    }
  };

  const getConnectionColor = (status: string): string => {
    switch (status) {
      case 'connected': return '#52c41a';
      case 'disconnected': return '#ff4d4f';
      case 'reconnecting': return '#fa8c16';
      default: return '#8c8c8c';
    }
  };

  const getHealthColor = (score: number): string => {
    if (score >= 90) return '#52c41a';
    if (score >= 70) return '#fa8c16';
    return '#ff4d4f';
  };

  const formatUptime = (hours: number): string => {
    if (hours < 1) return `${Math.round(hours * 60)}m`;
    if (hours < 24) return `${Math.round(hours)}h`;
    return `${Math.round(hours / 24)}d`;
  };

  const strategyColumns: ColumnsType<StrategyMonitorData> = [
    {
      title: 'Strategy',
      key: 'strategy',
      render: (_, record) => (
        <div>
          <Space>
            {getStateIcon(record.state)}
            <Text strong>{record.id}</Text>
          </Space>
          <br />
          <Tag color={getStateColor(record.state)} size="small">
            {record.state.toUpperCase()}
          </Tag>
        </div>
      )
    },
    {
      title: 'Health',
      dataIndex: 'health_score',
      key: 'health_score',
      render: (score: number) => (
        <div>
          <Progress
            type="circle"
            size="small"
            percent={score}
            strokeColor={getHealthColor(score)}
            format={() => 
              <HeartOutlined style={{ color: getHealthColor(score) }} />
            }
          />
          <div style={{ fontSize: 12, textAlign: 'center', marginTop: 4 }}>
            {score}%
          </div>
        </div>
      )
    },
    {
      title: 'Connection',
      dataIndex: 'connection_status',
      key: 'connection_status',
      render: (status: string, record) => (
        <div>
          <Badge
            color={getConnectionColor(status)}
            text={status.charAt(0).toUpperCase() + status.slice(1)}
          />
          <br />
          <Text type="secondary" style={{ fontSize: 12 }}>
            Latency: {record.latency_ms}ms
          </Text>
        </div>
      )
    },
    {
      title: 'Performance',
      key: 'performance',
      render: (_, record) => {
        const pnl = Number(record.performance_metrics.total_pnl);
        return (
          <div>
            <Text style={{ color: pnl >= 0 ? '#52c41a' : '#ff4d4f' }} strong>
              ${pnl.toFixed(2)}
            </Text>
            <br />
            <Text type="secondary" style={{ fontSize: 12 }}>
              {record.performance_metrics.total_trades} trades
            </Text>
          </div>
        );
      }
    },
    {
      title: 'Activity',
      key: 'activity',
      render: (_, record) => (
        <Space direction="vertical" size="small">
          <div>
            <Text type="secondary">Positions:</Text> {record.active_positions}
          </div>
          <div>
            <Text type="secondary">Orders:</Text> {record.pending_orders}
          </div>
          <div>
            <Text type="secondary">Recent:</Text> {record.recent_trades}
          </div>
        </Space>
      )
    },
    {
      title: 'Last Signal',
      key: 'last_signal',
      render: (_, record) => {
        if (!record.last_signal) {
          return <Text type="secondary">No signals</Text>;
        }
        
        const timeDiff = Date.now() - new Date(record.last_signal_time).getTime();
        const minutesAgo = Math.floor(timeDiff / (1000 * 60));
        
        return (
          <div>
            <Tag 
              color={
                record.last_signal === 'buy' ? 'green' : 
                record.last_signal === 'sell' ? 'red' : 'blue'
              }
              size="small"
            >
              {record.last_signal.toUpperCase()}
            </Tag>
            <br />
            <Text type="secondary" style={{ fontSize: 12 }}>
              {minutesAgo}m ago
            </Text>
          </div>
        );
      }
    },
    {
      title: 'Resources',
      key: 'resources',
      render: (_, record) => (
        <Space direction="vertical" size="small">
          <div>
            <Text type="secondary">CPU:</Text> {record.cpu_usage}%
          </div>
          <div>
            <Text type="secondary">Memory:</Text> {record.memory_usage}%
          </div>
          <div>
            <Text type="secondary">Uptime:</Text> {formatUptime(record.uptime_hours)}
          </div>
        </Space>
      )
    },
    {
      title: 'Actions',
      key: 'actions',
      render: (_, record) => (
        <Button
          type="text"
          icon={<EyeOutlined />}
          onClick={() => setSelectedStrategy(record.id)}
        >
          Details
        </Button>
      )
    }
  ];

  const getSignalIcon = (type: string, executed: boolean) => {
    const baseIcon = type === 'buy' ? '🟢' : type === 'sell' ? '🔴' : '🔵';
    return executed ? `${baseIcon}✅` : `${baseIcon}⏳`;
  };

  const renderRecentSignals = () => {
    const recentSignals = signals.slice(0, 10);
    
    return (
      <Timeline
        items={recentSignals.map(signal => ({
          dot: getSignalIcon(signal.signal_type, signal.executed),
          color: signal.executed ? 'green' : 'blue',
          children: (
            <div>
              <Text strong>
                {signal.signal_type.toUpperCase()} {signal.instrument}
              </Text>
              <br />
              <Text type="secondary" style={{ fontSize: 12 }}>
                Strategy: {signal.strategy_id} | 
                Confidence: {(signal.confidence * 100).toFixed(0)}% | 
                {new Date(signal.generated_at).toLocaleTimeString()}
              </Text>
              {signal.reasoning && (
                <>
                  <br />
                  <Text type="secondary" style={{ fontSize: 12 }}>
                    {signal.reasoning}
                  </Text>
                </>
              )}
            </div>
          )
        }))}
      />
    );
  };

  const renderActivePositions = () => {
    return (
      <Table
        dataSource={positions}
        rowKey="id"
        size="small"
        pagination={false}
        columns={[
          {
            title: 'Instrument',
            dataIndex: 'instrument',
            key: 'instrument'
          },
          {
            title: 'Side',
            dataIndex: 'side',
            key: 'side',
            render: (side: string) => (
              <Tag color={side === 'long' ? 'green' : 'red'}>
                {side.toUpperCase()}
              </Tag>
            )
          },
          {
            title: 'Size',
            dataIndex: 'quantity',
            key: 'quantity',
            render: (qty: number) => qty.toLocaleString()
          },
          {
            title: 'Entry',
            dataIndex: 'entry_price',
            key: 'entry_price',
            render: (price: number) => `$${price.toFixed(2)}`
          },
          {
            title: 'Current',
            dataIndex: 'current_price',
            key: 'current_price',
            render: (price: number) => `$${price.toFixed(2)}`
          },
          {
            title: 'P&L',
            dataIndex: 'unrealized_pnl',
            key: 'unrealized_pnl',
            render: (pnl: number) => (
              <Text style={{ color: pnl >= 0 ? '#52c41a' : '#ff4d4f' }}>
                ${pnl.toFixed(2)}
              </Text>
            )
          },
          {
            title: 'Duration',
            dataIndex: 'duration_hours',
            key: 'duration_hours',
            render: (hours: number) => formatUptime(hours)
          }
        ]}
      />
    );
  };

  return (
    <div className={`real-time-monitor ${className || ''}`}>
      <Row gutter={[16, 16]}>
        {/* Header Stats */}
        <Col span={24}>
          <Row gutter={16}>
            <Col xs={6}>
              <Card>
                <Statistic
                  title="Active Strategies"
                  value={strategies.filter(s => s.state === 'running').length}
                  suffix={`/ ${strategies.length}`}
                  valueStyle={{ color: '#52c41a' }}
                  prefix={<PlayCircleOutlined />}
                />
              </Card>
            </Col>
            <Col xs={6}>
              <Card>
                <Statistic
                  title="Active Positions"
                  value={positions.length}
                  valueStyle={{ color: '#1890ff' }}
                  prefix={<LineChartOutlined />}
                />
              </Card>
            </Col>
            <Col xs={6}>
              <Card>
                <Statistic
                  title="Recent Signals"
                  value={signals.filter(s => 
                    Date.now() - new Date(s.generated_at).getTime() < 3600000
                  ).length}
                  suffix="/ 1h"
                  valueStyle={{ color: '#fa8c16' }}
                  prefix={<FireOutlined />}
                />
              </Card>
            </Col>
            <Col xs={6}>
              <Card>
                <Statistic
                  title="Avg Health"
                  value={Math.round(
                    strategies.reduce((sum, s) => sum + s.health_score, 0) / strategies.length || 0
                  )}
                  suffix="%"
                  valueStyle={{ 
                    color: strategies.length > 0 ? 
                      getHealthColor(strategies.reduce((sum, s) => sum + s.health_score, 0) / strategies.length) :
                      '#8c8c8c'
                  }}
                  prefix={<HeartOutlined />}
                />
              </Card>
            </Col>
          </Row>
        </Col>

        {/* Strategy Monitor Table */}
        <Col span={24}>
          <Card 
            title={
              <Space>
                <Text strong>Strategy Monitor</Text>
                <Badge count={strategies.filter(s => s.state === 'error').length} />
                {loading && <Spin size="small" />}
              </Space>
            }
            extra={
              <Text type="secondary" style={{ fontSize: 12 }}>
                Updates every 2 seconds
              </Text>
            }
          >
            <Table
              columns={strategyColumns}
              dataSource={strategies}
              rowKey="id"
              pagination={false}
              size="middle"
              scroll={{ x: 1000 }}
            />
          </Card>
        </Col>

        {/* Signal Activity and Positions */}
        <Col xs={24} lg={12}>
          <Card title="Recent Signal Activity" size="small">
            {signals.length > 0 ? (
              renderRecentSignals()
            ) : (
              <Text type="secondary">No recent signals</Text>
            )}
          </Card>
        </Col>

        <Col xs={24} lg={12}>
          <Card title="Active Positions" size="small">
            {positions.length > 0 ? (
              renderActivePositions()
            ) : (
              <Text type="secondary">No active positions</Text>
            )}
          </Card>
        </Col>
      </Row>
    </div>
  );
};