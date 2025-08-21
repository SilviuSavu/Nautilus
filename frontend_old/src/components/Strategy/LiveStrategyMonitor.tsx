import React, { useState, useEffect, useRef } from 'react';
import {
  Card,
  Row,
  Col,
  Statistic,
  Progress,
  Badge,
  Alert,
  Button,
  Space,
  Tooltip,
  Table,
  Typography,
  Divider,
  Tag,
  Avatar,
  Modal,
  Form,
  Input,
  Switch,
  notification
} from 'antd';
import {
  MonitorOutlined,
  PlayCircleOutlined,
  PauseCircleOutlined,
  StopOutlined,
  WarningOutlined,
  CheckCircleOutlined,
  ExclamationCircleOutlined,
  ReloadOutlined,
  LineChartOutlined,
  DollarOutlined,
  ArrowUpOutlined,
  ArrowDownOutlined,
  AlertOutlined
} from '@ant-design/icons';
import type {
  LiveStrategy,
  LiveStrategyState,
  Position,
  StrategyAlert,
  ControlStrategyRequest,
  EmergencyAction,
  LiveMonitorProps
} from '../../types/deployment';

const { Text, Title } = Typography;

const LiveStrategyMonitor: React.FC<LiveMonitorProps> = ({
  strategyInstanceId,
  compact = false,
  showAlerts = true,
  refreshInterval = 5000
}) => {
  const [strategy, setStrategy] = useState<LiveStrategy | null>(null);
  const [loading, setLoading] = useState(true);
  const [controlling, setControlling] = useState(false);
  const [emergencyModalVisible, setEmergencyModalVisible] = useState(false);
  const [controlForm] = Form.useForm();
  const intervalRef = useRef<NodeJS.Timeout>();

  useEffect(() => {
    if (strategyInstanceId) {
      loadStrategyData();
      startRealTimeUpdates();
    }

    return () => {
      if (intervalRef.current) {
        clearInterval(intervalRef.current);
      }
    };
  }, [strategyInstanceId, refreshInterval]);

  const loadStrategyData = async () => {
    if (!strategyInstanceId) return;

    try {
      setLoading(true);
      const response = await fetch(`/api/v1/nautilus/strategies/live/${strategyInstanceId}`);
      const data = await response.json();
      setStrategy(data);
    } catch (error) {
      console.error('Error loading strategy data:', error);
      notification.error({
        message: 'Load Error',
        description: 'Failed to load strategy data'
      });
    } finally {
      setLoading(false);
    }
  };

  const startRealTimeUpdates = () => {
    if (intervalRef.current) {
      clearInterval(intervalRef.current);
    }

    intervalRef.current = setInterval(async () => {
      if (!strategyInstanceId) return;

      try {
        const response = await fetch(`/api/v1/nautilus/strategies/live/${strategyInstanceId}/metrics`);
        const data = await response.json();
        
        setStrategy(prev => prev ? {
          ...prev,
          performanceMetrics: data.performanceMetrics,
          riskMetrics: data.riskMetrics,
          currentPosition: data.positions[0] || prev.currentPosition,
          alerts: data.alerts,
          lastHeartbeat: new Date(data.timestamp),
          healthStatus: data.healthStatus
        } : null);
      } catch (error) {
        console.error('Error updating strategy metrics:', error);
      }
    }, refreshInterval);
  };

  const controlStrategy = async (action: string, reason: string, force = false) => {
    if (!strategyInstanceId) return;

    setControlling(true);
    try {
      const request: ControlStrategyRequest = {
        action: action as any,
        reason,
        force
      };

      const response = await fetch(`/api/v1/nautilus/strategies/live/${strategyInstanceId}/control`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(request)
      });

      const result = await response.json();
      
      if (result.success) {
        notification.success({
          message: 'Control Action Successful',
          description: `Strategy ${action} completed successfully`
        });
        setStrategy(prev => prev ? { ...prev, state: result.newState } : null);
      } else {
        notification.error({
          message: 'Control Action Failed',
          description: result.message || 'Unknown error occurred'
        });
      }
    } catch (error) {
      console.error('Control action failed:', error);
      notification.error({
        message: 'Control Error',
        description: 'Failed to execute control action'
      });
    } finally {
      setControlling(false);
    }
  };

  const handleEmergencyStop = () => {
    Modal.confirm({
      title: '🚨 EMERGENCY STOP',
      icon: <ExclamationCircleOutlined />,
      content: (
        <div>
          <p><strong>This will immediately stop the strategy and close all positions.</strong></p>
          <p>This action cannot be undone. Are you sure you want to proceed?</p>
        </div>
      ),
      okText: 'EMERGENCY STOP',
      okType: 'danger',
      cancelText: 'Cancel',
      onOk: () => {
        controlStrategy('emergency_stop', 'Manual emergency stop triggered', true);
      }
    });
  };

  const getStateColor = (state: LiveStrategyState): string => {
    switch (state) {
      case 'running': return 'green';
      case 'paused': return 'orange';
      case 'stopped': return 'red';
      case 'error': return 'red';
      case 'emergency_stopped': return 'red';
      case 'deploying': return 'blue';
      default: return 'gray';
    }
  };

  const getHealthColor = (health: string): string => {
    switch (health) {
      case 'healthy': return 'green';
      case 'warning': return 'orange';
      case 'critical': return 'red';
      default: return 'gray';
    }
  };

  const formatCurrency = (value: number): string => {
    return new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency: 'USD'
    }).format(value);
  };

  const formatPercentage = (value: number): string => {
    return `${(value * 100).toFixed(2)}%`;
  };

  if (!strategy) {
    return (
      <Card loading={loading}>
        <div className="text-center py-8">
          {strategyInstanceId ? 'Loading strategy data...' : 'No strategy selected'}
        </div>
      </Card>
    );
  }

  const alertColumns = [
    {
      title: 'Time',
      dataIndex: 'timestamp',
      render: (time: Date) => new Date(time).toLocaleTimeString(),
      width: 80
    },
    {
      title: 'Severity',
      dataIndex: 'severity',
      render: (severity: string) => (
        <Tag color={severity === 'critical' ? 'red' : severity === 'error' ? 'orange' : severity === 'warning' ? 'yellow' : 'blue'}>
          {severity.toUpperCase()}
        </Tag>
      ),
      width: 100
    },
    {
      title: 'Type',
      dataIndex: 'type',
      render: (type: string) => type.replace('_', ' ').toUpperCase(),
      width: 150
    },
    {
      title: 'Message',
      dataIndex: 'message',
      ellipsis: true
    }
  ];

  if (compact) {
    return (
      <Card
        size="small"
        title={
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-2">
              <Badge status={getStateColor(strategy.state) as any} />
              <span className="font-medium">{strategy.strategyId}</span>
            </div>
            <Tag color={getHealthColor(strategy.healthStatus.overall)}>
              {strategy.healthStatus.overall.toUpperCase()}
            </Tag>
          </div>
        }
        extra={
          <Space size="small">
            <Tooltip title="Refresh">
              <Button size="small" icon={<ReloadOutlined />} onClick={loadStrategyData} />
            </Tooltip>
            {strategy.state === 'running' && (
              <Button
                size="small"
                icon={<PauseCircleOutlined />}
                onClick={() => controlStrategy('pause', 'Manual pause')}
              >
                Pause
              </Button>
            )}
            {strategy.state === 'paused' && (
              <Button
                size="small"
                icon={<PlayCircleOutlined />}
                onClick={() => controlStrategy('resume', 'Manual resume')}
              >
                Resume
              </Button>
            )}
          </Space>
        }
      >
        <Row gutter={8}>
          <Col span={8}>
            <Statistic
              title="P&L"
              value={strategy.realizedPnL + strategy.unrealizedPnL}
              precision={2}
              prefix="$"
              valueStyle={{ color: (strategy.realizedPnL + strategy.unrealizedPnL) >= 0 ? '#3f8600' : '#cf1322' }}
            />
          </Col>
          <Col span={8}>
            <Statistic
              title="Position"
              value={strategy.currentPosition.quantity}
              suffix={strategy.currentPosition.side}
            />
          </Col>
          <Col span={8}>
            <Statistic
              title="Drawdown"
              value={strategy.riskMetrics.currentDrawdown}
              precision={2}
              suffix="%"
              valueStyle={{ color: strategy.riskMetrics.currentDrawdown > 5 ? '#cf1322' : '#1890ff' }}
            />
          </Col>
        </Row>
      </Card>
    );
  }

  return (
    <div className="live-strategy-monitor">
      <Card
        title={
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-3">
              <MonitorOutlined className="text-blue-600" />
              <div>
                <Title level={4} className="mb-0">{strategy.strategyId}</Title>
                <Text type="secondary">Version {strategy.version}</Text>
              </div>
            </div>
            <div className="flex items-center space-x-3">
              <Badge
                status={getStateColor(strategy.state) as any}
                text={strategy.state.replace('_', ' ').toUpperCase()}
              />
              <Tag color={getHealthColor(strategy.healthStatus.overall)}>
                {strategy.healthStatus.overall.toUpperCase()}
              </Tag>
            </div>
          </div>
        }
        extra={
          <Space>
            <Button icon={<ReloadOutlined />} onClick={loadStrategyData} loading={loading}>
              Refresh
            </Button>
            {strategy.state === 'running' && (
              <Button
                icon={<PauseCircleOutlined />}
                onClick={() => controlStrategy('pause', 'Manual pause')}
                loading={controlling}
              >
                Pause
              </Button>
            )}
            {strategy.state === 'paused' && (
              <Button
                icon={<PlayCircleOutlined />}
                onClick={() => controlStrategy('resume', 'Manual resume')}
                loading={controlling}
              >
                Resume
              </Button>
            )}
            <Button
              danger
              icon={<AlertOutlined />}
              onClick={handleEmergencyStop}
              loading={controlling}
            >
              Emergency Stop
            </Button>
          </Space>
        }
      >
        {/* Performance Metrics */}
        <Row gutter={16} className="mb-6">
          <Col span={6}>
            <Card size="small" className="text-center">
              <Statistic
                title="Total P&L"
                value={strategy.realizedPnL + strategy.unrealizedPnL}
                precision={2}
                prefix="$"
                valueStyle={{ 
                  color: (strategy.realizedPnL + strategy.unrealizedPnL) >= 0 ? '#3f8600' : '#cf1322',
                  fontSize: '20px'
                }}
              />
              <div className="mt-2">
                <Text type="secondary">
                  Realized: {formatCurrency(strategy.realizedPnL)} | 
                  Unrealized: {formatCurrency(strategy.unrealizedPnL)}
                </Text>
              </div>
            </Card>
          </Col>
          <Col span={6}>
            <Card size="small" className="text-center">
              <Statistic
                title="Daily P&L"
                value={strategy.performanceMetrics.dailyPnL}
                precision={2}
                prefix="$"
                valueStyle={{ 
                  color: strategy.performanceMetrics.dailyPnL >= 0 ? '#3f8600' : '#cf1322',
                  fontSize: '20px'
                }}
              />
              <div className="mt-2">
                <Text type="secondary">
                  Win Rate: {formatPercentage(strategy.performanceMetrics.win_rate)}
                </Text>
              </div>
            </Card>
          </Col>
          <Col span={6}>
            <Card size="small" className="text-center">
              <Statistic
                title="Drawdown"
                value={strategy.riskMetrics.currentDrawdown}
                precision={2}
                suffix="%"
                valueStyle={{ 
                  color: strategy.riskMetrics.currentDrawdown > 5 ? '#cf1322' : '#1890ff',
                  fontSize: '20px'
                }}
              />
              <div className="mt-2">
                <Text type="secondary">
                  Max Today: {formatPercentage(strategy.riskMetrics.maxDrawdownToday)}
                </Text>
              </div>
            </Card>
          </Col>
          <Col span={6}>
            <Card size="small" className="text-center">
              <Statistic
                title="Total Trades"
                value={strategy.performanceMetrics.total_trades}
                valueStyle={{ fontSize: '20px' }}
              />
              <div className="mt-2">
                <Text type="secondary">
                  Avg Size: {formatCurrency(strategy.performanceMetrics.avgTradeSize)}
                </Text>
              </div>
            </Card>
          </Col>
        </Row>

        {/* Position and Risk Information */}
        <Row gutter={16} className="mb-6">
          <Col span={12}>
            <Card title="Current Position" size="small">
              <Row gutter={8}>
                <Col span={12}>
                  <Statistic
                    title="Instrument"
                    value={strategy.currentPosition.instrument}
                    valueStyle={{ fontSize: '16px' }}
                  />
                </Col>
                <Col span={12}>
                  <Statistic
                    title="Side"
                    value={strategy.currentPosition.side}
                    valueStyle={{ 
                      color: strategy.currentPosition.side === 'LONG' ? '#3f8600' : 
                             strategy.currentPosition.side === 'SHORT' ? '#cf1322' : '#1890ff',
                      fontSize: '16px'
                    }}
                  />
                </Col>
              </Row>
              <Row gutter={8} className="mt-4">
                <Col span={12}>
                  <Statistic
                    title="Quantity"
                    value={strategy.currentPosition.quantity}
                    valueStyle={{ fontSize: '16px' }}
                  />
                </Col>
                <Col span={12}>
                  <Statistic
                    title="Avg Price"
                    value={strategy.currentPosition.avgPrice}
                    precision={4}
                    valueStyle={{ fontSize: '16px' }}
                  />
                </Col>
              </Row>
            </Card>
          </Col>
          <Col span={12}>
            <Card title="Risk Metrics" size="small">
              <Row gutter={8}>
                <Col span={12}>
                  <Statistic
                    title="VaR"
                    value={strategy.riskMetrics.valueAtRisk}
                    precision={2}
                    prefix="$"
                    valueStyle={{ fontSize: '16px' }}
                  />
                </Col>
                <Col span={12}>
                  <Statistic
                    title="Leverage"
                    value={strategy.riskMetrics.leverageRatio}
                    precision={2}
                    suffix="x"
                    valueStyle={{ fontSize: '16px' }}
                  />
                </Col>
              </Row>
              <Row gutter={8} className="mt-4">
                <Col span={12}>
                  <Statistic
                    title="Correlation"
                    value={strategy.riskMetrics.correlationToPortfolio}
                    precision={3}
                    valueStyle={{ fontSize: '16px' }}
                  />
                </Col>
                <Col span={12}>
                  <Statistic
                    title="Concentration"
                    value={strategy.riskMetrics.concentrationRisk}
                    precision={2}
                    suffix="%"
                    valueStyle={{ fontSize: '16px' }}
                  />
                </Col>
              </Row>
            </Card>
          </Col>
        </Row>

        {/* Health Status */}
        <Card title="System Health" size="small" className="mb-6">
          <Row gutter={16}>
            <Col span={6}>
              <div className="text-center">
                <Badge status={strategy.healthStatus.heartbeat === 'active' ? 'success' : 'error'} />
                <div className="mt-1">
                  <Text strong>Heartbeat</Text>
                  <div>{strategy.healthStatus.heartbeat}</div>
                </div>
              </div>
            </Col>
            <Col span={6}>
              <div className="text-center">
                <Badge status={strategy.healthStatus.dataFeed === 'connected' ? 'success' : 'error'} />
                <div className="mt-1">
                  <Text strong>Data Feed</Text>
                  <div>{strategy.healthStatus.dataFeed}</div>
                </div>
              </div>
            </Col>
            <Col span={6}>
              <div className="text-center">
                <Badge status={strategy.healthStatus.orderExecution === 'normal' ? 'success' : 'error'} />
                <div className="mt-1">
                  <Text strong>Order Execution</Text>
                  <div>{strategy.healthStatus.orderExecution}</div>
                </div>
              </div>
            </Col>
            <Col span={6}>
              <div className="text-center">
                <Badge status={strategy.healthStatus.riskCompliance === 'compliant' ? 'success' : 'error'} />
                <div className="mt-1">
                  <Text strong>Risk Compliance</Text>
                  <div>{strategy.healthStatus.riskCompliance}</div>
                </div>
              </div>
            </Col>
          </Row>
          <Divider />
          <Text type="secondary">
            Last Health Check: {strategy.healthStatus.lastHealthCheck?.toLocaleString() || 'Never'}
          </Text>
        </Card>

        {/* Alerts */}
        {showAlerts && strategy.alerts.length > 0 && (
          <Card title="Active Alerts" size="small">
            <Table
              dataSource={strategy.alerts.filter(alert => !alert.acknowledged)}
              columns={alertColumns}
              pagination={false}
              size="small"
              rowKey="alertId"
              scroll={{ y: 200 }}
            />
          </Card>
        )}
      </Card>
    </div>
  );
};

export default LiveStrategyMonitor;