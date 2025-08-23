import React, { useState, useEffect, useRef } from 'react';
import {
  Card,
  Row,
  Col,
  Statistic,
  Progress,
  Switch,
  Button,
  Alert,
  Space,
  Badge,
  Typography,
  Tooltip,
  Divider,
  Tag,
  List,
  Avatar,
  Spin,
  notification
} from 'antd';
import {
  DashboardOutlined,
  ThunderboltOutlined,
  LineChartOutlined,
  WarningOutlined,
  CheckCircleOutlined,
  CloseCircleOutlined,
  PlayCircleOutlined,
  PauseCircleOutlined,
  ReloadOutlined,
  BellOutlined,
  RiseOutlined,
  FallOutlined,
  MonitorOutlined,
  FireOutlined,
  SafetyCertificateOutlined,
  AlertOutlined
} from '@ant-design/icons';

import { RealTimeRiskMetrics } from './types/riskTypes';
import { riskService } from './services/riskService';
import { useWebSocketManager } from '../../hooks/useWebSocketManager';
import { useRiskMonitoring } from '../../hooks/risk/useRiskMonitoring';

const { Title, Text } = Typography;

interface RealTimeRiskMonitorProps {
  portfolioId: string;
  className?: string;
}

interface RiskTrendData {
  timestamp: Date;
  value: number;
  type: string;
}

const RealTimeRiskMonitor: React.FC<RealTimeRiskMonitorProps> = ({
  portfolioId,
  className
}) => {
  console.log('🎯 RealTimeRiskMonitor rendering for portfolio:', portfolioId);

  const [metrics, setMetrics] = useState<RealTimeRiskMetrics | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [monitoringActive, setMonitoringActive] = useState(false);
  const [lastUpdate, setLastUpdate] = useState<Date | null>(null);
  const [riskTrends, setRiskTrends] = useState<RiskTrendData[]>([]);
  const [criticalAlertCount, setCriticalAlertCount] = useState(0);

  // WebSocket connection for real-time updates
  const { isConnected, sendMessage, lastMessage } = useWebSocketManager({
    url: `${import.meta.env.VITE_WS_URL}/ws/risk/monitor/${portfolioId}`,
    autoConnect: monitoringActive,
    reconnectAttempts: 5,
    reconnectInterval: 3000
  });

  // Risk monitoring hook
  const {
    realTimeMetrics,
    criticalAlerts,
    breachedLimits,
    overallRiskScore,
    isConnected: riskConnected
  } = useRiskMonitoring({ 
    portfolioId, 
    enableRealTime: monitoringActive 
  });

  const updateIntervalRef = useRef<NodeJS.Timeout | null>(null);

  const fetchRiskMetrics = async () => {
    try {
      setError(null);
      const data = await riskService.getRealTimeMetrics(portfolioId);
      setMetrics(data);
      setLastUpdate(new Date());

      // Update trends data
      if (data) {
        const newTrendPoint: RiskTrendData = {
          timestamp: new Date(),
          value: data.overall_risk_score,
          type: 'risk_score'
        };
        setRiskTrends(prev => [...prev.slice(-19), newTrendPoint]);
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to fetch risk metrics');
      console.error('Risk metrics fetch error:', err);
    } finally {
      setLoading(false);
    }
  };

  const handleMonitoringToggle = async (enabled: boolean) => {
    try {
      if (enabled) {
        await riskService.startRealTimeMonitoring(portfolioId, {
          update_frequency_seconds: 1,
          enable_alerts: true,
          enable_auto_actions: true
        });
        notification.success({
          message: 'Real-Time Monitoring Started',
          description: 'Risk monitoring is now active with 1-second updates',
          duration: 3
        });
      } else {
        await riskService.stopRealTimeMonitoring(portfolioId);
        notification.info({
          message: 'Real-Time Monitoring Stopped',
          description: 'Risk monitoring has been deactivated',
          duration: 3
        });
      }
      setMonitoringActive(enabled);
    } catch (error) {
      console.error('Failed to toggle monitoring:', error);
      notification.error({
        message: 'Monitoring Toggle Failed',
        description: 'Unable to change monitoring status',
        duration: 4
      });
    }
  };

  // Handle WebSocket messages
  useEffect(() => {
    if (lastMessage) {
      try {
        const message = JSON.parse(lastMessage);
        if (message.type === 'risk_update' && message.data.risk_metrics) {
          setMetrics(message.data.risk_metrics);
          setLastUpdate(new Date());
        } else if (message.type === 'alert_triggered') {
          setCriticalAlertCount(prev => prev + 1);
          notification.warning({
            message: 'Risk Alert Triggered',
            description: message.data.alert.message,
            duration: 6
          });
        }
      } catch (error) {
        console.error('Error parsing WebSocket message:', error);
      }
    }
  }, [lastMessage]);

  // Initial data fetch
  useEffect(() => {
    fetchRiskMetrics();
    
    // Check monitoring status
    riskService.getMonitoringStatus(portfolioId).then(status => {
      setMonitoringActive(status.monitoring_active);
      setCriticalAlertCount(status.alert_count_24h);
    }).catch(console.error);
  }, [portfolioId]);

  // Setup periodic updates when monitoring is active
  useEffect(() => {
    if (monitoringActive && !isConnected) {
      // Fallback to polling if WebSocket is not connected
      updateIntervalRef.current = setInterval(fetchRiskMetrics, 5000);
    } else if (updateIntervalRef.current) {
      clearInterval(updateIntervalRef.current);
      updateIntervalRef.current = null;
    }

    return () => {
      if (updateIntervalRef.current) {
        clearInterval(updateIntervalRef.current);
      }
    };
  }, [monitoringActive, isConnected, portfolioId]);

  const getRiskColor = (score: number): string => {
    if (score >= 80) return '#ff4d4f'; // Critical
    if (score >= 60) return '#fa8c16'; // High
    if (score >= 40) return '#faad14'; // Medium
    if (score >= 20) return '#52c41a'; // Low
    return '#1890ff'; // Very Low
  };

  const getRiskStatus = (score: number): { text: string; status: 'success' | 'warning' | 'error' | 'processing' } => {
    if (score >= 80) return { text: 'Critical', status: 'error' };
    if (score >= 60) return { text: 'High', status: 'warning' };
    if (score >= 40) return { text: 'Medium', status: 'processing' };
    return { text: 'Low', status: 'success' };
  };

  const formatCurrency = (value: number): string => {
    return new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency: 'USD',
      minimumFractionDigits: 0,
      maximumFractionDigits: 0
    }).format(value);
  };

  const formatPercentage = (value: number): string => {
    return `${value.toFixed(2)}%`;
  };

  const connectionStatus = isConnected || riskConnected;
  const currentMetrics = realTimeMetrics || metrics;
  const currentRiskScore = overallRiskScore || currentMetrics?.overall_risk_score || 0;

  return (
    <div className={className}>
      {/* Header Controls */}
      <Card size="small" style={{ marginBottom: 16 }}>
        <Row align="middle" justify="space-between">
          <Col>
            <Space>
              <Title level={4} style={{ margin: 0 }}>
                <MonitorOutlined /> Real-Time Risk Monitor
              </Title>
              <Badge 
                status={connectionStatus ? 'processing' : 'error'}
                text={connectionStatus ? 'Live Connection' : 'Disconnected'}
                style={{ fontSize: '12px' }}
              />
            </Space>
          </Col>
          <Col>
            <Space>
              <Tooltip title={monitoringActive ? 'Stop monitoring' : 'Start monitoring'}>
                <Switch
                  checked={monitoringActive}
                  onChange={handleMonitoringToggle}
                  checkedChildren={<PlayCircleOutlined />}
                  unCheckedChildren={<PauseCircleOutlined />}
                  loading={loading}
                />
              </Tooltip>
              <Tooltip title="Refresh metrics">
                <Button 
                  icon={<ReloadOutlined />}
                  onClick={fetchRiskMetrics}
                  size="small"
                  loading={loading}
                />
              </Tooltip>
              {lastUpdate && (
                <Text type="secondary" style={{ fontSize: '12px' }}>
                  Updated: {lastUpdate.toLocaleTimeString()}
                </Text>
              )}
            </Space>
          </Col>
        </Row>
      </Card>

      {/* Error Alert */}
      {error && (
        <Alert
          message="Risk Monitoring Error"
          description={error}
          type="error"
          showIcon
          closable
          onClose={() => setError(null)}
          style={{ marginBottom: 16 }}
        />
      )}

      {/* Critical Alerts Banner */}
      {criticalAlerts.length > 0 && (
        <Alert
          message={
            <Space>
              <FireOutlined />
              <Text strong>CRITICAL RISK ALERT</Text>
            </Space>
          }
          description={`${criticalAlerts.length} critical risk condition(s) require immediate attention`}
          type="error"
          showIcon
          style={{ marginBottom: 16 }}
          action={
            <Button size="small" danger>
              View Alerts
            </Button>
          }
        />
      )}

      {/* Main Risk Metrics */}
      <Row gutter={[16, 16]}>
        {/* Overall Risk Score */}
        <Col xs={24} sm={12} md={8}>
          <Card>
            <Statistic
              title={
                <Space>
                  <SafetyCertificateOutlined />
                  Overall Risk Score
                </Space>
              }
              value={currentRiskScore}
              precision={1}
              suffix="%"
              loading={loading}
              prefix={
                <Progress
                  type="circle"
                  percent={currentRiskScore}
                  size={60}
                  strokeColor={getRiskColor(currentRiskScore)}
                  showInfo={false}
                  style={{ marginRight: 8 }}
                />
              }
              valueStyle={{ 
                color: getRiskColor(currentRiskScore),
                fontSize: '24px',
                fontWeight: 'bold'
              }}
            />
            <div style={{ marginTop: 8 }}>
              <Badge 
                {...getRiskStatus(currentRiskScore)}
                text={`Risk Level: ${getRiskStatus(currentRiskScore).text}`}
              />
            </div>
          </Card>
        </Col>

        {/* Portfolio Value */}
        <Col xs={24} sm={12} md={8}>
          <Card>
            <Statistic
              title="Portfolio Value"
              value={currentMetrics?.portfolio_value || 0}
              precision={0}
              prefix="$"
              loading={loading}
              formatter={(value) => `${Number(value).toLocaleString()}`}
            />
            <div style={{ marginTop: 8, display: 'flex', alignItems: 'center' }}>
              <Text type="secondary" style={{ fontSize: '12px' }}>
                Exposure: {formatCurrency(currentMetrics?.total_exposure || 0)}
              </Text>
            </div>
          </Card>
        </Col>

        {/* VaR 95% */}
        <Col xs={24} sm={12} md={8}>
          <Card>
            <Statistic
              title="1-Day VaR (95%)"
              value={currentMetrics?.var_95_current || 0}
              precision={0}
              prefix="$"
              loading={loading}
              formatter={(value) => `${Number(value).toLocaleString()}`}
              valueStyle={{ color: '#ff4d4f' }}
            />
            <div style={{ marginTop: 8 }}>
              <Text type="secondary" style={{ fontSize: '12px' }}>
                99%: {formatCurrency(currentMetrics?.var_99_current || 0)}
              </Text>
            </div>
          </Card>
        </Col>

        {/* Leverage & Margin */}
        <Col xs={24} sm={12} md={8}>
          <Card>
            <Statistic
              title="Leverage Ratio"
              value={currentMetrics?.leverage_ratio || 0}
              precision={2}
              suffix="x"
              loading={loading}
              valueStyle={{ 
                color: (currentMetrics?.leverage_ratio || 0) > 3 ? '#ff4d4f' : '#52c41a'
              }}
            />
            <div style={{ marginTop: 8 }}>
              <Text type="secondary" style={{ fontSize: '12px' }}>
                Margin: {formatPercentage(currentMetrics?.margin_utilization || 0)}
              </Text>
            </div>
          </Card>
        </Col>

        {/* Volatility */}
        <Col xs={24} sm={12} md={8}>
          <Card>
            <Statistic
              title="24h Volatility"
              value={currentMetrics?.volatility_24h || 0}
              precision={2}
              suffix="%"
              loading={loading}
              prefix={<RiseOutlined />}
              valueStyle={{ 
                color: (currentMetrics?.volatility_24h || 0) > 20 ? '#ff4d4f' : '#1890ff'
              }}
            />
            <div style={{ marginTop: 8 }}>
              <Text type="secondary" style={{ fontSize: '12px' }}>
                Drawdown: {formatPercentage(currentMetrics?.drawdown_current || 0)}
              </Text>
            </div>
          </Card>
        </Col>

        {/* Positions & Orders */}
        <Col xs={24} sm={12} md={8}>
          <Card>
            <Statistic
              title="Active Positions"
              value={currentMetrics?.position_count || 0}
              loading={loading}
              prefix={<LineChartOutlined />}
            />
            <div style={{ marginTop: 8 }}>
              <Text type="secondary" style={{ fontSize: '12px' }}>
                Orders: {currentMetrics?.active_orders_count || 0}
              </Text>
            </div>
          </Card>
        </Col>
      </Row>

      {/* Risk Component Scores */}
      <Card 
        title={
          <Space>
            <DashboardOutlined />
            Risk Component Analysis
          </Space>
        }
        style={{ marginTop: 16 }}
      >
        <Row gutter={[16, 16]}>
          <Col xs={24} sm={12} md={6}>
            <div style={{ textAlign: 'center' }}>
              <Progress
                type="circle"
                percent={currentMetrics?.concentration_risk_score || 0}
                size={80}
                strokeColor={getRiskColor(currentMetrics?.concentration_risk_score || 0)}
              />
              <div style={{ marginTop: 8 }}>
                <Text strong>Concentration</Text>
              </div>
            </div>
          </Col>
          <Col xs={24} sm={12} md={6}>
            <div style={{ textAlign: 'center' }}>
              <Progress
                type="circle"
                percent={currentMetrics?.correlation_risk_score || 0}
                size={80}
                strokeColor={getRiskColor(currentMetrics?.correlation_risk_score || 0)}
              />
              <div style={{ marginTop: 8 }}>
                <Text strong>Correlation</Text>
              </div>
            </div>
          </Col>
          <Col xs={24} sm={12} md={6}>
            <div style={{ textAlign: 'center' }}>
              <Progress
                type="circle"
                percent={currentMetrics?.liquidity_risk_score || 0}
                size={80}
                strokeColor={getRiskColor(currentMetrics?.liquidity_risk_score || 0)}
              />
              <div style={{ marginTop: 8 }}>
                <Text strong>Liquidity</Text>
              </div>
            </div>
          </Col>
          <Col xs={24} sm={12} md={6}>
            <div style={{ textAlign: 'center', position: 'relative' }}>
              <Progress
                type="circle"
                percent={Math.min((currentMetrics?.leverage_ratio || 0) * 20, 100)}
                size={80}
                strokeColor={getRiskColor(Math.min((currentMetrics?.leverage_ratio || 0) * 20, 100))}
              />
              <div style={{ marginTop: 8 }}>
                <Text strong>Leverage</Text>
              </div>
              {(currentMetrics?.leverage_ratio || 0) > 4 && (
                <Tag color="red" style={{ position: 'absolute', top: -5, right: 10 }}>
                  HIGH
                </Tag>
              )}
            </div>
          </Col>
        </Row>
      </Card>

      {/* Recent Alerts */}
      {(criticalAlerts.length > 0 || breachedLimits.length > 0) && (
        <Card
          title={
            <Space>
              <AlertOutlined />
              Active Risk Issues
              <Badge count={criticalAlerts.length + breachedLimits.length} />
            </Space>
          }
          style={{ marginTop: 16 }}
          size="small"
        >
          <List
            dataSource={[
              ...criticalAlerts.map(alert => ({
                id: alert.id,
                type: 'alert',
                severity: alert.severity,
                message: alert.message,
                time: alert.triggered_at
              })),
              ...breachedLimits.map(limit => ({
                id: limit.id,
                type: 'limit',
                severity: 'critical' as const,
                message: `Limit breach: ${limit.name}`,
                time: limit.last_breach || new Date()
              }))
            ]}
            renderItem={(item) => (
              <List.Item>
                <List.Item.Meta
                  avatar={
                    <Avatar 
                      icon={item.type === 'alert' ? <BellOutlined /> : <WarningOutlined />}
                      style={{ 
                        backgroundColor: item.severity === 'critical' ? '#ff4d4f' : '#fa8c16' 
                      }}
                    />
                  }
                  title={
                    <Space>
                      <Text strong>{item.message}</Text>
                      <Tag color={item.severity === 'critical' ? 'red' : 'orange'}>
                        {item.severity.toUpperCase()}
                      </Tag>
                    </Space>
                  }
                  description={
                    <Text type="secondary" style={{ fontSize: '12px' }}>
                      {new Date(item.time).toLocaleString()}
                    </Text>
                  }
                />
              </List.Item>
            )}
          />
        </Card>
      )}

      {/* Connection Status */}
      <Card size="small" style={{ marginTop: 16 }}>
        <Row align="middle" justify="space-between">
          <Col>
            <Space>
              <Badge 
                status={connectionStatus ? 'processing' : 'error'}
                text={connectionStatus ? 'Real-time connection active' : 'Connection offline'}
              />
              {monitoringActive && (
                <Tag color="green">
                  <PlayCircleOutlined /> Monitoring Active
                </Tag>
              )}
            </Space>
          </Col>
          <Col>
            <Space size="large">
              <div>
                <Text type="secondary" style={{ fontSize: '12px' }}>
                  Update Frequency: {monitoringActive ? '1 second' : 'Manual'}
                </Text>
              </div>
              <div>
                <Text type="secondary" style={{ fontSize: '12px' }}>
                  Alerts 24h: {criticalAlertCount}
                </Text>
              </div>
            </Space>
          </Col>
        </Row>
      </Card>
    </div>
  );
};

export default RealTimeRiskMonitor;