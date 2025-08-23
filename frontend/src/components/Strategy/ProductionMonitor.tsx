import React, { useState, useEffect, useCallback, useRef } from 'react';
import {
  Card,
  Row,
  Col,
  Statistic,
  Alert,
  Button,
  Space,
  Typography,
  Progress,
  Timeline,
  Table,
  Badge,
  Tag,
  Switch,
  Select,
  Tooltip,
  Modal,
  Form,
  InputNumber,
  notification,
  Tabs,
  List,
  Avatar,
  Divider
} from 'antd';
import {
  MonitorOutlined,
  ThunderboltOutlined,
  AlertOutlined,
  CheckCircleOutlined,
  ExclamationCircleOutlined,
  ClockCircleOutlined,
  DollarCircleOutlined,
  LineChartOutlined,
  BarChartOutlined,
  ArrowUpOutlined,
  ArrowDownOutlined,
  ReloadOutlined,
  SettingOutlined,
  BellOutlined,
  FireOutlined,
  ShieldOutlined,
  RobotOutlined,
  PlayCircleOutlined,
  PauseCircleOutlined,
  StopOutlined,
  EyeOutlined,
  FullscreenOutlined,
  DownloadOutlined
} from '@ant-design/icons';
import { Line, Area, Column } from '@ant-design/charts';
import dayjs from 'dayjs';
import type { ColumnType } from 'antd/es/table';
import type {
  ProductionMonitorProps,
  ProductionMonitoringDashboard,
  RealTimeMetrics,
  MonitoringAlert,
  PerformanceIndicator,
  ProductionMonitoringRequest
} from './types/deploymentTypes';

const { Title, Text, Paragraph } = Typography;
const { Option } = Select;
const { TabPane } = Tabs;

const API_BASE = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8001';

interface MetricHistory {
  timestamp: string;
  pnl_unrealized: number;
  pnl_realized: number;
  position_count: number;
  latency_ms: number;
  cpu_usage: number;
  memory_usage: number;
}

interface TradeExecution {
  trade_id: string;
  timestamp: Date;
  symbol: string;
  side: 'buy' | 'sell';
  quantity: number;
  price: number;
  pnl: number;
  execution_time_ms: number;
  venue: string;
}

const ProductionMonitor: React.FC<ProductionMonitorProps> = ({
  strategyId,
  deploymentId,
  refreshInterval = 5000,
  showAlerts = true,
  onAlert
}) => {
  const [dashboard, setDashboard] = useState<ProductionMonitoringDashboard | null>(null);
  const [metricHistory, setMetricHistory] = useState<MetricHistory[]>([]);
  const [recentTrades, setRecentTrades] = useState<TradeExecution[]>([]);
  const [loading, setLoading] = useState(false);
  const [autoRefresh, setAutoRefresh] = useState(true);
  const [selectedMetric, setSelectedMetric] = useState<string>('pnl_unrealized');
  const [alertsFilter, setAlertsFilter] = useState<string>('all');
  const [showConfigModal, setShowConfigModal] = useState(false);
  const [isFullscreen, setIsFullscreen] = useState(false);
  const [connectionStatus, setConnectionStatus] = useState<'connected' | 'disconnected' | 'connecting'>('disconnected');
  
  const intervalRef = useRef<NodeJS.Timeout>();
  const wsRef = useRef<WebSocket>();

  const loadMonitoringData = useCallback(async () => {
    if (!strategyId || !deploymentId) return;
    
    setLoading(true);
    try {
      const response = await fetch(`${API_BASE}/api/v1/strategies/production/${deploymentId}/monitor`);
      if (!response.ok) throw new Error('Failed to load monitoring data');
      
      const data: ProductionMonitoringDashboard = await response.json();
      setDashboard(data);
      
      // Load additional data
      await Promise.all([
        loadMetricHistory(),
        loadRecentTrades()
      ]);
      
    } catch (error) {
      console.error('Error loading monitoring data:', error);
      notification.error({
        message: 'Failed to Load Monitoring Data',
        description: error instanceof Error ? error.message : 'Unknown error'
      });
    } finally {
      setLoading(false);
    }
  }, [strategyId, deploymentId]);

  const loadMetricHistory = async () => {
    try {
      const response = await fetch(`${API_BASE}/api/v1/strategies/production/${deploymentId}/metrics/history?hours=24`);
      if (response.ok) {
        const data = await response.json();
        setMetricHistory(data.metrics || []);
      }
    } catch (error) {
      console.error('Error loading metric history:', error);
    }
  };

  const loadRecentTrades = async () => {
    try {
      const response = await fetch(`${API_BASE}/api/v1/strategies/production/${deploymentId}/trades/recent?limit=50`);
      if (response.ok) {
        const data = await response.json();
        setRecentTrades(data.trades || []);
      }
    } catch (error) {
      console.error('Error loading recent trades:', error);
    }
  };

  const connectWebSocket = useCallback(() => {
    if (!deploymentId) return;
    
    setConnectionStatus('connecting');
    
    const wsUrl = `${API_BASE.replace('http', 'ws')}/ws/production/${deploymentId}/monitor`;
    const ws = new WebSocket(wsUrl);
    
    ws.onopen = () => {
      setConnectionStatus('connected');
      console.log('Production monitor WebSocket connected');
    };
    
    ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        
        if (data.type === 'metrics_update') {
          setDashboard(prev => prev ? { ...prev, real_time_metrics: data.metrics } : null);
        } else if (data.type === 'alert') {
          const alert: MonitoringAlert = data.alert;
          setDashboard(prev => prev ? { 
            ...prev, 
            alerts: [alert, ...prev.alerts].slice(0, 100) 
          } : null);
          
          if (showAlerts) {
            notification[alert.severity === 'critical' || alert.severity === 'high' ? 'error' : 'warning']({
              message: `${alert.type.toUpperCase()} Alert`,
              description: alert.message,
              duration: alert.severity === 'critical' ? 0 : 10
            });
          }
          
          onAlert?.(alert);
        } else if (data.type === 'trade_execution') {
          const trade: TradeExecution = data.trade;
          setRecentTrades(prev => [trade, ...prev].slice(0, 50));
        }
      } catch (error) {
        console.error('Error parsing WebSocket message:', error);
      }
    };
    
    ws.onclose = () => {
      setConnectionStatus('disconnected');
      console.log('Production monitor WebSocket disconnected');
      
      // Attempt to reconnect after 5 seconds
      setTimeout(() => {
        if (autoRefresh) {
          connectWebSocket();
        }
      }, 5000);
    };
    
    ws.onerror = (error) => {
      console.error('Production monitor WebSocket error:', error);
      setConnectionStatus('disconnected');
    };
    
    wsRef.current = ws;
  }, [deploymentId, autoRefresh, showAlerts, onAlert]);

  useEffect(() => {
    loadMonitoringData();
    
    if (autoRefresh) {
      connectWebSocket();
      intervalRef.current = setInterval(loadMonitoringData, refreshInterval);
    }
    
    return () => {
      if (intervalRef.current) {
        clearInterval(intervalRef.current);
      }
      if (wsRef.current) {
        wsRef.current.close();
      }
    };
  }, [loadMonitoringData, connectWebSocket, autoRefresh, refreshInterval]);

  const acknowledgeAlert = async (alertId: string) => {
    try {
      const response = await fetch(`${API_BASE}/api/v1/strategies/production/alerts/${alertId}/acknowledge`, {
        method: 'POST'
      });
      
      if (response.ok) {
        setDashboard(prev => prev ? {
          ...prev,
          alerts: prev.alerts.map(alert =>
            alert.alert_id === alertId ? { ...alert, acknowledged: true } : alert
          )
        } : null);
        
        notification.success({
          message: 'Alert Acknowledged',
          description: 'Alert has been acknowledged successfully'
        });
      }
    } catch (error) {
      console.error('Error acknowledging alert:', error);
    }
  };

  const resolveAlert = async (alertId: string, notes: string) => {
    try {
      const response = await fetch(`${API_BASE}/api/v1/strategies/production/alerts/${alertId}/resolve`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ resolution_notes: notes })
      });
      
      if (response.ok) {
        setDashboard(prev => prev ? {
          ...prev,
          alerts: prev.alerts.map(alert =>
            alert.alert_id === alertId ? { ...alert, resolved: true, resolution_notes: notes } : alert
          )
        } : null);
        
        notification.success({
          message: 'Alert Resolved',
          description: 'Alert has been resolved successfully'
        });
      }
    } catch (error) {
      console.error('Error resolving alert:', error);
    }
  };

  const getHealthScoreColor = (score: number): string => {
    if (score >= 90) return '#52c41a';
    if (score >= 70) return '#faad14';
    if (score >= 50) return '#ff7a45';
    return '#ff4d4f';
  };

  const getIndicatorStatus = (status: string) => {
    switch (status) {
      case 'good': return { color: '#52c41a', icon: <CheckCircleOutlined /> };
      case 'warning': return { color: '#faad14', icon: <ExclamationCircleOutlined /> };
      case 'critical': return { color: '#ff4d4f', icon: <AlertOutlined /> };
      default: return { color: '#d9d9d9', icon: <ClockCircleOutlined /> };
    }
  };

  const renderMetricsOverview = () => {
    if (!dashboard?.real_time_metrics) return null;
    
    const metrics = dashboard.real_time_metrics;
    
    return (
      <Row gutter={[16, 16]} className="mb-4">
        <Col xs={24} sm={8} md={6}>
          <Card size="small">
            <Statistic
              title="Unrealized P&L"
              value={metrics.pnl_unrealized}
              precision={2}
              prefix="$"
              valueStyle={{ color: metrics.pnl_unrealized >= 0 ? '#3f8600' : '#cf1322' }}
              suffix={
                <div style={{ fontSize: '12px', color: '#8c8c8c' }}>
                  {metrics.pnl_unrealized >= 0 ? <ArrowUpOutlined /> : <ArrowDownOutlined />}
                </div>
              }
            />
          </Card>
        </Col>
        
        <Col xs={24} sm={8} md={6}>
          <Card size="small">
            <Statistic
              title="Realized P&L"
              value={metrics.pnl_realized}
              precision={2}
              prefix="$"
              valueStyle={{ color: metrics.pnl_realized >= 0 ? '#3f8600' : '#cf1322' }}
            />
          </Card>
        </Col>
        
        <Col xs={24} sm={8} md={6}>
          <Card size="small">
            <Statistic
              title="Active Positions"
              value={metrics.position_count}
              prefix={<LineChartOutlined />}
            />
          </Card>
        </Col>
        
        <Col xs={24} sm={8} md={6}>
          <Card size="small">
            <Statistic
              title="Orders Today"
              value={metrics.order_count}
              prefix={<BarChartOutlined />}
            />
          </Card>
        </Col>
        
        <Col xs={24} sm={8} md={6}>
          <Card size="small">
            <Statistic
              title="Fill Rate"
              value={metrics.fill_rate * 100}
              precision={1}
              suffix="%"
              valueStyle={{ color: metrics.fill_rate >= 0.95 ? '#3f8600' : '#cf1322' }}
            />
          </Card>
        </Col>
        
        <Col xs={24} sm={8} md={6}>
          <Card size="small">
            <Statistic
              title="Avg Latency"
              value={metrics.latency_ms}
              precision={0}
              suffix="ms"
              prefix={<ThunderboltOutlined />}
              valueStyle={{ color: metrics.latency_ms <= 100 ? '#3f8600' : '#cf1322' }}
            />
          </Card>
        </Col>
        
        <Col xs={24} sm={8} md={6}>
          <Card size="small">
            <Statistic
              title="CPU Usage"
              value={metrics.cpu_usage}
              precision={1}
              suffix="%"
              valueStyle={{ color: metrics.cpu_usage <= 80 ? '#3f8600' : '#cf1322' }}
            />
          </Card>
        </Col>
        
        <Col xs={24} sm={8} md={6}>
          <Card size="small">
            <Statistic
              title="Memory Usage"
              value={metrics.memory_usage}
              precision={1}
              suffix="%"
              valueStyle={{ color: metrics.memory_usage <= 80 ? '#3f8600' : '#cf1322' }}
            />
          </Card>
        </Col>
      </Row>
    );
  };

  const renderHealthScore = () => {
    if (!dashboard) return null;
    
    return (
      <Card title="System Health" className="mb-4">
        <Row gutter={16}>
          <Col span={8}>
            <div className="text-center">
              <Progress
                type="circle"
                percent={dashboard.health_score}
                format={() => `${dashboard.health_score}`}
                strokeColor={getHealthScoreColor(dashboard.health_score)}
                width={120}
              />
              <div className="mt-2">
                <Text strong>Overall Health Score</Text>
              </div>
            </div>
          </Col>
          
          <Col span={16}>
            <div className="space-y-2">
              {dashboard.performance_indicators.map((indicator, index) => {
                const status = getIndicatorStatus(indicator.status);
                return (
                  <div key={index} className="flex items-center justify-between">
                    <Space>
                      <span style={{ color: status.color }}>{status.icon}</span>
                      <Text>{indicator.name}</Text>
                    </Space>
                    <Space>
                      <Text strong>{indicator.value} {indicator.unit}</Text>
                      {indicator.benchmark && (
                        <Text type="secondary">(vs {indicator.benchmark})</Text>
                      )}
                      <Tag color={
                        indicator.trend === 'improving' ? 'success' :
                        indicator.trend === 'degrading' ? 'error' : 'default'
                      }>
                        {indicator.trend.toUpperCase()}
                      </Tag>
                    </Space>
                  </div>
                );
              })}
            </div>
          </Col>
        </Row>
      </Card>
    );
  };

  const renderMetricsChart = () => {
    if (metricHistory.length === 0) return null;
    
    const chartData = metricHistory.map(point => ({
      timestamp: dayjs(point.timestamp).format('HH:mm'),
      value: point[selectedMetric as keyof MetricHistory] as number
    }));
    
    const config = {
      data: chartData,
      xField: 'timestamp',
      yField: 'value',
      smooth: true,
      height: 250,
      color: selectedMetric.includes('pnl') ? (
        chartData[chartData.length - 1]?.value >= 0 ? '#52c41a' : '#ff4d4f'
      ) : '#1890ff'
    };
    
    return (
      <Card
        title="Metrics Trend"
        extra={
          <Select
            value={selectedMetric}
            onChange={setSelectedMetric}
            style={{ width: 150 }}
            size="small"
          >
            <Option value="pnl_unrealized">Unrealized P&L</Option>
            <Option value="pnl_realized">Realized P&L</Option>
            <Option value="position_count">Position Count</Option>
            <Option value="latency_ms">Latency</Option>
            <Option value="cpu_usage">CPU Usage</Option>
            <Option value="memory_usage">Memory Usage</Option>
          </Select>
        }
        className="mb-4"
      >
        <Line {...config} />
      </Card>
    );
  };

  const renderAlerts = () => {
    if (!dashboard?.alerts) return null;
    
    const filteredAlerts = dashboard.alerts.filter(alert => {
      if (alertsFilter === 'all') return true;
      if (alertsFilter === 'unresolved') return !alert.resolved;
      if (alertsFilter === 'critical') return alert.severity === 'critical';
      return alert.severity === alertsFilter;
    });
    
    const alertColumns: ColumnType<MonitoringAlert>[] = [
      {
        title: 'Severity',
        dataIndex: 'severity',
        key: 'severity',
        width: 100,
        render: (severity: string) => (
          <Tag color={
            severity === 'critical' ? 'red' :
            severity === 'high' ? 'orange' :
            severity === 'medium' ? 'yellow' : 'blue'
          }>
            {severity.toUpperCase()}
          </Tag>
        )
      },
      {
        title: 'Type',
        dataIndex: 'type',
        key: 'type',
        width: 100,
        render: (type: string) => (
          <Tag>{type.replace('_', ' ').toUpperCase()}</Tag>
        )
      },
      {
        title: 'Message',
        dataIndex: 'message',
        key: 'message',
        ellipsis: true
      },
      {
        title: 'Time',
        dataIndex: 'timestamp',
        key: 'timestamp',
        width: 120,
        render: (timestamp: Date) => dayjs(timestamp).format('HH:mm:ss')
      },
      {
        title: 'Status',
        key: 'status',
        width: 100,
        render: (_, alert) => (
          <Space>
            {alert.acknowledged && <Badge status="processing" text="ACK" />}
            {alert.resolved && <Badge status="success" text="RESOLVED" />}
          </Space>
        )
      },
      {
        title: 'Actions',
        key: 'actions',
        width: 120,
        render: (_, alert) => (
          <Space size="small">
            {!alert.acknowledged && (
              <Button
                size="small"
                onClick={() => acknowledgeAlert(alert.alert_id)}
              >
                ACK
              </Button>
            )}
            {!alert.resolved && (
              <Button
                size="small"
                type="primary"
                onClick={() => {
                  Modal.confirm({
                    title: 'Resolve Alert',
                    content: (
                      <div>
                        <p>Resolution notes:</p>
                        <textarea
                          id="resolution-notes"
                          rows={3}
                          style={{ width: '100%' }}
                          placeholder="Enter resolution notes..."
                        />
                      </div>
                    ),
                    onOk: () => {
                      const notes = (document.getElementById('resolution-notes') as HTMLTextAreaElement)?.value || '';
                      resolveAlert(alert.alert_id, notes);
                    }
                  });
                }}
              >
                Resolve
              </Button>
            )}
          </Space>
        )
      }
    ];
    
    return (
      <Card
        title="Alerts & Notifications"
        extra={
          <Select
            value={alertsFilter}
            onChange={setAlertsFilter}
            style={{ width: 120 }}
            size="small"
          >
            <Option value="all">All</Option>
            <Option value="unresolved">Unresolved</Option>
            <Option value="critical">Critical</Option>
            <Option value="high">High</Option>
          </Select>
        }
        className="mb-4"
      >
        <Table
          columns={alertColumns}
          dataSource={filteredAlerts}
          rowKey="alert_id"
          size="small"
          pagination={{ pageSize: 10 }}
        />
      </Card>
    );
  };

  const renderRecentTrades = () => {
    if (recentTrades.length === 0) return null;
    
    const tradeColumns: ColumnType<TradeExecution>[] = [
      {
        title: 'Time',
        dataIndex: 'timestamp',
        key: 'timestamp',
        width: 80,
        render: (timestamp: Date) => dayjs(timestamp).format('HH:mm:ss')
      },
      {
        title: 'Symbol',
        dataIndex: 'symbol',
        key: 'symbol',
        width: 80
      },
      {
        title: 'Side',
        dataIndex: 'side',
        key: 'side',
        width: 60,
        render: (side: string) => (
          <Tag color={side === 'buy' ? 'green' : 'red'}>
            {side.toUpperCase()}
          </Tag>
        )
      },
      {
        title: 'Quantity',
        dataIndex: 'quantity',
        key: 'quantity',
        width: 80,
        align: 'right'
      },
      {
        title: 'Price',
        dataIndex: 'price',
        key: 'price',
        width: 80,
        align: 'right',
        render: (price: number) => `$${price.toFixed(2)}`
      },
      {
        title: 'P&L',
        dataIndex: 'pnl',
        key: 'pnl',
        width: 80,
        align: 'right',
        render: (pnl: number) => (
          <Text style={{ color: pnl >= 0 ? '#3f8600' : '#cf1322' }}>
            ${pnl.toFixed(2)}
          </Text>
        )
      },
      {
        title: 'Latency',
        dataIndex: 'execution_time_ms',
        key: 'execution_time_ms',
        width: 80,
        align: 'right',
        render: (latency: number) => `${latency}ms`
      },
      {
        title: 'Venue',
        dataIndex: 'venue',
        key: 'venue',
        width: 80
      }
    ];
    
    return (
      <Card title="Recent Trades" className="mb-4">
        <Table
          columns={tradeColumns}
          dataSource={recentTrades}
          rowKey="trade_id"
          size="small"
          pagination={{ pageSize: 15 }}
          scroll={{ y: 400 }}
        />
      </Card>
    );
  };

  if (loading && !dashboard) {
    return <Card loading={true} style={{ minHeight: '400px' }} />;
  }

  return (
    <div className={`production-monitor ${isFullscreen ? 'fullscreen' : ''}`}>
      <Card
        title={
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-2">
              <MonitorOutlined />
              <span>Production Monitor</span>
              <Badge
                status={connectionStatus === 'connected' ? 'success' : 
                       connectionStatus === 'connecting' ? 'processing' : 'error'}
                text={connectionStatus}
              />
              {dashboard && (
                <Tag color={getHealthScoreColor(dashboard.health_score)}>
                  Health: {dashboard.health_score}%
                </Tag>
              )}
            </div>
            <Space>
              <Switch
                checked={autoRefresh}
                onChange={setAutoRefresh}
                checkedChildren="Auto"
                unCheckedChildren="Manual"
              />
              <Button
                icon={<ReloadOutlined />}
                onClick={loadMonitoringData}
                loading={loading}
              />
              <Button
                icon={<SettingOutlined />}
                onClick={() => setShowConfigModal(true)}
              />
              <Button
                icon={isFullscreen ? <ExclamationCircleOutlined /> : <FullscreenOutlined />}
                onClick={() => setIsFullscreen(!isFullscreen)}
              />
            </Space>
          </div>
        }
      >
        {renderMetricsOverview()}
        
        <Row gutter={16}>
          <Col span={16}>
            {renderHealthScore()}
            {renderMetricsChart()}
            {showAlerts && renderAlerts()}
          </Col>
          
          <Col span={8}>
            {renderRecentTrades()}
          </Col>
        </Row>
      </Card>
    </div>
  );
};

export default ProductionMonitor;