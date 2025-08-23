/**
 * RealTimeAnalyticsDashboard - Comprehensive Real-time Analytics Dashboard for Sprint 3
 * 
 * Features:
 * - Real-time P&L tracking with sub-second updates
 * - Risk metrics dashboard with VaR, exposure limits, and stress testing
 * - Strategy performance analytics and comparison
 * - Trade analytics with execution quality and slippage analysis
 * - WebSocket integration for live streaming updates
 * - Export capabilities for reports
 * - Responsive design for different screen sizes
 */

import React, { useState, useEffect, useCallback, useMemo } from 'react';
import {
  Card,
  Row,
  Col,
  Statistic,
  Typography,
  Space,
  Badge,
  Progress,
  Table,
  Tag,
  Select,
  Button,
  Alert,
  Tooltip,
  Switch,
  Divider,
  Timeline,
  List,
  Tabs,
  Spin,
  Modal,
  Dropdown,
  Menu,
  Drawer,
  Form,
  InputNumber,
  DatePicker,
  Collapse,
  Avatar
} from 'antd';
import {
  LineChartOutlined,
  RiseOutlined,
  FallOutlined,
  DashboardOutlined,
  ThunderboltOutlined,
  AlertOutlined,
  ReloadOutlined,
  SettingOutlined,
  PlayCircleOutlined,
  PauseCircleOutlined,
  FireOutlined,
  DownloadOutlined,
  FullscreenOutlined,
  WarningOutlined,
  DollarOutlined,
  PercentageOutlined,
  StockOutlined,
  BarChartOutlined,
  PieChartOutlined,
  RadarChartOutlined,
  ClockCircleOutlined,
  BellOutlined,
  CheckCircleOutlined,
  CloseCircleOutlined,
  SyncOutlined,
  EyeOutlined,
  ExportOutlined,
  FilterOutlined
} from '@ant-design/icons';
import { Line, Area, Column, Pie, Gauge, Heatmap } from '@ant-design/charts';
import dayjs from 'dayjs';

// Import custom hooks
import { useRealTimeAnalytics } from '../../hooks/analytics/useRealTimeAnalytics';
import { usePerformanceMetrics } from '../../hooks/analytics/usePerformanceMetrics';
import { useRiskAnalytics } from '../../hooks/analytics/useRiskAnalytics';
import { useStrategyAnalytics } from '../../hooks/analytics/useStrategyAnalytics';
import { useExecutionAnalytics } from '../../hooks/analytics/useExecutionAnalytics';
import { useWebSocketManager } from '../../hooks/useWebSocketManager';
import { useMessageBus } from '../../hooks/useMessageBus';

// Import types
import type { 
  PerformanceMetric, 
  AnalyticsStreamData,
  RealTimeRiskMetrics,
  WebSocketConnectionState 
} from '../../types/sprint3';

const { Text, Title } = Typography;
const { Option } = Select;
const { TabPane } = Tabs;
const { Panel } = Collapse;
const { RangePicker } = DatePicker;

interface RealTimeAnalyticsDashboardProps {
  portfolioId?: string;
  showStreaming?: boolean;
  updateInterval?: number;
  compactMode?: boolean;
  enableExports?: boolean;
  className?: string;
}

interface DashboardSettings {
  updateInterval: number;
  autoRefresh: boolean;
  showAlerts: boolean;
  metricsToShow: string[];
  chartTimeRange: string;
  riskThresholds: {
    var95: number;
    leverage: number;
    concentration: number;
  };
}

const RealTimeAnalyticsDashboard: React.FC<RealTimeAnalyticsDashboardProps> = ({
  portfolioId = 'default-portfolio',
  showStreaming = true,
  updateInterval = 250,
  compactMode = false,
  enableExports = true,
  className
}) => {
  // State management
  const [activeTab, setActiveTab] = useState<string>('overview');
  const [settings, setSettings] = useState<DashboardSettings>({
    updateInterval: updateInterval,
    autoRefresh: true,
    showAlerts: true,
    metricsToShow: ['pnl', 'sharpe', 'drawdown', 'var95', 'exposure'],
    chartTimeRange: '1H',
    riskThresholds: {
      var95: 50000,
      leverage: 3.0,
      concentration: 20.0
    }
  });
  const [selectedTimeRange, setSelectedTimeRange] = useState<'1M' | '5M' | '15M' | '1H' | '4H' | '1D'>('1H');
  const [selectedStrategy, setSelectedStrategy] = useState<string>('all');
  const [isSettingsVisible, setIsSettingsVisible] = useState(false);
  const [isFullscreen, setIsFullscreen] = useState(false);
  const [alertsCount, setAlertsCount] = useState(0);
  const [performanceData, setPerformanceData] = useState<any[]>([]);
  const [riskData, setRiskData] = useState<any[]>([]);

  // Custom hooks
  const realTimeAnalytics = useRealTimeAnalytics({
    portfolioId,
    updateInterval: settings.updateInterval,
    enableStreaming: showStreaming,
    autoStart: settings.autoRefresh
  });

  const performanceMetrics = usePerformanceMetrics({
    portfolioId,
    autoRefresh: settings.autoRefresh,
    refreshInterval: settings.updateInterval * 4 // Less frequent for heavy calculations
  });

  const riskAnalytics = useRiskAnalytics({
    autoRefresh: settings.autoRefresh,
    refreshInterval: settings.updateInterval * 2,
    enableRealTimeMonitoring: true,
    alertThresholds: settings.riskThresholds
  });

  const strategyAnalytics = useStrategyAnalytics({
    portfolioId,
    enableRealTime: true,
    updateInterval: settings.updateInterval
  });

  const executionAnalytics = useExecutionAnalytics({
    portfolioId,
    realTimeEnabled: true,
    updateInterval: settings.updateInterval
  });

  const websocketManager = useWebSocketManager();
  const messageBus = useMessageBus();

  // Performance stats
  const performanceStats = useMemo(() => {
    return realTimeAnalytics.getPerformanceStats();
  }, [realTimeAnalytics]);

  // Connection status
  const connectionStatus = useMemo(() => {
    return {
      websocket: websocketManager.connectionStatus,
      messageBus: messageBus.connectionStatus,
      realTimeAnalytics: realTimeAnalytics.isConnected,
      lastUpdate: realTimeAnalytics.lastUpdate
    };
  }, [websocketManager.connectionStatus, messageBus.connectionStatus, realTimeAnalytics.isConnected, realTimeAnalytics.lastUpdate]);

  // Key performance metrics
  const keyMetrics = useMemo(() => {
    const current = realTimeAnalytics.currentData;
    const trends = realTimeAnalytics.getTrends();
    
    if (!current) return [];

    return [
      {
        id: 'total_pnl',
        title: 'Total P&L',
        value: current.pnl.total,
        prefix: '$',
        precision: 2,
        trend: trends.pnl_trend > 0 ? 'up' : trends.pnl_trend < 0 ? 'down' : 'stable',
        change: current.pnl.daily_change,
        changePercent: current.pnl.daily_change_pct,
        color: current.pnl.total >= 0 ? '#52c41a' : '#ff4d4f'
      },
      {
        id: 'unrealized_pnl',
        title: 'Unrealized P&L',
        value: current.pnl.unrealized,
        prefix: '$',
        precision: 2,
        trend: current.pnl.unrealized >= 0 ? 'up' : 'down',
        change: current.pnl.unrealized,
        changePercent: (current.pnl.unrealized / Math.abs(current.pnl.total) * 100) || 0,
        color: current.pnl.unrealized >= 0 ? '#52c41a' : '#ff4d4f'
      },
      {
        id: 'sharpe_ratio',
        title: 'Sharpe Ratio',
        value: current.performance.sharpe_ratio,
        precision: 3,
        trend: trends.sharpe_trend > 0 ? 'up' : trends.sharpe_trend < 0 ? 'down' : 'stable',
        change: trends.sharpe_trend,
        changePercent: (trends.sharpe_trend / current.performance.sharpe_ratio * 100) || 0,
        color: current.performance.sharpe_ratio > 1 ? '#52c41a' : current.performance.sharpe_ratio > 0.5 ? '#faad14' : '#ff4d4f'
      },
      {
        id: 'max_drawdown',
        title: 'Max Drawdown',
        value: current.risk_metrics.max_drawdown,
        suffix: '%',
        precision: 2,
        trend: current.risk_metrics.max_drawdown > -5 ? 'up' : 'down',
        change: current.risk_metrics.max_drawdown,
        changePercent: 0,
        color: current.risk_metrics.max_drawdown > -5 ? '#52c41a' : current.risk_metrics.max_drawdown > -10 ? '#faad14' : '#ff4d4f'
      },
      {
        id: 'var_95',
        title: 'VaR 95%',
        value: Math.abs(current.risk_metrics.var_1d),
        prefix: '$',
        precision: 0,
        trend: Math.abs(current.risk_metrics.var_1d) < settings.riskThresholds.var95 ? 'up' : 'down',
        change: current.risk_metrics.var_1d,
        changePercent: (Math.abs(current.risk_metrics.var_1d) / settings.riskThresholds.var95 * 100),
        color: Math.abs(current.risk_metrics.var_1d) < settings.riskThresholds.var95 * 0.8 ? '#52c41a' : 
               Math.abs(current.risk_metrics.var_1d) < settings.riskThresholds.var95 ? '#faad14' : '#ff4d4f'
      },
      {
        id: 'exposure',
        title: 'Net Exposure',
        value: current.positions.net_exposure,
        prefix: '$',
        precision: 0,
        trend: trends.exposure_trend > 0 ? 'up' : trends.exposure_trend < 0 ? 'down' : 'stable',
        change: trends.exposure_trend,
        changePercent: (trends.exposure_trend / current.positions.net_exposure * 100) || 0,
        color: '#1890ff'
      },
      {
        id: 'leverage',
        title: 'Leverage',
        value: current.positions.leverage,
        suffix: 'x',
        precision: 2,
        trend: current.positions.leverage < 2 ? 'up' : 'down',
        change: current.positions.leverage,
        changePercent: (current.positions.leverage / settings.riskThresholds.leverage * 100),
        color: current.positions.leverage < 2 ? '#52c41a' : current.positions.leverage < 3 ? '#faad14' : '#ff4d4f'
      },
      {
        id: 'fill_rate',
        title: 'Fill Rate',
        value: current.execution.fill_rate,
        suffix: '%',
        precision: 1,
        trend: current.execution.fill_rate > 95 ? 'up' : 'down',
        change: current.execution.fill_rate,
        changePercent: current.execution.fill_rate,
        color: current.execution.fill_rate > 95 ? '#52c41a' : current.execution.fill_rate > 90 ? '#faad14' : '#ff4d4f'
      }
    ];
  }, [realTimeAnalytics.currentData, realTimeAnalytics.getTrends, settings.riskThresholds]);

  // Chart data preparation
  const chartData = useMemo(() => {
    const historical = realTimeAnalytics.historicalData;
    const timeRangeMinutes = {
      '1M': 1,
      '5M': 5,
      '15M': 15,
      '1H': 60,
      '4H': 240,
      '1D': 1440
    }[selectedTimeRange];

    const cutoffTime = dayjs().subtract(timeRangeMinutes, 'minute');
    
    return historical
      .filter(item => dayjs(item.timestamp).isAfter(cutoffTime))
      .map((item, index) => ({
        timestamp: dayjs(item.timestamp).format('HH:mm:ss'),
        time: dayjs(item.timestamp).valueOf(),
        totalPnL: item.pnl.total,
        unrealizedPnL: item.pnl.unrealized,
        realizedPnL: item.pnl.realized,
        sharpeRatio: item.performance.sharpe_ratio,
        var95: Math.abs(item.risk_metrics.var_1d),
        leverage: item.positions.leverage,
        netExposure: item.positions.net_exposure,
        index
      }))
      .slice(-100); // Keep last 100 points for performance
  }, [realTimeAnalytics.historicalData, selectedTimeRange]);

  // Alert monitoring
  useEffect(() => {
    if (riskAnalytics.riskAlerts) {
      const activeAlerts = riskAnalytics.riskAlerts.filter(alert => 
        dayjs(alert.timestamp).isAfter(dayjs().subtract(1, 'hour'))
      );
      setAlertsCount(activeAlerts.length);
    }
  }, [riskAnalytics.riskAlerts]);

  // Export functionality
  const handleExport = useCallback(async (format: 'pdf' | 'excel' | 'csv' | 'json') => {
    try {
      const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8001';
      const response = await fetch(`${API_BASE_URL}/api/v1/analytics/export`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          portfolio_id: portfolioId,
          format,
          data_types: ['performance', 'risk', 'execution'],
          time_range: selectedTimeRange,
          include_charts: true
        }),
      });

      if (response.ok) {
        const blob = await response.blob();
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.style.display = 'none';
        a.href = url;
        a.download = `analytics-report-${portfolioId}-${dayjs().format('YYYY-MM-DD-HH-mm-ss')}.${format}`;
        document.body.appendChild(a);
        a.click();
        window.URL.revokeObjectURL(url);
        document.body.removeChild(a);
      }
    } catch (error) {
      console.error('Export failed:', error);
    }
  }, [portfolioId, selectedTimeRange]);

  // Render connection status indicator
  const renderConnectionStatus = () => (
    <Space>
      <Badge 
        status={connectionStatus.websocket === 'connected' ? 'success' : 'error'} 
        text="WebSocket" 
      />
      <Badge 
        status={connectionStatus.messageBus === 'connected' ? 'success' : 'error'} 
        text="MessageBus" 
      />
      <Badge 
        status={connectionStatus.realTimeAnalytics ? 'processing' : 'default'} 
        text="Analytics" 
      />
      {connectionStatus.lastUpdate && (
        <Text type="secondary" style={{ fontSize: '11px' }}>
          Last: {dayjs(connectionStatus.lastUpdate).format('HH:mm:ss')}
        </Text>
      )}
    </Space>
  );

  // Render performance metrics cards
  const renderMetricsCards = () => (
    <Row gutter={[16, 16]}>
      {keyMetrics.map((metric) => (
        <Col xs={24} sm={12} md={6} key={metric.id}>
          <Card size="small" style={{ height: '100%' }}>
            <Statistic
              title={
                <Space>
                  <Text style={{ fontSize: '12px', fontWeight: 'normal' }}>
                    {metric.title}
                  </Text>
                  <div style={{ color: metric.color, fontSize: '14px' }}>
                    {metric.trend === 'up' ? <RiseOutlined /> : 
                     metric.trend === 'down' ? <FallOutlined /> : 
                     <DashboardOutlined />}
                  </div>
                </Space>
              }
              value={metric.value}
              prefix={metric.prefix}
              suffix={metric.suffix}
              precision={metric.precision}
              valueStyle={{ 
                color: metric.color, 
                fontSize: compactMode ? '16px' : '20px',
                fontWeight: 'bold'
              }}
            />
            {!compactMode && (
              <div style={{ marginTop: '8px', fontSize: '11px' }}>
                <Text type={metric.change >= 0 ? 'success' : 'danger'}>
                  {metric.change >= 0 ? '+' : ''}{metric.change.toFixed(2)}
                  {' '}({metric.changePercent.toFixed(1)}%)
                </Text>
              </div>
            )}
          </Card>
        </Col>
      ))}
    </Row>
  );

  // Render main chart
  const renderMainChart = () => {
    if (!chartData.length) {
      return (
        <div style={{ 
          height: 400, 
          display: 'flex', 
          alignItems: 'center', 
          justifyContent: 'center',
          color: '#999'
        }}>
          <Space direction="vertical" align="center">
            <SyncOutlined spin style={{ fontSize: '48px' }} />
            <Text type="secondary">Loading real-time data...</Text>
          </Space>
        </div>
      );
    }

    const config = {
      data: chartData,
      height: 400,
      xField: 'timestamp',
      yField: activeTab === 'pnl' ? 'totalPnL' : 
              activeTab === 'risk' ? 'var95' : 
              activeTab === 'execution' ? 'leverage' : 'totalPnL',
      smooth: true,
      animation: {
        appear: {
          animation: 'wave-in',
          duration: 1000,
        },
      },
      point: {
        size: 2,
        shape: 'circle',
      },
      tooltip: {
        formatter: (datum: any) => {
          return {
            name: activeTab === 'pnl' ? 'Total P&L' : 
                  activeTab === 'risk' ? 'VaR 95%' : 
                  activeTab === 'execution' ? 'Leverage' : 'Value',
            value: `${datum.totalPnL >= 0 ? '$' : '-$'}${Math.abs(datum.totalPnL).toFixed(2)}`,
          };
        },
      },
      xAxis: {
        tickCount: 8,
        label: {
          style: { fontSize: 10 },
        },
      },
      yAxis: {
        label: {
          formatter: (value: string) => {
            const num = parseFloat(value);
            if (activeTab === 'pnl') return `$${(num/1000).toFixed(0)}K`;
            if (activeTab === 'risk') return `$${(num/1000).toFixed(0)}K`;
            return num.toFixed(1);
          },
          style: { fontSize: 10 },
        },
      },
    };

    return <Line {...config} />;
  };

  // Render risk metrics heatmap
  const renderRiskHeatmap = () => {
    if (!realTimeAnalytics.currentData) return null;

    const riskData = [
      { metric: 'VaR 95%', value: Math.abs(realTimeAnalytics.currentData.risk_metrics.var_1d) },
      { metric: 'VaR 99%', value: Math.abs(realTimeAnalytics.currentData.risk_metrics.var_5d) },
      { metric: 'Expected Shortfall', value: realTimeAnalytics.currentData.risk_metrics.expected_shortfall },
      { metric: 'Beta', value: realTimeAnalytics.currentData.risk_metrics.beta },
      { metric: 'Volatility', value: realTimeAnalytics.currentData.risk_metrics.volatility },
      { metric: 'Max Drawdown', value: Math.abs(realTimeAnalytics.currentData.risk_metrics.max_drawdown) },
    ].map(item => ({
      ...item,
      normalized: item.value / Math.max(...riskData.map(d => d.value)) * 100,
      status: item.value < 50 ? 'low' : item.value < 100 ? 'medium' : 'high'
    }));

    return (
      <Row gutter={[8, 8]}>
        {riskData.map((item, index) => (
          <Col span={8} key={index}>
            <Card size="small" style={{ textAlign: 'center' }}>
              <Progress
                type="dashboard"
                percent={item.normalized}
                width={60}
                strokeColor={{
                  '0%': '#52c41a',
                  '50%': '#faad14',
                  '100%': '#ff4d4f',
                }}
                format={() => item.value.toFixed(1)}
              />
              <div style={{ marginTop: 8, fontSize: '11px', fontWeight: 'bold' }}>
                {item.metric}
              </div>
            </Card>
          </Col>
        ))}
      </Row>
    );
  };

  // Render alerts panel
  const renderAlerts = () => {
    if (!riskAnalytics.riskAlerts?.length) {
      return (
        <Alert
          message="All systems normal"
          description="No active risk alerts"
          type="success"
          showIcon
        />
      );
    }

    const recentAlerts = riskAnalytics.riskAlerts
      .filter(alert => dayjs(alert.timestamp).isAfter(dayjs().subtract(1, 'hour')))
      .slice(0, 5);

    return (
      <Timeline>
        {recentAlerts.map((alert) => (
          <Timeline.Item
            key={alert.id}
            color={
              alert.severity === 'critical' ? 'red' :
              alert.severity === 'high' ? 'orange' :
              alert.severity === 'medium' ? 'blue' : 'green'
            }
            dot={
              alert.severity === 'critical' ? <WarningOutlined /> :
              <BellOutlined />
            }
          >
            <div>
              <Text strong>{alert.message}</Text>
              <br />
              <Text type="secondary" style={{ fontSize: '11px' }}>
                {dayjs(alert.timestamp).format('HH:mm:ss')} - 
                Current: {alert.current_value.toFixed(2)} | 
                Threshold: {alert.threshold_value.toFixed(2)}
              </Text>
            </div>
          </Timeline.Item>
        ))}
      </Timeline>
    );
  };

  // Settings panel
  const renderSettings = () => (
    <Drawer
      title="Dashboard Settings"
      placement="right"
      onClose={() => setIsSettingsVisible(false)}
      visible={isSettingsVisible}
      width={400}
    >
      <Form layout="vertical">
        <Form.Item label="Update Interval (ms)">
          <InputNumber
            value={settings.updateInterval}
            onChange={(value) => setSettings(prev => ({ ...prev, updateInterval: value || 1000 }))}
            min={100}
            max={10000}
            step={100}
          />
        </Form.Item>
        
        <Form.Item label="Auto Refresh">
          <Switch
            checked={settings.autoRefresh}
            onChange={(checked) => setSettings(prev => ({ ...prev, autoRefresh: checked }))}
          />
        </Form.Item>

        <Form.Item label="Show Alerts">
          <Switch
            checked={settings.showAlerts}
            onChange={(checked) => setSettings(prev => ({ ...prev, showAlerts: checked }))}
          />
        </Form.Item>

        <Divider>Risk Thresholds</Divider>

        <Form.Item label="VaR 95% Threshold ($)">
          <InputNumber
            value={settings.riskThresholds.var95}
            onChange={(value) => setSettings(prev => ({
              ...prev,
              riskThresholds: { ...prev.riskThresholds, var95: value || 50000 }
            }))}
            min={1000}
            max={1000000}
            step={1000}
          />
        </Form.Item>

        <Form.Item label="Leverage Threshold">
          <InputNumber
            value={settings.riskThresholds.leverage}
            onChange={(value) => setSettings(prev => ({
              ...prev,
              riskThresholds: { ...prev.riskThresholds, leverage: value || 3.0 }
            }))}
            min={1}
            max={10}
            step={0.1}
          />
        </Form.Item>

        <Form.Item label="Concentration Threshold (%)">
          <InputNumber
            value={settings.riskThresholds.concentration}
            onChange={(value) => setSettings(prev => ({
              ...prev,
              riskThresholds: { ...prev.riskThresholds, concentration: value || 20 }
            }))}
            min={5}
            max={100}
            step={5}
          />
        </Form.Item>
      </Form>
    </Drawer>
  );

  if (compactMode) {
    return (
      <Card
        className={className}
        title={
          <Space>
            <ThunderboltOutlined style={{ color: connectionStatus.realTimeAnalytics ? '#52c41a' : '#999' }} />
            Real-time Analytics
            {connectionStatus.realTimeAnalytics && <Badge status="processing" text="Live" />}
            {alertsCount > 0 && <Badge count={alertsCount} />}
          </Space>
        }
        size="small"
        extra={renderConnectionStatus()}
      >
        {renderMetricsCards()}
      </Card>
    );
  }

  return (
    <div className={className} style={{ width: '100%' }}>
      {/* Header Controls */}
      <Card size="small" style={{ marginBottom: 16 }}>
        <Row gutter={[16, 8]} align="middle">
          <Col>
            <Space>
              <ThunderboltOutlined 
                style={{ 
                  color: connectionStatus.realTimeAnalytics ? '#52c41a' : '#999',
                  fontSize: '18px'
                }} 
              />
              <Title level={4} style={{ margin: 0 }}>
                Real-time Analytics Dashboard
              </Title>
              {connectionStatus.realTimeAnalytics && <Badge status="processing" text="Live Streaming" />}
              {alertsCount > 0 && (
                <Badge 
                  count={alertsCount} 
                  style={{ backgroundColor: '#ff4d4f' }}
                />
              )}
            </Space>
          </Col>
          
          <Col flex="auto">
            {renderConnectionStatus()}
          </Col>
          
          <Col>
            <Space>
              <Text type="secondary" style={{ fontSize: '11px' }}>
                Updates: {realTimeAnalytics.updateCount} | 
                Latency: {performanceStats.avg_latency.toFixed(1)}ms |
                Rate: {performanceStats.update_frequency.toFixed(1)}/s
              </Text>
            </Space>
          </Col>

          <Col>
            <Space>
              <Select
                value={selectedTimeRange}
                onChange={setSelectedTimeRange}
                size="small"
                style={{ width: 80 }}
              >
                <Option value="1M">1M</Option>
                <Option value="5M">5M</Option>
                <Option value="15M">15M</Option>
                <Option value="1H">1H</Option>
                <Option value="4H">4H</Option>
                <Option value="1D">1D</Option>
              </Select>
              
              <Button
                size="small"
                type={settings.autoRefresh ? "default" : "primary"}
                icon={settings.autoRefresh ? <PauseCircleOutlined /> : <PlayCircleOutlined />}
                onClick={() => setSettings(prev => ({ ...prev, autoRefresh: !prev.autoRefresh }))}
              >
                {settings.autoRefresh ? 'Pause' : 'Start'}
              </Button>
              
              <Button
                size="small"
                icon={<ReloadOutlined />}
                onClick={() => realTimeAnalytics.reset()}
              >
                Reset
              </Button>

              <Button
                size="small"
                icon={<SettingOutlined />}
                onClick={() => setIsSettingsVisible(true)}
              />

              {enableExports && (
                <Dropdown
                  overlay={
                    <Menu>
                      <Menu.Item key="pdf" icon={<DownloadOutlined />} onClick={() => handleExport('pdf')}>
                        Export PDF
                      </Menu.Item>
                      <Menu.Item key="excel" icon={<DownloadOutlined />} onClick={() => handleExport('excel')}>
                        Export Excel
                      </Menu.Item>
                      <Menu.Item key="csv" icon={<DownloadOutlined />} onClick={() => handleExport('csv')}>
                        Export CSV
                      </Menu.Item>
                      <Menu.Item key="json" icon={<DownloadOutlined />} onClick={() => handleExport('json')}>
                        Export JSON
                      </Menu.Item>
                    </Menu>
                  }
                  placement="bottomRight"
                >
                  <Button size="small" icon={<ExportOutlined />}>
                    Export
                  </Button>
                </Dropdown>
              )}

              <Button
                size="small"
                icon={<FullscreenOutlined />}
                onClick={() => setIsFullscreen(!isFullscreen)}
              />
            </Space>
          </Col>
        </Row>
      </Card>

      {/* Performance Metrics Cards */}
      <Card title="Key Performance Metrics" size="small" style={{ marginBottom: 16 }}>
        {renderMetricsCards()}
      </Card>

      {/* Main Content Tabs */}
      <Tabs 
        activeKey={activeTab} 
        onChange={setActiveTab}
        type="card"
        size="small"
      >
        <TabPane 
          tab={
            <Space>
              <LineChartOutlined />
              P&L Analysis
            </Space>
          } 
          key="pnl"
        >
          <Row gutter={[16, 16]}>
            <Col xs={24} lg={16}>
              <Card title="Real-time P&L Chart" size="small">
                {renderMainChart()}
              </Card>
            </Col>
            <Col xs={24} lg={8}>
              <Card title="P&L Breakdown" size="small" style={{ height: '450px' }}>
                {realTimeAnalytics.currentData && (
                  <Space direction="vertical" style={{ width: '100%' }}>
                    <Statistic
                      title="Realized P&L"
                      value={realTimeAnalytics.currentData.pnl.realized}
                      prefix="$"
                      precision={2}
                      valueStyle={{ color: realTimeAnalytics.currentData.pnl.realized >= 0 ? '#52c41a' : '#ff4d4f' }}
                    />
                    <Statistic
                      title="Unrealized P&L"
                      value={realTimeAnalytics.currentData.pnl.unrealized}
                      prefix="$"
                      precision={2}
                      valueStyle={{ color: realTimeAnalytics.currentData.pnl.unrealized >= 0 ? '#52c41a' : '#ff4d4f' }}
                    />
                    <Statistic
                      title="Daily Change"
                      value={realTimeAnalytics.currentData.pnl.daily_change}
                      prefix="$"
                      precision={2}
                      valueStyle={{ color: realTimeAnalytics.currentData.pnl.daily_change >= 0 ? '#52c41a' : '#ff4d4f' }}
                    />
                    <Statistic
                      title="Daily Change %"
                      value={realTimeAnalytics.currentData.pnl.daily_change_pct}
                      suffix="%"
                      precision={2}
                      valueStyle={{ color: realTimeAnalytics.currentData.pnl.daily_change_pct >= 0 ? '#52c41a' : '#ff4d4f' }}
                    />
                  </Space>
                )}
              </Card>
            </Col>
          </Row>
        </TabPane>

        <TabPane 
          tab={
            <Space>
              <WarningOutlined />
              Risk Metrics
              {alertsCount > 0 && <Badge count={alertsCount} size="small" />}
            </Space>
          } 
          key="risk"
        >
          <Row gutter={[16, 16]}>
            <Col xs={24} lg={12}>
              <Card title="Risk Heatmap" size="small">
                {renderRiskHeatmap()}
              </Card>
            </Col>
            <Col xs={24} lg={12}>
              <Card title="Risk Alerts" size="small" style={{ height: '300px', overflow: 'auto' }}>
                {renderAlerts()}
              </Card>
            </Col>
          </Row>
        </TabPane>

        <TabPane 
          tab={
            <Space>
              <StockOutlined />
              Strategy Performance
            </Space>
          } 
          key="strategy"
        >
          <Row gutter={[16, 16]}>
            <Col span={24}>
              <Card title="Strategy Analytics" size="small">
                <Alert
                  message="Strategy analytics integration in progress"
                  description="Real-time strategy performance tracking, comparison, and attribution analysis will be available here."
                  type="info"
                  showIcon
                  style={{ marginBottom: 16 }}
                />
              </Card>
            </Col>
          </Row>
        </TabPane>

        <TabPane 
          tab={
            <Space>
              <BarChartOutlined />
              Execution Quality
            </Space>
          } 
          key="execution"
        >
          <Row gutter={[16, 16]}>
            <Col xs={24} lg={16}>
              <Card title="Execution Metrics" size="small">
                {realTimeAnalytics.currentData && (
                  <Row gutter={[16, 16]}>
                    <Col span={6}>
                      <Statistic
                        title="Fill Rate"
                        value={realTimeAnalytics.currentData.execution.fill_rate}
                        suffix="%"
                        precision={1}
                        valueStyle={{ 
                          color: realTimeAnalytics.currentData.execution.fill_rate > 95 ? '#52c41a' : 
                                 realTimeAnalytics.currentData.execution.fill_rate > 90 ? '#faad14' : '#ff4d4f' 
                        }}
                      />
                    </Col>
                    <Col span={6}>
                      <Statistic
                        title="Avg Slippage"
                        value={realTimeAnalytics.currentData.execution.avg_slippage}
                        suffix=" bps"
                        precision={2}
                        valueStyle={{ 
                          color: realTimeAnalytics.currentData.execution.avg_slippage < 5 ? '#52c41a' : 
                                 realTimeAnalytics.currentData.execution.avg_slippage < 10 ? '#faad14' : '#ff4d4f' 
                        }}
                      />
                    </Col>
                    <Col span={6}>
                      <Statistic
                        title="Implementation Shortfall"
                        value={realTimeAnalytics.currentData.execution.implementation_shortfall}
                        suffix=" bps"
                        precision={2}
                        valueStyle={{ color: '#1890ff' }}
                      />
                    </Col>
                    <Col span={6}>
                      <Statistic
                        title="Market Impact"
                        value={realTimeAnalytics.currentData.execution.market_impact}
                        suffix=" bps"
                        precision={2}
                        valueStyle={{ color: '#722ed1' }}
                      />
                    </Col>
                  </Row>
                )}
              </Card>
            </Col>
            <Col xs={24} lg={8}>
              <Card title="Execution Summary" size="small">
                <Alert
                  message="Execution analytics available"
                  description="Real-time execution quality monitoring is active with sub-second updates."
                  type="success"
                  showIcon
                />
              </Card>
            </Col>
          </Row>
        </TabPane>
      </Tabs>

      {/* Settings Drawer */}
      {renderSettings()}

      {/* Error Display */}
      {realTimeAnalytics.error && (
        <Alert
          message="Real-time Analytics Error"
          description={realTimeAnalytics.error}
          type="error"
          closable
          style={{ marginTop: 16 }}
        />
      )}
    </div>
  );
};

export default RealTimeAnalyticsDashboard;