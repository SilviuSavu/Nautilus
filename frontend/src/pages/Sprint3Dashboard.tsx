import React, { useState, useEffect, lazy, Suspense, memo, useCallback, useMemo } from 'react';
import {
  Row,
  Col,
  Typography,
  Space,
  Card,
  Tabs,
  Button,
  Alert,
  Divider,
  Badge,
  Switch,
  Select,
  Tooltip,
  Spin
} from 'antd';
import {
  ThunderboltOutlined,
  DashboardOutlined,
  SettingOutlined,
  MonitorOutlined,
  ReloadOutlined,
  FullscreenOutlined,
  CompressOutlined,
  InfoCircleOutlined,
  BellOutlined,
  StarOutlined,
  WifiOutlined,
  BarChartOutlined
} from '@ant-design/icons';
import Sprint3StatusWidget from '../components/Sprint3/Sprint3StatusWidget';
import FeatureNavigation from '../components/Sprint3/FeatureNavigation';
import SystemHealthOverview from '../components/Sprint3/SystemHealthOverview';
import QuickActions from '../components/Sprint3/QuickActions';
import { WebSocketMonitor } from '../components/Infrastructure';

// Lazy load heavy components for better performance
const WebSocketMonitoringSuite = lazy(() => import('../components/WebSocket/WebSocketMonitoringSuite'));
const ConnectionHealthDashboard = lazy(() => import('../components/Monitoring/ConnectionHealthDashboard'));
const RealTimeAnalyticsDashboard = lazy(() => import('../components/Performance/RealTimeAnalyticsDashboard'));
const RiskDashboardSprint3 = lazy(() => import('../components/Risk/RiskDashboardSprint3'));
const RealTimeRiskMonitor = lazy(() => import('../components/Risk/RealTimeRiskMonitor'));
const DeploymentOrchestrator = lazy(() => import('../components/Strategy/DeploymentOrchestrator'));
const AdvancedDeploymentPipeline = lazy(() => import('../components/Strategy/AdvancedDeploymentPipeline'));
const Sprint3SystemMonitor = lazy(() => import('../components/Monitoring/Sprint3SystemMonitor'));
const PrometheusMetricsDashboard = lazy(() => import('../components/Monitoring/PrometheusMetricsDashboard'));
const GrafanaEmbedDashboard = lazy(() => import('../components/Monitoring/GrafanaEmbedDashboard'));
import { useMessageBus } from '../hooks/useMessageBus';
import { useEngineWebSocket } from '../hooks/useEngineWebSocket';
import type { Sprint3StatusResponse } from '../types/sprint3';
import ErrorBoundary from '../components/ErrorBoundary';

const { Title, Text, Paragraph } = Typography;
const { TabPane } = Tabs;

interface Sprint3DashboardProps {
  defaultTab?: string;
  compactMode?: boolean;
  autoRefresh?: boolean;
}

// Loading component for lazy-loaded components
const ComponentLoader: React.FC = memo(() => (
  <div style={{ 
    display: 'flex', 
    justifyContent: 'center', 
    alignItems: 'center', 
    minHeight: '200px',
    padding: '20px'
  }}>
    <Spin size="large" tip="Loading component..." />
  </div>
));
ComponentLoader.displayName = 'ComponentLoader';

const Sprint3Dashboard: React.FC<Sprint3DashboardProps> = memo(({
  defaultTab = 'overview',
  compactMode = false,
  autoRefresh = true
}) => {
  const [activeTab, setActiveTab] = useState(defaultTab);
  const [fullscreen, setFullscreen] = useState(false);
  const [viewMode, setViewMode] = useState<'standard' | 'compact' | 'detailed'>('standard');
  const [refreshInterval, setRefreshInterval] = useState(30000);
  const [showNotifications, setShowNotifications] = useState(true);

  const messageBus = useMessageBus();
  const engineWs = useEngineWebSocket();

  // Handle feature navigation
  const handleFeatureClick = useCallback((featureId: string) => {
    console.log('Feature clicked:', featureId);
    
    // Route to specific feature based on ID
    switch (featureId) {
      case 'websocket-monitor':
      case 'message-streaming':
        setActiveTab('infrastructure');
        break;
      case 'realtime-dashboard':
      case 'metric-streaming':
      case 'alert-engine':
        setActiveTab('analytics');
        break;
      case 'dynamic-limits':
      case 'breach-detection':
      case 'risk-analytics':
        setActiveTab('risk');
        break;
      case 'deployment-pipeline':
      case 'testing-framework':
      case 'rollback-system':
        setActiveTab('deployment');
        break;
      case 'system-health':
      case 'performance-profiler':
        setActiveTab('monitoring');
        break;
      default:
        console.log('Unknown feature:', featureId);
    }
  }, []);

  // Handle quick actions
  const handleQuickAction = useCallback((actionId: string) => {
    console.log('Quick action executed:', actionId);
  }, []);

  // Toggle fullscreen mode
  const toggleFullscreen = useCallback(() => {
    setFullscreen(prev => !prev);
  }, []);

  // Executive Summary Component
  const ExecutiveSummary = memo(() => (
    <Card 
      title={
        <Space>
          <StarOutlined style={{ color: '#722ed1' }} />
          <Title level={4} style={{ margin: 0 }}>
            Sprint 3 Executive Summary
          </Title>
        </Space>
      }
      extra={
        <Space>
          <Text type="secondary" style={{ fontSize: '12px' }}>
            Real-time Enterprise Trading Platform
          </Text>
        </Space>
      }
    >
      <Row gutter={[24, 16]}>
        <Col xs={24} lg={12}>
          <div style={{ marginBottom: '16px' }}>
            <Title level={5} style={{ color: '#1890ff', marginBottom: '8px' }}>
              🚀 Platform Status
            </Title>
            <Paragraph style={{ marginBottom: '12px' }}>
              <Text strong>Sprint 3</Text> introduces enterprise-grade infrastructure with real-time WebSocket communication, 
              advanced analytics, dynamic risk management, and automated strategy deployment pipelines.
            </Paragraph>
            <div style={{ display: 'flex', gap: '8px', flexWrap: 'wrap' }}>
              <Badge status="success" text="WebSocket Infrastructure" />
              <Badge status="success" text="Real-time Analytics" />
              <Badge status="processing" text="Risk Management" />
              <Badge status="success" text="Strategy Deployment" />
            </div>
          </div>

          <div>
            <Title level={5} style={{ color: '#52c41a', marginBottom: '8px' }}>
              📊 Key Capabilities
            </Title>
            <ul style={{ paddingLeft: '20px', margin: 0 }}>
              <li><Text>High-performance WebSocket infrastructure with {messageBus.messagesReceived} messages processed</Text></li>
              <li><Text>Real-time analytics streaming with sub-100ms latency</Text></li>
              <li><Text>Dynamic risk limit management and breach detection</Text></li>
              <li><Text>Automated strategy deployment with approval workflows</Text></li>
              <li><Text>Comprehensive system health monitoring and alerting</Text></li>
            </ul>
          </div>
        </Col>

        <Col xs={24} lg={12}>
          <Sprint3StatusWidget 
            size="default"
            showDetails={true}
            onFeatureClick={handleFeatureClick}
          />
        </Col>
      </Row>

      {showNotifications && (
        <Alert
          message="Sprint 3 Platform Active"
          description={
            <Space direction="vertical" style={{ width: '100%' }}>
              <Text>
                All Sprint 3 systems are operational. WebSocket infrastructure is handling real-time data streams, 
                analytics engines are processing market data, and risk management systems are actively monitoring positions.
              </Text>
              <Space>
                <Badge status="success" text={`${messageBus.connectionStatus === 'connected' ? 'Connected' : 'Disconnected'} to MessageBus`} />
                <Badge status={engineWs.isConnected ? 'success' : 'error'} text={`Engine WebSocket ${engineWs.isConnected ? 'Connected' : 'Disconnected'}`} />
                <Badge status="processing" text="Live Analytics Active" />
              </Space>
            </Space>
          }
          type="success"
          showIcon
          closable
          onClose={() => setShowNotifications(false)}
          style={{ marginTop: 16 }}
        />
      )}
    </Card>
  ));
  ExecutiveSummary.displayName = 'ExecutiveSummary';

  const containerStyle = useMemo(() => ({
    width: '100%',
    height: fullscreen ? '100vh' : 'auto',
    overflow: fullscreen ? 'auto' : 'visible',
    position: fullscreen ? 'fixed' : 'relative',
    top: fullscreen ? 0 : 'auto',
    left: fullscreen ? 0 : 'auto',
    zIndex: fullscreen ? 1000 : 'auto',
    backgroundColor: fullscreen ? '#fff' : 'transparent',
    padding: fullscreen ? '16px' : '0'
  } as React.CSSProperties), [fullscreen]);

  return (
    <div style={containerStyle}>
      {/* Header */}
      <div style={{ 
        display: 'flex', 
        justifyContent: 'space-between', 
        alignItems: 'center',
        marginBottom: '24px' 
      }}>
        <Space>
          <ThunderboltOutlined style={{ fontSize: '24px', color: '#722ed1' }} />
          <Title level={2} style={{ margin: 0 }}>
            Sprint 3 Dashboard
          </Title>
          <Badge count="Enterprise" style={{ backgroundColor: '#722ed1' }} />
        </Space>
        
        <Space>
          {/* View Mode Selector */}
          <Select
            value={viewMode}
            onChange={setViewMode}
            size="small"
            style={{ width: 120 }}
          >
            <Select.Option value="compact">Compact</Select.Option>
            <Select.Option value="standard">Standard</Select.Option>
            <Select.Option value="detailed">Detailed</Select.Option>
          </Select>

          {/* Auto-refresh Toggle */}
          <Tooltip title="Auto-refresh data">
            <Switch 
              checked={autoRefresh}
              checkedChildren="Auto"
              unCheckedChildren="Manual"
              size="small"
            />
          </Tooltip>

          {/* Refresh Interval */}
          <Select
            value={refreshInterval}
            onChange={setRefreshInterval}
            size="small"
            style={{ width: 80 }}
          >
            <Select.Option value={5000}>5s</Select.Option>
            <Select.Option value={15000}>15s</Select.Option>
            <Select.Option value={30000}>30s</Select.Option>
            <Select.Option value={60000}>1m</Select.Option>
          </Select>

          {/* Controls */}
          <Button 
            size="small" 
            icon={<ReloadOutlined />}
            onClick={() => {
              messageBus.clearMessages();
              engineWs.requestEngineStatus();
            }}
          >
            Refresh
          </Button>
          
          <Button 
            size="small" 
            icon={fullscreen ? <CompressOutlined /> : <FullscreenOutlined />}
            onClick={toggleFullscreen}
          />
          
          <Button size="small" icon={<SettingOutlined />} />
        </Space>
      </div>

      {/* Main Content */}
      <Tabs 
        activeKey={activeTab}
        onChange={setActiveTab}
        size="small"
        tabBarStyle={{ marginBottom: '16px' }}
      >
        {/* Overview Tab */}
        <TabPane
          tab={
            <Space>
              <DashboardOutlined />
              Overview
            </Space>
          }
          key="overview"
        >
          <Row gutter={[16, 16]}>
            <Col xs={24}>
              <ExecutiveSummary />
            </Col>
            
            <Col xs={24} lg={16}>
              <FeatureNavigation
                onFeatureClick={handleFeatureClick}
                onQuickAction={handleQuickAction}
                showQuickActions={viewMode !== 'compact'}
                compactMode={viewMode === 'compact'}
              />
            </Col>

            {viewMode !== 'compact' && (
              <Col xs={24} lg={8}>
                <QuickActions
                  onActionExecuted={handleQuickAction}
                  layout="list"
                  showStatus={true}
                />
              </Col>
            )}
          </Row>
        </TabPane>

        {/* Infrastructure Tab */}
        <TabPane
          tab={
            <Space>
              <MonitorOutlined />
              Infrastructure
              <Badge status="success" />
            </Space>
          }
          key="infrastructure"
        >
          <ErrorBoundary
            fallbackTitle="Infrastructure Monitoring Error"
            fallbackMessage="The infrastructure monitoring system encountered an error. This may be due to WebSocket connectivity or monitoring service issues."
          >
            <Tabs 
              size="small" 
              defaultActiveKey="websocket"
              tabBarStyle={{ marginBottom: '16px' }}
            >
              <Tabs.TabPane 
                tab={
                  <Space>
                    <WifiOutlined />
                    WebSocket Infrastructure
                  </Space>
                }
                key="websocket"
              >
                <Row gutter={[16, 16]}>
                  <Col xs={24} lg={16}>
                    <Suspense fallback={<ComponentLoader />}>
                      <WebSocketMonitoringSuite
                        showDetailedMetrics={viewMode === 'detailed'}
                        compactMode={viewMode === 'compact'}
                        autoRefresh={autoRefresh}
                        refreshInterval={refreshInterval}
                      />
                    </Suspense>
                  </Col>
                  <Col xs={24} lg={8}>
                    <Suspense fallback={<ComponentLoader />}>
                      <ConnectionHealthDashboard
                        compactMode={viewMode === 'compact'}
                        autoRefresh={autoRefresh}
                      />
                    </Suspense>
                  </Col>
                </Row>
              </Tabs.TabPane>
              <Tabs.TabPane 
                tab={
                  <Space>
                    <MonitorOutlined />
                    Legacy Monitor
                  </Space>
                }
                key="legacy"
              >
                <WebSocketMonitor
                  showDetailedMetrics={viewMode === 'detailed'}
                  compactMode={viewMode === 'compact'}
                  autoRefresh={autoRefresh}
                  refreshInterval={refreshInterval}
                />
              </Tabs.TabPane>
            </Tabs>
          </ErrorBoundary>
        </TabPane>

        {/* System Health Tab */}
        <TabPane
          tab={
            <Space>
              <InfoCircleOutlined />
              System Health
              {messageBus.connectionStatus !== 'connected' && (
                <Badge status="warning" />
              )}
            </Space>
          }
          key="monitoring"
        >
          <ErrorBoundary
            fallbackTitle="System Health Monitoring Error"
            fallbackMessage="The system health monitoring encountered an error. This may be due to monitoring service connectivity or metrics collection issues."
          >
            <Tabs 
              size="small" 
              defaultActiveKey="overview"
              tabBarStyle={{ marginBottom: '16px' }}
            >
              <Tabs.TabPane 
                tab={
                  <Space>
                    <DashboardOutlined />
                    System Overview
                  </Space>
                }
                key="overview"
              >
                <Row gutter={[16, 16]}>
                  <Col xs={24} lg={16}>
                    <Suspense fallback={<ComponentLoader />}>
                      <Sprint3SystemMonitor
                        autoRefresh={autoRefresh}
                        refreshInterval={refreshInterval}
                        showAlerts={true}
                        compactMode={viewMode === 'compact'}
                        showAdvancedMetrics={viewMode === 'detailed'}
                      />
                    </Suspense>
                  </Col>
                  <Col xs={24} lg={8}>
                    <SystemHealthOverview
                      autoRefresh={autoRefresh}
                      refreshInterval={refreshInterval}
                      showAlerts={true}
                      compactMode={viewMode === 'compact'}
                    />
                  </Col>
                </Row>
              </Tabs.TabPane>
              <Tabs.TabPane 
                tab={
                  <Space>
                    <BarChartOutlined />
                    Prometheus Metrics
                  </Space>
                }
                key="prometheus"
              >
                <Suspense fallback={<ComponentLoader />}>
                  <PrometheusMetricsDashboard
                    compactMode={viewMode === 'compact'}
                    autoRefresh={autoRefresh}
                    refreshInterval={refreshInterval}
                    showAdvancedQueries={viewMode === 'detailed'}
                  />
                </Suspense>
              </Tabs.TabPane>
              <Tabs.TabPane 
                tab={
                  <Space>
                    <MonitorOutlined />
                    Grafana Dashboard
                  </Space>
                }
                key="grafana"
              >
                <Suspense fallback={<ComponentLoader />}>
                  <GrafanaEmbedDashboard
                    dashboardId="sprint3-overview"
                    height={viewMode === 'compact' ? 400 : 600}
                    autoRefresh={autoRefresh}
                    refreshInterval={refreshInterval}
                  />
                </Suspense>
              </Tabs.TabPane>
            </Tabs>
          </ErrorBoundary>
        </TabPane>

        {/* Analytics Tab */}
        <TabPane
          tab={
            <Space>
              <DashboardOutlined />
              Real-time Analytics
              <Badge status="success" />
            </Space>
          }
          key="analytics"
        >
          <ErrorBoundary
            fallbackTitle="Real-time Analytics Error"
            fallbackMessage="The real-time analytics dashboard encountered an error. This may be due to data streaming issues or calculation problems."
          >
            <Suspense fallback={<ComponentLoader />}>
              <RealTimeAnalyticsDashboard
                portfolioId="default"
                showStreaming={true}
                compactMode={viewMode === 'compact'}
                autoRefresh={autoRefresh}
                refreshInterval={refreshInterval}
                showAdvancedMetrics={viewMode === 'detailed'}
              />
            </Suspense>
          </ErrorBoundary>
        </TabPane>

        {/* Risk Management Tab */}
        <TabPane
          tab={
            <Space>
              <BellOutlined />
              Risk Management
              <Badge status="success" />
            </Space>
          }
          key="risk"
        >
          <ErrorBoundary
            fallbackTitle="Risk Management Error"
            fallbackMessage="The risk management system encountered an error. This may be due to risk engine connectivity or calculation issues."
          >
            <Row gutter={[16, 16]}>
              <Col xs={24} lg={16}>
                <Suspense fallback={<ComponentLoader />}>
                  <RiskDashboardSprint3
                    portfolioId="default"
                    showAdvancedMetrics={viewMode === 'detailed'}
                    compactMode={viewMode === 'compact'}
                    autoRefresh={autoRefresh}
                    refreshInterval={refreshInterval}
                  />
                </Suspense>
              </Col>
              <Col xs={24} lg={8}>
                <Suspense fallback={<ComponentLoader />}>
                  <RealTimeRiskMonitor
                    portfolioId="default"
                    showAlerts={true}
                    compactMode={viewMode === 'compact'}
                    autoRefresh={autoRefresh}
                  />
                </Suspense>
              </Col>
            </Row>
          </ErrorBoundary>
        </TabPane>

        {/* Strategy Deployment Tab */}
        <TabPane
          tab={
            <Space>
              <ThunderboltOutlined />
              Strategy Deployment
              <Badge status="success" />
            </Space>
          }
          key="deployment"
        >
          <ErrorBoundary
            fallbackTitle="Strategy Deployment Error"
            fallbackMessage="The strategy deployment system encountered an error. This may be due to pipeline connectivity or approval workflow issues."
          >
            <Tabs 
              size="small" 
              defaultActiveKey="orchestrator"
              tabBarStyle={{ marginBottom: '16px' }}
            >
              <Tabs.TabPane 
                tab={
                  <Space>
                    <ThunderboltOutlined />
                    Deployment Orchestrator
                  </Space>
                }
                key="orchestrator"
              >
                <Suspense fallback={<ComponentLoader />}>
                  <DeploymentOrchestrator
                    compactMode={viewMode === 'compact'}
                    showAdvancedControls={viewMode === 'detailed'}
                    autoRefresh={autoRefresh}
                  />
                </Suspense>
              </Tabs.TabPane>
              <Tabs.TabPane 
                tab={
                  <Space>
                    <DashboardOutlined />
                    Advanced Pipeline
                  </Space>
                }
                key="pipeline"
              >
                <Suspense fallback={<ComponentLoader />}>
                  <AdvancedDeploymentPipeline
                    compactMode={viewMode === 'compact'}
                    showDetailedMetrics={viewMode === 'detailed'}
                    autoRefresh={autoRefresh}
                    refreshInterval={refreshInterval}
                  />
                </Suspense>
              </Tabs.TabPane>
            </Tabs>
          </ErrorBoundary>
        </TabPane>
      </Tabs>

      {/* Footer Status Bar */}
      {viewMode !== 'compact' && (
        <Card size="small" style={{ marginTop: 16, backgroundColor: '#fafafa' }}>
          <Row gutter={[16, 8]} align="middle">
            <Col xs={24} sm={12} md={8}>
              <Space size="small">
                <Text type="secondary" style={{ fontSize: '11px' }}>Sprint 3 Status:</Text>
                <Badge status="success" text="Active" />
                <Text type="secondary" style={{ fontSize: '11px' }}>
                  WebSocket: {messageBus.messagesReceived} msgs
                </Text>
              </Space>
            </Col>
            <Col xs={24} sm={12} md={8}>
              <Space size="small">
                <Text type="secondary" style={{ fontSize: '11px' }}>System Health:</Text>
                <Badge 
                  status={messageBus.connectionStatus === 'connected' && engineWs.isConnected ? 'success' : 'warning'} 
                  text={messageBus.connectionStatus === 'connected' && engineWs.isConnected ? 'Healthy' : 'Warning'} 
                />
              </Space>
            </Col>
            <Col xs={24} sm={12} md={8}>
              <Space size="small">
                <Text type="secondary" style={{ fontSize: '11px' }}>
                  Last Update: {new Date().toLocaleTimeString()}
                </Text>
                {autoRefresh && (
                  <Text type="secondary" style={{ fontSize: '11px' }}>
                    • Auto-refresh: {refreshInterval / 1000}s
                  </Text>
                )}
              </Space>
            </Col>
          </Row>
        </Card>
      )}
    </div>
  );
});
Sprint3Dashboard.displayName = 'Sprint3Dashboard';

export default Sprint3Dashboard;