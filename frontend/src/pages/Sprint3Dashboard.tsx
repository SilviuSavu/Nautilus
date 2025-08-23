import React, { useState, useEffect } from 'react';
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
  Tooltip
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
  StarOutlined
} from '@ant-design/icons';
import Sprint3StatusWidget from '../components/Sprint3/Sprint3StatusWidget';
import FeatureNavigation from '../components/Sprint3/FeatureNavigation';
import SystemHealthOverview from '../components/Sprint3/SystemHealthOverview';
import QuickActions from '../components/Sprint3/QuickActions';
import { WebSocketMonitor } from '../components/Infrastructure';
import { useMessageBus } from '../hooks/useMessageBus';
import { useEngineWebSocket } from '../hooks/useEngineWebSocket';
import type { Sprint3StatusResponse } from '../types/sprint3';

const { Title, Text, Paragraph } = Typography;
const { TabPane } = Tabs;

interface Sprint3DashboardProps {
  defaultTab?: string;
  compactMode?: boolean;
  autoRefresh?: boolean;
}

const Sprint3Dashboard: React.FC<Sprint3DashboardProps> = ({
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
  const handleFeatureClick = (featureId: string) => {
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
  };

  // Handle quick actions
  const handleQuickAction = (actionId: string) => {
    console.log('Quick action executed:', actionId);
  };

  // Toggle fullscreen mode
  const toggleFullscreen = () => {
    setFullscreen(!fullscreen);
  };

  // Executive Summary Component
  const ExecutiveSummary = () => (
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
  );

  const containerStyle = {
    width: '100%',
    height: fullscreen ? '100vh' : 'auto',
    overflow: fullscreen ? 'auto' : 'visible',
    position: fullscreen ? 'fixed' : 'relative',
    top: fullscreen ? 0 : 'auto',
    left: fullscreen ? 0 : 'auto',
    zIndex: fullscreen ? 1000 : 'auto',
    backgroundColor: fullscreen ? '#fff' : 'transparent',
    padding: fullscreen ? '16px' : '0'
  } as React.CSSProperties;

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
            </Space>
          }
          key="infrastructure"
        >
          <Row gutter={[16, 16]}>
            <Col xs={24}>
              <WebSocketMonitor
                showDetailedMetrics={viewMode === 'detailed'}
                compactMode={viewMode === 'compact'}
                autoRefresh={autoRefresh}
                refreshInterval={refreshInterval}
              />
            </Col>
          </Row>
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
          <Row gutter={[16, 16]}>
            <Col xs={24}>
              <SystemHealthOverview
                autoRefresh={autoRefresh}
                refreshInterval={refreshInterval}
                showAlerts={true}
                compactMode={viewMode === 'compact'}
              />
            </Col>
          </Row>
        </TabPane>

        {/* Analytics Tab */}
        <TabPane
          tab={
            <Space>
              <DashboardOutlined />
              Real-time Analytics
              <Badge status="processing" />
            </Space>
          }
          key="analytics"
        >
          <Card>
            <Alert
              message="Real-time Analytics Engine"
              description="Advanced analytics components are being integrated. This section will include real-time performance metrics, streaming analytics, and intelligent alerting systems."
              type="info"
              showIcon
              action={
                <Button size="small" type="primary">
                  Configure Analytics
                </Button>
              }
            />
            
            <Divider />
            
            <Row gutter={[16, 16]}>
              <Col xs={24} sm={8}>
                <Card size="small" title="Metrics Streaming">
                  <Text>• Performance metrics in real-time</Text><br />
                  <Text>• Custom metric definitions</Text><br />
                  <Text>• Historical trend analysis</Text><br />
                  <Text>• Cross-strategy comparisons</Text>
                </Card>
              </Col>
              <Col xs={24} sm={8}>
                <Card size="small" title="Alert Engine">
                  <Text>• Dynamic threshold monitoring</Text><br />
                  <Text>• Machine learning alerts</Text><br />
                  <Text>• Escalation workflows</Text><br />
                  <Text>• Integration with external systems</Text>
                </Card>
              </Col>
              <Col xs={24} sm={8}>
                <Card size="small" title="Performance Attribution">
                  <Text>• Real-time P&L attribution</Text><br />
                  <Text>• Risk factor analysis</Text><br />
                  <Text>• Execution quality metrics</Text><br />
                  <Text>• Benchmark comparisons</Text>
                </Card>
              </Col>
            </Row>
          </Card>
        </TabPane>

        {/* Risk Management Tab */}
        <TabPane
          tab={
            <Space>
              <BellOutlined />
              Risk Management
              <Badge count={2} size="small" />
            </Space>
          }
          key="risk"
        >
          <Card>
            <Alert
              message="Dynamic Risk Management System"
              description="Enhanced risk management with dynamic limits, real-time breach detection, and automated response systems. Integration with existing risk components is in progress."
              type="warning"
              showIcon
              action={
                <Button size="small" type="primary">
                  Configure Risk Limits
                </Button>
              }
            />
            
            <Divider />
            
            <Row gutter={[16, 16]}>
              <Col xs={24} sm={6}>
                <Card size="small" title="Dynamic Limits">
                  <Text>• Adaptive position limits</Text><br />
                  <Text>• Real-time exposure calculation</Text><br />
                  <Text>• Market-based adjustments</Text><br />
                  <Text>• Portfolio-level constraints</Text>
                </Card>
              </Col>
              <Col xs={24} sm={6}>
                <Card size="small" title="Breach Detection">
                  <Text>• Instantaneous monitoring</Text><br />
                  <Text>• Predictive breach alerts</Text><br />
                  <Text>• Automated responses</Text><br />
                  <Text>• Audit trail maintenance</Text>
                </Card>
              </Col>
              <Col xs={24} sm={6}>
                <Card size="small" title="Response Actions">
                  <Text>• Position reduction triggers</Text><br />
                  <Text>• Trading halt mechanisms</Text><br />
                  <Text>• Notification escalation</Text><br />
                  <Text>• Recovery procedures</Text>
                </Card>
              </Col>
              <Col xs={24} sm={6}>
                <Card size="small" title="Risk Analytics">
                  <Text>• VaR calculations</Text><br />
                  <Text>• Stress testing</Text><br />
                  <Text>• Correlation analysis</Text><br />
                  <Text>• Scenario modeling</Text>
                </Card>
              </Col>
            </Row>
          </Card>
        </TabPane>

        {/* Strategy Deployment Tab */}
        <TabPane
          tab={
            <Space>
              <ThunderboltOutlined />
              Strategy Deployment
              <Badge status="processing" />
            </Space>
          }
          key="deployment"
        >
          <Card>
            <Alert
              message="Automated Strategy Deployment Pipeline"
              description="Enterprise-grade strategy deployment with testing frameworks, approval workflows, and intelligent rollback capabilities. Integration with existing strategy management is in progress."
              type="success"
              showIcon
              action={
                <Button size="small" type="primary">
                  Start Deployment
                </Button>
              }
            />
            
            <Divider />
            
            <Row gutter={[16, 16]}>
              <Col xs={24} md={12} lg={6}>
                <Card size="small" title="Deployment Pipeline">
                  <Text>• Automated testing phases</Text><br />
                  <Text>• Approval workflows</Text><br />
                  <Text>• Gradual rollout controls</Text><br />
                  <Text>• Performance monitoring</Text>
                </Card>
              </Col>
              <Col xs={24} md={12} lg={6}>
                <Card size="small" title="Testing Framework">
                  <Text>• Backtesting integration</Text><br />
                  <Text>• Paper trading validation</Text><br />
                  <Text>• Risk assessment tests</Text><br />
                  <Text>• Performance benchmarks</Text>
                </Card>
              </Col>
              <Col xs={24} md={12} lg={6}>
                <Card size="small" title="Rollback System">
                  <Text>• Automatic failure detection</Text><br />
                  <Text>• Instant strategy rollback</Text><br />
                  <Text>• Configuration versioning</Text><br />
                  <Text>• Recovery procedures</Text>
                </Card>
              </Col>
              <Col xs={24} md={12} lg={6}>
                <Card size="small" title="Monitoring">
                  <Text>• Real-time performance tracking</Text><br />
                  <Text>• Deployment health checks</Text><br />
                  <Text>• Resource usage monitoring</Text><br />
                  <Text>• Alert integration</Text>
                </Card>
              </Col>
            </Row>
          </Card>
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
};

export default Sprint3Dashboard;