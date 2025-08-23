/**
 * Sprint 3: Grafana Embed Dashboard
 * Embedded Grafana dashboards for comprehensive visualization
 * 7-panel trading overview with real-time charts and analytics
 */

import React, { useState, useEffect } from 'react';
import {
  Card,
  Row,
  Col,
  Typography,
  Space,
  Button,
  Select,
  Alert,
  Spin,
  Tag,
  Tooltip,
  Tabs,
  Switch,
  notification,
  Divider
} from 'antd';
import {
  FullscreenOutlined,
  FullscreenExitOutlined,
  ReloadOutlined,
  SettingOutlined,
  EyeOutlined,
  DashboardOutlined,
  LineChartOutlined,
  BarChartOutlined,
  PieChartOutlined,
  ApiOutlined,
  ThunderboltOutlined,
  DatabaseOutlined
} from '@ant-design/icons';

const { Title, Text } = Typography;
const { Option } = Select;
const { TabPane } = Tabs;

interface GrafanaDashboard {
  uid: string;
  title: string;
  url: string;
  description: string;
  tags: string[];
  is_starred: boolean;
  folder_title: string;
  panels_count: number;
}

interface GrafanaEmbedDashboardProps {
  className?: string;
}

export const GrafanaEmbedDashboard: React.FC<GrafanaEmbedDashboardProps> = ({
  className
}) => {
  const [dashboards, setDashboards] = useState<GrafanaDashboard[]>([]);
  const [selectedDashboard, setSelectedDashboard] = useState<string>('nautilus-overview');
  const [fullscreen, setFullscreen] = useState(false);
  const [refreshRate, setRefreshRate] = useState('30s');
  const [theme, setTheme] = useState<'light' | 'dark'>('dark');
  const [autoRefresh, setAutoRefresh] = useState(true);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [grafanaStatus, setGrafanaStatus] = useState<'online' | 'offline' | 'checking'>('checking');

  // Pre-configured Sprint 3 Nautilus dashboards
  const predefinedDashboards: GrafanaDashboard[] = [
    {
      uid: 'nautilus-overview',
      title: 'Nautilus Trading Platform Overview',
      url: '/d/nautilus-overview/nautilus-trading-platform-overview',
      description: 'Comprehensive 7-panel overview of trading platform performance',
      tags: ['trading', 'overview', 'sprint3'],
      is_starred: true,
      folder_title: 'Nautilus Trading',
      panels_count: 7
    },
    {
      uid: 'websocket-monitoring',
      title: 'WebSocket Infrastructure',
      url: '/d/websocket-monitoring/websocket-infrastructure-monitoring',
      description: '1000+ concurrent connections monitoring and message throughput',
      tags: ['websocket', 'real-time', 'infrastructure'],
      is_starred: true,
      folder_title: 'Infrastructure',
      panels_count: 8
    },
    {
      uid: 'risk-management',
      title: 'Risk Management System',
      url: '/d/risk-management/risk-management-system',
      description: 'Real-time risk monitoring, limits, and breach detection',
      tags: ['risk', 'compliance', 'monitoring'],
      is_starred: true,
      folder_title: 'Risk & Compliance',
      panels_count: 6
    },
    {
      uid: 'trading-performance',
      title: 'Trading Performance Analytics',
      url: '/d/trading-performance/trading-performance-analytics',
      description: 'P&L, execution quality, and strategy performance metrics',
      tags: ['trading', 'performance', 'analytics'],
      is_starred: true,
      folder_title: 'Trading Analytics',
      panels_count: 10
    },
    {
      uid: 'system-resources',
      title: 'System Resources & Infrastructure',
      url: '/d/system-resources/system-resources-infrastructure',
      description: 'CPU, memory, disk, network, and container resource monitoring',
      tags: ['system', 'resources', 'infrastructure'],
      is_starred: false,
      folder_title: 'System Monitoring',
      panels_count: 12
    },
    {
      uid: 'database-performance',
      title: 'Database Performance (TimescaleDB)',
      url: '/d/database-performance/timescaledb-performance-monitoring',
      description: 'TimescaleDB optimization, queries, and time-series data performance',
      tags: ['database', 'timescaledb', 'performance'],
      is_starred: false,
      folder_title: 'Database',
      panels_count: 9
    }
  ];

  const checkGrafanaStatus = async () => {
    try {
      setGrafanaStatus('checking');
      const grafanaUrl = import.meta.env.VITE_GRAFANA_URL || 'http://localhost:3002';
      const response = await fetch(`${grafanaUrl}/api/health`, {
        method: 'GET',
        mode: 'no-cors' // Handle CORS for Grafana health check
      });
      setGrafanaStatus('online');
      setError(null);
    } catch (err) {
      setGrafanaStatus('offline');
      setError('Grafana service is not accessible. Please ensure Grafana is running on port 3002.');
      console.error('Grafana health check failed:', err);
    }
  };

  const fetchGrafanaDashboards = async () => {
    try {
      setLoading(true);
      // Use predefined dashboards as fallback since Grafana API might have CORS restrictions
      setDashboards(predefinedDashboards);
      setError(null);
    } catch (err) {
      setError('Failed to load dashboard list');
      console.error('Dashboard fetch error:', err);
      // Use predefined dashboards as fallback
      setDashboards(predefinedDashboards);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    checkGrafanaStatus();
    fetchGrafanaDashboards();
  }, []);

  const handleDashboardChange = (dashboardUid: string) => {
    setSelectedDashboard(dashboardUid);
    notification.success({
      message: 'Dashboard Switched',
      description: `Switched to ${dashboards.find(d => d.uid === dashboardUid)?.title}`,
      duration: 2
    });
  };

  const handleRefresh = () => {
    // Force iframe refresh by updating the src
    const iframe = document.getElementById('grafana-iframe') as HTMLIFrameElement;
    if (iframe) {
      const currentSrc = iframe.src;
      iframe.src = '';
      setTimeout(() => {
        iframe.src = currentSrc;
      }, 100);
    }
    
    notification.success({
      message: 'Dashboard Refreshed',
      description: 'Grafana dashboard has been refreshed',
      duration: 2
    });
  };

  const toggleFullscreen = () => {
    setFullscreen(!fullscreen);
    if (!fullscreen) {
      notification.info({
        message: 'Fullscreen Mode',
        description: 'Press ESC or click the button to exit fullscreen',
        duration: 3
      });
    }
  };

  const buildGrafanaUrl = (dashboard: GrafanaDashboard) => {
    const grafanaBaseUrl = import.meta.env.VITE_GRAFANA_URL || 'http://localhost:3002';
    const params = new URLSearchParams({
      refresh: refreshRate,
      theme: theme,
      kiosk: fullscreen ? 'tv' : '',
      from: 'now-1h',
      to: 'now'
    });

    return `${grafanaBaseUrl}${dashboard.url}?${params.toString()}`;
  };

  const selectedDashboardData = dashboards.find(d => d.uid === selectedDashboard);

  const renderDashboardInfo = () => (
    <Row gutter={[16, 8]} style={{ marginBottom: 16 }}>
      <Col xs={24} sm={12} md={8}>
        <Space>
          <Tag color="blue">{selectedDashboardData?.panels_count} Panels</Tag>
          <Tag color="green">{selectedDashboardData?.folder_title}</Tag>
          {selectedDashboardData?.is_starred && <Tag color="gold">★ Starred</Tag>}
        </Space>
      </Col>
      <Col xs={24} sm={12} md={16}>
        <Text type="secondary">{selectedDashboardData?.description}</Text>
      </Col>
    </Row>
  );

  const renderControls = () => (
    <Row justify="space-between" align="middle" style={{ marginBottom: 16 }}>
      <Col>
        <Space wrap>
          <Select
            value={selectedDashboard}
            onChange={handleDashboardChange}
            style={{ width: 300 }}
            size="small"
          >
            {dashboards.map(dashboard => (
              <Option key={dashboard.uid} value={dashboard.uid}>
                <Space>
                  {dashboard.is_starred && '★'}
                  {dashboard.title}
                </Space>
              </Option>
            ))}
          </Select>

          <Select
            value={refreshRate}
            onChange={setRefreshRate}
            style={{ width: 100 }}
            size="small"
          >
            <Option value="5s">5s</Option>
            <Option value="10s">10s</Option>
            <Option value="30s">30s</Option>
            <Option value="1m">1m</Option>
            <Option value="5m">5m</Option>
            <Option value="15m">15m</Option>
          </Select>

          <Select
            value={theme}
            onChange={setTheme}
            style={{ width: 80 }}
            size="small"
          >
            <Option value="light">Light</Option>
            <Option value="dark">Dark</Option>
          </Select>
        </Space>
      </Col>
      
      <Col>
        <Space>
          <Tooltip title="Auto-refresh">
            <Switch
              checked={autoRefresh}
              onChange={setAutoRefresh}
              size="small"
            />
          </Tooltip>

          <Button
            size="small"
            icon={<ReloadOutlined />}
            onClick={handleRefresh}
          >
            Refresh
          </Button>

          <Button
            size="small"
            icon={fullscreen ? <FullscreenExitOutlined /> : <FullscreenOutlined />}
            onClick={toggleFullscreen}
          >
            {fullscreen ? 'Exit' : 'Fullscreen'}
          </Button>
        </Space>
      </Col>
    </Row>
  );

  const renderGrafanaStatus = () => (
    <Alert
      message={
        <Space>
          <span>Grafana Status:</span>
          <Tag color={
            grafanaStatus === 'online' ? 'green' : 
            grafanaStatus === 'offline' ? 'red' : 'blue'
          }>
            {grafanaStatus.toUpperCase()}
          </Tag>
        </Space>
      }
      type={grafanaStatus === 'online' ? 'success' : grafanaStatus === 'offline' ? 'error' : 'info'}
      style={{ marginBottom: 16 }}
      showIcon={false}
    />
  );

  if (loading) {
    return (
      <div style={{ textAlign: 'center', padding: 60 }}>
        <Spin size="large" />
        <div style={{ marginTop: 16 }}>
          <Text type="secondary">Loading Grafana dashboards...</Text>
        </div>
      </div>
    );
  }

  return (
    <div className={`grafana-embed-dashboard ${className || ''}`} style={{ height: fullscreen ? '100vh' : 'auto' }}>
      {/* Header */}
      {!fullscreen && (
        <>
          <div style={{ marginBottom: 24 }}>
            <Title level={3} style={{ margin: 0 }}>
              <EyeOutlined style={{ marginRight: 8, color: '#ff6600' }} />
              Grafana Analytics Dashboards
            </Title>
            <Text type="secondary">
              Enterprise-grade visualization and monitoring dashboards
            </Text>
          </div>

          {renderGrafanaStatus()}
          {renderControls()}
          {selectedDashboardData && renderDashboardInfo()}
        </>
      )}

      {/* Error Display */}
      {error && (
        <Alert
          message="Grafana Connection Error"
          description={error}
          type="error"
          style={{ marginBottom: 16 }}
          action={
            <Button size="small" onClick={checkGrafanaStatus}>
              Retry
            </Button>
          }
        />
      )}

      {/* Dashboard Cards for Selection */}
      {!fullscreen && (
        <Row gutter={[16, 16]} style={{ marginBottom: 24 }}>
          {dashboards.slice(0, 4).map(dashboard => (
            <Col xs={12} md={6} key={dashboard.uid}>
              <Card
                size="small"
                hoverable
                onClick={() => handleDashboardChange(dashboard.uid)}
                style={{
                  border: selectedDashboard === dashboard.uid ? '2px solid #1890ff' : '1px solid #d9d9d9',
                  cursor: 'pointer'
                }}
              >
                <div style={{ textAlign: 'center' }}>
                  <div style={{ fontSize: '24px', marginBottom: 8 }}>
                    {dashboard.uid.includes('overview') && <DashboardOutlined />}
                    {dashboard.uid.includes('websocket') && <ApiOutlined />}
                    {dashboard.uid.includes('risk') && <ThunderboltOutlined />}
                    {dashboard.uid.includes('performance') && <LineChartOutlined />}
                    {dashboard.uid.includes('system') && <BarChartOutlined />}
                    {dashboard.uid.includes('database') && <DatabaseOutlined />}
                  </div>
                  <Text strong style={{ fontSize: '12px' }}>{dashboard.title.substring(0, 20)}...</Text>
                  <div style={{ marginTop: 4 }}>
                    <Tag color="blue" style={{ fontSize: '10px' }}>
                      {dashboard.panels_count} panels
                    </Tag>
                  </div>
                </div>
              </Card>
            </Col>
          ))}
        </Row>
      )}

      {/* Embedded Grafana Dashboard */}
      {selectedDashboardData && grafanaStatus === 'online' && (
        <Card 
          style={{ 
            padding: fullscreen ? 0 : 'inherit',
            border: fullscreen ? 'none' : 'inherit',
            height: fullscreen ? '100vh' : '700px'
          }}
        >
          <iframe
            id="grafana-iframe"
            src={buildGrafanaUrl(selectedDashboardData)}
            width="100%"
            height={fullscreen ? '100vh' : '650px'}
            frameBorder="0"
            title={selectedDashboardData.title}
            style={{ border: 'none' }}
            onLoad={() => {
              console.log('Grafana dashboard loaded:', selectedDashboardData.title);
            }}
            onError={() => {
              console.error('Grafana dashboard failed to load');
              setError('Failed to load Grafana dashboard');
            }}
          />
        </Card>
      )}

      {/* Offline Fallback */}
      {grafanaStatus === 'offline' && (
        <Card style={{ height: '500px', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
          <div style={{ textAlign: 'center' }}>
            <EyeOutlined style={{ fontSize: '48px', color: '#d9d9d9', marginBottom: 16 }} />
            <Title level={4} type="secondary">Grafana Dashboard Unavailable</Title>
            <Text type="secondary">
              Grafana service is not accessible. Please ensure Grafana is running.
            </Text>
            <div style={{ marginTop: 16 }}>
              <Button type="primary" onClick={checkGrafanaStatus}>
                Retry Connection
              </Button>
            </div>
          </div>
        </Card>
      )}

      {/* Fullscreen Exit Overlay */}
      {fullscreen && (
        <div style={{
          position: 'fixed',
          top: 20,
          right: 20,
          zIndex: 1000,
          background: 'rgba(0, 0, 0, 0.8)',
          borderRadius: 8,
          padding: 8
        }}>
          <Space>
            <Button
              type="primary"
              size="small"
              icon={<FullscreenExitOutlined />}
              onClick={toggleFullscreen}
            >
              Exit Fullscreen
            </Button>
            <Button
              size="small"
              icon={<ReloadOutlined />}
              onClick={handleRefresh}
            >
              Refresh
            </Button>
          </Space>
        </div>
      )}
    </div>
  );
};

export default GrafanaEmbedDashboard;