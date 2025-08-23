import React, { useState, useEffect } from 'react';
import {
  Card,
  Tabs,
  Row,
  Col,
  Statistic,
  Alert,
  Button,
  Switch,
  Space,
  Tooltip,
  Badge,
  Typography,
  Progress,
  Tag,
  Timeline,
  List,
  Avatar,
  Divider,
  Modal,
  Spin
} from 'antd';
import {
  DashboardOutlined,
  MonitorOutlined,
  ThunderboltOutlined,
  RobotOutlined,
  WarningOutlined,
  BellOutlined,
  FileTextOutlined,
  SettingOutlined,
  SafetyCertificateOutlined,
  LineChartOutlined,
  BarChartOutlined,
  PlayCircleOutlined,
  PauseCircleOutlined,
  ReloadOutlined,
  FireOutlined,
  CheckCircleOutlined,
  ClockCircleOutlined,
  RiseOutlined,
  SafetyCertificateOutlined,
  ExperimentOutlined,
  AlertOutlined,
  EyeOutlined,
  SyncOutlined
} from '@ant-design/icons';

// Sprint 3 Enhanced Components
import RealTimeRiskMonitor from './RealTimeRiskMonitor';
import AdvancedBreachDetector from './AdvancedBreachDetector';
import RiskReportGenerator from './RiskReportGenerator';
import RiskLimitConfigPanel from './RiskLimitConfigPanel';

// Existing components
import RiskMetrics from './RiskMetrics';
import ExposureAnalysis from './ExposureAnalysis';
import DynamicLimitEngine from './DynamicLimitEngine';
import RiskAlertCenter from './RiskAlertCenter';
import VaRCalculator from './VaRCalculator';
import ComplianceReporting from './ComplianceReporting';

// Hooks
import { useRiskMonitoring } from '../../hooks/risk/useRiskMonitoring';
import { useDynamicLimits } from '../../hooks/risk/useDynamicLimits';
import { useBreachDetection } from '../../hooks/risk/useBreachDetection';
import { useRiskReporting } from '../../hooks/risk/useRiskReporting';
import { useWebSocketManager } from '../../hooks/useWebSocketManager';

import { riskService } from './services/riskService';
import { RealTimeRiskMetrics, RiskComponentPerformance } from './types/riskTypes';

const { Title, Text } = Typography;

interface RiskDashboardSprint3Props {
  portfolioId: string;
  className?: string;
}

interface SystemHealthData {
  overall_status: 'healthy' | 'degraded' | 'unhealthy';
  components: RiskComponentPerformance[];
  active_monitoring_sessions: number;
  total_limits_monitored: number;
  alerts_processed_24h: number;
  ml_model_accuracy: number;
  last_health_check: Date;
}

const RiskDashboardSprint3: React.FC<RiskDashboardSprint3Props> = ({
  portfolioId,
  className
}) => {
  console.log('🎯 RiskDashboardSprint3 rendering for portfolio:', portfolioId);

  const [realTimeEnabled, setRealTimeEnabled] = useState(false);
  const [systemHealth, setSystemHealth] = useState<SystemHealthData | null>(null);
  const [systemMetrics, setSystemMetrics] = useState<any>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [activeTab, setActiveTab] = useState('overview');
  const [healthModalVisible, setHealthModalVisible] = useState(false);

  // Sprint 3 Enhanced Hooks
  const {
    realTimeMetrics,
    criticalAlerts,
    breachedLimits,
    overallRiskScore: monitoringRiskScore,
    isConnected: monitoringConnected
  } = useRiskMonitoring({ 
    portfolioId, 
    enableRealTime: realTimeEnabled 
  });

  const {
    limits,
    breachedLimits: limitBreaches,
    activeLimits,
    riskScore: limitRiskScore
  } = useDynamicLimits({ portfolioId });

  const {
    highRiskPredictions,
    imminentBreaches,
    overallRiskScore: breachRiskScore,
    mlModelAccuracy
  } = useBreachDetection({ 
    portfolioId, 
    enableRealTime: realTimeEnabled 
  });

  const {
    reports,
    generatingReports,
    completedReports,
    failedReports
  } = useRiskReporting({ portfolioId });

  // WebSocket for system-wide updates
  const { isConnected: wsConnected, lastMessage } = useWebSocketManager({
    url: `${import.meta.env.VITE_WS_URL}/ws/system/health`,
    autoConnect: true,
    reconnectAttempts: 5
  });

  const fetchSystemHealth = async () => {
    try {
      setError(null);
      const [healthData, metricsData] = await Promise.all([
        riskService.getRiskSystemHealth(),
        riskService.getRiskSystemMetrics('24h')
      ]);
      setSystemHealth(healthData);
      setSystemMetrics(metricsData);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to fetch system health');
      console.error('System health fetch error:', err);
    } finally {
      setLoading(false);
    }
  };

  const handleRealTimeToggle = async (enabled: boolean) => {
    try {
      setRealTimeEnabled(enabled);
      if (enabled) {
        await riskService.startRealTimeMonitoring(portfolioId, {
          update_frequency_seconds: 5,
          enable_alerts: true,
          enable_auto_actions: true
        });
      } else {
        await riskService.stopRealTimeMonitoring(portfolioId);
      }
    } catch (error) {
      console.error('Failed to toggle real-time monitoring:', error);
    }
  };

  // Handle WebSocket system updates
  useEffect(() => {
    if (lastMessage) {
      try {
        const message = JSON.parse(lastMessage);
        if (message.type === 'system_health_update') {
          setSystemHealth(message.data);
        }
      } catch (error) {
        console.error('Error parsing system WebSocket message:', error);
      }
    }
  }, [lastMessage]);

  // Initial data fetch
  useEffect(() => {
    fetchSystemHealth();
    
    // Auto-refresh system health every 30 seconds
    const interval = setInterval(fetchSystemHealth, 30000);
    return () => clearInterval(interval);
  }, [portfolioId]);

  // Calculate comprehensive risk score
  const comprehensiveRiskScore = Math.round((
    (monitoringRiskScore || 0) * 0.4 +
    (limitRiskScore || 0) * 0.3 +
    (breachRiskScore || 0) * 0.3
  ));

  // Get system status color
  const getSystemStatusColor = (status: string) => {
    switch (status) {
      case 'healthy': return '#52c41a';
      case 'degraded': return '#faad14';
      case 'unhealthy': return '#ff4d4f';
      default: return '#d9d9d9';
    }
  };

  // Connection status
  const overallConnection = monitoringConnected || wsConnected;
  const criticalIssues = criticalAlerts.length + limitBreaches.length + imminentBreaches.length;

  const tabItems = [
    {
      key: 'overview',
      label: (
        <Space>
          <DashboardOutlined />
          Enhanced Overview
          {comprehensiveRiskScore > 70 && (
            <Badge status="error" />
          )}
        </Space>
      ),
      children: (
        <div>
          <RiskMetrics portfolioId={portfolioId} />
        </div>
      )
    },
    {
      key: 'realtime',
      label: (
        <Space>
          <MonitorOutlined />
          Real-Time Monitor
          {overallConnection && <Badge status="processing" />}
        </Space>
      ),
      children: (
        <RealTimeRiskMonitor portfolioId={portfolioId} />
      )
    },
    {
      key: 'ml_breach',
      label: (
        <Space>
          <RobotOutlined />
          ML Breach Detection
          <Badge count={highRiskPredictions.length} size="small" />
        </Space>
      ),
      children: (
        <AdvancedBreachDetector portfolioId={portfolioId} />
      )
    },
    {
      key: 'dynamic_limits',
      label: (
        <Space>
          <ThunderboltOutlined />
          Dynamic Limits
          <Badge count={limitBreaches.length} size="small" />
        </Space>
      ),
      children: (
        <DynamicLimitEngine portfolioId={portfolioId} />
      )
    },
    {
      key: 'limit_config',
      label: (
        <Space>
          <SettingOutlined />
          Limit Configuration
          <Badge count={activeLimits.length} size="small" />
        </Space>
      ),
      children: (
        <RiskLimitConfigPanel portfolioId={portfolioId} />
      )
    },
    {
      key: 'alerts',
      label: (
        <Space>
          <BellOutlined />
          Alert Center
          <Badge count={criticalAlerts.length} size="small" />
        </Space>
      ),
      children: (
        <RiskAlertCenter portfolioId={portfolioId} />
      )
    },
    {
      key: 'exposure',
      label: (
        <Space>
          <BarChartOutlined />
          Exposure Analysis
        </Space>
      ),
      children: (
        <ExposureAnalysis portfolioId={portfolioId} />
      )
    },
    {
      key: 'var_calculator',
      label: (
        <Space>
          <LineChartOutlined />
          VaR Calculator
        </Space>
      ),
      children: (
        <VaRCalculator portfolioId={portfolioId} />
      )
    },
    {
      key: 'reporting',
      label: (
        <Space>
          <FileTextOutlined />
          Advanced Reporting
          <Badge count={generatingReports.length} size="small" />
        </Space>
      ),
      children: (
        <RiskReportGenerator portfolioId={portfolioId} />
      )
    },
    {
      key: 'compliance',
      label: (
        <Space>
          <SafetyCertificateOutlined />
          Compliance
        </Space>
      ),
      children: (
        <ComplianceReporting portfolioId={portfolioId} />
      )
    }
  ];

  if (loading) {
    return (
      <div className={className}>
        <Card>
          <div style={{ textAlign: 'center', padding: '50px 0' }}>
            <Spin size="large" />
            <div style={{ marginTop: 16 }}>
              <Text>Loading Sprint 3 Risk Dashboard...</Text>
            </div>
          </div>
        </Card>
      </div>
    );
  }

  return (
    <div className={className}>
      {/* Enhanced Header with Sprint 3 Statistics */}
      <Card style={{ marginBottom: 16 }}>
        <Row gutter={16} align="middle">
          <Col xs={24} sm={4}>
            <Statistic
              title="Portfolio Value"
              value={realTimeMetrics?.portfolio_value || 1000000}
              precision={0}
              prefix="$"
              formatter={(value) => `${Number(value).toLocaleString()}`}
            />
          </Col>
          
          <Col xs={24} sm={4}>
            <Statistic
              title="Risk Score"
              value={comprehensiveRiskScore}
              precision={0}
              suffix="%"
              prefix={<SafetyCertificateOutlined />}
              valueStyle={{ 
                color: comprehensiveRiskScore > 70 ? '#ff4d4f' : 
                       comprehensiveRiskScore > 40 ? '#faad14' : '#52c41a'
              }}
            />
          </Col>

          <Col xs={24} sm={4}>
            <Statistic
              title="Active Limits"
              value={activeLimits.length}
              prefix={<ThunderboltOutlined />}
              suffix={
                limitBreaches.length > 0 && (
                  <Badge count={limitBreaches.length} style={{ backgroundColor: '#ff4d4f' }} />
                )
              }
            />
          </Col>

          <Col xs={24} sm={4}>
            <Statistic
              title="ML Predictions"
              value={highRiskPredictions.length}
              prefix={<RobotOutlined />}
              valueStyle={{ color: highRiskPredictions.length > 0 ? '#fa8c16' : undefined }}
              suffix={mlModelAccuracy && (
                <Tooltip title={`ML Model Accuracy: ${Math.round(mlModelAccuracy * 100)}%`}>
                  <Tag size="small" color="blue">
                    {Math.round(mlModelAccuracy * 100)}%
                  </Tag>
                </Tooltip>
              )}
            />
          </Col>

          <Col xs={24} sm={4}>
            <Statistic
              title="System Status"
              value={systemHealth?.overall_status.toUpperCase() || 'UNKNOWN'}
              prefix={
                <div 
                  style={{ 
                    width: 8, 
                    height: 8, 
                    borderRadius: '50%', 
                    backgroundColor: systemHealth ? getSystemStatusColor(systemHealth.overall_status) : '#d9d9d9',
                    display: 'inline-block',
                    marginRight: 4
                  }} 
                />
              }
              valueStyle={{ 
                color: systemHealth ? getSystemStatusColor(systemHealth.overall_status) : '#d9d9d9',
                fontSize: '14px'
              }}
            />
          </Col>

          <Col xs={24} sm={4} style={{ textAlign: 'right' }}>
            <Space direction="vertical" size={0}>
              <Space>
                <Tooltip title={realTimeEnabled ? 'Disable real-time monitoring' : 'Enable real-time monitoring'}>
                  <Switch
                    checked={realTimeEnabled}
                    onChange={handleRealTimeToggle}
                    checkedChildren={<PlayCircleOutlined />}
                    unCheckedChildren={<PauseCircleOutlined />}
                  />
                </Tooltip>
                
                <Tooltip title="System health details">
                  <Button 
                    icon={<EyeOutlined />}
                    onClick={() => setHealthModalVisible(true)}
                    size="small"
                  />
                </Tooltip>

                <Tooltip title="Refresh system data">
                  <Button 
                    icon={<ReloadOutlined />}
                    onClick={fetchSystemHealth}
                    loading={loading}
                    size="small"
                  />
                </Tooltip>
              </Space>
              
              <div style={{ marginTop: 8 }}>
                <Space size={16}>
                  <Badge 
                    status={overallConnection ? 'processing' : 'error'} 
                    text={overallConnection ? 'Live' : 'Offline'}
                    style={{ fontSize: '12px' }}
                  />
                  <Text type="secondary" style={{ fontSize: '12px' }}>
                    Reports: {completedReports.length}
                  </Text>
                </Space>
              </div>
            </Space>
          </Col>
        </Row>
      </Card>

      {/* Error Alert */}
      {error && (
        <Alert
          message="System Error"
          description={error}
          type="error"
          showIcon
          closable
          onClose={() => setError(null)}
          style={{ marginBottom: 16 }}
        />
      )}

      {/* Critical Issues Alert */}
      {criticalIssues > 0 && (
        <Alert
          message={
            <Space>
              <FireOutlined />
              <Text strong>CRITICAL RISK ISSUES DETECTED</Text>
            </Space>
          }
          description={
            <div>
              {criticalAlerts.length > 0 && (
                <div>• {criticalAlerts.length} critical alert{criticalAlerts.length > 1 ? 's' : ''} active</div>
              )}
              {limitBreaches.length > 0 && (
                <div>• {limitBreaches.length} risk limit{limitBreaches.length > 1 ? 's' : ''} breached</div>
              )}
              {imminentBreaches.length > 0 && (
                <div>• {imminentBreaches.length} imminent breach{imminentBreaches.length > 1 ? 'es' : ''} predicted</div>
              )}
            </div>
          }
          type="error"
          showIcon
          style={{ marginBottom: 16 }}
          action={
            <Space>
              <Button size="small" danger onClick={() => setActiveTab('alerts')}>
                View Alerts
              </Button>
              <Button size="small" type="primary" onClick={() => setActiveTab('ml_breach')}>
                ML Predictions
              </Button>
            </Space>
          }
        />
      )}

      {/* Sprint 3 Performance Summary */}
      {systemMetrics && (
        <Card size="small" style={{ marginBottom: 16 }}>
          <Title level={5}>
            <Space>
              <RiseOutlined />
              System Performance (24h)
            </Space>
          </Title>
          <Row gutter={16}>
            <Col xs={24} sm={4}>
              <Text type="secondary">Latency (P95)</Text>
              <div>
                <Text strong>{systemMetrics.calculation_latency_p95}ms</Text>
              </div>
            </Col>
            <Col xs={24} sm={4}>
              <Text type="secondary">Throughput</Text>
              <div>
                <Text strong>{systemMetrics.throughput_calculations_per_second}/s</Text>
              </div>
            </Col>
            <Col xs={24} sm={4}>
              <Text type="secondary">Error Rate</Text>
              <div>
                <Text strong style={{ color: systemMetrics.error_rate > 0.05 ? '#ff4d4f' : '#52c41a' }}>
                  {(systemMetrics.error_rate * 100).toFixed(2)}%
                </Text>
              </div>
            </Col>
            <Col xs={24} sm={4}>
              <Text type="secondary">Cache Hit Rate</Text>
              <div>
                <Progress 
                  percent={Math.round(systemMetrics.cache_hit_rate * 100)}
                  size="small"
                  showInfo={false}
                />
                <Text strong>{Math.round(systemMetrics.cache_hit_rate * 100)}%</Text>
              </div>
            </Col>
            <Col xs={24} sm={4}>
              <Text type="secondary">WebSocket Connections</Text>
              <div>
                <Text strong>{systemMetrics.active_websocket_connections}</Text>
              </div>
            </Col>
            <Col xs={24} sm={4}>
              <Text type="secondary">Memory Usage</Text>
              <div>
                <Text strong>{systemMetrics.memory_usage_mb}MB</Text>
              </div>
            </Col>
          </Row>
        </Card>
      )}

      {/* Enhanced Main Content Tabs */}
      <Card>
        <Tabs
          activeKey={activeTab}
          onChange={setActiveTab}
          items={tabItems}
          tabBarStyle={{ marginBottom: 16 }}
          className="risk-sprint3-tabs"
          tabBarExtraContent={
            <Space>
              {realTimeEnabled && (
                <Badge 
                  status="processing" 
                  text="Real-time monitoring active"
                  style={{ fontSize: '12px' }}
                />
              )}
              <Badge 
                color={comprehensiveRiskScore > 70 ? '#ff4d4f' : comprehensiveRiskScore > 40 ? '#faad14' : '#52c41a'}
                text={`Risk Score: ${comprehensiveRiskScore}%`}
                style={{ fontSize: '12px' }}
              />
              {systemHealth && (
                <Badge 
                  color={getSystemStatusColor(systemHealth.overall_status)}
                  text={`System: ${systemHealth.overall_status}`}
                  style={{ fontSize: '12px' }}
                />
              )}
            </Space>
          }
        />
      </Card>

      {/* System Health Modal */}
      <Modal
        title={
          <Space>
            <MonitorOutlined />
            System Health Details
          </Space>
        }
        open={healthModalVisible}
        onCancel={() => setHealthModalVisible(false)}
        footer={[
          <Button key="close" onClick={() => setHealthModalVisible(false)}>
            Close
          </Button>
        ]}
        width={800}
      >
        {systemHealth && (
          <div>
            <Row gutter={16} style={{ marginBottom: 16 }}>
              <Col span={8}>
                <Card size="small">
                  <Statistic
                    title="Overall Status"
                    value={systemHealth.overall_status.toUpperCase()}
                    valueStyle={{ color: getSystemStatusColor(systemHealth.overall_status) }}
                  />
                </Card>
              </Col>
              <Col span={8}>
                <Card size="small">
                  <Statistic
                    title="Active Sessions"
                    value={systemHealth.active_monitoring_sessions}
                    prefix={<MonitorOutlined />}
                  />
                </Card>
              </Col>
              <Col span={8}>
                <Card size="small">
                  <Statistic
                    title="ML Accuracy"
                    value={Math.round(systemHealth.ml_model_accuracy * 100)}
                    suffix="%"
                    prefix={<RobotOutlined />}
                    valueStyle={{ color: systemHealth.ml_model_accuracy > 0.8 ? '#52c41a' : '#faad14' }}
                  />
                </Card>
              </Col>
            </Row>

            <Divider />

            <Title level={5}>Component Health</Title>
            <List
              dataSource={systemHealth.components}
              renderItem={(component: RiskComponentPerformance) => (
                <List.Item>
                  <List.Item.Meta
                    avatar={
                      <Avatar 
                        style={{ 
                          backgroundColor: component.success_rate > 0.95 ? '#52c41a' : 
                                         component.success_rate > 0.90 ? '#faad14' : '#ff4d4f'
                        }}
                      >
                        {component.component_name[0].toUpperCase()}
                      </Avatar>
                    }
                    title={component.component_name.replace('_', ' ').toUpperCase()}
                    description={
                      <Space>
                        <Text type="secondary">
                          Success: {(component.success_rate * 100).toFixed(1)}%
                        </Text>
                        <Text type="secondary">
                          Latency: {component.calculation_time_ms}ms
                        </Text>
                        <Text type="secondary">
                          Errors (24h): {component.error_count_24h}
                        </Text>
                        {component.cache_hit_rate && (
                          <Text type="secondary">
                            Cache: {(component.cache_hit_rate * 100).toFixed(0)}%
                          </Text>
                        )}
                      </Space>
                    }
                  />
                  <div>
                    <Progress
                      percent={Math.round(component.success_rate * 100)}
                      size="small"
                      strokeColor={
                        component.success_rate > 0.95 ? '#52c41a' : 
                        component.success_rate > 0.90 ? '#faad14' : '#ff4d4f'
                      }
                    />
                  </div>
                </List.Item>
              )}
            />
          </div>
        )}
      </Modal>
    </div>
  );
};

export default RiskDashboardSprint3;