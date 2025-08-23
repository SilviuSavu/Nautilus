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
  Typography
} from 'antd';
import { 
  DashboardOutlined, 
  BarChartOutlined, 
  AlertOutlined, 
  SettingOutlined,
  ReloadOutlined,
  PlayCircleOutlined,
  PauseCircleOutlined,
  ThunderboltOutlined,
  WarningOutlined,
  FileTextOutlined,
  MonitorOutlined,
  BellOutlined,
  SafetyCertificateOutlined,
  CalculatorOutlined,
  ExperimentOutlined,
  LineChartOutlined,
  TableOutlined
} from '@ant-design/icons';

// Existing components
import ExposureAnalysis from './ExposureAnalysis';
import RiskMetrics from './RiskMetrics';
import AlertSystem from './AlertSystem';

// New Sprint 3 components
import DynamicLimitEngine from './DynamicLimitEngine';
import BreachDetector from './BreachDetector';
import RiskReporter from './RiskReporter';
import LimitMonitor from './LimitMonitor';
import RiskAlertCenter from './RiskAlertCenter';
import ComplianceReporting from './ComplianceReporting';
import VaRCalculator from './VaRCalculator';

// New hooks
import { useRiskMonitoring } from '../../hooks/risk/useRiskMonitoring';
import { useDynamicLimits } from '../../hooks/risk/useDynamicLimits';
import { useBreachDetection } from '../../hooks/risk/useBreachDetection';
import { useRiskReporting } from '../../hooks/risk/useRiskReporting';

import { PortfolioRisk } from './types/riskTypes';
import { riskService } from './services/riskService';

const { Title } = Typography;


interface RiskDashboardProps {
  portfolioId: string;
  className?: string;
}

const RiskDashboard: React.FC<RiskDashboardProps> = ({ 
  portfolioId, 
  className 
}) => {
  console.log('🎯 RiskDashboard component rendering for portfolio:', portfolioId);
  
  const [portfolioRisk, setPortfolioRisk] = useState<PortfolioRisk | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [realTimeEnabled, setRealTimeEnabled] = useState(false);
  const [lastUpdate, setLastUpdate] = useState<Date | null>(null);

  // Sprint 3 hooks for enhanced functionality
  const {
    realTimeMetrics,
    criticalAlerts,
    breachedLimits,
    overallRiskScore: monitoringRiskScore,
    isConnected
  } = useRiskMonitoring({ portfolioId, enableRealTime: realTimeEnabled });

  const {
    limits,
    breachedLimits: limitBreaches,
    activeLimits,
    riskScore: limitRiskScore
  } = useDynamicLimits({ portfolioId });

  const {
    highRiskPredictions,
    imminentBreaches,
    overallRiskScore: breachRiskScore
  } = useBreachDetection({ portfolioId, enableRealTime: realTimeEnabled });

  const {
    reports,
    generatingReports
  } = useRiskReporting({ portfolioId });

  const fetchPortfolioRisk = async () => {
    try {
      setError(null);
      const data = await riskService.getPortfolioRisk(portfolioId);
      setPortfolioRisk(data);
      setLastUpdate(new Date());
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to fetch portfolio risk data');
    } finally {
      setLoading(false);
    }
  };

  const handleRealTimeToggle = async (enabled: boolean) => {
    try {
      if (enabled) {
        await riskService.enableRealTimeMonitoring(portfolioId);
      } else {
        await riskService.disableRealTimeMonitoring(portfolioId);
      }
      setRealTimeEnabled(enabled);
    } catch (error) {
      console.error('Failed to toggle real-time monitoring:', error);
    }
  };

  useEffect(() => {
    fetchPortfolioRisk();
  }, [portfolioId]);

  useEffect(() => {
    let interval: NodeJS.Timeout | null = null;
    
    if (realTimeEnabled) {
      interval = setInterval(fetchPortfolioRisk, 30000); // Update every 30 seconds
    }
    
    return () => {
      if (interval) clearInterval(interval);
    };
  }, [realTimeEnabled, portfolioId]);

  const formatCurrency = (value: string): string => {
    return `$${parseFloat(value).toLocaleString()}`;
  };

  const getRiskLevel = (value: number, thresholds: [number, number]): string => {
    const [medium, high] = thresholds;
    if (value >= high) return 'High';
    if (value >= medium) return 'Medium';
    return 'Low';
  };

  const getRiskColor = (value: number, thresholds: [number, number]): string => {
    const [medium, high] = thresholds;
    if (value >= high) return '#ff4d4f';
    if (value >= medium) return '#faad14';
    return '#52c41a';
  };

  // Enhanced tab items with Sprint 3 components
  const tabItems = [
    {
      label: (
        <span>
          <DashboardOutlined />
          Overview
        </span>
      ),
      key: 'overview',
      children: (
        <div>
          <RiskMetrics portfolioId={portfolioId} />
        </div>
      )
    },
    {
      label: (
        <span>
          <MonitorOutlined />
          Real-Time Monitor
          {isConnected && <Badge status="processing" size="small" style={{ marginLeft: 4 }} />}
        </span>
      ),
      key: 'monitor',
      children: (
        <div>
          <LimitMonitor portfolioId={portfolioId} />
        </div>
      )
    },
    {
      label: (
        <span>
          <ThunderboltOutlined />
          Dynamic Limits
          <Badge count={limitBreaches.length} size="small" style={{ marginLeft: 4 }} />
        </span>
      ),
      key: 'limits',
      children: (
        <div>
          <DynamicLimitEngine portfolioId={portfolioId} />
        </div>
      )
    },
    {
      label: (
        <span>
          <WarningOutlined />
          Breach Detection
          <Badge count={highRiskPredictions.length} size="small" style={{ marginLeft: 4 }} />
        </span>
      ),
      key: 'breach_detection',
      children: (
        <div>
          <BreachDetector portfolioId={portfolioId} />
        </div>
      )
    },
    {
      label: (
        <span>
          <BellOutlined />
          Alert Center
          <Badge count={criticalAlerts.length} size="small" style={{ marginLeft: 4 }} />
        </span>
      ),
      key: 'alerts',
      children: (
        <div>
          <RiskAlertCenter portfolioId={portfolioId} />
        </div>
      )
    },
    {
      label: (
        <span>
          <BarChartOutlined />
          Exposure Analysis
        </span>
      ),
      key: 'exposure',
      children: (
        <div>
          <ExposureAnalysis portfolioId={portfolioId} />
        </div>
      )
    },
    {
      label: (
        <span>
          <CalculatorOutlined />
          VaR Calculator
        </span>
      ),
      key: 'var_calculator',
      children: (
        <div>
          <VaRCalculator portfolioId={portfolioId} />
        </div>
      )
    },
    {
      label: (
        <span>
          <FileTextOutlined />
          Risk Reporting
          <Badge count={generatingReports.length} size="small" style={{ marginLeft: 4 }} />
        </span>
      ),
      key: 'reporting',
      children: (
        <div>
          <RiskReporter portfolioId={portfolioId} />
        </div>
      )
    },
    {
      label: (
        <span>
          <SafetyCertificateOutlined />
          Compliance
        </span>
      ),
      key: 'compliance',
      children: (
        <div>
          <ComplianceReporting portfolioId={portfolioId} />
        </div>
      )
    }
  ];

  // Calculate comprehensive risk score from all sources
  const comprehensiveRiskScore = Math.round((
    (monitoringRiskScore || 0) * 0.4 +
    (limitRiskScore || 0) * 0.3 +
    (breachRiskScore || 0) * 0.3
  ));

  return (
    <div className={className}>
      {/* Enhanced Header with Sprint 3 Statistics */}
      <Card style={{ marginBottom: 16 }}>
        <Row gutter={16} align="middle">
          <Col span={3}>
            <Statistic
              title="Portfolio Value"
              value={realTimeMetrics?.portfolio_value || (portfolioRisk ? parseFloat(portfolioRisk.total_exposure) : 0)}
              precision={0}
              prefix="$"
              loading={loading}
            />
          </Col>
          <Col span={3}>
            <Statistic
              title="Overall Risk Score"
              value={comprehensiveRiskScore}
              precision={0}
              suffix="%"
              loading={loading}
              prefix={<ThunderboltOutlined />}
              valueStyle={{ 
                color: comprehensiveRiskScore > 70 ? '#ff4d4f' : 
                       comprehensiveRiskScore > 40 ? '#faad14' : '#52c41a'
              }}
            />
          </Col>
          <Col span={3}>
            <Statistic
              title="1-Day VaR (95%)"
              value={realTimeMetrics?.var_95_current || (portfolioRisk ? parseFloat(portfolioRisk.var_1d) : 0)}
              precision={0}
              prefix="$"
              loading={loading}
              valueStyle={{ 
                color: portfolioRisk ? getRiskColor(
                  parseFloat(portfolioRisk.var_1d) / parseFloat(portfolioRisk.total_exposure) * 100,
                  [2, 5]
                ) : undefined
              }}
            />
          </Col>
          <Col span={3}>
            <Statistic
              title="Active Limits"
              value={activeLimits.length}
              loading={loading}
              prefix={<SettingOutlined />}
              suffix={
                limitBreaches.length > 0 && (
                  <Badge count={limitBreaches.length} style={{ backgroundColor: '#ff4d4f' }} />
                )
              }
            />
          </Col>
          <Col span={3}>
            <Statistic
              title="Critical Alerts"
              value={criticalAlerts.length}
              loading={loading}
              prefix={<BellOutlined />}
              valueStyle={{ color: criticalAlerts.length > 0 ? '#ff4d4f' : undefined }}
            />
          </Col>
          <Col span={3}>
            <Statistic
              title="Imminent Breaches"
              value={imminentBreaches.length}
              loading={loading}
              prefix={<WarningOutlined />}
              valueStyle={{ color: imminentBreaches.length > 0 ? '#fa8c16' : undefined }}
            />
          </Col>
          <Col span={6} style={{ textAlign: 'right' }}>
            <Space direction="vertical" size={0}>
              <Space>
                <Tooltip title={realTimeEnabled ? 'Disable real-time updates' : 'Enable real-time updates'}>
                  <Switch
                    checked={realTimeEnabled}
                    onChange={handleRealTimeToggle}
                    checkedChildren={<PlayCircleOutlined />}
                    unCheckedChildren={<PauseCircleOutlined />}
                  />
                </Tooltip>
                <Tooltip title="Refresh data">
                  <Button 
                    icon={<ReloadOutlined />} 
                    onClick={fetchPortfolioRisk}
                    loading={loading}
                  />
                </Tooltip>
                <Tooltip title={isConnected ? 'Real-time connection active' : 'Real-time connection offline'}>
                  <Badge 
                    status={isConnected ? 'processing' : 'error'} 
                    text={isConnected ? 'Live' : 'Offline'}
                    style={{ fontSize: '12px' }}
                  />
                </Tooltip>
              </Space>
              {lastUpdate && (
                <div style={{ fontSize: '12px', color: '#666', marginTop: 4 }}>
                  Last updated: {lastUpdate.toLocaleTimeString()}
                </div>
              )}
            </Space>
          </Col>
        </Row>
      </Card>

      {/* Error Alert */}
      {error && (
        <Alert
          message="Error Loading Risk Data"
          description={error}
          type="error"
          showIcon
          closable
          onClose={() => setError(null)}
          style={{ marginBottom: 16 }}
          action={
            <Button size="small" onClick={fetchPortfolioRisk}>
              Retry
            </Button>
          }
        />
      )}

      {/* Critical Alerts Banner */}
      {(criticalAlerts.length > 0 || limitBreaches.length > 0 || imminentBreaches.length > 0) && (
        <Alert
          message={
            <Space>
              <WarningOutlined />
              <Title level={5} style={{ margin: 0, color: '#fff' }}>
                Risk Management Alert
              </Title>
            </Space>
          }
          description={
            <div>
              {criticalAlerts.length > 0 && (
                <div>• {criticalAlerts.length} critical risk alert{criticalAlerts.length > 1 ? 's' : ''} require immediate attention</div>
              )}
              {limitBreaches.length > 0 && (
                <div>• {limitBreaches.length} risk limit{limitBreaches.length > 1 ? 's have' : ' has'} been breached</div>
              )}
              {imminentBreaches.length > 0 && (
                <div>• {imminentBreaches.length} limit breach{imminentBreaches.length > 1 ? 'es are' : ' is'} predicted within 30 minutes</div>
              )}
            </div>
          }
          type="error"
          showIcon
          style={{ marginBottom: 16 }}
          action={
            <Space>
              <Button size="small" danger>
                View Alerts
              </Button>
              <Button size="small" type="primary">
                Escalate
              </Button>
            </Space>
          }
        />
      )}

      {/* Enhanced Main Content Tabs */}
      <Card>
        <Tabs
          defaultActiveKey="overview"
          items={tabItems}
          tabBarStyle={{ marginBottom: 16 }}
          className="risk-internal-tabs"
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
            </Space>
          }
        />
      </Card>
    </div>
  );
};

export default RiskDashboard;