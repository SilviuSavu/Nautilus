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
  Badge
} from 'antd';
import { 
  DashboardOutlined, 
  BarChartOutlined, 
  AlertOutlined, 
  SettingOutlined,
  ReloadOutlined,
  PlayCircleOutlined,
  PauseCircleOutlined
} from '@ant-design/icons';
import ExposureAnalysis from './ExposureAnalysis';
import RiskMetrics from './RiskMetrics';
import AlertSystem from './AlertSystem';
import { PortfolioRisk } from './types/riskTypes';
import { riskService } from './services/riskService';


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
          <AlertOutlined />
          <Badge count={0} size="small">
            Alerts & Limits
          </Badge>
        </span>
      ),
      key: 'alerts',
      children: (
        <div>
          <AlertSystem portfolioId={portfolioId} />
        </div>
      )
    }
  ];

  return (
    <div className={className}>
      {/* Header with Summary Stats */}
      <Card style={{ marginBottom: 16 }}>
        <Row gutter={16} align="middle">
          <Col span={4}>
            <Statistic
              title="Portfolio Value"
              value={portfolioRisk ? parseFloat(portfolioRisk.total_exposure) : 0}
              precision={0}
              prefix="$"
              loading={loading}
            />
          </Col>
          <Col span={4}>
            <Statistic
              title="1-Day VaR (95%)"
              value={portfolioRisk ? parseFloat(portfolioRisk.var_1d) : 0}
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
          <Col span={4}>
            <Statistic
              title="Expected Shortfall"
              value={portfolioRisk ? parseFloat(portfolioRisk.expected_shortfall) : 0}
              precision={0}
              prefix="$"
              loading={loading}
              valueStyle={{ color: '#ff4d4f' }}
            />
          </Col>
          <Col span={3}>
            <Statistic
              title="Portfolio Beta"
              value={portfolioRisk ? portfolioRisk.beta : 0}
              precision={2}
              loading={loading}
              valueStyle={{ 
                color: portfolioRisk ? getRiskColor(
                  Math.abs(portfolioRisk.beta - 1) * 100,
                  [20, 50]
                ) : undefined
              }}
            />
          </Col>
          <Col span={3}>
            <Statistic
              title="Concentration Risk"
              value={portfolioRisk ? portfolioRisk.concentration_risk.length : 0}
              loading={loading}
              suffix="positions"
            />
          </Col>
          <Col span={6} style={{ textAlign: 'right' }}>
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
            </Space>
            {lastUpdate && (
              <div style={{ fontSize: '12px', color: '#666', marginTop: 4 }}>
                Last updated: {lastUpdate.toLocaleTimeString()}
              </div>
            )}
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

      {/* Main Content Tabs */}
      <Card>
        <Tabs
          defaultActiveKey="overview"
          items={tabItems}
          tabBarStyle={{ marginBottom: 16 }}
          tabBarExtraContent={
            realTimeEnabled && (
              <Badge 
                status="processing" 
                text="Real-time monitoring active"
                style={{ fontSize: '12px' }}
              />
            )
          }
        />
      </Card>
    </div>
  );
};

export default RiskDashboard;