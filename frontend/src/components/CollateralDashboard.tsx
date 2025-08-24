/**
 * Collateral Management Dashboard
 * ==============================
 * 
 * React component for displaying real-time margin monitoring,
 * cross-margining optimization results, and margin alerts.
 */

import React, { useState, useEffect, useCallback } from 'react';
import { Card, Row, Col, Progress, Alert, Button, Table, Tag, Statistic, Space, Tooltip } from 'antd';
import { 
  ExclamationCircleOutlined, 
  CheckCircleOutlined, 
  WarningOutlined, 
  ThunderboltOutlined,
  DollarCircleOutlined,
  PercentageOutlined,
  ClockCircleOutlined,
  FireOutlined
} from '@ant-design/icons';
import { 
  collateralService, 
  type Portfolio, 
  type MarginRequirement, 
  type MarginAlert, 
  type OptimizationResult,
  type MonitoringStatus 
} from '../services/collateralService';

interface CollateralDashboardProps {
  portfolio: Portfolio;
}

const CollateralDashboard: React.FC<CollateralDashboardProps> = ({ portfolio }) => {
  const [loading, setLoading] = useState(true);
  const [marginData, setMarginData] = useState<{
    margin_requirement: MarginRequirement;
    optimization?: OptimizationResult;
  } | null>(null);
  const [alerts, setAlerts] = useState<MarginAlert[]>([]);
  const [monitoring, setMonitoring] = useState<MonitoringStatus | null>(null);
  const [alertStream, setAlertStream] = useState<EventSource | null>(null);

  // Load initial margin data
  const loadMarginData = useCallback(async () => {
    try {
      setLoading(true);
      const result = await collateralService.calculateMargin(portfolio, true, false);
      setMarginData(result);
      
      // Check monitoring status
      const statusResult = await collateralService.getMonitoringStatus(portfolio.id);
      setMonitoring(statusResult.monitoring_status);
    } catch (error) {
      console.error('Error loading margin data:', error);
    } finally {
      setLoading(false);
    }
  }, [portfolio]);

  // Start real-time monitoring
  const startMonitoring = async () => {
    try {
      await collateralService.startMonitoring(portfolio);
      
      // Start alert stream
      const stream = collateralService.subscribeToMarginAlerts(
        portfolio.id,
        (alert) => {
          setAlerts(prev => [alert, ...prev.slice(0, 9)]); // Keep last 10 alerts
        },
        (error) => {
          console.error('Alert stream error:', error);
        },
        () => {
          // Heartbeat - could update last seen time
        }
      );
      
      setAlertStream(stream);
      await loadMarginData(); // Refresh data
    } catch (error) {
      console.error('Error starting monitoring:', error);
    }
  };

  // Stop monitoring
  const stopMonitoring = async () => {
    try {
      await collateralService.stopMonitoring(portfolio.id);
      alertStream?.close();
      setAlertStream(null);
      setAlerts([]);
      await loadMarginData(); // Refresh data
    } catch (error) {
      console.error('Error stopping monitoring:', error);
    }
  };

  useEffect(() => {
    loadMarginData();
    
    return () => {
      // Cleanup alert stream on unmount
      alertStream?.close();
    };
  }, [loadMarginData]);

  if (loading) {
    return (
      <Card>
        <div style={{ textAlign: 'center', padding: '50px 0' }}>
          Loading collateral data...
        </div>
      </Card>
    );
  }

  const margin = marginData?.margin_requirement;
  const optimization = marginData?.optimization;

  // Get margin utilization risk level and color
  const getRiskColor = (utilization: number) => {
    if (utilization >= 0.95) return '#ff0000'; // Emergency
    if (utilization >= 0.90) return '#ff6600'; // Critical  
    if (utilization >= 0.80) return '#ffcc00'; // Warning
    if (utilization >= 0.60) return '#0099ff'; // Medium
    return '#00cc00'; // Low risk
  };

  const getRiskLevel = (utilization: number) => {
    if (utilization >= 0.95) return 'EMERGENCY';
    if (utilization >= 0.90) return 'CRITICAL';
    if (utilization >= 0.80) return 'HIGH';
    if (utilization >= 0.60) return 'MEDIUM';
    return 'LOW';
  };

  const getAlertIcon = (severity: string) => {
    switch (severity) {
      case 'emergency': return <FireOutlined style={{ color: '#ff0000' }} />;
      case 'critical': return <ExclamationCircleOutlined style={{ color: '#ff6600' }} />;
      case 'warning': return <WarningOutlined style={{ color: '#ffcc00' }} />;
      case 'info': return <ThunderboltOutlined style={{ color: '#0099ff' }} />;
      default: return <CheckCircleOutlined style={{ color: '#00cc00' }} />;
    }
  };

  const alertColumns = [
    {
      title: 'Time',
      dataIndex: 'timestamp',
      key: 'timestamp',
      width: 100,
      render: (timestamp: string) => new Date(timestamp).toLocaleTimeString(),
    },
    {
      title: 'Severity',
      dataIndex: 'severity',
      key: 'severity',
      width: 80,
      render: (severity: string) => (
        <Tag color={severity === 'emergency' ? 'red' : severity === 'critical' ? 'orange' : severity === 'warning' ? 'gold' : 'blue'}>
          {severity.toUpperCase()}
        </Tag>
      ),
    },
    {
      title: 'Message',
      dataIndex: 'message',
      key: 'message',
      render: (message: string, record: MarginAlert) => (
        <Space>
          {getAlertIcon(record.severity)}
          {message}
        </Space>
      ),
    },
    {
      title: 'Action Required',
      dataIndex: 'required_action_amount',
      key: 'action',
      width: 120,
      render: (amount: number) => amount ? collateralService.formatCurrency(amount) : 'Monitor',
    },
  ];

  return (
    <div>
      {/* Header with monitoring controls */}
      <Card style={{ marginBottom: 16 }}>
        <Row justify="space-between" align="middle">
          <Col>
            <h2>🚨 Collateral Management - {portfolio.name}</h2>
          </Col>
          <Col>
            <Space>
              <Button 
                type={monitoring?.is_monitoring ? "default" : "primary"} 
                onClick={monitoring?.is_monitoring ? stopMonitoring : startMonitoring}
                loading={loading}
              >
                {monitoring?.is_monitoring ? 'Stop Monitoring' : 'Start Real-time Monitoring'}
              </Button>
              <Button onClick={loadMarginData} loading={loading}>
                Refresh
              </Button>
            </Space>
          </Col>
        </Row>
      </Card>

      {/* Main metrics row */}
      <Row gutter={[16, 16]} style={{ marginBottom: 16 }}>
        <Col span={6}>
          <Card>
            <Statistic
              title="Total Margin Required"
              value={margin?.total_margin || 0}
              formatter={(value) => collateralService.formatCurrency(Number(value))}
              prefix={<DollarCircleOutlined />}
            />
          </Card>
        </Col>
        <Col span={6}>
          <Card>
            <Statistic
              title="Margin Utilization"
              value={margin?.margin_utilization_percent || 0}
              suffix="%"
              precision={1}
              valueStyle={{ color: getRiskColor(margin?.margin_utilization || 0) }}
              prefix={<PercentageOutlined />}
            />
            <div style={{ marginTop: 8 }}>
              <Tag color={getRiskColor(margin?.margin_utilization || 0)}>
                {getRiskLevel(margin?.margin_utilization || 0)} RISK
              </Tag>
            </div>
          </Card>
        </Col>
        <Col span={6}>
          <Card>
            <Statistic
              title="Margin Excess"
              value={margin?.margin_excess || 0}
              formatter={(value) => collateralService.formatCurrency(Number(value))}
              valueStyle={{ color: (margin?.margin_excess || 0) > 0 ? '#3f8600' : '#cf1322' }}
              prefix={<DollarCircleOutlined />}
            />
          </Card>
        </Col>
        <Col span={6}>
          <Card>
            <Statistic
              title="Time to Margin Call"
              value={margin?.time_to_margin_call_minutes 
                ? collateralService.formatTimeToMarginCall(margin.time_to_margin_call_minutes)
                : 'N/A'
              }
              prefix={<ClockCircleOutlined />}
            />
          </Card>
        </Col>
      </Row>

      {/* Margin utilization progress bar */}
      <Card title="Margin Utilization Status" style={{ marginBottom: 16 }}>
        <Progress
          percent={Math.min((margin?.margin_utilization_percent || 0), 100)}
          status={margin?.margin_utilization_percent && margin.margin_utilization_percent > 90 ? 'exception' : 'active'}
          strokeColor={{
            '0%': '#00cc00',
            '60%': '#0099ff', 
            '80%': '#ffcc00',
            '90%': '#ff6600',
            '95%': '#ff0000',
          }}
          format={(percent) => `${percent?.toFixed(1)}%`}
        />
        <div style={{ marginTop: 8, display: 'flex', justifyContent: 'space-between', fontSize: '12px', color: '#666' }}>
          <span>Safe (0-60%)</span>
          <span>Medium (60-80%)</span>
          <span>High (80-90%)</span>
          <span>Critical (90-95%)</span>
          <span>Emergency (95%+)</span>
        </div>
      </Card>

      {/* Optimization results */}
      {optimization && (
        <Card title="Cross-Margining Optimization Results" style={{ marginBottom: 16 }}>
          <Row gutter={[16, 16]}>
            <Col span={8}>
              <Statistic
                title="Original Margin"
                value={optimization.original_margin}
                formatter={(value) => collateralService.formatCurrency(Number(value))}
              />
            </Col>
            <Col span={8}>
              <Statistic
                title="Optimized Margin"
                value={optimization.optimized_margin}
                formatter={(value) => collateralService.formatCurrency(Number(value))}
                valueStyle={{ color: '#3f8600' }}
              />
            </Col>
            <Col span={8}>
              <Statistic
                title="Capital Efficiency Improvement"
                value={optimization.capital_efficiency_improvement}
                suffix="%"
                precision={1}
                valueStyle={{ color: '#3f8600' }}
                prefix={<ThunderboltOutlined />}
              />
            </Col>
          </Row>
          
          <div style={{ marginTop: 16 }}>
            <Alert
              message={`Margin Savings: ${collateralService.formatCurrency(optimization.margin_savings)}`}
              description={`Through cross-margining optimization, you can save ${collateralService.formatCurrency(optimization.margin_savings)} in margin requirements, improving capital efficiency by ${optimization.capital_efficiency_improvement.toFixed(1)}%.`}
              type="success"
              showIcon
            />
          </div>
        </Card>
      )}

      {/* Real-time alerts */}
      {monitoring?.is_monitoring && (
        <Card title="Real-time Margin Alerts" style={{ marginBottom: 16 }}>
          {alerts.length > 0 ? (
            <Table
              columns={alertColumns}
              dataSource={alerts}
              rowKey="timestamp"
              pagination={false}
              size="small"
              scroll={{ y: 200 }}
            />
          ) : (
            <div style={{ textAlign: 'center', padding: '20px 0', color: '#666' }}>
              No alerts - margin levels are healthy
            </div>
          )}
        </Card>
      )}

      {/* Position breakdown */}
      <Card title="Position Margin Breakdown">
        <Table
          columns={[
            { title: 'Symbol', dataIndex: 'symbol', key: 'symbol' },
            { 
              title: 'Asset Class', 
              dataIndex: 'asset_class', 
              key: 'asset_class',
              render: (assetClass: string) => (
                <Tag color="blue">{assetClass.toUpperCase()}</Tag>
              )
            },
            { 
              title: 'Market Value', 
              dataIndex: 'market_value', 
              key: 'market_value',
              render: (value: number, record: any) => 
                collateralService.formatCurrency(value * record.quantity)
            },
            { 
              title: 'Margin Required', 
              key: 'margin',
              render: (record: any) => {
                const positionMargin = margin?.position_margins[record.id];
                return positionMargin ? 
                  collateralService.formatCurrency(positionMargin.initial_margin) : 
                  'Calculating...';
              }
            },
          ]}
          dataSource={portfolio.positions}
          rowKey="id"
          pagination={{ pageSize: 10 }}
          size="small"
        />
      </Card>
    </div>
  );
};

export default CollateralDashboard;