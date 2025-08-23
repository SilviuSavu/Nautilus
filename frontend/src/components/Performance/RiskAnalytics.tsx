/**
 * RiskAnalytics Component - Sprint 3 Integration
 * Comprehensive risk analytics with VaR, stress testing, and exposure analysis
 */

import React, { useState, useEffect, useMemo } from 'react';
import { 
  Card, Row, Col, Table, Tabs, Select, Button, Space, Typography, Alert, 
  Progress, Statistic, Tag, Tooltip, Switch, InputNumber, Modal, Descriptions
} from 'antd';
import { 
  WarningOutlined,
  ThunderboltOutlined,
  BarChartOutlined,
  RadarChartOutlined,
  TableOutlined,
  SettingOutlined,
  AlertOutlined,
  EyeOutlined,
  DownloadOutlined,
  PlayCircleOutlined,
  StopOutlined
} from '@ant-design/icons';
import { Column, Heatmap, Radar, Gauge, Line } from '@ant-design/plots';
import useRiskAnalytics from '../../hooks/analytics/useRiskAnalytics';

const { Title, Text } = Typography;
const { TabPane } = Tabs;
const { Option } = Select;

interface RiskAnalyticsProps {
  portfolioId: string;
  className?: string;
  height?: number;
  showRealTimeMonitoring?: boolean;
  defaultConfidenceLevels?: number[];
}

interface StressScenario {
  name: string;
  description: string;
  market_shock: number;
  volatility_multiplier: number;
  correlation_adjustment: number;
}

const RiskAnalytics: React.FC<RiskAnalyticsProps> = ({
  portfolioId,
  className,
  height = 1000,
  showRealTimeMonitoring = true,
  defaultConfidenceLevels = [0.95, 0.99],
}) => {
  // State
  const [activeTab, setActiveTab] = useState('overview');
  const [confidenceLevels, setConfidenceLevels] = useState(defaultConfidenceLevels);
  const [varMethods, setVarMethods] = useState<('historical' | 'parametric' | 'monte_carlo')[]>(['historical', 'parametric']);
  const [customScenarios, setCustomScenarios] = useState<StressScenario[]>([
    {
      name: 'Market Crash',
      description: '2008-style financial crisis',
      market_shock: -0.20,
      volatility_multiplier: 2.0,
      correlation_adjustment: 0.3,
    },
    {
      name: 'Flash Crash',
      description: 'Rapid market decline',
      market_shock: -0.10,
      volatility_multiplier: 3.0,
      correlation_adjustment: 0.5,
    },
    {
      name: 'Interest Rate Shock',
      description: 'Rapid rate increase',
      market_shock: -0.08,
      volatility_multiplier: 1.5,
      correlation_adjustment: 0.2,
    },
  ]);
  const [showSettings, setShowSettings] = useState(false);
  const [selectedExposureType, setSelectedExposureType] = useState<'sector' | 'currency' | 'geographic' | 'all'>('sector');
  
  const {
    result,
    isAnalyzing,
    error,
    riskAlerts,
    isMonitoring,
    analyzeRisk,
    startMonitoring,
    stopMonitoring,
    calculatePortfolioVar,
    runStressTest,
    getExposureAnalysis,
    checkRiskLimits,
  } = useRiskAnalytics({
    autoRefresh: true,
    refreshInterval: 30000,
    enableRealTimeMonitoring: showRealTimeMonitoring,
    alertThresholds: {
      var_95_threshold: 10000,
      concentration_threshold: 0.15,
      leverage_threshold: 3.0,
    },
  });
  
  // Fetch initial risk analysis
  useEffect(() => {
    if (portfolioId) {
      analyzeRisk({
        portfolio_id: portfolioId,
        confidence_levels: confidenceLevels,
        var_methods: varMethods,
        stress_scenarios: customScenarios,
        include_component_var: true,
        include_marginal_var: true,
        rolling_window: 252,
      });
    }
  }, [portfolioId, confidenceLevels, varMethods, customScenarios]);
  
  // VaR Analysis Chart Data
  const varChartData = useMemo(() => {
    if (!result || !result.var_analysis) return [];
    
    return result.var_analysis.map(va => ([
      { method: 'Historical', confidence: `${(va.confidence_level * 100).toFixed(0)}%`, value: Math.abs(va.historical_var) },
      { method: 'Parametric', confidence: `${(va.confidence_level * 100).toFixed(0)}%`, value: Math.abs(va.parametric_var) },
      { method: 'Monte Carlo', confidence: `${(va.confidence_level * 100).toFixed(0)}%`, value: Math.abs(va.monte_carlo_var) },
    ])).flat();
  }, [result]);
  
  // Stress Test Chart Data
  const stressTestData = useMemo(() => {
    if (!result || !result.stress_testing) return [];
    
    return result.stress_testing.map(st => ({
      scenario: st.scenario_name,
      loss_percentage: Math.abs(st.percentage_loss * 100),
      loss_amount: Math.abs(st.absolute_loss),
    }));
  }, [result]);
  
  // Exposure Data
  const exposureData = useMemo(() => {
    if (!result || !result.exposure_analysis) return [];
    
    const exposures = result.exposure_analysis.sector_exposure || {};
    return Object.entries(exposures).map(([sector, data]) => ({
      sector,
      long_exposure: (data as any).long_exposure,
      short_exposure: Math.abs((data as any).short_exposure),
      net_exposure: (data as any).net_exposure,
      percentage: (data as any).percentage_of_portfolio,
    }));
  }, [result]);
  
  // Component VaR Data
  const componentVarData = useMemo(() => {
    if (!result || !result.component_var) return [];
    
    return result.component_var
      .sort((a, b) => Math.abs(b.component_var) - Math.abs(a.component_var))
      .slice(0, 15);
  }, [result]);
  
  // Correlation Heatmap Data
  const correlationData = useMemo(() => {
    if (!result || !result.correlation_matrix) return [];
    
    const { assets, correlation_data } = result.correlation_matrix;
    const heatmapData = [];
    
    for (let i = 0; i < assets.length; i++) {
      for (let j = 0; j < assets.length; j++) {
        heatmapData.push({
          x: assets[i],
          y: assets[j],
          value: correlation_data[i][j],
        });
      }
    }
    
    return heatmapData;
  }, [result]);
  
  // Risk Metrics Time Series (if available)
  const riskTimeSeriesData = useMemo(() => {
    if (!result || !result.risk_metrics_time_series) return [];
    
    return result.risk_metrics_time_series.map(rm => ({
      date: rm.date,
      var_95: Math.abs(rm.var_95),
      var_99: Math.abs(rm.var_99),
      expected_shortfall: Math.abs(rm.expected_shortfall_95),
      volatility: rm.volatility * 100,
    }));
  }, [result]);
  
  // Chart Configurations
  const varChartConfig = {
    data: varChartData,
    xField: 'confidence',
    yField: 'value',
    seriesField: 'method',
    height: 300,
    color: ['#1890ff', '#52c41a', '#faad14'],
    label: {
      position: 'top' as const,
      formatter: (data: any) => `$${(data.value / 1000).toFixed(0)}k`,
    },
    yAxis: {
      label: {
        formatter: (v: string) => `$${(Number(v) / 1000).toFixed(0)}k`,
      },
    },
    tooltip: {
      formatter: (data: any) => ({
        name: data.method,
        value: `$${data.value.toLocaleString()}`,
      }),
    },
  };
  
  const stressTestConfig = {
    data: stressTestData,
    xField: 'scenario',
    yField: 'loss_percentage',
    height: 300,
    color: '#ff4d4f',
    label: {
      position: 'top' as const,
      formatter: (data: any) => `${data.loss_percentage.toFixed(1)}%`,
    },
    yAxis: {
      label: {
        formatter: (v: string) => `${v}%`,
      },
    },
    xAxis: {
      label: {
        autoRotate: true,
      },
    },
  };
  
  const exposureConfig = {
    data: exposureData,
    xField: 'sector',
    yField: ['long_exposure', 'short_exposure'],
    isStack: true,
    height: 300,
    color: ['#52c41a', '#ff4d4f'],
    yAxis: {
      label: {
        formatter: (v: string) => `$${(Number(v) / 1000000).toFixed(0)}M`,
      },
    },
    xAxis: {
      label: {
        autoRotate: true,
      },
    },
  };
  
  const correlationHeatmapConfig = {
    data: correlationData,
    xField: 'x',
    yField: 'y',
    colorField: 'value',
    height: 400,
    color: ['#1890ff', '#ffffff', '#ff4d4f'],
    tooltip: {
      formatter: (data: any) => ({
        name: `${data.x} vs ${data.y}`,
        value: data.value.toFixed(3),
      }),
    },
  };
  
  const riskTimeSeriesConfig = {
    data: riskTimeSeriesData,
    xField: 'date',
    yField: 'var_95',
    height: 250,
    smooth: true,
    color: '#ff4d4f',
    yAxis: {
      label: {
        formatter: (v: string) => `$${(Number(v) / 1000).toFixed(0)}k`,
      },
    },
    xAxis: {
      type: 'time' as const,
    },
  };
  
  // Table columns
  const componentVarColumns = [
    {
      title: 'Asset',
      dataIndex: 'asset',
      key: 'asset',
      width: 120,
    },
    {
      title: 'Individual VaR',
      dataIndex: 'individual_var',
      key: 'individual_var',
      render: (value: number) => `$${Math.abs(value).toLocaleString()}`,
      sorter: (a: any, b: any) => Math.abs(b.individual_var) - Math.abs(a.individual_var),
    },
    {
      title: 'Component VaR',
      dataIndex: 'component_var',
      key: 'component_var',
      render: (value: number) => `$${Math.abs(value).toLocaleString()}`,
      sorter: (a: any, b: any) => Math.abs(b.component_var) - Math.abs(a.component_var),
    },
    {
      title: 'Marginal VaR',
      dataIndex: 'marginal_var',
      key: 'marginal_var',
      render: (value: number) => `$${Math.abs(value).toLocaleString()}`,
      sorter: (a: any, b: any) => Math.abs(b.marginal_var) - Math.abs(a.marginal_var),
    },
    {
      title: 'Contribution %',
      dataIndex: 'percentage_contribution',
      key: 'percentage_contribution',
      render: (value: number) => `${(value * 100).toFixed(2)}%`,
      sorter: (a: any, b: any) => b.percentage_contribution - a.percentage_contribution,
    },
  ];
  
  const exposureColumns = [
    {
      title: 'Sector',
      dataIndex: 'sector',
      key: 'sector',
    },
    {
      title: 'Long Exposure',
      dataIndex: 'long_exposure',
      key: 'long_exposure',
      render: (value: number) => (
        <span style={{ color: '#52c41a' }}>
          ${value.toLocaleString()}
        </span>
      ),
      sorter: (a: any, b: any) => b.long_exposure - a.long_exposure,
    },
    {
      title: 'Short Exposure',
      dataIndex: 'short_exposure',
      key: 'short_exposure',
      render: (value: number) => (
        <span style={{ color: '#ff4d4f' }}>
          ${value.toLocaleString()}
        </span>
      ),
      sorter: (a: any, b: any) => b.short_exposure - a.short_exposure,
    },
    {
      title: 'Net Exposure',
      dataIndex: 'net_exposure',
      key: 'net_exposure',
      render: (value: number) => (
        <span style={{ color: value >= 0 ? '#52c41a' : '#ff4d4f' }}>
          ${value.toLocaleString()}
        </span>
      ),
      sorter: (a: any, b: any) => b.net_exposure - a.net_exposure,
    },
    {
      title: 'Portfolio %',
      dataIndex: 'percentage',
      key: 'percentage',
      render: (value: number) => (
        <span>
          {(value * 100).toFixed(2)}%
          <Progress 
            percent={Math.abs(value) * 100} 
            size="small" 
            showInfo={false}
            strokeColor={value >= 0 ? '#52c41a' : '#ff4d4f'}
          />
        </span>
      ),
      sorter: (a: any, b: any) => Math.abs(b.percentage) - Math.abs(a.percentage),
    },
  ];
  
  return (
    <div className={className} style={{ height }}>
      <Row gutter={[16, 16]}>
        {/* Header with controls */}
        <Col span={24}>
          <Card size="small">
            <Row justify="space-between" align="middle">
              <Col>
                <Space>
                  <WarningOutlined style={{ color: '#ff4d4f', fontSize: '18px' }} />
                  <Title level={4} style={{ margin: 0 }}>
                    Risk Analytics
                  </Title>
                  {showRealTimeMonitoring && (
                    <Tag color={isMonitoring ? 'green' : 'default'}>
                      {isMonitoring ? 'Live Monitoring' : 'Static Analysis'}
                    </Tag>
                  )}
                  {riskAlerts.length > 0 && (
                    <Tag color="red">
                      {riskAlerts.length} Alert{riskAlerts.length > 1 ? 's' : ''}
                    </Tag>
                  )}
                </Space>
              </Col>
              <Col>
                <Space>
                  <Button 
                    icon={<SettingOutlined />} 
                    size="small"
                    onClick={() => setShowSettings(true)}
                  >
                    Settings
                  </Button>
                  {showRealTimeMonitoring && (
                    <>
                      {!isMonitoring ? (
                        <Button 
                          type="primary"
                          icon={<PlayCircleOutlined />} 
                          size="small"
                          onClick={() => startMonitoring(portfolioId)}
                        >
                          Start Monitoring
                        </Button>
                      ) : (
                        <Button 
                          danger
                          icon={<StopOutlined />} 
                          size="small"
                          onClick={stopMonitoring}
                        >
                          Stop Monitoring
                        </Button>
                      )}
                    </>
                  )}
                  <Button 
                    icon={<DownloadOutlined />} 
                    size="small"
                    onClick={() => {
                      const blob = new Blob([JSON.stringify(result, null, 2)], { type: 'application/json' });
                      const url = URL.createObjectURL(blob);
                      const a = document.createElement('a');
                      a.href = url;
                      a.download = `risk_analysis_${portfolioId}.json`;
                      a.click();
                      URL.revokeObjectURL(url);
                    }}
                  >
                    Export
                  </Button>
                </Space>
              </Col>
            </Row>
          </Card>
        </Col>
        
        {/* Risk Alerts */}
        {riskAlerts.length > 0 && (
          <Col span={24}>
            <Card size="small">
              <Space direction="vertical" style={{ width: '100%' }}>
                {riskAlerts.slice(0, 3).map((alert) => (
                  <Alert
                    key={alert.id}
                    message={alert.message}
                    type={alert.severity === 'critical' ? 'error' : 'warning'}
                    showIcon
                    icon={<AlertOutlined />}
                    closable
                    style={{ marginBottom: 8 }}
                    description={
                      <Text type="secondary">
                        Current: {alert.current_value.toFixed(2)} | 
                        Threshold: {alert.threshold_value.toFixed(2)} | 
                        Time: {alert.timestamp.toLocaleTimeString()}
                      </Text>
                    }
                  />
                ))}
                {riskAlerts.length > 3 && (
                  <Button type="link" size="small">
                    View all {riskAlerts.length} alerts
                  </Button>
                )}
              </Space>
            </Card>
          </Col>
        )}
        
        {/* Portfolio Risk Summary */}
        {result && (
          <Col span={24}>
            <Row gutter={[16, 16]}>
              <Col xs={12} sm={6}>
                <Card size="small">
                  <Statistic
                    title="Portfolio Value"
                    value={result.portfolio_metrics.portfolio_value}
                    formatter={(value) => `$${Number(value).toLocaleString()}`}
                    valueStyle={{ fontSize: '16px' }}
                  />
                </Card>
              </Col>
              <Col xs={12} sm={6}>
                <Card size="small">
                  <Statistic
                    title="Net Exposure"
                    value={result.portfolio_metrics.net_exposure}
                    formatter={(value) => `$${Number(value).toLocaleString()}`}
                    valueStyle={{ fontSize: '16px' }}
                  />
                </Card>
              </Col>
              <Col xs={12} sm={6}>
                <Card size="small">
                  <Statistic
                    title="Leverage"
                    value={result.portfolio_metrics.leverage}
                    precision={2}
                    suffix="x"
                    valueStyle={{ 
                      fontSize: '16px',
                      color: result.portfolio_metrics.leverage > 3 ? '#ff4d4f' : 
                             result.portfolio_metrics.leverage > 2 ? '#faad14' : '#52c41a'
                    }}
                  />
                </Card>
              </Col>
              <Col xs={12} sm={6}>
                <Card size="small">
                  <Statistic
                    title="Portfolio Beta"
                    value={result.portfolio_metrics.beta}
                    precision={2}
                    valueStyle={{ fontSize: '16px' }}
                  />
                </Card>
              </Col>
            </Row>
          </Col>
        )}
        
        {/* Main Content Tabs */}
        <Col span={24}>
          <Card>
            <Tabs activeKey={activeTab} onChange={setActiveTab}>
              <TabPane tab={<Space><BarChartOutlined />VaR Analysis</Space>} key="var">
                <Row gutter={[16, 16]}>
                  {/* VaR Summary */}
                  {result && result.var_analysis && (
                    <Col span={24}>
                      <Row gutter={[16, 16]}>
                        {result.var_analysis.map((va, index) => (
                          <Col xs={24} sm={12} md={8} key={index}>
                            <Card size="small" title={`VaR ${(va.confidence_level * 100).toFixed(0)}%`}>
                              <Space direction="vertical" style={{ width: '100%' }}>
                                <Statistic
                                  title="Historical VaR"
                                  value={Math.abs(va.historical_var)}
                                  formatter={(value) => `$${Number(value).toLocaleString()}`}
                                  valueStyle={{ fontSize: '14px', color: '#ff4d4f' }}
                                />
                                <Statistic
                                  title="Parametric VaR"
                                  value={Math.abs(va.parametric_var)}
                                  formatter={(value) => `$${Number(value).toLocaleString()}`}
                                  valueStyle={{ fontSize: '14px', color: '#1890ff' }}
                                />
                                <Statistic
                                  title="Expected Shortfall"
                                  value={Math.abs(va.expected_shortfall)}
                                  formatter={(value) => `$${Number(value).toLocaleString()}`}
                                  valueStyle={{ fontSize: '14px', color: '#faad14' }}
                                />
                              </Space>
                            </Card>
                          </Col>
                        ))}
                      </Row>
                    </Col>
                  )}
                  
                  <Col span={24}>
                    <Card title="VaR Comparison by Method" size="small">
                      {varChartData.length > 0 ? (
                        <Column {...varChartConfig} />
                      ) : (
                        <div style={{ height: 300, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                          No VaR data available
                        </div>
                      )}
                    </Card>
                  </Col>
                  
                  {riskTimeSeriesData.length > 0 && (
                    <Col span={24}>
                      <Card title="VaR Time Series" size="small">
                        <Line {...riskTimeSeriesConfig} />
                      </Card>
                    </Col>
                  )}
                </Row>
              </TabPane>
              
              <TabPane tab={<Space><ThunderboltOutlined />Stress Testing</Space>} key="stress">
                <Row gutter={[16, 16]}>
                  <Col span={24}>
                    <Card title="Stress Test Results" size="small">
                      {stressTestData.length > 0 ? (
                        <Column {...stressTestConfig} />
                      ) : (
                        <div style={{ height: 300, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                          No stress test data available
                        </div>
                      )}
                    </Card>
                  </Col>
                  
                  {result && result.stress_testing && (
                    <Col span={24}>
                      <Table
                        dataSource={result.stress_testing}
                        size="small"
                        pagination={false}
                        scroll={{ x: 800 }}
                        columns={[
                          {
                            title: 'Scenario',
                            dataIndex: 'scenario_name',
                            key: 'scenario_name',
                            width: 150,
                          },
                          {
                            title: 'Description',
                            dataIndex: 'description',
                            key: 'description',
                          },
                          {
                            title: 'Base Value',
                            dataIndex: 'base_portfolio_value',
                            key: 'base_portfolio_value',
                            render: (value: number) => `$${value.toLocaleString()}`,
                          },
                          {
                            title: 'Stressed Value',
                            dataIndex: 'stressed_portfolio_value',
                            key: 'stressed_portfolio_value',
                            render: (value: number) => `$${value.toLocaleString()}`,
                          },
                          {
                            title: 'Absolute Loss',
                            dataIndex: 'absolute_loss',
                            key: 'absolute_loss',
                            render: (value: number) => (
                              <span style={{ color: '#ff4d4f' }}>
                                ${Math.abs(value).toLocaleString()}
                              </span>
                            ),
                          },
                          {
                            title: 'Loss %',
                            dataIndex: 'percentage_loss',
                            key: 'percentage_loss',
                            render: (value: number) => (
                              <span style={{ color: '#ff4d4f' }}>
                                {(Math.abs(value) * 100).toFixed(2)}%
                              </span>
                            ),
                          },
                        ]}
                      />
                    </Col>
                  )}
                </Row>
              </TabPane>
              
              <TabPane tab={<Space><RadarChartOutlined />Exposure Analysis</Space>} key="exposure">
                <Row gutter={[16, 16]}>
                  <Col span={24}>
                    <Space style={{ marginBottom: 16 }}>
                      <Text>Exposure Type:</Text>
                      <Select 
                        value={selectedExposureType} 
                        onChange={setSelectedExposureType}
                        style={{ width: 150 }}
                        size="small"
                      >
                        <Option value="sector">Sector</Option>
                        <Option value="currency">Currency</Option>
                        <Option value="geographic">Geographic</Option>
                        <Option value="all">All</Option>
                      </Select>
                    </Space>
                  </Col>
                  
                  <Col xs={24} lg={16}>
                    <Card title="Exposure by Sector" size="small">
                      {exposureData.length > 0 ? (
                        <Column {...exposureConfig} />
                      ) : (
                        <div style={{ height: 300, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                          No exposure data available
                        </div>
                      )}
                    </Card>
                  </Col>
                  
                  <Col xs={24} lg={8}>
                    <Card title="Risk Decomposition" size="small">
                      {result && result.risk_decomposition && (
                        <Space direction="vertical" style={{ width: '100%' }}>
                          <div>
                            <Text type="secondary">Systematic Risk</Text>
                            <Progress 
                              percent={(result.risk_decomposition.systematic_risk / result.risk_decomposition.total_risk) * 100}
                              format={percent => `${percent?.toFixed(1)}%`}
                              strokeColor="#1890ff"
                            />
                          </div>
                          <div>
                            <Text type="secondary">Idiosyncratic Risk</Text>
                            <Progress 
                              percent={(result.risk_decomposition.idiosyncratic_risk / result.risk_decomposition.total_risk) * 100}
                              format={percent => `${percent?.toFixed(1)}%`}
                              strokeColor="#52c41a"
                            />
                          </div>
                          <div>
                            <Text type="secondary">Concentration Risk</Text>
                            <Progress 
                              percent={result.risk_decomposition.concentration_risk * 100}
                              format={percent => `${percent?.toFixed(1)}%`}
                              strokeColor="#faad14"
                            />
                          </div>
                          <div>
                            <Text type="secondary">Liquidity Risk</Text>
                            <Progress 
                              percent={result.risk_decomposition.liquidity_risk_score * 100}
                              format={percent => `${percent?.toFixed(1)}%`}
                              strokeColor="#ff4d4f"
                            />
                          </div>
                        </Space>
                      )}
                    </Card>
                  </Col>
                  
                  <Col span={24}>
                    <Card title="Detailed Exposure Breakdown" size="small">
                      <Table
                        dataSource={exposureData}
                        columns={exposureColumns}
                        rowKey="sector"
                        size="small"
                        pagination={false}
                        scroll={{ y: 300 }}
                      />
                    </Card>
                  </Col>
                </Row>
              </TabPane>
              
              <TabPane tab={<Space><TableOutlined />Component Analysis</Space>} key="component">
                <Row gutter={[16, 16]}>
                  {componentVarData.length > 0 && (
                    <Col span={24}>
                      <Card title="Component VaR Analysis" size="small">
                        <Table
                          dataSource={componentVarData}
                          columns={componentVarColumns}
                          rowKey="asset"
                          size="small"
                          pagination={false}
                          scroll={{ x: 800, y: 400 }}
                        />
                      </Card>
                    </Col>
                  )}
                  
                  {correlationData.length > 0 && (
                    <Col span={24}>
                      <Card title="Asset Correlation Heatmap" size="small">
                        <Heatmap {...correlationHeatmapConfig} />
                        {result && result.correlation_matrix && (
                          <div style={{ marginTop: 16 }}>
                            <Space>
                              <Text type="secondary">
                                Average Correlation: {(result.correlation_matrix.avg_correlation * 100).toFixed(1)}%
                              </Text>
                              <Text type="secondary">
                                Condition Number: {result.correlation_matrix.condition_number.toFixed(2)}
                              </Text>
                            </Space>
                          </div>
                        )}
                      </Card>
                    </Col>
                  )}
                </Row>
              </TabPane>
            </Tabs>
          </Card>
        </Col>
      </Row>
      
      {/* Settings Modal */}
      <Modal
        title="Risk Analytics Settings"
        open={showSettings}
        onOk={() => setShowSettings(false)}
        onCancel={() => setShowSettings(false)}
        width={600}
      >
        <Space direction="vertical" style={{ width: '100%' }}>
          <div>
            <Text strong>Confidence Levels</Text>
            <Space>
              {[0.90, 0.95, 0.99].map(level => (
                <Button
                  key={level}
                  type={confidenceLevels.includes(level) ? 'primary' : 'default'}
                  size="small"
                  onClick={() => {
                    if (confidenceLevels.includes(level)) {
                      setConfidenceLevels(confidenceLevels.filter(cl => cl !== level));
                    } else {
                      setConfidenceLevels([...confidenceLevels, level]);
                    }
                  }}
                >
                  {(level * 100).toFixed(0)}%
                </Button>
              ))}
            </Space>
          </div>
          
          <div>
            <Text strong>VaR Methods</Text>
            <Space>
              {[
                { key: 'historical', label: 'Historical' },
                { key: 'parametric', label: 'Parametric' },
                { key: 'monte_carlo', label: 'Monte Carlo' },
              ].map(method => (
                <Button
                  key={method.key}
                  type={varMethods.includes(method.key as any) ? 'primary' : 'default'}
                  size="small"
                  onClick={() => {
                    if (varMethods.includes(method.key as any)) {
                      setVarMethods(varMethods.filter(vm => vm !== method.key));
                    } else {
                      setVarMethods([...varMethods, method.key as any]);
                    }
                  }}
                >
                  {method.label}
                </Button>
              ))}
            </Space>
          </div>
        </Space>
      </Modal>
    </div>
  );
};

export default RiskAnalytics;