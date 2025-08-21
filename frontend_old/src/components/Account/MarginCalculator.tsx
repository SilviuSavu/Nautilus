/**
 * Margin Calculator component for detailed margin analysis
 */

import React, { useState, useMemo } from 'react';
import { Card, Row, Col, Statistic, Progress, Alert, Space, Typography, Slider, InputNumber } from 'antd';
import { 
  CalculatorOutlined, 
  WarningOutlined, 
  InfoCircleOutlined,
  ExclamationCircleOutlined
} from '@ant-design/icons';
import { usePositionData } from '../../hooks/usePositionData';

const { Title, Text } = Typography;

interface MarginCalculatorProps {
  className?: string;
}

interface MarginScenario {
  name: string;
  marginUsage: number;
  riskLevel: 'low' | 'medium' | 'high' | 'critical';
  description: string;
}

export const MarginCalculator: React.FC<MarginCalculatorProps> = ({ className }) => {
  const { accountBalances } = usePositionData();
  const [simulatedExposure, setSimulatedExposure] = useState<number>(0);
  
  const primaryAccount = accountBalances[0];

  const marginScenarios = useMemo((): MarginScenario[] => {
    if (!primaryAccount) return [];
    
    return [
      {
        name: 'Conservative',
        marginUsage: 30,
        riskLevel: 'low',
        description: 'Safe margin usage with plenty of buffer for market volatility'
      },
      {
        name: 'Moderate',
        marginUsage: 50,
        riskLevel: 'medium',
        description: 'Balanced approach with moderate risk tolerance'
      },
      {
        name: 'Aggressive',
        marginUsage: 70,
        riskLevel: 'high',
        description: 'Higher leverage with increased risk of margin calls'
      },
      {
        name: 'Maximum',
        marginUsage: 90,
        riskLevel: 'critical',
        description: 'Near maximum leverage - high risk of margin call'
      }
    ];
  }, [primaryAccount]);

  if (!primaryAccount) {
    return (
      <Card className={className}>
        <Alert 
          message="No Account Data" 
          description="Margin calculator requires account information" 
          type="info" 
          showIcon 
        />
      </Card>
    );
  }

  const totalMargin = primaryAccount.marginUsed + primaryAccount.marginAvailable;
  const currentMarginUsage = totalMargin > 0 ? (primaryAccount.marginUsed / totalMargin) * 100 : 0;
  
  // Calculate additional buying power based on simulated exposure
  const additionalMarginRequired = simulatedExposure * 0.3; // Assume 30% margin requirement
  const projectedMarginUsed = primaryAccount.marginUsed + additionalMarginRequired;
  const projectedMarginUsage = totalMargin > 0 ? (projectedMarginUsed / totalMargin) * 100 : 0;

  const formatCurrency = (value: number) => {
    return new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency: primaryAccount.currency,
      minimumFractionDigits: 2,
      maximumFractionDigits: 2
    }).format(value);
  };

  const getMarginRiskColor = (usage: number) => {
    if (usage < 30) return '#52c41a';
    if (usage < 50) return '#faad14';
    if (usage < 70) return '#fa8c16';
    if (usage < 90) return '#f5222d';
    return '#a8071a';
  };

  const getMarginRiskLevel = (usage: number): string => {
    if (usage < 30) return 'Low Risk';
    if (usage < 50) return 'Medium Risk';
    if (usage < 70) return 'High Risk';
    if (usage < 90) return 'Very High Risk';
    return 'Critical Risk';
  };

  return (
    <div className={className}>
      <Title level={4}>
        <CalculatorOutlined /> Margin Calculator
      </Title>

      <Row gutter={[16, 16]}>
        {/* Current Margin Status */}
        <Col span={24}>
          <Card title="Current Margin Status">
            <Row gutter={[16, 16]}>
              <Col xs={24} md={8}>
                <Statistic
                  title="Margin Used"
                  value={primaryAccount.marginUsed}
                  formatter={(value) => formatCurrency(Number(value))}
                />
              </Col>
              <Col xs={24} md={8}>
                <Statistic
                  title="Margin Available"
                  value={primaryAccount.marginAvailable}
                  formatter={(value) => formatCurrency(Number(value))}
                />
              </Col>
              <Col xs={24} md={8}>
                <Space direction="vertical" style={{ width: '100%' }}>
                  <Text strong>Current Usage: {currentMarginUsage.toFixed(1)}%</Text>
                  <Progress
                    percent={currentMarginUsage}
                    strokeColor={getMarginRiskColor(currentMarginUsage)}
                    format={(percent) => `${percent?.toFixed(1)}%`}
                  />
                  <Text type="secondary">{getMarginRiskLevel(currentMarginUsage)}</Text>
                </Space>
              </Col>
            </Row>
          </Card>
        </Col>

        {/* Margin Scenarios */}
        <Col span={24}>
          <Card title="Margin Usage Scenarios">
            <Row gutter={[16, 16]}>
              {marginScenarios.map((scenario) => {
                const requiredMargin = (scenario.marginUsage / 100) * totalMargin;
                const availableExposure = (requiredMargin - primaryAccount.marginUsed) / 0.3; // Assume 30% margin req
                
                return (
                  <Col xs={24} sm={12} md={6} key={scenario.name}>
                    <Card 
                      size="small"
                      style={{ 
                        borderColor: getMarginRiskColor(scenario.marginUsage),
                        borderWidth: 2 
                      }}
                    >
                      <Space direction="vertical" style={{ width: '100%' }}>
                        <Text strong>{scenario.name}</Text>
                        <Progress
                          percent={scenario.marginUsage}
                          strokeColor={getMarginRiskColor(scenario.marginUsage)}
                          size="small"
                        />
                        <Text type="secondary" style={{ fontSize: '12px' }}>
                          {scenario.description}
                        </Text>
                        {availableExposure > 0 && (
                          <Text type="secondary">
                            Additional exposure: {formatCurrency(availableExposure)}
                          </Text>
                        )}
                      </Space>
                    </Card>
                  </Col>
                );
              })}
            </Row>
          </Card>
        </Col>

        {/* Exposure Simulator */}
        <Col span={24}>
          <Card title="Position Size Simulator">
            <Row gutter={[16, 16]}>
              <Col xs={24} md={12}>
                <Space direction="vertical" style={{ width: '100%' }}>
                  <Text strong>Simulate Additional Exposure:</Text>
                  <Slider
                    min={0}
                    max={primaryAccount.buyingPower}
                    step={1000}
                    value={simulatedExposure}
                    onChange={setSimulatedExposure}
                    tipFormatter={(value) => formatCurrency(value || 0)}
                  />
                  <InputNumber
                    style={{ width: '100%' }}
                    value={simulatedExposure}
                    onChange={(value) => setSimulatedExposure(value || 0)}
                    formatter={(value) => `$ ${value}`.replace(/\B(?=(\d{3})+(?!\d))/g, ',')}
                    parser={(value) => value?.replace(/\$\s?|(,*)/g, '') as unknown as number}
                    min={0}
                    max={primaryAccount.buyingPower}
                  />
                </Space>
              </Col>
              <Col xs={24} md={12}>
                <Space direction="vertical" style={{ width: '100%' }}>
                  <Text strong>Projected Margin Usage:</Text>
                  <Progress
                    percent={Math.min(projectedMarginUsage, 100)}
                    strokeColor={getMarginRiskColor(projectedMarginUsage)}
                    format={(percent) => `${percent?.toFixed(1)}%`}
                  />
                  <Text type="secondary">
                    Additional margin required: {formatCurrency(additionalMarginRequired)}
                  </Text>
                  {projectedMarginUsage > 80 && (
                    <Alert
                      message="High Risk"
                      description="This position size would result in high margin usage"
                      type="warning"
                      showIcon
                      icon={<WarningOutlined />}
                    />
                  )}
                  {projectedMarginUsage > 100 && (
                    <Alert
                      message="Insufficient Margin"
                      description="This position size exceeds available margin"
                      type="error"
                      showIcon
                      icon={<ExclamationCircleOutlined />}
                    />
                  )}
                </Space>
              </Col>
            </Row>
          </Card>
        </Col>

        {/* Risk Warnings */}
        <Col span={24}>
          <Card>
            <Alert
              message="Margin Trading Risks"
              description={
                <Space direction="vertical">
                  <Text>• Margin trading involves significant risk and may not be suitable for all investors</Text>
                  <Text>• You may lose more than your initial investment</Text>
                  <Text>• Margin calls may force you to liquidate positions at unfavorable prices</Text>
                  <Text>• Interest charges apply to margin loans</Text>
                </Space>
              }
              type="info"
              showIcon
              icon={<InfoCircleOutlined />}
            />
          </Card>
        </Col>
      </Row>
    </div>
  );
};

export default MarginCalculator;