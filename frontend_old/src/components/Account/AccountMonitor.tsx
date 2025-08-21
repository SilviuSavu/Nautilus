/**
 * Account Monitor component for real-time balance display
 */

import React from 'react';
import { Card, Row, Col, Progress, Typography, Space, Statistic, Alert, Tag } from 'antd';
import { 
  DollarOutlined, 
  ArrowUpOutlined, 
  ArrowDownOutlined,
  WarningOutlined,
  CheckCircleOutlined
} from '@ant-design/icons';
import { usePositionData } from '../../hooks/usePositionData';

const { Title, Text } = Typography;

interface AccountMonitorProps {
  className?: string;
  showTitle?: boolean;
  currency?: string;
}

export const AccountMonitor: React.FC<AccountMonitorProps> = ({ 
  className,
  showTitle = true,
  currency = 'USD'
}) => {
  const { 
    accountBalances, 
    positionSummary, 
    isLoading, 
    error,
    hasAccounts,
    lastUpdate
  } = usePositionData();

  const primaryAccount = accountBalances.find(acc => acc.currency === currency) || accountBalances[0];

  if (error) {
    return (
      <Alert 
        message="Account Data Error" 
        description={error} 
        type="error" 
        showIcon 
        className={className}
      />
    );
  }

  if (isLoading && !hasAccounts) {
    return (
      <Card loading={true} className={className}>
        <Title level={4}>Account Information</Title>
      </Card>
    );
  }

  if (!primaryAccount) {
    return (
      <Card className={className}>
        {showTitle && <Title level={4}>Account Information</Title>}
        <Alert 
          message="No Account Data" 
          description="Account information is not available" 
          type="info" 
          showIcon 
        />
      </Card>
    );
  }

  const marginUsagePercent = primaryAccount.marginAvailable > 0 
    ? (primaryAccount.marginUsed / (primaryAccount.marginUsed + primaryAccount.marginAvailable)) * 100
    : 0;

  const marginStatus = marginUsagePercent > 80 ? 'exception' : marginUsagePercent > 60 ? 'normal' : 'success';

  const formatCurrency = (value: number, currency: string = primaryAccount.currency) => {
    return new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency: currency,
      minimumFractionDigits: 2,
      maximumFractionDigits: 2
    }).format(value);
  };

  const formatLastUpdate = () => {
    if (!lastUpdate) return 'Never';
    const date = new Date(lastUpdate);
    return date.toLocaleTimeString();
  };

  return (
    <div className={className}>
      {showTitle && <Title level={4}>Account Information</Title>}
      
      <Row gutter={[16, 16]}>
        {/* Account Balance Overview */}
        <Col span={24}>
          <Card>
            <Row gutter={[16, 16]}>
              <Col xs={24} sm={12} md={6}>
                <Statistic
                  title="Account Balance"
                  value={primaryAccount.balance}
                  formatter={(value) => formatCurrency(Number(value))}
                  prefix={<DollarOutlined />}
                  valueStyle={{ color: primaryAccount.balance >= 0 ? '#3f8600' : '#cf1322' }}
                />
              </Col>
              <Col xs={24} sm={12} md={6}>
                <Statistic
                  title="Available Balance"
                  value={primaryAccount.availableBalance}
                  formatter={(value) => formatCurrency(Number(value))}
                  valueStyle={{ color: primaryAccount.availableBalance >= 0 ? '#3f8600' : '#cf1322' }}
                />
              </Col>
              <Col xs={24} sm={12} md={6}>
                <Statistic
                  title="Account Equity"
                  value={primaryAccount.equity}
                  formatter={(value) => formatCurrency(Number(value))}
                  prefix={primaryAccount.equity >= primaryAccount.balance ? <ArrowUpOutlined /> : <ArrowDownOutlined />}
                  valueStyle={{ color: primaryAccount.equity >= primaryAccount.balance ? '#3f8600' : '#cf1322' }}
                />
              </Col>
              <Col xs={24} sm={12} md={6}>
                <Statistic
                  title="Buying Power"
                  value={primaryAccount.buyingPower}
                  formatter={(value) => formatCurrency(Number(value))}
                  valueStyle={{ color: primaryAccount.buyingPower > 0 ? '#3f8600' : '#cf1322' }}
                />
              </Col>
            </Row>
          </Card>
        </Col>

        {/* Margin Information */}
        <Col span={24}>
          <Card title="Margin Usage">
            <Row gutter={[16, 16]}>
              <Col xs={24} md={12}>
                <Space direction="vertical" style={{ width: '100%' }}>
                  <div>
                    <Text strong>Margin Used: </Text>
                    <Text>{formatCurrency(primaryAccount.marginUsed)}</Text>
                  </div>
                  <div>
                    <Text strong>Margin Available: </Text>
                    <Text>{formatCurrency(primaryAccount.marginAvailable)}</Text>
                  </div>
                  <Progress
                    percent={marginUsagePercent}
                    status={marginStatus}
                    format={(percent) => `${percent?.toFixed(1)}%`}
                  />
                  {marginUsagePercent > 80 && (
                    <Alert
                      message="High Margin Usage"
                      description="Your margin usage is above 80%. Consider reducing positions or adding funds."
                      type="warning"
                      showIcon
                      icon={<WarningOutlined />}
                    />
                  )}
                </Space>
              </Col>
              <Col xs={24} md={12}>
                <Space direction="vertical">
                  <div>
                    <Text strong>Currency: </Text>
                    <Tag color="blue">{primaryAccount.currency}</Tag>
                  </div>
                  <div>
                    <Text strong>Last Update: </Text>
                    <Text type="secondary">{formatLastUpdate()}</Text>
                  </div>
                  <div>
                    <Text strong>Status: </Text>
                    <Tag color={primaryAccount.availableBalance > 0 ? 'green' : 'red'} icon={<CheckCircleOutlined />}>
                      {primaryAccount.availableBalance > 0 ? 'Active' : 'Restricted'}
                    </Tag>
                  </div>
                </Space>
              </Col>
            </Row>
          </Card>
        </Col>

        {/* P&L Summary (if available) */}
        {positionSummary && (
          <Col span={24}>
            <Card title="Portfolio P&L">
              <Row gutter={[16, 16]}>
                <Col xs={24} sm={8}>
                  <Statistic
                    title="Unrealized P&L"
                    value={positionSummary.pnl.unrealizedPnl}
                    formatter={(value) => formatCurrency(Number(value), positionSummary.pnl.currency)}
                    valueStyle={{ color: positionSummary.pnl.unrealizedPnl >= 0 ? '#3f8600' : '#cf1322' }}
                    prefix={positionSummary.pnl.unrealizedPnl >= 0 ? <ArrowUpOutlined /> : <ArrowDownOutlined />}
                  />
                </Col>
                <Col xs={24} sm={8}>
                  <Statistic
                    title="Realized P&L"
                    value={positionSummary.pnl.realizedPnl}
                    formatter={(value) => formatCurrency(Number(value), positionSummary.pnl.currency)}
                    valueStyle={{ color: positionSummary.pnl.realizedPnl >= 0 ? '#3f8600' : '#cf1322' }}
                  />
                </Col>
                <Col xs={24} sm={8}>
                  <Statistic
                    title="Total P&L"
                    value={positionSummary.pnl.totalPnl}
                    formatter={(value) => formatCurrency(Number(value), positionSummary.pnl.currency)}
                    valueStyle={{ color: positionSummary.pnl.totalPnl >= 0 ? '#3f8600' : '#cf1322' }}
                    prefix={positionSummary.pnl.totalPnl >= 0 ? <ArrowUpOutlined /> : <ArrowDownOutlined />}
                  />
                </Col>
              </Row>
            </Card>
          </Col>
        )}

        {/* Multi-Currency Accounts */}
        {accountBalances.length > 1 && (
          <Col span={24}>
            <Card title="Multi-Currency Balances">
              <Row gutter={[16, 16]}>
                {accountBalances.map((account) => (
                  <Col xs={24} sm={12} md={8} key={account.currency}>
                    <Card size="small">
                      <Space direction="vertical" style={{ width: '100%' }}>
                        <Text strong>{account.currency}</Text>
                        <Statistic
                          title="Balance"
                          value={account.balance}
                          formatter={(value) => formatCurrency(Number(value), account.currency)}
                        />
                        <Statistic
                          title="Available"
                          value={account.availableBalance}
                          formatter={(value) => formatCurrency(Number(value), account.currency)}
                        />
                      </Space>
                    </Card>
                  </Col>
                ))}
              </Row>
            </Card>
          </Col>
        )}
      </Row>
    </div>
  );
};

export default AccountMonitor;