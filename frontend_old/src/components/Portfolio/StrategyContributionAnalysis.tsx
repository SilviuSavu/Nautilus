/**
 * Strategy Contribution Analysis Component
 */

import React, { useState, useEffect, useMemo } from 'react';
import { Card, Table, Progress, Row, Col, Statistic, Select, Switch, Alert, Space, Tag } from 'antd';
import { ColumnType } from 'antd/es/table';
import { TrophyOutlined, RiseOutlined, FallOutlined, DollarOutlined } from '@ant-design/icons';
import { 
  portfolioAggregationService, 
  PortfolioAggregation, 
  StrategyPnL 
} from '../../services/portfolioAggregationService';
import {
  attributionCalculator,
  AttributionBreakdown,
  AttributionAnalysis
} from '../../services/performanceAttributionCalculator';

const { Option } = Select;

interface StrategyContributionAnalysisProps {
  period?: '1D' | '1W' | '1M' | '3M' | '6M' | '1Y';
  showAttribution?: boolean;
  maxStrategies?: number;
}

const StrategyContributionAnalysis: React.FC<StrategyContributionAnalysisProps> = ({
  period = '1M',
  showAttribution = true,
  maxStrategies = 10
}) => {
  const [portfolioData, setPortfolioData] = useState<PortfolioAggregation | null>(null);
  const [attributionData, setAttributionData] = useState<AttributionAnalysis | null>(null);
  const [selectedPeriod, setSelectedPeriod] = useState(period);
  const [sortBy, setSortBy] = useState<'contribution' | 'pnl' | 'return' | 'weight'>('contribution');
  const [showPositiveOnly, setShowPositiveOnly] = useState(false);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const handleAggregationUpdate = (aggregation: PortfolioAggregation) => {
      setPortfolioData(aggregation);
      
      // Calculate attribution analysis
      if (showAttribution && aggregation.strategies.length > 0) {
        const attribution = attributionCalculator.calculateStrategyAttribution(
          aggregation.strategies,
          aggregation.total_pnl,
          selectedPeriod
        );
        
        setAttributionData({
          strategy_attributions: attribution,
          sector_attributions: [],
          factor_attributions: [],
          total_active_return: attribution.reduce((sum, attr) => sum + attr.active_return, 0),
          total_tracking_error: Math.sqrt(attribution.reduce((sum, attr) => sum + Math.pow(attr.tracking_error, 2), 0)),
          information_ratio: 0,
          period: selectedPeriod
        });
      }
      
      setLoading(false);
    };

    portfolioAggregationService.addAggregationHandler(handleAggregationUpdate);

    const initialData = portfolioAggregationService.getPortfolioAggregation();
    if (initialData) {
      handleAggregationUpdate(initialData);
    }

    return () => {
      portfolioAggregationService.removeAggregationHandler(handleAggregationUpdate);
    };
  }, [selectedPeriod, showAttribution]);

  // Process and sort strategy data
  const sortedStrategies = useMemo(() => {
    if (!portfolioData) return [];
    
    let strategies = [...portfolioData.strategies];
    
    // Filter positive only if enabled
    if (showPositiveOnly) {
      strategies = strategies.filter(s => s.total_pnl > 0);
    }
    
    // Sort by selected criteria
    switch (sortBy) {
      case 'contribution':
        strategies.sort((a, b) => Math.abs(b.contribution_percent) - Math.abs(a.contribution_percent));
        break;
      case 'pnl':
        strategies.sort((a, b) => b.total_pnl - a.total_pnl);
        break;
      case 'return':
        strategies.sort((a, b) => {
          const aReturn = a.total_pnl / (a.weight || 1);
          const bReturn = b.total_pnl / (b.weight || 1);
          return bReturn - aReturn;
        });
        break;
      case 'weight':
        strategies.sort((a, b) => b.weight - a.weight);
        break;
    }
    
    return strategies.slice(0, maxStrategies);
  }, [portfolioData, sortBy, showPositiveOnly, maxStrategies]);

  // Calculate summary statistics
  const summaryStats = useMemo(() => {
    if (!portfolioData) return null;
    
    const totalStrategies = portfolioData.strategies.length;
    const profitableStrategies = portfolioData.strategies.filter(s => s.total_pnl > 0).length;
    const topContributor = portfolioData.strategies.reduce((top, strategy) => 
      Math.abs(strategy.contribution_percent) > Math.abs(top.contribution_percent) ? strategy : top,
      portfolioData.strategies[0]
    );
    const concentrationRisk = portfolioData.strategies
      .slice(0, 3)
      .reduce((sum, s) => sum + Math.abs(s.contribution_percent), 0);

    return {
      totalStrategies,
      profitableStrategies,
      profitableRate: totalStrategies > 0 ? (profitableStrategies / totalStrategies) * 100 : 0,
      topContributor,
      concentrationRisk
    };
  }, [portfolioData]);

  // Table columns configuration
  const columns: ColumnType<StrategyPnL>[] = [
    {
      title: 'Strategy',
      dataIndex: 'strategy_name',
      key: 'strategy_name',
      width: 150,
      fixed: 'left',
      render: (name: string, record: StrategyPnL) => (
        <div>
          <div style={{ fontWeight: 'bold' }}>{name}</div>
          <div style={{ fontSize: '12px', color: '#666' }}>
            {record.positions_count} positions
          </div>
        </div>
      ),
    },
    {
      title: 'P&L',
      dataIndex: 'total_pnl',
      key: 'total_pnl',
      width: 120,
      sorter: (a, b) => a.total_pnl - b.total_pnl,
      render: (pnl: number) => (
        <div style={{ 
          color: pnl >= 0 ? '#52c41a' : '#ff4d4f',
          fontWeight: 'bold'
        }}>
          {pnl >= 0 ? '+' : ''}${pnl.toLocaleString()}
        </div>
      ),
    },
    {
      title: 'Contribution %',
      dataIndex: 'contribution_percent',
      key: 'contribution_percent',
      width: 150,
      sorter: (a, b) => a.contribution_percent - b.contribution_percent,
      render: (contribution: number) => (
        <div>
          <Progress
            percent={Math.abs(contribution)}
            showInfo={false}
            strokeColor={contribution >= 0 ? '#52c41a' : '#ff4d4f'}
            trailColor="#f0f0f0"
            size="small"
          />
          <div style={{ 
            textAlign: 'center', 
            fontSize: '12px',
            color: contribution >= 0 ? '#52c41a' : '#ff4d4f',
            fontWeight: 'bold'
          }}>
            {contribution >= 0 ? '+' : ''}{contribution.toFixed(1)}%
          </div>
        </div>
      ),
    },
    {
      title: 'Weight %',
      dataIndex: 'weight',
      key: 'weight',
      width: 100,
      sorter: (a, b) => a.weight - b.weight,
      render: (weight: number) => `${weight.toFixed(1)}%`,
    },
    {
      title: 'Unrealized',
      dataIndex: 'unrealized_pnl',
      key: 'unrealized_pnl',
      width: 100,
      render: (unrealized: number) => (
        <span style={{ color: '#faad14' }}>
          ${unrealized.toLocaleString()}
        </span>
      ),
    },
    {
      title: 'Realized',
      dataIndex: 'realized_pnl',
      key: 'realized_pnl',
      width: 100,
      render: (realized: number) => (
        <span style={{ color: '#1890ff' }}>
          ${realized.toLocaleString()}
        </span>
      ),
    },
    {
      title: 'Status',
      key: 'status',
      width: 100,
      render: (_, record: StrategyPnL) => {
        const performance = record.total_pnl >= 0 ? 'profit' : 'loss';
        const contribution = Math.abs(record.contribution_percent);
        
        let tag = 'default';
        let text = 'Normal';
        
        if (contribution > 20) {
          tag = 'red';
          text = 'High Impact';
        } else if (contribution > 10) {
          tag = 'orange';
          text = 'Medium Impact';
        } else if (record.total_pnl > 0) {
          tag = 'green';
          text = 'Profitable';
        }
        
        return <Tag color={tag}>{text}</Tag>;
      },
    },
  ];

  // Attribution columns if enabled
  const attributionColumns: ColumnType<AttributionBreakdown>[] = [
    {
      title: 'Active Return',
      dataIndex: 'active_return',
      key: 'active_return',
      width: 100,
      render: (activeReturn: number) => (
        <span style={{ color: activeReturn >= 0 ? '#52c41a' : '#ff4d4f' }}>
          {activeReturn >= 0 ? '+' : ''}{activeReturn.toFixed(2)}%
        </span>
      ),
    },
    {
      title: 'Tracking Error',
      dataIndex: 'tracking_error',
      key: 'tracking_error',
      width: 100,
      render: (trackingError: number) => `${trackingError.toFixed(2)}%`,
    },
    {
      title: 'Info Ratio',
      dataIndex: 'information_ratio',
      key: 'information_ratio',
      width: 100,
      render: (infoRatio: number) => (
        <span style={{ 
          color: infoRatio > 0.5 ? '#52c41a' : infoRatio > 0 ? '#faad14' : '#ff4d4f' 
        }}>
          {infoRatio.toFixed(2)}
        </span>
      ),
    },
    {
      title: 'Allocation Effect',
      dataIndex: 'allocation_effect',
      key: 'allocation_effect',
      width: 120,
      render: (effect: number) => (
        <span style={{ color: effect >= 0 ? '#52c41a' : '#ff4d4f' }}>
          {effect >= 0 ? '+' : ''}{effect.toFixed(3)}%
        </span>
      ),
    },
    {
      title: 'Selection Effect',
      dataIndex: 'selection_effect',
      key: 'selection_effect',
      width: 120,
      render: (effect: number) => (
        <span style={{ color: effect >= 0 ? '#52c41a' : '#ff4d4f' }}>
          {effect >= 0 ? '+' : ''}{effect.toFixed(3)}%
        </span>
      ),
    },
  ];

  const allColumns = showAttribution && attributionData 
    ? [...columns, ...attributionColumns]
    : columns;

  if (loading) {
    return (
      <Card title="Strategy Contribution Analysis" loading={true}>
        <div style={{ height: 400 }}>Loading strategy data...</div>
      </Card>
    );
  }

  if (!portfolioData || portfolioData.strategies.length === 0) {
    return (
      <Card title="Strategy Contribution Analysis">
        <Alert
          message="No Strategy Data"
          description="No active strategies found in the portfolio."
          type="info"
          showIcon
        />
      </Card>
    );
  }

  return (
    <Card
      title="Strategy Contribution Analysis"
      extra={
        <Space>
          <Select 
            value={selectedPeriod} 
            onChange={setSelectedPeriod}
            style={{ width: 80 }}
          >
            <Option value="1D">1D</Option>
            <Option value="1W">1W</Option>
            <Option value="1M">1M</Option>
            <Option value="3M">3M</Option>
            <Option value="6M">6M</Option>
            <Option value="1Y">1Y</Option>
          </Select>
          <Select 
            value={sortBy} 
            onChange={setSortBy}
            style={{ width: 120 }}
          >
            <Option value="contribution">Contribution</Option>
            <Option value="pnl">P&L</Option>
            <Option value="return">Return</Option>
            <Option value="weight">Weight</Option>
          </Select>
          <Switch
            checked={showPositiveOnly}
            onChange={setShowPositiveOnly}
            checkedChildren="Profitable"
            unCheckedChildren="All"
          />
        </Space>
      }
    >
      {/* Summary Statistics */}
      {summaryStats && (
        <Row gutter={16} style={{ marginBottom: 16 }}>
          <Col span={6}>
            <Statistic
              title="Total Strategies"
              value={summaryStats.totalStrategies}
              prefix={<TrophyOutlined />}
              valueStyle={{ color: '#1890ff' }}
            />
          </Col>
          <Col span={6}>
            <Statistic
              title="Profitable Rate"
              value={summaryStats.profitableRate}
              precision={1}
              suffix="%"
              prefix={<RiseOutlined />}
              valueStyle={{ 
                color: summaryStats.profitableRate >= 60 ? '#3f8600' : 
                       summaryStats.profitableRate >= 40 ? '#faad14' : '#cf1322'
              }}
            />
          </Col>
          <Col span={6}>
            <Statistic
              title="Top Contributor"
              value={summaryStats.topContributor.contribution_percent}
              precision={1}
              suffix="%"
              prefix={<DollarOutlined />}
              valueStyle={{ 
                color: summaryStats.topContributor.total_pnl >= 0 ? '#3f8600' : '#cf1322'
              }}
            />
            <div style={{ fontSize: '12px', color: '#666' }}>
              {summaryStats.topContributor.strategy_name}
            </div>
          </Col>
          <Col span={6}>
            <Statistic
              title="Top 3 Concentration"
              value={summaryStats.concentrationRisk}
              precision={1}
              suffix="%"
              prefix={<FallOutlined />}
              valueStyle={{ 
                color: summaryStats.concentrationRisk > 75 ? '#cf1322' : 
                       summaryStats.concentrationRisk > 50 ? '#faad14' : '#3f8600'
              }}
            />
          </Col>
        </Row>
      )}

      {/* Strategy Table */}
      <Table
        columns={allColumns}
        dataSource={sortedStrategies}
        rowKey="strategy_id"
        pagination={false}
        scroll={{ x: 1200 }}
        size="small"
        bordered
        rowClassName={(record) => 
          record.total_pnl >= 0 ? 'strategy-profitable' : 'strategy-loss'
        }
      />

      {/* Attribution Analysis Summary */}
      {showAttribution && attributionData && (
        <div style={{ marginTop: 16 }}>
          <Row gutter={16}>
            <Col span={8}>
              <Card size="small" title="Active Return">
                <Statistic
                  value={attributionData.total_active_return}
                  precision={2}
                  suffix="%"
                  valueStyle={{ 
                    color: attributionData.total_active_return >= 0 ? '#3f8600' : '#cf1322'
                  }}
                />
              </Card>
            </Col>
            <Col span={8}>
              <Card size="small" title="Tracking Error">
                <Statistic
                  value={attributionData.total_tracking_error}
                  precision={2}
                  suffix="%"
                  valueStyle={{ color: '#1890ff' }}
                />
              </Card>
            </Col>
            <Col span={8}>
              <Card size="small" title="Information Ratio">
                <Statistic
                  value={attributionData.information_ratio}
                  precision={2}
                  valueStyle={{ 
                    color: attributionData.information_ratio > 0.5 ? '#3f8600' : 
                           attributionData.information_ratio > 0 ? '#faad14' : '#cf1322'
                  }}
                />
              </Card>
            </Col>
          </Row>
        </div>
      )}

      <style jsx>{`
        .strategy-profitable {
          background-color: rgba(82, 196, 26, 0.05);
        }
        .strategy-loss {
          background-color: rgba(255, 77, 79, 0.05);
        }
      `}</style>
    </Card>
  );
};

export default StrategyContributionAnalysis;