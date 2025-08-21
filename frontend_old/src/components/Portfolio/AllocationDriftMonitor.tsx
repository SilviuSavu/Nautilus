/**
 * Allocation Drift Monitor Component for tracking allocation changes
 */

import React, { useState, useEffect, useMemo } from 'react';
import { Card, Table, Progress, Alert, Row, Col, Statistic, Tag, Select, Button } from 'antd';
import { ColumnType } from 'antd/es/table';
import { WarningOutlined, RiseOutlined, FallOutlined, SyncOutlined } from '@ant-design/icons';
import { portfolioAggregationService, PortfolioAggregation } from '../../services/portfolioAggregationService';

const { Option } = Select;

interface AllocationDrift {
  category: string;
  target_allocation: number;
  current_allocation: number;
  drift_amount: number;
  drift_percentage: number;
  risk_level: 'low' | 'medium' | 'high' | 'critical';
  rebalance_needed: boolean;
  recommendation: string;
}

interface AllocationDriftMonitorProps {
  driftThreshold?: number;
  analysisType?: 'asset_class' | 'sector' | 'strategy';
}

const AllocationDriftMonitor: React.FC<AllocationDriftMonitorProps> = ({
  driftThreshold = 5,
  analysisType = 'asset_class'
}) => {
  const [portfolioData, setPortfolioData] = useState<PortfolioAggregation | null>(null);
  const [driftData, setDriftData] = useState<AllocationDrift[]>([]);
  const [selectedType, setSelectedType] = useState(analysisType);
  const [threshold, setThreshold] = useState(driftThreshold);
  const [loading, setLoading] = useState(true);

  // Target allocations (mock data - would come from strategy configuration)
  const targetAllocations: Record<string, Record<string, number>> = {
    asset_class: {
      'Stocks': 60,
      'Bonds': 25,
      'Cash': 10,
      'Other': 5
    },
    sector: {
      'Technology': 25,
      'Financials': 20,
      'Healthcare': 15,
      'Consumer Goods': 15,
      'Energy': 10,
      'Other': 15
    },
    strategy: {
      'Growth Strategy': 40,
      'Value Strategy': 30,
      'Momentum Strategy': 20,
      'Defensive Strategy': 10
    }
  };

  useEffect(() => {
    const handleAggregationUpdate = (aggregation: PortfolioAggregation) => {
      setPortfolioData(aggregation);
      calculateDrift(aggregation);
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
  }, [selectedType, threshold]);

  const calculateDrift = (aggregation: PortfolioAggregation) => {
    const targets = targetAllocations[selectedType] || {};
    const currentAllocations = getCurrentAllocations(aggregation);
    
    const drifts: AllocationDrift[] = Object.entries(targets).map(([category, target]) => {
      const current = currentAllocations[category] || 0;
      const driftAmount = current - target;
      const driftPercentage = target > 0 ? (driftAmount / target) * 100 : 0;
      
      const riskLevel = Math.abs(driftPercentage) > 50 ? 'critical' :
                       Math.abs(driftPercentage) > 25 ? 'high' :
                       Math.abs(driftPercentage) > threshold ? 'medium' : 'low';
      
      const rebalanceNeeded = Math.abs(driftAmount) > threshold;
      
      const recommendation = getRecommendation(driftAmount, driftPercentage, category);

      return {
        category,
        target_allocation: target,
        current_allocation: current,
        drift_amount: driftAmount,
        drift_percentage: driftPercentage,
        risk_level: riskLevel,
        rebalance_needed: rebalanceNeeded,
        recommendation
      };
    });

    setDriftData(drifts.sort((a, b) => Math.abs(b.drift_amount) - Math.abs(a.drift_amount)));
  };

  const getCurrentAllocations = (aggregation: PortfolioAggregation): Record<string, number> => {
    // Mock current allocations based on type
    switch (selectedType) {
      case 'asset_class':
        return {
          'Stocks': 65 + Math.random() * 10,
          'Bonds': 20 + Math.random() * 10,
          'Cash': 8 + Math.random() * 5,
          'Other': 7 + Math.random() * 3
        };
      case 'sector':
        return {
          'Technology': 30 + Math.random() * 10,
          'Financials': 18 + Math.random() * 8,
          'Healthcare': 12 + Math.random() * 6,
          'Consumer Goods': 16 + Math.random() * 8,
          'Energy': 8 + Math.random() * 4,
          'Other': 16 + Math.random() * 8
        };
      case 'strategy':
        return aggregation.strategies.reduce((acc, strategy) => {
          acc[strategy.strategy_name] = strategy.weight;
          return acc;
        }, {} as Record<string, number>);
      default:
        return {};
    }
  };

  const getRecommendation = (driftAmount: number, driftPercentage: number, category: string): string => {
    if (Math.abs(driftAmount) < 2) return 'No action needed';
    
    if (driftAmount > 0) {
      if (Math.abs(driftPercentage) > 25) {
        return `Significantly overweight - sell ${Math.abs(driftAmount).toFixed(1)}% immediately`;
      }
      return `Overweight - consider reducing by ${Math.abs(driftAmount).toFixed(1)}%`;
    } else {
      if (Math.abs(driftPercentage) > 25) {
        return `Significantly underweight - buy ${Math.abs(driftAmount).toFixed(1)}% immediately`;
      }
      return `Underweight - consider increasing by ${Math.abs(driftAmount).toFixed(1)}%`;
    }
  };

  const summaryStats = useMemo(() => {
    if (driftData.length === 0) return null;

    const totalDrift = driftData.reduce((sum, item) => sum + Math.abs(item.drift_amount), 0);
    const rebalanceNeeded = driftData.filter(item => item.rebalance_needed).length;
    const highRiskItems = driftData.filter(item => item.risk_level === 'high' || item.risk_level === 'critical').length;
    const avgDrift = totalDrift / driftData.length;

    return {
      totalDrift,
      rebalanceNeeded,
      highRiskItems,
      avgDrift,
      totalCategories: driftData.length
    };
  }, [driftData]);

  const columns: ColumnType<AllocationDrift>[] = [
    {
      title: 'Category',
      dataIndex: 'category',
      key: 'category',
      width: 120,
      render: (category: string) => <strong>{category}</strong>,
    },
    {
      title: 'Target vs Current',
      key: 'allocation_comparison',
      width: 200,
      render: (_, record: AllocationDrift) => (
        <div>
          <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 4 }}>
            <span>Target: {record.target_allocation.toFixed(1)}%</span>
            <span>Current: {record.current_allocation.toFixed(1)}%</span>
          </div>
          <Progress
            percent={record.current_allocation}
            showInfo={false}
            strokeColor={record.drift_amount > 0 ? '#ff4d4f' : record.drift_amount < 0 ? '#1890ff' : '#52c41a'}
            trailColor="#f0f0f0"
            size="small"
          />
        </div>
      ),
    },
    {
      title: 'Drift',
      key: 'drift',
      width: 100,
      render: (_, record: AllocationDrift) => (
        <div style={{ textAlign: 'center' }}>
          <div style={{ 
            color: record.drift_amount > 0 ? '#ff4d4f' : record.drift_amount < 0 ? '#1890ff' : '#52c41a',
            fontWeight: 'bold'
          }}>
            {record.drift_amount > 0 ? '+' : ''}{record.drift_amount.toFixed(1)}%
          </div>
          <div style={{ fontSize: '11px', color: '#666' }}>
            ({record.drift_percentage > 0 ? '+' : ''}{record.drift_percentage.toFixed(0)}%)
          </div>
        </div>
      ),
      sorter: (a, b) => Math.abs(b.drift_amount) - Math.abs(a.drift_amount),
    },
    {
      title: 'Risk Level',
      dataIndex: 'risk_level',
      key: 'risk_level',
      width: 100,
      render: (level: string) => {
        const colors = {
          low: 'green',
          medium: 'orange', 
          high: 'red',
          critical: 'red'
        };
        return <Tag color={colors[level as keyof typeof colors]}>{level.toUpperCase()}</Tag>;
      },
    },
    {
      title: 'Action',
      key: 'action',
      width: 100,
      render: (_, record: AllocationDrift) => (
        <div style={{ textAlign: 'center' }}>
          {record.rebalance_needed ? (
            <Tag color="warning" icon={<SyncOutlined />}>
              Rebalance
            </Tag>
          ) : (
            <Tag color="success">
              On Track
            </Tag>
          )}
        </div>
      ),
    },
    {
      title: 'Recommendation',
      dataIndex: 'recommendation',
      key: 'recommendation',
      render: (rec: string) => <span style={{ fontSize: '12px' }}>{rec}</span>,
    },
  ];

  if (loading) {
    return (
      <Card title="Allocation Drift Monitor" loading={true}>
        <div style={{ height: 300 }}>Loading drift analysis...</div>
      </Card>
    );
  }

  return (
    <Card
      title="Allocation Drift Monitor"
      extra={
        <div style={{ display: 'flex', gap: 8 }}>
          <Select value={selectedType} onChange={setSelectedType} style={{ width: 120 }}>
            <Option value="asset_class">Asset Class</Option>
            <Option value="sector">Sector</Option>
            <Option value="strategy">Strategy</Option>
          </Select>
          <Select value={threshold} onChange={setThreshold} style={{ width: 100 }}>
            <Option value={2}>2% Threshold</Option>
            <Option value={5}>5% Threshold</Option>
            <Option value={10}>10% Threshold</Option>
          </Select>
        </div>
      }
    >
      {/* Alert for high drift */}
      {summaryStats && summaryStats.highRiskItems > 0 && (
        <Alert
          message={`High Drift Detected in ${summaryStats.highRiskItems} Categories`}
          description="Immediate rebalancing recommended to maintain target allocation."
          type="warning"
          showIcon
          icon={<WarningOutlined />}
          style={{ marginBottom: 16 }}
        />
      )}

      {/* Summary Statistics */}
      {summaryStats && (
        <Row gutter={16} style={{ marginBottom: 16 }}>
          <Col span={6}>
            <Statistic
              title="Total Drift"
              value={summaryStats.totalDrift}
              precision={1}
              suffix="%"
              valueStyle={{ color: summaryStats.totalDrift > 20 ? '#ff4d4f' : '#1890ff' }}
            />
          </Col>
          <Col span={6}>
            <Statistic
              title="Rebalance Needed"
              value={summaryStats.rebalanceNeeded}
              suffix={`/ ${summaryStats.totalCategories}`}
              valueStyle={{ color: summaryStats.rebalanceNeeded > 0 ? '#faad14' : '#52c41a' }}
            />
          </Col>
          <Col span={6}>
            <Statistic
              title="High Risk Items"
              value={summaryStats.highRiskItems}
              valueStyle={{ color: summaryStats.highRiskItems > 0 ? '#ff4d4f' : '#52c41a' }}
            />
          </Col>
          <Col span={6}>
            <Statistic
              title="Average Drift"
              value={summaryStats.avgDrift}
              precision={1}
              suffix="%"
              valueStyle={{ color: '#722ed1' }}
            />
          </Col>
        </Row>
      )}

      {/* Drift Table */}
      <Table
        columns={columns}
        dataSource={driftData}
        rowKey="category"
        pagination={false}
        size="small"
        bordered
        rowClassName={(record) => {
          if (record.risk_level === 'critical') return 'critical-drift';
          if (record.risk_level === 'high') return 'high-drift';
          if (record.rebalance_needed) return 'rebalance-needed';
          return '';
        }}
      />

      {/* Rebalancing Actions */}
      {summaryStats && summaryStats.rebalanceNeeded > 0 && (
        <Card size="small" title="Rebalancing Actions" style={{ marginTop: 16 }} type="inner">
          <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap' }}>
            <Button type="primary" icon={<SyncOutlined />}>
              Auto Rebalance All
            </Button>
            <Button icon={<RiseOutlined />}>
              Rebalance Overweight Only
            </Button>
            <Button icon={<FallOutlined />}>
              Rebalance Underweight Only
            </Button>
          </div>
        </Card>
      )}

      <style jsx>{`
        .critical-drift {
          background-color: rgba(255, 77, 79, 0.1);
          border-left: 3px solid #ff4d4f;
        }
        .high-drift {
          background-color: rgba(255, 122, 69, 0.05);
        }
        .rebalance-needed {
          background-color: rgba(250, 173, 20, 0.05);
        }
      `}</style>
    </Card>
  );
};

export default AllocationDriftMonitor;