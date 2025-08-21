/**
 * Diversification Analysis Component for sector/geographic breakdown
 */

import React, { useState, useEffect, useMemo } from 'react';
import { Card, Row, Col, Progress, Table, Select, Alert, Statistic, Tag, Space } from 'antd';
import { ColumnType } from 'antd/es/table';
import { GlobalOutlined, PieChartOutlined, WarningOutlined } from '@ant-design/icons';
import { portfolioAggregationService, PortfolioAggregation } from '../../services/portfolioAggregationService';
import { Position } from '../../types/position';

const { Option } = Select;

interface DiversificationMetric {
  category: string;
  allocation: number;
  concentration_risk: number;
  diversification_score: number;
  positions_count: number;
  categories: string[];
}

interface ConcentrationRisk {
  level: 'low' | 'medium' | 'high' | 'critical';
  score: number;
  description: string;
  recommendations: string[];
}

interface DiversificationAnalysisProps {
  analysisType?: 'sector' | 'geography' | 'currency' | 'size';
}

const DiversificationAnalysis: React.FC<DiversificationAnalysisProps> = ({
  analysisType = 'sector'
}) => {
  const [portfolioData, setPortfolioData] = useState<PortfolioAggregation | null>(null);
  const [allPositions, setAllPositions] = useState<Position[]>([]);
  const [selectedAnalysis, setSelectedAnalysis] = useState(analysisType);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const handleAggregationUpdate = (aggregation: PortfolioAggregation) => {
      setPortfolioData(aggregation);
      const positions = portfolioAggregationService.getAllPositions();
      setAllPositions(positions);
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
  }, []);

  // Sector/Industry mappings
  const sectorMappings: Record<string, string> = {
    'AAPL': 'Technology', 'MSFT': 'Technology', 'GOOGL': 'Technology', 'AMZN': 'Technology',
    'JPM': 'Financials', 'BAC': 'Financials', 'WFC': 'Financials',
    'JNJ': 'Healthcare', 'PFE': 'Healthcare', 'UNH': 'Healthcare',
    'KO': 'Consumer Goods', 'PG': 'Consumer Goods', 'WMT': 'Consumer Goods',
    'XOM': 'Energy', 'CVX': 'Energy'
  };

  const geographyMappings: Record<string, string> = {
    'NASDAQ': 'North America', 'NYSE': 'North America',
    'LSE': 'Europe', 'EUREX': 'Europe',
    'TSE': 'Asia Pacific', 'HKEX': 'Asia Pacific'
  };

  const sizeMappings: Record<string, string> = {
    'large': 'Large Cap (>$10B)',
    'mid': 'Mid Cap ($2B-$10B)', 
    'small': 'Small Cap (<$2B)'
  };

  // Calculate diversification metrics
  const diversificationMetrics = useMemo(() => {
    if (!allPositions.length) return [];

    const totalValue = allPositions.reduce((sum, pos) => 
      sum + Math.abs(pos.currentPrice * pos.quantity), 0);

    const groupings: Record<string, {value: number, positions: Position[]}> = {};

    allPositions.forEach(position => {
      const value = Math.abs(position.currentPrice * position.quantity);
      let category = 'Other';

      switch (selectedAnalysis) {
        case 'sector':
          category = sectorMappings[position.symbol] || 'Other';
          break;
        case 'geography':
          category = geographyMappings[position.venue] || 'Other';
          break;
        case 'currency':
          category = position.currency;
          break;
        case 'size':
          const marketCap = value * Math.random() * 100; // Mock market cap calculation
          category = marketCap > 10000000000 ? 'Large Cap (>$10B)' :
                    marketCap > 2000000000 ? 'Mid Cap ($2B-$10B)' : 'Small Cap (<$2B)';
          break;
      }

      if (!groupings[category]) {
        groupings[category] = { value: 0, positions: [] };
      }
      groupings[category].value += value;
      groupings[category].positions.push(position);
    });

    const metrics: DiversificationMetric[] = Object.entries(groupings).map(([category, data]) => {
      const allocation = (data.value / totalValue) * 100;
      const concentrationRisk = allocation > 50 ? 100 : allocation > 30 ? 75 : allocation > 20 ? 50 : 25;
      const diversificationScore = Math.max(0, 100 - concentrationRisk);
      
      return {
        category,
        allocation,
        concentration_risk: concentrationRisk,
        diversification_score: diversificationScore,
        positions_count: data.positions.length,
        categories: [category]
      };
    });

    return metrics.sort((a, b) => b.allocation - a.allocation);
  }, [allPositions, selectedAnalysis]);

  // Calculate overall concentration risk
  const concentrationRisk = useMemo((): ConcentrationRisk => {
    if (diversificationMetrics.length === 0) {
      return {
        level: 'low',
        score: 0,
        description: 'No positions to analyze',
        recommendations: []
      };
    }

    const topAllocation = diversificationMetrics[0]?.allocation || 0;
    const top3Allocation = diversificationMetrics.slice(0, 3).reduce((sum, m) => sum + m.allocation, 0);
    const herfindahlIndex = diversificationMetrics.reduce((sum, m) => sum + Math.pow(m.allocation, 2), 0);

    let level: 'low' | 'medium' | 'high' | 'critical';
    let score: number;
    let description: string;
    let recommendations: string[] = [];

    if (herfindahlIndex > 5000 || topAllocation > 70) {
      level = 'critical';
      score = 90;
      description = 'Very high concentration risk detected';
      recommendations = [
        'Immediately diversify top holdings',
        'Consider sector/geographic rebalancing',
        'Reduce position sizes to under 20% each'
      ];
    } else if (herfindahlIndex > 2500 || topAllocation > 50) {
      level = 'high';
      score = 70;
      description = 'High concentration risk';
      recommendations = [
        'Reduce largest positions',
        'Add exposure to underrepresented sectors',
        'Consider geographic diversification'
      ];
    } else if (herfindahlIndex > 1000 || topAllocation > 30) {
      level = 'medium';
      score = 40;
      description = 'Moderate concentration risk';
      recommendations = [
        'Monitor top holdings',
        'Consider adding more categories',
        'Maintain balanced allocation'
      ];
    } else {
      level = 'low';
      score = 20;
      description = 'Well diversified portfolio';
      recommendations = [
        'Maintain current diversification',
        'Regular rebalancing',
        'Monitor for concentration drift'
      ];
    }

    return { level, score, description, recommendations };
  }, [diversificationMetrics]);

  // Table columns
  const columns: ColumnType<DiversificationMetric>[] = [
    {
      title: 'Category',
      dataIndex: 'category',
      key: 'category',
      width: 150,
      render: (category: string) => (
        <div style={{ fontWeight: 'bold' }}>{category}</div>
      ),
    },
    {
      title: 'Allocation',
      dataIndex: 'allocation',
      key: 'allocation',
      width: 200,
      render: (allocation: number) => (
        <div>
          <Progress
            percent={allocation}
            showInfo={false}
            strokeColor={
              allocation > 50 ? '#ff4d4f' :
              allocation > 30 ? '#faad14' :
              allocation > 20 ? '#1890ff' : '#52c41a'
            }
            size="small"
          />
          <div style={{ textAlign: 'center', fontSize: '12px', marginTop: '4px' }}>
            {allocation.toFixed(1)}%
          </div>
        </div>
      ),
      sorter: (a, b) => a.allocation - b.allocation,
    },
    {
      title: 'Risk Level',
      dataIndex: 'concentration_risk',
      key: 'concentration_risk',
      width: 120,
      render: (risk: number) => {
        let color = '#52c41a';
        let text = 'Low';
        
        if (risk > 75) {
          color = '#ff4d4f';
          text = 'Critical';
        } else if (risk > 50) {
          color = '#ff7a45';
          text = 'High';
        } else if (risk > 25) {
          color = '#faad14';
          text = 'Medium';
        }
        
        return <Tag color={color}>{text}</Tag>;
      },
    },
    {
      title: 'Diversification Score',
      dataIndex: 'diversification_score',
      key: 'diversification_score',
      width: 100,
      render: (score: number) => (
        <span style={{ 
          color: score > 75 ? '#52c41a' : score > 50 ? '#faad14' : '#ff4d4f',
          fontWeight: 'bold'
        }}>
          {score.toFixed(0)}/100
        </span>
      ),
    },
    {
      title: 'Positions',
      dataIndex: 'positions_count',
      key: 'positions_count',
      width: 80,
      render: (count: number) => count,
      sorter: (a, b) => a.positions_count - b.positions_count,
    },
  ];

  // Risk level colors
  const getRiskColor = (level: string) => {
    switch (level) {
      case 'critical': return '#ff4d4f';
      case 'high': return '#ff7a45';
      case 'medium': return '#faad14';
      case 'low': return '#52c41a';
      default: return '#d9d9d9';
    }
  };

  if (loading) {
    return (
      <Card title="Diversification Analysis" loading={true}>
        <div style={{ height: 400 }}>Loading diversification data...</div>
      </Card>
    );
  }

  if (!allPositions.length) {
    return (
      <Card title="Diversification Analysis">
        <Alert
          message="No Position Data"
          description="No positions found to analyze diversification."
          type="info"
          showIcon
        />
      </Card>
    );
  }

  return (
    <Card
      title="Diversification Analysis"
      extra={
        <Space>
          <Select 
            value={selectedAnalysis} 
            onChange={setSelectedAnalysis}
            style={{ width: 120 }}
          >
            <Option value="sector">
              <PieChartOutlined /> Sector
            </Option>
            <Option value="geography">
              <GlobalOutlined /> Geography
            </Option>
            <Option value="currency">Currency</Option>
            <Option value="size">Market Cap</Option>
          </Select>
        </Space>
      }
    >
      {/* Overall Risk Assessment */}
      <Alert
        message={`Concentration Risk: ${concentrationRisk.level.toUpperCase()}`}
        description={concentrationRisk.description}
        type={
          concentrationRisk.level === 'critical' ? 'error' :
          concentrationRisk.level === 'high' ? 'warning' :
          concentrationRisk.level === 'medium' ? 'info' : 'success'
        }
        showIcon
        icon={concentrationRisk.level === 'critical' || concentrationRisk.level === 'high' ? 
              <WarningOutlined /> : undefined}
        style={{ marginBottom: 16 }}
      />

      {/* Summary Statistics */}
      <Row gutter={16} style={{ marginBottom: 16 }}>
        <Col span={6}>
          <Statistic
            title="Categories"
            value={diversificationMetrics.length}
            valueStyle={{ color: '#1890ff' }}
            suffix="categories"
          />
        </Col>
        <Col span={6}>
          <Statistic
            title="Largest Allocation"
            value={diversificationMetrics[0]?.allocation || 0}
            precision={1}
            suffix="%"
            valueStyle={{ 
              color: getRiskColor(
                (diversificationMetrics[0]?.allocation || 0) > 50 ? 'critical' :
                (diversificationMetrics[0]?.allocation || 0) > 30 ? 'high' :
                (diversificationMetrics[0]?.allocation || 0) > 20 ? 'medium' : 'low'
              )
            }}
          />
        </Col>
        <Col span={6}>
          <Statistic
            title="Risk Score"
            value={concentrationRisk.score}
            suffix="/100"
            valueStyle={{ color: getRiskColor(concentrationRisk.level) }}
          />
        </Col>
        <Col span={6}>
          <Statistic
            title="Avg Diversification"
            value={diversificationMetrics.reduce((sum, m) => sum + m.diversification_score, 0) / diversificationMetrics.length}
            precision={0}
            suffix="/100"
            valueStyle={{ color: '#722ed1' }}
          />
        </Col>
      </Row>

      {/* Diversification Breakdown Table */}
      <Table
        columns={columns}
        dataSource={diversificationMetrics}
        rowKey="category"
        pagination={false}
        size="small"
        bordered
        title={() => `${selectedAnalysis.charAt(0).toUpperCase() + selectedAnalysis.slice(1)} Diversification`}
      />

      {/* Recommendations */}
      {concentrationRisk.recommendations.length > 0 && (
        <Card 
          size="small" 
          title="Recommendations" 
          style={{ marginTop: 16 }}
          type="inner"
        >
          <ul style={{ margin: 0, paddingLeft: 20 }}>
            {concentrationRisk.recommendations.map((rec, index) => (
              <li key={index} style={{ marginBottom: 4 }}>
                {rec}
              </li>
            ))}
          </ul>
        </Card>
      )}

      {/* Analysis Interpretation */}
      <Alert
        style={{ marginTop: 16 }}
        message="Analysis Interpretation"
        description={
          <div>
            <p><strong>Concentration Risk:</strong> Measures how much of the portfolio is concentrated in specific categories.</p>
            <p><strong>Diversification Score:</strong> Higher scores indicate better diversification within each category.</p>
            <p><strong>Best Practices:</strong> No single category should exceed 30% allocation for optimal risk management.</p>
          </div>
        }
        type="info"
        showIcon
      />
    </Card>
  );
};

export default DiversificationAnalysis;