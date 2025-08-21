/**
 * Strategy Correlation Matrix Component with heatmap visualization
 */

import React, { useState, useEffect, useMemo } from 'react';
import { Card, Select, Switch, Row, Col, Statistic, Alert, Tooltip, Space, Button } from 'antd';
import { InfoCircleOutlined, ZoomInOutlined, ZoomOutOutlined } from '@ant-design/icons';
import { 
  portfolioAggregationService, 
  PortfolioAggregation, 
  StrategyPnL 
} from '../../services/portfolioAggregationService';

const { Option } = Select;

interface CorrelationData {
  strategy1: string;
  strategy2: string;
  correlation: number;
  pValue: number;
  significance: 'high' | 'medium' | 'low' | 'none';
}

interface StrategyCorrelationMatrixProps {
  timeframe?: '1M' | '3M' | '6M' | '1Y';
  showPValues?: boolean;
  minCorrelation?: number;
}

const StrategyCorrelationMatrix: React.FC<StrategyCorrelationMatrixProps> = ({
  timeframe = '3M',
  showPValues = false,
  minCorrelation = 0.1
}) => {
  const [portfolioData, setPortfolioData] = useState<PortfolioAggregation | null>(null);
  const [correlationMatrix, setCorrelationMatrix] = useState<number[][]>([]);
  const [correlationData, setCorrelationData] = useState<CorrelationData[]>([]);
  const [selectedTimeframe, setSelectedTimeframe] = useState(timeframe);
  const [showSignificanceOnly, setShowSignificanceOnly] = useState(false);
  const [cellSize, setCellSize] = useState(60);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const handleAggregationUpdate = (aggregation: PortfolioAggregation) => {
      setPortfolioData(aggregation);
      calculateCorrelationMatrix(aggregation);
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
  }, [selectedTimeframe]);

  // Calculate correlation matrix and statistics
  const calculateCorrelationMatrix = (aggregation: PortfolioAggregation) => {
    const strategies = aggregation.strategies;
    const strategyCount = strategies.length;
    
    if (strategyCount < 2) {
      setCorrelationMatrix([]);
      setCorrelationData([]);
      return;
    }

    // Generate mock historical returns for each strategy
    const strategyReturns = strategies.map(strategy => 
      generateMockReturns(strategy, selectedTimeframe)
    );

    // Calculate correlation matrix
    const matrix: number[][] = [];
    const correlationPairs: CorrelationData[] = [];

    for (let i = 0; i < strategyCount; i++) {
      matrix[i] = [];
      for (let j = 0; j < strategyCount; j++) {
        if (i === j) {
          matrix[i][j] = 1.0; // Perfect correlation with self
        } else {
          const correlation = calculateCorrelation(strategyReturns[i], strategyReturns[j]);
          matrix[i][j] = correlation;
          
          // Store correlation data for analysis (avoid duplicates)
          if (i < j) {
            const pValue = calculatePValue(correlation, strategyReturns[i].length);
            const significance = getSignificanceLevel(pValue);
            
            correlationPairs.push({
              strategy1: strategies[i].strategy_name,
              strategy2: strategies[j].strategy_name,
              correlation,
              pValue,
              significance
            });
          }
        }
      }
    }

    setCorrelationMatrix(matrix);
    setCorrelationData(correlationPairs);
  };

  // Generate mock returns for strategy
  const generateMockReturns = (strategy: StrategyPnL, timeframe: string): number[] => {
    const periods = getTimeframePeriods(timeframe);
    const returns: number[] = [];
    
    // Base parameters on strategy performance
    const avgReturn = strategy.total_pnl > 0 ? 0.002 : -0.001; // Daily avg return
    const volatility = Math.max(0.01, Math.abs(strategy.total_pnl) / 100000 * 0.02); // Volatility based on P&L
    
    for (let i = 0; i < periods; i++) {
      // Generate return with some autocorrelation and random component
      const randomComponent = (Math.random() - 0.5) * volatility * 2;
      const trendComponent = avgReturn + (Math.random() - 0.5) * avgReturn * 0.5;
      returns.push(trendComponent + randomComponent);
    }
    
    return returns;
  };

  // Calculate Pearson correlation coefficient
  const calculateCorrelation = (x: number[], y: number[]): number => {
    if (x.length !== y.length || x.length < 2) return 0;
    
    const n = x.length;
    const sumX = x.reduce((a, b) => a + b, 0);
    const sumY = y.reduce((a, b) => a + b, 0);
    const sumXY = x.reduce((sum, xi, i) => sum + xi * y[i], 0);
    const sumX2 = x.reduce((sum, xi) => sum + xi * xi, 0);
    const sumY2 = y.reduce((sum, yi) => sum + yi * yi, 0);
    
    const numerator = n * sumXY - sumX * sumY;
    const denominator = Math.sqrt((n * sumX2 - sumX * sumX) * (n * sumY2 - sumY * sumY));
    
    return denominator === 0 ? 0 : numerator / denominator;
  };

  // Calculate p-value for correlation significance
  const calculatePValue = (correlation: number, sampleSize: number): number => {
    if (sampleSize < 3) return 1;
    
    const t = correlation * Math.sqrt((sampleSize - 2) / (1 - correlation * correlation));
    // Simplified p-value calculation (actual would use t-distribution)
    return Math.max(0.001, Math.min(0.999, Math.abs(t) > 2 ? 0.05 : 0.2));
  };

  // Get significance level
  const getSignificanceLevel = (pValue: number): 'high' | 'medium' | 'low' | 'none' => {
    if (pValue < 0.01) return 'high';
    if (pValue < 0.05) return 'medium';
    if (pValue < 0.1) return 'low';
    return 'none';
  };

  const getTimeframePeriods = (timeframe: string): number => {
    switch (timeframe) {
      case '1M': return 30;
      case '3M': return 90;
      case '6M': return 180;
      case '1Y': return 365;
      default: return 90;
    }
  };

  // Get color for correlation value
  const getCorrelationColor = (correlation: number): string => {
    const absCorr = Math.abs(correlation);
    
    if (correlation > 0) {
      // Positive correlation - shades of red
      if (absCorr > 0.8) return '#8b0000'; // Dark red
      if (absCorr > 0.6) return '#dc143c'; // Crimson
      if (absCorr > 0.4) return '#ff6b6b'; // Light red
      if (absCorr > 0.2) return '#ffa8a8'; // Very light red
      return '#ffe0e0'; // Pale red
    } else {
      // Negative correlation - shades of blue
      if (absCorr > 0.8) return '#000080'; // Dark blue
      if (absCorr > 0.6) return '#0066cc'; // Blue
      if (absCorr > 0.4) return '#4da6ff'; // Light blue
      if (absCorr > 0.2) return '#b3d9ff'; // Very light blue
      return '#e6f3ff'; // Pale blue
    }
  };

  // Get text color for correlation cell
  const getTextColor = (correlation: number): string => {
    return Math.abs(correlation) > 0.5 ? '#ffffff' : '#000000';
  };

  // Calculate summary statistics
  const summaryStats = useMemo(() => {
    if (correlationData.length === 0) return null;
    
    const correlations = correlationData.map(d => d.correlation);
    const avgCorrelation = correlations.reduce((sum, corr) => sum + corr, 0) / correlations.length;
    const maxCorrelation = Math.max(...correlations);
    const minCorrelation = Math.min(...correlations);
    
    const highCorrelations = correlationData.filter(d => Math.abs(d.correlation) > 0.7).length;
    const significantCorrelations = correlationData.filter(d => d.significance !== 'none').length;
    
    return {
      avgCorrelation,
      maxCorrelation,
      minCorrelation,
      highCorrelations,
      significantCorrelations,
      totalPairs: correlationData.length
    };
  }, [correlationData]);

  const renderCorrelationMatrix = () => {
    if (!portfolioData || correlationMatrix.length === 0) {
      return (
        <Alert
          message="Insufficient Data"
          description="Need at least 2 strategies to calculate correlations."
          type="info"
          showIcon
        />
      );
    }

    const strategies = portfolioData.strategies;
    const filteredData = showSignificanceOnly 
      ? correlationData.filter(d => d.significance !== 'none')
      : correlationData;

    return (
      <div style={{ overflowX: 'auto', overflowY: 'auto', maxHeight: '600px' }}>
        <div style={{ 
          display: 'inline-block',
          minWidth: (strategies.length + 1) * cellSize,
          minHeight: (strategies.length + 1) * cellSize
        }}>
          {/* Header row */}
          <div style={{ display: 'flex' }}>
            <div style={{ width: cellSize, height: cellSize }}></div>
            {strategies.map((strategy, index) => (
              <div
                key={index}
                style={{
                  width: cellSize,
                  height: cellSize,
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  backgroundColor: '#f5f5f5',
                  border: '1px solid #d9d9d9',
                  fontSize: '10px',
                  fontWeight: 'bold',
                  textAlign: 'center',
                  padding: '2px',
                  writingMode: 'vertical-lr',
                  textOrientation: 'mixed'
                }}
              >
                {strategy.strategy_name}
              </div>
            ))}
          </div>
          
          {/* Matrix rows */}
          {strategies.map((strategy, i) => (
            <div key={i} style={{ display: 'flex' }}>
              {/* Row header */}
              <div
                style={{
                  width: cellSize,
                  height: cellSize,
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  backgroundColor: '#f5f5f5',
                  border: '1px solid #d9d9d9',
                  fontSize: '10px',
                  fontWeight: 'bold',
                  textAlign: 'center',
                  padding: '2px'
                }}
              >
                {strategy.strategy_name}
              </div>
              
              {/* Correlation cells */}
              {strategies.map((_, j) => {
                const correlation = correlationMatrix[i][j];
                const backgroundColor = getCorrelationColor(correlation);
                const textColor = getTextColor(correlation);
                
                return (
                  <Tooltip
                    key={j}
                    title={
                      <div>
                        <div>{strategies[i].strategy_name} vs {strategies[j].strategy_name}</div>
                        <div>Correlation: {correlation.toFixed(3)}</div>
                        {i !== j && showPValues && (
                          <div>P-value: {correlationData.find(d => 
                            (d.strategy1 === strategies[i].strategy_name && d.strategy2 === strategies[j].strategy_name) ||
                            (d.strategy1 === strategies[j].strategy_name && d.strategy2 === strategies[i].strategy_name)
                          )?.pValue.toFixed(3) || 'N/A'}</div>
                        )}
                      </div>
                    }
                  >
                    <div
                      style={{
                        width: cellSize,
                        height: cellSize,
                        display: 'flex',
                        alignItems: 'center',
                        justifyContent: 'center',
                        backgroundColor,
                        color: textColor,
                        border: '1px solid #d9d9d9',
                        fontSize: '11px',
                        fontWeight: 'bold',
                        cursor: 'pointer'
                      }}
                    >
                      {correlation.toFixed(2)}
                    </div>
                  </Tooltip>
                );
              })}
            </div>
          ))}
        </div>
      </div>
    );
  };

  const renderCorrelationLegend = () => (
    <div style={{ marginTop: 16 }}>
      <div style={{ marginBottom: 8, fontWeight: 'bold' }}>Correlation Scale:</div>
      <div style={{ display: 'flex', alignItems: 'center', gap: '8px', flexWrap: 'wrap' }}>
        <div style={{ display: 'flex', alignItems: 'center' }}>
          <div style={{ 
            width: 20, 
            height: 20, 
            backgroundColor: '#000080',
            marginRight: 4,
            border: '1px solid #ccc'
          }}></div>
          <span style={{ fontSize: '12px' }}>Strong Negative (-0.8 to -1.0)</span>
        </div>
        <div style={{ display: 'flex', alignItems: 'center' }}>
          <div style={{ 
            width: 20, 
            height: 20, 
            backgroundColor: '#4da6ff',
            marginRight: 4,
            border: '1px solid #ccc'
          }}></div>
          <span style={{ fontSize: '12px' }}>Moderate Negative (-0.4 to -0.8)</span>
        </div>
        <div style={{ display: 'flex', alignItems: 'center' }}>
          <div style={{ 
            width: 20, 
            height: 20, 
            backgroundColor: '#f0f0f0',
            marginRight: 4,
            border: '1px solid #ccc'
          }}></div>
          <span style={{ fontSize: '12px' }}>Weak (-0.4 to 0.4)</span>
        </div>
        <div style={{ display: 'flex', alignItems: 'center' }}>
          <div style={{ 
            width: 20, 
            height: 20, 
            backgroundColor: '#ff6b6b',
            marginRight: 4,
            border: '1px solid #ccc'
          }}></div>
          <span style={{ fontSize: '12px' }}>Moderate Positive (0.4 to 0.8)</span>
        </div>
        <div style={{ display: 'flex', alignItems: 'center' }}>
          <div style={{ 
            width: 20, 
            height: 20, 
            backgroundColor: '#8b0000',
            marginRight: 4,
            border: '1px solid #ccc'
          }}></div>
          <span style={{ fontSize: '12px' }}>Strong Positive (0.8 to 1.0)</span>
        </div>
      </div>
    </div>
  );

  if (loading) {
    return (
      <Card title="Strategy Correlation Matrix" loading={true}>
        <div style={{ height: 400 }}>Loading correlation data...</div>
      </Card>
    );
  }

  return (
    <Card
      title="Strategy Correlation Matrix"
      extra={
        <Space>
          <Select 
            value={selectedTimeframe} 
            onChange={setSelectedTimeframe}
            style={{ width: 80 }}
          >
            <Option value="1M">1M</Option>
            <Option value="3M">3M</Option>
            <Option value="6M">6M</Option>
            <Option value="1Y">1Y</Option>
          </Select>
          
          <Switch
            checked={showSignificanceOnly}
            onChange={setShowSignificanceOnly}
            checkedChildren="Significant"
            unCheckedChildren="All"
          />
          
          <Switch
            checked={showPValues}
            onChange={setShowPValues}
            checkedChildren="P-Values"
            unCheckedChildren="Correlations"
          />
          
          <Button 
            icon={<ZoomOutOutlined />} 
            size="small"
            onClick={() => setCellSize(Math.max(40, cellSize - 10))}
          />
          <Button 
            icon={<ZoomInOutlined />} 
            size="small"
            onClick={() => setCellSize(Math.min(100, cellSize + 10))}
          />
        </Space>
      }
    >
      {/* Summary Statistics */}
      {summaryStats && (
        <Row gutter={16} style={{ marginBottom: 16 }}>
          <Col span={6}>
            <Statistic
              title="Average Correlation"
              value={summaryStats.avgCorrelation}
              precision={3}
              valueStyle={{ 
                color: Math.abs(summaryStats.avgCorrelation) < 0.3 ? '#52c41a' : '#faad14'
              }}
            />
          </Col>
          <Col span={6}>
            <Statistic
              title="Highest Correlation"
              value={summaryStats.maxCorrelation}
              precision={3}
              valueStyle={{ 
                color: summaryStats.maxCorrelation > 0.8 ? '#ff4d4f' : '#faad14'
              }}
            />
          </Col>
          <Col span={6}>
            <Statistic
              title="High Correlations"
              value={summaryStats.highCorrelations}
              suffix={`/ ${summaryStats.totalPairs}`}
              valueStyle={{ 
                color: summaryStats.highCorrelations > summaryStats.totalPairs * 0.3 ? '#ff4d4f' : '#52c41a'
              }}
            />
          </Col>
          <Col span={6}>
            <Statistic
              title="Significant Pairs"
              value={summaryStats.significantCorrelations}
              suffix={`/ ${summaryStats.totalPairs}`}
              valueStyle={{ color: '#1890ff' }}
            />
          </Col>
        </Row>
      )}

      {/* Correlation Matrix */}
      {renderCorrelationMatrix()}
      
      {/* Legend */}
      {renderCorrelationLegend()}
      
      {/* Interpretation Note */}
      <Alert
        style={{ marginTop: 16 }}
        message="Interpretation"
        description={
          <div>
            <p><strong>Diversification Benefits:</strong> Lower correlations (closer to 0) indicate better diversification.</p>
            <p><strong>Risk Concentration:</strong> High positive correlations (&gt;0.7) suggest strategies may move together during market stress.</p>
            <p><strong>Hedging Opportunities:</strong> Negative correlations indicate potential hedging relationships between strategies.</p>
          </div>
        }
        type="info"
        showIcon
        icon={<InfoCircleOutlined />}
      />
    </Card>
  );
};

export default StrategyCorrelationMatrix;