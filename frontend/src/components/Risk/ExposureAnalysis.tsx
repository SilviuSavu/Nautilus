import React, { useState, useEffect } from 'react';
import { Card, Row, Col, Statistic, Progress, Table, Spin, Alert, Tag } from 'antd';
import { Pie, Column } from '@ant-design/plots';
import { ExposureAnalysis as ExposureData, InstrumentExposure, SectorExposure } from './types/riskTypes';
import { riskService } from './services/riskService';

interface ExposureAnalysisProps {
  portfolioId: string;
  refreshInterval?: number;
}

const ExposureAnalysis: React.FC<ExposureAnalysisProps> = ({ 
  portfolioId, 
  refreshInterval = 30000 
}) => {
  const [exposureData, setExposureData] = useState<ExposureData | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const fetchExposureData = async () => {
    try {
      setError(null);
      const data = await riskService.getExposureAnalysis(portfolioId);
      setExposureData(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to fetch exposure data');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchExposureData();
    
    if (refreshInterval > 0) {
      const interval = setInterval(fetchExposureData, refreshInterval);
      return () => clearInterval(interval);
    }
  }, [portfolioId, refreshInterval]);

  const getRiskColor = (percentage: number): string => {
    if (percentage >= 25) return '#ff4d4f';
    if (percentage >= 15) return '#faad14';
    if (percentage >= 10) return '#1890ff';
    return '#52c41a';
  };

  const getRiskLevel = (percentage: number): string => {
    if (percentage >= 25) return 'High';
    if (percentage >= 15) return 'Medium';
    if (percentage >= 10) return 'Low';
    return 'Minimal';
  };

  const instrumentColumns = [
    {
      title: 'Symbol',
      dataIndex: 'symbol',
      key: 'symbol',
      render: (symbol: string) => <strong>{symbol}</strong>
    },
    {
      title: 'Market Value',
      dataIndex: 'market_value',
      key: 'market_value',
      render: (value: string) => `$${parseFloat(value).toLocaleString()}`
    },
    {
      title: 'Portfolio %',
      dataIndex: 'percentage_of_portfolio',
      key: 'percentage',
      render: (percentage: number) => (
        <div>
          <Progress 
            percent={percentage} 
            size="small" 
            strokeColor={getRiskColor(percentage)}
            showInfo={false}
          />
          <span style={{ marginLeft: 8 }}>{percentage.toFixed(1)}%</span>
        </div>
      )
    },
    {
      title: 'Risk Level',
      dataIndex: 'percentage_of_portfolio',
      key: 'risk_level',
      render: (percentage: number) => (
        <Tag color={getRiskColor(percentage)}>
          {getRiskLevel(percentage)}
        </Tag>
      )
    },
    {
      title: 'Unrealized P&L',
      dataIndex: 'unrealized_pnl',
      key: 'pnl',
      render: (pnl: string) => {
        const value = parseFloat(pnl);
        return (
          <span style={{ color: value >= 0 ? '#52c41a' : '#ff4d4f' }}>
            ${value.toLocaleString()}
          </span>
        );
      }
    }
  ];

  const preparePieData = (data: InstrumentExposure[]) => {
    return data.map(item => ({
      type: item.symbol,
      value: item.percentage_of_portfolio,
      market_value: parseFloat(item.market_value)
    }));
  };

  const prepareSectorData = (data: SectorExposure[]) => {
    return data.map(sector => ({
      sector: sector.sector,
      percentage: sector.percentage_of_portfolio,
      exposure: parseFloat(sector.total_exposure)
    }));
  };

  if (loading) {
    return (
      <Card title="Portfolio Exposure Analysis">
        <div style={{ textAlign: 'center', padding: '50px 0' }}>
          <Spin size="large" />
          <p style={{ marginTop: 16 }}>Loading exposure data...</p>
        </div>
      </Card>
    );
  }

  if (error) {
    return (
      <Card title="Portfolio Exposure Analysis">
        <Alert
          message="Error Loading Data"
          description={error}
          type="error"
          showIcon
          action={
            <button onClick={fetchExposureData}>Retry</button>
          }
        />
      </Card>
    );
  }

  if (!exposureData) {
    return (
      <Card title="Portfolio Exposure Analysis">
        <Alert
          message="No Data Available"
          description="No exposure data found for this portfolio."
          type="info"
          showIcon
        />
      </Card>
    );
  }

  const pieConfig = {
    data: preparePieData(exposureData.by_instrument),
    angleField: 'value',
    colorField: 'type',
    radius: 0.75,
    label: {
      type: 'spider',
      labelHeight: 28,
      content: '{name}\n{percentage}',
      style: {
        fontSize: 12,
        textAlign: 'center'
      }
    },
    interactions: [{ type: 'element-selected' }, { type: 'element-active' }],
    tooltip: {
      formatter: (datum: any) => ({
        name: datum.type,
        value: `$${datum.market_value.toLocaleString()} (${datum.value.toFixed(1)}%)`
      })
    }
  };

  const columnConfig = {
    data: prepareSectorData(exposureData.by_sector),
    xField: 'sector',
    yField: 'percentage',
    seriesField: 'sector',
    color: ({ percentage }: any) => getRiskColor(percentage),
    meta: {
      percentage: {
        alias: 'Portfolio %'
      }
    },
    tooltip: {
      formatter: (datum: any) => ({
        name: datum.sector,
        value: `${datum.percentage.toFixed(1)}% ($${datum.exposure.toLocaleString()})`
      })
    }
  };

  return (
    <div>
      <Row gutter={[16, 16]}>
        {/* Summary Statistics */}
        <Col span={24}>
          <Card title="Exposure Summary">
            <Row gutter={16}>
              <Col span={6}>
                <Statistic
                  title="Total Exposure"
                  value={parseFloat(exposureData.total_exposure)}
                  precision={0}
                  prefix="$"
                />
              </Col>
              <Col span={6}>
                <Statistic
                  title="Long Exposure"
                  value={parseFloat(exposureData.long_exposure)}
                  precision={0}
                  prefix="$"
                  valueStyle={{ color: '#52c41a' }}
                />
              </Col>
              <Col span={6}>
                <Statistic
                  title="Short Exposure"
                  value={parseFloat(exposureData.short_exposure)}
                  precision={0}
                  prefix="$"
                  valueStyle={{ color: '#ff4d4f' }}
                />
              </Col>
              <Col span={6}>
                <Statistic
                  title="Net Exposure"
                  value={parseFloat(exposureData.net_exposure)}
                  precision={0}
                  prefix="$"
                  valueStyle={{ 
                    color: parseFloat(exposureData.net_exposure) >= 0 ? '#52c41a' : '#ff4d4f' 
                  }}
                />
              </Col>
            </Row>
          </Card>
        </Col>

        {/* Instrument Concentration */}
        <Col span={12}>
          <Card title="Position Concentration" size="small">
            <Pie {...pieConfig} height={300} />
          </Card>
        </Col>

        {/* Sector Allocation */}
        <Col span={12}>
          <Card title="Sector Allocation" size="small">
            <Column {...columnConfig} height={300} />
          </Card>
        </Col>

        {/* Detailed Holdings */}
        <Col span={24}>
          <Card title="Detailed Holdings" size="small">
            <Table
              columns={instrumentColumns}
              dataSource={exposureData.by_instrument.map((item, index) => ({
                ...item,
                key: index
              }))}
              pagination={{ 
                pageSize: 10,
                showSizeChanger: true,
                showQuickJumper: true,
                showTotal: (total, range) => 
                  `${range[0]}-${range[1]} of ${total} positions`
              }}
              size="small"
              scroll={{ x: 800 }}
            />
          </Card>
        </Col>
      </Row>
    </div>
  );
};

export default ExposureAnalysis;