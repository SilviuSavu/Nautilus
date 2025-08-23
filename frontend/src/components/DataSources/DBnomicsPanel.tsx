import React, { useState, useEffect } from 'react';
import {
  Card,
  Row,
  Col,
  Statistic,
  Tag,
  Button,
  Space,
  Typography,
  Alert,
  List,
  Tooltip,
  Progress,
  Select,
  Input,
  Table,
  Badge,
  Spin
} from 'antd';
import {
  BarChartOutlined,
  GlobalOutlined,
  SearchOutlined,
  DatabaseOutlined,
  LineChartOutlined,
  ReloadOutlined,
  InfoCircleOutlined,
  TrophyOutlined
} from '@ant-design/icons';

import { dbnomicsService, type DBnomicsProvider, type DBnomicsSeries } from '../../services/dbnomicsService';

const { Title, Text, Paragraph } = Typography;
const { Option } = Select;

interface DBnomicsPanelProps {
  enabled: boolean;
  onToggle: (enabled: boolean) => void;
}

const DBnomicsPanel: React.FC<DBnomicsPanelProps> = ({ enabled, onToggle }) => {
  const [loading, setLoading] = useState(false);
  const [healthStatus, setHealthStatus] = useState<any>(null);
  const [providers, setProviders] = useState<DBnomicsProvider[]>([]);
  const [popularSeries, setPopularSeries] = useState<DBnomicsSeries[]>([]);
  const [statistics, setStatistics] = useState<any>(null);
  const [selectedCategory, setSelectedCategory] = useState<string>('inflation');
  const [searchKeyword, setSearchKeyword] = useState('');

  const categories = [
    { key: 'inflation', label: '📈 Inflation', description: 'Consumer price indices and inflation rates' },
    { key: 'employment', label: '👥 Employment', description: 'Unemployment rates and labor statistics' },
    { key: 'growth', label: '📊 Economic Growth', description: 'GDP growth and economic output' },
    { key: 'monetary', label: '🏦 Monetary Policy', description: 'Central bank rates and monetary indicators' },
    { key: 'trade', label: '🌍 Trade', description: 'Trade balances and international commerce' }
  ];

  const popularProviders = [
    { code: 'IMF', name: 'International Monetary Fund', flag: '🌍' },
    { code: 'OECD', name: 'Organisation for Economic Co-operation and Development', flag: '🏛️' },
    { code: 'ECB', name: 'European Central Bank', flag: '🇪🇺' },
    { code: 'EUROSTAT', name: 'European Union Statistics', flag: '📊' },
    { code: 'BIS', name: 'Bank for International Settlements', flag: '🏦' },
    { code: 'WB', name: 'World Bank', flag: '🌐' }
  ];

  useEffect(() => {
    if (enabled) {
      loadData();
    }
  }, [enabled]);

  const loadData = async () => {
    setLoading(true);
    try {
      // Load health status
      const health = await dbnomicsService.checkHealth();
      setHealthStatus(health);

      if (health.api_available) {
        // Load providers
        const providersData = await dbnomicsService.getProviders();
        setProviders(providersData.slice(0, 10)); // Show top 10

        // Load statistics
        const stats = await dbnomicsService.getStatistics();
        setStatistics(stats);

        // Load popular economic indicators
        const indicators = await dbnomicsService.getEconomicIndicators(selectedCategory as any);
        setPopularSeries(indicators.slice(0, 5));
      }
    } catch (error) {
      console.error('Failed to load DBnomics data:', error);
    } finally {
      setLoading(false);
    }
  };

  const handleCategoryChange = async (category: string) => {
    setSelectedCategory(category);
    try {
      const indicators = await dbnomicsService.getEconomicIndicators(category as any);
      setPopularSeries(indicators.slice(0, 5));
    } catch (error) {
      console.error('Failed to load category data:', error);
    }
  };

  const seriesColumns = [
    {
      title: 'Series',
      dataIndex: 'series_name',
      key: 'name',
      render: (text: string, record: DBnomicsSeries) => (
        <div>
          <Text strong style={{ fontSize: 12 }}>{text}</Text>
          <br />
          <Text type="secondary" style={{ fontSize: 10 }}>
            {record.provider_code}/{record.dataset_code}
          </Text>
        </div>
      )
    },
    {
      title: 'Frequency',
      dataIndex: 'frequency',
      key: 'frequency',
      width: 80,
      render: (text: string) => <Tag>{text}</Tag>
    },
    {
      title: 'Latest',
      dataIndex: 'last_date',
      key: 'date',
      width: 100,
      render: (text: string) => (
        <Text style={{ fontSize: 11 }}>{text?.substring(0, 10)}</Text>
      )
    },
    {
      title: 'Coverage',
      key: 'coverage',
      width: 100,
      render: (_: any, record: DBnomicsSeries) => (
        <Text style={{ fontSize: 11 }}>
          {record.observations_count?.toLocaleString()} obs
        </Text>
      )
    }
  ];

  if (!enabled) {
    return (
      <Card>
        <div style={{ textAlign: 'center', padding: '40px 20px' }}>
          <BarChartOutlined style={{ fontSize: 48, color: '#9254de', marginBottom: 16 }} />
          <Title level={4}>DBnomics Economic Data</Title>
          <Paragraph type="secondary">
            Access economic and statistical data from 80+ official providers worldwide including
            IMF, OECD, ECB, World Bank, and national statistical offices.
          </Paragraph>
          <Button 
            type="primary" 
            icon={<GlobalOutlined />}
            onClick={() => onToggle(true)}
            size="large"
          >
            Enable DBnomics
          </Button>
        </div>
      </Card>
    );
  }

  return (
    <Card
      title={
        <Space>
          <BarChartOutlined style={{ color: '#9254de' }} />
          <span>DBnomics Economic Data</span>
          <Badge 
            status={healthStatus?.api_available ? 'success' : 'error'} 
            text={healthStatus?.api_available ? 'Connected' : 'Offline'}
          />
        </Space>
      }
      extra={
        <Space>
          <Button
            icon={<ReloadOutlined />}
            onClick={loadData}
            loading={loading}
            size="small"
          >
            Refresh
          </Button>
          <Button
            danger
            onClick={() => onToggle(false)}
            size="small"
          >
            Disable
          </Button>
        </Space>
      }
    >
      {loading && (
        <div style={{ textAlign: 'center', padding: '20px' }}>
          <Spin size="large" />
          <div style={{ marginTop: 16 }}>
            <Text>Loading DBnomics data...</Text>
          </div>
        </div>
      )}

      {!loading && healthStatus && (
        <>
          {/* Connection Status */}
          {!healthStatus.api_available && (
            <Alert
              message="DBnomics API Unavailable"
              description="The DBnomics service is currently unavailable. Economic data features will be limited."
              type="warning"
              showIcon
              style={{ marginBottom: 16 }}
            />
          )}

          {healthStatus.api_available && (
            <>
              {/* Statistics Overview */}
              <Row gutter={[16, 16]} style={{ marginBottom: 24 }}>
                <Col xs={12} sm={6}>
                  <Statistic
                    title="Providers"
                    value={statistics?.totalProviders || 80}
                    prefix={<GlobalOutlined />}
                    valueStyle={{ fontSize: 16 }}
                  />
                </Col>
                <Col xs={12} sm={6}>
                  <Statistic
                    title="Datasets"
                    value={statistics?.totalDatasets || '1K+'}
                    prefix={<DatabaseOutlined />}
                    valueStyle={{ fontSize: 16 }}
                  />
                </Col>
                <Col xs={12} sm={6}>
                  <Statistic
                    title="Series"
                    value="800M+"
                    prefix={<LineChartOutlined />}
                    valueStyle={{ fontSize: 16 }}
                  />
                </Col>
                <Col xs={12} sm={6}>
                  <Statistic
                    title="Coverage"
                    value="Global"
                    prefix={<TrophyOutlined />}
                    valueStyle={{ fontSize: 16 }}
                  />
                </Col>
              </Row>

              {/* Popular Providers */}
              <Card size="small" style={{ marginBottom: 16 }}>
                <Title level={5} style={{ marginBottom: 12 }}>
                  <GlobalOutlined /> Popular Providers
                </Title>
                <Row gutter={[8, 8]}>
                  {popularProviders.map(provider => (
                    <Col key={provider.code} xs={12} sm={8} md={4}>
                      <Tooltip title={provider.name}>
                        <Tag 
                          style={{ 
                            width: '100%', 
                            textAlign: 'center',
                            cursor: 'pointer',
                            fontSize: 11,
                            padding: '4px 8px'
                          }}
                          color="blue"
                        >
                          {provider.flag} {provider.code}
                        </Tag>
                      </Tooltip>
                    </Col>
                  ))}
                </Row>
              </Card>

              {/* Category Selection */}
              <Card size="small" style={{ marginBottom: 16 }}>
                <Space wrap style={{ marginBottom: 12 }}>
                  <Text strong>Economic Categories:</Text>
                  {categories.map(category => (
                    <Tag
                      key={category.key}
                      color={selectedCategory === category.key ? 'purple' : 'default'}
                      style={{ cursor: 'pointer' }}
                      onClick={() => handleCategoryChange(category.key)}
                    >
                      {category.label}
                    </Tag>
                  ))}
                </Space>
                
                <Text type="secondary" style={{ fontSize: 12 }}>
                  {categories.find(c => c.key === selectedCategory)?.description}
                </Text>
              </Card>

              {/* Popular Series for Selected Category */}
              <Card size="small">
                <Title level={5} style={{ marginBottom: 12 }}>
                  <LineChartOutlined /> Popular {categories.find(c => c.key === selectedCategory)?.label} Series
                </Title>
                
                {popularSeries.length > 0 ? (
                  <Table
                    columns={seriesColumns}
                    dataSource={popularSeries}
                    pagination={false}
                    size="small"
                    rowKey={(record) => `${record.provider_code}/${record.dataset_code}/${record.series_code}`}
                    style={{ fontSize: 11 }}
                  />
                ) : (
                  <Text type="secondary">No series found for this category</Text>
                )}
              </Card>

              {/* Usage Instructions */}
              <Alert
                style={{ marginTop: 16 }}
                message={
                  <Space>
                    <InfoCircleOutlined />
                    <Text strong>DBnomics Integration Active</Text>
                  </Space>
                }
                description={
                  <div>
                    <Text>
                      DBnomics provides economic and statistical time series from 80+ official sources. 
                      Use this data to enhance your trading strategies with macroeconomic context.
                    </Text>
                    <div style={{ marginTop: 8 }}>
                      <Text strong>Available via API:</Text>
                      <ul style={{ margin: '4px 0', paddingLeft: 16 }}>
                        <li><code>/api/v1/dbnomics/series</code> - Fetch time series data</li>
                        <li><code>/api/v1/dbnomics/providers</code> - List data providers</li>
                        <li><code>/api/v1/dbnomics/search</code> - Search series by criteria</li>
                      </ul>
                    </div>
                  </div>
                }
                type="info"
                showIcon={false}
              />
            </>
          )}
        </>
      )}
    </Card>
  );
};

export default DBnomicsPanel;