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
  TrophyOutlined,
  GlobalOutlined,
  SearchOutlined,
  DatabaseOutlined,
  LineChartOutlined,
  ReloadOutlined,
  InfoCircleOutlined,
  DollarOutlined,
  RiseOutlined,
  FundOutlined,
  StockOutlined
} from '@ant-design/icons';

import { 
  tradingEconomicsService, 
  type TradingEconomicsCountry,
  type TradingEconomicsIndicator,
  type TradingEconomicsCalendarEvent,
  type TradingEconomicsMarket 
} from '../../services/tradingEconomicsService';

const { Title, Text, Paragraph } = Typography;
const { Option } = Select;

interface TradingEconomicsPanelProps {
  enabled: boolean;
  onToggle: (enabled: boolean) => void;
}

const TradingEconomicsPanel: React.FC<TradingEconomicsPanelProps> = ({ enabled, onToggle }) => {
  const [loading, setLoading] = useState(false);
  const [healthStatus, setHealthStatus] = useState<any>(null);
  const [countries, setCountries] = useState<TradingEconomicsCountry[]>([]);
  const [majorIndicators, setMajorIndicators] = useState<any>(null);
  const [marketOverview, setMarketOverview] = useState<any>(null);
  const [statistics, setStatistics] = useState<any>(null);
  const [selectedCategory, setSelectedCategory] = useState<string>('gdp');
  const [searchKeyword, setSearchKeyword] = useState('');

  const categories = [
    { key: 'gdp', label: '📊 GDP', description: 'Gross domestic product growth and output' },
    { key: 'inflation', label: '📈 Inflation', description: 'Consumer price indices and inflation rates' },
    { key: 'employment', label: '👥 Employment', description: 'Unemployment rates and labor market data' },
    { key: 'interest_rates', label: '🏦 Interest Rates', description: 'Central bank rates and monetary policy' },
    { key: 'trade', label: '🌍 Trade', description: 'Trade balance and international commerce' }
  ];

  const majorCountries = [
    { code: 'united states', name: 'United States', flag: '🇺🇸' },
    { code: 'united kingdom', name: 'United Kingdom', flag: '🇬🇧' },
    { code: 'germany', name: 'Germany', flag: '🇩🇪' },
    { code: 'japan', name: 'Japan', flag: '🇯🇵' },
    { code: 'china', name: 'China', flag: '🇨🇳' },
    { code: 'france', name: 'France', flag: '🇫🇷' }
  ];

  const marketTypes = [
    { key: 'currencies', label: '💱 Currencies', icon: <DollarOutlined />, description: 'Foreign exchange rates' },
    { key: 'commodities', label: '🥇 Commodities', icon: <TrophyOutlined />, description: 'Gold, oil, and commodity prices' },
    { key: 'stocks', label: '📊 Stocks', icon: <RiseOutlined />, description: 'Global stock market indices' },
    { key: 'bonds', label: '📋 Bonds', icon: <FundOutlined />, description: 'Government and corporate bonds' }
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
      const health = await tradingEconomicsService.checkHealth();
      setHealthStatus(health);

      if (health.status === 'healthy' || health.status === 'mock_mode') {
        // Load countries
        const countriesData = await tradingEconomicsService.getCountries();
        setCountries(countriesData.slice(0, 10)); // Show top 10

        // Load statistics
        const stats = await tradingEconomicsService.getStatistics();
        setStatistics(stats);

        // Load major economic indicators
        const indicators = await tradingEconomicsService.getMajorIndicators();
        setMajorIndicators(indicators);

        // Load market overview
        const markets = await tradingEconomicsService.getMarketOverview();
        setMarketOverview(markets);
      }
    } catch (error) {
      console.error('Failed to load Trading Economics data:', error);
    } finally {
      setLoading(false);
    }
  };

  const handleCategoryChange = async (category: string) => {
    setSelectedCategory(category);
    // Category data is already loaded in majorIndicators
  };

  const indicatorColumns = [
    {
      title: 'Indicator',
      dataIndex: 'Title',
      key: 'title',
      render: (text: string, record: TradingEconomicsIndicator) => (
        <div>
          <Text strong style={{ fontSize: 12 }}>{text}</Text>
          <br />
          <Text type="secondary" style={{ fontSize: 10 }}>
            {record.Country} - {record.Category}
          </Text>
        </div>
      )
    },
    {
      title: 'Latest',
      dataIndex: 'LatestValue',
      key: 'latest',
      width: 80,
      render: (value: number) => (
        <Text style={{ fontSize: 11 }}>
          {value !== null ? value.toLocaleString() : 'N/A'}
        </Text>
      )
    },
    {
      title: 'Previous',
      dataIndex: 'PreviousValue',
      key: 'previous',
      width: 80,
      render: (value: number) => (
        <Text style={{ fontSize: 11 }}>
          {value !== null ? value.toLocaleString() : 'N/A'}
        </Text>
      )
    },
    {
      title: 'Unit',
      dataIndex: 'Unit',
      key: 'unit',
      width: 60,
      render: (text: string) => <Tag style={{ fontSize: 10 }}>{text}</Tag>
    }
  ];

  const marketColumns = [
    {
      title: 'Symbol',
      dataIndex: 'Symbol',
      key: 'symbol',
      render: (text: string, record: TradingEconomicsMarket) => (
        <div>
          <Text strong style={{ fontSize: 12 }}>{text}</Text>
          <br />
          <Text type="secondary" style={{ fontSize: 10 }}>{record.Name}</Text>
        </div>
      )
    },
    {
      title: 'Last',
      dataIndex: 'Last',
      key: 'last',
      width: 80,
      render: (value: number) => (
        <Text style={{ fontSize: 11 }}>
          {value !== null ? value.toFixed(4) : 'N/A'}
        </Text>
      )
    },
    {
      title: 'Change',
      dataIndex: 'DailyPercentualChange',
      key: 'change',
      width: 80,
      render: (value: number) => {
        const color = value > 0 ? '#52c41a' : value < 0 ? '#ff4d4f' : '#666';
        return (
          <Text style={{ fontSize: 11, color }}>
            {value !== null ? `${value > 0 ? '+' : ''}${value.toFixed(2)}%` : 'N/A'}
          </Text>
        );
      }
    }
  ];

  if (!enabled) {
    return (
      <Card>
        <div style={{ textAlign: 'center', padding: '40px 20px' }}>
          <TrophyOutlined style={{ fontSize: 48, color: '#f5222d', marginBottom: 16 }} />
          <Title level={4}>Trading Economics</Title>
          <Paragraph type="secondary">
            Access 300,000+ economic indicators across 196 countries. Get real-time economic calendar,
            market data, forecasts, and comprehensive economic analysis from Trading Economics.
          </Paragraph>
          <Button 
            type="primary" 
            icon={<GlobalOutlined />}
            onClick={() => onToggle(true)}
            size="large"
          >
            Enable Trading Economics
          </Button>
        </div>
      </Card>
    );
  }

  return (
    <Card
      title={
        <Space>
          <TrophyOutlined style={{ color: '#f5222d' }} />
          <span>Trading Economics</span>
          <Badge 
            status={healthStatus?.status === 'healthy' || healthStatus?.status === 'mock_mode' ? 'success' : 'error'} 
            text={
              healthStatus?.status === 'healthy' ? 'Connected' : 
              healthStatus?.status === 'mock_mode' ? 'Mock Mode' : 'Offline'
            }
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
            <Text>Loading Trading Economics data...</Text>
          </div>
        </div>
      )}

      {!loading && healthStatus && (
        <>
          {/* Connection Status */}
          {healthStatus.status === 'error' && (
            <Alert
              message="Trading Economics API Unavailable"
              description={`The Trading Economics service is currently unavailable: ${healthStatus.error || 'Unknown error'}`}
              type="warning"
              showIcon
              style={{ marginBottom: 16 }}
            />
          )}

          {healthStatus.using_guest_access && (
            <Alert
              message="Using Guest Access"
              description="Trading Economics is running with limited guest access. Consider upgrading to a paid plan for full data access."
              type="info"
              showIcon
              style={{ marginBottom: 16 }}
            />
          )}

          {(healthStatus.status === 'healthy' || healthStatus.status === 'mock_mode') && (
            <>
              {/* Statistics Overview */}
              <Row gutter={[16, 16]} style={{ marginBottom: 24 }}>
                <Col xs={12} sm={6}>
                  <Statistic
                    title="Countries"
                    value={196}
                    prefix={<GlobalOutlined />}
                    valueStyle={{ fontSize: 16 }}
                  />
                </Col>
                <Col xs={12} sm={6}>
                  <Statistic
                    title="Indicators"
                    value="300K+"
                    prefix={<DatabaseOutlined />}
                    valueStyle={{ fontSize: 16 }}
                  />
                </Col>
                <Col xs={12} sm={6}>
                  <Statistic
                    title="Markets"
                    value="All Major"
                    prefix={<StockOutlined />}
                    valueStyle={{ fontSize: 16 }}
                  />
                </Col>
                <Col xs={12} sm={6}>
                  <Statistic
                    title="Rate Limit"
                    value={statistics?.rate_limit_status?.requests_made || 0}
                    suffix={`/${statistics?.rate_limit_status?.requests_limit || 500}`}
                    prefix={<LineChartOutlined />}
                    valueStyle={{ fontSize: 16 }}
                  />
                </Col>
              </Row>

              {/* Major Countries */}
              <Card size="small" style={{ marginBottom: 16 }}>
                <Title level={5} style={{ marginBottom: 12 }}>
                  <GlobalOutlined /> Major Countries
                </Title>
                <Row gutter={[8, 8]}>
                  {majorCountries.map(country => (
                    <Col key={country.code} xs={12} sm={8} md={4}>
                      <Tooltip title={country.name}>
                        <Tag 
                          style={{ 
                            width: '100%', 
                            textAlign: 'center',
                            cursor: 'pointer',
                            fontSize: 11,
                            padding: '4px 8px'
                          }}
                          color="red"
                        >
                          {country.flag} {country.name}
                        </Tag>
                      </Tooltip>
                    </Col>
                  ))}
                </Row>
              </Card>

              {/* Market Types */}
              <Card size="small" style={{ marginBottom: 16 }}>
                <Title level={5} style={{ marginBottom: 12 }}>
                  <StockOutlined /> Market Categories
                </Title>
                <Row gutter={[8, 8]}>
                  {marketTypes.map(market => (
                    <Col key={market.key} xs={12} sm={6}>
                      <Tag 
                        style={{ 
                          width: '100%', 
                          textAlign: 'center',
                          cursor: 'pointer',
                          fontSize: 11,
                          padding: '8px'
                        }}
                        color="orange"
                        icon={market.icon}
                      >
                        {market.label}
                      </Tag>
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
                      color={selectedCategory === category.key ? 'red' : 'default'}
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

              {/* Major Economic Indicators */}
              {majorIndicators && (
                <Row gutter={[16, 16]} style={{ marginBottom: 16 }}>
                  <Col xs={24} sm={12}>
                    <Card size="small">
                      <Title level={5} style={{ marginBottom: 12 }}>
                        <LineChartOutlined /> Major Economic Indicators
                      </Title>
                      
                      {majorIndicators[selectedCategory as keyof typeof majorIndicators]?.length > 0 ? (
                        <Table
                          columns={indicatorColumns}
                          dataSource={majorIndicators[selectedCategory as keyof typeof majorIndicators]}
                          pagination={false}
                          size="small"
                          rowKey={(record) => `${record.Country}-${record.Category}-${record.Title}`}
                          style={{ fontSize: 11 }}
                        />
                      ) : (
                        <Text type="secondary">No indicators found for this category</Text>
                      )}
                    </Card>
                  </Col>

                  <Col xs={24} sm={12}>
                    <Card size="small">
                      <Title level={5} style={{ marginBottom: 12 }}>
                        <DollarOutlined /> Market Overview
                      </Title>
                      
                      {marketOverview?.currencies?.length > 0 ? (
                        <Table
                          columns={marketColumns}
                          dataSource={marketOverview.currencies}
                          pagination={false}
                          size="small"
                          rowKey={(record) => record.Symbol}
                          style={{ fontSize: 11 }}
                        />
                      ) : (
                        <Text type="secondary">No market data available</Text>
                      )}
                    </Card>
                  </Col>
                </Row>
              )}

              {/* Usage Instructions */}
              <Alert
                style={{ marginTop: 16 }}
                message={
                  <Space>
                    <InfoCircleOutlined />
                    <Text strong>Trading Economics Integration Active</Text>
                  </Space>
                }
                description={
                  <div>
                    <Text>
                      Trading Economics provides 300,000+ economic indicators from 196 countries. 
                      Use this comprehensive data to enhance your trading strategies with global economic context.
                    </Text>
                    <div style={{ marginTop: 8 }}>
                      <Text strong>Available via API:</Text>
                      <ul style={{ margin: '4px 0', paddingLeft: 16 }}>
                        <li><code>/api/v1/trading-economics/indicators</code> - Economic indicators data</li>
                        <li><code>/api/v1/trading-economics/calendar</code> - Economic calendar events</li>
                        <li><code>/api/v1/trading-economics/markets</code> - Market data by category</li>
                        <li><code>/api/v1/trading-economics/search</code> - Search indicators</li>
                      </ul>
                    </div>
                    <div style={{ marginTop: 8 }}>
                      <Text type="secondary">
                        Rate Limit: {statistics?.rate_limit_status?.requests_made || 0} / {statistics?.rate_limit_status?.requests_limit || 500} requests per minute
                      </Text>
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

export default TradingEconomicsPanel;