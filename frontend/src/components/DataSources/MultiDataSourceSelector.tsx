import React, { useState, useEffect } from 'react';
import {
  Card,
  Button,
  Space,
  Typography,
  Badge,
  Row,
  Col,
  Tooltip,
  Statistic,
  Tag,
  Alert,
  Progress,
  Switch,
  Divider,
  Spin,
  notification
} from 'antd';
import {
  DatabaseOutlined,
  CloudOutlined,
  ApiOutlined,
  BankOutlined,
  LineChartOutlined,
  DollarCircleOutlined,
  FileTextOutlined,
  BarChartOutlined,
  CheckCircleOutlined,
  ExclamationCircleOutlined,
  LoadingOutlined,
  SyncOutlined,
  ThunderboltOutlined,
  GlobalOutlined
} from '@ant-design/icons';

const { Title, Text } = Typography;

export interface DataSource {
  id: string;
  name: string;
  description: string;
  icon: React.ReactNode;
  color: string;
  status: 'connected' | 'disconnected' | 'error' | 'loading';
  enabled: boolean;
  capabilities: string[];
  apiCalls?: {
    used: number;
    limit: number;
    resetTime?: string;
  };
  dataStats?: {
    instruments?: number | string;
    lastUpdate?: string;
    coverage?: string;
  };
  endpoints: string[];
  priority: number;
}

interface MultiDataSourceSelectorProps {
  onDataSourceToggle: (sourceId: string, enabled: boolean) => void;
  onRefreshAll: () => void;
  className?: string;
}

const MultiDataSourceSelector: React.FC<MultiDataSourceSelectorProps> = ({
  onDataSourceToggle,
  onRefreshAll,
  className
}) => {
  const [dataSources, setDataSources] = useState<DataSource[]>([]);
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);
  const [lastRefresh, setLastRefresh] = useState<Date | null>(null);

  const initializeDataSources = (): DataSource[] => [
    {
      id: 'ibkr',
      name: 'Interactive Brokers',
      description: 'Professional trading platform with real-time market data',
      icon: <DatabaseOutlined />,
      color: '#1890ff',
      status: 'loading',
      enabled: true,
      capabilities: ['Real-time Market Data', 'Order Execution', 'Historical Data', 'Multi-Asset Support'],
      endpoints: ['/api/v1/market-data/historical/bars', '/api/v1/ib/backfill'],
      priority: 1
    },
    {
      id: 'alpha_vantage',
      name: 'Alpha Vantage',
      description: 'Financial market data API with comprehensive coverage',
      icon: <LineChartOutlined />,
      color: '#52c41a',
      status: 'loading',
      enabled: true,
      capabilities: ['Stock Quotes', 'Company Fundamentals', 'Technical Indicators', 'Symbol Search'],
      apiCalls: {
        used: 0,
        limit: 500,
        resetTime: '24:00:00'
      },
      endpoints: ['/api/v1/nautilus-data/alpha-vantage/quote', '/api/v1/nautilus-data/alpha-vantage/search'],
      priority: 2
    },
    {
      id: 'fred',
      name: 'FRED Economic Data',
      description: 'Federal Reserve Economic Data for macro analysis',
      icon: <BankOutlined />,
      color: '#722ed1',
      status: 'loading',
      enabled: true,
      capabilities: ['Economic Indicators', 'Macro Factors', 'Interest Rates', 'Inflation Data'],
      dataStats: {
        instruments: 32,
        coverage: '50+ years'
      },
      endpoints: ['/api/v1/nautilus-data/fred/macro-factors', '/api/v1/nautilus-data/fred/series'],
      priority: 3
    },
    {
      id: 'edgar',
      name: 'SEC EDGAR',
      description: 'SEC filing data and regulatory information',
      icon: <FileTextOutlined />,
      color: '#fa8c16',
      status: 'loading',
      enabled: true,
      capabilities: ['SEC Filings', 'Company Facts', 'Insider Trading', 'Regulatory Compliance'],
      dataStats: {
        instruments: 7861,
        coverage: 'All public companies'
      },
      endpoints: ['/api/v1/edgar/companies/search', '/api/v1/edgar/ticker/{ticker}/resolve'],
      priority: 4
    },
    {
      id: 'trading_economics',
      name: 'Trading Economics',
      description: '300,000+ economic indicators across 196 countries with real-time data',
      icon: <GlobalOutlined />,
      color: '#f5222d',
      status: 'loading',
      enabled: false,  // Disabled by default since it's new
      capabilities: ['Economic Indicators', 'Market Data', 'Economic Calendar', 'Forecasts', 'Global Coverage'],
      apiCalls: {
        used: 0,
        limit: 500,
        resetTime: '1:00:00'
      },
      dataStats: {
        instruments: 300000,
        coverage: '196 countries',
        lastUpdate: 'Real-time'
      },
      endpoints: ['/api/v1/trading-economics/health', '/api/v1/trading-economics/indicators', '/api/v1/trading-economics/calendar'],
      priority: 5
    },
    {
      id: 'yfinance',
      name: 'Yahoo Finance',
      description: 'Free financial data with rate limiting protection',
      icon: <CloudOutlined />,
      color: '#eb2f96',
      status: 'loading',
      enabled: true,
      capabilities: ['Real-time Quotes', 'Historical Data', 'Market Information', 'Bulk Operations', 'Symbol Search'],
      dataStats: {
        instruments: 'Global',
        coverage: 'Free tier with rate limits'
      },
      endpoints: ['/api/v1/yfinance/quote/{symbol}', '/api/v1/yfinance/historical/{symbol}'],
      priority: 6
    },
    {
      id: 'backfill',
      name: 'Data Backfill Service',
      description: 'Automated historical data collection and gap filling',
      icon: <SyncOutlined />,
      color: '#13c2c2',
      status: 'loading',
      enabled: true,
      capabilities: ['Gap Detection', 'Batch Processing', 'Multi-Source Coordination', 'Progress Tracking'],
      endpoints: ['/api/v1/historical/backfill/start', '/api/v1/historical/backfill/status'],
      priority: 7
    },
    {
      id: 'dbnomics',
      name: 'DBnomics',
      description: 'Economic and statistical data from 80+ official providers worldwide',
      icon: <BarChartOutlined />,
      color: '#9254de',
      status: 'loading',
      enabled: false,  // Disabled by default since it's new
      capabilities: ['Economic Indicators', 'Statistical Data', 'Central Bank Data', 'Multi-Country Coverage'],
      dataStats: {
        instruments: 800000000,
        coverage: '80+ official providers',
        lastUpdate: 'Real-time'
      },
      endpoints: ['/api/v1/dbnomics/health', '/api/v1/dbnomics/series', '/api/v1/dbnomics/providers'],
      priority: 8
    }
  ];

  const checkDataSourceHealth = async (source: DataSource): Promise<DataSource> => {
    try {
      // DBnomics now has real backend endpoints

      const apiUrl = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8001';
      
      // Check different health endpoints based on source
      let healthEndpoint = '';
      switch (source.id) {
        case 'ibkr':
          healthEndpoint = '/health';
          break;
        case 'alpha_vantage':
        case 'fred':
          healthEndpoint = '/api/v1/nautilus-data/health';
          break;
        case 'edgar':
          healthEndpoint = '/api/v1/edgar/health';
          break;
        case 'trading_economics':
          healthEndpoint = '/api/v1/trading-economics/health';
          break;
        case 'yfinance':
          healthEndpoint = '/api/v1/yfinance/health';
          break;
        case 'backfill':
          healthEndpoint = '/api/v1/historical/backfill/status';
          break;
        default:
          healthEndpoint = '/health';
      }

      const response = await fetch(`${apiUrl}${healthEndpoint}`, {
        method: 'GET',
        headers: { 'Content-Type': 'application/json' }
      });

      if (response.ok) {
        const data = await response.json();
        
        // Update source based on response
        const updatedSource = { ...source };
        
        if (source.id === 'alpha_vantage' && data.find) {
          const avData = data.find((item: any) => item.source === 'Alpha Vantage');
          if (avData) {
            updatedSource.status = avData.status === 'operational' ? 'connected' : 'error';
            if (updatedSource.apiCalls) {
              // In a real implementation, you'd get this from the API
              updatedSource.apiCalls.used = Math.floor(Math.random() * 100);
            }
          }
        } else if (source.id === 'fred' && data.find) {
          const fredData = data.find((item: any) => item.source === 'FRED');
          if (fredData) {
            updatedSource.status = fredData.status === 'operational' ? 'connected' : 'error';
          }
        } else if (source.id === 'edgar') {
          updatedSource.status = data.status === 'healthy' ? 'connected' : 'error';
          if (data.statistics && updatedSource.dataStats) {
            updatedSource.dataStats.instruments = data.statistics.total_companies || 7861;
          }
        } else if (source.id === 'backfill') {
          updatedSource.status = data.controller ? 'connected' : 'error';
        } else if (source.id === 'dbnomics') {
          // DBnomics health check response structure
          updatedSource.status = data.status === 'healthy' || data.api_available ? 'connected' : 'error';
          if (data.providers && updatedSource.dataStats) {
            updatedSource.dataStats.instruments = `${data.providers.length}+ providers`;
          }
        } else {
          updatedSource.status = 'connected';
        }
        
        return updatedSource;
      } else {
        return { ...source, status: 'error' };
      }
    } catch (error) {
      console.error(`Health check failed for ${source.name}:`, error);
      return { ...source, status: 'error' };
    }
  };

  const refreshDataSources = async () => {
    setRefreshing(true);
    try {
      const sources = initializeDataSources();
      setDataSources(sources);

      // Check health for each source
      const healthChecks = sources.map(source => checkDataSourceHealth(source));
      const updatedSources = await Promise.all(healthChecks);
      
      setDataSources(updatedSources);
      setLastRefresh(new Date());
      
      notification.success({
        message: 'Data Sources Refreshed',
        description: 'All data source statuses have been updated.',
        duration: 3
      });
    } catch (error) {
      console.error('Error refreshing data sources:', error);
      notification.error({
        message: 'Refresh Failed',
        description: 'Failed to refresh data source statuses.',
        duration: 3
      });
    } finally {
      setRefreshing(false);
    }
  };

  useEffect(() => {
    const loadDataSources = async () => {
      setLoading(true);
      await refreshDataSources();
      setLoading(false);
    };

    loadDataSources();

    // Auto-refresh every 30 seconds
    const interval = setInterval(refreshDataSources, 30000);
    return () => clearInterval(interval);
  }, []);

  const handleToggleDataSource = async (sourceId: string, enabled: boolean) => {
    // Optimistically update the UI first
    setDataSources(prev => 
      prev.map(source => 
        source.id === sourceId ? { ...source, enabled, status: 'loading' } : source
      )
    );

    try {
      // DBnomics now has backend integration via MessageBus

      const apiUrl = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8001';
      
      // Call the backend to enable/disable the data source
      const endpoint = enabled 
        ? `/api/v1/multi-datasource/enable/${sourceId}`
        : `/api/v1/multi-datasource/disable/${sourceId}`;
      
      const response = await fetch(`${apiUrl}${endpoint}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' }
      });

      if (response.ok) {
        const result = await response.json();
        console.log(`✅ ${sourceId} ${enabled ? 'enabled' : 'disabled'}:`, result.message);
        
        // Update the data source status
        setDataSources(prev => 
          prev.map(source => 
            source.id === sourceId 
              ? { ...source, enabled, status: 'connected' } 
              : source
          )
        );

        // Call the parent component handler
        onDataSourceToggle(sourceId, enabled);

        // Show success notification
        notification.success({
          message: `Data Source ${enabled ? 'Enabled' : 'Disabled'}`,
          description: `${sourceId.toUpperCase()} has been ${enabled ? 'enabled' : 'disabled'} successfully.`,
          duration: 3
        });

      } else {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
      }

    } catch (error) {
      console.error(`❌ Failed to ${enabled ? 'enable' : 'disable'} ${sourceId}:`, error);
      
      // Revert the optimistic update
      setDataSources(prev => 
        prev.map(source => 
          source.id === sourceId 
            ? { ...source, enabled: !enabled, status: 'error' } 
            : source
        )
      );

      // Show error notification
      notification.error({
        message: `Failed to ${enabled ? 'Enable' : 'Disable'} Data Source`,
        description: `Could not ${enabled ? 'enable' : 'disable'} ${sourceId.toUpperCase()}. ${error instanceof Error ? error.message : 'Unknown error'}`,
        duration: 5
      });
    }
  };

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'connected':
        return <CheckCircleOutlined style={{ color: '#52c41a' }} />;
      case 'error':
        return <ExclamationCircleOutlined style={{ color: '#ff4d4f' }} />;
      case 'loading':
        return <LoadingOutlined style={{ color: '#1890ff' }} />;
      default:
        return <ExclamationCircleOutlined style={{ color: '#d9d9d9' }} />;
    }
  };

  const getStatusBadge = (status: string) => {
    switch (status) {
      case 'connected':
        return <Badge status="success" text="Connected" />;
      case 'error':
        return <Badge status="error" text="Error" />;
      case 'loading':
        return <Badge status="processing" text="Checking..." />;
      default:
        return <Badge status="default" text="Disconnected" />;
    }
  };

  const enabledSources = dataSources.filter(source => source.enabled);
  const connectedSources = dataSources.filter(source => source.status === 'connected' && source.enabled);

  if (loading) {
    return (
      <Card className={className}>
        <div style={{ textAlign: 'center', padding: '40px' }}>
          <Spin size="large" />
          <div style={{ marginTop: 16 }}>
            <Text>Loading data sources...</Text>
          </div>
        </div>
      </Card>
    );
  }

  return (
    <Card 
      title={
        <Space>
          <GlobalOutlined />
          <span>Multi-Source Data Selection</span>
          <Tag color="blue">{connectedSources.length}/{enabledSources.length} Active</Tag>
        </Space>
      }
      extra={
        <Space>
          {lastRefresh && (
            <Text type="secondary" style={{ fontSize: 12 }}>
              Last updated: {lastRefresh.toLocaleTimeString()}
            </Text>
          )}
          <Button
            icon={<SyncOutlined spin={refreshing} />}
            onClick={refreshDataSources}
            loading={refreshing}
            size="small"
          >
            Refresh
          </Button>
          <Button
            icon={<ThunderboltOutlined />}
            onClick={onRefreshAll}
            type="primary"
            size="small"
          >
            Refresh All
          </Button>
        </Space>
      }
      className={className}
    >
      {/* Summary Statistics */}
      <Row gutter={[16, 16]} style={{ marginBottom: 24 }}>
        <Col xs={24} sm={8} md={6}>
          <Statistic
            title="Total Sources"
            value={dataSources.length}
            prefix={<DatabaseOutlined />}
          />
        </Col>
        <Col xs={24} sm={8} md={6}>
          <Statistic
            title="Enabled"
            value={enabledSources.length}
            valueStyle={{ color: '#52c41a' }}
            prefix={<CheckCircleOutlined />}
          />
        </Col>
        <Col xs={24} sm={8} md={6}>
          <Statistic
            title="Connected"
            value={connectedSources.length}
            valueStyle={{ color: '#1890ff' }}
            prefix={<ApiOutlined />}
          />
        </Col>
        <Col xs={24} sm={8} md={6}>
          <Statistic
            title="Coverage"
            value={`${Math.round((connectedSources.length / dataSources.length) * 100)}%`}
            valueStyle={{ color: connectedSources.length === dataSources.length ? '#52c41a' : '#fa8c16' }}
            suffix="%"
          />
        </Col>
      </Row>

      {/* Multi-Source Alert */}
      <Alert
        message="Multi-Source Configuration Active"
        description="You can enable multiple data sources simultaneously. The system will intelligently route requests based on data availability, rate limits, and source capabilities. Priority is given to professional sources (IBKR) followed by supplementary sources (Alpha Vantage, FRED, DBnomics)."
        type="info"
        showIcon
        style={{ marginBottom: 24 }}
      />

      {/* Data Source Grid */}
      <Row gutter={[16, 16]}>
        {dataSources
          .sort((a, b) => a.priority - b.priority)
          .map((source) => (
            <Col xs={24} sm={12} md={8} lg={6} key={source.id}>
              <Card
                size="small"
                style={{
                  border: source.enabled ? `2px solid ${source.color}` : '1px solid #d9d9d9',
                  backgroundColor: source.enabled ? '#fafafa' : '#ffffff',
                  height: '100%'
                }}
                bodyStyle={{ height: '100%', display: 'flex', flexDirection: 'column' }}
              >
                <div style={{ textAlign: 'center', marginBottom: 12 }}>
                  <div
                    style={{
                      fontSize: 24,
                      color: source.color,
                      marginBottom: 8
                    }}
                  >
                    {source.icon}
                  </div>
                  <Title level={5} style={{ margin: 0 }}>
                    {source.name}
                  </Title>
                  <Text type="secondary" style={{ fontSize: 11 }}>
                    {source.description}
                  </Text>
                </div>

                <div style={{ marginBottom: 12 }}>
                  {getStatusBadge(source.status)}
                </div>

                {/* API Usage (for Alpha Vantage) */}
                {source.apiCalls && (
                  <div style={{ marginBottom: 12 }}>
                    <Text strong style={{ fontSize: 12 }}>API Usage</Text>
                    <Progress
                      percent={Math.round((source.apiCalls.used / source.apiCalls.limit) * 100)}
                      size="small"
                      status={source.apiCalls.used / source.apiCalls.limit > 0.8 ? 'exception' : 'normal'}
                    />
                    <Text style={{ fontSize: 11 }}>
                      {source.apiCalls.used}/{source.apiCalls.limit} calls
                    </Text>
                  </div>
                )}

                {/* Data Stats */}
                {source.dataStats && (
                  <div style={{ marginBottom: 12 }}>
                    <Text strong style={{ fontSize: 12 }}>Coverage</Text>
                    <div style={{ fontSize: 11 }}>
                      {source.dataStats.instruments && (
                        <div>📊 {typeof source.dataStats.instruments === 'number' ? source.dataStats.instruments.toLocaleString() : source.dataStats.instruments} instruments</div>
                      )}
                      {source.dataStats.coverage && (
                        <div>📅 {source.dataStats.coverage}</div>
                      )}
                    </div>
                  </div>
                )}

                {/* Capabilities */}
                <div style={{ marginBottom: 12, flex: 1 }}>
                  <Text strong style={{ fontSize: 12 }}>Capabilities</Text>
                  <div style={{ marginTop: 4 }}>
                    {source.capabilities.map((capability, index) => (
                      <Tag key={index} style={{ fontSize: 10, marginBottom: 2 }}>
                        {capability}
                      </Tag>
                    ))}
                  </div>
                </div>

                {/* Toggle Switch */}
                <div style={{ textAlign: 'center', marginTop: 'auto' }}>
                  <Switch
                    checked={source.enabled}
                    onChange={(enabled) => handleToggleDataSource(source.id, enabled)}
                    checkedChildren="ON"
                    unCheckedChildren="OFF"
                    disabled={source.status === 'loading'}
                  />
                </div>
              </Card>
            </Col>
          ))}
      </Row>

      {/* Active Sources Summary */}
      {enabledSources.length > 0 && (
        <>
          <Divider>Active Data Sources</Divider>
          <Space wrap>
            {enabledSources.map((source) => (
              <Tag
                key={source.id}
                color={source.color}
                icon={getStatusIcon(source.status)}
              >
                {source.name}
              </Tag>
            ))}
          </Space>
        </>
      )}
    </Card>
  );
};

export default MultiDataSourceSelector;