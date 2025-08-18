import React, { useState, useEffect } from 'react';
import { Card, Row, Col, Statistic, Badge, Table, Button, Space, Typography, Alert, Spin, Tabs, Input, Select, Modal, Form, InputNumber, notification } from 'antd';
import { 
  WifiOutlined, 
  DollarOutlined, 
  TrophyOutlined, 
  ShoppingCartOutlined,
  ReloadOutlined,
  ExclamationCircleOutlined,
  CheckCircleOutlined,
  LineChartOutlined,
  SearchOutlined,
  SettingOutlined,
  StockOutlined,
  BarChartOutlined,
  EyeOutlined,
  EyeInvisibleOutlined,
  EditOutlined,
  DeleteOutlined
} from '@ant-design/icons';
import { useMessageBus } from '../hooks/useMessageBus';

const { Title, Text } = Typography;

interface IBConnectionStatus {
  connected: boolean;
  gateway_type: string;
  host: string;
  port: number;
  client_id: number;
  account_id?: string;
  connection_time?: string;
  last_heartbeat?: string;
  error_message?: string;
}

interface IBAccountData {
  account_id: string;
  net_liquidation?: number;
  total_cash_value?: number;
  buying_power?: number;
  maintenance_margin?: number;
  initial_margin?: number;
  excess_liquidity?: number;
  currency: string;
  timestamp?: string;
}

interface IBPosition {
  position_key: string;
  account_id: string;
  contract_id: string;
  symbol: string;
  position: number;
  avg_cost?: number;
  market_price?: number;
  market_value?: number;
  unrealized_pnl?: number;
  realized_pnl?: number;
  timestamp?: string;
}

interface IBOrder {
  order_id: string;
  client_id: number;
  account_id: string;
  contract_id: string;
  symbol: string;
  action: string;
  order_type: string;
  total_quantity: number;
  filled_quantity: number;
  remaining_quantity: number;
  limit_price?: number;
  stop_price?: number;
  status: string;
  avg_fill_price?: number;
  commission?: number;
  timestamp?: string;
}

interface MarketDataSubscription {
  symbol: string;
  contract_id: string;
  data_type: string;
  subscribed: boolean;
  last_price?: number;
  bid?: number;
  ask?: number;
  volume?: number;
  timestamp?: string;
}

interface InstrumentSearchResult {
  contract_id: number;
  symbol: string;
  sec_type: string;
  exchange: string;
  currency: string;
  description?: string;
  strike?: number;
  expiry?: string;
  right?: string;
}

export const IBDashboard: React.FC = () => {
  const [connectionStatus, setConnectionStatus] = useState<IBConnectionStatus | null>(null);
  const [accountData, setAccountData] = useState<IBAccountData | null>(null);
  const [positions, setPositions] = useState<IBPosition[]>([]);
  const [orders, setOrders] = useState<IBOrder[]>([]);
  const [marketDataSubscriptions, setMarketDataSubscriptions] = useState<MarketDataSubscription[]>([]);
  const [instrumentSearchResults, setInstrumentSearchResults] = useState<InstrumentSearchResult[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [searchModalVisible, setSearchModalVisible] = useState(false);
  const [subscriptionModalVisible, setSubscriptionModalVisible] = useState(false);
  const [orderModifyModalVisible, setOrderModifyModalVisible] = useState(false);
  const [selectedOrder, setSelectedOrder] = useState<IBOrder | null>(null);

  const { latestMessage } = useMessageBus();

  // Handle WebSocket messages for IB data
  useEffect(() => {
    if (latestMessage) {
      try {
        // Check if latestMessage is already parsed or needs parsing
        let message;
        if (typeof latestMessage === 'string') {
          message = JSON.parse(latestMessage);
        } else if (typeof latestMessage === 'object' && latestMessage.payload) {
          // It's a MessageBusMessage object
          message = latestMessage.payload;
        } else {
          message = latestMessage;
        }
        
        if (message && message.type) {
          switch (message.type) {
            case 'ib_connection':
              if (message.data) {
                setConnectionStatus(message.data);
              }
              break;
            case 'ib_account':
              if (message.data) {
                setAccountData(message.data);
              }
              break;
            case 'ib_positions':
              if (message.data) {
                // Convert positions object to array
                const positionsArray = Object.entries(message.data).map(([key, pos]: [string, any]) => ({
                  ...pos,
                  position_key: key
                }));
                setPositions(positionsArray);
              }
              break;
            case 'ib_order':
              if (message.data) {
                // Update orders list with new order data
                setOrders(prevOrders => {
                  const existingIndex = prevOrders.findIndex(o => o.order_id === message.data.order_id);
                  if (existingIndex >= 0) {
                    const newOrders = [...prevOrders];
                    newOrders[existingIndex] = message.data;
                    return newOrders;
                  } else {
                    return [...prevOrders, message.data];
                  }
                });
              }
              break;
            case 'ib_market_data':
              if (message.data) {
                // Update market data subscriptions with real-time data
                setMarketDataSubscriptions(prev => 
                  prev.map(sub => 
                    sub.symbol === message.data.symbol 
                      ? { ...sub, ...message.data, timestamp: new Date().toISOString() }
                      : sub
                  )
                );
              }
              break;
          }
        }
      } catch (err) {
        console.error('Error parsing WebSocket message:', err, 'Message was:', latestMessage);
      }
    }
  }, [latestMessage]);

  // Fetch initial data
  useEffect(() => {
    const fetchIBData = async () => {
      console.log('🚀 Starting fetchIBData...');
      setLoading(true);
      setError(null);
      
      try {
        // Use the correct API base URL
        const apiUrl = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000';
        
        // Fetch connection status
        try {
          const connResponse = await fetch(`${apiUrl}/api/v1/ib/connection/status`, {
            credentials: 'include'
          });
          if (connResponse.ok) {
            const connData = await connResponse.json();
            console.log('IB Connection Data:', connData);
            setConnectionStatus(connData);
          } else {
            console.warn('IB Connection endpoint not available:', connResponse.status);
            // Set a default disconnected status instead of failing
            setConnectionStatus({
              connected: false,
              gateway_type: 'Unknown',
              host: 'localhost',
              port: 7496,
              client_id: 2,
              error_message: 'Connection endpoint not available'
            });
          }
        } catch (connError) {
          console.warn('Failed to fetch IB connection status:', connError);
          setConnectionStatus({
            connected: false,
            gateway_type: 'Unknown',
            host: 'localhost',
            port: 7496,
            client_id: 2,
            error_message: 'Failed to connect to IB Gateway'
          });
        }

        // Fetch account data
        try {
          const accountResponse = await fetch(`${apiUrl}/api/v1/ib/account`, {
            credentials: 'include'
          });
          if (accountResponse.ok) {
            const accountResult = await accountResponse.json();
            if (!accountResult.message) {
              setAccountData(accountResult);
            }
          }
        } catch (accountError) {
          console.warn('Failed to fetch IB account data:', accountError);
        }

        // Fetch positions
        try {
          const positionsResponse = await fetch(`${apiUrl}/api/v1/ib/positions`, {
            credentials: 'include'
          });
          if (positionsResponse.ok) {
            const positionsResult = await positionsResponse.json();
            setPositions(positionsResult.positions || []);
          }
        } catch (positionsError) {
          console.warn('Failed to fetch IB positions:', positionsError);
        }

        // Fetch orders
        try {
          const ordersResponse = await fetch(`${apiUrl}/api/v1/ib/orders`, {
            credentials: 'include'
          });
          if (ordersResponse.ok) {
            const ordersResult = await ordersResponse.json();
            setOrders(ordersResult.orders || []);
          }
        } catch (ordersError) {
          console.warn('Failed to fetch IB orders:', ordersError);
        }

      } catch (err) {
        console.error('Error in fetchIBData:', err);
        // Don't set error state for individual API failures
        // The component can still display with partial data
      } finally {
        setLoading(false);
      }
    };

    fetchIBData();
  }, []);

  const handleRefreshAccount = async () => {
    try {
      await fetch('/api/v1/ib/account/refresh', {
        method: 'POST',
        credentials: 'include'
      });
    } catch (err) {
      console.error('Error refreshing account data:', err);
    }
  };

  const handleRefreshPositions = async () => {
    try {
      await fetch('/api/v1/ib/positions/refresh', {
        method: 'POST',
        credentials: 'include'
      });
    } catch (err) {
      console.error('Error refreshing positions:', err);
    }
  };

  const handleRefreshOrders = async () => {
    try {
      await fetch('/api/v1/ib/orders/refresh', {
        method: 'POST',
        credentials: 'include'
      });
    } catch (err) {
      console.error('Error refreshing orders:', err);
    }
  };

  const handleInstrumentSearch = async (values: any) => {
    try {
      const response = await fetch('/api/v1/ib/search/contracts', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        credentials: 'include',
        body: JSON.stringify(values)
      });
      if (response.ok) {
        const results = await response.json();
        setInstrumentSearchResults(results.contracts || []);
      }
    } catch (err) {
      console.error('Error searching instruments:', err);
    }
  };

  const handleMarketDataSubscribe = async (symbol: string, contractId: string, dataType: string) => {
    try {
      const response = await fetch('/api/v1/ib/market-data/subscribe', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        credentials: 'include',
        body: JSON.stringify({ symbol, contract_id: contractId, data_type: dataType })
      });
      if (response.ok) {
        setMarketDataSubscriptions(prev => [
          ...prev.filter(sub => sub.symbol !== symbol),
          { symbol, contract_id: contractId, data_type: dataType, subscribed: true }
        ]);
        notification.success({ message: `Subscribed to ${symbol} market data` });
      }
    } catch (err) {
      console.error('Error subscribing to market data:', err);
      notification.error({ message: 'Failed to subscribe to market data' });
    }
  };

  const handleMarketDataUnsubscribe = async (symbol: string) => {
    try {
      await fetch('/api/v1/ib/market-data/unsubscribe', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        credentials: 'include',
        body: JSON.stringify({ symbol })
      });
      setMarketDataSubscriptions(prev => prev.filter(sub => sub.symbol !== symbol));
      notification.success({ message: `Unsubscribed from ${symbol} market data` });
    } catch (err) {
      console.error('Error unsubscribing from market data:', err);
      notification.error({ message: 'Failed to unsubscribe from market data' });
    }
  };

  const handleOrderModify = async (orderId: string, values: any) => {
    try {
      const response = await fetch(`/api/v1/ib/orders/${orderId}/modify`, {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        credentials: 'include',
        body: JSON.stringify(values)
      });
      if (response.ok) {
        notification.success({ message: 'Order modified successfully' });
        setOrderModifyModalVisible(false);
        handleRefreshOrders();
      }
    } catch (err) {
      console.error('Error modifying order:', err);
      notification.error({ message: 'Failed to modify order' });
    }
  };

  const handleOrderCancel = async (orderId: string) => {
    try {
      const response = await fetch(`/api/v1/ib/orders/${orderId}/cancel`, {
        method: 'DELETE',
        credentials: 'include'
      });
      if (response.ok) {
        notification.success({ message: 'Order cancelled successfully' });
        handleRefreshOrders();
      }
    } catch (err) {
      console.error('Error cancelling order:', err);
      notification.error({ message: 'Failed to cancel order' });
    }
  };

  const positionColumns = [
    {
      title: 'Symbol',
      dataIndex: 'symbol',
      key: 'symbol',
      width: 120,
    },
    {
      title: 'Position',
      dataIndex: 'position',
      key: 'position',
      width: 100,
      render: (value: number) => value.toFixed(0),
    },
    {
      title: 'Avg Cost',
      dataIndex: 'avg_cost',
      key: 'avg_cost',
      width: 100,
      render: (value: number) => value ? `$${value.toFixed(2)}` : '-',
    },
    {
      title: 'Market Price',
      dataIndex: 'market_price',
      key: 'market_price',
      width: 110,
      render: (value: number) => value ? `$${value.toFixed(2)}` : '-',
    },
    {
      title: 'Market Value',
      dataIndex: 'market_value',
      key: 'market_value',
      width: 110,
      render: (value: number) => value ? `$${value.toFixed(2)}` : '-',
    },
    {
      title: 'Unrealized P&L',
      dataIndex: 'unrealized_pnl',
      key: 'unrealized_pnl',
      width: 120,
      render: (value: number) => {
        if (!value) return '-';
        const color = value >= 0 ? '#52c41a' : '#ff4d4f';
        return <Text style={{ color }}>${value.toFixed(2)}</Text>;
      },
    },
  ];

  const orderColumns = [
    {
      title: 'Order ID',
      dataIndex: 'order_id',
      key: 'order_id',
      width: 100,
      render: (value: string) => value.substring(0, 8) + '...',
    },
    {
      title: 'Symbol',
      dataIndex: 'symbol',
      key: 'symbol',
      width: 100,
    },
    {
      title: 'Action',
      dataIndex: 'action',
      key: 'action',
      width: 80,
      render: (value: string) => (
        <Badge 
          color={value === 'BUY' ? 'green' : 'red'} 
          text={value} 
        />
      ),
    },
    {
      title: 'Type',
      dataIndex: 'order_type',
      key: 'order_type',
      width: 80,
    },
    {
      title: 'Quantity',
      dataIndex: 'total_quantity',
      key: 'total_quantity',
      width: 100,
      render: (value: number) => value.toFixed(0),
    },
    {
      title: 'Filled',
      dataIndex: 'filled_quantity',
      key: 'filled_quantity',
      width: 80,
      render: (value: number) => value.toFixed(0),
    },
    {
      title: 'Status',
      dataIndex: 'status',
      key: 'status',
      width: 100,
      render: (value: string) => {
        let color = 'default';
        if (value === 'Filled') color = 'success';
        else if (value === 'Cancelled') color = 'error';
        else if (value === 'Submitted') color = 'processing';
        return <Badge status={color as any} text={value} />;
      },
    },
    {
      title: 'Price',
      dataIndex: 'limit_price',
      key: 'limit_price',
      width: 80,
      render: (value: number, record: IBOrder) => {
        if (record.order_type === 'MKT') return 'Market';
        return value ? `$${value.toFixed(2)}` : '-';
      },
    },
    {
      title: 'Actions',
      key: 'actions',
      width: 120,
      render: (_, record: IBOrder) => (
        <Space>
          <Button 
            size="small" 
            icon={<EditOutlined />} 
            onClick={() => {
              setSelectedOrder(record);
              setOrderModifyModalVisible(true);
            }}
            disabled={record.status === 'Filled' || record.status === 'Cancelled'}
          />
          <Button 
            size="small" 
            danger 
            icon={<DeleteOutlined />} 
            onClick={() => handleOrderCancel(record.order_id)}
            disabled={record.status === 'Filled' || record.status === 'Cancelled'}
          />
        </Space>
      ),
    },
  ];

  const marketDataColumns = [
    {
      title: 'Symbol',
      dataIndex: 'symbol',
      key: 'symbol',
      width: 100,
    },
    {
      title: 'Data Type',
      dataIndex: 'data_type',
      key: 'data_type',
      width: 100,
    },
    {
      title: 'Last Price',
      dataIndex: 'last_price',
      key: 'last_price',
      width: 100,
      render: (value: number) => value ? `$${value.toFixed(2)}` : '-',
    },
    {
      title: 'Bid',
      dataIndex: 'bid',
      key: 'bid',
      width: 80,
      render: (value: number) => value ? `$${value.toFixed(2)}` : '-',
    },
    {
      title: 'Ask',
      dataIndex: 'ask',
      key: 'ask',
      width: 80,
      render: (value: number) => value ? `$${value.toFixed(2)}` : '-',
    },
    {
      title: 'Volume',
      dataIndex: 'volume',
      key: 'volume',
      width: 100,
      render: (value: number) => value ? value.toLocaleString() : '-',
    },
    {
      title: 'Actions',
      key: 'actions',
      width: 100,
      render: (_, record: MarketDataSubscription) => (
        <Button 
          size="small" 
          danger 
          icon={<EyeInvisibleOutlined />} 
          onClick={() => handleMarketDataUnsubscribe(record.symbol)}
        >
          Unsubscribe
        </Button>
      ),
    },
  ];

  const instrumentSearchColumns = [
    {
      title: 'Symbol',
      dataIndex: 'symbol',
      key: 'symbol',
      width: 100,
    },
    {
      title: 'Type',
      dataIndex: 'sec_type',
      key: 'sec_type',
      width: 80,
    },
    {
      title: 'Exchange',
      dataIndex: 'exchange',
      key: 'exchange',
      width: 100,
    },
    {
      title: 'Currency',
      dataIndex: 'currency',
      key: 'currency',
      width: 80,
    },
    {
      title: 'Description',
      dataIndex: 'description',
      key: 'description',
      width: 200,
      render: (value: string) => value || '-',
    },
    {
      title: 'Actions',
      key: 'actions',
      width: 150,
      render: (_, record: InstrumentSearchResult) => (
        <Space>
          <Button 
            size="small" 
            icon={<EyeOutlined />} 
            onClick={() => {
              setSubscriptionModalVisible(true);
              // Pre-fill with this instrument
            }}
          >
            Subscribe
          </Button>
        </Space>
      ),
    },
  ];

  console.log('IBDashboard render - connectionStatus:', connectionStatus);
  console.log('IBDashboard render - loading:', loading);

  if (loading) {
    return (
      <div style={{ textAlign: 'center', padding: '50px' }}>
        <Spin size="large" />
        <div style={{ marginTop: 16 }}>Loading Interactive Brokers data...</div>
      </div>
    );
  }

  return (
    <div style={{ padding: '24px' }}>
      <Title level={2}>
        <TrophyOutlined style={{ marginRight: '8px' }} />
        Interactive Brokers Dashboard
      </Title>

      {error && (
        <Alert
          message="Error"
          description={error}
          type="error"
          style={{ marginBottom: '24px' }}
          showIcon
        />
      )}

      {/* Connection Status */}
      <Card 
        title={
          <Space>
            <WifiOutlined />
            Connection Status
          </Space>
        }
        style={{ marginBottom: '24px' }}
      >
        <Row gutter={16}>
          <Col span={6}>
            <Statistic
              title="Status"
              value={connectionStatus?.connected ? 'Connected' : 'Disconnected'}
              prefix={
                connectionStatus?.connected ? 
                <CheckCircleOutlined style={{ color: '#52c41a' }} /> : 
                <ExclamationCircleOutlined style={{ color: '#ff4d4f' }} />
              }
              valueStyle={{ 
                color: connectionStatus?.connected ? '#52c41a' : '#ff4d4f',
                fontSize: '18px'
              }}
            />
          </Col>
          <Col span={6}>
            <Statistic
              title="Gateway Type"
              value={connectionStatus?.gateway_type || 'Unknown'}
            />
          </Col>
          <Col span={6}>
            <Statistic
              title="Host:Port"
              value={connectionStatus ? `${connectionStatus.host}:${connectionStatus.port}` : 'Unknown'}
            />
          </Col>
          <Col span={6}>
            <Statistic
              title="Account ID"
              value={connectionStatus?.account_id || 'Not set'}
            />
          </Col>
        </Row>
        {connectionStatus?.error_message && (
          <Alert
            message="Connection Error"
            description={connectionStatus.error_message}
            type="error"
            style={{ marginTop: '16px' }}
            showIcon
          />
        )}
      </Card>

      {/* Account Summary */}
      <Card 
        title={
          <Space>
            <DollarOutlined />
            Account Summary
            <Button 
              type="text" 
              icon={<ReloadOutlined />} 
              onClick={handleRefreshAccount}
              size="small"
            />
          </Space>
        }
        style={{ marginBottom: '24px' }}
      >
        {accountData ? (
          <Row gutter={16}>
            <Col span={6}>
              <Statistic
                title="Net Liquidation"
                value={accountData.net_liquidation}
                precision={2}
                prefix="$"
                valueStyle={{ color: '#3f8600' }}
              />
            </Col>
            <Col span={6}>
              <Statistic
                title="Total Cash"
                value={accountData.total_cash_value}
                precision={2}
                prefix="$"
              />
            </Col>
            <Col span={6}>
              <Statistic
                title="Buying Power"
                value={accountData.buying_power}
                precision={2}
                prefix="$"
              />
            </Col>
            <Col span={6}>
              <Statistic
                title="Excess Liquidity"
                value={accountData.excess_liquidity}
                precision={2}
                prefix="$"
              />
            </Col>
          </Row>
        ) : (
          <Alert
            message="No account data available"
            description="Account data will appear here when connected to Interactive Brokers"
            type="info"
            showIcon
          />
        )}
      </Card>

      {/* Main Content Tabs */}
      <Tabs defaultActiveKey="overview" items={[
        {
          key: 'overview',
          label: (
            <Space>
              <TrophyOutlined />
              Portfolio Overview
            </Space>
          ),
          children: (
            <Row gutter={16}>
              {/* Positions */}
              <Col span={12}>
                <Card 
                  title={
                    <Space>
                      <TrophyOutlined />
                      Positions ({positions.length})
                      <Button 
                        type="text" 
                        icon={<ReloadOutlined />} 
                        onClick={handleRefreshPositions}
                        size="small"
                      />
                    </Space>
                  }
                  style={{ height: '400px' }}
                >
                  <Table
                    columns={positionColumns}
                    dataSource={positions}
                    rowKey="position_key"
                    pagination={false}
                    scroll={{ y: 280 }}
                    size="small"
                  />
                </Card>
              </Col>

              {/* Orders */}
              <Col span={12}>
                <Card 
                  title={
                    <Space>
                      <ShoppingCartOutlined />
                      Orders ({orders.length})
                      <Button 
                        type="text" 
                        icon={<ReloadOutlined />} 
                        onClick={handleRefreshOrders}
                        size="small"
                      />
                    </Space>
                  }
                  style={{ height: '400px' }}
                >
                  <Table
                    columns={orderColumns}
                    dataSource={orders}
                    rowKey="order_id"
                    pagination={false}
                    scroll={{ y: 280 }}
                    size="small"
                  />
                </Card>
              </Col>
            </Row>
          ),
        },
        {
          key: 'market-data',
          label: (
            <Space>
              <LineChartOutlined />
              Market Data
            </Space>
          ),
          children: (
            <div>
              <div style={{ marginBottom: '16px' }}>
                <Space>
                  <Button 
                    type="primary" 
                    icon={<EyeOutlined />} 
                    onClick={() => setSubscriptionModalVisible(true)}
                  >
                    Subscribe to Market Data
                  </Button>
                  <Button 
                    icon={<SearchOutlined />} 
                    onClick={() => setSearchModalVisible(true)}
                  >
                    Search Instruments
                  </Button>
                </Space>
              </div>
              <Card 
                title={`Market Data Subscriptions (${marketDataSubscriptions.length})`}
                style={{ height: '500px' }}
              >
                <Table
                  columns={marketDataColumns}
                  dataSource={marketDataSubscriptions}
                  rowKey="symbol"
                  pagination={false}
                  scroll={{ y: 380 }}
                  size="small"
                />
              </Card>
            </div>
          ),
        },
        {
          key: 'instruments',
          label: (
            <Space>
              <StockOutlined />
              Instrument Discovery
            </Space>
          ),
          children: (
            <div>
              <div style={{ marginBottom: '16px' }}>
                <Button 
                  type="primary" 
                  icon={<SearchOutlined />} 
                  onClick={() => setSearchModalVisible(true)}
                >
                  Search Instruments
                </Button>
              </div>
              <Card 
                title={`Search Results (${instrumentSearchResults.length})`}
                style={{ height: '500px' }}
              >
                <Table
                  columns={instrumentSearchColumns}
                  dataSource={instrumentSearchResults}
                  rowKey="contract_id"
                  pagination={{ pageSize: 20 }}
                  scroll={{ y: 350 }}
                  size="small"
                />
              </Card>
            </div>
          ),
        },
        {
          key: 'analytics',
          label: (
            <Space>
              <BarChartOutlined />
              Analytics
            </Space>
          ),
          children: (
            <Row gutter={16}>
              <Col span={8}>
                <Card title="Performance Metrics">
                  <Statistic
                    title="Total P&L"
                    value={positions.reduce((sum, pos) => sum + (pos.unrealized_pnl || 0), 0)}
                    precision={2}
                    prefix="$"
                    valueStyle={{ color: positions.reduce((sum, pos) => sum + (pos.unrealized_pnl || 0), 0) >= 0 ? '#52c41a' : '#ff4d4f' }}
                  />
                  <Statistic
                    title="Active Positions"
                    value={positions.filter(pos => pos.position !== 0).length}
                    style={{ marginTop: '16px' }}
                  />
                  <Statistic
                    title="Open Orders"
                    value={orders.filter(order => order.status === 'Submitted' || order.status === 'PreSubmitted').length}
                    style={{ marginTop: '16px' }}
                  />
                </Card>
              </Col>
              <Col span={8}>
                <Card title="Risk Metrics">
                  <Statistic
                    title="Total Market Value"
                    value={positions.reduce((sum, pos) => sum + (pos.market_value || 0), 0)}
                    precision={2}
                    prefix="$"
                  />
                  <Statistic
                    title="Largest Position"
                    value={Math.max(...positions.map(pos => Math.abs(pos.market_value || 0)))}
                    precision={2}
                    prefix="$"
                    style={{ marginTop: '16px' }}
                  />
                </Card>
              </Col>
              <Col span={8}>
                <Card title="Trading Activity">
                  <Statistic
                    title="Orders Today"
                    value={orders.filter(order => {
                      if (!order.timestamp) return false;
                      const orderDate = new Date(order.timestamp).toDateString();
                      const today = new Date().toDateString();
                      return orderDate === today;
                    }).length}
                  />
                  <Statistic
                    title="Market Data Feeds"
                    value={marketDataSubscriptions.length}
                    style={{ marginTop: '16px' }}
                  />
                </Card>
              </Col>
            </Row>
          ),
        },
      ]} />
      {/* Instrument Search Modal */}
      <Modal
        title="Search Instruments"
        open={searchModalVisible}
        onCancel={() => setSearchModalVisible(false)}
        footer={null}
        width={800}
      >
        <Form
          layout="vertical"
          onFinish={handleInstrumentSearch}
          initialValues={{ sec_type: 'STK', exchange: 'SMART', currency: 'USD' }}
        >
          <Row gutter={16}>
            <Col span={8}>
              <Form.Item label="Symbol" name="symbol" rules={[{ required: true }]}>
                <Input placeholder="e.g. AAPL, MSFT" />
              </Form.Item>
            </Col>
            <Col span={8}>
              <Form.Item label="Security Type" name="sec_type">
                <Select>
                  <Select.Option value="STK">Stock</Select.Option>
                  <Select.Option value="OPT">Option</Select.Option>
                  <Select.Option value="FUT">Future</Select.Option>
                  <Select.Option value="CASH">Forex</Select.Option>
                  <Select.Option value="IND">Index</Select.Option>
                  <Select.Option value="BOND">Bond</Select.Option>
                </Select>
              </Form.Item>
            </Col>
            <Col span={8}>
              <Form.Item label="Exchange" name="exchange">
                <Select>
                  <Select.Option value="SMART">SMART</Select.Option>
                  <Select.Option value="NYSE">NYSE</Select.Option>
                  <Select.Option value="NASDAQ">NASDAQ</Select.Option>
                  <Select.Option value="IDEALPRO">IDEALPRO (Forex)</Select.Option>
                  <Select.Option value="CME">CME</Select.Option>
                  <Select.Option value="GLOBEX">GLOBEX</Select.Option>
                </Select>
              </Form.Item>
            </Col>
          </Row>
          <Row gutter={16}>
            <Col span={8}>
              <Form.Item label="Currency" name="currency">
                <Select>
                  <Select.Option value="USD">USD</Select.Option>
                  <Select.Option value="EUR">EUR</Select.Option>
                  <Select.Option value="GBP">GBP</Select.Option>
                  <Select.Option value="JPY">JPY</Select.Option>
                  <Select.Option value="CAD">CAD</Select.Option>
                </Select>
              </Form.Item>
            </Col>
            <Col span={8}>
              <Form.Item label="Expiry (Options/Futures)" name="expiry">
                <Input placeholder="YYYYMMDD" />
              </Form.Item>
            </Col>
            <Col span={8}>
              <Form.Item label="Strike (Options)" name="strike">
                <InputNumber placeholder="Strike price" style={{ width: '100%' }} />
              </Form.Item>
            </Col>
          </Row>
          <Form.Item>
            <Button type="primary" htmlType="submit" icon={<SearchOutlined />}>
              Search
            </Button>
          </Form.Item>
        </Form>
      </Modal>

      {/* Market Data Subscription Modal */}
      <Modal
        title="Subscribe to Market Data"
        open={subscriptionModalVisible}
        onCancel={() => setSubscriptionModalVisible(false)}
        footer={null}
      >
        <Form
          layout="vertical"
          onFinish={(values) => {
            handleMarketDataSubscribe(values.symbol, values.contract_id || '', values.data_type);
            setSubscriptionModalVisible(false);
          }}
          initialValues={{ data_type: 'TRADES' }}
        >
          <Form.Item label="Symbol" name="symbol" rules={[{ required: true }]}>
            <Input placeholder="e.g. AAPL, MSFT" />
          </Form.Item>
          <Form.Item label="Contract ID (optional)" name="contract_id">
            <Input placeholder="Leave empty for default contract" />
          </Form.Item>
          <Form.Item label="Data Type" name="data_type">
            <Select>
              <Select.Option value="TRADES">Trades</Select.Option>
              <Select.Option value="MIDPOINT">Midpoint</Select.Option>
              <Select.Option value="BID">Bid</Select.Option>
              <Select.Option value="ASK">Ask</Select.Option>
              <Select.Option value="BID_ASK">Bid/Ask</Select.Option>
              <Select.Option value="ADJUSTED_LAST">Adjusted Last</Select.Option>
            </Select>
          </Form.Item>
          <Form.Item>
            <Button type="primary" htmlType="submit" icon={<EyeOutlined />}>
              Subscribe
            </Button>
          </Form.Item>
        </Form>
      </Modal>

      {/* Order Modification Modal */}
      <Modal
        title="Modify Order"
        open={orderModifyModalVisible}
        onCancel={() => setOrderModifyModalVisible(false)}
        footer={null}
      >
        {selectedOrder && (
          <Form
            layout="vertical"
            onFinish={(values) => handleOrderModify(selectedOrder.order_id, values)}
            initialValues={{
              quantity: selectedOrder.total_quantity,
              limit_price: selectedOrder.limit_price,
              stop_price: selectedOrder.stop_price,
            }}
          >
            <Alert
              message={`Modifying order for ${selectedOrder.symbol}`}
              type="info"
              style={{ marginBottom: '16px' }}
            />
            <Form.Item label="Quantity" name="quantity">
              <InputNumber min={1} style={{ width: '100%' }} />
            </Form.Item>
            {selectedOrder.order_type !== 'MKT' && (
              <Form.Item label="Limit Price" name="limit_price">
                <InputNumber min={0.01} step={0.01} precision={2} style={{ width: '100%' }} />
              </Form.Item>
            )}
            {(selectedOrder.order_type === 'STP' || selectedOrder.order_type === 'STP_LMT') && (
              <Form.Item label="Stop Price" name="stop_price">
                <InputNumber min={0.01} step={0.01} precision={2} style={{ width: '100%' }} />
              </Form.Item>
            )}
            <Form.Item>
              <Space>
                <Button type="primary" htmlType="submit" icon={<EditOutlined />}>
                  Modify Order
                </Button>
                <Button onClick={() => setOrderModifyModalVisible(false)}>
                  Cancel
                </Button>
              </Space>
            </Form.Item>
          </Form>
        )}
      </Modal>
    </div>
  );
};

export default IBDashboard;