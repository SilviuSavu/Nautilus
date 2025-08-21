import React, { useState, useEffect } from 'react';
import {
  Table,
  Card,
  Button,
  Space,
  DatePicker,
  Select,
  Input,
  Row,
  Col,
  Typography,
  Tag,
  Statistic,
  Modal,
  Descriptions,
  Spin,
  Alert,
} from 'antd';
import { ExportOutlined, FilterFilled, ReloadOutlined, EyeOutlined, DollarCircleOutlined, TrophyOutlined } from '@ant-design/icons';
import { ColumnsType } from 'antd/es/table';
import dayjs, { Dayjs } from 'dayjs';

const { Title, Text } = Typography;
const { RangePicker } = DatePicker;
const { Option } = Select;

interface Trade {
  trade_id: string;
  account_id: string;
  venue: string;
  symbol: string;
  side: string;
  quantity: string;
  price: string;
  commission: string;
  execution_time: string;
  order_id?: string;
  execution_id?: string;
  strategy?: string;
  notes?: string;
}

interface TradeSummary {
  total_trades: number;
  winning_trades: number;
  losing_trades: number;
  win_rate: number;
  total_pnl: string;
  total_commission: string;
  net_pnl: string;
  average_win: string;
  average_loss: string;
  profit_factor: number;
  max_drawdown: string;
  sharpe_ratio?: number;
  start_date: string;
  end_date: string;
}

interface TradeFilters {
  account_id?: string;
  venue?: string;
  symbol?: string;
  strategy?: string;
  dateRange?: [Dayjs, Dayjs];
  limit: number;
  offset: number;
}

export const TradeHistoryTable: React.FC = () => {
  const [trades, setTrades] = useState<Trade[]>([]);
  const [loading, setLoading] = useState(false);
  const [summary, setSummary] = useState<TradeSummary | null>(null);
  const [selectedTrade, setSelectedTrade] = useState<Trade | null>(null);
  const [detailModalVisible, setDetailModalVisible] = useState(false);
  const [filters, setFilters] = useState<TradeFilters>({
    limit: 100,
    offset: 0
  });
  const [symbols, setSymbols] = useState<string[]>([]);
  const [strategies, setStrategies] = useState<string[]>([]);
  const [venues] = useState<string[]>(['IB', 'BINANCE', 'BYBIT']);

  // Fetch trades with current filters
  const fetchTrades = async () => {
    setLoading(true);
    try {
      const params = new URLSearchParams();
      
      if (filters.account_id) params.append('account_id', filters.account_id);
      if (filters.venue) params.append('venue', filters.venue);
      if (filters.symbol) params.append('symbol', filters.symbol);
      if (filters.strategy) params.append('strategy', filters.strategy);
      if (filters.dateRange && filters.dateRange[0] && filters.dateRange[1]) {
        params.append('start_date', filters.dateRange[0].toISOString());
        params.append('end_date', filters.dateRange[1].toISOString());
      }
      params.append('limit', filters.limit.toString());
      params.append('offset', filters.offset.toString());

      const response = await fetch(`/api/v1/trades/?${params.toString()}`);
      if (response.ok) {
        const data = await response.json();
        setTrades(data);
      } else {
        console.error('Failed to fetch trades');
      }
    } catch (error) {
      console.error('Error fetching trades:', error);
    } finally {
      setLoading(false);
    }
  };

  // Fetch trade summary
  const fetchSummary = async () => {
    try {
      const params = new URLSearchParams();
      
      if (filters.account_id) params.append('account_id', filters.account_id);
      if (filters.venue) params.append('venue', filters.venue);
      if (filters.symbol) params.append('symbol', filters.symbol);
      if (filters.strategy) params.append('strategy', filters.strategy);
      if (filters.dateRange && filters.dateRange[0] && filters.dateRange[1]) {
        params.append('start_date', filters.dateRange[0].toISOString());
        params.append('end_date', filters.dateRange[1].toISOString());
      }

      const response = await fetch(`/api/v1/trades/summary?${params.toString()}`);
      if (response.ok) {
        const data = await response.json();
        setSummary(data);
      }
    } catch (error) {
      console.error('Error fetching summary:', error);
    }
  };

  // Fetch available symbols and strategies
  const fetchMetadata = async () => {
    try {
      const [symbolsRes, strategiesRes] = await Promise.all([
        fetch('/api/v1/trades/symbols'),
        fetch('/api/v1/trades/strategies')
      ]);

      if (symbolsRes.ok) {
        const symbolsData = await symbolsRes.json();
        setSymbols(symbolsData.symbols.map((s: any) => s.symbol));
      }

      if (strategiesRes.ok) {
        const strategiesData = await strategiesRes.json();
        setStrategies(strategiesData.strategies.map((s: any) => s.strategy));
      }
    } catch (error) {
      console.error('Error fetching metadata:', error);
    }
  };

  // Export trades
  const exportTrades = async (format: 'csv' | 'json' | 'excel') => {
    try {
      const params = new URLSearchParams();
      params.append('format', format);
      
      if (filters.account_id) params.append('account_id', filters.account_id);
      if (filters.venue) params.append('venue', filters.venue);
      if (filters.symbol) params.append('symbol', filters.symbol);
      if (filters.strategy) params.append('strategy', filters.strategy);
      if (filters.dateRange && filters.dateRange[0] && filters.dateRange[1]) {
        params.append('start_date', filters.dateRange[0].toISOString());
        params.append('end_date', filters.dateRange[1].toISOString());
      }
      params.append('limit', '5000'); // Higher limit for export

      const response = await fetch(`/api/v1/trades/export?${params.toString()}`);
      if (response.ok) {
        const blob = await response.blob();
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `trades_${dayjs().format('YYYYMMDD_HHmmss')}.${format}`;
        document.body.appendChild(a);
        a.click();
        window.URL.revokeObjectURL(url);
        document.body.removeChild(a);
      }
    } catch (error) {
      console.error('Error exporting trades:', error);
    }
  };

  useEffect(() => {
    fetchTrades();
    fetchSummary();
  }, [filters]);

  useEffect(() => {
    fetchMetadata();
  }, []);

  // Table columns
  const columns: ColumnsType<Trade> = [
    {
      title: 'Time',
      dataIndex: 'execution_time',
      key: 'execution_time',
      width: 180,
      render: (time: string) => dayjs(time).format('YYYY-MM-DD HH:mm:ss'),
      sorter: (a, b) => dayjs(a.execution_time).unix() - dayjs(b.execution_time).unix(),
    },
    {
      title: 'Symbol',
      dataIndex: 'symbol',
      key: 'symbol',
      width: 100,
      render: (symbol: string) => <Text strong>{symbol}</Text>,
      sorter: (a, b) => a.symbol.localeCompare(b.symbol),
    },
    {
      title: 'Side',
      dataIndex: 'side',
      key: 'side',
      width: 80,
      render: (side: string) => (
        <Tag color={side === 'BUY' ? 'green' : 'red'}>
          {side}
        </Tag>
      ),
    },
    {
      title: 'Quantity',
      dataIndex: 'quantity',
      key: 'quantity',
      width: 120,
      align: 'right',
      render: (qty: string) => parseFloat(qty).toLocaleString(),
      sorter: (a, b) => parseFloat(a.quantity) - parseFloat(b.quantity),
    },
    {
      title: 'Price',
      dataIndex: 'price',
      key: 'price',
      width: 120,
      align: 'right',
      render: (price: string) => `$${parseFloat(price).toFixed(2)}`,
      sorter: (a, b) => parseFloat(a.price) - parseFloat(b.price),
    },
    {
      title: 'Value',
      key: 'value',
      width: 140,
      align: 'right',
      render: (_, record) => {
        const value = parseFloat(record.quantity) * parseFloat(record.price);
        return `$${value.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`;
      },
      sorter: (a, b) => {
        const aValue = parseFloat(a.quantity) * parseFloat(a.price);
        const bValue = parseFloat(b.quantity) * parseFloat(b.price);
        return aValue - bValue;
      },
    },
    {
      title: 'Commission',
      dataIndex: 'commission',
      key: 'commission',
      width: 120,
      align: 'right',
      render: (commission: string) => `$${parseFloat(commission).toFixed(2)}`,
    },
    {
      title: 'Venue',
      dataIndex: 'venue',
      key: 'venue',
      width: 80,
      render: (venue: string) => <Tag>{venue}</Tag>,
    },
    {
      title: 'Strategy',
      dataIndex: 'strategy',
      key: 'strategy',
      width: 120,
      render: (strategy?: string) => strategy ? <Tag color="blue">{strategy}</Tag> : '-',
    },
    {
      title: 'Actions',
      key: 'actions',
      width: 100,
      render: (_, record) => (
        <Button
          type="link"
          icon={<EyeOutlined />}
          onClick={() => {
            setSelectedTrade(record);
            setDetailModalVisible(true);
          }}
        >
          Details
        </Button>
      ),
    },
  ];

  return (
    <div>
      {/* Summary Statistics */}
      {summary && (
        <Card style={{ marginBottom: 16 }}>
          <Row gutter={[16, 16]}>
            <Col span={6}>
              <Statistic
                title="Total Trades"
                value={summary.total_trades}
                prefix={<TrophyOutlined />}
              />
            </Col>
            <Col span={6}>
              <Statistic
                title="Win Rate"
                value={summary.win_rate}
                suffix="%"
                precision={1}
                valueStyle={{ color: summary.win_rate > 50 ? '#3f8600' : '#cf1322' }}
              />
            </Col>
            <Col span={6}>
              <Statistic
                title="Net P&L"
                value={parseFloat(summary.net_pnl)}
                prefix={<DollarCircleOutlined />}
                precision={2}
                valueStyle={{ color: parseFloat(summary.net_pnl) >= 0 ? '#3f8600' : '#cf1322' }}
              />
            </Col>
            <Col span={6}>
              <Statistic
                title="Total Commission"
                value={parseFloat(summary.total_commission)}
                prefix={<DollarCircleOutlined />}
                precision={2}
                valueStyle={{ color: '#666' }}
              />
            </Col>
          </Row>
        </Card>
      )}

      {/* Filters */}
      <Card style={{ marginBottom: 16 }}>
        <Title level={4}><FilterFilled /> Filters</Title>
        <Row gutter={[16, 16]}>
          <Col span={6}>
            <label>Date Range:</label>
            <RangePicker
              style={{ width: '100%' }}
              value={filters.dateRange}
              onChange={(dates) => setFilters({ ...filters, dateRange: dates || undefined })}
              showTime
            />
          </Col>
          <Col span={4}>
            <label>Symbol:</label>
            <Select
              style={{ width: '100%' }}
              placeholder="All symbols"
              allowClear
              value={filters.symbol}
              onChange={(value) => setFilters({ ...filters, symbol: value })}
            >
              {symbols.map(symbol => (
                <Option key={symbol} value={symbol}>{symbol}</Option>
              ))}
            </Select>
          </Col>
          <Col span={4}>
            <label>Venue:</label>
            <Select
              style={{ width: '100%' }}
              placeholder="All venues"
              allowClear
              value={filters.venue}
              onChange={(value) => setFilters({ ...filters, venue: value })}
            >
              {venues.map(venue => (
                <Option key={venue} value={venue}>{venue}</Option>
              ))}
            </Select>
          </Col>
          <Col span={4}>
            <label>Strategy:</label>
            <Select
              style={{ width: '100%' }}
              placeholder="All strategies"
              allowClear
              value={filters.strategy}
              onChange={(value) => setFilters({ ...filters, strategy: value })}
            >
              {strategies.map(strategy => (
                <Option key={strategy} value={strategy}>{strategy}</Option>
              ))}
            </Select>
          </Col>
          <Col span={6}>
            <label>Actions:</label>
            <Space>
              <Button 
                icon={<ReloadOutlined />} 
                onClick={() => {
                  fetchTrades();
                  fetchSummary();
                }}
              >
                Refresh
              </Button>
              <Button 
                icon={<ExportOutlined />} 
                onClick={() => exportTrades('csv')}
              >
                Export CSV
              </Button>
              <Button 
                icon={<ExportOutlined />} 
                onClick={() => exportTrades('json')}
              >
                Export JSON
              </Button>
            </Space>
          </Col>
        </Row>
      </Card>

      {/* Trades Table */}
      <Card>
        <Title level={4}>Trade History</Title>
        <Table
          columns={columns}
          dataSource={trades}
          rowKey="trade_id"
          loading={loading}
          pagination={{
            current: Math.floor(filters.offset / filters.limit) + 1,
            pageSize: filters.limit,
            total: 1000, // Would need total count from API
            onChange: (page, pageSize) => {
              setFilters({
                ...filters,
                offset: (page - 1) * (pageSize || filters.limit),
                limit: pageSize || filters.limit
              });
            },
            showSizeChanger: true,
            showQuickJumper: true,
            showTotal: (total, range) => `${range[0]}-${range[1]} of ${total} trades`
          }}
          scroll={{ x: 1200 }}
          size="small"
        />
      </Card>

      {/* Trade Detail Modal */}
      <Modal
        title="Trade Details"
        open={detailModalVisible}
        onCancel={() => setDetailModalVisible(false)}
        footer={null}
        width={800}
      >
        {selectedTrade && (
          <Descriptions bordered column={2}>
            <Descriptions.Item label="Trade ID" span={2}>
              <Text code>{selectedTrade.trade_id}</Text>
            </Descriptions.Item>
            <Descriptions.Item label="Execution Time">
              {dayjs(selectedTrade.execution_time).format('YYYY-MM-DD HH:mm:ss')}
            </Descriptions.Item>
            <Descriptions.Item label="Account">
              {selectedTrade.account_id}
            </Descriptions.Item>
            <Descriptions.Item label="Symbol">
              <Text strong>{selectedTrade.symbol}</Text>
            </Descriptions.Item>
            <Descriptions.Item label="Venue">
              <Tag>{selectedTrade.venue}</Tag>
            </Descriptions.Item>
            <Descriptions.Item label="Side">
              <Tag color={selectedTrade.side === 'BUY' ? 'green' : 'red'}>
                {selectedTrade.side}
              </Tag>
            </Descriptions.Item>
            <Descriptions.Item label="Quantity">
              {parseFloat(selectedTrade.quantity).toLocaleString()}
            </Descriptions.Item>
            <Descriptions.Item label="Price">
              ${parseFloat(selectedTrade.price).toFixed(4)}
            </Descriptions.Item>
            <Descriptions.Item label="Value">
              ${(parseFloat(selectedTrade.quantity) * parseFloat(selectedTrade.price)).toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}
            </Descriptions.Item>
            <Descriptions.Item label="Commission">
              ${parseFloat(selectedTrade.commission).toFixed(2)}
            </Descriptions.Item>
            <Descriptions.Item label="Order ID">
              {selectedTrade.order_id ? <Text code>{selectedTrade.order_id}</Text> : '-'}
            </Descriptions.Item>
            <Descriptions.Item label="Execution ID">
              {selectedTrade.execution_id ? <Text code>{selectedTrade.execution_id}</Text> : '-'}
            </Descriptions.Item>
            <Descriptions.Item label="Strategy" span={2}>
              {selectedTrade.strategy ? <Tag color="blue">{selectedTrade.strategy}</Tag> : 'Not specified'}
            </Descriptions.Item>
            {selectedTrade.notes && (
              <Descriptions.Item label="Notes" span={2}>
                {selectedTrade.notes}
              </Descriptions.Item>
            )}
          </Descriptions>
        )}
      </Modal>
    </div>
  );
};