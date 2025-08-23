/**
 * Message Protocol Viewer Component
 * Sprint 3: Advanced WebSocket Infrastructure
 * 
 * Debug and view WebSocket messages with protocol analysis, filtering, and inspection tools.
 * Provides comprehensive message debugging capabilities for development and monitoring.
 */

import React, { useState, useEffect, useCallback, useRef } from 'react';
import { 
  Card, 
  Table, 
  Typography, 
  Space, 
  Button, 
  Input, 
  Select, 
  Tag, 
  Drawer, 
  Descriptions,
  Badge,
  Switch,
  Tooltip,
  Alert,
  Row,
  Col,
  Progress,
  Empty,
  Divider
} from 'antd';
import { 
  BugOutlined,
  ClearOutlined,
  SearchOutlined,
  FilterOutlined,
  EyeOutlined,
  DownloadOutlined,
  PlayCircleOutlined,
  PauseCircleOutlined,
  SettingOutlined,
  ExclamationCircleOutlined,
  CheckCircleOutlined,
  CloseCircleOutlined
} from '@ant-design/icons';
import { useWebSocketManager } from '../../hooks/useWebSocketManager';

const { Title, Text, Paragraph } = Typography;
const { Option } = Select;
const { Search } = Input;

interface WebSocketMessage {
  id: string;
  timestamp: string;
  type: string;
  direction: 'incoming' | 'outgoing';
  data: any;
  size: number;
  latency?: number;
  error?: string;
  priority?: number;
  version?: string;
  correlationId?: string;
}

interface MessageProtocolViewerProps {
  className?: string;
  maxMessages?: number;
  autoScroll?: boolean;
  showRawData?: boolean;
  enableRecording?: boolean;
}

export const MessageProtocolViewer: React.FC<MessageProtocolViewerProps> = ({
  className,
  maxMessages = 1000,
  autoScroll = true,
  showRawData = false,
  enableRecording = true
}) => {
  const {
    connectionState,
    messageHistory,
    clearMessageHistory,
    getMessageStats,
    recordMessages,
    stopRecording,
    isRecording
  } = useWebSocketManager();

  const [messages, setMessages] = useState<WebSocketMessage[]>([]);
  const [filteredMessages, setFilteredMessages] = useState<WebSocketMessage[]>([]);
  const [selectedMessage, setSelectedMessage] = useState<WebSocketMessage | null>(null);
  const [drawerVisible, setDrawerVisible] = useState(false);
  const [isAutoScrollEnabled, setIsAutoScrollEnabled] = useState(autoScroll);
  const [searchTerm, setSearchTerm] = useState('');
  const [typeFilter, setTypeFilter] = useState<string>('all');
  const [directionFilter, setDirectionFilter] = useState<string>('all');
  const [showErrors, setShowErrors] = useState(false);

  const tableRef = useRef<HTMLDivElement>(null);

  // Message types for filtering
  const messageTypes = [
    'market_data',
    'trade_updates',
    'risk_alert',
    'engine_status',
    'system_health',
    'performance_update',
    'order_update',
    'position_update',
    'heartbeat',
    'subscription',
    'error',
    'command'
  ];

  // Update messages when message history changes
  useEffect(() => {
    if (messageHistory) {
      const formattedMessages = messageHistory.map((msg: any, index: number) => ({
        id: msg.messageId || `msg-${Date.now()}-${index}`,
        timestamp: msg.timestamp || new Date().toISOString(),
        type: msg.type || 'unknown',
        direction: msg.direction || 'incoming',
        data: msg.data || msg,
        size: JSON.stringify(msg).length,
        latency: msg.latency,
        error: msg.error,
        priority: msg.priority,
        version: msg.version || '2.0',
        correlationId: msg.correlationId
      }));

      setMessages(formattedMessages.slice(-maxMessages));
    }
  }, [messageHistory, maxMessages]);

  // Filter messages based on search and filters
  useEffect(() => {
    let filtered = messages;

    // Search filter
    if (searchTerm) {
      filtered = filtered.filter(msg => 
        msg.type.toLowerCase().includes(searchTerm.toLowerCase()) ||
        JSON.stringify(msg.data).toLowerCase().includes(searchTerm.toLowerCase()) ||
        (msg.correlationId && msg.correlationId.includes(searchTerm))
      );
    }

    // Type filter
    if (typeFilter !== 'all') {
      filtered = filtered.filter(msg => msg.type === typeFilter);
    }

    // Direction filter
    if (directionFilter !== 'all') {
      filtered = filtered.filter(msg => msg.direction === directionFilter);
    }

    // Error filter
    if (showErrors) {
      filtered = filtered.filter(msg => msg.error);
    }

    setFilteredMessages(filtered);
  }, [messages, searchTerm, typeFilter, directionFilter, showErrors]);

  // Auto scroll to bottom
  useEffect(() => {
    if (isAutoScrollEnabled && tableRef.current) {
      tableRef.current.scrollTop = tableRef.current.scrollHeight;
    }
  }, [filteredMessages, isAutoScrollEnabled]);

  // Handle message inspection
  const handleInspectMessage = (message: WebSocketMessage) => {
    setSelectedMessage(message);
    setDrawerVisible(true);
  };

  // Handle export messages
  const handleExportMessages = () => {
    const dataStr = JSON.stringify(filteredMessages, null, 2);
    const dataBlob = new Blob([dataStr], { type: 'application/json' });
    const url = URL.createObjectURL(dataBlob);
    const link = document.createElement('a');
    link.href = url;
    link.download = `websocket-messages-${new Date().toISOString().slice(0, 10)}.json`;
    link.click();
    URL.revokeObjectURL(url);
  };

  // Toggle recording
  const handleToggleRecording = () => {
    if (isRecording) {
      stopRecording();
    } else {
      recordMessages();
    }
  };

  // Get message type color
  const getMessageTypeColor = (type: string) => {
    const colorMap: Record<string, string> = {
      market_data: 'blue',
      trade_updates: 'green',
      risk_alert: 'red',
      engine_status: 'purple',
      system_health: 'orange',
      performance_update: 'cyan',
      order_update: 'magenta',
      position_update: 'gold',
      heartbeat: 'gray',
      subscription: 'lime',
      error: 'red',
      command: 'volcano'
    };
    return colorMap[type] || 'default';
  };

  // Get direction icon
  const getDirectionIcon = (direction: string) => {
    return direction === 'incoming' ? '↓' : '↑';
  };

  // Format message size
  const formatSize = (bytes: number): string => {
    if (bytes < 1024) return `${bytes}B`;
    if (bytes < 1048576) return `${(bytes / 1024).toFixed(1)}KB`;
    return `${(bytes / 1048576).toFixed(1)}MB`;
  };

  // Format latency
  const formatLatency = (latency?: number): string => {
    if (!latency) return 'N/A';
    return latency < 1000 ? `${latency.toFixed(0)}ms` : `${(latency / 1000).toFixed(1)}s`;
  };

  // Get message stats
  const messageStats = getMessageStats ? getMessageStats() : {
    totalMessages: messages.length,
    errorCount: messages.filter(m => m.error).length,
    averageLatency: messages.reduce((acc, m) => acc + (m.latency || 0), 0) / messages.length || 0,
    messageTypes: {}
  };

  // Table columns
  const columns = [
    {
      title: 'Time',
      dataIndex: 'timestamp',
      key: 'timestamp',
      width: 120,
      render: (timestamp: string) => (
        <Text style={{ fontSize: '12px' }}>
          {new Date(timestamp).toLocaleTimeString()}
        </Text>
      ),
    },
    {
      title: 'Type',
      dataIndex: 'type',
      key: 'type',
      width: 140,
      render: (type: string) => (
        <Tag color={getMessageTypeColor(type)} style={{ fontSize: '11px' }}>
          {type}
        </Tag>
      ),
    },
    {
      title: 'Dir',
      dataIndex: 'direction',
      key: 'direction',
      width: 50,
      render: (direction: string) => (
        <Tooltip title={direction}>
          <Text style={{ fontSize: '14px' }}>
            {getDirectionIcon(direction)}
          </Text>
        </Tooltip>
      ),
    },
    {
      title: 'Size',
      dataIndex: 'size',
      key: 'size',
      width: 70,
      render: (size: number) => (
        <Text style={{ fontSize: '11px' }}>
          {formatSize(size)}
        </Text>
      ),
    },
    {
      title: 'Latency',
      dataIndex: 'latency',
      key: 'latency',
      width: 80,
      render: (latency: number) => (
        <Text style={{ fontSize: '11px' }}>
          {formatLatency(latency)}
        </Text>
      ),
    },
    {
      title: 'Data Preview',
      dataIndex: 'data',
      key: 'data',
      ellipsis: true,
      render: (data: any, record: WebSocketMessage) => (
        <Space>
          <Text style={{ fontSize: '11px' }} ellipsis>
            {typeof data === 'object' ? JSON.stringify(data).substring(0, 100) + '...' : String(data)}
          </Text>
          {record.error && (
            <Tooltip title={record.error}>
              <ExclamationCircleOutlined style={{ color: '#ff4d4f' }} />
            </Tooltip>
          )}
        </Space>
      ),
    },
    {
      title: 'Actions',
      key: 'actions',
      width: 80,
      render: (_, record: WebSocketMessage) => (
        <Button
          type="text"
          size="small"
          icon={<EyeOutlined />}
          onClick={() => handleInspectMessage(record)}
        />
      ),
    },
  ];

  return (
    <div className={className}>
      <Card
        title={
          <Space>
            <BugOutlined />
            <span>Message Protocol Viewer</span>
            <Badge 
              count={filteredMessages.length} 
              overflowCount={999} 
              style={{ backgroundColor: connectionState === 'connected' ? '#52c41a' : '#d9d9d9' }}
            />
          </Space>
        }
        size="small"
        extra={
          <Space>
            {enableRecording && (
              <Button
                type={isRecording ? "default" : "primary"}
                size="small"
                icon={isRecording ? <PauseCircleOutlined /> : <PlayCircleOutlined />}
                onClick={handleToggleRecording}
              >
                {isRecording ? 'Stop' : 'Record'}
              </Button>
            )}
            
            <Button
              size="small"
              icon={<DownloadOutlined />}
              onClick={handleExportMessages}
              disabled={filteredMessages.length === 0}
            >
              Export
            </Button>
            
            <Button
              size="small"
              icon={<ClearOutlined />}
              onClick={clearMessageHistory}
            >
              Clear
            </Button>
          </Space>
        }
      >
        {/* Statistics Overview */}
        <Row gutter={[16, 8]} style={{ marginBottom: 16 }}>
          <Col span={6}>
            <Space>
              <CheckCircleOutlined style={{ color: '#52c41a' }} />
              <Text strong>Total: {messageStats.totalMessages.toLocaleString()}</Text>
            </Space>
          </Col>
          
          <Col span={6}>
            <Space>
              {messageStats.errorCount > 0 ? (
                <CloseCircleOutlined style={{ color: '#ff4d4f' }} />
              ) : (
                <CheckCircleOutlined style={{ color: '#52c41a' }} />
              )}
              <Text strong>Errors: {messageStats.errorCount}</Text>
            </Space>
          </Col>
          
          <Col span={6}>
            <Text strong>Avg Latency: {formatLatency(messageStats.averageLatency)}</Text>
          </Col>
          
          <Col span={6}>
            <Text strong>Rate: {messages.length > 0 ? (messages.length / 60).toFixed(1) : '0'}/sec</Text>
          </Col>
        </Row>

        {/* Controls */}
        <Row gutter={[8, 8]} style={{ marginBottom: 16 }}>
          <Col span={8}>
            <Search
              placeholder="Search messages..."
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
              size="small"
              allowClear
            />
          </Col>
          
          <Col span={4}>
            <Select
              value={typeFilter}
              onChange={setTypeFilter}
              size="small"
              style={{ width: '100%' }}
              placeholder="Message Type"
            >
              <Option value="all">All Types</Option>
              {messageTypes.map(type => (
                <Option key={type} value={type}>
                  {type}
                </Option>
              ))}
            </Select>
          </Col>
          
          <Col span={4}>
            <Select
              value={directionFilter}
              onChange={setDirectionFilter}
              size="small"
              style={{ width: '100%' }}
            >
              <Option value="all">All Directions</Option>
              <Option value="incoming">Incoming</Option>
              <Option value="outgoing">Outgoing</Option>
            </Select>
          </Col>
          
          <Col span={3}>
            <Space>
              <Switch 
                size="small"
                checked={showErrors}
                onChange={setShowErrors}
              />
              <Text style={{ fontSize: '12px' }}>Errors Only</Text>
            </Space>
          </Col>
          
          <Col span={5}>
            <Space>
              <Switch 
                size="small"
                checked={isAutoScrollEnabled}
                onChange={setIsAutoScrollEnabled}
              />
              <Text style={{ fontSize: '12px' }}>Auto Scroll</Text>
            </Space>
          </Col>
        </Row>

        {/* Connection Status Warning */}
        {connectionState !== 'connected' && (
          <Alert
            message={`WebSocket ${connectionState}`}
            description="Message capture is limited when not connected"
            type="warning"
            showIcon
            style={{ marginBottom: 16 }}
          />
        )}

        {/* Messages Table */}
        <div 
          ref={tableRef} 
          style={{ 
            height: '400px', 
            overflowY: 'auto',
            border: '1px solid #f0f0f0',
            borderRadius: '6px'
          }}
        >
          {filteredMessages.length === 0 ? (
            <Empty 
              description="No messages captured"
              style={{ padding: '40px 20px' }}
            />
          ) : (
            <Table
              dataSource={filteredMessages}
              columns={columns}
              rowKey="id"
              size="small"
              pagination={false}
              scroll={{ y: 350 }}
              showHeader={false}
              rowClassName={(record) => record.error ? 'error-row' : ''}
            />
          )}
        </div>

        {/* Message Detail Drawer */}
        <Drawer
          title="Message Details"
          placement="right"
          onClose={() => setDrawerVisible(false)}
          open={drawerVisible}
          width={600}
          extra={
            selectedMessage && (
              <Tag color={getMessageTypeColor(selectedMessage.type)}>
                {selectedMessage.type}
              </Tag>
            )
          }
        >
          {selectedMessage && (
            <Space direction="vertical" size="large" style={{ width: '100%' }}>
              <Descriptions title="Message Information" column={2} size="small">
                <Descriptions.Item label="ID">{selectedMessage.id}</Descriptions.Item>
                <Descriptions.Item label="Type">{selectedMessage.type}</Descriptions.Item>
                <Descriptions.Item label="Direction">{selectedMessage.direction}</Descriptions.Item>
                <Descriptions.Item label="Timestamp">
                  {new Date(selectedMessage.timestamp).toLocaleString()}
                </Descriptions.Item>
                <Descriptions.Item label="Size">{formatSize(selectedMessage.size)}</Descriptions.Item>
                <Descriptions.Item label="Latency">{formatLatency(selectedMessage.latency)}</Descriptions.Item>
                <Descriptions.Item label="Priority">{selectedMessage.priority || 'Normal'}</Descriptions.Item>
                <Descriptions.Item label="Version">{selectedMessage.version || '2.0'}</Descriptions.Item>
                {selectedMessage.correlationId && (
                  <Descriptions.Item label="Correlation ID" span={2}>
                    <Text copyable>{selectedMessage.correlationId}</Text>
                  </Descriptions.Item>
                )}
              </Descriptions>

              {selectedMessage.error && (
                <Alert
                  message="Message Error"
                  description={selectedMessage.error}
                  type="error"
                  showIcon
                />
              )}

              <Divider />

              <Title level={5}>Message Data</Title>
              <div style={{ 
                backgroundColor: '#f6f6f6', 
                padding: '12px', 
                borderRadius: '4px',
                maxHeight: '300px',
                overflow: 'auto'
              }}>
                <pre style={{ 
                  margin: 0, 
                  fontSize: '12px',
                  lineHeight: '1.4'
                }}>
                  {JSON.stringify(selectedMessage.data, null, 2)}
                </pre>
              </div>

              {showRawData && (
                <>
                  <Divider />
                  <Title level={5}>Raw Message</Title>
                  <div style={{ 
                    backgroundColor: '#f6f6f6', 
                    padding: '12px', 
                    borderRadius: '4px',
                    maxHeight: '200px',
                    overflow: 'auto'
                  }}>
                    <pre style={{ 
                      margin: 0, 
                      fontSize: '12px',
                      lineHeight: '1.4'
                    }}>
                      {JSON.stringify(selectedMessage, null, 2)}
                    </pre>
                  </div>
                </>
              )}
            </Space>
          )}
        </Drawer>
      </Card>

      <style>{`
        .error-row {
          background-color: #fff2f0 !important;
        }
        .error-row:hover {
          background-color: #ffebe8 !important;
        }
      `}</style>
    </div>
  );
};

export default MessageProtocolViewer;