/**
 * Real-time MessageBus data viewer component
 */

import React, { useState, useEffect } from 'react'
import { Card, Table, Tabs, Typography, Space, Tag, Button, Input, Select } from 'antd'
import { MessageOutlined, ClearOutlined } from '@ant-design/icons'
import { useMessageBus } from '../hooks/useMessageBus'
import { MessageBusMessage } from '../types/messagebus'

const { Text, Title } = Typography
const { TabPane } = Tabs
const { Search } = Input
const { Option } = Select

interface MessageBusViewerProps {
  maxDisplayMessages?: number;
  showFilters?: boolean;
}

const MessageBusViewer: React.FC<MessageBusViewerProps> = ({ 
  maxDisplayMessages = 50,
  showFilters = true 
}) => {
  const { messages, getStats, getMessagesByTopic, clearMessages } = useMessageBus()
  const [filteredMessages, setFilteredMessages] = useState<MessageBusMessage[]>([])
  const [selectedTopic, setSelectedTopic] = useState<string>('all')
  const [searchText, setSearchText] = useState<string>('')
  const [activeTab, setActiveTab] = useState<string>('messages')

  const stats = getStats()

  // Filter messages based on topic and search text
  useEffect(() => {
    let filtered = selectedTopic === 'all' ? messages : getMessagesByTopic(selectedTopic)
    
    if (searchText) {
      filtered = filtered.filter(msg => 
        msg.topic.toLowerCase().includes(searchText.toLowerCase()) ||
        msg.message_type.toLowerCase().includes(searchText.toLowerCase()) ||
        JSON.stringify(msg.payload).toLowerCase().includes(searchText.toLowerCase())
      )
    }

    // Limit to max display messages
    setFilteredMessages(filtered.slice(-maxDisplayMessages).reverse())
  }, [messages, selectedTopic, searchText, maxDisplayMessages, getMessagesByTopic])

  // Get unique topics for filter dropdown
  const uniqueTopics = Array.from(new Set(messages.map(msg => msg.topic))).sort()

  const messageColumns = [
    {
      title: 'Time',
      dataIndex: 'timestamp',
      key: 'timestamp',
      width: 120,
      render: (timestamp: number) => new Date(timestamp / 1000000).toLocaleTimeString()
    },
    {
      title: 'Topic',
      dataIndex: 'topic',
      key: 'topic',
      width: 200,
      render: (topic: string) => <Tag color="blue">{topic}</Tag>
    },
    {
      title: 'Type',
      dataIndex: 'message_type',
      key: 'message_type',
      width: 150,
      render: (type: string) => <Tag color="green">{type}</Tag>
    },
    {
      title: 'Payload',
      dataIndex: 'payload',
      key: 'payload',
      render: (payload: any) => (
        <div style={{ maxWidth: 300, maxHeight: 60, overflow: 'auto' }}>
          <Text code style={{ fontSize: '11px', whiteSpace: 'pre-wrap' }}>
            {JSON.stringify(payload, null, 1)}
          </Text>
        </div>
      )
    }
  ]

  const topicStatsColumns = [
    {
      title: 'Topic',
      dataIndex: 'topic',
      key: 'topic',
      render: (topic: string) => <Tag color="blue">{topic}</Tag>
    },
    {
      title: 'Message Count',
      dataIndex: 'count',
      key: 'count',
      sorter: (a: any, b: any) => a.count - b.count,
      defaultSortOrder: 'descend' as const
    },
    {
      title: 'Latest Message',
      dataIndex: 'latest',
      key: 'latest',
      render: (timestamp: number) => new Date(timestamp / 1000000).toLocaleString()
    }
  ]

  const topicStatsData = Object.entries(stats.topicCounts).map(([topic, count]) => {
    const topicMessages = getMessagesByTopic(topic)
    const latestMessage = topicMessages[topicMessages.length - 1]
    
    return {
      key: topic,
      topic,
      count,
      latest: latestMessage?.timestamp || 0
    }
  })

  return (
    <Card
      title={
        <Space>
          <MessageOutlined />
          <Title level={4} style={{ margin: 0 }}>MessageBus Data Viewer</Title>
          <Text type="secondary">({stats.totalMessages} total, {stats.bufferedMessages} buffered)</Text>
        </Space>
      }
      extra={
        <Button 
          icon={<ClearOutlined />} 
          onClick={clearMessages}
          size="small"
        >
          Clear All
        </Button>
      }
    >
      <Tabs activeKey={activeTab} onChange={setActiveTab}>
        <TabPane tab="Real-time Messages" key="messages">
          {showFilters && (
            <Space style={{ marginBottom: 16, width: '100%' }} wrap>
              <Search
                placeholder="Search messages..."
                value={searchText}
                onChange={(e) => setSearchText(e.target.value)}
                style={{ width: 250 }}
                allowClear
              />
              <Select
                value={selectedTopic}
                onChange={setSelectedTopic}
                style={{ width: 200 }}
                placeholder="Filter by topic"
              >
                <Option value="all">All Topics</Option>
                {uniqueTopics.map(topic => (
                  <Option key={topic} value={topic}>{topic}</Option>
                ))}
              </Select>
              <Text type="secondary">
                Showing {filteredMessages.length} of {stats.bufferedMessages} messages
              </Text>
            </Space>
          )}

          <Table
            dataSource={filteredMessages.map((msg, index) => ({
              ...msg,
              key: `${msg.timestamp}-${index}`
            }))}
            columns={messageColumns}
            pagination={{
              pageSize: 20,
              showSizeChanger: true,
              showQuickJumper: true,
              showTotal: (total, range) => `${range[0]}-${range[1]} of ${total} messages`
            }}
            size="small"
            scroll={{ x: 800, y: 400 }}
            loading={stats.totalMessages === 0}
          />
        </TabPane>

        <TabPane tab="Topic Statistics" key="stats">
          <Table
            dataSource={topicStatsData}
            columns={topicStatsColumns}
            pagination={false}
            size="small"
            loading={stats.uniqueTopics === 0}
          />
          
          <div style={{ marginTop: 16 }}>
            <Space wrap>
              <Text><strong>Total Messages:</strong> {stats.totalMessages}</Text>
              <Text><strong>Buffered Messages:</strong> {stats.bufferedMessages}</Text>
              <Text><strong>Unique Topics:</strong> {stats.uniqueTopics}</Text>
            </Space>
          </div>
        </TabPane>
      </Tabs>
    </Card>
  )
}

export default MessageBusViewer