import React, { useState, useEffect } from 'react'
import { Card, List, Tag, Space, Button, Tooltip, Empty, Tabs } from 'antd'
import { HistoryOutlined, StarOutlined, DeleteOutlined, ClockCircleOutlined } from '@ant-design/icons'
import { searchHistoryService, SearchHistoryItem } from './services/searchHistoryService'

interface SearchHistoryProps {
  onSearchSelect: (query: string, filters?: SearchHistoryItem['filters']) => void
  maxItems?: number
  showFilters?: boolean
}

export const SearchHistory: React.FC<SearchHistoryProps> = ({
  onSearchSelect,
  maxItems = 10,
  showFilters = true
}) => {
  const [recentSearches, setRecentSearches] = useState<SearchHistoryItem[]>([])
  const [popularSearches, setPopularSearches] = useState<SearchHistoryItem[]>([])
  const [activeTab, setActiveTab] = useState('recent')

  useEffect(() => {
    loadSearchHistory()
  }, [maxItems])

  const loadSearchHistory = () => {
    setRecentSearches(searchHistoryService.getRecentSearches(maxItems))
    setPopularSearches(searchHistoryService.getPopularSearches(maxItems))
  }

  const handleSearchSelect = (item: SearchHistoryItem) => {
    onSearchSelect(item.query, item.filters)
  }

  const handleRemoveItem = (item: SearchHistoryItem, event: React.MouseEvent) => {
    event.stopPropagation()
    searchHistoryService.removeFromHistory(item.query, item.filters)
    loadSearchHistory()
  }

  const formatTimestamp = (timestamp: string) => {
    const date = new Date(timestamp)
    const now = new Date()
    const diffHours = (now.getTime() - date.getTime()) / (1000 * 60 * 60)

    if (diffHours < 1) {
      return 'Just now'
    } else if (diffHours < 24) {
      return `${Math.floor(diffHours)}h ago`
    } else if (diffHours < 168) { // 7 days
      return `${Math.floor(diffHours / 24)}d ago`
    } else {
      return date.toLocaleDateString()
    }
  }

  const renderFilters = (filters?: SearchHistoryItem['filters']) => {
    if (!filters || !showFilters) return null

    const filterTags = []
    
    if (filters.assetClasses && filters.assetClasses.length > 0) {
      filterTags.push(
        <Tag key="asset" color="blue">
          {filters.assetClasses.length === 1 ? 
            filters.assetClasses[0] : 
            `${filters.assetClasses.length} assets`}
        </Tag>
      )
    }
    
    if (filters.venues && filters.venues.length > 0) {
      filterTags.push(
        <Tag key="venue" color="green">
          {filters.venues.length === 1 ? 
            filters.venues[0] : 
            `${filters.venues.length} venues`}
        </Tag>
      )
    }
    
    if (filters.currencies && filters.currencies.length > 0) {
      filterTags.push(
        <Tag key="currency" color="orange">
          {filters.currencies.length === 1 ? 
            filters.currencies[0] : 
            `${filters.currencies.length} currencies`}
        </Tag>
      )
    }

    return filterTags.length > 0 ? (
      <div className="mt-1">
        <Space size="small">
          {filterTags}
        </Space>
      </div>
    ) : null
  }

  const renderSearchItem = (item: SearchHistoryItem) => (
    <List.Item
      key={`${item.query}-${item.timestamp}`}
      className="cursor-pointer hover:bg-gray-50 px-3 py-2"
      onClick={() => handleSearchSelect(item)}
      actions={[
        <Tooltip key="remove" title="Remove from history">
          <Button
            type="text"
                        icon={<DeleteOutlined />}
            onClick={(e) => handleRemoveItem(item, e)}
            className="text-gray-400 hover:text-red-500"
          />
        </Tooltip>
      ]}
    >
      <div className="w-full">
        <div className="flex justify-between items-start">
          <div className="flex-1">
            <div className="font-medium text-gray-800">
              {item.query}
            </div>
            {renderFilters(item.filters)}
          </div>
          <div className="text-xs text-gray-500 ml-2">
            <Space direction="vertical" size="small" align="end">
              <span>{formatTimestamp(item.timestamp)}</span>
              <span>{item.resultsCount} results</span>
            </Space>
          </div>
        </div>
      </div>
    </List.Item>
  )

  const tabItems = [
    {
      key: 'recent',
      label: (
        <Space>
          <ClockCircleOutlined />
          Recent ({recentSearches.length})
        </Space>
      ),
      children: recentSearches.length > 0 ? (
        <List
          dataSource={recentSearches}
          renderItem={renderSearchItem}
                    className="max-h-64 overflow-y-auto"
        />
      ) : (
        <Empty
          image={Empty.PRESENTED_IMAGE_SIMPLE}
          description="No recent searches"
          className="py-4"
        />
      )
    },
    {
      key: 'popular',
      label: (
        <Space>
          <StarOutlined />
          Popular ({popularSearches.length})
        </Space>
      ),
      children: popularSearches.length > 0 ? (
        <List
          dataSource={popularSearches}
          renderItem={renderSearchItem}
                    className="max-h-64 overflow-y-auto"
        />
      ) : (
        <Empty
          image={Empty.PRESENTED_IMAGE_SIMPLE}
          description="No popular searches yet"
          className="py-4"
        />
      )
    }
  ]

  if (recentSearches.length === 0 && popularSearches.length === 0) {
    return null
  }

  return (
    <Card
      title={
        <Space>
          <HistoryOutlined />
          Search History
        </Space>
      }
            className="mb-4"
      extra={
        <Button
          type="link" 
                    onClick={() => {
            searchHistoryService.clearHistory()
            loadSearchHistory()
          }}
          className="text-gray-400"
        >
          Clear All
        </Button>
      }
    >
      <Tabs
        activeKey={activeTab}
        onChange={setActiveTab}
                items={tabItems}
      />
    </Card>
  )
}

export default SearchHistory