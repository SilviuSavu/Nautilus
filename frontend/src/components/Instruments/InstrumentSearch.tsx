import React, { useState, useEffect, useRef } from 'react'
import { Input, List, Card, Tag, Space, Badge, Tooltip, Empty, Button, Row, Col, Drawer, Dropdown } from 'antd'
import { SearchOutlined, HeartOutlined, HeartFilled, ClockCircleOutlined, FilterOutlined, SettingOutlined, PlusOutlined, FolderOutlined, QuestionCircleOutlined } from '@ant-design/icons'
import { useInstrumentStore } from './hooks/useInstrumentStore'
import { Instrument, InstrumentSearchResult } from './types/instrumentTypes'
import { VenueStatusIndicator } from './VenueStatusIndicator'
import { AssetClassFilter } from './AssetClassFilter'
import { VenueFilter } from './VenueFilter'
import { SearchResultsRanking } from './SearchResultsRanking'
import { VirtualizedInstrumentList } from './VirtualizedInstrumentList'
import { useKeyboardShortcuts } from './hooks/useKeyboardShortcuts'
import { KeyboardShortcutsHelp } from './KeyboardShortcutsHelp'
import { SearchHistory } from './SearchHistory'
import { searchHistoryService } from './services/searchHistoryService'

const { Search } = Input

interface InstrumentSearchProps {
  onInstrumentSelect?: (instrument: Instrument) => void
  placeholder?: string
  showFavorites?: boolean
  showRecentSelections?: boolean
  maxResults?: number
  className?: string
  showFilters?: boolean
  showAdvancedSettings?: boolean
  useVirtualScrolling?: boolean
}

export const InstrumentSearch: React.FC<InstrumentSearchProps> = ({
  onInstrumentSelect,
  placeholder = "Search instruments across all venues...",
  showFavorites = true,
  showRecentSelections = true,
  maxResults = 50,
  className,
  showFilters = true,
  showAdvancedSettings = true,
  useVirtualScrolling = true
}) => {
  const [searchQuery, setSearchQuery] = useState('')
  const [isSearching, setIsSearching] = useState(false)
  const [searchResults, setSearchResults] = useState<InstrumentSearchResult[]>([])
  const [showResults, setShowResults] = useState(false)
  const [showFilterDrawer, setShowFilterDrawer] = useState(false)
  const [showSettingsDrawer, setShowSettingsDrawer] = useState(false)
  const [showKeyboardHelp, setShowKeyboardHelp] = useState(false)
  const [selectedIndex, setSelectedIndex] = useState(-1)
  const [isSearchFocused, setIsSearchFocused] = useState(false)
  const searchRef = useRef<HTMLDivElement>(null)
  const searchInputRef = useRef<any>(null)

  const {
    searchInstruments,
    addToFavorites,
    removeFromFavorites,
    favorites,
    recentSelections,
    addToRecentSelections,
    isLoading,
    venueStatus,
    searchFilters,
    updateVenueStatus,
    lastSearchTime,
    watchlists,
    addToWatchlist
  } = useInstrumentStore()

  const handleInstrumentClick = (instrument: Instrument) => {
    addToRecentSelections(instrument)
    onInstrumentSelect?.(instrument)
    setShowResults(false)
    setSearchQuery('')
  }

  const handleFavoriteToggle = (instrument: Instrument, event: React.MouseEvent) => {
    event.stopPropagation()
    const isFavorite = favorites.some(fav => fav.id === instrument.id)
    if (isFavorite) {
      removeFromFavorites(instrument.id)
    } else {
      addToFavorites(instrument)
    }
  }

  const handleAddToWatchlist = (instrument: Instrument, watchlistId: string, event: React.MouseEvent) => {
    event.stopPropagation()
    addToWatchlist(watchlistId, instrument)
    console.log(`Added ${instrument.symbol} to watchlist ${watchlistId}`)
  }

  const handleSearchHistorySelect = (query: string, filters?: any) => {
    setSearchQuery(query)
    // Apply filters if they exist
    // Note: This would require updating the searchFilters in the store
    // For now, we'll just set the query and let the user apply filters manually
  }

  // Initialize keyboard shortcuts
  useKeyboardShortcuts({
    searchResults: searchResults.map(r => r.instrument),
    selectedIndex,
    setSelectedIndex,
    onInstrumentSelect: handleInstrumentClick,
    onToggleFilters: () => setShowFilterDrawer(!showFilterDrawer),
    onToggleSettings: () => setShowSettingsDrawer(!showSettingsDrawer),
    onClearSearch: () => {
      setSearchQuery('')
      setSearchResults([])
      setShowResults(false)
      setSelectedIndex(-1)
    },
    onFocusSearch: () => {
      searchInputRef.current?.focus()
    },
    searchInputRef,
    isSearchFocused,
    maxResults
  })

  // Handle search with debouncing
  useEffect(() => {
    if (!searchQuery.trim()) {
      setSearchResults([])
      setShowResults(false)
      return
    }

    const timeoutId = setTimeout(async () => {
      setIsSearching(true)
      try {
        const results = await searchInstruments(searchQuery, maxResults)
        setSearchResults(results)
        setShowResults(true)
        setSelectedIndex(-1) // Reset selection when results change
        
        // Add to search history
        searchHistoryService.addToHistory(searchQuery, results.length, {
          assetClasses: searchFilters.assetClasses,
          venues: searchFilters.venues,
          currencies: searchFilters.currencies
        })
      } catch (error) {
        console.error('Search failed:', error)
        setSearchResults([])
        setShowResults(true) // Show empty results with error message
      } finally {
        setIsSearching(false)
      }
    }, 300) // 300ms debounce

    return () => clearTimeout(timeoutId)
  }, [searchQuery, maxResults, searchInstruments])

  // Handle clicks outside to close results
  useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      if (searchRef.current && !searchRef.current.contains(event.target as Node)) {
        setShowResults(false)
      }
    }

    document.addEventListener('mousedown', handleClickOutside)
    return () => document.removeEventListener('mousedown', handleClickOutside)
  }, [])

  const getAssetClassColor = (assetClass: string) => {
    const colorMap: Record<string, string> = {
      'STK': 'blue',
      'CASH': 'green',
      'FUT': 'orange',
      'IND': 'purple',
      'OPT': 'red',
      'BOND': 'cyan',
      'CRYPTO': 'gold',
      'COMMODITY': 'magenta'
    }
    return colorMap[assetClass] || 'default'
  }

  const renderInstrumentItem = (instrument: Instrument, index?: number) => {
    const isFavorite = favorites.some(fav => fav.id === instrument.id)
    const venueConnectionStatus = venueStatus[instrument.venue]
    const isSelected = index !== undefined && index === selectedIndex

    return (
      <List.Item
        key={`instrument-${instrument.id}`}
        onClick={() => handleInstrumentClick(instrument)}
        style={{ 
          cursor: 'pointer', 
          padding: '12px 16px',
          backgroundColor: isSelected ? '#e6f3ff' : undefined,
          borderLeft: isSelected ? '3px solid #1890ff' : '3px solid transparent'
        }}
        className={`hover:bg-gray-50 ${isSelected ? 'bg-blue-50' : ''}`}
        actions={[
          <div style={{ fontSize: '11px', color: '#999' }}>
            Score: {searchResults.find(r => r.instrument.id === instrument.id)?.relevanceScore?.toFixed(1) || 'N/A'}
          </div>
        ]}
      >
        <div className="w-full">
          <div className="flex justify-between items-start">
            <div className="flex-1">
              <Space size="small">
                <Tag color={getAssetClassColor(instrument.assetClass)}>
                  {instrument.assetClass}
                </Tag>
                <strong>{instrument.symbol}</strong>
                <span>{instrument.venue}</span>
                <VenueStatusIndicator 
                  venue={instrument.venue}
                  status={venueConnectionStatus}
                  showName={false}
                  size="small"
                />
              </Space>
              <div className="mt-2">
                <div className="text-sm text-gray-600">
                  {instrument.name}
                </div>
                <div className="text-xs text-gray-500 mt-1">
                  <Space size="small">
                    <span>{instrument.venue}</span>
                    <span>•</span>
                    <span>{instrument.currency}</span>
                    {instrument.sessionInfo && (
                      <>
                        <span>•</span>
                        <span>
                          {instrument.sessionInfo.isOpen ? (
                            <Badge status="success" text="Open" />
                          ) : (
                            <Badge status="default" text="Closed" />
                          )}
                        </span>
                      </>
                    )}
                  </Space>
                </div>
              </div>
            </div>
            <div className="flex items-center space-x-2">
              <Tooltip title={isFavorite ? "Remove from favorites" : "Add to favorites"}>
                <Button
                  type="text"
                  size="small"
                  icon={isFavorite ? <HeartFilled /> : <HeartOutlined />}
                  onClick={(e) => handleFavoriteToggle(instrument, e)}
                  className={isFavorite ? "text-red-500" : "text-gray-400"}
                />
              </Tooltip>
              <Dropdown
                menu={{
                  items: watchlists.map(watchlist => ({
                    key: watchlist.id,
                    label: (
                      <Space>
                        <FolderOutlined />
                        {watchlist.name}
                        <Badge count={watchlist.items.length} size="small" />
                      </Space>
                    ),
                    onClick: ({ domEvent }) => handleAddToWatchlist(instrument, watchlist.id, domEvent as React.MouseEvent<Element>)
                  }))
                }}
                trigger={['click']}
              >
                <Button
                  type="text"
                  size="small"
                  icon={<PlusOutlined />}
                  onClick={(e) => e.stopPropagation()}
                  className="text-gray-400"
                />
              </Dropdown>
            </div>
          </div>
        </div>
      </List.Item>
    )
  }

  const renderSearchResults = () => {
    if (!showResults) return null

    return (
      <Card
        className="absolute top-full left-0 right-0 z-50 mt-1 max-h-96 overflow-hidden shadow-lg"
        bodyStyle={{ padding: 0 }}
      >
        {isSearching ? (
          <div className="p-4 text-center text-gray-500">
            Searching across all venues...
          </div>
        ) : searchResults.length > 0 ? (
          useVirtualScrolling && searchResults.length > 20 ? (
            <VirtualizedInstrumentList
              searchResults={searchResults}
              height={320}
              itemHeight={80}
              favorites={favorites}
              venueStatus={venueStatus}
              watchlists={watchlists}
              onInstrumentClick={handleInstrumentClick}
              onFavoriteToggle={handleFavoriteToggle}
              onAddToWatchlist={handleAddToWatchlist}
            />
          ) : (
            <List
              dataSource={searchResults.map(result => result.instrument)}
              renderItem={(instrument, index) => renderInstrumentItem(instrument, index)}
              className="max-h-80 overflow-y-auto"
            />
          )
        ) : searchQuery.trim() ? (
          <div className="p-4">
            <Empty
              image={Empty.PRESENTED_IMAGE_SIMPLE}
              description={
                <div>
                  <div data-testid="no-instruments-found">No instruments found</div>
                  <div className="text-xs text-gray-400 mt-1">
                    Try searching by symbol, name, or venue
                  </div>
                  {searchFilters.assetClasses.length > 0 || searchFilters.venues.length > 0 ? (
                    <div className="text-xs text-blue-400 mt-1">
                      Active filters may be limiting results
                    </div>
                  ) : null}
                  <div className="text-xs text-gray-400 mt-1">
                    Search time: {lastSearchTime.toFixed(0)}ms
                  </div>
                </div>
              }
            />
          </div>
        ) : null}
      </Card>
    )
  }

  const renderFavorites = () => {
    if (!showFavorites || favorites.length === 0) return null

    return (
      <Card
        title={
          <Space>
            <HeartFilled className="text-red-500" />
            Favorite Instruments
          </Space>
        }
        size="small"
        className="mb-4"
      >
        <List
          dataSource={favorites.slice(0, 5)}
          renderItem={renderInstrumentItem}
          size="small"
        />
        {favorites.length > 5 && (
          <div className="text-center mt-2">
            <Button type="link" size="small">
              View all {favorites.length} favorites
            </Button>
          </div>
        )}
      </Card>
    )
  }

  const renderRecentSelections = () => {
    if (!showRecentSelections || recentSelections.length === 0) return null

    return (
      <Card
        title={
          <Space>
            <ClockCircleOutlined />
            Recent Selections
          </Space>
        }
        size="small"
        className="mb-4"
      >
        <List
          dataSource={recentSelections.slice(0, 3)}
          renderItem={renderInstrumentItem}
          size="small"
        />
      </Card>
    )
  }

  return (
    <div className={className}>
      <Row gutter={[16, 16]}>
        <Col xs={24} lg={showFilters ? 16 : 24}>
          <div ref={searchRef} className="relative">
            <Space.Compact style={{ width: '100%' }}>
              <Search
                ref={searchInputRef}
                placeholder={placeholder}
                prefix={<SearchOutlined />}
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                onFocus={() => {
                  setIsSearchFocused(true)
                  if (searchQuery.trim() && searchResults.length > 0) {
                    setShowResults(true)
                  }
                }}
                onBlur={() => {
                  // Delay setting to false to allow keyboard events to process
                  setTimeout(() => setIsSearchFocused(false), 100)
                }}
                loading={isSearching || isLoading}
                size="large"
                allowClear
                style={{ flex: 1 }}
              />
              {showFilters && (
                <Button
                  icon={<FilterOutlined />}
                  size="large"
                  onClick={() => setShowFilterDrawer(true)}
                  type={searchFilters.assetClasses.length > 0 || searchFilters.venues.length > 0 ? 'primary' : 'default'}
                >
                  Filters
                </Button>
              )}
              {showAdvancedSettings && (
                <Button
                  icon={<SettingOutlined />}
                  size="large"
                  onClick={() => setShowSettingsDrawer(true)}
                >
                  Ranking
                </Button>
              )}
              <Button
                icon={<QuestionCircleOutlined />}
                size="large"
                onClick={() => setShowKeyboardHelp(true)}
                title="Keyboard shortcuts"
              />
            </Space.Compact>
            {renderSearchResults()}
          </div>
          
          {!searchQuery && (
            <div className="mt-4">
              <SearchHistory 
                onSearchSelect={handleSearchHistorySelect}
                maxItems={5}
                showFilters={true}
              />
              {renderFavorites()}
              {renderRecentSelections()}
            </div>
          )}
        </Col>
        
        {showFilters && (
          <Col xs={24} lg={8}>
            <Space direction="vertical" style={{ width: '100%' }} size="middle">
              <AssetClassFilter showCounts compactMode={false} />
              <VenueFilter showCounts compactMode={false} />
            </Space>
          </Col>
        )}
      </Row>

      {/* Filter Drawer for Mobile */}
      <Drawer
        title="Search Filters"
        placement="bottom"
        onClose={() => setShowFilterDrawer(false)}
        open={showFilterDrawer}
        height={"70%"}
      >
        <Space direction="vertical" style={{ width: '100%' }} size="middle">
          <AssetClassFilter showCounts compactMode={false} />
          <VenueFilter showCounts compactMode={false} />
        </Space>
      </Drawer>

      {/* Advanced Settings Drawer */}
      <Drawer
        title="Search Ranking Settings"
        placement="right"
        onClose={() => setShowSettingsDrawer(false)}
        open={showSettingsDrawer}
        width={400}
      >
        <SearchResultsRanking />
      </Drawer>

      {/* Keyboard Shortcuts Help */}
      <KeyboardShortcutsHelp
        visible={showKeyboardHelp}
        onClose={() => setShowKeyboardHelp(false)}
      />
    </div>
  )
}