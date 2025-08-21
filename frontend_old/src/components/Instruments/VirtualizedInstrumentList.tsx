import React, { useMemo, useCallback } from 'react'
import { FixedSizeList as List } from 'react-window'
import { List as AntList, Space, Tag, Badge, Tooltip, Button, Dropdown } from 'antd'
import { HeartOutlined, HeartFilled, PlusOutlined, FolderOutlined } from '@ant-design/icons'
import { Instrument, InstrumentSearchResult } from './types/instrumentTypes'
import { VenueStatusIndicator } from './VenueStatusIndicator'

interface VirtualizedInstrumentListProps {
  searchResults: InstrumentSearchResult[]
  height?: number
  itemHeight?: number
  favorites: Instrument[]
  venueStatus: Record<string, any>
  watchlists: Array<{ id: string; name: string; items: any[] }>
  onInstrumentClick: (instrument: Instrument) => void
  onFavoriteToggle: (instrument: Instrument, event: React.MouseEvent) => void
  onAddToWatchlist: (instrument: Instrument, watchlistId: string, event: React.MouseEvent) => void
}

interface ListItemProps {
  index: number
  style: React.CSSProperties
  data: {
    searchResults: InstrumentSearchResult[]
    favorites: Instrument[]
    venueStatus: Record<string, any>
    watchlists: Array<{ id: string; name: string; items: any[] }>
    onInstrumentClick: (instrument: Instrument) => void
    onFavoriteToggle: (instrument: Instrument, event: React.MouseEvent) => void
    onAddToWatchlist: (instrument: Instrument, watchlistId: string, event: React.MouseEvent) => void
  }
}

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

const ListItem: React.FC<ListItemProps> = ({ index, style, data }) => {
  const {
    searchResults,
    favorites,
    venueStatus,
    watchlists,
    onInstrumentClick,
    onFavoriteToggle,
    onAddToWatchlist
  } = data

  const result = searchResults[index]
  if (!result) return null

  const { instrument } = result
  const isFavorite = favorites.some(fav => fav.id === instrument.id)
  const venueConnectionStatus = venueStatus[instrument.venue]

  return (
    <div style={style}>
      <AntList.Item
        key={`virtual-instrument-${instrument.id}`}
        onClick={() => onInstrumentClick(instrument)}
        style={{ 
          cursor: 'pointer', 
          padding: '12px 16px',
          margin: 0,
          border: 'none',
          borderBottom: '1px solid #f0f0f0'
        }}
        className="hover:bg-gray-50"
        actions={[
          <div key="score" style={{ fontSize: '11px', color: '#999' }}>
            Score: {result.relevanceScore?.toFixed(1) || 'N/A'}
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
                  onClick={(e) => onFavoriteToggle(instrument, e)}
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
                    onClick: ({ domEvent }) => onAddToWatchlist(instrument, watchlist.id, domEvent)
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
      </AntList.Item>
    </div>
  )
}

export const VirtualizedInstrumentList: React.FC<VirtualizedInstrumentListProps> = ({
  searchResults,
  height = 320, // 20rem in pixels
  itemHeight = 80,
  favorites,
  venueStatus,
  watchlists,
  onInstrumentClick,
  onFavoriteToggle,
  onAddToWatchlist
}) => {
  const listData = useMemo(() => ({
    searchResults,
    favorites,
    venueStatus,
    watchlists,
    onInstrumentClick,
    onFavoriteToggle,
    onAddToWatchlist
  }), [searchResults, favorites, venueStatus, watchlists, onInstrumentClick, onFavoriteToggle, onAddToWatchlist])

  const itemData = useCallback((index: number) => ({
    index,
    data: listData
  }), [listData])

  if (searchResults.length === 0) {
    return null
  }

  return (
    <div className="virtualized-instrument-list">
      <List
        height={height}
        itemCount={searchResults.length}
        itemSize={itemHeight}
        itemData={listData}
        style={{ 
          border: 'none',
          outline: 'none'
        }}
      >
        {ListItem}
      </List>
    </div>
  )
}

export default VirtualizedInstrumentList