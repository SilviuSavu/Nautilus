import React from 'react'
import { Card, Select, Space, Switch, Tooltip, Slider } from 'antd'
import { SortAscendingOutlined } from '@ant-design/icons'
import { useInstrumentStore } from './hooks/useInstrumentStore'

const { Option } = Select

interface SearchResultsRankingProps {
  className?: string
}

export const SearchResultsRanking: React.FC<SearchResultsRankingProps> = ({ className }) => {
  const { searchFilters, updateSearchFilters } = useInstrumentStore()

  const handleSortChange = (sortBy: string) => {
    updateSearchFilters({ 
      sortBy: sortBy as 'relevance' | 'symbol' | 'name' | 'volume' | 'venue'
    })
  }

  const handleSortOrderChange = (ascending: boolean) => {
    updateSearchFilters({ sortOrder: ascending ? 'asc' : 'desc' })
  }

  const handleBoostFactorChange = (field: string, value: number) => {
    const currentBoosts = searchFilters.boostFactors || {}
    updateSearchFilters({
      boostFactors: {
        ...currentBoosts,
        [field]: value
      }
    })
  }

  const boostFactors = searchFilters.boostFactors || {
    exactSymbolMatch: 10,
    symbolPrefix: 5,
    nameMatch: 3,
    venueMatch: 1,
    assetClassBoost: 2
  }

  return (
    <Card
      title={
        <Space>
          <SortAscendingOutlined />
          Search Ranking
        </Space>
      }
      size="small"
      className={className}
    >
      <Space direction="vertical" style={{ width: '100%' }} size="small">
        {/* Sort By */}
        <div>
          <div style={{ marginBottom: '8px', fontSize: '12px', fontWeight: 500 }}>
            Sort Results By:
          </div>
          <Select
            value={searchFilters.sortBy || 'relevance'}
            onChange={handleSortChange}
            style={{ width: '100%' }}
            size="small"
          >
            <Option value="relevance">Relevance Score</Option>
            <Option value="symbol">Symbol (A-Z)</Option>
            <Option value="name">Company Name</Option>
            <Option value="venue">Venue</Option>
            <Option value="volume">Trading Volume</Option>
          </Select>
        </div>

        {/* Sort Order */}
        <div>
          <Space style={{ width: '100%', justifyContent: 'space-between' }}>
            <span style={{ fontSize: '12px', fontWeight: 500 }}>
              Sort Order:
            </span>
            <Switch
              checked={searchFilters.sortOrder !== 'desc'}
              onChange={handleSortOrderChange}
              checkedChildren="A-Z"
              unCheckedChildren="Z-A"
              size="small"
            />
          </Space>
        </div>

        {/* Boost Factors */}
        <div style={{ marginTop: '12px' }}>
          <div style={{ marginBottom: '8px', fontSize: '12px', fontWeight: 500 }}>
            Relevance Boost Factors:
          </div>
          
          <Space direction="vertical" style={{ width: '100%' }} size="small">
            <div>
              <div style={{ fontSize: '11px', color: '#666', marginBottom: '4px' }}>
                Exact Symbol Match: {boostFactors.exactSymbolMatch}x
              </div>
              <Slider
                min={1}
                max={20}
                value={boostFactors.exactSymbolMatch}
                onChange={(value) => handleBoostFactorChange('exactSymbolMatch', value)}
                size="small"
              />
            </div>

            <div>
              <div style={{ fontSize: '11px', color: '#666', marginBottom: '4px' }}>
                Symbol Prefix: {boostFactors.symbolPrefix}x
              </div>
              <Slider
                min={1}
                max={10}
                value={boostFactors.symbolPrefix}
                onChange={(value) => handleBoostFactorChange('symbolPrefix', value)}
                size="small"
              />
            </div>

            <div>
              <div style={{ fontSize: '11px', color: '#666', marginBottom: '4px' }}>
                Name Match: {boostFactors.nameMatch}x
              </div>
              <Slider
                min={1}
                max={10}
                value={boostFactors.nameMatch}
                onChange={(value) => handleBoostFactorChange('nameMatch', value)}
                size="small"
              />
            </div>

            <div>
              <div style={{ fontSize: '11px', color: '#666', marginBottom: '4px' }}>
                Asset Class Boost: {boostFactors.assetClassBoost}x
              </div>
              <Slider
                min={1}
                max={5}
                value={boostFactors.assetClassBoost}
                onChange={(value) => handleBoostFactorChange('assetClassBoost', value)}
                size="small"
              />
            </div>
          </Space>
        </div>

        {/* Additional Options */}
        <div style={{ marginTop: '12px' }}>
          <Space direction="vertical" style={{ width: '100%' }}>
            <Tooltip title="Prioritize instruments from connected venues">
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <span style={{ fontSize: '12px' }}>Boost Connected Venues</span>
                <Switch
                  checked={searchFilters.boostConnectedVenues || false}
                  onChange={(checked) => updateSearchFilters({ boostConnectedVenues: checked })}
                  size="small"
                />
              </div>
            </Tooltip>

            <Tooltip title="Show instruments from favorite list first">
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <span style={{ fontSize: '12px' }}>Boost Favorites</span>
                <Switch
                  checked={searchFilters.boostFavorites || false}
                  onChange={(checked) => updateSearchFilters({ boostFavorites: checked })}
                  size="small"
                />
              </div>
            </Tooltip>

            <Tooltip title="Include fuzzy matching for typos">
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <span style={{ fontSize: '12px' }}>Fuzzy Matching</span>
                <Switch
                  checked={searchFilters.enableFuzzySearch !== false}
                  onChange={(checked) => updateSearchFilters({ enableFuzzySearch: checked })}
                  size="small"
                />
              </div>
            </Tooltip>
          </Space>
        </div>
      </Space>
    </Card>
  )
}