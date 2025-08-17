import React from 'react'
import { Card, Checkbox, Space, Tag, Tooltip, Divider, Button } from 'antd'
import { FilterOutlined, ClearOutlined } from '@ant-design/icons'
import { AssetClassFilter as AssetClassFilterType } from './types/instrumentTypes'
import { useInstrumentStore } from './hooks/useInstrumentStore'

interface AssetClassFilterProps {
  className?: string
  showCounts?: boolean
  compactMode?: boolean
}

const ASSET_CLASS_CONFIG = [
  {
    code: 'STK',
    name: 'Stocks',
    description: 'Common stocks and equity securities',
    color: 'blue',
    icon: '📈'
  },
  {
    code: 'CASH',
    name: 'Forex',
    description: 'Foreign exchange currency pairs',
    color: 'green',
    icon: '💱'
  },
  {
    code: 'FUT',
    name: 'Futures',
    description: 'Futures contracts and commodities',
    color: 'orange',
    icon: '⚡'
  },
  {
    code: 'IND',
    name: 'Indices',
    description: 'Stock market indices',
    color: 'purple',
    icon: '📊'
  },
  {
    code: 'OPT',
    name: 'Options',
    description: 'Options contracts',
    color: 'red',
    icon: '🎯'
  },
  {
    code: 'BOND',
    name: 'Bonds',
    description: 'Government and corporate bonds',
    color: 'cyan',
    icon: '🏦'
  },
  {
    code: 'CRYPTO',
    name: 'Crypto ETFs',
    description: 'Cryptocurrency ETFs and trusts',
    color: 'gold',
    icon: '₿'
  }
]

export const AssetClassFilter: React.FC<AssetClassFilterProps> = ({
  className,
  showCounts = true,
  compactMode = false
}) => {
  const { 
    searchFilters, 
    updateSearchFilters, 
    resetSearchFilters,
    getAssetClassCounts 
  } = useInstrumentStore()

  const assetClassCounts = getAssetClassCounts()

  const handleAssetClassToggle = (assetClass: string, checked: boolean) => {
    const currentClasses = searchFilters.assetClasses || []
    
    const updatedClasses = checked
      ? [...currentClasses, assetClass]
      : currentClasses.filter(ac => ac !== assetClass)
    
    updateSearchFilters({ assetClasses: updatedClasses })
  }

  const handleSelectAll = () => {
    const allAssetClasses = ASSET_CLASS_CONFIG.map(config => config.code)
    updateSearchFilters({ assetClasses: allAssetClasses })
  }

  const handleClearAll = () => {
    updateSearchFilters({ assetClasses: [] })
  }

  const isAllSelected = searchFilters.assetClasses.length === ASSET_CLASS_CONFIG.length
  const hasAnySelected = searchFilters.assetClasses.length > 0

  if (compactMode) {
    return (
      <div className={className}>
        <Space wrap>
          {ASSET_CLASS_CONFIG.map(config => {
            const isSelected = searchFilters.assetClasses.includes(config.code)
            const count = assetClassCounts[config.code] || 0
            
            return (
              <Tooltip key={config.code} title={config.description}>
                <Tag
                  color={isSelected ? config.color : 'default'}
                  style={{ 
                    cursor: 'pointer',
                    userSelect: 'none',
                    opacity: isSelected ? 1 : 0.6
                  }}
                  onClick={() => handleAssetClassToggle(config.code, !isSelected)}
                >
                  {config.icon} {config.code}
                  {showCounts && count > 0 && ` (${count})`}
                </Tag>
              </Tooltip>
            )
          })}
        </Space>
      </div>
    )
  }

  return (
    <Card
      title={
        <Space>
          <FilterOutlined />
          Asset Class Filter
        </Space>
      }
      size="small"
      className={className}
      extra={
        <Space>
          <Button 
            type="link" 
            size="small"
            onClick={handleSelectAll}
            disabled={isAllSelected}
          >
            All
          </Button>
          <Button 
            type="link" 
            size="small"
            icon={<ClearOutlined />}
            onClick={handleClearAll}
            disabled={!hasAnySelected}
          >
            Clear
          </Button>
        </Space>
      }
    >
      <Space direction="vertical" style={{ width: '100%' }} size="small">
        {ASSET_CLASS_CONFIG.map(config => {
          const isSelected = searchFilters.assetClasses.includes(config.code)
          const count = assetClassCounts[config.code] || 0
          
          return (
            <div key={config.code} style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
              <Checkbox
                checked={isSelected}
                onChange={(e) => handleAssetClassToggle(config.code, e.target.checked)}
                disabled={count === 0 && !isSelected}
              >
                <Space>
                  <span style={{ fontSize: '16px' }}>{config.icon}</span>
                  <Tag color={config.color} style={{ minWidth: '45px', textAlign: 'center' }}>
                    {config.code}
                  </Tag>
                  <span>{config.name}</span>
                </Space>
              </Checkbox>
              {showCounts && (
                <span style={{ 
                  color: count > 0 ? '#666' : '#ccc',
                  fontSize: '12px',
                  minWidth: '30px',
                  textAlign: 'right'
                }}>
                  {count}
                </span>
              )}
            </div>
          )
        })}
        
        {hasAnySelected && (
          <>
            <Divider style={{ margin: '8px 0' }} />
            <div style={{ fontSize: '12px', color: '#666', textAlign: 'center' }}>
              {searchFilters.assetClasses.length} of {ASSET_CLASS_CONFIG.length} asset classes selected
            </div>
          </>
        )}
      </Space>
    </Card>
  )
}