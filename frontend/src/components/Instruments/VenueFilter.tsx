import React from 'react'
import { Card, Checkbox, Space, Badge, Tooltip, Button, Divider } from 'antd'
import { BankOutlined, ClearOutlined, GlobalOutlined } from '@ant-design/icons'
import { VenueInfo } from './types/instrumentTypes'
import { useInstrumentStore } from './hooks/useInstrumentStore'

interface VenueFilterProps {
  className?: string
  showCounts?: boolean
  compactMode?: boolean
}

const VENUE_CONFIG = [
  {
    code: 'NASDAQ',
    name: 'NASDAQ Stock Market',
    country: 'US',
    flag: '🇺🇸',
    primaryAssets: ['STK', 'ETF']
  },
  {
    code: 'NYSE',
    name: 'New York Stock Exchange',
    country: 'US',
    flag: '🇺🇸',
    primaryAssets: ['STK']
  },
  {
    code: 'IDEALPRO',
    name: 'IDEALPRO FX',
    country: 'Global',
    flag: '🌍',
    primaryAssets: ['CASH']
  },
  {
    code: 'GLOBEX',
    name: 'CME Globex',
    country: 'US',
    flag: '🇺🇸',
    primaryAssets: ['FUT', 'OPT']
  },
  {
    code: 'NYMEX',
    name: 'NYMEX',
    country: 'US',
    flag: '🇺🇸',
    primaryAssets: ['FUT']
  },
  {
    code: 'CBOT',
    name: 'Chicago Board of Trade',
    country: 'US',
    flag: '🇺🇸',
    primaryAssets: ['FUT']
  },
  {
    code: 'SMART',
    name: 'IB Smart Routing',
    country: 'Global',
    flag: '🌐',
    primaryAssets: ['STK', 'ETF', 'OPT']
  },
  {
    code: 'ARCA',
    name: 'NYSE Arca',
    country: 'US',
    flag: '🇺🇸',
    primaryAssets: ['STK', 'ETF']
  }
]

export const VenueFilter: React.FC<VenueFilterProps> = ({
  className,
  showCounts = true,
  compactMode = false
}) => {
  const { 
    searchFilters, 
    updateSearchFilters,
    venueStatus,
    getVenueCounts,
    updateVenueStatus
  } = useInstrumentStore()

  const venueCounts = getVenueCounts()

  const handleVenueToggle = (venue: string, checked: boolean) => {
    const currentVenues = searchFilters.venues || []
    
    const updatedVenues = checked
      ? [...currentVenues, venue]
      : currentVenues.filter(v => v !== venue)
    
    updateSearchFilters({ venues: updatedVenues })
  }

  const handleSelectAll = () => {
    const allVenues = VENUE_CONFIG.map(config => config.code)
    updateSearchFilters({ venues: allVenues })
  }

  const handleClearAll = () => {
    updateSearchFilters({ venues: [] })
  }

  const handleConnectedOnly = () => {
    const connectedVenues = VENUE_CONFIG
      .filter(config => {
        const status = venueStatus[config.code]
        return status?.connectionStatus === 'connected'
      })
      .map(config => config.code)
    
    updateSearchFilters({ 
      venues: connectedVenues,
      onlyConnectedVenues: true 
    })
  }

  const getConnectionStatus = (venueCode: string) => {
    const status = venueStatus[venueCode]
    return status?.connectionStatus || 'unknown'
  }

  const getStatusBadge = (venueCode: string) => {
    const status = getConnectionStatus(venueCode)
    
    switch (status) {
      case 'connected':
        return <Badge status="success" />
      case 'connecting':
        return <Badge status="processing" />
      case 'error':
        return <Badge status="error" />
      case 'maintenance':
        return <Badge status="warning" />
      case 'disconnected':
        return <Badge status="default" />
      default:
        return <Badge status="default" />
    }
  }

  const isAllSelected = searchFilters.venues.length === VENUE_CONFIG.length
  const hasAnySelected = searchFilters.venues.length > 0

  if (compactMode) {
    return (
      <div className={className}>
        <Space wrap>
          {VENUE_CONFIG.map(config => {
            const isSelected = searchFilters.venues.includes(config.code)
            const count = venueCounts[config.code] || 0
            const status = getConnectionStatus(config.code)
            
            return (
              <Tooltip 
                key={config.code} 
                title={
                  <div>
                    <div><strong>{config.name}</strong></div>
                    <div>Status: {status}</div>
                    <div>Assets: {config.primaryAssets.join(', ')}</div>
                  </div>
                }
              >
                <Badge dot={status === 'connected'} color="green">
                  <span
                    style={{ 
                      cursor: 'pointer',
                      padding: '4px 8px',
                      borderRadius: '4px',
                      border: `1px solid ${isSelected ? '#1890ff' : '#d9d9d9'}`,
                      backgroundColor: isSelected ? '#e6f7ff' : 'white',
                      userSelect: 'none',
                      fontSize: '12px'
                    }}
                    onClick={() => handleVenueToggle(config.code, !isSelected)}
                  >
                    {config.flag} {config.code}
                    {showCounts && count > 0 && ` (${count})`}
                  </span>
                </Badge>
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
          <BankOutlined />
          Venue Filter
        </Space>
      }
      size="small"
      className={className}
      extra={
        <Space>
          <Tooltip title="Select only connected venues">
            <Button 
              type="link" 
              size="small"
              icon={<GlobalOutlined />}
              onClick={handleConnectedOnly}
            >
              Connected
            </Button>
          </Tooltip>
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
        {VENUE_CONFIG.map(config => {
          const isSelected = searchFilters.venues.includes(config.code)
          const count = venueCounts[config.code] || 0
          const status = getConnectionStatus(config.code)
          
          return (
            <div key={config.code} style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
              <Checkbox
                checked={isSelected}
                onChange={(e) => handleVenueToggle(config.code, e.target.checked)}
                disabled={count === 0 && !isSelected}
              >
                <Space>
                  {getStatusBadge(config.code)}
                  <span style={{ fontSize: '14px' }}>{config.flag}</span>
                  <span style={{ fontWeight: 500 }}>{config.code}</span>
                  <span style={{ color: '#666', fontSize: '12px' }}>
                    {config.name}
                  </span>
                </Space>
              </Checkbox>
              <Space>
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
                <Tooltip title={`Status: ${status}`}>
                  <span style={{ 
                    fontSize: '10px',
                    color: status === 'connected' ? '#52c41a' : '#8c8c8c',
                    textTransform: 'uppercase',
                    fontWeight: 500
                  }}>
                    {status}
                  </span>
                </Tooltip>
              </Space>
            </div>
          )
        })}
        
        {hasAnySelected && (
          <>
            <Divider style={{ margin: '8px 0' }} />
            <div style={{ fontSize: '12px', color: '#666', textAlign: 'center' }}>
              {searchFilters.venues.length} of {VENUE_CONFIG.length} venues selected
              {searchFilters.onlyConnectedVenues && (
                <span style={{ color: '#52c41a' }}> (connected only)</span>
              )}
            </div>
          </>
        )}
        
        <Divider style={{ margin: '8px 0' }} />
        <div style={{ textAlign: 'center' }}>
          <Button 
            type="link" 
            size="small"
            onClick={() => updateVenueStatus()}
          >
            Refresh Status
          </Button>
        </div>
      </Space>
    </Card>
  )
}