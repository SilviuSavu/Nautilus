import React from 'react'
import { Select, Space, Tag } from 'antd'
import { useChartStore } from './hooks/useChartStore'
import { Instrument } from './types/chartTypes'

const { Option, OptGroup } = Select

// Predefined instruments for all asset classes
const PREDEFINED_INSTRUMENTS: Record<string, Instrument[]> = {
  stocks: [
    { symbol: 'AAPL', venue: 'NASDAQ', assetClass: 'STK', currency: 'USD', id: 'AAPL-STK', name: 'Apple Inc.' },
    { symbol: 'MSFT', venue: 'NASDAQ', assetClass: 'STK', currency: 'USD', id: 'MSFT-STK', name: 'Microsoft Corporation' },
    { symbol: 'GOOGL', venue: 'NASDAQ', assetClass: 'STK', currency: 'USD', id: 'GOOGL-STK', name: 'Alphabet Inc.' },
    { symbol: 'AMZN', venue: 'NASDAQ', assetClass: 'STK', currency: 'USD', id: 'AMZN-STK', name: 'Amazon.com Inc.' },
    { symbol: 'TSLA', venue: 'NASDAQ', assetClass: 'STK', currency: 'USD', id: 'TSLA-STK', name: 'Tesla Inc.' },
    { symbol: 'NVDA', venue: 'NASDAQ', assetClass: 'STK', currency: 'USD', id: 'NVDA-STK', name: 'NVIDIA Corporation' }
  ],
  forex: [
    { symbol: 'EURUSD', venue: 'IDEALPRO', assetClass: 'CASH', currency: 'USD', id: 'EURUSD-CASH', name: 'Euro / US Dollar' },
    { symbol: 'GBPUSD', venue: 'IDEALPRO', assetClass: 'CASH', currency: 'USD', id: 'GBPUSD-CASH', name: 'British Pound / US Dollar' },
    { symbol: 'USDJPY', venue: 'IDEALPRO', assetClass: 'CASH', currency: 'JPY', id: 'USDJPY-CASH', name: 'US Dollar / Japanese Yen' },
    { symbol: 'USDCHF', venue: 'IDEALPRO', assetClass: 'CASH', currency: 'CHF', id: 'USDCHF-CASH', name: 'US Dollar / Swiss Franc' },
    { symbol: 'AUDUSD', venue: 'IDEALPRO', assetClass: 'CASH', currency: 'USD', id: 'AUDUSD-CASH', name: 'Australian Dollar / US Dollar' },
    { symbol: 'USDCAD', venue: 'IDEALPRO', assetClass: 'CASH', currency: 'CAD', id: 'USDCAD-CASH', name: 'US Dollar / Canadian Dollar' }
  ],
  futures: [
    { symbol: 'ES', venue: 'GLOBEX', assetClass: 'FUT', currency: 'USD', id: 'ES-FUT', name: 'E-mini S&P 500' },
    { symbol: 'NQ', venue: 'GLOBEX', assetClass: 'FUT', currency: 'USD', id: 'NQ-FUT', name: 'E-mini NASDAQ-100' },
    { symbol: 'YM', venue: 'GLOBEX', assetClass: 'FUT', currency: 'USD', id: 'YM-FUT', name: 'E-mini Dow Jones' },
    { symbol: 'RTY', venue: 'GLOBEX', assetClass: 'FUT', currency: 'USD', id: 'RTY-FUT', name: 'E-mini Russell 2000' },
    { symbol: 'CL', venue: 'NYMEX', assetClass: 'FUT', currency: 'USD', id: 'CL-FUT', name: 'Crude Oil' },
    { symbol: 'NG', venue: 'NYMEX', assetClass: 'FUT', currency: 'USD', id: 'NG-FUT', name: 'Natural Gas' },
    { symbol: 'GC', venue: 'NYMEX', assetClass: 'FUT', currency: 'USD', id: 'GC-FUT', name: 'Gold' },
    { symbol: 'SI', venue: 'NYMEX', assetClass: 'FUT', currency: 'USD', id: 'SI-FUT', name: 'Silver' }
  ],
  indices: [
    { symbol: 'SPX', venue: 'CBOE', assetClass: 'IND', currency: 'USD', id: 'SPX-IND', name: 'S&P 500 Index' },
    { symbol: 'NDX', venue: 'NASDAQ', assetClass: 'IND', currency: 'USD', id: 'NDX-IND', name: 'NASDAQ-100 Index' },
    { symbol: 'DJX', venue: 'CBOE', assetClass: 'IND', currency: 'USD', id: 'DJX-IND', name: 'Dow Jones Industrial Average' },
    { symbol: 'VIX', venue: 'CBOE', assetClass: 'IND', currency: 'USD', id: 'VIX-IND', name: 'CBOE Volatility Index' }
  ],
  etfs: [
    { symbol: 'SPY', venue: 'ARCA', assetClass: 'STK', currency: 'USD', id: 'SPY-STK', name: 'SPDR S&P 500 ETF' },
    { symbol: 'QQQ', venue: 'NASDAQ', assetClass: 'STK', currency: 'USD', id: 'QQQ-STK', name: 'Invesco QQQ Trust' },
    { symbol: 'IWM', venue: 'ARCA', assetClass: 'STK', currency: 'USD', id: 'IWM-STK', name: 'iShares Russell 2000 ETF' },
    { symbol: 'TLT', venue: 'NASDAQ', assetClass: 'STK', currency: 'USD', id: 'TLT-STK', name: 'iShares 20+ Year Treasury Bond ETF' },
    { symbol: 'GLD', venue: 'ARCA', assetClass: 'STK', currency: 'USD', id: 'GLD-STK', name: 'SPDR Gold Trust' },
    { symbol: 'USO', venue: 'ARCA', assetClass: 'STK', currency: 'USD', id: 'USO-STK', name: 'United States Oil Fund' }
  ]
}

interface InstrumentSelectorProps {
  className?: string
}

export const InstrumentSelector: React.FC<InstrumentSelectorProps> = ({ className }) => {
  const { currentInstrument, setCurrentInstrument } = useChartStore()

  const handleInstrumentChange = (value: string) => {
    // Find the instrument in all categories
    for (const category of Object.values(PREDEFINED_INSTRUMENTS)) {
      const instrument = category.find(inst => inst.id === value)
      if (instrument) {
        setCurrentInstrument(instrument)
        break
      }
    }
  }

  const getAssetClassColor = (assetClass: string) => {
    const colorMap: Record<string, string> = {
      'STK': 'blue',
      'CASH': 'green', 
      'FUT': 'orange',
      'IND': 'purple',
      'OPT': 'red',
      'BOND': 'cyan'
    }
    return colorMap[assetClass] || 'default'
  }

  return (
    <div className={className}>
      <Space direction="vertical" style={{ width: '100%' }}>
        <Select
          value={currentInstrument?.id}
          onChange={handleInstrumentChange}
          placeholder="Search or select instrument..."
          style={{ width: '100%', minWidth: 300 }}
          showSearch
          allowClear
          filterOption={(input, option) => {
            if (!option?.value) return false
            // Find the instrument by ID to get symbol and name for search
            for (const category of Object.values(PREDEFINED_INSTRUMENTS)) {
              const instrument = category.find(inst => inst.id === option.value)
              if (instrument) {
                const searchText = `${instrument.symbol} ${instrument.name}`.toLowerCase()
                return searchText.includes(input.toLowerCase())
              }
            }
            return false
          }}
        >
          <OptGroup label="📈 Stocks">
            {PREDEFINED_INSTRUMENTS.stocks.map(instrument => (
              <Option key={instrument.id} value={instrument.id}>
                <Space>
                  <Tag color={getAssetClassColor(instrument.assetClass)}>
                    {instrument.assetClass}
                  </Tag>
                  <strong>{instrument.symbol}</strong>
                  <span style={{ color: '#888' }}>{instrument.name}</span>
                </Space>
              </Option>
            ))}
          </OptGroup>
          
          <OptGroup label="💱 Forex">
            {PREDEFINED_INSTRUMENTS.forex.map(instrument => (
              <Option key={instrument.id} value={instrument.id}>
                <Space>
                  <Tag color={getAssetClassColor(instrument.assetClass)}>
                    {instrument.assetClass}
                  </Tag>
                  <strong>{instrument.symbol}</strong>
                  <span style={{ color: '#888' }}>{instrument.name}</span>
                </Space>
              </Option>
            ))}
          </OptGroup>
          
          <OptGroup label="⚡ Futures">
            {PREDEFINED_INSTRUMENTS.futures.map(instrument => (
              <Option key={instrument.id} value={instrument.id}>
                <Space>
                  <Tag color={getAssetClassColor(instrument.assetClass)}>
                    {instrument.assetClass}
                  </Tag>
                  <strong>{instrument.symbol}</strong>
                  <span style={{ color: '#888' }}>{instrument.name}</span>
                </Space>
              </Option>
            ))}
          </OptGroup>
          
          <OptGroup label="📊 Indices">
            {PREDEFINED_INSTRUMENTS.indices.map(instrument => (
              <Option key={instrument.id} value={instrument.id}>
                <Space>
                  <Tag color={getAssetClassColor(instrument.assetClass)}>
                    {instrument.assetClass}
                  </Tag>
                  <strong>{instrument.symbol}</strong>
                  <span style={{ color: '#888' }}>{instrument.name}</span>
                </Space>
              </Option>
            ))}
          </OptGroup>
          
          <OptGroup label="🏦 ETFs">
            {PREDEFINED_INSTRUMENTS.etfs.map(instrument => (
              <Option key={instrument.id} value={instrument.id}>
                <Space>
                  <Tag color={getAssetClassColor(instrument.assetClass)}>
                    {instrument.assetClass}
                  </Tag>
                  <strong>{instrument.symbol}</strong>
                  <span style={{ color: '#888' }}>{instrument.name}</span>
                </Space>
              </Option>
            ))}
          </OptGroup>
        </Select>
        
        {currentInstrument && (
          <div style={{ padding: '8px 12px', background: '#f5f5f5', borderRadius: '6px' }}>
            <Space>
              <Tag color={getAssetClassColor(currentInstrument.assetClass)}>
                {currentInstrument.assetClass}
              </Tag>
              <span><strong>{currentInstrument.symbol}</strong></span>
              <span style={{ color: '#888' }}>•</span>
              <span style={{ color: '#666' }}>{currentInstrument.venue}</span>
              <span style={{ color: '#888' }}>•</span>
              <span style={{ color: '#666' }}>{currentInstrument.currency}</span>
            </Space>
          </div>
        )}
      </Space>
    </div>
  )
}