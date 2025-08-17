import React, { useEffect, useRef, useState } from 'react'
import { createChart } from 'lightweight-charts'
import { Spin, Alert } from 'antd'
import { useChartStore } from './Chart/hooks/useChartStore'

interface HistoricalBar {
  time: string
  open: number
  high: number
  low: number
  close: number
  volume?: number
}

interface HistoricalDataResponse {
  symbol: string
  timeframe: string
  candles: HistoricalBar[]
  total: number
  start_date?: string
  end_date?: string
  source?: string
}

export const TestChart: React.FC = () => {
  const chartContainerRef = useRef<HTMLDivElement>(null)
  const chartRef = useRef<any>(null)
  const candleSeriesRef = useRef<any>(null)
  const { currentInstrument, timeframe } = useChartStore()
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [dataSource, setDataSource] = useState<string>('')

  const fetchHistoricalData = async (symbol: string, timeframe: string): Promise<HistoricalDataResponse> => {
    const apiUrl = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000'
    const url = `${apiUrl}/api/v1/market-data/historical/bars?symbol=${symbol}&timeframe=${timeframe}&limit=100`
    
    console.log('📊 Making API request to:', url)
    console.log('📊 Environment VITE_API_BASE_URL:', import.meta.env.VITE_API_BASE_URL)
    
    const response = await fetch(url)
    
    console.log('📊 API Response status:', response.status, response.statusText)
    
    if (!response.ok) {
      const errorText = await response.text()
      console.log('📊 API Error response:', errorText)
      throw new Error(`HTTP ${response.status}: ${response.statusText} - ${errorText}`)
    }
    
    const data = await response.json()
    console.log('📊 API Response data:', data)
    return data
  }

  // Create chart once on mount
  useEffect(() => {
    if (!chartContainerRef.current) return

    console.log('📊 Creating chart container')
    
    const chart = createChart(chartContainerRef.current, {
      width: 800,
      height: 400,
      layout: {
        background: { color: '#ffffff' },
        textColor: '#333',
      },
      grid: {
        vertLines: { color: '#f0f0f0' },
        horzLines: { color: '#f0f0f0' },
      },
      timeScale: {
        borderColor: '#cccccc',
      },
      rightPriceScale: {
        borderColor: '#cccccc',
      },
    })

    const candleSeries = chart.addCandlestickSeries({
      upColor: '#00ff88',
      downColor: '#ff4444',
      borderDownColor: '#ff4444',
      borderUpColor: '#00ff88',
      wickDownColor: '#ff4444',
      wickUpColor: '#00ff88',
    })

    chartRef.current = chart
    candleSeriesRef.current = candleSeries
    
    console.log('📊 Chart created successfully')
    
    return () => {
      console.log('📊 Cleaning up chart')
      if (chartRef.current) {
        chartRef.current.remove()
        chartRef.current = null
        candleSeriesRef.current = null
      }
    }
  }, []) // Only run once

  // Load data when instrument or timeframe changes
  useEffect(() => {
    if (!currentInstrument?.symbol || !candleSeriesRef.current) return

    console.log('📊 Loading data for:', currentInstrument.symbol, timeframe)
    
    const loadData = async () => {
      setLoading(true)
      setError(null)
      setDataSource('')
      
      try {
        const data = await fetchHistoricalData(currentInstrument.symbol, timeframe)
        console.log('📊 Data received:', data.candles?.length, 'candles')
        
        if (data.candles && data.candles.length > 0 && candleSeriesRef.current) {
          // Convert data to lightweight-charts format
          const chartData = data.candles.map(candle => {
            let time: string | number
            
            if (candle.time.includes('  ')) {
              const dateStr = candle.time.replace('  ', ' ')
              const date = new Date(dateStr.replace(/(\d{4})(\d{2})(\d{2}) (\d{2}):(\d{2}):(\d{2})/, '$1-$2-$3T$4:$5:$6'))
              time = Math.floor(date.getTime() / 1000)
            } else {
              time = candle.time
            }
            
            return {
              time,
              open: candle.open,
              high: candle.high,
              low: candle.low,
              close: candle.close,
            }
          })

          candleSeriesRef.current.setData(chartData)
          setDataSource(data.source || 'Backend')
          console.log('📊 Chart data updated successfully')
          
        } else {
          throw new Error(`No data available for ${currentInstrument.symbol} (${timeframe})`)
        }
      } catch (err) {
        console.error('📊 Data loading error:', err)
        setError(err instanceof Error ? err.message : 'Failed to load data')
        setDataSource('Error')
      } finally {
        setLoading(false)
      }
    }

    loadData()
  }, [currentInstrument, timeframe])

  return (
    <div>
      <div style={{ display: 'flex', alignItems: 'center', gap: '16px', marginBottom: '8px' }}>
        <h3 style={{ margin: 0 }}>
          Real Market Data - {currentInstrument?.symbol || 'No Instrument'} ({timeframe})
        </h3>
        {loading && <Spin size="small" />}
        {dataSource && (
          <span style={{ fontSize: '12px', color: '#666', fontStyle: 'italic' }}>
            Source: {dataSource}
          </span>
        )}
      </div>
      
      {error && (
        <Alert
          type="error"
          message="Failed to Load Real Market Data"
          description={error}
          style={{ marginBottom: '8px' }}
          showIcon
        />
      )}
      
      <div 
        ref={chartContainerRef} 
        style={{ 
          width: '800px', 
          height: '400px', 
          border: '1px solid #ccc',
          position: 'relative',
          backgroundColor: loading ? '#f9f9f9' : '#ffffff'
        }} 
      >
        {loading && (
          <div style={{
            position: 'absolute',
            top: '50%',
            left: '50%',
            transform: 'translate(-50%, -50%)',
            textAlign: 'center'
          }}>
            <Spin size="large" />
            <div style={{ marginTop: '16px', color: '#666' }}>
              Loading {currentInstrument?.symbol} ({timeframe}) data...
            </div>
          </div>
        )}
        {!loading && error && (
          <div style={{
            position: 'absolute',
            top: '50%',
            left: '50%',
            transform: 'translate(-50%, -50%)',
            textAlign: 'center',
            color: '#999'
          }}>
            <div style={{ fontSize: '18px', marginBottom: '8px' }}>📊</div>
            <div>No chart data available</div>
            <div style={{ fontSize: '12px', marginTop: '4px' }}>
              Check console for details
            </div>
          </div>
        )}
      </div>
    </div>
  )
}