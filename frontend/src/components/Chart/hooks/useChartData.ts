import { useEffect, useCallback, useRef } from 'react'
import { useChartStore } from './useChartStore'
import { OHLCVData, Instrument, Timeframe } from '../types/chartTypes'

interface HistoricalDataResponse {
  symbol: string
  timeframe: string
  candles: OHLCVData[]
  total: number
  start_date?: string
  end_date?: string
  source?: string
}

// Use empty string for API_BASE_URL when using Vite proxy
const API_BASE_URL = ''

export const useChartData = () => {
  const {
    currentInstrument,
    timeframe,
    setChartData,
    setLoading,
    setError
  } = useChartStore()

  const abortControllerRef = useRef<AbortController | null>(null)

  // Fetch historical data from backend API
  const fetchHistoricalData = useCallback(async (
    instrument: Instrument,
    timeframe: Timeframe
  ): Promise<OHLCVData[]> => {
    // Cancel any pending request
    if (abortControllerRef.current) {
      abortControllerRef.current.abort()
    }

    abortControllerRef.current = new AbortController()
    const signal = abortControllerRef.current.signal

    try {
      setLoading(true)
      setError(null)

      const params = new URLSearchParams({
        symbol: instrument.symbol,
        timeframe,
        asset_class: instrument.assetClass,
        exchange: instrument.venue,
        currency: instrument.currency
      })
      
      console.log('📡 Making API request:', `${API_BASE_URL}/api/v1/market-data/historical/bars?${params}`)

      const response = await fetch(
        `${API_BASE_URL}/api/v1/market-data/historical/bars?${params}`,
        { signal }
      )

      if (!response.ok) {
        throw new Error(`Failed to fetch historical data: ${response.status} ${response.statusText}`)
      }

      const result: HistoricalDataResponse = await response.json()
      console.log('📊 API Response received:', { 
        symbol: result.symbol, 
        candleCount: result.candles?.length || 0,
        firstCandle: result.candles?.[0],
        timeframe: result.timeframe 
      })

      return result.candles || []
    } catch (error) {
      if (error instanceof Error && error.name === 'AbortError') {
        // Request was cancelled, don't set error
        return []
      }

      const errorMessage = error instanceof Error ? error.message : 'Unknown error occurred'
      setError({
        type: 'data',
        message: `Failed to load historical data: ${errorMessage}`,
        timestamp: new Date().toISOString()
      })
      throw error
    } finally {
      setLoading(false)
      abortControllerRef.current = null
    }
  }, [setLoading, setError])


  // Load chart data for current instrument and timeframe
  const loadChartData = useCallback(async () => {
    console.log('🔍 loadChartData called with instrument:', currentInstrument)
    if (!currentInstrument) {
      console.log('❌ No current instrument, setting empty data')
      setChartData({
        candles: [],
        volume: []
      })
      return
    }

    try {
      // Only fetch real data, no mock data fallback
      const candles = await fetchHistoricalData(currentInstrument, timeframe)
      
      console.log('📈 fetchHistoricalData returned:', candles.length, 'candles')
      
      if (candles.length === 0) {
        console.warn('❌ No historical data available for', currentInstrument.symbol)
        setChartData({
          candles: [],
          volume: []
        })
        // Don't set error for empty data - this is expected behavior for some instruments
        return
      }

      const volume = candles.map(candle => ({
        time: candle.time,
        value: candle.volume,
        color: candle.close > candle.open ? '#26a69a80' : '#ef535080'
      }))

      console.log('✅ Setting chart data:', { 
        candlesCount: candles.length, 
        volumeCount: volume.length,
        firstCandle: candles[0],
        firstVolume: volume[0]
      })
      
      setChartData({
        candles,
        volume
      })
      
      // Clear any previous errors on successful data load
      setError(null)

    } catch (error) {
      console.error('Failed to load chart data:', error)
      setChartData({
        candles: [],
        volume: []
      })
    }
  }, [currentInstrument, timeframe, fetchHistoricalData, setChartData, setError])

  // Load data when instrument or timeframe changes
  useEffect(() => {
    if (currentInstrument) {
      loadChartData()
    }
  }, [currentInstrument, timeframe, loadChartData])

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      if (abortControllerRef.current) {
        abortControllerRef.current.abort()
      }
    }
  }, [])

  return {
    loadChartData,
    fetchHistoricalData
  }
}