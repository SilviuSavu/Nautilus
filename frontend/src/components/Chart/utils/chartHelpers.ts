import { OHLCVData, Instrument } from '../types/chartTypes'

/**
 * Utility functions for chart data processing and normalization
 */

// Convert NautilusTrader timestamp (nanoseconds) to ISO string
export const convertNautilusTimestamp = (nanoseconds: number): string => {
  const milliseconds = nanoseconds / 1000000
  return new Date(milliseconds).toISOString()
}

// Convert ISO string to TradingView time (seconds since epoch)
export const convertToTradingViewTime = (isoString: string): number => {
  return Math.floor(new Date(isoString).getTime() / 1000)
}

// Normalize price values to ensure they are valid numbers
export const normalizePrice = (price: any): number => {
  const parsed = typeof price === 'string' ? parseFloat(price) : Number(price)
  return isNaN(parsed) || !isFinite(parsed) ? 0 : Math.max(parsed, 0.00001)
}

// Normalize volume values
export const normalizeVolume = (volume: any): number => {
  const parsed = typeof volume === 'string' ? parseFloat(volume) : Number(volume)
  return isNaN(parsed) || !isFinite(parsed) ? 0 : Math.max(parsed, 0)
}

// Validate OHLCV data integrity
export const validateOHLCVData = (data: Partial<OHLCVData>): data is OHLCVData => {
  if (!data.time || !data.open || !data.high || !data.low || !data.close) {
    return false
  }

  const open = normalizePrice(data.open)
  const high = normalizePrice(data.high)
  const low = normalizePrice(data.low)
  const close = normalizePrice(data.close)

  // Basic OHLC validation: high >= max(open, close) and low <= min(open, close)
  if (high < Math.max(open, close) || low > Math.min(open, close)) {
    return false
  }

  return true
}

// Normalize raw NautilusTrader bar data to OHLCV format
export const normalizeNautilusBarData = (rawData: any): OHLCVData | null => {
  try {
    if (!rawData) return null

    const normalized: Partial<OHLCVData> = {
      time: rawData.ts_event ? convertNautilusTimestamp(rawData.ts_event) : rawData.time,
      open: normalizePrice(rawData.open),
      high: normalizePrice(rawData.high),
      low: normalizePrice(rawData.low),
      close: normalizePrice(rawData.close),
      volume: normalizeVolume(rawData.volume || 0)
    }

    return validateOHLCVData(normalized) ? normalized as OHLCVData : null
  } catch (error) {
    console.error('Failed to normalize bar data:', error, rawData)
    return null
  }
}

// Normalize array of bar data
export const normalizeBarDataArray = (rawDataArray: any[]): OHLCVData[] => {
  if (!Array.isArray(rawDataArray)) {
    console.warn('Expected array but received:', typeof rawDataArray)
    return []
  }

  return rawDataArray
    .map(normalizeNautilusBarData)
    .filter((data): data is OHLCVData => data !== null)
    .sort((a, b) => new Date(a.time).getTime() - new Date(b.time).getTime())
}

// Convert quote tick to OHLCV (for very short timeframes)
export const convertQuoteTickToOHLCV = (
  quoteTick: any,
  previousCandle?: OHLCVData
): OHLCVData | null => {
  try {
    if (!quoteTick || (!quoteTick.bid && !quoteTick.ask && !quoteTick.price)) {
      return null
    }

    const price = normalizePrice(quoteTick.price || quoteTick.bid || quoteTick.ask)
    const time = quoteTick.ts_event 
      ? convertNautilusTimestamp(quoteTick.ts_event)
      : new Date().toISOString()

    // If we have a previous candle, update it; otherwise create new one
    if (previousCandle) {
      return {
        time: previousCandle.time, // Keep original time
        open: previousCandle.open,
        high: Math.max(previousCandle.high, price),
        low: Math.min(previousCandle.low, price),
        close: price,
        volume: previousCandle.volume // No volume change for quotes
      }
    }

    // Create new candle from quote
    return {
      time,
      open: price,
      high: price,
      low: price,
      close: price,
      volume: 0
    }
  } catch (error) {
    console.error('Failed to convert quote tick:', error, quoteTick)
    return null
  }
}

// Convert trade tick to OHLCV
export const convertTradeTickToOHLCV = (
  tradeTick: any,
  previousCandle?: OHLCVData
): OHLCVData | null => {
  try {
    if (!tradeTick || !tradeTick.price) {
      return null
    }

    const price = normalizePrice(tradeTick.price)
    const volume = normalizeVolume(tradeTick.size || tradeTick.quantity || 0)
    const time = tradeTick.ts_event 
      ? convertNautilusTimestamp(tradeTick.ts_event)
      : new Date().toISOString()

    // If we have a previous candle, update it
    if (previousCandle) {
      return {
        time: previousCandle.time, // Keep original time
        open: previousCandle.open,
        high: Math.max(previousCandle.high, price),
        low: Math.min(previousCandle.low, price),
        close: price,
        volume: previousCandle.volume + volume
      }
    }

    // Create new candle from trade
    return {
      time,
      open: price,
      high: price,
      low: price,
      close: price,
      volume
    }
  } catch (error) {
    console.error('Failed to convert trade tick:', error, tradeTick)
    return null
  }
}

// Parse instrument ID into components
export const parseInstrumentId = (instrumentId: string): Partial<Instrument> => {
  try {
    // NautilusTrader instrument IDs are typically in format: SYMBOL.VENUE
    const parts = instrumentId.split('.')
    if (parts.length >= 2) {
      return {
        symbol: parts[0],
        venue: parts[1],
        id: instrumentId
      }
    }

    // Fallback if format is different
    return {
      symbol: instrumentId,
      id: instrumentId
    }
  } catch (error) {
    console.error('Failed to parse instrument ID:', error, instrumentId)
    return {
      symbol: instrumentId,
      id: instrumentId
    }
  }
}

// Calculate technical indicators (simple implementations)
export const calculateSMA = (data: OHLCVData[], period: number): number[] => {
  const sma: number[] = []
  
  for (let i = 0; i < data.length; i++) {
    if (i < period - 1) {
      sma.push(NaN)
    } else {
      const sum = data.slice(i - period + 1, i + 1).reduce((acc, candle) => acc + candle.close, 0)
      sma.push(sum / period)
    }
  }
  
  return sma
}

export const calculateEMA = (data: OHLCVData[], period: number): number[] => {
  const ema: number[] = []
  const multiplier = 2 / (period + 1)
  
  for (let i = 0; i < data.length; i++) {
    if (i === 0) {
      ema.push(data[i].close)
    } else {
      const currentEMA = (data[i].close - ema[i - 1]) * multiplier + ema[i - 1]
      ema.push(currentEMA)
    }
  }
  
  return ema
}

// Format price for display
export const formatPrice = (price: number, precision: number = 4): string => {
  if (!isFinite(price) || isNaN(price)) return '0.0000'
  return price.toFixed(precision)
}

// Format volume for display
export const formatVolume = (volume: number): string => {
  if (!isFinite(volume) || isNaN(volume)) return '0'
  
  if (volume >= 1e9) {
    return `${(volume / 1e9).toFixed(2)}B`
  } else if (volume >= 1e6) {
    return `${(volume / 1e6).toFixed(2)}M`
  } else if (volume >= 1e3) {
    return `${(volume / 1e3).toFixed(2)}K`
  }
  
  return volume.toFixed(0)
}