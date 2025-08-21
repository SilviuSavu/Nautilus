/**
 * Chart Data Processors
 * Converts OHLCV data into various chart types (Renko, Point & Figure, etc.)
 */

import { OHLCVData } from '../components/Chart/types/chartTypes'
import { RenkoData, PointFigureData, VolumeProfileData } from '../types/charting'

export interface RenkoConfig {
  brickSize?: number
  autoCalculateBrickSize?: boolean
  source: 'close' | 'high_low' | 'open_close'
}

export interface PointFigureConfig {
  boxSize?: number
  reversalAmount?: number
  autoCalculateBoxSize?: boolean
  source: 'close' | 'high_low'
}

export interface VolumeProfileConfig {
  priceLevels?: number
  sessionType: 'daily' | 'weekly' | 'custom'
  showPOC?: boolean // Point of Control
}

export interface HeikinAshiData {
  time: string
  open: number
  high: number
  low: number
  close: number
  volume: number
}

class ChartDataProcessor {
  
  /**
   * Convert OHLCV data to Renko bricks
   */
  processRenkoData(data: OHLCVData[], config: RenkoConfig): RenkoData[] {
    if (data.length === 0) return []

    const brickSize = config.brickSize || this.calculateOptimalBrickSize(data)
    const renkoData: RenkoData[] = []
    
    let currentPrice = data[0].close
    let lastBrickPrice = Math.floor(currentPrice / brickSize) * brickSize
    let currentTime = data[0].time

    for (let i = 0; i < data.length; i++) {
      const candle = data[i]
      const prices = this.getPricesForRenko(candle, config.source)
      
      for (const price of prices) {
        const bricksNeeded = Math.floor(Math.abs(price - lastBrickPrice) / brickSize)
        
        if (bricksNeeded > 0) {
          const direction = price > lastBrickPrice ? 'up' : 'down'
          
          // Create bricks
          for (let j = 0; j < bricksNeeded; j++) {
            const brickOpen = lastBrickPrice
            const brickClose = direction === 'up' ? 
              lastBrickPrice + brickSize : 
              lastBrickPrice - brickSize
            
            renkoData.push({
              time: candle.time,
              open: brickOpen,
              close: brickClose,
              trend: direction,
              brickSize
            })
            
            lastBrickPrice = brickClose
          }
          
          currentTime = candle.time
        }
      }
    }

    return renkoData
  }

  /**
   * Convert OHLCV data to Point & Figure chart
   */
  processPointFigureData(data: OHLCVData[], config: PointFigureConfig): PointFigureData[] {
    if (data.length === 0) return []

    const boxSize = config.boxSize || this.calculateOptimalBoxSize(data)
    const reversalAmount = config.reversalAmount || 3
    const pfData: PointFigureData[] = []
    
    let currentColumn = 0
    let currentTrend: 'X' | 'O' | null = null
    let currentPrice = data[0].close
    let boxes: Array<{ price: number; type: 'X' | 'O' }> = []

    for (const candle of data) {
      const prices = this.getPricesForPointFigure(candle, config.source)
      
      for (const price of prices) {
        const boxLevel = Math.floor(price / boxSize) * boxSize
        
        if (currentTrend === null) {
          // Initialize first trend
          currentTrend = price > currentPrice ? 'X' : 'O'
          boxes.push({ price: boxLevel, type: currentTrend })
          currentPrice = price
          continue
        }

        const priceChange = Math.abs(price - currentPrice) / boxSize
        
        if (currentTrend === 'X' && price > currentPrice + boxSize) {
          // Continue uptrend
          const newBoxes = Math.floor((price - currentPrice) / boxSize)
          for (let i = 1; i <= newBoxes; i++) {
            boxes.push({ 
              price: currentPrice + (i * boxSize), 
              type: 'X' 
            })
          }
          currentPrice = price
        } else if (currentTrend === 'O' && price < currentPrice - boxSize) {
          // Continue downtrend  
          const newBoxes = Math.floor((currentPrice - price) / boxSize)
          for (let i = 1; i <= newBoxes; i++) {
            boxes.push({ 
              price: currentPrice - (i * boxSize), 
              type: 'O' 
            })
          }
          currentPrice = price
        } else if (priceChange >= reversalAmount * boxSize) {
          // Trend reversal
          pfData.push({
            column: currentColumn,
            boxes: [...boxes],
            time: candle.time
          })
          
          currentColumn++
          currentTrend = currentTrend === 'X' ? 'O' : 'X'
          boxes = []
          
          const newBoxes = Math.floor(priceChange)
          for (let i = 0; i < newBoxes; i++) {
            const boxPrice = currentTrend === 'X' ? 
              currentPrice + (i * boxSize) : 
              currentPrice - (i * boxSize)
            boxes.push({ price: boxPrice, type: currentTrend })
          }
          
          currentPrice = price
        }
      }
    }

    // Add final column
    if (boxes.length > 0) {
      pfData.push({
        column: currentColumn,
        boxes: [...boxes],
        time: data[data.length - 1].time
      })
    }

    return pfData
  }

  /**
   * Calculate Volume Profile data
   */
  processVolumeProfileData(data: OHLCVData[], config: VolumeProfileConfig): VolumeProfileData[] {
    if (data.length === 0) return []

    const priceLevels = config.priceLevels || 50
    const minPrice = Math.min(...data.map(d => d.low))
    const maxPrice = Math.max(...data.map(d => d.high))
    const priceStep = (maxPrice - minPrice) / priceLevels

    const volumeProfile: VolumeProfileData[] = []
    
    // Initialize price levels
    for (let i = 0; i < priceLevels; i++) {
      const priceLevel = minPrice + (i * priceStep)
      volumeProfile.push({
        priceLevel,
        volume: 0,
        buyVolume: 0,
        sellVolume: 0,
        pocLevel: false
      })
    }

    // Distribute volume across price levels
    for (const candle of data) {
      const priceRange = candle.high - candle.low
      const volumePerLevel = candle.volume / (priceRange / priceStep || 1)
      
      const startLevel = Math.floor((candle.low - minPrice) / priceStep)
      const endLevel = Math.floor((candle.high - minPrice) / priceStep)
      
      for (let level = startLevel; level <= endLevel && level < priceLevels; level++) {
        if (level >= 0) {
          volumeProfile[level].volume += volumePerLevel
          
          // Estimate buy/sell volume based on close vs open
          const bullishCandle = candle.close > candle.open
          if (bullishCandle) {
            volumeProfile[level].buyVolume += volumePerLevel * 0.6
            volumeProfile[level].sellVolume += volumePerLevel * 0.4
          } else {
            volumeProfile[level].buyVolume += volumePerLevel * 0.4
            volumeProfile[level].sellVolume += volumePerLevel * 0.6
          }
        }
      }
    }

    // Find Point of Control (POC) - highest volume level
    if (config.showPOC) {
      let maxVolumeIndex = 0
      let maxVolume = 0
      
      volumeProfile.forEach((level, index) => {
        if (level.volume > maxVolume) {
          maxVolume = level.volume
          maxVolumeIndex = index
        }
      })
      
      volumeProfile[maxVolumeIndex].pocLevel = true
    }

    return volumeProfile.filter(level => level.volume > 0)
  }

  /**
   * Convert OHLCV data to Heikin Ashi
   */
  processHeikinAshiData(data: OHLCVData[]): HeikinAshiData[] {
    if (data.length === 0) return []

    const haData: HeikinAshiData[] = []
    let prevHaOpen = (data[0].open + data[0].close) / 2
    let prevHaClose = (data[0].open + data[0].high + data[0].low + data[0].close) / 4

    for (const candle of data) {
      const haClose = (candle.open + candle.high + candle.low + candle.close) / 4
      const haOpen = (prevHaOpen + prevHaClose) / 2
      const haHigh = Math.max(candle.high, haOpen, haClose)
      const haLow = Math.min(candle.low, haOpen, haClose)

      haData.push({
        time: candle.time,
        open: haOpen,
        high: haHigh,
        low: haLow,
        close: haClose,
        volume: candle.volume
      })

      prevHaOpen = haOpen
      prevHaClose = haClose
    }

    return haData
  }

  /**
   * Calculate optimal brick size for Renko charts
   */
  private calculateOptimalBrickSize(data: OHLCVData[]): number {
    if (data.length === 0) return 1

    // Calculate average true range for last 14 periods
    const atrPeriod = Math.min(14, data.length)
    let atrSum = 0
    
    for (let i = 1; i < atrPeriod; i++) {
      const current = data[i]
      const previous = data[i - 1]
      
      const tr = Math.max(
        current.high - current.low,
        Math.abs(current.high - previous.close),
        Math.abs(current.low - previous.close)
      )
      
      atrSum += tr
    }
    
    const atr = atrSum / (atrPeriod - 1)
    
    // Use 50% of ATR as brick size
    return Math.round(atr * 0.5 * 100) / 100
  }

  /**
   * Calculate optimal box size for Point & Figure charts
   */
  private calculateOptimalBoxSize(data: OHLCVData[]): number {
    if (data.length === 0) return 1

    const avgPrice = data.reduce((sum, candle) => sum + candle.close, 0) / data.length
    
    // Use percentage of average price
    const percentageBoxSize = avgPrice * 0.005 // 0.5%
    
    return Math.round(percentageBoxSize * 100) / 100
  }

  /**
   * Get relevant prices for Renko calculation
   */
  private getPricesForRenko(candle: OHLCVData, source: RenkoConfig['source']): number[] {
    switch (source) {
      case 'close':
        return [candle.close]
      case 'high_low':
        return [candle.high, candle.low]
      case 'open_close':
        return [candle.open, candle.close]
      default:
        return [candle.close]
    }
  }

  /**
   * Get relevant prices for Point & Figure calculation
   */
  private getPricesForPointFigure(candle: OHLCVData, source: PointFigureConfig['source']): number[] {
    switch (source) {
      case 'close':
        return [candle.close]
      case 'high_low':
        return [candle.high, candle.low]
      default:
        return [candle.close]
    }
  }
}

// Export singleton instance
export const chartDataProcessor = new ChartDataProcessor()

// Export types and classes
export { ChartDataProcessor }
export default chartDataProcessor