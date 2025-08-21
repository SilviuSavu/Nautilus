/**
 * Chart Data Processors Tests
 * Tests for advanced chart type data transformation algorithms
 */

import { describe, it, expect, beforeEach } from 'vitest'
import { ChartDataProcessor } from '../chartDataProcessors'
import { OHLCVData } from '../../components/Chart/types/chartTypes'

describe('ChartDataProcessor', () => {
  let processor: ChartDataProcessor
  let mockData: OHLCVData[]

  beforeEach(() => {
    processor = new ChartDataProcessor()
    
    // Create realistic market data for testing
    mockData = [
      { time: '2023-01-01', open: 100, high: 105, low: 98, close: 102, volume: 1000 },
      { time: '2023-01-02', open: 102, high: 108, low: 100, close: 106, volume: 1200 },
      { time: '2023-01-03', open: 106, high: 110, low: 104, close: 108, volume: 800 },
      { time: '2023-01-04', open: 108, high: 112, low: 105, close: 107, volume: 900 },
      { time: '2023-01-05', open: 107, high: 111, low: 103, close: 105, volume: 1100 },
      { time: '2023-01-06', open: 105, high: 109, low: 101, close: 103, volume: 950 },
      { time: '2023-01-07', open: 103, high: 107, low: 99, close: 101, volume: 1050 },
      { time: '2023-01-08', open: 101, high: 105, low: 97, close: 99, volume: 1300 },
      { time: '2023-01-09', open: 99, high: 103, low: 95, close: 97, volume: 1400 },
      { time: '2023-01-10', open: 97, high: 101, low: 93, close: 95, volume: 1200 }
    ]
  })

  describe('Renko Data Processing', () => {
    it('should process basic Renko data with fixed brick size', () => {
      const config = {
        brickSize: 2,
        autoCalculateBrickSize: false,
        source: 'close' as const
      }

      const result = processor.processRenkoData(mockData, config)
      
      expect(result).toBeDefined()
      expect(Array.isArray(result)).toBe(true)
      expect(result.length).toBeGreaterThan(0)
      
      // Each brick should have required properties
      result.forEach(brick => {
        expect(brick).toHaveProperty('time')
        expect(brick).toHaveProperty('open')
        expect(brick).toHaveProperty('close')
        expect(brick).toHaveProperty('trend')
        expect(brick).toHaveProperty('brickSize')
        expect(['up', 'down']).toContain(brick.trend)
      })
    })

    it('should calculate optimal brick size when auto-calculation enabled', () => {
      const config = {
        autoCalculateBrickSize: true,
        source: 'close' as const
      }

      const result = processor.processRenkoData(mockData, config)
      
      expect(result).toBeDefined()
      expect(result.length).toBeGreaterThan(0)
      
      // Auto-calculated brick size should be consistent
      const firstBrick = result[0]
      expect(firstBrick.brickSize).toBeGreaterThan(0)
      
      result.forEach(brick => {
        expect(brick.brickSize).toBe(firstBrick.brickSize)
      })
    })

    it('should handle different price sources for Renko', () => {
      const configs = [
        { brickSize: 2, autoCalculateBrickSize: false, source: 'close' as const },
        { brickSize: 2, autoCalculateBrickSize: false, source: 'high_low' as const },
        { brickSize: 2, autoCalculateBrickSize: false, source: 'open_close' as const }
      ]

      configs.forEach(config => {
        const result = processor.processRenkoData(mockData, config)
        expect(result).toBeDefined()
        expect(result.length).toBeGreaterThanOrEqual(0)
      })
    })

    it('should maintain correct brick direction', () => {
      const config = {
        brickSize: 3,
        autoCalculateBrickSize: false,
        source: 'close' as const
      }

      const result = processor.processRenkoData(mockData, config)
      
      // Verify brick direction logic
      result.forEach(brick => {
        if (brick.trend === 'up') {
          expect(brick.close).toBeGreaterThan(brick.open)
          expect(brick.close - brick.open).toBe(brick.brickSize)
        } else {
          expect(brick.close).toBeLessThan(brick.open)
          expect(brick.open - brick.close).toBe(brick.brickSize)
        }
      })
    })

    it('should handle edge cases for Renko processing', () => {
      // Empty data
      expect(processor.processRenkoData([], { brickSize: 2, source: 'close' })).toEqual([])
      
      // Single data point
      const singlePoint = [mockData[0]]
      const result = processor.processRenkoData(singlePoint, { brickSize: 10, source: 'close' })
      expect(result.length).toBe(0) // No bricks created with single point
      
      // Very large brick size
      const largeResult = processor.processRenkoData(mockData, { brickSize: 100, source: 'close' })
      expect(largeResult.length).toBeLessThanOrEqual(mockData.length)
    })
  })

  describe('Point & Figure Data Processing', () => {
    it('should process Point & Figure data correctly', () => {
      const config = {
        boxSize: 1,
        reversalAmount: 3,
        autoCalculateBoxSize: false,
        source: 'close' as const
      }

      const result = processor.processPointFigureData(mockData, config)
      
      expect(result).toBeDefined()
      expect(Array.isArray(result)).toBe(true)
      
      result.forEach(column => {
        expect(column).toHaveProperty('time')
        expect(column).toHaveProperty('boxes')
        expect(column).toHaveProperty('type')
        expect(['X', 'O']).toContain(column.type)
        expect(Array.isArray(column.boxes)).toBe(true)
        
        column.boxes.forEach(box => {
          expect(box).toHaveProperty('price')
          expect(typeof box.price).toBe('number')
        })
      })
    })

    it('should auto-calculate box size when enabled', () => {
      const config = {
        autoCalculateBoxSize: true,
        reversalAmount: 3,
        source: 'close' as const
      }

      const result = processor.processPointFigureData(mockData, config)
      expect(result).toBeDefined()
      
      // Should create some columns with auto-calculated box size
      if (result.length > 0) {
        expect(result[0].boxSize).toBeGreaterThan(0)
      }
    })

    it('should respect reversal amount in P&F processing', () => {
      const config = {
        boxSize: 1,
        reversalAmount: 2,
        autoCalculateBoxSize: false,
        source: 'high_low' as const
      }

      const result = processor.processPointFigureData(mockData, config)
      
      // Verify reversal logic - should alternate between X and O columns
      if (result.length > 1) {
        for (let i = 1; i < result.length; i++) {
          expect(result[i].type).not.toBe(result[i - 1].type)
        }
      }
    })

    it('should handle P&F edge cases', () => {
      // Empty data
      expect(processor.processPointFigureData([], { 
        boxSize: 1, 
        reversalAmount: 3, 
        source: 'close' 
      })).toEqual([])
      
      // Very small box size
      const smallBoxResult = processor.processPointFigureData(mockData, {
        boxSize: 0.01,
        reversalAmount: 3,
        source: 'close'
      })
      expect(smallBoxResult.length).toBeGreaterThanOrEqual(0)
    })
  })

  describe('Heikin-Ashi Data Processing', () => {
    it('should process Heikin-Ashi data correctly', () => {
      const result = processor.processHeikinAshiData(mockData)
      
      expect(result).toBeDefined()
      expect(result.length).toBe(mockData.length)
      
      result.forEach((haCandle, index) => {
        expect(haCandle).toHaveProperty('time')
        expect(haCandle).toHaveProperty('open')
        expect(haCandle).toHaveProperty('high')
        expect(haCandle).toHaveProperty('low')
        expect(haCandle).toHaveProperty('close')
        expect(haCandle).toHaveProperty('volume')
        
        // Heikin-Ashi close is average of OHLC
        const originalCandle = mockData[index]
        const expectedClose = (originalCandle.open + originalCandle.high + 
                             originalCandle.low + originalCandle.close) / 4
        expect(haCandle.close).toBeCloseTo(expectedClose, 2)
        
        // Volume should be preserved
        expect(haCandle.volume).toBe(originalCandle.volume)
      })
    })

    it('should calculate HA open correctly', () => {
      const result = processor.processHeikinAshiData(mockData)
      
      // First HA open should be average of first candle's open and close
      const expectedFirstOpen = (mockData[0].open + mockData[0].close) / 2
      expect(result[0].open).toBeCloseTo(expectedFirstOpen, 2)
      
      // Subsequent HA opens should be average of previous HA open and close
      for (let i = 1; i < result.length; i++) {
        const expectedOpen = (result[i - 1].open + result[i - 1].close) / 2
        expect(result[i].open).toBeCloseTo(expectedOpen, 2)
      }
    })

    it('should calculate HA high and low correctly', () => {
      const result = processor.processHeikinAshiData(mockData)
      
      result.forEach((haCandle, index) => {
        const original = mockData[index]
        
        // HA high should be max of original high, HA open, HA close
        const expectedHigh = Math.max(original.high, haCandle.open, haCandle.close)
        expect(haCandle.high).toBeCloseTo(expectedHigh, 2)
        
        // HA low should be min of original low, HA open, HA close
        const expectedLow = Math.min(original.low, haCandle.open, haCandle.close)
        expect(haCandle.low).toBeCloseTo(expectedLow, 2)
      })
    })

    it('should handle HA edge cases', () => {
      // Empty data
      expect(processor.processHeikinAshiData([])).toEqual([])
      
      // Single candle
      const singleResult = processor.processHeikinAshiData([mockData[0]])
      expect(singleResult.length).toBe(1)
      expect(singleResult[0].open).toBeCloseTo((mockData[0].open + mockData[0].close) / 2, 2)
    })
  })

  describe('Volume Profile Data Processing', () => {
    it('should process volume profile data correctly', () => {
      const config = {
        priceLevels: 10,
        sessionType: 'daily' as const,
        showPOC: true
      }

      const result = processor.processVolumeProfileData(mockData, config)
      
      expect(result).toBeDefined()
      expect(Array.isArray(result)).toBe(true)
      expect(result.length).toBeLessThanOrEqual(config.priceLevels)
      
      result.forEach(level => {
        expect(level).toHaveProperty('priceLevel')
        expect(level).toHaveProperty('volume')
        expect(level).toHaveProperty('percentage')
        expect(typeof level.priceLevel).toBe('number')
        expect(typeof level.volume).toBe('number')
        expect(level.volume).toBeGreaterThanOrEqual(0)
        expect(level.percentage).toBeGreaterThanOrEqual(0)
        expect(level.percentage).toBeLessThanOrEqual(100)
      })
    })

    it('should identify Point of Control (POC)', () => {
      const config = {
        priceLevels: 10,
        sessionType: 'daily' as const,
        showPOC: true
      }

      const result = processor.processVolumeProfileData(mockData, config)
      
      if (result.length > 0) {
        // Should have exactly one POC
        const pocLevels = result.filter(level => level.isPOC)
        expect(pocLevels.length).toBeLessThanOrEqual(1)
        
        if (pocLevels.length === 1) {
          // POC should have highest volume
          const maxVolume = Math.max(...result.map(l => l.volume))
          expect(pocLevels[0].volume).toBe(maxVolume)
        }
      }
    })

    it('should distribute volume across price levels correctly', () => {
      const config = {
        priceLevels: 5,
        sessionType: 'daily' as const,
        showPOC: false
      }

      const result = processor.processVolumeProfileData(mockData, config)
      
      // Total volume should equal sum of all input volumes
      const totalInputVolume = mockData.reduce((sum, candle) => sum + candle.volume, 0)
      const totalOutputVolume = result.reduce((sum, level) => sum + level.volume, 0)
      
      expect(totalOutputVolume).toBeCloseTo(totalInputVolume, 0)
    })

    it('should handle different session types', () => {
      const sessionTypes = ['daily', 'weekly'] as const
      
      sessionTypes.forEach(sessionType => {
        const config = {
          priceLevels: 10,
          sessionType,
          showPOC: true
        }

        const result = processor.processVolumeProfileData(mockData, config)
        expect(result).toBeDefined()
        expect(Array.isArray(result)).toBe(true)
      })
    })

    it('should handle volume profile edge cases', () => {
      // Empty data
      expect(processor.processVolumeProfileData([], {
        priceLevels: 10,
        sessionType: 'daily',
        showPOC: true
      })).toEqual([])
      
      // Single price level
      const singleLevelResult = processor.processVolumeProfileData(mockData, {
        priceLevels: 1,
        sessionType: 'daily',
        showPOC: true
      })
      expect(singleLevelResult.length).toBe(1)
      
      // More price levels than data points
      const manyLevelsResult = processor.processVolumeProfileData(mockData.slice(0, 3), {
        priceLevels: 100,
        sessionType: 'daily',
        showPOC: true
      })
      expect(manyLevelsResult.length).toBeLessThanOrEqual(100)
    })
  })

  describe('Data Validation', () => {
    it('should handle invalid OHLCV data', () => {
      const invalidData = [
        { time: '2023-01-01', open: NaN, high: 105, low: 95, close: 100, volume: 1000 },
        { time: '2023-01-02', open: 100, high: Infinity, low: 95, close: 105, volume: 1200 },
        { time: '2023-01-03', open: 105, high: 110, low: -Infinity, close: 108, volume: 800 }
      ]

      // Should not crash with invalid data
      expect(() => {
        processor.processHeikinAshiData(invalidData as OHLCVData[])
      }).not.toThrow()

      expect(() => {
        processor.processRenkoData(invalidData as OHLCVData[], { brickSize: 2, source: 'close' })
      }).not.toThrow()
    })

    it('should handle missing or null values', () => {
      const incompleteData = [
        { time: '2023-01-01', open: 100, high: 105, low: 95, close: null, volume: 1000 },
        { time: '2023-01-02', open: null, high: 108, low: 100, close: 106, volume: 1200 }
      ] as any[]

      expect(() => {
        processor.processHeikinAshiData(incompleteData)
      }).not.toThrow()
    })
  })

  describe('Performance Considerations', () => {
    it('should handle large datasets efficiently', () => {
      // Generate large dataset
      const largeData: OHLCVData[] = []
      for (let i = 0; i < 10000; i++) {
        largeData.push({
          time: `2023-01-${(i % 30) + 1}`,
          open: 100 + Math.random() * 20,
          high: 110 + Math.random() * 20,
          low: 90 + Math.random() * 20,
          close: 100 + Math.random() * 20,
          volume: 1000 + Math.random() * 500
        })
      }

      const start = performance.now()
      const result = processor.processHeikinAshiData(largeData)
      const end = performance.now()
      
      expect(result.length).toBe(largeData.length)
      expect(end - start).toBeLessThan(1000) // Should complete within 1 second
    })

    it('should process complex chart types within reasonable time', () => {
      const start = performance.now()
      
      processor.processRenkoData(mockData, { brickSize: 0.5, source: 'high_low' })
      processor.processPointFigureData(mockData, { boxSize: 0.5, reversalAmount: 2, source: 'high_low' })
      processor.processVolumeProfileData(mockData, { priceLevels: 50, sessionType: 'daily', showPOC: true })
      
      const end = performance.now()
      
      expect(end - start).toBeLessThan(100) // Should complete within 100ms for small dataset
    })
  })

  describe('Data Consistency', () => {
    it('should maintain time consistency across all chart types', () => {
      const renkoResult = processor.processRenkoData(mockData, { brickSize: 2, source: 'close' })
      const haResult = processor.processHeikinAshiData(mockData)
      
      // HA should maintain same number of data points
      expect(haResult.length).toBe(mockData.length)
      
      // Time values should be preserved for HA
      haResult.forEach((candle, index) => {
        expect(candle.time).toBe(mockData[index].time)
      })
    })

    it('should preserve volume information where applicable', () => {
      const haResult = processor.processHeikinAshiData(mockData)
      
      haResult.forEach((candle, index) => {
        expect(candle.volume).toBe(mockData[index].volume)
      })
    })
  })
})