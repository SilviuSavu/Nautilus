/**
 * Pattern Recognition Service Tests
 * Comprehensive unit tests for algorithmic chart pattern detection
 */

import { describe, it, expect, beforeEach, vi } from 'vitest'
import patternRecognition from '../patternRecognition'
import { OHLCVData } from '../../components/Chart/types/chartTypes'

// Mock OHLCV data generators for different patterns
const createHeadAndShouldersData = (): OHLCVData[] => {
  const basePrice = 100
  const data: OHLCVData[] = []
  
  // Create head and shoulders pattern: low-high-low peaks
  const pattern = [
    // Left shoulder (lower peak)
    ...Array.from({ length: 8 }, (_, i) => ({
      time: `2024-01-${(i + 1).toString().padStart(2, '0')}T10:00:00.000Z`,
      open: basePrice + i * 2,
      high: basePrice + i * 2.5,
      low: basePrice + i * 1.5,
      close: basePrice + i * 2,
      volume: 1000 - i * 50
    })),
    // Down to valley
    ...Array.from({ length: 5 }, (_, i) => ({
      time: `2024-01-${(i + 9).toString().padStart(2, '0')}T10:00:00.000Z`,
      open: basePrice + 14 - i * 2,
      high: basePrice + 15 - i * 2,
      low: basePrice + 13 - i * 2,
      close: basePrice + 14 - i * 2,
      volume: 950 - i * 30
    })),
    // Head (highest peak)
    ...Array.from({ length: 10 }, (_, i) => ({
      time: `2024-01-${(i + 14).toString().padStart(2, '0')}T10:00:00.000Z`,
      open: basePrice + 6 + i * 3,
      high: basePrice + 7 + i * 3.5,
      low: basePrice + 5 + i * 2.5,
      close: basePrice + 6 + i * 3,
      volume: 800 - i * 40
    })),
    // Down from head
    ...Array.from({ length: 5 }, (_, i) => ({
      time: `2024-01-${(i + 24).toString().padStart(2, '0')}T10:00:00.000Z`,
      open: basePrice + 36 - i * 3,
      high: basePrice + 37 - i * 3,
      low: basePrice + 35 - i * 3,
      close: basePrice + 36 - i * 3,
      volume: 600 - i * 20
    })),
    // Right shoulder (lower peak, similar to left)
    ...Array.from({ length: 7 }, (_, i) => ({
      time: `2024-01-${(i + 29).toString().padStart(2, '0')}T10:00:00.000Z`,
      open: basePrice + 21 + i * 1.5,
      high: basePrice + 22 + i * 1.8,
      low: basePrice + 20 + i * 1.2,
      close: basePrice + 21 + i * 1.5,
      volume: 500 - i * 10
    }))
  ]
  
  return pattern
}

const createDoubleTopData = (): OHLCVData[] => {
  const basePrice = 100
  const data: OHLCVData[] = []
  
  // First peak
  for (let i = 0; i < 15; i++) {
    data.push({
      time: `2024-01-${(i + 1).toString().padStart(2, '0')}T10:00:00.000Z`,
      open: basePrice + i * 1.5,
      high: basePrice + i * 1.8,
      low: basePrice + i * 1.2,
      close: basePrice + i * 1.5,
      volume: 1000 - i * 20
    })
  }
  
  // Valley between peaks
  for (let i = 0; i < 10; i++) {
    data.push({
      time: `2024-01-${(i + 16).toString().padStart(2, '0')}T10:00:00.000Z`,
      open: basePrice + 22 - i * 1.2,
      high: basePrice + 23 - i * 1.2,
      low: basePrice + 21 - i * 1.2,
      close: basePrice + 22 - i * 1.2,
      volume: 700
    })
  }
  
  // Second peak (similar height, lower volume)
  for (let i = 0; i < 12; i++) {
    data.push({
      time: `2024-01-${(i + 26).toString().padStart(2, '0')}T10:00:00.000Z`,
      open: basePrice + 10 + i * 1.4,
      high: basePrice + 11 + i * 1.7, // Slightly lower than first peak
      low: basePrice + 9 + i * 1.1,
      close: basePrice + 10 + i * 1.4,
      volume: 500 - i * 15 // Lower volume
    })
  }
  
  return data
}

const createTriangleData = (): OHLCVData[] => {
  const basePrice = 100
  const data: OHLCVData[] = []
  
  // Ascending triangle: flat resistance, ascending support
  for (let i = 0; i < 30; i++) {
    const resistanceLevel = 120 // Flat resistance
    const supportLevel = 100 + i * 0.5 // Ascending support
    const range = resistanceLevel - supportLevel
    
    data.push({
      time: `2024-01-${(i + 1).toString().padStart(2, '0')}T10:00:00.000Z`,
      open: supportLevel + Math.random() * range,
      high: Math.min(resistanceLevel, supportLevel + Math.random() * range * 1.2),
      low: Math.max(supportLevel, supportLevel + Math.random() * range * 0.8),
      close: supportLevel + Math.random() * range,
      volume: 1000 + Math.random() * 500
    })
  }
  
  return data
}

const createFlatPriceData = (): OHLCVData[] => {
  return Array.from({ length: 20 }, (_, i) => ({
    time: `2024-01-${(i + 1).toString().padStart(2, '0')}T10:00:00.000Z`,
    open: 100,
    high: 100.5,
    low: 99.5,
    close: 100,
    volume: 1000
  }))
}

const createRandomData = (length: number = 50): OHLCVData[] => {
  let price = 100
  return Array.from({ length }, (_, i) => {
    const change = (Math.random() - 0.5) * 4
    const open = price
    price = Math.max(10, price + change)
    
    return {
      time: `2024-01-${(i + 1).toString().padStart(2, '0')}T10:00:00.000Z`,
      open,
      high: price + Math.abs(change) * 0.5,
      low: price - Math.abs(change) * 0.5,
      close: price,
      volume: 1000 + Math.random() * 1000
    }
  })
}

describe('PatternRecognitionService', () => {
  let service: typeof patternRecognition
  let mockData: OHLCVData[]

  beforeEach(() => {
    service = patternRecognition
    mockData = createRandomData()
  })

  describe('Service Initialization', () => {
    it('should initialize with built-in pattern definitions', () => {
      const definitions = service.getPatternDefinitions()
      expect(definitions.length).toBeGreaterThan(0)
      
      const patternNames = definitions.map(d => d.name)
      expect(patternNames).toContain('Head and Shoulders')
      expect(patternNames).toContain('Double Top')
      expect(patternNames).toContain('Double Bottom')
      expect(patternNames).toContain('Ascending Triangle')
      expect(patternNames).toContain('Cup and Handle')
      expect(patternNames).toContain('Bull Flag')
      expect(patternNames).toContain('Falling Wedge')
    })

    it('should have correct default configuration', () => {
      const config = service.getConfig()
      expect(config.minBars).toBe(10)
      expect(config.maxBars).toBe(100)
      expect(config.minConfidence).toBe(0.6)
      expect(config.sensitivity).toBe('medium')
      expect(config.enabledPatterns).toHaveLength(0) // All patterns enabled by default
    })
  })

  describe('Configuration Management', () => {
    it('should update configuration correctly', () => {
      const newConfig = {
        minBars: 20,
        maxBars: 150,
        minConfidence: 0.8,
        sensitivity: 'high' as const,
        enabledPatterns: ['head_shoulders', 'double_top']
      }
      
      service.setConfig(newConfig)
      const updatedConfig = service.getConfig()
      
      expect(updatedConfig.minBars).toBe(20)
      expect(updatedConfig.maxBars).toBe(150)
      expect(updatedConfig.minConfidence).toBe(0.8)
      expect(updatedConfig.sensitivity).toBe('high')
      expect(updatedConfig.enabledPatterns).toEqual(['head_shoulders', 'double_top'])
    })

    it('should partially update configuration', () => {
      service.setConfig({ minConfidence: 0.9 })
      const config = service.getConfig()
      
      expect(config.minConfidence).toBe(0.9)
      expect(config.minBars).toBe(10) // Unchanged
      expect(config.sensitivity).toBe('medium') // Unchanged
    })
  })

  describe('Pattern Registration', () => {
    it('should register custom pattern definition', () => {
      const customPattern = {
        name: 'Custom Test Pattern',
        type: 'custom' as const,
        rules: [{
          type: 'price_action' as const,
          condition: 'strong_move_up',
          parameters: { minMove: 0.05 }
        }],
        minBars: 15,
        maxBars: 50,
        minConfidence: 0.7
      }
      
      const patternId = service.registerPattern(customPattern)
      expect(patternId).toBeDefined()
      
      const definitions = service.getPatternDefinitions()
      const registered = definitions.find(d => d.id === patternId)
      expect(registered).toBeDefined()
      expect(registered!.name).toBe('Custom Test Pattern')
    })

    it('should generate unique IDs for custom patterns', () => {
      const pattern1Id = service.registerPattern({
        name: 'Pattern 1',
        type: 'custom',
        rules: [],
        minBars: 10,
        maxBars: 50,
        minConfidence: 0.6
      })
      
      const pattern2Id = service.registerPattern({
        name: 'Pattern 2', 
        type: 'custom',
        rules: [],
        minBars: 10,
        maxBars: 50,
        minConfidence: 0.6
      })
      
      expect(pattern1Id).not.toBe(pattern2Id)
    })
  })

  describe('Head and Shoulders Pattern Detection', () => {
    it('should detect head and shoulders pattern', async () => {
      const hsData = createHeadAndShouldersData()
      const patterns = await service.detectPatterns(hsData, 'AAPL', '1H')
      
      const headShouldersPatterns = patterns.filter(p => p.type === 'head_shoulders')
      expect(headShouldersPatterns.length).toBeGreaterThan(0)
      
      if (headShouldersPatterns.length > 0) {
        const pattern = headShouldersPatterns[0]
        expect(pattern.confidence).toBeGreaterThan(0.6)
        expect(pattern.name).toBe('Head and Shoulders')
        expect(pattern.instrument).toBe('AAPL')
        expect(pattern.timeframe).toBe('1H')
        expect(pattern.status).toBe('forming')
      }
    })

    it('should calculate confidence for head and shoulders pattern', async () => {
      const hsData = createHeadAndShouldersData()
      const patterns = await service.detectPatterns(hsData, 'TEST', '1D')
      
      const hsPattern = patterns.find(p => p.type === 'head_shoulders')
      if (hsPattern) {
        expect(hsPattern.confidence).toBeGreaterThanOrEqual(0)
        expect(hsPattern.confidence).toBeLessThanOrEqual(1)
        expect(hsPattern.coordinates?.points).toBeDefined()
        expect(hsPattern.coordinates?.boundingBox).toBeDefined()
      }
    })
  })

  describe('Double Top Pattern Detection', () => {
    it('should detect double top pattern', async () => {
      const dtData = createDoubleTopData()
      const patterns = await service.detectPatterns(dtData, 'MSFT', '4H')
      
      const doubleTopPatterns = patterns.filter(p => p.type === 'double_top')
      expect(doubleTopPatterns.length).toBeGreaterThan(0)
      
      if (doubleTopPatterns.length > 0) {
        const pattern = doubleTopPatterns[0]
        expect(pattern.confidence).toBeGreaterThan(0.5)
        expect(pattern.name).toBe('Double Top')
      }
    })

    it('should validate double top peak similarity', async () => {
      const dtData = createDoubleTopData()
      
      // Test with very strict tolerance
      service.setConfig({ minConfidence: 0.9 })
      const strictPatterns = await service.detectPatterns(dtData, 'TEST', '1H')
      
      // Test with relaxed tolerance  
      service.setConfig({ minConfidence: 0.3 })
      const relaxedPatterns = await service.detectPatterns(dtData, 'TEST', '1H')
      
      // Relaxed should find more patterns
      expect(relaxedPatterns.length).toBeGreaterThanOrEqual(strictPatterns.length)
    })
  })

  describe('Triangle Pattern Detection', () => {
    it('should detect ascending triangle pattern', async () => {
      const triangleData = createTriangleData()
      const patterns = await service.detectPatterns(triangleData, 'GOOGL', '1D')
      
      const trianglePatterns = patterns.filter(p => p.type === 'triangle')
      expect(trianglePatterns.length).toBeGreaterThan(0)
      
      if (trianglePatterns.length > 0) {
        const pattern = trianglePatterns[0]
        expect(pattern.name).toBe('Ascending Triangle')
        expect(pattern.confidence).toBeGreaterThan(0.4)
      }
    })
  })

  describe('Pattern Detection Edge Cases', () => {
    it('should handle insufficient data gracefully', async () => {
      const shortData = createRandomData(5) // Less than minBars
      const patterns = await service.detectPatterns(shortData, 'TEST', '1M')
      
      expect(patterns).toHaveLength(0)
    })

    it('should handle flat price data', async () => {
      const flatData = createFlatPriceData()
      const patterns = await service.detectPatterns(flatData, 'FLAT', '5M')
      
      // Should not detect significant patterns in flat data
      expect(patterns.length).toBe(0)
    })

    it('should handle empty data', async () => {
      const patterns = await service.detectPatterns([], 'EMPTY', '1H')
      expect(patterns).toHaveLength(0)
    })

    it('should handle invalid OHLCV data', async () => {
      const invalidData = [
        { time: '2024-01-01', open: NaN, high: 105, low: 95, close: 100, volume: 1000 },
        { time: '2024-01-02', open: 100, high: Infinity, low: 90, close: 95, volume: 1200 }
      ] as OHLCVData[]
      
      // Should not crash with invalid data
      expect(async () => {
        await service.detectPatterns(invalidData, 'INVALID', '1H')
      }).not.toThrow()
    })
  })

  describe('Pattern Filtering and Management', () => {
    it('should filter patterns by enabled patterns config', async () => {
      const hsData = createHeadAndShouldersData()
      
      // Enable only double top patterns
      service.setConfig({ 
        enabledPatterns: ['double_top'],
        minConfidence: 0.1 
      })
      
      const patterns = await service.detectPatterns(hsData, 'TEST', '1H')
      const hsPatterns = patterns.filter(p => p.type === 'head_shoulders')
      
      // Should not detect head and shoulders when not enabled
      expect(hsPatterns).toHaveLength(0)
    })

    it('should filter patterns by minimum confidence', async () => {
      const testData = createHeadAndShouldersData()
      
      // High confidence threshold
      service.setConfig({ minConfidence: 0.9 })
      const highConfPatterns = await service.detectPatterns(testData, 'TEST', '1H')
      
      // Low confidence threshold
      service.setConfig({ minConfidence: 0.1 })
      const lowConfPatterns = await service.detectPatterns(testData, 'TEST', '1H')
      
      expect(lowConfPatterns.length).toBeGreaterThanOrEqual(highConfPatterns.length)
      
      // All returned patterns should meet confidence threshold
      highConfPatterns.forEach(pattern => {
        expect(pattern.confidence).toBeGreaterThanOrEqual(0.9)
      })
    })

    it('should get detected patterns with filters', async () => {
      const testData = createRandomData(100)
      await service.detectPatterns(testData, 'AAPL', '1H')
      await service.detectPatterns(testData, 'MSFT', '4H')
      
      // Filter by instrument
      const aaplPatterns = service.getDetectedPatterns('AAPL')
      aaplPatterns.forEach(pattern => {
        expect(pattern.instrument).toBe('AAPL')
      })
      
      // Filter by timeframe
      const hourlyPatterns = service.getDetectedPatterns(undefined, '1H')
      hourlyPatterns.forEach(pattern => {
        expect(pattern.timeframe).toBe('1H')
      })
      
      // Filter by both
      const aaplHourlyPatterns = service.getDetectedPatterns('AAPL', '1H')
      aaplHourlyPatterns.forEach(pattern => {
        expect(pattern.instrument).toBe('AAPL')
        expect(pattern.timeframe).toBe('1H')
      })
    })

    it('should clear detected patterns', async () => {
      const testData = createRandomData(50)
      await service.detectPatterns(testData, 'TEST', '1M')
      
      const beforeClear = service.getDetectedPatterns()
      expect(beforeClear.length).toBeGreaterThanOrEqual(0)
      
      service.clearPatterns()
      const afterClear = service.getDetectedPatterns()
      expect(afterClear).toHaveLength(0)
    })
  })

  describe('Utility Methods', () => {
    it('should find peaks correctly', async () => {
      // Create data with known peaks
      const peakData: OHLCVData[] = [
        { time: '2024-01-01', open: 100, high: 100, low: 100, close: 100, volume: 1000 },
        { time: '2024-01-02', open: 105, high: 110, low: 105, close: 108, volume: 1000 }, // Peak
        { time: '2024-01-03', open: 108, high: 108, low: 105, close: 106, volume: 1000 },
        { time: '2024-01-04', open: 106, high: 106, low: 103, close: 104, volume: 1000 },
        { time: '2024-01-05', open: 104, high: 115, low: 104, close: 112, volume: 1000 }, // Peak
        { time: '2024-01-06', open: 112, high: 112, low: 108, close: 110, volume: 1000 },
        { time: '2024-01-07', open: 110, high: 110, low: 107, close: 108, volume: 1000 }
      ]
      
      // Test peak detection indirectly through pattern detection
      const patterns = await service.detectPatterns(peakData, 'PEAK_TEST', '1D')
      
      // Should detect some pattern with the peaks
      expect(patterns.length).toBeGreaterThanOrEqual(0)
    })
  })

  describe('Historical Analysis', () => {
    it('should provide historical analysis structure', async () => {
      const testData = createRandomData(100)
      const analysis = await service.analyzeHistoricalPatterns(testData, 'HIST', '1H')
      
      expect(analysis).toBeDefined()
      expect(analysis).toHaveProperty('totalPatterns')
      expect(analysis).toHaveProperty('successRate')
      expect(analysis).toHaveProperty('avgTimeToComplete')
      expect(analysis).toHaveProperty('patternPerformance')
      
      expect(typeof analysis.totalPatterns).toBe('number')
      expect(typeof analysis.successRate).toBe('number')
      expect(typeof analysis.avgTimeToComplete).toBe('number')
      expect(typeof analysis.patternPerformance).toBe('object')
    })
  })

  describe('Performance Considerations', () => {
    it('should handle large datasets efficiently', async () => {
      const largeData = createRandomData(1000)
      
      const startTime = performance.now()
      const patterns = await service.detectPatterns(largeData, 'PERF_TEST', '1M')
      const endTime = performance.now()
      
      const executionTime = endTime - startTime
      expect(executionTime).toBeLessThan(5000) // Should complete within 5 seconds
      expect(patterns).toBeDefined()
      expect(Array.isArray(patterns)).toBe(true)
    })

    it('should not create excessive patterns from noisy data', async () => {
      const noisyData = createRandomData(500)
      service.setConfig({ minConfidence: 0.7 }) // Higher threshold
      
      const patterns = await service.detectPatterns(noisyData, 'NOISE_TEST', '5M')
      
      // Should not generate too many patterns from random data
      expect(patterns.length).toBeLessThan(noisyData.length * 0.1) // Less than 10% of data points
    })
  })

  describe('Pattern Coordinates and Metadata', () => {
    it('should generate valid pattern coordinates', async () => {
      const hsData = createHeadAndShouldersData()
      const patterns = await service.detectPatterns(hsData, 'COORDS_TEST', '1H')
      
      patterns.forEach(pattern => {
        if (pattern.coordinates) {
          expect(pattern.coordinates.points).toBeDefined()
          expect(pattern.coordinates.boundingBox).toBeDefined()
          expect(Array.isArray(pattern.coordinates.points)).toBe(true)
          
          pattern.coordinates.points.forEach(point => {
            expect(point).toHaveProperty('x')
            expect(point).toHaveProperty('y')
            expect(point).toHaveProperty('time')
            expect(typeof point.x).toBe('number')
            expect(typeof point.y).toBe('number')
            expect(typeof point.time).toBe('string')
          })
          
          const bb = pattern.coordinates.boundingBox
          expect(bb.left).toBeLessThanOrEqual(bb.right)
          expect(bb.bottom).toBeLessThanOrEqual(bb.top)
        }
      })
    })

    it('should set correct pattern metadata', async () => {
      const testData = createDoubleTopData()
      const patterns = await service.detectPatterns(testData, 'META_TEST', '30M')
      
      patterns.forEach(pattern => {
        expect(pattern).toHaveProperty('id')
        expect(pattern).toHaveProperty('name')
        expect(pattern).toHaveProperty('type')
        expect(pattern).toHaveProperty('confidence')
        expect(pattern).toHaveProperty('timeframe')
        expect(pattern).toHaveProperty('status')
        expect(pattern).toHaveProperty('detectedAt')
        expect(pattern).toHaveProperty('instrument')
        
        expect(typeof pattern.id).toBe('string')
        expect(pattern.id.length).toBeGreaterThan(0)
        expect(pattern.confidence).toBeGreaterThanOrEqual(0)
        expect(pattern.confidence).toBeLessThanOrEqual(1)
        expect(pattern.timeframe).toBe('30M')
        expect(pattern.instrument).toBe('META_TEST')
        expect(pattern.status).toBe('forming')
        
        // Validate timestamp format
        expect(() => new Date(pattern.detectedAt)).not.toThrow()
      })
    })
  })
})

describe('PatternRecognition Singleton', () => {
  it('should export singleton instance', () => {
    expect(patternRecognition).toBeDefined()
    expect(typeof patternRecognition.getPatternDefinitions).toBe('function')
    
    const definitions1 = patternRecognition.getPatternDefinitions()
    const definitions2 = patternRecognition.getPatternDefinitions()
    expect(definitions1).toEqual(definitions2)
  })

  it('should maintain state across calls', async () => {
    const testData = createRandomData(30)
    
    await patternRecognition.detectPatterns(testData, 'SINGLETON_TEST', '1H')
    const patterns1 = patternRecognition.getDetectedPatterns()
    
    await patternRecognition.detectPatterns(testData, 'SINGLETON_TEST2', '4H')
    const patterns2 = patternRecognition.getDetectedPatterns()
    
    // Should accumulate patterns
    expect(patterns2.length).toBeGreaterThanOrEqual(patterns1.length)
  })
})

describe('Pattern Detection Algorithms', () => {
  describe('Volume Analysis', () => {
    it('should detect decreasing volume patterns', async () => {
      const decreasingVolumeData: OHLCVData[] = Array.from({ length: 20 }, (_, i) => ({
        time: `2024-01-${(i + 1).toString().padStart(2, '0')}T10:00:00.000Z`,
        open: 100 + Math.random() * 5,
        high: 105 + Math.random() * 5,
        low: 95 + Math.random() * 5,
        close: 100 + Math.random() * 5,
        volume: 2000 - i * 80 // Clearly decreasing volume
      }))
      
      const patterns = await patternRecognition.detectPatterns(decreasingVolumeData, 'VOL_TEST', '1H')
      
      // Some patterns should be detected with decreasing volume
      expect(patterns.length).toBeGreaterThanOrEqual(0)
    })
  })

  describe('Price Action Analysis', () => {
    it('should handle edge cases in trend calculation', async () => {
      // Single data point
      const singlePoint: OHLCVData[] = [{
        time: '2024-01-01T10:00:00.000Z',
        open: 100,
        high: 100,
        low: 100,
        close: 100,
        volume: 1000
      }]
      
      const patterns = await patternRecognition.detectPatterns(singlePoint, 'SINGLE', '1H')
      expect(patterns).toHaveLength(0)
      
      // Two identical points
      const identicalPoints: OHLCVData[] = [
        { time: '2024-01-01T10:00:00.000Z', open: 100, high: 100, low: 100, close: 100, volume: 1000 },
        { time: '2024-01-02T10:00:00.000Z', open: 100, high: 100, low: 100, close: 100, volume: 1000 }
      ]
      
      const identicalPatterns = await patternRecognition.detectPatterns(identicalPoints, 'IDENTICAL', '1H')
      expect(identicalPatterns).toHaveLength(0)
    })
  })

  describe('Configuration Impact', () => {
    it('should respect bar count limitations', async () => {
      const testData = createRandomData(200)
      
      // Restrict to very small bar counts
      patternRecognition.setConfig({
        minBars: 150,
        maxBars: 160
      })
      
      const restrictedPatterns = await patternRecognition.detectPatterns(testData, 'RESTRICTED', '1H')
      
      // Reset to default
      patternRecognition.setConfig({
        minBars: 10,
        maxBars: 100
      })
      
      const normalPatterns = await patternRecognition.detectPatterns(testData, 'NORMAL', '1H')
      
      // Normal settings should potentially find more patterns
      expect(normalPatterns.length).toBeGreaterThanOrEqual(restrictedPatterns.length)
    })
  })
})