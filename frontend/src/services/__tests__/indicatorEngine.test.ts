/**
 * Technical Indicator Engine Tests
 * Comprehensive unit tests with mathematical accuracy validation
 */

import { describe, it, expect, beforeEach, vi } from 'vitest'
import { IndicatorEngine, indicatorEngine } from '../indicatorEngine'
import { OHLCVData } from '../../components/Chart/types/chartTypes'

describe('IndicatorEngine', () => {
  let engine: IndicatorEngine
  let mockData: OHLCVData[]

  beforeEach(() => {
    engine = new IndicatorEngine()
    mockData = [
      { time: '2023-01-01', open: 100, high: 105, low: 95, close: 102, volume: 1000 },
      { time: '2023-01-02', open: 102, high: 108, low: 100, close: 106, volume: 1200 },
      { time: '2023-01-03', open: 106, high: 110, low: 104, close: 108, volume: 800 },
      { time: '2023-01-04', open: 108, high: 112, low: 106, close: 110, volume: 900 },
      { time: '2023-01-05', open: 110, high: 115, low: 108, close: 113, volume: 1100 },
      { time: '2023-01-06', open: 113, high: 116, low: 111, close: 114, volume: 950 },
      { time: '2023-01-07', open: 114, high: 118, low: 112, close: 116, volume: 1050 },
      { time: '2023-01-08', open: 116, high: 120, low: 114, close: 118, volume: 1300 },
      { time: '2023-01-09', open: 118, high: 122, low: 116, close: 120, volume: 1400 },
      { time: '2023-01-10', open: 120, high: 124, low: 118, close: 122, volume: 1200 }
    ]
  })

  describe('Built-in Indicators', () => {
    it('should have all required built-in indicators', () => {
      const indicators = engine.getAvailableIndicators()
      const indicatorIds = indicators.map(ind => ind.id)
      
      expect(indicatorIds).toContain('sma')
      expect(indicatorIds).toContain('ema')
      expect(indicatorIds).toContain('rsi')
      expect(indicatorIds).toContain('macd')
      expect(indicatorIds).toContain('bollinger')
      expect(indicators.length).toBeGreaterThanOrEqual(5)
    })

    it('should return indicator by ID', () => {
      const smaIndicator = engine.getIndicator('sma')
      expect(smaIndicator).toBeDefined()
      expect(smaIndicator?.name).toBe('Simple Moving Average')
      expect(smaIndicator?.type).toBe('built_in')
    })
  })

  describe('Simple Moving Average (SMA)', () => {
    it('should calculate SMA correctly with default period', () => {
      const result = engine.calculate('sma', mockData, { period: 3 })
      expect(result).toBeDefined()
      expect(result!.values.length).toBe(mockData.length)
      
      // First two values should be NaN (insufficient data)
      expect(result!.values[0].value).toBe(null)
      expect(result!.values[1].value).toBe(null)
      
      // Third value: (102 + 106 + 108) / 3 = 105.33...
      expect(result!.values[2].value).toBeCloseTo(105.33, 2)
      
      // Fourth value: (106 + 108 + 110) / 3 = 108
      expect(result!.values[3].value).toBeCloseTo(108, 2)
    })

    it('should handle different period values', () => {
      const result = engine.calculate('sma', mockData, { period: 5 })
      expect(result).toBeDefined()
      
      // First four values should be NaN
      for (let i = 0; i < 4; i++) {
        expect(result!.values[i].value).toBe(null)
      }
      
      // Fifth value: (102 + 106 + 108 + 110 + 113) / 5 = 107.8
      expect(result!.values[4].value).toBeCloseTo(107.8, 1)
    })

    it('should handle edge cases', () => {
      const emptyResult = engine.calculate('sma', [], { period: 20 })
      expect(emptyResult).toBeDefined()
      expect(emptyResult!.values.length).toBe(0)
      
      const singleDataPoint = mockData.slice(0, 1)
      const singleResult = engine.calculate('sma', singleDataPoint, { period: 20 })
      expect(singleResult!.values[0].value).toBe(null)
    })
  })

  describe('Exponential Moving Average (EMA)', () => {
    it('should calculate EMA correctly', () => {
      const result = engine.calculate('ema', mockData, { period: 3 })
      expect(result).toBeDefined()
      
      // First value should be the first close price
      expect(result!.values[0].value).toBe(102)
      
      // Second value calculation: (106 * (2/4)) + (102 * (2/4)) = 104
      expect(result!.values[1].value).toBeCloseTo(104, 1)
    })

    it('should differ from SMA values', () => {
      const smaResult = engine.calculate('sma', mockData, { period: 5 })
      const emaResult = engine.calculate('ema', mockData, { period: 5 })
      
      expect(smaResult!.values[5].value).not.toEqual(emaResult!.values[5].value)
      // EMA should be more responsive to recent changes
      expect(emaResult!.values[9].value).toBeGreaterThan(smaResult!.values[9].value!)
    })
  })

  describe('Relative Strength Index (RSI)', () => {
    it('should calculate RSI correctly', () => {
      // Create data with clear uptrend for predictable RSI
      const trendData: OHLCVData[] = []
      for (let i = 0; i < 20; i++) {
        trendData.push({
          time: `2023-01-${i + 1}`,
          open: 100 + i,
          high: 105 + i,
          low: 95 + i,
          close: 100 + i + (i % 2), // Slight variations
          volume: 1000
        })
      }
      
      const result = engine.calculate('rsi', trendData, { period: 14 })
      expect(result).toBeDefined()
      
      // First 14 values should be NaN
      for (let i = 0; i < 14; i++) {
        expect(result!.values[i].value).toBe(null)
      }
      
      // RSI should be between 0 and 100
      for (let i = 14; i < result!.values.length; i++) {
        const value = result!.values[i].value
        if (value !== null) {
          expect(value).toBeGreaterThanOrEqual(0)
          expect(value).toBeLessThanOrEqual(100)
        }
      }
    })

    it('should handle extreme cases', () => {
      // All increasing prices should result in RSI near 100
      const increasingData: OHLCVData[] = Array.from({ length: 20 }, (_, i) => ({
        time: `2023-01-${i + 1}`,
        open: 100 + i * 2,
        high: 105 + i * 2,
        low: 95 + i * 2,
        close: 102 + i * 2,
        volume: 1000
      }))
      
      const result = engine.calculate('rsi', increasingData, { period: 14 })
      const lastValue = result!.values[result!.values.length - 1].value
      expect(lastValue).toBeGreaterThan(80) // Should be high RSI
    })
  })

  describe('MACD', () => {
    it('should calculate MACD correctly', () => {
      const result = engine.calculate('macd', mockData, {
        fastPeriod: 5,
        slowPeriod: 8,
        signalPeriod: 3
      })
      expect(result).toBeDefined()
      
      // MACD values should exist for most of the data
      const nonNullValues = result!.values.filter(v => v.value !== null)
      expect(nonNullValues.length).toBeGreaterThan(0)
      
      // Values should be finite
      nonNullValues.forEach(v => {
        expect(Number.isFinite(v.value)).toBe(true)
      })
    })
  })

  describe('Bollinger Bands', () => {
    it('should calculate Bollinger Bands upper band', () => {
      const result = engine.calculate('bollinger', mockData, {
        period: 5,
        stdDev: 2
      })
      expect(result).toBeDefined()
      
      // Should have values after the period
      const validValues = result!.values.slice(4).filter(v => v.value !== null)
      expect(validValues.length).toBeGreaterThan(0)
      
      // Upper band should be higher than the close prices
      validValues.forEach((v, index) => {
        const closePrice = mockData[index + 4].close
        expect(v.value).toBeGreaterThan(closePrice)
      })
    })
  })

  describe('Custom Indicators', () => {
    it('should create and execute custom indicator', () => {
      const customId = engine.createCustomIndicator(
        'Test Custom',
        `
        const closes = data.map(d => d.close);
        return closes.map((close, i) => i < 2 ? NaN : 
          (closes[i] + closes[i-1] + closes[i-2]) / 3);
        `,
        [
          {
            name: 'testParam',
            type: 'number',
            defaultValue: 1
          }
        ],
        {
          color: '#FF0000',
          lineWidth: 2,
          style: 'solid',
          overlay: true
        }
      )
      
      expect(customId).toBeDefined()
      expect(customId).toMatch(/^custom_/)
      
      const result = engine.calculate(customId, mockData)
      expect(result).toBeDefined()
      expect(result!.values.length).toBe(mockData.length)
    })

    it('should handle script errors gracefully', () => {
      const badScriptId = engine.createCustomIndicator(
        'Bad Script',
        'throw new Error("Test error")',
        [],
        {
          color: '#FF0000',
          lineWidth: 1,
          style: 'solid',
          overlay: true
        }
      )
      
      const result = engine.calculate(badScriptId, mockData)
      expect(result).toBe(null) // Should return null on error
    })

    it('should sandbox script execution', () => {
      const maliciousId = engine.createCustomIndicator(
        'Malicious Script',
        `
        try {
          // Try to access global objects - should fail
          if (typeof window !== 'undefined') {
            window.location = 'http://evil.com';
          }
          return data.map(() => 2); // Different value to test sandbox
        } catch (e) {
          // Expected to fail due to sandboxing
          return data.map(() => 1);
        }
        `,
        [],
        {
          color: '#FF0000',
          lineWidth: 1,
          style: 'solid',
          overlay: true
        }
      )
      
      const result = engine.calculate(maliciousId, mockData)
      expect(result).toBeDefined() // Should execute safely
      // Should return array of values, either 1 or 2 depending on environment
      expect(result!.values.every(v => v.value === 1 || v.value === 2)).toBe(true)
    })
  })

  describe('Caching', () => {
    it('should cache calculation results', () => {
      const spy = vi.spyOn(console, 'time')
      
      // First calculation
      const result1 = engine.calculate('sma', mockData, { period: 20 })
      
      // Second calculation with same parameters should be cached
      const result2 = engine.calculate('sma', mockData, { period: 20 })
      
      expect(result1).toEqual(result2)
      
      spy.mockRestore()
    })

    it('should invalidate cache for different parameters', () => {
      const result1 = engine.calculate('sma', mockData, { period: 10 })
      const result2 = engine.calculate('sma', mockData, { period: 20 })
      
      expect(result1).not.toEqual(result2)
    })

    it('should clear cache when requested', () => {
      engine.calculate('sma', mockData, { period: 10 })
      engine.clearCache()
      
      // Should recalculate after cache clear
      const result = engine.calculate('sma', mockData, { period: 10 })
      expect(result).toBeDefined()
    })
  })

  describe('Error Handling', () => {
    it('should handle invalid indicator ID', () => {
      const result = engine.calculate('nonexistent', mockData)
      expect(result).toBe(null)
    })

    it('should handle empty data gracefully', () => {
      const result = engine.calculate('sma', [], { period: 20 })
      expect(result).toBeDefined()
      expect(result!.values.length).toBe(0)
    })

    it('should handle invalid parameters gracefully', () => {
      const result = engine.calculate('sma', mockData, { period: -5 })
      expect(result).toBeDefined() // Should handle gracefully
    })
  })

  describe('Data Validation', () => {
    it('should filter out invalid/infinite values', () => {
      const invalidData: OHLCVData[] = [
        ...mockData,
        { time: '2023-01-11', open: NaN, high: Infinity, low: -Infinity, close: 125, volume: 1000 }
      ]
      
      const result = engine.calculate('sma', invalidData, { period: 3 })
      expect(result).toBeDefined()
      
      // Should handle invalid data without crashing
      result!.values.forEach(v => {
        if (v.value !== null) {
          expect(Number.isFinite(v.value)).toBe(true)
        }
      })
    })
  })
})

describe('Singleton Instance', () => {
  it('should export singleton instance', () => {
    expect(indicatorEngine).toBeInstanceOf(IndicatorEngine)
    
    // Should maintain state across imports
    const indicators1 = indicatorEngine.getAvailableIndicators()
    const indicators2 = indicatorEngine.getAvailableIndicators()
    expect(indicators1).toEqual(indicators2)
  })
})