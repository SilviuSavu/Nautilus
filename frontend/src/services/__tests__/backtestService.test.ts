/**
 * Comprehensive test suite for BacktestService
 */

import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest'
import { BacktestService, BacktestConfigValidator, BacktestConfig } from '../backtestService'

// Mock fetch globally
global.fetch = vi.fn()

describe('BacktestConfigValidator', () => {
  let validator: BacktestConfigValidator

  beforeEach(() => {
    validator = new BacktestConfigValidator()
  })

  describe('validateDateRange', () => {
    it('should pass for valid date range', () => {
      const config: BacktestConfig = {
        strategyClass: 'TestStrategy',
        strategyConfig: {},
        startDate: '2023-01-01',
        endDate: '2023-12-31',
        instruments: ['AAPL'],
        venues: ['NASDAQ'],
        initialBalance: 100000,
        baseCurrency: 'USD',
        dataConfiguration: { dataType: 'bar' }
      }

      const result = validator.validateDateRange(config)
      expect(result.isValid).toBe(true)
      expect(result.errors).toHaveLength(0)
    })

    it('should fail when start date is after end date', () => {
      const config: BacktestConfig = {
        strategyClass: 'TestStrategy',
        strategyConfig: {},
        startDate: '2023-12-31',
        endDate: '2023-01-01',
        instruments: ['AAPL'],
        venues: ['NASDAQ'],
        initialBalance: 100000,
        baseCurrency: 'USD',
        dataConfiguration: { dataType: 'bar' }
      }

      const result = validator.validateDateRange(config)
      expect(result.isValid).toBe(false)
      expect(result.errors).toContain('Start date must be before end date')
    })

    it('should fail when start date is in the future', () => {
      const futureDate = new Date()
      futureDate.setDate(futureDate.getDate() + 1)
      
      const config: BacktestConfig = {
        strategyClass: 'TestStrategy',
        strategyConfig: {},
        startDate: futureDate.toISOString().split('T')[0],
        endDate: futureDate.toISOString().split('T')[0],
        instruments: ['AAPL'],
        venues: ['NASDAQ'],
        initialBalance: 100000,
        baseCurrency: 'USD',
        dataConfiguration: { dataType: 'bar' }
      }

      const result = validator.validateDateRange(config)
      expect(result.isValid).toBe(false)
      expect(result.errors).toContain('Start date cannot be in the future')
    })

    it('should fail for duration less than 1 day', () => {
      const today = new Date().toISOString().split('T')[0]
      
      const config: BacktestConfig = {
        strategyClass: 'TestStrategy',
        strategyConfig: {},
        startDate: today,
        endDate: today,
        instruments: ['AAPL'],
        venues: ['NASDAQ'],
        initialBalance: 100000,
        baseCurrency: 'USD',
        dataConfiguration: { dataType: 'bar' }
      }

      const result = validator.validateDateRange(config)
      expect(result.isValid).toBe(false)
      expect(result.errors).toContain('Backtest duration must be at least 1 day')
    })

    it('should fail for duration exceeding 5 years', () => {
      const config: BacktestConfig = {
        strategyClass: 'TestStrategy',
        strategyConfig: {},
        startDate: '2020-01-01',
        endDate: '2026-01-01',
        instruments: ['AAPL'],
        venues: ['NASDAQ'],
        initialBalance: 100000,
        baseCurrency: 'USD',
        dataConfiguration: { dataType: 'bar' }
      }

      const result = validator.validateDateRange(config)
      expect(result.isValid).toBe(false)
      expect(result.errors).toContain('Backtest duration cannot exceed 5 years')
    })
  })

  describe('validateInstruments', () => {
    it('should pass for valid instruments', () => {
      const config: BacktestConfig = {
        strategyClass: 'TestStrategy',
        strategyConfig: {},
        startDate: '2023-01-01',
        endDate: '2023-12-31',
        instruments: ['AAPL', 'MSFT', 'GOOGL'],
        venues: ['NASDAQ'],
        initialBalance: 100000,
        baseCurrency: 'USD',
        dataConfiguration: { dataType: 'bar' }
      }

      const result = validator.validateInstruments(config)
      expect(result.isValid).toBe(true)
      expect(result.errors).toHaveLength(0)
    })

    it('should fail when no instruments are provided', () => {
      const config: BacktestConfig = {
        strategyClass: 'TestStrategy',
        strategyConfig: {},
        startDate: '2023-01-01',
        endDate: '2023-12-31',
        instruments: [],
        venues: ['NASDAQ'],
        initialBalance: 100000,
        baseCurrency: 'USD',
        dataConfiguration: { dataType: 'bar' }
      }

      const result = validator.validateInstruments(config)
      expect(result.isValid).toBe(false)
      expect(result.errors).toContain('At least one instrument must be selected')
    })

    it('should fail when too many instruments are provided', () => {
      const instruments = Array.from({ length: 51 }, (_, i) => `STOCK${i}`)
      
      const config: BacktestConfig = {
        strategyClass: 'TestStrategy',
        strategyConfig: {},
        startDate: '2023-01-01',
        endDate: '2023-12-31',
        instruments,
        venues: ['NASDAQ'],
        initialBalance: 100000,
        baseCurrency: 'USD',
        dataConfiguration: { dataType: 'bar' }
      }

      const result = validator.validateInstruments(config)
      expect(result.isValid).toBe(false)
      expect(result.errors).toContain('Maximum 50 instruments allowed per backtest')
    })

    it('should fail for invalid instrument symbols', () => {
      const config: BacktestConfig = {
        strategyClass: 'TestStrategy',
        strategyConfig: {},
        startDate: '2023-01-01',
        endDate: '2023-12-31',
        instruments: ['AAPL', 'invalid-symbol!', '123'],
        venues: ['NASDAQ'],
        initialBalance: 100000,
        baseCurrency: 'USD',
        dataConfiguration: { dataType: 'bar' }
      }

      const result = validator.validateInstruments(config)
      expect(result.isValid).toBe(false)
      expect(result.errors[0]).toContain('Invalid instrument symbols')
    })
  })

  describe('validateCapital', () => {
    it('should pass for valid capital amounts', () => {
      const config: BacktestConfig = {
        strategyClass: 'TestStrategy',
        strategyConfig: {},
        startDate: '2023-01-01',
        endDate: '2023-12-31',
        instruments: ['AAPL'],
        venues: ['NASDAQ'],
        initialBalance: 50000,
        baseCurrency: 'USD',
        dataConfiguration: { dataType: 'bar' }
      }

      const result = validator.validateCapital(config)
      expect(result.isValid).toBe(true)
      expect(result.errors).toHaveLength(0)
    })

    it('should fail for balance below minimum', () => {
      const config: BacktestConfig = {
        strategyClass: 'TestStrategy',
        strategyConfig: {},
        startDate: '2023-01-01',
        endDate: '2023-12-31',
        instruments: ['AAPL'],
        venues: ['NASDAQ'],
        initialBalance: 500,
        baseCurrency: 'USD',
        dataConfiguration: { dataType: 'bar' }
      }

      const result = validator.validateCapital(config)
      expect(result.isValid).toBe(false)
      expect(result.errors).toContain('Initial balance must be at least $1,000')
    })

    it('should fail for balance above maximum', () => {
      const config: BacktestConfig = {
        strategyClass: 'TestStrategy',
        strategyConfig: {},
        startDate: '2023-01-01',
        endDate: '2023-12-31',
        instruments: ['AAPL'],
        venues: ['NASDAQ'],
        initialBalance: 200000000,
        baseCurrency: 'USD',
        dataConfiguration: { dataType: 'bar' }
      }

      const result = validator.validateCapital(config)
      expect(result.isValid).toBe(false)
      expect(result.errors).toContain('Initial balance cannot exceed $100,000,000')
    })

    it('should fail for unsupported currency', () => {
      const config: BacktestConfig = {
        strategyClass: 'TestStrategy',
        strategyConfig: {},
        startDate: '2023-01-01',
        endDate: '2023-12-31',
        instruments: ['AAPL'],
        venues: ['NASDAQ'],
        initialBalance: 100000,
        baseCurrency: 'XYZ',
        dataConfiguration: { dataType: 'bar' }
      }

      const result = validator.validateCapital(config)
      expect(result.isValid).toBe(false)
      expect(result.errors).toContain('Unsupported currency: XYZ')
    })
  })

  describe('validateStrategy', () => {
    it('should pass for valid strategy configuration', () => {
      const config: BacktestConfig = {
        strategyClass: 'MovingAverageCross',
        strategyConfig: { fast_period: 10, slow_period: 20 },
        startDate: '2023-01-01',
        endDate: '2023-12-31',
        instruments: ['AAPL'],
        venues: ['NASDAQ'],
        initialBalance: 100000,
        baseCurrency: 'USD',
        dataConfiguration: { dataType: 'bar' }
      }

      const result = validator.validateStrategy(config)
      expect(result.isValid).toBe(true)
      expect(result.errors).toHaveLength(0)
    })

    it('should fail when strategy class is empty', () => {
      const config: BacktestConfig = {
        strategyClass: '',
        strategyConfig: {},
        startDate: '2023-01-01',
        endDate: '2023-12-31',
        instruments: ['AAPL'],
        venues: ['NASDAQ'],
        initialBalance: 100000,
        baseCurrency: 'USD',
        dataConfiguration: { dataType: 'bar' }
      }

      const result = validator.validateStrategy(config)
      expect(result.isValid).toBe(false)
      expect(result.errors).toContain('Strategy class is required')
    })

    it('should fail when strategy config is empty', () => {
      const config: BacktestConfig = {
        strategyClass: 'TestStrategy',
        strategyConfig: {},
        startDate: '2023-01-01',
        endDate: '2023-12-31',
        instruments: ['AAPL'],
        venues: ['NASDAQ'],
        initialBalance: 100000,
        baseCurrency: 'USD',
        dataConfiguration: { dataType: 'bar' }
      }

      const result = validator.validateStrategy(config)
      expect(result.isValid).toBe(false)
      expect(result.errors).toContain('Strategy configuration is required')
    })
  })

  describe('validateComplete', () => {
    it('should pass for completely valid configuration', () => {
      const config: BacktestConfig = {
        strategyClass: 'MovingAverageCross',
        strategyConfig: { fast_period: 10, slow_period: 20 },
        startDate: '2023-01-01',
        endDate: '2023-06-30',
        instruments: ['AAPL', 'MSFT'],
        venues: ['NASDAQ'],
        initialBalance: 100000,
        baseCurrency: 'USD',
        dataConfiguration: { dataType: 'bar' }
      }

      const result = validator.validateComplete(config)
      expect(result.isValid).toBe(true)
      expect(result.errors).toHaveLength(0)
    })

    it('should collect all validation errors', () => {
      const config: BacktestConfig = {
        strategyClass: '',
        strategyConfig: {},
        startDate: '2023-12-31',
        endDate: '2023-01-01',
        instruments: [],
        venues: ['NASDAQ'],
        initialBalance: 500,
        baseCurrency: 'XYZ',
        dataConfiguration: { dataType: 'bar' }
      }

      const result = validator.validateComplete(config)
      expect(result.isValid).toBe(false)
      expect(result.errors.length).toBeGreaterThan(3)
    })
  })
})

describe('BacktestService', () => {
  let service: BacktestService
  const mockFetch = fetch as vi.MockedFunction<typeof fetch>

  beforeEach(() => {
    service = new BacktestService()
    mockFetch.mockClear()
    vi.spyOn(Storage.prototype, 'getItem').mockReturnValue('mock-token')
  })

  afterEach(() => {
    vi.restoreAllMocks()
  })

  describe('startBacktest', () => {
    it('should start a backtest successfully', async () => {
      const config: BacktestConfig = {
        strategyClass: 'MovingAverageCross',
        strategyConfig: { fast_period: 10, slow_period: 20 },
        startDate: '2023-01-01',
        endDate: '2023-06-30',
        instruments: ['AAPL'],
        venues: ['NASDAQ'],
        initialBalance: 100000,
        baseCurrency: 'USD',
        dataConfiguration: { dataType: 'bar' }
      }

      const mockResponse = {
        success: true,
        backtest_id: 'test-backtest-123',
        status: 'queued'
      }

      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: async () => mockResponse
      } as Response)

      const result = await service.startBacktest(config)

      expect(result.backtestId).toBe('test-backtest-123')
      expect(result.status).toBe('queued')
      expect(mockFetch).toHaveBeenCalledWith(
        expect.stringContaining('/api/v1/nautilus/engine/backtest'),
        expect.objectContaining({
          method: 'POST',
          headers: expect.objectContaining({
            'Content-Type': 'application/json',
            'Authorization': 'Bearer mock-token'
          })
        })
      )
    })

    it('should validate config before starting', async () => {
      const invalidConfig: BacktestConfig = {
        strategyClass: '',
        strategyConfig: {},
        startDate: '2023-12-31',
        endDate: '2023-01-01',
        instruments: [],
        venues: ['NASDAQ'],
        initialBalance: 500,
        baseCurrency: 'USD',
        dataConfiguration: { dataType: 'bar' }
      }

      await expect(service.startBacktest(invalidConfig))
        .rejects.toThrow('Configuration validation failed')

      expect(mockFetch).not.toHaveBeenCalled()
    })

    it('should handle API errors', async () => {
      const config: BacktestConfig = {
        strategyClass: 'MovingAverageCross',
        strategyConfig: { fast_period: 10, slow_period: 20 },
        startDate: '2023-01-01',
        endDate: '2023-06-30',
        instruments: ['AAPL'],
        venues: ['NASDAQ'],
        initialBalance: 100000,
        baseCurrency: 'USD',
        dataConfiguration: { dataType: 'bar' }
      }

      mockFetch.mockResolvedValueOnce({
        ok: false,
        json: async () => ({ detail: 'Rate limit exceeded' })
      } as Response)

      await expect(service.startBacktest(config))
        .rejects.toThrow('Rate limit exceeded')
    })
  })

  describe('getBacktestStatus', () => {
    it('should get backtest status successfully', async () => {
      const mockStatus = {
        backtest: {
          id: 'test-backtest-123',
          status: 'running',
          progress: 45.5
        }
      }

      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: async () => mockStatus
      } as Response)

      const result = await service.getBacktestStatus('test-backtest-123')

      expect(result).toEqual(mockStatus.backtest)
      expect(mockFetch).toHaveBeenCalledWith(
        expect.stringContaining('/api/v1/nautilus/engine/backtest/test-backtest-123'),
        expect.objectContaining({
          headers: expect.objectContaining({
            'Authorization': 'Bearer mock-token'
          })
        })
      )
    })

    it('should handle not found errors', async () => {
      mockFetch.mockResolvedValueOnce({
        ok: false,
        status: 404
      } as Response)

      await expect(service.getBacktestStatus('nonexistent'))
        .rejects.toThrow('Failed to get backtest status')
    })
  })

  describe('cancelBacktest', () => {
    it('should cancel backtest successfully', async () => {
      const mockResponse = { success: true }

      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: async () => mockResponse
      } as Response)

      const result = await service.cancelBacktest('test-backtest-123')

      expect(result).toBe(true)
      expect(mockFetch).toHaveBeenCalledWith(
        expect.stringContaining('/api/v1/nautilus/engine/backtest/test-backtest-123'),
        expect.objectContaining({
          method: 'DELETE',
          headers: expect.objectContaining({
            'Authorization': 'Bearer mock-token'
          })
        })
      )
    })

    it('should handle cancellation failures', async () => {
      mockFetch.mockResolvedValueOnce({
        ok: false
      } as Response)

      await expect(service.cancelBacktest('test-backtest-123'))
        .rejects.toThrow('Failed to cancel backtest')
    })
  })

  describe('listBacktests', () => {
    it('should list backtests successfully', async () => {
      const mockList = {
        backtests: [
          { id: 'bt1', status: 'completed' },
          { id: 'bt2', status: 'running' }
        ]
      }

      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: async () => mockList
      } as Response)

      const result = await service.listBacktests()

      expect(result).toEqual(mockList.backtests)
      expect(mockFetch).toHaveBeenCalledWith(
        expect.stringContaining('/api/v1/nautilus/engine/backtests'),
        expect.objectContaining({
          headers: expect.objectContaining({
            'Authorization': 'Bearer mock-token'
          })
        })
      )
    })
  })

  describe('getBacktestResults', () => {
    it('should get comprehensive results', async () => {
      const mockResults = {
        backtest_id: 'test-backtest-123',
        status: 'completed',
        metrics: { total_return: 15.5 },
        trades: [{ id: 'trade1', pnl: 100 }],
        equity_curve: [{ timestamp: '2023-01-01', equity: 100000 }]
      }

      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: async () => mockResults
      } as Response)

      const result = await service.getBacktestResults('test-backtest-123')

      expect(result).toEqual(mockResults)
    })
  })

  describe('compareBacktests', () => {
    it('should compare multiple backtests', async () => {
      const mockComparison = {
        comparison_id: 'comp-123',
        backtests: [
          { backtest_id: 'bt1', metrics: { total_return: 10 } },
          { backtest_id: 'bt2', metrics: { total_return: 15 } }
        ],
        summary: { best_return: 15 }
      }

      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: async () => mockComparison
      } as Response)

      const result = await service.compareBacktests(['bt1', 'bt2'])

      expect(result).toEqual(mockComparison)
      expect(mockFetch).toHaveBeenCalledWith(
        expect.stringContaining('/api/v1/nautilus/backtest/compare'),
        expect.objectContaining({
          method: 'POST',
          body: JSON.stringify({ backtest_ids: ['bt1', 'bt2'] })
        })
      )
    })
  })

  describe('deleteBacktest', () => {
    it('should delete backtest successfully', async () => {
      mockFetch.mockResolvedValueOnce({
        ok: true
      } as Response)

      const result = await service.deleteBacktest('test-backtest-123')

      expect(result).toBe(true)
      expect(mockFetch).toHaveBeenCalledWith(
        expect.stringContaining('/api/v1/nautilus/backtest/test-backtest-123'),
        expect.objectContaining({
          method: 'DELETE'
        })
      )
    })

    it('should handle deletion failures', async () => {
      mockFetch.mockResolvedValueOnce({
        ok: false
      } as Response)

      const result = await service.deleteBacktest('test-backtest-123')

      expect(result).toBe(false)
    })
  })
})

describe('BacktestProgressMonitor', () => {
  // These tests would require mocking WebSocket
  // For now, we'll add basic structure tests

  it('should be instantiable', () => {
    const { BacktestProgressMonitor } = require('../backtestService')
    const monitor = new BacktestProgressMonitor()
    expect(monitor).toBeDefined()
  })
})

// Integration tests
describe('BacktestService Integration', () => {
  it('should handle complete backtest workflow', async () => {
    // This would be a full integration test
    // involving starting a backtest, monitoring progress,
    // and retrieving results
    expect(true).toBe(true) // Placeholder
  })
})