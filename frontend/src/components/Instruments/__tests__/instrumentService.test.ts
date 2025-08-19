import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest'
import { instrumentService } from '../services/instrumentService'
import { Instrument, SearchFilters } from '../types/instrumentTypes'

// Mock fetch
const mockFetch = vi.fn()
global.fetch = mockFetch

// Mock console.log to reduce test noise
vi.spyOn(console, 'log').mockImplementation(() => {})

describe('InstrumentService', () => {
  const service = instrumentService
  
  const mockInstrument: Instrument = {
    id: 'AAPL-STK',
    symbol: 'AAPL',
    name: 'Apple Inc',
    venue: 'SMART',
    assetClass: 'STK',
    currency: 'USD',
    contractId: 265598
  }

  beforeEach(() => {
    vi.clearAllMocks()
    service.clearCache() // Clear cache between tests
  })

  afterEach(() => {
    vi.restoreAllMocks()
  })

  describe('searchInstruments', () => {
    it('should return empty array for empty query', async () => {
      const results = await service.searchInstruments('')
      expect(results).toEqual([])
    })

    it('should return empty array for whitespace-only query', async () => {
      const results = await service.searchInstruments('   ')
      expect(results).toEqual([])
    })

    it('should call IB backend endpoint with correct parameters', async () => {
      const mockResponse = {
        instruments: [mockInstrument],
        total: 1,
        query: 'AAPL',
        timestamp: '2025-08-18T10:00:00Z'
      }

      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: async () => mockResponse,
        status: 200,
        headers: new Headers({ 'content-type': 'application/json' })
      })

      const results = await service.searchInstruments('AAPL')

      expect(mockFetch).toHaveBeenCalledWith(
        expect.stringContaining('/api/v1/ib/instruments/search/AAPL'),
        expect.objectContaining({
          method: 'GET',
          headers: expect.objectContaining({
            'Accept': 'application/json',
            'Content-Type': 'application/json'
          })
        })
      )
      
      expect(results).toHaveLength(1)
      expect(results[0].instrument.symbol).toBe('AAPL')
    })

    it('should apply asset class filter', async () => {
      const filters: SearchFilters = {
        assetClasses: ['FUT'],
        venues: [],
        currencies: []
      }

      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: async () => ({ instruments: [], total: 0, query: 'AAPL', timestamp: '2025-08-18T10:00:00Z' })
      })

      await service.searchInstruments('AAPL', { filters })

      const url = mockFetch.mock.calls[0][0] as string
      expect(url).toContain('sec_type=FUT')
    })

    it('should default to STK when no asset class filter specified', async () => {
      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: async () => ({ instruments: [], total: 0, query: 'AAPL', timestamp: '2025-08-18T10:00:00Z' })
      })

      await service.searchInstruments('AAPL')

      const url = mockFetch.mock.calls[0][0] as string
      expect(url).toContain('sec_type=STK')
    })

    it('should apply venue filter', async () => {
      const filters: SearchFilters = {
        assetClasses: [],
        venues: ['NYSE'],
        currencies: []
      }

      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: async () => ({ instruments: [], total: 0, query: 'AAPL', timestamp: '2025-08-18T10:00:00Z' })
      })

      await service.searchInstruments('AAPL', { filters })

      const url = mockFetch.mock.calls[0][0] as string
      expect(url).toContain('exchange=NYSE')
    })

    it('should apply currency filter', async () => {
      const filters: SearchFilters = {
        assetClasses: [],
        venues: [],
        currencies: ['EUR']
      }

      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: async () => ({ instruments: [], total: 0, query: 'AAPL', timestamp: '2025-08-18T10:00:00Z' })
      })

      await service.searchInstruments('AAPL', { filters })

      const url = mockFetch.mock.calls[0][0] as string
      expect(url).toContain('currency=EUR')
    })

    it('should respect maxResults parameter', async () => {
      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: async () => ({ instruments: [], total: 0, query: 'AAPL', timestamp: '2025-08-18T10:00:00Z' })
      })

      await service.searchInstruments('AAPL', { maxResults: 25 })

      const url = mockFetch.mock.calls[0][0] as string
      expect(url).toContain('max_results=25')
    })

    it('should handle backend errors gracefully', async () => {
      mockFetch.mockResolvedValueOnce({
        ok: false,
        status: 422,
        statusText: 'Unprocessable Entity',
        text: async () => 'Contract search timeout'
      })

      // Mock fallback data
      const fallbackSpy = vi.spyOn(service as any, 'fetchInstrumentsFromBackend')
      fallbackSpy.mockResolvedValue([])

      const results = await service.searchInstruments('NONEXISTENT')
      
      expect(results).toEqual([])
      expect(fallbackSpy).toHaveBeenCalled()
    })

    it('should handle network errors', async () => {
      mockFetch.mockRejectedValueOnce(new Error('Network error'))

      // Mock fallback data
      const fallbackSpy = vi.spyOn(service as any, 'fetchInstrumentsFromBackend')
      fallbackSpy.mockResolvedValue([])

      const results = await service.searchInstruments('AAPL')
      
      expect(results).toEqual([])
      expect(fallbackSpy).toHaveBeenCalled()
    })

    it('should calculate relevance scores', async () => {
      const mockInstruments = [
        { ...mockInstrument, symbol: 'AAPL', name: 'Apple Inc' },
        { ...mockInstrument, symbol: 'MSFT', name: 'Microsoft Corp', id: 'MSFT-STK' }
      ]

      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: async () => ({
          instruments: mockInstruments,
          total: 2,
          query: 'AAPL',
          timestamp: '2025-08-18T10:00:00Z'
        })
      })

      const results = await service.searchInstruments('AAPL')
      
      expect(results).toHaveLength(2)
      
      // AAPL should have higher relevance score due to exact match
      const appleResult = results.find(r => r.instrument.symbol === 'AAPL')
      const msftResult = results.find(r => r.instrument.symbol === 'MSFT')
      
      expect(appleResult?.relevanceScore).toBeGreaterThan(msftResult?.relevanceScore || 0)
    })
  })

  describe('caching', () => {
    it('should cache search results', async () => {
      const mockResponse = {
        instruments: [mockInstrument],
        total: 1,
        query: 'AAPL',
        timestamp: '2025-08-18T10:00:00Z'
      }

      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: async () => mockResponse
      })

      // First call
      await service.searchInstruments('AAPL')
      expect(mockFetch).toHaveBeenCalledTimes(1)

      // Second call should use cache
      await service.searchInstruments('AAPL')
      expect(mockFetch).toHaveBeenCalledTimes(1)
    })
  })
})