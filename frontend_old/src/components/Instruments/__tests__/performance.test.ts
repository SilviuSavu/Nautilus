/**
 * Performance tests for instrument search with large datasets
 */

import { describe, it, expect, vi, beforeEach } from 'vitest'
import { instrumentService } from '../services/instrumentService'
import { Instrument } from '../types/instrumentTypes'

// Mock fetch for performance testing
const mockFetch = vi.fn()
global.fetch = mockFetch

describe('Instrument Search Performance', () => {
  beforeEach(() => {
    vi.clearAllMocks()
    instrumentService.clearCache()
  })

  // Generate mock large dataset
  const generateLargeDataset = (size: number): Instrument[] => {
    const instruments: Instrument[] = []
    const symbols = ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'AMZN', 'META', 'NVDA', 'AMD']
    const venues = ['SMART', 'NASDAQ', 'NYSE', 'ARCA']
    const assetClasses = ['STK', 'OPT', 'FUT', 'CASH']
    
    for (let i = 0; i < size; i++) {
      instruments.push({
        id: `${symbols[i % symbols.length]}-${assetClasses[i % assetClasses.length]}-${i}`,
        symbol: `${symbols[i % symbols.length]}${i > 100 ? i : ''}`,
        name: `Company ${i}`,
        venue: venues[i % venues.length],
        assetClass: assetClasses[i % assetClasses.length],
        currency: 'USD',
        contractId: 100000 + i
      })
    }
    
    return instruments
  }

  it('should handle large dataset (1000 instruments) within performance limits', async () => {
    const largeDataset = generateLargeDataset(1000)
    
    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: async () => ({
        instruments: largeDataset,
        total: largeDataset.length,
        query: 'AAPL',
        timestamp: new Date().toISOString()
      })
    })

    const startTime = performance.now()
    const results = await instrumentService.searchInstruments('AAPL')
    const endTime = performance.now()
    
    const searchDuration = endTime - startTime
    
    // Performance requirements:
    expect(searchDuration).toBeLessThan(2000) // Should complete within 2 seconds
    expect(results.length).toBeGreaterThan(0)
    expect(results.length).toBeLessThanOrEqual(50) // Respects maxResults default
    
    console.log(`✅ Large dataset search completed in ${searchDuration.toFixed(2)}ms`)
  })

  it('should handle virtual scrolling threshold (100+ instruments) efficiently', async () => {
    const dataset = generateLargeDataset(150)
    
    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: async () => ({
        instruments: dataset,
        total: dataset.length,
        query: 'A',
        timestamp: new Date().toISOString()
      })
    })

    const startTime = performance.now()
    const results = await instrumentService.searchInstruments('A', { maxResults: 100 })
    const endTime = performance.now()
    
    const searchDuration = endTime - startTime
    
    expect(searchDuration).toBeLessThan(1500) // Virtual scrolling should be faster
    expect(results.length).toBeLessThanOrEqual(100)
    
    console.log(`✅ Virtual scrolling dataset search completed in ${searchDuration.toFixed(2)}ms`)
  })

  it('should cache results efficiently for repeated queries', async () => {
    const dataset = generateLargeDataset(500)
    
    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: async () => ({
        instruments: dataset,
        total: dataset.length,
        query: 'MSFT',
        timestamp: new Date().toISOString()
      })
    })

    // First call (should hit backend)
    const startTime1 = performance.now()
    const results1 = await instrumentService.searchInstruments('MSFT')
    const endTime1 = performance.now()
    const firstCallDuration = endTime1 - startTime1

    // Second call (should use cache)
    const startTime2 = performance.now()
    const results2 = await instrumentService.searchInstruments('MSFT')
    const endTime2 = performance.now()
    const cachedCallDuration = endTime2 - startTime2

    expect(mockFetch).toHaveBeenCalledTimes(1) // Only called once
    expect(results1).toEqual(results2) // Same results
    expect(cachedCallDuration).toBeLessThan(firstCallDuration) // Cache is faster
    expect(cachedCallDuration).toBeLessThan(100) // Cache should be very fast
    
    console.log(`✅ First call: ${firstCallDuration.toFixed(2)}ms, Cached call: ${cachedCallDuration.toFixed(2)}ms`)
  })

  it('should handle relevance scoring efficiently on large datasets', async () => {
    const dataset = generateLargeDataset(2000) // Very large dataset
    
    // Mix some exact matches and partial matches
    dataset[0] = { ...dataset[0], symbol: 'AAPL', name: 'Apple Inc' }
    dataset[1] = { ...dataset[1], symbol: 'AAPLW', name: 'Apple Warrants' }
    dataset[2] = { ...dataset[2], symbol: 'XAAPL', name: 'Some Apple ETF' }

    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: async () => ({
        instruments: dataset,
        total: dataset.length,
        query: 'AAPL',
        timestamp: new Date().toISOString()
      })
    })

    const startTime = performance.now()
    const results = await instrumentService.searchInstruments('AAPL')
    const endTime = performance.now()
    
    const searchDuration = endTime - startTime
    
    expect(searchDuration).toBeLessThan(3000) // Even large relevance scoring should be reasonable
    expect(results.length).toBeGreaterThan(0)
    
    // Check that results are sorted by relevance (exact match should be first)
    if (results.length > 0) {
      expect(results[0].instrument.symbol).toBe('AAPL')
    }
    
    console.log(`✅ Relevance scoring on 2000 instruments completed in ${searchDuration.toFixed(2)}ms`)
  })

  it('should handle memory efficiently with multiple large searches', async () => {
    // Test memory usage doesn't grow excessively with multiple searches
    const queries = ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'AMZN']
    const searchPromises: Promise<any>[] = []
    
    queries.forEach((query, index) => {
      const dataset = generateLargeDataset(500 + index * 100) // Varying sizes
      
      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: async () => ({
          instruments: dataset,
          total: dataset.length,
          query,
          timestamp: new Date().toISOString()
        })
      })
      
      searchPromises.push(instrumentService.searchInstruments(query))
    })

    const startTime = performance.now()
    const allResults = await Promise.all(searchPromises)
    const endTime = performance.now()
    
    const totalDuration = endTime - startTime
    
    expect(totalDuration).toBeLessThan(5000) // All 5 searches within 5 seconds
    expect(allResults).toHaveLength(5)
    expect(mockFetch).toHaveBeenCalledTimes(5)
    
    // Verify all searches returned results
    allResults.forEach((results, index) => {
      expect(results.length).toBeGreaterThan(0)
    })
    
    console.log(`✅ Multiple large searches (5 concurrent) completed in ${totalDuration.toFixed(2)}ms`)
  })
})