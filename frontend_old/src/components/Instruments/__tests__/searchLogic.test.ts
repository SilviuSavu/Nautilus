/**
 * Unit tests for search and filtering logic
 * These test the core algorithms without complex mocking
 */

import { describe, it, expect } from 'vitest'
import { Instrument, SearchFilters } from '../types/instrumentTypes'

// Core search logic functions extracted for testing
export function calculateRelevanceScore(query: string, instrument: Instrument): number {
  if (!query || !instrument) return 0
  
  const normalizedQuery = query.toLowerCase().trim()
  const symbol = instrument.symbol?.toLowerCase() || ''
  const name = instrument.name?.toLowerCase() || ''
  const venue = instrument.venue?.toLowerCase() || ''
  
  let score = 0
  
  // Exact symbol match gets highest score
  if (symbol === normalizedQuery) {
    score += 100
  }
  
  // Symbol starts with query
  if (symbol.startsWith(normalizedQuery)) {
    score += 50
  }
  
  // Symbol contains query
  if (symbol.includes(normalizedQuery)) {
    score += 20
  }
  
  // Name contains query
  if (name.includes(normalizedQuery)) {
    score += 10
  }
  
  // Venue contains query
  if (venue.includes(normalizedQuery)) {
    score += 5
  }
  
  return score
}

export function applySearchFilters(instruments: Instrument[], filters: SearchFilters): Instrument[] {
  if (!filters) return instruments
  
  let filtered = instruments
  
  // Filter by asset classes
  if (filters.assetClasses && filters.assetClasses.length > 0) {
    filtered = filtered.filter(inst => 
      filters.assetClasses.includes(inst.assetClass)
    )
  }
  
  // Filter by venues
  if (filters.venues && filters.venues.length > 0) {
    filtered = filtered.filter(inst => 
      filters.venues.includes(inst.venue)
    )
  }
  
  // Filter by currencies
  if (filters.currencies && filters.currencies.length > 0) {
    filtered = filtered.filter(inst => 
      filters.currencies.includes(inst.currency)
    )
  }
  
  return filtered
}

export function sortByRelevance(results: Array<{instrument: Instrument, relevanceScore: number}>): Array<{instrument: Instrument, relevanceScore: number}> {
  return results.sort((a, b) => b.relevanceScore - a.relevanceScore)
}

describe('Search and Filtering Logic', () => {
  const mockInstruments: Instrument[] = [
    {
      id: 'AAPL-STK',
      symbol: 'AAPL',
      name: 'Apple Inc',
      venue: 'SMART',
      assetClass: 'STK',
      currency: 'USD'
    },
    {
      id: 'MSFT-STK',
      symbol: 'MSFT',
      name: 'Microsoft Corporation',
      venue: 'NASDAQ',
      assetClass: 'STK',
      currency: 'USD'
    },
    {
      id: 'ES-FUT',
      symbol: 'ES',
      name: 'E-mini S&P 500',
      venue: 'CME',
      assetClass: 'FUT',
      currency: 'USD'
    },
    {
      id: 'EURUSD-CASH',
      symbol: 'EURUSD',
      name: 'Euro US Dollar',
      venue: 'IDEALPRO',
      assetClass: 'CASH',
      currency: 'USD'
    }
  ]

  describe('calculateRelevanceScore', () => {
    it('should return 0 for empty query', () => {
      const score = calculateRelevanceScore('', mockInstruments[0])
      expect(score).toBe(0)
    })

    it('should give highest score for exact symbol match', () => {
      const score = calculateRelevanceScore('AAPL', mockInstruments[0])
      expect(score).toBe(170) // 100 (exact) + 50 (starts) + 20 (contains)
    })

    it('should give high score for symbol prefix match', () => {
      const score = calculateRelevanceScore('AA', mockInstruments[0])
      expect(score).toBe(70) // 50 (starts) + 20 (contains)
    })

    it('should give medium score for symbol substring match', () => {
      const instrument = { ...mockInstruments[0], symbol: 'XAAPL' }
      const score = calculateRelevanceScore('AAPL', instrument)
      expect(score).toBe(20)
    })

    it('should give low score for name match', () => {
      const score = calculateRelevanceScore('apple', mockInstruments[0])
      expect(score).toBe(10)
    })

    it('should be case insensitive', () => {
      const score1 = calculateRelevanceScore('aapl', mockInstruments[0])
      const score2 = calculateRelevanceScore('AAPL', mockInstruments[0])
      expect(score1).toBe(score2)
    })

    it('should handle missing properties gracefully', () => {
      const instrument = { id: 'test', symbol: 'TEST' } as Instrument
      const score = calculateRelevanceScore('test', instrument)
      expect(score).toBeGreaterThan(0)
    })
  })

  describe('applySearchFilters', () => {
    it('should return all instruments when no filters applied', () => {
      const filters: SearchFilters = { assetClasses: [], venues: [], currencies: [] }
      const result = applySearchFilters(mockInstruments, filters)
      expect(result).toEqual(mockInstruments)
    })

    it('should filter by asset class', () => {
      const filters: SearchFilters = { 
        assetClasses: ['STK'], 
        venues: [], 
        currencies: [] 
      }
      const result = applySearchFilters(mockInstruments, filters)
      expect(result).toHaveLength(2)
      expect(result.every(inst => inst.assetClass === 'STK')).toBe(true)
    })

    it('should filter by venue', () => {
      const filters: SearchFilters = { 
        assetClasses: [], 
        venues: ['SMART'], 
        currencies: [] 
      }
      const result = applySearchFilters(mockInstruments, filters)
      expect(result).toHaveLength(1)
      expect(result[0].venue).toBe('SMART')
    })

    it('should filter by currency', () => {
      const filters: SearchFilters = { 
        assetClasses: [], 
        venues: [], 
        currencies: ['USD'] 
      }
      const result = applySearchFilters(mockInstruments, filters)
      expect(result).toHaveLength(4) // All test instruments are USD
    })

    it('should apply multiple filters simultaneously', () => {
      const filters: SearchFilters = { 
        assetClasses: ['STK'], 
        venues: ['SMART'], 
        currencies: ['USD'] 
      }
      const result = applySearchFilters(mockInstruments, filters)
      expect(result).toHaveLength(1)
      expect(result[0].symbol).toBe('AAPL')
    })

    it('should return empty array when filters exclude all instruments', () => {
      const filters: SearchFilters = { 
        assetClasses: ['BOND'], 
        venues: [], 
        currencies: [] 
      }
      const result = applySearchFilters(mockInstruments, filters)
      expect(result).toHaveLength(0)
    })
  })

  describe('sortByRelevance', () => {
    it('should sort results by relevance score in descending order', () => {
      const results = [
        { instrument: mockInstruments[0], relevanceScore: 50 },
        { instrument: mockInstruments[1], relevanceScore: 100 },
        { instrument: mockInstruments[2], relevanceScore: 10 }
      ]
      
      const sorted = sortByRelevance(results)
      
      expect(sorted[0].relevanceScore).toBe(100)
      expect(sorted[1].relevanceScore).toBe(50)
      expect(sorted[2].relevanceScore).toBe(10)
    })

    it('should handle empty array', () => {
      const result = sortByRelevance([])
      expect(result).toEqual([])
    })

    it('should handle single item', () => {
      const results = [{ instrument: mockInstruments[0], relevanceScore: 50 }]
      const sorted = sortByRelevance(results)
      expect(sorted).toEqual(results)
    })
  })

  describe('Integration: Search Pipeline', () => {
    it('should correctly process a complete search workflow', () => {
      const query = 'AAPL'
      const filters: SearchFilters = { 
        assetClasses: ['STK'], 
        venues: [], 
        currencies: [] 
      }
      
      // 1. Apply filters
      const filtered = applySearchFilters(mockInstruments, filters)
      expect(filtered).toHaveLength(2) // AAPL and MSFT
      
      // 2. Calculate relevance scores
      const withScores = filtered.map(instrument => ({
        instrument,
        relevanceScore: calculateRelevanceScore(query, instrument)
      }))
      
      // 3. Sort by relevance
      const sorted = sortByRelevance(withScores)
      
      // AAPL should be first (exact match = 170 points total)
      expect(sorted[0].instrument.symbol).toBe('AAPL')
      expect(sorted[0].relevanceScore).toBe(170)
      
      // MSFT should be second (no match = 0 points)
      expect(sorted[1].instrument.symbol).toBe('MSFT')
      expect(sorted[1].relevanceScore).toBe(0)
    })
  })
})