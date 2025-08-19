import { Instrument, InstrumentSearchResult } from '../types/instrumentTypes'

export interface OfflineCache {
  instruments: Instrument[]
  lastUpdate: string
  version: string
}

class OfflineService {
  private readonly CACHE_KEY = 'instrument-offline-cache'
  private readonly CACHE_VERSION = '1.0.0'
  private readonly MAX_CACHE_AGE = 24 * 60 * 60 * 1000 // 24 hours

  // Save instruments to offline cache
  saveToOfflineCache(instruments: Instrument[]): void {
    try {
      const cache: OfflineCache = {
        instruments,
        lastUpdate: new Date().toISOString(),
        version: this.CACHE_VERSION
      }
      
      localStorage.setItem(this.CACHE_KEY, JSON.stringify(cache))
      console.log(`Saved ${instruments.length} instruments to offline cache`)
    } catch (error) {
      console.warn('Failed to save instruments to offline cache:', error)
    }
  }

  // Load instruments from offline cache
  loadFromOfflineCache(): Instrument[] | null {
    try {
      const cached = localStorage.getItem(this.CACHE_KEY)
      if (!cached) return null

      const cache: OfflineCache = JSON.parse(cached)
      
      // Check version compatibility
      if (cache.version !== this.CACHE_VERSION) {
        console.warn('Offline cache version mismatch, clearing cache')
        this.clearOfflineCache()
        return null
      }

      // Check cache age
      const cacheAge = Date.now() - new Date(cache.lastUpdate).getTime()
      if (cacheAge > this.MAX_CACHE_AGE) {
        console.warn('Offline cache is stale, clearing cache')
        this.clearOfflineCache()
        return null
      }

      console.log(`Loaded ${cache.instruments.length} instruments from offline cache`)
      return cache.instruments
    } catch (error) {
      console.warn('Failed to load instruments from offline cache:', error)
      return null
    }
  }

  // Clear offline cache
  clearOfflineCache(): void {
    try {
      localStorage.removeItem(this.CACHE_KEY)
      console.log('Cleared offline cache')
    } catch (error) {
      console.warn('Failed to clear offline cache:', error)
    }
  }

  // Check if we have a valid offline cache
  hasValidCache(): boolean {
    const instruments = this.loadFromOfflineCache()
    return instruments !== null && instruments.length > 0
  }

  // Get cache info
  getCacheInfo(): { hasCache: boolean; lastUpdate?: string; instrumentCount?: number } {
    try {
      const cached = localStorage.getItem(this.CACHE_KEY)
      if (!cached) return { hasCache: false }

      const cache: OfflineCache = JSON.parse(cached)
      return {
        hasCache: true,
        lastUpdate: cache.lastUpdate,
        instrumentCount: cache.instruments.length
      }
    } catch (error) {
      return { hasCache: false }
    }
  }

  // Search instruments offline
  searchOffline(
    query: string,
    maxResults: number = 50,
    filters?: {
      assetClasses?: string[]
      venues?: string[]
      currencies?: string[]
    }
  ): InstrumentSearchResult[] {
    const instruments = this.loadFromOfflineCache()
    if (!instruments) return []

    if (!query.trim()) return []

    const normalizedQuery = query.toLowerCase().trim()
    let filteredInstruments = instruments

    // Apply filters
    if (filters?.assetClasses && filters.assetClasses.length > 0) {
      filteredInstruments = filteredInstruments.filter(
        inst => filters.assetClasses!.includes(inst.assetClass)
      )
    }
    
    if (filters?.venues && filters.venues.length > 0) {
      filteredInstruments = filteredInstruments.filter(
        inst => filters.venues!.includes(inst.venue)
      )
    }
    
    if (filters?.currencies && filters.currencies.length > 0) {
      filteredInstruments = filteredInstruments.filter(
        inst => filters.currencies!.includes(inst.currency)
      )
    }

    // Calculate relevance scores and create search results
    const searchResults = filteredInstruments
      .map(instrument => {
        const relevanceScore = this.calculateOfflineRelevanceScore(normalizedQuery, instrument)
        if (relevanceScore === 0) return null
        
        return {
          instrument,
          relevanceScore,
          matchType: this.getMatchType(normalizedQuery, instrument),
          highlightedText: this.getHighlightedText(normalizedQuery, instrument)
        } as InstrumentSearchResult
      })
      .filter((result): result is InstrumentSearchResult => result !== null)
      .sort((a, b) => b.relevanceScore - a.relevanceScore)
      .slice(0, maxResults)

    return searchResults
  }

  private calculateOfflineRelevanceScore(query: string, instrument: Instrument): number {
    const symbol = instrument.symbol.toLowerCase()
    const name = instrument.name.toLowerCase()
    const venue = instrument.venue.toLowerCase()
    
    let score = 0
    
    // Exact symbol match gets highest score
    if (symbol === query) {
      score += 100
    } else if (symbol.startsWith(query)) {
      score += 80
    } else if (symbol.includes(query)) {
      score += 60
    }
    
    // Name matching
    if (name.includes(query)) {
      score += 40
    }
    
    // Venue matching
    if (venue.includes(query)) {
      score += 20
    }
    
    // Boost score for common asset classes
    if (['STK', 'CASH', 'FUT'].includes(instrument.assetClass)) {
      score += 5
    }
    
    return score
  }

  private getMatchType(query: string, instrument: Instrument): 'symbol' | 'name' | 'venue' | 'alias' {
    if (instrument.symbol.toLowerCase().includes(query)) {
      return 'symbol'
    }
    if (instrument.name.toLowerCase().includes(query)) {
      return 'name'
    }
    if (instrument.venue.toLowerCase().includes(query)) {
      return 'venue'
    }
    return 'alias'
  }

  private getHighlightedText(query: string, instrument: Instrument): string {
    // Simple highlighting - in production, you'd want more sophisticated highlighting
    if (instrument.symbol.toLowerCase().includes(query)) {
      return instrument.symbol
    }
    if (instrument.name.toLowerCase().includes(query)) {
      return instrument.name
    }
    return instrument.symbol
  }

  // Check if device is online
  isOnline(): boolean {
    return navigator.onLine
  }

  // Update cache with new search results (called when online search succeeds)
  updateCacheWithSearchResults(searchResults: InstrumentSearchResult[]): void {
    // Extract unique instruments from search results
    const newInstruments = searchResults.map(result => result.instrument)
    
    // Load existing cache
    const existingInstruments = this.loadFromOfflineCache() || []
    
    // Merge new instruments with existing ones, avoiding duplicates
    const mergedInstruments = [...existingInstruments]
    
    newInstruments.forEach(newInstrument => {
      const existingIndex = mergedInstruments.findIndex(existing => existing.id === newInstrument.id)
      if (existingIndex >= 0) {
        // Update existing instrument with newer data
        mergedInstruments[existingIndex] = newInstrument
      } else {
        // Add new instrument
        mergedInstruments.push(newInstrument)
      }
    })
    
    // Limit cache size (keep most recent 1000 instruments)
    const limitedInstruments = mergedInstruments.slice(-1000)
    
    // Save updated cache
    this.saveToOfflineCache(limitedInstruments)
  }
}

export const offlineService = new OfflineService()