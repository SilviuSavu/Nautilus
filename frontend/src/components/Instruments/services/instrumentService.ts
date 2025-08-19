import { 
  Instrument, 
  InstrumentSearchResult, 
  VenueInfo, 
  SearchFilters,
  InstrumentSearchResponse,
  VenueStatusResponse,
  InstrumentDetailsResponse
} from '../types/instrumentTypes'
import { offlineService } from './offlineService'

class InstrumentService {
  private baseUrl: string
  private cache: Map<string, { data: any; timestamp: number; ttl: number }> = new Map()

  constructor() {
    this.baseUrl = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000'
    console.log('InstrumentService baseUrl:', this.baseUrl)
  }

  // Cache management
  private getCachedData<T>(key: string): T | null {
    const cached = this.cache.get(key)
    if (!cached) return null
    
    if (Date.now() - cached.timestamp > cached.ttl) {
      this.cache.delete(key)
      return null
    }
    
    return cached.data as T
  }

  private setCachedData<T>(key: string, data: T, ttlMs: number = 300000): void { // 5 minutes default
    this.cache.set(key, {
      data,
      timestamp: Date.now(),
      ttl: ttlMs
    })
  }

  // Fuzzy search implementation
  private calculateRelevanceScore(query: string, instrument: Instrument): number {
    if (!query || !instrument) return 0
    
    const normalizedQuery = query.toLowerCase().trim()
    const symbol = instrument.symbol?.toLowerCase() || ''
    const name = instrument.name?.toLowerCase() || ''
    const venue = instrument.venue?.toLowerCase() || ''
    
    let score = 0
    
    // Exact symbol match gets highest score
    if (symbol === normalizedQuery) {
      score += 100
    } else if (symbol.startsWith(normalizedQuery)) {
      score += 80
    } else if (symbol.includes(normalizedQuery)) {
      score += 60
    }
    
    // Name matching
    if (name.includes(normalizedQuery)) {
      score += 40
    }
    
    // Venue matching
    if (venue.includes(normalizedQuery)) {
      score += 20
    }
    
    // Boost score for common asset classes
    if (['STK', 'CASH', 'FUT'].includes(instrument.assetClass)) {
      score += 5
    }
    
    return score
  }

  // Enhanced instrument search with backend IB integration
  async searchInstruments(
    query: string, 
    options: {
      maxResults?: number
      filters?: SearchFilters
      favorites?: string[]
      venueStatus?: Record<string, VenueInfo>
    } = {}
  ): Promise<InstrumentSearchResult[]> {
    const { maxResults = 50, filters } = options
    
    if (!query.trim()) {
      return []
    }

    // Check if we're offline and have cached data
    if (!offlineService.isOnline() && offlineService.hasValidCache()) {
      console.log('🔌 Offline mode: using cached instrument data')
      return offlineService.searchOffline(query, maxResults, {
        assetClasses: filters?.assetClasses,
        venues: filters?.venues,
        currencies: filters?.currencies
      })
    }

    const cacheKey = `search:${query}:${JSON.stringify(filters)}`
    const cached = this.getCachedData<InstrumentSearchResult[]>(cacheKey)
    if (cached) {
      return cached.slice(0, maxResults)
    }

    try {
      // Try to search using the new IB instrument search endpoint first
      let searchResults: InstrumentSearchResult[] = []
      
      try {
        console.log(`Starting IB instrument search for query: "${query}" (stocks first for speed)`)
        const searchUrl = new URL(`${this.baseUrl}/api/v1/ib/instruments/search/${encodeURIComponent(query)}`)
        
        // Add query parameters for filtering
        if (filters?.assetClasses && filters.assetClasses.length === 1) {
          searchUrl.searchParams.set('sec_type', filters.assetClasses[0])
        } else {
          // Default to stocks only for fast response and relevant results
          searchUrl.searchParams.set('sec_type', 'STK')
          console.log(`🎯 Defaulting to stocks only (STK) for faster, cleaner results`)
        }
        if (filters?.venues && filters.venues.length === 1) {
          searchUrl.searchParams.set('exchange', filters.venues[0])
        }
        if (filters?.currencies && filters.currencies.length === 1) {
          searchUrl.searchParams.set('currency', filters.currencies[0])
        }
        searchUrl.searchParams.set('max_results', maxResults.toString())

        console.log(`Searching IB instruments with URL: ${searchUrl.toString()}`)
        console.log(`Base URL is: ${this.baseUrl}`)
        console.log(`About to make fetch call...`)
        
        let searchResponse: Response
        try {
          console.log(`🔥 Making fetch with timeout...`)
          
          // Create AbortController for timeout
          const controller = new AbortController()
          const timeoutId = setTimeout(() => {
            console.error(`❌ Request timeout after 45 seconds`)
            controller.abort()
          }, 45000) // 45 second timeout for slow IB responses
          
          searchResponse = await fetch(searchUrl.toString(), {
            method: 'GET',
            headers: {
              'Accept': 'application/json',
              'Content-Type': 'application/json'
            },
            signal: controller.signal
          })
          
          clearTimeout(timeoutId)
          console.log(`✅ Fetch completed successfully!`)
          console.log(`IB search response status: ${searchResponse.status}`)
          console.log(`IB search response ok: ${searchResponse.ok}`)
          console.log(`IB search response headers:`, searchResponse.headers ? Object.fromEntries(searchResponse.headers.entries()) : 'No headers')
        } catch (fetchError) {
          console.error(`❌ Fetch failed with error:`, fetchError)
          console.error(`❌ Falling back due to fetch error`)
          // Don't throw, let it fall back to local search
          searchResponse = { ok: false } as Response
        }
        
        console.log(`Checking searchResponse:`, searchResponse)
        console.log(`searchResponse defined:`, !!searchResponse)
        
        if (!searchResponse || !searchResponse.ok) {
          console.error(`IB search failed with status ${searchResponse?.status}: ${searchResponse?.statusText}`)
          try {
            const errorText = await searchResponse?.text()
            console.error(`IB search error body: ${errorText}`)
          } catch (e) {
            console.error(`Could not read error response body:`, e)
          }
          // Skip to fallback search
          searchResults = []
        } else if (searchResponse.ok) {
          const searchData = await searchResponse.json()
          console.log(`✅ IB search response received`)
          console.log(`IB search returned ${searchData.instruments?.length || 0} instruments`)
          console.log(`Raw IB search data keys:`, Object.keys(searchData || {}))
          console.log(`Full searchData:`, JSON.stringify(searchData, null, 2))
          console.log(`searchData.instruments type:`, typeof searchData.instruments)
          console.log(`searchData.instruments isArray:`, Array.isArray(searchData.instruments))
          if (searchData.instruments) {
            console.log(`searchData.instruments.length:`, searchData.instruments.length)
          }
          
          // Convert IB search results to our format
          if (searchData.instruments && Array.isArray(searchData.instruments) && searchData.instruments.length > 0) {
            console.log(`✅ Processing ${searchData.instruments.length} instruments`)
            searchResults = searchData.instruments.map((ibInstrument: any, index: number) => {
            const instrument: Instrument = {
              id: `${ibInstrument.symbol}-${ibInstrument.sec_type}`,
              symbol: ibInstrument.symbol,
              name: ibInstrument.name || ibInstrument.description || ibInstrument.symbol,
              venue: ibInstrument.exchange,
              assetClass: ibInstrument.sec_type,
              currency: ibInstrument.currency,
              contractId: ibInstrument.contract_id,
              localSymbol: ibInstrument.local_symbol,
              tradingClass: ibInstrument.trading_class,
              multiplier: ibInstrument.multiplier,
              expiry: ibInstrument.expiry,
              strike: ibInstrument.strike,
              right: ibInstrument.right,
              primaryExchange: ibInstrument.primary_exchange,
              description: ibInstrument.description,
              minTick: ibInstrument.min_tick,
              priceMagnifier: ibInstrument.price_magnifier,
              orderTypes: ibInstrument.order_types || [],
              validExchanges: ibInstrument.valid_exchanges || [],
              marketHours: ibInstrument.market_hours,
              liquidHours: ibInstrument.liquid_hours,
              timezone: ibInstrument.timezone
            }

            // Calculate relevance score based on search query
            let relevanceScore = this.calculateRelevanceScore(query, instrument)
            
            // Apply boost factors
            relevanceScore = this.applyBoostFactors(relevanceScore, query, instrument, options)
            
              return {
                instrument,
                relevanceScore,
                matchType: this.getMatchType(query, instrument),
                highlightedText: this.getHighlightedText(query, instrument)
              }
            })
            
            // Sort by relevance score
            searchResults.sort((a, b) => b.relevanceScore - a.relevanceScore)
            
            console.log(`IB search found ${searchResults.length} results for "${query}"`)
          } else {
            console.error(`IB search response does not contain instruments array:`, searchData)
            searchResults = []
          }
        }
      } catch (error) {
        console.warn('IB instrument search failed, falling back to local search:', error)
        console.warn('Error type:', typeof error)
        console.warn('Error message:', error?.message)
        console.warn('Error stack:', error?.stack)
      }

      // Fallback to local search if IB search failed or returned no results
      if (searchResults.length === 0) {
        console.log(`Falling back to local search for "${query}"`)
        
        // Get real instruments from backend
        const realInstruments = await this.fetchInstrumentsFromBackend()
        
        // Filter instruments based on search filters
        let filteredInstruments = realInstruments
        
        if (filters?.assetClasses && filters.assetClasses.length > 0) {
          filteredInstruments = filteredInstruments.filter(
            inst => filters.assetClasses.includes(inst.assetClass)
          )
        }
        
        if (filters?.venues && filters.venues.length > 0) {
          filteredInstruments = filteredInstruments.filter(
            inst => filters.venues.includes(inst.venue)
          )
        }
        
        if (filters?.currencies && filters.currencies.length > 0) {
          filteredInstruments = filteredInstruments.filter(
            inst => filters.currencies.includes(inst.currency)
          )
        }

        // Calculate relevance scores and create search results
        searchResults = filteredInstruments
          .map(instrument => {
            let relevanceScore = this.calculateRelevanceScore(query, instrument)
            if (relevanceScore === 0 && filters?.enableFuzzySearch !== false) {
              // Apply fuzzy search if enabled
              relevanceScore = this.calculateFuzzyScore(query, instrument)
            }
            
            if (relevanceScore === 0) return null
            
            // Apply boost factors
            relevanceScore = this.applyBoostFactors(relevanceScore, query, instrument, options)
            
            return {
              instrument,
              relevanceScore,
              matchType: this.getMatchType(query, instrument),
              highlightedText: this.getHighlightedText(query, instrument)
            }
          })
          .filter((result): result is InstrumentSearchResult => result !== null)
          .sort((a, b) => this.compareResults(a, b, filters))
          .slice(0, maxResults)
      }

      // Cache the results
      this.setCachedData(cacheKey, searchResults, 300000) // 5 minutes

      // Update offline cache with successful search results
      if (searchResults.length > 0 && offlineService.isOnline()) {
        offlineService.updateCacheWithSearchResults(searchResults)
      }

      console.log(`🎯 Final search results for "${query}":`, searchResults.length, 'instruments')
      if (searchResults.length > 0) {
        console.log(`🎯 First result:`, searchResults[0].instrument.symbol)
      }
      return searchResults
    } catch (error) {
      console.error('Failed to search instruments:', error)
      throw error
    }
  }

  private getMatchType(query: string, instrument: Instrument): 'symbol' | 'name' | 'venue' | 'alias' {
    if (!query || !instrument) return 'alias'
    
    const normalizedQuery = query.toLowerCase()
    
    if (instrument.symbol?.toLowerCase().includes(normalizedQuery)) {
      return 'symbol'
    }
    if (instrument.name?.toLowerCase().includes(normalizedQuery)) {
      return 'name'
    }
    if (instrument.venue?.toLowerCase().includes(normalizedQuery)) {
      return 'venue'
    }
    return 'alias'
  }

  private getHighlightedText(query: string, instrument: Instrument): string {
    const normalizedQuery = query.toLowerCase()
    
    // Simple highlighting - in production, you'd want more sophisticated highlighting
    if (instrument.symbol.toLowerCase().includes(normalizedQuery)) {
      return instrument.symbol
    }
    if (instrument.name.toLowerCase().includes(normalizedQuery)) {
      return instrument.name
    }
    return instrument.symbol
  }

  // Fuzzy search for handling typos
  private calculateFuzzyScore(query: string, instrument: Instrument): number {
    const normalizedQuery = query.toLowerCase()
    const symbol = instrument.symbol.toLowerCase()
    const name = instrument.name.toLowerCase()
    
    // Simple fuzzy matching - calculate edit distance
    const symbolDistance = this.levenshteinDistance(normalizedQuery, symbol)
    const nameDistance = this.levenshteinDistance(normalizedQuery, name)
    
    const maxLength = Math.max(normalizedQuery.length, symbol.length, name.length)
    
    // Convert distance to similarity score (0-100)
    const symbolSimilarity = Math.max(0, (maxLength - symbolDistance) / maxLength * 100)
    const nameSimilarity = Math.max(0, (maxLength - nameDistance) / maxLength * 100)
    
    // Only consider reasonable matches (>50% similarity)
    const threshold = 50
    if (symbolSimilarity > threshold) {
      return symbolSimilarity * 0.8 // Prefer symbol matches
    }
    if (nameSimilarity > threshold) {
      return nameSimilarity * 0.6 // Name matches get lower score
    }
    
    return 0
  }

  // Simple Levenshtein distance calculation
  private levenshteinDistance(str1: string, str2: string): number {
    const matrix = Array(str2.length + 1).fill(null).map(() => Array(str1.length + 1).fill(null))
    
    for (let i = 0; i <= str1.length; i++) matrix[0][i] = i
    for (let j = 0; j <= str2.length; j++) matrix[j][0] = j
    
    for (let j = 1; j <= str2.length; j++) {
      for (let i = 1; i <= str1.length; i++) {
        const indicator = str1[i - 1] === str2[j - 1] ? 0 : 1
        matrix[j][i] = Math.min(
          matrix[j][i - 1] + 1,     // deletion
          matrix[j - 1][i] + 1,     // insertion
          matrix[j - 1][i - 1] + indicator // substitution
        )
      }
    }
    
    return matrix[str2.length][str1.length]
  }

  // Apply boost factors to relevance scores
  private applyBoostFactors(
    baseScore: number, 
    query: string, 
    instrument: Instrument, 
    options: {
      filters?: SearchFilters
      favorites?: string[]
      venueStatus?: Record<string, VenueInfo>
    }
  ): number {
    let boostedScore = baseScore
    const { filters, favorites, venueStatus } = options
    const boostFactors = filters?.boostFactors || {
      exactSymbolMatch: 10,
      symbolPrefix: 5,
      nameMatch: 3,
      venueMatch: 1,
      assetClassBoost: 2
    }
    
    const normalizedQuery = query.toLowerCase()
    const symbol = instrument.symbol.toLowerCase()
    const name = instrument.name.toLowerCase()
    
    // Apply exact match boost
    if (symbol === normalizedQuery) {
      boostedScore *= boostFactors.exactSymbolMatch
    } else if (symbol.startsWith(normalizedQuery)) {
      boostedScore *= boostFactors.symbolPrefix
    } else if (name.includes(normalizedQuery)) {
      boostedScore *= boostFactors.nameMatch
    }
    
    // Boost favorites
    if (filters?.boostFavorites && favorites?.includes(instrument.id)) {
      boostedScore *= 3
    }
    
    // Boost connected venues
    if (filters?.boostConnectedVenues && venueStatus) {
      const venue = venueStatus[instrument.venue]
      if (venue?.connectionStatus === 'connected') {
        boostedScore *= 2
      }
    }
    
    // Boost common asset classes
    if (['STK', 'CASH', 'FUT'].includes(instrument.assetClass)) {
      boostedScore *= boostFactors.assetClassBoost
    }
    
    return boostedScore
  }

  // Compare results for sorting
  private compareResults(
    a: InstrumentSearchResult, 
    b: InstrumentSearchResult, 
    filters?: SearchFilters
  ): number {
    const sortBy = filters?.sortBy || 'relevance'
    const sortOrder = filters?.sortOrder || 'desc'
    const multiplier = sortOrder === 'asc' ? 1 : -1
    
    switch (sortBy) {
      case 'symbol':
        return multiplier * a.instrument.symbol.localeCompare(b.instrument.symbol)
      case 'name':
        return multiplier * a.instrument.name.localeCompare(b.instrument.name)
      case 'venue':
        return multiplier * a.instrument.venue.localeCompare(b.instrument.venue)
      case 'volume':
        // For now, use a simple heuristic based on asset class
        const volumeA = this.getEstimatedVolume(a.instrument)
        const volumeB = this.getEstimatedVolume(b.instrument)
        return multiplier * (volumeB - volumeA)
      case 'relevance':
      default:
        return b.relevanceScore - a.relevanceScore // Always descending for relevance
    }
  }

  // Estimate trading volume based on instrument characteristics
  private getEstimatedVolume(instrument: Instrument): number {
    // Simple heuristic - in production, you'd use real volume data
    let volume = 1000000 // Base volume
    
    // Boost for popular venues
    if (['NASDAQ', 'NYSE'].includes(instrument.venue)) volume *= 2
    if (instrument.venue === 'SMART') volume *= 1.5
    
    // Boost for common asset classes
    if (instrument.assetClass === 'STK') volume *= 1.5
    if (instrument.assetClass === 'CASH') volume *= 2
    if (instrument.assetClass === 'FUT') volume *= 1.2
    
    // Add some randomness for demo purposes
    return volume * (0.5 + Math.random())
  }

  // Fetch real instruments from backend using IB instrument search
  async fetchInstrumentsFromBackend(): Promise<Instrument[]> {
    try {
      const instruments: Instrument[] = []
      
      // Popular instruments endpoint removed to prevent fallback to mock data

      // Fallback: Fetch forex pairs from old endpoint if popular instruments failed
      if (instruments.length === 0) {
        try {
          const forexResponse = await fetch(`${this.baseUrl}/api/v1/ib/forex-pairs`)
          if (forexResponse && forexResponse.ok) {
            const forexData = await forexResponse.json()
            const forexInstruments = forexData.forex_pairs.map((pair: string) => ({
              id: `${pair.replace('/', '')}-CASH`,
              symbol: pair.replace('/', ''),
              name: `${pair} Currency Pair`,
              venue: 'IDEALPRO',
              assetClass: 'CASH',
              currency: pair.split('/')[1] || 'USD'
            }))
            instruments.push(...forexInstruments)
          }
        } catch (error) {
          console.warn('Failed to fetch forex pairs:', error)
        }
      }

      return instruments
    } catch (error) {
      console.error('Failed to fetch instruments from backend:', error)
      // Return empty array if backend fails - don't fall back to mock data
      return []
    }
  }

  // Get venue status information
  async getVenueStatus(): Promise<Record<string, VenueInfo>> {
    const cacheKey = 'venue-status'
    const cached = this.getCachedData<Record<string, VenueInfo>>(cacheKey)
    if (cached) {
      return cached
    }

    try {
      // Get real venue status from backend
      const venueStatus: Record<string, VenueInfo> = {}
      
      try {
        // Check IB connection status
        const ibStatusResponse = await fetch(`${this.baseUrl}/api/v1/ib/connection/status`)
        if (ibStatusResponse && ibStatusResponse.ok) {
          const ibStatus = await ibStatusResponse.json()
          
          // Add venue status based on real backend data
          const venues = ['NASDAQ', 'NYSE', 'IDEALPRO', 'GLOBEX', 'SMART']
          venues.forEach(venue => {
            venueStatus[venue] = {
              code: venue,
              name: this.getVenueName(venue),
              country: 'US',
              timezone: venue === 'GLOBEX' ? 'America/Chicago' : 'America/New_York',
              assetClasses: this.getVenueAssetClasses(venue),
              isConnected: ibStatus.connected || false,
              connectionStatus: ibStatus.connected ? 'connected' : 'disconnected',
              lastHeartbeat: new Date().toISOString()
            }
          })
        }
      } catch (error) {
        console.warn('Failed to get real venue status:', error)
        // Return empty object if backend fails - don't use mock data
      }

      this.setCachedData(cacheKey, venueStatus, 60000) // 1 minute cache
      return venueStatus
    } catch (error) {
      console.error('Failed to get venue status:', error)
      throw error
    }
  }

  // Get detailed instrument information
  async getInstrumentDetails(instrumentId: string): Promise<InstrumentDetailsResponse> {
    try {
      // Get real instrument from backend
      const instruments = await this.fetchInstrumentsFromBackend()
      const instrument = instruments.find(inst => inst.id === instrumentId)
      if (!instrument) {
        throw new Error('Instrument not found')
      }

      return {
        instrument,
        marketData: {
          lastPrice: 150.25,
          bid: 150.20,
          ask: 150.30,
          volume: 1000000,
          change: 2.15,
          changePercent: 1.45,
          lastUpdate: new Date().toISOString()
        },
        sessionInfo: {
          isOpen: true,
          timezone: 'America/New_York',
          marketHours: {
            open: '09:30',
            close: '16:00',
            days: ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']
          }
        }
      }
    } catch (error) {
      console.error('Failed to get instrument details:', error)
      throw error
    }
  }

  // Helper methods for venue information
  private getVenueName(venue: string): string {
    const venueNames: Record<string, string> = {
      'NASDAQ': 'NASDAQ Stock Market',
      'NYSE': 'New York Stock Exchange', 
      'IDEALPRO': 'IDEALPRO FX',
      'GLOBEX': 'CME Globex',
      'SMART': 'IB Smart Routing'
    }
    return venueNames[venue] || venue
  }

  private getVenueAssetClasses(venue: string): string[] {
    const venueAssetClasses: Record<string, string[]> = {
      'NASDAQ': ['STK', 'ETF'],
      'NYSE': ['STK'],
      'IDEALPRO': ['CASH'],
      'GLOBEX': ['FUT', 'OPT'],
      'SMART': ['STK', 'ETF', 'OPT']
    }
    return venueAssetClasses[venue] || []
  }

  // Clear all caches
  clearCache(): void {
    this.cache.clear()
  }

  // Get offline cache status
  getOfflineCacheStatus() {
    return {
      isOnline: offlineService.isOnline(),
      cacheInfo: offlineService.getCacheInfo()
    }
  }

  // Clear offline cache
  clearOfflineCache(): void {
    offlineService.clearOfflineCache()
  }
}

export const instrumentService = new InstrumentService()