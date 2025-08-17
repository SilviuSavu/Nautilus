import { 
  Instrument, 
  InstrumentSearchResult, 
  VenueInfo, 
  SearchFilters,
  InstrumentSearchResponse,
  VenueStatusResponse,
  InstrumentDetailsResponse
} from '../types/instrumentTypes'

class InstrumentService {
  private baseUrl: string
  private cache: Map<string, { data: any; timestamp: number; ttl: number }> = new Map()

  constructor() {
    this.baseUrl = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000'
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
    const normalizedQuery = query.toLowerCase().trim()
    const symbol = instrument.symbol.toLowerCase()
    const name = instrument.name.toLowerCase()
    const venue = instrument.venue.toLowerCase()
    
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

  // Enhanced instrument search with advanced filtering and ranking
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

    const cacheKey = `search:${query}:${JSON.stringify(filters)}`
    const cached = this.getCachedData<InstrumentSearchResult[]>(cacheKey)
    if (cached) {
      return cached.slice(0, maxResults)
    }

    try {
      // For now, we'll use the predefined instruments and implement real API call later
      const predefinedInstruments = this.getPredefinedInstruments()
      
      // Filter instruments based on search filters
      let filteredInstruments = predefinedInstruments
      
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
      const searchResults: InstrumentSearchResult[] = filteredInstruments
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

      // Cache the results
      this.setCachedData(cacheKey, searchResults, 300000) // 5 minutes

      return searchResults
    } catch (error) {
      console.error('Failed to search instruments:', error)
      throw error
    }
  }

  private getMatchType(query: string, instrument: Instrument): 'symbol' | 'name' | 'venue' | 'alias' {
    const normalizedQuery = query.toLowerCase()
    
    if (instrument.symbol.toLowerCase().includes(normalizedQuery)) {
      return 'symbol'
    }
    if (instrument.name.toLowerCase().includes(normalizedQuery)) {
      return 'name'
    }
    if (instrument.venue.toLowerCase().includes(normalizedQuery)) {
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

  // Get venue status information
  async getVenueStatus(): Promise<Record<string, VenueInfo>> {
    const cacheKey = 'venue-status'
    const cached = this.getCachedData<Record<string, VenueInfo>>(cacheKey)
    if (cached) {
      return cached
    }

    try {
      // For now, return mock venue status
      const venueStatus: Record<string, VenueInfo> = {
        'NASDAQ': {
          code: 'NASDAQ',
          name: 'NASDAQ Stock Market',
          country: 'US',
          timezone: 'America/New_York',
          assetClasses: ['STK', 'ETF'],
          isConnected: true,
          connectionStatus: 'connected',
          lastHeartbeat: new Date().toISOString()
        },
        'NYSE': {
          code: 'NYSE',
          name: 'New York Stock Exchange',
          country: 'US',
          timezone: 'America/New_York',
          assetClasses: ['STK'],
          isConnected: true,
          connectionStatus: 'connected',
          lastHeartbeat: new Date().toISOString()
        },
        'IDEALPRO': {
          code: 'IDEALPRO',
          name: 'IDEALPRO FX',
          country: 'US',
          timezone: 'America/New_York',
          assetClasses: ['CASH'],
          isConnected: true,
          connectionStatus: 'connected',
          lastHeartbeat: new Date().toISOString()
        },
        'GLOBEX': {
          code: 'GLOBEX',
          name: 'CME Globex',
          country: 'US',
          timezone: 'America/Chicago',
          assetClasses: ['FUT', 'OPT'],
          isConnected: true,
          connectionStatus: 'connected',
          lastHeartbeat: new Date().toISOString()
        },
        'SMART': {
          code: 'SMART',
          name: 'IB Smart Routing',
          country: 'Global',
          timezone: 'UTC',
          assetClasses: ['STK', 'ETF', 'OPT'],
          isConnected: true,
          connectionStatus: 'connected',
          lastHeartbeat: new Date().toISOString()
        }
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
      // TODO: Implement real API call
      // For now, return mock data
      const instrument = this.getPredefinedInstruments().find(inst => inst.id === instrumentId)
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

  // Predefined instruments (expanding the existing list)
  getPredefinedInstruments(): Instrument[] {
    return [
      // US Stocks
      { id: 'AAPL-STK', symbol: 'AAPL', name: 'Apple Inc.', venue: 'NASDAQ', assetClass: 'STK', currency: 'USD' },
      { id: 'MSFT-STK', symbol: 'MSFT', name: 'Microsoft Corporation', venue: 'NASDAQ', assetClass: 'STK', currency: 'USD' },
      { id: 'GOOGL-STK', symbol: 'GOOGL', name: 'Alphabet Inc.', venue: 'NASDAQ', assetClass: 'STK', currency: 'USD' },
      { id: 'AMZN-STK', symbol: 'AMZN', name: 'Amazon.com Inc.', venue: 'NASDAQ', assetClass: 'STK', currency: 'USD' },
      { id: 'TSLA-STK', symbol: 'TSLA', name: 'Tesla Inc.', venue: 'NASDAQ', assetClass: 'STK', currency: 'USD' },
      { id: 'NVDA-STK', symbol: 'NVDA', name: 'NVIDIA Corporation', venue: 'NASDAQ', assetClass: 'STK', currency: 'USD' },
      { id: 'META-STK', symbol: 'META', name: 'Meta Platforms Inc.', venue: 'NASDAQ', assetClass: 'STK', currency: 'USD' },
      { id: 'NFLX-STK', symbol: 'NFLX', name: 'Netflix Inc.', venue: 'NASDAQ', assetClass: 'STK', currency: 'USD' },
      { id: 'ADBE-STK', symbol: 'ADBE', name: 'Adobe Inc.', venue: 'NASDAQ', assetClass: 'STK', currency: 'USD' },
      { id: 'CRM-STK', symbol: 'CRM', name: 'Salesforce Inc.', venue: 'NYSE', assetClass: 'STK', currency: 'USD' },
      
      // ETFs
      { id: 'SPY-ETF', symbol: 'SPY', name: 'SPDR S&P 500 ETF', venue: 'ARCA', assetClass: 'STK', currency: 'USD' },
      { id: 'QQQ-ETF', symbol: 'QQQ', name: 'Invesco QQQ Trust', venue: 'NASDAQ', assetClass: 'STK', currency: 'USD' },
      { id: 'IWM-ETF', symbol: 'IWM', name: 'iShares Russell 2000 ETF', venue: 'ARCA', assetClass: 'STK', currency: 'USD' },
      { id: 'VTI-ETF', symbol: 'VTI', name: 'Vanguard Total Stock Market ETF', venue: 'ARCA', assetClass: 'STK', currency: 'USD' },
      
      // Forex
      { id: 'EURUSD-CASH', symbol: 'EURUSD', name: 'Euro / US Dollar', venue: 'IDEALPRO', assetClass: 'CASH', currency: 'USD' },
      { id: 'GBPUSD-CASH', symbol: 'GBPUSD', name: 'British Pound / US Dollar', venue: 'IDEALPRO', assetClass: 'CASH', currency: 'USD' },
      { id: 'USDJPY-CASH', symbol: 'USDJPY', name: 'US Dollar / Japanese Yen', venue: 'IDEALPRO', assetClass: 'CASH', currency: 'JPY' },
      { id: 'USDCHF-CASH', symbol: 'USDCHF', name: 'US Dollar / Swiss Franc', venue: 'IDEALPRO', assetClass: 'CASH', currency: 'CHF' },
      { id: 'AUDUSD-CASH', symbol: 'AUDUSD', name: 'Australian Dollar / US Dollar', venue: 'IDEALPRO', assetClass: 'CASH', currency: 'USD' },
      { id: 'USDCAD-CASH', symbol: 'USDCAD', name: 'US Dollar / Canadian Dollar', venue: 'IDEALPRO', assetClass: 'CASH', currency: 'CAD' },
      
      // Futures
      { id: 'ES-FUT', symbol: 'ES', name: 'E-mini S&P 500', venue: 'GLOBEX', assetClass: 'FUT', currency: 'USD' },
      { id: 'NQ-FUT', symbol: 'NQ', name: 'E-mini NASDAQ-100', venue: 'GLOBEX', assetClass: 'FUT', currency: 'USD' },
      { id: 'YM-FUT', symbol: 'YM', name: 'E-mini Dow Jones', venue: 'GLOBEX', assetClass: 'FUT', currency: 'USD' },
      { id: 'RTY-FUT', symbol: 'RTY', name: 'E-mini Russell 2000', venue: 'GLOBEX', assetClass: 'FUT', currency: 'USD' },
      { id: 'CL-FUT', symbol: 'CL', name: 'Crude Oil', venue: 'NYMEX', assetClass: 'FUT', currency: 'USD' },
      { id: 'NG-FUT', symbol: 'NG', name: 'Natural Gas', venue: 'NYMEX', assetClass: 'FUT', currency: 'USD' },
      { id: 'GC-FUT', symbol: 'GC', name: 'Gold', venue: 'NYMEX', assetClass: 'FUT', currency: 'USD' },
      { id: 'SI-FUT', symbol: 'SI', name: 'Silver', venue: 'NYMEX', assetClass: 'FUT', currency: 'USD' },
      
      // Bonds
      { id: 'ZN-FUT', symbol: 'ZN', name: '10-Year Treasury Note', venue: 'CBOT', assetClass: 'FUT', currency: 'USD' },
      { id: 'ZB-FUT', symbol: 'ZB', name: '30-Year Treasury Bond', venue: 'CBOT', assetClass: 'FUT', currency: 'USD' },
      
      // International Stocks (examples)
      { id: 'ASML-STK', symbol: 'ASML', name: 'ASML Holding N.V.', venue: 'NASDAQ', assetClass: 'STK', currency: 'USD' },
      { id: 'TSM-STK', symbol: 'TSM', name: 'Taiwan Semiconductor', venue: 'NYSE', assetClass: 'STK', currency: 'USD' },
      
      // Crypto-related instruments (ETFs and futures)
      { id: 'BITO-ETF', symbol: 'BITO', name: 'ProShares Bitcoin Strategy ETF', venue: 'NYSE', assetClass: 'STK', currency: 'USD' },
      { id: 'GBTC-ETF', symbol: 'GBTC', name: 'Grayscale Bitcoin Trust', venue: 'OTC', assetClass: 'STK', currency: 'USD' }
    ]
  }

  // Clear all caches
  clearCache(): void {
    this.cache.clear()
  }
}

export const instrumentService = new InstrumentService()