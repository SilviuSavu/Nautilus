export interface Instrument {
  id: string
  symbol: string
  name: string
  venue: string
  assetClass: string
  currency: string
  exchange?: string
  description?: string
  minTick?: number
  contractMultiplier?: number
  tradingClass?: string
  sessionInfo?: TradingSessionInfo
  lastPrice?: number
  lastUpdate?: string
  // IB-specific fields
  contractId?: number
  localSymbol?: string
  multiplier?: string
  expiry?: string
  strike?: number
  right?: string // 'C' for Call, 'P' for Put
  primaryExchange?: string
  priceMagnifier?: number
  orderTypes?: string[]
  validExchanges?: string[]
  marketHours?: string
  liquidHours?: string
  timezone?: string
}

export interface TradingSessionInfo {
  isOpen: boolean
  nextOpen?: string
  nextClose?: string
  marketHours?: MarketHours
  timezone?: string
}

export interface MarketHours {
  open: string
  close: string
  days: string[]
}

export interface InstrumentSearchResult {
  instrument: Instrument
  relevanceScore: number
  matchType: 'symbol' | 'name' | 'venue' | 'alias'
  highlightedText?: string
}

export interface VenueInfo {
  code: string
  name: string
  country: string
  timezone: string
  assetClasses: string[]
  isConnected: boolean
  connectionStatus: VenueConnectionStatus
  lastHeartbeat?: string
  errorMessage?: string
}

export type VenueConnectionStatus = 
  | 'connected' 
  | 'disconnected' 
  | 'connecting' 
  | 'error' 
  | 'maintenance'

export interface AssetClassFilter {
  code: string
  name: string
  enabled: boolean
  count?: number
}

export interface SearchFilters {
  assetClasses: string[]
  venues: string[]
  currencies: string[]
  minPrice?: number
  maxPrice?: number
  onlyFavorites?: boolean
  onlyConnectedVenues?: boolean
  
  // Ranking and sorting
  sortBy?: 'relevance' | 'symbol' | 'name' | 'volume' | 'venue'
  sortOrder?: 'asc' | 'desc'
  
  // Boost factors for relevance scoring
  boostFactors?: {
    exactSymbolMatch: number
    symbolPrefix: number
    nameMatch: number
    venueMatch: number
    assetClassBoost: number
  }
  
  // Additional boost options
  boostConnectedVenues?: boolean
  boostFavorites?: boolean
  enableFuzzySearch?: boolean
}

export interface InstrumentCatalog {
  instruments: Instrument[]
  venues: VenueInfo[]
  assetClasses: AssetClassFilter[]
  totalInstruments: number
  lastUpdated: string
}

export interface WatchlistItem {
  instrument: Instrument
  addedAt: string
  notes?: string
  alerts?: PriceAlert[]
}

export interface PriceAlert {
  id: string
  type: 'above' | 'below'
  price: number
  isActive: boolean
  createdAt: string
  triggeredAt?: string
}

export interface Watchlist {
  id: string
  name: string
  description?: string
  items: WatchlistItem[]
  createdAt: string
  updatedAt: string
  isDefault?: boolean
}

// API Response types
export interface InstrumentSearchResponse {
  query: string
  results: InstrumentSearchResult[]
  totalCount: number
  searchTime: number
  filters?: SearchFilters
}

export interface VenueStatusResponse {
  venues: Record<string, VenueInfo>
  lastUpdated: string
}

export interface InstrumentDetailsResponse {
  instrument: Instrument
  marketData?: {
    lastPrice: number
    bid: number
    ask: number
    volume: number
    change: number
    changePercent: number
    lastUpdate: string
  }
  sessionInfo: TradingSessionInfo
}