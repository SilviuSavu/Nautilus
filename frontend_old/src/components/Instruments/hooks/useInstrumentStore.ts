import { create } from 'zustand'
import { persist } from 'zustand/middleware'
import { 
  Instrument, 
  InstrumentSearchResult, 
  VenueInfo, 
  SearchFilters,
  Watchlist,
  WatchlistItem
} from '../types/instrumentTypes'
import { instrumentService } from '../services/instrumentService'
import { realtimeService, RealtimePrice, VenueStatusUpdate } from '../services/realtimeService'

interface InstrumentStore {
  // Search state
  searchQuery: string
  searchResults: InstrumentSearchResult[]
  isLoading: boolean
  lastSearchTime: number
  
  // Favorites and recent selections (persisted)
  favorites: Instrument[]
  recentSelections: Instrument[]
  watchlists: Watchlist[]
  
  // Venue status
  venueStatus: Record<string, VenueInfo>
  lastVenueUpdate: number
  
  // Real-time data
  realtimePrices: Map<string, RealtimePrice>
  realtimeVenueStatus: Map<string, VenueStatusUpdate>
  isRealtimeConnected: boolean
  lastPriceUpdate: number
  
  // Search filters
  searchFilters: SearchFilters
  
  // Actions
  searchInstruments: (query: string, maxResults?: number) => Promise<InstrumentSearchResult[]>
  updateSearchQuery: (query: string) => void
  clearSearchResults: () => void
  
  // Favorites management
  addToFavorites: (instrument: Instrument) => void
  removeFromFavorites: (instrumentId: string) => void
  isFavorite: (instrumentId: string) => boolean
  
  // Recent selections
  addToRecentSelections: (instrument: Instrument) => void
  clearRecentSelections: () => void
  
  // Watchlist management
  createWatchlist: (name: string, description?: string) => Watchlist
  deleteWatchlist: (watchlistId: string) => void
  addToWatchlist: (watchlistId: string, instrument: Instrument) => void
  removeFromWatchlist: (watchlistId: string, instrumentId: string) => void
  updateWatchlistItem: (watchlistId: string, instrumentId: string, updates: Partial<Pick<WatchlistItem, 'notes' | 'alerts'>>) => void
  getWatchlist: (watchlistId: string) => Watchlist | undefined
  
  // Venue status
  updateVenueStatus: () => Promise<void>
  getVenueInfo: (venue: string) => VenueInfo | undefined
  
  // Filter management
  updateSearchFilters: (filters: Partial<SearchFilters>) => void
  resetSearchFilters: () => void
  
  // Count utilities
  getAssetClassCounts: () => Record<string, number>
  getVenueCounts: () => Record<string, number>
  
  // Real-time methods
  subscribeToRealtimeUpdates: () => void
  unsubscribeFromRealtimeUpdates: () => void
  getRealtimePrice: (instrumentId: string) => RealtimePrice | undefined
  getRealtimeVenueStatus: (venue: string) => VenueStatusUpdate | undefined
  subscribeToInstrument: (instrument: Instrument) => void
  unsubscribeFromInstrument: (instrumentId: string) => void
  
  // Cache management
  clearCache: () => void
}

const defaultSearchFilters: SearchFilters = {
  assetClasses: [],
  venues: [],
  currencies: [],
  onlyFavorites: false,
  onlyConnectedVenues: false
}

export const useInstrumentStore = create<InstrumentStore>()(
  persist(
    (set, get) => ({
      // Initial state
      searchQuery: '',
      searchResults: [],
      isLoading: false,
      lastSearchTime: 0,
      favorites: [],
      recentSelections: [],
      watchlists: [
        {
          id: 'default',
          name: 'My Watchlist',
          description: 'Default watchlist',
          items: [],
          createdAt: new Date().toISOString(),
          updatedAt: new Date().toISOString(),
          isDefault: true
        }
      ],
      venueStatus: {},
      lastVenueUpdate: 0,
      realtimePrices: new Map(),
      realtimeVenueStatus: new Map(),
      isRealtimeConnected: false,
      lastPriceUpdate: 0,
      searchFilters: defaultSearchFilters,

      // Search actions
      searchInstruments: async (query: string, maxResults = 50) => {
        set({ isLoading: true })
        
        try {
          const startTime = performance.now()
          const filters = get().searchFilters
          
          const results = await instrumentService.searchInstruments(query, {
            maxResults,
            filters
          })
          
          const searchTime = performance.now() - startTime
          
          set({ 
            searchResults: results,
            lastSearchTime: searchTime,
            isLoading: false 
          })
          
          return results
        } catch (error) {
          console.error('Failed to search instruments:', error)
          set({ 
            searchResults: [],
            isLoading: false 
          })
          throw error
        }
      },

      updateSearchQuery: (query: string) => {
        set({ searchQuery: query })
      },

      clearSearchResults: () => {
        set({ 
          searchResults: [],
          searchQuery: '',
          lastSearchTime: 0 
        })
      },

      // Favorites management
      addToFavorites: (instrument: Instrument) => {
        const favorites = get().favorites
        const exists = favorites.some(fav => fav.id === instrument.id)
        
        if (!exists) {
          set({ 
            favorites: [instrument, ...favorites].slice(0, 100) // Limit to 100 favorites
          })
        }
      },

      removeFromFavorites: (instrumentId: string) => {
        const favorites = get().favorites
        set({ 
          favorites: favorites.filter(fav => fav.id !== instrumentId)
        })
      },

      isFavorite: (instrumentId: string) => {
        return get().favorites.some(fav => fav.id === instrumentId)
      },

      // Recent selections
      addToRecentSelections: (instrument: Instrument) => {
        const recent = get().recentSelections
        const filtered = recent.filter(item => item.id !== instrument.id)
        
        set({ 
          recentSelections: [instrument, ...filtered].slice(0, 20) // Limit to 20 recent
        })
      },

      clearRecentSelections: () => {
        set({ recentSelections: [] })
      },

      // Watchlist management
      createWatchlist: (name: string, description?: string) => {
        const watchlist: Watchlist = {
          id: `watchlist_${Date.now()}`,
          name,
          description,
          items: [],
          createdAt: new Date().toISOString(),
          updatedAt: new Date().toISOString()
        }
        
        const watchlists = get().watchlists
        set({ watchlists: [...watchlists, watchlist] })
        
        return watchlist
      },

      deleteWatchlist: (watchlistId: string) => {
        const watchlists = get().watchlists
        // Prevent deletion of default watchlist
        if (watchlistId === 'default') return
        
        set({ 
          watchlists: watchlists.filter(w => w.id !== watchlistId)
        })
      },

      addToWatchlist: (watchlistId: string, instrument: Instrument) => {
        const watchlists = get().watchlists
        const updatedWatchlists = watchlists.map(watchlist => {
          if (watchlist.id === watchlistId) {
            const exists = watchlist.items.some(item => item.instrument.id === instrument.id)
            if (!exists) {
              const newItem: WatchlistItem = {
                instrument,
                addedAt: new Date().toISOString()
              }
              return {
                ...watchlist,
                items: [...watchlist.items, newItem],
                updatedAt: new Date().toISOString()
              }
            }
          }
          return watchlist
        })
        
        set({ watchlists: updatedWatchlists })
      },

      removeFromWatchlist: (watchlistId: string, instrumentId: string) => {
        const watchlists = get().watchlists
        const updatedWatchlists = watchlists.map(watchlist => {
          if (watchlist.id === watchlistId) {
            return {
              ...watchlist,
              items: watchlist.items.filter(item => item.instrument.id !== instrumentId),
              updatedAt: new Date().toISOString()
            }
          }
          return watchlist
        })
        
        set({ watchlists: updatedWatchlists })
      },

      updateWatchlistItem: (watchlistId: string, instrumentId: string, updates: Partial<Pick<WatchlistItem, 'notes' | 'alerts'>>) => {
        const watchlists = get().watchlists
        const updatedWatchlists = watchlists.map(watchlist => {
          if (watchlist.id === watchlistId) {
            return {
              ...watchlist,
              items: watchlist.items.map(item => {
                if (item.instrument.id === instrumentId) {
                  return {
                    ...item,
                    ...updates
                  }
                }
                return item
              }),
              updatedAt: new Date().toISOString()
            }
          }
          return watchlist
        })
        
        set({ watchlists: updatedWatchlists })
      },

      getWatchlist: (watchlistId: string) => {
        return get().watchlists.find(w => w.id === watchlistId)
      },

      // Venue status
      updateVenueStatus: async () => {
        try {
          const venueStatus = await instrumentService.getVenueStatus()
          set({ 
            venueStatus,
            lastVenueUpdate: Date.now()
          })
        } catch (error) {
          console.error('Failed to update venue status:', error)
        }
      },

      getVenueInfo: (venue: string) => {
        return get().venueStatus[venue]
      },

      // Filter management
      updateSearchFilters: (filters: Partial<SearchFilters>) => {
        const currentFilters = get().searchFilters
        set({ 
          searchFilters: { ...currentFilters, ...filters }
        })
      },

      resetSearchFilters: () => {
        set({ searchFilters: defaultSearchFilters })
      },

      // Count utilities
      getAssetClassCounts: () => {
        // For now, return static counts until we load instruments asynchronously
        // This prevents the crash while we implement proper async loading
        return {
          'STK': 5,
          'CASH': 20,
          'FUT': 2,
          'ETF': 4
        }
      },

      getVenueCounts: () => {
        // For now, return static counts until we load instruments asynchronously
        // This prevents the crash while we implement proper async loading
        return {
          'NASDAQ': 5,
          'IDEALPRO': 20,
          'NYSE': 2,
          'GLOBEX': 4
        }
      },

      // Real-time methods
      subscribeToRealtimeUpdates: () => {
        // Subscribe to price updates
        realtimeService.subscribe('price', (update) => {
          if (update.type === 'price') {
            const priceData = update.data as RealtimePrice
            set((state) => ({
              realtimePrices: new Map(state.realtimePrices.set(priceData.instrumentId, priceData)),
              lastPriceUpdate: Date.now(),
              isRealtimeConnected: true
            }))
          }
        })

        // Subscribe to venue status updates
        realtimeService.subscribe('venue_status', (update) => {
          if (update.type === 'venue_status') {
            const venueData = update.data as VenueStatusUpdate
            set((state) => ({
              realtimeVenueStatus: new Map(state.realtimeVenueStatus.set(venueData.venue, venueData))
            }))
          }
        })

        // Subscribe to connection status
        realtimeService.subscribe('heartbeat', () => {
          const connectionStatus = realtimeService.getConnectionStatus()
          set({
            isRealtimeConnected: connectionStatus.isConnected
          })
        })
      },

      unsubscribeFromRealtimeUpdates: () => {
        // This would need to be implemented in the realtime service
        // For now, we'll just clear the data
        set({
          realtimePrices: new Map(),
          realtimeVenueStatus: new Map(),
          isRealtimeConnected: false,
          lastPriceUpdate: 0
        })
      },

      getRealtimePrice: (instrumentId: string) => {
        return get().realtimePrices.get(instrumentId)
      },

      getRealtimeVenueStatus: (venue: string) => {
        return get().realtimeVenueStatus.get(venue)
      },

      subscribeToInstrument: (instrument: Instrument) => {
        realtimeService.subscribeToInstrument(instrument.id)
        // Also subscribe to venue status for this instrument's venue
        realtimeService.subscribeToVenue(instrument.venue)
      },

      unsubscribeFromInstrument: (instrumentId: string) => {
        realtimeService.unsubscribeFromInstrument(instrumentId)
      },

      // Cache management
      clearCache: () => {
        set({
          searchResults: [],
          searchQuery: '',
          lastSearchTime: 0,
          venueStatus: {},
          lastVenueUpdate: 0,
          realtimePrices: new Map(),
          realtimeVenueStatus: new Map(),
          lastPriceUpdate: 0
        })
      }
    }),
    {
      name: 'instrument-store',
      // Only persist user preferences, not search results
      partialize: (state) => ({
        favorites: state.favorites,
        recentSelections: state.recentSelections,
        watchlists: state.watchlists,
        searchFilters: state.searchFilters
      })
    }
  )
)