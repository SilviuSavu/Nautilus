import { render, screen, fireEvent, waitFor } from '@testing-library/react'
import { describe, it, expect, vi, beforeEach } from 'vitest'
import { InstrumentSearch } from '../InstrumentSearch'
import { useInstrumentStore } from '../hooks/useInstrumentStore'
import { InstrumentSearchResult, Instrument } from '../types/instrumentTypes'

// Mock the store
vi.mock('../hooks/useInstrumentStore')
const mockUseInstrumentStore = vi.mocked(useInstrumentStore)

// Mock instrument data
const mockInstrument: Instrument = {
  id: 'AAPL-STK',
  symbol: 'AAPL',
  name: 'Apple Inc',
  venue: 'SMART',
  assetClass: 'STK',
  currency: 'USD',
  contractId: 265598
}

const mockSearchResult: InstrumentSearchResult = {
  instrument: mockInstrument,
  relevanceScore: 100,
  matchType: 'exact_symbol'
}

describe('InstrumentSearch', () => {
  const mockSearchInstruments = vi.fn()
  const mockAddToFavorites = vi.fn()
  const mockRemoveFromFavorites = vi.fn()
  const mockAddToRecentSelections = vi.fn()
  const mockAddToWatchlist = vi.fn()

  beforeEach(() => {
    vi.clearAllMocks()
    
    mockUseInstrumentStore.mockReturnValue({
      searchInstruments: mockSearchInstruments,
      addToFavorites: mockAddToFavorites,
      removeFromFavorites: mockRemoveFromFavorites,
      favorites: [],
      recentSelections: [],
      addToRecentSelections: mockAddToRecentSelections,
      isLoading: false,
      venueStatus: {},
      searchFilters: {
        assetClasses: [],
        venues: [],
        currencies: []
      },
      updateVenueStatus: vi.fn(),
      lastSearchTime: 150,
      watchlists: [{
        id: 'default',
        name: 'Default Watchlist',
        items: [],
        createdAt: new Date().toISOString(),
        updatedAt: new Date().toISOString()
      }],
      addToWatchlist: mockAddToWatchlist,
      getAssetClassCounts: vi.fn(() => ({ STK: 5, FUT: 2, CASH: 3 })),
      getVenueCounts: vi.fn(() => ({ SMART: 5, NYSE: 3, NASDAQ: 2 })),
      searchQuery: '',
      searchResults: [],
      lastVenueUpdate: Date.now(),
      realtimePrices: new Map(),
      realtimeVenueStatus: new Map(),
      isRealtimeConnected: false,
      lastPriceUpdate: Date.now(),
      updateSearchQuery: vi.fn(),
      clearSearchResults: vi.fn(),
      isFavorite: vi.fn(() => false),
      clearRecentSelections: vi.fn(),
      updateSearchFilters: vi.fn(),
      createWatchlist: vi.fn(),
      updateWatchlist: vi.fn(),
      deleteWatchlist: vi.fn(),
      removeFromWatchlist: vi.fn(),
      subscribeToRealtimeData: vi.fn(),
      unsubscribeFromRealtimeData: vi.fn(),
      updateRealtimePrice: vi.fn(),
      updateRealtimeVenueStatus: vi.fn(),
      setIsRealtimeConnected: vi.fn(),
      getInstrumentDetails: vi.fn()
    })
  })

  it('should render search input with placeholder', () => {
    render(<InstrumentSearch />)
    
    const searchInput = screen.getByPlaceholderText(/search instruments across all venues/i)
    expect(searchInput).toBeInTheDocument()
  })

  it('should call searchInstruments when user types', async () => {
    mockSearchInstruments.mockResolvedValue([mockSearchResult])
    
    render(<InstrumentSearch />)
    
    const searchInput = screen.getByPlaceholderText(/search instruments across all venues/i)
    fireEvent.change(searchInput, { target: { value: 'AAPL' } })
    
    // Wait for debounced search
    await waitFor(() => {
      expect(mockSearchInstruments).toHaveBeenCalledWith('AAPL', 50)
    }, { timeout: 1000 })
  })

  it('should display search results', async () => {
    mockSearchInstruments.mockResolvedValue([mockSearchResult])
    
    render(<InstrumentSearch />)
    
    const searchInput = screen.getByPlaceholderText(/search instruments across all venues/i)
    fireEvent.change(searchInput, { target: { value: 'AAPL' } })
    
    await waitFor(() => {
      expect(screen.getByText('AAPL')).toBeInTheDocument()
      expect(screen.getByText('Apple Inc')).toBeInTheDocument()
      expect(screen.getByText('STK')).toBeInTheDocument()
    })
  })

  it('should call onInstrumentSelect when instrument is clicked', async () => {
    const mockOnSelect = vi.fn()
    mockSearchInstruments.mockResolvedValue([mockSearchResult])
    
    render(<InstrumentSearch onInstrumentSelect={mockOnSelect} />)
    
    const searchInput = screen.getByPlaceholderText(/search instruments across all venues/i)
    fireEvent.change(searchInput, { target: { value: 'AAPL' } })
    
    await waitFor(() => {
      expect(screen.getByText('AAPL')).toBeInTheDocument()
    })
    
    fireEvent.click(screen.getByText('AAPL'))
    
    expect(mockOnSelect).toHaveBeenCalledWith(mockInstrument)
    expect(mockAddToRecentSelections).toHaveBeenCalledWith(mockInstrument)
  })

  it('should show "No instruments found" for empty results', async () => {
    mockSearchInstruments.mockResolvedValue([])
    
    render(<InstrumentSearch />)
    
    const searchInput = screen.getByPlaceholderText(/search instruments across all venues/i)
    fireEvent.change(searchInput, { target: { value: 'NONEXIST' } })
    
    await waitFor(() => {
      expect(screen.getByTestId('no-instruments-found')).toBeInTheDocument()
    })
  })

  it('should toggle favorites when heart icon is clicked', async () => {
    mockSearchInstruments.mockResolvedValue([mockSearchResult])
    
    render(<InstrumentSearch />)
    
    const searchInput = screen.getByPlaceholderText(/search instruments across all venues/i)
    fireEvent.change(searchInput, { target: { value: 'AAPL' } })
    
    await waitFor(() => {
      expect(screen.getByText('AAPL')).toBeInTheDocument()
    })
    
    const favoriteButton = screen.getByRole('button', { name: /add to favorites/i })
    fireEvent.click(favoriteButton)
    
    expect(mockAddToFavorites).toHaveBeenCalledWith(mockInstrument)
  })

  it('should display favorites section when available', () => {
    mockUseInstrumentStore.mockReturnValue({
      ...mockUseInstrumentStore(),
      favorites: [mockInstrument]
    })
    
    render(<InstrumentSearch />)
    
    expect(screen.getByText('Favorite Instruments')).toBeInTheDocument()
    expect(screen.getByText('AAPL')).toBeInTheDocument()
  })

  it('should display recent selections section when available', () => {
    mockUseInstrumentStore.mockReturnValue({
      ...mockUseInstrumentStore(),
      recentSelections: [mockInstrument]
    })
    
    render(<InstrumentSearch />)
    
    expect(screen.getByText('Recent Selections')).toBeInTheDocument()
    expect(screen.getByText('AAPL')).toBeInTheDocument()
  })

  it('should handle search errors gracefully', async () => {
    const consoleError = vi.spyOn(console, 'error').mockImplementation(() => {})
    mockSearchInstruments.mockRejectedValue(new Error('Search failed'))
    
    render(<InstrumentSearch />)
    
    const searchInput = screen.getByPlaceholderText(/search instruments across all venues/i)
    fireEvent.change(searchInput, { target: { value: 'AAPL' } })
    
    await waitFor(() => {
      expect(consoleError).toHaveBeenCalledWith('Search failed:', expect.any(Error))
    })
    
    consoleError.mockRestore()
  })
})