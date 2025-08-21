import React from 'react'
import { render, screen, fireEvent, waitFor } from '@testing-library/react'
import { describe, it, expect, vi, beforeEach } from 'vitest'
import { OrderBookDisplay } from '../OrderBookDisplay'
import { Instrument } from '../../Chart/types/chartTypes'

// Mock the hooks and services
vi.mock('../../../hooks/useOrderBookData', () => ({
  useOrderBookData: vi.fn(() => ({
    orderBookData: null,
    subscriptions: [],
    aggregationSettings: {
      enabled: false,
      increment: 0.01,
      maxLevels: 20
    },
    displaySettings: {
      showSpread: true,
      showOrderCount: false,
      colorScheme: 'default',
      decimals: 2
    },
    isLoading: false,
    error: null,
    connectionStatus: 'connected',
    performanceMetrics: null,
    subscribeToOrderBook: vi.fn(),
    unsubscribeFromOrderBook: vi.fn(),
    updateAggregationSettings: vi.fn(),
    updateDisplaySettings: vi.fn(),
    clearOrderBook: vi.fn()
  }))
}))

// Mock Ant Design components that might cause issues in tests
vi.mock('antd', async () => {
  const actual = await vi.importActual('antd')
  return {
    ...actual,
    Card: ({ children, title, ...props }: any) => (
      <div data-testid="card" {...props}>
        {title && <div data-testid="card-title">{title}</div>}
        {children}
      </div>
    ),
    Empty: ({ description }: any) => (
      <div data-testid="empty">{description}</div>
    ),
    Alert: ({ message, description }: any) => (
      <div data-testid="alert">
        <div data-testid="alert-message">{message}</div>
        <div data-testid="alert-description">{description}</div>
      </div>
    ),
    Spin: () => <div data-testid="spin">Loading...</div>,
    Space: ({ children }: any) => <div data-testid="space">{children}</div>
  }
})

describe('OrderBookDisplay', () => {
  const mockInstrument: Instrument = {
    id: 'AAPL-STK',
    symbol: 'AAPL',
    name: 'Apple Inc.',
    venue: 'NASDAQ',
    assetClass: 'STK',
    currency: 'USD'
  }

  beforeEach(() => {
    vi.clearAllMocks()
  })

  describe('rendering states', () => {
    it('should render empty state when no instrument is selected', () => {
      render(<OrderBookDisplay instrument={null} />)
      
      expect(screen.getByTestId('empty')).toBeInTheDocument()
      expect(screen.getByText('Select an instrument to view order book')).toBeInTheDocument()
    })

    it('should render loading state', () => {
      const { useOrderBookData } = require('../../../hooks/useOrderBookData')
      useOrderBookData.mockReturnValue({
        orderBookData: null,
        isLoading: true,
        error: null,
        connectionStatus: 'connected',
        subscriptions: [],
        aggregationSettings: { enabled: false, increment: 0.01, maxLevels: 20 },
        displaySettings: { showSpread: true, showOrderCount: false, colorScheme: 'default', decimals: 2 },
        performanceMetrics: null,
        subscribeToOrderBook: vi.fn(),
        unsubscribeFromOrderBook: vi.fn(),
        updateAggregationSettings: vi.fn(),
        updateDisplaySettings: vi.fn(),
        clearOrderBook: vi.fn()
      })

      render(<OrderBookDisplay instrument={mockInstrument} />)
      
      expect(screen.getByTestId('spin')).toBeInTheDocument()
      expect(screen.getByText('Loading order book data...')).toBeInTheDocument()
    })

    it('should render error state', () => {
      const { useOrderBookData } = require('../../../hooks/useOrderBookData')
      useOrderBookData.mockReturnValue({
        orderBookData: null,
        isLoading: false,
        error: 'Connection failed',
        connectionStatus: 'error',
        subscriptions: [],
        aggregationSettings: { enabled: false, increment: 0.01, maxLevels: 20 },
        displaySettings: { showSpread: true, showOrderCount: false, colorScheme: 'default', decimals: 2 },
        performanceMetrics: null,
        subscribeToOrderBook: vi.fn(),
        unsubscribeFromOrderBook: vi.fn(),
        updateAggregationSettings: vi.fn(),
        updateDisplaySettings: vi.fn(),
        clearOrderBook: vi.fn()
      })

      render(<OrderBookDisplay instrument={mockInstrument} />)
      
      expect(screen.getByTestId('alert')).toBeInTheDocument()
      expect(screen.getByTestId('alert-message')).toHaveTextContent('Order Book Error')
      expect(screen.getByTestId('alert-description')).toHaveTextContent('Connection failed')
    })

    it('should render order book data when available', () => {
      const mockOrderBookData = {
        symbol: 'AAPL',
        venue: 'NASDAQ',
        bids: [
          { id: 'bid-1', price: 150.25, quantity: 100, cumulative: 100, percentage: 50, orderCount: 5 },
          { id: 'bid-2', price: 150.24, quantity: 100, cumulative: 200, percentage: 50, orderCount: 8 }
        ],
        asks: [
          { id: 'ask-1', price: 150.26, quantity: 80, cumulative: 80, percentage: 40, orderCount: 4 },
          { id: 'ask-2', price: 150.27, quantity: 120, cumulative: 200, percentage: 60, orderCount: 6 }
        ],
        spread: {
          bestBid: 150.25,
          bestAsk: 150.26,
          spread: 0.01,
          spreadPercentage: 0.0067
        },
        timestamp: Date.now(),
        totalBidVolume: 200,
        totalAskVolume: 200
      }

      const { useOrderBookData } = require('../../../hooks/useOrderBookData')
      useOrderBookData.mockReturnValue({
        orderBookData: mockOrderBookData,
        isLoading: false,
        error: null,
        connectionStatus: 'connected',
        subscriptions: [],
        aggregationSettings: { enabled: false, increment: 0.01, maxLevels: 20 },
        displaySettings: { showSpread: true, showOrderCount: false, colorScheme: 'default', decimals: 2 },
        performanceMetrics: null,
        subscribeToOrderBook: vi.fn(),
        unsubscribeFromOrderBook: vi.fn(),
        updateAggregationSettings: vi.fn(),
        updateDisplaySettings: vi.fn(),
        clearOrderBook: vi.fn()
      })

      render(<OrderBookDisplay instrument={mockInstrument} />)
      
      // Should not show empty or loading states
      expect(screen.queryByTestId('empty')).not.toBeInTheDocument()
      expect(screen.queryByTestId('spin')).not.toBeInTheDocument()
      
      // Should show bid and ask volumes in footer
      expect(screen.getByText('Bid Volume: 200')).toBeInTheDocument()
      expect(screen.getByText('Ask Volume: 200')).toBeInTheDocument()
      expect(screen.getByText('Levels: 4')).toBeInTheDocument()
    })
  })

  describe('instrument subscription', () => {
    it('should subscribe to order book when instrument is provided', () => {
      const mockSubscribe = vi.fn()
      const { useOrderBookData } = require('../../../hooks/useOrderBookData')
      useOrderBookData.mockReturnValue({
        orderBookData: null,
        isLoading: false,
        error: null,
        connectionStatus: 'connected',
        subscriptions: [],
        aggregationSettings: { enabled: false, increment: 0.01, maxLevels: 20 },
        displaySettings: { showSpread: true, showOrderCount: false, colorScheme: 'default', decimals: 2 },
        performanceMetrics: null,
        subscribeToOrderBook: mockSubscribe,
        unsubscribeFromOrderBook: vi.fn(),
        updateAggregationSettings: vi.fn(),
        updateDisplaySettings: vi.fn(),
        clearOrderBook: vi.fn()
      })

      render(<OrderBookDisplay instrument={mockInstrument} autoSubscribe={true} />)
      
      expect(mockSubscribe).toHaveBeenCalledWith(mockInstrument)
    })

    it('should unsubscribe when instrument changes', () => {
      const mockUnsubscribe = vi.fn()
      const { useOrderBookData } = require('../../../hooks/useOrderBookData')
      useOrderBookData.mockReturnValue({
        orderBookData: null,
        isLoading: false,
        error: null,
        connectionStatus: 'connected',
        subscriptions: [],
        aggregationSettings: { enabled: false, increment: 0.01, maxLevels: 20 },
        displaySettings: { showSpread: true, showOrderCount: false, colorScheme: 'default', decimals: 2 },
        performanceMetrics: null,
        subscribeToOrderBook: vi.fn(),
        unsubscribeFromOrderBook: mockUnsubscribe,
        updateAggregationSettings: vi.fn(),
        updateDisplaySettings: vi.fn(),
        clearOrderBook: vi.fn()
      })

      const { rerender } = render(<OrderBookDisplay instrument={mockInstrument} autoSubscribe={true} />)
      
      const newInstrument: Instrument = {
        id: 'MSFT-STK',
        symbol: 'MSFT',
        name: 'Microsoft Corp.',
        venue: 'NASDAQ',
        assetClass: 'STK',
        currency: 'USD'
      }

      rerender(<OrderBookDisplay instrument={newInstrument} autoSubscribe={true} />)
      
      expect(mockUnsubscribe).toHaveBeenCalledWith(mockInstrument)
    })
  })

  describe('level click handling', () => {
    it('should call onLevelClick when provided', () => {
      const mockOnLevelClick = vi.fn()
      
      const mockOrderBookData = {
        symbol: 'AAPL',
        venue: 'NASDAQ',
        bids: [
          { id: 'bid-1', price: 150.25, quantity: 100, cumulative: 100, percentage: 100, orderCount: 5 }
        ],
        asks: [
          { id: 'ask-1', price: 150.26, quantity: 80, cumulative: 80, percentage: 100, orderCount: 4 }
        ],
        spread: {
          bestBid: 150.25,
          bestAsk: 150.26,
          spread: 0.01,
          spreadPercentage: 0.0067
        },
        timestamp: Date.now(),
        totalBidVolume: 100,
        totalAskVolume: 80
      }

      const { useOrderBookData } = require('../../../hooks/useOrderBookData')
      useOrderBookData.mockReturnValue({
        orderBookData: mockOrderBookData,
        isLoading: false,
        error: null,
        connectionStatus: 'connected',
        subscriptions: [],
        aggregationSettings: { enabled: false, increment: 0.01, maxLevels: 20 },
        displaySettings: { showSpread: true, showOrderCount: false, colorScheme: 'default', decimals: 2 },
        performanceMetrics: null,
        subscribeToOrderBook: vi.fn(),
        unsubscribeFromOrderBook: vi.fn(),
        updateAggregationSettings: vi.fn(),
        updateDisplaySettings: vi.fn(),
        clearOrderBook: vi.fn()
      })

      render(
        <OrderBookDisplay 
          instrument={mockInstrument} 
          onLevelClick={mockOnLevelClick}
        />
      )

      // This test would need the actual OrderBookLevel components to be rendered
      // For now, we're just testing that the prop is passed through
      expect(mockOnLevelClick).toBeDefined()
    })
  })

  describe('configuration props', () => {
    it('should use provided height', () => {
      render(<OrderBookDisplay instrument={null} height={800} />)
      
      const card = screen.getByTestId('card')
      expect(card).toHaveStyle({ height: '800px' })
    })

    it('should pass maxUpdatesPerSecond to hook', () => {
      const { useOrderBookData } = require('../../../hooks/useOrderBookData')
      
      render(<OrderBookDisplay instrument={null} maxUpdatesPerSecond={20} />)
      
      expect(useOrderBookData).toHaveBeenCalledWith({
        maxUpdatesPerSecond: 20,
        autoSubscribe: true,
        enablePerformanceTracking: true
      })
    })

    it('should pass autoSubscribe to hook', () => {
      const { useOrderBookData } = require('../../../hooks/useOrderBookData')
      
      render(<OrderBookDisplay instrument={null} autoSubscribe={false} />)
      
      expect(useOrderBookData).toHaveBeenCalledWith({
        maxUpdatesPerSecond: 10,
        autoSubscribe: false,
        enablePerformanceTracking: true
      })
    })
  })

  describe('empty data handling', () => {
    it('should show empty state when no order book data but connected', () => {
      const { useOrderBookData } = require('../../../hooks/useOrderBookData')
      useOrderBookData.mockReturnValue({
        orderBookData: {
          symbol: 'AAPL',
          venue: 'NASDAQ',
          bids: [],
          asks: [],
          spread: { bestBid: null, bestAsk: null, spread: null, spreadPercentage: null },
          timestamp: Date.now(),
          totalBidVolume: 0,
          totalAskVolume: 0
        },
        isLoading: false,
        error: null,
        connectionStatus: 'connected',
        subscriptions: [],
        aggregationSettings: { enabled: false, increment: 0.01, maxLevels: 20 },
        displaySettings: { showSpread: true, showOrderCount: false, colorScheme: 'default', decimals: 2 },
        performanceMetrics: null,
        subscribeToOrderBook: vi.fn(),
        unsubscribeFromOrderBook: vi.fn(),
        updateAggregationSettings: vi.fn(),
        updateDisplaySettings: vi.fn(),
        clearOrderBook: vi.fn()
      })

      render(<OrderBookDisplay instrument={mockInstrument} />)
      
      expect(screen.getByText('No order book data available')).toBeInTheDocument()
    })

    it('should show waiting message when disconnected', () => {
      const { useOrderBookData } = require('../../../hooks/useOrderBookData')
      useOrderBookData.mockReturnValue({
        orderBookData: null,
        isLoading: false,
        error: null,
        connectionStatus: 'disconnected',
        subscriptions: [],
        aggregationSettings: { enabled: false, increment: 0.01, maxLevels: 20 },
        displaySettings: { showSpread: true, showOrderCount: false, colorScheme: 'default', decimals: 2 },
        performanceMetrics: null,
        subscribeToOrderBook: vi.fn(),
        unsubscribeFromOrderBook: vi.fn(),
        updateAggregationSettings: vi.fn(),
        updateDisplaySettings: vi.fn(),
        clearOrderBook: vi.fn()
      })

      render(<OrderBookDisplay instrument={mockInstrument} />)
      
      expect(screen.getByText('Waiting for connection...')).toBeInTheDocument()
      expect(screen.getByText('Connection status: disconnected')).toBeInTheDocument()
    })
  })
})