import React from 'react'
import { render, screen, waitFor, act } from '@testing-library/react'
import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest'
import { OrderBookDisplay } from '../OrderBookDisplay'
import { Instrument } from '../../Chart/types/chartTypes'

// Mock the WebSocket service
const mockWebSocketService = {
  connect: vi.fn(),
  disconnect: vi.fn(),
  send: vi.fn(),
  addMessageHandler: vi.fn(),
  removeMessageHandler: vi.fn(),
  addStatusHandler: vi.fn(),
  removeStatusHandler: vi.fn(),
  getConnectionState: vi.fn(() => 'connected'),
  getPerformanceMetrics: vi.fn(() => ({
    messageLatency: [1, 2, 3],
    averageLatency: 2,
    maxLatency: 3,
    messagesProcessed: 10,
    messagesPerSecond: 5,
    lastUpdateTime: Date.now()
  }))
}

vi.mock('../../../services/websocket', () => ({
  webSocketService: mockWebSocketService,
  WebSocketService: vi.fn().mockImplementation(() => mockWebSocketService)
}))

// Mock orderBookService
const mockOrderBookService = {
  processOrderBookData: vi.fn(),
  validateOrderBookData: vi.fn(() => true),
  normalizeIBOrderBookData: vi.fn(),
  throttleUpdates: vi.fn((callback) => callback),
  getPerformanceMetrics: vi.fn(() => ({
    updateLatency: [1, 2, 3],
    averageLatency: 2,
    maxLatency: 3,
    updatesPerSecond: 10,
    lastUpdateTime: Date.now()
  })),
  resetPerformanceMetrics: vi.fn(),
  createOrderBookMessage: vi.fn(),
  getOptimalAggregationIncrement: vi.fn(() => 0.01)
}

vi.mock('../../../services/orderBookService', () => ({
  orderBookService: mockOrderBookService
}))

// Don't mock the hook directly, let it use the mocked services
vi.unmock('../../../hooks/useOrderBookData')

describe('OrderBook Integration Tests', () => {
  const mockInstrument: Instrument = {
    id: 'AAPL-STK',
    symbol: 'AAPL',
    name: 'Apple Inc.',
    venue: 'NASDAQ',
    assetClass: 'STK',
    currency: 'USD'
  }

  let messageHandler: ((message: any) => void) | null = null
  let statusHandler: ((status: any) => void) | null = null

  beforeEach(() => {
    vi.clearAllMocks()
    
    // Capture the message and status handlers
    mockWebSocketService.addMessageHandler.mockImplementation((handler) => {
      messageHandler = handler
    })
    
    mockWebSocketService.addStatusHandler.mockImplementation((handler) => {
      statusHandler = handler
    })

    // Mock successful data processing
    mockOrderBookService.processOrderBookData.mockReturnValue({
      symbol: 'AAPL',
      venue: 'NASDAQ',
      bids: [
        { id: 'bid-1', price: 150.25, quantity: 100, cumulative: 100, percentage: 50, orderCount: 5 }
      ],
      asks: [
        { id: 'ask-1', price: 150.26, quantity: 80, cumulative: 80, percentage: 50, orderCount: 4 }
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
    })
  })

  afterEach(() => {
    messageHandler = null
    statusHandler = null
  })

  describe('WebSocket Integration', () => {
    it('should establish WebSocket connection and subscribe to order book data', async () => {
      render(<OrderBookDisplay instrument={mockInstrument} autoSubscribe={true} />)

      // Should connect to WebSocket
      expect(mockWebSocketService.addMessageHandler).toHaveBeenCalled()
      expect(mockWebSocketService.addStatusHandler).toHaveBeenCalled()

      // Should subscribe to order book for the instrument
      await waitFor(() => {
        expect(mockWebSocketService.send).toHaveBeenCalledWith({
          type: 'subscribe_order_book',
          symbol: 'AAPL',
          venue: 'NASDAQ',
          asset_class: 'STK',
          currency: 'USD'
        })
      })
    })

    it('should handle incoming order book messages', async () => {
      render(<OrderBookDisplay instrument={mockInstrument} autoSubscribe={true} />)

      // Wait for initial setup
      await waitFor(() => {
        expect(messageHandler).not.toBeNull()
      })

      // Simulate incoming order book message
      const orderBookMessage = {
        type: 'order_book_update',
        symbol: 'AAPL',
        venue: 'NASDAQ',
        bids: [
          { price: 150.25, quantity: 100, orderCount: 5 }
        ],
        asks: [
          { price: 150.26, quantity: 80, orderCount: 4 }
        ],
        timestamp: Date.now()
      }

      act(() => {
        messageHandler!(orderBookMessage)
      })

      // Should validate and process the data
      await waitFor(() => {
        expect(mockOrderBookService.validateOrderBookData).toHaveBeenCalled()
        expect(mockOrderBookService.processOrderBookData).toHaveBeenCalled()
      })

      // Should display the processed data
      expect(screen.getByText('Bid Volume: 100')).toBeInTheDocument()
      expect(screen.getByText('Ask Volume: 80')).toBeInTheDocument()
    })

    it('should handle connection status changes', async () => {
      render(<OrderBookDisplay instrument={mockInstrument} autoSubscribe={true} />)

      await waitFor(() => {
        expect(statusHandler).not.toBeNull()
      })

      // Simulate connection loss
      act(() => {
        statusHandler!('disconnected')
      })

      await waitFor(() => {
        expect(screen.getByText('Connection status: disconnected')).toBeInTheDocument()
      })

      // Simulate reconnection
      act(() => {
        statusHandler!('connected')
      })

      await waitFor(() => {
        expect(screen.getByText('Live Data')).toBeInTheDocument()
      })
    })

    it('should unsubscribe when component unmounts', async () => {
      const { unmount } = render(<OrderBookDisplay instrument={mockInstrument} autoSubscribe={true} />)

      await waitFor(() => {
        expect(mockWebSocketService.send).toHaveBeenCalledWith(
          expect.objectContaining({
            type: 'subscribe_order_book'
          })
        )
      })

      unmount()

      // Should send unsubscribe message
      expect(mockWebSocketService.send).toHaveBeenCalledWith({
        type: 'unsubscribe_order_book',
        symbol: 'AAPL',
        venue: 'NASDAQ'
      })
    })

    it('should handle subscription changes when instrument changes', async () => {
      const { rerender } = render(<OrderBookDisplay instrument={mockInstrument} autoSubscribe={true} />)

      await waitFor(() => {
        expect(mockWebSocketService.send).toHaveBeenCalledWith(
          expect.objectContaining({
            type: 'subscribe_order_book',
            symbol: 'AAPL'
          })
        )
      })

      const newInstrument: Instrument = {
        id: 'MSFT-STK',
        symbol: 'MSFT',
        name: 'Microsoft Corp.',
        venue: 'NASDAQ',
        assetClass: 'STK',
        currency: 'USD'
      }

      rerender(<OrderBookDisplay instrument={newInstrument} autoSubscribe={true} />)

      await waitFor(() => {
        // Should unsubscribe from old instrument
        expect(mockWebSocketService.send).toHaveBeenCalledWith({
          type: 'unsubscribe_order_book',
          symbol: 'AAPL',
          venue: 'NASDAQ'
        })

        // Should subscribe to new instrument
        expect(mockWebSocketService.send).toHaveBeenCalledWith(
          expect.objectContaining({
            type: 'subscribe_order_book',
            symbol: 'MSFT'
          })
        )
      })
    })
  })

  describe('Data Processing Integration', () => {
    it('should validate incoming data before processing', async () => {
      render(<OrderBookDisplay instrument={mockInstrument} autoSubscribe={true} />)

      await waitFor(() => {
        expect(messageHandler).not.toBeNull()
      })

      const invalidMessage = {
        type: 'order_book_update',
        symbol: 'AAPL',
        venue: 'NASDAQ',
        bids: 'invalid',
        asks: 'invalid',
        timestamp: Date.now()
      }

      // Mock validation failure
      mockOrderBookService.validateOrderBookData.mockReturnValueOnce(false)

      act(() => {
        messageHandler!(invalidMessage)
      })

      await waitFor(() => {
        expect(mockOrderBookService.validateOrderBookData).toHaveBeenCalled()
        // Should not process invalid data
        expect(mockOrderBookService.processOrderBookData).not.toHaveBeenCalled()
      })
    })

    it('should apply aggregation settings to processed data', async () => {
      render(<OrderBookDisplay instrument={mockInstrument} autoSubscribe={true} />)

      await waitFor(() => {
        expect(messageHandler).not.toBeNull()
      })

      const orderBookMessage = {
        type: 'order_book_update',
        symbol: 'AAPL',
        venue: 'NASDAQ',
        bids: [{ price: 150.25, quantity: 100 }],
        asks: [{ price: 150.26, quantity: 80 }],
        timestamp: Date.now()
      }

      act(() => {
        messageHandler!(orderBookMessage)
      })

      await waitFor(() => {
        expect(mockOrderBookService.processOrderBookData).toHaveBeenCalledWith(
          expect.any(Object),
          expect.objectContaining({
            enabled: false,
            increment: 0.01,
            maxLevels: 20
          })
        )
      })
    })

    it('should throttle high-frequency updates', async () => {
      render(<OrderBookDisplay instrument={mockInstrument} maxUpdatesPerSecond={5} />)

      await waitFor(() => {
        expect(messageHandler).not.toBeNull()
      })

      // Send multiple rapid updates
      const orderBookMessage = {
        type: 'order_book_update',
        symbol: 'AAPL',
        venue: 'NASDAQ',
        bids: [{ price: 150.25, quantity: 100 }],
        asks: [{ price: 150.26, quantity: 80 }],
        timestamp: Date.now()
      }

      act(() => {
        messageHandler!(orderBookMessage)
        messageHandler!(orderBookMessage)
        messageHandler!(orderBookMessage)
      })

      // Should use throttled updates
      expect(mockOrderBookService.throttleUpdates).toHaveBeenCalled()
    })
  })

  describe('Performance Monitoring Integration', () => {
    it('should track performance metrics', async () => {
      render(<OrderBookDisplay instrument={mockInstrument} autoSubscribe={true} />)

      await waitFor(() => {
        expect(messageHandler).not.toBeNull()
      })

      const orderBookMessage = {
        type: 'order_book_update',
        symbol: 'AAPL',
        venue: 'NASDAQ',
        bids: [{ price: 150.25, quantity: 100 }],
        asks: [{ price: 150.26, quantity: 80 }],
        timestamp: Date.now()
      }

      act(() => {
        messageHandler!(orderBookMessage)
      })

      await waitFor(() => {
        expect(mockOrderBookService.getPerformanceMetrics).toHaveBeenCalled()
      })
    })
  })

  describe('Error Handling Integration', () => {
    it('should handle processing errors gracefully', async () => {
      render(<OrderBookDisplay instrument={mockInstrument} autoSubscribe={true} />)

      await waitFor(() => {
        expect(messageHandler).not.toBeNull()
      })

      // Mock processing error
      mockOrderBookService.processOrderBookData.mockImplementationOnce(() => {
        throw new Error('Processing failed')
      })

      const orderBookMessage = {
        type: 'order_book_update',
        symbol: 'AAPL',
        venue: 'NASDAQ',
        bids: [{ price: 150.25, quantity: 100 }],
        asks: [{ price: 150.26, quantity: 80 }],
        timestamp: Date.now()
      }

      // Should not crash when processing fails
      expect(() => {
        act(() => {
          messageHandler!(orderBookMessage)
        })
      }).not.toThrow()
    })

    it('should display error messages when service fails', async () => {
      render(<OrderBookDisplay instrument={mockInstrument} autoSubscribe={true} />)

      await waitFor(() => {
        expect(messageHandler).not.toBeNull()
      })

      // Mock service error
      mockOrderBookService.processOrderBookData.mockImplementationOnce(() => {
        throw new Error('Service unavailable')
      })

      const orderBookMessage = {
        type: 'order_book_update',
        symbol: 'AAPL',
        venue: 'NASDAQ',
        bids: [{ price: 150.25, quantity: 100 }],
        asks: [{ price: 150.26, quantity: 80 }],
        timestamp: Date.now()
      }

      act(() => {
        messageHandler!(orderBookMessage)
      })

      // Error should be displayed
      await waitFor(() => {
        expect(screen.getByText('Order Book Error')).toBeInTheDocument()
      })
    })
  })

  describe('Real-time Updates Integration', () => {
    it('should handle rapid successive updates', async () => {
      render(<OrderBookDisplay instrument={mockInstrument} autoSubscribe={true} />)

      await waitFor(() => {
        expect(messageHandler).not.toBeNull()
      })

      // Send multiple updates with different timestamps
      const updates = [
        {
          type: 'order_book_update',
          symbol: 'AAPL',
          venue: 'NASDAQ',
          bids: [{ price: 150.25, quantity: 100 }],
          asks: [{ price: 150.26, quantity: 80 }],
          timestamp: Date.now()
        },
        {
          type: 'order_book_update',
          symbol: 'AAPL',
          venue: 'NASDAQ',
          bids: [{ price: 150.24, quantity: 150 }],
          asks: [{ price: 150.27, quantity: 90 }],
          timestamp: Date.now() + 100
        }
      ]

      act(() => {
        updates.forEach(update => messageHandler!(update))
      })

      await waitFor(() => {
        expect(mockOrderBookService.processOrderBookData).toHaveBeenCalledTimes(2)
      })
    })

    it('should ignore messages for non-subscribed instruments', async () => {
      render(<OrderBookDisplay instrument={mockInstrument} autoSubscribe={true} />)

      await waitFor(() => {
        expect(messageHandler).not.toBeNull()
      })

      const wrongInstrumentMessage = {
        type: 'order_book_update',
        symbol: 'MSFT', // Different symbol
        venue: 'NASDAQ',
        bids: [{ price: 300.25, quantity: 100 }],
        asks: [{ price: 300.26, quantity: 80 }],
        timestamp: Date.now()
      }

      act(() => {
        messageHandler!(wrongInstrumentMessage)
      })

      // Should not process data for non-subscribed instrument
      expect(mockOrderBookService.processOrderBookData).not.toHaveBeenCalled()
    })
  })
})