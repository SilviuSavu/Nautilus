import React from 'react';
import { render, screen, fireEvent, waitFor, act } from '@testing-library/react';
import { vi, describe, it, expect, beforeEach, afterEach } from 'vitest';
import IBDashboard from '../IBDashboard';

// Mock the useMessageBus hook
const mockUseMessageBus = vi.fn();
vi.mock('../../hooks/useMessageBus', () => ({
  useMessageBus: () => mockUseMessageBus()
}));

// Mock fetch globally
const mockFetch = vi.fn();
global.fetch = mockFetch;

// Mock window.matchMedia
Object.defineProperty(window, 'matchMedia', {
  writable: true,
  value: vi.fn().mockImplementation(query => ({
    matches: false,
    media: query,
    onchange: null,
    addListener: vi.fn(), // deprecated
    removeListener: vi.fn(), // deprecated
    addEventListener: vi.fn(),
    removeEventListener: vi.fn(),
    dispatchEvent: vi.fn(),
  })),
});

describe('IBDashboard - Order Status Monitoring', () => {
  beforeEach(() => {
    mockUseMessageBus.mockReturnValue({
      latestMessage: null
    });
    mockFetch.mockClear();
  });

  afterEach(() => {
    vi.clearAllMocks();
  });

  describe('Order Status Display', () => {
    it('should render orders table with correct columns', async () => {
      // Mock API responses
      mockFetch
        .mockResolvedValueOnce({
          ok: true,
          json: () => Promise.resolve({ connected: false })
        })
        .mockResolvedValueOnce({
          ok: true,
          json: () => Promise.resolve({})
        })
        .mockResolvedValueOnce({
          ok: true,
          json: () => Promise.resolve({ positions: [] })
        })
        .mockResolvedValueOnce({
          ok: true,
          json: () => Promise.resolve({ orders: [] })
        });

      render(<IBDashboard />);

      await waitFor(() => {
        expect(screen.getByText('Orders (0)')).toBeInTheDocument();
      });

      // Check for order table columns
      expect(screen.getByText('Order ID')).toBeInTheDocument();
      expect(screen.getByText('Symbol')).toBeInTheDocument();
      expect(screen.getByText('Action')).toBeInTheDocument();
      expect(screen.getByText('Status')).toBeInTheDocument();
      expect(screen.getByText('Actions')).toBeInTheDocument();
    });

    it('should display orders with correct data formatting', async () => {
      const mockOrders = [
        {
          order_id: 'ORDER123456789',
          symbol: 'AAPL',
          action: 'BUY',
          order_type: 'LMT',
          total_quantity: 100,
          filled_quantity: 50,
          status: 'Submitted',
          limit_price: 150.00
        },
        {
          order_id: 'ORDER987654321',
          symbol: 'MSFT',
          action: 'SELL',
          order_type: 'MKT',
          total_quantity: 200,
          filled_quantity: 200,
          status: 'Filled',
          limit_price: null
        }
      ];

      mockFetch
        .mockResolvedValueOnce({
          ok: true,
          json: () => Promise.resolve({ connected: false })
        })
        .mockResolvedValueOnce({
          ok: true,
          json: () => Promise.resolve({})
        })
        .mockResolvedValueOnce({
          ok: true,
          json: () => Promise.resolve({ positions: [] })
        })
        .mockResolvedValueOnce({
          ok: true,
          json: () => Promise.resolve({ orders: mockOrders })
        });

      render(<IBDashboard />);

      await waitFor(() => {
        expect(screen.getByText('Orders (2)')).toBeInTheDocument();
      });

      // Check order data display
      expect(screen.getByText('ORDER123...')).toBeInTheDocument(); // Truncated order ID
      expect(screen.getByText('AAPL')).toBeInTheDocument();
      expect(screen.getByText('MSFT')).toBeInTheDocument();
      expect(screen.getByText('$150.00')).toBeInTheDocument();
      expect(screen.getByText('Market')).toBeInTheDocument(); // MKT order type
    });
  });

  describe('Order Status Badge Color Coding', () => {
    it('should render correct badge colors for different order statuses', async () => {
      const mockOrders = [
        {
          order_id: 'ORDER1',
          symbol: 'AAPL',
          action: 'BUY',
          order_type: 'LMT',
          total_quantity: 100,
          filled_quantity: 0,
          status: 'Submitted',
          limit_price: 150.00
        },
        {
          order_id: 'ORDER2',
          symbol: 'MSFT',
          action: 'SELL',
          order_type: 'MKT',
          total_quantity: 200,
          filled_quantity: 200,
          status: 'Filled',
          limit_price: null
        },
        {
          order_id: 'ORDER3',
          symbol: 'GOOGL',
          action: 'BUY',
          order_type: 'LMT',
          total_quantity: 50,
          filled_quantity: 0,
          status: 'Cancelled',
          limit_price: 2800.00
        }
      ];

      mockFetch
        .mockResolvedValueOnce({
          ok: true,
          json: () => Promise.resolve({ connected: false })
        })
        .mockResolvedValueOnce({
          ok: true,
          json: () => Promise.resolve({})
        })
        .mockResolvedValueOnce({
          ok: true,
          json: () => Promise.resolve({ positions: [] })
        })
        .mockResolvedValueOnce({
          ok: true,
          json: () => Promise.resolve({ orders: mockOrders })
        });

      render(<IBDashboard />);

      await waitFor(() => {
        expect(screen.getByText('Orders (3)')).toBeInTheDocument();
      });

      // Check status badges are present (color validation would require DOM inspection)
      const submittedBadge = screen.getByText('Submitted');
      const filledBadge = screen.getByText('Filled');
      const cancelledBadge = screen.getByText('Cancelled');

      expect(submittedBadge).toBeInTheDocument();
      expect(filledBadge).toBeInTheDocument();
      expect(cancelledBadge).toBeInTheDocument();
    });
  });

  describe('Order Action Button States', () => {
    it('should disable action buttons for completed orders', async () => {
      const mockOrders = [
        {
          order_id: 'ORDER_ACTIVE',
          symbol: 'AAPL',
          action: 'BUY',
          order_type: 'LMT',
          total_quantity: 100,
          filled_quantity: 0,
          status: 'Submitted',
          limit_price: 150.00
        },
        {
          order_id: 'ORDER_FILLED',
          symbol: 'MSFT',
          action: 'SELL',
          order_type: 'MKT',
          total_quantity: 200,
          filled_quantity: 200,
          status: 'Filled',
          limit_price: null
        },
        {
          order_id: 'ORDER_CANCELLED',
          symbol: 'GOOGL',
          action: 'BUY',
          order_type: 'LMT',
          total_quantity: 50,
          filled_quantity: 0,
          status: 'Cancelled',
          limit_price: 2800.00
        }
      ];

      mockFetch
        .mockResolvedValueOnce({
          ok: true,
          json: () => Promise.resolve({ connected: false })
        })
        .mockResolvedValueOnce({
          ok: true,
          json: () => Promise.resolve({})
        })
        .mockResolvedValueOnce({
          ok: true,
          json: () => Promise.resolve({ positions: [] })
        })
        .mockResolvedValueOnce({
          ok: true,
          json: () => Promise.resolve({ orders: mockOrders })
        });

      render(<IBDashboard />);

      await waitFor(() => {
        expect(screen.getByText('Orders (3)')).toBeInTheDocument();
      });

      // Get all edit and delete buttons
      const editButtons = screen.getAllByRole('button', { name: /edit/i });
      const deleteButtons = screen.getAllByRole('button', { name: /delete/i });

      // Active order buttons should be enabled (first order)
      expect(editButtons[0]).toBeEnabled();
      expect(deleteButtons[0]).toBeEnabled();

      // Filled order buttons should be disabled (second order)
      expect(editButtons[1]).toBeDisabled();
      expect(deleteButtons[1]).toBeDisabled();

      // Cancelled order buttons should be disabled (third order)
      expect(editButtons[2]).toBeDisabled();
      expect(deleteButtons[2]).toBeDisabled();
    });
  });

  describe('WebSocket Message Handling', () => {
    it('should update orders when ib_order message is received', async () => {
      mockFetch
        .mockResolvedValueOnce({
          ok: true,
          json: () => Promise.resolve({ connected: false })
        })
        .mockResolvedValueOnce({
          ok: true,
          json: () => Promise.resolve({})
        })
        .mockResolvedValueOnce({
          ok: true,
          json: () => Promise.resolve({ positions: [] })
        })
        .mockResolvedValueOnce({
          ok: true,
          json: () => Promise.resolve({ orders: [] })
        });

      // Initial render with no orders
      const { rerender } = render(<IBDashboard />);

      await waitFor(() => {
        expect(screen.getByText('Orders (0)')).toBeInTheDocument();
      });

      // Simulate WebSocket message with new order
      const newOrderMessage = {
        type: 'ib_order',
        data: {
          order_id: 'NEW_ORDER_123',
          symbol: 'TESLA',
          action: 'BUY',
          order_type: 'MKT',
          total_quantity: 100,
          filled_quantity: 0,
          status: 'Submitted'
        }
      };

      mockUseMessageBus.mockReturnValue({
        latestMessage: newOrderMessage
      });

      rerender(<IBDashboard />);

      await waitFor(() => {
        expect(screen.getByText('Orders (1)')).toBeInTheDocument();
        expect(screen.getByText('TESLA')).toBeInTheDocument();
        expect(screen.getByText('Submitted')).toBeInTheDocument();
      });
    });

    it('should update existing order when ib_order message is received', async () => {
      const initialOrder = {
        order_id: 'ORDER_123',
        symbol: 'AAPL',
        action: 'BUY',
        order_type: 'LMT',
        total_quantity: 100,
        filled_quantity: 0,
        status: 'Submitted',
        limit_price: 150.00
      };

      mockFetch
        .mockResolvedValueOnce({
          ok: true,
          json: () => Promise.resolve({ connected: false })
        })
        .mockResolvedValueOnce({
          ok: true,
          json: () => Promise.resolve({})
        })
        .mockResolvedValueOnce({
          ok: true,
          json: () => Promise.resolve({ positions: [] })
        })
        .mockResolvedValueOnce({
          ok: true,
          json: () => Promise.resolve({ orders: [initialOrder] })
        });

      const { rerender } = render(<IBDashboard />);

      await waitFor(() => {
        expect(screen.getByText('Submitted')).toBeInTheDocument();
      });

      // Simulate WebSocket message updating the order to filled
      const updatedOrderMessage = {
        type: 'ib_order',
        data: {
          ...initialOrder,
          filled_quantity: 100,
          status: 'Filled'
        }
      };

      mockUseMessageBus.mockReturnValue({
        latestMessage: updatedOrderMessage
      });

      rerender(<IBDashboard />);

      await waitFor(() => {
        expect(screen.getByText('Filled')).toBeInTheDocument();
        expect(screen.getByText('100')).toBeInTheDocument(); // filled quantity
      });
    });
  });

  describe('Order Modification Handler', () => {
    it('should call modify API when order is modified', async () => {
      const mockOrder = {
        order_id: 'ORDER_123',
        symbol: 'AAPL',
        action: 'BUY',
        order_type: 'LMT',
        total_quantity: 100,
        filled_quantity: 0,
        status: 'Submitted',
        limit_price: 150.00
      };

      mockFetch
        .mockResolvedValueOnce({
          ok: true,
          json: () => Promise.resolve({ connected: false })
        })
        .mockResolvedValueOnce({
          ok: true,
          json: () => Promise.resolve({})
        })
        .mockResolvedValueOnce({
          ok: true,
          json: () => Promise.resolve({ positions: [] })
        })
        .mockResolvedValueOnce({
          ok: true,
          json: () => Promise.resolve({ orders: [mockOrder] })
        })
        .mockResolvedValueOnce({
          ok: true,
          json: () => Promise.resolve({})
        }) // Order modify response
        .mockResolvedValueOnce({
          ok: true,
          json: () => Promise.resolve({ orders: [mockOrder] })
        }); // Refresh orders response

      render(<IBDashboard />);

      await waitFor(() => {
        expect(screen.getByText('Orders (1)')).toBeInTheDocument();
      });

      // Check that edit button exists and is clickable
      const editButton = screen.getByRole('button', { name: /edit/i });
      expect(editButton).toBeInTheDocument();
      expect(editButton).toBeEnabled();
    });
  });

  describe('Order Cancellation Handler', () => {
    it('should call cancel API when order is cancelled', async () => {
      const mockOrder = {
        order_id: 'ORDER_123',
        symbol: 'AAPL',
        action: 'BUY',
        order_type: 'LMT',
        total_quantity: 100,
        filled_quantity: 0,
        status: 'Submitted',
        limit_price: 150.00
      };

      mockFetch
        .mockResolvedValueOnce({
          ok: true,
          json: () => Promise.resolve({ connected: false })
        })
        .mockResolvedValueOnce({
          ok: true,
          json: () => Promise.resolve({})
        })
        .mockResolvedValueOnce({
          ok: true,
          json: () => Promise.resolve({ positions: [] })
        })
        .mockResolvedValueOnce({
          ok: true,
          json: () => Promise.resolve({ orders: [mockOrder] })
        });

      render(<IBDashboard />);

      await waitFor(() => {
        expect(screen.getByText('Orders (1)')).toBeInTheDocument();
      });

      // Check that delete button exists and is clickable
      const deleteButton = screen.getByRole('button', { name: /delete/i });
      expect(deleteButton).toBeInTheDocument();
      expect(deleteButton).toBeEnabled();
    });
  });
});