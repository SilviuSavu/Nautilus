import { renderHook, act } from '@testing-library/react';
import { vi, describe, it, expect, beforeEach, afterEach } from 'vitest';
import { useOrderManagement, IBOrder } from '../useOrderManagement';

// Mock notification  
vi.mock('antd', async () => {
  const actual = await vi.importActual('antd');
  return {
    ...actual,
    notification: {
      success: vi.fn(),
      error: vi.fn(),
    },
  };
});

// Mock fetch globally
const mockFetch = vi.fn();
global.fetch = mockFetch;

describe('useOrderManagement', () => {
  beforeEach(() => {
    mockFetch.mockClear();
    vi.clearAllMocks();
  });

  afterEach(() => {
    vi.clearAllMocks();
  });

  describe('modifyOrder', () => {
    it('should successfully modify an order', async () => {
      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: () => Promise.resolve({ success: true })
      });

      const { result } = renderHook(() => useOrderManagement());

      let modifyResult;
      await act(async () => {
        modifyResult = await result.current.modifyOrder('ORDER_123', {
          quantity: 150,
          limit_price: 155.00
        });
      });

      expect(mockFetch).toHaveBeenCalledWith(
        '/api/v1/ib/orders/ORDER_123/modify',
        expect.objectContaining({
          method: 'PUT',
          headers: { 'Content-Type': 'application/json' },
          credentials: 'include',
          body: JSON.stringify({
            quantity: 150,
            limit_price: 155.00
          })
        })
      );

      expect(modifyResult).toEqual({ success: true });
    });

    it('should handle modify order failure', async () => {
      mockFetch.mockResolvedValueOnce({
        ok: false,
        json: () => Promise.resolve({ detail: 'Order not found' })
      });

      const { result } = renderHook(() => useOrderManagement());

      let modifyResult;
      await act(async () => {
        modifyResult = await result.current.modifyOrder('ORDER_123', {
          quantity: 150
        });
      });

      expect(modifyResult).toEqual({ 
        success: false, 
        error: { detail: 'Order not found' }
      });
    });

    it('should handle network error during modify', async () => {
      const networkError = new Error('Network error');
      mockFetch.mockRejectedValueOnce(networkError);

      const { result } = renderHook(() => useOrderManagement());

      let modifyResult;
      await act(async () => {
        modifyResult = await result.current.modifyOrder('ORDER_123', {
          quantity: 150
        });
      });

      expect(modifyResult).toEqual({ 
        success: false, 
        error: networkError
      });
    });
  });

  describe('cancelOrder', () => {
    it('should successfully cancel an order', async () => {
      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: () => Promise.resolve({ success: true })
      });

      const { result } = renderHook(() => useOrderManagement());

      let cancelResult;
      await act(async () => {
        cancelResult = await result.current.cancelOrder('ORDER_123');
      });

      expect(mockFetch).toHaveBeenCalledWith(
        '/api/v1/ib/orders/ORDER_123/cancel',
        expect.objectContaining({
          method: 'DELETE',
          credentials: 'include'
        })
      );

      expect(cancelResult).toEqual({ success: true });
    });

    it('should handle cancel order failure', async () => {
      mockFetch.mockResolvedValueOnce({
        ok: false,
        json: () => Promise.resolve({ detail: 'Order cannot be cancelled' })
      });

      const { result } = renderHook(() => useOrderManagement());

      let cancelResult;
      await act(async () => {
        cancelResult = await result.current.cancelOrder('ORDER_123');
      });

      expect(cancelResult).toEqual({ 
        success: false, 
        error: { detail: 'Order cannot be cancelled' }
      });
    });
  });

  describe('utility functions', () => {
    it('should correctly identify actionable orders', () => {
      const { result } = renderHook(() => useOrderManagement());

      const submittedOrder: IBOrder = {
        order_id: 'ORDER_1',
        client_id: 1,
        account_id: 'ACC1',
        contract_id: '123',
        symbol: 'AAPL',
        action: 'BUY',
        order_type: 'LMT',
        total_quantity: 100,
        filled_quantity: 0,
        remaining_quantity: 100,
        status: 'Submitted'
      };

      const filledOrder: IBOrder = {
        ...submittedOrder,
        order_id: 'ORDER_2',
        status: 'Filled',
        filled_quantity: 100,
        remaining_quantity: 0
      };

      const cancelledOrder: IBOrder = {
        ...submittedOrder,
        order_id: 'ORDER_3',
        status: 'Cancelled'
      };

      expect(result.current.isOrderActionable(submittedOrder)).toBe(true);
      expect(result.current.isOrderActionable(filledOrder)).toBe(false);
      expect(result.current.isOrderActionable(cancelledOrder)).toBe(false);
    });

    it('should return correct status badge props', () => {
      const { result } = renderHook(() => useOrderManagement());

      expect(result.current.getStatusBadgeProps('Submitted')).toEqual({
        status: 'processing',
        text: 'Submitted'
      });

      expect(result.current.getStatusBadgeProps('Filled')).toEqual({
        status: 'success',
        text: 'Filled'
      });

      expect(result.current.getStatusBadgeProps('Cancelled')).toEqual({
        status: 'error',
        text: 'Cancelled'
      });

      expect(result.current.getStatusBadgeProps('Unknown')).toEqual({
        status: 'default',
        text: 'Unknown'
      });
    });

    it('should return correct action badge props', () => {
      const { result } = renderHook(() => useOrderManagement());

      expect(result.current.getActionBadgeProps('BUY')).toEqual({
        color: 'green',
        text: 'BUY'
      });

      expect(result.current.getActionBadgeProps('SELL')).toEqual({
        color: 'red',
        text: 'SELL'
      });
    });

    it('should format order ID correctly', () => {
      const { result } = renderHook(() => useOrderManagement());

      expect(result.current.formatOrderId('ORDER123')).toBe('ORDER123');
      expect(result.current.formatOrderId('ORDER123456789')).toBe('ORDER123...');
    });

    it('should format price correctly', () => {
      const { result } = renderHook(() => useOrderManagement());

      expect(result.current.formatPrice(150.50, 'LMT')).toBe('$150.50');
      expect(result.current.formatPrice(null, 'LMT')).toBe('-');
      expect(result.current.formatPrice(undefined, 'LMT')).toBe('-');
      expect(result.current.formatPrice(150.50, 'MKT')).toBe('Market');
    });
  });

  describe('loading states', () => {
    it('should track modifying state', async () => {
      mockFetch.mockImplementationOnce(() => 
        new Promise(resolve => setTimeout(() => resolve({
          ok: true,
          json: () => Promise.resolve({ success: true })
        }), 100))
      );

      const { result } = renderHook(() => useOrderManagement());

      expect(result.current.isModifying).toBe(false);

      act(() => {
        result.current.modifyOrder('ORDER_123', { quantity: 150 });
      });

      expect(result.current.isModifying).toBe(true);

      await act(async () => {
        await new Promise(resolve => setTimeout(resolve, 150));
      });

      expect(result.current.isModifying).toBe(false);
    });

    it('should track cancelling state per order', async () => {
      mockFetch.mockImplementationOnce(() => 
        new Promise(resolve => setTimeout(() => resolve({
          ok: true,
          json: () => Promise.resolve({ success: true })
        }), 100))
      );

      const { result } = renderHook(() => useOrderManagement());

      expect(result.current.isCancelling).toEqual({});

      act(() => {
        result.current.cancelOrder('ORDER_123');
      });

      expect(result.current.isCancelling).toEqual({ ORDER_123: true });

      await act(async () => {
        await new Promise(resolve => setTimeout(resolve, 150));
      });

      expect(result.current.isCancelling).toEqual({ ORDER_123: false });
    });
  });
});