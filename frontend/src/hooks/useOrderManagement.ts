import { useState, useCallback } from 'react';
import { notification } from 'antd';

export interface IBOrder {
  order_id: string;
  client_id: number;
  account_id: string;
  contract_id: string;
  symbol: string;
  action: string;
  order_type: string;
  total_quantity: number;
  filled_quantity: number;
  remaining_quantity: number;
  limit_price?: number;
  stop_price?: number;
  status: string;
  avg_fill_price?: number;
  commission?: number;
  timestamp?: string;
}

export interface OrderModificationData {
  quantity?: number;
  limit_price?: number;
  stop_price?: number;
}

export const useOrderManagement = () => {
  const [isModifying, setIsModifying] = useState(false);
  const [isCancelling, setIsCancelling] = useState<{ [orderId: string]: boolean }>({});

  const modifyOrder = useCallback(async (orderId: string, values: OrderModificationData) => {
    setIsModifying(true);
    try {
      const response = await fetch(`/api/v1/ib/orders/${orderId}/modify`, {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        credentials: 'include',
        body: JSON.stringify(values)
      });
      
      if (response.ok) {
        notification.success({ message: 'Order modified successfully' });
        return { success: true };
      } else {
        const errorData = await response.json();
        notification.error({ 
          message: 'Failed to modify order',
          description: errorData.detail || 'Unknown error occurred'
        });
        return { success: false, error: errorData };
      }
    } catch (err) {
      console.error('Error modifying order:', err);
      notification.error({ message: 'Failed to modify order' });
      return { success: false, error: err };
    } finally {
      setIsModifying(false);
    }
  }, []);

  const cancelOrder = useCallback(async (orderId: string) => {
    setIsCancelling(prev => ({ ...prev, [orderId]: true }));
    try {
      const response = await fetch(`/api/v1/ib/orders/${orderId}/cancel`, {
        method: 'DELETE',
        credentials: 'include'
      });
      
      if (response.ok) {
        notification.success({ message: 'Order cancelled successfully' });
        return { success: true };
      } else {
        const errorData = await response.json();
        notification.error({ 
          message: 'Failed to cancel order',
          description: errorData.detail || 'Unknown error occurred'
        });
        return { success: false, error: errorData };
      }
    } catch (err) {
      console.error('Error cancelling order:', err);
      notification.error({ message: 'Failed to cancel order' });
      return { success: false, error: err };
    } finally {
      setIsCancelling(prev => ({ ...prev, [orderId]: false }));
    }
  }, []);

  const refreshOrders = useCallback(async () => {
    try {
      await fetch('/api/v1/ib/orders/refresh', {
        method: 'POST',
        credentials: 'include'
      });
      return { success: true };
    } catch (err) {
      console.error('Error refreshing orders:', err);
      notification.error({ message: 'Failed to refresh orders' });
      return { success: false, error: err };
    }
  }, []);

  const isOrderActionable = useCallback((order: IBOrder) => {
    return order.status !== 'Filled' && order.status !== 'Cancelled';
  }, []);

  const getStatusBadgeProps = useCallback((status: string) => {
    switch (status) {
      case 'Filled':
        return { status: 'success' as const, text: status };
      case 'Cancelled':
        return { status: 'error' as const, text: status };
      case 'Submitted':
        return { status: 'processing' as const, text: status };
      default:
        return { status: 'default' as const, text: status };
    }
  }, []);

  const getActionBadgeProps = useCallback((action: string) => {
    return {
      color: action === 'BUY' ? 'green' : 'red',
      text: action
    };
  }, []);

  const formatOrderId = useCallback((orderId: string) => {
    return orderId.length > 8 ? orderId.substring(0, 8) + '...' : orderId;
  }, []);

  const formatPrice = useCallback((price: number | null | undefined, orderType: string) => {
    if (orderType === 'MKT') return 'Market';
    return price ? `$${price.toFixed(2)}` : '-';
  }, []);

  return {
    modifyOrder,
    cancelOrder,
    refreshOrders,
    isOrderActionable,
    getStatusBadgeProps,
    getActionBadgeProps,
    formatOrderId,
    formatPrice,
    isModifying,
    isCancelling
  };
};