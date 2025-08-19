import React from 'react';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import { vi, describe, it, expect, beforeEach } from 'vitest';
import { OrderModificationModal } from '../OrderModificationModal';
import { IBOrder } from '../../hooks/useOrderManagement';

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

describe('OrderModificationModal', () => {
  const mockOrder: IBOrder = {
    order_id: 'ORDER_123456789',
    client_id: 1,
    account_id: 'ACC123',
    contract_id: '456',
    symbol: 'AAPL',
    action: 'BUY',
    order_type: 'LMT',
    total_quantity: 100,
    filled_quantity: 0,
    remaining_quantity: 100,
    limit_price: 150.00,
    status: 'Submitted'
  };

  const mockOnCancel = vi.fn();
  const mockOnModify = vi.fn();

  beforeEach(() => {
    mockOnCancel.mockClear();
    mockOnModify.mockClear();
  });

  it('should render modal when visible is true and order is provided', () => {
    render(
      <OrderModificationModal
        visible={true}
        order={mockOrder}
        onCancel={mockOnCancel}
        onModify={mockOnModify}
      />
    );

    expect(screen.getByText('Modify Order')).toBeInTheDocument();
    expect(screen.getByText('Modifying order for AAPL')).toBeInTheDocument();
    expect(screen.getByText(/Order ID: ORDER_123456789/)).toBeInTheDocument();
    expect(screen.getByText(/Current Status: Submitted/)).toBeInTheDocument();
  });

  it('should not render when visible is false', () => {
    render(
      <OrderModificationModal
        visible={false}
        order={mockOrder}
        onCancel={mockOnCancel}
        onModify={mockOnModify}
      />
    );

    expect(screen.queryByText('Modify Order')).not.toBeInTheDocument();
  });

  it('should not render when order is null', () => {
    render(
      <OrderModificationModal
        visible={true}
        order={null}
        onCancel={mockOnCancel}
        onModify={mockOnModify}
      />
    );

    expect(screen.queryByText('Modify Order')).not.toBeInTheDocument();
  });

  it('should pre-populate form with order data', () => {
    render(
      <OrderModificationModal
        visible={true}
        order={mockOrder}
        onCancel={mockOnCancel}
        onModify={mockOnModify}
      />
    );

    const quantityInput = screen.getByDisplayValue('100');
    const limitPriceInput = screen.getByDisplayValue('150');

    expect(quantityInput).toBeInTheDocument();
    expect(limitPriceInput).toBeInTheDocument();
  });

  it('should show limit price field for non-market orders', () => {
    render(
      <OrderModificationModal
        visible={true}
        order={mockOrder}
        onCancel={mockOnCancel}
        onModify={mockOnModify}
      />
    );

    expect(screen.getByLabelText('Limit Price')).toBeInTheDocument();
  });

  it('should hide limit price field for market orders', () => {
    const marketOrder = { ...mockOrder, order_type: 'MKT', limit_price: undefined };

    render(
      <OrderModificationModal
        visible={true}
        order={marketOrder}
        onCancel={mockOnCancel}
        onModify={mockOnModify}
      />
    );

    expect(screen.queryByLabelText('Limit Price')).not.toBeInTheDocument();
  });

  it('should show stop price field for stop orders', () => {
    const stopOrder = { 
      ...mockOrder, 
      order_type: 'STP_LMT', 
      stop_price: 145.00 
    };

    render(
      <OrderModificationModal
        visible={true}
        order={stopOrder}
        onCancel={mockOnCancel}
        onModify={mockOnModify}
      />
    );

    expect(screen.getByLabelText('Stop Price')).toBeInTheDocument();
    expect(screen.getByDisplayValue('145')).toBeInTheDocument();
  });

  it('should call onModify when form is submitted with valid data', async () => {
    mockOnModify.mockResolvedValueOnce({ success: true });

    render(
      <OrderModificationModal
        visible={true}
        order={mockOrder}
        onCancel={mockOnCancel}
        onModify={mockOnModify}
      />
    );

    // Change quantity
    const quantityInput = screen.getByDisplayValue('100');
    fireEvent.change(quantityInput, { target: { value: '150' } });

    // Change limit price
    const limitPriceInput = screen.getByDisplayValue('150');
    fireEvent.change(limitPriceInput, { target: { value: '155.50' } });

    // Submit form
    const modifyButton = screen.getByRole('button', { name: /modify order/i });
    fireEvent.click(modifyButton);

    await waitFor(() => {
      expect(mockOnModify).toHaveBeenCalledWith('ORDER_123456789', {
        quantity: 150,
        limit_price: 155.50,
        stop_price: undefined
      });
    });
  });

  it('should call onCancel and reset form when modal is cancelled', async () => {
    render(
      <OrderModificationModal
        visible={true}
        order={mockOrder}
        onCancel={mockOnCancel}
        onModify={mockOnModify}
      />
    );

    const cancelButton = screen.getByRole('button', { name: /cancel/i });
    fireEvent.click(cancelButton);

    expect(mockOnCancel).toHaveBeenCalled();
  });

  it('should close modal and reset form after successful modification', async () => {
    mockOnModify.mockResolvedValueOnce({ success: true });

    render(
      <OrderModificationModal
        visible={true}
        order={mockOrder}
        onCancel={mockOnCancel}
        onModify={mockOnModify}
      />
    );

    // Submit form
    const modifyButton = screen.getByRole('button', { name: /modify order/i });
    fireEvent.click(modifyButton);

    await waitFor(() => {
      expect(mockOnCancel).toHaveBeenCalled();
    });
  });

  it('should not close modal when modification fails', async () => {
    mockOnModify.mockResolvedValueOnce({ success: false });

    render(
      <OrderModificationModal
        visible={true}
        order={mockOrder}
        onCancel={mockOnCancel}
        onModify={mockOnModify}
      />
    );

    // Submit form
    const modifyButton = screen.getByRole('button', { name: /modify order/i });
    fireEvent.click(modifyButton);

    await waitFor(() => {
      expect(mockOnModify).toHaveBeenCalled();
    });

    // Modal should still be open
    expect(mockOnCancel).not.toHaveBeenCalled();
    expect(screen.getByText('Modify Order')).toBeInTheDocument();
  });

  it('should show loading state when loading prop is true', () => {
    render(
      <OrderModificationModal
        visible={true}
        order={mockOrder}
        loading={true}
        onCancel={mockOnCancel}
        onModify={mockOnModify}
      />
    );

    const modifyButton = screen.getByRole('button', { name: /modify order/i });
    expect(modifyButton).toBeDisabled();
  });

  it('should validate required quantity field', async () => {
    render(
      <OrderModificationModal
        visible={true}
        order={mockOrder}
        onCancel={mockOnCancel}
        onModify={mockOnModify}
      />
    );

    // Clear quantity field
    const quantityInput = screen.getByDisplayValue('100');
    fireEvent.change(quantityInput, { target: { value: '' } });

    // Try to submit
    const modifyButton = screen.getByRole('button', { name: /modify order/i });
    fireEvent.click(modifyButton);

    await waitFor(() => {
      expect(screen.getByText('Please enter quantity')).toBeInTheDocument();
    });

    expect(mockOnModify).not.toHaveBeenCalled();
  });

  it('should validate minimum quantity', async () => {
    render(
      <OrderModificationModal
        visible={true}
        order={mockOrder}
        onCancel={mockOnCancel}
        onModify={mockOnModify}
      />
    );

    // Set invalid quantity
    const quantityInput = screen.getByDisplayValue('100');
    fireEvent.change(quantityInput, { target: { value: '0' } });

    // Try to submit
    const modifyButton = screen.getByRole('button', { name: /modify order/i });
    fireEvent.click(modifyButton);

    await waitFor(() => {
      expect(screen.getByText('Quantity must be at least 1')).toBeInTheDocument();
    });

    expect(mockOnModify).not.toHaveBeenCalled();
  });

  it('should validate minimum price for limit orders', async () => {
    render(
      <OrderModificationModal
        visible={true}
        order={mockOrder}
        onCancel={mockOnCancel}
        onModify={mockOnModify}
      />
    );

    // Set invalid price
    const limitPriceInput = screen.getByDisplayValue('150');
    fireEvent.change(limitPriceInput, { target: { value: '0' } });

    // Try to submit
    const modifyButton = screen.getByRole('button', { name: /modify order/i });
    fireEvent.click(modifyButton);

    await waitFor(() => {
      expect(screen.getByText('Price must be greater than 0')).toBeInTheDocument();
    });

    expect(mockOnModify).not.toHaveBeenCalled();
  });
});