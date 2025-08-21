/**
 * Unit tests for usePositionData hook
 */

import { renderHook, act } from '@testing-library/react';
import { describe, it, expect, beforeEach, vi } from 'vitest';
import { usePositionData } from '../usePositionData';
import { positionService } from '../../services/positionService';

// Mock the position service
vi.mock('../../services/positionService', () => ({
  positionService: {
    getPositions: vi.fn(),
    getAccountBalances: vi.fn(),
    getPositionSummary: vi.fn(),
    getAlerts: vi.fn(),
    acknowledgeAlert: vi.fn(),
    addPositionHandler: vi.fn(),
    addAccountHandler: vi.fn(),
    addSummaryHandler: vi.fn(),
    addAlertHandler: vi.fn(),
    removePositionHandler: vi.fn(),
    removeAccountHandler: vi.fn(),
    removeSummaryHandler: vi.fn(),
    removeAlertHandler: vi.fn(),
  }
}));

const mockPositionService = positionService as any;

describe('usePositionData', () => {
  beforeEach(() => {
    vi.clearAllMocks();
    
    // Setup default mock returns
    mockPositionService.getPositions.mockReturnValue([]);
    mockPositionService.getAccountBalances.mockReturnValue([]);
    mockPositionService.getPositionSummary.mockReturnValue({
      totalPositions: 0,
      longPositions: 0,
      shortPositions: 0,
      totalExposure: 0,
      netExposure: 0,
      grossExposure: 0,
      pnl: {
        unrealizedPnl: 0,
        realizedPnl: 0,
        totalPnl: 0,
        currency: 'USD',
        dailyPnl: 0,
        pnlPercentage: 0
      },
      currency: 'USD'
    });
    mockPositionService.getAlerts.mockReturnValue([]);
  });

  it('should initialize with empty state when no data available', () => {
    const { result } = renderHook(() => usePositionData());

    expect(result.current.positions).toEqual([]);
    expect(result.current.accountBalances).toEqual([]);
    expect(result.current.alerts).toEqual([]);
    expect(result.current.isLoading).toBe(true);
    expect(result.current.error).toBeNull();
    expect(result.current.hasPositions).toBe(false);
    expect(result.current.hasAccounts).toBe(false);
    expect(result.current.hasAlerts).toBe(false);
  });

  it('should setup event handlers on mount', () => {
    renderHook(() => usePositionData());

    expect(mockPositionService.addPositionHandler).toHaveBeenCalledTimes(1);
    expect(mockPositionService.addAccountHandler).toHaveBeenCalledTimes(1);
    expect(mockPositionService.addSummaryHandler).toHaveBeenCalledTimes(1);
    expect(mockPositionService.addAlertHandler).toHaveBeenCalledTimes(1);
  });

  it('should cleanup event handlers on unmount', () => {
    const { unmount } = renderHook(() => usePositionData());

    unmount();

    expect(mockPositionService.removePositionHandler).toHaveBeenCalledTimes(1);
    expect(mockPositionService.removeAccountHandler).toHaveBeenCalledTimes(1);
    expect(mockPositionService.removeSummaryHandler).toHaveBeenCalledTimes(1);
    expect(mockPositionService.removeAlertHandler).toHaveBeenCalledTimes(1);
  });

  it('should acknowledge alerts correctly', () => {
    const { result } = renderHook(() => usePositionData());

    act(() => {
      result.current.acknowledgeAlert('alert-1');
    });

    expect(mockPositionService.acknowledgeAlert).toHaveBeenCalledWith('alert-1');
  });

  it('should refresh data correctly', () => {
    mockPositionService.getPositions.mockReturnValue([
      {
        id: 'pos-1',
        symbol: 'AAPL',
        venue: 'NASDAQ',
        side: 'LONG',
        quantity: 100,
        averagePrice: 150,
        unrealizedPnl: 500,
        realizedPnl: 0,
        currentPrice: 155,
        currency: 'USD',
        timestamp: Date.now(),
        openTimestamp: Date.now()
      }
    ]);

    const { result } = renderHook(() => usePositionData());

    act(() => {
      result.current.refreshData();
    });

    expect(result.current.positions).toHaveLength(1);
    expect(result.current.positions[0].symbol).toBe('AAPL');
    expect(result.current.isLoading).toBe(false);
  });

  it('should filter positions by symbol correctly', () => {
    const mockPositions = [
      {
        id: 'pos-1',
        symbol: 'AAPL',
        venue: 'NASDAQ',
        side: 'LONG' as const,
        quantity: 100,
        averagePrice: 150,
        unrealizedPnl: 500,
        realizedPnl: 0,
        currentPrice: 155,
        currency: 'USD',
        timestamp: Date.now(),
        openTimestamp: Date.now()
      },
      {
        id: 'pos-2',
        symbol: 'GOOGL',
        venue: 'NASDAQ',
        side: 'SHORT' as const,
        quantity: 50,
        averagePrice: 2800,
        unrealizedPnl: -1000,
        realizedPnl: 0,
        currentPrice: 2820,
        currency: 'USD',
        timestamp: Date.now(),
        openTimestamp: Date.now()
      }
    ];

    mockPositionService.getPositions.mockReturnValue(mockPositions);

    const { result } = renderHook(() => usePositionData());

    act(() => {
      result.current.refreshData();
    });

    const aaplPositions = result.current.getPositionsBySymbol('AAPL');
    expect(aaplPositions).toHaveLength(1);
    expect(aaplPositions[0].symbol).toBe('AAPL');

    const googlPositions = result.current.getPositionsBySymbol('GOOGL');
    expect(googlPositions).toHaveLength(1);
    expect(googlPositions[0].symbol).toBe('GOOGL');

    const nonExistentPositions = result.current.getPositionsBySymbol('TSLA');
    expect(nonExistentPositions).toHaveLength(0);
  });

  it('should get position by ID correctly', () => {
    const mockPositions = [
      {
        id: 'pos-1',
        symbol: 'AAPL',
        venue: 'NASDAQ',
        side: 'LONG' as const,
        quantity: 100,
        averagePrice: 150,
        unrealizedPnl: 500,
        realizedPnl: 0,
        currentPrice: 155,
        currency: 'USD',
        timestamp: Date.now(),
        openTimestamp: Date.now()
      }
    ];

    mockPositionService.getPositions.mockReturnValue(mockPositions);

    const { result } = renderHook(() => usePositionData());

    act(() => {
      result.current.refreshData();
    });

    const position = result.current.getPositionById('pos-1');
    expect(position).toBeDefined();
    expect(position?.symbol).toBe('AAPL');

    const nonExistentPosition = result.current.getPositionById('pos-999');
    expect(nonExistentPosition).toBeUndefined();
  });

  it('should calculate total portfolio value correctly', () => {
    const mockSummary = {
      totalPositions: 1,
      longPositions: 1,
      shortPositions: 0,
      totalExposure: 15500,
      netExposure: 15500,
      grossExposure: 15500,
      pnl: {
        unrealizedPnl: 500,
        realizedPnl: 0,
        totalPnl: 500,
        currency: 'USD',
        dailyPnl: 100,
        pnlPercentage: 3.23
      },
      currency: 'USD'
    };

    mockPositionService.getPositionSummary.mockReturnValue(mockSummary);

    const { result } = renderHook(() => usePositionData());

    act(() => {
      result.current.refreshData();
    });

    const totalValue = result.current.getTotalPortfolioValue();
    expect(totalValue).toBe(15500);
  });

  it('should count unacknowledged alerts correctly', () => {
    const mockAlerts = [
      {
        id: 'alert-1',
        positionId: 'pos-1',
        type: 'price_change' as const,
        message: 'Price changed',
        severity: 'info' as const,
        timestamp: Date.now(),
        acknowledged: false
      },
      {
        id: 'alert-2',
        positionId: 'pos-1',
        type: 'pnl_threshold' as const,
        message: 'P&L threshold',
        severity: 'warning' as const,
        timestamp: Date.now(),
        acknowledged: true
      }
    ];

    mockPositionService.getAlerts.mockReturnValue(mockAlerts);

    const { result } = renderHook(() => usePositionData());

    act(() => {
      result.current.refreshData();
    });

    expect(result.current.getUnacknowledgedAlertsCount()).toBe(1);
    expect(result.current.hasUnacknowledgedAlerts).toBe(true);
  });
});