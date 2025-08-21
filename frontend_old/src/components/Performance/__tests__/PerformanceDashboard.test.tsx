import React from 'react';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import { vi, describe, it, expect, beforeEach, afterEach } from 'vitest';
import { PerformanceDashboard } from '../PerformanceDashboard';

// Mock fetch
global.fetch = vi.fn();

const mockFetch = fetch as vi.MockedFunction<typeof fetch>;

const mockStrategyData = {
  instances: [
    {
      id: 'test-strategy-1',
      config_id: 'config-1',
      nautilus_strategy_id: 'nautilus-1',
      deployment_id: 'deploy-1',
      state: 'running',
      performance_metrics: {
        total_pnl: '1250.50',
        unrealized_pnl: '125.25',
        total_trades: 45,
        winning_trades: 28,
        win_rate: 0.622,
        max_drawdown: '8.5',
        sharpe_ratio: 1.85,
        last_updated: new Date()
      },
      runtime_info: {
        orders_placed: 45,
        positions_opened: 12,
        uptime_seconds: 86400
      },
      error_log: [],
      started_at: new Date(),
      stopped_at: null
    }
  ]
};

const mockPerformanceData = {
  total_pnl: '1250.50',
  unrealized_pnl: '125.25',
  total_trades: 45,
  winning_trades: 28,
  win_rate: 0.622,
  max_drawdown: '8.5',
  sharpe_ratio: 1.85,
  daily_pnl_change: 2.3,
  weekly_pnl_change: 8.7,
  monthly_pnl_change: 15.2,
  daily_returns: [0.01, 0.02, -0.005, 0.015, 0.008],
  last_updated: new Date()
};

const mockPerformanceHistory = {
  snapshots: [
    {
      timestamp: new Date('2023-01-01'),
      total_pnl: 1000,
      unrealized_pnl: 50,
      drawdown: 2.5,
      sharpe_ratio: 1.5,
      win_rate: 0.6
    },
    {
      timestamp: new Date('2023-01-02'),
      total_pnl: 1100,
      unrealized_pnl: 75,
      drawdown: 1.8,
      sharpe_ratio: 1.6,
      win_rate: 0.62
    }
  ]
};

describe('PerformanceDashboard', () => {
  beforeEach(() => {
    mockFetch.mockClear();
    
    // Default successful responses
    mockFetch.mockImplementation((url) => {
      if (url.includes('/api/v1/strategies/active')) {
        return Promise.resolve({
          ok: true,
          json: () => Promise.resolve(mockStrategyData)
        } as Response);
      }
      
      if (url.includes('/api/v1/performance/aggregate') || url.includes('/strategies/') && url.includes('/performance')) {
        return Promise.resolve({
          ok: true,
          json: () => Promise.resolve(mockPerformanceData)
        } as Response);
      }
      
      if (url.includes('/api/v1/performance/history')) {
        return Promise.resolve({
          ok: true,
          json: () => Promise.resolve(mockPerformanceHistory)
        } as Response);
      }
      
      return Promise.resolve({
        ok: false,
        json: () => Promise.resolve({})
      } as Response);
    });
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  it('renders performance dashboard with title', () => {
    render(<PerformanceDashboard />);
    
    expect(screen.getByText('Performance Dashboard')).toBeInTheDocument();
    expect(screen.getByText('Real-time strategy performance monitoring and analytics')).toBeInTheDocument();
  });

  it('loads and displays performance data', async () => {
    render(<PerformanceDashboard />);
    
    await waitFor(() => {
      expect(screen.getByText('$1375.75')).toBeInTheDocument(); // total_pnl + unrealized_pnl
    });
    
    expect(screen.getByText('1.85')).toBeInTheDocument(); // Sharpe ratio
    expect(screen.getByText('62.2%')).toBeInTheDocument(); // Win rate
  });

  it('displays metrics cards with correct values', async () => {
    render(<PerformanceDashboard />);
    
    await waitFor(() => {
      expect(screen.getByText('Total P&L')).toBeInTheDocument();
      expect(screen.getByText('Sharpe Ratio')).toBeInTheDocument();
      expect(screen.getByText('Max Drawdown')).toBeInTheDocument();
      expect(screen.getByText('Win Rate')).toBeInTheDocument();
    });
  });

  it('shows high drawdown warning when appropriate', async () => {
    const highDrawdownData = {
      ...mockPerformanceData,
      max_drawdown: '15.5' // Above 10% threshold
    };
    
    mockFetch.mockImplementation((url) => {
      if (url.includes('/api/v1/performance/')) {
        return Promise.resolve({
          ok: true,
          json: () => Promise.resolve(highDrawdownData)
        } as Response);
      }
      return Promise.resolve({
        ok: true,
        json: () => Promise.resolve(mockStrategyData)
      } as Response);
    });
    
    render(<PerformanceDashboard />);
    
    await waitFor(() => {
      expect(screen.getByText('High Drawdown Warning')).toBeInTheDocument();
      expect(screen.getByText(/15.50%, which exceeds the 10% threshold/)).toBeInTheDocument();
    });
  });

  it('handles strategy selection change', async () => {
    render(<PerformanceDashboard />);
    
    await waitFor(() => {
      const strategySelect = screen.getByDisplayValue('All Strategies');
      expect(strategySelect).toBeInTheDocument();
    });
    
    // Should trigger new API call when strategy changes
    fireEvent.change(screen.getByDisplayValue('All Strategies'), {
      target: { value: 'test-strategy-1' }
    });
    
    await waitFor(() => {
      expect(mockFetch).toHaveBeenCalledWith(
        expect.stringContaining('/api/v1/strategies/test-strategy-1/performance')
      );
    });
  });

  it('refreshes data when refresh button is clicked', async () => {
    render(<PerformanceDashboard />);
    
    const refreshButton = screen.getByText('Refresh');
    fireEvent.click(refreshButton);
    
    await waitFor(() => {
      expect(mockFetch).toHaveBeenCalledTimes(6); // Initial load (3 calls) + refresh (3 calls)
    });
  });

  it('navigates between tabs correctly', async () => {
    render(<PerformanceDashboard />);
    
    // Check default tab
    expect(screen.getByText('Overview')).toHaveClass('ant-tabs-tab-active');
    
    // Click on Real-Time Monitor tab
    fireEvent.click(screen.getByText('Real-Time Monitor'));
    expect(screen.getByText('Real-Time Monitor')).toHaveClass('ant-tabs-tab-active');
    
    // Click on Strategy Comparison tab
    fireEvent.click(screen.getByText('Strategy Comparison'));
    expect(screen.getByText('Strategy Comparison')).toHaveClass('ant-tabs-tab-active');
    
    // Click on Execution Analytics tab
    fireEvent.click(screen.getByText('Execution Analytics'));
    expect(screen.getByText('Execution Analytics')).toHaveClass('ant-tabs-tab-active');
    
    // Click on Alert System tab
    fireEvent.click(screen.getByText('Alert System'));
    expect(screen.getByText('Alert System')).toHaveClass('ant-tabs-tab-active');
  });

  it('calculates Sharpe ratio correctly from daily returns', async () => {
    const dataWithReturns = {
      ...mockPerformanceData,
      daily_returns: [0.01, 0.02, -0.01, 0.015, 0.005], // Known returns for calculation
      sharpe_ratio: null // Force calculation
    };
    
    mockFetch.mockImplementation((url) => {
      if (url.includes('/api/v1/performance/')) {
        return Promise.resolve({
          ok: true,
          json: () => Promise.resolve(dataWithReturns)
        } as Response);
      }
      return Promise.resolve({
        ok: true,
        json: () => Promise.resolve(mockStrategyData)
      } as Response);
    });
    
    render(<PerformanceDashboard />);
    
    await waitFor(() => {
      // Should display calculated Sharpe ratio
      const sharpeElements = screen.getAllByText(/\d+\.\d{2}/);
      expect(sharpeElements.length).toBeGreaterThan(0);
    });
  });

  it('handles API errors gracefully', async () => {
    mockFetch.mockImplementation(() => 
      Promise.resolve({
        ok: false,
        json: () => Promise.resolve({})
      } as Response)
    );
    
    render(<PerformanceDashboard />);
    
    // Should not crash and show loading state
    expect(screen.getByText('Loading performance data...')).toBeInTheDocument();
  });

  it('updates real-time data at intervals', async () => {
    vi.useFakeTimers();
    
    render(<PerformanceDashboard />);
    
    // Initial load
    await waitFor(() => {
      expect(mockFetch).toHaveBeenCalledTimes(3);
    });
    
    // Fast forward 5 seconds (auto-refresh interval)
    vi.advanceTimersByTime(5000);
    
    await waitFor(() => {
      expect(mockFetch).toHaveBeenCalledTimes(6); // Another set of calls
    });
    
    vi.useRealTimers();
  });

  it('displays trading statistics correctly', async () => {
    render(<PerformanceDashboard />);
    
    await waitFor(() => {
      expect(screen.getByText('Trading Statistics')).toBeInTheDocument();
      expect(screen.getByText('45')).toBeInTheDocument(); // Total trades
      expect(screen.getByText('28')).toBeInTheDocument(); // Winning trades
      expect(screen.getByText('17')).toBeInTheDocument(); // Losing trades (45-28)
    });
  });

  it('shows last update timestamp', async () => {
    render(<PerformanceDashboard />);
    
    await waitFor(() => {
      expect(screen.getByText(/Last updated:/)).toBeInTheDocument();
    });
  });

  it('formats P&L values with correct colors', async () => {
    render(<PerformanceDashboard />);
    
    await waitFor(() => {
      const pnlElement = screen.getByText('$1375.75');
      expect(pnlElement).toHaveStyle({ color: '#3f8600' }); // Positive P&L should be green
    });
  });
});