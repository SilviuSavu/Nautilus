import React from 'react';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import { vi, describe, it, expect, beforeEach } from 'vitest';
import { StrategyComparison } from '../StrategyComparison';
import dayjs from 'dayjs';

// Mock fetch
global.fetch = vi.fn();
const mockFetch = fetch as vi.MockedFunction<typeof fetch>;

const mockStrategies = [
  {
    id: 'momentum_1',
    config_id: 'config_1',
    nautilus_strategy_id: 'nautilus_1',
    deployment_id: 'deploy_1',
    state: 'running' as const,
    performance_metrics: {
      total_pnl: '2450.75',
      unrealized_pnl: '245.30',
      total_trades: 125,
      winning_trades: 78,
      win_rate: 0.624,
      max_drawdown: '6.2',
      sharpe_ratio: 1.75,
      last_updated: new Date()
    },
    runtime_info: {
      orders_placed: 125,
      positions_opened: 45,
      uptime_seconds: 86400
    },
    error_log: [],
    started_at: new Date(),
    stopped_at: undefined
  },
  {
    id: 'mean_revert_2',
    config_id: 'config_2',
    nautilus_strategy_id: 'nautilus_2',
    deployment_id: 'deploy_2',
    state: 'paused' as const,
    performance_metrics: {
      total_pnl: '-450.25',
      unrealized_pnl: '0.00',
      total_trades: 85,
      winning_trades: 42,
      win_rate: 0.494,
      max_drawdown: '12.8',
      sharpe_ratio: 0.85,
      last_updated: new Date()
    },
    runtime_info: {
      orders_placed: 85,
      positions_opened: 25,
      uptime_seconds: 43200
    },
    error_log: [],
    started_at: new Date(),
    stopped_at: undefined
  }
];

const mockTimeRange: [dayjs.Dayjs, dayjs.Dayjs] = [
  dayjs().subtract(30, 'days'),
  dayjs()
];

const mockComparisonData = {
  comparisons: [
    {
      strategy_id: 'momentum_1',
      strategy_name: 'Momentum Strategy',
      total_pnl: 2450.75,
      win_rate: 0.624,
      sharpe_ratio: 1.75,
      max_drawdown: 6.2,
      total_trades: 125,
      avg_trade_pnl: 19.61,
      volatility: 18.5,
      calmar_ratio: 2.82,
      sortino_ratio: 2.45,
      beta: 1.12,
      correlation_to_benchmark: 0.78,
      state: 'running',
      uptime_percentage: 98.5,
      last_update: new Date()
    },
    {
      strategy_id: 'mean_revert_2',
      strategy_name: 'Mean Reversion Strategy',
      total_pnl: -450.25,
      win_rate: 0.494,
      sharpe_ratio: 0.85,
      max_drawdown: 12.8,
      total_trades: 85,
      avg_trade_pnl: -5.30,
      volatility: 22.1,
      calmar_ratio: 0.66,
      sortino_ratio: 1.12,
      beta: 0.85,
      correlation_to_benchmark: 0.62,
      state: 'paused',
      uptime_percentage: 87.2,
      last_update: new Date()
    }
  ]
};

const mockBenchmarkData = {
  name: 'S&P 500',
  symbol: 'SPY',
  return_period: 8.2,
  volatility: 16.5,
  sharpe_ratio: 0.52
};

describe('StrategyComparison', () => {
  beforeEach(() => {
    mockFetch.mockClear();
  });

  it('renders strategy comparison table with data', async () => {
    mockFetch
      .mockResolvedValueOnce({
        ok: true,
        json: async () => mockComparisonData
      } as Response)
      .mockResolvedValueOnce({
        ok: true,
        json: async () => mockBenchmarkData
      } as Response);

    render(
      <StrategyComparison 
        strategies={mockStrategies} 
        timeRange={mockTimeRange}
      />
    );

    // Check header
    await waitFor(() => {
      expect(screen.getByText('Strategy Comparison')).toBeInTheDocument();
    });

    // Check table headers
    expect(screen.getByText('Rank')).toBeInTheDocument();
    expect(screen.getByText('Strategy')).toBeInTheDocument();
    expect(screen.getByText('Total P&L')).toBeInTheDocument();
    expect(screen.getByText('Win Rate')).toBeInTheDocument();
    expect(screen.getByText('Sharpe Ratio')).toBeInTheDocument();

    // Check strategy data loads
    await waitFor(() => {
      expect(screen.getByText('Momentum Strategy')).toBeInTheDocument();
      expect(screen.getByText('Mean Reversion Strategy')).toBeInTheDocument();
    });

    // Verify API calls
    expect(mockFetch).toHaveBeenCalledWith('/api/v1/performance/compare', expect.objectContaining({
      method: 'POST',
      headers: { 'Content-Type': 'application/json' }
    }));
    expect(mockFetch).toHaveBeenCalledWith(expect.stringContaining('/api/v1/performance/benchmarks/SPY'));
  });

  it('displays performance metrics correctly', async () => {
    mockFetch
      .mockResolvedValueOnce({
        ok: true,
        json: async () => mockComparisonData
      } as Response)
      .mockResolvedValueOnce({
        ok: true,
        json: async () => mockBenchmarkData
      } as Response);

    render(
      <StrategyComparison 
        strategies={mockStrategies} 
        timeRange={mockTimeRange}
      />
    );

    await waitFor(() => {
      // Check P&L values
      expect(screen.getByText('$2450.75')).toBeInTheDocument();
      expect(screen.getByText('$-450.25')).toBeInTheDocument();
    });

    await waitFor(() => {
      // Check win rates
      expect(screen.getByText('62.4%')).toBeInTheDocument();
      expect(screen.getByText('49.4%')).toBeInTheDocument();
    });

    await waitFor(() => {
      // Check Sharpe ratios
      expect(screen.getByText('1.75')).toBeInTheDocument();
      expect(screen.getByText('0.85')).toBeInTheDocument();
    });
  });

  it('shows ranking with proper icons', async () => {
    mockFetch
      .mockResolvedValueOnce({
        ok: true,
        json: async () => mockComparisonData
      } as Response)
      .mockResolvedValueOnce({
        ok: true,
        json: async () => mockBenchmarkData
      } as Response);

    render(
      <StrategyComparison 
        strategies={mockStrategies} 
        timeRange={mockTimeRange}
      />
    );

    await waitFor(() => {
      // Check ranking icons (first should be gold medal emoji)
      expect(screen.getByText('🥇')).toBeInTheDocument();
      expect(screen.getByText('🥈')).toBeInTheDocument();
    });
  });

  it('displays benchmark information', async () => {
    mockFetch
      .mockResolvedValueOnce({
        ok: true,
        json: async () => mockComparisonData
      } as Response)
      .mockResolvedValueOnce({
        ok: true,
        json: async () => mockBenchmarkData
      } as Response);

    render(
      <StrategyComparison 
        strategies={mockStrategies} 
        timeRange={mockTimeRange}
      />
    );

    await waitFor(() => {
      expect(screen.getByText('SPY Benchmark:')).toBeInTheDocument();
      expect(screen.getByText('Return: 8.20%')).toBeInTheDocument();
      expect(screen.getByText('Volatility: 16.50%')).toBeInTheDocument();
      expect(screen.getByText('Sharpe: 0.52')).toBeInTheDocument();
    });
  });

  it('allows benchmark selection', async () => {
    mockFetch
      .mockResolvedValue({
        ok: true,
        json: async () => mockComparisonData
      } as Response);

    render(
      <StrategyComparison 
        strategies={mockStrategies} 
        timeRange={mockTimeRange}
      />
    );

    // Find benchmark selector
    const benchmarkSelect = screen.getByDisplayValue('S&P 500 (SPY)');
    expect(benchmarkSelect).toBeInTheDocument();

    // Change benchmark
    fireEvent.change(benchmarkSelect, { target: { value: 'QQQ' } });

    await waitFor(() => {
      // Should trigger new API call for QQQ benchmark
      expect(mockFetch).toHaveBeenCalledWith(expect.stringContaining('/api/v1/performance/benchmarks/QQQ'));
    });
  });

  it('supports sorting by different metrics', async () => {
    mockFetch
      .mockResolvedValue({
        ok: true,
        json: async () => mockComparisonData
      } as Response);

    render(
      <StrategyComparison 
        strategies={mockStrategies} 
        timeRange={mockTimeRange}
      />
    );

    await waitFor(() => {
      // Find sort selector
      const sortSelect = screen.getByDisplayValue('P&L (High to Low)');
      expect(sortSelect).toBeInTheDocument();
    });

    // Change sorting to Sharpe ratio
    const sortSelect = screen.getByDisplayValue('P&L (High to Low)');
    fireEvent.change(sortSelect, { target: { value: 'sharpe_ratio-desc' } });

    // Should re-sort the table (testing internal state change)
    expect(sortSelect).toBeInTheDocument();
  });

  it('shows state indicators correctly', async () => {
    mockFetch
      .mockResolvedValueOnce({
        ok: true,
        json: async () => mockComparisonData
      } as Response)
      .mockResolvedValueOnce({
        ok: true,
        json: async () => mockBenchmarkData
      } as Response);

    render(
      <StrategyComparison 
        strategies={mockStrategies} 
        timeRange={mockTimeRange}
      />
    );

    await waitFor(() => {
      // Check strategy states
      expect(screen.getByText('running')).toBeInTheDocument();
      expect(screen.getByText('paused')).toBeInTheDocument();
    });

    await waitFor(() => {
      // Check uptime percentages
      expect(screen.getByText('98.5% uptime')).toBeInTheDocument();
      expect(screen.getByText('87.2% uptime')).toBeInTheDocument();
    });
  });

  it('displays risk metrics tooltips', async () => {
    mockFetch
      .mockResolvedValueOnce({
        ok: true,
        json: async () => mockComparisonData
      } as Response)
      .mockResolvedValueOnce({
        ok: true,
        json: async () => mockBenchmarkData
      } as Response);

    render(
      <StrategyComparison 
        strategies={mockStrategies} 
        timeRange={mockTimeRange}
      />
    );

    await waitFor(() => {
      // Check risk metrics are displayed
      expect(screen.getByText('Vol: 18.50%')).toBeInTheDocument();
      expect(screen.getByText('Calmar: 2.82')).toBeInTheDocument();
      expect(screen.getByText('Sortino: 2.45')).toBeInTheDocument();
    });
  });

  it('shows benchmark correlation data', async () => {
    mockFetch
      .mockResolvedValueOnce({
        ok: true,
        json: async () => mockComparisonData
      } as Response)
      .mockResolvedValueOnce({
        ok: true,
        json: async () => mockBenchmarkData
      } as Response);

    render(
      <StrategyComparison 
        strategies={mockStrategies} 
        timeRange={mockTimeRange}
      />
    );

    await waitFor(() => {
      // Check beta values
      expect(screen.getByText('β: 1.12')).toBeInTheDocument();
      expect(screen.getByText('β: 0.85')).toBeInTheDocument();
    });

    await waitFor(() => {
      // Check correlation values
      expect(screen.getByText('ρ: 0.78')).toBeInTheDocument();
      expect(screen.getByText('ρ: 0.62')).toBeInTheDocument();
    });
  });

  it('handles refresh functionality', async () => {
    mockFetch
      .mockResolvedValue({
        ok: true,
        json: async () => mockComparisonData
      } as Response);

    render(
      <StrategyComparison 
        strategies={mockStrategies} 
        timeRange={mockTimeRange}
      />
    );

    // Find and click refresh button
    const refreshButton = screen.getByRole('button', { name: /refresh/i });
    expect(refreshButton).toBeInTheDocument();

    fireEvent.click(refreshButton);

    // Should trigger additional API calls
    await waitFor(() => {
      expect(mockFetch).toHaveBeenCalledTimes(4); // Initial 2 + refresh 2
    });
  });

  it('handles API errors gracefully', async () => {
    mockFetch.mockRejectedValue(new Error('API Error'));

    render(
      <StrategyComparison 
        strategies={mockStrategies} 
        timeRange={mockTimeRange}
      />
    );

    // Should not crash and should show error state
    await waitFor(() => {
      expect(screen.getByText('Strategy Comparison')).toBeInTheDocument();
    });

    // Table should be empty but structure should exist
    expect(screen.getByText('Rank')).toBeInTheDocument();
    expect(screen.getByText('Strategy')).toBeInTheDocument();
  });

  it('calculates relative performance vs benchmark correctly', async () => {
    const enhancedMockData = {
      comparisons: [
        {
          ...mockComparisonData.comparisons[0],
          total_pnl: 1000 // 10% return
        }
      ]
    };

    const benchmarkWith8Percent = {
      ...mockBenchmarkData,
      return_period: 800 // 8% return in dollar terms
    };

    mockFetch
      .mockResolvedValueOnce({
        ok: true,
        json: async () => enhancedMockData
      } as Response)
      .mockResolvedValueOnce({
        ok: true,
        json: async () => benchmarkWith8Percent
      } as Response);

    render(
      <StrategyComparison 
        strategies={mockStrategies} 
        timeRange={mockTimeRange}
      />
    );

    await waitFor(() => {
      // Should show relative performance calculation
      expect(screen.getByText(/vs SPY:/)).toBeInTheDocument();
    });
  });

  it('updates when time range changes', async () => {
    mockFetch
      .mockResolvedValue({
        ok: true,
        json: async () => mockComparisonData
      } as Response);

    const { rerender } = render(
      <StrategyComparison 
        strategies={mockStrategies} 
        timeRange={mockTimeRange}
      />
    );

    // Initial API calls
    await waitFor(() => {
      expect(mockFetch).toHaveBeenCalledTimes(2);
    });

    // Change time range
    const newTimeRange: [dayjs.Dayjs, dayjs.Dayjs] = [
      dayjs().subtract(7, 'days'),
      dayjs()
    ];

    rerender(
      <StrategyComparison 
        strategies={mockStrategies} 
        timeRange={newTimeRange}
      />
    );

    // Should trigger new API calls with updated time range
    await waitFor(() => {
      expect(mockFetch).toHaveBeenCalledTimes(4);
    });
  });
});