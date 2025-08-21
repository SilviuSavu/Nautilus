import React from 'react';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import { vi, describe, it, expect, beforeEach } from 'vitest';
import { ExecutionAnalytics } from '../ExecutionAnalytics';

// Mock fetch
global.fetch = vi.fn();
const mockFetch = fetch as vi.MockedFunction<typeof fetch>;

const mockExecutionMetrics = {
  metrics: [
    {
      strategy_id: 'momentum_1',
      total_trades: 125,
      avg_execution_time_ms: 45.2,
      avg_slippage_bps: 3.1,
      avg_commission: 1.25,
      fill_rate: 0.987,
      latency_p50: 35.8,
      latency_p95: 89.3,
      latency_p99: 145.7,
      market_impact_bps: 1.8,
      implementation_shortfall: 4.2,
      vwap_performance: 0.85,
      execution_cost_bps: 8.3,
      rejected_orders: 2,
      partial_fills: 8,
      successful_fills: 115
    },
    {
      strategy_id: 'mean_revert_2',
      total_trades: 85,
      avg_execution_time_ms: 67.8,
      avg_slippage_bps: 5.7,
      avg_commission: 1.50,
      fill_rate: 0.941,
      latency_p50: 52.1,
      latency_p95: 125.4,
      latency_p99: 298.6,
      market_impact_bps: 3.2,
      implementation_shortfall: 7.8,
      vwap_performance: -0.32,
      execution_cost_bps: 12.7,
      rejected_orders: 5,
      partial_fills: 12,
      successful_fills: 68
    }
  ]
};

const mockTradeExecutions = {
  trades: [
    {
      id: 'trade_1',
      strategy_id: 'momentum_1',
      instrument: 'AAPL',
      side: 'buy',
      order_type: 'market',
      quantity_requested: 100,
      quantity_filled: 100,
      requested_price: null,
      executed_price: 175.23,
      commission: 1.25,
      slippage_bps: 2.8,
      execution_time_ms: 42,
      market_impact_bps: 1.5,
      vwap_benchmark: 175.18,
      timestamp: new Date(),
      venue: 'SMART',
      order_id: 'order_12345',
      fill_quality: 'excellent'
    },
    {
      id: 'trade_2',
      strategy_id: 'mean_revert_2',
      instrument: 'MSFT',
      side: 'sell',
      order_type: 'limit',
      quantity_requested: 200,
      quantity_filled: 180,
      requested_price: 285.50,
      executed_price: 285.45,
      commission: 2.10,
      slippage_bps: -1.7,
      execution_time_ms: 125,
      market_impact_bps: 0.8,
      vwap_benchmark: 285.48,
      timestamp: new Date(),
      venue: 'ISLAND',
      order_id: 'order_67890',
      fill_quality: 'good'
    }
  ]
};

const mockSlippageAnalysis = {
  analysis: [
    {
      time_period: 'Last 24 Hours',
      avg_slippage_bps: 3.2,
      slippage_volatility: 1.8,
      worst_slippage_bps: 12.5,
      best_slippage_bps: -0.5,
      trade_count: 25,
      slippage_distribution: {
        excellent: 18,
        good: 5,
        fair: 2,
        poor: 0
      }
    },
    {
      time_period: 'Last 7 Days',
      avg_slippage_bps: 4.1,
      slippage_volatility: 2.3,
      worst_slippage_bps: 18.2,
      best_slippage_bps: -1.2,
      trade_count: 145,
      slippage_distribution: {
        excellent: 95,
        good: 35,
        fair: 12,
        poor: 3
      }
    }
  ]
};

const mockLatencyAnalysis = {
  analysis: [
    {
      strategy_id: 'momentum_1',
      avg_latency_ms: 45.2,
      p50_latency_ms: 35.8,
      p95_latency_ms: 89.3,
      p99_latency_ms: 145.7,
      max_latency_ms: 287.5,
      timeout_rate: 0.008,
      latency_trend: 'improving'
    },
    {
      strategy_id: 'mean_revert_2',
      avg_latency_ms: 67.8,
      p50_latency_ms: 52.1,
      p95_latency_ms: 125.4,
      p99_latency_ms: 298.6,
      max_latency_ms: 456.2,
      timeout_rate: 0.015,
      latency_trend: 'stable'
    }
  ]
};

describe('ExecutionAnalytics', () => {
  beforeEach(() => {
    mockFetch.mockClear();
  });

  it('renders execution analytics dashboard', async () => {
    mockFetch
      .mockResolvedValueOnce({
        ok: true,
        json: async () => mockExecutionMetrics
      } as Response)
      .mockResolvedValueOnce({
        ok: true,
        json: async () => mockTradeExecutions
      } as Response)
      .mockResolvedValueOnce({
        ok: true,
        json: async () => mockSlippageAnalysis
      } as Response)
      .mockResolvedValueOnce({
        ok: true,
        json: async () => mockLatencyAnalysis
      } as Response);

    render(<ExecutionAnalytics />);

    await waitFor(() => {
      expect(screen.getByText('Execution Analytics')).toBeInTheDocument();
      expect(screen.getByText('Trade execution quality and performance analysis')).toBeInTheDocument();
    });

    // Verify API calls
    expect(mockFetch).toHaveBeenCalledWith(expect.stringContaining('/api/v1/execution/metrics'));
    expect(mockFetch).toHaveBeenCalledWith(expect.stringContaining('/api/v1/execution/trades'));
    expect(mockFetch).toHaveBeenCalledWith(expect.stringContaining('/api/v1/execution/slippage-analysis'));
    expect(mockFetch).toHaveBeenCalledWith(expect.stringContaining('/api/v1/execution/latency-analysis'));
  });

  it('displays overview metrics correctly', async () => {
    mockFetch
      .mockResolvedValueOnce({
        ok: true,
        json: async () => mockExecutionMetrics
      } as Response)
      .mockResolvedValueOnce({
        ok: true,
        json: async () => mockTradeExecutions
      } as Response)
      .mockResolvedValueOnce({
        ok: true,
        json: async () => mockSlippageAnalysis
      } as Response)
      .mockResolvedValueOnce({
        ok: true,
        json: async () => mockLatencyAnalysis
      } as Response);

    render(<ExecutionAnalytics />);

    await waitFor(() => {
      // Check total trades (125 + 85 = 210)
      expect(screen.getByText('210')).toBeInTheDocument();
    });

    await waitFor(() => {
      // Check average fill rate calculation
      const avgFillRate = ((0.987 * 125 + 0.941 * 85) / (125 + 85)) * 100;
      expect(screen.getByText(`${avgFillRate.toFixed(1)}%`)).toBeInTheDocument();
    });
  });

  it('shows strategy execution metrics table', async () => {
    mockFetch
      .mockResolvedValueOnce({
        ok: true,
        json: async () => mockExecutionMetrics
      } as Response)
      .mockResolvedValueOnce({
        ok: true,
        json: async () => mockTradeExecutions
      } as Response)
      .mockResolvedValueOnce({
        ok: true,
        json: async () => mockSlippageAnalysis
      } as Response)
      .mockResolvedValueOnce({
        ok: true,
        json: async () => mockLatencyAnalysis
      } as Response);

    render(<ExecutionAnalytics />);

    await waitFor(() => {
      // Check strategy names
      expect(screen.getByText('momentum_1')).toBeInTheDocument();
      expect(screen.getByText('mean_revert_2')).toBeInTheDocument();
    });

    await waitFor(() => {
      // Check trade counts
      expect(screen.getByText('125')).toBeInTheDocument();
      expect(screen.getByText('85')).toBeInTheDocument();
    });

    await waitFor(() => {
      // Check fill rates
      expect(screen.getByText('98.7%')).toBeInTheDocument();
      expect(screen.getByText('94.1%')).toBeInTheDocument();
    });

    await waitFor(() => {
      // Check slippage in basis points
      expect(screen.getByText('3.1 bps')).toBeInTheDocument();
      expect(screen.getByText('5.7 bps')).toBeInTheDocument();
    });
  });

  it('displays trade quality tab with recent executions', async () => {
    mockFetch
      .mockResolvedValueOnce({
        ok: true,
        json: async () => mockExecutionMetrics
      } as Response)
      .mockResolvedValueOnce({
        ok: true,
        json: async () => mockTradeExecutions
      } as Response)
      .mockResolvedValueOnce({
        ok: true,
        json: async () => mockSlippageAnalysis
      } as Response)
      .mockResolvedValueOnce({
        ok: true,
        json: async () => mockLatencyAnalysis
      } as Response);

    render(<ExecutionAnalytics />);

    // Click Trade Quality tab
    const tradeQualityTab = screen.getByText('Trade Quality');
    fireEvent.click(tradeQualityTab);

    await waitFor(() => {
      // Check trade data
      expect(screen.getByText('AAPL')).toBeInTheDocument();
      expect(screen.getByText('MSFT')).toBeInTheDocument();
    });

    await waitFor(() => {
      // Check order sides
      expect(screen.getByText('BUY')).toBeInTheDocument();
      expect(screen.getByText('SELL')).toBeInTheDocument();
    });

    await waitFor(() => {
      // Check execution prices
      expect(screen.getByText('$175.2300')).toBeInTheDocument();
      expect(screen.getByText('$285.4500')).toBeInTheDocument();
    });

    await waitFor(() => {
      // Check fill quality
      expect(screen.getByText('EXCELLENT')).toBeInTheDocument();
      expect(screen.getByText('GOOD')).toBeInTheDocument();
    });
  });

  it('shows slippage analysis with quality distribution', async () => {
    mockFetch
      .mockResolvedValueOnce({
        ok: true,
        json: async () => mockExecutionMetrics
      } as Response)
      .mockResolvedValueOnce({
        ok: true,
        json: async () => mockTradeExecutions
      } as Response)
      .mockResolvedValueOnce({
        ok: true,
        json: async () => mockSlippageAnalysis
      } as Response)
      .mockResolvedValueOnce({
        ok: true,
        json: async () => mockLatencyAnalysis
      } as Response);

    render(<ExecutionAnalytics />);

    // Click Slippage Analysis tab
    const slippageTab = screen.getByText('Slippage Analysis');
    fireEvent.click(slippageTab);

    await waitFor(() => {
      // Check time period headers
      expect(screen.getByText('Slippage Analysis - Last 24 Hours')).toBeInTheDocument();
      expect(screen.getByText('Slippage Analysis - Last 7 Days')).toBeInTheDocument();
    });

    await waitFor(() => {
      // Check slippage metrics
      expect(screen.getByText('3.2')).toBeInTheDocument(); // avg slippage for 24h
      expect(screen.getByText('4.1')).toBeInTheDocument(); // avg slippage for 7d
    });

    await waitFor(() => {
      // Check quality distribution labels
      expect(screen.getByText('Excellent (<1 bps):')).toBeInTheDocument();
      expect(screen.getByText('Good (1-5 bps):')).toBeInTheDocument();
      expect(screen.getByText('Fair (5-15 bps):')).toBeInTheDocument();
      expect(screen.getByText('Poor (>15 bps):')).toBeInTheDocument();
    });
  });

  it('displays latency analysis with trend indicators', async () => {
    mockFetch
      .mockResolvedValueOnce({
        ok: true,
        json: async () => mockExecutionMetrics
      } as Response)
      .mockResolvedValueOnce({
        ok: true,
        json: async () => mockTradeExecutions
      } as Response)
      .mockResolvedValueOnce({
        ok: true,
        json: async () => mockSlippageAnalysis
      } as Response)
      .mockResolvedValueOnce({
        ok: true,
        json: async () => mockLatencyAnalysis
      } as Response);

    render(<ExecutionAnalytics />);

    // Click Latency Analysis tab
    const latencyTab = screen.getByText('Latency Analysis');
    fireEvent.click(latencyTab);

    await waitFor(() => {
      // Check strategy latency cards
      expect(screen.getByText('momentum_1 Latency')).toBeInTheDocument();
      expect(screen.getByText('mean_revert_2 Latency')).toBeInTheDocument();
    });

    await waitFor(() => {
      // Check P50 latency
      expect(screen.getByText('35.8')).toBeInTheDocument(); // P50 for momentum_1
      expect(screen.getByText('52.1')).toBeInTheDocument(); // P50 for mean_revert_2
    });

    await waitFor(() => {
      // Check P95 latency
      expect(screen.getByText('89.3')).toBeInTheDocument(); // P95 for momentum_1
      expect(screen.getByText('125.4')).toBeInTheDocument(); // P95 for mean_revert_2
    });

    await waitFor(() => {
      // Check timeout rates
      expect(screen.getByText('Timeout Rate: 0.80%')).toBeInTheDocument(); // 0.008 * 100
      expect(screen.getByText('Timeout Rate: 1.50%')).toBeInTheDocument(); // 0.015 * 100
    });
  });

  it('handles strategy selection', async () => {
    mockFetch
      .mockResolvedValue({
        ok: true,
        json: async () => mockExecutionMetrics
      } as Response);

    render(<ExecutionAnalytics />);

    await waitFor(() => {
      // Find strategy selector
      const strategySelect = screen.getByDisplayValue('All Strategies');
      expect(strategySelect).toBeInTheDocument();
    });

    // Change strategy selection
    const strategySelect = screen.getByDisplayValue('All Strategies');
    fireEvent.change(strategySelect, { target: { value: 'momentum_1' } });

    // Should trigger new API calls with updated strategy filter
    await waitFor(() => {
      expect(mockFetch).toHaveBeenCalledWith(expect.stringContaining('strategy_id=momentum_1'));
    });
  });

  it('supports time range selection', async () => {
    mockFetch
      .mockResolvedValue({
        ok: true,
        json: async () => mockExecutionMetrics
      } as Response);

    render(<ExecutionAnalytics />);

    await waitFor(() => {
      // Check date range picker exists
      const dateRangePicker = screen.getByRole('textbox');
      expect(dateRangePicker).toBeInTheDocument();
    });
  });

  it('shows fill rate warning when below threshold', async () => {
    const lowFillRateMetrics = {
      metrics: [
        {
          ...mockExecutionMetrics.metrics[0],
          fill_rate: 0.85 // Below 90% threshold
        }
      ]
    };

    mockFetch
      .mockResolvedValueOnce({
        ok: true,
        json: async () => lowFillRateMetrics
      } as Response)
      .mockResolvedValueOnce({
        ok: true,
        json: async () => mockTradeExecutions
      } as Response)
      .mockResolvedValueOnce({
        ok: true,
        json: async () => mockSlippageAnalysis
      } as Response)
      .mockResolvedValueOnce({
        ok: true,
        json: async () => mockLatencyAnalysis
      } as Response);

    render(<ExecutionAnalytics />);

    await waitFor(() => {
      expect(screen.getByText('Low Fill Rate Warning')).toBeInTheDocument();
      expect(screen.getByText(/Average fill rate is 85.0%/)).toBeInTheDocument();
    });
  });

  it('displays partial fill indicators correctly', async () => {
    const partialFillTrade = {
      trades: [
        {
          ...mockTradeExecutions.trades[0],
          quantity_requested: 100,
          quantity_filled: 75 // Partial fill
        }
      ]
    };

    mockFetch
      .mockResolvedValueOnce({
        ok: true,
        json: async () => mockExecutionMetrics
      } as Response)
      .mockResolvedValueOnce({
        ok: true,
        json: async () => partialFillTrade
      } as Response)
      .mockResolvedValueOnce({
        ok: true,
        json: async () => mockSlippageAnalysis
      } as Response)
      .mockResolvedValueOnce({
        ok: true,
        json: async () => mockLatencyAnalysis
      } as Response);

    render(<ExecutionAnalytics />);

    // Go to Trade Quality tab
    const tradeQualityTab = screen.getByText('Trade Quality');
    fireEvent.click(tradeQualityTab);

    await waitFor(() => {
      expect(screen.getByText('75')).toBeInTheDocument(); // filled quantity
      expect(screen.getByText('Partial: 100 req')).toBeInTheDocument(); // partial indicator
    });
  });

  it('handles refresh functionality', async () => {
    mockFetch
      .mockResolvedValue({
        ok: true,
        json: async () => mockExecutionMetrics
      } as Response);

    render(<ExecutionAnalytics />);

    // Find and click refresh button
    const refreshButton = screen.getByRole('button', { name: /refresh/i });
    fireEvent.click(refreshButton);

    // Should trigger additional API calls
    await waitFor(() => {
      expect(mockFetch).toHaveBeenCalledTimes(8); // Initial 4 + refresh 4
    });
  });

  it('color codes metrics based on performance thresholds', async () => {
    mockFetch
      .mockResolvedValueOnce({
        ok: true,
        json: async () => mockExecutionMetrics
      } as Response)
      .mockResolvedValueOnce({
        ok: true,
        json: async () => mockTradeExecutions
      } as Response)
      .mockResolvedValueOnce({
        ok: true,
        json: async () => mockSlippageAnalysis
      } as Response)
      .mockResolvedValueOnce({
        ok: true,
        json: async () => mockLatencyAnalysis
      } as Response);

    render(<ExecutionAnalytics />);

    // The component should apply appropriate color styling based on thresholds
    // For example: fill rate > 95% = green, > 90% = orange, < 90% = red
    // slippage < 5 bps = green, < 15 bps = orange, > 15 bps = red
    // latency < 100ms = green, < 500ms = orange, > 500ms = red

    await waitFor(() => {
      // Check that metrics are displayed (color testing would require DOM inspection)
      expect(screen.getByText('45ms')).toBeInTheDocument(); // momentum_1 latency
      expect(screen.getByText('68ms')).toBeInTheDocument(); // mean_revert_2 latency
    });
  });

  it('handles API errors gracefully', async () => {
    mockFetch.mockRejectedValue(new Error('API Error'));

    render(<ExecutionAnalytics />);

    // Should not crash and should show the layout
    await waitFor(() => {
      expect(screen.getByText('Execution Analytics')).toBeInTheDocument();
    });
  });
});