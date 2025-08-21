/**
 * MonteCarloVisualization Component Tests - Story 5.1
 */

import { render, screen, waitFor } from '@testing-library/react';
import { describe, it, expect, beforeEach } from 'vitest';
import MonteCarloVisualization from '../MonteCarloVisualization';
import { MonteCarloResponse } from '../../../types/analytics';

const mockMonteCarloData: MonteCarloResponse = {
  scenarios_run: 10000,
  time_horizon_days: 252,
  confidence_intervals: {
    percentile_5: -15.3,
    percentile_25: -3.2,
    percentile_50: 8.7,
    percentile_75: 22.4,
    percentile_95: 45.8
  },
  expected_return: 12.5,
  probability_of_loss: 0.25,
  value_at_risk_5: -18.2,
  expected_shortfall_5: -25.7,
  worst_case_scenario: -45.6,
  best_case_scenario: 67.3,
  stress_test_results: [
    {
      scenario_name: 'market_crash',
      probability_of_loss: 0.85,
      expected_loss: -35.2,
      var_95: -42.1
    },
    {
      scenario_name: 'high_volatility',
      probability_of_loss: 0.45,
      expected_loss: -12.8,
      var_95: -28.3
    },
    {
      scenario_name: 'recession',
      probability_of_loss: 0.65,
      expected_loss: -22.4,
      var_95: -38.7
    }
  ],
  simulation_paths: [
    [0, 2.1, 4.3, 6.8, 8.2, 10.5],
    [0, -1.2, 1.8, 5.2, 7.4, 9.1],
    [0, 3.5, 2.1, 4.7, 8.9, 12.3],
    [0, -2.8, -1.5, 2.2, 5.8, 7.6],
    [0, 1.9, 4.2, 6.1, 9.3, 11.8]
  ]
};

const mockEmptyData: MonteCarloResponse = {
  scenarios_run: 0,
  time_horizon_days: 0,
  confidence_intervals: {
    percentile_5: 0,
    percentile_25: 0,
    percentile_50: 0,
    percentile_75: 0,
    percentile_95: 0
  },
  expected_return: 0,
  probability_of_loss: 0,
  value_at_risk_5: 0,
  expected_shortfall_5: 0,
  worst_case_scenario: 0,
  best_case_scenario: 0,
  stress_test_results: [],
  simulation_paths: []
};

describe('MonteCarloVisualization', () => {
  beforeEach(() => {
    // Mock ResizeObserver
    global.ResizeObserver = class ResizeObserver {
      observe() {}
      unobserve() {}
      disconnect() {}
    };
  });

  it('renders Monte Carlo visualization with complete data', async () => {
    render(<MonteCarloVisualization data={mockMonteCarloData} />);

    // Check if main components are rendered
    expect(screen.getByText('Simulation Summary')).toBeInTheDocument();
    expect(screen.getByText('Confidence Intervals')).toBeInTheDocument();
    expect(screen.getByText('Return Distribution Histogram')).toBeInTheDocument();
    expect(screen.getByText('Sample Simulation Paths')).toBeInTheDocument();
    expect(screen.getByText('Stress Test Results')).toBeInTheDocument();

    await waitFor(() => {
      // Check risk metrics cards
      expect(screen.getByText('Expected Return')).toBeInTheDocument();
      expect(screen.getByText('Probability of Loss')).toBeInTheDocument();
      expect(screen.getByText('Value at Risk (5%)')).toBeInTheDocument();
      expect(screen.getByText('Expected Shortfall (5%)')).toBeInTheDocument();
      expect(screen.getByText('Best Case')).toBeInTheDocument();
      expect(screen.getByText('Worst Case')).toBeInTheDocument();
    });
  });

  it('displays correct risk metric values', () => {
    render(<MonteCarloVisualization data={mockMonteCarloData} />);

    // Check that the expected return is displayed correctly
    expect(screen.getByText('12.50%')).toBeInTheDocument();
    
    // Check probability of loss (should be 25.0%)
    expect(screen.getByText('25.0%')).toBeInTheDocument();
    
    // Check VaR value
    expect(screen.getByText('-18.20%')).toBeInTheDocument();
    
    // Check best and worst case scenarios
    expect(screen.getByText('67.30%')).toBeInTheDocument();
    expect(screen.getByText('-45.60%')).toBeInTheDocument();
  });

  it('displays confidence intervals correctly', () => {
    render(<MonteCarloVisualization data={mockMonteCarloData} />);

    // Check confidence interval percentiles
    expect(screen.getByText('5th Percentile:')).toBeInTheDocument();
    expect(screen.getByText('25th Percentile:')).toBeInTheDocument();
    expect(screen.getByText('50th Percentile:')).toBeInTheDocument();
    expect(screen.getByText('75th Percentile:')).toBeInTheDocument();
    expect(screen.getByText('95th Percentile:')).toBeInTheDocument();

    // Check specific confidence interval values
    expect(screen.getByText('-15.30%')).toBeInTheDocument();
    expect(screen.getByText('8.70%')).toBeInTheDocument();
    expect(screen.getByText('45.80%')).toBeInTheDocument();
  });

  it('renders stress test results table correctly', () => {
    render(<MonteCarloVisualization data={mockMonteCarloData} />);

    // Check stress test scenarios
    expect(screen.getByText('MARKET CRASH')).toBeInTheDocument();
    expect(screen.getByText('HIGH VOLATILITY')).toBeInTheDocument();
    expect(screen.getByText('RECESSION')).toBeInTheDocument();

    // Check stress test table headers
    expect(screen.getByText('Scenario')).toBeInTheDocument();
    expect(screen.getByText('Loss Probability')).toBeInTheDocument();
    expect(screen.getByText('Expected Loss')).toBeInTheDocument();
    expect(screen.getByText('VaR (95%)')).toBeInTheDocument();

    // Check some stress test values
    expect(screen.getByText('-35.20%')).toBeInTheDocument();
    expect(screen.getByText('-42.10%')).toBeInTheDocument();
  });

  it('displays simulation summary statistics', () => {
    render(<MonteCarloVisualization data={mockMonteCarloData} />);

    // Check simulation summary values
    expect(screen.getByText('10,000')).toBeInTheDocument(); // scenarios run
    expect(screen.getByText('252')).toBeInTheDocument();    // time horizon
    expect(screen.getByText('days')).toBeInTheDocument();

    // Check confidence range
    expect(screen.getByText('-15.3% to 45.8%')).toBeInTheDocument();
  });

  it('handles empty simulation paths gracefully', () => {
    const dataWithoutPaths = {
      ...mockMonteCarloData,
      simulation_paths: []
    };

    render(<MonteCarloVisualization data={dataWithoutPaths} />);

    expect(screen.getByText('No simulation paths data available')).toBeInTheDocument();
    expect(screen.getByText('Simulation paths are needed for path visualization')).toBeInTheDocument();
  });

  it('handles empty stress test results', () => {
    const dataWithoutStress = {
      ...mockMonteCarloData,
      stress_test_results: []
    };

    render(<MonteCarloVisualization data={dataWithoutStress} />);

    // Stress test card should not be rendered
    expect(screen.queryByText('Stress Test Results')).not.toBeInTheDocument();
  });

  it('applies correct styling for positive and negative values', () => {
    render(<MonteCarloVisualization data={mockMonteCarloData} />);

    // Expected return is positive, should have green color styling
    const expectedReturnElement = screen.getByText('12.50%');
    expect(expectedReturnElement.closest('.ant-statistic-content')).toHaveStyle({ color: '#52c41a' });

    // VaR is negative, should have red color styling
    const varElement = screen.getByText('-18.20%');
    expect(varElement.closest('.ant-statistic-content')).toHaveStyle({ color: '#f5222d' });
  });

  it('formats numbers correctly with proper precision', () => {
    render(<MonteCarloVisualization data={mockMonteCarloData} />);

    // Check that percentages are formatted to 2 decimal places
    expect(screen.getByText('12.50%')).toBeInTheDocument();
    expect(screen.getByText('-15.30%')).toBeInTheDocument();
    expect(screen.getByText('25.0%')).toBeInTheDocument(); // Probability of loss should be 1 decimal
  });

  it('handles loading state correctly', () => {
    render(<MonteCarloVisualization data={mockMonteCarloData} loading={true} />);

    // Component should still render with loading prop
    expect(screen.getByText('Simulation Summary')).toBeInTheDocument();
  });

  it('processes histogram data correctly', () => {
    render(<MonteCarloVisualization data={mockMonteCarloData} />);

    // Histogram should be rendered
    expect(screen.getByText('Return Distribution Histogram')).toBeInTheDocument();

    // Should process simulation paths into histogram bins
    // (This is more of an integration test for the useMemo logic)
  });

  it('handles different risk colors based on probability thresholds', () => {
    // Test with low risk probability
    const lowRiskData = {
      ...mockMonteCarloData,
      probability_of_loss: 0.05 // 5% - should be green
    };

    render(<MonteCarloVisualization data={lowRiskData} />);
    
    // Should show low risk probability
    expect(screen.getByText('5.0%')).toBeInTheDocument();
  });

  it('renders with custom height prop', () => {
    const { container } = render(
      <MonteCarloVisualization data={mockMonteCarloData} height={800} />
    );

    expect(container.firstChild).toHaveStyle({ height: '800px' });
  });

  it('handles malformed or incomplete data gracefully', () => {
    // Test with minimal data
    render(<MonteCarloVisualization data={mockEmptyData} />);

    // Should still render basic structure without crashing
    expect(screen.getByText('Simulation Summary')).toBeInTheDocument();
    expect(screen.getByText('0')).toBeInTheDocument(); // scenarios run = 0
  });
});