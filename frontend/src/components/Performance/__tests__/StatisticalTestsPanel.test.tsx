/**
 * StatisticalTestsPanel Component Tests - Story 5.1
 */

import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import { describe, it, expect, beforeEach, vi } from 'vitest';
import StatisticalTestsPanel from '../StatisticalTestsPanel';
import { StatisticalTestsResponse } from '../../../types/analytics';

const mockStatisticalTestsData: StatisticalTestsResponse = {
  sharpe_ratio_test: {
    sharpe_ratio: 1.25,
    t_statistic: 2.84,
    p_value: 0.0045,
    is_significant: true,
    confidence_interval: [0.82, 1.68]
  },
  alpha_significance_test: {
    alpha: 0.035,
    t_statistic: 2.12,
    p_value: 0.034,
    is_significant: true,
    confidence_interval: [0.008, 0.062]
  },
  beta_stability_test: {
    beta: 1.15,
    rolling_beta_std: 0.18,
    stability_score: 0.75,
    regime_changes_detected: 2
  },
  performance_persistence: {
    persistence_score: 0.68,
    consecutive_winning_periods: 8,
    consistency_rating: 'Medium'
  },
  bootstrap_results: [
    {
      metric: 'Sharpe Ratio',
      bootstrap_mean: 1.23,
      bootstrap_std: 0.15,
      confidence_interval_95: [0.93, 1.53]
    },
    {
      metric: 'Alpha',
      bootstrap_mean: 0.034,
      bootstrap_std: 0.012,
      confidence_interval_95: [0.010, 0.058]
    },
    {
      metric: 'Beta',
      bootstrap_mean: 1.16,
      bootstrap_std: 0.08,
      confidence_interval_95: [1.00, 1.32]
    }
  ]
};

const mockNonSignificantData: StatisticalTestsResponse = {
  ...mockStatisticalTestsData,
  sharpe_ratio_test: {
    ...mockStatisticalTestsData.sharpe_ratio_test,
    p_value: 0.125,
    is_significant: false
  },
  alpha_significance_test: {
    ...mockStatisticalTestsData.alpha_significance_test,
    p_value: 0.089,
    is_significant: false
  }
};

describe('StatisticalTestsPanel', () => {
  const mockOnRunTests = vi.fn();

  beforeEach(() => {
    // Mock ResizeObserver
    global.ResizeObserver = class ResizeObserver {
      observe() {}
      unobserve() {}
      disconnect() {}
    };
    
    mockOnRunTests.mockClear();
  });

  it('renders statistical tests panel with all components', async () => {
    render(
      <StatisticalTestsPanel 
        data={mockStatisticalTestsData}
        onRunTests={mockOnRunTests}
      />
    );

    // Check if main components are rendered
    expect(screen.getByText('Test Configuration')).toBeInTheDocument();
    expect(screen.getByText('Sharpe Ratio Significance Test')).toBeInTheDocument();
    expect(screen.getByText('Alpha Significance Test')).toBeInTheDocument();
    expect(screen.getByText('Beta Stability Analysis')).toBeInTheDocument();
    expect(screen.getByText('Performance Persistence Analysis')).toBeInTheDocument();
    expect(screen.getByText('Bootstrap Confidence Intervals')).toBeInTheDocument();

    await waitFor(() => {
      // Check test control elements
      expect(screen.getByText('Test Type:')).toBeInTheDocument();
      expect(screen.getByText('Significance Level:')).toBeInTheDocument();
      expect(screen.getByText('Run Tests')).toBeInTheDocument();
    });
  });

  it('displays Sharpe ratio test results correctly', () => {
    render(<StatisticalTestsPanel data={mockStatisticalTestsData} />);

    // Check Sharpe ratio value
    expect(screen.getByText('1.2500')).toBeInTheDocument();

    // Check t-statistic
    expect(screen.getByText('2.840')).toBeInTheDocument();

    // Check p-value
    expect(screen.getByText('0.0045')).toBeInTheDocument();

    // Check significance status
    expect(screen.getByText('Significant')).toBeInTheDocument();

    // Check confidence interval
    expect(screen.getByText('[0.8200, 1.6800]')).toBeInTheDocument();
  });

  it('displays alpha significance test results correctly', () => {
    render(<StatisticalTestsPanel data={mockStatisticalTestsData} />);

    // Check alpha value (should be displayed as percentage)
    expect(screen.getByText('3.5000%')).toBeInTheDocument();

    // Check significance status for alpha
    expect(screen.getByText('Significant Alpha')).toBeInTheDocument();

    // Check alpha confidence interval
    expect(screen.getByText('[0.8000%, 6.2000%]')).toBeInTheDocument();
  });

  it('displays beta stability analysis correctly', () => {
    render(<StatisticalTestsPanel data={mockStatisticalTestsData} />);

    // Check beta value
    expect(screen.getByText('1.150')).toBeInTheDocument();

    // Check rolling beta standard deviation
    expect(screen.getByText('0.1800')).toBeInTheDocument();

    // Check regime changes
    expect(screen.getByText('2')).toBeInTheDocument();

    // Check stability score display
    expect(screen.getByText('0.750')).toBeInTheDocument();
  });

  it('displays performance persistence analysis correctly', () => {
    render(<StatisticalTestsPanel data={mockStatisticalTestsData} />);

    // Check persistence score
    expect(screen.getByText('0.680')).toBeInTheDocument();

    // Check consecutive winning periods
    expect(screen.getByText('8')).toBeInTheDocument();

    // Check consistency rating
    expect(screen.getByText('Medium')).toBeInTheDocument();
    expect(screen.getByText('Medium Consistency')).toBeInTheDocument();
  });

  it('renders bootstrap results table correctly', () => {
    render(<StatisticalTestsPanel data={mockStatisticalTestsData} />);

    // Check table headers
    expect(screen.getByText('Metric')).toBeInTheDocument();
    expect(screen.getByText('Bootstrap Mean')).toBeInTheDocument();
    expect(screen.getByText('Bootstrap Std Dev')).toBeInTheDocument();
    expect(screen.getByText('95% Confidence Interval')).toBeInTheDocument();

    // Check bootstrap data
    expect(screen.getByText('Sharpe Ratio')).toBeInTheDocument();
    expect(screen.getByText('Alpha')).toBeInTheDocument();
    expect(screen.getByText('Beta')).toBeInTheDocument();

    // Check bootstrap values
    expect(screen.getByText('1.2300')).toBeInTheDocument();
    expect(screen.getByText('0.0340')).toBeInTheDocument();
    expect(screen.getByText('[0.9300, 1.5300]')).toBeInTheDocument();
  });

  it('handles test configuration controls correctly', async () => {
    render(
      <StatisticalTestsPanel 
        data={mockStatisticalTestsData}
        onRunTests={mockOnRunTests}
      />
    );

    // Test type selector should be present
    const testTypeSelect = screen.getByDisplayValue('Sharpe Ratio');
    expect(testTypeSelect).toBeInTheDocument();

    // Significance level input should be present
    const significanceInput = screen.getByDisplayValue('0.050');
    expect(significanceInput).toBeInTheDocument();

    // Run tests button should be present
    const runTestsButton = screen.getByRole('button', { name: /run tests/i });
    expect(runTestsButton).toBeInTheDocument();

    // Click run tests button
    fireEvent.click(runTestsButton);

    expect(mockOnRunTests).toHaveBeenCalledWith('sharpe', 0.05);
  });

  it('changes test type selection correctly', async () => {
    render(
      <StatisticalTestsPanel 
        data={mockStatisticalTestsData}
        onRunTests={mockOnRunTests}
      />
    );

    // Find test type selector
    const testTypeSelect = screen.getByDisplayValue('Sharpe Ratio');
    
    // Open dropdown and select Alpha
    fireEvent.mouseDown(testTypeSelect);
    
    await waitFor(() => {
      const alphaOption = screen.getByText('Alpha');
      fireEvent.click(alphaOption);
    });

    // Run tests should be called with new selection when button clicked
    const runTestsButton = screen.getByRole('button', { name: /run tests/i });
    fireEvent.click(runTestsButton);

    expect(mockOnRunTests).toHaveBeenCalledWith('alpha', 0.05);
  });

  it('handles significance level changes', async () => {
    render(
      <StatisticalTestsPanel 
        data={mockStatisticalTestsData}
        onRunTests={mockOnRunTests}
      />
    );

    // Find significance level input
    const significanceInput = screen.getByDisplayValue('0.050');

    // Change significance level
    fireEvent.change(significanceInput, { target: { value: '0.01' } });

    // Run tests button
    const runTestsButton = screen.getByRole('button', { name: /run tests/i });
    fireEvent.click(runTestsButton);

    expect(mockOnRunTests).toHaveBeenCalledWith('sharpe', 0.01);
  });

  it('toggles confidence intervals display', async () => {
    render(<StatisticalTestsPanel data={mockStatisticalTestsData} />);

    // Find the CI toggle switch
    const ciToggle = screen.getByRole('switch');
    expect(ciToggle).toBeChecked();

    // Bootstrap table should be visible
    expect(screen.getByText('Bootstrap Confidence Intervals')).toBeInTheDocument();

    // Toggle off
    fireEvent.click(ciToggle);

    // Bootstrap table should be hidden (component behavior)
    // Note: This depends on implementation - the table might still be rendered but hidden
  });

  it('applies correct colors for significant vs non-significant results', () => {
    render(<StatisticalTestsPanel data={mockStatisticalTestsData} />);

    // Significant results should have green badges
    const significantBadges = screen.getAllByText('Significant');
    expect(significantBadges.length).toBeGreaterThan(0);

    // P-values should be color-coded based on significance
    // This tests the styling logic
    const pValueElement = screen.getByText('0.0045');
    expect(pValueElement).toHaveStyle({ color: '#52c41a' });
  });

  it('handles non-significant results correctly', () => {
    render(<StatisticalTestsPanel data={mockNonSignificantData} />);

    // Should show non-significant status
    expect(screen.getByText('Not Significant')).toBeInTheDocument();
    expect(screen.getByText('No Significant Alpha')).toBeInTheDocument();

    // P-values should be styled differently for non-significant results
    const pValueElement = screen.getByText('0.125');
    expect(pValueElement).toHaveStyle({ color: '#f5222d' });
  });

  it('handles loading state correctly', () => {
    render(<StatisticalTestsPanel data={mockStatisticalTestsData} loading={true} />);

    // Run tests button should show loading state
    const runTestsButton = screen.getByRole('button', { name: /run tests/i });
    expect(runTestsButton).toHaveClass('ant-btn-loading');
  });

  it('displays correct interpretation messages', () => {
    render(<StatisticalTestsPanel data={mockStatisticalTestsData} />);

    // Should show interpretation for significant Sharpe ratio
    expect(screen.getByText(/genuine risk-adjusted outperformance/)).toBeInTheDocument();

    // Should show interpretation for beta stability
    expect(screen.getByText(/consistent market sensitivity/)).toBeInTheDocument();

    // Should show interpretation for performance persistence
    expect(screen.getByText(/generally consistent results/)).toBeInTheDocument();
  });

  it('formats significance levels with asterisks correctly', () => {
    const highlySignificantData = {
      ...mockStatisticalTestsData,
      sharpe_ratio_test: {
        ...mockStatisticalTestsData.sharpe_ratio_test,
        p_value: 0.0005
      }
    };

    render(<StatisticalTestsPanel data={highlySignificantData} />);

    // Should display significance level markers (this depends on implementation)
    expect(screen.getByText('Significant')).toBeInTheDocument();
  });

  it('handles different consistency ratings correctly', () => {
    const highConsistencyData = {
      ...mockStatisticalTestsData,
      performance_persistence: {
        ...mockStatisticalTestsData.performance_persistence,
        consistency_rating: 'High' as const
      }
    };

    render(<StatisticalTestsPanel data={highConsistencyData} />);

    expect(screen.getByText('High')).toBeInTheDocument();
    expect(screen.getByText('High Consistency')).toBeInTheDocument();

    // Should show positive interpretation message
    expect(screen.getByText(/Strong performance persistence/)).toBeInTheDocument();
  });

  it('renders with custom height prop', () => {
    const { container } = render(
      <StatisticalTestsPanel data={mockStatisticalTestsData} height={800} />
    );

    expect(container.firstChild).toHaveStyle({ height: '800px' });
  });

  it('handles empty bootstrap results gracefully', () => {
    const dataWithoutBootstrap = {
      ...mockStatisticalTestsData,
      bootstrap_results: []
    };

    render(<StatisticalTestsPanel data={dataWithoutBootstrap} />);

    // Bootstrap section should not be rendered or should show empty state
    // This depends on implementation - component might hide the section
    expect(screen.getByText('Test Configuration')).toBeInTheDocument();
  });
});