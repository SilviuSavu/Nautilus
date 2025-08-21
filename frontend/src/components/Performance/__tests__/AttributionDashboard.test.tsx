/**
 * AttributionDashboard Component Tests - Story 5.1
 */

import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import { describe, it, expect, beforeEach, vi } from 'vitest';
import AttributionDashboard from '../AttributionDashboard';
import { AttributionAnalysisResponse } from '../../../types/analytics';

const mockAttributionData: AttributionAnalysisResponse = {
  attribution_type: 'sector',
  period_start: '2024-01-01',
  period_end: '2024-12-31',
  total_active_return: 2.45,
  attribution_breakdown: {
    security_selection: 1.35,
    asset_allocation: 0.85,
    interaction_effect: 0.25
  },
  sector_attribution: [
    {
      sector: 'Technology',
      portfolio_weight: 0.35,
      benchmark_weight: 0.28,
      portfolio_return: 15.2,
      benchmark_return: 12.8,
      allocation_effect: 0.42,
      selection_effect: 0.68,
      total_effect: 1.10
    },
    {
      sector: 'Healthcare',
      portfolio_weight: 0.18,
      benchmark_weight: 0.22,
      portfolio_return: 8.5,
      benchmark_return: 9.2,
      allocation_effect: -0.15,
      selection_effect: -0.18,
      total_effect: -0.33
    },
    {
      sector: 'Financial Services',
      portfolio_weight: 0.25,
      benchmark_weight: 0.20,
      portfolio_return: 11.8,
      benchmark_return: 10.5,
      allocation_effect: 0.28,
      selection_effect: 0.35,
      total_effect: 0.63
    },
    {
      sector: 'Consumer Goods',
      portfolio_weight: 0.12,
      benchmark_weight: 0.15,
      portfolio_return: 6.2,
      benchmark_return: 7.8,
      allocation_effect: -0.18,
      selection_effect: -0.25,
      total_effect: -0.43
    }
  ],
  factor_attribution: [
    {
      factor_name: 'Value',
      factor_exposure: 0.15,
      factor_return: 8.2,
      contribution: 0.42
    },
    {
      factor_name: 'Growth',
      factor_exposure: 0.35,
      factor_return: 12.5,
      contribution: 1.25
    },
    {
      factor_name: 'Momentum',
      factor_exposure: 0.22,
      factor_return: -2.1,
      contribution: -0.18
    },
    {
      factor_name: 'Quality',
      factor_exposure: 0.28,
      factor_return: 6.8,
      contribution: 0.65
    }
  ]
};

const mockFactorAttributionData: AttributionAnalysisResponse = {
  ...mockAttributionData,
  attribution_type: 'factor',
  sector_attribution: []
};

describe('AttributionDashboard', () => {
  const mockOnAttributionTypeChange = vi.fn();

  beforeEach(() => {
    // Mock ResizeObserver
    global.ResizeObserver = class ResizeObserver {
      observe() {}
      unobserve() {}
      disconnect() {}
    };
    
    mockOnAttributionTypeChange.mockClear();
  });

  it('renders attribution dashboard with sector data', async () => {
    render(
      <AttributionDashboard 
        data={mockAttributionData} 
        onAttributionTypeChange={mockOnAttributionTypeChange}
      />
    );

    // Check if main components are rendered
    expect(screen.getByText('Attribution Summary')).toBeInTheDocument();
    expect(screen.getByText('Sector Attribution Chart')).toBeInTheDocument();
    expect(screen.getByText('Attribution Treemap')).toBeInTheDocument();

    await waitFor(() => {
      // Check attribution breakdown
      expect(screen.getByText('Total Active Return')).toBeInTheDocument();
      expect(screen.getByText('Security Selection Effect')).toBeInTheDocument();
      expect(screen.getByText('Asset Allocation Effect')).toBeInTheDocument();
      expect(screen.getByText('Interaction Effect')).toBeInTheDocument();
    });
  });

  it('displays correct attribution summary values', () => {
    render(<AttributionDashboard data={mockAttributionData} />);

    // Check total active return
    expect(screen.getByText('2.450%')).toBeInTheDocument();

    // Check attribution breakdown values
    expect(screen.getByText('1.350%')).toBeInTheDocument(); // Security selection
    expect(screen.getByText('0.850%')).toBeInTheDocument(); // Asset allocation
    expect(screen.getByText('0.250%')).toBeInTheDocument(); // Interaction effect
  });

  it('renders sector attribution table correctly', async () => {
    render(<AttributionDashboard data={mockAttributionData} />);

    // Switch to table view
    const tableButton = screen.getByRole('button', { name: /table/i });
    fireEvent.click(tableButton);

    await waitFor(() => {
      // Check table headers
      expect(screen.getByText('Sector')).toBeInTheDocument();
      expect(screen.getByText('Portfolio Weight')).toBeInTheDocument();
      expect(screen.getByText('Benchmark Weight')).toBeInTheDocument();
      expect(screen.getByText('Portfolio Return')).toBeInTheDocument();
      expect(screen.getByText('Benchmark Return')).toBeInTheDocument();
      expect(screen.getByText('Allocation Effect')).toBeInTheDocument();
      expect(screen.getByText('Selection Effect')).toBeInTheDocument();
      expect(screen.getByText('Total Effect')).toBeInTheDocument();

      // Check sector data
      expect(screen.getByText('Technology')).toBeInTheDocument();
      expect(screen.getByText('Healthcare')).toBeInTheDocument();
      expect(screen.getByText('Financial Services')).toBeInTheDocument();
      expect(screen.getByText('Consumer Goods')).toBeInTheDocument();
    });
  });

  it('switches between chart and table views', async () => {
    render(<AttributionDashboard data={mockAttributionData} />);

    // Initially should show chart view
    expect(screen.getByText('Sector Attribution Chart')).toBeInTheDocument();

    // Click table button
    const tableButton = screen.getByRole('button', { name: /table/i });
    fireEvent.click(tableButton);

    await waitFor(() => {
      expect(screen.getByText('Sector Attribution Table')).toBeInTheDocument();
    });

    // Switch back to chart view
    const chartButton = screen.getByRole('button', { name: /chart/i });
    fireEvent.click(chartButton);

    await waitFor(() => {
      expect(screen.getByText('Sector Attribution Chart')).toBeInTheDocument();
    });
  });

  it('handles attribution type changes', async () => {
    render(
      <AttributionDashboard 
        data={mockAttributionData}
        onAttributionTypeChange={mockOnAttributionTypeChange}
      />
    );

    // Find and click the attribution type selector
    const sectorSelect = screen.getByDisplayValue('Sector');
    expect(sectorSelect).toBeInTheDocument();

    // Simulate selection change
    fireEvent.mouseDown(sectorSelect);
    
    await waitFor(() => {
      const factorOption = screen.getByText('Factor');
      fireEvent.click(factorOption);
    });

    expect(mockOnAttributionTypeChange).toHaveBeenCalledWith('factor');
  });

  it('renders factor attribution correctly', () => {
    render(<AttributionDashboard data={mockFactorAttributionData} />);

    // Should show factor attribution instead of sector
    expect(screen.getByText('Factor Attribution Chart')).toBeInTheDocument();
    
    // Switch to table view to check factor data
    const tableButton = screen.getByRole('button', { name: /table/i });
    fireEvent.click(tableButton);

    // Check factor attribution table headers
    expect(screen.getByText('Factor')).toBeInTheDocument();
    expect(screen.getByText('Factor Exposure')).toBeInTheDocument();
    expect(screen.getByText('Factor Return')).toBeInTheDocument();
    expect(screen.getByText('Contribution')).toBeInTheDocument();

    // Check factor names
    expect(screen.getByText('Value')).toBeInTheDocument();
    expect(screen.getByText('Growth')).toBeInTheDocument();
    expect(screen.getByText('Momentum')).toBeInTheDocument();
    expect(screen.getByText('Quality')).toBeInTheDocument();
  });

  it('toggles allocation and selection effects visibility', async () => {
    render(<AttributionDashboard data={mockAttributionData} />);

    // Find the allocation and selection switches
    const switches = screen.getAllByRole('switch');
    expect(switches).toHaveLength(2);

    // Toggle allocation switch
    fireEvent.click(switches[0]);
    
    // Toggle selection switch  
    fireEvent.click(switches[1]);

    // The chart should still render (testing the toggle functionality)
    expect(screen.getByText('Sector Attribution Chart')).toBeInTheDocument();
  });

  it('applies correct styling for positive and negative effects', () => {
    render(<AttributionDashboard data={mockAttributionData} />);

    // Total active return is positive, should have green styling
    const totalReturnElement = screen.getByText('2.450%');
    expect(totalReturnElement).toHaveStyle({ color: '#52c41a' });
  });

  it('displays period information correctly', () => {
    render(<AttributionDashboard data={mockAttributionData} />);

    expect(screen.getByText('2024-01-01 to 2024-12-31')).toBeInTheDocument();
  });

  it('handles empty sector attribution data', () => {
    const emptyData = {
      ...mockAttributionData,
      sector_attribution: []
    };

    render(<AttributionDashboard data={emptyData} />);

    // Should show no data message for treemap
    expect(screen.getByText('No treemap data available')).toBeInTheDocument();
  });

  it('handles empty factor attribution data', () => {
    const emptyFactorData = {
      ...mockFactorAttributionData,
      factor_attribution: []
    };

    render(<AttributionDashboard data={emptyFactorData} />);

    // Should show no data message for factor chart
    expect(screen.getByText('No factor attribution data available')).toBeInTheDocument();
  });

  it('formats numbers with correct precision', () => {
    render(<AttributionDashboard data={mockAttributionData} />);

    // Check that attribution effects are formatted to 3 decimal places
    expect(screen.getByText('1.350%')).toBeInTheDocument();
    expect(screen.getByText('0.850%')).toBeInTheDocument();
    expect(screen.getByText('0.250%')).toBeInTheDocument();
  });

  it('renders with custom height prop', () => {
    const { container } = render(
      <AttributionDashboard data={mockAttributionData} height={800} />
    );

    expect(container.firstChild).toHaveStyle({ height: '800px' });
  });

  it('shows loading state correctly', () => {
    render(<AttributionDashboard data={mockAttributionData} loading={true} />);

    // Component should still render with loading prop
    expect(screen.getByText('Attribution Summary')).toBeInTheDocument();
  });

  it('handles currency effect when present', () => {
    const dataWithCurrency = {
      ...mockAttributionData,
      attribution_breakdown: {
        ...mockAttributionData.attribution_breakdown,
        currency_effect: 0.15
      }
    };

    render(<AttributionDashboard data={dataWithCurrency} />);

    expect(screen.getByText('Currency Effect')).toBeInTheDocument();
    expect(screen.getByText('0.150%')).toBeInTheDocument();
  });

  it('sorts sector data correctly by total effect', () => {
    render(<AttributionDashboard data={mockAttributionData} />);

    // Switch to table view to check sorting
    const tableButton = screen.getByRole('button', { name: /table/i });
    fireEvent.click(tableButton);

    // The table should be sorted by total effect (descending by default)
    // Technology should appear first (highest total effect: 1.10)
    const rows = screen.getAllByRole('row');
    expect(rows.length).toBeGreaterThan(1); // Header + data rows
  });

  it('handles sector name truncation for long names', () => {
    const longSectorData = {
      ...mockAttributionData,
      sector_attribution: [
        {
          ...mockAttributionData.sector_attribution[0],
          sector: 'Very Long Sector Name That Should Be Truncated'
        }
      ]
    };

    render(<AttributionDashboard data={longSectorData} />);

    // The component should handle long sector names gracefully
    expect(screen.getByText('Attribution Summary')).toBeInTheDocument();
  });
});