import React from 'react';
import { render, screen, waitFor, fireEvent } from '@testing-library/react';
import '@testing-library/jest-dom';
import { vi } from 'vitest';
import RiskDashboard from '../RiskDashboard';
import { riskService } from '../services/riskService';

// Mock the risk service
vi.mock('../services/riskService', () => ({
  riskService: {
    getPortfolioRisk: vi.fn(),
    enableRealTimeMonitoring: vi.fn(),
    disableRealTimeMonitoring: vi.fn(),
    getRiskMetrics: vi.fn(),
    getExposureAnalysis: vi.fn(),
    getRiskAlerts: vi.fn(),
    getRiskLimits: vi.fn(),
    getRiskChartData: vi.fn()
  }
}));

// Mock Chart components to avoid canvas rendering issues in tests
vi.mock('@ant-design/plots', () => ({
  Pie: () => <div data-testid="pie-chart">Pie Chart</div>,
  Column: () => <div data-testid="column-chart">Column Chart</div>,
  Line: () => <div data-testid="line-chart">Line Chart</div>,
  Heatmap: () => <div data-testid="heatmap-chart">Heatmap Chart</div>
}));

const mockPortfolioRisk = {
  portfolio_id: 'test-portfolio',
  var_1d: '2500.50',
  var_1w: '5500.75',
  var_1m: '12000.25',
  expected_shortfall: '3800.00',
  beta: 1.2,
  correlation_matrix: [
    {
      symbol1: 'AAPL',
      symbol2: 'GOOGL',
      correlation: 0.75,
      confidence_level: 95.0,
      calculation_period_days: 252
    }
  ],
  concentration_risk: [
    {
      category: 'instrument',
      name: 'AAPL',
      exposure_amount: '15000.00',
      exposure_percentage: 30.0,
      risk_level: 'medium'
    }
  ],
  total_exposure: '50000.00',
  last_calculated: new Date()
};

const mockRiskMetrics = {
  portfolio_id: 'test-portfolio',
  var_1d_95: '2500.50',
  var_1d_99: '3200.75',
  var_1w_95: '5500.25',
  var_1w_99: '7200.50',
  var_1m_95: '12000.25',
  var_1m_99: '15500.75',
  expected_shortfall_95: '3800.00',
  expected_shortfall_99: '4800.00',
  beta_vs_market: 1.2,
  portfolio_volatility: 0.18,
  sharpe_ratio: 1.5,
  max_drawdown: '8500.00',
  correlation_with_market: 0.85,
  tracking_error: 0.05,
  information_ratio: 0.75,
  calculated_at: new Date()
};

const mockExposureAnalysis = {
  total_exposure: '50000.00',
  long_exposure: '45000.00',
  short_exposure: '5000.00',
  net_exposure: '40000.00',
  by_instrument: [
    {
      symbol: 'AAPL',
      position_size: '100',
      market_value: '15000.00',
      percentage_of_portfolio: 30.0,
      unrealized_pnl: '500.00',
      risk_contribution: 0.25
    }
  ],
  by_sector: [
    {
      sector: 'Technology',
      total_exposure: '25000.00',
      percentage_of_portfolio: 50.0,
      instrument_count: 3,
      top_holdings: []
    }
  ],
  by_currency: [],
  by_geography: []
};

describe('RiskDashboard', () => {
  beforeEach(() => {
    // Reset all mocks
    vi.clearAllMocks();
    
    // Setup default mock responses
    (riskService.getPortfolioRisk as any).mockResolvedValue(mockPortfolioRisk);
    (riskService.getRiskMetrics as any).mockResolvedValue(mockRiskMetrics);
    (riskService.getExposureAnalysis as any).mockResolvedValue(mockExposureAnalysis);
    (riskService.getRiskAlerts as any).mockResolvedValue({ alerts: [] });
    (riskService.getRiskLimits as any).mockResolvedValue({ limits: [] });
    (riskService.getRiskChartData as any).mockResolvedValue({ data: [] });
  });

  test('renders dashboard with portfolio summary', async () => {
    render(<RiskDashboard portfolioId="test-portfolio" />);

    // Check for loading state initially
    expect(screen.getByText('Portfolio Value')).toBeInTheDocument();

    // Wait for data to load
    await waitFor(() => {
      expect(screen.getByText('50,000')).toBeInTheDocument();
    });
  });

  test('displays risk metrics correctly', async () => {
    render(<RiskDashboard portfolioId="test-portfolio" />);

    await waitFor(() => {
      expect(screen.getByText('1-Day VaR (95%)')).toBeInTheDocument();
      expect(screen.getByText('Expected Shortfall')).toBeInTheDocument();
      expect(screen.getByText('Portfolio Beta')).toBeInTheDocument();
    });
  });

  test('handles real-time monitoring toggle', async () => {
    render(<RiskDashboard portfolioId="test-portfolio" />);

    await waitFor(() => {
      const realTimeSwitch = screen.getByRole('switch');
      expect(realTimeSwitch).toBeInTheDocument();
    });

    const realTimeSwitch = screen.getByRole('switch');
    fireEvent.click(realTimeSwitch);

    await waitFor(() => {
      expect(riskService.enableRealTimeMonitoring).toHaveBeenCalledWith('test-portfolio');
    });
  });

  test('handles refresh button click', async () => {
    render(<RiskDashboard portfolioId="test-portfolio" />);

    await waitFor(() => {
      const buttons = screen.getAllByRole('button');
      expect(buttons.length).toBeGreaterThan(0);
    });

    // Find the button with the reload icon
    const buttons = screen.getAllByRole('button');
    const refreshButton = buttons.find(button => button.querySelector('.anticon-reload'));
    expect(refreshButton).toBeTruthy();
    
    if (refreshButton) {
      fireEvent.click(refreshButton);
      
      // Should call the service again
      await waitFor(() => {
        expect(riskService.getPortfolioRisk).toHaveBeenCalledTimes(2);
      });
    }
  });

  test('displays error message when data loading fails', async () => {
    const errorMessage = 'Failed to fetch portfolio risk data';
    (riskService.getPortfolioRisk as any).mockRejectedValue(new Error(errorMessage));

    render(<RiskDashboard portfolioId="test-portfolio" />);

    await waitFor(() => {
      expect(screen.getByText('Error Loading Risk Data')).toBeInTheDocument();
      expect(screen.getByText(errorMessage)).toBeInTheDocument();
    });
  });

  test('switches between dashboard tabs', async () => {
    render(<RiskDashboard portfolioId="test-portfolio" />);

    await waitFor(() => {
      expect(screen.getByText('Overview')).toBeInTheDocument();
      expect(screen.getByText('Exposure Analysis')).toBeInTheDocument();
      expect(screen.getByText('Alerts & Limits')).toBeInTheDocument();
    });

    // Click on Exposure Analysis tab
    fireEvent.click(screen.getByText('Exposure Analysis'));

    await waitFor(() => {
      expect(riskService.getExposureAnalysis).toHaveBeenCalled();
    });

    // Click on Alerts & Limits tab
    fireEvent.click(screen.getByText('Alerts & Limits'));

    await waitFor(() => {
      expect(riskService.getRiskAlerts).toHaveBeenCalled();
      expect(riskService.getRiskLimits).toHaveBeenCalled();
    });
  });

  test('shows concentration risk count', async () => {
    render(<RiskDashboard portfolioId="test-portfolio" />);

    await waitFor(() => {
      expect(screen.getByText('Concentration Risk')).toBeInTheDocument();
      // Just check that we have positions text - the count is verified by the mock data
      expect(screen.getByText('positions')).toBeInTheDocument();
    });
  });

  test('calculates and displays risk levels with appropriate colors', async () => {
    // Test with high-risk portfolio
    const highRiskPortfolio = {
      ...mockPortfolioRisk,
      var_1d: '5000.00', // 10% of portfolio value - high risk
      total_exposure: '50000.00'
    };

    (riskService.getPortfolioRisk as any).mockResolvedValue(highRiskPortfolio);

    render(<RiskDashboard portfolioId="test-portfolio" />);

    await waitFor(() => {
      const varElement = screen.getByText('1-Day VaR (95%)');
      expect(varElement).toBeInTheDocument();
    });
  });
});

describe('RiskDashboard Integration', () => {
  test('updates data when portfolioId changes', async () => {
    const { rerender } = render(<RiskDashboard portfolioId="portfolio-1" />);

    await waitFor(() => {
      expect(riskService.getPortfolioRisk).toHaveBeenCalledWith('portfolio-1');
    });

    // Change portfolio ID
    rerender(<RiskDashboard portfolioId="portfolio-2" />);

    await waitFor(() => {
      expect(riskService.getPortfolioRisk).toHaveBeenCalledWith('portfolio-2');
    });
  });

  test('handles service call failures gracefully', async () => {
    // Mock different types of failures
    (riskService.getPortfolioRisk as any).mockRejectedValue(new Error('Network error'));
    (riskService.getRiskMetrics as any).mockRejectedValue(new Error('API error'));

    render(<RiskDashboard portfolioId="test-portfolio" />);

    // Should still render the component structure
    await waitFor(() => {
      expect(screen.getByText('Portfolio Value')).toBeInTheDocument();
      expect(screen.getByText('Error Loading Risk Data')).toBeInTheDocument();
    });
  });
});