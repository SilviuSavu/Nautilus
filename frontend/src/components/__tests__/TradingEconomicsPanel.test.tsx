import { render, screen, waitFor, fireEvent } from '@testing-library/react';
import { vi, describe, it, expect, beforeEach, afterEach } from 'vitest';
import TradingEconomicsPanel from '../DataSources/TradingEconomicsPanel';

// Mock fetch globally
const mockFetch = vi.fn();
global.fetch = mockFetch;

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

// Mock Ant Design notification
vi.mock('antd', async () => {
  const actual = await vi.importActual('antd');
  return {
    ...actual,
    notification: {
      success: vi.fn(),
      error: vi.fn(),
      warning: vi.fn(),
      info: vi.fn(),
    },
  };
});

describe('TradingEconomicsPanel', () => {
  const mockProps = {
    enabled: true,
    onToggle: vi.fn(),
  };

  const mockHealthResponse = {
    service: 'trading_economics',
    status: 'healthy',
    package_available: true,
    using_guest_access: false,
    last_check: '2023-10-26T10:00:00Z',
    rate_limit_status: {
      requests_made: 10,
      requests_limit: 500,
      window_seconds: 60,
      time_until_reset: 50
    }
  };

  const mockCountriesResponse = {
    success: true,
    count: 5,
    countries: [
      { Country: 'United States', CountryGroup: 'North America' },
      { Country: 'United Kingdom', CountryGroup: 'Europe' },
      { Country: 'Germany', CountryGroup: 'Europe' },
      { Country: 'Japan', CountryGroup: 'Asia' },
      { Country: 'China', CountryGroup: 'Asia' }
    ]
  };

  const mockMajorIndicators = {
    gdp: [
      {
        Country: 'United States',
        Category: 'GDP',
        Title: 'GDP Growth Rate',
        LatestValue: 2.1,
        PreviousValue: 2.4,
        Forecast: 2.0,
        Unit: 'Percent',
        Frequency: 'Quarterly'
      }
    ],
    inflation: [
      {
        Country: 'United States',
        Category: 'Inflation',
        Title: 'Consumer Price Index',
        LatestValue: 3.2,
        PreviousValue: 3.4,
        Forecast: 3.1,
        Unit: 'Percent',
        Frequency: 'Monthly'
      }
    ],
    unemployment: [],
    interest_rates: []
  };

  const mockMarketOverview = {
    currencies: [
      {
        Symbol: 'USD/EUR',
        Name: 'US Dollar vs Euro',
        Country: 'United States',
        Group: 'currencies',
        Last: 0.85,
        DailyChange: -0.01,
        DailyPercentualChange: -1.2
      }
    ],
    commodities: [],
    stocks: [],
    bonds: []
  };

  const mockStatistics = {
    total_requests: 100,
    cache_size: 5,
    rate_limit_status: {
      requests_made: 10,
      requests_limit: 500,
      window_seconds: 60,
      time_until_reset: 50
    },
    package_available: true,
    using_guest_access: false,
    last_reset: '2023-10-26T09:00:00Z'
  };

  beforeEach(() => {
    mockFetch.mockClear();
    mockProps.onToggle.mockClear();
    
    // Set up default environment variable
    import.meta.env.VITE_API_BASE_URL = 'http://localhost:8001';
  });

  afterEach(() => {
    vi.clearAllMocks();
  });

  describe('Disabled State', () => {
    it('should render enable button when disabled', () => {
      render(<TradingEconomicsPanel enabled={false} onToggle={mockProps.onToggle} />);
      
      expect(screen.getByText('Trading Economics')).toBeInTheDocument();
      expect(screen.getByText(/300,000\+ economic indicators/)).toBeInTheDocument();
      expect(screen.getByRole('button', { name: /Enable Trading Economics/i })).toBeInTheDocument();
    });

    it('should call onToggle when enable button is clicked', () => {
      render(<TradingEconomicsPanel enabled={false} onToggle={mockProps.onToggle} />);
      
      const enableButton = screen.getByRole('button', { name: /Enable Trading Economics/i });
      fireEvent.click(enableButton);
      
      expect(mockProps.onToggle).toHaveBeenCalledWith(true);
    });
  });

  describe('Enabled State - Loading', () => {
    it('should show loading spinner when enabled', async () => {
      // Mock pending API calls
      mockFetch.mockImplementation(() => new Promise(() => {}));

      render(<TradingEconomicsPanel enabled={true} onToggle={mockProps.onToggle} />);

      expect(screen.getByText('Loading Trading Economics data...')).toBeInTheDocument();
      expect(screen.getByRole('img', { name: 'loading' })).toBeInTheDocument();
    });
  });

  describe('Enabled State - Healthy', () => {
    beforeEach(() => {
      // Mock all successful API responses
      mockFetch
        .mockResolvedValueOnce({
          ok: true,
          json: () => Promise.resolve(mockHealthResponse)
        })
        .mockResolvedValueOnce({
          ok: true,
          json: () => Promise.resolve(mockCountriesResponse)
        })
        .mockResolvedValueOnce({
          ok: true,
          json: () => Promise.resolve({ success: true, statistics: mockStatistics })
        })
        .mockResolvedValueOnce({
          ok: true,
          json: () => Promise.resolve(mockMajorIndicators.gdp)
        })
        .mockResolvedValueOnce({
          ok: true,
          json: () => Promise.resolve(mockMajorIndicators.inflation)
        })
        .mockResolvedValueOnce({
          ok: true,
          json: () => Promise.resolve(mockMajorIndicators.unemployment)
        })
        .mockResolvedValueOnce({
          ok: true,
          json: () => Promise.resolve(mockMajorIndicators.interest_rates)
        })
        .mockResolvedValueOnce({
          ok: true,
          json: () => Promise.resolve(mockMarketOverview.currencies)
        })
        .mockResolvedValueOnce({
          ok: true,
          json: () => Promise.resolve(mockMarketOverview.commodities)
        })
        .mockResolvedValueOnce({
          ok: true,
          json: () => Promise.resolve(mockMarketOverview.stocks)
        })
        .mockResolvedValueOnce({
          ok: true,
          json: () => Promise.resolve(mockMarketOverview.bonds)
        });
    });

    it('should display header with connection status', async () => {
      render(<TradingEconomicsPanel enabled={true} onToggle={mockProps.onToggle} />);

      await waitFor(() => {
        expect(screen.getByText('Trading Economics')).toBeInTheDocument();
        expect(screen.getByText('Connected')).toBeInTheDocument();
      });
    });

    it('should display statistics overview', async () => {
      render(<TradingEconomicsPanel enabled={true} onToggle={mockProps.onToggle} />);

      await waitFor(() => {
        expect(screen.getByText('Countries')).toBeInTheDocument();
        expect(screen.getByText('196')).toBeInTheDocument();
        expect(screen.getByText('Indicators')).toBeInTheDocument();
        expect(screen.getByText('300K+')).toBeInTheDocument();
        expect(screen.getByText('Markets')).toBeInTheDocument();
        expect(screen.getByText('All Major')).toBeInTheDocument();
        expect(screen.getByText('Rate Limit')).toBeInTheDocument();
      });
    });

    it('should display major countries', async () => {
      render(<TradingEconomicsPanel enabled={true} onToggle={mockProps.onToggle} />);

      await waitFor(() => {
        expect(screen.getByText('Major Countries')).toBeInTheDocument();
        expect(screen.getByText(/🇺🇸 United States/)).toBeInTheDocument();
        expect(screen.getByText(/🇬🇧 United Kingdom/)).toBeInTheDocument();
        expect(screen.getByText(/🇩🇪 Germany/)).toBeInTheDocument();
        expect(screen.getByText(/🇯🇵 Japan/)).toBeInTheDocument();
        expect(screen.getByText(/🇨🇳 China/)).toBeInTheDocument();
      });
    });

    it('should display market categories', async () => {
      render(<TradingEconomicsPanel enabled={true} onToggle={mockProps.onToggle} />);

      await waitFor(() => {
        expect(screen.getByText('Market Categories')).toBeInTheDocument();
        expect(screen.getByText(/💱 Currencies/)).toBeInTheDocument();
        expect(screen.getByText(/🥇 Commodities/)).toBeInTheDocument();
        expect(screen.getByText(/📊 Stocks/)).toBeInTheDocument();
        expect(screen.getByText(/📋 Bonds/)).toBeInTheDocument();
      });
    });

    it('should display economic categories with default selection', async () => {
      render(<TradingEconomicsPanel enabled={true} onToggle={mockProps.onToggle} />);

      await waitFor(() => {
        expect(screen.getByText('Economic Categories:')).toBeInTheDocument();
        expect(screen.getByText('📊 GDP')).toBeInTheDocument();
        expect(screen.getByText('📈 Inflation')).toBeInTheDocument();
        expect(screen.getByText('👥 Employment')).toBeInTheDocument();
        expect(screen.getByText('🏦 Interest Rates')).toBeInTheDocument();
        expect(screen.getByText('🌍 Trade')).toBeInTheDocument();
      });
    });

    it('should handle category selection', async () => {
      render(<TradingEconomicsPanel enabled={true} onToggle={mockProps.onToggle} />);

      await waitFor(() => {
        const inflationTag = screen.getByText('📈 Inflation');
        fireEvent.click(inflationTag);
        expect(screen.getByText('Consumer price indices and inflation rates')).toBeInTheDocument();
      });
    });

    it('should display usage instructions', async () => {
      render(<TradingEconomicsPanel enabled={true} onToggle={mockProps.onToggle} />);

      await waitFor(() => {
        expect(screen.getByText('Trading Economics Integration Active')).toBeInTheDocument();
        expect(screen.getByText(/Trading Economics provides 300,000\+/)).toBeInTheDocument();
        expect(screen.getByText('/api/v1/trading-economics/indicators')).toBeInTheDocument();
        expect(screen.getByText('/api/v1/trading-economics/calendar')).toBeInTheDocument();
        expect(screen.getByText('/api/v1/trading-economics/markets')).toBeInTheDocument();
        expect(screen.getByText('/api/v1/trading-economics/search')).toBeInTheDocument();
      });
    });
  });

  describe('Enabled State - Error Handling', () => {
    it('should handle API error gracefully', async () => {
      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: () => Promise.resolve({
          service: 'trading_economics',
          status: 'error',
          package_available: false,
          using_guest_access: true,
          last_check: '2023-10-26T10:00:00Z',
          error: 'API connection failed'
        })
      });

      render(<TradingEconomicsPanel enabled={true} onToggle={mockProps.onToggle} />);

      await waitFor(() => {
        expect(screen.getByText('Trading Economics API Unavailable')).toBeInTheDocument();
        expect(screen.getByText(/API connection failed/)).toBeInTheDocument();
      });
    });

    it('should handle guest access mode', async () => {
      const guestHealthResponse = {
        ...mockHealthResponse,
        using_guest_access: true
      };

      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: () => Promise.resolve(guestHealthResponse)
      });

      render(<TradingEconomicsPanel enabled={true} onToggle={mockProps.onToggle} />);

      await waitFor(() => {
        expect(screen.getByText('Using Guest Access')).toBeInTheDocument();
        expect(screen.getByText(/Consider upgrading to a paid plan/)).toBeInTheDocument();
      });
    });

    it('should handle mock mode status', async () => {
      const mockModeResponse = {
        ...mockHealthResponse,
        status: 'mock_mode',
        package_available: false
      };

      mockFetch
        .mockResolvedValueOnce({
          ok: true,
          json: () => Promise.resolve(mockModeResponse)
        })
        .mockResolvedValueOnce({
          ok: true,
          json: () => Promise.resolve(mockCountriesResponse)
        })
        .mockResolvedValueOnce({
          ok: true,
          json: () => Promise.resolve({ success: true, statistics: mockStatistics })
        })
        .mockResolvedValueOnce({
          ok: true,
          json: () => Promise.resolve(mockMajorIndicators.gdp)
        })
        .mockResolvedValueOnce({
          ok: true,
          json: () => Promise.resolve(mockMajorIndicators.inflation)
        })
        .mockResolvedValueOnce({
          ok: true,
          json: () => Promise.resolve(mockMajorIndicators.unemployment)
        })
        .mockResolvedValueOnce({
          ok: true,
          json: () => Promise.resolve(mockMajorIndicators.interest_rates)
        })
        .mockResolvedValueOnce({
          ok: true,
          json: () => Promise.resolve(mockMarketOverview.currencies)
        })
        .mockResolvedValueOnce({
          ok: true,
          json: () => Promise.resolve(mockMarketOverview.commodities)
        })
        .mockResolvedValueOnce({
          ok: true,
          json: () => Promise.resolve(mockMarketOverview.stocks)
        })
        .mockResolvedValueOnce({
          ok: true,
          json: () => Promise.resolve(mockMarketOverview.bonds)
        });

      render(<TradingEconomicsPanel enabled={true} onToggle={mockProps.onToggle} />);

      await waitFor(() => {
        expect(screen.getByText('Mock Mode')).toBeInTheDocument();
      });
    });
  });

  describe('Control Actions', () => {
    beforeEach(() => {
      // Mock successful health response
      mockFetch.mockResolvedValue({
        ok: true,
        json: () => Promise.resolve(mockHealthResponse)
      });
    });

    it('should call onToggle when disable button is clicked', async () => {
      render(<TradingEconomicsPanel enabled={true} onToggle={mockProps.onToggle} />);

      await waitFor(() => {
        const disableButton = screen.getByRole('button', { name: /Disable/i });
        fireEvent.click(disableButton);
        expect(mockProps.onToggle).toHaveBeenCalledWith(false);
      });
    });

    it('should refresh data when refresh button is clicked', async () => {
      render(<TradingEconomicsPanel enabled={true} onToggle={mockProps.onToggle} />);

      await waitFor(() => {
        const refreshButton = screen.getByRole('button', { name: /Refresh/i });
        fireEvent.click(refreshButton);
        // Should trigger new API calls
      });

      // Verify that health endpoint was called again
      expect(mockFetch).toHaveBeenCalledWith('http://localhost:8001/api/v1/trading-economics/health');
    });
  });

  describe('Data Display', () => {
    beforeEach(() => {
      // Mock all successful API responses for data display
      mockFetch
        .mockResolvedValueOnce({
          ok: true,
          json: () => Promise.resolve(mockHealthResponse)
        })
        .mockResolvedValueOnce({
          ok: true,
          json: () => Promise.resolve(mockCountriesResponse)
        })
        .mockResolvedValueOnce({
          ok: true,
          json: () => Promise.resolve({ success: true, statistics: mockStatistics })
        })
        .mockResolvedValueOnce({
          ok: true,
          json: () => Promise.resolve(mockMajorIndicators.gdp)
        })
        .mockResolvedValueOnce({
          ok: true,
          json: () => Promise.resolve(mockMajorIndicators.inflation)
        })
        .mockResolvedValueOnce({
          ok: true,
          json: () => Promise.resolve(mockMajorIndicators.unemployment)
        })
        .mockResolvedValueOnce({
          ok: true,
          json: () => Promise.resolve(mockMajorIndicators.interest_rates)
        })
        .mockResolvedValueOnce({
          ok: true,
          json: () => Promise.resolve(mockMarketOverview.currencies)
        })
        .mockResolvedValueOnce({
          ok: true,
          json: () => Promise.resolve(mockMarketOverview.commodities)
        })
        .mockResolvedValueOnce({
          ok: true,
          json: () => Promise.resolve(mockMarketOverview.stocks)
        })
        .mockResolvedValueOnce({
          ok: true,
          json: () => Promise.resolve(mockMarketOverview.bonds)
        });
    });

    it('should display economic indicators table', async () => {
      render(<TradingEconomicsPanel enabled={true} onToggle={mockProps.onToggle} />);

      await waitFor(() => {
        expect(screen.getByText('Major Economic Indicators')).toBeInTheDocument();
        expect(screen.getByText('GDP Growth Rate')).toBeInTheDocument();
        expect(screen.getByText('2.1')).toBeInTheDocument();
        expect(screen.getByText('Percent')).toBeInTheDocument();
      });
    });

    it('should display market data table', async () => {
      render(<TradingEconomicsPanel enabled={true} onToggle={mockProps.onToggle} />);

      await waitFor(() => {
        expect(screen.getByText('Market Overview')).toBeInTheDocument();
        expect(screen.getByText('USD/EUR')).toBeInTheDocument();
        expect(screen.getByText('0.8500')).toBeInTheDocument();
        expect(screen.getByText('-1.20%')).toBeInTheDocument();
      });
    });

    it('should handle empty data gracefully', async () => {
      // Mock empty responses
      mockFetch
        .mockResolvedValueOnce({
          ok: true,
          json: () => Promise.resolve(mockHealthResponse)
        })
        .mockResolvedValueOnce({
          ok: true,
          json: () => Promise.resolve({ success: true, countries: [], count: 0 })
        })
        .mockResolvedValueOnce({
          ok: true,
          json: () => Promise.resolve({ success: true, statistics: null })
        })
        .mockResolvedValueOnce({
          ok: true,
          json: () => Promise.resolve([])
        })
        .mockResolvedValueOnce({
          ok: true,
          json: () => Promise.resolve([])
        })
        .mockResolvedValueOnce({
          ok: true,
          json: () => Promise.resolve([])
        })
        .mockResolvedValueOnce({
          ok: true,
          json: () => Promise.resolve([])
        })
        .mockResolvedValueOnce({
          ok: true,
          json: () => Promise.resolve([])
        })
        .mockResolvedValueOnce({
          ok: true,
          json: () => Promise.resolve([])
        })
        .mockResolvedValueOnce({
          ok: true,
          json: () => Promise.resolve([])
        })
        .mockResolvedValueOnce({
          ok: true,
          json: () => Promise.resolve([])
        });

      render(<TradingEconomicsPanel enabled={true} onToggle={mockProps.onToggle} />);

      await waitFor(() => {
        expect(screen.getByText('No indicators found for this category')).toBeInTheDocument();
        expect(screen.getByText('No market data available')).toBeInTheDocument();
      });
    });
  });

  describe('Rate Limit Display', () => {
    it('should display rate limit information', async () => {
      const healthWithRateLimit = {
        ...mockHealthResponse,
        rate_limit_status: {
          requests_made: 45,
          requests_limit: 500,
          window_seconds: 60,
          time_until_reset: 15
        }
      };

      const statsWithRateLimit = {
        ...mockStatistics,
        rate_limit_status: {
          requests_made: 45,
          requests_limit: 500,
          window_seconds: 60,
          time_until_reset: 15
        }
      };

      mockFetch
        .mockResolvedValueOnce({
          ok: true,
          json: () => Promise.resolve(healthWithRateLimit)
        })
        .mockResolvedValueOnce({
          ok: true,
          json: () => Promise.resolve(mockCountriesResponse)
        })
        .mockResolvedValueOnce({
          ok: true,
          json: () => Promise.resolve({ success: true, statistics: statsWithRateLimit })
        });

      render(<TradingEconomicsPanel enabled={true} onToggle={mockProps.onToggle} />);

      await waitFor(() => {
        expect(screen.getByText('45')).toBeInTheDocument();
        expect(screen.getByText('/500')).toBeInTheDocument();
        expect(screen.getByText(/Rate Limit: 45 \/ 500 requests per minute/)).toBeInTheDocument();
      });
    });
  });
});