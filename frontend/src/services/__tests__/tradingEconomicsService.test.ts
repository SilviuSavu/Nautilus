import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import { tradingEconomicsService } from '../tradingEconomicsService';

// Mock fetch globally
const mockFetch = vi.fn();
global.fetch = mockFetch;

describe('TradingEconomicsService', () => {
  beforeEach(() => {
    mockFetch.mockClear();
    // Set up default environment variable
    import.meta.env.VITE_API_BASE_URL = 'http://localhost:8001';
  });

  afterEach(() => {
    vi.clearAllMocks();
  });

  describe('Health Check', () => {
    it('should return health status on successful check', async () => {
      const mockHealthResponse = {
        service: 'trading_economics',
        status: 'healthy',
        package_available: true,
        using_guest_access: false,
        last_check: '2023-10-26T10:00:00Z'
      };

      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: () => Promise.resolve(mockHealthResponse)
      });

      const result = await tradingEconomicsService.checkHealth();

      expect(mockFetch).toHaveBeenCalledWith('http://localhost:8001/api/v1/trading-economics/health');
      expect(result).toEqual(mockHealthResponse);
    });

    it('should handle health check error', async () => {
      mockFetch.mockRejectedValueOnce(new Error('Network error'));

      const result = await tradingEconomicsService.checkHealth();

      expect(result.status).toBe('error');
      expect(result.error).toBe('Network error');
      expect(result.service).toBe('trading_economics');
    });

    it('should handle HTTP error response', async () => {
      mockFetch.mockResolvedValueOnce({
        ok: false,
        status: 500,
        statusText: 'Internal Server Error'
      });

      const result = await tradingEconomicsService.checkHealth();

      expect(result.status).toBe('error');
      expect(result.error).toBe('HTTP 500: Internal Server Error');
    });
  });

  describe('Countries Data', () => {
    it('should fetch countries successfully', async () => {
      const mockCountriesResponse = {
        success: true,
        count: 2,
        countries: [
          { Country: 'United States', CountryGroup: 'North America' },
          { Country: 'United Kingdom', CountryGroup: 'Europe' }
        ]
      };

      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: () => Promise.resolve(mockCountriesResponse)
      });

      const result = await tradingEconomicsService.getCountries();

      expect(mockFetch).toHaveBeenCalledWith('http://localhost:8001/api/v1/trading-economics/countries');
      expect(result).toEqual(mockCountriesResponse.countries);
    });

    it('should handle countries fetch error', async () => {
      mockFetch.mockRejectedValueOnce(new Error('Fetch error'));

      const result = await tradingEconomicsService.getCountries();

      expect(result).toEqual([]);
    });
  });

  describe('Indicators Data', () => {
    it('should fetch indicators successfully', async () => {
      const mockIndicatorsResponse = {
        success: true,
        count: 1,
        indicators: [
          {
            Country: 'United States',
            Category: 'GDP',
            Title: 'GDP Growth Rate',
            LatestValue: 2.1,
            Unit: 'Percent'
          }
        ]
      };

      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: () => Promise.resolve(mockIndicatorsResponse)
      });

      const result = await tradingEconomicsService.getIndicators();

      expect(mockFetch).toHaveBeenCalledWith('http://localhost:8001/api/v1/trading-economics/indicators');
      expect(result).toEqual(mockIndicatorsResponse.indicators);
    });

    it('should fetch country indicators successfully', async () => {
      const country = 'united states';
      const mockResponse = {
        success: true,
        country,
        data: [
          { Country: 'United States', Category: 'GDP', LatestValue: 2.1 }
        ]
      };

      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: () => Promise.resolve(mockResponse)
      });

      const result = await tradingEconomicsService.getCountryIndicators(country);

      expect(mockFetch).toHaveBeenCalledWith(`http://localhost:8001/api/v1/trading-economics/indicators/${encodeURIComponent(country)}`);
      expect(result).toEqual(mockResponse.data);
    });

    it('should fetch country indicators with category filter', async () => {
      const country = 'united states';
      const category = 'gdp';
      const mockResponse = {
        success: true,
        country,
        category,
        data: [{ Country: 'United States', Category: 'GDP', LatestValue: 2.1 }]
      };

      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: () => Promise.resolve(mockResponse)
      });

      const result = await tradingEconomicsService.getCountryIndicators(country, category);

      expect(mockFetch).toHaveBeenCalledWith(`http://localhost:8001/api/v1/trading-economics/indicators/${encodeURIComponent(country)}?category=${category}`);
      expect(result).toEqual(mockResponse.data);
    });

    it('should fetch specific indicator successfully', async () => {
      const country = 'united states';
      const indicator = 'gdp';
      const mockResponse = {
        success: true,
        country,
        indicator,
        data: [{ Country: 'United States', Category: 'GDP', LatestValue: 2.1 }]
      };

      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: () => Promise.resolve(mockResponse)
      });

      const result = await tradingEconomicsService.getSpecificIndicator(country, indicator);

      expect(mockFetch).toHaveBeenCalledWith(`http://localhost:8001/api/v1/trading-economics/indicator/${encodeURIComponent(country)}/${encodeURIComponent(indicator)}`);
      expect(result).toEqual(mockResponse.data);
    });

    it('should fetch specific indicator with date filters', async () => {
      const country = 'united states';
      const indicator = 'gdp';
      const startDate = '2023-01-01';
      const endDate = '2023-12-31';
      const mockResponse = {
        success: true,
        country,
        indicator,
        data: [{ Country: 'United States', Category: 'GDP', LatestValue: 2.1 }]
      };

      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: () => Promise.resolve(mockResponse)
      });

      const result = await tradingEconomicsService.getSpecificIndicator(country, indicator, startDate, endDate);

      expect(mockFetch).toHaveBeenCalledWith(`http://localhost:8001/api/v1/trading-economics/indicator/${encodeURIComponent(country)}/${encodeURIComponent(indicator)}?start_date=${startDate}&end_date=${endDate}`);
      expect(result).toEqual(mockResponse.data);
    });
  });

  describe('Calendar Data', () => {
    it('should fetch calendar events successfully', async () => {
      const mockCalendarResponse = {
        success: true,
        events: [
          {
            Event: 'GDP Growth Rate',
            Country: 'United States',
            Date: '2023-10-26',
            Importance: 'high'
          }
        ]
      };

      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: () => Promise.resolve(mockCalendarResponse)
      });

      const result = await tradingEconomicsService.getCalendar();

      expect(mockFetch).toHaveBeenCalledWith('http://localhost:8001/api/v1/trading-economics/calendar');
      expect(result).toEqual(mockCalendarResponse.events);
    });

    it('should fetch calendar events with filters', async () => {
      const filters = {
        country: 'united states',
        importance: 'high',
        startDate: '2023-01-01',
        endDate: '2023-12-31'
      };

      const mockCalendarResponse = {
        success: true,
        events: [
          {
            Event: 'GDP Growth Rate',
            Country: 'United States',
            Date: '2023-10-26',
            Importance: 'high'
          }
        ]
      };

      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: () => Promise.resolve(mockCalendarResponse)
      });

      const result = await tradingEconomicsService.getCalendar(filters);

      const expectedUrl = `http://localhost:8001/api/v1/trading-economics/calendar?country=${filters.country}&importance=${filters.importance}&start_date=${filters.startDate}&end_date=${filters.endDate}`;
      expect(mockFetch).toHaveBeenCalledWith(expectedUrl);
      expect(result).toEqual(mockCalendarResponse.events);
    });
  });

  describe('Market Data', () => {
    it('should fetch market data successfully', async () => {
      const marketType = 'currencies';
      const mockMarketResponse = {
        success: true,
        market_type: marketType,
        data: [
          {
            Symbol: 'USD/EUR',
            Name: 'US Dollar vs Euro',
            Last: 0.85,
            Group: 'currencies'
          }
        ]
      };

      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: () => Promise.resolve(mockMarketResponse)
      });

      const result = await tradingEconomicsService.getMarkets(marketType);

      expect(mockFetch).toHaveBeenCalledWith(`http://localhost:8001/api/v1/trading-economics/markets/${encodeURIComponent(marketType)}`);
      expect(result).toEqual(mockMarketResponse.data);
    });

    it('should fetch market data with country filter', async () => {
      const marketType = 'currencies';
      const country = 'united states';
      const mockMarketResponse = {
        success: true,
        market_type: marketType,
        country,
        data: []
      };

      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: () => Promise.resolve(mockMarketResponse)
      });

      const result = await tradingEconomicsService.getMarkets(marketType, country);

      expect(mockFetch).toHaveBeenCalledWith(`http://localhost:8001/api/v1/trading-economics/markets/${encodeURIComponent(marketType)}?country=${country}`);
      expect(result).toEqual(mockMarketResponse.data);
    });
  });

  describe('Forecast Data', () => {
    it('should fetch forecast data successfully', async () => {
      const country = 'united states';
      const indicator = 'gdp';
      const mockForecastResponse = {
        success: true,
        country,
        indicator,
        forecast: {
          Country: 'United States',
          Category: 'GDP',
          YearEnd: 2024,
          q1: 2.1,
          q2: 2.3
        }
      };

      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: () => Promise.resolve(mockForecastResponse)
      });

      const result = await tradingEconomicsService.getForecast(country, indicator);

      expect(mockFetch).toHaveBeenCalledWith(`http://localhost:8001/api/v1/trading-economics/forecast/${encodeURIComponent(country)}/${encodeURIComponent(indicator)}`);
      expect(result).toEqual(mockForecastResponse.forecast);
    });

    it('should return null on forecast fetch error', async () => {
      mockFetch.mockRejectedValueOnce(new Error('Fetch error'));

      const result = await tradingEconomicsService.getForecast('united states', 'gdp');

      expect(result).toBeNull();
    });
  });

  describe('Search', () => {
    it('should search indicators successfully', async () => {
      const term = 'gdp';
      const mockSearchResponse = {
        success: true,
        search_term: term,
        results: [
          {
            Country: 'United States',
            Category: 'GDP',
            Title: 'GDP Growth Rate',
            LatestValue: 2.1
          }
        ]
      };

      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: () => Promise.resolve(mockSearchResponse)
      });

      const result = await tradingEconomicsService.search(term);

      expect(mockFetch).toHaveBeenCalledWith(`http://localhost:8001/api/v1/trading-economics/search?term=${term}`);
      expect(result).toEqual(mockSearchResponse.results);
    });

    it('should search indicators with category filter', async () => {
      const term = 'inflation';
      const category = 'indicators';
      const mockSearchResponse = {
        success: true,
        search_term: term,
        category,
        results: []
      };

      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: () => Promise.resolve(mockSearchResponse)
      });

      const result = await tradingEconomicsService.search(term, category);

      expect(mockFetch).toHaveBeenCalledWith(`http://localhost:8001/api/v1/trading-economics/search?term=${term}&category=${category}`);
      expect(result).toEqual(mockSearchResponse.results);
    });
  });

  describe('Statistics and Operations', () => {
    it('should fetch statistics successfully', async () => {
      const mockStatsResponse = {
        success: true,
        statistics: {
          total_requests: 100,
          cache_size: 5,
          rate_limit_status: {
            requests_made: 10,
            requests_limit: 500
          }
        }
      };

      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: () => Promise.resolve(mockStatsResponse)
      });

      const result = await tradingEconomicsService.getStatistics();

      expect(mockFetch).toHaveBeenCalledWith('http://localhost:8001/api/v1/trading-economics/statistics');
      expect(result).toEqual(mockStatsResponse.statistics);
    });

    it('should refresh cache successfully', async () => {
      const mockRefreshResponse = {
        success: true,
        cache_refresh: {
          cache_refreshed: true,
          timestamp: '2023-10-26T10:00:00Z'
        }
      };

      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: () => Promise.resolve(mockRefreshResponse)
      });

      const result = await tradingEconomicsService.refreshCache();

      expect(mockFetch).toHaveBeenCalledWith('http://localhost:8001/api/v1/trading-economics/cache/refresh', {
        method: 'POST'
      });
      expect(result).toBe(true);
    });

    it('should fetch supported functions successfully', async () => {
      const mockFunctionsResponse = {
        success: true,
        functions: [
          'getCountries',
          'getIndicatorData',
          'getCalendarData',
          'getMarketsData'
        ]
      };

      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: () => Promise.resolve(mockFunctionsResponse)
      });

      const result = await tradingEconomicsService.getSupportedFunctions();

      expect(mockFetch).toHaveBeenCalledWith('http://localhost:8001/api/v1/trading-economics/supported-functions');
      expect(result).toEqual(mockFunctionsResponse.functions);
    });
  });

  describe('High-Level Methods', () => {
    it('should fetch major indicators successfully', async () => {
      const mockGdpResponse = [
        { Country: 'United States', Category: 'GDP', LatestValue: 2.1 }
      ];
      const mockInflationResponse = [
        { Country: 'United States', Category: 'Inflation', LatestValue: 3.2 }
      ];
      const mockEmploymentResponse = [
        { Country: 'United States', Category: 'Employment', LatestValue: 3.7 }
      ];
      const mockRatesResponse = [
        { Country: 'United States', Category: 'Interest Rate', LatestValue: 5.25 }
      ];

      mockFetch
        .mockResolvedValueOnce({
          ok: true,
          json: () => Promise.resolve({ data: mockGdpResponse })
        })
        .mockResolvedValueOnce({
          ok: true,
          json: () => Promise.resolve({ data: mockInflationResponse })
        })
        .mockResolvedValueOnce({
          ok: true,
          json: () => Promise.resolve({ data: mockEmploymentResponse })
        })
        .mockResolvedValueOnce({
          ok: true,
          json: () => Promise.resolve({ data: mockRatesResponse })
        });

      const result = await tradingEconomicsService.getMajorIndicators();

      expect(result.gdp).toEqual(mockGdpResponse.slice(0, 3));
      expect(result.inflation).toEqual(mockInflationResponse.slice(0, 3));
      expect(result.unemployment).toEqual(mockEmploymentResponse.slice(0, 3));
      expect(result.interest_rates).toEqual(mockRatesResponse.slice(0, 3));
    });

    it('should fetch market overview successfully', async () => {
      const mockCurrenciesResponse = [
        { Symbol: 'USD/EUR', Group: 'currencies', Last: 0.85 }
      ];
      const mockCommoditiesResponse = [
        { Symbol: 'GOLD', Group: 'commodities', Last: 1950.0 }
      ];
      const mockStocksResponse = [
        { Symbol: 'SPX', Group: 'stocks', Last: 4300.0 }
      ];
      const mockBondsResponse = [
        { Symbol: '10Y', Group: 'bonds', Last: 4.5 }
      ];

      mockFetch
        .mockResolvedValueOnce({
          ok: true,
          json: () => Promise.resolve({ data: mockCurrenciesResponse })
        })
        .mockResolvedValueOnce({
          ok: true,
          json: () => Promise.resolve({ data: mockCommoditiesResponse })
        })
        .mockResolvedValueOnce({
          ok: true,
          json: () => Promise.resolve({ data: mockStocksResponse })
        })
        .mockResolvedValueOnce({
          ok: true,
          json: () => Promise.resolve({ data: mockBondsResponse })
        });

      const result = await tradingEconomicsService.getMarketOverview();

      expect(result.currencies).toEqual(mockCurrenciesResponse.slice(0, 5));
      expect(result.commodities).toEqual(mockCommoditiesResponse.slice(0, 5));
      expect(result.stocks).toEqual(mockStocksResponse.slice(0, 5));
      expect(result.bonds).toEqual(mockBondsResponse.slice(0, 5));
    });
  });

  describe('Error Handling', () => {
    it('should handle network errors gracefully in all methods', async () => {
      mockFetch.mockRejectedValue(new Error('Network error'));

      const countries = await tradingEconomicsService.getCountries();
      const indicators = await tradingEconomicsService.getIndicators();
      const calendar = await tradingEconomicsService.getCalendar();
      const markets = await tradingEconomicsService.getMarkets('currencies');
      const forecast = await tradingEconomicsService.getForecast('us', 'gdp');
      const search = await tradingEconomicsService.search('gdp');
      const statistics = await tradingEconomicsService.getStatistics();
      const cacheRefresh = await tradingEconomicsService.refreshCache();
      const functions = await tradingEconomicsService.getSupportedFunctions();

      expect(countries).toEqual([]);
      expect(indicators).toEqual([]);
      expect(calendar).toEqual([]);
      expect(markets).toEqual([]);
      expect(forecast).toBeNull();
      expect(search).toEqual([]);
      expect(statistics).toBeNull();
      expect(cacheRefresh).toBe(false);
      expect(functions).toEqual([]);
    });

    it('should handle HTTP errors gracefully', async () => {
      mockFetch.mockResolvedValue({
        ok: false,
        status: 500,
        statusText: 'Internal Server Error'
      });

      const result = await tradingEconomicsService.getCountries();
      expect(result).toEqual([]);
    });

    it('should handle malformed JSON responses', async () => {
      mockFetch.mockResolvedValue({
        ok: true,
        json: () => Promise.reject(new Error('Invalid JSON'))
      });

      const result = await tradingEconomicsService.getCountries();
      expect(result).toEqual([]);
    });
  });
});