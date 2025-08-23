/**
 * Trading Economics Service
 * Specialized service for interacting with Trading Economics global economic data
 * Provides access to 300,000+ economic indicators across 196 countries
 */

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL;

interface TradingEconomicsCountry {
  Country: string;
  CountryGroup: string;
  ISO?: string;
  Flag?: string;
}

interface TradingEconomicsIndicator {
  Country: string;
  Category: string;
  Title: string;
  LatestValue: number | null;
  PreviousValue: number | null;
  Forecast: number | null;
  Unit: string;
  Frequency: string;
  LastUpdate: string;
  Group?: string;
  Importance?: string;
}

interface TradingEconomicsCalendarEvent {
  Event: string;
  Country: string;
  Category: string;
  Date: string;
  Importance: 'high' | 'medium' | 'low';
  Actual: number | null;
  Previous: number | null;
  Forecast: number | null;
  Revised: number | null;
  Unit: string;
}

interface TradingEconomicsMarket {
  Symbol: string;
  Name: string;
  Country: string;
  Group: string;
  Last: number | null;
  DailyChange: number | null;
  DailyPercentualChange: number | null;
  WeeklyChange: number | null;
  MonthlyChange: number | null;
  YearlyChange: number | null;
  Date: string;
}

interface TradingEconomicsForecast {
  Country: string;
  Category: string;
  YearEnd: number;
  q1: number | null;
  q2: number | null;
  q3: number | null;
  q4: number | null;
}

interface TradingEconomicsSearchResult {
  Country: string;
  Category: string;
  Title: string;
  LatestValue: number | null;
  Unit: string;
  Symbol?: string;
}

interface TradingEconomicsHealthStatus {
  service: string;
  status: 'healthy' | 'degraded' | 'error' | 'mock_mode';
  package_available: boolean;
  using_guest_access: boolean;
  last_check: string;
  rate_limit_status?: {
    requests_made: number;
    requests_limit: number;
    window_seconds: number;
    time_until_reset: number;
  };
  error?: string;
}

interface TradingEconomicsStatistics {
  total_requests: number;
  cache_size: number;
  rate_limit_status: {
    requests_made: number;
    requests_limit: number;
    window_seconds: number;
    time_until_reset: number;
  };
  package_available: boolean;
  using_guest_access: boolean;
  last_reset: string;
}

class TradingEconomicsService {
  private baseURL = `${API_BASE_URL}/api/v1/trading-economics`;

  /**
   * Check Trading Economics service health and connectivity
   */
  async checkHealth(): Promise<TradingEconomicsHealthStatus> {
    try {
      const response = await fetch(`${this.baseURL}/health`);
      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
      }
      return await response.json();
    } catch (error) {
      console.error('Trading Economics health check failed:', error);
      return {
        service: 'trading_economics',
        status: 'error',
        package_available: false,
        using_guest_access: true,
        last_check: new Date().toISOString(),
        error: error instanceof Error ? error.message : 'Unknown error'
      };
    }
  }

  /**
   * Get all available countries with economic data
   */
  async getCountries(): Promise<TradingEconomicsCountry[]> {
    try {
      const response = await fetch(`${this.baseURL}/countries`);
      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
      }
      const data = await response.json();
      return data.countries || [];
    } catch (error) {
      console.error('Failed to fetch countries:', error);
      return [];
    }
  }

  /**
   * Get all available economic indicators
   */
  async getIndicators(): Promise<TradingEconomicsIndicator[]> {
    try {
      const response = await fetch(`${this.baseURL}/indicators`);
      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
      }
      const data = await response.json();
      return data.indicators || [];
    } catch (error) {
      console.error('Failed to fetch indicators:', error);
      return [];
    }
  }

  /**
   * Get economic indicators for a specific country
   */
  async getCountryIndicators(country: string, category?: string): Promise<TradingEconomicsIndicator[]> {
    try {
      const params = new URLSearchParams();
      if (category) params.append('category', category);
      
      const url = `${this.baseURL}/indicators/${encodeURIComponent(country)}${params.toString() ? '?' + params.toString() : ''}`;
      const response = await fetch(url);
      
      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
      }
      const data = await response.json();
      return data.data || [];
    } catch (error) {
      console.error(`Failed to fetch indicators for ${country}:`, error);
      return [];
    }
  }

  /**
   * Get specific economic indicator data for a country
   */
  async getSpecificIndicator(
    country: string, 
    indicator: string,
    startDate?: string,
    endDate?: string
  ): Promise<TradingEconomicsIndicator[]> {
    try {
      const params = new URLSearchParams();
      if (startDate) params.append('start_date', startDate);
      if (endDate) params.append('end_date', endDate);
      
      const url = `${this.baseURL}/indicator/${encodeURIComponent(country)}/${encodeURIComponent(indicator)}${params.toString() ? '?' + params.toString() : ''}`;
      const response = await fetch(url);
      
      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
      }
      const data = await response.json();
      return data.data || [];
    } catch (error) {
      console.error(`Failed to fetch ${indicator} for ${country}:`, error);
      return [];
    }
  }

  /**
   * Get economic calendar events
   */
  async getCalendar(filters?: {
    country?: string;
    category?: string;
    importance?: string;
    startDate?: string;
    endDate?: string;
  }): Promise<TradingEconomicsCalendarEvent[]> {
    try {
      const params = new URLSearchParams();
      if (filters?.country) params.append('country', filters.country);
      if (filters?.category) params.append('category', filters.category);
      if (filters?.importance) params.append('importance', filters.importance);
      if (filters?.startDate) params.append('start_date', filters.startDate);
      if (filters?.endDate) params.append('end_date', filters.endDate);
      
      const url = `${this.baseURL}/calendar${params.toString() ? '?' + params.toString() : ''}`;
      const response = await fetch(url);
      
      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
      }
      const data = await response.json();
      return data.events || [];
    } catch (error) {
      console.error('Failed to fetch calendar:', error);
      return [];
    }
  }

  /**
   * Get market data by type
   */
  async getMarkets(marketType: string, country?: string): Promise<TradingEconomicsMarket[]> {
    try {
      const params = new URLSearchParams();
      if (country) params.append('country', country);
      
      const url = `${this.baseURL}/markets/${encodeURIComponent(marketType)}${params.toString() ? '?' + params.toString() : ''}`;
      const response = await fetch(url);
      
      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
      }
      const data = await response.json();
      return data.data || [];
    } catch (error) {
      console.error(`Failed to fetch market data for ${marketType}:`, error);
      return [];
    }
  }

  /**
   * Get forecast data for specific indicator
   */
  async getForecast(country: string, indicator: string): Promise<TradingEconomicsForecast | null> {
    try {
      const url = `${this.baseURL}/forecast/${encodeURIComponent(country)}/${encodeURIComponent(indicator)}`;
      const response = await fetch(url);
      
      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
      }
      const data = await response.json();
      return data.forecast || null;
    } catch (error) {
      console.error(`Failed to fetch forecast for ${indicator} in ${country}:`, error);
      return null;
    }
  }

  /**
   * Search economic indicators by term
   */
  async search(term: string, category?: string): Promise<TradingEconomicsSearchResult[]> {
    try {
      const params = new URLSearchParams({ term });
      if (category) params.append('category', category);
      
      const url = `${this.baseURL}/search?${params.toString()}`;
      const response = await fetch(url);
      
      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
      }
      const data = await response.json();
      return data.results || [];
    } catch (error) {
      console.error(`Failed to search for '${term}':`, error);
      return [];
    }
  }

  /**
   * Get Trading Economics service statistics
   */
  async getStatistics(): Promise<TradingEconomicsStatistics | null> {
    try {
      const response = await fetch(`${this.baseURL}/statistics`);
      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
      }
      const data = await response.json();
      return data.statistics || null;
    } catch (error) {
      console.error('Failed to fetch statistics:', error);
      return null;
    }
  }

  /**
   * Refresh Trading Economics data cache
   */
  async refreshCache(): Promise<boolean> {
    try {
      const response = await fetch(`${this.baseURL}/cache/refresh`, {
        method: 'POST'
      });
      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
      }
      const data = await response.json();
      return data.success || false;
    } catch (error) {
      console.error('Failed to refresh cache:', error);
      return false;
    }
  }

  /**
   * Get list of supported Trading Economics API functions
   */
  async getSupportedFunctions(): Promise<string[]> {
    try {
      const response = await fetch(`${this.baseURL}/supported-functions`);
      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
      }
      const data = await response.json();
      return data.functions || [];
    } catch (error) {
      console.error('Failed to fetch supported functions:', error);
      return [];
    }
  }

  /**
   * Get major economic indicators for dashboard display
   */
  async getMajorIndicators(): Promise<{
    gdp: TradingEconomicsIndicator[];
    inflation: TradingEconomicsIndicator[];
    unemployment: TradingEconomicsIndicator[];
    interest_rates: TradingEconomicsIndicator[];
  }> {
    try {
      const [gdp, inflation, unemployment, rates] = await Promise.all([
        this.getCountryIndicators('united states', 'gdp'),
        this.getCountryIndicators('united states', 'inflation'),
        this.getCountryIndicators('united states', 'employment'),
        this.getCountryIndicators('united states', 'interest rate')
      ]);

      return {
        gdp: gdp.slice(0, 3),
        inflation: inflation.slice(0, 3),
        unemployment: unemployment.slice(0, 3),
        interest_rates: rates.slice(0, 3)
      };
    } catch (error) {
      console.error('Failed to fetch major indicators:', error);
      return { gdp: [], inflation: [], unemployment: [], interest_rates: [] };
    }
  }

  /**
   * Get market overview data
   */
  async getMarketOverview(): Promise<{
    currencies: TradingEconomicsMarket[];
    commodities: TradingEconomicsMarket[];
    stocks: TradingEconomicsMarket[];
    bonds: TradingEconomicsMarket[];
  }> {
    try {
      const [currencies, commodities, stocks, bonds] = await Promise.all([
        this.getMarkets('currencies'),
        this.getMarkets('commodities'),
        this.getMarkets('stocks'),
        this.getMarkets('bonds')
      ]);

      return {
        currencies: currencies.slice(0, 5),
        commodities: commodities.slice(0, 5),
        stocks: stocks.slice(0, 5),
        bonds: bonds.slice(0, 5)
      };
    } catch (error) {
      console.error('Failed to fetch market overview:', error);
      return { currencies: [], commodities: [], stocks: [], bonds: [] };
    }
  }
}

// Export singleton instance
export const tradingEconomicsService = new TradingEconomicsService();

// Export types
export type {
  TradingEconomicsCountry,
  TradingEconomicsIndicator,
  TradingEconomicsCalendarEvent,
  TradingEconomicsMarket,
  TradingEconomicsForecast,
  TradingEconomicsSearchResult,
  TradingEconomicsHealthStatus,
  TradingEconomicsStatistics
};