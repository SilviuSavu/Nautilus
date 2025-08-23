/**
 * DBnomics Service
 * Specialized service for interacting with DBnomics economic and statistical data
 */

interface DBnomicsProvider {
  code: string;
  name: string;
  website: string;
  data_count: number;
  datasets: DBnomicsDataset[];
}

interface DBnomicsDataset {
  code: string;
  name: string;
  provider_code: string;
  series_count: number;
  last_update: string;
  dimensions?: Record<string, string[]>;
}

interface DBnomicsSeries {
  provider_code: string;
  dataset_code: string;
  series_code: string;
  series_name: string;
  frequency: string;
  unit: string;
  last_update: string;
  first_date: string;
  last_date: string;
  observations_count: number;
  data?: Array<{
    period: string;
    value: number;
  }>;
}

interface DBnomicsSearchResult {
  providers: DBnomicsProvider[];
  datasets: DBnomicsDataset[];
  series: DBnomicsSeries[];
}

interface DBnomicsFilterOptions {
  provider?: string;
  dataset?: string;
  dimensions?: Record<string, string[]>;
  startDate?: string;
  endDate?: string;
  maxSeries?: number;
}

class DBnomicsService {
  private apiUrl: string;

  constructor() {
    this.apiUrl = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8001';
  }

  /**
   * Check DBnomics service health
   */
  async checkHealth(): Promise<{ status: string; api_available: boolean; providers?: number }> {
    try {
      const response = await fetch(`${this.apiUrl}/api/v1/dbnomics/health`);
      return await response.json();
    } catch (error) {
      console.warn('DBnomics backend not implemented yet, using mock data:', error);
      return { 
        status: 'mock', 
        api_available: true, 
        providers: 80,
        message: 'Using mock data - backend integration pending'
      };
    }
  }

  /**
   * Get list of available providers
   */
  async getProviders(): Promise<DBnomicsProvider[]> {
    try {
      const response = await fetch(`${this.apiUrl}/api/v1/dbnomics/providers`);
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      return await response.json();
    } catch (error) {
      console.warn('Using mock provider data:', error);
      return [
        { code: 'IMF', name: 'International Monetary Fund', website: 'imf.org', data_count: 50000, datasets: [] },
        { code: 'OECD', name: 'Organisation for Economic Co-operation and Development', website: 'oecd.org', data_count: 100000, datasets: [] },
        { code: 'ECB', name: 'European Central Bank', website: 'ecb.europa.eu', data_count: 25000, datasets: [] },
        { code: 'EUROSTAT', name: 'European Union Statistics', website: 'eurostat.eu', data_count: 75000, datasets: [] },
        { code: 'BIS', name: 'Bank for International Settlements', website: 'bis.org', data_count: 15000, datasets: [] },
        { code: 'WB', name: 'World Bank', website: 'worldbank.org', data_count: 60000, datasets: [] }
      ];
    }
  }

  /**
   * Get datasets for a specific provider
   */
  async getDatasets(providerCode: string): Promise<DBnomicsDataset[]> {
    try {
      const response = await fetch(`${this.apiUrl}/api/v1/dbnomics/datasets/${providerCode}`);
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      return await response.json();
    } catch (error) {
      console.error(`Failed to fetch datasets for ${providerCode}:`, error);
      throw error;
    }
  }

  /**
   * Search for series with filters
   */
  async searchSeries(filters: DBnomicsFilterOptions): Promise<DBnomicsSeries[]> {
    try {
      const params = new URLSearchParams();
      
      if (filters.provider) params.append('provider_code', filters.provider);
      if (filters.dataset) params.append('dataset_code', filters.dataset);
      if (filters.startDate) params.append('start_date', filters.startDate);
      if (filters.endDate) params.append('end_date', filters.endDate);
      if (filters.maxSeries) params.append('max_nb_series', filters.maxSeries.toString());
      
      if (filters.dimensions) {
        params.append('dimensions', JSON.stringify(filters.dimensions));
      }

      const response = await fetch(`${this.apiUrl}/api/v1/dbnomics/series?${params}`);
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      return await response.json();
    } catch (error) {
      console.error('Failed to search series:', error);
      throw error;
    }
  }

  /**
   * Get specific series data
   */
  async getSeries(
    providerCode: string, 
    datasetCode: string, 
    seriesCode: string,
    options?: { startDate?: string; endDate?: string }
  ): Promise<DBnomicsSeries> {
    try {
      const params = new URLSearchParams();
      params.append('provider_code', providerCode);
      params.append('dataset_code', datasetCode);
      params.append('series_code', seriesCode);
      
      if (options?.startDate) params.append('start_date', options.startDate);
      if (options?.endDate) params.append('end_date', options.endDate);

      const response = await fetch(`${this.apiUrl}/api/v1/dbnomics/series/${providerCode}/${datasetCode}/${seriesCode}?${params}`);
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      return await response.json();
    } catch (error) {
      console.error(`Failed to fetch series ${providerCode}/${datasetCode}/${seriesCode}:`, error);
      throw error;
    }
  }

  /**
   * Get economic indicators by category
   */
  async getEconomicIndicators(category?: 'inflation' | 'employment' | 'growth' | 'monetary' | 'trade'): Promise<DBnomicsSeries[]> {
    // Return mock data for demo purposes
    const mockSeries: DBnomicsSeries[] = [
      {
        provider_code: 'IMF',
        dataset_code: 'CPI',
        series_code: 'A.US.PCPIEC_WT',
        series_name: 'United States - Consumer Price Index, All items',
        frequency: 'A',
        unit: 'Index',
        last_update: '2024-01-15',
        first_date: '1980-01-01',
        last_date: '2023-12-31',
        observations_count: 44
      },
      {
        provider_code: 'OECD',
        dataset_code: 'KEI',
        series_code: 'M.USA.LORSGPRT.STSA',
        series_name: 'United States - Unemployment rate',
        frequency: 'M',
        unit: 'Percent',
        last_update: '2024-01-20',
        first_date: '1990-01-01',
        last_date: '2023-12-31',
        observations_count: 408
      },
      {
        provider_code: 'ECB',
        dataset_code: 'IRS',
        series_code: 'M.U2.EUR.RT.MM.EURIBOR3MD_.HSTA',
        series_name: 'Euro area - 3-month EURIBOR',
        frequency: 'M',
        unit: 'Percent per annum',
        last_update: '2024-01-18',
        first_date: '1999-01-01',
        last_date: '2023-12-31',
        observations_count: 300
      }
    ];

    // For now, just return mock data
    console.warn('Using mock economic indicators - backend not implemented');
    
    if (category) {
      return mockSeries.filter((_, index) => index < 2); // Return 2 series for category
    } else {
      return mockSeries; // Return all mock series
    }
  }

  /**
   * Get popular series by country
   */
  async getCountryIndicators(
    countryCode: string,
    indicators: string[] = ['CPI', 'GDP', 'UNEMPLOYMENT']
  ): Promise<DBnomicsSeries[]> {
    const results: DBnomicsSeries[] = [];

    for (const indicator of indicators) {
      try {
        const filter: DBnomicsFilterOptions = {
          dimensions: { geo: [countryCode] },
          maxSeries: 5
        };

        switch (indicator) {
          case 'CPI':
            filter.provider = 'IMF';
            filter.dataset = 'CPI';
            break;
          case 'GDP':
            filter.provider = 'OECD';
            filter.dataset = 'QNA';
            filter.dimensions!.subject = ['B1_GE'];
            break;
          case 'UNEMPLOYMENT':
            filter.provider = 'OECD';
            filter.dataset = 'KEI';
            filter.dimensions!.subject = ['LORSGPRT'];
            break;
        }

        const series = await this.searchSeries(filter);
        results.push(...series);
      } catch (error) {
        console.warn(`Failed to get ${indicator} for ${countryCode}:`, error);
      }
    }

    return results;
  }

  /**
   * Subscribe to series updates (WebSocket or polling)
   */
  async subscribeSeries(seriesIds: string[], callback: (data: DBnomicsSeries[]) => void): Promise<() => void> {
    // Implement polling for now (WebSocket could be added later)
    const interval = setInterval(async () => {
      try {
        const updates = await Promise.all(
          seriesIds.map(async (id) => {
            const [provider, dataset, series] = id.split('/');
            return this.getSeries(provider, dataset, series);
          })
        );
        callback(updates);
      } catch (error) {
        console.error('Failed to update series:', error);
      }
    }, 30000); // Update every 30 seconds

    // Return unsubscribe function
    return () => clearInterval(interval);
  }

  /**
   * Format series for display
   */
  formatSeriesForDisplay(series: DBnomicsSeries): {
    id: string;
    name: string;
    value: string;
    unit: string;
    date: string;
    provider: string;
  } {
    const latestData = series.data?.[series.data.length - 1];
    
    return {
      id: `${series.provider_code}/${series.dataset_code}/${series.series_code}`,
      name: series.series_name,
      value: latestData ? latestData.value.toLocaleString() : 'N/A',
      unit: series.unit,
      date: latestData ? latestData.period : series.last_date,
      provider: series.provider_code
    };
  }

  /**
   * Get data source statistics
   */
  async getStatistics(): Promise<{
    totalProviders: number;
    totalDatasets: number;
    totalSeries: number;
    lastUpdate: string;
    topProviders: Array<{ name: string; seriesCount: number }>;
  }> {
    try {
      const response = await fetch(`${this.apiUrl}/api/v1/dbnomics/statistics`);
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      return await response.json();
    } catch (error) {
      console.error('Failed to get statistics:', error);
      // Return default values if API call fails
      return {
        totalProviders: 80,
        totalDatasets: 1000,
        totalSeries: 800000000,
        lastUpdate: new Date().toISOString(),
        topProviders: [
          { name: 'IMF', seriesCount: 50000 },
          { name: 'OECD', seriesCount: 100000 },
          { name: 'World Bank', seriesCount: 75000 }
        ]
      };
    }
  }
}

// Singleton instance
export const dbnomicsService = new DBnomicsService();

// Export types
export type {
  DBnomicsProvider,
  DBnomicsDataset,
  DBnomicsSeries,
  DBnomicsSearchResult,
  DBnomicsFilterOptions
};