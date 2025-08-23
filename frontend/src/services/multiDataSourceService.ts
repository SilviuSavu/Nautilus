/**
 * Multi-Data Source Service
 * Coordinates multiple data sources simultaneously and provides intelligent routing
 */

import type { DataSource } from '../components/DataSources';

interface DataSourceConfig {
  enabled: boolean;
  priority: number;
  fallbackEnabled: boolean;
  rateLimit?: {
    calls: number;
    window: number; // in seconds
    used: number;
    resetTime: Date;
  };
}

interface DataRequest {
  type: 'quote' | 'historical' | 'search' | 'fundamentals' | 'economic' | 'statistical';
  symbol?: string;
  timeframe?: string;
  startDate?: Date;
  endDate?: Date;
  keywords?: string;
  urgency?: 'low' | 'medium' | 'high';
  // DBnomics specific parameters
  provider?: string;
  dataset?: string;
  series?: string;
  dimensions?: Record<string, string[]>;
}

interface DataResponse {
  source: string;
  data: any;
  timestamp: Date;
  fromCache: boolean;
  nextSource?: string; // Fallback source if this fails
}

class MultiDataSourceService {
  private configurations: Map<string, DataSourceConfig> = new Map();
  private cache: Map<string, { data: any; timestamp: Date; ttl: number }> = new Map();
  private requestQueue: Array<{ request: DataRequest; resolve: Function; reject: Function }> = [];
  private isProcessing = false;

  constructor() {
    this.initializeConfigurations();
    // Queue processor starts automatically when requests are queued
  }

  private initializeConfigurations() {
    // Default configurations for each data source
    const defaultConfigs: Array<[string, DataSourceConfig]> = [
      ['ibkr', { enabled: true, priority: 1, fallbackEnabled: true }],
      ['alpha_vantage', { 
        enabled: true, 
        priority: 2, 
        fallbackEnabled: true,
        rateLimit: {
          calls: 5,
          window: 60,
          used: 0,
          resetTime: new Date(Date.now() + 60000)
        }
      }],
      ['fred', { enabled: true, priority: 3, fallbackEnabled: true }],
      ['edgar', { enabled: false, priority: 4, fallbackEnabled: true }],
      ['dbnomics', { enabled: false, priority: 5, fallbackEnabled: true }],
      ['yfinance', { enabled: false, priority: 6, fallbackEnabled: true }]
    ];

    defaultConfigs.forEach(([id, config]) => {
      this.configurations.set(id, config);
    });
  }

  /**
   * Configure a data source
   */
  public configureDataSource(sourceId: string, config: Partial<DataSourceConfig>) {
    const current = this.configurations.get(sourceId) || {
      enabled: false,
      priority: 999,
      fallbackEnabled: false
    };
    
    this.configurations.set(sourceId, { ...current, ...config });
    
    // Emit configuration change event
    this.emitConfigurationChange();
  }

  /**
   * Enable or disable a data source
   */
  public toggleDataSource(sourceId: string, enabled: boolean) {
    this.configureDataSource(sourceId, { enabled });
  }

  /**
   * Get current configuration for all sources
   */
  public getConfigurations(): Array<{ id: string; config: DataSourceConfig }> {
    return Array.from(this.configurations.entries()).map(([id, config]) => ({
      id,
      config
    }));
  }

  /**
   * Get enabled sources sorted by priority
   */
  private getEnabledSources(): string[] {
    return Array.from(this.configurations.entries())
      .filter(([_, config]) => config.enabled)
      .sort((a, b) => a[1].priority - b[1].priority)
      .map(([id, _]) => id);
  }

  /**
   * Check if a source can handle the request type
   */
  private canSourceHandle(sourceId: string, requestType: string): boolean {
    const capabilities: Record<string, string[]> = {
      ibkr: ['quote', 'historical', 'fundamentals'],
      alpha_vantage: ['quote', 'historical', 'search', 'fundamentals'],
      fred: ['economic', 'historical'],
      edgar: ['fundamentals', 'search'],
      dbnomics: ['economic', 'statistical', 'historical', 'search'],
      yfinance: ['quote', 'historical', 'fundamentals']
    };

    return capabilities[sourceId]?.includes(requestType) || false;
  }

  /**
   * Check rate limits for a source
   */
  private checkRateLimit(sourceId: string): boolean {
    const config = this.configurations.get(sourceId);
    if (!config?.rateLimit) return true;

    const now = new Date();
    if (now > config.rateLimit.resetTime) {
      // Reset the rate limit
      config.rateLimit.used = 0;
      config.rateLimit.resetTime = new Date(now.getTime() + config.rateLimit.window * 1000);
    }

    return config.rateLimit.used < config.rateLimit.calls;
  }

  /**
   * Update rate limit usage
   */
  private updateRateLimit(sourceId: string) {
    const config = this.configurations.get(sourceId);
    if (config?.rateLimit) {
      config.rateLimit.used++;
    }
  }

  /**
   * Generate cache key for a request
   */
  private getCacheKey(request: DataRequest): string {
    return JSON.stringify({
      type: request.type,
      symbol: request.symbol,
      timeframe: request.timeframe,
      keywords: request.keywords
    });
  }

  /**
   * Check cache for data
   */
  private getFromCache(request: DataRequest): any | null {
    const cacheKey = this.getCacheKey(request);
    const cached = this.cache.get(cacheKey);
    
    if (cached && Date.now() - cached.timestamp.getTime() < cached.ttl) {
      return {
        source: 'cache',
        data: cached.data,
        timestamp: cached.timestamp,
        fromCache: true
      };
    }
    
    return null;
  }

  /**
   * Store data in cache
   */
  private storeInCache(request: DataRequest, data: any, ttl: number = 30000) {
    const cacheKey = this.getCacheKey(request);
    this.cache.set(cacheKey, {
      data,
      timestamp: new Date(),
      ttl
    });
  }

  /**
   * Request data from a specific source using persistent API client
   */
  private async requestFromSource(sourceId: string, request: DataRequest): Promise<any> {
    // Import persistent API client for reliable connections
    const { persistentApiClient } = await import('./persistentApiClient');
    
    // Map request types to endpoints
    const endpoints: Record<string, Record<string, string>> = {
      ibkr: {
        quote: '/api/v1/market-data/quote',
        historical: '/api/v1/market-data/historical/bars'
      },
      alpha_vantage: {
        quote: '/api/v1/nautilus-data/alpha-vantage/quote',
        search: '/api/v1/nautilus-data/alpha-vantage/search',
        fundamentals: '/api/v1/alpha-vantage/company'
      },
      fred: {
        economic: '/api/v1/fred/macro-factors',
        historical: '/api/v1/fred/series'
      },
      edgar: {
        search: '/api/v1/edgar/companies/search',
        fundamentals: '/api/v1/edgar/ticker'
      },
      dbnomics: {
        economic: '/api/v1/dbnomics/series',
        statistical: '/api/v1/dbnomics/series',
        historical: '/api/v1/dbnomics/series',
        search: '/api/v1/dbnomics/providers'
      },
      yfinance: {
        quote: '/api/v1/yfinance/quote',
        historical: '/api/v1/yfinance/historical'
      }
    };

    const endpoint = endpoints[sourceId]?.[request.type];
    if (!endpoint) {
      throw new Error(`No endpoint for ${sourceId} ${request.type}`);
    }

    // Build parameters object
    const params: Record<string, any> = {};
    
    if (request.symbol) {
      if (endpoint.includes('{symbol}') || endpoint.includes('{ticker}')) {
        // Handle URL path parameters
        const finalEndpoint = endpoint.replace('{symbol}', request.symbol).replace('{ticker}', request.symbol);
        return persistentApiClient.get(finalEndpoint);
      } else {
        params.symbol = request.symbol;
      }
    }
    
    if (request.keywords) {
      params.keywords = request.keywords;
      params.q = request.keywords;
    }
    
    if (request.timeframe) {
      params.timeframe = request.timeframe;
    }

    // DBnomics specific parameters
    if (sourceId === 'dbnomics') {
      if (request.provider) params.provider_code = request.provider;
      if (request.dataset) params.dataset_code = request.dataset;
      if (request.series) params.series_code = request.series;
      if (request.dimensions) params.dimensions = JSON.stringify(request.dimensions);
      if (request.startDate) params.start_date = request.startDate.toISOString().split('T')[0];
      if (request.endDate) params.end_date = request.endDate.toISOString().split('T')[0];
    }

    // Use persistent API client with automatic retry and connection management
    return persistentApiClient.get(endpoint, Object.keys(params).length > 0 ? params : undefined);
  }

  /**
   * Execute a data request using the multi-source strategy
   */
  public async executeRequest(request: DataRequest): Promise<DataResponse> {
    // Check cache first
    const cached = this.getFromCache(request);
    if (cached) {
      return cached;
    }

    // Get enabled sources that can handle this request
    const availableSources = this.getEnabledSources()
      .filter(sourceId => this.canSourceHandle(sourceId, request.type))
      .filter(sourceId => this.checkRateLimit(sourceId));

    if (availableSources.length === 0) {
      throw new Error(`No available data sources for ${request.type} request`);
    }

    // Try each source in priority order
    let lastError: Error | null = null;
    
    for (const sourceId of availableSources) {
      try {
        this.updateRateLimit(sourceId);
        const data = await this.requestFromSource(sourceId, request);
        
        // Store in cache
        this.storeInCache(request, data);
        
        return {
          source: sourceId,
          data,
          timestamp: new Date(),
          fromCache: false,
          nextSource: availableSources[availableSources.indexOf(sourceId) + 1]
        };
        
      } catch (error) {
        console.warn(`Data request failed for ${sourceId}:`, error);
        lastError = error as Error;
        continue;
      }
    }

    throw lastError || new Error('All data sources failed');
  }

  /**
   * Queue a request for batch processing
   */
  public queueRequest(request: DataRequest): Promise<DataResponse> {
    return new Promise((resolve, reject) => {
      this.requestQueue.push({ request, resolve, reject });
      
      if (!this.isProcessing) {
        this.processQueue();
      }
    });
  }

  /**
   * Process the request queue
   */
  private async processQueue() {
    if (this.isProcessing || this.requestQueue.length === 0) return;
    
    this.isProcessing = true;
    
    while (this.requestQueue.length > 0) {
      const batch = this.requestQueue.splice(0, 5); // Process 5 at a time
      
      await Promise.allSettled(
        batch.map(async ({ request, resolve, reject }) => {
          try {
            const result = await this.executeRequest(request);
            resolve(result);
          } catch (error) {
            reject(error);
          }
        })
      );
      
      // Small delay between batches to respect rate limits
      await new Promise(resolve => setTimeout(resolve, 1000));
    }
    
    this.isProcessing = false;
  }

  /**
   * Get real-time status of all data sources
   */
  public async getDataSourceStatus(): Promise<Array<{ id: string; status: string; enabled: boolean; rateLimitStatus?: any }>> {
    const sources = this.getConfigurations();
    
    return sources.map(({ id, config }) => ({
      id,
      status: config.enabled ? 'enabled' : 'disabled',
      enabled: config.enabled,
      rateLimitStatus: config.rateLimit ? {
        used: config.rateLimit.used,
        limit: config.rateLimit.calls,
        resetTime: config.rateLimit.resetTime
      } : undefined
    }));
  }

  /**
   * Clear cache
   */
  public clearCache() {
    this.cache.clear();
  }

  /**
   * Emit configuration change event
   */
  private emitConfigurationChange() {
    // In a real app, this would emit an event that components can listen to
    window.dispatchEvent(new CustomEvent('dataSourceConfigChanged', {
      detail: this.getConfigurations()
    }));
  }

  /**
   * Get cache statistics
   */
  public getCacheStats() {
    return {
      size: this.cache.size,
      items: Array.from(this.cache.entries()).map(([key, value]) => ({
        key,
        timestamp: value.timestamp,
        age: Date.now() - value.timestamp.getTime()
      }))
    };
  }
}

// Singleton instance
export const multiDataSourceService = new MultiDataSourceService();

// Export types
export type { DataRequest, DataResponse, DataSourceConfig };