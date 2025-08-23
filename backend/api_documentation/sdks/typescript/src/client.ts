/**
 * Main Nautilus Trading Platform TypeScript Client
 * Comprehensive API client with full TypeScript support
 */

import axios, { AxiosInstance, AxiosResponse, AxiosError } from 'axios';
import { EventEmitter } from 'events';
import { AuthManager } from './auth';
import { WebSocketClient } from './websocket';
import { 
  NautilusException, 
  AuthenticationError, 
  RateLimitError,
  ValidationError,
  NetworkError 
} from './exceptions';
import {
  NautilusConfig,
  MarketData,
  HistoricalDataParams,
  HistoricalDataResponse,
  RiskLimit,
  CreateRiskLimitRequest,
  Strategy,
  DeployStrategyRequest,
  DeployStrategyResponse,
  HealthCheck,
  PerformanceAnalytics,
  RiskAnalytics,
  PaginationParams,
  ApiResponse,
  LoginCredentials,
  LoginResponse
} from './types';

/**
 * Market Data API client
 */
export class MarketDataAPI {
  constructor(private client: NautilusClient) {}

  /**
   * Get real-time quote for a symbol
   */
  async getQuote(symbol: string, source?: string): Promise<MarketData> {
    const params = source ? { source } : undefined;
    const response = await this.client.request<MarketData>(
      'GET',
      `/api/v1/market-data/quote/${symbol}`,
      { params }
    );
    return response.data;
  }

  /**
   * Get historical market data
   */
  async getHistoricalData(
    symbol: string,
    params: HistoricalDataParams = {}
  ): Promise<HistoricalDataResponse> {
    const response = await this.client.request<HistoricalDataResponse>(
      'GET',
      `/api/v1/market-data/historical/${symbol}`,
      { params }
    );
    return response.data;
  }

  /**
   * Search for symbols
   */
  async searchSymbols(query: string): Promise<string[]> {
    const response = await this.client.request<{ symbols: string[] }>(
      'GET',
      '/api/v1/market-data/search',
      { params: { q: query } }
    );
    return response.data.symbols;
  }
}

/**
 * Risk Management API client
 */
export class RiskManagementAPI {
  constructor(private client: NautilusClient) {}

  /**
   * Create a new risk limit
   */
  async createLimit(request: CreateRiskLimitRequest): Promise<RiskLimit> {
    const response = await this.client.request<RiskLimit>(
      'POST',
      '/api/v1/risk/limits',
      { data: request }
    );
    return response.data;
  }

  /**
   * Get all risk limits
   */
  async getLimits(): Promise<RiskLimit[]> {
    const response = await this.client.request<{ limits: RiskLimit[] }>(
      'GET',
      '/api/v1/risk/limits'
    );
    return response.data.limits;
  }

  /**
   * Check specific risk limit
   */
  async checkLimit(limitId: string): Promise<any> {
    const response = await this.client.request(
      'GET',
      `/api/v1/risk/limits/${limitId}/check`
    );
    return response.data;
  }

  /**
   * Get risk breaches
   */
  async getBreaches(params?: PaginationParams): Promise<any[]> {
    const response = await this.client.request<{ breaches: any[] }>(
      'GET',
      '/api/v1/risk/breaches',
      { params }
    );
    return response.data.breaches;
  }

  /**
   * Update risk limit
   */
  async updateLimit(limitId: string, updates: Partial<CreateRiskLimitRequest>): Promise<RiskLimit> {
    const response = await this.client.request<RiskLimit>(
      'PUT',
      `/api/v1/risk/limits/${limitId}`,
      { data: updates }
    );
    return response.data;
  }

  /**
   * Delete risk limit
   */
  async deleteLimit(limitId: string): Promise<void> {
    await this.client.request(
      'DELETE',
      `/api/v1/risk/limits/${limitId}`
    );
  }
}

/**
 * Strategy Management API client
 */
export class StrategyAPI {
  constructor(private client: NautilusClient) {}

  /**
   * Deploy a trading strategy
   */
  async deploy(request: DeployStrategyRequest): Promise<DeployStrategyResponse> {
    const response = await this.client.request<DeployStrategyResponse>(
      'POST',
      '/api/v1/strategies/deploy',
      { data: request }
    );
    return response.data;
  }

  /**
   * Get all strategies
   */
  async getStrategies(): Promise<Strategy[]> {
    const response = await this.client.request<{ strategies: Strategy[] }>(
      'GET',
      '/api/v1/strategies'
    );
    return response.data.strategies;
  }

  /**
   * Get specific strategy
   */
  async getStrategy(strategyId: string): Promise<Strategy> {
    const response = await this.client.request<Strategy>(
      'GET',
      `/api/v1/strategies/${strategyId}`
    );
    return response.data;
  }

  /**
   * Get deployment status
   */
  async getDeploymentStatus(deploymentId: string): Promise<any> {
    const response = await this.client.request(
      'GET',
      `/api/v1/strategies/pipeline/${deploymentId}/status`
    );
    return response.data;
  }

  /**
   * Pause/resume strategy
   */
  async pauseStrategy(strategyId: string): Promise<void> {
    await this.client.request(
      'POST',
      `/api/v1/strategies/${strategyId}/pause`
    );
  }

  async resumeStrategy(strategyId: string): Promise<void> {
    await this.client.request(
      'POST',
      `/api/v1/strategies/${strategyId}/resume`
    );
  }

  /**
   * Rollback strategy
   */
  async rollback(deploymentId: string): Promise<any> {
    const response = await this.client.request(
      'POST',
      `/api/v1/strategies/rollback/${deploymentId}`
    );
    return response.data;
  }
}

/**
 * Analytics API client
 */
export class AnalyticsAPI {
  constructor(private client: NautilusClient) {}

  /**
   * Get performance analytics
   */
  async getPerformance(portfolioId?: string, params?: any): Promise<PerformanceAnalytics> {
    const queryParams = { portfolio_id: portfolioId, ...params };
    const response = await this.client.request<PerformanceAnalytics>(
      'GET',
      '/api/v1/analytics/performance',
      { params: queryParams }
    );
    return response.data;
  }

  /**
   * Get risk analytics
   */
  async getRisk(portfolioId: string): Promise<RiskAnalytics> {
    const response = await this.client.request<RiskAnalytics>(
      'GET',
      `/api/v1/analytics/risk/${portfolioId}`
    );
    return response.data;
  }

  /**
   * Get strategy analytics
   */
  async getStrategyAnalytics(strategyId: string): Promise<any> {
    const response = await this.client.request(
      'GET',
      `/api/v1/analytics/strategy/${strategyId}`
    );
    return response.data;
  }
}

/**
 * System API client
 */
export class SystemAPI {
  constructor(private client: NautilusClient) {}

  /**
   * Get system health
   */
  async getHealth(): Promise<HealthCheck> {
    const response = await this.client.request<HealthCheck>(
      'GET',
      '/health',
      { auth: false }
    );
    return response.data;
  }

  /**
   * Get system metrics
   */
  async getMetrics(): Promise<any> {
    const response = await this.client.request(
      'GET',
      '/api/v1/system/metrics'
    );
    return response.data;
  }

  /**
   * Get system status
   */
  async getStatus(): Promise<any> {
    const response = await this.client.request(
      'GET',
      '/api/v1/system/status'
    );
    return response.data;
  }
}

/**
 * Main Nautilus Trading Platform Client
 */
export class NautilusClient extends EventEmitter {
  private axiosInstance: AxiosInstance;
  private authManager: AuthManager;
  private wsClient: WebSocketClient;
  
  // API modules
  public readonly marketData: MarketDataAPI;
  public readonly risk: RiskManagementAPI;
  public readonly strategies: StrategyAPI;
  public readonly analytics: AnalyticsAPI;
  public readonly system: SystemAPI;

  constructor(config: Partial<NautilusConfig> = {}) {
    super();
    
    const fullConfig: NautilusConfig = {
      baseUrl: 'http://localhost:8001',
      timeout: 30000,
      maxRetries: 3,
      retryDelay: 1000,
      wsUrl: 'ws://localhost:8001',
      ...config
    };

    // Initialize HTTP client
    this.axiosInstance = axios.create({
      baseURL: fullConfig.baseUrl,
      timeout: fullConfig.timeout,
      headers: {
        'Content-Type': 'application/json',
        'User-Agent': 'Nautilus-TypeScript-SDK/3.0.0'
      }
    });

    // Initialize auth manager
    this.authManager = new AuthManager(fullConfig, this.axiosInstance);

    // Initialize WebSocket client
    this.wsClient = new WebSocketClient(fullConfig, this.authManager);

    // Initialize API modules
    this.marketData = new MarketDataAPI(this);
    this.risk = new RiskManagementAPI(this);
    this.strategies = new StrategyAPI(this);
    this.analytics = new AnalyticsAPI(this);
    this.system = new SystemAPI(this);

    // Setup request/response interceptors
    this.setupInterceptors();

    // Setup retry logic
    this.setupRetryLogic(fullConfig);
  }

  /**
   * Setup request/response interceptors
   */
  private setupInterceptors(): void {
    // Request interceptor - add auth headers
    this.axiosInstance.interceptors.request.use(
      async (config) => {
        // Add authentication headers if needed
        if (config.headers && !config.headers['skip-auth']) {
          const authHeaders = await this.authManager.getAuthHeaders();
          Object.assign(config.headers, authHeaders);
        }
        return config;
      },
      (error) => Promise.reject(error)
    );

    // Response interceptor - handle errors
    this.axiosInstance.interceptors.response.use(
      (response) => response,
      async (error: AxiosError) => {
        if (error.response?.status === 401) {
          // Try to refresh token
          try {
            await this.authManager.refreshToken();
            // Retry the original request
            return this.axiosInstance.request(error.config!);
          } catch (refreshError) {
            this.emit('authError', new AuthenticationError('Authentication failed'));
            throw new AuthenticationError('Authentication failed');
          }
        }

        return Promise.reject(this.handleError(error));
      }
    );
  }

  /**
   * Setup retry logic for failed requests
   */
  private setupRetryLogic(config: NautilusConfig): void {
    let retryCount = 0;
    
    this.axiosInstance.interceptors.response.use(
      (response) => response,
      async (error: AxiosError) => {
        if (retryCount < config.maxRetries && this.shouldRetry(error)) {
          retryCount++;
          await this.delay(config.retryDelay * retryCount);
          return this.axiosInstance.request(error.config!);
        }
        retryCount = 0;
        return Promise.reject(error);
      }
    );
  }

  /**
   * Determine if request should be retried
   */
  private shouldRetry(error: AxiosError): boolean {
    if (!error.response) return true; // Network error
    
    const status = error.response.status;
    return status >= 500 || status === 408 || status === 429;
  }

  /**
   * Delay utility for retry logic
   */
  private delay(ms: number): Promise<void> {
    return new Promise(resolve => setTimeout(resolve, ms));
  }

  /**
   * Handle and transform errors
   */
  private handleError(error: AxiosError): Error {
    if (!error.response) {
      return new NetworkError('Network error occurred');
    }

    const { status, data } = error.response;
    const errorData = data as any;

    switch (status) {
      case 401:
        return new AuthenticationError(errorData?.message || 'Authentication required');
      case 403:
        return new AuthenticationError(errorData?.message || 'Access forbidden');
      case 429:
        const retryAfter = error.response.headers['retry-after'];
        return new RateLimitError(
          errorData?.message || 'Rate limit exceeded',
          parseInt(retryAfter) || 60
        );
      case 422:
        return new ValidationError(errorData?.message || 'Validation failed', errorData?.details);
      default:
        return new NautilusException(
          errorData?.message || `HTTP ${status} error`,
          status,
          errorData
        );
    }
  }

  /**
   * Make authenticated API request
   */
  async request<T = any>(
    method: string,
    endpoint: string,
    options: {
      params?: any;
      data?: any;
      headers?: Record<string, string>;
      auth?: boolean;
    } = {}
  ): Promise<ApiResponse<T>> {
    const { params, data, headers = {}, auth = true } = options;

    if (!auth) {
      headers['skip-auth'] = 'true';
    }

    try {
      const response: AxiosResponse<T> = await this.axiosInstance.request({
        method: method.toUpperCase(),
        url: endpoint,
        params,
        data,
        headers
      });

      return {
        data: response.data,
        status: response.status,
        headers: response.headers
      };
    } catch (error) {
      this.emit('error', error);
      throw error;
    }
  }

  /**
   * Authentication methods
   */
  async login(credentials: LoginCredentials): Promise<LoginResponse> {
    const response = await this.request<LoginResponse>(
      'POST',
      '/api/v1/auth/login',
      { 
        data: credentials,
        auth: false 
      }
    );

    // Store tokens
    await this.authManager.setTokens(
      response.data.access_token,
      response.data.refresh_token
    );

    this.emit('authenticated', response.data);
    return response.data;
  }

  async logout(): Promise<void> {
    await this.authManager.clearTokens();
    this.emit('logout');
  }

  /**
   * WebSocket methods
   */
  async connectWebSocket(): Promise<void> {
    return this.wsClient.connect();
  }

  async disconnectWebSocket(): Promise<void> {
    return this.wsClient.disconnect();
  }

  /**
   * Subscribe to real-time market data
   */
  async subscribeToMarketData(
    symbols: string[],
    callback: (data: MarketData) => void
  ): Promise<void> {
    await this.wsClient.subscribe('market_data', { symbols }, callback);
  }

  /**
   * Subscribe to trade updates
   */
  async subscribeToTrades(callback: (data: any) => void): Promise<void> {
    await this.wsClient.subscribe('trade_updates', {}, callback);
  }

  /**
   * Subscribe to risk alerts
   */
  async subscribeToRiskAlerts(callback: (data: any) => void): Promise<void> {
    await this.wsClient.subscribe('risk_alerts', {}, callback);
  }

  /**
   * Get current authentication status
   */
  isAuthenticated(): boolean {
    return this.authManager.isAuthenticated();
  }

  /**
   * Get current access token
   */
  getAccessToken(): string | null {
    return this.authManager.getAccessToken();
  }

  /**
   * Close all connections and cleanup
   */
  async close(): Promise<void> {
    await this.wsClient.disconnect();
    this.removeAllListeners();
  }
}

// Export as default for convenience
export default NautilusClient;

// Usage example
/*
const client = new NautilusClient({
  baseUrl: 'http://localhost:8001',
  timeout: 30000
});

// Authentication
await client.login({
  username: 'trader@nautilus.com',
  password: 'password'
});

// Get market data
const quote = await client.marketData.getQuote('AAPL');
console.log(`AAPL: $${quote.price}`);

// Create risk limit
const riskLimit = await client.risk.createLimit({
  type: 'position_limit',
  value: 1000000,
  symbol: 'AAPL',
  warning_threshold: 0.8
});

// Deploy strategy
const deployment = await client.strategies.deploy({
  name: 'EMA_Cross',
  version: '1.0.0',
  description: 'EMA crossover strategy',
  parameters: {
    fast_ema: 12,
    slow_ema: 26
  }
});

// Real-time data
await client.subscribeToMarketData(['AAPL', 'GOOGL'], (data) => {
  console.log('Market update:', data);
});

// Cleanup
await client.close();
*/