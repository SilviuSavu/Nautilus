/**
 * Direct Access Client for Critical Trading Operations
 * Provides direct connections to critical engines bypassing the API gateway
 * for sub-50ms latency requirements in trading operations.
 */

interface EngineConfig {
  name: string;
  url: string;
  port: number;
  maxLatencyMs: number;
  priority: 'critical' | 'high' | 'normal';
  retryCount: number;
  timeoutMs: number;
}

interface DirectResponse<T = any> {
  success: boolean;
  data?: T;
  error?: string;
  metadata: {
    responseTimeMs: number;
    engineUsed: string;
    directAccess: boolean;
    fallbackUsed: boolean;
    retryCount: number;
    timestamp: number;
  };
}

interface PerformanceMetrics {
  totalRequests: number;
  successfulRequests: number;
  failedRequests: number;
  averageLatencyMs: number;
  p95LatencyMs: number;
  p99LatencyMs: number;
  directAccessRate: number;
  fallbackRate: number;
  lastUpdated: number;
}

interface HealthStatus {
  engine: string;
  status: 'healthy' | 'degraded' | 'unhealthy';
  latencyMs: number;
  successRate: number;
  lastCheck: number;
}

enum OperationType {
  CRITICAL_TRADING = 'critical_trading',
  REAL_TIME_ANALYTICS = 'real_time_analytics',
  RISK_CALCULATION = 'risk_calculation',
  ML_PREDICTION = 'ml_prediction'
}

class LatencyMonitor {
  private latencies: number[] = [];
  private maxSamples = 1000;
  
  record(latency: number): void {
    this.latencies.push(latency);
    if (this.latencies.length > this.maxSamples) {
      this.latencies = this.latencies.slice(-this.maxSamples);
    }
  }
  
  getAverage(): number {
    if (this.latencies.length === 0) return 0;
    return this.latencies.reduce((sum, lat) => sum + lat, 0) / this.latencies.length;
  }
  
  getPercentile(percentile: number): number {
    if (this.latencies.length === 0) return 0;
    const sorted = [...this.latencies].sort((a, b) => a - b);
    const index = Math.ceil((percentile / 100) * sorted.length) - 1;
    return sorted[Math.max(0, index)];
  }
  
  clear(): void {
    this.latencies = [];
  }
}

class DirectAccessClient {
  private engines: Map<string, EngineConfig> = new Map();
  private healthStatus: Map<string, HealthStatus> = new Map();
  private performanceMetrics: Map<string, PerformanceMetrics> = new Map();
  private latencyMonitors: Map<string, LatencyMonitor> = new Map();
  private healthCheckInterval: NodeJS.Timeout | null = null;
  private initialized = false;
  
  // Gateway fallback client
  private gatewayBaseUrl: string;
  
  constructor(gatewayBaseUrl = 'http://localhost:8001') {
    this.gatewayBaseUrl = gatewayBaseUrl;
    this.initializeEngines();
  }
  
  private initializeEngines(): void {
    const criticalEngines: EngineConfig[] = [
      {
        name: 'strategy',
        url: 'http://localhost:8700',
        port: 8700,
        maxLatencyMs: 50,
        priority: 'critical',
        retryCount: 1,
        timeoutMs: 5000
      },
      {
        name: 'risk',
        url: 'http://localhost:8200',
        port: 8200,
        maxLatencyMs: 100,
        priority: 'critical',
        retryCount: 2,
        timeoutMs: 10000
      },
      {
        name: 'analytics',
        url: 'http://localhost:8100',
        port: 8100,
        maxLatencyMs: 200,
        priority: 'high',
        retryCount: 2,
        timeoutMs: 15000
      },
      {
        name: 'ml',
        url: 'http://localhost:8400',
        port: 8400,
        maxLatencyMs: 300,
        priority: 'high',
        retryCount: 1,
        timeoutMs: 30000
      }
    ];
    
    criticalEngines.forEach(engine => {
      this.engines.set(engine.name, engine);
      this.latencyMonitors.set(engine.name, new LatencyMonitor());
      this.performanceMetrics.set(engine.name, {
        totalRequests: 0,
        successfulRequests: 0,
        failedRequests: 0,
        averageLatencyMs: 0,
        p95LatencyMs: 0,
        p99LatencyMs: 0,
        directAccessRate: 100,
        fallbackRate: 0,
        lastUpdated: Date.now()
      });
    });
    
    console.log('🚀 Direct Access Client initialized with', criticalEngines.length, 'critical engines');
  }
  
  async initialize(): Promise<void> {
    if (this.initialized) return;
    
    // Perform initial health check
    await this.checkAllEnginesHealth();
    
    // Start periodic health monitoring
    this.healthCheckInterval = setInterval(() => {
      this.checkAllEnginesHealth().catch(console.error);
    }, 30000); // Check every 30 seconds
    
    this.initialized = true;
    console.log('✅ Direct Access Client fully initialized');
  }
  
  async shutdown(): Promise<void> {
    if (this.healthCheckInterval) {
      clearInterval(this.healthCheckInterval);
      this.healthCheckInterval = null;
    }
    this.initialized = false;
    console.log('🔄 Direct Access Client shutdown complete');
  }
  
  private async checkAllEnginesHealth(): Promise<void> {
    const healthChecks = Array.from(this.engines.values()).map(engine =>
      this.checkEngineHealth(engine.name)
    );
    
    await Promise.allSettled(healthChecks);
  }
  
  private async checkEngineHealth(engineName: string): Promise<void> {
    const engine = this.engines.get(engineName);
    if (!engine) return;
    
    const startTime = performance.now();
    
    try {
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), 5000); // 5s health check timeout
      
      const response = await fetch(`${engine.url}/health`, {
        method: 'GET',
        signal: controller.signal,
        headers: {
          'Content-Type': 'application/json',
        }
      });
      
      clearTimeout(timeoutId);
      const latency = performance.now() - startTime;
      
      let status: 'healthy' | 'degraded' | 'unhealthy' = 'unhealthy';
      
      if (response.ok) {
        if (latency <= engine.maxLatencyMs) {
          status = 'healthy';
        } else if (latency <= engine.maxLatencyMs * 2) {
          status = 'degraded';
        }
      }
      
      this.healthStatus.set(engineName, {
        engine: engineName,
        status,
        latencyMs: latency,
        successRate: status === 'healthy' ? 100 : (status === 'degraded' ? 75 : 0),
        lastCheck: Date.now()
      });
      
      if (status === 'healthy') {
        console.log(`✅ Engine ${engineName} healthy (${latency.toFixed(1)}ms)`);
      } else if (status === 'degraded') {
        console.warn(`⚠️ Engine ${engineName} degraded (${latency.toFixed(1)}ms)`);
      } else {
        console.error(`❌ Engine ${engineName} unhealthy`);
      }
      
    } catch (error) {
      const latency = performance.now() - startTime;
      
      this.healthStatus.set(engineName, {
        engine: engineName,
        status: 'unhealthy',
        latencyMs: latency,
        successRate: 0,
        lastCheck: Date.now()
      });
      
      console.error(`❌ Health check failed for ${engineName}:`, error);
    }
  }
  
  private isEngineHealthy(engineName: string): boolean {
    const health = this.healthStatus.get(engineName);
    return health ? health.status === 'healthy' : false;
  }
  
  private async executeDirectRequest<T>(
    engineName: string,
    endpoint: string,
    options: RequestInit = {}
  ): Promise<DirectResponse<T>> {
    const engine = this.engines.get(engineName);
    if (!engine) {
      throw new Error(`Unknown engine: ${engineName}`);
    }
    
    const startTime = performance.now();
    let retryCount = 0;
    let lastError: Error | null = null;
    
    // Update metrics
    const metrics = this.performanceMetrics.get(engineName)!;
    metrics.totalRequests++;
    
    for (let attempt = 0; attempt <= engine.retryCount; attempt++) {
      try {
        const controller = new AbortController();
        const timeoutId = setTimeout(() => controller.abort(), engine.timeoutMs);
        
        const response = await fetch(`${engine.url}${endpoint}`, {
          ...options,
          signal: controller.signal,
          headers: {
            'Content-Type': 'application/json',
            ...options.headers
          }
        });
        
        clearTimeout(timeoutId);
        const responseTime = performance.now() - startTime;
        
        if (!response.ok) {
          throw new Error(`HTTP ${response.status}: ${response.statusText}`);
        }
        
        const data = await response.json();
        
        // Record successful request
        metrics.successfulRequests++;
        this.latencyMonitors.get(engineName)!.record(responseTime);
        this.updateMetrics(engineName);
        
        console.log(`⚡ Direct access to ${engineName}${endpoint} (${responseTime.toFixed(1)}ms)`);
        
        return {
          success: true,
          data,
          metadata: {
            responseTimeMs: responseTime,
            engineUsed: engineName,
            directAccess: true,
            fallbackUsed: false,
            retryCount: attempt,
            timestamp: Date.now()
          }
        };
        
      } catch (error) {
        lastError = error as Error;
        retryCount = attempt + 1;
        
        if (attempt < engine.retryCount) {
          // Brief exponential backoff
          const delay = Math.min(100 * Math.pow(2, attempt), 1000);
          await new Promise(resolve => setTimeout(resolve, delay));
          console.warn(`🔄 Retrying ${engineName}${endpoint} (attempt ${attempt + 2})`);
        }
      }
    }
    
    // All direct attempts failed - record failure
    metrics.failedRequests++;
    this.updateMetrics(engineName);
    
    const responseTime = performance.now() - startTime;
    console.error(`❌ Direct access failed for ${engineName}${endpoint} after ${retryCount} attempts`);
    
    return {
      success: false,
      error: lastError?.message || 'Unknown error',
      metadata: {
        responseTimeMs: responseTime,
        engineUsed: engineName,
        directAccess: true,
        fallbackUsed: false,
        retryCount: retryCount,
        timestamp: Date.now()
      }
    };
  }
  
  private async executeGatewayFallback<T>(
    engineName: string,
    endpoint: string,
    options: RequestInit = {}
  ): Promise<DirectResponse<T>> {
    const startTime = performance.now();
    
    try {
      const response = await fetch(`${this.gatewayBaseUrl}/api/v1/${engineName}${endpoint}`, {
        ...options,
        headers: {
          'Content-Type': 'application/json',
          ...options.headers
        }
      });
      
      const responseTime = performance.now() - startTime;
      
      if (!response.ok) {
        throw new Error(`Gateway HTTP ${response.status}: ${response.statusText}`);
      }
      
      const data = await response.json();
      
      console.log(`🔄 Gateway fallback for ${engineName}${endpoint} (${responseTime.toFixed(1)}ms)`);
      
      return {
        success: true,
        data,
        metadata: {
          responseTimeMs: responseTime,
          engineUsed: engineName,
          directAccess: false,
          fallbackUsed: true,
          retryCount: 0,
          timestamp: Date.now()
        }
      };
      
    } catch (error) {
      const responseTime = performance.now() - startTime;
      
      console.error(`❌ Gateway fallback failed for ${engineName}${endpoint}:`, error);
      
      return {
        success: false,
        error: (error as Error).message,
        metadata: {
          responseTimeMs: responseTime,
          engineUsed: engineName,
          directAccess: false,
          fallbackUsed: true,
          retryCount: 0,
          timestamp: Date.now()
        }
      };
    }
  }
  
  private updateMetrics(engineName: string): void {
    const metrics = this.performanceMetrics.get(engineName)!;
    const latencyMonitor = this.latencyMonitors.get(engineName)!;
    
    metrics.averageLatencyMs = latencyMonitor.getAverage();
    metrics.p95LatencyMs = latencyMonitor.getPercentile(95);
    metrics.p99LatencyMs = latencyMonitor.getPercentile(99);
    
    if (metrics.totalRequests > 0) {
      metrics.directAccessRate = (metrics.successfulRequests / metrics.totalRequests) * 100;
      metrics.fallbackRate = 100 - metrics.directAccessRate;
    }
    
    metrics.lastUpdated = Date.now();
  }
  
  // Public API methods
  
  async request<T = any>(
    engineName: string,
    endpoint: string,
    operationType: OperationType,
    options: RequestInit = {}
  ): Promise<DirectResponse<T>> {
    if (!this.initialized) {
      await this.initialize();
    }
    
    // Check if this is a critical engine with direct access available
    const engine = this.engines.get(engineName);
    if (!engine) {
      // Non-critical engine - use gateway
      return this.executeGatewayFallback<T>(engineName, endpoint, options);
    }
    
    // For critical engines, try direct access first
    if (this.isEngineHealthy(engineName)) {
      const directResult = await this.executeDirectRequest<T>(engineName, endpoint, options);
      
      // If direct access succeeded and meets latency requirements, return it
      if (directResult.success && directResult.metadata.responseTimeMs <= engine.maxLatencyMs) {
        return directResult;
      }
      
      // Direct access failed or too slow - try fallback
      console.warn(`⚠️ Direct access for ${engineName} failed or too slow, trying fallback`);
    }
    
    // Use gateway fallback
    return this.executeGatewayFallback<T>(engineName, endpoint, options);
  }
  
  // Convenience methods for common operations
  
  async executeTradingOrder(orderData: any): Promise<DirectResponse> {
    return this.request('strategy', '/execute', OperationType.CRITICAL_TRADING, {
      method: 'POST',
      body: JSON.stringify(orderData)
    });
  }
  
  async calculateRisk(portfolioData: any): Promise<DirectResponse> {
    return this.request('risk', '/calculate-var', OperationType.RISK_CALCULATION, {
      method: 'POST',
      body: JSON.stringify(portfolioData)
    });
  }
  
  async getAnalytics(params?: any): Promise<DirectResponse> {
    const query = params ? `?${new URLSearchParams(params).toString()}` : '';
    return this.request('analytics', `/real-time${query}`, OperationType.REAL_TIME_ANALYTICS);
  }
  
  async predictPrice(marketData: any): Promise<DirectResponse> {
    return this.request('ml', '/predict', OperationType.ML_PREDICTION, {
      method: 'POST',
      body: JSON.stringify(marketData)
    });
  }
  
  // Status and metrics methods
  
  getEngineHealth(engineName?: string): HealthStatus[] | HealthStatus | null {
    if (engineName) {
      return this.healthStatus.get(engineName) || null;
    }
    return Array.from(this.healthStatus.values());
  }
  
  getPerformanceMetrics(engineName?: string): PerformanceMetrics[] | PerformanceMetrics | null {
    if (engineName) {
      return this.performanceMetrics.get(engineName) || null;
    }
    return Array.from(this.performanceMetrics.values());
  }
  
  getSystemSummary() {
    const engines = Array.from(this.engines.keys());
    const healthyEngines = engines.filter(name => this.isEngineHealthy(name));
    const totalMetrics = Array.from(this.performanceMetrics.values());
    
    const totalRequests = totalMetrics.reduce((sum, m) => sum + m.totalRequests, 0);
    const totalSuccessful = totalMetrics.reduce((sum, m) => sum + m.successfulRequests, 0);
    const avgLatency = totalMetrics.reduce((sum, m) => sum + m.averageLatencyMs, 0) / totalMetrics.length || 0;
    
    return {
      totalEngines: engines.length,
      healthyEngines: healthyEngines.length,
      systemHealthRate: (healthyEngines.length / engines.length) * 100,
      totalRequests,
      successRate: totalRequests > 0 ? (totalSuccessful / totalRequests) * 100 : 0,
      averageLatencyMs: avgLatency,
      directAccessAvailable: engines.length > 0,
      lastUpdated: Date.now()
    };
  }
  
  async forceHealthCheck(engineName?: string): Promise<void> {
    if (engineName) {
      await this.checkEngineHealth(engineName);
    } else {
      await this.checkAllEnginesHealth();
    }
  }
}

// Export singleton instance
export const directAccessClient = new DirectAccessClient();
export { OperationType, type DirectResponse, type HealthStatus, type PerformanceMetrics };
export default DirectAccessClient;