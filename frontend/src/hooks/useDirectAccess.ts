/**
 * React Hook for Direct Access Client Integration
 * Provides React components with direct engine access capabilities
 * and performance monitoring for critical trading operations.
 */

import { useState, useEffect, useCallback, useRef } from 'react';
import { directAccessClient, OperationType, type DirectResponse, type HealthStatus, type PerformanceMetrics } from '../services/DirectAccessClient';

interface UseDirectAccessOptions {
  enableHealthMonitoring?: boolean;
  healthCheckInterval?: number;
  enablePerformanceTracking?: boolean;
  autoInitialize?: boolean;
}

interface DirectAccessState {
  initialized: boolean;
  connecting: boolean;
  systemHealth: {
    totalEngines: number;
    healthyEngines: number;
    systemHealthRate: number;
    averageLatencyMs: number;
    directAccessAvailable: boolean;
  };
  engineHealth: HealthStatus[];
  performanceMetrics: PerformanceMetrics[];
  lastError: string | null;
}

interface DirectAccessActions {
  initialize: () => Promise<void>;
  shutdown: () => Promise<void>;
  executeTradingOrder: (orderData: any) => Promise<DirectResponse>;
  calculateRisk: (portfolioData: any) => Promise<DirectResponse>;
  getAnalytics: (params?: any) => Promise<DirectResponse>;
  predictPrice: (marketData: any) => Promise<DirectResponse>;
  request: <T = any>(engineName: string, endpoint: string, operationType: OperationType, options?: RequestInit) => Promise<DirectResponse<T>>;
  forceHealthCheck: (engineName?: string) => Promise<void>;
  clearError: () => void;
}

const defaultOptions: UseDirectAccessOptions = {
  enableHealthMonitoring: true,
  healthCheckInterval: 30000, // 30 seconds
  enablePerformanceTracking: true,
  autoInitialize: true
};

export function useDirectAccess(options: UseDirectAccessOptions = {}): [DirectAccessState, DirectAccessActions] {
  const opts = { ...defaultOptions, ...options };
  const [state, setState] = useState<DirectAccessState>({
    initialized: false,
    connecting: false,
    systemHealth: {
      totalEngines: 0,
      healthyEngines: 0,
      systemHealthRate: 0,
      averageLatencyMs: 0,
      directAccessAvailable: false
    },
    engineHealth: [],
    performanceMetrics: [],
    lastError: null
  });
  
  const healthCheckIntervalRef = useRef<NodeJS.Timeout | null>(null);
  const isMountedRef = useRef(true);
  
  // Update state helper
  const updateState = useCallback((updates: Partial<DirectAccessState>) => {
    if (isMountedRef.current) {
      setState(prev => ({ ...prev, ...updates }));
    }
  }, []);
  
  // Initialize direct access client
  const initialize = useCallback(async () => {
    if (state.initialized || state.connecting) return;
    
    try {
      updateState({ connecting: true, lastError: null });
      
      await directAccessClient.initialize();
      
      // Get initial system status
      const systemSummary = directAccessClient.getSystemSummary();
      const engineHealth = directAccessClient.getEngineHealth() as HealthStatus[];
      const performanceMetrics = directAccessClient.getPerformanceMetrics() as PerformanceMetrics[];
      
      updateState({
        initialized: true,
        connecting: false,
        systemHealth: systemSummary,
        engineHealth: engineHealth || [],
        performanceMetrics: performanceMetrics || []
      });
      
      console.log('✅ Direct Access Hook initialized');
      
    } catch (error) {
      console.error('❌ Failed to initialize Direct Access:', error);
      updateState({
        connecting: false,
        lastError: error instanceof Error ? error.message : 'Initialization failed'
      });
    }
  }, [state.initialized, state.connecting, updateState]);
  
  // Shutdown direct access client
  const shutdown = useCallback(async () => {
    if (!state.initialized) return;
    
    try {
      await directAccessClient.shutdown();
      
      if (healthCheckIntervalRef.current) {
        clearInterval(healthCheckIntervalRef.current);
        healthCheckIntervalRef.current = null;
      }
      
      updateState({
        initialized: false,
        connecting: false,
        systemHealth: {
          totalEngines: 0,
          healthyEngines: 0,
          systemHealthRate: 0,
          averageLatencyMs: 0,
          directAccessAvailable: false
        },
        engineHealth: [],
        performanceMetrics: []
      });
      
      console.log('🔄 Direct Access Hook shutdown');
      
    } catch (error) {
      console.error('❌ Error during shutdown:', error);
      updateState({
        lastError: error instanceof Error ? error.message : 'Shutdown failed'
      });
    }
  }, [state.initialized, updateState]);
  
  // Update system status
  const updateSystemStatus = useCallback(async () => {
    if (!state.initialized) return;
    
    try {
      const systemSummary = directAccessClient.getSystemSummary();
      const engineHealth = directAccessClient.getEngineHealth() as HealthStatus[];
      const performanceMetrics = directAccessClient.getPerformanceMetrics() as PerformanceMetrics[];
      
      updateState({
        systemHealth: systemSummary,
        engineHealth: engineHealth || [],
        performanceMetrics: performanceMetrics || []
      });
      
    } catch (error) {
      console.error('❌ Error updating system status:', error);
    }
  }, [state.initialized, updateState]);
  
  // Force health check
  const forceHealthCheck = useCallback(async (engineName?: string) => {
    if (!state.initialized) {
      throw new Error('Direct Access not initialized');
    }
    
    try {
      await directAccessClient.forceHealthCheck(engineName);
      await updateSystemStatus();
    } catch (error) {
      console.error('❌ Health check failed:', error);
      updateState({
        lastError: error instanceof Error ? error.message : 'Health check failed'
      });
    }
  }, [state.initialized, updateState, updateSystemStatus]);
  
  // Trading operations with error handling
  const executeTradingOrder = useCallback(async (orderData: any): Promise<DirectResponse> => {
    if (!state.initialized) {
      throw new Error('Direct Access not initialized');
    }
    
    try {
      const result = await directAccessClient.executeTradingOrder(orderData);
      
      // Update performance metrics after request
      setTimeout(updateSystemStatus, 100);
      
      return result;
    } catch (error) {
      updateState({
        lastError: error instanceof Error ? error.message : 'Trading order failed'
      });
      throw error;
    }
  }, [state.initialized, updateState, updateSystemStatus]);
  
  const calculateRisk = useCallback(async (portfolioData: any): Promise<DirectResponse> => {
    if (!state.initialized) {
      throw new Error('Direct Access not initialized');
    }
    
    try {
      const result = await directAccessClient.calculateRisk(portfolioData);
      setTimeout(updateSystemStatus, 100);
      return result;
    } catch (error) {
      updateState({
        lastError: error instanceof Error ? error.message : 'Risk calculation failed'
      });
      throw error;
    }
  }, [state.initialized, updateState, updateSystemStatus]);
  
  const getAnalytics = useCallback(async (params?: any): Promise<DirectResponse> => {
    if (!state.initialized) {
      throw new Error('Direct Access not initialized');
    }
    
    try {
      const result = await directAccessClient.getAnalytics(params);
      setTimeout(updateSystemStatus, 100);
      return result;
    } catch (error) {
      updateState({
        lastError: error instanceof Error ? error.message : 'Analytics request failed'
      });
      throw error;
    }
  }, [state.initialized, updateState, updateSystemStatus]);
  
  const predictPrice = useCallback(async (marketData: any): Promise<DirectResponse> => {
    if (!state.initialized) {
      throw new Error('Direct Access not initialized');
    }
    
    try {
      const result = await directAccessClient.predictPrice(marketData);
      setTimeout(updateSystemStatus, 100);
      return result;
    } catch (error) {
      updateState({
        lastError: error instanceof Error ? error.message : 'Price prediction failed'
      });
      throw error;
    }
  }, [state.initialized, updateState, updateSystemStatus]);
  
  // Generic request method
  const request = useCallback(async <T = any>(
    engineName: string,
    endpoint: string,
    operationType: OperationType,
    options?: RequestInit
  ): Promise<DirectResponse<T>> => {
    if (!state.initialized) {
      throw new Error('Direct Access not initialized');
    }
    
    try {
      const result = await directAccessClient.request<T>(engineName, endpoint, operationType, options);
      setTimeout(updateSystemStatus, 100);
      return result;
    } catch (error) {
      updateState({
        lastError: error instanceof Error ? error.message : 'Request failed'
      });
      throw error;
    }
  }, [state.initialized, updateState, updateSystemStatus]);
  
  // Clear error
  const clearError = useCallback(() => {
    updateState({ lastError: null });
  }, [updateState]);
  
  // Auto-initialize on mount
  useEffect(() => {
    if (opts.autoInitialize && !state.initialized && !state.connecting) {
      initialize().catch(console.error);
    }
  }, [opts.autoInitialize, state.initialized, state.connecting, initialize]);
  
  // Set up health monitoring interval
  useEffect(() => {
    if (state.initialized && opts.enableHealthMonitoring) {
      healthCheckIntervalRef.current = setInterval(() => {
        updateSystemStatus().catch(console.error);
      }, opts.healthCheckInterval);
      
      return () => {
        if (healthCheckIntervalRef.current) {
          clearInterval(healthCheckIntervalRef.current);
          healthCheckIntervalRef.current = null;
        }
      };
    }
  }, [state.initialized, opts.enableHealthMonitoring, opts.healthCheckInterval, updateSystemStatus]);
  
  // Cleanup on unmount
  useEffect(() => {
    isMountedRef.current = true;
    
    return () => {
      isMountedRef.current = false;
      if (healthCheckIntervalRef.current) {
        clearInterval(healthCheckIntervalRef.current);
        healthCheckIntervalRef.current = null;
      }
    };
  }, []);
  
  // Return state and actions
  return [
    state,
    {
      initialize,
      shutdown,
      executeTradingOrder,
      calculateRisk,
      getAnalytics,
      predictPrice,
      request,
      forceHealthCheck,
      clearError
    }
  ];
}

// Convenience hook for trading operations only
export function useDirectTrading() {
  const [state, actions] = useDirectAccess({
    enableHealthMonitoring: true,
    healthCheckInterval: 15000, // More frequent for trading
    enablePerformanceTracking: true,
    autoInitialize: true
  });
  
  return {
    // State
    ready: state.initialized && state.systemHealth.directAccessAvailable,
    connecting: state.connecting,
    latency: state.systemHealth.averageLatencyMs,
    healthRate: state.systemHealth.systemHealthRate,
    error: state.lastError,
    
    // Critical operations only
    executeOrder: actions.executeTradingOrder,
    calculateRisk: actions.calculateRisk,
    
    // Utilities
    forceHealthCheck: actions.forceHealthCheck,
    clearError: actions.clearError
  };
}

// Performance monitoring hook
export function useDirectAccessMetrics() {
  const [state] = useDirectAccess({
    enableHealthMonitoring: true,
    enablePerformanceTracking: true,
    autoInitialize: true
  });
  
  // Calculate aggregate metrics
  const aggregateMetrics = state.performanceMetrics.reduce((acc, metric) => {
    acc.totalRequests += metric.totalRequests;
    acc.successfulRequests += metric.successfulRequests;
    acc.failedRequests += metric.failedRequests;
    acc.averageLatency += metric.averageLatencyMs;
    acc.p95Latency += metric.p95LatencyMs;
    acc.p99Latency += metric.p99LatencyMs;
    acc.directAccessRate += metric.directAccessRate;
    return acc;
  }, {
    totalRequests: 0,
    successfulRequests: 0,
    failedRequests: 0,
    averageLatency: 0,
    p95Latency: 0,
    p99Latency: 0,
    directAccessRate: 0
  });
  
  const engineCount = state.performanceMetrics.length;
  if (engineCount > 0) {
    aggregateMetrics.averageLatency /= engineCount;
    aggregateMetrics.p95Latency /= engineCount;
    aggregateMetrics.p99Latency /= engineCount;
    aggregateMetrics.directAccessRate /= engineCount;
  }
  
  const successRate = aggregateMetrics.totalRequests > 0 
    ? (aggregateMetrics.successfulRequests / aggregateMetrics.totalRequests) * 100 
    : 0;
  
  return {
    engines: state.performanceMetrics,
    aggregate: {
      ...aggregateMetrics,
      successRate,
      engineCount,
      healthyEngines: state.systemHealth.healthyEngines
    },
    systemHealth: state.systemHealth,
    lastUpdated: Date.now()
  };
}

export default useDirectAccess;