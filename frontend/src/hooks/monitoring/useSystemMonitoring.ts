/**
 * Story 5.2: System Performance Monitoring Hook
 * React hook for managing system monitoring state and API calls
 */

import { useState, useEffect, useCallback, useRef } from 'react';
import { systemMonitoringService } from '../../services/monitoring/SystemMonitoringService';
import {
  LatencyMonitoringResponse,
  SystemMonitoringResponse,
  ConnectionMonitoringResponse,
  AlertsMonitoringResponse,
  PerformanceTrendsResponse,
  AlertConfigurationRequest,
  AlertConfigurationResponse,
  MonitoringState,
  UseMonitoringConfig
} from '../../types/monitoring';

interface UseSystemMonitoringReturn extends MonitoringState {
  // Data refresh functions
  refreshLatencyMetrics: (venue?: string, timeframe?: string) => Promise<void>;
  refreshSystemMetrics: (metrics?: string[], period?: string) => Promise<void>;
  refreshConnectionMetrics: (venue?: string, includeHistory?: boolean) => Promise<void>;
  refreshAlerts: (status?: string, severity?: string) => Promise<void>;
  refreshPerformanceTrends: (period?: string) => Promise<void>;
  refreshAllMetrics: () => Promise<void>;

  // Alert management
  configureAlert: (request: AlertConfigurationRequest) => Promise<AlertConfigurationResponse>;

  // Control functions
  startAutoRefresh: () => void;
  stopAutoRefresh: () => void;
  isAutoRefreshActive: boolean;

  // Real-time metrics
  realTimeLatency: number | null;
  realTimeCpuUsage: number | null;
  realTimeMemoryUsage: number | null;
}

const DEFAULT_CONFIG: UseMonitoringConfig = {
  refreshInterval: 10000, // 10 seconds
  autoRefresh: true,
  includeHistory: true,
  venueFilter: ['all']
};

export const useSystemMonitoring = (
  config: UseMonitoringConfig = {}
): UseSystemMonitoringReturn => {
  const configWithDefaults = { ...DEFAULT_CONFIG, ...config };
  const intervalRef = useRef<NodeJS.Timeout | null>(null);
  const [isAutoRefreshActive, setIsAutoRefreshActive] = useState(configWithDefaults.autoRefresh || false);

  // Core monitoring state
  const [state, setState] = useState<MonitoringState>({
    latencyMetrics: [],
    systemMetrics: null,
    connectionQuality: [],
    activeAlerts: [],
    performanceTrends: null,
    loading: false,
    error: null,
    lastUpdate: null
  });

  // Real-time metrics state
  const [realTimeMetrics, setRealTimeMetrics] = useState({
    realTimeLatency: null as number | null,
    realTimeCpuUsage: null as number | null,
    realTimeMemoryUsage: null as number | null
  });

  // Error handling helper
  const handleError = useCallback((error: Error, operation: string) => {
    console.error(`Error in ${operation}:`, error);
    setState(prev => ({
      ...prev,
      error: error.message,
      loading: false
    }));
  }, []);

  // Refresh latency metrics
  const refreshLatencyMetrics = useCallback(async (
    venue: string = 'all',
    timeframe: string = '1h'
  ) => {
    try {
      setState(prev => ({ ...prev, loading: true, error: null }));
      
      const response: LatencyMonitoringResponse = await systemMonitoringService.getLatencyMetrics(venue, timeframe);
      
      setState(prev => ({
        ...prev,
        latencyMetrics: response.venue_latencies,
        loading: false,
        lastUpdate: new Date()
      }));

      // Update real-time latency (average of all venues)
      const avgLatency = response.venue_latencies.reduce((sum, venue) => 
        sum + venue.order_execution_latency.avg_ms, 0
      ) / response.venue_latencies.length;

      setRealTimeMetrics(prev => ({
        ...prev,
        realTimeLatency: avgLatency
      }));

    } catch (error) {
      handleError(error as Error, 'refreshLatencyMetrics');
    }
  }, [handleError]);

  // Refresh system metrics
  const refreshSystemMetrics = useCallback(async (
    metrics: string[] = ['cpu', 'memory', 'network'],
    period: string = 'realtime'
  ) => {
    try {
      const response: SystemMonitoringResponse = await systemMonitoringService.getSystemMetrics(metrics, period);
      
      setState(prev => ({
        ...prev,
        systemMetrics: response,
        lastUpdate: new Date()
      }));

      // Update real-time CPU and memory
      setRealTimeMetrics(prev => ({
        ...prev,
        realTimeCpuUsage: response.cpu_metrics.usage_percent,
        realTimeMemoryUsage: response.memory_metrics.usage_percent
      }));

    } catch (error) {
      handleError(error as Error, 'refreshSystemMetrics');
    }
  }, [handleError]);

  // Refresh connection metrics
  const refreshConnectionMetrics = useCallback(async (
    venue: string = 'all',
    includeHistory: boolean = true
  ) => {
    try {
      const response: ConnectionMonitoringResponse = await systemMonitoringService.getConnectionMetrics(venue, includeHistory);
      
      setState(prev => ({
        ...prev,
        connectionQuality: response.venue_connections,
        lastUpdate: new Date()
      }));

    } catch (error) {
      handleError(error as Error, 'refreshConnectionMetrics');
    }
  }, [handleError]);

  // Refresh alerts
  const refreshAlerts = useCallback(async (
    status: string = 'active',
    severity: string = 'all'
  ) => {
    try {
      const response: AlertsMonitoringResponse = await systemMonitoringService.getAlerts(status, severity);
      
      setState(prev => ({
        ...prev,
        activeAlerts: response.active_alerts,
        lastUpdate: new Date()
      }));

    } catch (error) {
      handleError(error as Error, 'refreshAlerts');
    }
  }, [handleError]);

  // Refresh performance trends
  const refreshPerformanceTrends = useCallback(async (
    period: string = '7d'
  ) => {
    try {
      const response: PerformanceTrendsResponse = await systemMonitoringService.getPerformanceTrends(period);
      
      setState(prev => ({
        ...prev,
        performanceTrends: response,
        lastUpdate: new Date()
      }));

    } catch (error) {
      handleError(error as Error, 'refreshPerformanceTrends');
    }
  }, [handleError]);

  // Refresh all metrics
  const refreshAllMetrics = useCallback(async () => {
    setState(prev => ({ ...prev, loading: true, error: null }));
    
    try {
      await Promise.all([
        refreshLatencyMetrics(),
        refreshSystemMetrics(),
        refreshConnectionMetrics(),
        refreshAlerts(),
        refreshPerformanceTrends()
      ]);
    } catch (error) {
      handleError(error as Error, 'refreshAllMetrics');
    } finally {
      setState(prev => ({ ...prev, loading: false }));
    }
  }, [refreshLatencyMetrics, refreshSystemMetrics, refreshConnectionMetrics, refreshAlerts, refreshPerformanceTrends, handleError]);

  // Configure alert
  const configureAlert = useCallback(async (
    request: AlertConfigurationRequest
  ): Promise<AlertConfigurationResponse> => {
    try {
      const response = await systemMonitoringService.configureAlert(request);
      
      // Refresh alerts after configuration
      if (response.status === 'created') {
        await refreshAlerts();
      }
      
      return response;
    } catch (error) {
      handleError(error as Error, 'configureAlert');
      throw error;
    }
  }, [refreshAlerts, handleError]);

  // Start auto refresh
  const startAutoRefresh = useCallback(() => {
    if (intervalRef.current) {
      clearInterval(intervalRef.current);
    }

    intervalRef.current = setInterval(() => {
      refreshAllMetrics();
    }, configWithDefaults.refreshInterval);

    setIsAutoRefreshActive(true);
  }, [refreshAllMetrics, configWithDefaults.refreshInterval]);

  // Stop auto refresh
  const stopAutoRefresh = useCallback(() => {
    if (intervalRef.current) {
      clearInterval(intervalRef.current);
      intervalRef.current = null;
    }
    setIsAutoRefreshActive(false);
  }, []);

  // Initialize data and auto-refresh
  useEffect(() => {
    // Initial data load
    refreshAllMetrics();

    // Start auto-refresh if enabled
    if (configWithDefaults.autoRefresh) {
      startAutoRefresh();
    }

    // Cleanup on unmount
    return () => {
      stopAutoRefresh();
    };
  }, []); // Empty dependency array for mount/unmount only

  // Cleanup interval on unmount
  useEffect(() => {
    return () => {
      if (intervalRef.current) {
        clearInterval(intervalRef.current);
      }
    };
  }, []);

  return {
    // State
    ...state,
    ...realTimeMetrics,

    // Methods
    refreshLatencyMetrics,
    refreshSystemMetrics,
    refreshConnectionMetrics,
    refreshAlerts,
    refreshPerformanceTrends,
    refreshAllMetrics,
    configureAlert,

    // Control
    startAutoRefresh,
    stopAutoRefresh,
    isAutoRefreshActive
  };
};

export default useSystemMonitoring;