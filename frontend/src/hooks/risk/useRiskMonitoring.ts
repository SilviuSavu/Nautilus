import { useState, useEffect, useCallback, useRef } from 'react';
import { riskService } from '../../components/Risk/services/riskService';
import { RiskMetrics, PortfolioRisk, RiskAlert, RiskLimit } from '../../components/Risk/types/riskTypes';

export interface RiskMonitoringState {
  realTimeMetrics: RealTimeRiskMetrics | null;
  historicalTrends: RiskTrend[];
  limitUtilization: LimitUtilization[];
  alerts: RiskAlert[];
  systemHealth: SystemHealth;
  monitoringConfig: MonitoringConfiguration;
  loading: {
    metrics: boolean;
    trends: boolean;
    alerts: boolean;
    config: boolean;
  };
  error: string | null;
  lastUpdated: Date | null;
  connectionStatus: 'connected' | 'disconnected' | 'reconnecting';
}

export interface RealTimeRiskMetrics {
  portfolio_id: string;
  timestamp: Date;
  var_95_current: number;
  var_99_current: number;
  expected_shortfall: number;
  portfolio_value: number;
  daily_pnl: number;
  intraday_var: number;
  stress_test_pnl?: number;
  beta: number;
  volatility: number;
  max_drawdown: number;
  sharpe_ratio: number;
  var_utilization_percentage: number;
  concentration_score: number;
  correlation_risk_score: number;
  liquidity_risk_score: number;
  overall_risk_score: number; // 0-100
}

export interface RiskTrend {
  timestamp: Date;
  metric_name: string;
  value: number;
  percentile_rank?: number;
  moving_average_5d?: number;
  moving_average_20d?: number;
  volatility_score?: number;
}

export interface LimitUtilization {
  limit_id: string;
  limit_name: string;
  limit_type: RiskLimit['limit_type'];
  threshold_value: number;
  current_value: number;
  utilization_percentage: number;
  warning_threshold: number;
  breach_probability: number;
  time_to_breach_estimate?: number; // minutes
  status: 'normal' | 'warning' | 'breach' | 'critical';
  trend_direction: 'up' | 'down' | 'stable';
}

export interface SystemHealth {
  risk_engine_status: 'healthy' | 'degraded' | 'offline';
  data_feed_status: 'healthy' | 'stale' | 'offline';
  calculation_latency_ms: number;
  last_calculation: Date;
  error_count_24h: number;
  warning_count_24h: number;
  uptime_percentage: number;
  memory_usage_percentage: number;
  cpu_usage_percentage: number;
}

export interface MonitoringConfiguration {
  portfolio_id: string;
  update_frequency_seconds: number;
  enabled_metrics: string[];
  alert_thresholds: {
    var_breach_threshold: number;
    concentration_threshold: number;
    correlation_threshold: number;
    volatility_threshold: number;
  };
  notification_settings: {
    email_enabled: boolean;
    dashboard_enabled: boolean;
    webhook_enabled: boolean;
    sms_enabled: boolean;
  };
  escalation_rules: {
    auto_escalate_minutes: number;
    escalation_levels: string[];
  };
}

export interface UseRiskMonitoringProps {
  portfolioId: string;
  enableRealTime?: boolean;
  updateFrequency?: number; // seconds
  enableTrends?: boolean;
  trendPeriod?: number; // days
}

export const useRiskMonitoring = ({
  portfolioId,
  enableRealTime = true,
  updateFrequency = 5,
  enableTrends = true,
  trendPeriod = 30
}: UseRiskMonitoringProps) => {
  const [state, setState] = useState<RiskMonitoringState>({
    realTimeMetrics: null,
    historicalTrends: [],
    limitUtilization: [],
    alerts: [],
    systemHealth: {
      risk_engine_status: 'healthy',
      data_feed_status: 'healthy',
      calculation_latency_ms: 0,
      last_calculation: new Date(),
      error_count_24h: 0,
      warning_count_24h: 0,
      uptime_percentage: 100,
      memory_usage_percentage: 0,
      cpu_usage_percentage: 0
    },
    monitoringConfig: {
      portfolio_id: portfolioId,
      update_frequency_seconds: updateFrequency,
      enabled_metrics: ['var', 'concentration', 'correlation', 'volatility'],
      alert_thresholds: {
        var_breach_threshold: 0.95,
        concentration_threshold: 0.25,
        correlation_threshold: 0.8,
        volatility_threshold: 0.3
      },
      notification_settings: {
        email_enabled: true,
        dashboard_enabled: true,
        webhook_enabled: false,
        sms_enabled: false
      },
      escalation_rules: {
        auto_escalate_minutes: 15,
        escalation_levels: ['team_lead', 'risk_manager', 'cro']
      }
    },
    loading: {
      metrics: true,
      trends: true,
      alerts: true,
      config: true
    },
    error: null,
    lastUpdated: null,
    connectionStatus: 'disconnected'
  });

  const wsRef = useRef<WebSocket | null>(null);
  const refreshTimeoutRef = useRef<NodeJS.Timeout>();
  const metricsBufferRef = useRef<RealTimeRiskMetrics[]>([]);

  const updateState = useCallback((updates: Partial<RiskMonitoringState>) => {
    setState(prev => ({
      ...prev,
      ...updates,
      loading: { ...prev.loading, ...updates.loading }
    }));
  }, []);

  const fetchCurrentMetrics = useCallback(async (silent = false) => {
    if (!silent) {
      updateState({ loading: { ...state.loading, metrics: true } });
    }

    try {
      const portfolioRisk = await riskService.getPortfolioRisk(portfolioId);
      const riskMetrics = await riskService.getRiskMetrics(portfolioId);
      
      // Convert to real-time format
      const realTimeMetrics: RealTimeRiskMetrics = {
        portfolio_id: portfolioId,
        timestamp: new Date(),
        var_95_current: parseFloat(riskMetrics.var_1d_95),
        var_99_current: parseFloat(riskMetrics.var_1d_99),
        expected_shortfall: parseFloat(riskMetrics.expected_shortfall_95),
        portfolio_value: parseFloat(portfolioRisk.total_exposure),
        daily_pnl: 0, // Would come from position service
        intraday_var: parseFloat(riskMetrics.var_1d_95) * 0.6, // Approximation
        beta: riskMetrics.beta_vs_market,
        volatility: riskMetrics.portfolio_volatility,
        max_drawdown: parseFloat(riskMetrics.max_drawdown),
        sharpe_ratio: riskMetrics.sharpe_ratio,
        var_utilization_percentage: (parseFloat(riskMetrics.var_1d_95) / parseFloat(portfolioRisk.total_exposure)) * 100,
        concentration_score: portfolioRisk.concentration_risk.length > 0 ? 
          Math.max(...portfolioRisk.concentration_risk.map(c => c.exposure_percentage)) : 0,
        correlation_risk_score: Math.abs(riskMetrics.correlation_with_market) * 100,
        liquidity_risk_score: 25, // Mock value
        overall_risk_score: Math.min(100, Math.max(0, 
          (parseFloat(riskMetrics.var_1d_95) / parseFloat(portfolioRisk.total_exposure)) * 200
        ))
      };

      updateState({
        realTimeMetrics,
        error: null,
        lastUpdated: new Date(),
        loading: { ...state.loading, metrics: false }
      });

      // Add to buffer for trends
      metricsBufferRef.current = [
        realTimeMetrics,
        ...metricsBufferRef.current.slice(0, 999) // Keep last 1000 points
      ];

      return realTimeMetrics;
    } catch (error) {
      if (error instanceof Error) {
        updateState({
          error: error.message,
          loading: { ...state.loading, metrics: false }
        });
      }
      throw error;
    }
  }, [portfolioId, state.loading, updateState]);

  const fetchHistoricalTrends = useCallback(async (silent = false) => {
    if (!silent) {
      updateState({ loading: { ...state.loading, trends: true } });
    }

    try {
      // Generate mock trend data - would be replaced with actual API call
      const trends: RiskTrend[] = [];
      const now = Date.now();
      
      for (let i = 0; i < trendPeriod; i++) {
        const timestamp = new Date(now - (i * 24 * 60 * 60 * 1000)); // Daily points
        
        trends.push({
          timestamp,
          metric_name: 'var_95',
          value: Math.random() * 50000 + 10000,
          percentile_rank: Math.random() * 100,
          moving_average_5d: Math.random() * 45000 + 12000,
          moving_average_20d: Math.random() * 40000 + 15000,
          volatility_score: Math.random() * 100
        });
      }

      updateState({
        historicalTrends: trends.reverse(),
        loading: { ...state.loading, trends: false }
      });

      return trends;
    } catch (error) {
      if (error instanceof Error) {
        updateState({
          error: error.message,
          loading: { ...state.loading, trends: false }
        });
      }
      throw error;
    }
  }, [trendPeriod, updateState, state.loading]);

  const fetchLimitUtilization = useCallback(async () => {
    try {
      const limits = await riskService.getRiskLimits(portfolioId);
      const portfolioRisk = await riskService.getPortfolioRisk(portfolioId);
      
      const utilization: LimitUtilization[] = limits.map(limit => {
        let currentValue = 0;
        let thresholdValue = parseFloat(limit.threshold_value);
        
        // Determine current value based on limit type
        switch (limit.limit_type) {
          case 'var':
            currentValue = parseFloat(portfolioRisk.var_1d);
            break;
          case 'concentration':
            currentValue = portfolioRisk.concentration_risk.length > 0 ?
              Math.max(...portfolioRisk.concentration_risk.map(c => c.exposure_percentage)) : 0;
            thresholdValue = parseFloat(limit.threshold_value);
            break;
          case 'leverage':
            currentValue = 1.5; // Mock value
            break;
          default:
            currentValue = Math.random() * thresholdValue;
        }

        const utilizationPercentage = (currentValue / thresholdValue) * 100;
        const warningThreshold = parseFloat(limit.warning_threshold);
        
        let status: LimitUtilization['status'] = 'normal';
        if (currentValue >= thresholdValue) status = 'breach';
        else if (currentValue >= warningThreshold) status = 'warning';
        else if (utilizationPercentage > 80) status = 'critical';

        return {
          limit_id: limit.id,
          limit_name: limit.name,
          limit_type: limit.limit_type,
          threshold_value: thresholdValue,
          current_value: currentValue,
          utilization_percentage: utilizationPercentage,
          warning_threshold: warningThreshold,
          breach_probability: Math.max(0, Math.min(1, (currentValue - warningThreshold) / (thresholdValue - warningThreshold))),
          time_to_breach_estimate: utilizationPercentage > 90 ? Math.floor(Math.random() * 60) + 5 : undefined,
          status,
          trend_direction: Math.random() > 0.5 ? 'up' : Math.random() > 0.5 ? 'down' : 'stable'
        };
      });

      updateState({ limitUtilization: utilization });
      return utilization;
    } catch (error) {
      console.error('Error fetching limit utilization:', error);
      return [];
    }
  }, [portfolioId, updateState]);

  const fetchAlerts = useCallback(async () => {
    try {
      const alerts = await riskService.getRiskAlerts(portfolioId);
      updateState({ alerts });
      return alerts;
    } catch (error) {
      console.error('Error fetching alerts:', error);
      return [];
    }
  }, [portfolioId, updateState]);

  const updateSystemHealth = useCallback(() => {
    const health: SystemHealth = {
      risk_engine_status: wsRef.current?.readyState === WebSocket.OPEN ? 'healthy' : 'offline',
      data_feed_status: state.lastUpdated && Date.now() - state.lastUpdated.getTime() < 30000 ? 'healthy' : 'stale',
      calculation_latency_ms: Math.floor(Math.random() * 100) + 10,
      last_calculation: state.lastUpdated || new Date(),
      error_count_24h: Math.floor(Math.random() * 5),
      warning_count_24h: Math.floor(Math.random() * 20),
      uptime_percentage: 99.5 + Math.random() * 0.5,
      memory_usage_percentage: Math.floor(Math.random() * 30) + 60,
      cpu_usage_percentage: Math.floor(Math.random() * 40) + 20
    };

    updateState({ systemHealth: health });
  }, [state.lastUpdated, updateState]);

  const acknowledgeAlert = useCallback(async (alertId: string, acknowledgedBy: string) => {
    try {
      await riskService.acknowledgeAlert(alertId, acknowledgedBy);
      updateState({
        alerts: state.alerts.map(alert =>
          alert.id === alertId
            ? { ...alert, acknowledged: true, acknowledged_by: acknowledgedBy, acknowledged_at: new Date() }
            : alert
        )
      });
      return true;
    } catch (error) {
      if (error instanceof Error) {
        updateState({ error: error.message });
      }
      throw error;
    }
  }, [state.alerts, updateState]);

  const updateConfiguration = useCallback(async (config: Partial<MonitoringConfiguration>) => {
    updateState({ loading: { ...state.loading, config: true } });

    try {
      const updatedConfig = { ...state.monitoringConfig, ...config };
      
      // Mock API call - would update backend configuration
      await new Promise(resolve => setTimeout(resolve, 1000));

      updateState({
        monitoringConfig: updatedConfig,
        loading: { ...state.loading, config: false }
      });

      return updatedConfig;
    } catch (error) {
      if (error instanceof Error) {
        updateState({
          error: error.message,
          loading: { ...state.loading, config: false }
        });
      }
      throw error;
    }
  }, [state.monitoringConfig, state.loading, updateState]);

  // WebSocket connection for real-time updates
  useEffect(() => {
    if (!enableRealTime) return;

    const connectWebSocket = () => {
      updateState({ connectionStatus: 'reconnecting' });
      
      const wsUrl = `${import.meta.env.VITE_WS_URL || 'ws://localhost:8001'}/ws/risk/monitoring/${portfolioId}`;
      wsRef.current = new WebSocket(wsUrl);

      wsRef.current.onopen = () => {
        console.log('Risk monitoring WebSocket connected');
        updateState({ connectionStatus: 'connected' });
        
        // Send configuration
        wsRef.current?.send(JSON.stringify({
          type: 'configure',
          data: {
            update_frequency: updateFrequency,
            metrics: state.monitoringConfig.enabled_metrics
          }
        }));
      };

      wsRef.current.onmessage = (event) => {
        try {
          const message = JSON.parse(event.data);
          
          switch (message.type) {
            case 'metrics_update':
              updateState({
                realTimeMetrics: message.data.metrics,
                lastUpdated: new Date()
              });
              break;
              
            case 'alert_triggered':
              updateState({
                alerts: [message.data.alert, ...state.alerts]
              });
              break;

            case 'limit_utilization':
              updateState({
                limitUtilization: message.data.utilization
              });
              break;

            case 'system_health':
              updateState({
                systemHealth: message.data.health
              });
              break;
          }
        } catch (error) {
          console.error('Error parsing WebSocket message:', error);
        }
      };

      wsRef.current.onclose = () => {
        console.log('Risk monitoring WebSocket disconnected');
        updateState({ connectionStatus: 'disconnected' });
        // Attempt to reconnect after 5 seconds
        setTimeout(connectWebSocket, 5000);
      };

      wsRef.current.onerror = (error) => {
        console.error('Risk monitoring WebSocket error:', error);
        updateState({ connectionStatus: 'disconnected' });
      };
    };

    connectWebSocket();

    return () => {
      if (wsRef.current) {
        wsRef.current.close();
      }
    };
  }, [enableRealTime, portfolioId, updateFrequency, state.monitoringConfig.enabled_metrics, updateState, state.alerts]);

  // Polling for non-real-time mode
  useEffect(() => {
    if (!enableRealTime) {
      const refresh = () => {
        Promise.all([
          fetchCurrentMetrics(true),
          fetchLimitUtilization(),
          fetchAlerts()
        ]).catch(console.error);
      };

      refreshTimeoutRef.current = setTimeout(function tick() {
        refresh();
        refreshTimeoutRef.current = setTimeout(tick, updateFrequency * 1000);
      }, updateFrequency * 1000);

      return () => {
        if (refreshTimeoutRef.current) {
          clearTimeout(refreshTimeoutRef.current);
        }
      };
    }
  }, [enableRealTime, updateFrequency, fetchCurrentMetrics, fetchLimitUtilization, fetchAlerts]);

  // System health updates
  useEffect(() => {
    const healthInterval = setInterval(updateSystemHealth, 10000); // Every 10 seconds
    return () => clearInterval(healthInterval);
  }, [updateSystemHealth]);

  // Initial fetch
  useEffect(() => {
    Promise.all([
      fetchCurrentMetrics(),
      enableTrends ? fetchHistoricalTrends() : Promise.resolve(),
      fetchLimitUtilization(),
      fetchAlerts()
    ]).catch(console.error);

    return () => {
      if (refreshTimeoutRef.current) {
        clearTimeout(refreshTimeoutRef.current);
      }
    };
  }, [portfolioId, fetchCurrentMetrics, fetchHistoricalTrends, fetchLimitUtilization, fetchAlerts, enableTrends]);

  // Computed values
  const criticalAlerts = state.alerts.filter(alert => alert.severity === 'critical' && !alert.acknowledged);
  const breachedLimits = state.limitUtilization.filter(util => util.status === 'breach');
  const warningLimits = state.limitUtilization.filter(util => util.status === 'warning');
  
  const avgUtilization = state.limitUtilization.length > 0 
    ? state.limitUtilization.reduce((sum, util) => sum + util.utilization_percentage, 0) / state.limitUtilization.length
    : 0;

  const riskTrendDirection = state.realTimeMetrics && metricsBufferRef.current.length > 1
    ? metricsBufferRef.current[0].overall_risk_score > metricsBufferRef.current[1].overall_risk_score ? 'up' : 'down'
    : 'stable';

  return {
    // State
    ...state,
    
    // Computed values
    criticalAlerts,
    breachedLimits,
    warningLimits,
    avgUtilization,
    riskTrendDirection,
    
    // Actions
    fetchCurrentMetrics,
    fetchHistoricalTrends,
    fetchLimitUtilization,
    fetchAlerts,
    acknowledgeAlert,
    updateConfiguration,
    
    // Utilities
    refresh: () => Promise.all([
      fetchCurrentMetrics(),
      fetchLimitUtilization(),
      fetchAlerts()
    ]),
    
    isConnected: enableRealTime && state.connectionStatus === 'connected',
    
    getMetricTrend: (metricName: string) => 
      state.historicalTrends.filter(trend => trend.metric_name === metricName),
      
    getLimitUtilization: (limitId: string) =>
      state.limitUtilization.find(util => util.limit_id === limitId)
  };
};

export default useRiskMonitoring;