import { useState, useEffect, useCallback, useRef } from 'react';
import { riskService } from '../../components/Risk/services/riskService';
import { RiskAlert, RiskLimit } from '../../components/Risk/types/riskTypes';

export interface BreachPrediction {
  limit_id: string;
  limit_name: string;
  limit_type: RiskLimit['limit_type'];
  current_value: number;
  threshold_value: number;
  warning_threshold: number;
  probability_of_breach: number; // 0-1
  time_to_breach_minutes?: number;
  confidence_score: number; // 0-1
  contributing_factors: string[];
  recommended_actions: string[];
  severity: 'low' | 'medium' | 'high' | 'critical';
}

export interface BreachPattern {
  limit_type: RiskLimit['limit_type'];
  breach_frequency: number;
  average_recovery_time_minutes: number;
  common_triggers: string[];
  seasonal_patterns?: {
    hour_of_day: number[];
    day_of_week: number[];
    market_conditions: string[];
  };
  correlation_with_market: number;
}

export interface BreachDetectionState {
  predictions: BreachPrediction[];
  patterns: BreachPattern[];
  activeAlerts: RiskAlert[];
  recentBreaches: Array<{
    timestamp: Date;
    limit_id: string;
    limit_name: string;
    severity: string;
    recovery_time_minutes?: number;
  }>;
  mlModelInfo: {
    model_version: string;
    last_trained: Date;
    accuracy_score: number;
    prediction_horizon_minutes: number;
  } | null;
  loading: {
    predictions: boolean;
    patterns: boolean;
    alerts: boolean;
  };
  error: string | null;
  lastUpdated: Date | null;
}

export interface UseBreachDetectionProps {
  portfolioId: string;
  enableRealTime?: boolean;
  predictionHorizon?: number; // minutes
  refreshInterval?: number; // milliseconds
}

export const useBreachDetection = ({
  portfolioId,
  enableRealTime = true,
  predictionHorizon = 60,
  refreshInterval = 5000
}: UseBreachDetectionProps) => {
  const [state, setState] = useState<BreachDetectionState>({
    predictions: [],
    patterns: [],
    activeAlerts: [],
    recentBreaches: [],
    mlModelInfo: null,
    loading: {
      predictions: true,
      patterns: true,
      alerts: true
    },
    error: null,
    lastUpdated: null
  });

  const wsRef = useRef<WebSocket | null>(null);
  const refreshTimeoutRef = useRef<NodeJS.Timeout>();
  const abortControllerRef = useRef<AbortController>();

  const updateState = useCallback((updates: Partial<BreachDetectionState>) => {
    setState(prev => ({
      ...prev,
      ...updates,
      loading: { ...prev.loading, ...updates.loading }
    }));
  }, []);

  const fetchPredictions = useCallback(async (silent = false) => {
    if (!silent) {
      updateState({ loading: { ...state.loading, predictions: true } });
    }

    try {
      // Mock API call - would be replaced with actual Sprint 3 backend endpoint
      const mockPredictions: BreachPrediction[] = [
        {
          limit_id: 'limit-1',
          limit_name: '1-Day VaR Limit',
          limit_type: 'var',
          current_value: 48500,
          threshold_value: 50000,
          warning_threshold: 45000,
          probability_of_breach: 0.75,
          time_to_breach_minutes: 23,
          confidence_score: 0.89,
          contributing_factors: ['Market volatility spike', 'Portfolio concentration in tech sector'],
          recommended_actions: ['Reduce position size in AAPL', 'Hedge with SPY puts'],
          severity: 'high'
        },
        {
          limit_id: 'limit-2',
          limit_name: 'Position Concentration',
          limit_type: 'concentration',
          current_value: 18.5,
          threshold_value: 20.0,
          warning_threshold: 15.0,
          probability_of_breach: 0.42,
          time_to_breach_minutes: 156,
          confidence_score: 0.73,
          contributing_factors: ['Single position growth', 'Market correlation increase'],
          recommended_actions: ['Diversify holdings', 'Set stop losses'],
          severity: 'medium'
        }
      ];

      updateState({
        predictions: mockPredictions,
        error: null,
        lastUpdated: new Date(),
        loading: { ...state.loading, predictions: false }
      });

      return mockPredictions;
    } catch (error) {
      if (error instanceof Error) {
        updateState({
          error: error.message,
          loading: { ...state.loading, predictions: false }
        });
      }
      throw error;
    }
  }, [updateState, state.loading]);

  const fetchPatterns = useCallback(async (silent = false) => {
    if (!silent) {
      updateState({ loading: { ...state.loading, patterns: true } });
    }

    try {
      // Mock API call - would be replaced with actual Sprint 3 backend endpoint
      const mockPatterns: BreachPattern[] = [
        {
          limit_type: 'var',
          breach_frequency: 0.12, // 12% of trading days
          average_recovery_time_minutes: 45,
          common_triggers: ['Market gap down', 'Earnings surprises', 'Fed announcements'],
          seasonal_patterns: {
            hour_of_day: [9, 10, 15, 16], // Market open and close hours
            day_of_week: [1, 5], // Monday and Friday more volatile
            market_conditions: ['High VIX', 'Low liquidity', 'Earnings season']
          },
          correlation_with_market: 0.68
        },
        {
          limit_type: 'concentration',
          breach_frequency: 0.05,
          average_recovery_time_minutes: 120,
          common_triggers: ['Stock specific news', 'Sector rotation'],
          correlation_with_market: 0.23
        }
      ];

      updateState({
        patterns: mockPatterns,
        error: null,
        loading: { ...state.loading, patterns: false }
      });

      return mockPatterns;
    } catch (error) {
      if (error instanceof Error) {
        updateState({
          error: error.message,
          loading: { ...state.loading, patterns: false }
        });
      }
      throw error;
    }
  }, [updateState, state.loading]);

  const fetchActiveAlerts = useCallback(async (silent = false) => {
    if (!silent) {
      updateState({ loading: { ...state.loading, alerts: true } });
    }

    try {
      const alerts = await riskService.getRiskAlerts(portfolioId);
      const activeAlerts = alerts.filter(alert => !alert.acknowledged);

      updateState({
        activeAlerts,
        error: null,
        loading: { ...state.loading, alerts: false }
      });

      return activeAlerts;
    } catch (error) {
      if (error instanceof Error) {
        updateState({
          error: error.message,
          loading: { ...state.loading, alerts: false }
        });
      }
      throw error;
    }
  }, [portfolioId, updateState, state.loading]);

  const acknowledgeAlert = useCallback(async (alertId: string, acknowledgedBy: string) => {
    try {
      await riskService.acknowledgeAlert(alertId, acknowledgedBy);
      
      updateState({
        activeAlerts: state.activeAlerts.map(alert => 
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
  }, [state.activeAlerts, updateState]);

  const dismissAlert = useCallback(async (alertId: string) => {
    try {
      await riskService.dismissAlert(alertId);
      
      updateState({
        activeAlerts: state.activeAlerts.filter(alert => alert.id !== alertId)
      });

      return true;
    } catch (error) {
      if (error instanceof Error) {
        updateState({ error: error.message });
      }
      throw error;
    }
  }, [state.activeAlerts, updateState]);

  const clearError = useCallback(() => {
    updateState({ error: null });
  }, [updateState]);

  // WebSocket connection for real-time updates
  useEffect(() => {
    if (!enableRealTime) return;

    const connectWebSocket = () => {
      const wsUrl = `${import.meta.env.VITE_WS_URL || 'ws://localhost:8001'}/ws/risk/breach-detection/${portfolioId}`;
      wsRef.current = new WebSocket(wsUrl);

      wsRef.current.onopen = () => {
        console.log('Breach detection WebSocket connected');
      };

      wsRef.current.onmessage = (event) => {
        try {
          const message = JSON.parse(event.data);
          
          switch (message.type) {
            case 'prediction_update':
              updateState({ 
                predictions: message.data.predictions,
                lastUpdated: new Date()
              });
              break;
              
            case 'breach_alert':
              const newAlert = message.data.alert;
              updateState({
                activeAlerts: [...state.activeAlerts, newAlert],
                recentBreaches: [
                  {
                    timestamp: new Date(),
                    limit_id: newAlert.metadata?.limit_id || 'unknown',
                    limit_name: newAlert.metadata?.limit_name || 'Unknown Limit',
                    severity: newAlert.severity,
                    recovery_time_minutes: undefined
                  },
                  ...state.recentBreaches.slice(0, 9) // Keep last 10
                ]
              });
              break;

            case 'pattern_update':
              updateState({ patterns: message.data.patterns });
              break;

            case 'model_info':
              updateState({ mlModelInfo: message.data.model_info });
              break;
          }
        } catch (error) {
          console.error('Error parsing WebSocket message:', error);
        }
      };

      wsRef.current.onclose = () => {
        console.log('Breach detection WebSocket disconnected');
        // Attempt to reconnect after 5 seconds
        setTimeout(connectWebSocket, 5000);
      };

      wsRef.current.onerror = (error) => {
        console.error('Breach detection WebSocket error:', error);
      };
    };

    connectWebSocket();

    return () => {
      if (wsRef.current) {
        wsRef.current.close();
      }
    };
  }, [enableRealTime, portfolioId, updateState, state.activeAlerts, state.recentBreaches]);

  // Auto-refresh for non-real-time mode
  useEffect(() => {
    if (!enableRealTime && refreshInterval > 0) {
      const refresh = () => {
        Promise.all([
          fetchPredictions(true),
          fetchActiveAlerts(true)
        ]).catch(console.error);
      };

      refreshTimeoutRef.current = setTimeout(function tick() {
        refresh();
        refreshTimeoutRef.current = setTimeout(tick, refreshInterval);
      }, refreshInterval);

      return () => {
        if (refreshTimeoutRef.current) {
          clearTimeout(refreshTimeoutRef.current);
        }
      };
    }
  }, [enableRealTime, refreshInterval, fetchPredictions, fetchActiveAlerts]);

  // Initial fetch
  useEffect(() => {
    Promise.all([
      fetchPredictions(),
      fetchPatterns(),
      fetchActiveAlerts()
    ]).catch(console.error);

    return () => {
      abortControllerRef.current?.abort();
      if (refreshTimeoutRef.current) {
        clearTimeout(refreshTimeoutRef.current);
      }
    };
  }, [portfolioId, fetchPredictions, fetchPatterns, fetchActiveAlerts]);

  // Computed values
  const highRiskPredictions = state.predictions.filter(p => 
    p.severity === 'high' || p.severity === 'critical'
  );

  const imminentBreaches = state.predictions.filter(p => 
    p.time_to_breach_minutes && p.time_to_breach_minutes < 30
  );

  const criticalAlerts = state.activeAlerts.filter(alert => 
    alert.severity === 'critical'
  );

  const overallRiskScore = Math.min(100, Math.max(0,
    state.predictions.reduce((acc, pred) => {
      const severityWeight = {
        low: 1,
        medium: 2,
        high: 3,
        critical: 4
      }[pred.severity];
      return acc + (pred.probability_of_breach * severityWeight * 25);
    }, 0)
  ));

  return {
    // State
    ...state,
    
    // Computed values
    highRiskPredictions,
    imminentBreaches,
    criticalAlerts,
    overallRiskScore,
    
    // Actions
    fetchPredictions,
    fetchPatterns,
    fetchActiveAlerts,
    acknowledgeAlert,
    dismissAlert,
    clearError,
    
    // Utilities
    refresh: () => Promise.all([
      fetchPredictions(),
      fetchPatterns(),
      fetchActiveAlerts()
    ]),
    
    isConnected: enableRealTime && wsRef.current?.readyState === WebSocket.OPEN,
    
    getPredictionForLimit: (limitId: string) => 
      state.predictions.find(p => p.limit_id === limitId),
      
    getPatternForLimitType: (limitType: RiskLimit['limit_type']) =>
      state.patterns.find(p => p.limit_type === limitType)
  };
};

export default useBreachDetection;