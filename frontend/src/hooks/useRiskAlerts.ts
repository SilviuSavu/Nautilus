/**
 * useRiskAlerts - WebSocket hook for real-time risk monitoring and alerts
 * Sprint 3 Priority 1: WebSocket Streaming Infrastructure
 * 
 * Connects to /ws/risk/alerts endpoint for live risk alerts and monitoring
 */

import { useState, useEffect, useRef, useCallback } from 'react';

interface RiskAlert {
  type: 'risk_breach' | 'var_exceeded' | 'position_limit' | 'correlation_alert' | 'volatility_spike' | 'drawdown_alert';
  severity: 'LOW' | 'MEDIUM' | 'HIGH' | 'CRITICAL';
  data: {
    alert_id: string;
    portfolio_id?: string;
    strategy_id?: string;
    instrument_id?: string;
    symbol?: string;
    description: string;
    current_value: number;
    threshold_value: number;
    breach_magnitude: number;
    risk_metric: 'var' | 'position_size' | 'concentration' | 'correlation' | 'volatility' | 'drawdown';
    breach_percentage: number;
    recommended_action?: string;
    time_to_breach?: number;
    historical_context?: {
      previous_breaches: number;
      avg_recovery_time: number;
      max_breach_magnitude: number;
    };
    metadata?: Record<string, any>;
  };
  timestamp: string;
  connection_id?: string;
  acknowledged?: boolean;
  resolved?: boolean;
}

interface RiskMetrics {
  portfolio_var: number;
  position_concentration: number;
  correlation_risk: number;
  volatility_regime: number;
  max_drawdown: number;
  leverage_ratio: number;
  liquidity_score: number;
}

interface RiskAlertsState {
  isConnected: boolean;
  activeAlerts: RiskAlert[];
  recentAlerts: RiskAlert[];
  riskMetrics: RiskMetrics | null;
  alertCounts: Record<string, number>;
  lastUpdate: Date | null;
  connectionError: string | null;
  connectionAttempts: number;
}

interface UseRiskAlertsOptions {
  autoReconnect?: boolean;
  reconnectInterval?: number;
  maxReconnectAttempts?: number;
  heartbeatInterval?: number;
  maxRecentAlerts?: number;
  severityFilter?: string[];
  portfolioFilter?: string[];
  autoAcknowledgeInfo?: boolean;
}

export const useRiskAlerts = (options: UseRiskAlertsOptions = {}) => {
  const {
    autoReconnect = true,
    reconnectInterval = 5000,
    maxReconnectAttempts = 10,
    heartbeatInterval = 30000,
    maxRecentAlerts = 50,
    severityFilter = ['LOW', 'MEDIUM', 'HIGH', 'CRITICAL'],
    portfolioFilter = [],
    autoAcknowledgeInfo = false
  } = options;

  const [state, setState] = useState<RiskAlertsState>({
    isConnected: false,
    activeAlerts: [],
    recentAlerts: [],
    riskMetrics: null,
    alertCounts: {},
    lastUpdate: null,
    connectionError: null,
    connectionAttempts: 0
  });

  const wsRef = useRef<WebSocket | null>(null);
  const reconnectTimeoutRef = useRef<NodeJS.Timeout>();
  const heartbeatTimeoutRef = useRef<NodeJS.Timeout>();
  const mountedRef = useRef(true);

  // Get WebSocket URL for risk alerts
  const getWebSocketUrl = useCallback(() => {
    const apiBaseUrl = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8001';
    const protocol = apiBaseUrl.startsWith('https') ? 'wss:' : 'ws:';
    const host = apiBaseUrl.replace(/^https?:\/\//, '');
    return `${protocol}//${host}/ws/risk/alerts`;
  }, []);

  // Send heartbeat to keep connection alive
  const sendHeartbeat = useCallback(() => {
    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
      try {
        wsRef.current.send(JSON.stringify({
          type: 'heartbeat',
          timestamp: new Date().toISOString()
        }));
      } catch (error) {
        console.error('Failed to send risk alerts heartbeat:', error);
      }
    }
  }, []);

  // Start heartbeat interval
  const startHeartbeat = useCallback(() => {
    if (heartbeatTimeoutRef.current) {
      clearInterval(heartbeatTimeoutRef.current);
    }
    
    heartbeatTimeoutRef.current = setInterval(sendHeartbeat, heartbeatInterval);
  }, [sendHeartbeat, heartbeatInterval]);

  // Stop heartbeat interval
  const stopHeartbeat = useCallback(() => {
    if (heartbeatTimeoutRef.current) {
      clearInterval(heartbeatTimeoutRef.current);
      heartbeatTimeoutRef.current = undefined;
    }
  }, []);

  // Check if alert should be included based on filters
  const shouldIncludeAlert = useCallback((alert: RiskAlert) => {
    // Severity filter
    if (!severityFilter.includes(alert.severity)) {
      return false;
    }

    // Portfolio filter
    if (portfolioFilter.length > 0 && alert.data.portfolio_id && 
        !portfolioFilter.includes(alert.data.portfolio_id)) {
      return false;
    }

    return true;
  }, [severityFilter, portfolioFilter]);

  // Calculate alert priority score for sorting
  const getAlertPriorityScore = useCallback((alert: RiskAlert) => {
    const severityScores = { CRITICAL: 4, HIGH: 3, MEDIUM: 2, LOW: 1 };
    const typeScores = { 
      var_exceeded: 4, 
      risk_breach: 3, 
      position_limit: 3, 
      drawdown_alert: 2, 
      correlation_alert: 2, 
      volatility_spike: 1 
    };
    
    return severityScores[alert.severity] * 10 + typeScores[alert.type];
  }, []);

  // Connect to WebSocket
  const connect = useCallback(() => {
    if (!mountedRef.current) return;

    try {
      const wsUrl = getWebSocketUrl();
      console.log('Connecting to risk alerts WebSocket:', wsUrl);
      
      const ws = new WebSocket(wsUrl);
      wsRef.current = ws;

      // Connection opened
      ws.onopen = () => {
        if (!mountedRef.current) return;
        
        console.log('Risk alerts WebSocket connected');
        setState(prev => ({
          ...prev,
          isConnected: true,
          connectionError: null,
          connectionAttempts: 0
        }));

        // Subscribe to risk alerts with filters
        try {
          ws.send(JSON.stringify({
            type: 'subscribe',
            event_types: ['risk_alert', 'risk_metrics', 'alert_resolved', 'alert_acknowledged'],
            filters: {
              severity: severityFilter,
              portfolios: portfolioFilter
            },
            options: {
              include_metrics: true,
              historical_alerts: false
            },
            timestamp: new Date().toISOString()
          }));
        } catch (error) {
          console.error('Failed to subscribe to risk alerts:', error);
        }

        startHeartbeat();
      };

      // Message received
      ws.onmessage = (event) => {
        if (!mountedRef.current) return;

        try {
          const message = JSON.parse(event.data);
          
          if (message.type === 'risk_alert') {
            const alert: RiskAlert = message;
            
            if (!shouldIncludeAlert(alert)) {
              return;
            }

            setState(prev => {
              // Auto-acknowledge INFO level alerts if enabled
              if (autoAcknowledgeInfo && alert.severity === 'LOW') {
                alert.acknowledged = true;
              }

              // Add to recent alerts
              const newRecentAlerts = [alert, ...prev.recentAlerts].slice(0, maxRecentAlerts);
              
              // Add to active alerts if not resolved/acknowledged
              const newActiveAlerts = alert.acknowledged || alert.resolved 
                ? prev.activeAlerts 
                : [alert, ...prev.activeAlerts].sort((a, b) => 
                    getAlertPriorityScore(b) - getAlertPriorityScore(a)
                  );

              // Update alert counts
              const alertCounts = { ...prev.alertCounts };
              alertCounts[alert.severity] = (alertCounts[alert.severity] || 0) + 1;
              alertCounts[alert.type] = (alertCounts[alert.type] || 0) + 1;

              return {
                ...prev,
                activeAlerts: newActiveAlerts,
                recentAlerts: newRecentAlerts,
                alertCounts,
                lastUpdate: new Date(alert.timestamp)
              };
            });
          } else if (message.type === 'risk_metrics') {
            setState(prev => ({
              ...prev,
              riskMetrics: message.data,
              lastUpdate: new Date(message.timestamp)
            }));
          } else if (message.type === 'alert_resolved' || message.type === 'alert_acknowledged') {
            const alertId = message.data.alert_id;
            setState(prev => ({
              ...prev,
              activeAlerts: prev.activeAlerts.filter(a => a.data.alert_id !== alertId),
              lastUpdate: new Date(message.timestamp)
            }));
          }
          
        } catch (error) {
          console.error('Failed to parse risk alert message:', error);
        }
      };

      // Connection error
      ws.onerror = (error) => {
        if (!mountedRef.current) return;
        
        console.error('Risk alerts WebSocket error:', error);
        setState(prev => ({
          ...prev,
          connectionError: 'Risk alerts connection error'
        }));
      };

      // Connection closed
      ws.onclose = (event) => {
        if (!mountedRef.current) return;
        
        console.log('Risk alerts WebSocket closed:', event.code, event.reason);
        stopHeartbeat();
        
        setState(prev => ({
          ...prev,
          isConnected: false,
          connectionError: event.code !== 1000 ? `Connection closed: ${event.reason || event.code}` : null
        }));

        // Auto-reconnect if enabled and not manually closed
        if (autoReconnect && event.code !== 1000 && state.connectionAttempts < maxReconnectAttempts) {
          setState(prev => ({
            ...prev,
            connectionAttempts: prev.connectionAttempts + 1
          }));

          reconnectTimeoutRef.current = setTimeout(() => {
            if (mountedRef.current) {
              console.log(`Attempting to reconnect risk alerts... (${state.connectionAttempts + 1}/${maxReconnectAttempts})`);
              connect();
            }
          }, reconnectInterval);
        }
      };

    } catch (error) {
      console.error('Failed to create risk alerts WebSocket connection:', error);
      setState(prev => ({
        ...prev,
        connectionError: 'Failed to create WebSocket connection'
      }));
    }
  }, [getWebSocketUrl, autoReconnect, reconnectInterval, maxReconnectAttempts, state.connectionAttempts, startHeartbeat, stopHeartbeat, shouldIncludeAlert, severityFilter, portfolioFilter, maxRecentAlerts, autoAcknowledgeInfo, getAlertPriorityScore]);

  // Disconnect from WebSocket
  const disconnect = useCallback(() => {
    if (reconnectTimeoutRef.current) {
      clearTimeout(reconnectTimeoutRef.current);
      reconnectTimeoutRef.current = undefined;
    }

    stopHeartbeat();

    if (wsRef.current) {
      wsRef.current.close(1000, 'Manual disconnect');
      wsRef.current = null;
    }

    setState(prev => ({
      ...prev,
      isConnected: false,
      connectionError: null,
      connectionAttempts: 0
    }));
  }, [stopHeartbeat]);

  // Send message to WebSocket
  const sendMessage = useCallback((message: any) => {
    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
      try {
        wsRef.current.send(JSON.stringify({
          ...message,
          timestamp: new Date().toISOString()
        }));
        return true;
      } catch (error) {
        console.error('Failed to send risk alert message:', error);
        return false;
      }
    }
    return false;
  }, []);

  // Acknowledge alert
  const acknowledgeAlert = useCallback((alertId: string) => {
    const success = sendMessage({
      type: 'acknowledge_alert',
      alert_id: alertId
    });

    if (success) {
      setState(prev => ({
        ...prev,
        activeAlerts: prev.activeAlerts.filter(a => a.data.alert_id !== alertId)
      }));
    }

    return success;
  }, [sendMessage]);

  // Acknowledge all alerts of specific severity
  const acknowledgeAllBySeverity = useCallback((severity: string) => {
    const alertsToAck = state.activeAlerts.filter(a => a.severity === severity);
    let successCount = 0;

    alertsToAck.forEach(alert => {
      if (acknowledgeAlert(alert.data.alert_id)) {
        successCount++;
      }
    });

    return successCount;
  }, [state.activeAlerts, acknowledgeAlert]);

  // Clear acknowledged alerts from recent list
  const clearRecentAlerts = useCallback(() => {
    setState(prev => ({
      ...prev,
      recentAlerts: []
    }));
  }, []);

  // Update subscription filters
  const updateFilters = useCallback((newSeverity?: string[], newPortfolios?: string[]) => {
    return sendMessage({
      type: 'update_filters',
      filters: {
        severity: newSeverity || severityFilter,
        portfolios: newPortfolios || portfolioFilter
      }
    });
  }, [sendMessage, severityFilter, portfolioFilter]);

  // Get risk summary statistics
  const getRiskSummary = useCallback(() => {
    const criticalCount = state.activeAlerts.filter(a => a.severity === 'CRITICAL').length;
    const highCount = state.activeAlerts.filter(a => a.severity === 'HIGH').length;
    const mediumCount = state.activeAlerts.filter(a => a.severity === 'MEDIUM').length;
    const lowCount = state.activeAlerts.filter(a => a.severity === 'LOW').length;

    const avgBreachMagnitude = state.activeAlerts.length > 0 
      ? state.activeAlerts.reduce((sum, a) => sum + a.data.breach_magnitude, 0) / state.activeAlerts.length
      : 0;

    return {
      totalActive: state.activeAlerts.length,
      critical: criticalCount,
      high: highCount,
      medium: mediumCount,
      low: lowCount,
      avgBreachMagnitude,
      riskScore: criticalCount * 4 + highCount * 3 + mediumCount * 2 + lowCount * 1
    };
  }, [state.activeAlerts]);

  // Initialize connection on mount
  useEffect(() => {
    mountedRef.current = true;
    connect();

    return () => {
      mountedRef.current = false;
      disconnect();
    };
  }, [connect, disconnect]);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      if (reconnectTimeoutRef.current) {
        clearTimeout(reconnectTimeoutRef.current);
      }
      stopHeartbeat();
    };
  }, [stopHeartbeat]);

  return {
    // Connection state
    isConnected: state.isConnected,
    connectionError: state.connectionError,
    connectionAttempts: state.connectionAttempts,
    
    // Alert data
    activeAlerts: state.activeAlerts,
    recentAlerts: state.recentAlerts,
    riskMetrics: state.riskMetrics,
    alertCounts: state.alertCounts,
    lastUpdate: state.lastUpdate,
    
    // Connection control
    connect,
    disconnect,
    
    // Alert management
    acknowledgeAlert,
    acknowledgeAllBySeverity,
    clearRecentAlerts,
    
    // Message sending
    sendMessage,
    updateFilters,
    
    // Utilities
    getRiskSummary,
    sendHeartbeat
  };
};

export default useRiskAlerts;