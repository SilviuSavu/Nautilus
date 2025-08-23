/**
 * useConnectionHealth Hook
 * Sprint 3: Advanced WebSocket Infrastructure
 * 
 * Connection monitoring and health assessment hook with comprehensive metrics,
 * performance analysis, and connection quality scoring.
 */

import { useState, useEffect, useCallback, useRef } from 'react';
import { useWebSocketManager } from './useWebSocketManager';

interface ConnectionHealthMetrics {
  isHealthy: boolean;
  qualityScore: number;
  stabilityScore: number;
  reconnectCount: number;
  packetLoss: number;
  avgLatency: number;
  maxLatency: number;
  minLatency: number;
  jitter: number;
  throughput: number;
  errorRate: number;
  uptime: number;
  lastHealthCheck: string;
}

interface LatencyMeasurement {
  timestamp: number;
  latency: number;
  messageType: string;
}

interface ConnectionAlert {
  id: string;
  severity: 'info' | 'warning' | 'error' | 'critical';
  message: string;
  timestamp: string;
  resolved: boolean;
  metric?: string;
  value?: number;
  threshold?: number;
}

interface StabilityMetrics {
  connectionUptime: number;
  disconnectionCount: number;
  averageSessionDuration: number;
  connectionReliability: number;
  timeToReconnect: number[];
}

interface PerformanceThresholds {
  latency: {
    excellent: number;
    good: number;
    poor: number;
    critical: number;
  };
  throughput: {
    excellent: number;
    good: number;
    poor: number;
    critical: number;
  };
  errorRate: {
    excellent: number;
    good: number;
    poor: number;
    critical: number;
  };
  jitter: {
    excellent: number;
    good: number;
    poor: number;
    critical: number;
  };
}

const DEFAULT_THRESHOLDS: PerformanceThresholds = {
  latency: { excellent: 50, good: 100, poor: 300, critical: 1000 },
  throughput: { excellent: 100, good: 50, poor: 10, critical: 1 },
  errorRate: { excellent: 1, good: 3, poor: 10, critical: 20 },
  jitter: { excellent: 10, good: 25, poor: 50, critical: 100 }
};

export const useConnectionHealth = (
  thresholds: PerformanceThresholds = DEFAULT_THRESHOLDS,
  monitoringInterval: number = 5000,
  historySize: number = 100
) => {
  const {
    connectionState,
    connectionError,
    connectionAttempts,
    messageLatency,
    messagesReceived,
    messagesSent,
    getConnectionInfo,
    addMessageHandler
  } = useWebSocketManager();

  // Health state
  const [connectionHealth, setConnectionHealth] = useState<ConnectionHealthMetrics>({
    isHealthy: false,
    qualityScore: 0,
    stabilityScore: 0,
    reconnectCount: 0,
    packetLoss: 0,
    avgLatency: 0,
    maxLatency: 0,
    minLatency: Infinity,
    jitter: 0,
    throughput: 0,
    errorRate: 0,
    uptime: 0,
    lastHealthCheck: new Date().toISOString()
  });

  const [latencyHistory, setLatencyHistory] = useState<LatencyMeasurement[]>([]);
  const [connectionAlerts, setConnectionAlerts] = useState<ConnectionAlert[]>([]);
  const [stabilityMetrics, setStabilityMetrics] = useState<StabilityMetrics>({
    connectionUptime: 0,
    disconnectionCount: 0,
    averageSessionDuration: 0,
    connectionReliability: 100,
    timeToReconnect: []
  });

  // Refs for tracking
  const latencyMeasurementsRef = useRef<LatencyMeasurement[]>([]);
  const connectionEventsRef = useRef<Array<{ timestamp: number; event: string }>>([]);
  const disconnectionTimesRef = useRef<number[]>([]);
  const reconnectionTimesRef = useRef<number[]>([]);
  const messageCounterRef = useRef({ sent: 0, received: 0, errors: 0 });
  const healthCheckIntervalRef = useRef<NodeJS.Timeout>();
  const connectionStartTimeRef = useRef<number>(0);
  const lastDisconnectTimeRef = useRef<number>(0);

  // Generate alert ID
  const generateAlertId = useCallback(() => {
    return `alert_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
  }, []);

  // Add connection alert
  const addAlert = useCallback((
    severity: 'info' | 'warning' | 'error' | 'critical',
    message: string,
    metric?: string,
    value?: number,
    threshold?: number
  ) => {
    const alert: ConnectionAlert = {
      id: generateAlertId(),
      severity,
      message,
      timestamp: new Date().toISOString(),
      resolved: false,
      metric,
      value,
      threshold
    };

    setConnectionAlerts(prev => [alert, ...prev.slice(0, 49)]); // Keep last 50 alerts
    
    // Auto-resolve info alerts after 30 seconds
    if (severity === 'info') {
      setTimeout(() => {
        setConnectionAlerts(prev => 
          prev.map(a => a.id === alert.id ? { ...a, resolved: true } : a)
        );
      }, 30000);
    }
  }, [generateAlertId]);

  // Calculate jitter
  const calculateJitter = useCallback((latencies: number[]): number => {
    if (latencies.length < 2) return 0;
    
    let jitterSum = 0;
    for (let i = 1; i < latencies.length; i++) {
      jitterSum += Math.abs(latencies[i] - latencies[i - 1]);
    }
    
    return jitterSum / (latencies.length - 1);
  }, []);

  // Calculate quality score
  const calculateQualityScore = useCallback((metrics: Partial<ConnectionHealthMetrics>): number => {
    const weights = {
      latency: 0.3,
      jitter: 0.2,
      throughput: 0.2,
      errorRate: 0.2,
      stability: 0.1
    };

    // Latency score (0-100, lower is better)
    let latencyScore = 100;
    if (metrics.avgLatency !== undefined) {
      if (metrics.avgLatency <= thresholds.latency.excellent) latencyScore = 100;
      else if (metrics.avgLatency <= thresholds.latency.good) latencyScore = 80;
      else if (metrics.avgLatency <= thresholds.latency.poor) latencyScore = 60;
      else latencyScore = 20;
    }

    // Jitter score (0-100, lower is better)
    let jitterScore = 100;
    if (metrics.jitter !== undefined) {
      if (metrics.jitter <= thresholds.jitter.excellent) jitterScore = 100;
      else if (metrics.jitter <= thresholds.jitter.good) jitterScore = 80;
      else if (metrics.jitter <= thresholds.jitter.poor) jitterScore = 60;
      else jitterScore = 20;
    }

    // Throughput score (0-100, higher is better)
    let throughputScore = 100;
    if (metrics.throughput !== undefined) {
      if (metrics.throughput >= thresholds.throughput.excellent) throughputScore = 100;
      else if (metrics.throughput >= thresholds.throughput.good) throughputScore = 80;
      else if (metrics.throughput >= thresholds.throughput.poor) throughputScore = 60;
      else throughputScore = 20;
    }

    // Error rate score (0-100, lower is better)
    let errorScore = 100;
    if (metrics.errorRate !== undefined) {
      if (metrics.errorRate <= thresholds.errorRate.excellent) errorScore = 100;
      else if (metrics.errorRate <= thresholds.errorRate.good) errorScore = 80;
      else if (metrics.errorRate <= thresholds.errorRate.poor) errorScore = 60;
      else errorScore = 20;
    }

    // Stability score
    const stabilityScore = metrics.stabilityScore || 100;

    // Calculate weighted score
    const totalScore = (
      latencyScore * weights.latency +
      jitterScore * weights.jitter +
      throughputScore * weights.throughput +
      errorScore * weights.errorRate +
      stabilityScore * weights.stability
    );

    return Math.round(Math.max(0, Math.min(100, totalScore)));
  }, [thresholds]);

  // Update latency measurements
  const updateLatencyMeasurement = useCallback((latency: number, messageType: string = 'unknown') => {
    const measurement: LatencyMeasurement = {
      timestamp: Date.now(),
      latency,
      messageType
    };

    latencyMeasurementsRef.current.push(measurement);
    if (latencyMeasurementsRef.current.length > historySize) {
      latencyMeasurementsRef.current.shift();
    }

    setLatencyHistory([...latencyMeasurementsRef.current]);

    // Check for latency alerts
    if (latency > thresholds.latency.critical) {
      addAlert('critical', `Critical latency detected: ${latency.toFixed(0)}ms`, 'latency', latency, thresholds.latency.critical);
    } else if (latency > thresholds.latency.poor) {
      addAlert('warning', `High latency detected: ${latency.toFixed(0)}ms`, 'latency', latency, thresholds.latency.poor);
    }
  }, [historySize, thresholds.latency, addAlert]);

  // Calculate connection metrics
  const calculateConnectionMetrics = useCallback(() => {
    const now = Date.now();
    const connectionInfo = getConnectionInfo();
    const latencies = latencyMeasurementsRef.current.map(m => m.latency);
    
    // Basic latency statistics
    const avgLatency = latencies.length > 0 ? 
      latencies.reduce((sum, lat) => sum + lat, 0) / latencies.length : 0;
    const maxLatency = latencies.length > 0 ? Math.max(...latencies) : 0;
    const minLatency = latencies.length > 0 ? Math.min(...latencies) : 0;
    const jitter = calculateJitter(latencies);

    // Calculate throughput (messages per second)
    const uptime = connectionInfo.uptime || 0;
    const throughput = uptime > 0 ? 
      (connectionInfo.messagesReceived + connectionInfo.messagesSent) / (uptime / 1000) : 0;

    // Calculate error rate
    const totalMessages = connectionInfo.messagesReceived + connectionInfo.messagesSent;
    const errorRate = totalMessages > 0 ? 
      (messageCounterRef.current.errors / totalMessages) * 100 : 0;

    // Calculate packet loss (approximate)
    const expectedMessages = messageCounterRef.current.sent;
    const receivedMessages = messageCounterRef.current.received;
    const packetLoss = expectedMessages > 0 ? 
      Math.max(0, ((expectedMessages - receivedMessages) / expectedMessages) * 100) : 0;

    // Calculate stability score
    const reconnectPenalty = Math.min(50, connectionAttempts * 5);
    const uptimeFactor = Math.min(100, (uptime / (24 * 60 * 60 * 1000)) * 100); // 24 hours = 100%
    const stabilityScore = Math.max(0, 100 - reconnectPenalty + uptimeFactor * 0.5);

    const metrics: ConnectionHealthMetrics = {
      isHealthy: connectionState === 'connected' && avgLatency < thresholds.latency.poor,
      qualityScore: 0, // Will be calculated below
      stabilityScore,
      reconnectCount: connectionAttempts,
      packetLoss,
      avgLatency,
      maxLatency,
      minLatency: minLatency === Infinity ? 0 : minLatency,
      jitter,
      throughput,
      errorRate,
      uptime,
      lastHealthCheck: new Date().toISOString()
    };

    // Calculate quality score
    metrics.qualityScore = calculateQualityScore(metrics);

    return metrics;
  }, [
    getConnectionInfo,
    calculateJitter,
    connectionState,
    connectionAttempts,
    thresholds.latency.poor,
    calculateQualityScore
  ]);

  // Update stability metrics
  const updateStabilityMetrics = useCallback(() => {
    const now = Date.now();
    const connectionEvents = connectionEventsRef.current;
    
    // Count disconnections in the last hour
    const oneHourAgo = now - (60 * 60 * 1000);
    const recentDisconnections = disconnectionTimesRef.current.filter(time => time > oneHourAgo);
    
    // Calculate average session duration
    let totalSessionDuration = 0;
    let sessionCount = 0;
    
    for (let i = 0; i < connectionEvents.length - 1; i++) {
      if (connectionEvents[i].event === 'connected' && connectionEvents[i + 1].event === 'disconnected') {
        totalSessionDuration += connectionEvents[i + 1].timestamp - connectionEvents[i].timestamp;
        sessionCount++;
      }
    }
    
    const averageSessionDuration = sessionCount > 0 ? totalSessionDuration / sessionCount : 0;
    
    // Calculate connection reliability
    const totalAttempts = connectionEvents.filter(e => e.event === 'connecting').length;
    const successfulConnections = connectionEvents.filter(e => e.event === 'connected').length;
    const connectionReliability = totalAttempts > 0 ? (successfulConnections / totalAttempts) * 100 : 100;
    
    setStabilityMetrics({
      connectionUptime: connectionStartTimeRef.current > 0 ? now - connectionStartTimeRef.current : 0,
      disconnectionCount: recentDisconnections.length,
      averageSessionDuration,
      connectionReliability,
      timeToReconnect: reconnectionTimesRef.current.slice(-10) // Keep last 10 reconnection times
    });
  }, []);

  // Perform health check
  const performHealthCheck = useCallback(() => {
    const metrics = calculateConnectionMetrics();
    setConnectionHealth(metrics);
    updateStabilityMetrics();

    // Generate health alerts
    if (metrics.qualityScore < 30) {
      addAlert('critical', `Poor connection quality: ${metrics.qualityScore}%`, 'quality_score', metrics.qualityScore, 50);
    } else if (metrics.qualityScore < 60) {
      addAlert('warning', `Degraded connection quality: ${metrics.qualityScore}%`, 'quality_score', metrics.qualityScore, 80);
    }

    if (metrics.errorRate > thresholds.errorRate.poor) {
      addAlert('error', `High error rate: ${metrics.errorRate.toFixed(1)}%`, 'error_rate', metrics.errorRate, thresholds.errorRate.poor);
    }

    if (metrics.packetLoss > 5) {
      addAlert('warning', `Packet loss detected: ${metrics.packetLoss.toFixed(1)}%`, 'packet_loss', metrics.packetLoss, 5);
    }

  }, [calculateConnectionMetrics, updateStabilityMetrics, addAlert, thresholds.errorRate.poor]);

  // Track connection state changes
  useEffect(() => {
    const now = Date.now();
    
    connectionEventsRef.current.push({
      timestamp: now,
      event: connectionState
    });

    // Keep only recent events (last 24 hours)
    const oneDayAgo = now - (24 * 60 * 60 * 1000);
    connectionEventsRef.current = connectionEventsRef.current.filter(
      event => event.timestamp > oneDayAgo
    );

    // Handle specific state changes
    switch (connectionState) {
      case 'connected':
        if (connectionStartTimeRef.current === 0) {
          connectionStartTimeRef.current = now;
        }
        if (lastDisconnectTimeRef.current > 0) {
          const reconnectTime = now - lastDisconnectTimeRef.current;
          reconnectionTimesRef.current.push(reconnectTime);
          lastDisconnectTimeRef.current = 0;
        }
        addAlert('info', 'WebSocket connection established');
        break;
      
      case 'disconnected':
      case 'error':
        if (lastDisconnectTimeRef.current === 0) {
          lastDisconnectTimeRef.current = now;
          disconnectionTimesRef.current.push(now);
        }
        if (connectionState === 'error' && connectionError) {
          addAlert('error', `Connection error: ${connectionError}`);
        } else {
          addAlert('warning', 'WebSocket connection lost');
        }
        break;
      
      case 'reconnecting':
        addAlert('info', 'Attempting to reconnect...');
        break;
    }
  }, [connectionState, connectionError, addAlert]);

  // Track message latency
  useEffect(() => {
    if (messageLatency > 0) {
      updateLatencyMeasurement(messageLatency);
    }
  }, [messageLatency, updateLatencyMeasurement]);

  // Set up message handler for tracking
  useEffect(() => {
    const removeHandler = addMessageHandler(
      'health_monitor',
      (message) => {
        messageCounterRef.current.received += 1;
        
        if (message.error) {
          messageCounterRef.current.errors += 1;
        }
        
        if (message.latency) {
          updateLatencyMeasurement(message.latency, message.type);
        }
      }
    );

    return removeHandler;
  }, [addMessageHandler, updateLatencyMeasurement]);

  // Start periodic health checks
  useEffect(() => {
    healthCheckIntervalRef.current = setInterval(performHealthCheck, monitoringInterval);
    
    // Initial health check
    performHealthCheck();

    return () => {
      if (healthCheckIntervalRef.current) {
        clearInterval(healthCheckIntervalRef.current);
      }
    };
  }, [performHealthCheck, monitoringInterval]);

  // Resolve alert
  const resolveAlert = useCallback((alertId: string) => {
    setConnectionAlerts(prev => 
      prev.map(alert => 
        alert.id === alertId ? { ...alert, resolved: true } : alert
      )
    );
  }, []);

  // Clear resolved alerts
  const clearResolvedAlerts = useCallback(() => {
    setConnectionAlerts(prev => prev.filter(alert => !alert.resolved));
  }, []);

  // Get performance trend
  const getPerformanceTrend = useCallback((metric: keyof ConnectionHealthMetrics, timeRange: number = 300000) => {
    const now = Date.now();
    const cutoffTime = now - timeRange;
    
    switch (metric) {
      case 'avgLatency':
        return latencyMeasurementsRef.current
          .filter(m => m.timestamp > cutoffTime)
          .map(m => ({ timestamp: m.timestamp, value: m.latency }));
      
      default:
        return [];
    }
  }, []);

  return {
    // Health metrics
    connectionHealth,
    qualityScore: connectionHealth.qualityScore,
    stabilityMetrics,
    performanceMetrics: {
      latency: connectionHealth.avgLatency,
      jitter: connectionHealth.jitter,
      throughput: connectionHealth.throughput,
      errorRate: connectionHealth.errorRate,
      packetLoss: connectionHealth.packetLoss
    },

    // History and trends
    latencyHistory,
    connectionAlerts,
    alertSummary: connectionAlerts.filter(alert => !alert.resolved),

    // Control functions
    performHealthCheck,
    resolveAlert,
    clearResolvedAlerts,
    getPerformanceTrend,

    // Connection status
    isConnected: connectionState === 'connected',
    connectionState
  };
};

export default useConnectionHealth;