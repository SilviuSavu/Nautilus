/**
 * useWebSocketHealth Hook
 * Sprint 3: Advanced WebSocket Connection Health Monitoring
 * 
 * Comprehensive connection health monitoring with quality scoring,
 * stability tracking, performance metrics, and intelligent diagnostics.
 */

import { useState, useEffect, useCallback, useRef } from 'react';
import { useWebSocketManager } from './useWebSocketManager';
import type { ConnectionHealth, WebSocketPerformanceMetrics } from '../types/websocket';

export interface ConnectionQualityMetrics {
  latency: {
    current: number;
    average: number;
    min: number;
    max: number;
    p50: number;
    p95: number;
    p99: number;
    jitter: number;
  };
  throughput: {
    messagesPerSecond: number;
    bytesPerSecond: number;
    peakThroughput: number;
    averageThroughput: number;
  };
  reliability: {
    uptime: number;
    uptimePercentage: number;
    reconnectionCount: number;
    lastReconnection: string | null;
    connectionStability: number;
    errorRate: number;
    successRate: number;
  };
  performance: {
    qualityScore: number;
    stabilityScore: number;
    healthScore: number;
    performanceGrade: 'A' | 'B' | 'C' | 'D' | 'F';
  };
}

export interface ConnectionDiagnostics {
  issues: {
    id: string;
    type: 'latency' | 'stability' | 'throughput' | 'error';
    severity: 'low' | 'medium' | 'high' | 'critical';
    message: string;
    recommendation: string;
    detectedAt: string;
  }[];
  summary: {
    totalIssues: number;
    criticalIssues: number;
    recommendedActions: string[];
    overallHealth: 'excellent' | 'good' | 'fair' | 'poor' | 'critical';
  };
}

export interface UseWebSocketHealthOptions {
  monitoringInterval?: number;
  latencyThreshold?: number;
  stabilityThreshold?: number;
  errorThreshold?: number;
  enableAutoRecovery?: boolean;
  enableDiagnostics?: boolean;
  healthHistorySize?: number;
}

export interface UseWebSocketHealthReturn {
  // Connection health
  connectionHealth: ConnectionHealth;
  qualityMetrics: ConnectionQualityMetrics;
  diagnostics: ConnectionDiagnostics;
  
  // Health status
  isHealthy: boolean;
  healthGrade: 'A' | 'B' | 'C' | 'D' | 'F';
  healthTrend: 'improving' | 'stable' | 'declining';
  
  // Monitoring control
  startMonitoring: () => void;
  stopMonitoring: () => void;
  resetMetrics: () => void;
  runDiagnostics: () => Promise<ConnectionDiagnostics>;
  
  // Health history
  healthHistory: ConnectionQualityMetrics[];
  getHealthTrend: (minutes?: number) => 'improving' | 'stable' | 'declining';
  
  // Auto-recovery
  triggerRecovery: () => Promise<void>;
  enableAutoRecovery: () => void;
  disableAutoRecovery: () => void;
}

export function useWebSocketHealth(
  options: UseWebSocketHealthOptions = {}
): UseWebSocketHealthReturn {
  const {
    monitoringInterval = 1000,
    latencyThreshold = 1000,
    stabilityThreshold = 0.95,
    errorThreshold = 0.05,
    enableAutoRecovery = false,
    enableDiagnostics = true,
    healthHistorySize = 300
  } = options;
  
  // State
  const [isMonitoring, setIsMonitoring] = useState(false);
  const [healthHistory, setHealthHistory] = useState<ConnectionQualityMetrics[]>([]);
  const [isHealthy, setIsHealthy] = useState(false);
  const [healthGrade, setHealthGrade] = useState<'A' | 'B' | 'C' | 'D' | 'F'>('F');
  const [healthTrend, setHealthTrend] = useState<'improving' | 'stable' | 'declining'>('stable');
  const [autoRecoveryEnabled, setAutoRecoveryEnabled] = useState(enableAutoRecovery);
  
  const [connectionHealth, setConnectionHealth] = useState<ConnectionHealth>({
    isHealthy: false,
    qualityScore: 0,
    stabilityScore: 0,
    latencyStats: { current: 0, average: 0, min: 0, max: 0, p95: 0 },
    throughputStats: { current: 0, average: 0, peak: 0 },
    errorStats: { total: 0, rate: 0, recentErrors: [] },
    connectionStats: { uptime: 0, reconnections: 0 }
  });
  
  const [qualityMetrics, setQualityMetrics] = useState<ConnectionQualityMetrics>({
    latency: {
      current: 0, average: 0, min: 0, max: 0, p50: 0, p95: 0, p99: 0, jitter: 0
    },
    throughput: {
      messagesPerSecond: 0, bytesPerSecond: 0, peakThroughput: 0, averageThroughput: 0
    },
    reliability: {
      uptime: 0, uptimePercentage: 0, reconnectionCount: 0, lastReconnection: null,
      connectionStability: 0, errorRate: 0, successRate: 0
    },
    performance: {
      qualityScore: 0, stabilityScore: 0, healthScore: 0, performanceGrade: 'F'
    }
  });
  
  const [diagnostics, setDiagnostics] = useState<ConnectionDiagnostics>({
    issues: [],
    summary: {
      totalIssues: 0,
      criticalIssues: 0,
      recommendedActions: [],
      overallHealth: 'critical'
    }
  });
  
  // Refs
  const monitoringIntervalRef = useRef<NodeJS.Timeout>();
  const latencyBuffer = useRef<number[]>([]);
  const throughputBuffer = useRef<{ timestamp: number; messages: number; bytes: number }[]>([]);
  const errorBuffer = useRef<string[]>([]);
  const connectionStartTime = useRef<number>(0);
  const lastMetricsUpdate = useRef<number>(0);
  const isMountedRef = useRef(true);
  
  // WebSocket manager
  const {
    connectionState,
    messageLatency,
    messagesReceived,
    messagesSent,
    connectionAttempts,
    isReconnecting,
    reconnect,
    getConnectionInfo,
    getMessageStats
  } = useWebSocketManager();
  
  // Cleanup on unmount
  useEffect(() => {
    return () => {
      isMountedRef.current = false;
      if (monitoringIntervalRef.current) {
        clearInterval(monitoringIntervalRef.current);
      }
    };
  }, []);
  
  // Track connection state changes
  useEffect(() => {
    if (connectionState === 'connected' && connectionStartTime.current === 0) {
      connectionStartTime.current = Date.now();
    } else if (connectionState === 'disconnected') {
      connectionStartTime.current = 0;
    }
  }, [connectionState]);
  
  // Calculate percentiles
  const calculatePercentile = useCallback((values: number[], percentile: number): number => {
    if (values.length === 0) return 0;
    
    const sorted = [...values].sort((a, b) => a - b);
    const index = (percentile / 100) * (sorted.length - 1);
    const lower = Math.floor(index);
    const upper = Math.ceil(index);
    
    if (lower === upper) return sorted[lower];
    
    const weight = index - lower;
    return sorted[lower] * (1 - weight) + sorted[upper] * weight;
  }, []);
  
  // Calculate jitter
  const calculateJitter = useCallback((latencies: number[]): number => {
    if (latencies.length < 2) return 0;
    
    let jitterSum = 0;
    for (let i = 1; i < latencies.length; i++) {
      jitterSum += Math.abs(latencies[i] - latencies[i - 1]);
    }
    
    return jitterSum / (latencies.length - 1);
  }, []);
  
  // Update quality metrics
  const updateQualityMetrics = useCallback(() => {
    if (!isMountedRef.current) return;
    
    const now = Date.now();
    const connectionInfo = getConnectionInfo();
    const messageStats = getMessageStats();
    
    // Update latency buffer
    if (messageLatency > 0) {
      latencyBuffer.current.push(messageLatency);
      if (latencyBuffer.current.length > 100) {
        latencyBuffer.current.shift();
      }
    }
    
    // Update throughput buffer
    const timeSinceLastUpdate = lastMetricsUpdate.current > 0 ? now - lastMetricsUpdate.current : 0;
    if (timeSinceLastUpdate > 0) {
      throughputBuffer.current.push({
        timestamp: now,
        messages: messagesReceived + messagesSent,
        bytes: messageStats.totalMessages * 100 // Approximate
      });
      
      if (throughputBuffer.current.length > 60) {
        throughputBuffer.current.shift();
      }
    }
    
    // Calculate latency metrics
    const latencies = latencyBuffer.current;
    const latencyMetrics = {
      current: messageLatency,
      average: latencies.length > 0 ? latencies.reduce((sum, lat) => sum + lat, 0) / latencies.length : 0,
      min: latencies.length > 0 ? Math.min(...latencies) : 0,
      max: latencies.length > 0 ? Math.max(...latencies) : 0,
      p50: calculatePercentile(latencies, 50),
      p95: calculatePercentile(latencies, 95),
      p99: calculatePercentile(latencies, 99),
      jitter: calculateJitter(latencies)
    };
    
    // Calculate throughput metrics
    const throughputData = throughputBuffer.current;
    let messagesPerSecond = 0;
    let bytesPerSecond = 0;
    let peakThroughput = 0;
    
    if (throughputData.length > 1) {
      const timeSpan = (throughputData[throughputData.length - 1].timestamp - throughputData[0].timestamp) / 1000;
      const messageCount = throughputData[throughputData.length - 1].messages - throughputData[0].messages;
      const byteCount = throughputData[throughputData.length - 1].bytes - throughputData[0].bytes;
      
      messagesPerSecond = timeSpan > 0 ? messageCount / timeSpan : 0;
      bytesPerSecond = timeSpan > 0 ? byteCount / timeSpan : 0;
      peakThroughput = Math.max(...throughputData.map(d => d.messages));
    }
    
    // Calculate reliability metrics
    const uptime = connectionStartTime.current > 0 ? now - connectionStartTime.current : 0;
    const totalTime = uptime + (connectionAttempts * 5000); // Estimate downtime
    const uptimePercentage = totalTime > 0 ? (uptime / totalTime) * 100 : 0;
    const connectionStability = Math.max(0, Math.min(100, 100 - (connectionAttempts * 10)));
    const errorRate = messageStats.totalMessages > 0 ? messageStats.errorCount / messageStats.totalMessages : 0;
    const successRate = 1 - errorRate;
    
    // Calculate performance scores
    const qualityScore = Math.max(0, Math.min(100,
      100 - (latencyMetrics.average / 10) - (errorRate * 100) - (latencyMetrics.jitter / 5)
    ));
    
    const stabilityScore = Math.max(0, Math.min(100,
      uptimePercentage * 0.7 + connectionStability * 0.3
    ));
    
    const healthScore = (qualityScore + stabilityScore) / 2;
    
    let performanceGrade: 'A' | 'B' | 'C' | 'D' | 'F' = 'F';
    if (healthScore >= 90) performanceGrade = 'A';
    else if (healthScore >= 80) performanceGrade = 'B';
    else if (healthScore >= 70) performanceGrade = 'C';
    else if (healthScore >= 60) performanceGrade = 'D';
    
    // Update state
    const newQualityMetrics: ConnectionQualityMetrics = {
      latency: latencyMetrics,
      throughput: {
        messagesPerSecond,
        bytesPerSecond,
        peakThroughput,
        averageThroughput: messagesPerSecond
      },
      reliability: {
        uptime,
        uptimePercentage,
        reconnectionCount: connectionAttempts,
        lastReconnection: connectionAttempts > 0 ? new Date().toISOString() : null,
        connectionStability,
        errorRate,
        successRate
      },
      performance: {
        qualityScore,
        stabilityScore,
        healthScore,
        performanceGrade
      }
    };
    
    setQualityMetrics(newQualityMetrics);
    setHealthGrade(performanceGrade);
    setIsHealthy(healthScore > 70 && errorRate < errorThreshold && latencyMetrics.average < latencyThreshold);
    
    // Update connection health
    setConnectionHealth({
      isHealthy: healthScore > 70,
      qualityScore,
      stabilityScore,
      latencyStats: {
        current: latencyMetrics.current,
        average: latencyMetrics.average,
        min: latencyMetrics.min,
        max: latencyMetrics.max,
        p95: latencyMetrics.p95
      },
      throughputStats: {
        current: messagesPerSecond,
        average: messagesPerSecond,
        peak: peakThroughput
      },
      errorStats: {
        total: messageStats.errorCount,
        rate: errorRate,
        recentErrors: errorBuffer.current.slice(-5)
      },
      connectionStats: {
        uptime,
        reconnections: connectionAttempts,
        lastReconnection: connectionAttempts > 0 ? new Date().toISOString() : undefined
      }
    });
    
    // Add to history
    setHealthHistory(prev => {
      const newHistory = [...prev, newQualityMetrics];
      return newHistory.slice(-healthHistorySize);
    });
    
    lastMetricsUpdate.current = now;
  }, [
    messageLatency, messagesReceived, messagesSent, connectionAttempts,
    getConnectionInfo, getMessageStats, calculatePercentile, calculateJitter,
    errorThreshold, latencyThreshold, healthHistorySize
  ]);
  
  // Run diagnostics
  const runDiagnostics = useCallback(async (): Promise<ConnectionDiagnostics> => {
    const issues: ConnectionDiagnostics['issues'] = [];
    
    // Latency issues
    if (qualityMetrics.latency.average > latencyThreshold) {
      issues.push({
        id: 'high_latency',
        type: 'latency',
        severity: qualityMetrics.latency.average > latencyThreshold * 2 ? 'critical' : 'high',
        message: `Average latency (${qualityMetrics.latency.average.toFixed(0)}ms) exceeds threshold (${latencyThreshold}ms)`,
        recommendation: 'Check network connection and server performance',
        detectedAt: new Date().toISOString()
      });
    }
    
    // Jitter issues
    if (qualityMetrics.latency.jitter > 100) {
      issues.push({
        id: 'high_jitter',
        type: 'latency',
        severity: qualityMetrics.latency.jitter > 500 ? 'high' : 'medium',
        message: `Network jitter is high (${qualityMetrics.latency.jitter.toFixed(0)}ms)`,
        recommendation: 'Check for network congestion or unstable connection',
        detectedAt: new Date().toISOString()
      });
    }
    
    // Stability issues
    if (qualityMetrics.reliability.connectionStability < stabilityThreshold * 100) {
      issues.push({
        id: 'connection_instability',
        type: 'stability',
        severity: qualityMetrics.reliability.connectionStability < 50 ? 'critical' : 'high',
        message: `Connection stability is low (${qualityMetrics.reliability.connectionStability.toFixed(1)}%)`,
        recommendation: 'Check network stability and consider increasing reconnection intervals',
        detectedAt: new Date().toISOString()
      });
    }
    
    // Error rate issues
    if (qualityMetrics.reliability.errorRate > errorThreshold) {
      issues.push({
        id: 'high_error_rate',
        type: 'error',
        severity: qualityMetrics.reliability.errorRate > errorThreshold * 2 ? 'critical' : 'high',
        message: `Error rate is high (${(qualityMetrics.reliability.errorRate * 100).toFixed(1)}%)`,
        recommendation: 'Check server logs and validate message formats',
        detectedAt: new Date().toISOString()
      });
    }
    
    // Throughput issues
    if (qualityMetrics.throughput.messagesPerSecond < 1 && connectionState === 'connected') {
      issues.push({
        id: 'low_throughput',
        type: 'throughput',
        severity: 'medium',
        message: 'Message throughput is very low',
        recommendation: 'Check if data sources are active and subscriptions are working',
        detectedAt: new Date().toISOString()
      });
    }
    
    const criticalIssues = issues.filter(issue => issue.severity === 'critical').length;
    const totalIssues = issues.length;
    
    const recommendedActions = [
      ...new Set(issues.map(issue => issue.recommendation))
    ];
    
    let overallHealth: ConnectionDiagnostics['summary']['overallHealth'] = 'excellent';
    if (criticalIssues > 0) overallHealth = 'critical';
    else if (totalIssues > 3) overallHealth = 'poor';
    else if (totalIssues > 1) overallHealth = 'fair';
    else if (totalIssues > 0) overallHealth = 'good';
    
    const diagnosticsResult: ConnectionDiagnostics = {
      issues,
      summary: {
        totalIssues,
        criticalIssues,
        recommendedActions,
        overallHealth
      }
    };
    
    setDiagnostics(diagnosticsResult);
    return diagnosticsResult;
  }, [qualityMetrics, connectionState, latencyThreshold, stabilityThreshold, errorThreshold]);
  
  // Get health trend
  const getHealthTrend = useCallback((minutes = 5): 'improving' | 'stable' | 'declining' => {
    if (healthHistory.length < 10) return 'stable';
    
    const cutoffTime = Date.now() - (minutes * 60 * 1000);
    const recentHistory = healthHistory.filter((_, index) => 
      index >= healthHistory.length - Math.min(healthHistory.length, minutes * 60 / (monitoringInterval / 1000))
    );
    
    if (recentHistory.length < 3) return 'stable';
    
    const firstThird = recentHistory.slice(0, Math.floor(recentHistory.length / 3));
    const lastThird = recentHistory.slice(-Math.floor(recentHistory.length / 3));
    
    const firstAvgHealth = firstThird.reduce((sum, metrics) => sum + metrics.performance.healthScore, 0) / firstThird.length;
    const lastAvgHealth = lastThird.reduce((sum, metrics) => sum + metrics.performance.healthScore, 0) / lastThird.length;
    
    const difference = lastAvgHealth - firstAvgHealth;
    
    if (difference > 5) return 'improving';
    if (difference < -5) return 'declining';
    return 'stable';
  }, [healthHistory, monitoringInterval]);
  
  // Update trend
  useEffect(() => {
    const trend = getHealthTrend();
    setHealthTrend(trend);
  }, [healthHistory, getHealthTrend]);
  
  // Auto-recovery logic
  const triggerRecovery = useCallback(async () => {
    if (connectionState === 'connected' && qualityMetrics.performance.healthScore < 50) {
      await reconnect();
    }
  }, [connectionState, qualityMetrics.performance.healthScore, reconnect]);
  
  // Auto-recovery trigger
  useEffect(() => {
    if (autoRecoveryEnabled && !isHealthy && connectionState === 'connected') {
      const criticalIssues = diagnostics.issues.filter(issue => issue.severity === 'critical');
      if (criticalIssues.length > 0) {
        triggerRecovery();
      }
    }
  }, [autoRecoveryEnabled, isHealthy, connectionState, diagnostics.issues, triggerRecovery]);
  
  // Start monitoring
  const startMonitoring = useCallback(() => {
    if (isMonitoring) return;
    
    setIsMonitoring(true);
    monitoringIntervalRef.current = setInterval(updateQualityMetrics, monitoringInterval);
  }, [isMonitoring, updateQualityMetrics, monitoringInterval]);
  
  // Stop monitoring
  const stopMonitoring = useCallback(() => {
    setIsMonitoring(false);
    if (monitoringIntervalRef.current) {
      clearInterval(monitoringIntervalRef.current);
    }
  }, []);
  
  // Reset metrics
  const resetMetrics = useCallback(() => {
    latencyBuffer.current = [];
    throughputBuffer.current = [];
    errorBuffer.current = [];
    setHealthHistory([]);
    lastMetricsUpdate.current = 0;
  }, []);
  
  // Auto-start monitoring when connected
  useEffect(() => {
    if (connectionState === 'connected' && !isMonitoring) {
      startMonitoring();
    } else if (connectionState === 'disconnected' && isMonitoring) {
      stopMonitoring();
    }
  }, [connectionState, isMonitoring, startMonitoring, stopMonitoring]);
  
  // Run diagnostics periodically
  useEffect(() => {
    if (enableDiagnostics && isMonitoring) {
      const diagnosticsInterval = setInterval(runDiagnostics, 30000); // Every 30 seconds
      return () => clearInterval(diagnosticsInterval);
    }
  }, [enableDiagnostics, isMonitoring, runDiagnostics]);
  
  return {
    // Connection health
    connectionHealth,
    qualityMetrics,
    diagnostics,
    
    // Health status
    isHealthy,
    healthGrade,
    healthTrend,
    
    // Monitoring control
    startMonitoring,
    stopMonitoring,
    resetMetrics,
    runDiagnostics,
    
    // Health history
    healthHistory,
    getHealthTrend,
    
    // Auto-recovery
    triggerRecovery,
    enableAutoRecovery: () => setAutoRecoveryEnabled(true),
    disableAutoRecovery: () => setAutoRecoveryEnabled(false)
  };
}

export default useWebSocketHealth;