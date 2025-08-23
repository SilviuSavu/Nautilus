/**
 * useWebSocketReconnect Hook
 * Sprint 3: Advanced WebSocket Reconnection Management
 * 
 * Intelligent reconnection logic with exponential backoff, connection quality assessment,
 * smart retry strategies, and failover mechanisms for maximum reliability.
 */

import { useState, useEffect, useCallback, useRef } from 'react';
import { useWebSocketManager } from './useWebSocketManager';

export interface ReconnectionStrategy {
  name: 'exponential' | 'linear' | 'fibonacci' | 'custom';
  baseDelay: number;
  maxDelay: number;
  multiplier?: number;
  jitter?: boolean;
  customDelayFunction?: (attempt: number) => number;
}

export interface ReconnectionConfig {
  enabled: boolean;
  maxAttempts: number;
  strategy: ReconnectionStrategy;
  healthCheckInterval: number;
  connectionQualityThreshold: number;
  enableFailover: boolean;
  failoverUrls?: string[];
  enableCircuitBreaker: boolean;
  circuitBreakerThreshold: number;
  circuitBreakerTimeout: number;
}

export interface ReconnectionState {
  isReconnecting: boolean;
  currentAttempt: number;
  totalAttempts: number;
  nextRetryIn: number;
  lastReconnectionTime: string | null;
  consecutiveFailures: number;
  circuitBreakerOpen: boolean;
  currentUrl: string;
  failoverActive: boolean;
}

export interface ReconnectionStats {
  totalReconnections: number;
  successfulReconnections: number;
  failedReconnections: number;
  averageReconnectionTime: number;
  longestDowntime: number;
  shortestDowntime: number;
  successRate: number;
  reliability: number;
  uptime: number;
  downtimeHistory: { start: string; end: string; duration: number }[];
}

export interface UseWebSocketReconnectReturn {
  // Reconnection state
  reconnectionState: ReconnectionState;
  reconnectionStats: ReconnectionStats;
  
  // Configuration
  updateConfig: (config: Partial<ReconnectionConfig>) => void;
  getConfig: () => ReconnectionConfig;
  
  // Manual control
  forceReconnect: () => Promise<void>;
  cancelReconnection: () => void;
  pauseReconnection: () => void;
  resumeReconnection: () => void;
  
  // Circuit breaker
  openCircuitBreaker: () => void;
  closeCircuitBreaker: () => void;
  getCircuitBreakerState: () => 'closed' | 'open' | 'half-open';
  
  // Failover
  triggerFailover: () => Promise<void>;
  resetToOriginalUrl: () => Promise<void>;
  getActiveUrl: () => string;
  
  // Health monitoring
  checkConnectionHealth: () => Promise<boolean>;
  setHealthCheck: (healthCheckFn: () => Promise<boolean>) => void;
  
  // Statistics
  resetStats: () => void;
  exportStats: () => ReconnectionStats;
}

const DEFAULT_CONFIG: ReconnectionConfig = {
  enabled: true,
  maxAttempts: 10,
  strategy: {
    name: 'exponential',
    baseDelay: 1000,
    maxDelay: 30000,
    multiplier: 1.5,
    jitter: true
  },
  healthCheckInterval: 30000,
  connectionQualityThreshold: 0.7,
  enableFailover: false,
  failoverUrls: [],
  enableCircuitBreaker: true,
  circuitBreakerThreshold: 5,
  circuitBreakerTimeout: 60000
};

export function useWebSocketReconnect(
  initialConfig?: Partial<ReconnectionConfig>
): UseWebSocketReconnectReturn {
  // State
  const [config, setConfig] = useState<ReconnectionConfig>({
    ...DEFAULT_CONFIG,
    ...initialConfig
  });
  
  const [reconnectionState, setReconnectionState] = useState<ReconnectionState>({
    isReconnecting: false,
    currentAttempt: 0,
    totalAttempts: 0,
    nextRetryIn: 0,
    lastReconnectionTime: null,
    consecutiveFailures: 0,
    circuitBreakerOpen: false,
    currentUrl: '',
    failoverActive: false
  });
  
  const [reconnectionStats, setReconnectionStats] = useState<ReconnectionStats>({
    totalReconnections: 0,
    successfulReconnections: 0,
    failedReconnections: 0,
    averageReconnectionTime: 0,
    longestDowntime: 0,
    shortestDowntime: 0,
    successRate: 0,
    reliability: 0,
    uptime: 0,
    downtimeHistory: []
  });
  
  // Refs
  const reconnectionTimeoutRef = useRef<NodeJS.Timeout>();
  const healthCheckIntervalRef = useRef<NodeJS.Timeout>();
  const circuitBreakerTimeoutRef = useRef<NodeJS.Timeout>();
  const countdownIntervalRef = useRef<NodeJS.Timeout>();
  const isPausedRef = useRef(false);
  const lastConnectionTime = useRef<number>(0);
  const disconnectionStartTime = useRef<number>(0);
  const customHealthCheckRef = useRef<(() => Promise<boolean>) | null>(null);
  const originalUrlRef = useRef<string>('');
  const failoverIndexRef = useRef<number>(0);
  const reconnectionTimesRef = useRef<number[]>([]);
  const isMountedRef = useRef(true);
  
  // WebSocket manager
  const {
    connectionState,
    connectionError,
    connectionAttempts,
    connect,
    disconnect,
    reconnect,
    getConnectionInfo
  } = useWebSocketManager();
  
  // Cleanup on unmount
  useEffect(() => {
    return () => {
      isMountedRef.current = false;
      clearAllTimeouts();
    };
  }, []);
  
  // Clear all timeouts
  const clearAllTimeouts = useCallback(() => {
    if (reconnectionTimeoutRef.current) {
      clearTimeout(reconnectionTimeoutRef.current);
    }
    if (healthCheckIntervalRef.current) {
      clearInterval(healthCheckIntervalRef.current);
    }
    if (circuitBreakerTimeoutRef.current) {
      clearTimeout(circuitBreakerTimeoutRef.current);
    }
    if (countdownIntervalRef.current) {
      clearInterval(countdownIntervalRef.current);
    }
  }, []);
  
  // Calculate reconnection delay
  const calculateReconnectionDelay = useCallback((attempt: number): number => {
    const { strategy } = config;
    let delay = strategy.baseDelay;
    
    switch (strategy.name) {
      case 'exponential':
        delay = Math.min(
          strategy.baseDelay * Math.pow(strategy.multiplier || 2, attempt - 1),
          strategy.maxDelay
        );
        break;
        
      case 'linear':
        delay = Math.min(
          strategy.baseDelay + (attempt - 1) * (strategy.multiplier || 1000),
          strategy.maxDelay
        );
        break;
        
      case 'fibonacci':
        const fib = (n: number): number => n <= 1 ? 1 : fib(n - 1) + fib(n - 2);
        delay = Math.min(
          strategy.baseDelay * fib(attempt),
          strategy.maxDelay
        );
        break;
        
      case 'custom':
        if (strategy.customDelayFunction) {
          delay = Math.min(strategy.customDelayFunction(attempt), strategy.maxDelay);
        }
        break;
    }
    
    // Add jitter to prevent thundering herd
    if (strategy.jitter) {
      const jitterAmount = delay * 0.1;
      delay += (Math.random() * 2 - 1) * jitterAmount;
    }
    
    return Math.max(delay, 100); // Minimum 100ms delay
  }, [config]);
  
  // Update reconnection stats
  const updateStats = useCallback((success: boolean, reconnectionTime?: number) => {
    setReconnectionStats(prev => {
      const newStats = { ...prev };
      
      newStats.totalReconnections++;
      
      if (success) {
        newStats.successfulReconnections++;
        if (reconnectionTime) {
          reconnectionTimesRef.current.push(reconnectionTime);
          if (reconnectionTimesRef.current.length > 100) {
            reconnectionTimesRef.current.shift();
          }
          
          newStats.averageReconnectionTime = 
            reconnectionTimesRef.current.reduce((sum, time) => sum + time, 0) / 
            reconnectionTimesRef.current.length;
        }
      } else {
        newStats.failedReconnections++;
      }
      
      newStats.successRate = newStats.totalReconnections > 0 
        ? newStats.successfulReconnections / newStats.totalReconnections 
        : 0;
      
      newStats.reliability = Math.max(0, Math.min(1, newStats.successRate * 0.8 + 
        (1 - Math.min(prev.consecutiveFailures / 10, 1)) * 0.2));
      
      return newStats;
    });
  }, []);
  
  // Start countdown timer
  const startCountdown = useCallback((delay: number) => {
    let remaining = delay;
    
    setReconnectionState(prev => ({ ...prev, nextRetryIn: remaining }));
    
    countdownIntervalRef.current = setInterval(() => {
      remaining -= 1000;
      if (remaining <= 0) {
        clearInterval(countdownIntervalRef.current!);
        setReconnectionState(prev => ({ ...prev, nextRetryIn: 0 }));
      } else {
        setReconnectionState(prev => ({ ...prev, nextRetryIn: remaining }));
      }
    }, 1000);
  }, []);
  
  // Perform reconnection attempt
  const performReconnection = useCallback(async () => {
    if (!isMountedRef.current || !config.enabled || isPausedRef.current) {
      return;
    }
    
    const attemptStartTime = Date.now();
    
    setReconnectionState(prev => ({
      ...prev,
      currentAttempt: prev.currentAttempt + 1,
      totalAttempts: prev.totalAttempts + 1
    }));
    
    try {
      await reconnect();
      
      // Success
      const reconnectionTime = Date.now() - attemptStartTime;
      updateStats(true, reconnectionTime);
      
      setReconnectionState(prev => ({
        ...prev,
        isReconnecting: false,
        currentAttempt: 0,
        consecutiveFailures: 0,
        lastReconnectionTime: new Date().toISOString()
      }));
      
      // Close circuit breaker on success
      if (reconnectionState.circuitBreakerOpen) {
        closeCircuitBreaker();
      }
      
    } catch (error) {
      // Failure
      updateStats(false);
      
      setReconnectionState(prev => ({
        ...prev,
        consecutiveFailures: prev.consecutiveFailures + 1
      }));
      
      // Check circuit breaker
      if (config.enableCircuitBreaker && 
          reconnectionState.consecutiveFailures >= config.circuitBreakerThreshold) {
        openCircuitBreaker();
        return;
      }
      
      // Check max attempts
      if (reconnectionState.currentAttempt >= config.maxAttempts) {
        if (config.enableFailover && config.failoverUrls && config.failoverUrls.length > 0) {
          await triggerFailover();
        } else {
          setReconnectionState(prev => ({ ...prev, isReconnecting: false }));
        }
        return;
      }
      
      // Schedule next attempt
      scheduleReconnection();
    }
  }, [config, reconnectionState, reconnect, updateStats]);
  
  // Schedule next reconnection attempt
  const scheduleReconnection = useCallback(() => {
    if (!config.enabled || reconnectionState.circuitBreakerOpen) return;
    
    const delay = calculateReconnectionDelay(reconnectionState.currentAttempt + 1);
    startCountdown(delay);
    
    reconnectionTimeoutRef.current = setTimeout(() => {
      if (isMountedRef.current && !isPausedRef.current) {
        performReconnection();
      }
    }, delay);
  }, [config.enabled, reconnectionState, calculateReconnectionDelay, startCountdown, performReconnection]);
  
  // Trigger failover
  const triggerFailover = useCallback(async () => {
    if (!config.enableFailover || !config.failoverUrls || config.failoverUrls.length === 0) {
      return;
    }
    
    // Store original URL if not already stored
    if (!originalUrlRef.current) {
      const connectionInfo = getConnectionInfo();
      originalUrlRef.current = connectionInfo.connectionId || '';
    }
    
    // Try next failover URL
    failoverIndexRef.current = (failoverIndexRef.current + 1) % config.failoverUrls.length;
    const failoverUrl = config.failoverUrls[failoverIndexRef.current];
    
    setReconnectionState(prev => ({
      ...prev,
      currentUrl: failoverUrl,
      failoverActive: true,
      currentAttempt: 0,
      consecutiveFailures: 0
    }));
    
    try {
      // Disconnect and reconnect with new URL
      await disconnect();
      // Note: In a real implementation, you'd update the WebSocket URL here
      await connect();
    } catch (error) {
      // If failover fails, try next URL or give up
      if (failoverIndexRef.current < config.failoverUrls.length - 1) {
        await triggerFailover();
      }
    }
  }, [config, getConnectionInfo, disconnect, connect]);
  
  // Reset to original URL
  const resetToOriginalUrl = useCallback(async () => {
    if (!originalUrlRef.current) return;
    
    setReconnectionState(prev => ({
      ...prev,
      currentUrl: originalUrlRef.current,
      failoverActive: false,
      currentAttempt: 0
    }));
    
    await disconnect();
    // Note: In a real implementation, you'd reset the WebSocket URL here
    await connect();
  }, [disconnect, connect]);
  
  // Open circuit breaker
  const openCircuitBreaker = useCallback(() => {
    setReconnectionState(prev => ({ ...prev, circuitBreakerOpen: true }));
    
    // Schedule circuit breaker timeout
    circuitBreakerTimeoutRef.current = setTimeout(() => {
      if (isMountedRef.current) {
        setReconnectionState(prev => ({ ...prev, circuitBreakerOpen: false }));
        // Attempt reconnection when circuit breaker closes
        scheduleReconnection();
      }
    }, config.circuitBreakerTimeout);
  }, [config.circuitBreakerTimeout, scheduleReconnection]);
  
  // Close circuit breaker
  const closeCircuitBreaker = useCallback(() => {
    setReconnectionState(prev => ({ ...prev, circuitBreakerOpen: false }));
    
    if (circuitBreakerTimeoutRef.current) {
      clearTimeout(circuitBreakerTimeoutRef.current);
    }
  }, []);
  
  // Get circuit breaker state
  const getCircuitBreakerState = useCallback((): 'closed' | 'open' | 'half-open' => {
    if (reconnectionState.circuitBreakerOpen) return 'open';
    if (reconnectionState.consecutiveFailures > 0) return 'half-open';
    return 'closed';
  }, [reconnectionState]);
  
  // Check connection health
  const checkConnectionHealth = useCallback(async (): Promise<boolean> => {
    if (customHealthCheckRef.current) {
      return customHealthCheckRef.current();
    }
    
    // Default health check
    return connectionState === 'connected';
  }, [connectionState]);
  
  // Force reconnection
  const forceReconnect = useCallback(async () => {
    clearAllTimeouts();
    
    setReconnectionState(prev => ({
      ...prev,
      isReconnecting: true,
      currentAttempt: 0
    }));
    
    await performReconnection();
  }, [clearAllTimeouts, performReconnection]);
  
  // Cancel reconnection
  const cancelReconnection = useCallback(() => {
    clearAllTimeouts();
    setReconnectionState(prev => ({ ...prev, isReconnecting: false }));
  }, [clearAllTimeouts]);
  
  // Pause reconnection
  const pauseReconnection = useCallback(() => {
    isPausedRef.current = true;
    clearAllTimeouts();
  }, [clearAllTimeouts]);
  
  // Resume reconnection
  const resumeReconnection = useCallback(() => {
    isPausedRef.current = false;
    if (connectionState !== 'connected' && config.enabled) {
      scheduleReconnection();
    }
  }, [connectionState, config.enabled, scheduleReconnection]);
  
  // Update configuration
  const updateConfig = useCallback((newConfig: Partial<ReconnectionConfig>) => {
    setConfig(prev => ({ ...prev, ...newConfig }));
  }, []);
  
  // Get configuration
  const getConfig = useCallback(() => config, [config]);
  
  // Set custom health check
  const setHealthCheck = useCallback((healthCheckFn: () => Promise<boolean>) => {
    customHealthCheckRef.current = healthCheckFn;
  }, []);
  
  // Get active URL
  const getActiveUrl = useCallback(() => reconnectionState.currentUrl, [reconnectionState.currentUrl]);
  
  // Reset statistics
  const resetStats = useCallback(() => {
    setReconnectionStats({
      totalReconnections: 0,
      successfulReconnections: 0,
      failedReconnections: 0,
      averageReconnectionTime: 0,
      longestDowntime: 0,
      shortestDowntime: 0,
      successRate: 0,
      reliability: 0,
      uptime: 0,
      downtimeHistory: []
    });
    reconnectionTimesRef.current = [];
  }, []);
  
  // Export statistics
  const exportStats = useCallback(() => reconnectionStats, [reconnectionStats]);
  
  // Monitor connection state changes
  useEffect(() => {
    if (connectionState === 'connected') {
      lastConnectionTime.current = Date.now();
      
      // Stop reconnection if it was in progress
      if (reconnectionState.isReconnecting) {
        clearAllTimeouts();
        setReconnectionState(prev => ({
          ...prev,
          isReconnecting: false,
          currentAttempt: 0,
          consecutiveFailures: 0
        }));
      }
      
      // Record downtime if we were disconnected
      if (disconnectionStartTime.current > 0) {
        const downtime = Date.now() - disconnectionStartTime.current;
        setReconnectionStats(prev => ({
          ...prev,
          downtimeHistory: [...prev.downtimeHistory, {
            start: new Date(disconnectionStartTime.current).toISOString(),
            end: new Date().toISOString(),
            duration: downtime
          }].slice(-50), // Keep last 50 downtime records
          longestDowntime: Math.max(prev.longestDowntime, downtime),
          shortestDowntime: prev.shortestDowntime === 0 ? downtime : Math.min(prev.shortestDowntime, downtime)
        }));
        disconnectionStartTime.current = 0;
      }
      
    } else if (connectionState === 'disconnected' && lastConnectionTime.current > 0) {
      disconnectionStartTime.current = Date.now();
      
      // Start reconnection if enabled and not already reconnecting
      if (config.enabled && !reconnectionState.isReconnecting && !isPausedRef.current) {
        setReconnectionState(prev => ({ ...prev, isReconnecting: true, currentAttempt: 0 }));
        scheduleReconnection();
      }
    }
  }, [connectionState, reconnectionState.isReconnecting, config.enabled, clearAllTimeouts, scheduleReconnection]);
  
  // Health check monitoring
  useEffect(() => {
    if (config.healthCheckInterval > 0) {
      healthCheckIntervalRef.current = setInterval(async () => {
        if (connectionState === 'connected') {
          const isHealthy = await checkConnectionHealth();
          if (!isHealthy) {
            forceReconnect();
          }
        }
      }, config.healthCheckInterval);
      
      return () => {
        if (healthCheckIntervalRef.current) {
          clearInterval(healthCheckIntervalRef.current);
        }
      };
    }
  }, [config.healthCheckInterval, connectionState, checkConnectionHealth, forceReconnect]);
  
  // Calculate uptime
  useEffect(() => {
    const uptimeInterval = setInterval(() => {
      if (connectionState === 'connected' && lastConnectionTime.current > 0) {
        const uptime = Date.now() - lastConnectionTime.current;
        setReconnectionStats(prev => ({ ...prev, uptime }));
      }
    }, 1000);
    
    return () => clearInterval(uptimeInterval);
  }, [connectionState]);
  
  return {
    // Reconnection state
    reconnectionState,
    reconnectionStats,
    
    // Configuration
    updateConfig,
    getConfig,
    
    // Manual control
    forceReconnect,
    cancelReconnection,
    pauseReconnection,
    resumeReconnection,
    
    // Circuit breaker
    openCircuitBreaker,
    closeCircuitBreaker,
    getCircuitBreakerState,
    
    // Failover
    triggerFailover,
    resetToOriginalUrl,
    getActiveUrl,
    
    // Health monitoring
    checkConnectionHealth,
    setHealthCheck,
    
    // Statistics
    resetStats,
    exportStats
  };
}

export default useWebSocketReconnect;