/**
 * WebSocket Utilities
 * Sprint 3: Advanced WebSocket Management Utilities
 * 
 * Comprehensive utilities for WebSocket connection management,
 * message handling, performance optimization, and error recovery.
 */

import type { 
  WebSocketMessage, 
  ConnectionState, 
  MessageType,
  SubscriptionFilters,
  ConnectionHealth,
  WebSocketPerformanceMetrics
} from '../types/websocket';

// Connection Management Utilities
export class WebSocketConnectionManager {
  private connections: Map<string, WebSocket> = new Map();
  private connectionStates: Map<string, ConnectionState> = new Map();
  private reconnectTimers: Map<string, NodeJS.Timeout> = new Map();
  private healthChecks: Map<string, NodeJS.Timeout> = new Map();

  // Create a new WebSocket connection
  connect(
    connectionId: string,
    url: string,
    options: {
      protocols?: string[];
      autoReconnect?: boolean;
      reconnectInterval?: number;
      maxReconnectAttempts?: number;
    } = {}
  ): Promise<WebSocket> {
    return new Promise((resolve, reject) => {
      try {
        const ws = new WebSocket(url, options.protocols);
        
        ws.onopen = () => {
          this.connections.set(connectionId, ws);
          this.connectionStates.set(connectionId, 'connected');
          
          if (options.autoReconnect !== false) {
            this.startHealthCheck(connectionId, ws);
          }
          
          resolve(ws);
        };

        ws.onerror = (error) => {
          this.connectionStates.set(connectionId, 'error');
          reject(error);
        };

        ws.onclose = (event) => {
          this.connectionStates.set(connectionId, 'disconnected');
          this.connections.delete(connectionId);
          
          if (options.autoReconnect !== false && event.code !== 1000) {
            this.scheduleReconnect(connectionId, url, options);
          }
        };

        this.connectionStates.set(connectionId, 'connecting');
      } catch (error) {
        this.connectionStates.set(connectionId, 'error');
        reject(error);
      }
    });
  }

  // Disconnect a WebSocket connection
  disconnect(connectionId: string, code = 1000, reason = 'Client disconnect'): void {
    const ws = this.connections.get(connectionId);
    if (ws) {
      this.clearReconnectTimer(connectionId);
      this.clearHealthCheck(connectionId);
      ws.close(code, reason);
      this.connections.delete(connectionId);
      this.connectionStates.set(connectionId, 'disconnected');
    }
  }

  // Get connection state
  getConnectionState(connectionId: string): ConnectionState {
    return this.connectionStates.get(connectionId) || 'disconnected';
  }

  // Get WebSocket instance
  getConnection(connectionId: string): WebSocket | undefined {
    return this.connections.get(connectionId);
  }

  // Send message to specific connection
  sendMessage(connectionId: string, message: any): boolean {
    const ws = this.connections.get(connectionId);
    if (ws && ws.readyState === WebSocket.OPEN) {
      try {
        ws.send(JSON.stringify(message));
        return true;
      } catch (error) {
        console.error(`Failed to send message to ${connectionId}:`, error);
        return false;
      }
    }
    return false;
  }

  // Broadcast message to all connections
  broadcast(message: any, excludeConnections: string[] = []): number {
    let successCount = 0;
    
    this.connections.forEach((ws, connectionId) => {
      if (!excludeConnections.includes(connectionId)) {
        if (this.sendMessage(connectionId, message)) {
          successCount++;
        }
      }
    });
    
    return successCount;
  }

  // Private methods
  private scheduleReconnect(
    connectionId: string, 
    url: string, 
    options: any,
    attempt = 1
  ): void {
    if (options.maxReconnectAttempts && attempt > options.maxReconnectAttempts) {
      return;
    }

    const delay = Math.min((options.reconnectInterval || 5000) * Math.pow(2, attempt - 1), 30000);
    
    const timer = setTimeout(() => {
      this.connectionStates.set(connectionId, 'reconnecting');
      this.connect(connectionId, url, options)
        .catch(() => {
          this.scheduleReconnect(connectionId, url, options, attempt + 1);
        });
    }, delay);

    this.reconnectTimers.set(connectionId, timer);
  }

  private clearReconnectTimer(connectionId: string): void {
    const timer = this.reconnectTimers.get(connectionId);
    if (timer) {
      clearTimeout(timer);
      this.reconnectTimers.delete(connectionId);
    }
  }

  private startHealthCheck(connectionId: string, ws: WebSocket): void {
    const healthCheck = setInterval(() => {
      if (ws.readyState === WebSocket.OPEN) {
        try {
          ws.send(JSON.stringify({ type: 'ping', timestamp: Date.now() }));
        } catch (error) {
          this.clearHealthCheck(connectionId);
        }
      } else {
        this.clearHealthCheck(connectionId);
      }
    }, 30000);

    this.healthChecks.set(connectionId, healthCheck);
  }

  private clearHealthCheck(connectionId: string): void {
    const healthCheck = this.healthChecks.get(connectionId);
    if (healthCheck) {
      clearInterval(healthCheck);
      this.healthChecks.delete(connectionId);
    }
  }

  // Cleanup all connections
  cleanup(): void {
    this.connections.forEach((_, connectionId) => {
      this.disconnect(connectionId);
    });
    this.connections.clear();
    this.connectionStates.clear();
    this.reconnectTimers.clear();
    this.healthChecks.clear();
  }
}

// Message Processing Utilities
export class MessageProcessor {
  private messageHandlers: Map<string, (message: WebSocketMessage) => void> = new Map();
  private messageFilters: Map<string, (message: WebSocketMessage) => boolean> = new Map();
  private messageQueue: WebSocketMessage[] = [];
  private processingTimer: NodeJS.Timeout | null = null;
  private batchSize = 50;
  private processingInterval = 100; // ms

  // Add message handler
  addHandler(
    handlerId: string,
    handler: (message: WebSocketMessage) => void,
    filter?: (message: WebSocketMessage) => boolean
  ): () => void {
    this.messageHandlers.set(handlerId, handler);
    if (filter) {
      this.messageFilters.set(handlerId, filter);
    }

    return () => {
      this.messageHandlers.delete(handlerId);
      this.messageFilters.delete(handlerId);
    };
  }

  // Process incoming message
  processMessage(message: WebSocketMessage): void {
    this.messageQueue.push(message);
    this.startProcessing();
  }

  // Start batch processing
  private startProcessing(): void {
    if (this.processingTimer) return;

    this.processingTimer = setInterval(() => {
      this.processBatch();
    }, this.processingInterval);
  }

  // Process batch of messages
  private processBatch(): void {
    if (this.messageQueue.length === 0) {
      if (this.processingTimer) {
        clearInterval(this.processingTimer);
        this.processingTimer = null;
      }
      return;
    }

    const batch = this.messageQueue.splice(0, this.batchSize);
    
    batch.forEach(message => {
      this.messageHandlers.forEach((handler, handlerId) => {
        try {
          const filter = this.messageFilters.get(handlerId);
          if (!filter || filter(message)) {
            handler(message);
          }
        } catch (error) {
          console.error(`Error in message handler ${handlerId}:`, error);
        }
      });
    });
  }

  // Configure processing options
  configure(options: { batchSize?: number; processingInterval?: number }): void {
    if (options.batchSize) this.batchSize = options.batchSize;
    if (options.processingInterval) this.processingInterval = options.processingInterval;
  }

  // Clear all handlers and queue
  cleanup(): void {
    if (this.processingTimer) {
      clearInterval(this.processingTimer);
      this.processingTimer = null;
    }
    this.messageHandlers.clear();
    this.messageFilters.clear();
    this.messageQueue = [];
  }
}

// Performance Monitoring Utilities
export class WebSocketPerformanceMonitor {
  private metrics: WebSocketPerformanceMetrics;
  private startTime: number;
  private messageTimestamps: number[] = [];
  private latencyBuffer: number[] = [];
  private errorCount = 0;
  private connectionCount = 0;

  constructor() {
    this.startTime = Date.now();
    this.metrics = {
      messageLatency: 0,
      messagesPerSecond: 0,
      errorRate: 0,
      reconnectionCount: 0,
      uptime: 0,
      throughput: 0,
      packetLoss: 0,
      jitter: 0,
      qualityScore: 0
    };
  }

  // Record message received
  recordMessage(latency?: number): void {
    const now = Date.now();
    this.messageTimestamps.push(now);
    
    if (latency !== undefined) {
      this.latencyBuffer.push(latency);
      if (this.latencyBuffer.length > 1000) {
        this.latencyBuffer.shift();
      }
    }

    // Keep only last minute of timestamps
    const oneMinuteAgo = now - 60000;
    this.messageTimestamps = this.messageTimestamps.filter(t => t > oneMinuteAgo);
    
    this.updateMetrics();
  }

  // Record error
  recordError(): void {
    this.errorCount++;
    this.updateMetrics();
  }

  // Record reconnection
  recordReconnection(): void {
    this.connectionCount++;
    this.updateMetrics();
  }

  // Update metrics
  private updateMetrics(): void {
    const now = Date.now();
    const uptime = now - this.startTime;
    
    // Calculate messages per second
    const messagesPerSecond = this.messageTimestamps.length / Math.min(uptime / 1000, 60);
    
    // Calculate average latency
    const messageLatency = this.latencyBuffer.length > 0
      ? this.latencyBuffer.reduce((sum, lat) => sum + lat, 0) / this.latencyBuffer.length
      : 0;

    // Calculate jitter
    const jitter = this.calculateJitter();

    // Calculate error rate
    const totalMessages = this.messageTimestamps.length;
    const errorRate = totalMessages > 0 ? this.errorCount / totalMessages : 0;

    // Calculate quality score (0-100)
    const latencyScore = Math.max(0, 100 - (messageLatency / 10)); // Penalty for high latency
    const errorScore = Math.max(0, 100 - (errorRate * 1000)); // Penalty for errors
    const jitterScore = Math.max(0, 100 - jitter); // Penalty for jitter
    const qualityScore = (latencyScore + errorScore + jitterScore) / 3;

    this.metrics = {
      messageLatency,
      messagesPerSecond,
      errorRate,
      reconnectionCount: this.connectionCount,
      uptime,
      throughput: messagesPerSecond * 1000, // bytes per second estimate
      packetLoss: 0, // Would need additional tracking
      jitter,
      qualityScore
    };
  }

  // Calculate jitter
  private calculateJitter(): number {
    if (this.latencyBuffer.length < 2) return 0;
    
    let jitterSum = 0;
    for (let i = 1; i < this.latencyBuffer.length; i++) {
      jitterSum += Math.abs(this.latencyBuffer[i] - this.latencyBuffer[i - 1]);
    }
    
    return jitterSum / (this.latencyBuffer.length - 1);
  }

  // Get current metrics
  getMetrics(): WebSocketPerformanceMetrics {
    this.updateMetrics();
    return { ...this.metrics };
  }

  // Reset metrics
  reset(): void {
    this.startTime = Date.now();
    this.messageTimestamps = [];
    this.latencyBuffer = [];
    this.errorCount = 0;
    this.connectionCount = 0;
    this.updateMetrics();
  }
}

// Subscription Management Utilities
export class SubscriptionManager {
  private subscriptions: Map<string, {
    type: MessageType;
    filters: SubscriptionFilters;
    callback: (message: WebSocketMessage) => void;
    isActive: boolean;
    messageCount: number;
    lastActivity: string;
  }> = new Map();

  // Subscribe to message type
  subscribe(
    subscriptionId: string,
    messageType: MessageType,
    callback: (message: WebSocketMessage) => void,
    filters: SubscriptionFilters = {}
  ): () => void {
    this.subscriptions.set(subscriptionId, {
      type: messageType,
      filters,
      callback,
      isActive: true,
      messageCount: 0,
      lastActivity: new Date().toISOString()
    });

    return () => {
      this.unsubscribe(subscriptionId);
    };
  }

  // Unsubscribe from message type
  unsubscribe(subscriptionId: string): void {
    this.subscriptions.delete(subscriptionId);
  }

  // Process message for subscriptions
  processMessage(message: WebSocketMessage): void {
    this.subscriptions.forEach((subscription, id) => {
      if (subscription.isActive && this.matchesSubscription(message, subscription)) {
        try {
          subscription.callback(message);
          subscription.messageCount++;
          subscription.lastActivity = new Date().toISOString();
        } catch (error) {
          console.error(`Error in subscription callback ${id}:`, error);
        }
      }
    });
  }

  // Check if message matches subscription
  private matchesSubscription(
    message: WebSocketMessage,
    subscription: {
      type: MessageType;
      filters: SubscriptionFilters;
    }
  ): boolean {
    // Check message type
    if (message.type !== subscription.type) {
      return false;
    }

    // Apply filters
    const { filters } = subscription;

    if (filters.symbols && filters.symbols.length > 0) {
      const messageSymbol = (message as any).symbol;
      if (!messageSymbol || !filters.symbols.includes(messageSymbol)) {
        return false;
      }
    }

    if (filters.portfolio_ids && filters.portfolio_ids.length > 0) {
      const portfolioId = (message as any).portfolio_id;
      if (!portfolioId || !filters.portfolio_ids.includes(portfolioId)) {
        return false;
      }
    }

    if (filters.strategy_ids && filters.strategy_ids.length > 0) {
      const strategyId = (message as any).strategy_id;
      if (!strategyId || !filters.strategy_ids.includes(strategyId)) {
        return false;
      }
    }

    if (filters.severity) {
      const messageSeverity = (message as any).severity;
      if (!messageSeverity || messageSeverity !== filters.severity) {
        return false;
      }
    }

    return true;
  }

  // Get subscription statistics
  getSubscriptionStats(): Record<string, {
    type: MessageType;
    messageCount: number;
    lastActivity: string;
    isActive: boolean;
  }> {
    const stats: Record<string, any> = {};
    
    this.subscriptions.forEach((subscription, id) => {
      stats[id] = {
        type: subscription.type,
        messageCount: subscription.messageCount,
        lastActivity: subscription.lastActivity,
        isActive: subscription.isActive
      };
    });

    return stats;
  }

  // Activate/deactivate subscription
  setSubscriptionActive(subscriptionId: string, isActive: boolean): void {
    const subscription = this.subscriptions.get(subscriptionId);
    if (subscription) {
      subscription.isActive = isActive;
    }
  }

  // Clear all subscriptions
  clear(): void {
    this.subscriptions.clear();
  }
}

// Rate Limiting Utilities
export class RateLimiter {
  private buckets: Map<string, {
    tokens: number;
    lastRefill: number;
    capacity: number;
    refillRate: number;
  }> = new Map();

  // Check if action is allowed
  checkLimit(
    key: string,
    capacity: number,
    refillRate: number, // tokens per second
    tokensRequired = 1
  ): boolean {
    const now = Date.now();
    let bucket = this.buckets.get(key);

    if (!bucket) {
      bucket = {
        tokens: capacity,
        lastRefill: now,
        capacity,
        refillRate
      };
      this.buckets.set(key, bucket);
    }

    // Refill tokens
    const timePassed = (now - bucket.lastRefill) / 1000;
    const tokensToAdd = Math.floor(timePassed * bucket.refillRate);
    
    if (tokensToAdd > 0) {
      bucket.tokens = Math.min(bucket.capacity, bucket.tokens + tokensToAdd);
      bucket.lastRefill = now;
    }

    // Check if enough tokens
    if (bucket.tokens >= tokensRequired) {
      bucket.tokens -= tokensRequired;
      return true;
    }

    return false;
  }

  // Get remaining tokens
  getRemainingTokens(key: string): number {
    const bucket = this.buckets.get(key);
    return bucket ? bucket.tokens : 0;
  }

  // Reset rate limiter
  reset(key?: string): void {
    if (key) {
      this.buckets.delete(key);
    } else {
      this.buckets.clear();
    }
  }
}

// Error Recovery Utilities
export class ErrorRecoveryManager {
  private errorCounts: Map<string, number> = new Map();
  private lastErrors: Map<string, number> = new Map();
  private recoveryStrategies: Map<string, () => Promise<void>> = new Map();

  // Register recovery strategy
  registerStrategy(
    errorType: string,
    strategy: () => Promise<void>
  ): void {
    this.recoveryStrategies.set(errorType, strategy);
  }

  // Handle error with recovery
  async handleError(errorType: string, error: Error): Promise<void> {
    const now = Date.now();
    const count = this.errorCounts.get(errorType) || 0;
    const lastError = this.lastErrors.get(errorType) || 0;

    // Update error tracking
    this.errorCounts.set(errorType, count + 1);
    this.lastErrors.set(errorType, now);

    // Check if we should attempt recovery
    const timeSinceLastError = now - lastError;
    const shouldRecover = count < 5 && timeSinceLastError > 1000; // Max 5 attempts, 1 second apart

    if (shouldRecover) {
      const strategy = this.recoveryStrategies.get(errorType);
      if (strategy) {
        try {
          await strategy();
          console.log(`Recovery successful for error type: ${errorType}`);
          this.errorCounts.set(errorType, 0); // Reset on success
        } catch (recoveryError) {
          console.error(`Recovery failed for error type ${errorType}:`, recoveryError);
        }
      }
    }
  }

  // Reset error counts
  resetErrors(errorType?: string): void {
    if (errorType) {
      this.errorCounts.delete(errorType);
      this.lastErrors.delete(errorType);
    } else {
      this.errorCounts.clear();
      this.lastErrors.clear();
    }
  }

  // Get error statistics
  getErrorStats(): Record<string, { count: number; lastOccurrence: number }> {
    const stats: Record<string, any> = {};
    
    this.errorCounts.forEach((count, errorType) => {
      stats[errorType] = {
        count,
        lastOccurrence: this.lastErrors.get(errorType) || 0
      };
    });

    return stats;
  }
}

// Export utility functions
export const createWebSocketUrl = (
  baseUrl: string, 
  endpoint: string, 
  params: Record<string, string> = {}
): string => {
  const url = new URL(endpoint, baseUrl.replace('http', 'ws'));
  Object.entries(params).forEach(([key, value]) => {
    url.searchParams.set(key, value);
  });
  return url.toString();
};

export const validateWebSocketMessage = (message: any): message is WebSocketMessage => {
  return (
    typeof message === 'object' &&
    message !== null &&
    typeof message.type === 'string' &&
    typeof message.timestamp === 'string'
  );
};

export const calculateConnectionHealth = (
  metrics: WebSocketPerformanceMetrics,
  thresholds: {
    maxLatency?: number;
    maxErrorRate?: number;
    minQualityScore?: number;
  } = {}
): ConnectionHealth => {
  const {
    maxLatency = 1000,
    maxErrorRate = 0.05,
    minQualityScore = 70
  } = thresholds;

  const isHealthy = (
    metrics.messageLatency <= maxLatency &&
    metrics.errorRate <= maxErrorRate &&
    metrics.qualityScore >= minQualityScore
  );

  const stabilityScore = Math.max(0, 100 - (metrics.reconnectionCount * 10));

  return {
    isHealthy,
    qualityScore: metrics.qualityScore,
    stabilityScore,
    latencyStats: {
      current: metrics.messageLatency,
      average: metrics.messageLatency,
      min: 0, // Would need tracking
      max: 0, // Would need tracking
      p95: 0  // Would need tracking
    },
    throughputStats: {
      current: metrics.messagesPerSecond,
      average: metrics.messagesPerSecond,
      peak: 0 // Would need tracking
    },
    errorStats: {
      total: Math.floor(metrics.errorRate * 100),
      rate: metrics.errorRate,
      recentErrors: []
    },
    connectionStats: {
      uptime: metrics.uptime,
      reconnections: metrics.reconnectionCount,
      lastReconnection: undefined
    }
  };
};

export const formatWebSocketMessage = (
  type: MessageType,
  data: any,
  options: {
    priority?: number;
    correlationId?: string;
    version?: string;
  } = {}
): WebSocketMessage => {
  return {
    type,
    timestamp: new Date().toISOString(),
    message_id: `msg_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
    data,
    priority: options.priority || 2,
    correlation_id: options.correlationId,
    version: options.version || '2.0'
  } as WebSocketMessage;
};

// Global instances (use carefully in React contexts)
export const globalConnectionManager = new WebSocketConnectionManager();
export const globalMessageProcessor = new MessageProcessor();
export const globalSubscriptionManager = new SubscriptionManager();
export const globalRateLimiter = new RateLimiter();
export const globalErrorRecoveryManager = new ErrorRecoveryManager();