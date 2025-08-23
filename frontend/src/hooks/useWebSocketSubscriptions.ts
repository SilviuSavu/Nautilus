/**
 * useWebSocketSubscriptions Hook
 * Sprint 3: Advanced WebSocket Subscription Management
 * 
 * Intelligent subscription management with topic-based filtering,
 * rate limiting, priority queuing, and subscription lifecycle management.
 */

import { useState, useEffect, useCallback, useRef } from 'react';
import { useWebSocketManager } from './useWebSocketManager';
import type { 
  MessageType, 
  SubscriptionFilters, 
  SubscriptionConfig, 
  MessagePriority 
} from '../types/websocket';

export interface SubscriptionRequest {
  id: string;
  messageType: MessageType;
  filters?: SubscriptionFilters;
  priority?: MessagePriority;
  rateLimit?: number;
  autoReconnect?: boolean;
  onMessage?: (message: any) => void;
  onError?: (error: Error) => void;
}

export interface ActiveSubscription extends SubscriptionRequest {
  subscriptionId: string;
  isActive: boolean;
  messageCount: number;
  errorCount: number;
  lastActivity: string;
  createdAt: string;
  rateExceeded: boolean;
  health: 'healthy' | 'degraded' | 'error';
}

export interface SubscriptionMetrics {
  totalSubscriptions: number;
  activeSubscriptions: number;
  pausedSubscriptions: number;
  totalMessages: number;
  messagesPerSecond: number;
  errorRate: number;
  averageLatency: number;
  subscriptionHealth: number;
}

export interface BulkSubscriptionOptions {
  batchSize?: number;
  delayBetweenBatches?: number;
  onProgress?: (completed: number, total: number) => void;
  onBatchComplete?: (batch: SubscriptionRequest[], results: string[]) => void;
}

export interface UseWebSocketSubscriptionsReturn {
  // Subscriptions state
  subscriptions: ActiveSubscription[];
  metrics: SubscriptionMetrics;
  
  // Subscription control
  subscribe: (request: SubscriptionRequest) => Promise<string>;
  unsubscribe: (id: string) => Promise<boolean>;
  unsubscribeAll: () => Promise<void>;
  pauseSubscription: (id: string) => void;
  resumeSubscription: (id: string) => void;
  
  // Bulk operations
  subscribeBulk: (requests: SubscriptionRequest[], options?: BulkSubscriptionOptions) => Promise<string[]>;
  unsubscribeBulk: (ids: string[]) => Promise<boolean[]>;
  
  // Subscription management
  updateSubscription: (id: string, updates: Partial<SubscriptionRequest>) => Promise<boolean>;
  resubscribe: (id: string) => Promise<string>;
  getSubscription: (id: string) => ActiveSubscription | null;
  getSubscriptionsByType: (messageType: MessageType) => ActiveSubscription[];
  
  // Health and monitoring
  getHealthySubscriptions: () => ActiveSubscription[];
  getDegradedSubscriptions: () => ActiveSubscription[];
  getErrorSubscriptions: () => ActiveSubscription[];
  checkSubscriptionHealth: (id: string) => 'healthy' | 'degraded' | 'error';
  
  // Rate limiting
  setGlobalRateLimit: (messagesPerSecond: number) => void;
  setSubscriptionRateLimit: (id: string, messagesPerSecond: number) => void;
  resetRateLimits: () => void;
  
  // Utilities
  exportSubscriptions: () => SubscriptionRequest[];
  importSubscriptions: (subscriptions: SubscriptionRequest[]) => Promise<string[]>;
  clearMetrics: () => void;
}

export function useWebSocketSubscriptions(): UseWebSocketSubscriptionsReturn {
  // State
  const [subscriptions, setSubscriptions] = useState<ActiveSubscription[]>([]);
  const [metrics, setMetrics] = useState<SubscriptionMetrics>({
    totalSubscriptions: 0,
    activeSubscriptions: 0,
    pausedSubscriptions: 0,
    totalMessages: 0,
    messagesPerSecond: 0,
    errorRate: 0,
    averageLatency: 0,
    subscriptionHealth: 100
  });
  const [globalRateLimit, setGlobalRateLimit] = useState<number | null>(null);
  
  // Refs
  const subscriptionHandlers = useRef<Map<string, (message: any) => void>>(new Map());
  const rateLimitTrackers = useRef<Map<string, { count: number; window: number }>>(new Map());
  const messageStatsRef = useRef<{ timestamp: number; count: number }[]>([]);
  const metricsIntervalRef = useRef<NodeJS.Timeout>();
  const isMountedRef = useRef(true);
  
  // WebSocket manager
  const {
    connectionState,
    subscribe: wsSubscribe,
    unsubscribe: wsUnsubscribe,
    addMessageHandler,
    sendMessage,
    isConnected: wsConnected
  } = useWebSocketManager();
  
  // Cleanup on unmount
  useEffect(() => {
    return () => {
      isMountedRef.current = false;
      if (metricsIntervalRef.current) {
        clearInterval(metricsIntervalRef.current);
      }
    };
  }, []);
  
  // Generate subscription ID
  const generateSubscriptionId = useCallback(() => {
    return `sub_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
  }, []);
  
  // Check rate limit
  const checkRateLimit = useCallback((subscriptionId: string, limit: number): boolean => {
    const now = Date.now();
    const windowStart = now - 1000; // 1 second window
    
    const tracker = rateLimitTrackers.current.get(subscriptionId) || { count: 0, window: now };
    
    // Reset window if needed
    if (tracker.window < windowStart) {
      tracker.count = 0;
      tracker.window = now;
    }
    
    // Check limit
    if (tracker.count >= limit) {
      return false;
    }
    
    // Update tracker
    tracker.count++;
    rateLimitTrackers.current.set(subscriptionId, tracker);
    
    return true;
  }, []);
  
  // Update subscription metrics
  const updateSubscriptionMetrics = useCallback((subscriptionId: string, increment: 'message' | 'error') => {
    setSubscriptions(prev => prev.map(sub => {
      if (sub.subscriptionId === subscriptionId) {
        const updated = { ...sub };
        if (increment === 'message') {
          updated.messageCount++;
          updated.lastActivity = new Date().toISOString();
        } else {
          updated.errorCount++;
        }
        
        // Update health status
        const totalActivity = updated.messageCount + updated.errorCount;
        const errorRate = totalActivity > 0 ? updated.errorCount / totalActivity : 0;
        
        if (errorRate > 0.1) updated.health = 'error';
        else if (errorRate > 0.05 || updated.rateExceeded) updated.health = 'degraded';
        else updated.health = 'healthy';
        
        return updated;
      }
      return sub;
    }));
  }, []);
  
  // Process incoming message
  const processMessage = useCallback((message: any, subscription: ActiveSubscription) => {
    // Check rate limit
    if (subscription.rateLimit) {
      if (!checkRateLimit(subscription.subscriptionId, subscription.rateLimit)) {
        setSubscriptions(prev => prev.map(sub => 
          sub.subscriptionId === subscription.subscriptionId 
            ? { ...sub, rateExceeded: true, health: 'degraded' }
            : sub
        ));
        return;
      }
    }
    
    // Check global rate limit
    if (globalRateLimit && !checkRateLimit('global', globalRateLimit)) {
      return;
    }
    
    // Update message statistics
    messageStatsRef.current.push({ timestamp: Date.now(), count: 1 });
    if (messageStatsRef.current.length > 100) {
      messageStatsRef.current.shift();
    }
    
    // Update subscription metrics
    updateSubscriptionMetrics(subscription.subscriptionId, 'message');
    
    // Call subscription handler
    try {
      if (subscription.onMessage) {
        subscription.onMessage(message);
      }
    } catch (error) {
      updateSubscriptionMetrics(subscription.subscriptionId, 'error');
      if (subscription.onError) {
        subscription.onError(error instanceof Error ? error : new Error('Unknown error'));
      }
    }
  }, [checkRateLimit, globalRateLimit, updateSubscriptionMetrics]);
  
  // Subscribe to message type
  const subscribe = useCallback(async (request: SubscriptionRequest): Promise<string> => {
    if (!wsConnected) {
      throw new Error('WebSocket not connected');
    }
    
    try {
      // Create WebSocket subscription
      const subscriptionId = await wsSubscribe(request.messageType, request.filters);
      
      // Create active subscription
      const activeSubscription: ActiveSubscription = {
        ...request,
        subscriptionId,
        isActive: true,
        messageCount: 0,
        errorCount: 0,
        lastActivity: new Date().toISOString(),
        createdAt: new Date().toISOString(),
        rateExceeded: false,
        health: 'healthy'
      };
      
      // Add to subscriptions
      setSubscriptions(prev => [...prev, activeSubscription]);
      
      // Add message handler
      const handlerId = `${request.id}_handler`;
      const cleanup = addMessageHandler(
        handlerId,
        (message) => processMessage(message, activeSubscription),
        (message) => message.type === request.messageType
      );
      
      subscriptionHandlers.current.set(request.id, cleanup);
      
      return request.id;
      
    } catch (error) {
      throw new Error(`Failed to subscribe: ${error instanceof Error ? error.message : error}`);
    }
  }, [wsConnected, wsSubscribe, addMessageHandler, processMessage]);
  
  // Unsubscribe from message type
  const unsubscribe = useCallback(async (id: string): Promise<boolean> => {
    const subscription = subscriptions.find(sub => sub.id === id);
    if (!subscription) {
      return false;
    }
    
    try {
      // Remove WebSocket subscription
      await wsUnsubscribe(subscription.subscriptionId);
      
      // Remove message handler
      const cleanup = subscriptionHandlers.current.get(id);
      if (cleanup) {
        cleanup();
        subscriptionHandlers.current.delete(id);
      }
      
      // Remove from subscriptions
      setSubscriptions(prev => prev.filter(sub => sub.id !== id));
      
      return true;
    } catch (error) {
      return false;
    }
  }, [subscriptions, wsUnsubscribe]);
  
  // Unsubscribe from all
  const unsubscribeAll = useCallback(async () => {
    const unsubscribePromises = subscriptions.map(sub => unsubscribe(sub.id));
    await Promise.all(unsubscribePromises);
  }, [subscriptions, unsubscribe]);
  
  // Pause subscription
  const pauseSubscription = useCallback((id: string) => {
    setSubscriptions(prev => prev.map(sub =>
      sub.id === id ? { ...sub, isActive: false } : sub
    ));
  }, []);
  
  // Resume subscription
  const resumeSubscription = useCallback((id: string) => {
    setSubscriptions(prev => prev.map(sub =>
      sub.id === id ? { ...sub, isActive: true, rateExceeded: false } : sub
    ));
  }, []);
  
  // Bulk subscribe
  const subscribeBulk = useCallback(async (
    requests: SubscriptionRequest[],
    options: BulkSubscriptionOptions = {}
  ): Promise<string[]> => {
    const { batchSize = 10, delayBetweenBatches = 100, onProgress, onBatchComplete } = options;
    
    const results: string[] = [];
    const batches = [];
    
    // Create batches
    for (let i = 0; i < requests.length; i += batchSize) {
      batches.push(requests.slice(i, i + batchSize));
    }
    
    // Process batches
    for (let i = 0; i < batches.length; i++) {
      const batch = batches[i];
      const batchResults: string[] = [];
      
      // Process batch in parallel
      const subscribePromises = batch.map(async (request) => {
        try {
          const id = await subscribe(request);
          batchResults.push(id);
          return id;
        } catch (error) {
          batchResults.push('');
          return '';
        }
      });
      
      await Promise.all(subscribePromises);
      results.push(...batchResults);
      
      // Call progress callback
      if (onProgress) {
        onProgress((i + 1) * batchSize, requests.length);
      }
      
      // Call batch complete callback
      if (onBatchComplete) {
        onBatchComplete(batch, batchResults);
      }
      
      // Delay between batches (except last)
      if (i < batches.length - 1) {
        await new Promise(resolve => setTimeout(resolve, delayBetweenBatches));
      }
    }
    
    return results;
  }, [subscribe]);
  
  // Bulk unsubscribe
  const unsubscribeBulk = useCallback(async (ids: string[]): Promise<boolean[]> => {
    const promises = ids.map(id => unsubscribe(id));
    return Promise.all(promises);
  }, [unsubscribe]);
  
  // Update subscription
  const updateSubscription = useCallback(async (
    id: string, 
    updates: Partial<SubscriptionRequest>
  ): Promise<boolean> => {
    const subscription = subscriptions.find(sub => sub.id === id);
    if (!subscription) {
      return false;
    }
    
    // If message type or filters changed, need to resubscribe
    if (updates.messageType || updates.filters) {
      await unsubscribe(id);
      const newRequest: SubscriptionRequest = { ...subscription, ...updates };
      await subscribe(newRequest);
    } else {
      // Just update local state
      setSubscriptions(prev => prev.map(sub =>
        sub.id === id ? { ...sub, ...updates } : sub
      ));
    }
    
    return true;
  }, [subscriptions, unsubscribe, subscribe]);
  
  // Resubscribe
  const resubscribe = useCallback(async (id: string): Promise<string> => {
    const subscription = subscriptions.find(sub => sub.id === id);
    if (!subscription) {
      throw new Error(`Subscription ${id} not found`);
    }
    
    await unsubscribe(id);
    return subscribe(subscription);
  }, [subscriptions, unsubscribe, subscribe]);
  
  // Get subscription
  const getSubscription = useCallback((id: string): ActiveSubscription | null => {
    return subscriptions.find(sub => sub.id === id) || null;
  }, [subscriptions]);
  
  // Get subscriptions by type
  const getSubscriptionsByType = useCallback((messageType: MessageType): ActiveSubscription[] => {
    return subscriptions.filter(sub => sub.messageType === messageType);
  }, [subscriptions]);
  
  // Get healthy subscriptions
  const getHealthySubscriptions = useCallback((): ActiveSubscription[] => {
    return subscriptions.filter(sub => sub.health === 'healthy');
  }, [subscriptions]);
  
  // Get degraded subscriptions
  const getDegradedSubscriptions = useCallback((): ActiveSubscription[] => {
    return subscriptions.filter(sub => sub.health === 'degraded');
  }, [subscriptions]);
  
  // Get error subscriptions
  const getErrorSubscriptions = useCallback((): ActiveSubscription[] => {
    return subscriptions.filter(sub => sub.health === 'error');
  }, [subscriptions]);
  
  // Check subscription health
  const checkSubscriptionHealth = useCallback((id: string): 'healthy' | 'degraded' | 'error' => {
    const subscription = getSubscription(id);
    return subscription?.health || 'error';
  }, [getSubscription]);
  
  // Set subscription rate limit
  const setSubscriptionRateLimit = useCallback((id: string, messagesPerSecond: number) => {
    setSubscriptions(prev => prev.map(sub =>
      sub.id === id ? { ...sub, rateLimit: messagesPerSecond } : sub
    ));
  }, []);
  
  // Reset rate limits
  const resetRateLimits = useCallback(() => {
    rateLimitTrackers.current.clear();
    setSubscriptions(prev => prev.map(sub => ({ ...sub, rateExceeded: false })));
  }, []);
  
  // Export subscriptions
  const exportSubscriptions = useCallback((): SubscriptionRequest[] => {
    return subscriptions.map(sub => ({
      id: sub.id,
      messageType: sub.messageType,
      filters: sub.filters,
      priority: sub.priority,
      rateLimit: sub.rateLimit,
      autoReconnect: sub.autoReconnect,
      onMessage: sub.onMessage,
      onError: sub.onError
    }));
  }, [subscriptions]);
  
  // Import subscriptions
  const importSubscriptions = useCallback(async (importedSubscriptions: SubscriptionRequest[]): Promise<string[]> => {
    return subscribeBulk(importedSubscriptions);
  }, [subscribeBulk]);
  
  // Clear metrics
  const clearMetrics = useCallback(() => {
    setSubscriptions(prev => prev.map(sub => ({
      ...sub,
      messageCount: 0,
      errorCount: 0,
      health: 'healthy',
      rateExceeded: false
    })));
    messageStatsRef.current = [];
  }, []);
  
  // Update metrics periodically
  useEffect(() => {
    metricsIntervalRef.current = setInterval(() => {
      if (!isMountedRef.current) return;
      
      const now = Date.now();
      const recentMessages = messageStatsRef.current.filter(stat => now - stat.timestamp < 60000);
      const messagesPerSecond = recentMessages.length / 60;
      
      const totalSubscriptions = subscriptions.length;
      const activeSubscriptions = subscriptions.filter(sub => sub.isActive).length;
      const pausedSubscriptions = totalSubscriptions - activeSubscriptions;
      const totalMessages = subscriptions.reduce((sum, sub) => sum + sub.messageCount, 0);
      const totalErrors = subscriptions.reduce((sum, sub) => sum + sub.errorCount, 0);
      const errorRate = totalMessages > 0 ? totalErrors / totalMessages : 0;
      
      const healthyCount = subscriptions.filter(sub => sub.health === 'healthy').length;
      const subscriptionHealth = totalSubscriptions > 0 ? (healthyCount / totalSubscriptions) * 100 : 100;
      
      setMetrics({
        totalSubscriptions,
        activeSubscriptions,
        pausedSubscriptions,
        totalMessages,
        messagesPerSecond,
        errorRate,
        averageLatency: 0, // Would need to track this from message processing
        subscriptionHealth
      });
    }, 5000);
    
    return () => {
      if (metricsIntervalRef.current) {
        clearInterval(metricsIntervalRef.current);
      }
    };
  }, [subscriptions]);
  
  // Auto-resubscribe on reconnection
  useEffect(() => {
    if (connectionState === 'connected') {
      const autoReconnectSubscriptions = subscriptions.filter(sub => sub.autoReconnect && !sub.isActive);
      
      autoReconnectSubscriptions.forEach(async (sub) => {
        try {
          await resubscribe(sub.id);
        } catch (error) {
          console.error(`Failed to auto-resubscribe ${sub.id}:`, error);
        }
      });
    }
  }, [connectionState, subscriptions, resubscribe]);
  
  return {
    // Subscriptions state
    subscriptions,
    metrics,
    
    // Subscription control
    subscribe,
    unsubscribe,
    unsubscribeAll,
    pauseSubscription,
    resumeSubscription,
    
    // Bulk operations
    subscribeBulk,
    unsubscribeBulk,
    
    // Subscription management
    updateSubscription,
    resubscribe,
    getSubscription,
    getSubscriptionsByType,
    
    // Health and monitoring
    getHealthySubscriptions,
    getDegradedSubscriptions,
    getErrorSubscriptions,
    checkSubscriptionHealth,
    
    // Rate limiting
    setGlobalRateLimit,
    setSubscriptionRateLimit,
    resetRateLimits,
    
    // Utilities
    exportSubscriptions,
    importSubscriptions,
    clearMetrics
  };
}

export default useWebSocketSubscriptions;