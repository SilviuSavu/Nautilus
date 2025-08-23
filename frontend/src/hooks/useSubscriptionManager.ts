/**
 * useSubscriptionManager Hook
 * Sprint 3: Advanced WebSocket Infrastructure
 * 
 * Advanced subscription management with Redis integration, filtering, rate limiting,
 * and comprehensive subscription analytics and monitoring.
 */

import { useState, useEffect, useCallback, useRef } from 'react';
import { useWebSocketManager } from './useWebSocketManager';

interface Subscription {
  id: string;
  type: string;
  filters: Record<string, any>;
  isActive: boolean;
  messageCount: number;
  errorCount: number;
  createdAt: string;
  lastActivity: string;
  rateLimit?: number;
  queueSize?: number;
  priority?: number;
}

interface SubscriptionStats {
  activeCount: number;
  totalMessages: number;
  averageLatency: number;
  errorRate: number;
  messagesByType: Record<string, number>;
  subscriptionHealth: Record<string, number>;
}

interface SubscriptionAnalytics {
  subscriptionId: string;
  messageRate: number;
  errorRate: number;
  latencyStats: {
    min: number;
    max: number;
    average: number;
    p95: number;
  };
  healthScore: number;
  lastUpdate: string;
}

interface RateLimitConfig {
  messagesPerSecond: number;
  burstSize: number;
  windowSize: number;
}

export const useSubscriptionManager = () => {
  const {
    connectionState,
    sendMessage,
    addMessageHandler,
    subscribe: wsSubscribe,
    unsubscribe: wsUnsubscribe,
    getConnectionInfo
  } = useWebSocketManager();

  // State management
  const [subscriptions, setSubscriptions] = useState<Subscription[]>([]);
  const [subscriptionStats, setSubscriptionStats] = useState<SubscriptionStats>({
    activeCount: 0,
    totalMessages: 0,
    averageLatency: 0,
    errorRate: 0,
    messagesByType: {},
    subscriptionHealth: {}
  });

  // Refs for tracking
  const subscriptionHandlersRef = useRef<Map<string, string>>(new Map());
  const messageCountersRef = useRef<Map<string, { count: number; errors: number; lastSeen: number }>>(new Map());
  const rateLimitersRef = useRef<Map<string, { tokens: number; lastRefill: number; config: RateLimitConfig }>>(new Map());
  const analyticsRef = useRef<Map<string, SubscriptionAnalytics>>(new Map());

  // Generate subscription ID
  const generateSubscriptionId = useCallback(() => {
    return `sub_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
  }, []);

  // Rate limiter implementation
  const checkRateLimit = useCallback((subscriptionId: string, rateLimitConfig?: RateLimitConfig): boolean => {
    if (!rateLimitConfig) return true;

    const now = Date.now();
    const limiter = rateLimitersRef.current.get(subscriptionId) || {
      tokens: rateLimitConfig.burstSize,
      lastRefill: now,
      config: rateLimitConfig
    };

    // Refill tokens based on time elapsed
    const timeDelta = (now - limiter.lastRefill) / 1000;
    const tokensToAdd = timeDelta * rateLimitConfig.messagesPerSecond;
    limiter.tokens = Math.min(rateLimitConfig.burstSize, limiter.tokens + tokensToAdd);
    limiter.lastRefill = now;

    if (limiter.tokens >= 1) {
      limiter.tokens -= 1;
      rateLimitersRef.current.set(subscriptionId, limiter);
      return true;
    }

    return false;
  }, []);

  // Update subscription statistics
  const updateSubscriptionStats = useCallback(() => {
    const activeSubscriptions = subscriptions.filter(sub => sub.isActive);
    const totalMessages = subscriptions.reduce((sum, sub) => sum + sub.messageCount, 0);
    const totalErrors = subscriptions.reduce((sum, sub) => sum + sub.errorCount, 0);
    
    const messagesByType = subscriptions.reduce((acc, sub) => {
      acc[sub.type] = (acc[sub.type] || 0) + sub.messageCount;
      return acc;
    }, {} as Record<string, number>);

    const subscriptionHealth = subscriptions.reduce((acc, sub) => {
      const healthScore = sub.messageCount > 0 ? 
        (1 - (sub.errorCount / sub.messageCount)) * 100 : 100;
      acc[sub.id] = healthScore;
      return acc;
    }, {} as Record<string, number>);

    setSubscriptionStats({
      activeCount: activeSubscriptions.length,
      totalMessages,
      averageLatency: 0, // Will be calculated from analytics
      errorRate: totalMessages > 0 ? (totalErrors / totalMessages) * 100 : 0,
      messagesByType,
      subscriptionHealth
    });
  }, [subscriptions]);

  // Create message handler for subscription
  const createSubscriptionHandler = useCallback((subscription: Subscription) => {
    return (message: any) => {
      // Check rate limit
      const rateLimitConfig = subscription.rateLimit ? {
        messagesPerSecond: subscription.rateLimit,
        burstSize: subscription.rateLimit * 2,
        windowSize: 1000
      } : undefined;

      if (!checkRateLimit(subscription.id, rateLimitConfig)) {
        console.warn(`Rate limit exceeded for subscription ${subscription.id}`);
        return;
      }

      // Apply filters
      if (!applyMessageFilters(message, subscription.filters)) {
        return;
      }

      // Update counters
      const counter = messageCountersRef.current.get(subscription.id) || {
        count: 0,
        errors: 0,
        lastSeen: 0
      };

      counter.count += 1;
      counter.lastSeen = Date.now();

      if (message.error) {
        counter.errors += 1;
      }

      messageCountersRef.current.set(subscription.id, counter);

      // Update subscription in state
      setSubscriptions(prev => prev.map(sub => 
        sub.id === subscription.id 
          ? {
              ...sub,
              messageCount: counter.count,
              errorCount: counter.errors,
              lastActivity: new Date().toISOString()
            }
          : sub
      ));

      // Update analytics
      updateSubscriptionAnalytics(subscription.id, message);
    };
  }, [checkRateLimit]);

  // Apply message filters
  const applyMessageFilters = useCallback((message: any, filters: Record<string, any>): boolean => {
    // Symbol filter
    if (filters.symbols && filters.symbols.length > 0) {
      const messageSymbol = message.data?.symbol || message.symbol;
      if (messageSymbol && !filters.symbols.includes(messageSymbol)) {
        return false;
      }
    }

    // Portfolio filter
    if (filters.portfolio_ids && filters.portfolio_ids.length > 0) {
      const portfolioId = message.data?.portfolio_id || message.portfolio_id;
      if (portfolioId && !filters.portfolio_ids.includes(portfolioId)) {
        return false;
      }
    }

    // Strategy filter
    if (filters.strategy_ids && filters.strategy_ids.length > 0) {
      const strategyId = message.data?.strategy_id || message.strategy_id;
      if (strategyId && !filters.strategy_ids.includes(strategyId)) {
        return false;
      }
    }

    // User filter
    if (filters.user_id) {
      const userId = message.data?.user_id || message.user_id;
      if (userId && userId !== filters.user_id) {
        return false;
      }
    }

    // Severity filter (for alerts)
    if (filters.severity) {
      const severity = message.data?.severity || message.severity;
      if (severity && severity !== filters.severity) {
        return false;
      }
    }

    // Price range filter
    if (filters.min_price || filters.max_price) {
      const price = message.data?.price || message.price;
      if (price !== undefined) {
        if (filters.min_price && price < filters.min_price) {
          return false;
        }
        if (filters.max_price && price > filters.max_price) {
          return false;
        }
      }
    }

    return true;
  }, []);

  // Update subscription analytics
  const updateSubscriptionAnalytics = useCallback((subscriptionId: string, message: any) => {
    const now = Date.now();
    const analytics = analyticsRef.current.get(subscriptionId) || {
      subscriptionId,
      messageRate: 0,
      errorRate: 0,
      latencyStats: { min: Infinity, max: 0, average: 0, p95: 0 },
      healthScore: 100,
      lastUpdate: new Date().toISOString()
    };

    // Update message rate (messages per second)
    const timeSinceLastUpdate = (now - new Date(analytics.lastUpdate).getTime()) / 1000;
    analytics.messageRate = timeSinceLastUpdate > 0 ? 1 / timeSinceLastUpdate : 0;

    // Update latency stats
    if (message.latency !== undefined) {
      analytics.latencyStats.min = Math.min(analytics.latencyStats.min, message.latency);
      analytics.latencyStats.max = Math.max(analytics.latencyStats.max, message.latency);
      // Simple moving average for now
      analytics.latencyStats.average = (analytics.latencyStats.average + message.latency) / 2;
    }

    // Update error rate
    const counter = messageCountersRef.current.get(subscriptionId);
    if (counter && counter.count > 0) {
      analytics.errorRate = (counter.errors / counter.count) * 100;
      analytics.healthScore = Math.max(0, 100 - analytics.errorRate);
    }

    analytics.lastUpdate = new Date().toISOString();
    analyticsRef.current.set(subscriptionId, analytics);
  }, []);

  // Subscribe to a message type
  const subscribe = useCallback(async (
    type: string,
    filters: Record<string, any> = {},
    options: { rateLimit?: number; queueSize?: number; priority?: number } = {}
  ): Promise<string> => {
    if (connectionState !== 'connected') {
      throw new Error('WebSocket not connected');
    }

    const subscriptionId = generateSubscriptionId();
    
    const newSubscription: Subscription = {
      id: subscriptionId,
      type,
      filters,
      isActive: true,
      messageCount: 0,
      errorCount: 0,
      createdAt: new Date().toISOString(),
      lastActivity: new Date().toISOString(),
      rateLimit: options.rateLimit,
      queueSize: options.queueSize || 100,
      priority: options.priority || 2
    };

    // Add to subscriptions
    setSubscriptions(prev => [...prev, newSubscription]);

    // Create message handler
    const handler = createSubscriptionHandler(newSubscription);
    const handlerId = addMessageHandler(
      subscriptionId,
      handler,
      (message) => message.type === type
    );

    subscriptionHandlersRef.current.set(subscriptionId, handlerId);

    // Subscribe via WebSocket
    try {
      await wsSubscribe(type, filters);
      console.log(`Subscribed to ${type} with ID: ${subscriptionId}`);
      return subscriptionId;
    } catch (error) {
      // Remove subscription if WebSocket subscription failed
      setSubscriptions(prev => prev.filter(sub => sub.id !== subscriptionId));
      throw error;
    }
  }, [connectionState, generateSubscriptionId, createSubscriptionHandler, addMessageHandler, wsSubscribe]);

  // Unsubscribe from a subscription
  const unsubscribe = useCallback(async (subscriptionId: string): Promise<boolean> => {
    const subscription = subscriptions.find(sub => sub.id === subscriptionId);
    if (!subscription) {
      return false;
    }

    try {
      // Unsubscribe via WebSocket
      await wsUnsubscribe(subscriptionId);

      // Remove message handler
      const handlerId = subscriptionHandlersRef.current.get(subscriptionId);
      if (handlerId) {
        // Handler cleanup is handled by useWebSocketManager
        subscriptionHandlersRef.current.delete(subscriptionId);
      }

      // Remove from state
      setSubscriptions(prev => prev.filter(sub => sub.id !== subscriptionId));

      // Cleanup counters and analytics
      messageCountersRef.current.delete(subscriptionId);
      rateLimitersRef.current.delete(subscriptionId);
      analyticsRef.current.delete(subscriptionId);

      console.log(`Unsubscribed from ${subscriptionId}`);
      return true;

    } catch (error) {
      console.error(`Failed to unsubscribe from ${subscriptionId}:`, error);
      return false;
    }
  }, [subscriptions, wsUnsubscribe]);

  // Update subscription filters or settings
  const updateSubscription = useCallback(async (
    subscriptionId: string,
    updates: Partial<Pick<Subscription, 'filters' | 'rateLimit' | 'queueSize' | 'priority'>>
  ): Promise<boolean> => {
    const subscription = subscriptions.find(sub => sub.id === subscriptionId);
    if (!subscription) {
      return false;
    }

    try {
      // Update subscription locally
      setSubscriptions(prev => prev.map(sub =>
        sub.id === subscriptionId
          ? { ...sub, ...updates }
          : sub
      ));

      // Send update to backend if filters changed
      if (updates.filters) {
        await sendMessage({
          type: 'update_subscription',
          subscription_id: subscriptionId,
          filters: updates.filters
        });
      }

      // Update rate limiter if changed
      if (updates.rateLimit !== undefined) {
        if (updates.rateLimit > 0) {
          rateLimitersRef.current.set(subscriptionId, {
            tokens: updates.rateLimit * 2, // burst size
            lastRefill: Date.now(),
            config: {
              messagesPerSecond: updates.rateLimit,
              burstSize: updates.rateLimit * 2,
              windowSize: 1000
            }
          });
        } else {
          rateLimitersRef.current.delete(subscriptionId);
        }
      }

      return true;
    } catch (error) {
      console.error(`Failed to update subscription ${subscriptionId}:`, error);
      return false;
    }
  }, [subscriptions, sendMessage]);

  // Pause subscription
  const pauseSubscription = useCallback(async (subscriptionId: string): Promise<boolean> => {
    return updateSubscription(subscriptionId, { filters: { ...subscriptions.find(s => s.id === subscriptionId)?.filters, _paused: true } })
      .then(() => {
        setSubscriptions(prev => prev.map(sub =>
          sub.id === subscriptionId
            ? { ...sub, isActive: false }
            : sub
        ));
        return true;
      });
  }, [updateSubscription, subscriptions]);

  // Resume subscription
  const resumeSubscription = useCallback(async (subscriptionId: string): Promise<boolean> => {
    const subscription = subscriptions.find(sub => sub.id === subscriptionId);
    if (!subscription) return false;

    const { _paused, ...cleanFilters } = subscription.filters;
    
    return updateSubscription(subscriptionId, { filters: cleanFilters })
      .then(() => {
        setSubscriptions(prev => prev.map(sub =>
          sub.id === subscriptionId
            ? { ...sub, isActive: true }
            : sub
        ));
        return true;
      });
  }, [updateSubscription, subscriptions]);

  // Get subscription analytics
  const getSubscriptionAnalytics = useCallback((subscriptionId?: string): SubscriptionAnalytics | SubscriptionAnalytics[] => {
    if (subscriptionId) {
      return analyticsRef.current.get(subscriptionId) || {
        subscriptionId,
        messageRate: 0,
        errorRate: 0,
        latencyStats: { min: 0, max: 0, average: 0, p95: 0 },
        healthScore: 100,
        lastUpdate: new Date().toISOString()
      };
    }

    return Array.from(analyticsRef.current.values());
  }, []);

  // Clear subscription history
  const clearSubscriptionHistory = useCallback((subscriptionId?: string) => {
    if (subscriptionId) {
      messageCountersRef.current.delete(subscriptionId);
      analyticsRef.current.delete(subscriptionId);
      
      setSubscriptions(prev => prev.map(sub =>
        sub.id === subscriptionId
          ? { ...sub, messageCount: 0, errorCount: 0 }
          : sub
      ));
    } else {
      messageCountersRef.current.clear();
      analyticsRef.current.clear();
      
      setSubscriptions(prev => prev.map(sub => ({
        ...sub,
        messageCount: 0,
        errorCount: 0
      })));
    }
  }, []);

  // Update stats periodically
  useEffect(() => {
    const interval = setInterval(updateSubscriptionStats, 5000);
    return () => clearInterval(interval);
  }, [updateSubscriptionStats]);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      // Cleanup all subscriptions
      subscriptionHandlersRef.current.clear();
      messageCountersRef.current.clear();
      rateLimitersRef.current.clear();
      analyticsRef.current.clear();
    };
  }, []);

  return {
    // Subscription data
    subscriptions,
    subscriptionStats,
    
    // Subscription management
    subscribe,
    unsubscribe,
    updateSubscription,
    pauseSubscription,
    resumeSubscription,
    
    // Analytics and monitoring
    getSubscriptionAnalytics,
    clearSubscriptionHistory,
    
    // Connection status
    isConnected: connectionState === 'connected'
  };
};

export default useSubscriptionManager;