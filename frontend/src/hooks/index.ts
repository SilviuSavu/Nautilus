/**
 * Hooks Index
 * Sprint 3: Advanced WebSocket Infrastructure
 * 
 * Exports all WebSocket hooks and enhanced real-time data hooks
 */

// Core WebSocket Infrastructure Hooks
export { useWebSocketManager } from './useWebSocketManager';
export { useSubscriptionManager } from './useSubscriptionManager';
export { useRealTimeData } from './useRealTimeData';
export { useConnectionHealth } from './useConnectionHealth';

// Enhanced Market Data Hooks
export { useMarketData } from './useMarketData';
export { useTradeUpdatesEnhanced } from './useTradeUpdatesEnhanced';

// Legacy/Existing Hooks (maintained for backward compatibility)
export { useEngineWebSocket } from './useEngineWebSocket';
export { useTradeUpdates } from './useTradeUpdates';
export { useRiskAlerts } from './useRiskAlerts';

// Re-export as default exports for convenience
export { default as useWebSocketManager } from './useWebSocketManager';
export { default as useSubscriptionManager } from './useSubscriptionManager';
export { default as useRealTimeData } from './useRealTimeData';
export { default as useConnectionHealth } from './useConnectionHealth';
export { default as useMarketData } from './useMarketData';
export { default as useTradeUpdatesEnhanced } from './useTradeUpdatesEnhanced';
export { default as useEngineWebSocket } from './useEngineWebSocket';
export { default as useTradeUpdates } from './useTradeUpdates';
export { default as useRiskAlerts } from './useRiskAlerts';