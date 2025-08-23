/**
 * TypeScript Types for Sprint 3 WebSocket Protocols
 * 
 * Comprehensive type definitions for WebSocket messages, protocols,
 * and data structures used in the advanced WebSocket infrastructure.
 */

// Base Protocol Types
export type MessageType = 
  | 'connection_established'
  | 'connection_closed'
  | 'heartbeat'
  | 'heartbeat_response'
  | 'subscribe'
  | 'unsubscribe'
  | 'subscription_confirmed'
  | 'subscription_error'
  | 'engine_status'
  | 'market_data'
  | 'trade_updates'
  | 'system_health'
  | 'performance_update'
  | 'risk_alert'
  | 'order_update'
  | 'position_update'
  | 'strategy_performance'
  | 'execution_analytics'
  | 'risk_metrics'
  | 'event'
  | 'alert'
  | 'notification'
  | 'breach_alert'
  | 'system_alert'
  | 'ack'
  | 'error'
  | 'success'
  | 'command'
  | 'command_response';

export type ConnectionState = 'disconnected' | 'connecting' | 'connected' | 'reconnecting' | 'error';

export type MessagePriority = 1 | 2 | 3 | 4; // Low, Normal, High, Critical

export type AlertSeverity = 'low' | 'medium' | 'high' | 'critical';

export type OrderStatus = 'pending' | 'open' | 'filled' | 'partial' | 'cancelled' | 'rejected';

export type OrderSide = 'buy' | 'sell';

export type PositionSide = 'long' | 'short' | 'flat';

// Base Message Structure
export interface BaseWebSocketMessage {
  type: MessageType;
  version?: string;
  timestamp: string;
  message_id?: string;
  correlation_id?: string;
  priority?: MessagePriority;
}

// Connection Management Messages
export interface ConnectionMessage extends BaseWebSocketMessage {
  type: 'connection_established' | 'connection_closed';
  connection_id: string;
  user_id?: string;
  session_info?: Record<string, any>;
}

export interface HeartbeatMessage extends BaseWebSocketMessage {
  type: 'heartbeat' | 'heartbeat_response';
  client_timestamp?: string;
  server_timestamp?: string;
}

// Subscription Management Messages
export interface SubscriptionMessage extends BaseWebSocketMessage {
  type: 'subscribe' | 'unsubscribe' | 'subscription_confirmed';
  subscription_id: string;
  topic: string;
  parameters?: Record<string, any>;
  filters?: SubscriptionFilters;
}

export interface SubscriptionFilters {
  symbols?: string[];
  portfolio_ids?: string[];
  strategy_ids?: string[];
  user_id?: string;
  severity?: AlertSeverity;
  min_price?: number;
  max_price?: number;
  venue?: string;
  asset_class?: string;
}

export interface SubscriptionErrorMessage extends BaseWebSocketMessage {
  type: 'subscription_error';
  subscription_id: string;
  error_code: string;
  error_message: string;
  details?: Record<string, any>;
}

// Data Messages
export interface DataMessage extends BaseWebSocketMessage {
  data: Record<string, any>;
  source?: string;
  sequence?: number;
}

// Market Data Messages
export interface MarketDataMessage extends DataMessage {
  type: 'market_data';
  symbol: string;
  venue?: string;
  data_type?: 'tick' | 'quote' | 'bar' | 'trade';
  data: {
    price?: number;
    bid_price?: number;
    ask_price?: number;
    bid_size?: number;
    ask_size?: number;
    volume?: number;
    open?: number;
    high?: number;
    low?: number;
    close?: number;
    change?: number;
    change_percent?: number;
    last_trade_price?: number;
    last_trade_size?: number;
    last_trade_time?: string;
    timestamp: string;
  };
}

// Trade Update Messages
export interface TradeUpdateMessage extends DataMessage {
  type: 'trade_updates';
  user_id: string;
  trade_type?: 'execution' | 'fill' | 'partial_fill';
  data: {
    trade_id: string;
    order_id?: string;
    symbol: string;
    side: OrderSide;
    quantity: number;
    price: number;
    status: OrderStatus;
    filled_quantity?: number;
    remaining_quantity?: number;
    avg_fill_price?: number;
    commission?: number;
    fees?: number;
    execution_id?: string;
    venue?: string;
    execution_time?: string;
    slippage_bps?: number;
    timestamp: string;
  };
}

// Order Update Messages
export interface OrderUpdateMessage extends DataMessage {
  type: 'order_update';
  order_id: string;
  user_id: string;
  symbol: string;
  order_status: OrderStatus;
  data: {
    order_type?: string;
    side: OrderSide;
    quantity: number;
    price?: number;
    filled_quantity: number;
    remaining_quantity: number;
    avg_fill_price?: number;
    time_in_force?: string;
    order_time?: string;
    last_update_time?: string;
    cancel_reason?: string;
    reject_reason?: string;
    timestamp: string;
  };
}

// Position Update Messages
export interface PositionUpdateMessage extends DataMessage {
  type: 'position_update';
  portfolio_id: string;
  symbol: string;
  position_side: PositionSide;
  data: {
    quantity: number;
    avg_price: number;
    market_price?: number;
    unrealized_pnl: number;
    realized_pnl?: number;
    cost_basis?: number;
    market_value?: number;
    day_pnl?: number;
    total_pnl?: number;
    timestamp: string;
  };
}

// Performance Update Messages
export interface PerformanceUpdateMessage extends DataMessage {
  type: 'performance_update';
  portfolio_id: string;
  strategy_id?: string;
  data: {
    pnl: number;
    returns: number;
    sharpe_ratio?: number;
    max_drawdown?: number;
    win_rate?: number;
    profit_factor?: number;
    total_trades?: number;
    winning_trades?: number;
    losing_trades?: number;
    avg_trade_pnl?: number;
    best_trade?: number;
    worst_trade?: number;
    timestamp: string;
  };
}

// Risk Alert Messages
export interface RiskAlertMessage extends DataMessage {
  type: 'risk_alert';
  portfolio_id: string;
  risk_type: string;
  severity: AlertSeverity;
  data: {
    metric: string;
    current_value: number;
    threshold: number;
    limit_value?: number;
    breach_percentage?: number;
    message: string;
    recommended_action?: string;
    auto_action_taken?: boolean;
    timestamp: string;
  };
}

// Breach Alert Messages
export interface BreachAlertMessage extends DataMessage {
  type: 'breach_alert';
  portfolio_id: string;
  breach_type: string;
  severity: AlertSeverity;
  data: {
    limit_type: string;
    current_value: number;
    limit_value: number;
    breach_percentage: number;
    breach_duration?: number;
    auto_action?: string;
    manual_override?: boolean;
    timestamp: string;
  };
}

// Strategy Performance Messages
export interface StrategyPerformanceMessage extends DataMessage {
  type: 'strategy_performance';
  strategy_id: string;
  strategy_name: string;
  data: {
    total_return: number;
    sharpe_ratio: number;
    max_drawdown: number;
    trade_count: number;
    win_rate: number;
    profit_factor: number;
    calmar_ratio?: number;
    sortino_ratio?: number;
    var_95?: number;
    cvar_95?: number;
    alpha?: number;
    beta?: number;
    information_ratio?: number;
    tracking_error?: number;
    timestamp: string;
  };
}

// Execution Analytics Messages
export interface ExecutionAnalyticsMessage extends DataMessage {
  type: 'execution_analytics';
  execution_id: string;
  symbol: string;
  data: {
    fill_price: number;
    benchmark_price: number;
    slippage: number;
    slippage_bps: number;
    execution_shortfall?: number;
    market_impact?: number;
    timing_cost?: number;
    opportunity_cost?: number;
    venue_quality?: number;
    fill_ratio?: number;
    participation_rate?: number;
    execution_time_ms: number;
    timestamp: string;
  };
}

// Risk Metrics Messages
export interface RiskMetricsMessage extends DataMessage {
  type: 'risk_metrics';
  portfolio_id: string;
  metric_type: string;
  data: {
    value: number;
    percentile?: number;
    historical_avg?: number;
    trend?: 'increasing' | 'decreasing' | 'stable';
    confidence_level?: number;
    calculation_method?: string;
    lookback_period?: string;
    timestamp: string;
  };
}

// System Health Messages
export interface SystemHealthMessage extends DataMessage {
  type: 'system_health';
  component?: string;
  status?: 'healthy' | 'degraded' | 'error';
  data: {
    cpu_usage?: number;
    memory_usage?: number;
    disk_usage?: number;
    network_latency?: number;
    active_connections?: number;
    error_rate?: number;
    throughput?: number;
    uptime?: number;
    version?: string;
    last_restart?: string;
    health_score?: number;
    status: string;
    timestamp: string;
  };
}

// Engine Status Messages
export interface EngineStatusMessage extends DataMessage {
  type: 'engine_status';
  engine_id?: string;
  data: {
    state: 'starting' | 'running' | 'stopping' | 'stopped' | 'error';
    uptime: number;
    memory_usage?: number;
    cpu_usage?: number;
    active_strategies?: number;
    orders_processed?: number;
    trades_executed?: number;
    last_heartbeat?: string;
    error_count?: number;
    warning_count?: number;
    performance_score?: number;
    timestamp: string;
  };
}

// Event and Alert Messages
export interface EventMessage extends DataMessage {
  type: 'event';
  event_type: string;
  event_id: string;
  source: string;
  data: {
    title?: string;
    description: string;
    category?: string;
    tags?: string[];
    user_id?: string;
    session_id?: string;
    timestamp: string;
  };
}

export interface AlertMessage extends DataMessage {
  type: 'alert';
  alert_level: AlertSeverity;
  alert_category?: string;
  data: {
    title: string;
    message: string;
    source?: string;
    affected_component?: string;
    resolution_steps?: string[];
    auto_resolve?: boolean;
    expiry_time?: string;
    acknowledged?: boolean;
    timestamp: string;
  };
}

export interface SystemAlertMessage extends DataMessage {
  type: 'system_alert';
  component: string;
  alert_type: string;
  severity: AlertSeverity;
  data: {
    metric?: string;
    current_value?: number;
    threshold?: number;
    description: string;
    impact_level?: 'low' | 'medium' | 'high' | 'critical';
    estimated_resolution_time?: number;
    mitigation_actions?: string[];
    timestamp: string;
  };
}

// Error Messages
export interface ErrorMessage extends BaseWebSocketMessage {
  type: 'error';
  error_code: string;
  error_message: string;
  details?: Record<string, any>;
}

// Success/Acknowledgment Messages
export interface AckMessage extends BaseWebSocketMessage {
  type: 'ack' | 'success';
  ack_message_id?: string;
  status?: string;
  data?: Record<string, any>;
}

// Command Messages
export interface CommandMessage extends BaseWebSocketMessage {
  type: 'command';
  command: string;
  parameters?: Record<string, any>;
  target?: string;
}

export interface CommandResponseMessage extends DataMessage {
  type: 'command_response';
  command_id?: string;
  command: string;
  status: 'success' | 'error' | 'pending';
  data: {
    result?: any;
    error?: string;
    execution_time_ms?: number;
    timestamp: string;
  };
}

// Union Types
export type WebSocketMessage = 
  | ConnectionMessage
  | HeartbeatMessage
  | SubscriptionMessage
  | SubscriptionErrorMessage
  | MarketDataMessage
  | TradeUpdateMessage
  | OrderUpdateMessage
  | PositionUpdateMessage
  | PerformanceUpdateMessage
  | RiskAlertMessage
  | BreachAlertMessage
  | StrategyPerformanceMessage
  | ExecutionAnalyticsMessage
  | RiskMetricsMessage
  | SystemHealthMessage
  | EngineStatusMessage
  | EventMessage
  | AlertMessage
  | SystemAlertMessage
  | ErrorMessage
  | AckMessage
  | CommandMessage
  | CommandResponseMessage;

// WebSocket Connection Info
export interface WebSocketConnectionInfo {
  connectionId: string;
  sessionId: string;
  userId?: string;
  connectedAt: string;
  lastActivity: string;
  protocolVersion: string;
  uptime: number;
  messagesSent: number;
  messagesReceived: number;
  bytesTransferred: number;
  errorCount: number;
  subscriptionCount: number;
  connectionState: ConnectionState;
  connectionQuality: number;
}

// Subscription Configuration
export interface SubscriptionConfig {
  id: string;
  type: MessageType;
  filters: SubscriptionFilters;
  isActive: boolean;
  messageCount: number;
  errorCount: number;
  createdAt: string;
  lastActivity: string;
  rateLimit?: number;
  queueSize?: number;
  priority?: MessagePriority;
}

// Performance Metrics
export interface WebSocketPerformanceMetrics {
  messageLatency: number;
  messagesPerSecond: number;
  errorRate: number;
  reconnectionCount: number;
  uptime: number;
  throughput: number;
  packetLoss: number;
  jitter: number;
  qualityScore: number;
}

// Connection Health
export interface ConnectionHealth {
  isHealthy: boolean;
  qualityScore: number;
  stabilityScore: number;
  latencyStats: {
    current: number;
    average: number;
    min: number;
    max: number;
    p95: number;
  };
  throughputStats: {
    current: number;
    average: number;
    peak: number;
  };
  errorStats: {
    total: number;
    rate: number;
    recentErrors: string[];
  };
  connectionStats: {
    uptime: number;
    reconnections: number;
    lastReconnection?: string;
  };
}

// Stream Statistics
export interface StreamStatistics {
  streamId: string;
  messageType: MessageType;
  totalMessages: number;
  messagesPerSecond: number;
  averageLatency: number;
  errorCount: number;
  lastMessageTime: string;
  dataFreshness: number;
  subscriptionCount: number;
  isHealthy: boolean;
}

// WebSocket Manager Configuration
export interface WebSocketManagerConfig {
  url: string;
  autoReconnect: boolean;
  reconnectInterval: number;
  maxReconnectAttempts: number;
  heartbeatInterval: number;
  messageQueueSize: number;
  enableDebugLogging: boolean;
  compressionEnabled: boolean;
  binaryMode: boolean;
  protocols?: string[];
  headers?: Record<string, string>;
}

// Rate Limiting Configuration
export interface RateLimitConfig {
  messagesPerSecond: number;
  burstSize: number;
  windowSizeMs: number;
  enabled: boolean;
}

// Message Handler Configuration
export interface MessageHandlerConfig {
  id: string;
  messageTypes: MessageType[];
  filters?: (message: WebSocketMessage) => boolean;
  priority?: MessagePriority;
  rateLimit?: RateLimitConfig;
  errorHandling?: 'ignore' | 'retry' | 'escalate';
}

// WebSocket Event Types
export interface WebSocketEvents {
  'connection-state-changed': (state: ConnectionState) => void;
  'message-received': (message: WebSocketMessage) => void;
  'message-sent': (message: WebSocketMessage) => void;
  'subscription-added': (subscription: SubscriptionConfig) => void;
  'subscription-removed': (subscriptionId: string) => void;
  'error': (error: Error) => void;
  'performance-update': (metrics: WebSocketPerformanceMetrics) => void;
  'health-update': (health: ConnectionHealth) => void;
}

// Export utility types
export type MessageTypeOf<T> = T extends { type: infer U } ? U : never;
export type DataOf<T> = T extends { data: infer U } ? U : never;
export type RequiredMessage<T extends WebSocketMessage> = Required<T>;
export type OptionalMessage<T extends WebSocketMessage> = Partial<T>;

// Type guards
export function isMarketDataMessage(message: WebSocketMessage): message is MarketDataMessage {
  return message.type === 'market_data';
}

export function isTradeUpdateMessage(message: WebSocketMessage): message is TradeUpdateMessage {
  return message.type === 'trade_updates';
}

export function isRiskAlertMessage(message: WebSocketMessage): message is RiskAlertMessage {
  return message.type === 'risk_alert';
}

export function isErrorMessage(message: WebSocketMessage): message is ErrorMessage {
  return message.type === 'error';
}

export function isHeartbeatMessage(message: WebSocketMessage): message is HeartbeatMessage {
  return message.type === 'heartbeat' || message.type === 'heartbeat_response';
}

// Message validation functions
export function validateMessageStructure(message: any): message is WebSocketMessage {
  return (
    typeof message === 'object' &&
    message !== null &&
    typeof message.type === 'string' &&
    typeof message.timestamp === 'string'
  );
}

export function validateSubscriptionFilters(filters: any): filters is SubscriptionFilters {
  return (
    typeof filters === 'object' &&
    filters !== null &&
    (filters.symbols === undefined || Array.isArray(filters.symbols)) &&
    (filters.portfolio_ids === undefined || Array.isArray(filters.portfolio_ids)) &&
    (filters.strategy_ids === undefined || Array.isArray(filters.strategy_ids))
  );
}