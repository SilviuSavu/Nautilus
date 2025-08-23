/**
 * Sprint 3 Types - WebSocket Infrastructure, Real-time Analytics, Risk Management, Strategy Deployment
 * 
 * Core types for Sprint 3 integration including WebSocket infrastructure,
 * real-time analytics, dynamic risk management, and strategy deployment pipeline.
 */

// WebSocket Infrastructure Types
export interface WebSocketConnectionState {
  id: string;
  url: string;
  status: 'connected' | 'connecting' | 'disconnected' | 'error' | 'reconnecting';
  lastConnected?: string;
  lastError?: string;
  messageCount: number;
  latency?: number;
  reconnectAttempts: number;
  subscriptions: string[];
}

export interface WebSocketMetrics {
  totalConnections: number;
  activeConnections: number;
  messageRate: number;
  errorRate: number;
  averageLatency: number;
  uptime: number;
  lastUpdate: string;
}

export interface WebSocketMessage {
  id: string;
  type: string;
  topic: string;
  payload: any;
  timestamp: string;
  source: string;
  processed: boolean;
}

// Real-time Analytics Types
export interface RealTimeAnalyticsConfig {
  enabled: boolean;
  updateInterval: number;
  bufferSize: number;
  metrics: string[];
  alertThresholds: Record<string, number>;
}

export interface AnalyticsStreamData {
  timestamp: string;
  metric: string;
  value: number;
  symbol?: string;
  metadata?: Record<string, any>;
}

export interface PerformanceMetric {
  id: string;
  name: string;
  value: number;
  change: number;
  changePercent: number;
  trend: 'up' | 'down' | 'stable';
  timestamp: string;
  category: 'returns' | 'risk' | 'execution' | 'system';
}

// Risk Management Types
export interface DynamicRiskLimit {
  id: string;
  name: string;
  type: 'position_size' | 'exposure' | 'drawdown' | 'var' | 'custom';
  value: number;
  threshold: number;
  warningThreshold?: number;
  enabled: boolean;
  autoAdjust: boolean;
  lastUpdated: string;
  breachCount: number;
}

export interface RiskBreach {
  id: string;
  limitId: string;
  limitName: string;
  severity: 'info' | 'warning' | 'critical';
  actualValue: number;
  thresholdValue: number;
  timestamp: string;
  resolved: boolean;
  actions: RiskBreachAction[];
}

export interface RiskBreachAction {
  id: string;
  type: 'alert' | 'limit_order' | 'position_reduce' | 'system_halt';
  status: 'pending' | 'executing' | 'completed' | 'failed';
  details: string;
  timestamp: string;
}

export interface RealTimeRiskMetrics {
  var: number;
  drawdown: number;
  exposure: number;
  concentration: number;
  correlation: number;
  leverage: number;
  liquidity: number;
  timestamp: string;
}

// Strategy Deployment Types
export interface StrategyDeploymentPipeline {
  id: string;
  name: string;
  version: string;
  status: 'draft' | 'testing' | 'approval' | 'deploying' | 'active' | 'paused' | 'retired';
  stages: DeploymentStage[];
  currentStage: number;
  createdBy: string;
  createdAt: string;
  deployedAt?: string;
  lastModified: string;
}

export interface DeploymentStage {
  id: string;
  name: string;
  type: 'validation' | 'testing' | 'approval' | 'deployment' | 'monitoring';
  status: 'pending' | 'running' | 'completed' | 'failed' | 'skipped';
  startTime?: string;
  endTime?: string;
  duration?: number;
  logs: DeploymentLog[];
  requirements: DeploymentRequirement[];
  artifacts: DeploymentArtifact[];
}

export interface DeploymentRequirement {
  id: string;
  type: 'approval' | 'test' | 'metric' | 'resource';
  description: string;
  status: 'pending' | 'met' | 'failed';
  value?: string | number;
  threshold?: string | number;
}

export interface DeploymentLog {
  id: string;
  timestamp: string;
  level: 'info' | 'warning' | 'error';
  message: string;
  details?: Record<string, any>;
}

export interface DeploymentArtifact {
  id: string;
  name: string;
  type: 'config' | 'model' | 'report' | 'log';
  path: string;
  size: number;
  checksum: string;
  createdAt: string;
}

// System Health Types
export interface SystemHealthMetrics {
  overall: 'healthy' | 'warning' | 'critical';
  components: ComponentHealth[];
  lastUpdate: string;
  uptime: number;
  alerts: SystemAlert[];
}

export interface ComponentHealth {
  id: string;
  name: string;
  type: 'service' | 'database' | 'websocket' | 'api' | 'engine';
  status: 'healthy' | 'warning' | 'critical' | 'offline';
  metrics: ComponentMetric[];
  lastCheck: string;
  dependencies: string[];
}

export interface ComponentMetric {
  name: string;
  value: number;
  unit: string;
  threshold?: number;
  status: 'normal' | 'warning' | 'critical';
}

export interface SystemAlert {
  id: string;
  type: 'system' | 'performance' | 'security' | 'deployment';
  severity: 'info' | 'warning' | 'critical';
  message: string;
  source: string;
  timestamp: string;
  acknowledged: boolean;
  resolvedAt?: string;
}

// Feature Navigation Types
export interface Sprint3Feature {
  id: string;
  name: string;
  description: string;
  icon: string;
  path: string;
  category: 'websocket' | 'analytics' | 'risk' | 'strategy' | 'monitoring';
  status: 'active' | 'beta' | 'experimental';
  dependencies: string[];
  permissions: string[];
}

export interface QuickAction {
  id: string;
  name: string;
  description: string;
  icon: string;
  action: () => void | Promise<void>;
  category: 'system' | 'trading' | 'monitoring' | 'configuration';
  disabled?: boolean;
  loading?: boolean;
}

// Configuration Types
export interface Sprint3Config {
  websocket: {
    autoReconnect: boolean;
    maxReconnectAttempts: number;
    heartbeatInterval: number;
    bufferSize: number;
  };
  analytics: {
    realTimeEnabled: boolean;
    updateInterval: number;
    metricsBuffer: number;
    alertsEnabled: boolean;
  };
  risk: {
    dynamicLimitsEnabled: boolean;
    breachActionsEnabled: boolean;
    autoLimitAdjustment: boolean;
    alertsEnabled: boolean;
  };
  deployment: {
    autoApproval: boolean;
    testingRequired: boolean;
    rollbackEnabled: boolean;
    monitoringDuration: number;
  };
}

// API Response Types
export interface Sprint3StatusResponse {
  websocket: WebSocketMetrics;
  analytics: {
    enabled: boolean;
    activeStreams: number;
    metricsCount: number;
    lastUpdate: string;
  };
  risk: {
    activeLimits: number;
    breachCount: number;
    alertsCount: number;
    lastCheck: string;
  };
  deployment: {
    activePipelines: number;
    pendingApprovals: number;
    lastDeployment: string;
  };
  system: SystemHealthMetrics;
}

// Event Types
export interface Sprint3Event {
  type: 'websocket' | 'analytics' | 'risk' | 'deployment' | 'system';
  action: string;
  data: any;
  timestamp: string;
  source: string;
}

export type WebSocketEventType = 
  | 'connection_status_changed'
  | 'message_received'
  | 'subscription_updated'
  | 'error_occurred'
  | 'metrics_updated';

export type AnalyticsEventType =
  | 'metric_updated'
  | 'alert_triggered'
  | 'stream_started'
  | 'stream_stopped'
  | 'threshold_breached';

export type RiskEventType =
  | 'limit_breached'
  | 'limit_updated'
  | 'action_triggered'
  | 'breach_resolved'
  | 'metrics_calculated';

export type DeploymentEventType =
  | 'pipeline_created'
  | 'stage_completed'
  | 'approval_requested'
  | 'deployment_started'
  | 'deployment_completed'
  | 'rollback_initiated';

export type SystemEventType =
  | 'health_check_completed'
  | 'alert_created'
  | 'component_status_changed'
  | 'maintenance_scheduled';

// Utility Types
export type Sprint3EventHandler<T = any> = (event: T) => void | Promise<void>;

export interface Sprint3State {
  initialized: boolean;
  features: Record<string, boolean>;
  config: Sprint3Config;
  status: Sprint3StatusResponse;
  lastUpdate: string;
}