/**
 * Story 5.2: System Performance Monitoring Types
 * Comprehensive type definitions for system monitoring components
 */

// Core monitoring interfaces
export interface LatencyMetrics {
  venue_name: string;
  order_execution_latency: {
    min_ms: number;
    max_ms: number;
    avg_ms: number;
    p50_ms: number;
    p95_ms: number;
    p99_ms: number;
    samples: number;
  };
  market_data_latency: {
    tick_to_trade_ms: number;
    feed_latency_ms: number;
    processing_latency_ms: number;
    total_latency_ms: number;
  };
  connection_latency: {
    ping_ms: number;
    jitter_ms: number;
    packet_loss_percent: number;
  };
  last_updated: string;
}

export interface SystemMetrics {
  timestamp: string;
  cpu_metrics: {
    usage_percent: number;
    core_count: number;
    load_average_1m: number;
    load_average_5m: number;
    load_average_15m: number;
    per_core_usage: number[];
    temperature_celsius?: number;
  };
  memory_metrics: {
    total_gb: number;
    used_gb: number;
    available_gb: number;
    usage_percent: number;
    swap_total_gb: number;
    swap_used_gb: number;
    buffer_cache_gb: number;
  };
  network_metrics: {
    bytes_sent_per_sec: number;
    bytes_received_per_sec: number;
    packets_sent_per_sec: number;
    packets_received_per_sec: number;
    errors_per_sec: number;
    active_connections: number;
    bandwidth_utilization_percent: number;
  };
  disk_metrics: {
    total_space_gb: number;
    used_space_gb: number;
    available_space_gb: number;
    usage_percent: number;
    read_iops: number;
    write_iops: number;
    read_throughput_mbps: number;
    write_throughput_mbps: number;
  };
  process_metrics: {
    trading_engine_cpu_percent: number;
    trading_engine_memory_mb: number;
    database_cpu_percent: number;
    database_memory_mb: number;
    total_processes: number;
  };
}

export interface ConnectionQuality {
  venue_name: string;
  status: 'connected' | 'disconnected' | 'degraded' | 'reconnecting';
  quality_score: number;
  uptime_percent_24h: number;
  connection_duration_seconds: number;
  last_disconnect_time?: string;
  disconnect_count_24h: number;
  data_quality: {
    message_rate_per_sec: number;
    duplicate_messages_percent: number;
    out_of_sequence_percent: number;
    stale_data_percent: number;
  };
  performance_metrics: {
    response_time_ms: number;
    throughput_mbps: number;
    error_rate_percent: number;
  };
  reconnection_stats: {
    auto_reconnect_enabled: boolean;
    reconnect_attempts_24h: number;
    avg_reconnect_time_seconds: number;
    max_reconnect_time_seconds: number;
  };
}

export interface PerformanceAlert {
  alert_id: string;
  metric_name: string;
  current_value: number;
  threshold_value: number;
  severity: 'low' | 'medium' | 'high' | 'critical';
  triggered_at: string;
  venue_name?: string;
  description: string;
  auto_resolution_available: boolean;
  escalation_level: number;
  notification_sent: boolean;
}

export interface ResolvedAlert {
  alert_id: string;
  metric_name: string;
  resolved_at: string;
  resolution_method: 'auto' | 'manual' | 'timeout';
  duration_minutes: number;
}

// API Response interfaces
export interface LatencyMonitoringResponse {
  venue_latencies: LatencyMetrics[];
  overall_statistics: {
    best_venue: string;
    worst_venue: string;
    avg_execution_latency_ms: number;
    latency_trend: 'improving' | 'degrading' | 'stable';
  };
  timeframe: string;
}

export interface SystemMonitoringResponse extends SystemMetrics {}

export interface ConnectionMonitoringResponse {
  venue_connections: ConnectionQuality[];
  connection_history?: {
    timestamp: string;
    venue_name: string;
    event_type: 'connect' | 'disconnect' | 'reconnect' | 'error';
    details: string;
  }[];
  overall_health: {
    total_venues: number;
    connected_venues: number;
    degraded_venues: number;
    overall_score: number;
  };
}

export interface AlertsMonitoringResponse {
  active_alerts: PerformanceAlert[];
  resolved_alerts: ResolvedAlert[];
  alert_statistics: {
    total_alerts_24h: number;
    critical_alerts_24h: number;
    avg_resolution_time_minutes: number;
    most_frequent_alert_type: string;
  };
}

export interface PerformanceTrendsResponse {
  trend_analysis: {
    metric_name: string;
    current_value: number;
    trend_direction: 'improving' | 'degrading' | 'stable';
    change_percent_24h: number;
    change_percent_7d: number;
    predicted_value_24h: number;
    confidence_score: number;
  }[];
  capacity_planning: {
    cpu_exhaustion_prediction_days?: number;
    memory_exhaustion_prediction_days?: number;
    disk_exhaustion_prediction_days?: number;
    recommended_actions: string[];
  };
  anomalies_detected: {
    timestamp: string;
    metric_name: string;
    anomaly_score: number;
    description: string;
  }[];
}

// Alert configuration types
export interface AlertConfigurationRequest {
  metric_name: string;
  threshold_value: number;
  condition: 'greater_than' | 'less_than' | 'equals' | 'not_equals';
  severity: 'low' | 'medium' | 'high' | 'critical';
  enabled: boolean;
  venue_filter?: string[];
  notification_channels: {
    email: string[];
    slack_webhook?: string;
    teams_webhook?: string;
    sms_numbers?: string[];
  };
  escalation_rules: {
    escalate_after_minutes: number;
    escalation_contacts: string[];
  };
  auto_resolution: {
    enabled: boolean;
    resolution_threshold?: number;
    max_attempts?: number;
  };
}

export interface AlertConfigurationResponse {
  alert_rule_id: string;
  status: 'created' | 'updated' | 'error';
  message: string;
  validation_errors?: string[];
}

// Component state interfaces
export interface MonitoringState {
  latencyMetrics: LatencyMetrics[];
  systemMetrics: SystemMetrics | null;
  connectionQuality: ConnectionQuality[];
  activeAlerts: PerformanceAlert[];
  performanceTrends: PerformanceTrendsResponse | null;
  loading: boolean;
  error: string | null;
  lastUpdate: Date | null;
}

// Chart data interfaces
export interface LatencyChartData {
  venue: string;
  p50: number;
  p95: number;
  p99: number;
  avg: number;
}

export interface SystemUsageChartData {
  timestamp: string;
  cpu_usage: number;
  memory_usage: number;
  network_usage: number;
  disk_usage: number;
}

export interface ConnectionQualityChartData {
  venue: string;
  quality_score: number;
  uptime_percent: number;
  response_time: number;
  status: string;
}

// Hook configuration interfaces
export interface UseMonitoringConfig {
  refreshInterval?: number;
  venueFilter?: string[];
  autoRefresh?: boolean;
  includeHistory?: boolean;
}

// Service interfaces
export interface MonitoringService {
  getLatencyMetrics(venue?: string, timeframe?: string): Promise<LatencyMonitoringResponse>;
  getSystemMetrics(metrics?: string[], period?: string): Promise<SystemMonitoringResponse>;
  getConnectionMetrics(venue?: string, includeHistory?: boolean): Promise<ConnectionMonitoringResponse>;
  getAlerts(status?: string, severity?: string): Promise<AlertsMonitoringResponse>;
  getPerformanceTrends(period?: string): Promise<PerformanceTrendsResponse>;
  configureAlert(request: AlertConfigurationRequest): Promise<AlertConfigurationResponse>;
}

// Utility types
export type MetricType = 'latency' | 'system' | 'connection' | 'alert' | 'trend';
export type VenueName = 'IB' | 'Alpaca' | 'Binance' | 'all';
export type AlertSeverity = 'low' | 'medium' | 'high' | 'critical';
export type ConnectionStatus = 'connected' | 'disconnected' | 'degraded' | 'reconnecting';
export type TrendDirection = 'improving' | 'degrading' | 'stable';