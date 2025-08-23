// Sprint 3 Enhanced Deployment Pipeline Types

export interface DeploymentPipelineStage {
  id: string;
  name: string;
  type: 'validation' | 'backtesting' | 'paper_trading' | 'staging' | 'production';
  status: 'pending' | 'running' | 'completed' | 'failed' | 'skipped' | 'cancelled';
  duration_ms?: number;
  started_at?: Date;
  completed_at?: Date;
  error_message?: string;
  artifacts?: PipelineArtifact[];
  success_criteria?: StageSuccessCriteria;
  auto_advance: boolean;
  required_approvals?: string[];
  stage_config?: Record<string, any>;
}

export interface PipelineArtifact {
  id: string;
  type: 'backtest_report' | 'validation_log' | 'performance_metrics' | 'risk_report';
  url: string;
  created_at: Date;
  size_bytes?: number;
  metadata?: Record<string, any>;
}

export interface StageSuccessCriteria {
  min_trades?: number;
  max_drawdown?: number;
  min_sharpe_ratio?: number;
  max_var_95?: number;
  win_rate_threshold?: number;
  profit_factor_min?: number;
  validation_checks?: ValidationCheck[];
}

export interface ValidationCheck {
  name: string;
  description: string;
  required: boolean;
  passed?: boolean;
  details?: string;
}

export interface AdvancedDeploymentPipeline {
  pipeline_id: string;
  strategy_id: string;
  version: string;
  created_by: string;
  created_at: Date;
  status: 'draft' | 'running' | 'completed' | 'failed' | 'cancelled' | 'paused';
  stages: DeploymentPipelineStage[];
  current_stage?: string;
  configuration: PipelineConfiguration;
  progress: PipelineProgress;
  metadata?: Record<string, any>;
}

export interface PipelineConfiguration {
  deployment_strategy: 'direct' | 'blue_green' | 'canary' | 'rolling';
  auto_advance: boolean;
  rollback_triggers: RollbackTrigger[];
  notification_settings: NotificationSettings;
  resource_limits: ResourceLimits;
  timeout_settings: TimeoutSettings;
}

export interface RollbackTrigger {
  id: string;
  type: 'performance_based' | 'ml_prediction' | 'manual' | 'time_based';
  enabled: boolean;
  conditions: RollbackCondition[];
  action: 'pause' | 'rollback' | 'alert';
}

export interface RollbackCondition {
  metric: string;
  operator: 'gt' | 'lt' | 'gte' | 'lte' | 'eq' | 'neq';
  value: number;
  window_minutes?: number;
}

export interface NotificationSettings {
  channels: ('email' | 'slack' | 'webhook')[];
  recipients: string[];
  events: ('stage_complete' | 'stage_failed' | 'approval_required' | 'rollback_triggered')[];
}

export interface ResourceLimits {
  max_cpu_cores: number;
  max_memory_mb: number;
  max_disk_mb: number;
  max_network_mbps?: number;
}

export interface TimeoutSettings {
  validation_timeout_minutes: number;
  backtesting_timeout_minutes: number;
  paper_trading_timeout_minutes: number;
  staging_timeout_minutes: number;
  production_timeout_minutes: number;
}

export interface PipelineProgress {
  overall_progress: number;
  current_stage_progress: number;
  stages_completed: number;
  stages_total: number;
  estimated_completion?: Date;
  elapsed_minutes: number;
}

export interface DeploymentOrchestrator {
  active_pipelines: AdvancedDeploymentPipeline[];
  completed_pipelines: AdvancedDeploymentPipeline[];
  failed_pipelines: AdvancedDeploymentPipeline[];
  resource_usage: ResourceUsageStats;
  performance_metrics: OrchestratorMetrics;
}

export interface ResourceUsageStats {
  cpu_usage_percent: number;
  memory_usage_percent: number;
  disk_usage_percent: number;
  network_usage_mbps: number;
  concurrent_pipelines: number;
  max_concurrent_pipelines: number;
}

export interface OrchestratorMetrics {
  total_deployments: number;
  successful_deployments: number;
  failed_deployments: number;
  average_deployment_time_minutes: number;
  success_rate: number;
  rollback_rate: number;
  uptime_percent: number;
}

export interface EnhancedTestSuite {
  suite_id: string;
  name: string;
  version: string;
  tests: TestCase[];
  execution_settings: TestExecutionSettings;
  results?: TestSuiteResults;
}

export interface TestCase {
  id: string;
  name: string;
  type: 'syntax' | 'unit' | 'integration' | 'performance' | 'risk' | 'regression';
  description: string;
  enabled: boolean;
  timeout_seconds: number;
  test_config: Record<string, any>;
  success_criteria?: Record<string, any>;
}

export interface TestExecutionSettings {
  parallel_execution: boolean;
  max_parallel_tests: number;
  retry_failed_tests: boolean;
  max_retries: number;
  fail_fast: boolean;
  generate_reports: boolean;
}

export interface TestSuiteResults {
  suite_id: string;
  executed_at: Date;
  total_tests: number;
  passed_tests: number;
  failed_tests: number;
  skipped_tests: number;
  execution_time_seconds: number;
  test_results: TestResult[];
  summary: TestSummary;
}

export interface TestResult {
  test_id: string;
  status: 'passed' | 'failed' | 'skipped' | 'error';
  execution_time_seconds: number;
  error_message?: string;
  details?: Record<string, any>;
  artifacts?: string[];
}

export interface TestSummary {
  overall_status: 'passed' | 'failed' | 'partial';
  critical_failures: number;
  warnings: number;
  coverage_percent?: number;
  performance_score?: number;
  risk_score?: number;
}

export interface MLRollbackService {
  model_id: string;
  model_version: string;
  enabled: boolean;
  confidence_threshold: number;
  prediction_window_minutes: number;
  monitored_metrics: MonitoredMetric[];
  rollback_history: RollbackEvent[];
}

export interface MonitoredMetric {
  name: string;
  weight: number;
  threshold_type: 'anomaly' | 'trend' | 'absolute';
  threshold_value?: number;
  anomaly_sensitivity?: number;
}

export interface RollbackEvent {
  event_id: string;
  timestamp: Date;
  trigger_type: 'ml_prediction' | 'performance' | 'manual';
  confidence_score?: number;
  metrics_snapshot: Record<string, number>;
  rollback_executed: boolean;
  outcome: 'successful' | 'failed' | 'partial';
}

export interface ApprovalWorkflow {
  workflow_id: string;
  pipeline_id: string;
  stage_id: string;
  required_approvals: ApprovalRequirement[];
  current_approvals: Approval[];
  status: 'pending' | 'approved' | 'rejected' | 'expired';
  created_at: Date;
  expires_at?: Date;
}

export interface ApprovalRequirement {
  role: string;
  user_id?: string;
  required_count: number;
  approval_type: 'any' | 'all' | 'majority';
  escalation_after_hours?: number;
}

export interface Approval {
  approval_id: string;
  user_id: string;
  role: string;
  status: 'approved' | 'rejected';
  timestamp: Date;
  comments?: string;
  signature?: string;
}

export interface GitLikeVersion {
  version_id: string;
  strategy_id: string;
  commit_hash: string;
  version_number: string;
  branch: string;
  parent_version?: string;
  author: string;
  commit_message: string;
  timestamp: Date;
  tags: string[];
  diff?: VersionDiff;
  metadata?: Record<string, any>;
}

export interface VersionDiff {
  added_files: string[];
  modified_files: string[];
  deleted_files: string[];
  parameter_changes: ParameterChange[];
  configuration_changes: ConfigurationChange[];
}

export interface ParameterChange {
  parameter_name: string;
  old_value: any;
  new_value: any;
  change_type: 'added' | 'modified' | 'removed';
}

export interface ConfigurationChange {
  section: string;
  field: string;
  old_value: any;
  new_value: any;
  impact_level: 'low' | 'medium' | 'high' | 'critical';
}

export interface ProductionMonitoringDashboard {
  strategy_id: string;
  deployment_id: string;
  monitoring_start: Date;
  real_time_metrics: RealTimeMetrics;
  alerts: MonitoringAlert[];
  performance_indicators: PerformanceIndicator[];
  health_score: number;
}

export interface RealTimeMetrics {
  pnl_unrealized: number;
  pnl_realized: number;
  position_count: number;
  order_count: number;
  fill_rate: number;
  latency_ms: number;
  cpu_usage: number;
  memory_usage: number;
  last_updated: Date;
}

export interface MonitoringAlert {
  alert_id: string;
  type: 'performance' | 'risk' | 'system' | 'business';
  severity: 'low' | 'medium' | 'high' | 'critical';
  message: string;
  timestamp: Date;
  acknowledged: boolean;
  resolved: boolean;
  resolution_notes?: string;
}

export interface PerformanceIndicator {
  name: string;
  value: number;
  unit: string;
  status: 'good' | 'warning' | 'critical';
  trend: 'improving' | 'stable' | 'degrading';
  benchmark?: number;
}

// API Request/Response Types
export interface CreatePipelineRequest {
  strategy_id: string;
  version?: string;
  configuration: PipelineConfiguration;
  stages?: DeploymentPipelineStage[];
  auto_start?: boolean;
}

export interface PipelineStatusResponse {
  pipeline: AdvancedDeploymentPipeline;
  logs?: string[];
  recommendations?: string[];
}

export interface ExecuteTestSuiteRequest {
  strategy_id: string;
  test_suite: EnhancedTestSuite;
  environment?: 'development' | 'staging' | 'production';
}

export interface MLRollbackPredictionRequest {
  strategy_id: string;
  current_metrics: Record<string, number>;
  time_horizon_minutes: number;
}

export interface MLRollbackPredictionResponse {
  prediction: 'stable' | 'degrading' | 'critical';
  confidence_score: number;
  risk_factors: string[];
  recommendations: string[];
  time_to_rollback_minutes?: number;
}

export interface ApprovalWorkflowRequest {
  pipeline_id: string;
  stage_id: string;
  approval_requirements: ApprovalRequirement[];
  approval_context?: Record<string, any>;
}

export interface VersionControlRequest {
  strategy_id: string;
  operation: 'commit' | 'branch' | 'merge' | 'tag' | 'rollback';
  target?: string;
  message?: string;
  metadata?: Record<string, any>;
}

export interface ProductionMonitoringRequest {
  strategy_id: string;
  deployment_id: string;
  monitoring_config: {
    metrics_enabled: string[];
    alert_thresholds: Record<string, number>;
    notification_channels: string[];
  };
}

// Component Props Types
export interface AdvancedDeploymentPipelineProps {
  strategyId: string;
  initialConfiguration?: Partial<PipelineConfiguration>;
  onPipelineCreated?: (pipelineId: string) => void;
  onPipelineCompleted?: (pipelineId: string, success: boolean) => void;
  onClose?: () => void;
}

export interface DeploymentOrchestratorProps {
  maxConcurrentPipelines?: number;
  autoRefreshInterval?: number;
  showMetrics?: boolean;
  onPipelineSelect?: (pipelineId: string) => void;
}

export interface PipelineStatusMonitorProps {
  pipelineId: string;
  autoRefresh?: boolean;
  showLogs?: boolean;
  onStatusChange?: (status: string) => void;
}

export interface AutomatedTestingSuiteProps {
  strategyId: string;
  testSuite?: EnhancedTestSuite;
  environment?: 'development' | 'staging' | 'production';
  onTestComplete?: (results: TestSuiteResults) => void;
}

export interface RollbackServiceManagerProps {
  strategyId: string;
  deploymentId?: string;
  enableMLPredictions?: boolean;
  onRollbackTriggered?: (reason: string) => void;
}

export interface DeploymentApprovalEngineProps {
  workflowId?: string;
  pipelineId: string;
  stageId: string;
  requiredApprovals: ApprovalRequirement[];
  onApprovalComplete?: (approved: boolean) => void;
}

export interface StrategyVersionControlProps {
  strategyId: string;
  showDiffs?: boolean;
  allowBranching?: boolean;
  onVersionChange?: (versionId: string) => void;
}

export interface ProductionMonitorProps {
  strategyId: string;
  deploymentId: string;
  refreshInterval?: number;
  showAlerts?: boolean;
  onAlert?: (alert: MonitoringAlert) => void;
}