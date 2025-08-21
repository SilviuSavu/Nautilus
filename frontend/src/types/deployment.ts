/**
 * Type definitions for Strategy Deployment Pipeline
 * Extends existing strategy types for deployment-specific functionality
 */

import { StrategyConfig, PerformanceMetrics, StrategyState, ErrorEntry } from '../components/Strategy/types/strategyTypes';

// Core Deployment Types
export interface DeploymentRequest {
  strategyId: string;
  version: string;
  backtestId?: string;
  backtestResults: BacktestResults;
  proposedConfig: StrategyDeploymentConfig;
  riskAssessment: RiskAssessment;
  rolloutPlan: RolloutPlan;
  approvalRequired: boolean;
  requiredApprovals: string[];
  deploymentEnvironment: 'production' | 'staging' | 'development';
  containerName: string;
}

export interface BacktestResults {
  totalReturn: number;
  sharpeRatio: number;
  maxDrawdown: number;
  winRate: number;
  avgTrade: number;
  totalTrades: number;
  calmarRatio?: number;
  sortinoRatio?: number;
  volatility?: number;
  startDate?: Date;
  endDate?: Date;
}

export interface StrategyDeploymentConfig extends StrategyConfig {
  riskEngine: {
    enabled: boolean;
    maxOrderSize: number;
    maxNotionalPerOrder: number;
    maxDailyLoss: number;
    positionLimits: {
      maxPositions: number;
      maxPositionSize: number;
    };
  };
  venues: DeploymentVenue[];
  dataEngine: {
    timeBarsTimestampOnClose: boolean;
    validateDataSequence: boolean;
    bufferDeltas: boolean;
  };
  execEngine: {
    reconciliation: boolean;
    inflightCheckIntervalMs: number;
    snapshotOrders: boolean;
    snapshotPositions: boolean;
  };
  environment: {
    containerName: string;
    databaseUrl: string;
    redisUrl?: string;
    deploymentId: string;
    monitoringEnabled: boolean;
    loggingLevel: 'DEBUG' | 'INFO' | 'WARNING' | 'ERROR';
  };
}

export interface DeploymentVenue {
  name: string;
  venueType: 'ECN' | 'STP' | 'MM';
  accountId: string;
  routing: string;
  clientId: string;
  gatewayHost: string;
  gatewayPort: number;
}

export interface RiskAssessment {
  portfolioImpact: 'low' | 'medium' | 'high' | 'critical';
  correlationRisk: 'low' | 'medium' | 'high';
  maxDrawdownEstimate: number;
  varEstimate: number;
  liquidityRisk: 'low' | 'medium' | 'high';
  warnings: string[];
  blockers: string[];
  recommendations: string[];
}

// Rollout Management
export interface RolloutPlan {
  phases: RolloutPhase[];
  currentPhase: number;
  escalationCriteria: EscalationCriteria;
}

export interface RolloutPhase {
  name: string;
  positionSizePercent: number;
  duration: number; // seconds, -1 for indefinite
  successCriteria: SuccessCriteria;
  rollbackTriggers: RollbackTrigger[];
}

export interface SuccessCriteria {
  minTrades?: number;
  maxDrawdown?: number;
  pnlThreshold?: number;
  ongoing?: boolean;
}

export interface RollbackTrigger {
  type: 'max_loss' | 'consecutive_losses' | 'correlation_breach' | 'performance_deviation';
  threshold: number;
  enabled: boolean;
  action: 'pause' | 'stop' | 'reduce_size' | 'alert_only';
}

export interface EscalationCriteria {
  maxLossPercentage: number;
  consecutiveLosses: number;
  correlationThreshold: number;
}

// Deployment Lifecycle
export interface StrategyDeployment {
  deploymentId: string;
  strategyId: string;
  version: string;
  backtestId?: string;
  deploymentConfig: StrategyDeploymentConfig;
  rolloutPlan: RolloutPlan;
  status: DeploymentStatus;
  createdBy: string;
  createdAt: Date;
  approvedBy?: string;
  approvedAt?: Date;
  deployedAt?: Date;
  stoppedAt?: Date;
  approvalChain: DeploymentApproval[];
}

export type DeploymentStatus = 
  | 'draft'
  | 'pending_approval' 
  | 'approved' 
  | 'deploying' 
  | 'deployed' 
  | 'running' 
  | 'paused' 
  | 'stopped' 
  | 'failed' 
  | 'rolled_back';

export interface DeploymentApproval {
  approvalId: string;
  deploymentId: string;
  approverId: string;
  approverName: string;
  approvalLevel: number;
  status: 'pending' | 'approved' | 'rejected';
  comments?: string;
  approvedAt?: Date;
  requiredRole: 'developer' | 'senior_trader' | 'risk_manager' | 'compliance';
}

// Live Strategy Monitoring
export interface LiveStrategy {
  strategyInstanceId: string;
  deploymentId: string;
  strategyId: string;
  version: string;
  state: LiveStrategyState;
  currentPosition: Position;
  realizedPnL: number;
  unrealizedPnL: number;
  performanceMetrics: LivePerformanceMetrics;
  riskMetrics: LiveRiskMetrics;
  lastHeartbeat: Date;
  healthStatus: HealthStatus;
  alerts: StrategyAlert[];
}

export type LiveStrategyState = 
  | 'deploying' 
  | 'running' 
  | 'paused' 
  | 'stopped' 
  | 'error' 
  | 'emergency_stopped';

export interface Position {
  instrument: string;
  side: 'LONG' | 'SHORT' | 'FLAT';
  quantity: number;
  avgPrice: number;
  marketValue: number;
  unrealizedPnL: number;
  lastUpdated: Date;
}

export interface LivePerformanceMetrics extends PerformanceMetrics {
  dailyPnL: number;
  weeklyPnL: number;
  monthlyPnL: number;
  totalVolume: number;
  avgTradeSize: number;
  fillRate: number;
  slippageAvg: number;
  executionQuality: number;
  vsBacktestDeviation: number;
}

export interface LiveRiskMetrics {
  currentDrawdown: number;
  maxDrawdownToday: number;
  valueAtRisk: number;
  expectedShortfall: number;
  leverageRatio: number;
  concentrationRisk: number;
  correlationToPortfolio: number;
  lastRiskCheck: Date;
}

export interface HealthStatus {
  overall: 'healthy' | 'warning' | 'critical' | 'unknown';
  heartbeat: 'active' | 'stale' | 'disconnected';
  dataFeed: 'connected' | 'delayed' | 'disconnected';
  orderExecution: 'normal' | 'degraded' | 'failed';
  riskCompliance: 'compliant' | 'warning' | 'breach';
  lastHealthCheck: Date;
}

export interface StrategyAlert {
  alertId: string;
  strategyInstanceId: string;
  type: AlertType;
  severity: 'info' | 'warning' | 'error' | 'critical';
  message: string;
  timestamp: Date;
  acknowledged: boolean;
  acknowledgedBy?: string;
  resolvedAt?: Date;
}

export type AlertType = 
  | 'performance_deviation'
  | 'risk_limit_breach'
  | 'connection_lost'
  | 'order_execution_failed'
  | 'data_quality_issue'
  | 'heartbeat_missed'
  | 'manual_intervention_required';

// Emergency Controls
export interface EmergencyAction {
  actionId: string;
  strategyInstanceId: string;
  actionType: 'pause' | 'resume' | 'emergency_stop' | 'reduce_size' | 'close_positions';
  reason: string;
  initiatedBy: string;
  initiatedAt: Date;
  executedAt?: Date;
  status: 'pending' | 'executing' | 'completed' | 'failed';
  confirmationRequired: boolean;
  secondConfirmationRequired: boolean;
}

export interface EmergencyStopConfig {
  enabled: boolean;
  triggers: {
    maxDailyLoss: number;
    maxDrawdown: number;
    consecutiveLosses: number;
    manualTrigger: boolean;
  };
  actions: {
    closePositions: boolean;
    cancelOrders: boolean;
    notifyRiskTeam: boolean;
    createIncidentReport: boolean;
  };
}

// API Request/Response Types
export interface CreateDeploymentRequest {
  strategyId: string;
  version: string;
  proposedConfig: Partial<StrategyDeploymentConfig>;
  rolloutPlan: RolloutPlan;
  riskAssessment?: Partial<RiskAssessment>;
}

export interface CreateDeploymentResponse {
  deploymentId: string;
  status: DeploymentStatus;
  validationResult: {
    valid: boolean;
    errors: string[];
    warnings: string[];
  };
}

export interface ApproveDeploymentRequest {
  deploymentId: string;
  comments?: string;
  conditionalApproval?: boolean;
  conditions?: string[];
}

export interface DeployStrategyRequest {
  deploymentId: string;
  forceRestart?: boolean;
}

export interface DeployStrategyResponse {
  success: boolean;
  strategyInstanceId?: string;
  message?: string;
  estimatedStartTime?: Date;
}

export interface ControlStrategyRequest {
  action: 'pause' | 'resume' | 'stop' | 'emergency_stop';
  reason: string;
  force?: boolean;
}

export interface ControlStrategyResponse {
  success: boolean;
  newState: LiveStrategyState;
  message?: string;
  executedAt: Date;
}

export interface RollbackRequest {
  deploymentId: string;
  targetVersion: string;
  reason: string;
  immediate?: boolean;
}

export interface RollbackResponse {
  rollbackId: string;
  estimatedDuration: number;
  affectedStrategies: string[];
  rollbackPlan: RollbackStep[];
}

export interface RollbackStep {
  stepId: string;
  description: string;
  estimatedDuration: number;
  critical: boolean;
  status: 'pending' | 'executing' | 'completed' | 'failed';
}

// Event Types for Real-time Updates
export interface DeploymentEvent {
  eventId: string;
  deploymentId: string;
  strategyInstanceId?: string;
  eventType: string;
  eventData: any;
  timestamp: Date;
}

export interface StrategyMetricsUpdate {
  strategyInstanceId: string;
  performanceMetrics: LivePerformanceMetrics;
  riskMetrics: LiveRiskMetrics;
  positions: Position[];
  alerts: StrategyAlert[];
  timestamp: Date;
}

// UI Component Props
export interface DeploymentPipelineProps {
  strategyId?: string;
  onDeploymentCreated?: (deploymentId: string) => void;
  onClose?: () => void;
}

export interface LiveMonitorProps {
  strategyInstanceId?: string;
  compact?: boolean;
  showAlerts?: boolean;
  refreshInterval?: number;
}

export interface EmergencyControlsProps {
  strategies: LiveStrategy[];
  onEmergencyAction?: (action: EmergencyAction) => void;
  requireConfirmation?: boolean;
}

export interface ApprovalInterfaceProps {
  deployment: StrategyDeployment;
  currentUser: {
    id: string;
    role: string;
    approvalLevel: number;
  };
  onApprove?: (approved: boolean, comments?: string) => void;
}