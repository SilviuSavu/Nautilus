/**
 * useAdvancedDeploymentPipeline Hook
 * Sprint 3: Advanced Strategy Deployment with CI/CD Pipeline
 * 
 * Comprehensive deployment pipeline management with automated testing,
 * staged rollouts, canary deployments, and intelligent rollback mechanisms.
 */

import { useState, useEffect, useCallback, useRef } from 'react';
import { useWebSocketStream } from '../useWebSocketStream';

export type DeploymentStage = 
  | 'validation'
  | 'testing'
  | 'staging'
  | 'canary'
  | 'production'
  | 'monitoring'
  | 'completed'
  | 'failed'
  | 'cancelled'
  | 'rolled_back';

export type DeploymentStrategy = 
  | 'direct'
  | 'blue_green'
  | 'canary'
  | 'rolling'
  | 'a_b_test';

export type TestType = 
  | 'syntax'
  | 'unit'
  | 'integration'
  | 'backtest'
  | 'paper_trading'
  | 'stress_test'
  | 'compliance';

export interface DeploymentPipeline {
  id: string;
  name: string;
  strategyId: string;
  version: string;
  
  // Configuration
  strategy: DeploymentStrategy;
  stages: PipelineStage[];
  rollbackTriggers: RollbackTrigger[];
  
  // Current state
  currentStage: DeploymentStage;
  status: 'pending' | 'running' | 'paused' | 'completed' | 'failed' | 'cancelled';
  progress: number; // 0-100
  
  // Execution details
  startedAt?: string;
  completedAt?: string;
  duration?: number;
  deployedBy: string;
  
  // Environment details
  environments: {
    staging?: EnvironmentDetails;
    canary?: EnvironmentDetails;
    production?: EnvironmentDetails;
  };
  
  // Metrics and monitoring
  metrics: DeploymentMetrics;
  logs: DeploymentLog[];
  
  // Approval workflow
  approvals: ApprovalStep[];
  
  // Rollback information
  rollbackPlan?: RollbackPlan;
  previousVersion?: string;
}

export interface PipelineStage {
  id: string;
  name: string;
  type: DeploymentStage;
  order: number;
  
  // Configuration
  tests: TestConfiguration[];
  approvalRequired: boolean;
  autoPromote: boolean;
  timeout: number; // minutes
  
  // Execution state
  status: 'pending' | 'running' | 'completed' | 'failed' | 'skipped';
  startedAt?: string;
  completedAt?: string;
  duration?: number;
  
  // Results
  results: {
    testResults: TestResult[];
    artifacts: DeploymentArtifact[];
    metrics: Record<string, number>;
    logs: string[];
  };
}

export interface TestConfiguration {
  id: string;
  name: string;
  type: TestType;
  enabled: boolean;
  parameters: Record<string, any>;
  passingThreshold: number;
  criticalTest: boolean;
}

export interface TestResult {
  testId: string;
  name: string;
  type: TestType;
  status: 'pending' | 'running' | 'passed' | 'failed' | 'skipped';
  score: number;
  threshold: number;
  duration: number;
  output: string;
  artifacts: string[];
  timestamp: string;
}

export interface EnvironmentDetails {
  id: string;
  name: string;
  url: string;
  instances: number;
  trafficPercent: number;
  healthStatus: 'healthy' | 'degraded' | 'unhealthy';
  lastDeployment: string;
  version: string;
}

export interface DeploymentMetrics {
  deploymentTime: number;
  testExecutionTime: number;
  rollbackCount: number;
  successRate: number;
  
  // Performance metrics
  latency: {
    p50: number;
    p95: number;
    p99: number;
  };
  
  throughput: number;
  errorRate: number;
  
  // Resource utilization
  cpuUsage: number;
  memoryUsage: number;
  
  // Business metrics
  profitability?: number;
  sharpeRatio?: number;
  maxDrawdown?: number;
}

export interface DeploymentLog {
  id: string;
  timestamp: string;
  level: 'debug' | 'info' | 'warning' | 'error';
  stage: string;
  message: string;
  details?: Record<string, any>;
}

export interface ApprovalStep {
  id: string;
  stage: DeploymentStage;
  approver: string;
  required: boolean;
  status: 'pending' | 'approved' | 'rejected';
  timestamp?: string;
  comment?: string;
}

export interface RollbackTrigger {
  id: string;
  name: string;
  condition: string;
  threshold: number;
  enabled: boolean;
  severity: 'warning' | 'critical';
  autoRollback: boolean;
}

export interface RollbackPlan {
  id: string;
  strategy: 'immediate' | 'gradual' | 'blue_green_switch';
  steps: RollbackStep[];
  estimatedDuration: number;
  dataBackupRequired: boolean;
}

export interface RollbackStep {
  id: string;
  order: number;
  action: string;
  parameters: Record<string, any>;
  estimatedDuration: number;
  status: 'pending' | 'running' | 'completed' | 'failed';
}

export interface DeploymentArtifact {
  id: string;
  name: string;
  type: 'binary' | 'config' | 'report' | 'log';
  url: string;
  size: number;
  checksum: string;
  createdAt: string;
}

export interface DeploymentTemplate {
  id: string;
  name: string;
  description: string;
  category: 'equity' | 'fixed_income' | 'fx' | 'crypto' | 'multi_asset';
  stages: Omit<PipelineStage, 'id' | 'status' | 'startedAt' | 'completedAt' | 'duration' | 'results'>[];
  defaultStrategy: DeploymentStrategy;
  rollbackTriggers: Omit<RollbackTrigger, 'id'>[];
}

export interface UseAdvancedDeploymentPipelineOptions {
  strategyId?: string;
  enableRealTime?: boolean;
  autoApprove?: boolean;
  enableRollbackTriggers?: boolean;
  maxConcurrentDeployments?: number;
}

export interface UseAdvancedDeploymentPipelineReturn {
  // Pipeline data
  pipelines: DeploymentPipeline[];
  activePipelines: DeploymentPipeline[];
  templates: DeploymentTemplate[];
  
  // Current deployment
  currentPipeline: DeploymentPipeline | null;
  currentStage: PipelineStage | null;
  
  // Status
  isLoading: boolean;
  isDeploying: boolean;
  error: string | null;
  
  // Pipeline management
  createPipeline: (config: {
    strategyId: string;
    version: string;
    strategy: DeploymentStrategy;
    templateId?: string;
  }) => Promise<string>;
  
  startDeployment: (pipelineId: string) => Promise<void>;
  pauseDeployment: (pipelineId: string) => Promise<void>;
  resumeDeployment: (pipelineId: string) => Promise<void>;
  cancelDeployment: (pipelineId: string) => Promise<void>;
  
  // Stage management
  promoteToNextStage: (pipelineId: string) => Promise<void>;
  retryStage: (pipelineId: string, stageId: string) => Promise<void>;
  skipStage: (pipelineId: string, stageId: string, reason: string) => Promise<void>;
  
  // Testing
  runTests: (pipelineId: string, stageId: string, testIds?: string[]) => Promise<TestResult[]>;
  retryFailedTests: (pipelineId: string, stageId: string) => Promise<TestResult[]>;
  
  // Approval workflow
  submitForApproval: (pipelineId: string, stageId: string) => Promise<void>;
  approve: (pipelineId: string, approvalId: string, comment?: string) => Promise<void>;
  reject: (pipelineId: string, approvalId: string, reason: string) => Promise<void>;
  
  // Rollback management
  triggerRollback: (pipelineId: string, reason: string) => Promise<void>;
  executeRollbackPlan: (planId: string) => Promise<void>;
  
  // Environment management
  deployToEnvironment: (pipelineId: string, environment: string, trafficPercent?: number) => Promise<void>;
  switchTraffic: (pipelineId: string, fromEnv: string, toEnv: string, percent: number) => Promise<void>;
  
  // Templates
  createTemplate: (template: Omit<DeploymentTemplate, 'id'>) => Promise<string>;
  applyTemplate: (templateId: string, strategyId: string) => Promise<string>;
  
  // Monitoring and analytics
  getDeploymentMetrics: (pipelineId: string) => Promise<DeploymentMetrics>;
  getDeploymentHistory: (strategyId?: string) => Promise<DeploymentPipeline[]>;
  generateReport: (pipelineId: string, format: 'json' | 'pdf') => Promise<string | Blob>;
  
  // Real-time updates
  subscribeToUpdates: (pipelineId: string) => () => void;
  
  // Utilities
  refresh: () => Promise<void>;
  reset: () => void;
}

const DEFAULT_OPTIONS: Required<UseAdvancedDeploymentPipelineOptions> = {
  strategyId: '',
  enableRealTime: true,
  autoApprove: false,
  enableRollbackTriggers: true,
  maxConcurrentDeployments: 5
};

export function useAdvancedDeploymentPipeline(
  options: UseAdvancedDeploymentPipelineOptions = {}
): UseAdvancedDeploymentPipelineReturn {
  const config = { ...DEFAULT_OPTIONS, ...options };
  
  // State
  const [pipelines, setPipelines] = useState<DeploymentPipeline[]>([]);
  const [templates, setTemplates] = useState<DeploymentTemplate[]>([]);
  const [currentPipeline, setCurrentPipeline] = useState<DeploymentPipeline | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [isDeploying, setIsDeploying] = useState(false);
  const [error, setError] = useState<string | null>(null);
  
  // Refs
  const subscriptionsRef = useRef<Map<string, () => void>>(new Map());
  const isMountedRef = useRef(true);
  
  // API base URL
  const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8001';
  
  // WebSocket stream for real-time deployment updates
  const {
    isActive: isRealTimeActive,
    latestMessage,
    startStream,
    stopStream,
    error: streamError
  } = useWebSocketStream({
    streamId: 'deployment_pipeline',
    messageType: 'strategy_performance',
    bufferSize: 500,
    autoSubscribe: config.enableRealTime
  });
  
  // Cleanup on unmount
  useEffect(() => {
    return () => {
      isMountedRef.current = false;
      // Clean up all subscriptions
      subscriptionsRef.current.forEach(unsub => unsub());
      subscriptionsRef.current.clear();
    };
  }, []);
  
  // Process real-time deployment updates
  useEffect(() => {
    if (latestMessage && latestMessage.data) {
      const updateData = latestMessage.data;
      
      if (updateData.type === 'deployment_update' && updateData.pipeline_id) {
        const pipelineId = updateData.pipeline_id;
        
        setPipelines(prev => prev.map(pipeline => {
          if (pipeline.id === pipelineId) {
            const updated = { ...pipeline };
            
            // Update stage information
            if (updateData.stage_update) {
              updated.currentStage = updateData.stage_update.stage;
              updated.progress = updateData.stage_update.progress || pipeline.progress;
              
              updated.stages = pipeline.stages.map(stage => {
                if (stage.id === updateData.stage_update.stage_id) {
                  return {
                    ...stage,
                    status: updateData.stage_update.status,
                    ...updateData.stage_update.details
                  };
                }
                return stage;
              });
            }
            
            // Update metrics
            if (updateData.metrics) {
              updated.metrics = { ...updated.metrics, ...updateData.metrics };
            }
            
            // Add logs
            if (updateData.log) {
              updated.logs = [
                ...updated.logs,
                {
                  id: `log_${Date.now()}`,
                  timestamp: updateData.timestamp || new Date().toISOString(),
                  level: updateData.log.level || 'info',
                  stage: updateData.log.stage || updated.currentStage,
                  message: updateData.log.message,
                  details: updateData.log.details
                }
              ].slice(-1000); // Keep last 1000 logs
            }
            
            return updated;
          }
          return pipeline;
        }));
        
        // Update current pipeline if it's the one being updated
        if (currentPipeline?.id === pipelineId) {
          setCurrentPipeline(prev => {
            if (!prev) return null;
            return pipelines.find(p => p.id === pipelineId) || prev;
          });
        }
      }
    }
  }, [latestMessage, currentPipeline, pipelines]);
  
  // Utility functions
  const generateId = useCallback((prefix: string) => {
    return `${prefix}_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
  }, []);
  
  // Pipeline management
  const createPipeline = useCallback(async (config: {
    strategyId: string;
    version: string;
    strategy: DeploymentStrategy;
    templateId?: string;
  }): Promise<string> => {
    setIsLoading(true);
    
    try {
      const pipelineId = generateId('pipeline');
      
      const response = await fetch(`${API_BASE_URL}/api/v1/deployment/pipelines`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          id: pipelineId,
          strategy_id: config.strategyId,
          version: config.version,
          deployment_strategy: config.strategy,
          template_id: config.templateId
        })
      });
      
      if (!response.ok) {
        throw new Error(`Pipeline creation failed: ${response.statusText}`);
      }
      
      const result = await response.json();
      const newPipeline: DeploymentPipeline = result.pipeline;
      
      setPipelines(prev => [...prev, newPipeline]);
      
      return pipelineId;
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Pipeline creation failed');
      throw err;
    } finally {
      setIsLoading(false);
    }
  }, [generateId, API_BASE_URL]);
  
  const startDeployment = useCallback(async (pipelineId: string): Promise<void> => {
    setIsDeploying(true);
    
    try {
      const response = await fetch(`${API_BASE_URL}/api/v1/deployment/pipelines/${pipelineId}/start`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' }
      });
      
      if (!response.ok) {
        throw new Error(`Deployment start failed: ${response.statusText}`);
      }
      
      // Update pipeline status
      setPipelines(prev => prev.map(pipeline => 
        pipeline.id === pipelineId 
          ? { 
              ...pipeline, 
              status: 'running',
              startedAt: new Date().toISOString()
            }
          : pipeline
      ));
      
      const pipeline = pipelines.find(p => p.id === pipelineId);
      if (pipeline) {
        setCurrentPipeline(pipeline);
      }
      
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Deployment start failed');
      throw err;
    } finally {
      setIsDeploying(false);
    }
  }, [API_BASE_URL, pipelines]);
  
  const pauseDeployment = useCallback(async (pipelineId: string): Promise<void> => {
    try {
      const response = await fetch(`${API_BASE_URL}/api/v1/deployment/pipelines/${pipelineId}/pause`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' }
      });
      
      if (!response.ok) {
        throw new Error(`Deployment pause failed: ${response.statusText}`);
      }
      
      setPipelines(prev => prev.map(pipeline => 
        pipeline.id === pipelineId 
          ? { ...pipeline, status: 'paused' }
          : pipeline
      ));
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Deployment pause failed');
      throw err;
    }
  }, [API_BASE_URL]);
  
  const resumeDeployment = useCallback(async (pipelineId: string): Promise<void> => {
    try {
      const response = await fetch(`${API_BASE_URL}/api/v1/deployment/pipelines/${pipelineId}/resume`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' }
      });
      
      if (!response.ok) {
        throw new Error(`Deployment resume failed: ${response.statusText}`);
      }
      
      setPipelines(prev => prev.map(pipeline => 
        pipeline.id === pipelineId 
          ? { ...pipeline, status: 'running' }
          : pipeline
      ));
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Deployment resume failed');
      throw err;
    }
  }, [API_BASE_URL]);
  
  const cancelDeployment = useCallback(async (pipelineId: string): Promise<void> => {
    try {
      const response = await fetch(`${API_BASE_URL}/api/v1/deployment/pipelines/${pipelineId}/cancel`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' }
      });
      
      if (!response.ok) {
        throw new Error(`Deployment cancellation failed: ${response.statusText}`);
      }
      
      setPipelines(prev => prev.map(pipeline => 
        pipeline.id === pipelineId 
          ? { 
              ...pipeline, 
              status: 'cancelled',
              completedAt: new Date().toISOString()
            }
          : pipeline
      ));
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Deployment cancellation failed');
      throw err;
    }
  }, [API_BASE_URL]);
  
  // Stage management
  const promoteToNextStage = useCallback(async (pipelineId: string): Promise<void> => {
    try {
      const response = await fetch(`${API_BASE_URL}/api/v1/deployment/pipelines/${pipelineId}/promote`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' }
      });
      
      if (!response.ok) {
        throw new Error(`Stage promotion failed: ${response.statusText}`);
      }
      
      // Pipeline will be updated via WebSocket
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Stage promotion failed');
      throw err;
    }
  }, [API_BASE_URL]);
  
  const retryStage = useCallback(async (pipelineId: string, stageId: string): Promise<void> => {
    try {
      const response = await fetch(`${API_BASE_URL}/api/v1/deployment/pipelines/${pipelineId}/stages/${stageId}/retry`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' }
      });
      
      if (!response.ok) {
        throw new Error(`Stage retry failed: ${response.statusText}`);
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Stage retry failed');
      throw err;
    }
  }, [API_BASE_URL]);
  
  const skipStage = useCallback(async (pipelineId: string, stageId: string, reason: string): Promise<void> => {
    try {
      const response = await fetch(`${API_BASE_URL}/api/v1/deployment/pipelines/${pipelineId}/stages/${stageId}/skip`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ reason })
      });
      
      if (!response.ok) {
        throw new Error(`Stage skip failed: ${response.statusText}`);
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Stage skip failed');
      throw err;
    }
  }, [API_BASE_URL]);
  
  // Testing
  const runTests = useCallback(async (pipelineId: string, stageId: string, testIds?: string[]): Promise<TestResult[]> => {
    try {
      const response = await fetch(`${API_BASE_URL}/api/v1/deployment/pipelines/${pipelineId}/stages/${stageId}/tests`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ test_ids: testIds })
      });
      
      if (!response.ok) {
        throw new Error(`Test execution failed: ${response.statusText}`);
      }
      
      const result = await response.json();
      return result.test_results || [];
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Test execution failed');
      throw err;
    }
  }, [API_BASE_URL]);
  
  const retryFailedTests = useCallback(async (pipelineId: string, stageId: string): Promise<TestResult[]> => {
    const pipeline = pipelines.find(p => p.id === pipelineId);
    const stage = pipeline?.stages.find(s => s.id === stageId);
    
    if (!stage) {
      throw new Error('Stage not found');
    }
    
    const failedTestIds = stage.results.testResults
      .filter(test => test.status === 'failed')
      .map(test => test.testId);
    
    return runTests(pipelineId, stageId, failedTestIds);
  }, [pipelines, runTests]);
  
  // Approval workflow
  const submitForApproval = useCallback(async (pipelineId: string, stageId: string): Promise<void> => {
    try {
      const response = await fetch(`${API_BASE_URL}/api/v1/deployment/pipelines/${pipelineId}/stages/${stageId}/approve`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' }
      });
      
      if (!response.ok) {
        throw new Error(`Approval submission failed: ${response.statusText}`);
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Approval submission failed');
      throw err;
    }
  }, [API_BASE_URL]);
  
  const approve = useCallback(async (pipelineId: string, approvalId: string, comment?: string): Promise<void> => {
    try {
      const response = await fetch(`${API_BASE_URL}/api/v1/deployment/pipelines/${pipelineId}/approvals/${approvalId}`, {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ 
          status: 'approved',
          comment 
        })
      });
      
      if (!response.ok) {
        throw new Error(`Approval failed: ${response.statusText}`);
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Approval failed');
      throw err;
    }
  }, [API_BASE_URL]);
  
  const reject = useCallback(async (pipelineId: string, approvalId: string, reason: string): Promise<void> => {
    try {
      const response = await fetch(`${API_BASE_URL}/api/v1/deployment/pipelines/${pipelineId}/approvals/${approvalId}`, {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ 
          status: 'rejected',
          comment: reason 
        })
      });
      
      if (!response.ok) {
        throw new Error(`Rejection failed: ${response.statusText}`);
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Rejection failed');
      throw err;
    }
  }, [API_BASE_URL]);
  
  // Rollback management
  const triggerRollback = useCallback(async (pipelineId: string, reason: string): Promise<void> => {
    try {
      const response = await fetch(`${API_BASE_URL}/api/v1/deployment/pipelines/${pipelineId}/rollback`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ reason })
      });
      
      if (!response.ok) {
        throw new Error(`Rollback trigger failed: ${response.statusText}`);
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Rollback trigger failed');
      throw err;
    }
  }, [API_BASE_URL]);
  
  const executeRollbackPlan = useCallback(async (planId: string): Promise<void> => {
    try {
      const response = await fetch(`${API_BASE_URL}/api/v1/deployment/rollback-plans/${planId}/execute`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' }
      });
      
      if (!response.ok) {
        throw new Error(`Rollback execution failed: ${response.statusText}`);
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Rollback execution failed');
      throw err;
    }
  }, [API_BASE_URL]);
  
  // Environment management
  const deployToEnvironment = useCallback(async (pipelineId: string, environment: string, trafficPercent = 100): Promise<void> => {
    try {
      const response = await fetch(`${API_BASE_URL}/api/v1/deployment/pipelines/${pipelineId}/deploy/${environment}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ traffic_percent: trafficPercent })
      });
      
      if (!response.ok) {
        throw new Error(`Environment deployment failed: ${response.statusText}`);
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Environment deployment failed');
      throw err;
    }
  }, [API_BASE_URL]);
  
  const switchTraffic = useCallback(async (pipelineId: string, fromEnv: string, toEnv: string, percent: number): Promise<void> => {
    try {
      const response = await fetch(`${API_BASE_URL}/api/v1/deployment/pipelines/${pipelineId}/traffic`, {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ 
          from_environment: fromEnv,
          to_environment: toEnv,
          percentage: percent
        })
      });
      
      if (!response.ok) {
        throw new Error(`Traffic switch failed: ${response.statusText}`);
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Traffic switch failed');
      throw err;
    }
  }, [API_BASE_URL]);
  
  // Templates
  const createTemplate = useCallback(async (template: Omit<DeploymentTemplate, 'id'>): Promise<string> => {
    const templateId = generateId('template');
    
    try {
      const response = await fetch(`${API_BASE_URL}/api/v1/deployment/templates`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ ...template, id: templateId })
      });
      
      if (!response.ok) {
        throw new Error(`Template creation failed: ${response.statusText}`);
      }
      
      const newTemplate: DeploymentTemplate = { ...template, id: templateId };
      setTemplates(prev => [...prev, newTemplate]);
      
      return templateId;
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Template creation failed');
      throw err;
    }
  }, [generateId, API_BASE_URL]);
  
  const applyTemplate = useCallback(async (templateId: string, strategyId: string): Promise<string> => {
    const template = templates.find(t => t.id === templateId);
    if (!template) {
      throw new Error('Template not found');
    }
    
    return createPipeline({
      strategyId,
      version: '1.0.0',
      strategy: template.defaultStrategy,
      templateId
    });
  }, [templates, createPipeline]);
  
  // Monitoring and analytics
  const getDeploymentMetrics = useCallback(async (pipelineId: string): Promise<DeploymentMetrics> => {
    try {
      const response = await fetch(`${API_BASE_URL}/api/v1/deployment/pipelines/${pipelineId}/metrics`);
      
      if (!response.ok) {
        throw new Error(`Metrics retrieval failed: ${response.statusText}`);
      }
      
      const data = await response.json();
      return data.metrics;
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Metrics retrieval failed');
      throw err;
    }
  }, [API_BASE_URL]);
  
  const getDeploymentHistory = useCallback(async (strategyId?: string): Promise<DeploymentPipeline[]> => {
    try {
      const url = strategyId 
        ? `${API_BASE_URL}/api/v1/deployment/pipelines?strategy_id=${strategyId}`
        : `${API_BASE_URL}/api/v1/deployment/pipelines`;
      
      const response = await fetch(url);
      
      if (!response.ok) {
        throw new Error(`History retrieval failed: ${response.statusText}`);
      }
      
      const data = await response.json();
      return data.pipelines || [];
    } catch (err) {
      setError(err instanceof Error ? err.message : 'History retrieval failed');
      throw err;
    }
  }, [API_BASE_URL]);
  
  const generateReport = useCallback(async (pipelineId: string, format: 'json' | 'pdf'): Promise<string | Blob> => {
    try {
      const response = await fetch(`${API_BASE_URL}/api/v1/deployment/pipelines/${pipelineId}/report?format=${format}`);
      
      if (!response.ok) {
        throw new Error(`Report generation failed: ${response.statusText}`);
      }
      
      if (format === 'json') {
        const data = await response.json();
        return JSON.stringify(data, null, 2);
      } else {
        return await response.blob();
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Report generation failed');
      throw err;
    }
  }, [API_BASE_URL]);
  
  // Real-time subscriptions
  const subscribeToUpdates = useCallback((pipelineId: string) => {
    if (subscriptionsRef.current.has(pipelineId)) {
      return subscriptionsRef.current.get(pipelineId)!;
    }
    
    // Create subscription (this would integrate with WebSocket stream)
    const unsubscribe = () => {
      subscriptionsRef.current.delete(pipelineId);
    };
    
    subscriptionsRef.current.set(pipelineId, unsubscribe);
    return unsubscribe;
  }, []);
  
  // Control functions
  const refresh = useCallback(async () => {
    setIsLoading(true);
    
    try {
      const [pipelinesResponse, templatesResponse] = await Promise.all([
        fetch(`${API_BASE_URL}/api/v1/deployment/pipelines${config.strategyId ? `?strategy_id=${config.strategyId}` : ''}`),
        fetch(`${API_BASE_URL}/api/v1/deployment/templates`)
      ]);
      
      if (pipelinesResponse.ok) {
        const pipelinesData = await pipelinesResponse.json();
        setPipelines(pipelinesData.pipelines || []);
      }
      
      if (templatesResponse.ok) {
        const templatesData = await templatesResponse.json();
        setTemplates(templatesData.templates || []);
      }
      
      setError(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to refresh data');
    } finally {
      setIsLoading(false);
    }
  }, [API_BASE_URL, config.strategyId]);
  
  const reset = useCallback(() => {
    setPipelines([]);
    setTemplates([]);
    setCurrentPipeline(null);
    setError(null);
    subscriptionsRef.current.forEach(unsub => unsub());
    subscriptionsRef.current.clear();
  }, []);
  
  // Computed values
  const activePipelines = pipelines.filter(p => p.status === 'running' || p.status === 'paused');
  const currentStage = currentPipeline?.stages.find(s => s.type === currentPipeline.currentStage) || null;
  
  // Initial data load
  useEffect(() => {
    refresh();
  }, [refresh]);
  
  return {
    // Pipeline data
    pipelines,
    activePipelines,
    templates,
    
    // Current deployment
    currentPipeline,
    currentStage,
    
    // Status
    isLoading,
    isDeploying,
    error: error || streamError,
    
    // Pipeline management
    createPipeline,
    startDeployment,
    pauseDeployment,
    resumeDeployment,
    cancelDeployment,
    
    // Stage management
    promoteToNextStage,
    retryStage,
    skipStage,
    
    // Testing
    runTests,
    retryFailedTests,
    
    // Approval workflow
    submitForApproval,
    approve,
    reject,
    
    // Rollback management
    triggerRollback,
    executeRollbackPlan,
    
    // Environment management
    deployToEnvironment,
    switchTraffic,
    
    // Templates
    createTemplate,
    applyTemplate,
    
    // Monitoring and analytics
    getDeploymentMetrics,
    getDeploymentHistory,
    generateReport,
    
    // Real-time updates
    subscribeToUpdates,
    
    // Utilities
    refresh,
    reset
  };
}

export default useAdvancedDeploymentPipeline;