/**
 * Strategy Rollback Service Hook
 * Manages automated rollback with performance triggers
 */

import { useState, useCallback, useEffect } from 'react';
import { message } from 'antd';

export interface RollbackTrigger {
  triggerId: string;
  name: string;
  type: 'performance' | 'error_rate' | 'drawdown' | 'custom';
  enabled: boolean;
  conditions: {
    metric: string;
    operator: '>' | '<' | '>=' | '<=' | '==' | '!=';
    threshold: number;
    duration?: number; // seconds
    comparison?: 'absolute' | 'relative' | 'rolling_average';
  }[];
  actions: {
    rollback: boolean;
    notify: boolean;
    pauseTrading: boolean;
    emergencyStop: boolean;
  };
  metadata: Record<string, any>;
}

export interface RollbackPlan {
  planId: string;
  strategyId: string;
  environment: string;
  currentVersion: string;
  targetVersion: string;
  rollbackType: 'automatic' | 'manual' | 'emergency';
  estimatedDuration: number; // seconds
  validationSteps: string[];
  rollbackSteps: string[];
  prerequisites: string[];
  riskAssessment: {
    riskLevel: 'low' | 'medium' | 'high' | 'critical';
    potentialImpact: string;
    mitigationSteps: string[];
  };
}

export interface RollbackExecution {
  executionId: string;
  planId: string;
  strategyId: string;
  environment: string;
  fromVersion: string;
  toVersion: string;
  status: 'pending' | 'validating' | 'rolling_back' | 'verifying' | 'completed' | 'failed' | 'cancelled';
  triggerReason: string;
  executedBy: string;
  startedAt: Date;
  completedAt?: Date;
  duration?: number;
  steps: Array<{
    stepId: string;
    name: string;
    status: 'pending' | 'running' | 'completed' | 'failed' | 'skipped';
    startedAt?: Date;
    completedAt?: Date;
    output?: string;
    error?: string;
  }>;
  metrics: {
    performanceImpact: Record<string, number>;
    rollbackEffectiveness: number;
    systemStability: Record<string, boolean>;
  };
}

export interface PerformanceMonitor {
  monitorId: string;
  strategyId: string;
  environment: string;
  isActive: boolean;
  triggers: RollbackTrigger[];
  monitoringWindow: number; // seconds
  alertThresholds: Record<string, number>;
  lastCheck: Date;
  currentMetrics: Record<string, any>;
  alertHistory: Array<{
    timestamp: Date;
    alertType: string;
    severity: 'info' | 'warning' | 'error' | 'critical';
    message: string;
    triggered: boolean;
  }>;
}

export interface RollbackHistory {
  totalRollbacks: number;
  automaticRollbacks: number;
  manualRollbacks: number;
  successRate: number;
  averageDuration: number;
  recentRollbacks: RollbackExecution[];
  triggerFrequency: Record<string, number>;
  environmentStats: Record<string, any>;
}

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8001';

export const useRollbackService = () => {
  const [rollbackPlans, setRollbackPlans] = useState<RollbackPlan[]>([]);
  const [rollbackExecutions, setRollbackExecutions] = useState<RollbackExecution[]>([]);
  const [performanceMonitors, setPerformanceMonitors] = useState<PerformanceMonitor[]>([]);
  const [rollbackTriggers, setRollbackTriggers] = useState<RollbackTrigger[]>([]);
  const [rollbackHistory, setRollbackHistory] = useState<RollbackHistory | null>(null);
  const [activeMonitoring, setActiveMonitoring] = useState<Set<string>>(new Set());
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Fetch rollback plans
  const fetchRollbackPlans = useCallback(async (strategyId?: string, environment?: string) => {
    try {
      setLoading(true);
      const params = new URLSearchParams();
      if (strategyId) params.append('strategy_id', strategyId);
      if (environment) params.append('environment', environment);

      const response = await fetch(`${API_BASE_URL}/api/v1/strategy/rollback/plans?${params}`);
      const data = await response.json();
      
      setRollbackPlans(data);
    } catch (err) {
      console.error('Failed to fetch rollback plans:', err);
      setError('Failed to fetch rollback plans');
    } finally {
      setLoading(false);
    }
  }, []);

  // Fetch rollback executions
  const fetchRollbackExecutions = useCallback(async (strategyId?: string, status?: string) => {
    try {
      setLoading(true);
      const params = new URLSearchParams();
      if (strategyId) params.append('strategy_id', strategyId);
      if (status) params.append('status', status);

      const response = await fetch(`${API_BASE_URL}/api/v1/strategy/rollback/executions?${params}`);
      const data = await response.json();
      
      setRollbackExecutions(data.map((execution: any) => ({
        ...execution,
        startedAt: new Date(execution.startedAt),
        completedAt: execution.completedAt ? new Date(execution.completedAt) : undefined,
        steps: execution.steps.map((step: any) => ({
          ...step,
          startedAt: step.startedAt ? new Date(step.startedAt) : undefined,
          completedAt: step.completedAt ? new Date(step.completedAt) : undefined
        }))
      })));
    } catch (err) {
      console.error('Failed to fetch rollback executions:', err);
      setError('Failed to fetch rollback executions');
    } finally {
      setLoading(false);
    }
  }, []);

  // Fetch performance monitors
  const fetchPerformanceMonitors = useCallback(async (strategyId?: string) => {
    try {
      const params = new URLSearchParams();
      if (strategyId) params.append('strategy_id', strategyId);

      const response = await fetch(`${API_BASE_URL}/api/v1/strategy/rollback/monitors?${params}`);
      const data = await response.json();
      
      setPerformanceMonitors(data.map((monitor: any) => ({
        ...monitor,
        lastCheck: new Date(monitor.lastCheck),
        alertHistory: monitor.alertHistory.map((alert: any) => ({
          ...alert,
          timestamp: new Date(alert.timestamp)
        }))
      })));
    } catch (err) {
      console.error('Failed to fetch performance monitors:', err);
      setError('Failed to fetch performance monitors');
    }
  }, []);

  // Fetch rollback triggers
  const fetchRollbackTriggers = useCallback(async (strategyId?: string) => {
    try {
      const params = new URLSearchParams();
      if (strategyId) params.append('strategy_id', strategyId);

      const response = await fetch(`${API_BASE_URL}/api/v1/strategy/rollback/triggers?${params}`);
      const data = await response.json();
      
      setRollbackTriggers(data);
    } catch (err) {
      console.error('Failed to fetch rollback triggers:', err);
      setError('Failed to fetch rollback triggers');
    }
  }, []);

  // Fetch rollback history
  const fetchRollbackHistory = useCallback(async (strategyId?: string) => {
    try {
      const params = new URLSearchParams();
      if (strategyId) params.append('strategy_id', strategyId);

      const response = await fetch(`${API_BASE_URL}/api/v1/strategy/rollback/history?${params}`);
      const data = await response.json();
      
      setRollbackHistory({
        ...data,
        recentRollbacks: data.recentRollbacks.map((execution: any) => ({
          ...execution,
          startedAt: new Date(execution.startedAt),
          completedAt: execution.completedAt ? new Date(execution.completedAt) : undefined
        }))
      });
    } catch (err) {
      console.error('Failed to fetch rollback history:', err);
      setError('Failed to fetch rollback history');
    }
  }, []);

  // Create rollback plan
  const createRollbackPlan = useCallback(async (
    strategyId: string,
    environment: string,
    currentVersion: string,
    targetVersion: string,
    rollbackType: 'automatic' | 'manual' | 'emergency' = 'manual'
  ): Promise<RollbackPlan | null> => {
    try {
      setLoading(true);
      
      const planData = {
        strategy_id: strategyId,
        environment,
        current_version: currentVersion,
        target_version: targetVersion,
        rollback_type: rollbackType
      };

      const response = await fetch(`${API_BASE_URL}/api/v1/strategy/rollback/plans`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(planData)
      });

      if (!response.ok) {
        throw new Error(`Failed to create rollback plan: ${response.statusText}`);
      }

      const result = await response.json();
      message.success(`Rollback plan created successfully`);
      
      await fetchRollbackPlans();
      
      return result;
    } catch (err) {
      console.error('Failed to create rollback plan:', err);
      message.error(`Failed to create rollback plan: ${err}`);
      setError(`Failed to create rollback plan: ${err}`);
      return null;
    } finally {
      setLoading(false);
    }
  }, [fetchRollbackPlans]);

  // Execute rollback
  const executeRollback = useCallback(async (
    planId: string,
    reason: string,
    executedBy: string = 'user'
  ): Promise<RollbackExecution | null> => {
    try {
      setLoading(true);
      
      const executeData = {
        plan_id: planId,
        reason,
        executed_by: executedBy
      };

      const response = await fetch(`${API_BASE_URL}/api/v1/strategy/rollback/execute`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(executeData)
      });

      if (!response.ok) {
        throw new Error(`Failed to execute rollback: ${response.statusText}`);
      }

      const result = await response.json();
      message.success(`Rollback initiated successfully`);
      
      await fetchRollbackExecutions();
      
      return {
        ...result,
        startedAt: new Date(result.startedAt),
        completedAt: result.completedAt ? new Date(result.completedAt) : undefined,
        steps: result.steps.map((step: any) => ({
          ...step,
          startedAt: step.startedAt ? new Date(step.startedAt) : undefined,
          completedAt: step.completedAt ? new Date(step.completedAt) : undefined
        }))
      };
    } catch (err) {
      console.error('Failed to execute rollback:', err);
      message.error(`Failed to execute rollback: ${err}`);
      setError(`Failed to execute rollback: ${err}`);
      return null;
    } finally {
      setLoading(false);
    }
  }, [fetchRollbackExecutions]);

  // Emergency rollback
  const emergencyRollback = useCallback(async (
    strategyId: string,
    environment: string,
    reason: string,
    executedBy: string = 'user'
  ): Promise<RollbackExecution | null> => {
    try {
      setLoading(true);
      
      const emergencyData = {
        strategy_id: strategyId,
        environment,
        reason,
        executed_by: executedBy
      };

      const response = await fetch(`${API_BASE_URL}/api/v1/strategy/rollback/emergency`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(emergencyData)
      });

      if (!response.ok) {
        throw new Error(`Failed to execute emergency rollback: ${response.statusText}`);
      }

      const result = await response.json();
      message.warning(`Emergency rollback initiated`);
      
      await fetchRollbackExecutions();
      
      return {
        ...result,
        startedAt: new Date(result.startedAt),
        completedAt: result.completedAt ? new Date(result.completedAt) : undefined,
        steps: result.steps.map((step: any) => ({
          ...step,
          startedAt: step.startedAt ? new Date(step.startedAt) : undefined,
          completedAt: step.completedAt ? new Date(step.completedAt) : undefined
        }))
      };
    } catch (err) {
      console.error('Failed to execute emergency rollback:', err);
      message.error(`Failed to execute emergency rollback: ${err}`);
      setError(`Failed to execute emergency rollback: ${err}`);
      return null;
    } finally {
      setLoading(false);
    }
  }, [fetchRollbackExecutions]);

  // Create rollback trigger
  const createRollbackTrigger = useCallback(async (
    strategyId: string,
    triggerConfig: Omit<RollbackTrigger, 'triggerId'>
  ): Promise<RollbackTrigger | null> => {
    try {
      setLoading(true);
      
      const triggerData = {
        strategy_id: strategyId,
        ...triggerConfig
      };

      const response = await fetch(`${API_BASE_URL}/api/v1/strategy/rollback/triggers`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(triggerData)
      });

      if (!response.ok) {
        throw new Error(`Failed to create rollback trigger: ${response.statusText}`);
      }

      const result = await response.json();
      message.success(`Rollback trigger created successfully`);
      
      await fetchRollbackTriggers();
      
      return result;
    } catch (err) {
      console.error('Failed to create rollback trigger:', err);
      message.error(`Failed to create rollback trigger: ${err}`);
      setError(`Failed to create rollback trigger: ${err}`);
      return null;
    } finally {
      setLoading(false);
    }
  }, [fetchRollbackTriggers]);

  // Update rollback trigger
  const updateRollbackTrigger = useCallback(async (
    triggerId: string,
    updates: Partial<RollbackTrigger>
  ): Promise<boolean> => {
    try {
      setLoading(true);

      const response = await fetch(`${API_BASE_URL}/api/v1/strategy/rollback/triggers/${triggerId}`, {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(updates)
      });

      if (!response.ok) {
        throw new Error(`Failed to update rollback trigger: ${response.statusText}`);
      }

      message.success(`Rollback trigger updated successfully`);
      
      await fetchRollbackTriggers();
      
      return true;
    } catch (err) {
      console.error('Failed to update rollback trigger:', err);
      message.error(`Failed to update rollback trigger: ${err}`);
      setError(`Failed to update rollback trigger: ${err}`);
      return false;
    } finally {
      setLoading(false);
    }
  }, [fetchRollbackTriggers]);

  // Start performance monitoring
  const startPerformanceMonitoring = useCallback(async (
    strategyId: string,
    environment: string,
    monitoringConfig: {
      window: number;
      triggers: string[];
      alertThresholds: Record<string, number>;
    }
  ): Promise<PerformanceMonitor | null> => {
    try {
      setLoading(true);
      
      const monitorData = {
        strategy_id: strategyId,
        environment,
        monitoring_config: monitoringConfig
      };

      const response = await fetch(`${API_BASE_URL}/api/v1/strategy/rollback/monitors/start`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(monitorData)
      });

      if (!response.ok) {
        throw new Error(`Failed to start performance monitoring: ${response.statusText}`);
      }

      const result = await response.json();
      message.success(`Performance monitoring started`);
      
      setActiveMonitoring(prev => new Set([...prev, result.monitorId]));
      await fetchPerformanceMonitors();
      
      return {
        ...result,
        lastCheck: new Date(result.lastCheck),
        alertHistory: result.alertHistory.map((alert: any) => ({
          ...alert,
          timestamp: new Date(alert.timestamp)
        }))
      };
    } catch (err) {
      console.error('Failed to start performance monitoring:', err);
      message.error(`Failed to start performance monitoring: ${err}`);
      setError(`Failed to start performance monitoring: ${err}`);
      return null;
    } finally {
      setLoading(false);
    }
  }, [fetchPerformanceMonitors]);

  // Stop performance monitoring
  const stopPerformanceMonitoring = useCallback(async (
    monitorId: string
  ): Promise<boolean> => {
    try {
      setLoading(true);

      const response = await fetch(`${API_BASE_URL}/api/v1/strategy/rollback/monitors/${monitorId}/stop`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' }
      });

      if (!response.ok) {
        throw new Error(`Failed to stop performance monitoring: ${response.statusText}`);
      }

      message.success(`Performance monitoring stopped`);
      
      setActiveMonitoring(prev => {
        const newSet = new Set(prev);
        newSet.delete(monitorId);
        return newSet;
      });
      
      await fetchPerformanceMonitors();
      
      return true;
    } catch (err) {
      console.error('Failed to stop performance monitoring:', err);
      message.error(`Failed to stop performance monitoring: ${err}`);
      setError(`Failed to stop performance monitoring: ${err}`);
      return false;
    } finally {
      setLoading(false);
    }
  }, [fetchPerformanceMonitors]);

  // Cancel rollback execution
  const cancelRollbackExecution = useCallback(async (
    executionId: string
  ): Promise<boolean> => {
    try {
      setLoading(true);

      const response = await fetch(`${API_BASE_URL}/api/v1/strategy/rollback/executions/${executionId}/cancel`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' }
      });

      if (!response.ok) {
        throw new Error(`Failed to cancel rollback execution: ${response.statusText}`);
      }

      message.success(`Rollback execution cancelled`);
      
      await fetchRollbackExecutions();
      
      return true;
    } catch (err) {
      console.error('Failed to cancel rollback execution:', err);
      message.error(`Failed to cancel rollback execution: ${err}`);
      setError(`Failed to cancel rollback execution: ${err}`);
      return false;
    } finally {
      setLoading(false);
    }
  }, [fetchRollbackExecutions]);

  // Get rollback plan by ID
  const getRollbackPlan = useCallback((planId: string): RollbackPlan | null => {
    return rollbackPlans.find(plan => plan.planId === planId) || null;
  }, [rollbackPlans]);

  // Get rollback execution by ID
  const getRollbackExecution = useCallback((executionId: string): RollbackExecution | null => {
    return rollbackExecutions.find(exec => exec.executionId === executionId) || null;
  }, [rollbackExecutions]);

  // Get performance monitor by ID
  const getPerformanceMonitor = useCallback((monitorId: string): PerformanceMonitor | null => {
    return performanceMonitors.find(monitor => monitor.monitorId === monitorId) || null;
  }, [performanceMonitors]);

  // Check if monitoring is active
  const isMonitoringActive = useCallback((monitorId: string): boolean => {
    return activeMonitoring.has(monitorId);
  }, [activeMonitoring]);

  // Initialize data
  useEffect(() => {
    fetchRollbackPlans();
    fetchRollbackExecutions();
    fetchPerformanceMonitors();
    fetchRollbackTriggers();
    fetchRollbackHistory();
  }, [
    fetchRollbackPlans,
    fetchRollbackExecutions,
    fetchPerformanceMonitors,
    fetchRollbackTriggers,
    fetchRollbackHistory
  ]);

  return {
    // State
    rollbackPlans,
    rollbackExecutions,
    performanceMonitors,
    rollbackTriggers,
    rollbackHistory,
    activeMonitoring,
    loading,
    error,

    // Actions
    createRollbackPlan,
    executeRollback,
    emergencyRollback,
    createRollbackTrigger,
    updateRollbackTrigger,
    startPerformanceMonitoring,
    stopPerformanceMonitoring,
    cancelRollbackExecution,

    // Queries
    getRollbackPlan,
    getRollbackExecution,
    getPerformanceMonitor,
    isMonitoringActive,

    // Data fetching
    fetchRollbackPlans,
    fetchRollbackExecutions,
    fetchPerformanceMonitors,
    fetchRollbackTriggers,
    fetchRollbackHistory
  };
};