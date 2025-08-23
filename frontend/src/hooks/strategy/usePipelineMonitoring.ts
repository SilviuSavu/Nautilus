/**
 * Pipeline Monitoring Hook
 * Real-time pipeline monitoring and alerts
 */

import { useState, useCallback, useEffect, useRef } from 'react';
import { message } from 'antd';

export interface PipelineStage {
  stageId: string;
  name: string;
  type: 'test' | 'deploy' | 'verify' | 'approve' | 'rollback';
  status: 'pending' | 'running' | 'completed' | 'failed' | 'skipped' | 'cancelled';
  startedAt?: Date;
  completedAt?: Date;
  duration?: number;
  logs: string[];
  metrics: Record<string, any>;
  dependencies: string[];
  retryCount: number;
  maxRetries: number;
}

export interface PipelineExecution {
  executionId: string;
  pipelineId: string;
  strategyId: string;
  version: string;
  environment: string;
  status: 'pending' | 'running' | 'completed' | 'failed' | 'cancelled' | 'paused';
  stages: PipelineStage[];
  currentStage?: string;
  startedAt: Date;
  completedAt?: Date;
  totalDuration?: number;
  triggeredBy: string;
  metadata: Record<string, any>;
}

export interface PipelineAlert {
  alertId: string;
  pipelineId: string;
  executionId?: string;
  type: 'error' | 'warning' | 'info' | 'success';
  severity: 'low' | 'medium' | 'high' | 'critical';
  title: string;
  message: string;
  timestamp: Date;
  acknowledged: boolean;
  acknowledgedBy?: string;
  acknowledgedAt?: Date;
  actions: Array<{
    action: string;
    label: string;
    url?: string;
  }>;
}

export interface PipelineMetrics {
  pipelineId: string;
  strategyId: string;
  environment: string;
  totalExecutions: number;
  successfulExecutions: number;
  failedExecutions: number;
  successRate: number;
  averageDuration: number;
  lastExecution?: Date;
  currentStatus: string;
  performanceTrend: Array<{
    timestamp: Date;
    duration: number;
    success: boolean;
  }>;
  stageMetrics: Record<string, {
    averageDuration: number;
    failureRate: number;
    retryRate: number;
  }>;
}

export interface MonitoringConfig {
  pipelineId: string;
  enabled: boolean;
  alertRules: Array<{
    ruleId: string;
    name: string;
    condition: string;
    threshold: number;
    severity: 'low' | 'medium' | 'high' | 'critical';
    actions: string[];
  }>;
  notificationChannels: Array<{
    type: 'email' | 'slack' | 'webhook' | 'sms';
    endpoint: string;
    enabled: boolean;
  }>;
  retentionPeriod: number; // days
  samplingInterval: number; // seconds
}

export interface RealTimeUpdate {
  type: 'stage_update' | 'execution_update' | 'alert' | 'metric_update';
  timestamp: Date;
  pipelineId: string;
  executionId?: string;
  data: any;
}

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8001';
const WS_URL = import.meta.env.VITE_WS_URL || 'ws://localhost:8001';

export const usePipelineMonitoring = () => {
  const [pipelineExecutions, setPipelineExecutions] = useState<PipelineExecution[]>([]);
  const [pipelineAlerts, setPipelineAlerts] = useState<PipelineAlert[]>([]);
  const [pipelineMetrics, setPipelineMetrics] = useState<Record<string, PipelineMetrics>>({});
  const [monitoringConfigs, setMonitoringConfigs] = useState<Record<string, MonitoringConfig>>({});
  const [activeExecutions, setActiveExecutions] = useState<Set<string>>(new Set());
  const [realtimeUpdates, setRealtimeUpdates] = useState<RealTimeUpdate[]>([]);
  const [connectionStatus, setConnectionStatus] = useState<'connected' | 'disconnected' | 'reconnecting'>('disconnected');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const wsRef = useRef<WebSocket | null>(null);
  const reconnectTimeoutRef = useRef<NodeJS.Timeout | null>(null);
  const heartbeatIntervalRef = useRef<NodeJS.Timeout | null>(null);

  // Establish WebSocket connection
  const connectWebSocket = useCallback(() => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      return;
    }

    try {
      wsRef.current = new WebSocket(`${WS_URL}/ws/pipeline-monitoring`);

      wsRef.current.onopen = () => {
        console.log('Pipeline monitoring WebSocket connected');
        setConnectionStatus('connected');
        setError(null);

        // Start heartbeat
        heartbeatIntervalRef.current = setInterval(() => {
          if (wsRef.current?.readyState === WebSocket.OPEN) {
            wsRef.current.send(JSON.stringify({ type: 'ping' }));
          }
        }, 30000);
      };

      wsRef.current.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);
          handleRealtimeUpdate(data);
        } catch (err) {
          console.error('Failed to parse WebSocket message:', err);
        }
      };

      wsRef.current.onclose = () => {
        console.log('Pipeline monitoring WebSocket disconnected');
        setConnectionStatus('disconnected');
        
        if (heartbeatIntervalRef.current) {
          clearInterval(heartbeatIntervalRef.current);
          heartbeatIntervalRef.current = null;
        }

        // Attempt to reconnect after 5 seconds
        reconnectTimeoutRef.current = setTimeout(() => {
          setConnectionStatus('reconnecting');
          connectWebSocket();
        }, 5000);
      };

      wsRef.current.onerror = (error) => {
        console.error('Pipeline monitoring WebSocket error:', error);
        setError('WebSocket connection error');
      };
    } catch (err) {
      console.error('Failed to establish WebSocket connection:', err);
      setError('Failed to establish WebSocket connection');
    }
  }, []);

  // Handle real-time updates
  const handleRealtimeUpdate = useCallback((update: RealTimeUpdate) => {
    setRealtimeUpdates(prev => [update, ...prev.slice(0, 99)]); // Keep last 100 updates

    switch (update.type) {
      case 'execution_update':
        setPipelineExecutions(prev => 
          prev.map(exec => 
            exec.executionId === update.executionId
              ? { ...exec, ...update.data, lastUpdate: new Date() }
              : exec
          )
        );
        break;

      case 'stage_update':
        setPipelineExecutions(prev => 
          prev.map(exec => 
            exec.executionId === update.executionId
              ? {
                  ...exec,
                  stages: exec.stages.map(stage =>
                    stage.stageId === update.data.stageId
                      ? { ...stage, ...update.data }
                      : stage
                  )
                }
              : exec
          )
        );
        break;

      case 'alert':
        const alert: PipelineAlert = {
          ...update.data,
          timestamp: new Date(update.data.timestamp)
        };
        setPipelineAlerts(prev => [alert, ...prev]);
        
        // Show notification for high/critical alerts
        if (alert.severity === 'high' || alert.severity === 'critical') {
          message.error(`Pipeline Alert: ${alert.title}`);
        } else if (alert.severity === 'medium') {
          message.warning(`Pipeline Warning: ${alert.title}`);
        }
        break;

      case 'metric_update':
        setPipelineMetrics(prev => ({
          ...prev,
          [update.pipelineId]: {
            ...prev[update.pipelineId],
            ...update.data
          }
        }));
        break;
    }
  }, []);

  // Fetch pipeline executions
  const fetchPipelineExecutions = useCallback(async (filters?: {
    pipelineId?: string;
    strategyId?: string;
    environment?: string;
    status?: string;
    limit?: number;
  }) => {
    try {
      setLoading(true);
      const params = new URLSearchParams();
      if (filters?.pipelineId) params.append('pipeline_id', filters.pipelineId);
      if (filters?.strategyId) params.append('strategy_id', filters.strategyId);
      if (filters?.environment) params.append('environment', filters.environment);
      if (filters?.status) params.append('status', filters.status);
      if (filters?.limit) params.append('limit', filters.limit.toString());

      const response = await fetch(`${API_BASE_URL}/api/v1/strategy/pipeline/executions?${params}`);
      const data = await response.json();
      
      setPipelineExecutions(data.map((execution: any) => ({
        ...execution,
        startedAt: new Date(execution.startedAt),
        completedAt: execution.completedAt ? new Date(execution.completedAt) : undefined,
        stages: execution.stages.map((stage: any) => ({
          ...stage,
          startedAt: stage.startedAt ? new Date(stage.startedAt) : undefined,
          completedAt: stage.completedAt ? new Date(stage.completedAt) : undefined
        }))
      })));

      // Track active executions
      const activeIds = data
        .filter((exec: any) => ['pending', 'running', 'paused'].includes(exec.status))
        .map((exec: any) => exec.executionId);
      setActiveExecutions(new Set(activeIds));
    } catch (err) {
      console.error('Failed to fetch pipeline executions:', err);
      setError('Failed to fetch pipeline executions');
    } finally {
      setLoading(false);
    }
  }, []);

  // Fetch pipeline alerts
  const fetchPipelineAlerts = useCallback(async (filters?: {
    pipelineId?: string;
    severity?: string;
    acknowledged?: boolean;
    limit?: number;
  }) => {
    try {
      const params = new URLSearchParams();
      if (filters?.pipelineId) params.append('pipeline_id', filters.pipelineId);
      if (filters?.severity) params.append('severity', filters.severity);
      if (filters?.acknowledged !== undefined) params.append('acknowledged', filters.acknowledged.toString());
      if (filters?.limit) params.append('limit', filters.limit.toString());

      const response = await fetch(`${API_BASE_URL}/api/v1/strategy/pipeline/alerts?${params}`);
      const data = await response.json();
      
      setPipelineAlerts(data.map((alert: any) => ({
        ...alert,
        timestamp: new Date(alert.timestamp),
        acknowledgedAt: alert.acknowledgedAt ? new Date(alert.acknowledgedAt) : undefined
      })));
    } catch (err) {
      console.error('Failed to fetch pipeline alerts:', err);
      setError('Failed to fetch pipeline alerts');
    }
  }, []);

  // Fetch pipeline metrics
  const fetchPipelineMetrics = useCallback(async (pipelineId?: string) => {
    try {
      const params = new URLSearchParams();
      if (pipelineId) params.append('pipeline_id', pipelineId);

      const response = await fetch(`${API_BASE_URL}/api/v1/strategy/pipeline/metrics?${params}`);
      const data = await response.json();
      
      const metricsMap: Record<string, PipelineMetrics> = {};
      data.forEach((metrics: any) => {
        metricsMap[metrics.pipelineId] = {
          ...metrics,
          lastExecution: metrics.lastExecution ? new Date(metrics.lastExecution) : undefined,
          performanceTrend: metrics.performanceTrend.map((point: any) => ({
            ...point,
            timestamp: new Date(point.timestamp)
          }))
        };
      });
      
      setPipelineMetrics(metricsMap);
    } catch (err) {
      console.error('Failed to fetch pipeline metrics:', err);
      setError('Failed to fetch pipeline metrics');
    }
  }, []);

  // Fetch monitoring configurations
  const fetchMonitoringConfigs = useCallback(async (pipelineId?: string) => {
    try {
      const params = new URLSearchParams();
      if (pipelineId) params.append('pipeline_id', pipelineId);

      const response = await fetch(`${API_BASE_URL}/api/v1/strategy/pipeline/monitoring-config?${params}`);
      const data = await response.json();
      
      const configMap: Record<string, MonitoringConfig> = {};
      data.forEach((config: MonitoringConfig) => {
        configMap[config.pipelineId] = config;
      });
      
      setMonitoringConfigs(configMap);
    } catch (err) {
      console.error('Failed to fetch monitoring configurations:', err);
      setError('Failed to fetch monitoring configurations');
    }
  }, []);

  // Subscribe to pipeline monitoring
  const subscribeToPipeline = useCallback((pipelineId: string) => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify({
        type: 'subscribe',
        pipelineId
      }));
    }
  }, []);

  // Unsubscribe from pipeline monitoring
  const unsubscribeFromPipeline = useCallback((pipelineId: string) => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify({
        type: 'unsubscribe',
        pipelineId
      }));
    }
  }, []);

  // Acknowledge alert
  const acknowledgeAlert = useCallback(async (
    alertId: string,
    acknowledgedBy: string
  ): Promise<boolean> => {
    try {
      const response = await fetch(`${API_BASE_URL}/api/v1/strategy/pipeline/alerts/${alertId}/acknowledge`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ acknowledged_by: acknowledgedBy })
      });

      if (!response.ok) {
        throw new Error(`Failed to acknowledge alert: ${response.statusText}`);
      }

      setPipelineAlerts(prev => 
        prev.map(alert => 
          alert.alertId === alertId
            ? { 
                ...alert, 
                acknowledged: true, 
                acknowledgedBy, 
                acknowledgedAt: new Date() 
              }
            : alert
        )
      );

      message.success('Alert acknowledged successfully');
      return true;
    } catch (err) {
      console.error('Failed to acknowledge alert:', err);
      message.error(`Failed to acknowledge alert: ${err}`);
      setError(`Failed to acknowledge alert: ${err}`);
      return false;
    }
  }, []);

  // Pause pipeline execution
  const pauseExecution = useCallback(async (executionId: string): Promise<boolean> => {
    try {
      const response = await fetch(`${API_BASE_URL}/api/v1/strategy/pipeline/executions/${executionId}/pause`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' }
      });

      if (!response.ok) {
        throw new Error(`Failed to pause execution: ${response.statusText}`);
      }

      message.success('Pipeline execution paused');
      await fetchPipelineExecutions();
      return true;
    } catch (err) {
      console.error('Failed to pause execution:', err);
      message.error(`Failed to pause execution: ${err}`);
      setError(`Failed to pause execution: ${err}`);
      return false;
    }
  }, [fetchPipelineExecutions]);

  // Resume pipeline execution
  const resumeExecution = useCallback(async (executionId: string): Promise<boolean> => {
    try {
      const response = await fetch(`${API_BASE_URL}/api/v1/strategy/pipeline/executions/${executionId}/resume`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' }
      });

      if (!response.ok) {
        throw new Error(`Failed to resume execution: ${response.statusText}`);
      }

      message.success('Pipeline execution resumed');
      await fetchPipelineExecutions();
      return true;
    } catch (err) {
      console.error('Failed to resume execution:', err);
      message.error(`Failed to resume execution: ${err}`);
      setError(`Failed to resume execution: ${err}`);
      return false;
    }
  }, [fetchPipelineExecutions]);

  // Cancel pipeline execution
  const cancelExecution = useCallback(async (executionId: string): Promise<boolean> => {
    try {
      const response = await fetch(`${API_BASE_URL}/api/v1/strategy/pipeline/executions/${executionId}/cancel`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' }
      });

      if (!response.ok) {
        throw new Error(`Failed to cancel execution: ${response.statusText}`);
      }

      message.success('Pipeline execution cancelled');
      await fetchPipelineExecutions();
      return true;
    } catch (err) {
      console.error('Failed to cancel execution:', err);
      message.error(`Failed to cancel execution: ${err}`);
      setError(`Failed to cancel execution: ${err}`);
      return false;
    }
  }, [fetchPipelineExecutions]);

  // Retry failed stage
  const retryStage = useCallback(async (
    executionId: string,
    stageId: string
  ): Promise<boolean> => {
    try {
      const response = await fetch(`${API_BASE_URL}/api/v1/strategy/pipeline/executions/${executionId}/stages/${stageId}/retry`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' }
      });

      if (!response.ok) {
        throw new Error(`Failed to retry stage: ${response.statusText}`);
      }

      message.success('Stage retry initiated');
      await fetchPipelineExecutions();
      return true;
    } catch (err) {
      console.error('Failed to retry stage:', err);
      message.error(`Failed to retry stage: ${err}`);
      setError(`Failed to retry stage: ${err}`);
      return false;
    }
  }, [fetchPipelineExecutions]);

  // Get execution by ID
  const getExecution = useCallback((executionId: string): PipelineExecution | null => {
    return pipelineExecutions.find(exec => exec.executionId === executionId) || null;
  }, [pipelineExecutions]);

  // Get pipeline metrics by ID
  const getMetrics = useCallback((pipelineId: string): PipelineMetrics | null => {
    return pipelineMetrics[pipelineId] || null;
  }, [pipelineMetrics]);

  // Get unacknowledged alerts count
  const getUnacknowledgedAlertsCount = useCallback((pipelineId?: string): number => {
    return pipelineAlerts.filter(alert => 
      !alert.acknowledged && 
      (pipelineId ? alert.pipelineId === pipelineId : true)
    ).length;
  }, [pipelineAlerts]);

  // Get active executions count
  const getActiveExecutionsCount = useCallback((pipelineId?: string): number => {
    return pipelineExecutions.filter(exec => 
      activeExecutions.has(exec.executionId) &&
      (pipelineId ? exec.pipelineId === pipelineId : true)
    ).length;
  }, [pipelineExecutions, activeExecutions]);

  // Initialize and cleanup
  useEffect(() => {
    connectWebSocket();
    fetchPipelineExecutions();
    fetchPipelineAlerts();
    fetchPipelineMetrics();
    fetchMonitoringConfigs();

    return () => {
      if (wsRef.current) {
        wsRef.current.close();
      }
      if (reconnectTimeoutRef.current) {
        clearTimeout(reconnectTimeoutRef.current);
      }
      if (heartbeatIntervalRef.current) {
        clearInterval(heartbeatIntervalRef.current);
      }
    };
  }, [
    connectWebSocket,
    fetchPipelineExecutions,
    fetchPipelineAlerts,
    fetchPipelineMetrics,
    fetchMonitoringConfigs
  ]);

  return {
    // State
    pipelineExecutions,
    pipelineAlerts,
    pipelineMetrics,
    monitoringConfigs,
    activeExecutions,
    realtimeUpdates,
    connectionStatus,
    loading,
    error,

    // Actions
    subscribeToPipeline,
    unsubscribeFromPipeline,
    acknowledgeAlert,
    pauseExecution,
    resumeExecution,
    cancelExecution,
    retryStage,

    // Queries
    getExecution,
    getMetrics,
    getUnacknowledgedAlertsCount,
    getActiveExecutionsCount,

    // Data fetching
    fetchPipelineExecutions,
    fetchPipelineAlerts,
    fetchPipelineMetrics,
    fetchMonitoringConfigs,

    // WebSocket management
    connectWebSocket
  };
};