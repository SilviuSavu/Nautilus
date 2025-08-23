/**
 * Strategy Deployment Hook
 * Manages complete deployment lifecycle for strategies
 */

import { useState, useCallback, useEffect } from 'react';
import { message } from 'antd';

export interface DeploymentEnvironment {
  id: string;
  name: string;
  type: 'development' | 'testing' | 'staging' | 'production';
  status: 'active' | 'inactive' | 'maintenance';
  requiresApproval: boolean;
  autoPromote: boolean;
}

export interface DeploymentConfig {
  strategy: 'direct' | 'blue_green' | 'canary' | 'rolling';
  autoRollback: boolean;
  rollbackThreshold: number;
  canaryPercentage?: number;
  approvalRequired: boolean;
  notificationChannels: string[];
  resourceLimits: Record<string, any>;
  healthChecks: Record<string, any>;
}

export interface DeploymentRequest {
  requestId: string;
  strategyId: string;
  version: string;
  sourceEnvironment?: string;
  targetEnvironment: string;
  deploymentConfig: DeploymentConfig;
  requestedBy: string;
  requestedAt: Date;
  approvedBy?: string;
  approvedAt?: Date;
  approvalStatus: 'pending' | 'approved' | 'rejected';
}

export interface Deployment {
  deploymentId: string;
  requestId: string;
  strategyId: string;
  version: string;
  environment: string;
  status: 'pending' | 'testing' | 'deploying' | 'deployed' | 'failed' | 'rolled_back' | 'cancelled';
  deploymentConfig: DeploymentConfig;
  nautilusDeploymentId?: string;
  previousVersion?: string;
  canaryDeploymentId?: string;
  deployedBy: string;
  deployedAt: Date;
  completedAt?: Date;
  performanceMetrics: Record<string, any>;
  healthStatus: string;
  errorLog: Array<{
    timestamp: string;
    error: string;
    phase: string;
  }>;
}

export interface DeploymentPipeline {
  pipelineId: string;
  strategyId: string;
  version: string;
  environments: string[];
  currentStage: number;
  deployments: Deployment[];
  status: string;
  startedAt: Date;
  completedAt?: Date;
}

export interface DeploymentStats {
  totalDeployments: number;
  successfulDeployments: number;
  failedDeployments: number;
  successRate: number;
  environmentStats: Record<string, any>;
  pendingRequests: number;
  activePipelines: number;
  timestamp: string;
}

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8001';

export const useStrategyDeployment = () => {
  const [deployments, setDeployments] = useState<Deployment[]>([]);
  const [deploymentRequests, setDeploymentRequests] = useState<DeploymentRequest[]>([]);
  const [deploymentPipelines, setDeploymentPipelines] = useState<DeploymentPipeline[]>([]);
  const [environments, setEnvironments] = useState<DeploymentEnvironment[]>([]);
  const [deploymentStats, setDeploymentStats] = useState<DeploymentStats | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Fetch environments
  const fetchEnvironments = useCallback(async () => {
    try {
      const response = await fetch(`${API_BASE_URL}/api/v1/strategy/deployment/environments`);
      const data = await response.json();
      setEnvironments(data);
    } catch (err) {
      console.error('Failed to fetch environments:', err);
      setError('Failed to fetch deployment environments');
    }
  }, []);

  // Fetch deployments
  const fetchDeployments = useCallback(async (filters?: {
    strategyId?: string;
    environment?: string;
    status?: string;
  }) => {
    try {
      setLoading(true);
      const params = new URLSearchParams();
      if (filters?.strategyId) params.append('strategy_id', filters.strategyId);
      if (filters?.environment) params.append('environment', filters.environment);
      if (filters?.status) params.append('status', filters.status);

      const response = await fetch(`${API_BASE_URL}/api/v1/strategy/deployment/deployments?${params}`);
      const data = await response.json();
      setDeployments(data.map((d: any) => ({
        ...d,
        deployedAt: new Date(d.deployedAt),
        completedAt: d.completedAt ? new Date(d.completedAt) : undefined
      })));
    } catch (err) {
      console.error('Failed to fetch deployments:', err);
      setError('Failed to fetch deployments');
    } finally {
      setLoading(false);
    }
  }, []);

  // Fetch deployment requests
  const fetchDeploymentRequests = useCallback(async (filters?: {
    strategyId?: string;
    status?: string;
  }) => {
    try {
      const params = new URLSearchParams();
      if (filters?.strategyId) params.append('strategy_id', filters.strategyId);
      if (filters?.status) params.append('status', filters.status);

      const response = await fetch(`${API_BASE_URL}/api/v1/strategy/deployment/requests?${params}`);
      const data = await response.json();
      setDeploymentRequests(data.map((r: any) => ({
        ...r,
        requestedAt: new Date(r.requestedAt),
        approvedAt: r.approvedAt ? new Date(r.approvedAt) : undefined
      })));
    } catch (err) {
      console.error('Failed to fetch deployment requests:', err);
      setError('Failed to fetch deployment requests');
    }
  }, []);

  // Fetch deployment pipelines
  const fetchDeploymentPipelines = useCallback(async () => {
    try {
      const response = await fetch(`${API_BASE_URL}/api/v1/strategy/deployment/pipelines`);
      const data = await response.json();
      setDeploymentPipelines(data.map((p: any) => ({
        ...p,
        startedAt: new Date(p.startedAt),
        completedAt: p.completedAt ? new Date(p.completedAt) : undefined,
        deployments: p.deployments.map((d: any) => ({
          ...d,
          deployedAt: new Date(d.deployedAt),
          completedAt: d.completedAt ? new Date(d.completedAt) : undefined
        }))
      })));
    } catch (err) {
      console.error('Failed to fetch deployment pipelines:', err);
      setError('Failed to fetch deployment pipelines');
    }
  }, []);

  // Fetch deployment statistics
  const fetchDeploymentStats = useCallback(async () => {
    try {
      const response = await fetch(`${API_BASE_URL}/api/v1/strategy/deployment/statistics`);
      const data = await response.json();
      setDeploymentStats(data);
    } catch (err) {
      console.error('Failed to fetch deployment statistics:', err);
      setError('Failed to fetch deployment statistics');
    }
  }, []);

  // Create deployment request
  const createDeploymentRequest = useCallback(async (
    strategyId: string,
    version: string,
    targetEnvironment: string,
    deploymentConfig?: Partial<DeploymentConfig>,
    requestedBy: string = 'user'
  ): Promise<DeploymentRequest | null> => {
    try {
      setLoading(true);
      
      const requestData = {
        strategy_id: strategyId,
        version,
        target_environment: targetEnvironment,
        deployment_config: deploymentConfig || {},
        requested_by: requestedBy
      };

      const response = await fetch(`${API_BASE_URL}/api/v1/strategy/deployment/requests`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(requestData)
      });

      if (!response.ok) {
        throw new Error(`Failed to create deployment request: ${response.statusText}`);
      }

      const result = await response.json();
      message.success(`Deployment request created successfully`);
      
      await fetchDeploymentRequests();
      
      return {
        ...result,
        requestedAt: new Date(result.requestedAt),
        approvedAt: result.approvedAt ? new Date(result.approvedAt) : undefined
      };
    } catch (err) {
      console.error('Failed to create deployment request:', err);
      message.error(`Failed to create deployment request: ${err}`);
      setError(`Failed to create deployment request: ${err}`);
      return null;
    } finally {
      setLoading(false);
    }
  }, [fetchDeploymentRequests]);

  // Approve deployment request
  const approveDeploymentRequest = useCallback(async (
    requestId: string,
    approvedBy: string,
    approved: boolean = true
  ): Promise<boolean> => {
    try {
      setLoading(true);

      const response = await fetch(`${API_BASE_URL}/api/v1/strategy/deployment/requests/${requestId}/approve`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          approved_by: approvedBy,
          approved
        })
      });

      if (!response.ok) {
        throw new Error(`Failed to ${approved ? 'approve' : 'reject'} deployment request`);
      }

      message.success(`Deployment request ${approved ? 'approved' : 'rejected'} successfully`);
      
      await fetchDeploymentRequests();
      await fetchDeployments();
      
      return true;
    } catch (err) {
      console.error(`Failed to ${approved ? 'approve' : 'reject'} deployment request:`, err);
      message.error(`Failed to ${approved ? 'approve' : 'reject'} deployment request`);
      setError(`Failed to ${approved ? 'approve' : 'reject'} deployment request: ${err}`);
      return false;
    } finally {
      setLoading(false);
    }
  }, [fetchDeploymentRequests, fetchDeployments]);

  // Deploy strategy directly
  const deployStrategy = useCallback(async (
    strategyId: string,
    version: string,
    targetEnvironment: string,
    deploymentConfig?: Partial<DeploymentConfig>,
    requestedBy: string = 'user'
  ): Promise<Deployment | null> => {
    try {
      setLoading(true);
      
      const deployData = {
        strategy_id: strategyId,
        version,
        target_environment: targetEnvironment,
        deployment_config: deploymentConfig || {},
        requested_by: requestedBy
      };

      const response = await fetch(`${API_BASE_URL}/api/v1/strategy/deployment/deploy`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(deployData)
      });

      if (!response.ok) {
        throw new Error(`Failed to deploy strategy: ${response.statusText}`);
      }

      const result = await response.json();
      message.success(`Strategy deployed successfully`);
      
      await fetchDeployments();
      
      return {
        ...result,
        deployedAt: new Date(result.deployedAt),
        completedAt: result.completedAt ? new Date(result.completedAt) : undefined
      };
    } catch (err) {
      console.error('Failed to deploy strategy:', err);
      message.error(`Failed to deploy strategy: ${err}`);
      setError(`Failed to deploy strategy: ${err}`);
      return null;
    } finally {
      setLoading(false);
    }
  }, [fetchDeployments]);

  // Create deployment pipeline
  const createDeploymentPipeline = useCallback(async (
    strategyId: string,
    version: string,
    environments: string[]
  ): Promise<DeploymentPipeline | null> => {
    try {
      setLoading(true);
      
      const pipelineData = {
        strategy_id: strategyId,
        version,
        environments
      };

      const response = await fetch(`${API_BASE_URL}/api/v1/strategy/deployment/pipelines`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(pipelineData)
      });

      if (!response.ok) {
        throw new Error(`Failed to create deployment pipeline: ${response.statusText}`);
      }

      const result = await response.json();
      message.success(`Deployment pipeline created successfully`);
      
      await fetchDeploymentPipelines();
      
      return {
        ...result,
        startedAt: new Date(result.startedAt),
        completedAt: result.completedAt ? new Date(result.completedAt) : undefined,
        deployments: result.deployments.map((d: any) => ({
          ...d,
          deployedAt: new Date(d.deployedAt),
          completedAt: d.completedAt ? new Date(d.completedAt) : undefined
        }))
      };
    } catch (err) {
      console.error('Failed to create deployment pipeline:', err);
      message.error(`Failed to create deployment pipeline: ${err}`);
      setError(`Failed to create deployment pipeline: ${err}`);
      return null;
    } finally {
      setLoading(false);
    }
  }, [fetchDeploymentPipelines]);

  // Stop deployment
  const stopDeployment = useCallback(async (deploymentId: string): Promise<boolean> => {
    try {
      setLoading(true);

      const response = await fetch(`${API_BASE_URL}/api/v1/strategy/deployment/deployments/${deploymentId}/stop`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' }
      });

      if (!response.ok) {
        throw new Error(`Failed to stop deployment: ${response.statusText}`);
      }

      message.success(`Deployment stopped successfully`);
      await fetchDeployments();
      
      return true;
    } catch (err) {
      console.error('Failed to stop deployment:', err);
      message.error(`Failed to stop deployment: ${err}`);
      setError(`Failed to stop deployment: ${err}`);
      return false;
    } finally {
      setLoading(false);
    }
  }, [fetchDeployments]);

  // Get deployment by ID
  const getDeployment = useCallback((deploymentId: string): Deployment | null => {
    return deployments.find(d => d.deploymentId === deploymentId) || null;
  }, [deployments]);

  // Get deployment request by ID
  const getDeploymentRequest = useCallback((requestId: string): DeploymentRequest | null => {
    return deploymentRequests.find(r => r.requestId === requestId) || null;
  }, [deploymentRequests]);

  // Get deployment pipeline by ID
  const getDeploymentPipeline = useCallback((pipelineId: string): DeploymentPipeline | null => {
    return deploymentPipelines.find(p => p.pipelineId === pipelineId) || null;
  }, [deploymentPipelines]);

  // Initialize data
  useEffect(() => {
    fetchEnvironments();
    fetchDeployments();
    fetchDeploymentRequests();
    fetchDeploymentPipelines();
    fetchDeploymentStats();
  }, [
    fetchEnvironments,
    fetchDeployments,
    fetchDeploymentRequests,
    fetchDeploymentPipelines,
    fetchDeploymentStats
  ]);

  return {
    // State
    deployments,
    deploymentRequests,
    deploymentPipelines,
    environments,
    deploymentStats,
    loading,
    error,

    // Actions
    createDeploymentRequest,
    approveDeploymentRequest,
    deployStrategy,
    createDeploymentPipeline,
    stopDeployment,

    // Queries
    getDeployment,
    getDeploymentRequest,
    getDeploymentPipeline,

    // Refresh functions
    fetchDeployments,
    fetchDeploymentRequests,
    fetchDeploymentPipelines,
    fetchDeploymentStats,
    fetchEnvironments
  };
};