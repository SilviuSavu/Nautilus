/**
 * useAdvancedRiskLimits Hook
 * Sprint 3: Advanced Risk Limit Management
 * 
 * Comprehensive risk limit management with dynamic limit adjustment,
 * ML-based prediction, breach prevention, and hierarchical limit structures.
 */

import { useState, useEffect, useCallback, useRef } from 'react';
import { useWebSocketStream } from '../useWebSocketStream';
import type { StreamMessage } from '../useWebSocketStream';

export type LimitType = 
  | 'position_limit'
  | 'notional_limit'
  | 'var_limit'
  | 'drawdown_limit'
  | 'concentration_limit'
  | 'leverage_limit'
  | 'sector_limit'
  | 'country_limit'
  | 'currency_limit'
  | 'volatility_limit'
  | 'correlation_limit'
  | 'liquidity_limit';

export type LimitTimeframe = 'intraday' | 'daily' | 'weekly' | 'monthly' | 'quarterly' | 'annual';

export type LimitStatus = 'active' | 'warning' | 'breach' | 'suspended' | 'expired';

export type ActionType = 
  | 'alert'
  | 'reduce_position'
  | 'close_position'
  | 'block_trading'
  | 'escalate'
  | 'custom';

export interface RiskLimit {
  id: string;
  name: string;
  type: LimitType;
  description: string;
  
  // Limit configuration
  hardLimit: number;
  softLimit: number;
  warningThreshold: number;
  
  // Scope
  scope: {
    portfolioIds?: string[];
    strategyIds?: string[];
    symbols?: string[];
    sectors?: string[];
    assetClasses?: string[];
    geographies?: string[];
  };
  
  // Time configuration
  timeframe: LimitTimeframe;
  effectiveFrom: string;
  effectiveUntil?: string;
  
  // Current state
  currentValue: number;
  utilization: number; // as percentage
  status: LimitStatus;
  
  // Dynamic adjustment
  dynamicAdjustment: {
    enabled: boolean;
    model?: 'volatility' | 'var' | 'regime' | 'ml';
    parameters: Record<string, any>;
    lastAdjustment?: string;
    adjustmentHistory: {
      timestamp: string;
      oldLimit: number;
      newLimit: number;
      reason: string;
    }[];
  };
  
  // Actions
  breachActions: {
    type: ActionType;
    parameters: Record<string, any>;
    priority: number;
  }[];
  
  // Monitoring
  monitoring: {
    checkInterval: number; // milliseconds
    alertContacts: string[];
    escalationDelay: number;
    autoResolve: boolean;
  };
  
  // Metadata
  createdBy: string;
  createdAt: string;
  lastModified: string;
  version: number;
}

export interface LimitBreach {
  id: string;
  limitId: string;
  portfolioId?: string;
  strategyId?: string;
  
  // Breach details
  timestamp: string;
  breachType: 'soft' | 'hard';
  breachValue: number;
  limitValue: number;
  exceedanceAmount: number;
  exceedancePercentage: number;
  
  // Context
  triggeringEvent: {
    type: string;
    data: Record<string, any>;
  };
  
  // Actions taken
  actionsTaken: {
    action: ActionType;
    timestamp: string;
    result: 'success' | 'failed' | 'pending';
    details?: string;
  }[];
  
  // Resolution
  status: 'open' | 'acknowledged' | 'resolved' | 'escalated';
  acknowledgedBy?: string;
  acknowledgedAt?: string;
  resolvedBy?: string;
  resolvedAt?: string;
  resolution?: string;
  
  // Prediction
  predictedBreach?: {
    confidence: number;
    timeToBreach: number;
    contributingFactors: string[];
  };
}

export interface LimitHierarchy {
  id: string;
  name: string;
  level: 'firm' | 'desk' | 'portfolio' | 'strategy';
  parentId?: string;
  childIds: string[];
  limits: string[];
  aggregationRules: {
    type: 'sum' | 'max' | 'weighted_average';
    weights?: Record<string, number>;
  };
}

export interface LimitTemplate {
  id: string;
  name: string;
  description: string;
  category: 'equity' | 'fixed_income' | 'fx' | 'commodities' | 'multi_asset';
  limits: Omit<RiskLimit, 'id' | 'createdAt' | 'lastModified' | 'version' | 'currentValue' | 'utilization' | 'status'>[];
}

export interface UseAdvancedRiskLimitsOptions {
  portfolioId?: string;
  enableRealTime?: boolean;
  enablePrediction?: boolean;
  enableDynamicAdjustment?: boolean;
  checkInterval?: number;
  autoApplyActions?: boolean;
}

export interface UseAdvancedRiskLimitsReturn {
  // Limits data
  limits: RiskLimit[];
  breaches: LimitBreach[];
  hierarchies: LimitHierarchy[];
  templates: LimitTemplate[];
  
  // Status
  isLoading: boolean;
  error: string | null;
  isMonitoring: boolean;
  lastCheck: Date | null;
  
  // Limit management
  createLimit: (limit: Omit<RiskLimit, 'id' | 'createdAt' | 'lastModified' | 'version' | 'currentValue' | 'utilization' | 'status'>) => Promise<string>;
  updateLimit: (limitId: string, updates: Partial<RiskLimit>) => Promise<void>;
  deleteLimit: (limitId: string) => Promise<void>;
  activateLimit: (limitId: string) => Promise<void>;
  suspendLimit: (limitId: string) => Promise<void>;
  
  // Dynamic adjustment
  enableDynamicAdjustment: (limitId: string, model: string, parameters: Record<string, any>) => Promise<void>;
  disableDynamicAdjustment: (limitId: string) => Promise<void>;
  adjustLimit: (limitId: string, newLimit: number, reason: string) => Promise<void>;
  
  // Limit templates
  createTemplate: (template: Omit<LimitTemplate, 'id'>) => Promise<string>;
  applyTemplate: (templateId: string, scope: RiskLimit['scope']) => Promise<string[]>;
  
  // Hierarchy management
  createHierarchy: (hierarchy: Omit<LimitHierarchy, 'id'>) => Promise<string>;
  updateHierarchy: (hierarchyId: string, updates: Partial<LimitHierarchy>) => Promise<void>;
  
  // Breach management
  acknowledgeBreach: (breachId: string, comment?: string) => Promise<void>;
  resolveBreach: (breachId: string, resolution: string) => Promise<void>;
  escalateBreach: (breachId: string, reason: string) => Promise<void>;
  
  // Monitoring and prediction
  startMonitoring: () => void;
  stopMonitoring: () => void;
  checkAllLimits: () => Promise<LimitBreach[]>;
  predictBreaches: (horizonMinutes?: number) => Promise<{
    limitId: string;
    confidence: number;
    timeToBreach: number;
    factors: string[];
  }[]>;
  
  // Analysis
  getLimitUtilization: () => { limitId: string; utilization: number; trend: 'increasing' | 'decreasing' | 'stable' }[];
  getBreachStatistics: () => {
    totalBreaches: number;
    breachesByType: Record<LimitType, number>;
    averageResolutionTime: number;
    topBreachedLimits: { limitId: string; count: number }[];
  };
  
  // Stress testing
  runStressTest: (scenario: { name: string; shocks: Record<string, number> }) => Promise<{
    breachedLimits: string[];
    utilizationChanges: Record<string, number>;
  }>;
  
  // Reporting
  generateReport: (format: 'json' | 'csv' | 'pdf', filters?: {
    limitTypes?: LimitType[];
    timeframe?: { start: string; end: string };
    includeBreaches?: boolean;
  }) => Promise<string | Blob>;
  
  // Control
  refresh: () => Promise<void>;
  reset: () => void;
}

const DEFAULT_OPTIONS: Required<UseAdvancedRiskLimitsOptions> = {
  portfolioId: '',
  enableRealTime: true,
  enablePrediction: false,
  enableDynamicAdjustment: false,
  checkInterval: 5000,
  autoApplyActions: false
};

export function useAdvancedRiskLimits(
  options: UseAdvancedRiskLimitsOptions = {}
): UseAdvancedRiskLimitsReturn {
  const config = { ...DEFAULT_OPTIONS, ...options };
  
  // State
  const [limits, setLimits] = useState<RiskLimit[]>([]);
  const [breaches, setBreaches] = useState<LimitBreach[]>([]);
  const [hierarchies, setHierarchies] = useState<LimitHierarchy[]>([]);
  const [templates, setTemplates] = useState<LimitTemplate[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [isMonitoring, setIsMonitoring] = useState(false);
  const [lastCheck, setLastCheck] = useState<Date | null>(null);
  
  // Refs
  const monitoringIntervalRef = useRef<NodeJS.Timeout>();
  const limitValuesRef = useRef<Record<string, number[]>>({}); // For trend analysis
  const isMountedRef = useRef(true);
  
  // API base URL
  const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8001';
  
  // WebSocket stream for real-time limit updates
  const {
    isActive: isRealTimeActive,
    latestMessage,
    startStream,
    stopStream,
    error: streamError
  } = useWebSocketStream({
    streamId: 'risk_limits',
    messageType: 'risk_alert',
    bufferSize: 500,
    autoSubscribe: config.enableRealTime,
    filters: config.portfolioId ? { portfolio_ids: [config.portfolioId] } : undefined
  });
  
  // Cleanup on unmount
  useEffect(() => {
    return () => {
      isMountedRef.current = false;
      if (monitoringIntervalRef.current) {
        clearInterval(monitoringIntervalRef.current);
      }
    };
  }, []);
  
  // Process real-time limit updates
  useEffect(() => {
    if (latestMessage && latestMessage.data) {
      const alertData = latestMessage.data;
      
      // Update limit current value if it's a limit update
      if (alertData.type === 'limit_update' && alertData.limit_id) {
        setLimits(prev => prev.map(limit => 
          limit.id === alertData.limit_id 
            ? {
                ...limit,
                currentValue: alertData.current_value || limit.currentValue,
                utilization: calculateUtilization(alertData.current_value || limit.currentValue, limit.hardLimit),
                status: determineStatus(alertData.current_value || limit.currentValue, limit)
              }
            : limit
        ));
      }
      
      // Handle breach alerts
      if (alertData.type === 'limit_breach' && alertData.limit_id) {
        const breach: LimitBreach = {
          id: `breach_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
          limitId: alertData.limit_id,
          portfolioId: alertData.portfolio_id,
          strategyId: alertData.strategy_id,
          timestamp: alertData.timestamp || new Date().toISOString(),
          breachType: alertData.breach_type || 'soft',
          breachValue: alertData.current_value || 0,
          limitValue: alertData.limit_value || 0,
          exceedanceAmount: alertData.exceedance_amount || 0,
          exceedancePercentage: alertData.exceedance_percentage || 0,
          triggeringEvent: {
            type: alertData.triggering_event?.type || 'unknown',
            data: alertData.triggering_event?.data || {}
          },
          actionsTaken: [],
          status: 'open'
        };
        
        setBreaches(prev => [breach, ...prev].slice(0, 1000)); // Keep last 1000 breaches
        
        // Auto-apply actions if enabled
        if (config.autoApplyActions) {
          applyBreachActions(breach);
        }
      }
    }
  }, [latestMessage, config.autoApplyActions]);
  
  // Utility functions
  const calculateUtilization = useCallback((currentValue: number, limit: number): number => {
    return limit > 0 ? Math.abs(currentValue) / limit * 100 : 0;
  }, []);
  
  const determineStatus = useCallback((currentValue: number, limit: RiskLimit): LimitStatus => {
    const utilization = calculateUtilization(currentValue, limit.hardLimit);
    
    if (utilization >= 100) return 'breach';
    if (utilization >= (limit.warningThreshold || 80)) return 'warning';
    return 'active';
  }, [calculateUtilization]);
  
  // Generate unique ID
  const generateId = useCallback((prefix: string) => {
    return `${prefix}_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
  }, []);
  
  // Apply breach actions
  const applyBreachActions = useCallback(async (breach: LimitBreach) => {
    const limit = limits.find(l => l.id === breach.limitId);
    if (!limit) return;
    
    const actionResults: LimitBreach['actionsTaken'] = [];
    
    for (const action of limit.breachActions.sort((a, b) => a.priority - b.priority)) {
      try {
        const result = await executeAction(action, breach, limit);
        actionResults.push({
          action: action.type,
          timestamp: new Date().toISOString(),
          result: result ? 'success' : 'failed',
          details: result ? 'Action executed successfully' : 'Action failed'
        });
      } catch (error) {
        actionResults.push({
          action: action.type,
          timestamp: new Date().toISOString(),
          result: 'failed',
          details: error instanceof Error ? error.message : 'Unknown error'
        });
      }
    }
    
    // Update breach with actions taken
    setBreaches(prev => prev.map(b => 
      b.id === breach.id ? { ...b, actionsTaken: actionResults } : b
    ));
  }, [limits]);
  
  // Execute individual action
  const executeAction = useCallback(async (
    action: RiskLimit['breachActions'][0], 
    breach: LimitBreach, 
    limit: RiskLimit
  ): Promise<boolean> => {
    try {
      const response = await fetch(`${API_BASE_URL}/api/v1/risk/limits/${limit.id}/actions`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          action_type: action.type,
          parameters: action.parameters,
          breach_id: breach.id
        })
      });
      
      return response.ok;
    } catch (error) {
      console.error('Failed to execute action:', error);
      return false;
    }
  }, [API_BASE_URL]);
  
  // Limit management functions
  const createLimit = useCallback(async (limitData: Omit<RiskLimit, 'id' | 'createdAt' | 'lastModified' | 'version' | 'currentValue' | 'utilization' | 'status'>): Promise<string> => {
    const id = generateId('limit');
    const now = new Date().toISOString();
    
    const newLimit: RiskLimit = {
      ...limitData,
      id,
      currentValue: 0,
      utilization: 0,
      status: 'active',
      createdAt: now,
      lastModified: now,
      version: 1
    };
    
    try {
      const response = await fetch(`${API_BASE_URL}/api/v1/risk/limits`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(newLimit)
      });
      
      if (!response.ok) {
        throw new Error(`Failed to create limit: ${response.statusText}`);
      }
      
      setLimits(prev => [...prev, newLimit]);
      return id;
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to create limit');
      throw err;
    }
  }, [generateId, API_BASE_URL]);
  
  const updateLimit = useCallback(async (limitId: string, updates: Partial<RiskLimit>): Promise<void> => {
    try {
      const response = await fetch(`${API_BASE_URL}/api/v1/risk/limits/${limitId}`, {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          ...updates,
          lastModified: new Date().toISOString(),
          version: (limits.find(l => l.id === limitId)?.version || 0) + 1
        })
      });
      
      if (!response.ok) {
        throw new Error(`Failed to update limit: ${response.statusText}`);
      }
      
      setLimits(prev => prev.map(limit => 
        limit.id === limitId 
          ? { ...limit, ...updates, lastModified: new Date().toISOString(), version: limit.version + 1 }
          : limit
      ));
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to update limit');
      throw err;
    }
  }, [API_BASE_URL, limits]);
  
  const deleteLimit = useCallback(async (limitId: string): Promise<void> => {
    try {
      const response = await fetch(`${API_BASE_URL}/api/v1/risk/limits/${limitId}`, {
        method: 'DELETE'
      });
      
      if (!response.ok) {
        throw new Error(`Failed to delete limit: ${response.statusText}`);
      }
      
      setLimits(prev => prev.filter(limit => limit.id !== limitId));
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to delete limit');
      throw err;
    }
  }, [API_BASE_URL]);
  
  const activateLimit = useCallback(async (limitId: string): Promise<void> => {
    await updateLimit(limitId, { status: 'active' });
  }, [updateLimit]);
  
  const suspendLimit = useCallback(async (limitId: string): Promise<void> => {
    await updateLimit(limitId, { status: 'suspended' });
  }, [updateLimit]);
  
  // Dynamic adjustment functions
  const enableDynamicAdjustment = useCallback(async (
    limitId: string, 
    model: string, 
    parameters: Record<string, any>
  ): Promise<void> => {
    await updateLimit(limitId, {
      dynamicAdjustment: {
        enabled: true,
        model: model as any,
        parameters,
        adjustmentHistory: []
      }
    });
  }, [updateLimit]);
  
  const disableDynamicAdjustment = useCallback(async (limitId: string): Promise<void> => {
    const limit = limits.find(l => l.id === limitId);
    if (limit) {
      await updateLimit(limitId, {
        dynamicAdjustment: {
          ...limit.dynamicAdjustment,
          enabled: false
        }
      });
    }
  }, [limits, updateLimit]);
  
  const adjustLimit = useCallback(async (limitId: string, newLimit: number, reason: string): Promise<void> => {
    const limit = limits.find(l => l.id === limitId);
    if (!limit) return;
    
    const adjustment = {
      timestamp: new Date().toISOString(),
      oldLimit: limit.hardLimit,
      newLimit,
      reason
    };
    
    await updateLimit(limitId, {
      hardLimit: newLimit,
      dynamicAdjustment: {
        ...limit.dynamicAdjustment,
        lastAdjustment: adjustment.timestamp,
        adjustmentHistory: [...limit.dynamicAdjustment.adjustmentHistory, adjustment]
      }
    });
  }, [limits, updateLimit]);
  
  // Template functions
  const createTemplate = useCallback(async (template: Omit<LimitTemplate, 'id'>): Promise<string> => {
    const id = generateId('template');
    const newTemplate: LimitTemplate = { ...template, id };
    
    try {
      const response = await fetch(`${API_BASE_URL}/api/v1/risk/templates`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(newTemplate)
      });
      
      if (!response.ok) {
        throw new Error(`Failed to create template: ${response.statusText}`);
      }
      
      setTemplates(prev => [...prev, newTemplate]);
      return id;
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to create template');
      throw err;
    }
  }, [generateId, API_BASE_URL]);
  
  const applyTemplate = useCallback(async (templateId: string, scope: RiskLimit['scope']): Promise<string[]> => {
    const template = templates.find(t => t.id === templateId);
    if (!template) {
      throw new Error('Template not found');
    }
    
    const limitIds: string[] = [];
    
    for (const limitConfig of template.limits) {
      const id = await createLimit({
        ...limitConfig,
        scope,
        createdBy: 'template_application'
      });
      limitIds.push(id);
    }
    
    return limitIds;
  }, [templates, createLimit]);
  
  // Hierarchy functions
  const createHierarchy = useCallback(async (hierarchy: Omit<LimitHierarchy, 'id'>): Promise<string> => {
    const id = generateId('hierarchy');
    const newHierarchy: LimitHierarchy = { ...hierarchy, id };
    
    try {
      const response = await fetch(`${API_BASE_URL}/api/v1/risk/hierarchies`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(newHierarchy)
      });
      
      if (!response.ok) {
        throw new Error(`Failed to create hierarchy: ${response.statusText}`);
      }
      
      setHierarchies(prev => [...prev, newHierarchy]);
      return id;
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to create hierarchy');
      throw err;
    }
  }, [generateId, API_BASE_URL]);
  
  const updateHierarchy = useCallback(async (hierarchyId: string, updates: Partial<LimitHierarchy>): Promise<void> => {
    try {
      const response = await fetch(`${API_BASE_URL}/api/v1/risk/hierarchies/${hierarchyId}`, {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(updates)
      });
      
      if (!response.ok) {
        throw new Error(`Failed to update hierarchy: ${response.statusText}`);
      }
      
      setHierarchies(prev => prev.map(hierarchy => 
        hierarchy.id === hierarchyId ? { ...hierarchy, ...updates } : hierarchy
      ));
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to update hierarchy');
      throw err;
    }
  }, [API_BASE_URL]);
  
  // Breach management functions
  const acknowledgeBreach = useCallback(async (breachId: string, comment?: string): Promise<void> => {
    setBreaches(prev => prev.map(breach => 
      breach.id === breachId 
        ? {
            ...breach,
            status: 'acknowledged',
            acknowledgedBy: 'current_user', // Would get from auth context
            acknowledgedAt: new Date().toISOString(),
            resolution: comment
          }
        : breach
    ));
  }, []);
  
  const resolveBreach = useCallback(async (breachId: string, resolution: string): Promise<void> => {
    setBreaches(prev => prev.map(breach => 
      breach.id === breachId 
        ? {
            ...breach,
            status: 'resolved',
            resolvedBy: 'current_user', // Would get from auth context
            resolvedAt: new Date().toISOString(),
            resolution
          }
        : breach
    ));
  }, []);
  
  const escalateBreach = useCallback(async (breachId: string, reason: string): Promise<void> => {
    setBreaches(prev => prev.map(breach => 
      breach.id === breachId 
        ? {
            ...breach,
            status: 'escalated',
            resolution: reason
          }
        : breach
    ));
    
    // Send escalation notification
    try {
      await fetch(`${API_BASE_URL}/api/v1/risk/breaches/${breachId}/escalate`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ reason })
      });
    } catch (error) {
      console.error('Failed to send escalation:', error);
    }
  }, [API_BASE_URL]);
  
  // Monitoring functions
  const startMonitoring = useCallback(() => {
    if (isMonitoring) return;
    
    setIsMonitoring(true);
    
    if (config.enableRealTime) {
      startStream();
    }
    
    // Start periodic checks
    monitoringIntervalRef.current = setInterval(() => {
      if (isMountedRef.current) {
        checkAllLimits();
      }
    }, config.checkInterval);
  }, [isMonitoring, config.enableRealTime, config.checkInterval, startStream]);
  
  const stopMonitoring = useCallback(() => {
    setIsMonitoring(false);
    
    if (config.enableRealTime) {
      stopStream();
    }
    
    if (monitoringIntervalRef.current) {
      clearInterval(monitoringIntervalRef.current);
    }
  }, [config.enableRealTime, stopStream]);
  
  const checkAllLimits = useCallback(async (): Promise<LimitBreach[]> => {
    try {
      const response = await fetch(`${API_BASE_URL}/api/v1/risk/limits/check`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          portfolio_id: config.portfolioId,
          limit_ids: limits.map(l => l.id)
        })
      });
      
      if (!response.ok) {
        throw new Error(`Limit check failed: ${response.statusText}`);
      }
      
      const result = await response.json();
      setLastCheck(new Date());
      
      // Update limit values and statuses
      if (result.limit_values) {
        setLimits(prev => prev.map(limit => {
          const newValue = result.limit_values[limit.id];
          if (newValue !== undefined) {
            return {
              ...limit,
              currentValue: newValue,
              utilization: calculateUtilization(newValue, limit.hardLimit),
              status: determineStatus(newValue, limit)
            };
          }
          return limit;
        }));
      }
      
      // Process any new breaches
      const newBreaches: LimitBreach[] = result.breaches || [];
      if (newBreaches.length > 0) {
        setBreaches(prev => [...newBreaches, ...prev].slice(0, 1000));
        
        if (config.autoApplyActions) {
          newBreaches.forEach(breach => applyBreachActions(breach));
        }
      }
      
      return newBreaches;
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to check limits');
      return [];
    }
  }, [API_BASE_URL, config.portfolioId, config.autoApplyActions, limits, calculateUtilization, determineStatus, applyBreachActions]);
  
  const predictBreaches = useCallback(async (horizonMinutes = 60) => {
    if (!config.enablePrediction) return [];
    
    try {
      const response = await fetch(`${API_BASE_URL}/api/v1/risk/limits/predict`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          portfolio_id: config.portfolioId,
          horizon_minutes: horizonMinutes,
          limit_ids: limits.map(l => l.id)
        })
      });
      
      if (!response.ok) {
        throw new Error(`Prediction failed: ${response.statusText}`);
      }
      
      const predictions = await response.json();
      return predictions.predictions || [];
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to predict breaches');
      return [];
    }
  }, [API_BASE_URL, config.portfolioId, config.enablePrediction, limits]);
  
  // Analysis functions
  const getLimitUtilization = useCallback(() => {
    return limits.map(limit => {
      const history = limitValuesRef.current[limit.id] || [];
      let trend: 'increasing' | 'decreasing' | 'stable' = 'stable';
      
      if (history.length >= 2) {
        const recent = history.slice(-5);
        const first = recent[0];
        const last = recent[recent.length - 1];
        
        if (last > first * 1.05) trend = 'increasing';
        else if (last < first * 0.95) trend = 'decreasing';
      }
      
      return {
        limitId: limit.id,
        utilization: limit.utilization,
        trend
      };
    });
  }, [limits]);
  
  const getBreachStatistics = useCallback(() => {
    const breachesByType: Record<LimitType, number> = {} as any;
    let totalResolutionTime = 0;
    let resolvedBreaches = 0;
    
    const limitBreachCounts: Record<string, number> = {};
    
    breaches.forEach(breach => {
      const limit = limits.find(l => l.id === breach.limitId);
      if (limit) {
        breachesByType[limit.type] = (breachesByType[limit.type] || 0) + 1;
        limitBreachCounts[breach.limitId] = (limitBreachCounts[breach.limitId] || 0) + 1;
        
        if (breach.status === 'resolved' && breach.resolvedAt) {
          const resolutionTime = new Date(breach.resolvedAt).getTime() - new Date(breach.timestamp).getTime();
          totalResolutionTime += resolutionTime;
          resolvedBreaches++;
        }
      }
    });
    
    const topBreachedLimits = Object.entries(limitBreachCounts)
      .sort(([,a], [,b]) => b - a)
      .slice(0, 5)
      .map(([limitId, count]) => ({ limitId, count }));
    
    return {
      totalBreaches: breaches.length,
      breachesByType,
      averageResolutionTime: resolvedBreaches > 0 ? totalResolutionTime / resolvedBreaches : 0,
      topBreachedLimits
    };
  }, [breaches, limits]);
  
  // Stress testing
  const runStressTest = useCallback(async (scenario: { name: string; shocks: Record<string, number> }) => {
    try {
      const response = await fetch(`${API_BASE_URL}/api/v1/risk/limits/stress-test`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          portfolio_id: config.portfolioId,
          scenario,
          limit_ids: limits.map(l => l.id)
        })
      });
      
      if (!response.ok) {
        throw new Error(`Stress test failed: ${response.statusText}`);
      }
      
      const result = await response.json();
      return {
        breachedLimits: result.breached_limits || [],
        utilizationChanges: result.utilization_changes || {}
      };
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to run stress test');
      return { breachedLimits: [], utilizationChanges: {} };
    }
  }, [API_BASE_URL, config.portfolioId, limits]);
  
  // Reporting
  const generateReport = useCallback(async (
    format: 'json' | 'csv' | 'pdf',
    filters?: {
      limitTypes?: LimitType[];
      timeframe?: { start: string; end: string };
      includeBreaches?: boolean;
    }
  ): Promise<string | Blob> => {
    const data = {
      limits: filters?.limitTypes 
        ? limits.filter(l => filters.limitTypes!.includes(l.type))
        : limits,
      breaches: filters?.includeBreaches ? breaches : [],
      statistics: getBreachStatistics(),
      utilization: getLimitUtilization()
    };
    
    switch (format) {
      case 'json':
        return JSON.stringify(data, null, 2);
      case 'csv':
        const csvRows = [
          ['Limit ID', 'Name', 'Type', 'Hard Limit', 'Current Value', 'Utilization %', 'Status'],
          ...data.limits.map(l => [
            l.id, l.name, l.type, l.hardLimit, l.currentValue, l.utilization.toFixed(2), l.status
          ])
        ];
        return csvRows.map(row => row.join(',')).join('\n');
      case 'pdf':
        return new Blob([JSON.stringify(data)], { type: 'application/json' });
      default:
        return '';
    }
  }, [limits, breaches, getBreachStatistics, getLimitUtilization]);
  
  // Control functions
  const refresh = useCallback(async () => {
    setIsLoading(true);
    try {
      const [limitsResponse, breachesResponse, templatesResponse, hierarchiesResponse] = await Promise.all([
        fetch(`${API_BASE_URL}/api/v1/risk/limits${config.portfolioId ? `?portfolio_id=${config.portfolioId}` : ''}`),
        fetch(`${API_BASE_URL}/api/v1/risk/breaches${config.portfolioId ? `?portfolio_id=${config.portfolioId}` : ''}`),
        fetch(`${API_BASE_URL}/api/v1/risk/templates`),
        fetch(`${API_BASE_URL}/api/v1/risk/hierarchies`)
      ]);
      
      if (limitsResponse.ok) {
        const limitsData = await limitsResponse.json();
        setLimits(limitsData.limits || []);
      }
      
      if (breachesResponse.ok) {
        const breachesData = await breachesResponse.json();
        setBreaches(breachesData.breaches || []);
      }
      
      if (templatesResponse.ok) {
        const templatesData = await templatesResponse.json();
        setTemplates(templatesData.templates || []);
      }
      
      if (hierarchiesResponse.ok) {
        const hierarchiesData = await hierarchiesResponse.json();
        setHierarchies(hierarchiesData.hierarchies || []);
      }
      
      setError(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to refresh data');
    } finally {
      setIsLoading(false);
    }
  }, [API_BASE_URL, config.portfolioId]);
  
  const reset = useCallback(() => {
    setLimits([]);
    setBreaches([]);
    setHierarchies([]);
    setTemplates([]);
    setError(null);
    setLastCheck(null);
    limitValuesRef.current = {};
  }, []);
  
  // Track limit values for trend analysis
  useEffect(() => {
    limits.forEach(limit => {
      if (!limitValuesRef.current[limit.id]) {
        limitValuesRef.current[limit.id] = [];
      }
      limitValuesRef.current[limit.id].push(limit.currentValue);
      
      // Keep only last 100 values
      if (limitValuesRef.current[limit.id].length > 100) {
        limitValuesRef.current[limit.id].shift();
      }
    });
  }, [limits]);
  
  // Initial data load
  useEffect(() => {
    refresh();
  }, [refresh]);
  
  // Auto-start monitoring if enabled
  useEffect(() => {
    if (config.enableRealTime) {
      startMonitoring();
    }
    
    return () => {
      stopMonitoring();
    };
  }, [config.enableRealTime, startMonitoring, stopMonitoring]);
  
  return {
    // Limits data
    limits,
    breaches,
    hierarchies,
    templates,
    
    // Status
    isLoading,
    error: error || streamError,
    isMonitoring,
    lastCheck,
    
    // Limit management
    createLimit,
    updateLimit,
    deleteLimit,
    activateLimit,
    suspendLimit,
    
    // Dynamic adjustment
    enableDynamicAdjustment,
    disableDynamicAdjustment,
    adjustLimit,
    
    // Limit templates
    createTemplate,
    applyTemplate,
    
    // Hierarchy management
    createHierarchy,
    updateHierarchy,
    
    // Breach management
    acknowledgeBreach,
    resolveBreach,
    escalateBreach,
    
    // Monitoring and prediction
    startMonitoring,
    stopMonitoring,
    checkAllLimits,
    predictBreaches,
    
    // Analysis
    getLimitUtilization,
    getBreachStatistics,
    
    // Stress testing
    runStressTest,
    
    // Reporting
    generateReport,
    
    // Control
    refresh,
    reset
  };
}

export default useAdvancedRiskLimits;