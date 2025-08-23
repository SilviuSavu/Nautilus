import { useState, useEffect, useCallback, useRef } from 'react';
import { riskService } from '../../components/Risk/services/riskService';
import { RiskLimit, RiskLimitConfiguration, EscalationRule } from '../../components/Risk/types/riskTypes';

export interface DynamicLimitState {
  limits: RiskLimit[];
  configuration: RiskLimitConfiguration | null;
  loading: {
    limits: boolean;
    configuration: boolean;
    updating: boolean;
    deleting: Set<string>;
  };
  error: string | null;
  lastUpdated: Date | null;
  activeBreaches: number;
}

export interface UseDynamicLimitsProps {
  portfolioId: string;
  autoRefresh?: boolean;
  refreshInterval?: number; // milliseconds
}

export interface CreateLimitParams {
  name: string;
  limit_type: RiskLimit['limit_type'];
  threshold_value: string;
  warning_threshold: string;
  action: RiskLimit['action'];
  active?: boolean;
}

export interface UpdateLimitParams {
  threshold_value?: string;
  warning_threshold?: string;
  action?: RiskLimit['action'];
  active?: boolean;
}

export const useDynamicLimits = ({
  portfolioId,
  autoRefresh = true,
  refreshInterval = 5000
}: UseDynamicLimitsProps) => {
  const [state, setState] = useState<DynamicLimitState>({
    limits: [],
    configuration: null,
    loading: {
      limits: true,
      configuration: true,
      updating: false,
      deleting: new Set()
    },
    error: null,
    lastUpdated: null,
    activeBreaches: 0
  });

  const refreshTimeoutRef = useRef<NodeJS.Timeout>();
  const abortControllerRef = useRef<AbortController>();

  const updateState = useCallback((updates: Partial<DynamicLimitState>) => {
    setState(prev => ({
      ...prev,
      ...updates,
      loading: { ...prev.loading, ...updates.loading }
    }));
  }, []);

  const fetchLimits = useCallback(async (silent = false) => {
    if (!silent) {
      updateState({ loading: { ...state.loading, limits: true } });
    }

    try {
      abortControllerRef.current?.abort();
      abortControllerRef.current = new AbortController();

      const limits = await riskService.getRiskLimits(portfolioId);
      const activeBreaches = limits.filter(limit => limit.breach_count > 0).length;

      updateState({
        limits,
        activeBreaches,
        error: null,
        lastUpdated: new Date(),
        loading: { ...state.loading, limits: false }
      });

      return limits;
    } catch (error) {
      if (error instanceof Error && error.name !== 'AbortError') {
        updateState({
          error: error.message,
          loading: { ...state.loading, limits: false }
        });
      }
      throw error;
    }
  }, [portfolioId, updateState, state.loading]);

  const fetchConfiguration = useCallback(async (silent = false) => {
    if (!silent) {
      updateState({ loading: { ...state.loading, configuration: true } });
    }

    try {
      const config = await riskService.getRiskConfiguration(portfolioId);
      // Convert to RiskLimitConfiguration format
      const configuration: RiskLimitConfiguration = {
        portfolio_id: portfolioId,
        limits: state.limits,
        notification_settings: {
          email_alerts: true,
          dashboard_alerts: true
        },
        escalation_rules: []
      };

      updateState({
        configuration,
        error: null,
        loading: { ...state.loading, configuration: false }
      });

      return configuration;
    } catch (error) {
      if (error instanceof Error) {
        updateState({
          error: error.message,
          loading: { ...state.loading, configuration: false }
        });
      }
      throw error;
    }
  }, [portfolioId, updateState, state.loading, state.limits]);

  const createLimit = useCallback(async (limitParams: CreateLimitParams) => {
    updateState({ loading: { ...state.loading, updating: true } });

    try {
      const newLimit = await riskService.createRiskLimit({
        ...limitParams,
        portfolio_id: portfolioId,
        breach_count: 0,
        active: limitParams.active ?? true
      });

      updateState({
        limits: [...state.limits, newLimit],
        error: null,
        lastUpdated: new Date(),
        loading: { ...state.loading, updating: false }
      });

      return newLimit;
    } catch (error) {
      if (error instanceof Error) {
        updateState({
          error: error.message,
          loading: { ...state.loading, updating: false }
        });
      }
      throw error;
    }
  }, [portfolioId, state.limits, updateState, state.loading]);

  const updateLimit = useCallback(async (limitId: string, updates: UpdateLimitParams) => {
    updateState({ loading: { ...state.loading, updating: true } });

    try {
      const updatedLimit = await riskService.updateRiskLimit(limitId, updates);
      
      updateState({
        limits: state.limits.map(limit => 
          limit.id === limitId ? updatedLimit : limit
        ),
        error: null,
        lastUpdated: new Date(),
        loading: { ...state.loading, updating: false }
      });

      return updatedLimit;
    } catch (error) {
      if (error instanceof Error) {
        updateState({
          error: error.message,
          loading: { ...state.loading, updating: false }
        });
      }
      throw error;
    }
  }, [state.limits, updateState, state.loading]);

  const deleteLimit = useCallback(async (limitId: string) => {
    updateState({ 
      loading: { 
        ...state.loading, 
        deleting: new Set([...state.loading.deleting, limitId])
      }
    });

    try {
      await riskService.deleteRiskLimit(limitId);
      
      updateState({
        limits: state.limits.filter(limit => limit.id !== limitId),
        error: null,
        lastUpdated: new Date(),
        loading: {
          ...state.loading,
          deleting: new Set([...state.loading.deleting].filter(id => id !== limitId))
        }
      });

      return true;
    } catch (error) {
      if (error instanceof Error) {
        updateState({
          error: error.message,
          loading: {
            ...state.loading,
            deleting: new Set([...state.loading.deleting].filter(id => id !== limitId))
          }
        });
      }
      throw error;
    }
  }, [state.limits, state.loading, updateState]);

  const toggleLimit = useCallback(async (limitId: string, active: boolean) => {
    return updateLimit(limitId, { active });
  }, [updateLimit]);

  const configureNotifications = useCallback(async (
    notificationSettings: RiskLimitConfiguration['notification_settings'],
    escalationRules: EscalationRule[]
  ) => {
    updateState({ loading: { ...state.loading, updating: true } });

    try {
      const configuration: RiskLimitConfiguration = {
        portfolio_id: portfolioId,
        limits: state.limits,
        notification_settings: notificationSettings,
        escalation_rules: escalationRules
      };

      await riskService.configureRiskLimits(configuration);

      updateState({
        configuration,
        error: null,
        lastUpdated: new Date(),
        loading: { ...state.loading, updating: false }
      });

      return configuration;
    } catch (error) {
      if (error instanceof Error) {
        updateState({
          error: error.message,
          loading: { ...state.loading, updating: false }
        });
      }
      throw error;
    }
  }, [portfolioId, state.limits, state.loading, updateState]);

  const clearError = useCallback(() => {
    updateState({ error: null });
  }, [updateState]);

  // Auto-refresh logic
  useEffect(() => {
    if (autoRefresh && refreshInterval > 0) {
      const refresh = () => {
        fetchLimits(true).catch(console.error);
      };

      refreshTimeoutRef.current = setTimeout(function tick() {
        refresh();
        refreshTimeoutRef.current = setTimeout(tick, refreshInterval);
      }, refreshInterval);

      return () => {
        if (refreshTimeoutRef.current) {
          clearTimeout(refreshTimeoutRef.current);
        }
      };
    }
  }, [autoRefresh, refreshInterval, fetchLimits]);

  // Initial fetch
  useEffect(() => {
    fetchLimits();
    fetchConfiguration();

    return () => {
      abortControllerRef.current?.abort();
      if (refreshTimeoutRef.current) {
        clearTimeout(refreshTimeoutRef.current);
      }
    };
  }, [portfolioId, fetchLimits, fetchConfiguration]);

  // Computed values
  const activeLimits = state.limits.filter(limit => limit.active);
  const breachedLimits = state.limits.filter(limit => limit.breach_count > 0);
  const warningLimits = state.limits.filter(limit => 
    limit.active && limit.breach_count === 0 && 
    // Would need to implement warning threshold check logic here
    false // Placeholder
  );

  const limitsByType = state.limits.reduce((acc, limit) => {
    if (!acc[limit.limit_type]) {
      acc[limit.limit_type] = [];
    }
    acc[limit.limit_type].push(limit);
    return acc;
  }, {} as Record<RiskLimit['limit_type'], RiskLimit[]>);

  const riskScore = Math.min(100, Math.max(0, 
    (breachedLimits.length / Math.max(activeLimits.length, 1)) * 100
  ));

  return {
    // State
    ...state,
    
    // Computed values
    activeLimits,
    breachedLimits,
    warningLimits,
    limitsByType,
    riskScore,
    
    // Actions
    fetchLimits,
    fetchConfiguration,
    createLimit,
    updateLimit,
    deleteLimit,
    toggleLimit,
    configureNotifications,
    clearError,
    
    // Utilities
    refresh: () => Promise.all([fetchLimits(), fetchConfiguration()]),
    isDeleting: (limitId: string) => state.loading.deleting.has(limitId)
  };
};

export default useDynamicLimits;