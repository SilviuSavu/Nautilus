/**
 * React hook for position and account data management
 */

import { useState, useEffect, useCallback } from 'react';
import { 
  Position, 
  AccountBalance, 
  PositionSummary, 
  PositionAlert, 
  PositionState 
} from '../types/position';
import { positionService } from '../services/positionService';

export const usePositionData = () => {
  const [state, setState] = useState<PositionState>({
    positions: [],
    accountBalances: [],
    positionSummary: null,
    alerts: [],
    isLoading: true,
    error: null,
    lastUpdate: 0
  });

  /**
   * Update positions in state
   */
  const handlePositionsUpdate = useCallback((positions: Position[]) => {
    setState(prevState => ({
      ...prevState,
      positions,
      lastUpdate: Date.now(),
      isLoading: false
    }));
  }, []);

  /**
   * Update account balances in state
   */
  const handleAccountsUpdate = useCallback((accountBalances: AccountBalance[]) => {
    setState(prevState => ({
      ...prevState,
      accountBalances,
      lastUpdate: Date.now(),
      isLoading: false
    }));
  }, []);

  /**
   * Update position summary in state
   */
  const handleSummaryUpdate = useCallback((positionSummary: PositionSummary) => {
    setState(prevState => ({
      ...prevState,
      positionSummary,
      lastUpdate: Date.now()
    }));
  }, []);

  /**
   * Update alerts in state
   */
  const handleAlertsUpdate = useCallback((alerts: PositionAlert[]) => {
    setState(prevState => ({
      ...prevState,
      alerts,
      lastUpdate: Date.now()
    }));
  }, []);

  /**
   * Initialize position data and set up event handlers
   */
  useEffect(() => {
    // Get initial data
    const positions = positionService.getPositions();
    const accountBalances = positionService.getAccountBalances();
    const positionSummary = positionService.getPositionSummary();
    const alerts = positionService.getAlerts();

    setState({
      positions,
      accountBalances,
      positionSummary,
      alerts,
      isLoading: positions.length === 0 && accountBalances.length === 0,
      error: null,
      lastUpdate: Date.now()
    });

    // Set up event handlers
    positionService.addPositionHandler(handlePositionsUpdate);
    positionService.addAccountHandler(handleAccountsUpdate);
    positionService.addSummaryHandler(handleSummaryUpdate);
    positionService.addAlertHandler(handleAlertsUpdate);

    // Cleanup function
    return () => {
      positionService.removePositionHandler(handlePositionsUpdate);
      positionService.removeAccountHandler(handleAccountsUpdate);
      positionService.removeSummaryHandler(handleSummaryUpdate);
      positionService.removeAlertHandler(handleAlertsUpdate);
    };
  }, [handlePositionsUpdate, handleAccountsUpdate, handleSummaryUpdate, handleAlertsUpdate]);

  /**
   * Acknowledge an alert
   */
  const acknowledgeAlert = useCallback((alertId: string) => {
    positionService.acknowledgeAlert(alertId);
  }, []);

  /**
   * Get position by ID
   */
  const getPositionById = useCallback((positionId: string): Position | undefined => {
    return state.positions.find(position => position.id === positionId);
  }, [state.positions]);

  /**
   * Get positions by symbol
   */
  const getPositionsBySymbol = useCallback((symbol: string): Position[] => {
    return state.positions.filter(position => position.symbol === symbol);
  }, [state.positions]);

  /**
   * Get account balance by currency
   */
  const getAccountByCurrency = useCallback((currency: string): AccountBalance | undefined => {
    return state.accountBalances.find(account => account.currency === currency);
  }, [state.accountBalances]);

  /**
   * Get total portfolio value in base currency
   */
  const getTotalPortfolioValue = useCallback((baseCurrency: string = 'USD'): number => {
    if (!state.positionSummary) return 0;
    
    // Convert to requested base currency if different from summary currency
    if (state.positionSummary.currency !== baseCurrency) {
      return positionService.convertCurrency(
        state.positionSummary.totalExposure,
        state.positionSummary.currency,
        baseCurrency
      );
    }
    
    return state.positionSummary.totalExposure;
  }, [state.positionSummary]);

  /**
   * Get filtered alerts by severity
   */
  const getAlertsBySeverity = useCallback((severity: 'info' | 'warning' | 'critical'): PositionAlert[] => {
    return state.alerts.filter(alert => alert.severity === severity);
  }, [state.alerts]);

  /**
   * Get unacknowledged alerts count
   */
  const getUnacknowledgedAlertsCount = useCallback((): number => {
    return state.alerts.filter(alert => !alert.acknowledged).length;
  }, [state.alerts]);

  /**
   * Refresh data manually
   */
  const refreshData = useCallback(() => {
    setState(prevState => ({ ...prevState, isLoading: true }));
    
    // Get fresh data from service
    const positions = positionService.getPositions();
    const accountBalances = positionService.getAccountBalances();
    const positionSummary = positionService.getPositionSummary();
    const alerts = positionService.getAlerts();

    setState({
      positions,
      accountBalances,
      positionSummary,
      alerts,
      isLoading: false,
      error: null,
      lastUpdate: Date.now()
    });
  }, []);

  return {
    // State
    ...state,
    
    // Actions
    acknowledgeAlert,
    refreshData,
    
    // Computed values / getters
    getPositionById,
    getPositionsBySymbol,
    getAccountByCurrency,
    getTotalPortfolioValue,
    getAlertsBySeverity,
    getUnacknowledgedAlertsCount,
    
    // Convenience flags
    hasPositions: state.positions.length > 0,
    hasAccounts: state.accountBalances.length > 0,
    hasAlerts: state.alerts.length > 0,
    hasUnacknowledgedAlerts: state.alerts.some(alert => !alert.acknowledged)
  };
};