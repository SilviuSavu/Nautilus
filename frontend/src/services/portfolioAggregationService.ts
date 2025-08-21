/**
 * Portfolio Aggregation Service for multi-strategy P&L calculation
 */

import { Position, PnLCalculation } from '../types/position';
import { pnlEngine, PnLBreakdown } from './pnlCalculationEngine';

export interface StrategyPnL {
  strategy_id: string;
  strategy_name: string;
  total_pnl: number;
  unrealized_pnl: number;
  realized_pnl: number;
  positions_count: number;
  weight: number;
  contribution_percent: number;
  last_updated: string;
}

export interface PortfolioAggregation {
  total_pnl: number;
  unrealized_pnl: number;
  realized_pnl: number;
  total_return: number;
  sharpe_ratio: number;
  max_drawdown: number;
  volatility: number;
  beta: number;
  strategies: StrategyPnL[];
  last_updated: string;
}

export interface PortfolioMetrics {
  timeWeightedReturn: number;
  moneyWeightedReturn: number;
  totalExposure: number;
  activeStrategies: number;
  avgDailyReturn: number;
  winRate: number;
  profitFactor: number;
  downvideDeviation: number;
}

export class PortfolioAggregationService {
  private strategiesMap: Map<string, StrategyPnL> = new Map();
  private positionsByStrategy: Map<string, Position[]> = new Map();
  private aggregationHandlers: Set<(aggregation: PortfolioAggregation) => void> = new Set();
  private lastAggregation: PortfolioAggregation | null = null;

  /**
   * Add positions for a specific strategy
   */
  public addStrategyPositions(strategyId: string, strategyName: string, positions: Position[]): void {
    this.positionsByStrategy.set(strategyId, positions);
    
    // Calculate strategy-level P&L
    const strategyPnL = this.calculateStrategyPnL(strategyId, strategyName, positions);
    this.strategiesMap.set(strategyId, strategyPnL);
    
    // Trigger aggregation update
    this.calculatePortfolioAggregation();
  }

  /**
   * Update positions for an existing strategy
   */
  public updateStrategyPositions(strategyId: string, positions: Position[]): void {
    if (!this.positionsByStrategy.has(strategyId)) {
      console.warn(`Strategy ${strategyId} not found for position update`);
      return;
    }

    const existingStrategy = this.strategiesMap.get(strategyId);
    if (existingStrategy) {
      this.addStrategyPositions(strategyId, existingStrategy.strategy_name, positions);
    }
  }

  /**
   * Remove strategy from portfolio
   */
  public removeStrategy(strategyId: string): void {
    this.positionsByStrategy.delete(strategyId);
    this.strategiesMap.delete(strategyId);
    this.calculatePortfolioAggregation();
  }

  /**
   * Calculate P&L for a specific strategy
   */
  private calculateStrategyPnL(strategyId: string, strategyName: string, positions: Position[]): StrategyPnL {
    const portfolioPnL = pnlEngine.calculatePortfolioPnL(positions);
    const positionsCount = positions.length;
    
    // Calculate total portfolio value for weight calculation
    const totalPortfolioValue = this.calculateTotalPortfolioValue();
    const strategyValue = positions.reduce((sum, pos) => 
      sum + Math.abs(pos.currentPrice * pos.quantity), 0);
    
    const weight = totalPortfolioValue > 0 ? (strategyValue / totalPortfolioValue) * 100 : 0;

    return {
      strategy_id: strategyId,
      strategy_name: strategyName,
      total_pnl: portfolioPnL.totalPnl,
      unrealized_pnl: portfolioPnL.unrealizedPnl,
      realized_pnl: portfolioPnL.realizedPnl,
      positions_count: positionsCount,
      weight,
      contribution_percent: 0, // Will be calculated in portfolio aggregation
      last_updated: new Date().toISOString()
    };
  }

  /**
   * Calculate total portfolio value across all strategies
   */
  private calculateTotalPortfolioValue(): number {
    let totalValue = 0;
    
    for (const positions of this.positionsByStrategy.values()) {
      totalValue += positions.reduce((sum, pos) => 
        sum + Math.abs(pos.currentPrice * pos.quantity), 0);
    }
    
    return totalValue;
  }

  /**
   * Calculate portfolio-level aggregation
   */
  private calculatePortfolioAggregation(): void {
    const strategies = Array.from(this.strategiesMap.values());
    
    if (strategies.length === 0) {
      this.lastAggregation = null;
      this.notifyAggregationHandlers();
      return;
    }

    // Aggregate P&L across all strategies
    const totalPnL = strategies.reduce((sum, strategy) => sum + strategy.total_pnl, 0);
    const totalUnrealizedPnL = strategies.reduce((sum, strategy) => sum + strategy.unrealized_pnl, 0);
    const totalRealizedPnL = strategies.reduce((sum, strategy) => sum + strategy.realized_pnl, 0);

    // Calculate contribution percentages
    const updatedStrategies = strategies.map(strategy => ({
      ...strategy,
      contribution_percent: totalPnL !== 0 ? (strategy.total_pnl / totalPnL) * 100 : 0
    }));

    // Get all positions for portfolio-level calculations
    const allPositions: Position[] = [];
    for (const positions of this.positionsByStrategy.values()) {
      allPositions.push(...positions);
    }

    // Calculate portfolio metrics
    const metrics = this.calculatePortfolioMetrics(allPositions);
    
    this.lastAggregation = {
      total_pnl: totalPnL,
      unrealized_pnl: totalUnrealizedPnL,
      realized_pnl: totalRealizedPnL,
      total_return: metrics.timeWeightedReturn,
      sharpe_ratio: 0, // Will be calculated with historical data
      max_drawdown: 0, // Will be calculated with historical data
      volatility: 0, // Will be calculated with historical data
      beta: 0, // Will be calculated with benchmark data
      strategies: updatedStrategies,
      last_updated: new Date().toISOString()
    };

    this.notifyAggregationHandlers();
  }

  /**
   * Calculate advanced portfolio metrics
   */
  private calculatePortfolioMetrics(positions: Position[]): PortfolioMetrics {
    if (positions.length === 0) {
      return {
        timeWeightedReturn: 0,
        moneyWeightedReturn: 0,
        totalExposure: 0,
        activeStrategies: this.strategiesMap.size,
        avgDailyReturn: 0,
        winRate: 0,
        profitFactor: 0,
        downvideDeviation: 0
      };
    }

    const totalExposure = positions.reduce((sum, pos) => 
      sum + Math.abs(pos.currentPrice * pos.quantity), 0);
    
    const totalInvestment = positions.reduce((sum, pos) => 
      sum + (pos.averagePrice * pos.quantity), 0);
    
    const timeWeightedReturn = totalInvestment > 0 
      ? ((totalExposure - totalInvestment) / totalInvestment) * 100 
      : 0;

    // Calculate basic metrics (simplified - would need historical data for accuracy)
    const profitablePositions = positions.filter(pos => pos.unrealizedPnl + pos.realizedPnl > 0);
    const winRate = positions.length > 0 ? (profitablePositions.length / positions.length) * 100 : 0;

    const totalProfit = positions
      .filter(pos => pos.unrealizedPnl + pos.realizedPnl > 0)
      .reduce((sum, pos) => sum + pos.unrealizedPnl + pos.realizedPnl, 0);
    
    const totalLoss = Math.abs(positions
      .filter(pos => pos.unrealizedPnl + pos.realizedPnl < 0)
      .reduce((sum, pos) => sum + pos.unrealizedPnl + pos.realizedPnl, 0));
    
    const profitFactor = totalLoss > 0 ? totalProfit / totalLoss : totalProfit > 0 ? 999 : 0;

    return {
      timeWeightedReturn,
      moneyWeightedReturn: timeWeightedReturn, // Simplified
      totalExposure,
      activeStrategies: this.strategiesMap.size,
      avgDailyReturn: 0, // Would need historical data
      winRate,
      profitFactor,
      downvideDeviation: 0 // Would need historical data
    };
  }

  /**
   * Get current portfolio aggregation
   */
  public getPortfolioAggregation(): PortfolioAggregation | null {
    return this.lastAggregation;
  }

  /**
   * Get strategies breakdown
   */
  public getStrategies(): StrategyPnL[] {
    return Array.from(this.strategiesMap.values());
  }

  /**
   * Get positions for a specific strategy
   */
  public getStrategyPositions(strategyId: string): Position[] {
    return this.positionsByStrategy.get(strategyId) || [];
  }

  /**
   * Get all positions across all strategies
   */
  public getAllPositions(): Position[] {
    const allPositions: Position[] = [];
    for (const positions of this.positionsByStrategy.values()) {
      allPositions.push(...positions);
    }
    return allPositions;
  }

  /**
   * Calculate P&L attribution by strategy
   */
  public getStrategyAttribution(): StrategyPnL[] {
    return this.getStrategies().sort((a, b) => b.contribution_percent - a.contribution_percent);
  }

  /**
   * Get portfolio performance breakdown
   */
  public getPerformanceBreakdown(): PnLBreakdown[] {
    const allPositions = this.getAllPositions();
    const aggregation = this.getPortfolioAggregation();
    
    if (!aggregation || allPositions.length === 0) return [];
    
    return pnlEngine.calculatePnLBreakdown(allPositions, aggregation.total_pnl);
  }

  /**
   * Event handlers
   */
  public addAggregationHandler(handler: (aggregation: PortfolioAggregation) => void): void {
    this.aggregationHandlers.add(handler);
  }

  public removeAggregationHandler(handler: (aggregation: PortfolioAggregation) => void): void {
    this.aggregationHandlers.delete(handler);
  }

  /**
   * Notify aggregation handlers
   */
  private notifyAggregationHandlers(): void {
    if (!this.lastAggregation) return;
    
    this.aggregationHandlers.forEach(handler => {
      try {
        handler(this.lastAggregation!);
      } catch (error) {
        console.error('Error in aggregation handler:', error);
      }
    });
  }

  /**
   * Reset all data
   */
  public reset(): void {
    this.strategiesMap.clear();
    this.positionsByStrategy.clear();
    this.lastAggregation = null;
    this.notifyAggregationHandlers();
  }
}

// Export singleton instance
export const portfolioAggregationService = new PortfolioAggregationService();