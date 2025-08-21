/**
 * Performance Attribution Calculator for strategy-level attribution analysis
 */

import { Position } from '../types/position';
import { StrategyPnL } from './portfolioAggregationService';

export interface AttributionBreakdown {
  strategy_id: string;
  strategy_name: string;
  contribution_pnl: number;
  contribution_percent: number;
  active_return: number;
  tracking_error: number;
  information_ratio: number;
  weight: number;
  benchmark_weight?: number;
  allocation_effect: number;
  selection_effect: number;
  interaction_effect: number;
}

export interface SectorAttribution {
  sector: string;
  portfolio_weight: number;
  benchmark_weight: number;
  portfolio_return: number;
  benchmark_return: number;
  allocation_effect: number;
  selection_effect: number;
  total_effect: number;
}

export interface FactorAttribution {
  factor_name: string;
  exposure: number;
  factor_return: number;
  contribution: number;
  risk_contribution: number;
}

export interface AttributionAnalysis {
  strategy_attributions: AttributionBreakdown[];
  sector_attributions: SectorAttribution[];
  factor_attributions: FactorAttribution[];
  total_active_return: number;
  total_tracking_error: number;
  information_ratio: number;
  period: string;
}

export interface BenchmarkData {
  returns: number[];
  weights: Map<string, number>;
  sectorWeights: Map<string, number>;
  factorExposures: Map<string, number>;
}

export class PerformanceAttributionCalculator {
  private benchmarkData: BenchmarkData | null = null;
  private historicalReturns: Map<string, number[]> = new Map();
  private sectorMappings: Map<string, string> = new Map();

  constructor() {
    this.initializeDefaultSectorMappings();
  }

  /**
   * Initialize default sector mappings for common instruments
   */
  private initializeDefaultSectorMappings(): void {
    // Technology stocks
    const techStocks = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'TSLA', 'NVDA'];
    techStocks.forEach(symbol => this.sectorMappings.set(symbol, 'Technology'));

    // Financial stocks
    const financialStocks = ['JPM', 'BAC', 'WFC', 'GS', 'MS'];
    financialStocks.forEach(symbol => this.sectorMappings.set(symbol, 'Financials'));

    // Healthcare stocks
    const healthcareStocks = ['JNJ', 'PFE', 'UNH', 'ABBV', 'MRK'];
    healthcareStocks.forEach(symbol => this.sectorMappings.set(symbol, 'Healthcare'));

    // Consumer goods
    const consumerStocks = ['KO', 'PG', 'WMT', 'HD', 'MCD'];
    consumerStocks.forEach(symbol => this.sectorMappings.set(symbol, 'Consumer'));

    // Energy stocks
    const energyStocks = ['XOM', 'CVX', 'COP', 'EOG'];
    energyStocks.forEach(symbol => this.sectorMappings.set(symbol, 'Energy'));
  }

  /**
   * Set benchmark data for attribution analysis
   */
  public setBenchmarkData(benchmark: BenchmarkData): void {
    this.benchmarkData = benchmark;
  }

  /**
   * Calculate strategy attribution analysis
   */
  public calculateStrategyAttribution(
    strategies: StrategyPnL[],
    totalPortfolioPnL: number,
    period: string = '1M'
  ): AttributionBreakdown[] {
    if (strategies.length === 0) return [];

    const totalWeight = strategies.reduce((sum, strategy) => sum + strategy.weight, 0);

    return strategies.map(strategy => {
      // Calculate active return (strategy return vs benchmark)
      const strategyReturn = this.calculateStrategyReturn(strategy);
      const benchmarkReturn = this.getBenchmarkReturn(strategy.strategy_id, period);
      const activeReturn = strategyReturn - benchmarkReturn;

      // Calculate tracking error
      const trackingError = this.calculateTrackingError(strategy.strategy_id, period);

      // Information ratio
      const informationRatio = trackingError > 0 ? activeReturn / trackingError : 0;

      // Allocation and selection effects (Brinson model)
      const benchmarkWeight = this.getBenchmarkWeight(strategy.strategy_id);
      const allocationEffect = (strategy.weight / 100 - benchmarkWeight) * benchmarkReturn;
      const selectionEffect = benchmarkWeight * (strategyReturn - benchmarkReturn);
      const interactionEffect = (strategy.weight / 100 - benchmarkWeight) * (strategyReturn - benchmarkReturn);

      return {
        strategy_id: strategy.strategy_id,
        strategy_name: strategy.strategy_name,
        contribution_pnl: strategy.total_pnl,
        contribution_percent: strategy.contribution_percent,
        active_return: activeReturn,
        tracking_error: trackingError,
        information_ratio: informationRatio,
        weight: strategy.weight,
        benchmark_weight: benchmarkWeight * 100,
        allocation_effect: allocationEffect,
        selection_effect: selectionEffect,
        interaction_effect: interactionEffect
      };
    });
  }

  /**
   * Calculate sector attribution analysis
   */
  public calculateSectorAttribution(
    positions: Position[],
    period: string = '1M'
  ): SectorAttribution[] {
    if (positions.length === 0) return [];

    // Group positions by sector
    const sectorPositions = new Map<string, Position[]>();
    positions.forEach(position => {
      const sector = this.getSectorForSymbol(position.symbol);
      if (!sectorPositions.has(sector)) {
        sectorPositions.set(sector, []);
      }
      sectorPositions.get(sector)!.push(position);
    });

    const totalPortfolioValue = positions.reduce((sum, pos) => 
      sum + Math.abs(pos.currentPrice * pos.quantity), 0);

    const sectorAttributions: SectorAttribution[] = [];

    for (const [sector, sectorPos] of sectorPositions.entries()) {
      const sectorValue = sectorPos.reduce((sum, pos) => 
        sum + Math.abs(pos.currentPrice * pos.quantity), 0);
      
      const portfolioWeight = totalPortfolioValue > 0 ? (sectorValue / totalPortfolioValue) * 100 : 0;
      const benchmarkWeight = this.getBenchmarkSectorWeight(sector);
      
      // Calculate sector returns
      const portfolioReturn = this.calculateSectorReturn(sectorPos);
      const benchmarkReturn = this.getBenchmarkSectorReturn(sector, period);
      
      // Brinson attribution effects
      const allocationEffect = (portfolioWeight / 100 - benchmarkWeight) * benchmarkReturn;
      const selectionEffect = benchmarkWeight * (portfolioReturn - benchmarkReturn);
      const totalEffect = allocationEffect + selectionEffect;

      sectorAttributions.push({
        sector,
        portfolio_weight: portfolioWeight,
        benchmark_weight: benchmarkWeight * 100,
        portfolio_return: portfolioReturn,
        benchmark_return: benchmarkReturn,
        allocation_effect: allocationEffect,
        selection_effect: selectionEffect,
        total_effect: totalEffect
      });
    }

    return sectorAttributions.sort((a, b) => Math.abs(b.total_effect) - Math.abs(a.total_effect));
  }

  /**
   * Calculate factor attribution analysis
   */
  public calculateFactorAttribution(
    positions: Position[],
    period: string = '1M'
  ): FactorAttribution[] {
    const factorNames = ['Market', 'Size', 'Value', 'Quality', 'Momentum', 'Volatility'];
    const factorAttributions: FactorAttribution[] = [];

    for (const factorName of factorNames) {
      const exposure = this.calculateFactorExposure(positions, factorName);
      const factorReturn = this.getFactorReturn(factorName, period);
      const contribution = exposure * factorReturn;
      const riskContribution = this.calculateFactorRiskContribution(positions, factorName);

      factorAttributions.push({
        factor_name: factorName,
        exposure,
        factor_return: factorReturn,
        contribution,
        risk_contribution: riskContribution
      });
    }

    return factorAttributions.sort((a, b) => Math.abs(b.contribution) - Math.abs(a.contribution));
  }

  /**
   * Calculate comprehensive attribution analysis
   */
  public calculateFullAttribution(
    strategies: StrategyPnL[],
    positions: Position[],
    period: string = '1M'
  ): AttributionAnalysis {
    const strategyAttributions = this.calculateStrategyAttribution(strategies, 
      strategies.reduce((sum, s) => sum + s.total_pnl, 0), period);
    
    const sectorAttributions = this.calculateSectorAttribution(positions, period);
    const factorAttributions = this.calculateFactorAttribution(positions, period);

    // Calculate portfolio-level metrics
    const totalActiveReturn = strategyAttributions.reduce((sum, attr) => 
      sum + attr.active_return * (attr.weight / 100), 0);
    
    const weightedTrackingErrors = strategyAttributions.map(attr => 
      Math.pow(attr.tracking_error * (attr.weight / 100), 2));
    const totalTrackingError = Math.sqrt(weightedTrackingErrors.reduce((sum, te) => sum + te, 0));
    
    const informationRatio = totalTrackingError > 0 ? totalActiveReturn / totalTrackingError : 0;

    return {
      strategy_attributions: strategyAttributions,
      sector_attributions: sectorAttributions,
      factor_attributions: factorAttributions,
      total_active_return: totalActiveReturn,
      total_tracking_error: totalTrackingError,
      information_ratio: informationRatio,
      period
    };
  }

  /**
   * Helper methods for calculations
   */
  private calculateStrategyReturn(strategy: StrategyPnL): number {
    // Simplified return calculation - would use historical data in practice
    return strategy.total_pnl > 0 ? Math.random() * 10 + 2 : Math.random() * -5;
  }

  private getBenchmarkReturn(strategyId: string, period: string): number {
    // Mock benchmark return - would use actual benchmark data
    return Math.random() * 8 + 1;
  }

  private calculateTrackingError(strategyId: string, period: string): number {
    // Mock tracking error calculation - would use historical return variance
    return Math.random() * 3 + 0.5;
  }

  private getBenchmarkWeight(strategyId: string): number {
    if (!this.benchmarkData) return 0.1; // Default 10%
    return this.benchmarkData.weights.get(strategyId) || 0.1;
  }

  private getSectorForSymbol(symbol: string): string {
    return this.sectorMappings.get(symbol) || 'Other';
  }

  private getBenchmarkSectorWeight(sector: string): number {
    if (!this.benchmarkData) {
      // Default sector weights
      const defaultWeights: Record<string, number> = {
        'Technology': 0.25,
        'Financials': 0.15,
        'Healthcare': 0.15,
        'Consumer': 0.12,
        'Energy': 0.08,
        'Other': 0.25
      };
      return defaultWeights[sector] || 0.1;
    }
    return this.benchmarkData.sectorWeights.get(sector) || 0.1;
  }

  private calculateSectorReturn(positions: Position[]): number {
    if (positions.length === 0) return 0;
    
    const totalValue = positions.reduce((sum, pos) => 
      sum + Math.abs(pos.currentPrice * pos.quantity), 0);
    const totalPnL = positions.reduce((sum, pos) => 
      sum + pos.unrealizedPnl + pos.realizedPnl, 0);
    
    return totalValue > 0 ? (totalPnL / totalValue) * 100 : 0;
  }

  private getBenchmarkSectorReturn(sector: string, period: string): number {
    // Mock sector benchmark return
    const sectorReturns: Record<string, number> = {
      'Technology': Math.random() * 12 + 3,
      'Financials': Math.random() * 8 + 1,
      'Healthcare': Math.random() * 9 + 2,
      'Consumer': Math.random() * 7 + 1.5,
      'Energy': Math.random() * 15 - 2,
      'Other': Math.random() * 6 + 1
    };
    return sectorReturns[sector] || 5;
  }

  private calculateFactorExposure(positions: Position[], factorName: string): number {
    // Mock factor exposure calculation
    const factorMappings: Record<string, number> = {
      'Market': 0.95,
      'Size': Math.random() * 0.4 - 0.2,
      'Value': Math.random() * 0.3 - 0.15,
      'Quality': Math.random() * 0.25,
      'Momentum': Math.random() * 0.4 - 0.2,
      'Volatility': Math.random() * 0.3 - 0.15
    };
    return factorMappings[factorName] || 0;
  }

  private getFactorReturn(factorName: string, period: string): number {
    // Mock factor returns
    const factorReturns: Record<string, number> = {
      'Market': Math.random() * 8 + 2,
      'Size': Math.random() * 4 - 2,
      'Value': Math.random() * 6 - 1,
      'Quality': Math.random() * 5 + 1,
      'Momentum': Math.random() * 8 - 2,
      'Volatility': Math.random() * 3 - 1.5
    };
    return factorReturns[factorName] || 0;
  }

  private calculateFactorRiskContribution(positions: Position[], factorName: string): number {
    // Mock risk contribution calculation
    return Math.random() * 0.1;
  }

  /**
   * Add historical returns for strategy
   */
  public addHistoricalReturns(strategyId: string, returns: number[]): void {
    this.historicalReturns.set(strategyId, returns);
  }

  /**
   * Update sector mapping for symbol
   */
  public updateSectorMapping(symbol: string, sector: string): void {
    this.sectorMappings.set(symbol, sector);
  }

  /**
   * Get current sector mappings
   */
  public getSectorMappings(): Map<string, string> {
    return new Map(this.sectorMappings);
  }
}

// Export singleton instance
export const attributionCalculator = new PerformanceAttributionCalculator();