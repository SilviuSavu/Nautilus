/**
 * Unit tests for P&L Calculation Engine
 */

import { PnLCalculationEngine } from '../pnlCalculationEngine';
import { Position } from '../../types/position';

describe('PnLCalculationEngine', () => {
  let engine: PnLCalculationEngine;

  beforeEach(() => {
    engine = new PnLCalculationEngine({
      baseCurrency: 'USD',
      includeFees: true,
      useMarkToMarket: true
    });
  });

  describe('calculateUnrealizedPnL', () => {
    const mockPosition: Position = {
      id: 'test-1',
      symbol: 'AAPL',
      venue: 'NASDAQ',
      side: 'LONG',
      quantity: 100,
      averagePrice: 150.00,
      unrealizedPnl: 0,
      realizedPnl: 0,
      currentPrice: 155.00,
      currency: 'USD',
      timestamp: Date.now(),
      openTimestamp: Date.now() - 3600000
    };

    it('should calculate positive unrealized P&L for long position with price increase', () => {
      const pnl = engine.calculateUnrealizedPnL(mockPosition, 155.00);
      expect(pnl).toBe(500); // (155 - 150) * 100 = 500
    });

    it('should calculate negative unrealized P&L for long position with price decrease', () => {
      const pnl = engine.calculateUnrealizedPnL(mockPosition, 145.00);
      expect(pnl).toBe(-500); // (145 - 150) * 100 = -500
    });

    it('should calculate negative unrealized P&L for short position with price increase', () => {
      const shortPosition = { ...mockPosition, side: 'SHORT' as const };
      const pnl = engine.calculateUnrealizedPnL(shortPosition, 155.00);
      expect(pnl).toBe(-500); // (155 - 150) * 100 * -1 = -500
    });

    it('should calculate positive unrealized P&L for short position with price decrease', () => {
      const shortPosition = { ...mockPosition, side: 'SHORT' as const };
      const pnl = engine.calculateUnrealizedPnL(shortPosition, 145.00);
      expect(pnl).toBe(500); // (145 - 150) * 100 * -1 = 500
    });

    it('should include fees when configured', () => {
      const pnl = engine.calculateUnrealizedPnL(mockPosition, 155.00, 10);
      expect(pnl).toBe(490); // 500 - 10 = 490
    });
  });

  describe('calculateRealizedPnL', () => {
    it('should calculate positive realized P&L for profitable long trade', () => {
      const pnl = engine.calculateRealizedPnL(150, 155, 100, 'LONG', 'USD');
      expect(pnl).toBe(500);
    });

    it('should calculate negative realized P&L for losing long trade', () => {
      const pnl = engine.calculateRealizedPnL(150, 145, 100, 'LONG', 'USD');
      expect(pnl).toBe(-500);
    });

    it('should calculate positive realized P&L for profitable short trade', () => {
      const pnl = engine.calculateRealizedPnL(150, 145, 100, 'SHORT', 'USD');
      expect(pnl).toBe(500);
    });

    it('should include fees in calculation', () => {
      const pnl = engine.calculateRealizedPnL(150, 155, 100, 'LONG', 'USD', 5, 10);
      expect(pnl).toBe(485); // 500 - 5 - 10 = 485
    });
  });

  describe('calculatePortfolioPnL', () => {
    const mockPositions: Position[] = [
      {
        id: 'pos-1',
        symbol: 'AAPL',
        venue: 'NASDAQ',
        side: 'LONG',
        quantity: 100,
        averagePrice: 150,
        unrealizedPnl: 500,
        realizedPnl: 200,
        currentPrice: 155,
        currency: 'USD',
        timestamp: Date.now(),
        openTimestamp: Date.now()
      },
      {
        id: 'pos-2',
        symbol: 'GOOGL',
        venue: 'NASDAQ',
        side: 'SHORT',
        quantity: 50,
        averagePrice: 2800,
        unrealizedPnl: -1000,
        realizedPnl: 300,
        currentPrice: 2820,
        currency: 'USD',
        timestamp: Date.now(),
        openTimestamp: Date.now()
      }
    ];

    it('should calculate total portfolio P&L correctly', () => {
      const portfolioPnL = engine.calculatePortfolioPnL(mockPositions);
      
      expect(portfolioPnL.unrealizedPnl).toBe(-500); // 500 + (-1000)
      expect(portfolioPnL.realizedPnl).toBe(500); // 200 + 300
      expect(portfolioPnL.totalPnl).toBe(0); // -500 + 500
      expect(portfolioPnL.currency).toBe('USD');
    });

    it('should calculate P&L percentage correctly', () => {
      const portfolioPnL = engine.calculatePortfolioPnL(mockPositions);
      
      // Total exposure: (155 * 100) + (2820 * 50) = 15,500 + 141,000 = 156,500
      // Total P&L: 0
      // Percentage: 0%
      expect(portfolioPnL.pnlPercentage).toBe(0);
    });
  });

  describe('calculateMarkToMarket', () => {
    const mockPositions: Position[] = [
      {
        id: 'pos-1',
        symbol: 'AAPL',
        venue: 'NASDAQ',
        side: 'LONG',
        quantity: 100,
        averagePrice: 150,
        unrealizedPnl: 0,
        realizedPnl: 0,
        currentPrice: 155,
        currency: 'USD',
        timestamp: Date.now(),
        openTimestamp: Date.now()
      }
    ];

    it('should calculate mark-to-market values correctly', () => {
      const currentPrices = new Map([['AAPL', 155]]);
      const mtmValues = engine.calculateMarkToMarket(mockPositions, currentPrices);
      
      expect(mtmValues.get('pos-1')).toBe(15500); // 155 * 100
    });
  });

  describe('calculateRiskMetrics', () => {
    const mockReturns = [0.05, -0.02, 0.03, -0.01, 0.04, -0.03, 0.02];

    it('should calculate risk metrics correctly', () => {
      const riskMetrics = engine.calculateRiskMetrics([], mockReturns);
      
      expect(riskMetrics.sharpeRatio).toBeGreaterThan(0);
      expect(riskMetrics.volatility).toBeGreaterThan(0);
      expect(riskMetrics.maxDrawdown).toBeGreaterThanOrEqual(0);
      expect(riskMetrics.var95).toBeLessThanOrEqual(0);
    });

    it('should handle empty returns array', () => {
      const riskMetrics = engine.calculateRiskMetrics([], []);
      
      expect(riskMetrics.sharpeRatio).toBe(0);
      expect(riskMetrics.volatility).toBe(0);
      expect(riskMetrics.maxDrawdown).toBe(0);
      expect(riskMetrics.var95).toBe(0);
      expect(riskMetrics.expectedShortfall).toBe(0);
    });
  });

  describe('configuration', () => {
    it('should update configuration correctly', () => {
      engine.updateConfiguration({ baseCurrency: 'EUR', includeFees: false });
      const config = engine.getConfiguration();
      
      expect(config.baseCurrency).toBe('EUR');
      expect(config.includeFees).toBe(false);
      expect(config.useMarkToMarket).toBe(true); // Unchanged
    });
  });
});