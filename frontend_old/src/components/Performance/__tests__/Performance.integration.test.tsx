import React from 'react';
import { render, screen } from '@testing-library/react';
import { describe, it, expect, vi } from 'vitest';
import { MetricsCards } from '../MetricsCards';

// Simple integration test focusing on core functionality
describe('Performance Components Integration', () => {
  it('MetricsCards renders all required metrics', () => {
    const props = {
      totalPnL: 1250.50,
      unrealizedPnL: 125.25,
      winRate: 0.622,
      sharpeRatio: 1.85,
      maxDrawdown: 8.5,
      totalTrades: 45,
      winningTrades: 28,
      dailyChange: 2.3,
      weeklyChange: 8.7,
      monthlyChange: 15.2
    };

    render(<MetricsCards {...props} />);
    
    // Check that key metrics are displayed
    expect(screen.getByText('Total P&L')).toBeInTheDocument();
    expect(screen.getByText('Win Rate')).toBeInTheDocument();
    expect(screen.getByText('Sharpe Ratio')).toBeInTheDocument();
    expect(screen.getByText('Max Drawdown')).toBeInTheDocument();
  });

  it('MetricsCards shows correct numerical values', () => {
    const props = {
      totalPnL: 1000.00,
      unrealizedPnL: 50.00,
      winRate: 0.6,
      sharpeRatio: 1.5,
      maxDrawdown: 5.0,
      totalTrades: 100,
      winningTrades: 60,
      dailyChange: 1.0,
      weeklyChange: 5.0,
      monthlyChange: 10.0
    };

    render(<MetricsCards {...props} />);
    
    // Ant Design formats numbers with comma separators and splits into int/decimal
    expect(screen.getByText('1,000')).toBeInTheDocument(); // Integer part
    expect(screen.getByText('.00')).toBeInTheDocument(); // Decimal part
    expect(screen.getByText('60.0')).toBeInTheDocument(); // Win rate as percentage
    expect(screen.getByText('1.50')).toBeInTheDocument();
    expect(screen.getByText('5.00')).toBeInTheDocument();
  });

  it('MetricsCards handles negative values correctly', () => {
    const props = {
      totalPnL: -500.00,
      unrealizedPnL: -25.00,
      winRate: 0.3,
      sharpeRatio: -0.5,
      maxDrawdown: 15.0,
      totalTrades: 50,
      winningTrades: 15,
      dailyChange: -2.0,
      weeklyChange: -5.0,
      monthlyChange: -10.0
    };

    render(<MetricsCards {...props} />);
    
    expect(screen.getByText('-500')).toBeInTheDocument();
    expect(screen.getByText('-25')).toBeInTheDocument();
    expect(screen.getByText('30.0')).toBeInTheDocument(); // Win rate as percentage
    expect(screen.getByText('-0.50')).toBeInTheDocument();
  });

  it('MetricsCards calculates win rate percentage correctly', () => {
    const props = {
      totalPnL: 0,
      unrealizedPnL: 0,
      winRate: 0.75,
      sharpeRatio: 0,
      maxDrawdown: 0,
      totalTrades: 100,
      winningTrades: 75,
      dailyChange: 0,
      weeklyChange: 0,
      monthlyChange: 0
    };

    render(<MetricsCards {...props} />);
    
    expect(screen.getByText('75.0')).toBeInTheDocument();
    expect(screen.getByText('75 of 100 trades')).toBeInTheDocument();
  });

  it('handles zero and edge case values', () => {
    const props = {
      totalPnL: 0,
      unrealizedPnL: 0,
      winRate: 0,
      sharpeRatio: 0,
      maxDrawdown: 0,
      totalTrades: 0,
      winningTrades: 0,
      dailyChange: 0,
      weeklyChange: 0,
      monthlyChange: 0
    };

    render(<MetricsCards {...props} />);
    
    // Should not crash with zero values
    expect(screen.getByText('Total P&L')).toBeInTheDocument();
    expect(screen.getByText('0 of 0 trades')).toBeInTheDocument();
  });
});