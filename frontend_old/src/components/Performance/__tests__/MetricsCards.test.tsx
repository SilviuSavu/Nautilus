import React from 'react';
import { render, screen } from '@testing-library/react';
import { describe, it, expect } from 'vitest';
import { MetricsCards } from '../MetricsCards';

describe('MetricsCards', () => {
  const defaultProps = {
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

  it('renders all metric cards', () => {
    render(<MetricsCards {...defaultProps} />);
    
    expect(screen.getByText('Total P&L')).toBeInTheDocument();
    expect(screen.getByText('Unrealized P&L')).toBeInTheDocument();
    expect(screen.getByText('Win Rate')).toBeInTheDocument();
    expect(screen.getByText('Sharpe Ratio')).toBeInTheDocument();
    expect(screen.getByText('Max Drawdown')).toBeInTheDocument();
    expect(screen.getByText('Weekly Change')).toBeInTheDocument();
    expect(screen.getByText('Monthly Change')).toBeInTheDocument();
    expect(screen.getByText('Total Trades')).toBeInTheDocument();
  });

  it('displays correct values', () => {
    render(<MetricsCards {...defaultProps} />);
    
    expect(screen.getByText('1250.50')).toBeInTheDocument();
    expect(screen.getByText('125.25')).toBeInTheDocument();
    expect(screen.getByText('62.2')).toBeInTheDocument();
    expect(screen.getByText('1.85')).toBeInTheDocument();
    expect(screen.getByText('8.50')).toBeInTheDocument();
    expect(screen.getByText('45')).toBeInTheDocument();
  });

  it('shows correct colors for positive P&L', () => {
    render(<MetricsCards {...defaultProps} />);
    
    const totalPnLElement = screen.getByText('1250.50').closest('.ant-statistic-content');
    expect(totalPnLElement).toHaveStyle({ color: '#3f8600' });
    
    const unrealizedPnLElement = screen.getByText('125.25').closest('.ant-statistic-content');
    expect(unrealizedPnLElement).toHaveStyle({ color: '#3f8600' });
  });

  it('shows correct colors for negative P&L', () => {
    const negativeProps = {
      ...defaultProps,
      totalPnL: -500.25,
      unrealizedPnL: -75.50
    };
    
    render(<MetricsCards {...negativeProps} />);
    
    const totalPnLElement = screen.getByText('-500.25').closest('.ant-statistic-content');
    expect(totalPnLElement).toHaveStyle({ color: '#cf1322' });
    
    const unrealizedPnLElement = screen.getByText('-75.50').closest('.ant-statistic-content');
    expect(unrealizedPnLElement).toHaveStyle({ color: '#cf1322' });
  });

  it('shows correct win rate color for good performance', () => {
    render(<MetricsCards {...defaultProps} />);
    
    const winRateElement = screen.getByText('62.2').closest('.ant-statistic-content');
    expect(winRateElement).toHaveStyle({ color: '#3f8600' }); // > 50%
  });

  it('shows correct win rate color for poor performance', () => {
    const poorWinRateProps = {
      ...defaultProps,
      winRate: 0.35
    };
    
    render(<MetricsCards {...poorWinRateProps} />);
    
    const winRateElement = screen.getByText('35.0').closest('.ant-statistic-content');
    expect(winRateElement).toHaveStyle({ color: '#666666' }); // <= 50%
  });

  it('shows correct Sharpe ratio colors', () => {
    // Excellent Sharpe ratio > 1
    render(<MetricsCards {...defaultProps} />);
    let sharpeElement = screen.getByText('1.85').closest('.ant-statistic-content');
    expect(sharpeElement).toHaveStyle({ color: '#3f8600' });
    
    // Good Sharpe ratio between 0 and 1
    render(<MetricsCards {...{ ...defaultProps, sharpeRatio: 0.75 }} />);
    sharpeElement = screen.getByText('0.75').closest('.ant-statistic-content');
    expect(sharpeElement).toHaveStyle({ color: '#fa8c16' });
    
    // Poor Sharpe ratio < 0
    render(<MetricsCards {...{ ...defaultProps, sharpeRatio: -0.25 }} />);
    sharpeElement = screen.getByText('-0.25').closest('.ant-statistic-content');
    expect(sharpeElement).toHaveStyle({ color: '#cf1322' });
  });

  it('shows correct drawdown colors', () => {
    // Low drawdown < 5%
    const lowDrawdownProps = { ...defaultProps, maxDrawdown: 3.2 };
    render(<MetricsCards {...lowDrawdownProps} />);
    let drawdownElement = screen.getByText('3.20').closest('.ant-statistic-content');
    expect(drawdownElement).toHaveStyle({ color: '#3f8600' });
    
    // Medium drawdown 5-10%
    render(<MetricsCards {...defaultProps} />);
    drawdownElement = screen.getByText('8.50').closest('.ant-statistic-content');
    expect(drawdownElement).toHaveStyle({ color: '#fa8c16' });
    
    // High drawdown > 10%
    const highDrawdownProps = { ...defaultProps, maxDrawdown: 15.5 };
    render(<MetricsCards {...highDrawdownProps} />);
    drawdownElement = screen.getByText('15.50').closest('.ant-statistic-content');
    expect(drawdownElement).toHaveStyle({ color: '#cf1322' });
  });

  it('displays change arrows correctly', () => {
    render(<MetricsCards {...defaultProps} />);
    
    // Should show up arrows for positive changes
    const upArrows = screen.getAllByLabelText('arrow-up');
    expect(upArrows.length).toBeGreaterThan(0);
    
    // Test with negative changes
    const negativeChangeProps = {
      ...defaultProps,
      dailyChange: -1.5,
      weeklyChange: -3.2,
      monthlyChange: -8.1
    };
    
    render(<MetricsCards {...negativeChangeProps} />);
    
    const downArrows = screen.getAllByLabelText('arrow-down');
    expect(downArrows.length).toBeGreaterThan(0);
  });

  it('displays trade statistics correctly', () => {
    render(<MetricsCards {...defaultProps} />);
    
    expect(screen.getByText('28 of 45 trades')).toBeInTheDocument();
  });

  it('handles zero values correctly', () => {
    const zeroProps = {
      ...defaultProps,
      totalPnL: 0,
      unrealizedPnL: 0,
      dailyChange: 0,
      weeklyChange: 0,
      monthlyChange: 0
    };
    
    render(<MetricsCards {...zeroProps} />);
    
    expect(screen.getAllByText('0.00')).toHaveLength(5); // All zero values displayed
  });

  it('formats percentages correctly', () => {
    render(<MetricsCards {...defaultProps} />);
    
    expect(screen.getByText('2.30%')).toBeInTheDocument(); // Daily change
    expect(screen.getByText('8.70%')).toBeInTheDocument(); // Weekly change
    expect(screen.getByText('15.20%')).toBeInTheDocument(); // Monthly change
    expect(screen.getByText('8.50%')).toBeInTheDocument(); // Max drawdown
  });

  it('shows descriptive labels', () => {
    render(<MetricsCards {...defaultProps} />);
    
    expect(screen.getByText('Daily change')).toBeInTheDocument();
    expect(screen.getByText('Open positions')).toBeInTheDocument();
    expect(screen.getByText('Risk-adjusted return')).toBeInTheDocument();
    expect(screen.getByText('Maximum loss from peak')).toBeInTheDocument();
    expect(screen.getByText('7-day performance')).toBeInTheDocument();
    expect(screen.getByText('30-day performance')).toBeInTheDocument();
    expect(screen.getByText('Executed orders')).toBeInTheDocument();
  });

  it('handles large numbers correctly', () => {
    const largeNumberProps = {
      ...defaultProps,
      totalPnL: 1234567.89,
      totalTrades: 9999
    };
    
    render(<MetricsCards {...largeNumberProps} />);
    
    expect(screen.getByText('1234567.89')).toBeInTheDocument();
    expect(screen.getByText('9999')).toBeInTheDocument();
  });

  it('handles decimal precision correctly', () => {
    const precisionProps = {
      ...defaultProps,
      totalPnL: 1250.123456, // Should round to 2 decimal places
      sharpeRatio: 1.8567,    // Should round to 2 decimal places
      dailyChange: 2.3456     // Should round to 2 decimal places
    };
    
    render(<MetricsCards {...precisionProps} />);
    
    expect(screen.getByText('1250.12')).toBeInTheDocument();
    expect(screen.getByText('1.86')).toBeInTheDocument();
    expect(screen.getByText('2.35%')).toBeInTheDocument();
  });
});