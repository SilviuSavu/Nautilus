import React from 'react';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import { describe, it, expect, vi, beforeEach } from 'vitest';
import StrategyDeploymentPipeline from '../StrategyDeploymentPipeline';
import type { DeploymentPipelineProps } from '../../../types/deployment';

// Mock fetch
global.fetch = vi.fn();

const mockProps: DeploymentPipelineProps = {
  strategyId: 'test-strategy-123',
  onDeploymentCreated: vi.fn(),
  onClose: vi.fn()
};

const mockStrategyData = {
  id: 'test-strategy-123',
  name: 'Test Strategy',
  version: 2,
  template_id: 'ma_cross_001',
  parameters: {
    fast_period: 10,
    slow_period: 20
  },
  risk_settings: {
    max_position_size: '100000',
    stop_loss_atr: 2.0
  }
};

const mockBacktestResults = {
  totalReturn: 0.15,
  sharpeRatio: 1.25,
  maxDrawdown: 0.08,
  winRate: 0.6,
  avgTrade: 0.002,
  totalTrades: 150
};

const mockRiskAssessment = {
  risk_level: 'medium',
  portfolioImpact: 'low',
  correlationRisk: 'medium',
  maxDrawdownEstimate: 0.08,
  varEstimate: 0.04,
  liquidityRisk: 'low',
  warnings: ['Consider position size limits'],
  blockers: [],
  recommendations: ['Monitor correlation with existing positions']
};

describe('StrategyDeploymentPipeline', () => {
  beforeEach(() => {
    vi.clearAllMocks();
    
    // Mock successful strategy data fetch
    (global.fetch as any).mockImplementation((url: string) => {
      if (url.includes('/api/v1/strategies/test-strategy-123')) {
        return Promise.resolve({
          ok: true,
          json: () => Promise.resolve(mockStrategyData)
        });
      }
      
      if (url.includes('/backtests/latest')) {
        return Promise.resolve({
          ok: true,
          json: () => Promise.resolve(mockBacktestResults)
        });
      }
      
      if (url.includes('/risk-assessment')) {
        return Promise.resolve({
          ok: true,
          json: () => Promise.resolve(mockRiskAssessment)
        });
      }
      
      if (url.includes('/deployment/create')) {
        return Promise.resolve({
          ok: true,
          json: () => Promise.resolve({
            deploymentId: 'deployment-123',
            status: 'pending_approval'
          })
        });
      }
      
      return Promise.reject(new Error('Not found'));
    });
  });

  it('renders deployment pipeline with initial configuration step', async () => {
    render(<StrategyDeploymentPipeline {...mockProps} />);
    
    await waitFor(() => {
      expect(screen.getByText('Strategy Deployment Pipeline')).toBeInTheDocument();
      expect(screen.getByText('Configuration')).toBeInTheDocument();
      expect(screen.getByText('Risk Assessment')).toBeInTheDocument();
      expect(screen.getByText('Review')).toBeInTheDocument();
      expect(screen.getByText('Approval')).toBeInTheDocument();
    });
  });

  it('loads strategy data on mount', async () => {
    render(<StrategyDeploymentPipeline {...mockProps} />);
    
    await waitFor(() => {
      expect(global.fetch).toHaveBeenCalledWith('/api/v1/strategies/test-strategy-123');
      expect(global.fetch).toHaveBeenCalledWith('/api/v1/strategies/test-strategy-123/backtests/latest');
    });
  });

  it('allows filling out deployment configuration', async () => {
    render(<StrategyDeploymentPipeline {...mockProps} />);
    
    await waitFor(() => {
      expect(screen.getByDisplayValue('2.1.0')).toBeInTheDocument();
    });
    
    // Update version
    const versionInput = screen.getByDisplayValue('2.1.0');
    fireEvent.change(versionInput, { target: { value: '2.2.0' } });
    expect(versionInput).toHaveValue('2.2.0');
    
    // Select environment
    const environmentSelect = screen.getByRole('combobox', { name: /environment/i });
    fireEvent.click(environmentSelect);
    
    await waitFor(() => {
      const productionOption = screen.getByText('Production');
      fireEvent.click(productionOption);
    });
  });

  it('validates required fields before advancing', async () => {
    render(<StrategyDeploymentPipeline {...mockProps} />);
    
    await waitFor(() => {
      const nextButton = screen.getByRole('button', { name: /next/i });
      expect(nextButton).toBeInTheDocument();
    });
    
    // Clear required field
    const versionInput = screen.getByDisplayValue('2.1.0');
    fireEvent.change(versionInput, { target: { value: '' } });
    
    // Try to advance
    const nextButton = screen.getByRole('button', { name: /next/i });
    fireEvent.click(nextButton);
    
    await waitFor(() => {
      expect(screen.getByText('Please fill in all required fields')).toBeInTheDocument();
    });
  });

  it('performs risk assessment when advancing to step 2', async () => {
    render(<StrategyDeploymentPipeline {...mockProps} />);
    
    await waitFor(() => {
      const nextButton = screen.getByRole('button', { name: /next/i });
      fireEvent.click(nextButton);
    });
    
    await waitFor(() => {
      expect(global.fetch).toHaveBeenCalledWith(
        '/api/v1/nautilus/deployment/risk-assessment',
        expect.objectContaining({
          method: 'POST',
          headers: { 'Content-Type': 'application/json' }
        })
      );
    });
  });

  it('displays risk assessment results', async () => {
    render(<StrategyDeploymentPipeline {...mockProps} />);
    
    // Navigate to risk assessment step
    await waitFor(() => {
      const nextButton = screen.getByRole('button', { name: /next/i });
      fireEvent.click(nextButton);
    });
    
    // Wait for risk assessment to complete
    await waitFor(() => {
      expect(screen.getByText('Risk Assessment Complete')).toBeInTheDocument();
      expect(screen.getByText('MEDIUM')).toBeInTheDocument();
      expect(screen.getByText('8.00%')).toBeInTheDocument(); // Max drawdown
      expect(screen.getByText('4.00%')).toBeInTheDocument(); // VaR
    });
    
    // Check warnings
    expect(screen.getByText('Consider position size limits')).toBeInTheDocument();
    
    // Check recommendations
    expect(screen.getByText('Monitor correlation with existing positions')).toBeInTheDocument();
  });

  it('blocks advancement if risk assessment has blockers', async () => {
    const riskAssessmentWithBlockers = {
      ...mockRiskAssessment,
      blockers: ['Maximum drawdown exceeds 25% limit']
    };
    
    (global.fetch as any).mockImplementation((url: string) => {
      if (url.includes('/risk-assessment')) {
        return Promise.resolve({
          ok: true,
          json: () => Promise.resolve(riskAssessmentWithBlockers)
        });
      }
      return (global.fetch as any).getMockImplementation()(url);
    });
    
    render(<StrategyDeploymentPipeline {...mockProps} />);
    
    // Navigate to risk assessment step
    await waitFor(() => {
      const nextButton = screen.getByRole('button', { name: /next/i });
      fireEvent.click(nextButton);
    });
    
    await waitFor(() => {
      expect(screen.getByText('Maximum drawdown exceeds 25% limit')).toBeInTheDocument();
      
      // Next button should be disabled
      const nextButton = screen.getByRole('button', { name: /next/i });
      expect(nextButton).toBeDisabled();
    });
  });

  it('displays review information correctly', async () => {
    render(<StrategyDeploymentPipeline {...mockProps} />);
    
    // Navigate through steps
    await waitFor(() => {
      const nextButton = screen.getByRole('button', { name: /next/i });
      fireEvent.click(nextButton);
    });
    
    await waitFor(() => {
      const nextButton = screen.getByRole('button', { name: /next/i });
      fireEvent.click(nextButton);
    });
    
    // Should be on review step
    await waitFor(() => {
      expect(screen.getByText('Deployment Review')).toBeInTheDocument();
      expect(screen.getByText('Test Strategy')).toBeInTheDocument();
      expect(screen.getByText('2.1.0')).toBeInTheDocument();
    });
    
    // Check backtest results tab
    const backtestTab = screen.getByText('Backtest Results');
    fireEvent.click(backtestTab);
    
    await waitFor(() => {
      expect(screen.getByText('15.00%')).toBeInTheDocument(); // Total return
      expect(screen.getByText('1.25')).toBeInTheDocument(); // Sharpe ratio
    });
  });

  it('submits deployment request successfully', async () => {
    render(<StrategyDeploymentPipeline {...mockProps} />);
    
    // Navigate to final step
    await waitFor(() => {
      const nextButton = screen.getByRole('button', { name: /next/i });
      fireEvent.click(nextButton);
    });
    
    await waitFor(() => {
      const nextButton = screen.getByRole('button', { name: /next/i });
      fireEvent.click(nextButton);
    });
    
    await waitFor(() => {
      const nextButton = screen.getByRole('button', { name: /next/i });
      fireEvent.click(nextButton);
    });
    
    // Submit deployment request
    await waitFor(() => {
      const submitButton = screen.getByText('Submit for Approval');
      fireEvent.click(submitButton);
    });
    
    await waitFor(() => {
      expect(global.fetch).toHaveBeenCalledWith(
        '/api/v1/nautilus/deployment/create',
        expect.objectContaining({
          method: 'POST',
          headers: { 'Content-Type': 'application/json' }
        })
      );
      
      expect(mockProps.onDeploymentCreated).toHaveBeenCalledWith('deployment-123');
      expect(screen.getByText('Deployment request created successfully')).toBeInTheDocument();
    });
  });

  it('handles API errors gracefully', async () => {
    (global.fetch as any).mockRejectedValueOnce(new Error('API Error'));
    
    render(<StrategyDeploymentPipeline {...mockProps} />);
    
    await waitFor(() => {
      expect(screen.getByText('Failed to load strategy data')).toBeInTheDocument();
    });
  });

  it('allows navigation between steps', async () => {
    render(<StrategyDeploymentPipeline {...mockProps} />);
    
    // Go to step 2
    await waitFor(() => {
      const nextButton = screen.getByRole('button', { name: /next/i });
      fireEvent.click(nextButton);
    });
    
    // Go back to step 1
    await waitFor(() => {
      const prevButton = screen.getByRole('button', { name: /previous/i });
      fireEvent.click(prevButton);
    });
    
    // Should be back on configuration step
    await waitFor(() => {
      expect(screen.getByText('Deployment Configuration')).toBeInTheDocument();
    });
  });

  it('closes modal when close button is clicked', () => {
    render(<StrategyDeploymentPipeline {...mockProps} />);
    
    const closeButton = screen.getByText('Close');
    fireEvent.click(closeButton);
    
    expect(mockProps.onClose).toHaveBeenCalled();
  });
});