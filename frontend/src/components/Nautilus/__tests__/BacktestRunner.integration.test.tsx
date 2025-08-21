/**
 * Integration test suite for BacktestRunner component
 * Tests the complete workflow from configuration to results
 */

import React from 'react'
import { render, screen, fireEvent, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest'
import BacktestRunner from '../BacktestRunner'

// Mock the child components
vi.mock('../BacktestConfiguration', () => ({
  default: ({ onConfigChange }: any) => {
    React.useEffect(() => {
      // Simulate valid config being set
      onConfigChange({
        strategyClass: 'MovingAverageCross',
        strategyConfig: { fast_period: 10, slow_period: 20 },
        startDate: '2023-01-01',
        endDate: '2023-06-30',
        instruments: ['AAPL'],
        venues: ['NASDAQ'],
        initialBalance: 100000,
        baseCurrency: 'USD',
        dataConfiguration: { dataType: 'bar' }
      }, true)
    }, [])
    
    return (
      <div data-testid="backtest-configuration">
        <h3>Backtest Configuration Mock</h3>
      </div>
    )
  }
}))

vi.mock('../BacktestResults', () => ({
  default: ({ backtest, onExport, onCompare }: any) => (
    <div data-testid="backtest-results">
      <h3>Backtest Results Mock</h3>
      <div>Backtest ID: {backtest.backtestId}</div>
      <button onClick={() => onExport('pdf')}>Export PDF</button>
      <button onClick={() => onCompare([backtest.backtestId])}>Compare</button>
    </div>
  )
}))

// Mock backtestService
const mockBacktestService = {
  startBacktest: vi.fn(),
  listBacktests: vi.fn(),
  getBacktestResults: vi.fn(),
  cancelBacktest: vi.fn(),
  deleteBacktest: vi.fn()
}

vi.mock('../../../services/backtestService', () => ({
  backtestService: mockBacktestService,
  BacktestProgressMonitor: class MockProgressMonitor {
    subscribeToBacktest = vi.fn()
    unsubscribeFromBacktest = vi.fn()
    destroy = vi.fn()
  }
}))

// Mock fetch globally
global.fetch = vi.fn()

describe('BacktestRunner Integration', () => {
  beforeEach(() => {
    vi.clearAllMocks()
    
    // Default mock implementations
    mockBacktestService.listBacktests.mockResolvedValue([
      {
        backtestId: 'existing-backtest-1',
        status: 'completed',
        startTime: '2023-01-01T09:00:00Z',
        config: { strategyClass: 'TestStrategy' },
        metrics: { totalReturn: 10.5 }
      },
      {
        backtestId: 'existing-backtest-2',
        status: 'running',
        startTime: '2023-01-02T09:00:00Z',
        config: { strategyClass: 'AnotherStrategy' },
        progress: { percentage: 45 }
      }
    ])
    
    mockBacktestService.startBacktest.mockResolvedValue({
      backtestId: 'new-backtest-123',
      status: 'queued'
    })
    
    mockBacktestService.getBacktestResults.mockResolvedValue({
      backtestId: 'existing-backtest-1',
      status: 'completed',
      metrics: { totalReturn: 10.5 },
      trades: [],
      equityCurve: []
    })
  })

  afterEach(() => {
    vi.clearAllTimers()
  })

  const renderComponent = () => {
    return render(<BacktestRunner />)
  }

  it('should render the main interface', async () => {
    renderComponent()
    
    expect(screen.getByText('Backtesting Engine')).toBeInTheDocument()
    expect(screen.getByText('Run and analyze historical strategy backtests with real-time monitoring')).toBeInTheDocument()
  })

  it('should load existing backtests on mount', async () => {
    renderComponent()
    
    await waitFor(() => {
      expect(mockBacktestService.listBacktests).toHaveBeenCalled()
    })
    
    expect(screen.getByText(/existing-backtest-1/)).toBeInTheDocument()
    expect(screen.getByText(/existing-backtest-2/)).toBeInTheDocument()
  })

  it('should display summary statistics correctly', async () => {
    renderComponent()
    
    await waitFor(() => {
      expect(screen.getByText('Total Backtests')).toBeInTheDocument()
      expect(screen.getByText('2')).toBeInTheDocument() // Total count
      expect(screen.getByText('Completed')).toBeInTheDocument()
      expect(screen.getByText('1')).toBeInTheDocument() // Completed count
      expect(screen.getByText('Running')).toBeInTheDocument()
      expect(screen.getByText('1')).toBeInTheDocument() // Running count
    })
  })

  it('should open configuration drawer when configure button is clicked', async () => {
    renderComponent()
    
    const configureButton = screen.getByText('Configure Backtest')
    await userEvent.click(configureButton)
    
    expect(screen.getByText('Backtest Configuration')).toBeInTheDocument()
    expect(screen.getByTestId('backtest-configuration')).toBeInTheDocument()
  })

  it('should enable start button when valid configuration is provided', async () => {
    renderComponent()
    
    const configureButton = screen.getByText('Configure Backtest')
    await userEvent.click(configureButton)
    
    await waitFor(() => {
      const startButton = screen.getByText('Start Backtest')
      expect(startButton).not.toBeDisabled()
    })
  })

  it('should start a backtest when start button is clicked', async () => {
    renderComponent()
    
    // Open configuration drawer
    const configureButton = screen.getByText('Configure Backtest')
    await userEvent.click(configureButton)
    
    await waitFor(() => {
      const startButton = screen.getByText('Start Backtest')
      expect(startButton).not.toBeDisabled()
    })
    
    // Start backtest
    const startButton = screen.getByText('Start Backtest')
    await userEvent.click(startButton)
    
    await waitFor(() => {
      expect(mockBacktestService.startBacktest).toHaveBeenCalledWith(
        expect.objectContaining({
          strategyClass: 'MovingAverageCross',
          instruments: ['AAPL']
        })
      )
    })
  })

  it('should close configuration drawer after starting backtest', async () => {
    renderComponent()
    
    // Open configuration drawer
    const configureButton = screen.getByText('Configure Backtest')
    await userEvent.click(configureButton)
    
    // Start backtest
    const startButton = screen.getByText('Start Backtest')
    await userEvent.click(startButton)
    
    await waitFor(() => {
      expect(screen.queryByTestId('backtest-configuration')).not.toBeInTheDocument()
    })
  })

  it('should display progress for running backtests', async () => {
    renderComponent()
    
    await waitFor(() => {
      // Should show progress for running backtest
      expect(screen.getByText('running')).toBeInTheDocument()
    })
  })

  it('should allow viewing results for completed backtests', async () => {
    renderComponent()
    
    await waitFor(() => {
      const viewButtons = screen.getAllByTitle('View Results')
      expect(viewButtons.length).toBeGreaterThan(0)
    })
    
    // Click view results for the completed backtest
    const viewButton = screen.getAllByTitle('View Results')[0]
    await userEvent.click(viewButton)
    
    await waitFor(() => {
      expect(mockBacktestService.getBacktestResults).toHaveBeenCalledWith('existing-backtest-1')
    })
  })

  it('should open results drawer when viewing results', async () => {
    renderComponent()
    
    await waitFor(() => {
      const viewButton = screen.getAllByTitle('View Results')[0]
      userEvent.click(viewButton)
    })
    
    await waitFor(() => {
      expect(screen.getByText('Backtest Results')).toBeInTheDocument()
      expect(screen.getByTestId('backtest-results')).toBeInTheDocument()
    })
  })

  it('should allow cancelling running backtests', async () => {
    mockBacktestService.cancelBacktest.mockResolvedValue(true)
    
    renderComponent()
    
    await waitFor(() => {
      const cancelButtons = screen.getAllByTitle('Cancel Backtest')
      expect(cancelButtons.length).toBeGreaterThan(0)
    })
    
    // Click cancel for the running backtest
    const cancelButton = screen.getAllByTitle('Cancel Backtest')[0]
    await userEvent.click(cancelButton)
    
    await waitFor(() => {
      expect(mockBacktestService.cancelBacktest).toHaveBeenCalled()
    })
  })

  it('should allow deleting backtests', async () => {
    mockBacktestService.deleteBacktest.mockResolvedValue(true)
    
    renderComponent()
    
    await waitFor(() => {
      const deleteButtons = screen.getAllByTitle('Delete Backtest')
      expect(deleteButtons.length).toBeGreaterThan(0)
    })
    
    // Click delete for a backtest
    const deleteButton = screen.getAllByTitle('Delete Backtest')[0]
    await userEvent.click(deleteButton)
    
    await waitFor(() => {
      expect(mockBacktestService.deleteBacktest).toHaveBeenCalled()
    })
  })

  it('should refresh backtest list when refresh button is clicked', async () => {
    renderComponent()
    
    const refreshButton = screen.getByText('Refresh')
    await userEvent.click(refreshButton)
    
    await waitFor(() => {
      expect(mockBacktestService.listBacktests).toHaveBeenCalledTimes(2) // Once on mount, once on refresh
    })
  })

  it('should show active backtests count', async () => {
    renderComponent()
    
    await waitFor(() => {
      expect(screen.getByText('Active Backtests')).toBeInTheDocument()
      // Should show count of running backtests
      expect(screen.getByDisplayValue('1')).toBeInTheDocument()
    })
  })

  it('should handle export functionality in results', async () => {
    renderComponent()
    
    // Open results for a completed backtest
    await waitFor(() => {
      const viewButton = screen.getAllByTitle('View Results')[0]
      userEvent.click(viewButton)
    })
    
    await waitFor(() => {
      const exportButton = screen.getByText('Export PDF')
      userEvent.click(exportButton)
    })
    
    // Should show export message
    await waitFor(() => {
      expect(screen.getByText(/Exporting backtest results as PDF/)).toBeInTheDocument()
    })
  })

  it('should handle comparison functionality in results', async () => {
    renderComponent()
    
    // Open results for a completed backtest
    await waitFor(() => {
      const viewButton = screen.getAllByTitle('View Results')[0]
      userEvent.click(viewButton)
    })
    
    await waitFor(() => {
      const compareButton = screen.getByText('Compare')
      userEvent.click(compareButton)
    })
    
    // Should show comparison message
    await waitFor(() => {
      expect(screen.getByText(/Comparing 1 backtests/)).toBeInTheDocument()
    })
  })

  it('should show loading state while starting backtest', async () => {
    // Make startBacktest hang to simulate loading
    mockBacktestService.startBacktest.mockImplementation(() => new Promise(() => {}))
    
    renderComponent()
    
    const configureButton = screen.getByText('Configure Backtest')
    await userEvent.click(configureButton)
    
    const startButton = screen.getByText('Start Backtest')
    await userEvent.click(startButton)
    
    // Should show loading state
    expect(startButton).toBeDisabled()
  })

  it('should handle API errors gracefully', async () => {
    mockBacktestService.startBacktest.mockRejectedValue(new Error('API Error'))
    
    renderComponent()
    
    const configureButton = screen.getByText('Configure Backtest')
    await userEvent.click(configureButton)
    
    const startButton = screen.getByText('Start Backtest')
    await userEvent.click(startButton)
    
    await waitFor(() => {
      expect(screen.getByText(/Failed to start backtest/)).toBeInTheDocument()
    })
  })

  it('should poll for updates on running backtests', async () => {
    vi.useFakeTimers()
    
    renderComponent()
    
    // Fast-forward 5 seconds (polling interval)
    vi.advanceTimersByTime(5000)
    
    await waitFor(() => {
      expect(mockBacktestService.listBacktests).toHaveBeenCalledTimes(2) // Initial + poll
    })
    
    vi.useRealTimers()
  })

  it('should show strategy information in backtest list', async () => {
    renderComponent()
    
    await waitFor(() => {
      expect(screen.getByText('TestStrategy')).toBeInTheDocument()
      expect(screen.getByText('AnotherStrategy')).toBeInTheDocument()
    })
  })

  it('should display backtest duration correctly', async () => {
    renderComponent()
    
    await waitFor(() => {
      // Should show duration for completed backtests
      expect(screen.getByText(/\d+m/)).toBeInTheDocument() // Minutes format
    })
  })

  it('should show pagination in backtest table', async () => {
    // Mock more backtests to trigger pagination
    mockBacktestService.listBacktests.mockResolvedValue(
      Array.from({ length: 15 }, (_, i) => ({
        backtestId: `backtest-${i}`,
        status: 'completed',
        startTime: '2023-01-01T09:00:00Z',
        config: { strategyClass: 'TestStrategy' }
      }))
    )
    
    renderComponent()
    
    await waitFor(() => {
      expect(screen.getByText(/1-10 of 15 backtests/)).toBeInTheDocument()
    })
  })
})

// Error handling integration tests
describe('BacktestRunner Error Handling', () => {
  beforeEach(() => {
    vi.clearAllMocks()
  })

  it('should handle network errors gracefully', async () => {
    mockBacktestService.listBacktests.mockRejectedValue(new Error('Network Error'))
    
    render(<BacktestRunner />)
    
    await waitFor(() => {
      expect(screen.getByText(/Failed to load backtests/)).toBeInTheDocument()
    })
  })

  it('should handle empty backtest list', async () => {
    mockBacktestService.listBacktests.mockResolvedValue([])
    
    render(<BacktestRunner />)
    
    await waitFor(() => {
      expect(screen.getByText('Total Backtests')).toBeInTheDocument()
      expect(screen.getByText('0')).toBeInTheDocument()
    })
  })

  it('should handle malformed backtest data', async () => {
    mockBacktestService.listBacktests.mockResolvedValue([
      {
        backtestId: 'malformed-backtest',
        // Missing required fields
      }
    ])
    
    render(<BacktestRunner />)
    
    // Should render without crashing
    await waitFor(() => {
      expect(screen.getByText('Backtesting Engine')).toBeInTheDocument()
    })
  })
})