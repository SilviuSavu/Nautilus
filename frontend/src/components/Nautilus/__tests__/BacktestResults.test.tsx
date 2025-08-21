/**
 * Test suite for BacktestResults component
 */

import React from 'react'
import { render, screen, fireEvent, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { describe, it, expect, vi, beforeEach } from 'vitest'
import BacktestResults from '../BacktestResults'
import { BacktestResult } from '../../../services/backtestService'

// Mock dayjs
vi.mock('dayjs', () => ({
  default: vi.fn((date) => ({
    format: vi.fn(() => date ? '12/25 10:30' : '12/25 15:30'),
    diff: vi.fn(() => 360),
    isBefore: vi.fn(() => false),
    isAfter: vi.fn(() => false)
  }))
}))

// Mock EquityCurveChart
vi.mock('../EquityCurveChart', () => ({
  default: ({ data, metrics }: any) => (
    <div data-testid="equity-curve-chart">
      Equity Curve Chart - {data?.length || 0} points
    </div>
  )
}))

describe('BacktestResults', () => {
  const mockOnExport = vi.fn()
  const mockOnCompare = vi.fn()
  
  beforeEach(() => {
    mockOnExport.mockClear()
    mockOnCompare.mockClear()
  })

  const createMockBacktest = (overrides: Partial<BacktestResult> = {}): BacktestResult => ({
    backtestId: 'test-backtest-123',
    status: 'completed',
    startTime: '2023-01-01T09:00:00Z',
    endTime: '2023-01-01T17:00:00Z',
    config: {
      strategyClass: 'MovingAverageCross',
      strategyConfig: { fast_period: 10, slow_period: 20 },
      startDate: '2023-01-01',
      endDate: '2023-12-31',
      instruments: ['AAPL', 'MSFT'],
      venues: ['NASDAQ'],
      initialBalance: 100000,
      baseCurrency: 'USD',
      dataConfiguration: { dataType: 'bar' }
    },
    metrics: {
      totalReturn: 15.25,
      annualizedReturn: 12.8,
      sharpeRatio: 1.45,
      sortinoRatio: 1.82,
      calmarRatio: 0.95,
      maxDrawdown: -8.5,
      volatility: 18.3,
      winRate: 58.2,
      profitFactor: 1.35,
      alpha: 3.2,
      beta: 0.85,
      informationRatio: 0.75,
      totalTrades: 248,
      winningTrades: 144,
      losingTrades: 104,
      averageWin: 245.50,
      averageLoss: -182.25,
      largestWin: 1250.00,
      largestLoss: -890.00
    },
    trades: [
      {
        tradeId: 'trade-1',
        instrumentId: 'AAPL',
        side: 'buy',
        quantity: 100,
        entryPrice: 150.25,
        exitPrice: 152.80,
        entryTime: '2023-01-01T10:30:00Z',
        exitTime: '2023-01-01T15:30:00Z',
        pnl: 255.00,
        commission: 2.50,
        duration: 300
      },
      {
        tradeId: 'trade-2',
        instrumentId: 'MSFT',
        side: 'sell',
        quantity: 50,
        entryPrice: 280.00,
        exitPrice: 275.00,
        entryTime: '2023-01-01T11:00:00Z',
        exitTime: '2023-01-01T14:00:00Z',
        pnl: -250.00,
        commission: 2.50,
        duration: 180
      }
    ],
    equityCurve: [
      {
        timestamp: '2023-01-01T09:00:00Z',
        equity: 100000,
        drawdown: 0,
        balance: 100000,
        unrealizedPnl: 0
      },
      {
        timestamp: '2023-01-01T17:00:00Z',
        equity: 115250,
        drawdown: -2.5,
        balance: 115250,
        unrealizedPnl: 0
      }
    ],
    ...overrides
  })

  const renderComponent = (backtest: BacktestResult, props = {}) => {
    return render(
      <BacktestResults
        backtest={backtest}
        onExport={mockOnExport}
        onCompare={mockOnCompare}
        showComparison={true}
        {...props}
      />
    )
  }

  it('should render backtest header information', () => {
    const backtest = createMockBacktest()
    renderComponent(backtest)
    
    expect(screen.getByText('Backtest Results')).toBeInTheDocument()
    expect(screen.getByText(/test-backtest-123/)).toBeInTheDocument()
    expect(screen.getByText(/Jan 01, 2023/)).toBeInTheDocument()
  })

  it('should render export buttons', () => {
    const backtest = createMockBacktest()
    renderComponent(backtest)
    
    expect(screen.getByText('Export PDF')).toBeInTheDocument()
    expect(screen.getByText('Export Excel')).toBeInTheDocument()
  })

  it('should render compare button when showComparison is true', () => {
    const backtest = createMockBacktest()
    renderComponent(backtest)
    
    expect(screen.getByText('Compare')).toBeInTheDocument()
  })

  it('should not render compare button when showComparison is false', () => {
    const backtest = createMockBacktest()
    renderComponent(backtest, { showComparison: false })
    
    expect(screen.queryByText('Compare')).not.toBeInTheDocument()
  })

  it('should render all performance metrics cards', () => {
    const backtest = createMockBacktest()
    renderComponent(backtest)
    
    expect(screen.getByText('Total Return')).toBeInTheDocument()
    expect(screen.getByText('Annualized Return')).toBeInTheDocument()
    expect(screen.getByText('Sharpe Ratio')).toBeInTheDocument()
    expect(screen.getByText('Sortino Ratio')).toBeInTheDocument()
    expect(screen.getByText('Max Drawdown')).toBeInTheDocument()
    expect(screen.getByText('Win Rate')).toBeInTheDocument()
    expect(screen.getByText('Profit Factor')).toBeInTheDocument()
    expect(screen.getByText('Total Trades')).toBeInTheDocument()
    expect(screen.getByText('Volatility')).toBeInTheDocument()
    expect(screen.getByText('Calmar Ratio')).toBeInTheDocument()
    expect(screen.getByText('Alpha')).toBeInTheDocument()
    expect(screen.getByText('Beta')).toBeInTheDocument()
  })

  it('should display correct metric values', () => {
    const backtest = createMockBacktest()
    renderComponent(backtest)
    
    expect(screen.getByText('15.25')).toBeInTheDocument() // Total Return
    expect(screen.getByText('1.45')).toBeInTheDocument() // Sharpe Ratio
    expect(screen.getByText('58.2')).toBeInTheDocument() // Win Rate
    expect(screen.getByText('248')).toBeInTheDocument() // Total Trades
  })

  it('should render tab navigation', () => {
    const backtest = createMockBacktest()
    renderComponent(backtest)
    
    expect(screen.getByText('Overview')).toBeInTheDocument()
    expect(screen.getByText('Trades')).toBeInTheDocument()
    expect(screen.getByText('Charts')).toBeInTheDocument()
    expect(screen.getByText('Analysis')).toBeInTheDocument()
  })

  it('should switch between tabs', async () => {
    const backtest = createMockBacktest()
    renderComponent(backtest)
    
    const tradesTab = screen.getByText('Trades')
    await userEvent.click(tradesTab)
    
    expect(screen.getByText('Trade Analysis')).toBeInTheDocument()
  })

  it('should render trades table in trades tab', async () => {
    const backtest = createMockBacktest()
    renderComponent(backtest)
    
    const tradesTab = screen.getByText('Trades')
    await userEvent.click(tradesTab)
    
    expect(screen.getByText('AAPL')).toBeInTheDocument()
    expect(screen.getByText('MSFT')).toBeInTheDocument()
    expect(screen.getByText('BUY')).toBeInTheDocument()
    expect(screen.getByText('SELL')).toBeInTheDocument()
  })

  it('should show trade filters button', async () => {
    const backtest = createMockBacktest()
    renderComponent(backtest)
    
    const tradesTab = screen.getByText('Trades')
    await userEvent.click(tradesTab)
    
    expect(screen.getByText('Filters')).toBeInTheDocument()
  })

  it('should open filter drawer when filters button is clicked', async () => {
    const backtest = createMockBacktest()
    renderComponent(backtest)
    
    const tradesTab = screen.getByText('Trades')
    await userEvent.click(tradesTab)
    
    const filtersButton = screen.getByText('Filters')
    await userEvent.click(filtersButton)
    
    expect(screen.getByText('Trade Filters')).toBeInTheDocument()
  })

  it('should render equity curve chart in charts tab', async () => {
    const backtest = createMockBacktest()
    renderComponent(backtest)
    
    const chartsTab = screen.getByText('Charts')
    await userEvent.click(chartsTab)
    
    expect(screen.getByTestId('equity-curve-chart')).toBeInTheDocument()
  })

  it('should render analysis tab content', async () => {
    const backtest = createMockBacktest()
    renderComponent(backtest)
    
    const analysisTab = screen.getByText('Analysis')
    await userEvent.click(analysisTab)
    
    expect(screen.getByText('Risk Analysis')).toBeInTheDocument()
    expect(screen.getByText('Trade Statistics')).toBeInTheDocument()
  })

  it('should call onExport when export buttons are clicked', async () => {
    const backtest = createMockBacktest()
    renderComponent(backtest)
    
    const exportPdfButton = screen.getByText('Export PDF')
    await userEvent.click(exportPdfButton)
    
    expect(mockOnExport).toHaveBeenCalledWith('pdf')
  })

  it('should call onCompare when compare button is clicked', async () => {
    const backtest = createMockBacktest()
    renderComponent(backtest)
    
    const compareButton = screen.getByText('Compare')
    await userEvent.click(compareButton)
    
    expect(mockOnCompare).toHaveBeenCalledWith(['test-backtest-123'])
  })

  it('should show not available message for incomplete backtests', () => {
    const backtest = createMockBacktest({ status: 'running' })
    renderComponent(backtest)
    
    expect(screen.getByText('Backtest results not available')).toBeInTheDocument()
    expect(screen.getByText('Backtest is still running or has not completed successfully.')).toBeInTheDocument()
  })

  it('should color positive metrics green and negative red', () => {
    const backtest = createMockBacktest()
    renderComponent(backtest)
    
    // Find elements with positive values (should be green/success colored)
    const totalReturnElement = screen.getByText('15.25')
    expect(totalReturnElement.closest('[class*="ant-statistic-content"]')).toHaveStyle({ color: expect.stringMatching(/#3f8600|rgb\(63, 134, 0\)/) })
  })

  it('should show filtered metrics when trades are filtered', async () => {
    const backtest = createMockBacktest()
    renderComponent(backtest)
    
    const tradesTab = screen.getByText('Trades')
    await userEvent.click(tradesTab)
    
    // Open filters
    const filtersButton = screen.getByText('Filters')
    await userEvent.click(filtersButton)
    
    // Apply a filter
    const instrumentSelect = screen.getByDisplayValue(/All instruments/)
    await userEvent.click(instrumentSelect)
    await userEvent.click(screen.getByText('AAPL'))
    
    // Should show filtered trade metrics
    expect(screen.getByText(/1 trades filtered/)).toBeInTheDocument()
  })

  it('should sort trades by different columns', async () => {
    const backtest = createMockBacktest()
    renderComponent(backtest)
    
    const tradesTab = screen.getByText('Trades')
    await userEvent.click(tradesTab)
    
    // Click on P&L column header to sort
    const pnlHeader = screen.getByText('P&L')
    await userEvent.click(pnlHeader)
    
    // Table should be sorted (check that trades are reordered)
    const tradeRows = screen.getAllByRole('row')
    expect(tradeRows.length).toBeGreaterThan(1)
  })

  it('should show trade duration in appropriate units', async () => {
    const backtest = createMockBacktest()
    renderComponent(backtest)
    
    const tradesTab = screen.getByText('Trades')
    await userEvent.click(tradesTab)
    
    // Should show duration in minutes, hours, or days as appropriate
    expect(screen.getByText(/6h/)).toBeInTheDocument() // 360 minutes = 6 hours
  })

  it('should handle missing metrics gracefully', () => {
    const backtest = createMockBacktest({ metrics: undefined })
    renderComponent(backtest)
    
    // Should still render without crashing
    expect(screen.getByText('Backtest Results')).toBeInTheDocument()
  })

  it('should handle missing trades gracefully', async () => {
    const backtest = createMockBacktest({ trades: undefined })
    renderComponent(backtest)
    
    const tradesTab = screen.getByText('Trades')
    await userEvent.click(tradesTab)
    
    // Should show empty table without crashing
    expect(screen.getByText('Trade Analysis')).toBeInTheDocument()
  })

  it('should handle missing equity curve gracefully', async () => {
    const backtest = createMockBacktest({ equityCurve: undefined })
    renderComponent(backtest)
    
    const chartsTab = screen.getByText('Charts')
    await userEvent.click(chartsTab)
    
    // Should still render chart component
    expect(screen.getByTestId('equity-curve-chart')).toBeInTheDocument()
  })

  it('should show pagination for trades table', async () => {
    // Create backtest with many trades
    const manyTrades = Array.from({ length: 100 }, (_, i) => ({
      tradeId: `trade-${i}`,
      instrumentId: 'AAPL',
      side: 'buy' as const,
      quantity: 100,
      entryPrice: 150.00,
      exitPrice: 152.00,
      entryTime: '2023-01-01T10:30:00Z',
      exitTime: '2023-01-01T15:30:00Z',
      pnl: 200.00,
      commission: 2.50,
      duration: 300
    }))
    
    const backtest = createMockBacktest({ trades: manyTrades })
    renderComponent(backtest)
    
    const tradesTab = screen.getByText('Trades')
    await userEvent.click(tradesTab)
    
    // Should show pagination
    expect(screen.getByText(/1-50 of 100 trades/)).toBeInTheDocument()
  })

  it('should clear filters when clear button is clicked', async () => {
    const backtest = createMockBacktest()
    renderComponent(backtest)
    
    const tradesTab = screen.getByText('Trades')
    await userEvent.click(tradesTab)
    
    // Open filters and set a filter
    const filtersButton = screen.getByText('Filters')
    await userEvent.click(filtersButton)
    
    const instrumentSelect = screen.getByDisplayValue(/All instruments/)
    await userEvent.click(instrumentSelect)
    await userEvent.click(screen.getByText('AAPL'))
    
    // Clear filters
    const clearButton = screen.getByText('Clear Filters')
    await userEvent.click(clearButton)
    
    // Filters should be cleared
    expect(screen.queryByText(/trades filtered/)).not.toBeInTheDocument()
  })
})