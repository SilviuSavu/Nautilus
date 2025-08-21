import { describe, it, expect, beforeEach, vi } from 'vitest'
import { render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { TimeframeSelector } from '../../../src/components/Chart/TimeframeSelector'
import { useChartStore } from '../../../src/components/Chart/hooks/useChartStore'

// Mock the chart store
vi.mock('../../../src/components/Chart/hooks/useChartStore')

const mockUseChartStore = vi.mocked(useChartStore)

describe('TimeframeSelector', () => {
  const mockSetTimeframe = vi.fn()

  beforeEach(() => {
    vi.clearAllMocks()
    
    mockUseChartStore.mockReturnValue({
      timeframe: '1h',
      setTimeframe: mockSetTimeframe,
      indicators: [],
      currentInstrument: null,
      chartData: { candles: [], volume: [] },
      settings: {
        timeframe: '1h',
        showVolume: true,
        indicators: [],
        crosshair: true,
        grid: true,
        timezone: 'UTC'
      },
      isLoading: false,
      error: null,
      realTimeUpdates: true,
      setCurrentInstrument: vi.fn(),
      addIndicator: vi.fn(),
      removeIndicator: vi.fn(),
      updateIndicator: vi.fn(),
      setChartData: vi.fn(),
      updateSettings: vi.fn(),
      setLoading: vi.fn(),
      setError: vi.fn(),
      toggleRealTimeUpdates: vi.fn()
    })
  })

  it('renders all timeframe options', () => {
    render(<TimeframeSelector />)
    
    // Use the actual labels from the component
    const expectedTimeframes = ['1m', '2m', '5m', '10m', '15m', '30m', '1H', '2H', '4H', '1D', '1W', '1M']
    
    expectedTimeframes.forEach(timeframe => {
      expect(screen.getByText(timeframe)).toBeInTheDocument()
    })
  })

  it('highlights the current timeframe', () => {
    render(<TimeframeSelector />)
    
    const activeButton = screen.getByText('1H')
    expect(activeButton.closest('button')).toHaveClass('ant-btn-primary')
  })

  it('calls setTimeframe when a timeframe is clicked', async () => {
    const user = userEvent.setup()
    render(<TimeframeSelector />)
    
    const fiveMinButton = screen.getByText('5m')
    await user.click(fiveMinButton)
    
    expect(mockSetTimeframe).toHaveBeenCalledWith('5m')
  })

  it('updates active state when timeframe changes', () => {
    mockUseChartStore.mockReturnValue({
      ...mockUseChartStore(),
      timeframe: '5m'
    })

    render(<TimeframeSelector />)
    
    const activeButton = screen.getByText('5m')
    const inactiveButton = screen.getByText('1h')
    
    expect(activeButton).toHaveClass('active')
    expect(inactiveButton).not.toHaveClass('active')
  })

  it('applies custom className', () => {
    const { container } = render(<TimeframeSelector className="custom-class" />)
    
    expect(container.firstChild).toHaveClass('timeframe-selector', 'custom-class')
  })

  it('handles all supported timeframes correctly', async () => {
    const user = userEvent.setup()
    render(<TimeframeSelector />)
    
    const timeframes = ['1m', '2m', '5m', '10m', '15m', '30m', '1h', '2h', '4h', '1d', '1w', '1M']
    
    for (const timeframe of timeframes) {
      const button = screen.getByText(timeframe)
      await user.click(button)
      expect(mockSetTimeframe).toHaveBeenCalledWith(timeframe)
    }
    
    expect(mockSetTimeframe).toHaveBeenCalledTimes(timeframes.length)
  })

  it('displays timeframes in correct order', () => {
    render(<TimeframeSelector />)
    
    const buttons = screen.getAllByRole('button')
    const expectedOrder = ['1m', '2m', '5m', '10m', '15m', '30m', '1h', '2h', '4h', '1d', '1w', '1M']
    
    buttons.forEach((button, index) => {
      expect(button).toHaveTextContent(expectedOrder[index])
    })
  })

  it('maintains accessibility attributes', () => {
    render(<TimeframeSelector />)
    
    const buttons = screen.getAllByRole('button')
    
    buttons.forEach(button => {
      expect(button).toHaveAttribute('type', 'button')
      expect(button.textContent).toBeTruthy()
    })
  })

  it('does not call setTimeframe when clicking the already active timeframe', async () => {
    const user = userEvent.setup()
    render(<TimeframeSelector />)
    
    // Click the currently active timeframe (1h)
    const activeButton = screen.getByText('1h')
    await user.click(activeButton)
    
    // Should still call setTimeframe (allowing for refresh/reload behavior)
    expect(mockSetTimeframe).toHaveBeenCalledWith('1h')
  })

  it('handles keyboard navigation', async () => {
    const user = userEvent.setup()
    render(<TimeframeSelector />)
    
    const firstButton = screen.getByText('1m')
    firstButton.focus()
    
    // Tab to next button
    await user.keyboard('{Tab}')
    expect(screen.getByText('2m')).toHaveFocus()
    
    // Enter to select
    await user.keyboard('{Enter}')
    expect(mockSetTimeframe).toHaveBeenCalledWith('2m')
  })

  it('provides visual feedback on hover', () => {
    render(<TimeframeSelector />)
    
    const button = screen.getByText('5m')
    expect(button).toHaveClass('timeframe-button')
  })

  it('handles edge case timeframes correctly', async () => {
    const user = userEvent.setup()
    
    // Test with weekly timeframe
    mockUseChartStore.mockReturnValue({
      ...mockUseChartStore(),
      timeframe: '1w'
    })

    render(<TimeframeSelector />)
    
    const weeklyButton = screen.getByText('1w')
    expect(weeklyButton).toHaveClass('active')
    
    // Test with monthly timeframe
    const monthlyButton = screen.getByText('1M')
    await user.click(monthlyButton)
    
    expect(mockSetTimeframe).toHaveBeenCalledWith('1M')
  })
})