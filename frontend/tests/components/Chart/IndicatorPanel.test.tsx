import { describe, it, expect, beforeEach, vi } from 'vitest'
import { render, screen, fireEvent, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { IndicatorPanel } from '../../../src/components/Chart/IndicatorPanel'
import { useChartStore } from '../../../src/components/Chart/hooks/useChartStore'

// Mock the chart store
vi.mock('../../../src/components/Chart/hooks/useChartStore')

const mockUseChartStore = vi.mocked(useChartStore)

describe('IndicatorPanel', () => {
  const mockAddIndicator = vi.fn()
  const mockRemoveIndicator = vi.fn()
  const mockUpdateIndicator = vi.fn()

  beforeEach(() => {
    vi.clearAllMocks()
    
    mockUseChartStore.mockReturnValue({
      indicators: [],
      addIndicator: mockAddIndicator,
      removeIndicator: mockRemoveIndicator,
      updateIndicator: mockUpdateIndicator,
      currentInstrument: null,
      timeframe: '1h',
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
      setTimeframe: vi.fn(),
      setChartData: vi.fn(),
      updateSettings: vi.fn(),
      setLoading: vi.fn(),
      setError: vi.fn(),
      toggleRealTimeUpdates: vi.fn()
    })
  })

  it('renders indicator panel with collapsed state by default', () => {
    render(<IndicatorPanel />)
    
    expect(screen.getByText('📊 Indicators (0)')).toBeInTheDocument()
    expect(screen.queryByText('Add Indicator')).not.toBeInTheDocument()
  })

  it('expands panel when toggle button is clicked', async () => {
    const user = userEvent.setup()
    render(<IndicatorPanel />)
    
    const toggleButton = screen.getByLabelText('Expand indicators')
    await user.click(toggleButton)
    
    expect(screen.getByText('Add Indicator')).toBeInTheDocument()
    expect(screen.getByText('Add SMA(20)')).toBeInTheDocument()
  })

  it('adds SMA indicator when form is submitted', async () => {
    const user = userEvent.setup()
    render(<IndicatorPanel />)
    
    // Expand panel
    await user.click(screen.getByLabelText('Expand indicators'))
    
    // Submit form with default values
    await user.click(screen.getByText('Add SMA(20)'))
    
    expect(mockAddIndicator).toHaveBeenCalledWith(
      expect.objectContaining({
        type: 'SMA',
        period: 20,
        visible: true,
        color: expect.any(String)
      })
    )
  })

  it('adds EMA indicator when type is changed', async () => {
    const user = userEvent.setup()
    render(<IndicatorPanel />)
    
    // Expand panel
    await user.click(screen.getByLabelText('Expand indicators'))
    
    // Change indicator type to EMA
    const typeSelect = screen.getByDisplayValue('Simple Moving Average (SMA)')
    await user.selectOptions(typeSelect, 'EMA')
    
    // Submit form
    await user.click(screen.getByText('Add EMA(20)'))
    
    expect(mockAddIndicator).toHaveBeenCalledWith(
      expect.objectContaining({
        type: 'EMA',
        period: 20
      })
    )
  })

  it('allows custom period input', async () => {
    const user = userEvent.setup()
    render(<IndicatorPanel />)
    
    // Expand panel
    await user.click(screen.getByLabelText('Expand indicators'))
    
    // Change period
    const periodInput = screen.getByLabelText('Period:')
    await user.clear(periodInput)
    await user.type(periodInput, '50')
    
    // Wait for state update and find button
    await waitFor(() => {
      expect(screen.getByText('Add SMA(50)')).toBeInTheDocument()
    })
    
    await user.click(screen.getByText('Add SMA(50)'))
    
    expect(mockAddIndicator).toHaveBeenCalledWith(
      expect.objectContaining({
        period: 50
      })
    )
  })

  it('displays active indicators when present', () => {
    const mockIndicators = [
      {
        id: 'sma_20_1',
        type: 'SMA' as const,
        period: 20,
        color: '#FF6B6B',
        visible: true
      },
      {
        id: 'ema_50_2',
        type: 'EMA' as const,
        period: 50,
        color: '#4ECDC4',
        visible: false
      }
    ]

    mockUseChartStore.mockReturnValue({
      ...mockUseChartStore(),
      indicators: mockIndicators
    })

    render(<IndicatorPanel />)
    
    // Should show count in header
    expect(screen.getByText('📊 Indicators (2)')).toBeInTheDocument()
  })

  it('shows active indicators list when expanded', async () => {
    const user = userEvent.setup()
    const mockIndicators = [
      {
        id: 'sma_20_1',
        type: 'SMA' as const,
        period: 20,
        color: '#FF6B6B',
        visible: true
      }
    ]

    mockUseChartStore.mockReturnValue({
      ...mockUseChartStore(),
      indicators: mockIndicators
    })

    render(<IndicatorPanel />)
    
    // Expand panel
    await user.click(screen.getByLabelText('Expand indicators'))
    
    expect(screen.getByText('Active Indicators')).toBeInTheDocument()
    expect(screen.getByText('SMA(20)')).toBeInTheDocument()
  })

  it('removes indicator when remove button is clicked', async () => {
    const user = userEvent.setup()
    const mockIndicators = [
      {
        id: 'sma_20_1',
        type: 'SMA' as const,
        period: 20,
        color: '#FF6B6B',
        visible: true
      }
    ]

    mockUseChartStore.mockReturnValue({
      ...mockUseChartStore(),
      indicators: mockIndicators
    })

    render(<IndicatorPanel />)
    
    // Expand panel
    await user.click(screen.getByLabelText('Expand indicators'))
    
    // Click remove button
    const removeButton = screen.getByTitle('Remove indicator')
    await user.click(removeButton)
    
    expect(mockRemoveIndicator).toHaveBeenCalledWith('sma_20_1')
  })

  it('toggles indicator visibility', async () => {
    const user = userEvent.setup()
    const mockIndicators = [
      {
        id: 'sma_20_1',
        type: 'SMA' as const,
        period: 20,
        color: '#FF6B6B',
        visible: true
      }
    ]

    mockUseChartStore.mockReturnValue({
      ...mockUseChartStore(),
      indicators: mockIndicators
    })

    render(<IndicatorPanel />)
    
    // Expand panel
    await user.click(screen.getByLabelText('Expand indicators'))
    
    // Click visibility toggle
    const visibilityButton = screen.getByTitle('Hide indicator')
    await user.click(visibilityButton)
    
    expect(mockUpdateIndicator).toHaveBeenCalledWith('sma_20_1', { visible: false })
  })

  it('updates indicator color', async () => {
    const user = userEvent.setup()
    const mockIndicators = [
      {
        id: 'sma_20_1',
        type: 'SMA' as const,
        period: 20,
        color: '#FF6B6B',
        visible: true
      }
    ]

    mockUseChartStore.mockReturnValue({
      ...mockUseChartStore(),
      indicators: mockIndicators
    })

    render(<IndicatorPanel />)
    
    // Expand panel
    await user.click(screen.getByLabelText('Expand indicators'))
    
    // Change color
    const colorPicker = screen.getByTitle('Change color')
    fireEvent.change(colorPicker, { target: { value: '#00FF00' } })
    
    expect(mockUpdateIndicator).toHaveBeenCalledWith('sma_20_1', { color: '#00FF00' })
  })

  it('updates indicator period', async () => {
    const user = userEvent.setup()
    const mockIndicators = [
      {
        id: 'sma_20_1',
        type: 'SMA' as const,
        period: 20,
        color: '#FF6B6B',
        visible: true
      }
    ]

    mockUseChartStore.mockReturnValue({
      ...mockUseChartStore(),
      indicators: mockIndicators
    })

    render(<IndicatorPanel />)
    
    // Expand panel
    await user.click(screen.getByLabelText('Expand indicators'))
    
    // Change period
    const periodEdit = screen.getByTitle('Edit period')
    await user.clear(periodEdit)
    await user.type(periodEdit, '50')
    
    expect(mockUpdateIndicator).toHaveBeenCalledWith('sma_20_1', { period: 50 })
  })

  it('shows empty state when no indicators are present', async () => {
    const user = userEvent.setup()
    render(<IndicatorPanel />)
    
    // Expand panel
    await user.click(screen.getByLabelText('Expand indicators'))
    
    expect(screen.getByText('No indicators added yet. Add your first technical indicator above.')).toBeInTheDocument()
  })

  it('validates period input bounds', async () => {
    const user = userEvent.setup()
    render(<IndicatorPanel />)
    
    // Expand panel
    await user.click(screen.getByLabelText('Expand indicators'))
    
    // Try to set period to 0
    const periodInput = screen.getByLabelText('Period:')
    await user.clear(periodInput)
    await user.type(periodInput, '0')
    
    // Submit form - should default to 1
    await user.click(screen.getByText('Add SMA(1)'))
    
    expect(mockAddIndicator).toHaveBeenCalledWith(
      expect.objectContaining({
        period: 1
      })
    )
  })

  it('applies custom className', () => {
    const { container } = render(<IndicatorPanel className="custom-class" />)
    
    expect(container.firstChild).toHaveClass('indicator-panel', 'custom-class')
  })
})