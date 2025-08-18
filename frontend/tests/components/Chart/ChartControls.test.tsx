import { describe, it, expect, beforeEach, vi } from 'vitest'
import { render, screen, fireEvent } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { ChartControls } from '../../../src/components/Chart/ChartControls'
import { useChartStore } from '../../../src/components/Chart/hooks/useChartStore'

// Mock the chart store
vi.mock('../../../src/components/Chart/hooks/useChartStore')

const mockUseChartStore = vi.mocked(useChartStore)

describe('ChartControls', () => {
  const mockUpdateSettings = vi.fn()
  const mockToggleRealTimeUpdates = vi.fn()
  const mockOnZoomIn = vi.fn()
  const mockOnZoomOut = vi.fn()
  const mockOnZoomFit = vi.fn()
  const mockOnToggleFullscreen = vi.fn()
  const mockOnResetChart = vi.fn()

  const defaultSettings = {
    timeframe: '1h' as const,
    showVolume: true,
    indicators: [],
    crosshair: true,
    grid: true,
    timezone: 'UTC'
  }

  beforeEach(() => {
    vi.clearAllMocks()
    
    mockUseChartStore.mockReturnValue({
      settings: defaultSettings,
      realTimeUpdates: true,
      updateSettings: mockUpdateSettings,
      toggleRealTimeUpdates: mockToggleRealTimeUpdates,
      indicators: [],
      currentInstrument: null,
      timeframe: '1h',
      chartData: { candles: [], volume: [] },
      isLoading: false,
      error: null,
      setCurrentInstrument: vi.fn(),
      setTimeframe: vi.fn(),
      addIndicator: vi.fn(),
      removeIndicator: vi.fn(),
      updateIndicator: vi.fn(),
      setChartData: vi.fn(),
      setLoading: vi.fn(),
      setError: vi.fn()
    })
  })

  it('renders all control buttons', () => {
    render(
      <ChartControls
        onZoomIn={mockOnZoomIn}
        onZoomOut={mockOnZoomOut}
        onZoomFit={mockOnZoomFit}
        onToggleFullscreen={mockOnToggleFullscreen}
        onResetChart={mockOnResetChart}
      />
    )
    
    expect(screen.getByTitle('Zoom In (Ctrl/Cmd + +)')).toBeInTheDocument()
    expect(screen.getByTitle('Zoom Out (Ctrl/Cmd + -)')).toBeInTheDocument()
    expect(screen.getByTitle('Zoom to Fit (Ctrl/Cmd + 0)')).toBeInTheDocument()
    expect(screen.getByTitle('Reset Chart (R)')).toBeInTheDocument()
    expect(screen.getByTitle('Toggle Volume (V)')).toBeInTheDocument()
    expect(screen.getByTitle('Toggle Crosshair (C)')).toBeInTheDocument()
    expect(screen.getByTitle('Toggle Grid (G)')).toBeInTheDocument()
    expect(screen.getByTitle('Toggle Real-time Updates')).toBeInTheDocument()
    expect(screen.getByTitle('Chart Settings')).toBeInTheDocument()
    expect(screen.getByTitle('Toggle Fullscreen (F)')).toBeInTheDocument()
  })

  it('calls zoom handlers when buttons are clicked', async () => {
    const user = userEvent.setup()
    render(
      <ChartControls
        onZoomIn={mockOnZoomIn}
        onZoomOut={mockOnZoomOut}
        onZoomFit={mockOnZoomFit}
      />
    )
    
    await user.click(screen.getByTitle('Zoom In (Ctrl/Cmd + +)'))
    expect(mockOnZoomIn).toHaveBeenCalledTimes(1)
    
    await user.click(screen.getByTitle('Zoom Out (Ctrl/Cmd + -)'))
    expect(mockOnZoomOut).toHaveBeenCalledTimes(1)
    
    await user.click(screen.getByTitle('Zoom to Fit (Ctrl/Cmd + 0)'))
    expect(mockOnZoomFit).toHaveBeenCalledTimes(1)
  })

  it('calls reset and fullscreen handlers', async () => {
    const user = userEvent.setup()
    render(
      <ChartControls
        onResetChart={mockOnResetChart}
        onToggleFullscreen={mockOnToggleFullscreen}
      />
    )
    
    await user.click(screen.getByTitle('Reset Chart (R)'))
    expect(mockOnResetChart).toHaveBeenCalledTimes(1)
    
    await user.click(screen.getByTitle('Toggle Fullscreen (F)'))
    expect(mockOnToggleFullscreen).toHaveBeenCalledTimes(1)
  })

  it('toggles volume display', async () => {
    const user = userEvent.setup()
    render(<ChartControls />)
    
    const volumeButton = screen.getByTitle('Toggle Volume (V)')
    await user.click(volumeButton)
    
    expect(mockUpdateSettings).toHaveBeenCalledWith({ showVolume: false })
  })

  it('toggles crosshair display', async () => {
    const user = userEvent.setup()
    render(<ChartControls />)
    
    const crosshairButton = screen.getByTitle('Toggle Crosshair (C)')
    await user.click(crosshairButton)
    
    expect(mockUpdateSettings).toHaveBeenCalledWith({ crosshair: false })
  })

  it('toggles grid display', async () => {
    const user = userEvent.setup()
    render(<ChartControls />)
    
    const gridButton = screen.getByTitle('Toggle Grid (G)')
    await user.click(gridButton)
    
    expect(mockUpdateSettings).toHaveBeenCalledWith({ grid: false })
  })

  it('toggles real-time updates', async () => {
    const user = userEvent.setup()
    render(<ChartControls />)
    
    const liveButton = screen.getByTitle('Toggle Real-time Updates')
    await user.click(liveButton)
    
    expect(mockToggleRealTimeUpdates).toHaveBeenCalledTimes(1)
  })

  it('shows correct button states based on settings', () => {
    mockUseChartStore.mockReturnValue({
      ...mockUseChartStore(),
      settings: {
        ...defaultSettings,
        showVolume: false,
        crosshair: false,
        grid: false
      },
      realTimeUpdates: false
    })

    render(<ChartControls />)
    
    const volumeButton = screen.getByTitle('Toggle Volume (V)')
    const crosshairButton = screen.getByTitle('Toggle Crosshair (C)')
    const gridButton = screen.getByTitle('Toggle Grid (G)')
    const liveButton = screen.getByTitle('Toggle Real-time Updates')
    
    expect(volumeButton).not.toHaveClass('active')
    expect(crosshairButton).not.toHaveClass('active')
    expect(gridButton).not.toHaveClass('active')
    expect(liveButton).not.toHaveClass('active')
  })

  it('opens settings panel when settings button is clicked', async () => {
    const user = userEvent.setup()
    render(<ChartControls />)
    
    const settingsButton = screen.getByTitle('Chart Settings')
    await user.click(settingsButton)
    
    expect(screen.getByText('Chart Settings')).toBeInTheDocument()
    expect(screen.getByText('Timezone:')).toBeInTheDocument()
    expect(screen.getByText('Show Volume')).toBeInTheDocument()
    expect(screen.getByText('Keyboard Shortcuts')).toBeInTheDocument()
  })

  it('changes timezone in settings panel', async () => {
    const user = userEvent.setup()
    render(<ChartControls />)
    
    // Open settings panel
    await user.click(screen.getByTitle('Chart Settings'))
    
    // Change timezone
    const timezoneSelect = screen.getByDisplayValue('UTC')
    await user.selectOptions(timezoneSelect, 'America/New_York')
    
    expect(mockUpdateSettings).toHaveBeenCalledWith({ timezone: 'America/New_York' })
  })

  it('handles keyboard shortcuts for zoom controls', () => {
    render(
      <ChartControls
        onZoomIn={mockOnZoomIn}
        onZoomOut={mockOnZoomOut}
        onZoomFit={mockOnZoomFit}
      />
    )
    
    // Test Ctrl + Plus
    fireEvent.keyDown(document, { key: '+', ctrlKey: true })
    expect(mockOnZoomIn).toHaveBeenCalledTimes(1)
    
    // Test Ctrl + Minus
    fireEvent.keyDown(document, { key: '-', ctrlKey: true })
    expect(mockOnZoomOut).toHaveBeenCalledTimes(1)
    
    // Test Ctrl + 0
    fireEvent.keyDown(document, { key: '0', ctrlKey: true })
    expect(mockOnZoomFit).toHaveBeenCalledTimes(1)
  })

  it('handles keyboard shortcuts for view controls', () => {
    render(<ChartControls onToggleFullscreen={mockOnToggleFullscreen} onResetChart={mockOnResetChart} />)
    
    // Test F key for fullscreen
    fireEvent.keyDown(document, { key: 'f' })
    expect(mockOnToggleFullscreen).toHaveBeenCalledTimes(1)
    
    // Test R key for reset
    fireEvent.keyDown(document, { key: 'r' })
    expect(mockOnResetChart).toHaveBeenCalledTimes(1)
    
    // Test V key for volume
    fireEvent.keyDown(document, { key: 'v' })
    expect(mockUpdateSettings).toHaveBeenCalledWith({ showVolume: false })
    
    // Test C key for crosshair
    fireEvent.keyDown(document, { key: 'c' })
    expect(mockUpdateSettings).toHaveBeenCalledWith({ crosshair: false })
    
    // Test G key for grid
    fireEvent.keyDown(document, { key: 'g' })
    expect(mockUpdateSettings).toHaveBeenCalledWith({ grid: false })
  })

  it('prevents keyboard shortcuts when modifier keys are pressed', () => {
    render(<ChartControls onToggleFullscreen={mockOnToggleFullscreen} />)
    
    // Test F key with Ctrl - should not trigger fullscreen
    fireEvent.keyDown(document, { key: 'f', ctrlKey: true })
    expect(mockOnToggleFullscreen).not.toHaveBeenCalled()
    
    // Test F key with Shift - should not trigger fullscreen
    fireEvent.keyDown(document, { key: 'f', shiftKey: true })
    expect(mockOnToggleFullscreen).not.toHaveBeenCalled()
  })

  it('shows keyboard shortcuts in settings panel', async () => {
    const user = userEvent.setup()
    render(<ChartControls />)
    
    // Open settings panel
    await user.click(screen.getByTitle('Chart Settings'))
    
    expect(screen.getByText('Keyboard Shortcuts')).toBeInTheDocument()
    expect(screen.getByText('Ctrl/Cmd + +')).toBeInTheDocument()
    expect(screen.getByText('Zoom In')).toBeInTheDocument()
    expect(screen.getByText('F')).toBeInTheDocument()
    expect(screen.getByText('Fullscreen')).toBeInTheDocument()
  })

  it('applies custom className', () => {
    const { container } = render(<ChartControls className="custom-class" />)
    
    expect(container.firstChild).toHaveClass('chart-controls', 'custom-class')
  })

  it('toggles settings panel visibility', async () => {
    const user = userEvent.setup()
    render(<ChartControls />)
    
    const settingsButton = screen.getByTitle('Chart Settings')
    
    // Open settings panel
    await user.click(settingsButton)
    expect(screen.getByText('Chart Settings')).toBeInTheDocument()
    
    // Close settings panel
    await user.click(settingsButton)
    expect(screen.queryByText('Chart Settings')).not.toBeInTheDocument()
  })

  it('shows live indicator with pulse animation when real-time updates are active', () => {
    mockUseChartStore.mockReturnValue({
      ...mockUseChartStore(),
      realTimeUpdates: true
    })

    render(<ChartControls />)
    
    const liveButton = screen.getByTitle('Toggle Real-time Updates')
    expect(liveButton).toHaveClass('active', 'real-time')
    expect(liveButton).toHaveTextContent('🟢 Live')
  })

  it('shows paused indicator when real-time updates are disabled', () => {
    mockUseChartStore.mockReturnValue({
      ...mockUseChartStore(),
      realTimeUpdates: false
    })

    render(<ChartControls />)
    
    const liveButton = screen.getByTitle('Toggle Real-time Updates')
    expect(liveButton).not.toHaveClass('active')
    expect(liveButton).toHaveTextContent('⏸️ Live')
  })
})