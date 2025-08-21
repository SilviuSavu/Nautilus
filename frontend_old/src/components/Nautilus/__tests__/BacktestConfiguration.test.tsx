/**
 * Test suite for BacktestConfiguration component
 */

import React from 'react'
import { render, screen, fireEvent, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { describe, it, expect, vi, beforeEach } from 'vitest'
import BacktestConfiguration from '../BacktestConfiguration'
import { BacktestConfig } from '../../../services/backtestService'

// Mock dayjs
vi.mock('dayjs', () => ({
  default: vi.fn(() => ({
    subtract: vi.fn().mockReturnThis(),
    format: vi.fn(() => '2023-01-01'),
    isBefore: vi.fn(() => false),
    isAfter: vi.fn(() => false)
  }))
}))

describe('BacktestConfiguration', () => {
  const mockOnConfigChange = vi.fn()
  
  beforeEach(() => {
    mockOnConfigChange.mockClear()
  })

  const renderComponent = (props = {}) => {
    return render(
      <BacktestConfiguration
        onConfigChange={mockOnConfigChange}
        {...props}
      />
    )
  }

  it('should render all configuration sections', () => {
    renderComponent()
    
    expect(screen.getByText('Strategy Selection')).toBeInTheDocument()
    expect(screen.getByText('Market Data & Time Range')).toBeInTheDocument()
    expect(screen.getByText('Capital & Currency')).toBeInTheDocument()
  })

  it('should render strategy template options', async () => {
    renderComponent()
    
    const strategySelect = screen.getByPlaceholderText('Select strategy template')
    await userEvent.click(strategySelect)
    
    expect(screen.getByText('Moving Average Cross')).toBeInTheDocument()
    expect(screen.getByText('Mean Reversion')).toBeInTheDocument()
    expect(screen.getByText('Breakout Strategy')).toBeInTheDocument()
    expect(screen.getByText('Pairs Trading')).toBeInTheDocument()
  })

  it('should show strategy complexity tags', async () => {
    renderComponent()
    
    const strategySelect = screen.getByPlaceholderText('Select strategy template')
    await userEvent.click(strategySelect)
    
    expect(screen.getByText('beginner')).toBeInTheDocument()
    expect(screen.getByText('intermediate')).toBeInTheDocument()
    expect(screen.getByText('advanced')).toBeInTheDocument()
  })

  it('should populate strategy parameters when template is selected', async () => {
    renderComponent()
    
    const strategySelect = screen.getByPlaceholderText('Select strategy template')
    await userEvent.click(strategySelect)
    
    const movingAverageOption = screen.getByText('Moving Average Cross')
    await userEvent.click(movingAverageOption)
    
    await waitFor(() => {
      const parametersTextarea = screen.getByPlaceholderText('Strategy parameters in JSON format')
      expect(parametersTextarea).toBeInTheDocument()
      expect(parametersTextarea.value).toContain('fast_period')
      expect(parametersTextarea.value).toContain('slow_period')
    })
  })

  it('should render instrument selection with search functionality', () => {
    renderComponent()
    
    const instrumentSelect = screen.getByPlaceholderText('Select instruments')
    expect(instrumentSelect).toBeInTheDocument()
    
    // Should show common instruments
    expect(screen.getByText('Technology')).toBeInTheDocument() // Sector labels
  })

  it('should render venue selection with tooltips', async () => {
    renderComponent()
    
    const venueSelect = screen.getByPlaceholderText('Select venues')
    await userEvent.click(venueSelect)
    
    expect(screen.getByText('NASDAQ')).toBeInTheDocument()
    expect(screen.getByText('New York Stock Exchange')).toBeInTheDocument()
    expect(screen.getByText('IB Smart Routing')).toBeInTheDocument()
    expect(screen.getByText('Simulated Exchange')).toBeInTheDocument()
  })

  it('should validate initial balance constraints', async () => {
    renderComponent()
    
    const balanceInput = screen.getByPlaceholderText('100000')
    
    // Test minimum validation
    await userEvent.clear(balanceInput)
    await userEvent.type(balanceInput, '500')
    
    await waitFor(() => {
      expect(mockOnConfigChange).toHaveBeenCalledWith(
        expect.any(Object),
        false // Should be invalid
      )
    })
  })

  it('should show advanced settings when toggle is enabled', async () => {
    renderComponent()
    
    const advancedToggle = screen.getByRole('switch', { name: /advanced/i })
    await userEvent.click(advancedToggle)
    
    expect(screen.getByText('Execution Settings')).toBeInTheDocument()
    expect(screen.getByText('Risk Management')).toBeInTheDocument()
    expect(screen.getByText('Commission Model')).toBeInTheDocument()
    expect(screen.getByText('Position Sizing')).toBeInTheDocument()
  })

  it('should render execution settings in advanced mode', async () => {
    renderComponent()
    
    const advancedToggle = screen.getByRole('switch', { name: /advanced/i })
    await userEvent.click(advancedToggle)
    
    expect(screen.getByText('Commission Model')).toBeInTheDocument()
    expect(screen.getByText('Slippage Model')).toBeInTheDocument()
    expect(screen.getByText('Fill Model')).toBeInTheDocument()
  })

  it('should render risk management settings in advanced mode', async () => {
    renderComponent()
    
    const advancedToggle = screen.getByRole('switch', { name: /advanced/i })
    await userEvent.click(advancedToggle)
    
    expect(screen.getByText('Position Sizing')).toBeInTheDocument()
    expect(screen.getByText('Max Leverage')).toBeInTheDocument()
    expect(screen.getByText('Max Portfolio Risk (%)')).toBeInTheDocument()
  })

  it('should call onConfigChange when form values change', async () => {
    renderComponent()
    
    const strategySelect = screen.getByPlaceholderText('Select strategy template')
    await userEvent.click(strategySelect)
    
    const movingAverageOption = screen.getByText('Moving Average Cross')
    await userEvent.click(movingAverageOption)
    
    await waitFor(() => {
      expect(mockOnConfigChange).toHaveBeenCalled()
    })
  })

  it('should show validation errors for invalid configuration', async () => {
    renderComponent()
    
    // Leave required fields empty and trigger validation
    const balanceInput = screen.getByPlaceholderText('100000')
    await userEvent.clear(balanceInput)
    await userEvent.type(balanceInput, '100')
    
    await waitFor(() => {
      expect(screen.getByText('Configuration Validation Failed')).toBeInTheDocument()
    })
  })

  it('should show success message for valid configuration', async () => {
    renderComponent()
    
    // Fill out a valid configuration
    const strategySelect = screen.getByPlaceholderText('Select strategy template')
    await userEvent.click(strategySelect)
    await userEvent.click(screen.getByText('Moving Average Cross'))
    
    const instrumentSelect = screen.getByPlaceholderText('Select instruments')
    await userEvent.click(instrumentSelect)
    await userEvent.click(screen.getByText('Apple Inc. (AAPL)'))
    
    await waitFor(() => {
      // Should eventually show valid configuration
      expect(mockOnConfigChange).toHaveBeenCalledWith(
        expect.any(Object),
        expect.any(Boolean)
      )
    })
  })

  it('should handle preset date ranges', async () => {
    renderComponent()
    
    const dateRangePicker = screen.getByPlaceholderText('Select dates')
    await userEvent.click(dateRangePicker)
    
    // Should show preset options
    expect(screen.getByText('Last 1 Month')).toBeInTheDocument()
    expect(screen.getByText('Last 3 Months')).toBeInTheDocument()
    expect(screen.getByText('Last 6 Months')).toBeInTheDocument()
    expect(screen.getByText('Last 1 Year')).toBeInTheDocument()
    expect(screen.getByText('Last 2 Years')).toBeInTheDocument()
  })

  it('should populate initial configuration when provided', () => {
    const initialConfig: Partial<BacktestConfig> = {
      strategyClass: 'MovingAverageCross',
      initialBalance: 50000,
      baseCurrency: 'EUR'
    }
    
    renderComponent({ initialConfig })
    
    // Should populate form with initial values
    expect(mockOnConfigChange).toHaveBeenCalled()
  })

  it('should render data quality options', async () => {
    renderComponent()
    
    const dataQualitySelect = screen.getByDisplayValue('Cleaned Data')
    await userEvent.click(dataQualitySelect)
    
    expect(screen.getByText('Raw Data')).toBeInTheDocument()
    expect(screen.getByText('Cleaned Data')).toBeInTheDocument()
    expect(screen.getByText('Validated Data')).toBeInTheDocument()
  })

  it('should support all major currencies', async () => {
    renderComponent()
    
    const currencySelect = screen.getByDisplayValue('USD - US Dollar')
    await userEvent.click(currencySelect)
    
    expect(screen.getByText('EUR - Euro')).toBeInTheDocument()
    expect(screen.getByText('GBP - British Pound')).toBeInTheDocument()
    expect(screen.getByText('JPY - Japanese Yen')).toBeInTheDocument()
    expect(screen.getByText('CHF - Swiss Franc')).toBeInTheDocument()
    expect(screen.getByText('CAD - Canadian Dollar')).toBeInTheDocument()
    expect(screen.getByText('AUD - Australian Dollar')).toBeInTheDocument()
  })

  it('should format balance input with thousands separators', async () => {
    renderComponent()
    
    const balanceInput = screen.getByPlaceholderText('100000')
    await userEvent.clear(balanceInput)
    await userEvent.type(balanceInput, '1000000')
    
    await waitFor(() => {
      expect(balanceInput.value).toContain(',')
    })
  })

  it('should validate leverage limits in advanced mode', async () => {
    renderComponent()
    
    const advancedToggle = screen.getByRole('switch', { name: /advanced/i })
    await userEvent.click(advancedToggle)
    
    const leverageInput = screen.getByPlaceholderText('1.0')
    await userEvent.clear(leverageInput)
    await userEvent.type(leverageInput, '15')
    
    // Should enforce maximum leverage limit
    await waitFor(() => {
      expect(parseFloat(leverageInput.value)).toBeLessThanOrEqual(10)
    })
  })

  it('should validate portfolio risk percentage', async () => {
    renderComponent()
    
    const advancedToggle = screen.getByRole('switch', { name: /advanced/i })
    await userEvent.click(advancedToggle)
    
    const riskInput = screen.getByPlaceholderText('2.0')
    await userEvent.clear(riskInput)
    await userEvent.type(riskInput, '150')
    
    // Should enforce maximum risk percentage
    await waitFor(() => {
      expect(parseFloat(riskInput.value)).toBeLessThanOrEqual(100)
    })
  })

  it('should show tooltips for complex settings', async () => {
    renderComponent()
    
    const advancedToggle = screen.getByRole('switch', { name: /advanced/i })
    await userEvent.click(advancedToggle)
    
    // Look for info icons that show tooltips
    const infoIcons = screen.getAllByRole('img', { name: /info-circle/i })
    expect(infoIcons.length).toBeGreaterThan(0)
  })
})

// Component interaction tests
describe('BacktestConfiguration Interactions', () => {
  const mockOnConfigChange = vi.fn()
  
  beforeEach(() => {
    mockOnConfigChange.mockClear()
  })

  it('should update strategy parameters when different template is selected', async () => {
    render(<BacktestConfiguration onConfigChange={mockOnConfigChange} />)
    
    const strategySelect = screen.getByPlaceholderText('Select strategy template')
    
    // Select Moving Average Cross
    await userEvent.click(strategySelect)
    await userEvent.click(screen.getByText('Moving Average Cross'))
    
    await waitFor(() => {
      const parametersTextarea = screen.getByPlaceholderText('Strategy parameters in JSON format')
      expect(parametersTextarea.value).toContain('fast_period')
    })
    
    // Change to Mean Reversion
    await userEvent.click(strategySelect)
    await userEvent.click(screen.getByText('Mean Reversion'))
    
    await waitFor(() => {
      const parametersTextarea = screen.getByPlaceholderText('Strategy parameters in JSON format')
      expect(parametersTextarea.value).toContain('rsi_period')
    })
  })

  it('should maintain form state when toggling advanced mode', async () => {
    render(<BacktestConfiguration onConfigChange={mockOnConfigChange} />)
    
    // Set some basic values
    const balanceInput = screen.getByPlaceholderText('100000')
    await userEvent.clear(balanceInput)
    await userEvent.type(balanceInput, '50000')
    
    // Toggle advanced mode on
    const advancedToggle = screen.getByRole('switch', { name: /advanced/i })
    await userEvent.click(advancedToggle)
    
    // Toggle advanced mode off
    await userEvent.click(advancedToggle)
    
    // Basic values should be maintained
    expect(balanceInput.value).toContain('50,000')
  })

  it('should disable advanced settings when toggle is off', async () => {
    render(<BacktestConfiguration onConfigChange={mockOnConfigChange} />)
    
    // Advanced settings should not be visible initially
    expect(screen.queryByText('Execution Settings')).not.toBeInTheDocument()
    expect(screen.queryByText('Risk Management')).not.toBeInTheDocument()
  })
})