/**
 * Indicator Builder Component Tests
 * Comprehensive unit tests for parameter validation and UI logic
 */

import React from 'react'
import { render, screen, fireEvent, waitFor } from '@testing-library/react'
import { describe, it, expect, beforeEach, vi, Mock } from 'vitest'
import { message } from 'antd'
import { IndicatorBuilder } from '../IndicatorBuilder'
import { TechnicalIndicator } from '../../../services/indicatorEngine'

// Mock Ant Design message
vi.mock('antd', async () => {
  const actual = await vi.importActual('antd')
  return {
    ...actual,
    message: {
      success: vi.fn(),
      error: vi.fn(),
      warning: vi.fn(),
      info: vi.fn()
    }
  }
})

// Mock indicator engine
const mockIndicators: TechnicalIndicator[] = [
  {
    id: 'sma',
    name: 'Simple Moving Average',
    type: 'built_in',
    parameters: [
      {
        name: 'period',
        type: 'number',
        defaultValue: 20,
        min: 1,
        max: 200
      },
      {
        name: 'source',
        type: 'string',
        defaultValue: 'close',
        options: ['open', 'high', 'low', 'close']
      }
    ],
    calculation: {
      period: 20,
      source: 'close'
    },
    display: {
      color: '#FF6B6B',
      lineWidth: 2,
      style: 'solid',
      overlay: true
    }
  },
  {
    id: 'rsi',
    name: 'Relative Strength Index',
    type: 'built_in',
    parameters: [
      {
        name: 'period',
        type: 'number',
        defaultValue: 14,
        min: 2,
        max: 100
      },
      {
        name: 'overbought',
        type: 'number',
        defaultValue: 70,
        min: 50,
        max: 90
      },
      {
        name: 'showLevels',
        type: 'boolean',
        defaultValue: true
      },
      {
        name: 'levelColor',
        type: 'color',
        defaultValue: '#888888'
      }
    ],
    calculation: {
      period: 14,
      source: 'close'
    },
    display: {
      color: '#45B7D1',
      lineWidth: 2,
      style: 'solid',
      overlay: false
    }
  }
]

const mockIndicatorEngine = {
  getAvailableIndicators: vi.fn(() => mockIndicators),
  createCustomIndicator: vi.fn(),
  calculate: vi.fn()
}

vi.mock('../../../services/indicatorEngine', () => ({
  indicatorEngine: mockIndicatorEngine
}))

// Mock chart store
const mockChartStore = {
  addIndicator: vi.fn(),
  removeIndicator: vi.fn(),
  updateIndicator: vi.fn()
}

vi.mock('../../Chart/hooks/useChartStore', () => ({
  useChartStore: () => mockChartStore
}))

describe('IndicatorBuilder', () => {
  const defaultProps = {
    onIndicatorAdd: vi.fn(),
    onIndicatorRemove: vi.fn()
  }

  beforeEach(() => {
    vi.clearAllMocks()
    mockIndicatorEngine.getAvailableIndicators.mockReturnValue(mockIndicators)
  })

  describe('Component Initialization', () => {
    it('should render component with available indicators', () => {
      render(<IndicatorBuilder {...defaultProps} />)
      
      expect(screen.getByText('Technical Indicators')).toBeInTheDocument()
      expect(screen.getByText('Select Indicator:')).toBeInTheDocument()
      expect(screen.getByText('Parameters:')).toBeInTheDocument()
      expect(screen.getByText('Add Indicator')).toBeInTheDocument()
    })

    it('should load available indicators on mount', () => {
      render(<IndicatorBuilder {...defaultProps} />)
      
      expect(mockIndicatorEngine.getAvailableIndicators).toHaveBeenCalled()
      expect(screen.getByDisplayValue('Simple Moving Average')).toBeInTheDocument()
    })

    it('should initialize with first indicator selected', () => {
      render(<IndicatorBuilder {...defaultProps} />)
      
      // Should show SMA parameters
      expect(screen.getByDisplayValue('20')).toBeInTheDocument() // period
      expect(screen.getByDisplayValue('close')).toBeInTheDocument() // source
    })

    it('should show empty state when no indicators added', () => {
      render(<IndicatorBuilder {...defaultProps} />)
      
      expect(screen.getByText('Active Indicators (0):')).toBeInTheDocument()
      expect(screen.getByText('No indicators added yet')).toBeInTheDocument()
    })
  })

  describe('Indicator Selection', () => {
    it('should change indicator when selected from dropdown', async () => {
      render(<IndicatorBuilder {...defaultProps} />)
      
      const select = screen.getByDisplayValue('Simple Moving Average')
      fireEvent.mouseDown(select)
      
      await waitFor(() => {
        expect(screen.getByText('Relative Strength Index')).toBeInTheDocument()
      })
      
      fireEvent.click(screen.getByText('Relative Strength Index'))
      
      await waitFor(() => {
        // Should show RSI parameters
        expect(screen.getByDisplayValue('14')).toBeInTheDocument() // period
        expect(screen.getByDisplayValue('70')).toBeInTheDocument() // overbought
      })
    })

    it('should initialize parameters when indicator changes', async () => {
      render(<IndicatorBuilder {...defaultProps} />)
      
      const select = screen.getByDisplayValue('Simple Moving Average')
      fireEvent.mouseDown(select)
      fireEvent.click(screen.getByText('Relative Strength Index'))
      
      await waitFor(() => {
        expect(screen.getByDisplayValue('14')).toBeInTheDocument()
        expect(screen.getByDisplayValue('70')).toBeInTheDocument()
      })
    })

    it('should show custom indicator badge', () => {
      const customIndicator = {
        ...mockIndicators[0],
        id: 'custom_test',
        name: 'Custom Test',
        type: 'custom' as const
      }
      
      mockIndicatorEngine.getAvailableIndicators.mockReturnValue([...mockIndicators, customIndicator])
      
      render(<IndicatorBuilder {...defaultProps} />)
      
      const select = screen.getByDisplayValue('Simple Moving Average')
      fireEvent.mouseDown(select)
      
      expect(screen.getByText('(Custom)')).toBeInTheDocument()
    })
  })

  describe('Parameter Configuration', () => {
    it('should handle number parameter changes', () => {
      render(<IndicatorBuilder {...defaultProps} />)
      
      const periodInput = screen.getByDisplayValue('20')
      fireEvent.change(periodInput, { target: { value: '30' } })
      
      expect(periodInput).toHaveValue(30)
    })

    it('should handle boolean parameter changes', async () => {
      render(<IndicatorBuilder {...defaultProps} />)
      
      // Switch to RSI to get boolean parameter
      const select = screen.getByDisplayValue('Simple Moving Average')
      fireEvent.mouseDown(select)
      fireEvent.click(screen.getByText('Relative Strength Index'))
      
      await waitFor(() => {
        const switchElement = screen.getByRole('switch')
        expect(switchElement).toBeChecked()
        
        fireEvent.click(switchElement)
        expect(switchElement).not.toBeChecked()
      })
    })

    it('should handle color parameter changes', async () => {
      render(<IndicatorBuilder {...defaultProps} />)
      
      // Switch to RSI to get color parameter
      const select = screen.getByDisplayValue('Simple Moving Average')
      fireEvent.mouseDown(select)
      fireEvent.click(screen.getByText('Relative Strength Index'))
      
      await waitFor(() => {
        const colorPickers = screen.getAllByText('Color')
        expect(colorPickers).toHaveLength(1)
      })
    })

    it('should handle string parameter with options', () => {
      render(<IndicatorBuilder {...defaultProps} />)
      
      const sourceSelect = screen.getByDisplayValue('close')
      fireEvent.mouseDown(sourceSelect)
      
      expect(screen.getByText('open')).toBeInTheDocument()
      expect(screen.getByText('high')).toBeInTheDocument()
      expect(screen.getByText('low')).toBeInTheDocument()
      
      fireEvent.click(screen.getByText('high'))
      expect(screen.getByDisplayValue('high')).toBeInTheDocument()
    })

    it('should respect parameter constraints', () => {
      render(<IndicatorBuilder {...defaultProps} />)
      
      const periodInput = screen.getByDisplayValue('20')
      
      // Should not allow values outside min/max range
      fireEvent.change(periodInput, { target: { value: '300' } })
      // Input should be constrained to max value (200)
      fireEvent.blur(periodInput)
    })
  })

  describe('Adding Indicators', () => {
    it('should add indicator with correct parameters', () => {
      render(<IndicatorBuilder {...defaultProps} />)
      
      // Change period parameter
      const periodInput = screen.getByDisplayValue('20')
      fireEvent.change(periodInput, { target: { value: '50' } })
      
      // Add indicator
      const addButton = screen.getByText('Add Indicator')
      fireEvent.click(addButton)
      
      expect(defaultProps.onIndicatorAdd).toHaveBeenCalledWith('sma', {
        period: 50,
        source: 'close'
      })
      expect(message.success).toHaveBeenCalledWith('Added Simple Moving Average')
    })

    it('should show indicator in active list after adding', () => {
      render(<IndicatorBuilder {...defaultProps} />)
      
      const addButton = screen.getByText('Add Indicator')
      fireEvent.click(addButton)
      
      expect(screen.getByText('Active Indicators (1):')).toBeInTheDocument()
      expect(screen.getByText('Simple Moving Average(20)')).toBeInTheDocument()
    })

    it('should disable add button when no indicator selected', () => {
      mockIndicatorEngine.getAvailableIndicators.mockReturnValue([])
      
      render(<IndicatorBuilder {...defaultProps} />)
      
      const addButton = screen.getByText('Add Indicator')
      expect(addButton).toBeDisabled()
    })

    it('should generate unique instance IDs', () => {
      render(<IndicatorBuilder {...defaultProps} />)
      
      const addButton = screen.getByText('Add Indicator')
      
      // Add same indicator twice
      fireEvent.click(addButton)
      fireEvent.click(addButton)
      
      expect(defaultProps.onIndicatorAdd).toHaveBeenCalledTimes(2)
      expect(screen.getByText('Active Indicators (2):')).toBeInTheDocument()
    })
  })

  describe('Managing Active Indicators', () => {
    it('should remove indicator when delete button clicked', () => {
      render(<IndicatorBuilder {...defaultProps} />)
      
      // Add an indicator first
      const addButton = screen.getByText('Add Indicator')
      fireEvent.click(addButton)
      
      // Remove it
      const deleteButton = screen.getByLabelText('delete')
      fireEvent.click(deleteButton)
      
      expect(screen.getByText('Active Indicators (0):')).toBeInTheDocument()
      expect(screen.getByText('No indicators added yet')).toBeInTheDocument()
      expect(message.success).toHaveBeenCalledWith('Indicator removed')
    })

    it('should toggle visibility when eye button clicked', () => {
      render(<IndicatorBuilder {...defaultProps} />)
      
      // Add an indicator
      const addButton = screen.getByText('Add Indicator')
      fireEvent.click(addButton)
      
      // Toggle visibility
      const eyeButton = screen.getByLabelText('eye')
      fireEvent.click(eyeButton)
      
      // Should change to eye-invisible
      expect(screen.getByLabelText('eye-invisible')).toBeInTheDocument()
    })

    it('should change color when color picker is used', () => {
      render(<IndicatorBuilder {...defaultProps} />)
      
      // Add an indicator
      const addButton = screen.getByText('Add Indicator')
      fireEvent.click(addButton)
      
      // Find color picker in active indicators section
      const colorPickers = screen.getAllByText('Color')
      expect(colorPickers.length).toBeGreaterThan(0)
    })

    it('should show scrollable list when many indicators added', () => {
      render(<IndicatorBuilder {...defaultProps} />)
      
      const addButton = screen.getByText('Add Indicator')
      
      // Add multiple indicators
      for (let i = 0; i < 10; i++) {
        fireEvent.click(addButton)
      }
      
      expect(screen.getByText('Active Indicators (10):')).toBeInTheDocument()
      
      // List should be scrollable (maxHeight: 200px)
      const activeList = screen.getByText('Simple Moving Average(20)').closest('div')
      expect(activeList?.parentElement).toHaveStyle({ maxHeight: '200px', overflowY: 'auto' })
    })
  })

  describe('Custom Indicator Creation', () => {
    it('should open custom indicator modal when custom button clicked', () => {
      render(<IndicatorBuilder {...defaultProps} />)
      
      const customButton = screen.getByText('Custom')
      fireEvent.click(customButton)
      
      expect(screen.getByText('Create Custom Indicator')).toBeInTheDocument()
      expect(screen.getByPlaceholderText('Enter indicator name')).toBeInTheDocument()
      expect(screen.getByPlaceholderText(/Example: Simple custom indicator/)).toBeInTheDocument()
    })

    it('should create custom indicator with valid input', async () => {
      mockIndicatorEngine.createCustomIndicator.mockReturnValue('custom_123')
      mockIndicatorEngine.getAvailableIndicators.mockReturnValueOnce(mockIndicators)
        .mockReturnValueOnce([...mockIndicators, { id: 'custom_123', name: 'My Custom', type: 'scripted', parameters: [], calculation: { period: 1, source: 'close' }, display: { color: '#FF6B6B', lineWidth: 2, style: 'solid', overlay: true } }])
      
      render(<IndicatorBuilder {...defaultProps} />)
      
      // Open modal
      const customButton = screen.getByText('Custom')
      fireEvent.click(customButton)
      
      // Fill in form
      const nameInput = screen.getByPlaceholderText('Enter indicator name')
      const scriptInput = screen.getByPlaceholderText(/Example: Simple custom indicator/)
      
      fireEvent.change(nameInput, { target: { value: 'My Custom Indicator' } })
      fireEvent.change(scriptInput, { target: { value: 'return data.map(d => d.close)' } })
      
      // Submit
      const okButton = screen.getByText('OK')
      fireEvent.click(okButton)
      
      expect(mockIndicatorEngine.createCustomIndicator).toHaveBeenCalledWith(
        'My Custom Indicator',
        'return data.map(d => d.close)',
        [{ name: 'period', type: 'number', defaultValue: 20, min: 1, max: 200 }],
        {
          color: '#FF6B6B',
          lineWidth: 2,
          style: 'solid',
          overlay: true
        }
      )
      
      await waitFor(() => {
        expect(message.success).toHaveBeenCalledWith('Custom indicator created successfully')
      })
    })

    it('should show error for incomplete custom indicator form', () => {
      render(<IndicatorBuilder {...defaultProps} />)
      
      // Open modal
      const customButton = screen.getByText('Custom')
      fireEvent.click(customButton)
      
      // Submit without filling form
      const okButton = screen.getByText('OK')
      fireEvent.click(okButton)
      
      expect(message.error).toHaveBeenCalledWith('Please provide both name and script for custom indicator')
    })

    it('should handle custom indicator creation errors', () => {
      mockIndicatorEngine.createCustomIndicator.mockImplementation(() => {
        throw new Error('Script compilation failed')
      })
      
      render(<IndicatorBuilder {...defaultProps} />)
      
      // Open modal
      const customButton = screen.getByText('Custom')
      fireEvent.click(customButton)
      
      // Fill form
      const nameInput = screen.getByPlaceholderText('Enter indicator name')
      const scriptInput = screen.getByPlaceholderText(/Example: Simple custom indicator/)
      
      fireEvent.change(nameInput, { target: { value: 'Test' } })
      fireEvent.change(scriptInput, { target: { value: 'invalid script' } })
      
      // Submit
      const okButton = screen.getByText('OK')
      fireEvent.click(okButton)
      
      expect(message.error).toHaveBeenCalledWith('Error creating custom indicator: Script compilation failed')
    })

    it('should clear form after successful creation', async () => {
      mockIndicatorEngine.createCustomIndicator.mockReturnValue('custom_123')
      
      render(<IndicatorBuilder {...defaultProps} />)
      
      // Open modal and fill form
      const customButton = screen.getByText('Custom')
      fireEvent.click(customButton)
      
      const nameInput = screen.getByPlaceholderText('Enter indicator name')
      const scriptInput = screen.getByPlaceholderText(/Example: Simple custom indicator/)
      
      fireEvent.change(nameInput, { target: { value: 'Test' } })
      fireEvent.change(scriptInput, { target: { value: 'return []' } })
      
      // Submit
      const okButton = screen.getByText('OK')
      fireEvent.click(okButton)
      
      await waitFor(() => {
        expect(screen.queryByText('Create Custom Indicator')).not.toBeInTheDocument()
      })
    })
  })

  describe('Error Handling', () => {
    it('should handle missing indicators gracefully', () => {
      mockIndicatorEngine.getAvailableIndicators.mockReturnValue([])
      
      render(<IndicatorBuilder {...defaultProps} />)
      
      expect(screen.getByText('Technical Indicators')).toBeInTheDocument()
      expect(screen.getByText('Add Indicator')).toBeDisabled()
    })

    it('should handle indicator engine errors', () => {
      mockIndicatorEngine.getAvailableIndicators.mockImplementation(() => {
        throw new Error('Engine error')
      })
      
      // Should not crash
      expect(() => render(<IndicatorBuilder {...defaultProps} />)).not.toThrow()
    })

    it('should handle callback errors gracefully', () => {
      const errorProps = {
        ...defaultProps,
        onIndicatorAdd: vi.fn(() => { throw new Error('Callback error') })
      }
      
      render(<IndicatorBuilder {...errorProps} />)
      
      const addButton = screen.getByText('Add Indicator')
      
      // Should not crash when callback throws
      expect(() => fireEvent.click(addButton)).not.toThrow()
    })
  })

  describe('Component Props', () => {
    it('should apply custom className', () => {
      const { container } = render(
        <IndicatorBuilder {...defaultProps} className="custom-indicator-builder" />
      )
      
      expect(container.firstChild).toHaveClass('custom-indicator-builder')
    })

    it('should work without callback props', () => {
      render(<IndicatorBuilder />)
      
      const addButton = screen.getByText('Add Indicator')
      
      // Should not crash when callbacks are undefined
      expect(() => fireEvent.click(addButton)).not.toThrow()
    })

    it('should call onIndicatorRemove when indicator removed', () => {
      render(<IndicatorBuilder {...defaultProps} />)
      
      // Add and remove indicator
      fireEvent.click(screen.getByText('Add Indicator'))
      fireEvent.click(screen.getByLabelText('delete'))
      
      expect(defaultProps.onIndicatorRemove).toHaveBeenCalled()
    })
  })

  describe('UI State Management', () => {
    it('should maintain parameter state when switching indicators', async () => {
      render(<IndicatorBuilder {...defaultProps} />)
      
      // Change SMA period
      const periodInput = screen.getByDisplayValue('20')
      fireEvent.change(periodInput, { target: { value: '50' } })
      
      // Switch to RSI
      const select = screen.getByDisplayValue('Simple Moving Average')
      fireEvent.mouseDown(select)
      fireEvent.click(screen.getByText('Relative Strength Index'))
      
      // Switch back to SMA
      await waitFor(() => {
        const rsiSelect = screen.getByDisplayValue('Relative Strength Index')
        fireEvent.mouseDown(rsiSelect)
        fireEvent.click(screen.getByText('Simple Moving Average'))
      })
      
      // Period should be reset to default, not preserve previous value
      await waitFor(() => {
        expect(screen.getByDisplayValue('20')).toBeInTheDocument()
      })
    })

    it('should preserve active indicators list across re-renders', () => {
      const { rerender } = render(<IndicatorBuilder {...defaultProps} />)
      
      // Add indicator
      fireEvent.click(screen.getByText('Add Indicator'))
      expect(screen.getByText('Active Indicators (1):')).toBeInTheDocument()
      
      // Rerender component
      rerender(<IndicatorBuilder {...defaultProps} className="updated" />)
      
      // Active indicators should still be there
      expect(screen.getByText('Active Indicators (1):')).toBeInTheDocument()
    })

    it('should update indicator list after custom creation', async () => {
      const customIndicator = {
        id: 'custom_new',
        name: 'New Custom',
        type: 'scripted' as const,
        parameters: [],
        calculation: { period: 1, source: 'close' },
        display: { color: '#FF6B6B', lineWidth: 2, style: 'solid', overlay: true }
      }
      
      mockIndicatorEngine.createCustomIndicator.mockReturnValue('custom_new')
      mockIndicatorEngine.getAvailableIndicators
        .mockReturnValueOnce(mockIndicators)
        .mockReturnValueOnce([...mockIndicators, customIndicator])
      
      render(<IndicatorBuilder {...defaultProps} />)
      
      // Initially should not have custom indicator
      const select = screen.getByDisplayValue('Simple Moving Average')
      fireEvent.mouseDown(select)
      expect(screen.queryByText('New Custom')).not.toBeInTheDocument()
      fireEvent.click(select) // Close dropdown
      
      // Create custom indicator
      fireEvent.click(screen.getByText('Custom'))
      fireEvent.change(screen.getByPlaceholderText('Enter indicator name'), { target: { value: 'New Custom' } })
      fireEvent.change(screen.getByPlaceholderText(/Example: Simple custom indicator/), { target: { value: 'return []' } })
      fireEvent.click(screen.getByText('OK'))
      
      await waitFor(() => {
        expect(mockIndicatorEngine.getAvailableIndicators).toHaveBeenCalledTimes(2)
      })
    })
  })

  describe('Accessibility', () => {
    it('should have proper labels for form elements', () => {
      render(<IndicatorBuilder {...defaultProps} />)
      
      expect(screen.getByText('Select Indicator:')).toBeInTheDocument()
      expect(screen.getByText('Parameters:')).toBeInTheDocument()
      expect(screen.getByText('period:')).toBeInTheDocument()
      expect(screen.getByText('source:')).toBeInTheDocument()
    })

    it('should support keyboard navigation', () => {
      render(<IndicatorBuilder {...defaultProps} />)
      
      const addButton = screen.getByText('Add Indicator')
      addButton.focus()
      expect(document.activeElement).toBe(addButton)
    })

    it('should have proper button labels', () => {
      render(<IndicatorBuilder {...defaultProps} />)
      
      // Add indicator
      fireEvent.click(screen.getByText('Add Indicator'))
      
      expect(screen.getByLabelText('eye')).toBeInTheDocument()
      expect(screen.getByLabelText('delete')).toBeInTheDocument()
    })
  })

  describe('Performance', () => {
    it('should not re-initialize parameters unnecessarily', () => {
      const { rerender } = render(<IndicatorBuilder {...defaultProps} />)
      
      // Change period
      const periodInput = screen.getByDisplayValue('20')
      fireEvent.change(periodInput, { target: { value: '50' } })
      expect(periodInput).toHaveValue(50)
      
      // Rerender with same props
      rerender(<IndicatorBuilder {...defaultProps} />)
      
      // Value should be preserved
      expect(screen.getByDisplayValue('50')).toBeInTheDocument()
    })

    it('should handle large numbers of active indicators efficiently', () => {
      render(<IndicatorBuilder {...defaultProps} />)
      
      const addButton = screen.getByText('Add Indicator')
      
      // Add many indicators - should not cause performance issues
      for (let i = 0; i < 50; i++) {
        fireEvent.click(addButton)
      }
      
      expect(screen.getByText('Active Indicators (50):')).toBeInTheDocument()
    })
  })
})