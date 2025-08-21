/**
 * Layout Manager Component Tests
 * Comprehensive unit tests for layout configuration and persistence
 */

import React from 'react'
import { render, screen, fireEvent, waitFor } from '@testing-library/react'
import { describe, it, expect, beforeEach, vi, Mock } from 'vitest'
import { message } from 'antd'
import { LayoutManager } from '../LayoutManager'
import { ChartLayout } from '../../../types/charting'
import { LayoutTemplate } from '../../../services/chartLayoutService'

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

// Mock layouts and templates data
const mockLayouts: ChartLayout[] = [
  {
    id: 'layout1',
    name: 'Trading Dashboard',
    charts: [],
    layout: {
      rows: 2,
      columns: 2,
      chartPositions: []
    },
    synchronization: {
      crosshair: true,
      zoom: false,
      timeRange: true
    },
    theme: {
      id: 'default',
      name: 'Default',
      colors: {},
      fonts: {}
    }
  },
  {
    id: 'layout2',
    name: 'Analysis View',
    charts: [],
    layout: {
      rows: 1,
      columns: 3,
      chartPositions: []
    },
    synchronization: {
      crosshair: false,
      zoom: true,
      timeRange: false
    },
    theme: {
      id: 'dark',
      name: 'Dark',
      colors: {},
      fonts: {}
    }
  }
]

const mockTemplates: LayoutTemplate[] = [
  {
    id: 'template1',
    name: 'Single Chart',
    description: 'Basic single chart layout',
    category: 'trading',
    isBuiltIn: true,
    layout: {
      name: 'Single Chart Template',
      charts: [],
      layout: { rows: 1, columns: 1, chartPositions: [] },
      synchronization: { crosshair: false, zoom: false, timeRange: false },
      theme: { id: 'default', name: 'Default', colors: {}, fonts: {} }
    }
  },
  {
    id: 'template2',
    name: 'Four Chart Grid',
    description: '2x2 chart grid for comprehensive analysis',
    category: 'analysis',
    isBuiltIn: true,
    layout: {
      name: 'Four Chart Grid Template',
      charts: [],
      layout: { rows: 2, columns: 2, chartPositions: [] },
      synchronization: { crosshair: true, zoom: true, timeRange: true },
      theme: { id: 'default', name: 'Default', colors: {}, fonts: {} }
    }
  },
  {
    id: 'template3',
    name: 'My Custom Layout',
    description: 'Custom user template',
    category: 'custom',
    isBuiltIn: false,
    layout: {
      name: 'Custom Template',
      charts: [],
      layout: { rows: 3, columns: 1, chartPositions: [] },
      synchronization: { crosshair: false, zoom: false, timeRange: true },
      theme: { id: 'default', name: 'Default', colors: {}, fonts: {} }
    }
  }
]

// Mock chart layout service
const mockChartLayoutService = {
  getAllLayouts: vi.fn(() => mockLayouts),
  getAllTemplates: vi.fn(() => mockTemplates),
  getLayout: vi.fn(),
  getTemplate: vi.fn(),
  createLayout: vi.fn(),
  createLayoutFromTemplate: vi.fn(),
  createTemplate: vi.fn(),
  deleteLayout: vi.fn(),
  setActiveLayout: vi.fn(),
  setEventHandlers: vi.fn(),
  loadLayoutsFromStorage: vi.fn()
}

vi.mock('../../../services/chartLayoutService', () => ({
  chartLayoutService: mockChartLayoutService
}))

describe('LayoutManager', () => {
  const defaultProps = {
    onLayoutChange: vi.fn(),
    onLayoutCreate: vi.fn(),
    onLayoutDelete: vi.fn()
  }

  const mockCurrentLayout = mockLayouts[0]

  beforeEach(() => {
    vi.clearAllMocks()
    mockChartLayoutService.getAllLayouts.mockReturnValue(mockLayouts)
    mockChartLayoutService.getAllTemplates.mockReturnValue(mockTemplates)
  })

  describe('Component Initialization', () => {
    it('should render layout manager with title', () => {
      render(<LayoutManager {...defaultProps} />)
      
      expect(screen.getByText('Layout Manager')).toBeInTheDocument()
      expect(screen.getByText('Active Layout:')).toBeInTheDocument()
      expect(screen.getByText('Quick Start Templates:')).toBeInTheDocument()
    })

    it('should load layouts and templates on mount', () => {
      render(<LayoutManager {...defaultProps} />)
      
      expect(mockChartLayoutService.getAllLayouts).toHaveBeenCalled()
      expect(mockChartLayoutService.getAllTemplates).toHaveBeenCalled()
      expect(mockChartLayoutService.setEventHandlers).toHaveBeenCalled()
      expect(mockChartLayoutService.loadLayoutsFromStorage).toHaveBeenCalled()
    })

    it('should show available layouts in dropdown', async () => {
      render(<LayoutManager {...defaultProps} />)
      
      const select = screen.getByPlaceholderText('Select a layout')
      fireEvent.mouseDown(select)
      
      await waitFor(() => {
        expect(screen.getByText('Trading Dashboard')).toBeInTheDocument()
        expect(screen.getByText('Analysis View')).toBeInTheDocument()
      })
    })

    it('should set current layout as selected when provided', () => {
      render(
        <LayoutManager 
          {...defaultProps} 
          currentLayout={mockCurrentLayout}
        />
      )
      
      expect(screen.getByDisplayValue('Trading Dashboard')).toBeInTheDocument()
    })
  })

  describe('Layout Selection', () => {
    it('should call onLayoutChange when layout selected', async () => {
      mockChartLayoutService.getLayout.mockReturnValue(mockLayouts[1])
      
      render(<LayoutManager {...defaultProps} />)
      
      const select = screen.getByPlaceholderText('Select a layout')
      fireEvent.mouseDown(select)
      
      await waitFor(() => {
        fireEvent.click(screen.getByText('Analysis View'))
      })
      
      expect(mockChartLayoutService.getLayout).toHaveBeenCalledWith('layout2')
      expect(mockChartLayoutService.setActiveLayout).toHaveBeenCalledWith('layout2')
      expect(defaultProps.onLayoutChange).toHaveBeenCalledWith(mockLayouts[1])
    })

    it('should handle layout selection when layout not found', async () => {
      mockChartLayoutService.getLayout.mockReturnValue(null)
      
      render(<LayoutManager {...defaultProps} />)
      
      const select = screen.getByPlaceholderText('Select a layout')
      fireEvent.mouseDown(select)
      
      await waitFor(() => {
        fireEvent.click(screen.getByText('Analysis View'))
      })
      
      expect(defaultProps.onLayoutChange).not.toHaveBeenCalled()
    })

    it('should show layout grid descriptions', async () => {
      render(<LayoutManager {...defaultProps} />)
      
      const select = screen.getByPlaceholderText('Select a layout')
      fireEvent.mouseDown(select)
      
      await waitFor(() => {
        expect(screen.getByText('(2×2 Grid)')).toBeInTheDocument()
        expect(screen.getByText('(1×3 Grid)')).toBeInTheDocument()
      })
    })
  })

  describe('Layout Creation', () => {
    it('should open create layout modal when New Layout clicked', () => {
      render(<LayoutManager {...defaultProps} />)
      
      fireEvent.click(screen.getByText('New Layout'))
      
      expect(screen.getByText('Create New Layout')).toBeInTheDocument()
      expect(screen.getByPlaceholderText('Enter layout name')).toBeInTheDocument()
    })

    it('should create layout with form data', async () => {
      mockChartLayoutService.createLayout.mockReturnValue('new_layout_id')
      
      render(<LayoutManager {...defaultProps} />)
      
      // Open modal
      fireEvent.click(screen.getByText('New Layout'))
      
      // Fill form
      const nameInput = screen.getByPlaceholderText('Enter layout name')
      fireEvent.change(nameInput, { target: { value: 'Test Layout' } })
      
      const rowsSelect = screen.getByPlaceholderText('Rows')
      fireEvent.mouseDown(rowsSelect)
      fireEvent.click(screen.getByText('2'))
      
      const colsSelect = screen.getByPlaceholderText('Cols')
      fireEvent.mouseDown(colsSelect)
      fireEvent.click(screen.getByText('3'))
      
      const syncSelect = screen.getByPlaceholderText('Select synchronization options')
      fireEvent.mouseDown(syncSelect)
      fireEvent.click(screen.getByTitle('Crosshair'))
      fireEvent.click(screen.getByTitle('Zoom'))
      
      // Submit
      fireEvent.click(screen.getByText('OK'))
      
      await waitFor(() => {
        expect(mockChartLayoutService.createLayout).toHaveBeenCalledWith({
          name: 'Test Layout',
          charts: [],
          layout: {
            rows: 2,
            columns: 3,
            chartPositions: []
          },
          synchronization: {
            crosshair: true,
            zoom: true,
            timeRange: false
          },
          theme: expect.objectContaining({
            name: 'Default'
          })
        })
      })
      
      expect(defaultProps.onLayoutCreate).toHaveBeenCalledWith('new_layout_id')
      expect(message.success).toHaveBeenCalledWith('Layout created successfully')
    })

    it('should require layout name for creation', async () => {
      render(<LayoutManager {...defaultProps} />)
      
      fireEvent.click(screen.getByText('New Layout'))
      
      // Try to submit without name
      fireEvent.click(screen.getByText('OK'))
      
      await waitFor(() => {
        expect(screen.getByText('Please enter a layout name')).toBeInTheDocument()
      })
      
      expect(mockChartLayoutService.createLayout).not.toHaveBeenCalled()
    })

    it('should handle layout creation errors', async () => {
      const consoleError = vi.spyOn(console, 'error').mockImplementation(() => {})
      mockChartLayoutService.createLayout.mockImplementation(() => {
        throw new Error('Creation failed')
      })
      
      render(<LayoutManager {...defaultProps} />)
      
      fireEvent.click(screen.getByText('New Layout'))
      
      const nameInput = screen.getByPlaceholderText('Enter layout name')
      fireEvent.change(nameInput, { target: { value: 'Test Layout' } })
      
      fireEvent.click(screen.getByText('OK'))
      
      await waitFor(() => {
        expect(consoleError).toHaveBeenCalledWith('Failed to create layout:', expect.any(Error))
      })
      
      consoleError.mockRestore()
    })
  })

  describe('Template Management', () => {
    it('should show built-in templates', () => {
      render(<LayoutManager {...defaultProps} />)
      
      expect(screen.getByText('Built-in Templates:')).toBeInTheDocument()
      expect(screen.getByText('Single Chart')).toBeInTheDocument()
      expect(screen.getByText('Four Chart Grid')).toBeInTheDocument()
    })

    it('should show custom templates when available', () => {
      render(<LayoutManager {...defaultProps} />)
      
      expect(screen.getByText('Custom Templates:')).toBeInTheDocument()
      expect(screen.getByText('My Custom Layout')).toBeInTheDocument()
    })

    it('should create layout from template', () => {
      mockChartLayoutService.getTemplate.mockReturnValue(mockTemplates[0])
      mockChartLayoutService.createLayoutFromTemplate.mockReturnValue('template_layout_id')
      
      render(<LayoutManager {...defaultProps} />)
      
      fireEvent.click(screen.getByText('Single Chart'))
      
      expect(mockChartLayoutService.createLayoutFromTemplate).toHaveBeenCalledWith(
        'template1', 
        expect.stringContaining('Single Chart Template')
      )
      expect(defaultProps.onLayoutCreate).toHaveBeenCalledWith('template_layout_id')
      expect(message.success).toHaveBeenCalledWith('Created layout from Single Chart Template template')
    })

    it('should handle template creation failure', () => {
      mockChartLayoutService.getTemplate.mockReturnValue(mockTemplates[0])
      mockChartLayoutService.createLayoutFromTemplate.mockReturnValue(null)
      
      render(<LayoutManager {...defaultProps} />)
      
      fireEvent.click(screen.getByText('Single Chart'))
      
      expect(defaultProps.onLayoutCreate).not.toHaveBeenCalled()
      expect(message.success).not.toHaveBeenCalled()
    })

    it('should show tooltips for templates', async () => {
      render(<LayoutManager {...defaultProps} />)
      
      const templateButton = screen.getByText('Single Chart')
      fireEvent.mouseEnter(templateButton)
      
      await waitFor(() => {
        expect(screen.getByText('Basic single chart layout')).toBeInTheDocument()
      })
    })
  })

  describe('Layout Actions with Current Layout', () => {
    beforeEach(() => {
      mockChartLayoutService.getLayout.mockReturnValue(mockCurrentLayout)
    })

    it('should show action buttons when current layout exists', () => {
      render(
        <LayoutManager 
          {...defaultProps} 
          currentLayout={mockCurrentLayout}
        />
      )
      
      expect(screen.getByText('Save as Template')).toBeInTheDocument()
      expect(screen.getByText('Duplicate')).toBeInTheDocument()
      expect(screen.getByText('Delete')).toBeInTheDocument()
    })

    it('should duplicate current layout', () => {
      mockChartLayoutService.createLayout.mockReturnValue('duplicated_layout_id')
      
      render(
        <LayoutManager 
          {...defaultProps} 
          currentLayout={mockCurrentLayout}
        />
      )
      
      fireEvent.click(screen.getByText('Duplicate'))
      
      expect(mockChartLayoutService.createLayout).toHaveBeenCalledWith({
        ...mockCurrentLayout,
        name: 'Trading Dashboard (Copy)',
        id: undefined
      })
      expect(defaultProps.onLayoutCreate).toHaveBeenCalledWith('duplicated_layout_id')
      expect(message.success).toHaveBeenCalledWith('Layout duplicated successfully')
    })

    it('should delete layout with confirmation', async () => {
      mockChartLayoutService.deleteLayout.mockReturnValue(true)
      
      // Mock Modal.confirm to auto-confirm
      const mockConfirm = vi.fn((config) => {
        config.onOk()
      })
      vi.doMock('antd', async () => {
        const actual = await vi.importActual('antd')
        return {
          ...actual,
          Modal: {
            ...actual.Modal,
            confirm: mockConfirm
          }
        }
      })
      
      render(
        <LayoutManager 
          {...defaultProps} 
          currentLayout={mockCurrentLayout}
        />
      )
      
      fireEvent.click(screen.getByText('Delete'))
      
      expect(mockChartLayoutService.deleteLayout).toHaveBeenCalledWith('layout1')
      expect(defaultProps.onLayoutDelete).toHaveBeenCalledWith('layout1')
      expect(message.success).toHaveBeenCalledWith('Layout deleted successfully')
    })

    it('should save current layout as template', async () => {
      render(
        <LayoutManager 
          {...defaultProps} 
          currentLayout={mockCurrentLayout}
        />
      )
      
      fireEvent.click(screen.getByText('Save as Template'))
      
      expect(screen.getByText('Save as Template')).toBeInTheDocument()
      
      // Fill form
      const nameInput = screen.getByPlaceholderText('Enter template name')
      fireEvent.change(nameInput, { target: { value: 'My Template' } })
      
      const descInput = screen.getByPlaceholderText('Enter template description (optional)')
      fireEvent.change(descInput, { target: { value: 'Custom template description' } })
      
      // Select category
      const customRadio = screen.getByLabelText('Custom')
      fireEvent.click(customRadio)
      
      // Submit
      fireEvent.click(screen.getByText('OK'))
      
      await waitFor(() => {
        expect(mockChartLayoutService.createTemplate).toHaveBeenCalledWith(
          'My Template',
          'Custom template description',
          mockCurrentLayout,
          'custom'
        )
      })
      
      expect(message.success).toHaveBeenCalledWith('Template saved successfully')
    })

    it('should prevent saving template without current layout', () => {
      render(<LayoutManager {...defaultProps} />)
      
      // No current layout provided
      expect(screen.queryByText('Save as Template')).not.toBeInTheDocument()
    })

    it('should require template name when saving', async () => {
      render(
        <LayoutManager 
          {...defaultProps} 
          currentLayout={mockCurrentLayout}
        />
      )
      
      fireEvent.click(screen.getByText('Save as Template'))
      
      // Try to submit without name
      fireEvent.click(screen.getByText('OK'))
      
      await waitFor(() => {
        expect(screen.getByText('Please enter a template name')).toBeInTheDocument()
      })
      
      expect(mockChartLayoutService.createTemplate).not.toHaveBeenCalled()
    })
  })

  describe('Current Layout Info Display', () => {
    it('should show current layout information', () => {
      render(
        <LayoutManager 
          {...defaultProps} 
          currentLayout={mockCurrentLayout}
        />
      )
      
      expect(screen.getByText('Layout: Trading Dashboard')).toBeInTheDocument()
      expect(screen.getByText('Grid: 2×2 Grid')).toBeInTheDocument()
      expect(screen.getByText('Charts: 0')).toBeInTheDocument()
      expect(screen.getByText('Sync: Crosshair, Time')).toBeInTheDocument()
    })

    it('should show "None" when no synchronization enabled', () => {
      const layoutWithNoSync = {
        ...mockCurrentLayout,
        synchronization: {
          crosshair: false,
          zoom: false,
          timeRange: false
        }
      }
      
      render(
        <LayoutManager 
          {...defaultProps} 
          currentLayout={layoutWithNoSync}
        />
      )
      
      expect(screen.getByText('Sync: None')).toBeInTheDocument()
    })

    it('should not show layout info when no current layout', () => {
      render(<LayoutManager {...defaultProps} />)
      
      expect(screen.queryByText(/Layout:/)).not.toBeInTheDocument()
      expect(screen.queryByText(/Grid:/)).not.toBeInTheDocument()
    })
  })

  describe('Grid Description Utility', () => {
    it('should describe common grid layouts correctly', () => {
      const testCases = [
        { rows: 1, cols: 1, expected: 'Single Chart' },
        { rows: 1, cols: 2, expected: 'Dual Horizontal' },
        { rows: 2, cols: 1, expected: 'Dual Vertical' },
        { rows: 2, cols: 2, expected: '2×2 Grid' },
        { rows: 3, cols: 3, expected: '3×3 Grid' },
        { rows: 2, cols: 3, expected: '2×3 Grid' }
      ]
      
      testCases.forEach(({ rows, cols, expected }) => {
        const layoutWithGrid = {
          ...mockCurrentLayout,
          layout: { ...mockCurrentLayout.layout, rows, columns: cols }
        }
        
        const { rerender } = render(
          <LayoutManager 
            {...defaultProps} 
            currentLayout={layoutWithGrid}
          />
        )
        
        expect(screen.getByText(`Grid: ${expected}`)).toBeInTheDocument()
        
        // Clean up for next test
        rerender(<div />)
      })
    })
  })

  describe('Error Handling', () => {
    it('should handle service errors gracefully', () => {
      mockChartLayoutService.getAllLayouts.mockImplementation(() => {
        throw new Error('Service error')
      })
      
      // Should not crash
      expect(() => render(<LayoutManager {...defaultProps} />)).not.toThrow()
    })

    it('should handle missing callbacks gracefully', () => {
      render(<LayoutManager />)
      
      const select = screen.getByPlaceholderText('Select a layout')
      
      // Should not crash when callbacks are undefined
      expect(() => fireEvent.mouseDown(select)).not.toThrow()
    })

    it('should handle template creation errors', async () => {
      const consoleError = vi.spyOn(console, 'error').mockImplementation(() => {})
      mockChartLayoutService.createTemplate.mockImplementation(() => {
        throw new Error('Template creation failed')
      })
      
      render(
        <LayoutManager 
          {...defaultProps} 
          currentLayout={mockCurrentLayout}
        />
      )
      
      fireEvent.click(screen.getByText('Save as Template'))
      
      const nameInput = screen.getByPlaceholderText('Enter template name')
      fireEvent.change(nameInput, { target: { value: 'Test Template' } })
      
      fireEvent.click(screen.getByText('OK'))
      
      await waitFor(() => {
        expect(consoleError).toHaveBeenCalledWith('Failed to save template:', expect.any(Error))
      })
      
      consoleError.mockRestore()
    })
  })

  describe('Event Handler Integration', () => {
    it('should register layout service event handlers', () => {
      render(<LayoutManager {...defaultProps} />)
      
      expect(mockChartLayoutService.setEventHandlers).toHaveBeenCalledWith({
        onLayoutChange: expect.any(Function)
      })
    })

    it('should handle layout service changes', () => {
      render(<LayoutManager {...defaultProps} />)
      
      // Get the registered event handler
      const eventHandlers = mockChartLayoutService.setEventHandlers.mock.calls[0][0]
      const onLayoutChange = eventHandlers.onLayoutChange
      
      // Trigger the event
      onLayoutChange(mockCurrentLayout)
      
      expect(mockChartLayoutService.getAllLayouts).toHaveBeenCalled()
      expect(defaultProps.onLayoutChange).toHaveBeenCalledWith(mockCurrentLayout)
    })
  })

  describe('Responsive Design', () => {
    it('should handle different screen sizes', () => {
      // Mock useBreakpoint
      vi.doMock('antd', async () => {
        const actual = await vi.importActual('antd')
        return {
          ...actual,
          Grid: {
            useBreakpoint: () => ({ xs: true, sm: false, md: false })
          }
        }
      })
      
      render(<LayoutManager {...defaultProps} />)
      
      expect(screen.getByText('Layout Manager')).toBeInTheDocument()
    })
  })

  describe('Performance', () => {
    it('should not reload layouts unnecessarily', () => {
      const { rerender } = render(<LayoutManager {...defaultProps} />)
      
      const initialCalls = mockChartLayoutService.getAllLayouts.mock.calls.length
      
      // Rerender with same props
      rerender(<LayoutManager {...defaultProps} />)
      
      // Should not call getAllLayouts again unnecessarily
      expect(mockChartLayoutService.getAllLayouts.mock.calls.length).toBe(initialCalls)
    })

    it('should handle large numbers of layouts efficiently', () => {
      const manyLayouts = Array.from({ length: 100 }, (_, i) => ({
        ...mockCurrentLayout,
        id: `layout${i}`,
        name: `Layout ${i}`
      }))
      
      mockChartLayoutService.getAllLayouts.mockReturnValue(manyLayouts)
      
      const startTime = performance.now()
      render(<LayoutManager {...defaultProps} />)
      const endTime = performance.now()
      
      expect(endTime - startTime).toBeLessThan(100) // Should render quickly
      expect(screen.getByText('Layout Manager')).toBeInTheDocument()
    })
  })
})