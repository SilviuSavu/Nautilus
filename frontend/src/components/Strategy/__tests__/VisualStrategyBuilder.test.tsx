import React from 'react';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import { vi, describe, it, expect, beforeEach } from 'vitest';
import { DndContext } from '@dnd-kit/core';
import { VisualStrategyBuilder } from '../VisualStrategyBuilder';

// Mock @dnd-kit/core for testing
vi.mock('@dnd-kit/core', () => ({
  DndContext: ({ children }: any) => <div data-testid="dnd-context">{children}</div>,
  DragOverlay: ({ children }: any) => <div data-testid="drag-overlay">{children}</div>,
  useSensor: vi.fn(),
  useSensors: vi.fn(() => []),
  PointerSensor: vi.fn(),
  closestCenter: vi.fn(),
  defaultDropAnimationSideEffects: {}
}));

vi.mock('@dnd-kit/sortable', () => ({
  SortableContext: ({ children }: any) => <div data-testid="sortable-context">{children}</div>,
  arrayMove: vi.fn((items, oldIndex, newIndex) => {
    const result = [...items];
    result.splice(newIndex, 0, result.splice(oldIndex, 1)[0]);
    return result;
  }),
  verticalListSortingStrategy: {},
  useSortable: vi.fn(() => ({
    attributes: {},
    listeners: {},
    setNodeRef: vi.fn(),
    transform: null,
    transition: null,
    isDragging: false
  }))
}));

vi.mock('@dnd-kit/utilities', () => ({
  CSS: {
    Transform: {
      toString: vi.fn(() => 'transform: translate3d(0, 0, 0)')
    }
  }
}));

describe('VisualStrategyBuilder', () => {
  const mockOnStrategyChange = vi.fn();

  beforeEach(() => {
    vi.clearAllMocks();
  });

  it('renders empty state when no components are added', () => {
    render(
      <VisualStrategyBuilder onStrategyChange={mockOnStrategyChange} />
    );

    expect(screen.getByText('Visual Strategy Builder')).toBeInTheDocument();
    expect(screen.getByText('No Components Added')).toBeInTheDocument();
    expect(screen.getByText('Add Your First Component')).toBeInTheDocument();
  });

  it('shows component library when Add Component button is clicked', async () => {
    render(
      <VisualStrategyBuilder onStrategyChange={mockOnStrategyChange} />
    );

    const addButton = screen.getByText('Add Component');
    fireEvent.click(addButton);

    await waitFor(() => {
      expect(screen.getByText('Component Library')).toBeInTheDocument();
    });
  });

  it('displays validation summary with no errors initially', () => {
    render(
      <VisualStrategyBuilder onStrategyChange={mockOnStrategyChange} />
    );

    expect(screen.getByText('0/0 Valid')).toBeInTheDocument();
  });

  it('is readonly when readonly prop is true', () => {
    render(
      <VisualStrategyBuilder 
        onStrategyChange={mockOnStrategyChange} 
        readonly={true}
      />
    );

    expect(screen.queryByText('Add Component')).not.toBeInTheDocument();
    expect(screen.queryByText('Add Your First Component')).not.toBeInTheDocument();
  });

  it('renders DnD contexts for drag and drop functionality', () => {
    render(
      <VisualStrategyBuilder onStrategyChange={mockOnStrategyChange} />
    );

    expect(screen.getByTestId('dnd-context')).toBeInTheDocument();
  });

  it('calls onStrategyChange when strategy changes', () => {
    const { rerender } = render(
      <VisualStrategyBuilder onStrategyChange={mockOnStrategyChange} />
    );

    // The component should call onStrategyChange with empty arrays initially
    expect(mockOnStrategyChange).toHaveBeenCalledWith([], []);
  });

  it('shows component library with different categories', async () => {
    render(
      <VisualStrategyBuilder onStrategyChange={mockOnStrategyChange} />
    );

    fireEvent.click(screen.getByText('Add Component'));

    await waitFor(() => {
      expect(screen.getByText('Moving Average')).toBeInTheDocument();
      expect(screen.getByText('RSI')).toBeInTheDocument();
      expect(screen.getByText('Cross Above')).toBeInTheDocument();
      expect(screen.getByText('AND Gate')).toBeInTheDocument();
      expect(screen.getByText('Buy Order')).toBeInTheDocument();
      expect(screen.getByText('Stop Loss')).toBeInTheDocument();
    });
  });

  it('closes component library when drawer is closed', async () => {
    render(
      <VisualStrategyBuilder onStrategyChange={mockOnStrategyChange} />
    );

    fireEvent.click(screen.getByText('Add Component'));

    await waitFor(() => {
      expect(screen.getByText('Component Library')).toBeInTheDocument();
    });

    // Find and click the close button (usually has aria-label="Close")
    const closeButton = screen.getByLabelText('Close') || 
                       screen.getByRole('button', { name: /close/i });
    
    if (closeButton) {
      fireEvent.click(closeButton);
    }

    await waitFor(() => {
      expect(screen.queryByText('Component Library')).not.toBeInTheDocument();
    });
  });

  it('renders component categories correctly', async () => {
    render(
      <VisualStrategyBuilder onStrategyChange={mockOnStrategyChange} />
    );

    fireEvent.click(screen.getByText('Add Component'));

    await waitFor(() => {
      // Check that category headers are rendered
      expect(screen.getByText('Indicator')).toBeInTheDocument();
      expect(screen.getByText('Signal')).toBeInTheDocument();
      expect(screen.getByText('Condition')).toBeInTheDocument();
      expect(screen.getByText('Action')).toBeInTheDocument();
      expect(screen.getByText('Risk Control')).toBeInTheDocument();
    });
  });

  it('shows helpful text when no components are present', () => {
    render(
      <VisualStrategyBuilder onStrategyChange={mockOnStrategyChange} />
    );

    expect(screen.getByText('Start building your strategy by adding components from the library')).toBeInTheDocument();
  });

  it('has correct ARIA attributes for accessibility', () => {
    render(
      <VisualStrategyBuilder onStrategyChange={mockOnStrategyChange} />
    );

    const addButton = screen.getByRole('button', { name: /add component/i });
    expect(addButton).toBeInTheDocument();
  });

  it('handles component addition workflow', async () => {
    render(
      <VisualStrategyBuilder onStrategyChange={mockOnStrategyChange} />
    );

    // Click Add Component
    fireEvent.click(screen.getByText('Add Component'));

    await waitFor(() => {
      expect(screen.getByText('Component Library')).toBeInTheDocument();
    });

    // The component library should show template components
    expect(screen.getByText('Moving Average')).toBeInTheDocument();
    expect(screen.getByText('Simple or Exponential Moving Average')).toBeInTheDocument();
  });

  it('displays correct builder mode information', () => {
    render(
      <VisualStrategyBuilder onStrategyChange={mockOnStrategyChange} />
    );

    expect(screen.getByText('Build strategies using drag-and-drop components')).toBeInTheDocument();
  });

  it('maintains component library structure', async () => {
    render(
      <VisualStrategyBuilder onStrategyChange={mockOnStrategyChange} />
    );

    fireEvent.click(screen.getByText('Add Component'));

    await waitFor(() => {
      // Verify the library has the expected structure
      expect(screen.getByText('Drag components to build your strategy')).toBeInTheDocument();
      
      // Check for specific component types
      const movingAverage = screen.getByText('Moving Average');
      const rsi = screen.getByText('RSI');
      const crossAbove = screen.getByText('Cross Above');
      
      expect(movingAverage).toBeInTheDocument();
      expect(rsi).toBeInTheDocument();
      expect(crossAbove).toBeInTheDocument();
    });
  });
});