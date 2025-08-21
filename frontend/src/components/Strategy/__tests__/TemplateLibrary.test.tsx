import React from 'react';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import { vi, describe, it, expect, beforeEach } from 'vitest';
import { TemplateLibrary } from '../TemplateLibrary';
import strategyService from '../services/strategyService';

// Mock the strategy service
vi.mock('../services/strategyService', () => ({
  default: {
    searchTemplates: vi.fn(),
  }
}));

const mockTemplates = [
  {
    id: 'ma_cross_001',
    name: 'Moving Average Cross',
    category: 'trend_following',
    description: 'A trend-following strategy',
    python_class: 'MovingAverageCrossStrategy',
    parameters: [
      {
        name: 'fast_period',
        display_name: 'Fast Period',
        type: 'integer',
        required: true,
        default_value: 10,
        help_text: 'Fast MA period',
        group: 'Technical'
      }
    ],
    risk_parameters: [
      {
        name: 'max_position_size',
        display_name: 'Max Position',
        type: 'decimal',
        required: true,
        help_text: 'Maximum position size',
        group: 'Risk',
        impact_level: 'high'
      }
    ],
    example_configs: [
      {
        name: 'Conservative',
        description: 'Conservative settings',
        parameters: { fast_period: 15 }
      }
    ],
    created_at: '2024-01-01T00:00:00Z',
    updated_at: '2024-01-01T00:00:00Z'
  }
];

const mockResponse = {
  templates: mockTemplates,
  categories: ['trend_following']
};

describe('TemplateLibrary', () => {
  const mockOnTemplateSelect = vi.fn();

  beforeEach(() => {
    vi.clearAllMocks();
    (strategyService.searchTemplates as any).mockResolvedValue(mockResponse);
  });

  it('renders template library with search and filters', async () => {
    render(
      <TemplateLibrary onTemplateSelect={mockOnTemplateSelect} />
    );

    expect(screen.getByText('Strategy Template Library')).toBeInTheDocument();
    expect(screen.getByPlaceholderText(/Search templates/)).toBeInTheDocument();
    expect(screen.getByText('Category:')).toBeInTheDocument();
    
    await waitFor(() => {
      expect(screen.getByText('Moving Average Cross')).toBeInTheDocument();
    });
  });

  it('displays template cards with correct information', async () => {
    render(
      <TemplateLibrary onTemplateSelect={mockOnTemplateSelect} />
    );

    await waitFor(() => {
      expect(screen.getByText('Moving Average Cross')).toBeInTheDocument();
      expect(screen.getByText('A trend-following strategy')).toBeInTheDocument();
      expect(screen.getByText('trend following')).toBeInTheDocument();
      expect(screen.getByText('MovingAverageCrossStrategy')).toBeInTheDocument();
    });
  });

  it('calls onTemplateSelect when template is clicked', async () => {
    render(
      <TemplateLibrary onTemplateSelect={mockOnTemplateSelect} />
    );

    await waitFor(() => {
      const templateCard = screen.getByText('Moving Average Cross').closest('.ant-card');
      expect(templateCard).toBeInTheDocument();
    });

    const templateCard = screen.getByText('Moving Average Cross').closest('.ant-card');
    fireEvent.click(templateCard!);

    expect(mockOnTemplateSelect).toHaveBeenCalledWith(mockTemplates[0]);
  });

  it('handles search functionality', async () => {
    render(
      <TemplateLibrary onTemplateSelect={mockOnTemplateSelect} />
    );

    const searchInput = screen.getByPlaceholderText(/Search templates/);
    fireEvent.change(searchInput, { target: { value: 'moving' } });

    await waitFor(() => {
      expect(strategyService.searchTemplates).toHaveBeenCalledWith(
        expect.objectContaining({ search_text: 'moving' })
      );
    });
  });

  it('handles category filtering', async () => {
    render(
      <TemplateLibrary onTemplateSelect={mockOnTemplateSelect} />
    );

    // Wait for initial load
    await waitFor(() => {
      expect(screen.getByText('Moving Average Cross')).toBeInTheDocument();
    });

    const categorySelect = screen.getByText('All Categories');
    fireEvent.click(categorySelect);

    const categoryOption = screen.getByText('trend following');
    fireEvent.click(categoryOption);

    await waitFor(() => {
      expect(strategyService.searchTemplates).toHaveBeenCalledWith(
        expect.objectContaining({ category: 'trend_following' })
      );
    });
  });

  it('shows error state when loading fails', async () => {
    (strategyService.searchTemplates as any).mockRejectedValue(
      new Error('Network error')
    );

    render(
      <TemplateLibrary onTemplateSelect={mockOnTemplateSelect} />
    );

    await waitFor(() => {
      expect(screen.getByText('Failed to Load Templates')).toBeInTheDocument();
      expect(screen.getByText('Network error')).toBeInTheDocument();
    });
  });

  it('shows empty state when no templates found', async () => {
    (strategyService.searchTemplates as any).mockResolvedValue({
      templates: [],
      categories: []
    });

    render(
      <TemplateLibrary onTemplateSelect={mockOnTemplateSelect} />
    );

    await waitFor(() => {
      expect(screen.getByText('No strategy templates found')).toBeInTheDocument();
    });
  });

  it('highlights selected template', async () => {
    render(
      <TemplateLibrary 
        onTemplateSelect={mockOnTemplateSelect}
        selectedTemplateId="ma_cross_001"
      />
    );

    await waitFor(() => {
      const templateCard = screen.getByText('Moving Average Cross').closest('.ant-card');
      expect(templateCard).toHaveClass('selected');
    });
  });

  it('displays parameter and example counts', async () => {
    render(
      <TemplateLibrary onTemplateSelect={mockOnTemplateSelect} />
    );

    await waitFor(() => {
      // Check for parameter badges
      expect(screen.getByText('Parameters')).toBeInTheDocument();
      expect(screen.getByText('Risk Controls')).toBeInTheDocument();
      expect(screen.getByText('Examples')).toBeInTheDocument();
    });
  });
});